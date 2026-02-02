from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import dump
from torch import nn
from torch.utils.data import DataLoader

from ml_2.shared_architecture.cnn import CNN2DClassifier, CNNConfig
from ml_2.shared_architecture.fusion import FusionWeights, fuse_probabilities
from ml_2.shared_architecture.knn import KNNConfig, KNNPipeline
from ml_2.utils.data import DatasetConfig, FeatureMatrix, SampleRecord, build_feature_matrix, build_sample_records
from ml_2.utils.datasets import ImageConfig, SpectrogramDataset
from ml_2.utils.io import write_csv, write_json
from ml_2.utils.metrics import metrics_from_probabilities
from ml_2.utils.seed import set_seed
from ml_2.utils.splits import SplitIndices, stratified_split_three
from ml_2.utils.torch_utils import select_device

from .config import ExperimentConfig


def run_submodel_experiments(
    dataset_config: DatasetConfig,
    experiment_config: ExperimentConfig,
    output_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    if not dataset_config.image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {dataset_config.image_dir}")
    if not dataset_config.features_dir.exists():
        raise FileNotFoundError(f"Missing features directory: {dataset_config.features_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / dataset_config.name / timestamp
    cnn_dir = run_dir / "cnn"
    knn_dir = run_dir / "knn"
    fusion_dir = run_dir / "fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    cnn_dir.mkdir(parents=True, exist_ok=True)
    knn_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_config.name}] Loading sample records...")
    records = build_sample_records(dataset_config)
    print(f"[{dataset_config.name}] Loaded {len(records)} samples.")
    label_map = (
        list(dataset_config.labels)
        if dataset_config.labels
        else _infer_label_map(records)
    )
    if not label_map:
        raise ValueError("Label map is empty. Check dataset labels.")
    label_to_index = {label: idx for idx, label in enumerate(label_map)}
    labels_array = np.asarray([label_to_index[r.label] for r in records], dtype=np.int64)

    print(f"[{dataset_config.name}] Building train/val/test splits...")
    set_seed(experiment_config.split_seed)
    split_indices = stratified_split_three(
        labels_array,
        val_ratio=experiment_config.val_ratio,
        test_ratio=experiment_config.test_ratio,
        seed=experiment_config.split_seed,
    )
    if split_indices.train.size == 0:
        raise ValueError("Training split is empty. Adjust split ratios.")
    if split_indices.val.size == 0 or split_indices.test.size == 0:
        raise ValueError("Validation or test split is empty. Adjust split ratios.")

    split_payload = _serialize_split_indices(records, split_indices)
    write_json(run_dir / "splits.json", split_payload)

    print(f"[{dataset_config.name}] Loading feature vectors for KNN...")
    feature_matrix = build_feature_matrix(
        records,
        label_to_index,
        dataset_config.features_key,
    )
    feature_matrix = _align_feature_matrix(records, feature_matrix)
    print(f"[{dataset_config.name}] Feature matrix shape: {feature_matrix.features.shape}.")

    run_metadata = {
        "model": dataset_config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_config": _serialize_dataset_config(dataset_config),
        "label_map": label_map,
        "experiment_config": json.loads(json.dumps(asdict(experiment_config))),
    }
    write_json(run_dir / "run_config.json", run_metadata)

    print(f"[{dataset_config.name}] Starting CNN experiments...")
    cnn_results = _run_cnn_experiments(
        dataset_config,
        experiment_config,
        records,
        labels_array,
        split_indices,
        cnn_dir,
        label_map,
        label_to_index,
    )

    print(f"[{dataset_config.name}] Starting KNN experiments...")
    knn_results = _run_knn_experiments(
        dataset_config,
        experiment_config,
        feature_matrix,
        split_indices,
        knn_dir,
        label_map,
    )

    print(f"[{dataset_config.name}] Starting fusion experiments...")
    fusion_results = _run_fusion_experiments(
        experiment_config,
        labels_array,
        split_indices,
        cnn_results,
        knn_results,
        fusion_dir,
    )

    print(f"[{dataset_config.name}] Selecting best configuration...")
    best = _select_best_fusion(
        fusion_results,
        experiment_config.selection_metric,
        experiment_config.selection_split,
    )

    results_dir = results_root / dataset_config.name
    results_dir.mkdir(parents=True, exist_ok=True)
    best_summary = _write_best_artifacts(
        results_dir,
        run_dir,
        dataset_config.name,
        label_map,
        best,
    )

    summary = {
        "model": dataset_config.name,
        "run_dir": str(run_dir),
        "best": best_summary,
    }
    write_json(run_dir / "summary.json", summary)
    _update_results_registry(results_root, dataset_config.name, best_summary)

    print(f"[{dataset_config.name}] Done.")
    return summary


def _run_cnn_experiments(
    dataset_config: DatasetConfig,
    experiment_config: ExperimentConfig,
    records: list[SampleRecord],
    labels_array: np.ndarray,
    split_indices: SplitIndices,
    output_dir: Path,
    label_map: list[str],
    label_to_index: dict[str, int],
) -> list[dict[str, Any]]:
    image_config = ImageConfig(image_size=experiment_config.cnn_grid.image_size)
    device = select_device(experiment_config.training.device)

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    train_records = [records[idx] for idx in split_indices.train]
    val_records = [records[idx] for idx in split_indices.val]
    test_records = [records[idx] for idx in split_indices.test]

    train_labels = labels_array[split_indices.train]

    train_dataset = SpectrogramDataset(train_records, label_to_index, image_config)
    val_dataset = SpectrogramDataset(val_records, label_to_index, image_config)
    test_dataset = SpectrogramDataset(test_records, label_to_index, image_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=True,
        num_workers=experiment_config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=False,
        num_workers=experiment_config.training.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=False,
        num_workers=experiment_config.training.num_workers,
    )

    total_runs = (
        len(experiment_config.cnn_grid.conv_channels_options)
        * len(experiment_config.cnn_grid.kernel_sizes)
        * len(experiment_config.cnn_grid.dropouts)
        * len(experiment_config.cnn_grid.dense_units)
    )
    print(f"[{dataset_config.name}] CNN grid size: {total_runs} runs.")
    run_id = 0
    for conv_channels in experiment_config.cnn_grid.conv_channels_options:
        for kernel_size in experiment_config.cnn_grid.kernel_sizes:
            for dropout in experiment_config.cnn_grid.dropouts:
                for dense_units in experiment_config.cnn_grid.dense_units:
                    run_id += 1
                    print(
                        f"[{dataset_config.name}] CNN run {run_id}/{total_runs} "
                        f"(channels={conv_channels}, kernel={kernel_size}, "
                        f"dropout={dropout}, dense={dense_units})"
                    )
                    cnn_config = CNNConfig(
                        input_channels=3,
                        num_classes=len(label_map),
                        conv_channels=tuple(conv_channels),
                        kernel_size=kernel_size,
                        dropout=dropout,
                        dense_units=dense_units,
                    )
                    model = CNN2DClassifier(cnn_config)

                    class_weights = None
                    if experiment_config.training.use_class_weights:
                        class_weights = _compute_class_weights(
                            train_labels, len(label_map)
                        ).to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=experiment_config.training.learning_rate,
                        weight_decay=experiment_config.training.weight_decay,
                    )

                    history, best_state, best_epoch, best_val_loss = _train_cnn(
                        model,
                        train_loader,
                        val_loader,
                        criterion,
                        optimizer,
                        device,
                        experiment_config,
                    )
                    model.load_state_dict(best_state)

                    val_probs, val_labels = _predict_cnn_probs(model, val_loader, device)
                    test_probs, test_labels = _predict_cnn_probs(
                        model, test_loader, device
                    )

                    val_metrics = metrics_from_probabilities(
                        val_labels, val_probs, len(label_map)
                    )
                    test_metrics = metrics_from_probabilities(
                        test_labels, test_probs, len(label_map)
                    )

                    run_dir = output_dir / f"run_{run_id:03d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = run_dir / "model.pt"

                    _save_cnn_checkpoint(
                        checkpoint_path,
                        cnn_config,
                        label_map,
                        experiment_config.cnn_grid.image_size,
                        model.state_dict(),
                    )

                    metrics_payload = {
                        "model": dataset_config.name,
                        "run_id": run_id,
                        "config": _serialize_cnn_config(cnn_config, image_config),
                        "early_stopping": {
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                        },
                        "metrics": {
                            "val": _format_metrics(val_metrics, label_map),
                            "test": _format_metrics(test_metrics, label_map),
                        },
                    }
                    write_json(run_dir / "metrics.json", metrics_payload)
                    write_json(run_dir / "history.json", {"history": history})

                    results.append(
                        {
                            "run_id": run_id,
                            "checkpoint_path": str(checkpoint_path),
                            "config": metrics_payload["config"],
                            "metrics": metrics_payload["metrics"],
                            "val_probs": val_probs,
                            "test_probs": test_probs,
                        }
                    )

                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "conv_channels": "-".join(map(str, conv_channels)),
                            "kernel_size": kernel_size,
                            "dropout": dropout,
                            "dense_units": dense_units,
                            "val_f1": metrics_payload["metrics"]["val"]["f1_macro"],
                            "test_f1": metrics_payload["metrics"]["test"]["f1_macro"],
                        }
                    )

    write_json(output_dir / "summary.json", {"runs": summary_rows})
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        fieldnames=[
            "run_id",
            "conv_channels",
            "kernel_size",
            "dropout",
            "dense_units",
            "val_f1",
            "test_f1",
        ],
    )
    return results


def _run_knn_experiments(
    dataset_config: DatasetConfig,
    experiment_config: ExperimentConfig,
    feature_matrix: FeatureMatrix,
    split_indices: SplitIndices,
    output_dir: Path,
    label_map: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    x_train = feature_matrix.features[split_indices.train]
    y_train = feature_matrix.labels[split_indices.train]
    x_val = feature_matrix.features[split_indices.val]
    y_val = feature_matrix.labels[split_indices.val]
    x_test = feature_matrix.features[split_indices.test]
    y_test = feature_matrix.labels[split_indices.test]

    total_runs = (
        len(experiment_config.knn_grid.neighbors)
        * len(experiment_config.knn_grid.metrics)
        * len(experiment_config.knn_grid.weights)
        * len(experiment_config.knn_grid.p_values)
    )
    print(f"[{dataset_config.name}] KNN grid size: {total_runs} runs.")
    run_id = 0
    for neighbors in experiment_config.knn_grid.neighbors:
        for metric in experiment_config.knn_grid.metrics:
            for weights in experiment_config.knn_grid.weights:
                for p_value in experiment_config.knn_grid.p_values:
                    run_id += 1
                    print(
                        f"[{dataset_config.name}] KNN run {run_id}/{total_runs} "
                        f"(k={neighbors}, metric={metric}, weights={weights}, p={p_value})"
                    )
                    knn_config = KNNConfig(
                        n_neighbors=neighbors,
                        metric=metric,
                        weights=weights,
                        p=p_value,
                    )
                    pipeline = KNNPipeline(knn_config)
                    pipeline.fit(x_train, y_train)

                    val_probs = pipeline.predict_proba(x_val)
                    test_probs = pipeline.predict_proba(x_test)

                    val_metrics = metrics_from_probabilities(
                        y_val, val_probs, len(label_map)
                    )
                    test_metrics = metrics_from_probabilities(
                        y_test, test_probs, len(label_map)
                    )

                    run_dir = output_dir / f"run_{run_id:03d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    model_path = run_dir / "knn.pkl"

                    dump(
                        {
                            "config": asdict(knn_config),
                            "scaler": pipeline.scaler,
                            "model": pipeline.model,
                            "label_map": label_map,
                        },
                        model_path,
                    )

                    metrics_payload = {
                        "model": dataset_config.name,
                        "run_id": run_id,
                        "config": asdict(knn_config),
                        "metrics": {
                            "val": _format_metrics(val_metrics, label_map),
                            "test": _format_metrics(test_metrics, label_map),
                        },
                    }
                    write_json(run_dir / "metrics.json", metrics_payload)

                    results.append(
                        {
                            "run_id": run_id,
                            "model_path": str(model_path),
                            "config": metrics_payload["config"],
                            "metrics": metrics_payload["metrics"],
                            "val_probs": val_probs,
                            "test_probs": test_probs,
                        }
                    )

                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "neighbors": neighbors,
                            "metric": metric,
                            "weights": weights,
                            "p_value": p_value,
                            "val_f1": metrics_payload["metrics"]["val"]["f1_macro"],
                            "test_f1": metrics_payload["metrics"]["test"]["f1_macro"],
                        }
                    )

    write_json(output_dir / "summary.json", {"runs": summary_rows})
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        fieldnames=[
            "run_id",
            "neighbors",
            "metric",
            "weights",
            "p_value",
            "val_f1",
            "test_f1",
        ],
    )
    return results


def _run_fusion_experiments(
    experiment_config: ExperimentConfig,
    labels_array: np.ndarray,
    split_indices: SplitIndices,
    cnn_results: list[dict[str, Any]],
    knn_results: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    y_val = labels_array[split_indices.val]
    y_test = labels_array[split_indices.test]

    results: list[dict[str, Any]] = []
    total_runs = (
        len(cnn_results)
        * len(knn_results)
        * len(experiment_config.fusion.weight_pairs)
    )
    print(f"Fusion grid size: {total_runs} runs.")
    run_id = 0

    for cnn in cnn_results:
        for knn in knn_results:
            for weights_pair in experiment_config.fusion.weight_pairs:
                run_id += 1
                if total_runs <= 50 or run_id % 50 == 0 or run_id == total_runs:
                    print(f"Fusion run {run_id}/{total_runs} (cnn={cnn['run_id']}, knn={knn['run_id']})")
                weights = FusionWeights(*weights_pair)
                val_probs = fuse_probabilities(cnn["val_probs"], knn["val_probs"], weights)
                test_probs = fuse_probabilities(
                    cnn["test_probs"], knn["test_probs"], weights
                )

                val_metrics = metrics_from_probabilities(
                    y_val, val_probs, val_probs.shape[1]
                )
                test_metrics = metrics_from_probabilities(
                    y_test, test_probs, test_probs.shape[1]
                )

                payload = {
                    "run_id": run_id,
                    "cnn_run_id": cnn["run_id"],
                    "knn_run_id": knn["run_id"],
                    "weights": {"w_cnn": weights.w_cnn, "w_knn": weights.w_knn},
                    "metrics": {
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                }
                results.append(payload)

    write_json(output_dir / "fusion_results.json", {"runs": results})
    return results


def _select_best_fusion(
    fusion_results: list[dict[str, Any]],
    metric: str,
    split_name: str,
) -> dict[str, Any]:
    if not fusion_results:
        return {}

    split_name = split_name if split_name in ("val", "test") else "val"
    best = max(
        fusion_results,
        key=lambda item: item["metrics"][split_name].get(metric, 0.0),
    )
    return best


def _write_best_artifacts(
    results_dir: Path,
    run_dir: Path,
    model_name: str,
    label_map: list[str],
    best: dict[str, Any],
) -> dict[str, Any]:
    if not best:
        return {}

    cnn_run_id = best["cnn_run_id"]
    knn_run_id = best["knn_run_id"]
    cnn_src = _find_run_artifact(run_dir, "cnn", cnn_run_id, "model.pt")
    knn_src = _find_run_artifact(run_dir, "knn", knn_run_id, "knn.pkl")

    cnn_dst = results_dir / "best_cnn.pt"
    knn_dst = results_dir / "best_knn.pkl"
    if cnn_src:
        shutil.copy2(cnn_src, cnn_dst)
    if knn_src:
        shutil.copy2(knn_src, knn_dst)

    best_payload = {
        "model": model_name,
        "cnn_run_id": cnn_run_id,
        "knn_run_id": knn_run_id,
        "weights": best["weights"],
        "metrics": best["metrics"],
        "cnn_checkpoint": str(cnn_dst) if cnn_src else None,
        "knn_model": str(knn_dst) if knn_src else None,
    }

    write_json(results_dir / "best_config.json", best_payload)
    write_json(results_dir / "label_map.json", {"labels": label_map})
    return best_payload


def _find_run_artifact(
    base_dir: Path,
    model_type: str,
    run_id: int,
    filename: str,
) -> Path | None:
    run_dir = base_dir / model_type / f"run_{run_id:03d}" / filename
    if run_dir.exists():
        return run_dir
    return None


def _update_results_registry(
    results_root: Path,
    model_name: str,
    best_payload: dict[str, Any],
) -> None:
    registry_path = results_root / "registry.json"
    if registry_path.exists():
        try:
            with registry_path.open("r", encoding="utf-8") as handle:
                registry = json.load(handle)
        except json.JSONDecodeError:
            registry = {}
    else:
        registry = {}

    registry.setdefault("model_name", "CoughSenseModel_2-0")
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry.setdefault("submodels", {})
    registry["submodels"][model_name] = best_payload

    write_json(registry_path, registry)


def _serialize_dataset_config(config: DatasetConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "image_dir": str(config.image_dir),
        "features_dir": str(config.features_dir),
        "label_dir": str(config.label_dir) if config.label_dir else None,
        "label_key": config.label_key,
        "labels": list(config.labels) if config.labels else None,
        "image_extensions": list(config.image_extensions),
        "features_suffix": config.features_suffix,
        "label_suffix": config.label_suffix,
        "features_key": config.features_key,
        "require_both_modalities": config.require_both_modalities,
    }


def _infer_label_map(records: list[SampleRecord]) -> list[str]:
    seen: set[str] = set()
    label_map: list[str] = []
    for record in records:
        if record.label in seen:
            continue
        seen.add(record.label)
        label_map.append(record.label)
    return label_map


def _serialize_split_indices(
    records: list[SampleRecord],
    split_indices: SplitIndices,
) -> dict[str, list[str]]:
    return {
        "train_ids": [records[idx].sample_id for idx in split_indices.train],
        "val_ids": [records[idx].sample_id for idx in split_indices.val],
        "test_ids": [records[idx].sample_id for idx in split_indices.test],
    }


def _serialize_cnn_config(
    cnn_config: CNNConfig,
    image_config: ImageConfig,
) -> dict[str, Any]:
    payload = asdict(cnn_config)
    payload["image_size"] = list(image_config.image_size)
    return payload


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float32)
    for idx, count in enumerate(counts):
        if count > 0:
            weights[idx] = total / (num_classes * count)
        else:
            weights[idx] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def _train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    experiment_config: ExperimentConfig,
) -> tuple[list[dict[str, float]], dict[str, Any], int, float]:
    history: list[dict[str, float]] = []
    model.to(device)

    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, experiment_config.training.max_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = train_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        val_loss, val_acc = _evaluate_loss_accuracy(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"  Epoch {epoch}/{experiment_config.training.max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - experiment_config.training.early_stopping_min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= experiment_config.training.early_stopping_patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return history, best_state, best_epoch, best_val_loss


def _evaluate_loss_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (total_loss / total if total else 0.0, correct / total if total else 0.0)


def _predict_cnn_probs(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(labels.numpy())
    return np.vstack(probs_list), np.concatenate(labels_list)


def _format_metrics(
    metrics: dict[str, Any],
    label_map: list[str],
) -> dict[str, Any]:
    per_class = {}
    for idx, label in enumerate(label_map):
        per_class[label] = {
            "precision": metrics["precision_per_class"][idx],
            "recall": metrics["recall_per_class"][idx],
            "f1": metrics["f1_per_class"][idx],
        }
    return {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_map,
            "matrix": metrics["confusion_matrix"],
        },
    }


def _save_cnn_checkpoint(
    path: Path,
    config: CNNConfig,
    label_map: list[str],
    image_size: tuple[int, int],
    state_dict: dict[str, Any],
) -> None:
    payload = {
        "model_config": asdict(config),
        "label_map": label_map,
        "image_size": list(image_size),
        "state_dict": state_dict,
    }
    torch.save(payload, path)


def _align_feature_matrix(
    records: list[SampleRecord],
    feature_matrix: FeatureMatrix,
) -> FeatureMatrix:
    record_ids = [record.sample_id for record in records]
    index_lookup = {sid: idx for idx, sid in enumerate(feature_matrix.sample_ids)}
    missing = [sid for sid in record_ids if sid not in index_lookup]
    if missing:
        raise ValueError(f"Missing feature vectors for samples: {missing[:5]}")

    ordered_indices = [index_lookup[sid] for sid in record_ids]
    return FeatureMatrix(
        features=feature_matrix.features[ordered_indices],
        labels=feature_matrix.labels[ordered_indices],
        sample_ids=record_ids,
        label_to_index=feature_matrix.label_to_index,
        index_to_label=feature_matrix.index_to_label,
    )
