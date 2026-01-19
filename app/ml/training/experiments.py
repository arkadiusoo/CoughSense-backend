from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import DatasetConfig, ExperimentSplit, TrainingConfig
from .data import FeatureDataset, build_feature_dataset, stratified_split
from .metrics import classification_metrics
from .modeling import CNN1DClassifier, ModelConfig
from .trainer import evaluate, train_model
from .utils import compute_class_weights, select_device, set_seed


def run_experiments(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    splits: Sequence[ExperimentSplit],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_feature_dataset(dataset_config)

    results: list[dict] = []
    summary_rows: list[dict] = []

    for split in splits:
        split_seed = training_config.base_seed + split.seed
        set_seed(split_seed)

        train_idx, test_idx = stratified_split(
            dataset.labels, split.test_ratio, split_seed
        )

        split_dir = output_dir / split.tag
        split_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload, history, state_dict = _train_and_evaluate_split(
            dataset=dataset,
            dataset_config=dataset_config,
            training_config=training_config,
            train_idx=train_idx,
            test_idx=test_idx,
            split=split,
        )

        checkpoint_path = split_dir / "model.pt"
        label_map_path = split_dir / "label_map.json"
        model_config_path = split_dir / "model_config.json"
        metrics_path = split_dir / "metrics.json"
        history_path = split_dir / "history.json"

        _write_json(label_map_path, metrics_payload["label_map"])
        _write_json(model_config_path, metrics_payload["model_config"])
        _write_json(metrics_path, metrics_payload)
        _write_json(history_path, history)

        _save_checkpoint(
            checkpoint_path,
            metrics_payload["model_config"],
            metrics_payload["label_map"],
            state_dict,
        )

        summary_row = {
            "split_tag": split.tag,
            "train_ratio": split.train_ratio,
            "test_ratio": split.test_ratio,
            "seed": split_seed,
            "num_train": int(metrics_payload["train_summary"]["num_samples"]),
            "num_test": int(metrics_payload["test_summary"]["num_samples"]),
            "accuracy": metrics_payload["metrics"]["accuracy"],
            "precision_macro": metrics_payload["metrics"]["precision_macro"],
            "recall_macro": metrics_payload["metrics"]["recall_macro"],
            "f1_macro": metrics_payload["metrics"]["f1_macro"],
        }
        summary_rows.append(summary_row)

        results.append(
            {
                "split": split.tag,
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
                "label_map_path": str(label_map_path),
                "model_config_path": str(model_config_path),
            }
        )

    summary = {
        "model": dataset_config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": summary_rows,
        "best_split": _select_best_split(summary_rows),
    }

    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    _write_json(summary_path, summary)
    _write_summary_csv(summary_csv_path, summary_rows)

    _update_best_models_registry(output_dir.parent, dataset_config.name, summary)

    return {
        "summary_path": str(summary_path),
        "summary_csv_path": str(summary_csv_path),
        "results": results,
    }


def _train_and_evaluate_split(
    dataset: FeatureDataset,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split: ExperimentSplit,
) -> tuple[dict, list[dict[str, float]], dict]:
    device = select_device(training_config.device)

    x_train = torch.from_numpy(dataset.features[train_idx]).float().unsqueeze(1)
    y_train = torch.from_numpy(dataset.labels[train_idx]).long()
    x_test = torch.from_numpy(dataset.features[test_idx]).float().unsqueeze(1)
    y_test = torch.from_numpy(dataset.labels[test_idx]).long()

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    sampler = None
    if training_config.use_weighted_sampler:
        weights = compute_class_weights(y_train.tolist(), len(dataset_config.labels))
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=training_config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model_config = ModelConfig(
        input_features=dataset.features.shape[1],
        num_classes=len(dataset_config.labels),
        dropout=training_config.dropout,
    )
    model = CNN1DClassifier(model_config)

    class_weights = compute_class_weights(y_train.tolist(), len(dataset_config.labels))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=training_config.epochs,
    )

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = classification_metrics(
        y_true,
        y_pred,
        num_classes=len(dataset_config.labels),
    )

    per_class = {}
    for idx, label in dataset.index_to_label.items():
        per_class[label] = {
            "precision": metrics["precision_per_class"][idx],
            "recall": metrics["recall_per_class"][idx],
            "f1": metrics["f1_per_class"][idx],
        }

    label_map = [dataset.index_to_label[idx] for idx in range(len(dataset.index_to_label))]
    metrics_payload = {
        "model": dataset_config.name,
        "split": asdict(split),
        "seed": training_config.base_seed + split.seed,
        "label_map": label_map,
        "model_config": asdict(model_config),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "per_class": per_class,
            "confusion_matrix": {
                "labels": label_map,
                "matrix": metrics["confusion_matrix"],
            },
        },
        "train_summary": {
            "num_samples": int(len(train_idx)),
            "class_weights": class_weights.tolist(),
        },
        "test_summary": {
            "num_samples": int(len(test_idx)),
        },
    }

    return metrics_payload, history, model.state_dict()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_summary_csv(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "split_tag",
        "train_ratio",
        "test_ratio",
        "seed",
        "num_train",
        "num_test",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_best_split(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    best_row = max(rows, key=lambda item: item.get("f1_macro", 0.0))
    return best_row


def _save_checkpoint(
    path: Path,
    model_config: dict,
    label_map: dict,
    state_dict: dict,
) -> None:
    payload = {
        "model_config": model_config,
        "label_map": label_map,
        "state_dict": state_dict,
    }
    torch.save(payload, path)


def _update_best_models_registry(
    root_output_dir: Path,
    submodel_name: str,
    summary: dict,
) -> None:
    registry_path = root_output_dir / "best_models.json"
    if registry_path.exists():
        try:
            with registry_path.open("r", encoding="utf-8") as handle:
                registry = json.load(handle)
        except json.JSONDecodeError:
            registry = {}
    else:
        registry = {}

    registry.setdefault("model_name", "CoughSenseModel_1-0")
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry.setdefault("submodels", {})

    best_split = summary.get("best_split")
    if best_split:
        best_tag = best_split["split_tag"]
        submodel_dir = Path(summary.get("model", submodel_name))
        split_dir = submodel_dir / best_tag
        registry["submodels"][submodel_name] = {
            "split_tag": best_tag,
            "checkpoint_path": str(split_dir / "model.pt"),
            "label_map_path": str(split_dir / "label_map.json"),
            "model_config_path": str(split_dir / "model_config.json"),
            "metrics_path": str(split_dir / "metrics.json"),
        }

    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
