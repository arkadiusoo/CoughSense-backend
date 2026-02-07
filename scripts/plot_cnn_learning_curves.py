#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = ("signal_type", "cough_classifier", "breath_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CNN learning curves for CoughSenseModel 2.0 runs."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("ml_2/experiments"),
        help="Root directory with experiment runs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("ml_2/results"),
        help="Root directory with registry.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/learning_curves"),
        help="Directory to save plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = args.results_root / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry.json: {registry_path}")

    registry = json.loads(registry_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        best = registry.get("submodels", {}).get(model_name)
        if not best:
            print(f"[{model_name}] Missing entry in registry.json")
            continue

        cnn_run_id = best.get("cnn_run_id")
        if cnn_run_id is None:
            print(f"[{model_name}] Missing cnn_run_id")
            continue

        run_dir = _find_latest_run_dir(args.experiments_root / model_name)
        if run_dir is None:
            print(f"[{model_name}] No runs found in {args.experiments_root / model_name}")
            continue

        history_path = run_dir / "cnn" / f"run_{cnn_run_id:03d}" / "history.json"
        if not history_path.exists():
            print(f"[{model_name}] Missing history.json: {history_path}")
            continue

        history = json.loads(history_path.read_text()).get("history", [])
        if not history:
            print(f"[{model_name}] Empty history in {history_path}")
            continue

        _plot_history(model_name, history, args.output_dir)


def _find_latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not run_dirs:
        return None
    return run_dirs[-1]


def _plot_history(model_name: str, history: list[dict], output_dir: Path) -> None:
    epochs = [item.get("epoch") for item in history]
    train_loss = [item.get("train_loss") for item in history]
    val_loss = [item.get("val_loss") for item in history]
    train_acc = [item.get("train_accuracy") for item in history]
    val_acc = [item.get("val_accuracy") for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_accuracy")
    axes[1].plot(epochs, val_acc, label="val_accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / f"{model_name}_learning_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[{model_name}] Saved: {output_path}")


if __name__ == "__main__":
    main()
