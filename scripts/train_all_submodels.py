#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.training.config import DEFAULT_SPLITS, TrainingConfig
from app.ml.training.experiments import run_experiments
from app.ml.training.registry import (
    BREATH_CLASSIFIER_CONFIG,
    COUGH_CLASSIFIER_CONFIG,
    COUGH_VS_BREATH_CONFIG,
    DEFAULT_OUTPUT_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all CoughSense submodels sequentially.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        base_seed=args.seed,
        use_weighted_sampler=args.use_weighted_sampler,
        num_workers=args.num_workers,
        device=args.device,
    )

    for dataset_config in (
        COUGH_VS_BREATH_CONFIG,
        COUGH_CLASSIFIER_CONFIG,
        BREATH_CLASSIFIER_CONFIG,
    ):
        print(f"[pipeline] Training submodel: {dataset_config.name}")
        output_dir = args.output_dir / dataset_config.name
        run_experiments(
            dataset_config=dataset_config,
            training_config=training_config,
            splits=DEFAULT_SPLITS,
            output_dir=output_dir,
        )
        print(f"[pipeline] Completed submodel: {dataset_config.name}")


if __name__ == "__main__":
    main()
