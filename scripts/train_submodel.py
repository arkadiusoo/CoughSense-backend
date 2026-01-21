#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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


MODEL_CONFIGS = {
    "signal_type": COUGH_VS_BREATH_CONFIG,
    "cough_classifier": COUGH_CLASSIFIER_CONFIG,
    "breath_classifier": BREATH_CLASSIFIER_CONFIG,
}


def build_parser(default_model: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a CoughSense submodel with repeated stratified splits.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=default_model,
        required=default_model is None,
        help="Which submodel to train.",
    )
    parser.add_argument("--data-dir", type=Path, help="Override dataset directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device selection, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Enable weighted sampling to reduce class imbalance.",
    )
    return parser


def main(default_model: str | None = None) -> None:
    parser = build_parser(default_model=default_model)
    args = parser.parse_args()

    dataset_config = MODEL_CONFIGS[args.model]
    if args.data_dir:
        dataset_config = replace(dataset_config, data_dir=args.data_dir)

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

    output_dir = args.output_dir / dataset_config.name
    run_experiments(
        dataset_config=dataset_config,
        training_config=training_config,
        splits=DEFAULT_SPLITS,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
