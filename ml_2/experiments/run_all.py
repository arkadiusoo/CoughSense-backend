from __future__ import annotations

import argparse
from pathlib import Path

from ml_2.experiments.config import ExperimentConfig
from ml_2.experiments.pipeline import run_submodel_experiments
from ml_2.model_1_sound_type.config import build_dataset_config as build_sound_config
from ml_2.model_2_cough.config import build_dataset_config as build_cough_config
from ml_2.model_3_breath.config import build_dataset_config as build_breath_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoughSenseModel 2.0 experiments.")
    parser.add_argument("--sound-data", required=True, type=Path)
    parser.add_argument("--cough-data", required=True, type=Path)
    parser.add_argument("--breath-data", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("ml_2/experiments"))
    parser.add_argument("--results-root", type=Path, default=Path("ml_2/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = ExperimentConfig()

    sound_config = build_sound_config(args.sound_data)
    cough_config = build_cough_config(args.cough_data)
    breath_config = build_breath_config(args.breath_data)

    run_submodel_experiments(sound_config, experiment_config, args.output_root, args.results_root)
    run_submodel_experiments(cough_config, experiment_config, args.output_root, args.results_root)
    run_submodel_experiments(breath_config, experiment_config, args.output_root, args.results_root)


if __name__ == "__main__":
    main()
