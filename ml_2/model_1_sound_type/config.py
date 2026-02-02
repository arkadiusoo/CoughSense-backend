from __future__ import annotations

from pathlib import Path

from ml_2.utils.data import DatasetConfig


LABELS = ("cough", "breath")
LABEL_KEY = "type"


def build_dataset_config(data_dir: Path) -> DatasetConfig:
    return DatasetConfig(
        name="signal_type",
        image_dir=data_dir,
        features_dir=data_dir,
        label_dir=None,
        label_key=LABEL_KEY,
        labels=LABELS,
    )
