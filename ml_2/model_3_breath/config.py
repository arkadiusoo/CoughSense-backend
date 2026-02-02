from __future__ import annotations

from pathlib import Path

from ml_2.utils.data import DatasetConfig


LABELS = (
    "Healthy",
    "URTI",
    "Asthma",
    "COPD",
    "LRTI",
    "Bronchiectasis",
    "Pneumonia",
    "Bronchiolitis",
)
LABEL_KEY = "status"


def build_dataset_config(data_dir: Path) -> DatasetConfig:
    return DatasetConfig(
        name="breath_classifier",
        image_dir=data_dir,
        features_dir=data_dir,
        label_dir=None,
        label_key=LABEL_KEY,
        labels=LABELS,
    )
