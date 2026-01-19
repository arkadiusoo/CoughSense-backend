from __future__ import annotations

from pathlib import Path

from .config import DatasetConfig


DEFAULT_COUGH_VS_BREATH_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/coughVSbreath"
)
DEFAULT_LABELED_COUGH_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_cough"
)
DEFAULT_LABELED_BREATH_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_breath"
)

COUGH_VS_BREATH_CONFIG = DatasetConfig(
    name="signal_type",
    data_dir=DEFAULT_COUGH_VS_BREATH_DIR,
    label_key="type",
    labels=("cough", "breath"),
)

COUGH_CLASSIFIER_CONFIG = DatasetConfig(
    name="cough_classifier",
    data_dir=DEFAULT_LABELED_COUGH_DIR,
    label_key="status",
    labels=("COVID-19", "healthy", "symptomatic"),
)

BREATH_CLASSIFIER_CONFIG = DatasetConfig(
    name="breath_classifier",
    data_dir=DEFAULT_LABELED_BREATH_DIR,
    label_key="status",
    labels=(
        "Healthy",
        "URTI",
        "Asthma",
        "COPD",
        "LRTI",
        "Bronchiectasis",
        "Pneumonia",
        "Bronchiolitis",
    ),
)

DEFAULT_OUTPUT_DIR = Path("artifacts/coughsense_model_1_0")
