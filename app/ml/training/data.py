from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np

from app.audio.features import extract_features
from app.audio.io import AudioSignal

from .config import DatasetConfig


logger = logging.getLogger(__name__)


@dataclass
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    file_paths: list[Path]


@dataclass(frozen=True)
class SampleItem:
    audio_path: Path
    label: str


def build_feature_dataset(config: DatasetConfig) -> FeatureDataset:
    samples = _collect_samples(config)
    if not samples:
        raise ValueError(f"No samples found in {config.data_dir}.")

    label_to_index = {label: idx for idx, label in enumerate(config.labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    file_paths: list[Path] = []

    for sample in samples:
        if sample.label not in label_to_index:
            logger.warning("Unknown label '%s' in %s", sample.label, sample.audio_path)
            continue
        signal = _load_audio_signal(sample.audio_path)
        features = extract_features(signal)
        features_list.append(features)
        labels_list.append(label_to_index[sample.label])
        file_paths.append(sample.audio_path)

    if not features_list:
        raise ValueError("No valid samples after label filtering.")

    feature_matrix = np.stack(features_list).astype(np.float32)
    label_array = np.asarray(labels_list, dtype=np.int64)

    return FeatureDataset(
        features=feature_matrix,
        labels=label_array,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        file_paths=file_paths,
    )


def _collect_samples(config: DatasetConfig) -> list[SampleItem]:
    audio_files = sorted(config.data_dir.rglob("*.wav"))
    samples: list[SampleItem] = []
    for audio_path in audio_files:
        metadata_path = audio_path.with_suffix(".json")
        if not metadata_path.exists():
            logger.warning("Missing metadata for %s", audio_path)
            continue
        label = _read_label(metadata_path, config.label_key)
        if label is None:
            logger.warning("Missing label '%s' in %s", config.label_key, metadata_path)
            continue
        samples.append(SampleItem(audio_path=audio_path, label=label))
    return samples


def _read_label(metadata_path: Path, key: str) -> str | None:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", metadata_path, exc)
        return None
    value = data.get(key)
    if isinstance(value, str):
        return value
    return None


def _load_audio_signal(path: Path) -> AudioSignal:
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0 or sr is None:
        raise ValueError(f"Audio data is empty: {path}")
    return AudioSignal(samples=y, sample_rate=int(sr))


def stratified_split(
    labels: Sequence[int],
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(labels_array):
        class_indices = np.where(labels_array == class_id)[0]
        rng.shuffle(class_indices)
        if len(class_indices) <= 1:
            train_indices.extend(class_indices.tolist())
            continue
        desired_test = int(round(len(class_indices) * test_ratio))
        test_count = min(max(1, desired_test), len(class_indices) - 1)
        test_indices.extend(class_indices[:test_count].tolist())
        train_indices.extend(class_indices[test_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices), np.asarray(test_indices)
