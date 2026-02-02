from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    image_dir: Path
    features_dir: Path
    label_dir: Path | None
    label_key: str
    labels: Sequence[str]
    image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg")
    features_suffix: str = ".features.json"
    label_suffix: str = ".json"
    features_key: str = "features"
    require_both_modalities: bool = True


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    label: str
    image_path: Path | None
    features_path: Path | None


@dataclass(frozen=True)
class FeatureMatrix:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]


def build_sample_records(config: DatasetConfig) -> list[SampleRecord]:
    label_to_index = {label: idx for idx, label in enumerate(config.labels)}
    images = _collect_images(config.image_dir, config.image_extensions)
    features = _collect_features(config.features_dir, config.features_suffix)

    sample_ids = set(images)
    if config.require_both_modalities:
        sample_ids &= set(features)
    else:
        sample_ids |= set(features)

    if not sample_ids:
        raise ValueError(f"No samples found in {config.image_dir} and {config.features_dir}.")

    records: list[SampleRecord] = []
    skipped_missing_label = 0
    skipped_unknown = 0

    for sample_id in sorted(sample_ids):
        image_path = images.get(sample_id)
        features_path = features.get(sample_id)
        label_path = _resolve_label_path(config, sample_id, image_path, features_path)

        label = _read_label(label_path, config.label_key)
        if label is None:
            skipped_missing_label += 1
            continue
        if label not in label_to_index:
            skipped_unknown += 1
            logger.warning("Unknown label '%s' in %s", label, label_path)
            continue

        records.append(
            SampleRecord(
                sample_id=sample_id,
                label=label,
                image_path=image_path,
                features_path=features_path,
            )
        )

    if not records:
        raise ValueError("No valid samples after label filtering.")

    if skipped_missing_label or skipped_unknown:
        logger.info(
            "Skipped samples: missing_label=%s, unknown_label=%s",
            skipped_missing_label,
            skipped_unknown,
        )

    return records


def build_feature_matrix(
    records: Iterable[SampleRecord],
    label_map: dict[str, int],
    features_key: str,
) -> FeatureMatrix:
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    sample_ids: list[str] = []

    for record in records:
        if record.features_path is None:
            continue
        vector = _read_features(record.features_path, features_key)
        if vector.size == 0:
            continue
        features_list.append(vector)
        labels_list.append(label_map[record.label])
        sample_ids.append(record.sample_id)

    if not features_list:
        raise ValueError("No feature vectors found for KNN.")

    index_to_label = {idx: label for label, idx in label_map.items()}

    return FeatureMatrix(
        features=np.stack(features_list).astype(np.float32),
        labels=np.asarray(labels_list, dtype=np.int64),
        sample_ids=sample_ids,
        label_to_index=label_map,
        index_to_label=index_to_label,
    )


def _collect_images(root: Path, extensions: Sequence[str]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for ext in extensions:
        for path in root.rglob(f"*{ext}"):
            if not path.is_file():
                continue
            sample_id = _sample_id_from_path(path, root, ext)
            images[sample_id] = path
    return images


def _collect_features(root: Path, suffix: str) -> dict[str, Path]:
    features: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        sample_id = _sample_id_from_path(path, root, suffix)
        features[sample_id] = path
    return features


def _sample_id_from_path(path: Path, root: Path, suffix: str) -> str:
    relative = path.relative_to(root).as_posix()
    if relative.endswith(suffix):
        relative = relative[: -len(suffix)]
    else:
        relative = str(Path(relative).with_suffix(""))
    return relative


def _resolve_label_path(
    config: DatasetConfig,
    sample_id: str,
    image_path: Path | None,
    features_path: Path | None,
) -> Path:
    if config.label_dir is not None:
        return config.label_dir / f"{sample_id}{config.label_suffix}"
    if image_path is not None:
        return image_path.with_suffix(config.label_suffix)
    if features_path is not None:
        return _label_from_features_path(features_path, config.features_suffix, config.label_suffix)
    return config.image_dir / f"{sample_id}{config.label_suffix}"


def _label_from_features_path(
    path: Path,
    features_suffix: str,
    label_suffix: str,
) -> Path:
    name = path.name
    if name.endswith(features_suffix):
        base = name[: -len(features_suffix)]
        return path.parent / f"{base}{label_suffix}"
    return path.with_suffix(label_suffix)


def _read_label(path: Path, key: str) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read label from %s: %s", path, exc)
        return None
    value = payload.get(key)
    if isinstance(value, str):
        return value
    return None


def _read_features(path: Path, key: str) -> np.ndarray:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read features from %s: %s", path, exc)
        return np.array([], dtype=np.float32)

    vector = payload.get(key, [])
    if not isinstance(vector, list):
        return np.array([], dtype=np.float32)
    return np.asarray(vector, dtype=np.float32)
