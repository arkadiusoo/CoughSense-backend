from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    label_key: str
    labels: Sequence[str]


@dataclass(frozen=True)
class ExperimentSplit:
    train_ratio: float
    test_ratio: float
    seed: int
    tag: str


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    base_seed: int = 42
    use_weighted_sampler: bool = False
    num_workers: int = 0
    device: str | None = None


DEFAULT_SPLITS: Sequence[ExperimentSplit] = (
    ExperimentSplit(train_ratio=0.9, test_ratio=0.1, seed=101, tag="90_10"),
    ExperimentSplit(train_ratio=0.8, test_ratio=0.2, seed=202, tag="80_20"),
    ExperimentSplit(train_ratio=0.7, test_ratio=0.3, seed=303, tag="70_30"),
    ExperimentSplit(train_ratio=0.6, test_ratio=0.4, seed=404, tag="60_40"),
)
