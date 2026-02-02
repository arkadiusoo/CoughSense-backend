from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    device: str | None = None
    use_class_weights: bool = True


@dataclass(frozen=True)
class CNNGrid:
    conv_channels_options: Sequence[Sequence[int]] = ((16, 32), (16, 32, 64))
    kernel_sizes: Sequence[int] = (3, 5)
    dropouts: Sequence[float] = (0.3, 0.5)
    dense_units: Sequence[int] = (64, 128)
    image_size: tuple[int, int] = (224, 224)


@dataclass(frozen=True)
class KNNGrid:
    neighbors: Sequence[int] = (3, 5, 7, 9)
    metrics: Sequence[str] = ("minkowski", "euclidean", "manhattan")
    weights: Sequence[str] = ("distance", "uniform")
    p_values: Sequence[int] = (1, 2)


@dataclass(frozen=True)
class FusionConfig:
    weight_pairs: Sequence[tuple[float, float]] = (
        (0.5, 0.5),
        (0.7, 0.3),
        (0.3, 0.7),
        (0.9, 0.1),
        (0.1, 0.9),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cnn_grid: CNNGrid = field(default_factory=CNNGrid)
    knn_grid: KNNGrid = field(default_factory=KNNGrid)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    selection_metric: str = "f1_macro"
    selection_split: str = "val"
