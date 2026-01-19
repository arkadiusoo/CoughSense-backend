from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(labels)), minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)
