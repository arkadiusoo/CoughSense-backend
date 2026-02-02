from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionWeights:
    w_cnn: float
    w_knn: float

    def normalized(self) -> "FusionWeights":
        total = self.w_cnn + self.w_knn
        if total == 0:
            return FusionWeights(0.5, 0.5)
        return FusionWeights(self.w_cnn / total, self.w_knn / total)


def fuse_probabilities(
    cnn_probs: np.ndarray,
    knn_probs: np.ndarray,
    weights: FusionWeights,
) -> np.ndarray:
    if cnn_probs.shape != knn_probs.shape:
        raise ValueError("CNN and KNN probabilities must have the same shape.")
    normalized = weights.normalized()
    fused = normalized.w_cnn * cnn_probs + normalized.w_knn * knn_probs
    return np.clip(fused, 0.0, 1.0)
