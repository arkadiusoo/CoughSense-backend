from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KNNConfig:
    n_neighbors: int = 5
    metric: str = "minkowski"
    weights: Literal["uniform", "distance"] = "distance"
    p: int = 2


class KNNPipeline:
    def __init__(self, config: KNNConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            metric=config.metric,
            weights=config.weights,
            p=config.p,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)
