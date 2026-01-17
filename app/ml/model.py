from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class CoughClassifier:
    def predict(self, features: np.ndarray) -> Prediction:
        if features.size == 0:
            raise ValueError("Feature vector is empty.")
        # Placeholder for PyTorch inference.
        return Prediction(label="cough_positive", confidence=0.82)
