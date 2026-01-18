from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: Optional[float]


class CoughClassifier:
    model_name = "stub-v1"

    def predict(self, features: np.ndarray) -> Prediction:
        if features.size == 0:
            raise ValueError("Feature vector is empty.")
        # Placeholder for PyTorch inference.
        return Prediction(label="cough_positive", confidence=0.82)
