from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.ml.training.modeling import CNN1DClassifier, ModelConfig


logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/coughsense_model_1_0")
DEFAULT_REGISTRY_PATH = DEFAULT_ARTIFACTS_DIR / "best_models.json"
DEFAULT_MIN_SIGNAL_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: Optional[float]


class _StubModel:
    model_name = "stub-v1"

    def predict(self, features: np.ndarray) -> Prediction:
        if features.size == 0:
            raise ValueError("Feature vector is empty.")
        return Prediction(label="cough_positive", confidence=0.82)


class TorchAudioClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model_config = ModelConfig(**payload["model_config"])
        self.labels = list(payload["label_map"])
        self.model = CNN1DClassifier(model_config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> Prediction:
        probs = self.predict_proba(features)
        best_idx = int(np.argmax(probs))
        return Prediction(label=self.labels[best_idx], confidence=float(probs[best_idx]))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            raise ValueError("Feature vector is empty.")
        tensor = torch.from_numpy(features).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(1)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs


class CoughSenseModel_1_0:
    model_name = "CoughSenseModel_1-0"

    def __init__(
        self,
        signal_type_model: TorchAudioClassifier,
        cough_model: TorchAudioClassifier,
        breath_model: TorchAudioClassifier,
        min_signal_confidence: float = DEFAULT_MIN_SIGNAL_CONFIDENCE,
    ) -> None:
        self.signal_type_model = signal_type_model
        self.cough_model = cough_model
        self.breath_model = breath_model
        self.min_signal_confidence = min_signal_confidence

    @classmethod
    def from_registry(
        cls,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        min_signal_confidence: float = DEFAULT_MIN_SIGNAL_CONFIDENCE,
    ) -> "CoughSenseModel_1_0":
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Missing model registry: {registry_path}. Train models first."
            )

        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)

        submodels = registry.get("submodels", {})
        required = ("signal_type", "cough_classifier", "breath_classifier")
        missing = [name for name in required if name not in submodels]
        if missing:
            raise ValueError(
                f"Registry is missing submodels: {', '.join(missing)}"
            )

        def resolve(path_str: str) -> Path:
            return registry_path.parent / Path(path_str)

        signal_model = TorchAudioClassifier(resolve(submodels["signal_type"]["checkpoint_path"]))
        cough_model = TorchAudioClassifier(resolve(submodels["cough_classifier"]["checkpoint_path"]))
        breath_model = TorchAudioClassifier(resolve(submodels["breath_classifier"]["checkpoint_path"]))

        return cls(
            signal_type_model=signal_model,
            cough_model=cough_model,
            breath_model=breath_model,
            min_signal_confidence=min_signal_confidence,
        )

    def predict(self, features: np.ndarray) -> Prediction:
        signal_prediction = self.signal_type_model.predict(features)
        if (
            signal_prediction.confidence is None
            or signal_prediction.confidence < self.min_signal_confidence
        ):
            return Prediction(label="reject", confidence=signal_prediction.confidence)

        if signal_prediction.label == "cough":
            return self.cough_model.predict(features)
        if signal_prediction.label == "breath":
            return self.breath_model.predict(features)

        return Prediction(label="reject", confidence=signal_prediction.confidence)


class CoughClassifier:
    def __init__(self) -> None:
        self._model = self._load_model()

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def predict(self, features: np.ndarray) -> Prediction:
        return self._model.predict(features)

    def _load_model(self) -> object:
        try:
            return CoughSenseModel_1_0.from_registry()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Falling back to stub model: %s", exc)
            return _StubModel()
