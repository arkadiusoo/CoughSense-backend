from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from joblib import load
from PIL import Image

from ml_2.shared_architecture.cnn import CNN2DClassifier, CNNConfig
from ml_2.shared_architecture.fusion import FusionWeights, fuse_probabilities
from ml_2.shared_architecture.knn import KNNConfig, KNNPipeline


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: Optional[float]


class CNNPredictor:
    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.config = CNNConfig(**payload["model_config"])
        self.label_map = list(payload["label_map"])
        self.image_size = tuple(payload["image_size"])
        self.model = CNN2DClassifier(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)

    def predict_proba(self, image: Image.Image | Path) -> np.ndarray:
        tensor = self._prepare_image(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def _prepare_image(self, image: Image.Image | Path) -> torch.Tensor:
        if isinstance(image, Path):
            image = Image.open(image)
        image = image.convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)


class KNNPredictor:
    def __init__(self, model_path: Path) -> None:
        payload = load(model_path)
        self.config = KNNConfig(**payload["config"])
        self.label_map = list(payload["label_map"])
        self.pipeline = KNNPipeline(self.config)
        self.pipeline.scaler = payload["scaler"]
        self.pipeline.model = payload["model"]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.pipeline.predict_proba(features)[0]


class HybridSubmodel:
    def __init__(
        self,
        cnn: CNNPredictor,
        knn: KNNPredictor,
        fusion_weights: FusionWeights,
    ) -> None:
        self.cnn = cnn
        self.knn = knn
        self.fusion_weights = fusion_weights
        self.label_map = cnn.label_map

    @classmethod
    def from_result_dir(cls, result_dir: Path) -> "HybridSubmodel":
        config_path = result_dir / "best_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing best_config.json in {result_dir}")
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        cnn_path = Path(payload["cnn_checkpoint"])
        knn_path = Path(payload["knn_model"])
        weights = payload.get("weights", {"w_cnn": 0.5, "w_knn": 0.5})

        cnn = CNNPredictor(cnn_path)
        knn = KNNPredictor(knn_path)
        fusion_weights = FusionWeights(weights["w_cnn"], weights["w_knn"])
        return cls(cnn, knn, fusion_weights)

    def predict_proba(
        self,
        image: Image.Image | Path,
        features: np.ndarray,
    ) -> np.ndarray:
        cnn_probs = self.cnn.predict_proba(image)
        knn_probs = self.knn.predict_proba(features)
        fused = fuse_probabilities(
            cnn_probs.reshape(1, -1),
            knn_probs.reshape(1, -1),
            self.fusion_weights,
        )[0]
        return fused

    def predict(
        self,
        image: Image.Image | Path,
        features: np.ndarray,
    ) -> Prediction:
        probs = self.predict_proba(image, features)
        best_idx = int(np.argmax(probs))
        return Prediction(label=self.label_map[best_idx], confidence=float(probs[best_idx]))


class CoughSenseModel_2_0:
    model_name = "CoughSenseModel_2-0"

    def __init__(
        self,
        signal_type_model: HybridSubmodel,
        cough_model: HybridSubmodel,
        breath_model: HybridSubmodel,
        min_signal_confidence: float = 0.5,
    ) -> None:
        self.signal_type_model = signal_type_model
        self.cough_model = cough_model
        self.breath_model = breath_model
        self.min_signal_confidence = min_signal_confidence

    @classmethod
    def from_results(cls, results_root: Path) -> "CoughSenseModel_2_0":
        signal_model = HybridSubmodel.from_result_dir(results_root / "signal_type")
        cough_model = HybridSubmodel.from_result_dir(results_root / "cough_classifier")
        breath_model = HybridSubmodel.from_result_dir(results_root / "breath_classifier")
        return cls(signal_model, cough_model, breath_model)

    def predict(
        self,
        image: Image.Image | Path,
        features: np.ndarray,
    ) -> Prediction:
        signal_prediction = self.signal_type_model.predict(image, features)
        if (
            signal_prediction.confidence is None
            or signal_prediction.confidence < self.min_signal_confidence
        ):
            return Prediction(label="reject", confidence=signal_prediction.confidence)

        if signal_prediction.label == "cough":
            return self.cough_model.predict(image, features)
        if signal_prediction.label == "breath":
            return self.breath_model.predict(image, features)
        return Prediction(label="reject", confidence=signal_prediction.confidence)
