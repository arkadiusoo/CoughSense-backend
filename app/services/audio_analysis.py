import time
from pathlib import Path
import logging
from datetime import datetime, timezone
import json

from fastapi import HTTPException, UploadFile, status

from app.audio.conversion import convert_to_wav
from app.audio.features import extract_features
from app.audio.io import load_audio_bytes
from app.audio.spectrogram import (
    generate_spectrogram_image_from_wav_bytes,
    save_debug_spectrogram,
)
from app.audio.validation import validate_duration, validate_format
from app.db.repository import AnalysisRepository
from app.ml.model import CoughClassifier
from app.ifDev import (
    DEV_SPECTROGRAM_DIR,
    DEV_SUBMODEL_LOG_DIR,
    IF_DEV,
    SAVE_SUBMODEL_LOGS,
)
from ml_2.model import CoughSenseModel_2_0, Prediction as HybridPrediction


ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm"}
MIN_DURATION_SECONDS = 0.1
MAX_DURATION_SECONDS = 60.0
HYBRID_RESULTS_DIR = Path("ml_2/results")
SIGNAL_TYPE_DURATION_SECONDS = 20.0
COUGH_DURATION_SECONDS = 10.0
BREATH_DURATION_SECONDS = 20.0

logger = logging.getLogger(__name__)

repository = AnalysisRepository()


async def analyze_cough_audio(file: UploadFile) -> dict:
    _validate_upload(file)
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload.",
        )

    start_time = time.perf_counter()
    try:
        wav_bytes = convert_to_wav(
            audio_bytes,
            content_type=file.content_type or "",
            filename=file.filename or "",
        )
        signal = load_audio_bytes(wav_bytes)
        validate_duration(
            signal.duration_seconds,
            MIN_DURATION_SECONDS,
            MAX_DURATION_SECONDS,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    features = extract_features(signal)
    prediction = _predict(wav_bytes, features)
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    saved_analysis = repository.save_analysis(
        result_label=prediction.label,
        result_confidence=prediction.confidence,
        recording_duration_seconds=signal.duration_seconds,
        processing_time_ms=processing_time_ms,
        model_ai=classifier.model_name,
    )
    return {
        "result": {
            "label": prediction.label,
            # "confidence": prediction.confidence,
        },
        "analyzed_at": saved_analysis.analyzed_at,
    }


def _validate_upload(file: UploadFile) -> None:
    try:
        validate_format(
            content_type=file.content_type or "",
            filename=file.filename or "",
            allowed_content_types=ALLOWED_CONTENT_TYPES,
            allowed_extensions=ALLOWED_EXTENSIONS,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _load_classifier() -> CoughSenseModel_2_0 | CoughClassifier:
    try:
        return CoughSenseModel_2_0.from_results(HYBRID_RESULTS_DIR)
    except Exception as exc:
        logger.warning("Could not load CoughSenseModel_2.0 from %s: %s", HYBRID_RESULTS_DIR, exc)
        logger.warning("Falling back to legacy CoughClassifier.")
        return CoughClassifier()


def _predict(wav_bytes: bytes, features) -> HybridPrediction:
    if isinstance(classifier, CoughSenseModel_2_0):
        return _predict_hybrid(classifier, wav_bytes, features)
    return classifier.predict(features)


def _predict_hybrid(
    model: CoughSenseModel_2_0,
    wav_bytes: bytes,
    features,
) -> HybridPrediction:
    debug_payload: dict = {
        "model_name": model.model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submodels": {},
    }

    # Stage 1: generate spectrogram for signal_type model (20s, as in training script).
    signal_image = generate_spectrogram_image_from_wav_bytes(
        wav_bytes=wav_bytes,
        target_duration_seconds=SIGNAL_TYPE_DURATION_SECONDS,
    )
    _save_spectrogram_if_dev(signal_image, prefix="signal_type")

    signal_prediction, signal_details = model.signal_type_model.predict_with_details(
        signal_image, features
    )
    debug_payload["submodels"]["signal_type"] = signal_details

    if (
        signal_prediction.confidence is None
        or signal_prediction.confidence < model.min_signal_confidence
    ):
        debug_payload["final_prediction"] = {
            "label": "reject",
            "confidence": signal_prediction.confidence,
            "reason": "signal_type_confidence_below_threshold",
        }
        _save_submodel_debug_if_dev(debug_payload)
        return HybridPrediction(label="reject", confidence=signal_prediction.confidence)

    if signal_prediction.label == "cough":
        cough_prediction, cough_details = _evaluate_secondary_submodel(
            model=model.cough_model,
            wav_bytes=wav_bytes,
            features=features,
            target_duration_seconds=COUGH_DURATION_SECONDS,
            prefix="cough",
        )
        debug_payload["submodels"]["cough_classifier"] = cough_details
        debug_payload["final_prediction"] = {
            "label": cough_prediction.label,
            "confidence": cough_prediction.confidence,
            "path": "signal_type->cough_classifier",
        }
        _save_submodel_debug_if_dev(debug_payload)
        return cough_prediction

    if signal_prediction.label == "breath":
        breath_prediction, breath_details = _evaluate_secondary_submodel(
            model=model.breath_model,
            wav_bytes=wav_bytes,
            features=features,
            target_duration_seconds=BREATH_DURATION_SECONDS,
            prefix="breath",
        )
        debug_payload["submodels"]["breath_classifier"] = breath_details
        debug_payload["final_prediction"] = {
            "label": breath_prediction.label,
            "confidence": breath_prediction.confidence,
            "path": "signal_type->breath_classifier",
        }
        _save_submodel_debug_if_dev(debug_payload)
        return breath_prediction

    debug_payload["final_prediction"] = {
        "label": "reject",
        "confidence": signal_prediction.confidence,
        "reason": "unknown_signal_type_label",
    }
    _save_submodel_debug_if_dev(debug_payload)
    return HybridPrediction(label="reject", confidence=signal_prediction.confidence)


def _save_spectrogram_if_dev(image, prefix: str) -> None:
    if not IF_DEV:
        return
    path = save_debug_spectrogram(image, DEV_SPECTROGRAM_DIR, prefix)
    logger.info("Saved debug spectrogram: %s", path)


def _evaluate_secondary_submodel(
    model,
    wav_bytes: bytes,
    features,
    target_duration_seconds: float,
    prefix: str,
) -> tuple[HybridPrediction, dict]:
    image = generate_spectrogram_image_from_wav_bytes(
        wav_bytes=wav_bytes,
        target_duration_seconds=target_duration_seconds,
    )
    _save_spectrogram_if_dev(image, prefix=prefix)
    return model.predict_with_details(image, features)


def _save_submodel_debug_if_dev(payload: dict) -> None:
    if not IF_DEV or not SAVE_SUBMODEL_LOGS:
        return

    DEV_SUBMODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = DEV_SUBMODEL_LOG_DIR / f"submodel_debug_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved submodel debug output: %s", output_path)


classifier = _load_classifier()
