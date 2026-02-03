import time
from pathlib import Path
import logging

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
from app.ifDev import IF_DEV, DEV_SPECTROGRAM_DIR
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
    # Stage 1: generate spectrogram for signal_type model (20s, as in training script).
    signal_image = generate_spectrogram_image_from_wav_bytes(
        wav_bytes=wav_bytes,
        target_duration_seconds=SIGNAL_TYPE_DURATION_SECONDS,
    )
    _save_spectrogram_if_dev(signal_image, prefix="signal_type")

    signal_prediction = model.signal_type_model.predict(signal_image, features)
    if (
        signal_prediction.confidence is None
        or signal_prediction.confidence < model.min_signal_confidence
    ):
        return HybridPrediction(label="reject", confidence=signal_prediction.confidence)

    if signal_prediction.label == "cough":
        # Stage 2a: if cough, generate cough spectrogram (10s, as in training script).
        cough_image = generate_spectrogram_image_from_wav_bytes(
            wav_bytes=wav_bytes,
            target_duration_seconds=COUGH_DURATION_SECONDS,
        )
        _save_spectrogram_if_dev(cough_image, prefix="cough")
        return model.cough_model.predict(cough_image, features)

    if signal_prediction.label == "breath":
        # Stage 2b: if breath, generate breath spectrogram (20s, as in training script).
        breath_image = generate_spectrogram_image_from_wav_bytes(
            wav_bytes=wav_bytes,
            target_duration_seconds=BREATH_DURATION_SECONDS,
        )
        _save_spectrogram_if_dev(breath_image, prefix="breath")
        return model.breath_model.predict(breath_image, features)

    return HybridPrediction(label="reject", confidence=signal_prediction.confidence)


def _save_spectrogram_if_dev(image, prefix: str) -> None:
    if not IF_DEV:
        return
    path = save_debug_spectrogram(image, DEV_SPECTROGRAM_DIR, prefix)
    logger.info("Saved debug spectrogram: %s", path)


classifier = _load_classifier()
