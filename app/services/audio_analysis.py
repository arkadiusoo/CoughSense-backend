import time

from fastapi import HTTPException, UploadFile, status

from app.audio.conversion import convert_to_wav
from app.audio.features import extract_features
from app.audio.io import load_audio_bytes
from app.audio.validation import validate_duration, validate_format
from app.db.repository import AnalysisRepository
from app.ml.model import CoughClassifier


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


classifier = CoughClassifier()
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
    prediction = classifier.predict(features)
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
            "confidence": prediction.confidence,
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
