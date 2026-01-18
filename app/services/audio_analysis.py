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
    repository.save_analysis(
        duration_seconds=signal.duration_seconds,
        prediction=prediction,
        feature_vector_size=int(features.shape[0]),
    )
    return {
        "prediction": {
            "label": prediction.label,
            "confidence": prediction.confidence,
        },
        "duration_seconds": signal.duration_seconds,
        "feature_vector_size": int(features.shape[0]),
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
