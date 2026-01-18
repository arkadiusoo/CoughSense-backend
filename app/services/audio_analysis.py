from io import BytesIO
from typing import Iterable

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.db.repository import AnalysisRepository
from app.ml.model import CoughClassifier
from app.services.audio_conversion import convert_to_wav


ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm"}
MIN_DURATION_SECONDS = 0.2
MAX_DURATION_SECONDS = 15.0


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

    wav_bytes = convert_to_wav(
        audio_bytes,
        content_type=file.content_type or "",
        filename=file.filename or "",
    )
    y, sr = _load_audio(wav_bytes)
    duration_seconds = float(len(y) / sr)
    _validate_duration(duration_seconds)

    features = extract_features(y, sr)
    prediction = classifier.predict(features)
    repository.save_analysis(
        duration_seconds=duration_seconds,
        prediction=prediction,
        feature_vector_size=int(features.shape[0]),
    )
    print(features)
    return {
        "prediction": {
            "label": prediction.label,
            "confidence": prediction.confidence,
        },
        "duration_seconds": duration_seconds,
        "feature_vector_size": int(features.shape[0]),
    }


def _validate_upload(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    base_content_type = content_type.split(";", 1)[0].strip()
    if base_content_type and base_content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio format.",
        )

    filename = (file.filename or "").lower()
    if filename and not _has_allowed_extension(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension.",
        )


def _has_allowed_extension(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode audio file.",
        ) from exc
    if y.size == 0 or sr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio data is empty.",
        )
    return y, int(sr)


def _validate_duration(duration_seconds: float) -> None:
    if duration_seconds < MIN_DURATION_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio recording is too short.",
        )
    if duration_seconds > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio recording is too long.",
        )


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    feature_blocks: Iterable[np.ndarray] = (
        _summarize_feature(mfcc),
        _summarize_feature(chroma),
        _summarize_feature(spectral_centroid),
        _summarize_feature(spectral_bandwidth),
        _summarize_feature(spectral_rolloff),
        _summarize_feature(zcr),
    )

    vector = np.concatenate(list(feature_blocks)).astype(np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _summarize_feature(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)
    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)
    return np.concatenate([mean, std])
