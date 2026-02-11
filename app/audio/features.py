import json
from datetime import datetime, timezone
from typing import Iterable

import librosa
import numpy as np

from app.ifDev import DEV_FEATURES_DIR, IF_DEV, SAVE_FEATURE_BLOCKS
from app.audio.io import AudioSignal


def extract_features(signal: AudioSignal) -> np.ndarray:
    y = signal.samples
    sr = signal.sample_rate

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    mfcc_summary = _summarize_feature(mfcc)
    chroma_summary = _summarize_feature(chroma)
    spectral_centroid_summary = _summarize_feature(spectral_centroid)
    spectral_bandwidth_summary = _summarize_feature(spectral_bandwidth)
    spectral_rolloff_summary = _summarize_feature(spectral_rolloff)
    zcr_summary = _summarize_feature(zcr)

    feature_blocks: Iterable[np.ndarray] = (
        mfcc_summary,
        chroma_summary,
        spectral_centroid_summary,
        spectral_bandwidth_summary,
        spectral_rolloff_summary,
        zcr_summary,
    )

    vector = np.concatenate(list(feature_blocks)).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    if IF_DEV and SAVE_FEATURE_BLOCKS:
        _save_feature_debug_payload(
            signal=signal,
            mfcc=mfcc,
            chroma=chroma,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            zcr=zcr,
            mfcc_summary=mfcc_summary,
            chroma_summary=chroma_summary,
            spectral_centroid_summary=spectral_centroid_summary,
            spectral_bandwidth_summary=spectral_bandwidth_summary,
            spectral_rolloff_summary=spectral_rolloff_summary,
            zcr_summary=zcr_summary,
            vector=vector,
        )
    return vector


def _summarize_feature(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)
    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)
    return np.concatenate([mean, std])


def _save_feature_debug_payload(
    signal: AudioSignal,
    mfcc: np.ndarray,
    chroma: np.ndarray,
    spectral_centroid: np.ndarray,
    spectral_bandwidth: np.ndarray,
    spectral_rolloff: np.ndarray,
    zcr: np.ndarray,
    mfcc_summary: np.ndarray,
    chroma_summary: np.ndarray,
    spectral_centroid_summary: np.ndarray,
    spectral_bandwidth_summary: np.ndarray,
    spectral_rolloff_summary: np.ndarray,
    zcr_summary: np.ndarray,
    vector: np.ndarray,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio": {
            "sample_rate": signal.sample_rate,
            "duration_seconds": signal.duration_seconds,
            "num_samples": int(signal.samples.size),
        },
        "raw_features": {
            "mfcc": _array_to_list(mfcc),
            "chroma": _array_to_list(chroma),
            "spectral_centroid": _array_to_list(spectral_centroid),
            "spectral_bandwidth": _array_to_list(spectral_bandwidth),
            "spectral_rolloff": _array_to_list(spectral_rolloff),
            "zcr": _array_to_list(zcr),
        },
        "summaries": {
            "mfcc": _split_summary(mfcc_summary, channels=13),
            "chroma": _split_summary(chroma_summary, channels=12),
            "spectral_centroid": _split_summary(spectral_centroid_summary, channels=1),
            "spectral_bandwidth": _split_summary(spectral_bandwidth_summary, channels=1),
            "spectral_rolloff": _split_summary(spectral_rolloff_summary, channels=1),
            "zcr": _split_summary(zcr_summary, channels=1),
        },
        "feature_vector": {
            "size": int(vector.size),
            "values": _array_to_list(vector),
            "layout": [
                "mfcc_mean[13]",
                "mfcc_std[13]",
                "chroma_mean[12]",
                "chroma_std[12]",
                "spectral_centroid_mean[1]",
                "spectral_centroid_std[1]",
                "spectral_bandwidth_mean[1]",
                "spectral_bandwidth_std[1]",
                "spectral_rolloff_mean[1]",
                "spectral_rolloff_std[1]",
                "zcr_mean[1]",
                "zcr_std[1]",
            ],
        },
    }

    DEV_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = DEV_FEATURES_DIR / f"features_debug_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _array_to_list(array: np.ndarray) -> list:
    sanitized = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized.tolist()


def _split_summary(summary: np.ndarray, channels: int) -> dict[str, list]:
    return {
        "mean": _array_to_list(summary[:channels]),
        "std": _array_to_list(summary[channels:]),
    }
