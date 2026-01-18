from typing import Iterable

import librosa
import numpy as np

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
