from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioSignal:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples) / self.sample_rate)


def load_audio_bytes(audio_bytes: bytes) -> AudioSignal:
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as exc:
        raise ValueError("Could not decode audio file.") from exc
    if y.size == 0 or sr is None:
        raise ValueError("Audio data is empty.")
    return AudioSignal(samples=y, sample_rate=int(sr))
