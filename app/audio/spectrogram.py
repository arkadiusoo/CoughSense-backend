from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
from PIL import Image

from app.audio.io import AudioSignal


DEFAULT_SR = 16000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def generate_mel_spectrogram(
    signal: AudioSignal,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    target_sample_rate: int = DEFAULT_SR,
    target_duration_seconds: float | None = None,
) -> np.ndarray:
    samples = signal.samples.astype(np.float32)
    sample_rate = int(signal.sample_rate)
    if sample_rate != target_sample_rate:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=target_sample_rate
        )
        sample_rate = target_sample_rate
    return _compute_mel_spectrogram(
        audio=samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_duration_seconds=target_duration_seconds,
    )


def generate_spectrogram_image_from_wav_bytes(
    wav_bytes: bytes,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    target_sample_rate: int = DEFAULT_SR,
    target_duration_seconds: float | None = None,
) -> Image.Image:
    # Match training script behavior: decode with target SR directly, then pad/trim.
    audio, sample_rate = librosa.load(
        BytesIO(wav_bytes),
        sr=target_sample_rate,
        mono=True,
    )
    mel = _compute_mel_spectrogram(
        audio=audio.astype(np.float32),
        sample_rate=int(sample_rate),
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_duration_seconds=target_duration_seconds,
    )
    return spectrogram_to_image(mel)


def spectrogram_to_image(spectrogram: np.ndarray) -> Image.Image:
    normalized = spectrogram - float(np.min(spectrogram))
    max_value = float(np.max(normalized))
    if max_value > 0.0:
        normalized = normalized / max_value
    normalized = np.flipud(normalized)
    try:
        # Use the same rendering path as scripts/generate_spectogram.py.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Generating spectrogram image requires matplotlib to be installed."
        ) from exc

    buffer = BytesIO()
    plt.imsave(buffer, normalized, cmap="magma", format="png")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def generate_spectrogram_image(
    signal: AudioSignal,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    target_sample_rate: int = DEFAULT_SR,
    target_duration_seconds: float | None = None,
) -> Image.Image:
    mel = generate_mel_spectrogram(
        signal=signal,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_sample_rate=target_sample_rate,
        target_duration_seconds=target_duration_seconds,
    )
    return spectrogram_to_image(mel)


def save_debug_spectrogram(
    image: Image.Image,
    output_dir: Path,
    prefix: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{prefix}_{timestamp}.png"
    image.save(output_path)
    return output_path


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.size == target_len:
        return audio
    if audio.size > target_len:
        return audio[:target_len]
    pad_width = target_len - audio.size
    return np.pad(audio, (0, pad_width), mode="constant")


def _compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_duration_seconds: float | None,
) -> np.ndarray:
    if target_duration_seconds is not None:
        target_len = int(round(sample_rate * target_duration_seconds))
        audio = _pad_or_trim(audio, target_len)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
