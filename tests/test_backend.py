from io import BytesIO
import wave

import numpy as np

from app.audio.conversion import convert_to_wav
from app.audio.features import extract_features
from app.audio.io import load_audio_bytes


def test_tc_be_01_convert_to_wav(
    audio_mp3_bytes: bytes,
    audio_wav_bytes: bytes,
    tmp_path,
):
    wav_bytes = convert_to_wav(audio_mp3_bytes, "audio/mpeg", "sample.mp3")

    assert wav_bytes
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    output_path = tmp_path / "output.wav"
    output_path.write_bytes(wav_bytes)
    assert output_path.exists()

    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() > 0
        assert wav_file.getframerate() > 0

    converted_signal = load_audio_bytes(wav_bytes)
    expected_signal = load_audio_bytes(audio_wav_bytes)

    assert converted_signal.sample_rate == expected_signal.sample_rate

    min_len = min(len(converted_signal.samples), len(expected_signal.samples))
    assert min_len > 0

    converted_samples = converted_signal.samples[:min_len]
    expected_samples = expected_signal.samples[:min_len]

    diff = converted_samples - expected_samples
    rms_diff = float(np.sqrt(np.mean(diff**2)))
    rms_ref = float(np.sqrt(np.mean(expected_samples**2)))
    assert rms_diff / max(rms_ref, 1e-8) < 0.1


def test_tc_be_02_extract_features(audio_signal):
    vector = extract_features(audio_signal)

    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.dtype == np.float32
    assert vector.size > 0
