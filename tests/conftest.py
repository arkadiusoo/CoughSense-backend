from pathlib import Path
import shutil
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.audio.io import load_audio_bytes


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def audio_wav_path() -> Path:
    return TEST_DATA_DIR / "sample.wav"


@pytest.fixture(scope="session")
def audio_mp3_path() -> Path:
    return TEST_DATA_DIR / "sample.mp3"


@pytest.fixture(scope="session")
def audio_wav_bytes(audio_wav_path: Path) -> bytes:
    return audio_wav_path.read_bytes()


@pytest.fixture(scope="session")
def audio_mp3_bytes(audio_mp3_path: Path) -> bytes:
    return audio_mp3_path.read_bytes()


@pytest.fixture(scope="session")
def audio_bytes(audio_wav_bytes: bytes) -> bytes:
    return audio_wav_bytes


@pytest.fixture(scope="session")
def audio_signal(audio_wav_bytes: bytes):
    return load_audio_bytes(audio_wav_bytes)


@pytest.fixture(scope="session")
def require_ffmpeg_binary():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available")
