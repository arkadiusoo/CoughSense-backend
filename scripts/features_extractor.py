#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audio.features import extract_features
from app.audio.io import load_audio_bytes

DATA_DIRS = [
    Path(
        "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/coughVSbreath"
    ),
    Path(
        "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_breath"
    ),
    Path(
        "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_cough"
    ),
]

OUTPUT_SUFFIX = ".features.json"


def find_wavs(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.wav") if path.is_file()])


def write_features(wav_path: Path, out_path: Path) -> None:
    signal = load_audio_bytes(wav_path.read_bytes())
    vector = extract_features(signal)
    payload = {
        "features": vector.tolist(),
        "sample_rate": signal.sample_rate,
        "duration_seconds": signal.duration_seconds,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    total_written = 0
    for root in DATA_DIRS:
        if not root.exists():
            raise FileNotFoundError(f"Missing data directory: {root}")

        written = 0
        failures = 0
        for wav_path in find_wavs(root):
            out_path = wav_path.with_suffix(OUTPUT_SUFFIX)
            try:
                write_features(wav_path, out_path)
            except Exception:
                failures += 1
                continue
            written += 1

        total_written += written
        print(f"{root}: created {written} feature files, errors: {failures}.")

    print(f"Done. Created {total_written} feature files.")


if __name__ == "__main__":
    main()
