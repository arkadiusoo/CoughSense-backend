#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

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


def find_wavs(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.wav") if path.is_file()])


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def print_stats(root: Path, durations: np.ndarray, failures: int) -> None:
    if durations.size == 0:
        print(f"{root}")
        print("  files: 0")
        print(f"  files_with_errors: {failures}")
        print("  mean: n/a")
        print("  median: n/a")
        print("  min: n/a")
        print("  max: n/a")
        print("  std_dev: n/a")
        print("  p25: n/a")
        print("  p75: n/a")
        print("  p90: n/a")
        print("  p95: n/a")
        print("  total_duration_hours: 0.00h")
        return

    mean = float(durations.mean())
    median = float(np.median(durations))
    minimum = float(durations.min())
    maximum = float(durations.max())
    std_dev = float(durations.std())
    p25 = float(np.percentile(durations, 25))
    p75 = float(np.percentile(durations, 75))
    p90 = float(np.percentile(durations, 90))
    p95 = float(np.percentile(durations, 95))
    total_hours = float(durations.sum() / 3600.0)

    print(f"{root}")
    print(f"  files: {durations.size}")
    print(f"  files_with_errors: {failures}")
    print(f"  mean: {format_seconds(mean)}")
    print(f"  median: {format_seconds(median)}")
    print(f"  min: {format_seconds(minimum)}")
    print(f"  max: {format_seconds(maximum)}")
    print(f"  std_dev: {format_seconds(std_dev)}")
    print(f"  p25: {format_seconds(p25)}")
    print(f"  p75: {format_seconds(p75)}")
    print(f"  p90: {format_seconds(p90)}")
    print(f"  p95: {format_seconds(p95)}")
    print(f"  total_duration_hours: {total_hours:.2f}h")


def main() -> None:
    for root in DATA_DIRS:
        if not root.exists():
            raise FileNotFoundError(f"Missing data directory: {root}")

        durations: list[float] = []
        failures = 0
        for wav_path in find_wavs(root):
            try:
                duration = float(librosa.get_duration(path=str(wav_path)))
            except Exception:
                failures += 1
                continue
            durations.append(duration)

        print_stats(root, np.array(durations, dtype=float), failures)


if __name__ == "__main__":
    main()
