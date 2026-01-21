#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

COUGH_SRC = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/COUGHVID_dataset"
)
BREATH_SRC = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/Respiratory_Sound_Database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
)
DEST_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/coughVSbreath"
)


def find_wavs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    )


def next_index(dest: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.wav$")
    max_idx = 0
    for wav in dest.glob(f"{prefix}_*.wav"):
        match = pattern.match(wav.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def copy_with_label(files: list[Path], prefix: str, label: str, dest: Path) -> int:
    idx = next_index(dest, prefix)
    for src in files:
        stem = f"{prefix}_{idx}"
        dst_wav = dest / f"{stem}.wav"
        dst_json = dest / f"{stem}.json"
        shutil.copy2(src, dst_wav)
        with dst_json.open("w", encoding="utf-8") as handle:
            json.dump({"type": label}, handle)
        idx += 1
    return idx


def main() -> None:
    if not COUGH_SRC.exists():
        raise FileNotFoundError(f"Missing cough source: {COUGH_SRC}")
    if not BREATH_SRC.exists():
        raise FileNotFoundError(f"Missing breath source: {BREATH_SRC}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    cough_files = find_wavs(COUGH_SRC)
    breath_files = find_wavs(BREATH_SRC)

    copy_with_label(cough_files, "cough", "cough", DEST_DIR)
    copy_with_label(breath_files, "breath", "breath", DEST_DIR)

    print(f"Copied {len(cough_files)} cough files and {len(breath_files)} breath files.")


if __name__ == "__main__":
    main()
