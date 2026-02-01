#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

COUGH_SRC = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/COUGHVID_dataset"
)
DEST_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_cough"
)
LABEL_FIELDS = ("expert_labels_1", "expert_labels_2", "expert_labels_3")
TARGET_DIAGNOSES = {"COVID-19", "healthy_cough"}


def find_wav_json_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for wav in root.rglob("*.wav"):
        if not wav.is_file():
            continue
        json_path = wav.with_suffix(".json")
        if json_path.is_file():
            pairs.append((wav, json_path))
    return sorted(pairs, key=lambda item: item[0].name)


def safe_label_for_filename(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", label.strip())
    return cleaned.strip("-") or "unknown"


def next_index(dest: Path, status_slug: str) -> int:
    pattern = re.compile(rf"^cough_{re.escape(status_slug)}_(\d+)\.wav$")
    max_idx = 0
    for wav in dest.glob(f"cough_{status_slug}_*.wav"):
        match = pattern.match(wav.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def extract_diagnoses(value: object) -> set[str]:
    diagnoses: set[str] = set()
    if isinstance(value, dict):
        diagnosis = value.get("diagnosis")
        if diagnosis is not None:
            diagnoses.add(str(diagnosis))
        return diagnoses

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "diagnosis" in item:
                diagnoses.add(str(item["diagnosis"]))
        return diagnoses

    return diagnoses


def pick_target_diagnosis(data: dict) -> str | None:
    for label in LABEL_FIELDS:
        if label not in data:
            continue
        diagnoses = extract_diagnoses(data[label])
        matched = diagnoses & TARGET_DIAGNOSES
        if matched:
            return sorted(matched)[0]
    return None


def main() -> None:
    if not COUGH_SRC.exists():
        raise FileNotFoundError(f"Missing source: {COUGH_SRC}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    counters: dict[str, int] = {}

    for wav_path, json_path in find_wav_json_pairs(COUGH_SRC):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped += 1
            continue

        if not isinstance(data, dict):
            skipped += 1
            continue

        diagnosis_value = pick_target_diagnosis(data)
        if diagnosis_value is None:
            skipped += 1
            continue

        diagnosis_slug = safe_label_for_filename(diagnosis_value)
        if diagnosis_slug not in counters:
            counters[diagnosis_slug] = next_index(DEST_DIR, diagnosis_slug)
        index = counters[diagnosis_slug]
        stem = f"cough_{diagnosis_slug}_{index}"
        dst_wav = DEST_DIR / f"{stem}.wav"
        dst_json = DEST_DIR / f"{stem}.json"

        shutil.copy2(wav_path, dst_wav)
        with dst_json.open("w", encoding="utf-8") as handle:
            json.dump({"diagnosis": diagnosis_value}, handle)

        counters[diagnosis_slug] = index + 1
        copied += 1

    diagnoses = ", ".join(sorted(TARGET_DIAGNOSES))
    print(
        "Copied "
        f"{copied} pairs with expert diagnosis in [{diagnoses}]. "
        f"Skipped {skipped} pairs."
    )


if __name__ == "__main__":
    main()
