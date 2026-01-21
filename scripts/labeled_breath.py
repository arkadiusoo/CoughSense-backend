#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

CSV_PATH = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/Respiratory_Sound_Database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
)
AUDIO_DIR = Path(
"/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/Respiratory_Sound_Database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
)
DEST_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/labeled_breath"
)


def load_diagnosis_map(csv_path: Path) -> tuple[dict[str, str], int]:
    diagnosis: dict[str, str] = {}
    conflicts = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            patient_id = row[0].strip()
            status = row[1].strip()
            if not patient_id or not status:
                continue
            if patient_id in diagnosis and diagnosis[patient_id] != status:
                conflicts += 1
                continue
            diagnosis[patient_id] = status
    return diagnosis, conflicts


def patient_id_from_name(filename: str) -> str | None:
    match = re.match(r"^(\d+)", filename)
    if not match:
        return None
    return match.group(1)


def safe_status_for_filename(status: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", status.strip())
    return cleaned.strip("-") or "unknown"


def next_index(dest: Path, patient_id: str, status_slug: str) -> int:
    pattern = re.compile(
        rf"^breath_{re.escape(patient_id)}_{re.escape(status_slug)}_(\d+)\.wav$"
    )
    max_idx = 0
    for wav in dest.glob(f"breath_{patient_id}_{status_slug}_*.wav"):
        match = pattern.match(wav.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Missing audio directory: {AUDIO_DIR}")

    diagnosis, conflicts = load_diagnosis_map(CSV_PATH)
    print("Mapa pacjent -> choroba:")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2, sort_keys=True))
    if conflicts:
        print(f"Wykryto konflikty w CSV: {conflicts}", file=sys.stderr)

    answer = input("Czy mapa jest poprawna i mogę kontynuować kopiowanie? [t/N] ").strip()
    if answer.lower() not in {"t", "tak", "y", "yes"}:
        print("Przerwano na życzenie użytkownika.")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    counters: dict[tuple[str, str], int] = {}
    copied = 0
    skipped = 0
    missing = 0

    for wav_path in sorted(AUDIO_DIR.glob("*.wav")):
        patient_id = patient_id_from_name(wav_path.name)
        if not patient_id:
            skipped += 1
            continue
        status = diagnosis.get(patient_id)
        if not status:
            missing += 1
            continue

        status_slug = safe_status_for_filename(status)
        key = (patient_id, status_slug)
        if key not in counters:
            counters[key] = next_index(DEST_DIR, patient_id, status_slug)

        idx = counters[key]
        stem = f"breath_{patient_id}_{status_slug}_{idx}"
        dst_wav = DEST_DIR / f"{stem}.wav"
        dst_json = DEST_DIR / f"{stem}.json"

        shutil.copy2(wav_path, dst_wav)
        with dst_json.open("w", encoding="utf-8") as handle:
            json.dump({"status": status}, handle, ensure_ascii=False)

        counters[key] = idx + 1
        copied += 1

    print(
        "Gotowe. Skopiowano: "
        f"{copied}. Pominięto bez ID: {skipped}. Brak w mapie: {missing}."
    )


if __name__ == "__main__":
    main()
