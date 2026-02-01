#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/COUGHVID_dataset"
)
DEFAULT_LABELS = ("expert_labels_1", "expert_labels_2", "expert_labels_3")
DEFAULT_DIAGNOSES = ("COVID-19", "healthy_cough", "healthy_coughm")


def iter_json_files(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.json") if path.is_file()])


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count JSON files that contain expert_labels_1/2/3 with "
            "diagnosis values matching a target list."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root directory to scan for JSON files.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=list(DEFAULT_LABELS),
        help="Expert label field to inspect (repeatable).",
    )
    parser.add_argument(
        "--diagnosis",
        action="append",
        dest="diagnoses",
        default=list(DEFAULT_DIAGNOSES),
        help="Diagnosis value to match (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    labels = tuple(args.labels)
    target = {str(item) for item in args.diagnoses}

    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    json_files = iter_json_files(root)
    total_files = len(json_files)
    parsed_files = 0
    invalid_json = 0

    matched_files = 0
    per_label_counts = Counter({label: 0 for label in labels})
    per_diagnosis_counts: Counter[str] = Counter({diag: 0 for diag in target})

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_json += 1
            continue

        parsed_files += 1
        if not isinstance(data, dict):
            continue

        file_matched_any = False
        file_diag_matches: set[str] = set()

        for label in labels:
            if label not in data:
                continue
            diagnoses = extract_diagnoses(data[label])
            matched = diagnoses & target
            if matched:
                per_label_counts[label] += 1
                file_matched_any = True
                file_diag_matches |= matched

        if file_matched_any:
            matched_files += 1
            for diag in file_diag_matches:
                per_diagnosis_counts[diag] += 1

    print("Expert label diagnosis summary")
    print(f"Dataset root: {root}")
    print(f"JSON files: {total_files}")
    print(f"Parsed JSON files: {parsed_files}")
    print(f"Invalid JSON files: {invalid_json}")
    print(f"Target diagnoses: {', '.join(sorted(target))}")
    print(f"Matched files (any label): {matched_files}")
    print("Matched files per label:")
    for label in labels:
        print(f"  {label}: {per_label_counts[label]}")
    print("Matched files per diagnosis (any label):")
    for diag in sorted(target):
        print(f"  {diag}: {per_diagnosis_counts[diag]}")


if __name__ == "__main__":
    main()
