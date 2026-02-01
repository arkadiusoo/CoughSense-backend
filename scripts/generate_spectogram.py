#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
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
TEST_OUTPUT_DIR = Path(
    "/Volumes/SSD500GB/1.ssd_files/dane_do_inzynierki/training_data/test_spectograms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mel spectrograms for WAV files in training folders."
    )
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Process only 10 files per directory.",
    )
    parser.add_argument(
        "-random",
        "--random",
        action="store_true",
        help="Use random 10 files per directory (requires --test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    return parser.parse_args()


def find_wavs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


def select_files(files: list[Path], use_test: bool, use_random: bool, seed: int) -> list[Path]:
    if not use_test:
        return files
    limit = min(10, len(files))
    if use_random:
        rng = random.Random(seed)
        return sorted(rng.sample(files, k=limit))
    return files[:limit]


def compute_mel_spectrogram(
    wav_path: Path, n_mels: int, n_fft: int, hop_length: int
) -> np.ndarray:
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_spectrogram(array: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Saving PNG spectrograms requires matplotlib to be installed."
        ) from exc

    normalized = array - array.min()
    if normalized.max() > 0:
        normalized = normalized / normalized.max()
    normalized = np.flipud(normalized)
    plt.imsave(out_path, normalized, cmap="magma")


def main() -> None:
    args = parse_args()
    if args.random and not args.test:
        raise SystemExit("--random requires --test")

    total_created = 0
    if args.test:
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for root in DATA_DIRS:
        if not root.exists():
            raise FileNotFoundError(f"Missing data directory: {root}")

        wavs = find_wavs(root)
        selected = select_files(wavs, args.test, args.random, args.seed)

        created = 0
        for wav_path in selected:
            spectrogram = compute_mel_spectrogram(
                wav_path,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )
            if args.test:
                out_path = TEST_OUTPUT_DIR / wav_path.with_suffix(".png").name
            else:
                out_path = wav_path.with_suffix(".png")
            save_spectrogram(spectrogram, out_path)
            created += 1

        total_created += created
        if args.test:
            print(f"{root}: created {created} test spectrograms from {len(selected)} files.")
        else:
            print(f"{root}: created {created} spectrograms from {len(selected)} files.")

    if args.test:
        print(f"Done. Test spectrograms saved to: {TEST_OUTPUT_DIR}")
    else:
        print(f"Done. Created {total_created} spectrograms.")


if __name__ == "__main__":
    main()
