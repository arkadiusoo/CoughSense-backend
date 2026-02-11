#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Keep matplotlib cache inside project to avoid permission issues on some setups.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(".tmp/matplotlib")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render confusion matrices from best model metrics JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ml_2/best_models_metrics"),
        help="Directory with JSON metrics files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml_2/results/confusion_matrices"),
        help="Directory where chart images will be saved.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Which split to render (default: val).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output file format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Image DPI for raster outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    return parser.parse_args()


def load_matrix(file_path: Path, split: str) -> tuple[str, np.ndarray, list[str]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", {})
    split_metrics = metrics.get(split, {})
    confusion = split_metrics.get("confusion_matrix", {})
    labels = confusion.get("labels")
    matrix = confusion.get("matrix")

    if not labels or matrix is None:
        raise ValueError(f"Missing confusion matrix in {file_path}")

    np_matrix = np.array(matrix, dtype=np.int64)
    if np_matrix.ndim != 2:
        raise ValueError(f"Invalid matrix shape in {file_path}: {np_matrix.shape}")

    # Stored matrix is usually [actual x predicted]. We transpose to show:
    # rows -> predicted, columns -> actual.
    return file_path.stem, np_matrix.T, labels


def figure_size(class_count: int) -> tuple[float, float]:
    side = max(6.0, min(11.0, 1.25 * class_count + 2.0))
    return side, side


def render_matrix(
    title: str,
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    split: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=figure_size(len(labels)))
    image = ax.imshow(matrix, cmap="Reds", interpolation="nearest")

    ax.set_title(f"{title} ({split})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual value", fontsize=11)
    ax.set_ylabel("Predicted value", fontsize=11)

    indices = np.arange(len(labels))
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value > 0 else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Samples", rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {args.input_dir}")

    generated = []
    for file_path in files:
        stem, matrix, labels = load_matrix(file_path, args.split)
        output_path = args.output_dir / f"{stem}_{args.split}.{args.format}"
        render_matrix(
            title=stem.replace("_", " "),
            matrix=matrix,
            labels=labels,
            output_path=output_path,
            split=args.split,
            dpi=args.dpi,
        )
        generated.append(output_path)
        print(f"[OK] {output_path}")

    if args.show:
        for image_path in generated:
            image = plt.imread(image_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.title(image_path.name)
        plt.show()


if __name__ == "__main__":
    main()
