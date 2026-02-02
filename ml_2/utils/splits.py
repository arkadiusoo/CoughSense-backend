from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split_three(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitIndices:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(labels_array):
        class_indices = np.where(labels_array == class_id)[0]
        rng.shuffle(class_indices)
        count = len(class_indices)
        if count == 0:
            continue

        desired_test = int(round(count * test_ratio))
        test_count = _bounded_split_size(desired_test, count)

        remaining = count - test_count
        if remaining <= 1:
            val_count = 0
        else:
            desired_val = int(round(count * val_ratio))
            val_count = min(max(1, desired_val), remaining - 1)

        test_indices.extend(class_indices[:test_count].tolist())
        val_indices.extend(class_indices[test_count : test_count + val_count].tolist())
        train_indices.extend(class_indices[test_count + val_count :].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return SplitIndices(
        train=np.asarray(train_indices),
        val=np.asarray(val_indices),
        test=np.asarray(test_indices),
    )


def _bounded_split_size(desired: int, total: int) -> int:
    if total <= 1:
        return 0
    return min(max(1, desired), total - 1)
