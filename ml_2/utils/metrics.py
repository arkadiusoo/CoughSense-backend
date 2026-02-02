from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float | list[float] | list[list[int]]]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fp) != 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fn) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(precision + recall) != 0,
    )

    accuracy = float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()) if num_classes else 0.0,
        "recall_macro": float(recall.mean()) if num_classes else 0.0,
        "f1_macro": float(f1.mean()) if num_classes else 0.0,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def metrics_from_probabilities(
    y_true: np.ndarray,
    probas: np.ndarray,
    num_classes: int,
) -> dict[str, float | list[float] | list[list[int]]]:
    y_pred = np.argmax(probas, axis=1)
    return classification_metrics(y_true, y_pred, num_classes)
