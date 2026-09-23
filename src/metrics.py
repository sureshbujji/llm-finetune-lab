"""Classification metrics. Pure functions, no torch dependency."""

from __future__ import annotations


def accuracy(y_true: list[str], y_pred: list[str | None]) -> float:
    """Fraction correct. A None prediction (unparseable output) counts as wrong."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
    return correct / len(y_true)


def per_class_prf(
    y_true: list[str], y_pred: list[str | None], labels: list[str]
) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 per class. None predictions hurt recall only.

    Zero-division yields 0.0 (a class never predicted has precision 0.0;
    a class never correctly predicted has recall 0.0).
    """
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return result


def confusion_pairs(
    y_true: list[str], y_pred: list[str | None]
) -> list[tuple[str, str | None]]:
    """(true, predicted) pairs for error analysis, wrong ones first."""
    pairs = list(zip(y_true, y_pred))
    return sorted(pairs, key=lambda tp: tp[0] == tp[1])
