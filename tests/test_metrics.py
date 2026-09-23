"""Eval metric math: accuracy and per-class precision/recall/F1."""

import pytest

from src.metrics import accuracy, confusion_pairs, per_class_prf

LABELS = ["Critical", "High", "Medium", "Low"]


def test_accuracy_basic():
    y_true = ["Critical", "High", "Low", "Medium"]
    y_pred = ["Critical", "Low", "Low", None]  # None = unparseable -> wrong
    assert accuracy(y_true, y_pred) == 0.5


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_per_class_prf_hand_computed():
    y_true = ["Critical", "Critical", "High", "Low"]
    y_pred = ["Critical", "High", "High", "Low"]
    prf = per_class_prf(y_true, y_pred, LABELS)
    # Critical: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=2/3
    assert prf["Critical"]["precision"] == 1.0
    assert prf["Critical"]["recall"] == 0.5
    assert prf["Critical"]["f1"] == pytest.approx(2 / 3)
    assert prf["Critical"]["support"] == 2.0
    # High: tp=1, fp=1, fn=0 -> P=0.5, R=1.0
    assert prf["High"]["precision"] == 0.5
    assert prf["High"]["recall"] == 1.0
    # Medium: never appears -> all zeros, no ZeroDivisionError
    assert prf["Medium"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}


def test_none_predictions_hurt_recall_not_precision():
    y_true = ["Critical", "Critical"]
    y_pred = ["Critical", None]
    prf = per_class_prf(y_true, y_pred, LABELS)
    assert prf["Critical"]["precision"] == 1.0  # no false positives
    assert prf["Critical"]["recall"] == 0.5


def test_confusion_pairs_wrong_first():
    pairs = confusion_pairs(["a", "b"], ["a", "c"])
    assert pairs[0] == ("b", "c")
    assert pairs[1] == ("a", "a")
