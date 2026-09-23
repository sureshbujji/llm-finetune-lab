"""parse_prediction: earliest word-boundary label match."""

from src.eval_before_after import parse_prediction


def test_parses_single_label():
    assert parse_prediction("critical") == "Critical"
    assert parse_prediction("the answer is high") == "High"


def test_earliest_in_text_order_wins():
    # 'low' appears first in the text even though 'High' comes first in LABELS.
    assert parse_prediction("low severity, not high") == "Low"


def test_word_boundaries():
    assert parse_prediction("slow load times") is None  # 'low' inside 'slow'
    assert parse_prediction("highly unlikely") is None


def test_no_label_returns_none():
    assert parse_prediction("::::::::::") is None
    assert parse_prediction("") is None


def test_case_insensitive():
    assert parse_prediction("MEDIUM") == "Medium"
