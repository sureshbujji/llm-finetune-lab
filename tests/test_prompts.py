"""Prompt templating and label masking."""

import pytest

from src.prompts import LABELS, build_prompt, build_training_text, encode_with_mask
from src.tokenizer import SimpleTokenizer


@pytest.fixture()
def tok():
    return SimpleTokenizer.build(
        ["### Bug report Title: crash Description: boom ### Severity"] + LABELS
    )


def test_prompt_contains_fields_and_no_label():
    p = build_prompt("App crashes", "It crashes on launch.")
    assert "App crashes" in p
    assert "It crashes on launch." in p
    # The template header names the label set, but no label is *answered*:
    # nothing but whitespace follows the final colon.
    assert p.rstrip().endswith(":")
    assert "critical" not in p.rsplit(":", 1)[1].lower()


def test_training_text_appends_label():
    t = build_training_text("App crashes", "It crashes on launch.", "Critical")
    assert t.startswith(build_prompt("App crashes", "It crashes on launch."))
    assert t.rstrip().endswith("Critical")


def test_training_text_rejects_bad_label():
    with pytest.raises(ValueError):
        build_training_text("t", "d", "Catastrophic")


def test_encode_with_mask_masks_prompt_only(tok):
    input_ids, labels = encode_with_mask(tok, "App crashes", "It crashes.", "High")
    prompt_ids = tok.encode(build_prompt("App crashes", "It crashes."))
    label_ids = tok.encode("High")
    assert input_ids == prompt_ids + label_ids
    assert labels == [-100] * len(prompt_ids) + label_ids
    # Only label tokens contribute to the loss.
    assert all(v == -100 for v in labels[: len(prompt_ids)])
    assert all(v != -100 for v in labels[len(prompt_ids):])


def test_encode_with_mask_optional_eos(tok):
    input_ids, labels = encode_with_mask(
        tok, "App crashes", "It crashes.", "High", add_eos=True
    )
    assert input_ids[-1] == tok.eos_token_id
    assert labels[-1] == tok.eos_token_id
