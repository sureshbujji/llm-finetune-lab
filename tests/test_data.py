"""Data generation determinism + split integrity."""

import json
import os

from src.make_data import N_EVAL, N_TRAIN, SEED_EVAL, SEED_TRAIN, generate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generate_is_deterministic():
    assert generate(SEED_TRAIN, N_TRAIN) == generate(SEED_TRAIN, N_TRAIN)
    assert generate(SEED_EVAL, N_EVAL) == generate(SEED_EVAL, N_EVAL)


def test_different_seeds_differ():
    assert generate(SEED_TRAIN, N_TRAIN) != generate(SEED_EVAL, N_TRAIN)


def test_counts_and_valid_labels():
    train = generate(SEED_TRAIN, N_TRAIN)
    eval_ = generate(SEED_EVAL, N_EVAL)
    assert len(train) == N_TRAIN == 50
    assert len(eval_) == N_EVAL == 20
    for ex in train + eval_:
        assert ex["severity"] in ("Critical", "High", "Medium", "Low")
        assert ex["title"] and ex["description"]


def test_label_balance():
    from collections import Counter

    counts = Counter(ex["severity"] for ex in generate(SEED_TRAIN, N_TRAIN))
    assert set(counts) == {"Critical", "High", "Medium", "Low"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_no_train_eval_overlap():
    train = generate(SEED_TRAIN, N_TRAIN)
    train_keys = {(e["title"], e["description"]) for e in train}
    eval_ = generate(SEED_EVAL, N_EVAL, exclude=train_keys)
    for e in eval_:
        assert (e["title"], e["description"]) not in train_keys


def test_committed_files_match_generator():
    """data/*.jsonl on disk must equal what the generator produces."""
    train = generate(SEED_TRAIN, N_TRAIN)
    train_keys = {(e["title"], e["description"]) for e in train}
    assert _load("train.jsonl") == train
    assert _load("eval.jsonl") == generate(SEED_EVAL, N_EVAL, exclude=train_keys)
