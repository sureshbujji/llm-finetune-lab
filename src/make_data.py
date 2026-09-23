"""Generate the synthetic bug-severity dataset.

Deterministic: the same seeds always produce byte-identical JSONL files.
Re-run with `python src/make_data.py` to regenerate data/ from scratch.

Task: given a bug title + description, predict severity
(Critical / High / Medium / Low). Severity-correlated keywords make the
task learnable for a tiny model while staying realistic QA fare.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED_TRAIN = 42
SEED_EVAL = 2026
N_TRAIN = 50
N_EVAL = 20

SEVERITIES = ["Critical", "High", "Medium", "Low"]

# (title, description) templates per severity. Descriptions carry the
# severity signal a model must learn; contexts add benign variation.
_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Critical": [
        ("App crashes on launch",
         "The application crashes immediately on startup for every user after the 3.2 update. No workaround exists."),
        ("Data loss when session refreshes",
         "Unsaved form data is silently discarded when the session refreshes. Users lose hours of work."),
        ("Duplicate charges on payment retry",
         "Retrying a timed-out payment creates a second charge and corrupts the ledger state."),
        ("Auth tokens logged in plaintext",
         "Session tokens are written in plaintext to the debug console, enabling session hijack."),
        ("Sync worker deadlock under load",
         "The background sync worker deadlocks under load and the whole pipeline stalls indefinitely."),
        ("Deploy takes down production API",
         "The deploy script drops the API in all regions for several minutes on every release."),
    ],
    "High": [
        ("SSO login fails for enterprise users",
         "Single sign-on returns a 500 error for roughly half of enterprise accounts."),
        ("CSV export sums the wrong column",
         "The finance export totals the wrong column, so monthly reports are incorrect."),
        ("Search returns no results after upgrade",
         "Queries that worked last week now return empty result sets."),
        ("Order created twice on timeout",
         "A slow card authorization creates the order twice, double-charging the customer."),
        ("Push notifications silently dropped",
         "Android push notifications never arrive; no error is logged anywhere."),
        ("Monthly report job runs out of memory",
         "The reporting job fails with an out-of-memory error on large accounts."),
    ],
    "Medium": [
        ("Dashboard widgets flicker on refresh",
         "Widgets reload twice and flicker whenever the dashboard auto-refreshes."),
        ("Signup error message names no field",
         "The signup form shows an error without saying which field is invalid."),
        ("Reports page loads very slowly",
         "The reports page takes over twenty seconds to load with default filters."),
        ("Edited records show stale values",
         "After editing a record, the old values stay visible until a manual reload."),
        ("Confirmation dialog renders off-center",
         "The confirmation modal is misaligned on smaller laptop screens."),
        ("Table sort resets on pagination",
         "Sorting a table is lost as soon as you move to the next page."),
    ],
    "Low": [
        ("Typo in welcome email",
         "The onboarding email says 'Welcom' instead of 'Welcome'."),
        ("Submit button uses old brand color",
         "The settings page submit button still uses last year's brand blue."),
        ("Help tooltip truncated on narrow screens",
         "The tooltip on the help icon is cut off on narrow viewports."),
        ("Sidebar mixes capitalization styles",
         "Menu items mix title case and sentence case in the navigation sidebar."),
        ("Uneven card padding on home page",
         "Cards on the home page have extra padding on the right edge."),
        ("Footer copyright year is outdated",
         "The footer still shows last year in the copyright line."),
    ],
}

_CONTEXTS = [
    "Seen on build 4821.",
    "Reported by the QA team during regression.",
    "Reported by a customer on the support portal.",
    "Reproduced on staging with a clean profile.",
    "Seen on both web and mobile.",
    "Steps to reproduce are attached to the ticket.",
]


def generate(
    seed: int, n: int, exclude: set[tuple[str, str]] | None = None
) -> list[dict[str, str]]:
    """Generate n labeled bug reports deterministically from seed.

    `exclude` holds (title, description) pairs that must not be reused —
    used to keep the eval split strictly held-out from train.
    """
    rng = random.Random(seed)
    # Balanced label order, shuffled deterministically.
    severities = [SEVERITIES[i % len(SEVERITIES)] for i in range(n)]
    rng.shuffle(severities)

    examples: list[dict[str, str]] = []
    used: set[tuple[str, str]] = set(exclude or ())
    for severity in severities:
        title, desc = rng.choice(_TEMPLATES[severity])
        context = rng.choice(_CONTEXTS)
        description = f"{desc} {context}"
        # Guarantee unique (title, description) pairs (and no train/eval overlap).
        attempt = 0
        while (title, description) in used:
            context = _CONTEXTS[(_CONTEXTS.index(context) + 1) % len(_CONTEXTS)]
            description = f"{desc} {context}"
            attempt += 1
            if attempt > len(_CONTEXTS):  # pragma: no cover - defensive
                description = f"{desc} {context} (variant {attempt})"
        used.add((title, description))
        examples.append(
            {"title": title, "description": description, "severity": severity}
        )
    return examples


def write_jsonl(path: str, examples: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


from src.prompts import build_training_text
from src.tokenizer import SimpleTokenizer


def build_vocab_file(train: list[dict[str, str]], path: str) -> None:
    texts = [
        build_training_text(e["title"], e["description"], e["severity"]) for e in train
    ]
    tok = SimpleTokenizer.build(texts)
    tok.save(path)
    print(f"vocab size {len(tok)} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate synthetic data.")
    parser.add_argument("--seed-train", type=int, default=SEED_TRAIN)
    parser.add_argument("--seed-eval", type=int, default=SEED_EVAL)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = generate(args.seed_train, N_TRAIN)
    train_keys = {(e["title"], e["description"]) for e in train}
    eval_ = generate(args.seed_eval, N_EVAL, exclude=train_keys)

    # Held-out means held-out: no exact duplicates across splits.
    overlap = [e for e in eval_ if (e["title"], e["description"]) in train_keys]
    if overlap:  # pragma: no cover - defensive; exclude= above prevents this
        raise RuntimeError(f"train/eval overlap: {overlap}")

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), eval_)
    build_vocab_file(train, os.path.join(args.out_dir, "vocab.json"))
    print(f"wrote {len(train)} train + {len(eval_)} eval examples to {args.out_dir}/")


if __name__ == "__main__":
    main()
