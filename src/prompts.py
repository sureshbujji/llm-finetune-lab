"""Prompt templates for bug-severity classification as causal-LM fine-tuning.

Training text = prompt + label (loss is computed on the label tokens only).
Inference prompt = prompt with no label; the model generates the severity.
"""

from __future__ import annotations

LABELS = ["Critical", "High", "Medium", "Low"]

PROMPT_TEMPLATE = (
    "### Bug report\n"
    "Title: {title}\n"
    "Description: {description}\n"
    "\n"
    "### Severity (Critical, High, Medium, Low):\n"
)


def build_prompt(title: str, description: str) -> str:
    """Inference prompt: no label, model must generate it."""
    return PROMPT_TEMPLATE.format(title=title, description=description)


def build_training_text(title: str, description: str, severity: str) -> str:
    """Full training string: prompt followed by the gold label."""
    if severity not in LABELS:
        raise ValueError(f"unknown severity: {severity!r}")
    return build_prompt(title, description) + severity


def encode_with_mask(tokenizer, title: str, description: str, severity: str,
                     add_eos: bool = False):
    """Tokenize training example; mask prompt tokens with -100 in labels.

    Returns (input_ids, labels) where labels == -100 for every prompt token
    so cross-entropy is computed on the label tokens only.

    add_eos appends an end-of-sequence target after the label. It defaults to
    False: on this toy's tiny random-init model the EOS target dominates the
    marginal next-token distribution (EOS ends 100% of sequences, each label
    only ~25%) and greedy decoding collapses to immediate EOS. A real run
    from pretrained weights keeps the EOS target.
    """
    prompt_ids = tokenizer.encode(build_prompt(title, description))
    label_ids = tokenizer.encode(severity, add_eos=add_eos)
    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    return input_ids, labels
