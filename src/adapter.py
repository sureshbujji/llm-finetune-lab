"""LoRA adapter configuration and helpers (peft)."""

from __future__ import annotations

# LoRA hyperparameters. Rationale (see README "LoRA notes"):
#  - r=8: tiny rank; the task is a 4-way keyword->label map, so a small
#    subspace suffices. On a real GPU run with a bigger model/task I'd try 16.
#  - lora_alpha=16 (= 2*r): the standard scaling heuristic; the effective
#    update scale is alpha/r = 2.
#  - lora_dropout=0.05: mild regularization; set to 0.0 for the deterministic
#    smoke/seeded runs.
#  - target_modules c_attn + c_proj: the GPT-2 attention projections. For
#    LLaMA-family models these would be q_proj/v_proj (+ k/o on bigger runs).
LORA_KWARGS = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"],
)


def build_lora_config(dropout: float | None = None):
    from peft import LoraConfig, TaskType

    kwargs = dict(LORA_KWARGS)
    if dropout is not None:
        kwargs["lora_dropout"] = dropout
    return LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)


def apply_lora(model, dropout: float | None = None):
    """Wrap model with LoRA. Freezes base weights; only adapter trains."""
    from peft import get_peft_model

    return get_peft_model(model, build_lora_config(dropout))


def load_adapter(base_model, adapter_dir: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_dir)


def param_counts(model) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
