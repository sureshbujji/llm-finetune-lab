"""LoRA wiring: adapter params trainable, base model frozen.

Needs torch + transformers + peft; skipped gracefully when unavailable.
Uses the offline random-init tiny GPT-2 so no network is required.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from src.adapter import apply_lora, build_lora_config, param_counts
from src.model import build_base_model


def _tiny_model():
    return build_base_model("random", vocab_size=64, seed=7)


def test_lora_config_values():
    cfg = build_lora_config()
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == pytest.approx(0.05)
    assert set(cfg.target_modules) == {"c_attn", "c_proj"}


def test_only_adapter_params_trainable():
    model = apply_lora(_tiny_model())
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert trainable_names, "no trainable parameters found"
    assert frozen_names, "base model was not frozen"
    assert all("lora_" in n for n in trainable_names)
    assert all("lora_" not in n for n in frozen_names)


def test_trainable_fraction_is_small():
    model = apply_lora(_tiny_model())
    trainable, total = param_counts(model)
    assert 0 < trainable < 0.05 * total


def test_base_output_unchanged_by_adapter_init():
    """A freshly-applied LoRA adapter is a near-no-op: B is zero-initialized,
    so base-model logits should barely move before training."""
    torch.manual_seed(0)
    base = _tiny_model()
    adapted = apply_lora(_tiny_model())
    base.eval()
    adapted.eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        out_base = base(input_ids=x).logits
        out_adapt = adapted(input_ids=x).logits
    assert torch.allclose(out_base, out_adapt, atol=1e-4)
