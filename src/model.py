"""Model construction.

Two sources, one flag:
  --model-source random      tiny randomly-initialized GPT-2 (OFFLINE, default)
  --model-source pretrained  distilgpt2 weights + tokenizer from HuggingFace
                             (needs network ONCE to download, then cached)

The random path is fully offline and deterministic given --seed, which is
what CI and the committed report use.
"""

from __future__ import annotations

TINY_GPT2 = dict(
    n_layer=2,
    n_head=4,
    n_embd=128,
    n_inner=512,
    n_positions=256,
    n_ctx=256,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)

PRETRAINED_ID = "distilgpt2"


def build_base_model(source: str, vocab_size: int, seed: int):
    """Build the base causal LM. torch/transformers imported lazily so the
    pure-python modules stay importable without torch installed."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config

    if source == "random":
        torch.manual_seed(seed)
        config = GPT2Config(
            vocab_size=vocab_size,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=2,
            **TINY_GPT2,
        )
        model = GPT2LMHeadModel(config)
    elif source == "pretrained":
        from transformers import AutoModelForCausalLM

        # NOTE: downloads ~350MB once, then uses the HF cache offline.
        model = AutoModelForCausalLM.from_pretrained(PRETRAINED_ID)
    else:
        raise ValueError(f"unknown --model-source: {source!r}")
    model.config.use_cache = False
    return model


def build_tokenizer_for_source(source: str, vocab_path: str):
    """Return (tokenizer, vocab_size) for the chosen model source."""
    if source == "random":
        from .tokenizer import SimpleTokenizer

        tok = SimpleTokenizer.load(vocab_path)
        return tok, len(tok)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PRETRAINED_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, tok.vocab_size
