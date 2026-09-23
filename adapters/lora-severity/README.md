# Adapter: lora-severity

LoRA adapter for bug-severity classification, from the
llm-finetune-lab repo (seeded offline run — see
`reports/finetune_report.md`).

- **Base model:** randomly-initialized tiny GPT-2 (2 layers, 4 heads,
  128 hidden, 275-token word vocab), seed 123 — *not* pretrained weights.
- **LoRA:** r=8, alpha=16, dropout=0.05, targets `c_attn`+`c_proj`,
  bias=none. 22,528 trainable params (4.62% of 487,296).
- **Training:** 10 epochs x 50 synthetic bug reports, batch 8, AdamW lr=1e-3,
  label-only loss (prompt tokens masked with -100). CPU, ~17s.
- **Result:** held-out accuracy 0.00 (base) -> 0.45 (adapted), n=20.
- **Full config:** `training_config.json` in this directory.

Toy-scale artifact for methodology demonstration, not production use.
Regenerate: `python src/train.py` (fully offline, deterministic).
