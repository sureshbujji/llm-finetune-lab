"""LoRA fine-tuning on the bug-severity task.

Offline by default: randomly-initialized tiny GPT-2 + SimpleTokenizer,
no network. --model-source pretrained swaps in distilgpt2 (one-time
download, then cached).

  python src/train.py --smoke          # 1 optimizer step on 4 examples (CI)
  python src/train.py                  # 3 epochs x 50 examples (still CPU-fast)

Saves the LoRA adapter to adapters/lora-severity/ + training_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adapter import apply_lora, build_lora_config, param_counts
from src.model import build_base_model, build_tokenizer_for_source
from src.prompts import LABELS, encode_with_mask


def load_examples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collate(tokenizer, examples, pad_id: int):
    """Pad a batch of (input_ids, labels) pairs.

    Labels already carry -100 masking for prompt tokens (see
    prompts.encode_with_mask); padding positions are masked with -100 too.
    """
    import torch

    encoded = [
        encode_with_mask(tokenizer, ex["title"], ex["description"], ex["severity"])
        for ex in examples
    ]
    max_len = max(len(ids) for ids, _ in encoded)
    batch_ids, batch_labels, batch_masks = [], [], []
    for input_ids, labels in encoded:
        pad = max_len - len(input_ids)
        batch_ids.append(input_ids + [pad_id] * pad)
        batch_labels.append(labels + [-100] * pad)
        batch_masks.append([1] * len(input_ids) + [0] * pad)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_masks, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def train(args) -> dict:
    import torch

    torch.manual_seed(args.seed)
    tokenizer, vocab_size = build_tokenizer_for_source(args.model_source, args.vocab)
    model = build_base_model(args.model_source, vocab_size, args.seed)
    peft_model = apply_lora(model, dropout=0.0 if args.smoke else None)
    peft_model.train()

    trainable, total = param_counts(peft_model)
    print(
        f"trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.2f}%)"
    )

    examples = load_examples(args.train_file)
    if args.smoke:
        examples = examples[:4]
        epochs, batch_size, lr = 1, 4, 1e-3
    else:
        epochs, batch_size, lr = args.epochs, args.batch_size, args.lr

    pad_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=lr
    )

    step, running_loss = 0, 0.0
    for epoch in range(epochs):
        # Deterministic order (no shuffling) so seeded runs are reproducible.
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            input_ids, attn, labels = collate(tokenizer, batch, pad_id)
            out = peft_model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            running_loss += loss.item()
            if args.smoke:
                break
        if args.smoke:
            break

    avg_loss = running_loss / max(step, 1)
    print(f"done: {step} optimizer step(s), avg loss {avg_loss:.4f}")

    os.makedirs(args.adapter_out, exist_ok=True)
    peft_model.save_pretrained(args.adapter_out)
    lora_cfg = build_lora_config(dropout=0.0 if args.smoke else None)
    config = {
        "model_source": args.model_source,
        "lora": {
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
            "bias": lora_cfg.bias,
            "target_modules": list(lora_cfg.target_modules),
            "task_type": str(lora_cfg.task_type),
        },
        "seed": args.seed,
        "epochs": epochs,
        "n_train": len(examples),
        "batch_size": batch_size,
        "lr": lr,
        "steps": step,
        "avg_loss": avg_loss,
        "trainable_params": trainable,
        "total_params": total,
        "smoke": args.smoke,
    }
    with open(os.path.join(args.adapter_out, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"adapter saved to {args.adapter_out}/")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on bug severity.")
    parser.add_argument("--smoke", action="store_true",
                        help="1 optimizer step on 4 examples (for CI)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-source", choices=["random", "pretrained"],
                        default="random")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument("--adapter-out", default="adapters/lora-severity")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
