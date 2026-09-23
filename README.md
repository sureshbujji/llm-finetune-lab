# llm-finetune-lab

I built this to learn LoRA fine-tuning the way I learn everything as a QA lead: **by building the smallest real thing that can be measured before and after.** It fine-tunes a tiny GPT-2 with a LoRA adapter (real `transformers` + `peft`, real gradient steps) to classify bug reports by severity — then evaluates the base model against the adapted model on a held-out set and writes up the delta.

The headline result is deliberately toy-scale, and I'm honest about that below: a randomly-initialized 2-layer model trained on 50 synthetic bug reports on CPU goes from **0.00 → 0.45 accuracy** on 20 held-out examples. What transfers to real work is the methodology: seeded data, frozen-base verification, masked-label training, and a before/after eval with per-class metrics.

## Architecture

```
                    +---------------------+
                    | src/make_data.py    |  seeded RNG (42/2026)
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
   data/train.jsonl (50)            data/eval.jsonl (20)
   data/vocab.json (275 toks)       strictly held-out
              |                                 |
              +----------------+----------------+
                               |
                    +----------v----------+
                    | src/model.py        |  --model-source random:
                    | tiny GPT-2 (2 lyr)  |  random init, OFFLINE (default)
                    |                     |  --model-source pretrained:
                    |                     |  distilgpt2 (one-time download)
                    +----------+----------+
                               |
                    +----------v----------+
                    | src/adapter.py      |  peft LoRA: r=8, alpha=16,
                    | get_peft_model      |  targets c_attn/c_proj
                    +----------+----------+  base frozen, 4.62% trainable
                               |
                    +----------v----------+
                    | src/train.py        |  manual AdamW loop, label-only
                    |                     |  loss (-100 masked prompt)
                    +----------+----------+
                               |
                    adapters/lora-severity/  (adapter_model.safetensors)
                               |
              +----------------+----------------+
              |                                 |
     base model (seed 123)            base + LoRA adapter
              |                                 |
              +----------------+----------------+
                               |
                    +----------v----------+
                    | src/eval_before_after.py   |  greedy decode 20 held-out,
                    | accuracy + per-class P/R/F1|  parse label, score
                    +----------+----------+
                               |
                  reports/finetune_report.md  (committed, seeded run)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # CPU-only torch (~200MB, no CUDA wheels)

pytest                            # 25 tests, all offline
python src/make_data.py           # regenerate data/ (seeded, idempotent)
python src/train.py --smoke       # 1 optimizer step on 4 examples (what CI does)
python src/train.py               # full tiny run: 10 epochs x 50 examples, ~17s on CPU
python src/eval_before_after.py   # base vs LoRA -> reports/finetune_report.md
```

No API key, no network, no GPU needed for any of the above. The committed `adapters/lora-severity/` is the adapter from the seeded full run, so eval works right after install.

To use real pretrained weights instead (one config flag):

```bash
python src/train.py --model-source pretrained   # downloads distilgpt2 ONCE (~350MB), then cached
python src/eval_before_after.py --model-source pretrained
```

## Sample output

Seeded offline run (`python src/train.py` then `python src/eval_before_after.py`, seed 123):

```
trainable params: 22,528 / 487,296 (4.62%)
done: 70 optimizer step(s), avg loss 4.4442
base acc=0.00 lora acc=0.45 delta=+0.45
```

| Severity | Base F1 | LoRA F1 | Support |
|---|---|---|---|
| Critical | 0.00 | 0.67 | 5 |
| High | 0.00 | 0.00 | 5 |
| Medium | 0.00 | 0.00 | 5 |
| Low | 0.00 | 0.53 | 5 |

The base model (random init) generates gibberish — 0.00 is expected, not a bug. LoRA teaches it the output format and part of the keyword→severity mapping (Critical/Low mostly; High/Medium still confuse it — see the error analysis in `reports/finetune_report.md`). Full report: [reports/finetune_report.md](reports/finetune_report.md).

## Honest compute notes

| Step | Runs on | Network? |
|---|---|---|
| `pytest` (25 tests) | any CPU | never |
| `make_data.py` | any CPU | never |
| `train.py --smoke` / full tiny run | CPU, seconds | never |
| `eval_before_after.py` | CPU, seconds | never |
| `--model-source pretrained` | one-time ~350MB download, then cached | **yes, once** |

What I'd actually do with a GPU (e.g. a Colab T4): run the `--model-source pretrained` path — distilgpt2 already knows language, so LoRA only has to teach the task format and domain vocabulary, and the delta would be dramatically larger. The random-init path exists so the methodology is verifiable anywhere, including CI and offline machines; it is not a recipe for production accuracy.

## LoRA hyperparameters

In `src/adapter.py` (`LORA_KWARGS`):

- **r=8** — tiny rank; the task is a 4-way keyword→label map, so a small subspace suffices.
- **lora_alpha=16** (= 2r) — the standard scaling heuristic; effective update scale is alpha/r = 2.
- **lora_dropout=0.05** (0.0 in `--smoke`) — mild regularization; smoke forces 0.0 for determinism.
- **target_modules=[c_attn, c_proj]** — the GPT-2 attention projections. For LLaMA-family models these would be `q_proj`/`v_proj` (plus k/o on bigger runs).
- **bias="none"** — biases stay frozen; standard for LoRA.

What I'd change on a real GPU run: start from pretrained weights; sweep r ∈ {8, 16, 32} and lr ∈ {1e-4, 5e-4, 1e-3} against a real validation split with early stopping, keeping the best by eval F1; add target modules if r=8 underfits; replace greedy generation with log-likelihood scoring over the four label strings (more stable than free generation — see `parse_prediction` for why parsing free text is fragile).

One thing I learned building this: with a weak random-init model, **including EOS as a training target collapses greedy decoding** — EOS ends 100% of sequences while each label ends ~25%, so the model learns the marginal and emits EOS immediately. `encode_with_mask` therefore defaults to label-only targets (`add_eos=False`), documented in the docstring.

## When fine-tuning beats prompting (my decision guide)

- **Fine-tune** when the task is stable (fixed labels/format), you need it offline / low-latency / cheap at inference, the behavior must persist without a 2,000-token system prompt, or you're distilling a larger model's judgments into a small deployable model.
- **Prompt** when the task changes weekly, you have a handful of examples, or the base model already does the job — fine-tuning is the expensive way to learn what a prompt could have told you.
- **Neither — measure first**: if the bottleneck is knowledge (facts that change), that's retrieval, not weights. Fine-tuning doesn't fix stale knowledge; it fixes behavior.

This repo is the "measure first" habit made concrete: the eval harness exists before any claim about the adapter.

## Roadmap

- Add the `--model-source pretrained` report (distilgpt2, Colab GPU) next to the random-init one.
- Hyperparameter sweep script (r × lr) with early stopping on a validation split.
- Log-likelihood scoring over label strings as the primary eval metric.
- Larger, noisier synthetic set (label noise, adversarial near-duplicates) to test robustness.

## Layout

```
llm-finetune-lab/
  data/
    train.jsonl            50 synthetic bug reports (seed 42)
    eval.jsonl             20 held-out bug reports (seed 2026)
    vocab.json             275-token offline vocab built from train
  src/
    make_data.py           seeded data generator (+ vocab builder)
    prompts.py             prompt template + label-masked encoding
    tokenizer.py           tiny offline word-level tokenizer
    model.py               random tiny GPT-2 | distilgpt2 (one flag)
    adapter.py             LoRA config (r=8, alpha=16, c_attn/c_proj)
    train.py               LoRA training loop, CLI (--smoke)
    eval_before_after.py   base vs adapter eval -> reports/
    metrics.py             accuracy + per-class P/R/F1 (pure functions)
  adapters/lora-severity/  committed adapter from the seeded run
  reports/finetune_report.md   committed before/after report (seeded, offline)
  tests/                   25 tests: data determinism, prompts, LoRA wiring,
                           metric math, prediction parsing
  .github/workflows/ci.yml  pytest + smoke train + smoke eval (CPU, no network)
```

## License

MIT — see [LICENSE](LICENSE).
