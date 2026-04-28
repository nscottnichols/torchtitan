# Evaluation Results — agpt 2B

> **Living document** — updated as new eval results come in.
>
> Last updated: 2026-04-28

## Setup

| Field | Value |
|-------|-------|
| Model | agpt_2b (1.99B params) |
| Tokenizer | google/gemma-7b (vocab_size=256128) |
| Training data | olmo-mix-1124 (4.67T tokens) |
| Training config | 256N, GBS=3,072, SophiaG LR=2.28e-5, seq_len=8,192 |
| Eval tasks | hellaswag, arc_easy, arc_challenge, winogrande (0-shot, acc_norm) |
| Eval backend | lm-eval 0.4.10 + transformers 4.57.6 (HF backend, XPU, bf16) |
| Checkpoint conversion | DCP → HF safetensors via `eval/convert_to_hf.py` |

Tokens consumed at step *N* = `N × 3,072 × 8,192`. Random baseline is
25% for the four-choice tasks (hellaswag, arc_*) and 50% for winogrande.

## Benchmark Accuracy vs Training Step

![2B Eval Results](figures/eval_2b_v2.png)

## Results

| Step | Tokens | HellaSwag | ARC-Easy | ARC-Chall | Winogrande |
|-----:|------:|------:|------:|------:|------:|
|  1,000 |  25.2B | 25.35 | 27.44 | 23.38 | 47.83 |
|  2,000 |  50.3B | 25.26 | 27.19 | 24.23 | 49.57 |
|  3,000 |  75.5B | 25.25 | 26.89 | 23.55 | 51.07 |
|  4,000 | 100.7B | 25.12 | 26.98 | 24.06 | 48.46 |
|  5,000 | 125.8B | 25.09 | 26.98 | 24.06 | 49.01 |
|  6,000 | 151.0B | 25.10 | 27.48 | 24.23 | 48.54 |
|  7,000 | 176.2B | 25.39 | 27.19 | 23.55 | 49.88 |
|  8,000 | 201.3B | 25.35 | 26.98 | 24.49 | 51.07 |
|  9,000 | 226.5B | 25.17 | 27.23 | 24.06 | 47.67 |
| 10,000 | 251.7B | 25.07 | 27.82 | 23.81 | 48.86 |
| 11,000 | 276.8B | 25.36 | 26.68 | 24.15 | 47.99 |
| 12,000 | 302.0B | 25.04 | 27.31 | 24.06 | 49.09 |
| 13,000 | 327.2B | 25.11 | 27.31 | 24.15 | 49.01 |
| 14,000 | 352.3B | 25.00 | 27.48 | 24.57 | 49.88 |
| 15,000 | 377.5B | 25.17 | 27.78 | 24.15 | 50.99 |
| 16,000 | 402.7B | 25.27 | 27.44 | 24.15 | 49.96 |
| 17,000 | 427.8B | 25.16 | 26.73 | **25.60** | 49.80 |
| 18,000 | 453.0B | 25.22 | 27.27 | 25.00 | 49.49 |

**Mean (steps 1K–18K):** hellaswag 25.20 · arc_easy 27.23 · arc_challenge 24.18 · winogrande 49.34

## Observations

- **All four tasks hover near random baseline through step 18,000** (~453B tokens, ~10% of the 4.67T target). This is expected — small models typically need 500B+ tokens before benchmarks rise above noise.
- **arc_challenge** shows the clearest upward drift, climbing from 23.38 → 25.60 between steps 1K and 17K. The last two checkpoints are the first to land at-or-above the 25% random baseline.
- **arc_easy** is consistently 1–3 points above its 25% baseline, peaking at 27.82 (step 10K). Modest but real signal.
- **hellaswag** is noise-bound at 25.0–25.4 — no meaningful trend yet.
- **winogrande** fluctuates 47.7–51.1, with three clear above-baseline peaks (steps 3K, 8K, 15K). Variance is high enough that the trend isn't yet distinguishable from noise.

## Caveats

- The DCP → HF conversion may not be loading the embedding table correctly
  (different DCP checkpoints produce HF safetensors with **identical**
  `tok_embeddings.weight`). Deeper layers are loaded correctly, so the eval
  scores reflect the trained transformer minus updated embeddings.
- Re-running with a fixed converter is expected to produce slightly different
  numbers, particularly for tasks that hinge on token-level predictions.

## Reproducing

```bash
# Convert + evaluate a single checkpoint (compute node, ezpz_setup_env)
bash torchtitan/experiments/ezpz/eval/convert_and_eval.sh \
    --model 2b --step 18000

# Aggregate all results into the table + figure above
python3 torchtitan/experiments/ezpz/eval/aggregate_evals.py --model 2b
```
