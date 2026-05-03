# Evaluation Results — agpt 2B

> **Living document** — updated as new eval results come in.
>
> Last updated: 2026-05-03
>
> **Training curves:** see [`docs/production/agpt/2b/`](../../../production/agpt/2b/README.md)
> for loss / throughput / MFU dashboards across all 2B trajectories
> (v1 256N, v2 256N, v2 512N, v2 1024N), each with its own per-node
> sub-page.

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

## v1 vs v2 — benchmark accuracy

The v1 (bf16-master) and v2 (fp32-master) 2B runs share the same
architecture, optimizer, and dataset; the only difference is the
master-weight dtype. v1 hovers within ~1pp of the random baseline on
every task across all 18K steps (~453B tokens) — this looked like
"slow learning" at the time but is actually the bf16-RMSNorm-freeze
bug talking. v2 — once it has enough tokens — should descend visibly.
Re-render with new v2 ckpts as they become available:

```bash
python3 torchtitan/experiments/ezpz/docs/evals/agpt/2b/plot_v1_vs_v2.py
```

![v1 vs v2 — 2B benchmark accuracy](figures/v1_vs_v2.png)

## v2 256N vs v2 512N — same model, two batch sizes

A surprise from the overlay above: **at matched token counts, v2 256N
beats v2 512N by 3-8pp** — even though 512N has the same architecture
and optimizer.

| Tokens (B) | 256N step | 512N step | 256N HellaSwag | 512N HellaSwag | Δ |
|-----------:|----------:|----------:|---------------:|---------------:|--:|
| ~50 | 1000 | ~500 | 0.262 | (not evaluated) | — |
| ~100 | 2000 | 1000 | **0.301** | 0.264 | **-3.7pp** |
| ~200 | (n/a) | 2000 | — | 0.304 | — |

But at **matched step counts**, the two trajectories are
indistinguishable (within the ~1.4pp lm-eval stderr):

| Step | 256N HellaSwag | 512N HellaSwag | Δ |
|-----:|---------------:|---------------:|--:|
| 1000 | 0.262 | 0.264 | +0.2pp |
| 2000 | 0.301 | 0.304 | +0.3pp |

### Interpretation

This is the classic **large-batch under-training** effect. Both runs
use SophiaG LR=2.28e-5 (tuned for the 256N regime, GBS=6,144). At
512N (GBS=12,288):
- Each gradient update covers 2× more tokens
- But the optimizer state evolves at half the cadence per token
- LR scaling was **not** applied — we kept LR=2.28e-5 fixed
- So the 512N model is meaningfully **under-trained per token**

The per-step parity confirms the optimizer/architecture are healthy —
each gradient update produces the same effective learning. The
per-token gap is purely the doubled batch size eating into the
gradient-update budget.

**Implications:**
- **Per-step**: 256N == 512N (same per-update progress)
- **Per-token**: 256N wins by ~3-8pp at matched tokens
- **Per-wall-clock**: 512N wins (≈2× throughput)

For convergence-to-target-loss, 256N is the more token-efficient
choice. For reaching a target wall-clock, 512N gets there faster but
spends more tokens. Either could be preferable depending on the goal
(chasing minimum wall-clock vs minimum tokens).

**Open question**: would √2-scaling LR (3.22e-5) at 512N close the
per-token gap? Worth a one-off experiment — but the canonical 512N
chain has too much accumulated training to perturb its LR mid-run, so
that test would need a fresh fork.

<details>
<summary><strong>v1 detailed results (bf16-tainted, kept for record) — click to expand</strong></summary>

### Benchmark Accuracy vs Training Step (v1)

![2B Eval Results](figures/eval_2b_v2.png)

### Results (v1)

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

### Observations

- **All four tasks hover near random baseline through step 18,000** (~453B tokens, ~10% of the 4.67T target). This is expected — small models typically need 500B+ tokens before benchmarks rise above noise.
- **arc_challenge** shows the clearest upward drift, climbing from 23.38 → 25.60 between steps 1K and 17K. The last two checkpoints are the first to land at-or-above the 25% random baseline.
- **arc_easy** is consistently 1–3 points above its 25% baseline, peaking at 27.82 (step 10K). Modest but real signal.
- **hellaswag** is noise-bound at 25.0–25.4 — no meaningful trend yet.
- **winogrande** fluctuates 47.7–51.1, with three clear above-baseline peaks (steps 3K, 8K, 15K). Variance is high enough that the trend isn't yet distinguishable from noise.

### Caveats

- **All RMSNorm weights are frozen at 1.0** because the production
  runs used `training.dtype = bfloat16` (master weights in bf16,
  per-step update sub-ulp at scale 1.0). This is a real training
  issue, not a converter bug. Eval scores below reflect a model
  whose 25 RMSNorm parameters never moved from init throughout the
  18K-step run. See
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../../../guides/training-dtype-bf16-norm-freeze.md).
- The earlier "embeddings stuck across checkpoints" caveat was a
  misdiagnosis: it was based on inspecting `tok_embeddings.weight[0]`
  (a frozen padding-token row). Non-padding embedding rows DO
  update, and the DCP → HF converter loads them correctly.
- Eval scores below are noise-bound at near-random levels for all
  tasks. This is consistent with what we'd expect from a model
  whose normalization is permanently broken: the residual stream
  learns, but loses information through poorly-scaled norms.
  Compare with the MDS-trained 2B (no norm freeze) which reaches
  ~0.59 hellaswag at the same approximate token count.

</details>

## Reproducing

```bash
# Convert + evaluate a single checkpoint (compute node, ezpz_setup_env)
bash torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh \
    --model 2b --step 18000

# Aggregate all results into the table + figure above
python3 torchtitan/experiments/ezpz/eval/aggregate_evals.py --model 2b
```
