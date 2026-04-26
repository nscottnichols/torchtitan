# agpt_2b Loss Competition

**Goal:** Lowest training loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).

**W&B Report:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

## Rules

1. **Model:** `agpt_2b` (2048 dim, 12 layers, 256k vocab) — architecture tweaks allowed
2. **Hardware:** 2 nodes, 24 XPU tiles
3. **Steps:** exactly 1000
4. **Dataset:** `HuggingFaceFW/fineweb-edu` (streaming) — fixed for fairness
5. **Batch size:** LBS=2, GBS=48 — fixed for fairness
6. **Sequence length:** 8192 — fixed for fairness
7. All changes must stay within `experiments/ezpz/`
8. **Tunable:** optimizer, LR, LR schedule, gradient clipping, architecture tweaks

## Leaderboard

| Rank | Config | Optimizer | LR | Schedule | Final Loss | Steps | TPS/GPU | Status |
|------|--------|-----------|------|----------|------------|-------|---------|--------|
| 1 | `speedrun_2b_adamw` | AdamW | 1.3e-3 | WSD 20/800/200 | **3.801** | 1000 | 7,245 | Done |
| 2 | `speedrun_2b_muon` | Muon | 2.4e-3 | WSD 20/800/200 | 3.915* | 785 | 4,695 | Resubmitted (3h) |
| 3 | `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | WSD 20/800/200 | 4.719 | 1000 | 7,208 | Done |
| 4 | `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | WSD 20/800/200 | 4.756* | 773 | 4,596 | Resubmitted (3h) |
| 5 | `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | WSD 20/800/200 | 5.850 | 1000 | 7,344 | Done |

*Timed out at 1h walltime — loss at last completed step.

## In Progress

### Hparam Tweaks (submitted 2026-04-26)

| Config | Change vs Baseline | Job ID | Status |
|--------|-------------------|--------|--------|
| `speedrun_2b_adamw_short_decay` | decay_ratio=0.1 (10%) | 12465388 | Running |
| `speedrun_2b_adamw_cosine` | cosine decay | 12465389 | Running |
| `speedrun_2b_adamw_fast_warmup` | warmup=5, decay=10% | 12465390 | Running |
| `speedrun_2b_muon_short_decay` | Muon + decay=10% | 12465391 | Running |
| `speedrun_2b_muon_fast_warmup` | Muon + warmup=5, decay=10% | 12465392 | Running |

### New Optimizers (submitted 2026-04-26)

| Config | Optimizer | Paper | Job ID | Status |
|--------|-----------|-------|--------|--------|
| `speedrun_2b_mano` | Mano (manifold-normalized) | [arxiv 2601.23000](https://arxiv.org/abs/2601.23000) | 12465393 | Running |
| `speedrun_2b_spam` | SPAM (spike-aware Adam) | [arxiv 2501.06842](https://arxiv.org/abs/2501.06842) | 12465394 | Running |

### Architecture Tweaks (submitted 2026-04-26)

| Config | Change | Job ID | Status |
|--------|--------|--------|--------|
| `speedrun_2b_adamw_qknorm` | AdamW + QK-Norm | 12465395 | Running |
| `speedrun_2b_muon_qknorm` | Muon + QK-Norm | 12465396 | Running |

## Modifications Log

### Optimizers Added

| Optimizer | File | Key Idea | Source |
|-----------|------|----------|--------|
| **Mano** | `optimizer/mano.py` | Tangent-space projection on rotating Oblique manifold. Cheaper than Muon (vector ops vs Newton-Schulz). 1.75x faster wall-clock. | [arxiv 2601.23000](https://arxiv.org/abs/2601.23000) |
| **SPAM** | `optimizer/spam.py` | Spike-aware gradient clipping + periodic momentum reset every DeltaT steps. Prevents gradient spike damage. | [arxiv 2501.06842](https://arxiv.org/abs/2501.06842) |

### Architecture Tweaks

| Tweak | File | Key Idea | Source |
|-------|------|----------|--------|
| **QK-Norm** | `agpt/__init__.py` | RMSNorm on Q,K before attention dot product. Prevents attention entropy collapse in early training. | Gemma 2, NanoGPT speedrun |

### Infrastructure

| Feature | File | Description |
|---------|------|-------------|
| Generic HF datasets | `datasets.py` | `register_hf_dataset()` + auto-fallback for arbitrary HF hub paths |
| Competition configs | `competition/configs.py` | Speedrun configs with fixed dataset/LBS/seq_len |
| Submit script | `competition/submit_run.sh` | PBS submission with .venv setup |

## Quick Start

```bash
# Run a single config
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config speedrun_2b_muon

# Submit to PBS
qsub -l select=2 -N speedrun_2b_muon -v CONFIG=speedrun_2b_muon \
    torchtitan/experiments/ezpz/competition/submit_run.sh
```

## Ideas to Try Next

- **Cosine decay to zero** vs linear (in progress)
- **Shorter decay window** (10% vs 20%, in progress)
- **Mano + QK-Norm** combo
- **SPAM + Muon hybrid** — spike-aware clipping with Muon's orthogonal updates
- **Logit softcapping** — cap attention logits at 30.0 (Gemma 2)
- **WSM** — checkpoint merging instead of online decay ([arxiv 2507.17634](https://arxiv.org/html/2507.17634v2))
- **Data diversity** — ensure each batch has tokens from 16+ unique documents
- **ReLU-squared activation** — replace SiLU in FFN (NanoGPT speedrun)
