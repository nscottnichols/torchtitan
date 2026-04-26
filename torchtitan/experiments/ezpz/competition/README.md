# agpt_2b Loss Competition

**Goal:** Lowest training loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).

## Rules

1. **Model:** `agpt_2b` (2048 dim, 12 layers, 256k vocab) — no architecture changes
2. **Hardware:** 2 nodes, 24 XPU tiles
3. **Steps:** exactly 1000
4. **Dataset:** `HuggingFaceFW/fineweb-edu` (streaming) — fixed for fairness
5. **Batch size:** LBS=2, GBS=48 — fixed for fairness
6. **Sequence length:** 8192 — fixed for fairness
7. All changes must stay within `experiments/ezpz/`
8. **Tunable:** optimizer, LR, LR schedule, gradient clipping, compile options

## Configs

| Config | Optimizer | LR | Schedule |
|--------|-----------|------|----------|
| `speedrun_2b_adamw` | AdamW | 1.3e-3 | WSD (20/800/200) |
| `speedrun_2b_muon` | Muon | 2.4e-3 | WSD (20/800/200) |
| `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | WSD (20/800/200) |
| `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | WSD (20/800/200) |
| `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | WSD (20/800/200) |

All configs: LBS=2, GBS=48, seq_len=8192, compile=on, AC=none, fineweb-edu.

## Quick Start

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config speedrun_2b_muon
```

## Leaderboard

| Rank | Config | Optimizer | LR | Final Loss | Date | Notes |
|------|--------|-----------|------|------------|------|-------|
| — | — | — | — | — | — | _Run configs to populate_ |

## Ideas to Try

- **Cosine decay:** `--lr_scheduler.decay_type cosine`
- **Longer warmup:** `--lr_scheduler.warmup_steps 50`
- **Shorter decay:** `--lr_scheduler.decay_ratio 0.1` (decay last 10%)
- **ADOPT optimizer:** Available in `ezpz/optimizer/adopt.py`
- **Gradient clipping:** `--optimizer.max_grad_norm 1.0`
- **Different betas:** `--optimizer.betas 0.9,0.999`
