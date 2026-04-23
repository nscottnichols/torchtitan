# Production Training Runs — Aurora

> **Living document** — updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-04-23

## Overview

Full-scale production training of AuroraGPT dense models on the
[olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) dataset
(4.67T tokens) across Aurora compute nodes.

## Loss Curves

![Production Training Loss](experiments/agpt/aurora/figures/production_training_loss.png)

## Active Runs

### 2B @ 256N — SophiaG LR=2.28e-5

| Field | Value |
|-------|-------|
| Model | agpt_2b (1.99B params) |
| Nodes / GPUs | 256 / 3,072 |
| Parallelism | TP=1, FSDP=3072 |
| Compile | on |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 3,072 (LBS=1) |
| Total steps | 185,718 |
| Total tokens | 4.67T |
| Seq len | 8,192 |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072` |
| Checkpoint interval | 100 steps |

**Progress:**

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8444122 | 1–1431 | 12.94 → 6.13 | 489 | 1.8% | 47.12 GiB | Walltime (12h) |
| 8446337 | 1431→ | — | — | — | — | Queued |
| 8446338 | cont. | — | — | — | — | Queued (dep) |
| 8446339 | cont. | — | — | — | — | Queued (dep) |

**W&B:** [pjanidnw](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pjanidnw)

**Tokens consumed:** 1431 × 3072 × 8192 = **36.0T tokens** (0.8% of target)

**Note:** Low MFU (1.8%) is because the 2B model is too small for 3072-way
FSDP — communication dominates. TPS/GPU (489) is much lower than the 2N
baseline (5400). This is expected from the scaling study results.

---

### 20B @ 256N — SophiaG LR=2.28e-5

| Field | Value |
|-------|-------|
| Model | agpt_20b (20.7B params) |
| Nodes / GPUs | 256 / 3,072 |
| Parallelism | TP=1, FSDP=3072 |
| Compile | on |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 3,072 (LBS=1) |
| Total steps | 185,718 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs3072` |

**Progress:**

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8443212 | 1–100 | 12.93 → 11.84 | 280 | 14.0% | 40.95 GiB | Killed (qdel) |
| 8444123 | 100–581 | 11.84 → 7.35 | 248 | 12.4% | 43.95 GiB | Walltime (12h) |
| 8446340 | 581→ | — | — | — | — | Queued |
| 8446341 | cont. | — | — | — | — | Queued (dep) |
| 8446342 | cont. | — | — | — | — | Queued (dep) |

**W&B:** [q9oq5huj](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/q9oq5huj) (job 8443212), [pnkaurba](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pnkaurba) (job 8444123)

**Latest checkpoint:** step-500

**Tokens consumed:** 581 × 3072 × 8192 = **14.6B tokens** (0.3% of target)

---

### 20B @ 512N — SophiaG LR=2.28e-5

| Field | Value |
|-------|-------|
| Model | agpt_20b (20.7B params) |
| Nodes / GPUs | 512 / 6,144 |
| Parallelism | TP=1, FSDP=6144 |
| Compile | on |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 6,144 (LBS=1) |
| Total steps | 92,859 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs6144` |

**Progress:**

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8443819 | 1–458 | 12.94 → 7.09 | 41 | 2.1% | 54.14 GiB | Walltime (12h) |
| 8446343 | 458→ | — | — | — | — | Queued |
| 8446344 | cont. | — | — | — | — | Queued (dep) |

**W&B:** [8of5hse0](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/8of5hse0)

**Latest checkpoint:** step-400

**Tokens consumed:** 458 × 6144 × 8192 = **23.1B tokens** (0.5% of target)

**Note:** Very low TPS (41) — likely compile took most of the 12h walltime,
leaving limited time for actual training. The 20B model at 512N may need
compile=off for production viability.

---

### 80B @ 256N — AdamW LR=1.1e-5

| Field | Value |
|-------|-------|
| Model | agpt_80b (80.8B params) |
| Nodes / GPUs | 256 / 3,072 |
| Parallelism | TP=2, FSDP=1536 |
| Compile | on |
| Optimizer | AdamW, LR=1.1e-5 |
| GBS | 1,536 (LBS=1) |
| Total steps | 371,437 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-80b-AdamW-olmo-mix-1124-n256-gbs1536` |

**Progress:**

| Job ID | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|-------|------|---------|-----|--------|--------|
| 8444124 | 0 | — | — | — | — | Segfault (node) |
| 8446345 | 0→ | — | — | — | — | Queued (retry) |
| 8446346 | cont. | — | — | — | — | Queued (dep) |

**Expected:** 80 TPS/GPU, 15% MFU (from 256N test runs).
Compile takes ~7 min at 256N.

**Note:** Previous attempt (8444124) segfaulted on a single bad node.
AdamW is the only viable optimizer for 80B — SophiaG and Muon both
produce NaN due to bf16 overflow at dim=9216.

---

### 80B @ 512N — AdamW LR=1.1e-5 (no compile)

| Field | Value |
|-------|-------|
| Model | agpt_80b (80.8B params) |
| Nodes / GPUs | 512 / 6,144 |
| Parallelism | TP=2, FSDP=3072 |
| Compile | **off** (CPU OOM at 512N) |
| Optimizer | AdamW, LR=1.1e-5 |
| GBS | 3,072 (LBS=1) |
| Total steps | 185,718 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-80b-AdamW-olmo-mix-1124-n512-gbs3072` |

**Progress:**

| Job ID | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|-------|------|---------|-----|--------|--------|
| 8443820 | 0 | — | — | — | — | CPU OOM (compile) |
| 8446347 | 0→ | — | — | — | — | Queued (no-compile) |
| 8446348 | cont. | — | — | — | — | Queued (dep) |

**Expected:** 66 TPS/GPU, 12% MFU without compile (from 512N test).

---

### 2B @ 512N — SophiaG LR=2.28e-5 (no compile)

| Field | Value |
|-------|-------|
| Model | agpt_2b (1.99B params) |
| Nodes / GPUs | 512 / 6,144 |
| Parallelism | TP=1, FSDP=6144 |
| Compile | **off** (OOM at 512N) |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 6,144 (LBS=1) |
| Total steps | 92,859 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs6144` |

**Progress:**

| Job ID | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|-------|------|---------|-----|--------|--------|
| 8443818 | 0 | — | — | — | — | OOM (compile) |
| 8446349 | 0→ | — | — | — | — | Queued (no-compile) |
| 8446350 | cont. | — | — | — | — | Queued (dep) |

---

## Summary Table

| Run | Model | Nodes | Optimizer | LR | Compile | Steps Done | Loss | Status |
|-----|-------|-------|-----------|------|---------|------------|------|--------|
| 2B-256N | 2B | 256 | SophiaG | 2.28e-5 | on | 1431 | 6.13 | Continuing |
| 20B-256N | 20B | 256 | SophiaG | 2.28e-5 | on | 581 | 7.35 | Continuing |
| 20B-512N | 20B | 512 | SophiaG | 2.28e-5 | on | 458 | 7.09 | Continuing |
| 80B-256N | 80B | 256 | AdamW | 1.1e-5 | on | 0 | — | Retry queued |
| 80B-512N | 80B | 512 | AdamW | 1.1e-5 | off | 0 | — | Resubmit queued |
| 2B-512N | 2B | 512 | SophiaG | 2.28e-5 | off | 0 | — | Resubmit queued |

## Known Issues

1. **torch.compile OOM at 512N** — 2B OOMs on GPU, 80B OOMs on CPU. Use
   `--compile.no-enable` for 512N jobs.
2. **SophiaG/Muon broken at 80B** — bf16 overflow in Hessian/Newton-Schulz.
   Use AdamW with LR=1.1e-5.
3. **Transient segfaults** — single bad nodes crash the whole job. Retry
   usually works.
4. **Compile time at 256N** — ~7-15 min depending on model size. Eats into
   the 12h walltime.

## Job Chain Reference

```
2B-256N:  8446337 → 8446338 → 8446339
20B-256N: 8446340 → 8446341 → 8446342
20B-512N: 8446343 → 8446344
80B-256N: 8446345 → 8446346
80B-512N: 8446347 → 8446348
2B-512N:  8446349 → 8446350
```
