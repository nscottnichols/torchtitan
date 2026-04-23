# Production Training — agpt 2B

## 2B @ 256N — SophiaG LR=2.28e-5

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

### Progress

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8444122 | 1–1431 | 12.94 → 6.13 | 489 | 1.8% | 47.12 GiB | Walltime (12h) |
| 8446337 | 1431→ | — | — | — | — | Queued |
| 8446338 | cont. | — | — | — | — | Queued (dep) |
| 8446339 | cont. | — | — | — | — | Queued (dep) |

**W&B:** [pjanidnw](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pjanidnw)

**Tokens consumed:** 1431 × 3072 × 8192 = **36.0B tokens** (0.8% of target)

**Note:** Low MFU (1.8%) is because the 2B model is too small for 3072-way
FSDP — communication dominates. TPS/GPU (489) is much lower than the 2N
baseline (5400). This is expected from the scaling study results.

---

## 2B @ 512N — SophiaG LR=2.28e-5 (no compile)

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

### Progress

| Job ID | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|-------|------|---------|-----|--------|--------|
| 8443818 | 0 | — | — | — | — | OOM (compile) |
| 8446349 | 0→ | — | — | — | — | Queued (no-compile) |
| 8446350 | cont. | — | — | — | — | Queued (dep) |

### Job Chains

```
2B-256N: 8446337 → 8446338 → 8446339
2B-512N: 8446349 → 8446350
```
