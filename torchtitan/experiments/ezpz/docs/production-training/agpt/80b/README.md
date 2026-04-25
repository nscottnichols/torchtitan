# Production Training — agpt 80B

## 80B @ 256N — AdamW LR=1.1e-5

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

### Progress

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8444124 | 0 | — | — | — | — | Segfault (node) |
| 8446345 | 1–429+ | 12.94 → NaN (step 138) | 87 | 16.0% | 52.94 GiB | **Running** (7h, NaN) |
| 8446346 | cont. | — | — | — | — | Held (dep) |

**W&B:** [47pxgzf3](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/47pxgzf3)

**Issue:** Loss went NaN at step 138. Loss was 12.00 → NaN. The model
trained for 137 steps with good loss convergence (12.94 → 12.00) and
87 TPS / 16% MFU before diverging. LR=1.1e-5 may still be too high
for 80B at GBS=1536. Consider reducing to 5e-6 or 1e-6.

**Note:** AdamW is the only viable optimizer for 80B — SophiaG and Muon
both produce NaN due to bf16 overflow at dim=9216.

---

## 80B @ 512N — AdamW LR=1.1e-5 (no compile)

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

### Progress

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8443820 | 0 | — | — | — | — | CPU OOM (compile) |
| 8446347 | — | — | — | — | — | Queued |
| 8446348 | 1–320+ | 12.94 → NaN (step 15) | 62 | 11.3% | 56.33 GiB | **Running** (7h53m, NaN) |

**W&B:** [mbszs7ij](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mbszs7ij)

**Issue:** Loss went NaN at step 15. Only 14 steps converged before
diverging. Same LR issue as 256N but worse — higher GBS (3072) makes
the effective learning rate even more aggressive.

### Job Chains

```
80B-256N: 8446345 → 8446346
80B-512N: 8446347 → 8446348
```
