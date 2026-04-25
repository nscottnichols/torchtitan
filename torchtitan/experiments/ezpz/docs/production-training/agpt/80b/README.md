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
| 8446345 | 1–685+ | 12.94 → NaN (step 138) | 92 | 16.8% | 52.94 GiB | **Running** (NaN) |
| 8446346 | cont. | — | — | — | — | Held (dep) |

**W&B:** [47pxgzf3](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/47pxgzf3)

**Issue:** Loss went NaN at step 138. Loss was 12.00 → NaN. The model
trained for 137 steps with good loss convergence (12.94 → 12.00) and
87 TPS / 16% MFU before diverging.

**LR finder at 256N (2026-04-25):** Ran with init_lr=1e-9, max_lr=1e-4.
NaN appeared at step 30 (LR=3e-8). This is puzzlingly low — the
production run survived 137 steps at LR=1.1e-5. The NaN may be
data-dependent (bad batch) rather than LR-dependent.

**Investigation in progress:**
- Testing fixed LR=1e-6 for 200 steps on fresh 256N allocation (8451155)
- If this also NaNs, the issue is likely bf16 overflow in the gradient
  computation itself (same root cause as SophiaG/Muon at 80B)

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
| 8446348 | 1–495+ | 12.94 → NaN (step 15) | 65 | 11.9% | 56.33 GiB | **Running** (NaN) |

**W&B:** [mbszs7ij](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mbszs7ij)

**Issue:** Loss went NaN at step 15. Only 14 steps converged before
diverging. Same LR issue as 256N but worse — higher GBS (3072) makes
the effective learning rate even more aggressive.

### Job Chains

```
80B-256N: 8446345 → 8446346
80B-512N: 8446347 → 8446348
```
