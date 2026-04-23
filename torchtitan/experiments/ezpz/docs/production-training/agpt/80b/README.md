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

| Job ID | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|-------|------|---------|-----|--------|--------|
| 8443820 | 0 | — | — | — | — | CPU OOM (compile) |
| 8446347 | 0→ | — | — | — | — | Queued (no-compile) |
| 8446348 | cont. | — | — | — | — | Queued (dep) |

**Expected:** 66 TPS/GPU, 12% MFU without compile (from 512N test).

### Job Chains

```
80B-256N: 8446345 → 8446346
80B-512N: 8446347 → 8446348
```
