# Production Training — agpt 80B

## v2 — first production attempt submitted 2026-05-11

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 512N**](n512/README.md) (canonical chain attempt) | **8480361 Q** (522 nodes via failover wrapper, 512 active + 10 spare) | — | — | — |

Working config (proven in 4N smoke 12466025 on 2026-05-05): AdamW
LR=1e-6, TP=2, AC=full, compile=OFF, fp32-master. Loss descended
cleanly 12.98 → 10.46 over 20 steps in the smoke. Submit script:
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../../scripts/submit_agpt_80b_aurora_venv_failover.sh).

Production clone: `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/`.

---

<details>
<summary><strong>v1 — 80B history (NaN'd, kept for record) — click to expand</strong></summary>

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

---

## 80B @ 256N — AdamW LR=1e-6

| Field | Value |
|-------|-------|
| Model | agpt_80b (80.8B params) |
| Nodes / GPUs | 256 / 3,072 |
| Parallelism | TP=2, FSDP=1536 |
| Compile | on |
| Optimizer | AdamW, LR=1e-6 |
| GBS | 1,536 (LBS=1) |

### Progress

| Job ID | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|-------|---------------------|---------|-----|--------|--------|
| 8451225 | 1–51 | 12.94 → 12.91 | — | — | — | Complete (walltime) |
| 8451226 | 0 | — | — | — | — | Crashed (Gloo timeout, bad node) |

**Note:** LR=1e-6 stabilized the 80B model (51 steps without NaN).
Job 8451226 crashed during dataloader init due to an unreachable node
(`Gloo connectFullMesh failed ... No route to host`). Needs resubmit.

### Job Chains

```
80B-256N (LR=1.1e-5): 8446345 → 8446346
80B-256N (LR=1e-6): 8451225 → 8451226 (crashed)
80B-512N (LR=1.1e-5): 8446347 → 8446348
80B-512N (torch 2.13, LR=1e-6): 8451727 → 8451728
```

</details>
