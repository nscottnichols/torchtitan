# Production Training — agpt 2B @ 512 nodes

> **This is the canonical 2B production chain.** The 256N v2 ran once
> (no continuation chained); 1024N is queued but not yet started.
>
> **Eval scores:** see [`docs/evals/agpt/2b/`](../../../../evals/agpt/2b/README.md)
> for the v1-vs-v2 lm-eval comparison. v2 512N at 503B tokens beats
> v1 by **+19.8pp** on ARC-Easy and **+15.4pp** on HellaSwag.

## v2 — 2B @ 512N — SophiaG LR=2.28e-5 (fp32 master)

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/` |
| Submit script | [`scripts/submit_agpt_2b_aurora_venv.sh`](../../../../../scripts/submit_agpt_2b_aurora_venv.sh) (one script handles all v2 node counts via env vars) |
| Stack | torch 2.13 venv (yeet-env tarball mode) |
| Compile | on |
| GBS | 12,288 (LBS=2) |
| Total steps | 46,429 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288` |
| W&B chain | [i252kps9](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i252kps9) → [d4hlr8qe](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/d4hlr8qe) |

### Loss / Throughput / MFU

![2B v2 512N Training](figures/production_2b_v2_512n.png)

### Diagnostics

![2B v2 512N Diagnostics](figures/training_diagnostics_2b_v2_512n.png)

### Tokens vs Wall Clock

![2B v2 512N Tokens vs Time](figures/tokens_vs_time_2b_v2_512n.png)

### Progress (chain)

| Job ID | Date | Walltime | Steps | Loss (start → end) | TPS/GPU | MFU | Status |
|--------|------|---------:|------:|-------------------:|--------:|----:|--------|
| [`8460301`](#log-8460301) | 2026-05-01 | 6h | 1–1387 | 12.65 → 3.59 | ~2,700 | ~10% | NODE_FAIL after step 1387. 13 ckpts saved. |
| [`8463626`](#log-8463626) | 2026-05-03 | 12h | 1300–5073 | 3.59 → 2.97 | ~2,700 | ~10% | Walltime hit (NODE_FAIL at end). **50 ckpts saved (every 100 steps).** |
| [`8463627`](#log-8463627) | 2026-05-07 | 12h | 5000–6955 | 2.97 → 2.90 | varies | varies | Done (walltime, 12h00m18s). |
| [`8466847`](#log-8466847) | 2026-05-11 | 12h | 6900–13279 | 2.90 → **2.79** | ~2,800 | ~10% | Done (walltime, 12h00m13s). step-13200 ckpt saved. |
| [`8479988`](#log-8479988) | 2026-05-11 | 12h | 13200+ | — | — | — | **Queued** (resubmit, 2026-05-11 evening) — auto-resumes from step-13200 |
| [`8479989`](#log-8479989) | 2026-05-11 | 12h | (cont.) | — | — | — | Held (`afterany:8479988`) |

**Latest checkpoint:** step-13200 (8466847 saving every 100 steps)

**Cumulative steps:** 13,279

**Tokens consumed:** 13,279 × 12,288 × 8,192 = **1.34T tokens** (28.7% of 4.67T target — past the quarter mark)

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8460301"></a>`8460301` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512.o8460301` |
| <a id="log-8463626"></a>`8463626` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain1.o8463626` |
| <a id="log-8463627"></a>`8463627` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain2.o8463627` |
| <a id="log-8466847"></a>`8466847` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain3.o8466847` |
| <a id="log-8479988"></a>`8479988` | queued — log will land in `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/` on start |
| <a id="log-8479989"></a>`8479989` | held (`afterany:8479988`) |

---

<details>
<summary><strong>v1 — 2B @ 512N — SophiaG LR=2.28e-5 (bf16 master, BROKEN) — click to expand</strong></summary>

This run is kept for the record. It uses `--training.dtype=bfloat16`
and has **frozen RMSNorm weights** — see
[`docs/guides/training-dtype-bf16-norm-freeze.md`](../../../../guides/training-dtype-bf16-norm-freeze.md).
Don't draw conclusions from these loss curves.

| Field | Value |
|-------|-------|
| Model | agpt_2b (1.99B params) |
| Submit script | [`submit/aurora/submit_agpt_2b_n512.sh`](../../../../../submit/aurora/submit_agpt_2b_n512.sh) (v1 torch 2.10 layout) |
| Nodes / GPUs | 512 / 6,144 |
| Parallelism | TP=1, FSDP=6144 |
| Compile | **off** (OOM at 512N) |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 6,144 (LBS=1) |
| Total steps | 92,859 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs6144` |

### Progress

| Job ID | Date | Steps | Loss | TPS/GPU | MFU | Memory | Status |
|--------|------|-------|------|---------|-----|--------|--------|
| [`8443818`](#log-8443818) | 2026-04-22 | 0 | — | — | — | — | OOM (compile) |
| [`8446349`](#log-8446349) | 2026-04-25 | 0 | — | — | — | — | Segfault (signal 11) |
| [`8446350`](#log-8446350) | 2026-04-26 | — | — | — | — | — | Queued |

(No v1 512N training-curve figures were ever generated — the run never
got past the first step.)

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8443818"></a>`8443818` | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-2b-sophiag-n512.o8443818` |
| <a id="log-8446349"></a>`8446349` | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-2b-sophiag-n512.o8446349` |
| <a id="log-8446350"></a>`8446350` | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-2b-sophiag-n512.o8446350` |

### Job Chains (historical)

```
2B-256N (torch 2.10, LBS=1): 8446337 → 8446338 → 8446339 → 8451750 → 8451752
2B-512N (torch 2.10, LBS=1): 8446349 → 8446350
2B-512N (torch 2.13, LBS=2): 8451723 → 8451724 (killed — yeet-env saturated flare)
```

</details>
