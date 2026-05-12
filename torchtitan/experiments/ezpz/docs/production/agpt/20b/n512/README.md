# Production Training — agpt 20B @ 512 nodes

> **This is the canonical 20B production chain.**
>
> **Status (2026-05-11 night):** Stalled again. 8479579 reached step
> 803 then **silently hung** for 5h before being killed manually
> (no exit, no crash — W&B heartbeat continued unchanged). New
> failure mode not handled by the failover wrapper; see incident
> report at
> [`docs/experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md`](../../../../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md).
> 8479580 auto-released from hold and is Q to resume from step-800.
>
> **Eval scores:** see [`docs/evals/agpt/20b/`](../../../../evals/agpt/20b/README.md)
> for the v1-vs-v2 lm-eval comparison. ARC-Easy lifted **0.271 → 0.444**
> across steps 100→800 vs v1's flat ~0.27 baseline. HellaSwag norm
> +3pp above v1 by step 800.

## v2 — 20B @ 512N — SophiaG LR=2.28e-5 (fp32 master)

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/` |
| Submit script | [`scripts/submit_agpt_20b_aurora_venv.sh`](../../../../../scripts/submit_agpt_20b_aurora_venv.sh) (one script handles all v2 node counts via env vars) |
| Stack | torch 2.13 venv (yeet-env tarball mode) |
| Optimizer | SophiaG, LR=2.28e-5 |
| Compile | on |
| GBS | 12,288 (LBS=2) |
| Total steps | 46,429 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288` |
| W&B | [9tsyx5us](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/9tsyx5us) |

### Loss / Throughput / MFU

![20B v2 512N Training](figures/production_20b_v2_512n.png)

### Diagnostics

![20B v2 512N Diagnostics](figures/training_diagnostics_20b_v2_512n.png)

### Tokens vs Wall Clock

![20B v2 512N Tokens vs Time](figures/tokens_vs_time_20b_v2_512n.png)

### Progress (chain)

| Job ID | Date | Walltime | Steps | Loss (start → end) | TPS/GPU | MFU | Status |
|--------|------|---------:|------:|-------------------:|--------:|----:|--------|
| [`8460302`][8460302] | 2026-05-01 | 6h | 1–300 | 12.94 → 4.95 | ~355 | ~17.7% | Walltime hit (NODE_FAIL at end). 3 ckpts saved. |
| [`8463628`][8463628] | 2026-05-03 | 12h | 200–863 | 5.62 → **3.46** | ~355 | ~17.8% | Done (walltime, step-100..800 ckpts saved). |
| [`8466848`][8466848] | 2026-05-07 | — | — | — | — | — | **Crashed @ startup** (127s) — `MemoryError: std::bad_alloc` in `torch.distributed.broadcast` during `set_determinism`. Intermittent: didn't reproduce on retry. |
| [`8479579`][8479579] | 2026-05-11 | 12h | 800–803 | 3.46 → 3.53 | ~340 then 0 | ~17.6% then 0 | **Killed by qdel @ 5h56m** — silent hang after step 803 (logged 13:30, no further training output through 18:23). W&B heartbeat continued unchanged for 5h. New failure mode (no exit, no crash, no traceback). See [`docs/experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md`](../../../../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md). |
| [`8479580`][8479580] | 2026-05-11 | 12h | 800+ | — | — | — | **Queued** — auto-released from hold by `afterany:8479579`, will resume from step-800 |

**Latest checkpoint:** step-800 (244 GB on disk per ckpt)

**Cumulative steps:** 803 (8479579 advancing live)

**Tokens consumed:** 803 × 12,288 × 8,192 = **81B tokens** (1.7% of 4.67T target)

**Logs:**

- `8460302`: `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512.o8460302`
- `8463628`: `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain1.o8463628`
- `8466848`: `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain2.o8466848` (startup crash)
- `8479579`: `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain2.o8479579` (running)
- `8479580`: held (`afterany:8479579`)

> **Note on 8466848 crash:** `set_determinism` calls `torch.distributed.broadcast(seed_tensor, src=0)` and one rank hit `std::bad_alloc`. This is the same failure mode that killed both 1024N attempts (8463182, 8463183) on 2026-05-04 — but at 6,144 ranks (512N) instead of 12,288 (1024N). The previous 20B 512N run (8463628, 4 days earlier) succeeded at the same scale and same script, and so does the resubmit (8479579), so it's intermittent. See [`memory/project_1024n_init_crash.md`](.) — that memory's "1024N only" claim is stale; the bug fires unpredictably at 512N+ but is not reliably triggered.

---

<details>
<summary><strong>v1 — 20B @ 512N — SophiaG LR=2.28e-5 (bf16 master, BROKEN) — click to expand</strong></summary>

This run is kept for the record. It uses `--training.dtype=bfloat16`
and has **frozen RMSNorm weights** — see
[`docs/guides/training-dtype-bf16-norm-freeze.md`](../../../../guides/training-dtype-bf16-norm-freeze.md).
Don't draw conclusions from these loss curves.

| Field | Value |
|-------|-------|
| Model | agpt_20b (20.7B params) |
| Submit script | [`submit/aurora/submit_agpt_20b_n512.sh`](../../../../../submit/aurora/submit_agpt_20b_n512.sh) (v1 torch 2.10 layout) |
| Nodes / GPUs | 512 / 6,144 |
| Parallelism | TP=1, FSDP=6144 |
| Compile | on |
| Optimizer | SophiaG, LR=2.28e-5 |
| GBS | 6,144 (LBS=1) |
| Total steps | 92,859 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs6144` |

### Loss / Throughput / MFU (512N)

![20B v1 512N Training](figures/production_20b_v1_512n.png)

### Progress

| Job ID | Date | Steps | Loss (start → end) | TPS/GPU | MFU | Memory | Status |
|--------|------|-------|---------------------|---------|-----|--------|--------|
| [`8443819`][8443819] | 2026-04-22 | 1–458 | 12.94 → 7.09 | 41 | 2.1% | 54.14 GiB | Complete (walltime) |
| [`8446343`][8446343] | 2026-04-25 | 0 | — | — | — | — | Segfault (signal 11) |
| [`8446344`][8446344] | 2026-04-26 | — | — | — | — | — | Queued |

**W&B:** [8of5hse0](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/8of5hse0)

**Latest checkpoint:** step-400

**Tokens consumed:** 458 × 6144 × 8192 = **23.1B tokens** (0.5% of target)

**Note:** Very low TPS (41) — compile took most of the 12h walltime.
512N continuation (8446343) segfaulted on a bad node. 8446344 will retry.

**Logs:**

- `8443819`: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8443819`
- `8446343`: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8446343`
- `8446344`: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8446344`

### Job Chains (historical)

```
20B-256N (torch 2.10, LBS=1): 8446340 → 8446341 → 8446342 → 8451749 → 8451751
20B-512N (torch 2.10, LBS=1): 8446343 → 8446344
20B-512N (torch 2.13, LBS=2): 8451725 → 8451726 (killed — yeet-env saturated flare)
```

</details>

<!-- Job-ID reference-style link definitions -->

<!-- v2 chain -->
[8460302]: /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512.o8460302
[8463628]: /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain1.o8463628
[8466848]: /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain2.o8466848
[8479579]: /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-chain2.o8479579
[8479580]: /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/

<!-- v1 (historical) -->
[8443819]: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8443819
[8446343]: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8446343
[8446344]: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/agpt-20b-sophiag-n512.o8446344
