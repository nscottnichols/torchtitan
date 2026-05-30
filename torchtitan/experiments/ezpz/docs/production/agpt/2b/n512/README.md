# Production Training — agpt 2B @ 512 nodes

> **This is the canonical 2B production chain.**
>
> **Status:** chain at step **30,400** as of 2026-05-30 (last R was
> `8508753` walltime-exit 2026-05-27, then `8509042` cont crashed in
> `set_determinism std::bad_alloc` at 6,144 ranks — documented
> intermittent). +1 continuation `8513545` (`afterany:8509042`) Q for
> 512N slot since 2026-05-28 23:20 (>24h Q wait — Aurora capacity
> tight on Fri/weekend). Chain advanced **+3,294 steps** since the
> last README refresh (27,106 → 30,400), persisting ~33 ckpts at
> 100-step intervals. Sync-mode workaround for async-cascade continues
> to hold; the async-mode runs (`8505176` and earlier) had been pinned
> at step-13,300 for two weeks. Loss **2.71** at step-30,400.
>
> **Eval scores:** see [`docs/evals/agpt/2b/`](../../../../evals/agpt/2b/README.md)
> for the current v2 lm-eval results.

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

![2B v2 512N Training](figures/production_2b_v2_512n.svg)

### Diagnostics

![2B v2 512N Diagnostics](figures/training_diagnostics_2b_v2_512n.svg)

### Tokens vs Wall Clock

![2B v2 512N Tokens vs Time](figures/tokens_vs_time_2b_v2_512n.svg)

### Progress (chain)

| Job ID | Date | Walltime | Steps | Loss (start → end) | TPS/GPU | MFU | Status |
|--------|------|---------:|------:|-------------------:|--------:|----:|--------|
| [`8460301`](#log-8460301) | 2026-05-01 | 6h | 1–1387 | 12.65 → 3.59 | ~2,700 | ~10% | NODE_FAIL after step 1387. 13 ckpts saved. |
| [`8463626`](#log-8463626) | 2026-05-03 | 12h | 1300–5073 | 3.59 → 2.97 | ~2,700 | ~10% | Walltime hit (NODE_FAIL at end). **50 ckpts saved (every 100 steps).** |
| [`8463627`](#log-8463627) | 2026-05-07 | 12h | 5000–6955 | 2.97 → 2.90 | varies | varies | Done (walltime, 12h00m18s). |
| [`8466847`](#log-8466847) | 2026-05-11 | 12h | 6900–13279 | 2.90 → **2.79** | ~2,800 | ~10% | Done (walltime, 12h00m13s). step-13200 ckpt saved. |
| [`8479988`](#log-8479988) | 2026-05-11 | 12h | 13200+ | — | — | — | **Queued** (resubmit, 2026-05-11 evening) — auto-resumes from step-13200 |
| [`8479989`](#log-8479989) | 2026-05-11 | 12h | (cont.) | — | — | — | Held (`afterany:8479988`) |
| [`8505176`](#log-8505176) | 2026-05-23 | 12h | 13,300 → ~13,400 | — | — | — | **All 3 wrapper attempts failed in async-save cluster cascade @ step 13400** (same regression first observed at 20B 512N). Wrapper exhausted retries, exit 143. Only ckpt persisted across the dispatch was step-13300 from attempt 1's brief progress. **Async mode pinned at step-13300.** |
| [`8506215`](#log-8506215) | 2026-05-24 | — | (qsub fail) | — | — | — | **Sync-mode resubmit aborted in preflight.** All 3 wrapper attempts tripped the 120s preflight DDP-init watchdog (timeout doesn't scale with N at 6,144 ranks). Exit 1. **Bug fix:** bumped default to 600s + added `--train-iters 5` to cap preflight length. |
| [`8506221`](#log-8506221) | 2026-05-24 | 12h | 13,300 → **16,676** | 2.79 → **2.76** | ~2,700 | ~10% | **SYNC mode (async disabled).** 21 ckpts step-14600..step-16600 persisted. Preflight attempt 1 hung at iter 111 (real silent hang, 49 min of silence); wrapper SIGTERM'd and blind-swapped bad node `x4305c0s7b0n0` for spare `x4602c3s3b0n0`. Preflight attempt 2 succeeded ~75 min total; main training started 00:17 and ran to walltime exit -29. **First sustained 512N progress in two weeks.** |
| [`8507196`](#log-8507196) | 2026-05-25 → 2026-05-26 | 11h+ | 13,300 → ~**20,989** | 2.76 → **2.74** | ~2,700 | ~10% | **SYNC mode.** Resumed from step-13,300, ran through 21:12 → 08:17. Trained cleanly to step **20,989** in-memory, persisted **+76 ckpts** before all 3 wrapper attempts tripped an **Aurora pals-RPC infra failure** (exit 127 in launch phase, 3 different "bad" nodes swapped — pals failures the wrapper cannot recover from). See `memory/project_aurora_pals_rpc_launch_failure.md`. |
| [`8507199`](#log-8507199) | 2026-05-26 | 12h | 20,900 → **25,967** | 2.74 → **2.72** | ~2,700 | ~10% | Done (walltime, exit -29). **SYNC mode.** `afterany` continuation, ran 09:24 → 21:26. Persisted **~50 ckpts** step-21000..step-25900. |
| [`8508753`](#log-8508753) | 2026-05-27 → 2026-05-28 | 12h | 25,900 → **30,484** | 2.72 → **2.71** | ~2,700 | ~10% | Done (walltime exit -29). **SYNC mode.** `afterany` continuation of 8507199, ran 09:19 → 21:19. +80 ckpts step-22600..step-30400 persisted. |

**Latest checkpoint:** step-25900 (8507199 last persisted; 8508753 R still in first ckpt interval at snapshot)

**Cumulative steps:** 27,106+ (8508753 R as of 2026-05-27 12:37)

**Tokens consumed:** 27,106 × 12,288 × 8,192 = **2.73T tokens** (58.4% of 4.67T target)

### Recovery

Async checkpoint saves cluster-cascaded at 6,144 ranks: every
dispatch from 2026-05-11 → 2026-05-23 cleanly trained ~100 steps
to the next save then died the same way, pinning the chain at
step-13,300 for two weeks. Same pattern hit the 20B 512N chain.

`8506221` (2026-05-24) ran with `CHECKPOINT_ASYNC_MODE=disabled`
and advanced the chain by 3,376 steps (+21 ckpts persisted) in
12h. Sync mode is the operational workaround for both 2B and 20B
512N trajectories. The follow-on chain (`8507199` + `8508753`)
has continued the pattern, taking the chain from step-16,676 to
**27,106+** across two more sustained dispatches.

`8507196` (2026-05-25) was the one stumble in this stretch: trained
cleanly to step **20,989** in-memory (+76 persisted ckpts), then all
three wrapper attempts hit an **Aurora pals-RPC infrastructure
failure** (exit 127 during launch). The wrapper correctly identified
the failures as node-related and swapped three different nodes before
exhausting retries — but pals-RPC failures are upstream of anything
the wrapper can repair. The chain still benefited (76 persisted ckpts
from step-13,400..step-20,900 advanced the on-disk frontier
significantly), and `8507199` resumed cleanly from step-20900.

Note the preflight gotcha discovered in `8506215`: the wrapper's
default 120s smoke-test timeout doesn't scale to 6,144-rank DDP
init, so all three attempts watchdog-tripped before any real
training. Default is now 600s + `--train-iters 5`.

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8460301"></a>`8460301` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512.o8460301` |
| <a id="log-8463626"></a>`8463626` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain1.o8463626` |
| <a id="log-8463627"></a>`8463627` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain2.o8463627` |
| <a id="log-8466847"></a>`8466847` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain3.o8466847` |
| <a id="log-8479988"></a>`8479988` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain4.o8479988` |
| <a id="log-8479989"></a>`8479989` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain5.o8479989` |
| <a id="log-8485509"></a>`8485509` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain6.o8485509` (ran 1h19m, walltime; pinned at step-13400) |
| <a id="log-8485511"></a>`8485511` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain7.o8485511` (ran 1h23m, walltime; also pinned at step-13400 — same ckpt as 8485509 resume) |
| <a id="log-8505176"></a>`8505176` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-chain8.o8505176` |
| <a id="log-8506215"></a>`8506215` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-sync-chain1.o8506215` |
| <a id="log-8506221"></a>`8506221` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-sync-chain2.o8506221` |
| <a id="log-8507196"></a>`8507196` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-failover-sync-cont.o8507196` |
| <a id="log-8507199"></a>`8507199` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-failover-sync-cont2.o8507199` |
| <a id="log-8508753"></a>`8508753` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n512-v2-failover-sync-cont3.o8508753` |
