# Production Training Runs — Aurora

> **Living document** — updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-05-22

## Scaling Performance

- [`scaling-performance.md`](scaling-performance.md) — detailed
  experiment log from Apr 18-21 (compile scaling, 80B at 4-512N,
  interactive workflow validation).
- [`docs/scaling/yeet_env/`](../scaling/yeet_env/README.md) —
  yeet-env tarball broadcast scaling (8N to 4096N) on Aurora.

## Overview

Full-scale production training of AuroraGPT models on the
[olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) dataset
(4.67T tokens) across Aurora compute nodes.

**Restarted on 2026-04-30 (v2)** after discovering the bf16-master
RMSNorm-freeze bug. All current production runs use `dtype=float32`
master weights, plain CrossEntropyLoss, LBS=2 with the torch 2.13 venv
(yeet-env tarball mode). See
[`docs/guides/training-dtype-bf16-norm-freeze.md`](../guides/training-dtype-bf16-norm-freeze.md)
for the diagnosis.

## Active Runs

### Canonical chains (one per model)

| Model | Nodes | Cumulative steps | Loss | Tokens | Latest job | Status |
|-------|------:|-----------------:|-----:|-------:|------------|--------|
| 2B  | 512 | **13,400** | **2.79** | **1.35T** (28.9%) | [`8485509`](agpt/2b/n512/README.md#log-8485509) + [`8485511`](agpt/2b/n512/README.md#log-8485511) | Both walltime-finished after ~1h20m each; chain pinned at step-13400 across both runs (possible ckpt-save loop — investigate). |
| 20B | 512 | **1,000**  | **3.34** | **101B** (2.2%)   | [`8481645`](agpt/20b/n512/README.md#log-8481645) | **+200 fresh steps** (failover wrapper, 522 nodes). Killed by bad node `10.115.76.36` @ 3h26m; step-1000 ckpt saved. 8481647 Q to resume. |
| 80B | 512 | — | — | — | [`8485512`](agpt/80b/n512/README.md#log-8485512) | **Q** — fresh 80B with all 4 failover fixes live; **`8503077` Q at 2058N stress test** (2048 active + 10 spare). |

> **Failover wrapper validated 2026-05-21**: [`8481646`](agpt/20b/n256/README.md#log-8481646) (20B 256N) hit a real Aurora gloo crash, the wrapper auto-detected the bad node, swapped in a spare, and retried — **first end-to-end production proof of the swap-and-retry path**. See [failover writeup](../experiments/agpt/aurora/20260521-failover-validated-8481646.md).

### Active 256N trajectories

| Model | Nodes | Cumulative steps | Loss | Tokens | Latest job | Status |
|-------|------:|-----------------:|-----:|-------:|------------|--------|
| 2B  | 256 | **12,889** | **2.81** | **650B** (13.9%) | [`8470101`](agpt/2b/n256/README.md#log-8470101) | Done (walltime, 12h00m20s; cleanly walltime-finished) — caught up to canonical 512N per-step |
| 20B | 256 | **500** | **4.08** | 25B (0.54%) | [`8479581`](agpt/20b/n256/README.md#log-8479581) | **Crashed** (gloo TCP timeout @ 3h39m); 8479582 Q to resume from step-500 |

### Other jobs

| Job ID | Date | Model | Nodes | Walltime | Status |
|--------|------|-------|------:|---------:|--------|
| [`8463182`](agpt/2b/n1024/README.md#log-8463182) | 2026-05-04 | 2B | 1024 | 12h | **Crashed @ startup (211s, std::bad_alloc)** |
| [`8463183`](agpt/20b/n1024/README.md#log-8463183) | 2026-05-04 | 20B | 1024 | 12h | **Crashed @ startup (211s, SIGSEGV)** |
| [`8463659`](agpt/20b/n256/README.md#log-8463659) | 2026-05-04 | 20B | 256 | 12h | **NODE_FAIL** after step 364 (loss 4.61); step-300 ckpt saved |
| [`8466848`](agpt/20b/n512/README.md#log-8466848) | 2026-05-07 | 20B | 512 | — | **Crashed @ startup** (`set_determinism` `std::bad_alloc`); didn't reproduce on 8479579 retry |
| [`8467141`](agpt/2b/n512/README.md#log-8467141)/[`8467142`](agpt/2b/n512/README.md#log-8467142) | 2026-05-07/11 | 2B | 512 | 12h | √2-LR fork — chain1 ran 4h, chain2 ran 1h53m; both done. Tests `LR=3.22e-5` at GBS=12,288 (separate ckpt dir `gbs12288-lr3.22e-5`) |
| [`8470102`](agpt/20b/n256/README.md#log-8470102)/[`8470103`](agpt/20b/n256/README.md#log-8470103) | 2026-05-08 | 20B | 256 | — | Both **crashed** with gloo TCP timeouts at ~3h elapsed |

### Dense (agpt) — bf16-tainted (superseded, kept for record)

See per-model READMEs (`agpt/2b/`, `agpt/20b/`, `agpt/80b/`).

### MoE

| Run | Model | Nodes | Status |
|-----|-------|------:|--------|
| [10B_2B EP=12](moe/10b_2b_sdpa_ep/) | 10B_2B_sdpa | TBD | Planned |

## Known Issues

1. **bf16-master RMSNorm freeze (RESOLVED 2026-04-30):** Default
   `training.dtype` flipped from `bfloat16` to `float32` after we
   discovered RMSNorm.weight was frozen at 1.0 by sub-ULP updates at
   bf16. v2 runs fix this; checkpoints from before the fix are
   tainted. See [bf16-norm-freeze guide](../guides/training-dtype-bf16-norm-freeze.md).
2. **NODE_FAIL at end-of-walltime is common** — both v2 2B runs hit
   NODE_FAIL after 6 hours of clean training, with TPS dragging from
   ~5K → ~30 in the final few hundred steps before kill. Single bad
   node taking down the whole job. Mitigation: keep_latest_k=0 (keep
   all ckpts) so `step-N00` snapshots survive the failure.
3. **torch.compile OOM at 512N** — 2B OOMs on GPU, 80B OOMs on CPU.
   Use `--compile.no-enable` for 512+ node jobs.
4. **SophiaG/Muon broken at 80B** — bf16 overflow in Hessian/Newton-Schulz.
   Use AdamW only at 80B.
5. **80B AdamW LR=1.1e-5 → NaN** — loss diverges at step 138 (256N) and
   step 15 (512N). Pending v2 restart with LR=1e-6.
6. **yeet-env saturates flare at 512N (RESOLVED via tarball mode)** —
   the per-file rsync mode used to take hours and saturate Lustre. The
   tarball mode (`ezpz yeet-env --src .venv.tar.gz`, default in v2
   submit scripts) does the same broadcast in 70-420 seconds at
   8-2048N. See [yeet_env scaling](../scaling/yeet_env/README.md).
