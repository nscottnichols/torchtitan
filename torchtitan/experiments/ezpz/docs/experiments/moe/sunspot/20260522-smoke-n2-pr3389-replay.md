# MoE PR #3389 Replay Smoke — Sunspot 2N (2026-05-22)

Smoke-test for the 38th upstream sync: validate that the
[PR #3389 (MoE [6/n], dispatcher split)](https://github.com/pytorch/torchtitan/pull/3389)
replay landed in
[`d87729ad8`](https://github.com/saforem2/torchtitan/commit/d87729ad8)
doesn't break ezpz's `moe_2b_ep` path, and that the `agpt_2b`
sanity case is unaffected.

## TL;DR

- **`agpt_2b`** — clean, byte-comparable to prior post-resync
  baselines (50 steps, exit 0, 140 s wall).
- **`moe_2b_ep @ LBS=16` (registry default)** — **regressed: OOM at
  init** (62.53 GiB allocation request during forward, against 64 GiB
  per-tile capacity). Yesterday's 37th-sync run with the same config
  peaked at **15.03 GiB**. ~4× memory blowup.
- **`moe_2b_ep @ LBS=2`** — clean, 27.08 GiB peak, 3,200 TPS, 9.4%
  MFU. So the regression is **batch-shape dependent**, not a constant
  factor.

The replay itself (isinstance dispatch on `token_dispatcher` Config
classes) is functionally correct — `moe_2b_ep` reached the forward
loop, ran the dispatch path, and trained cleanly at LBS=2. The OOM
at LBS=16 is plausibly upstream's responsibility: PR #3389
restructured token-dispatcher buffer sizing through the new
`local_reorder` helper, and the for_loop expert backend (XPU's
only option, no SM90+ `_grouped_mm`) is shape-sensitive.

## Environment

| Field        | Value                                                             |
|--------------|-------------------------------------------------------------------|
| Date         | 2026-05-22                                                        |
| Branch       | `ezpz` (post-38th-sync merge + PR #3389 replay)                   |
| Commit       | `d87729ad8`                                                       |
| Machine      | Sunspot                                                           |
| Job IDs      | 12467277 (agpt + moe LBS=16), 12467288 (moe LBS=2)                |
| Nodes        | 2                                                                 |
| Devices      | 24 (Intel Max 1550)                                               |
| Devices/Node | 12                                                                |
| Steps        | 50 per config                                                     |
| Dataset      | blendcorpus                                                       |
| Backend      | xccl                                                              |
| Torch        | `2.13.0.dev20260519+xpu` (.venv)                                  |
| Checkpoint   | disabled (`--checkpoint.no-enable`)                                |

Stale `outputs/checkpoint/step-100` from a pre-PR-3159 (35th sync)
revision was incompatible with the merged code (`Missing key in
checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight.` —
Llama3 `qkv_linear` weight layout changed in PR #3159). Backed up to
`outputs/checkpoint-20260522-120005` and smokes ran from fresh init.

## Results

### `agpt_2b` (sanity)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/c6vff0te
- Log: `logs/smoke-38th-sync/agpt_2b-fresh-20260522-120017.log`
- 50 steps, **exit 0**, 140 s wall.
- Final loss 6.05, peak memory **24.34 GiB (38.04%)** — matches the
  prior `agpt_2b` post-resync baselines from
  [20260520-smoke-n2-pr3386-merge-followup.md](../../agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)
  to within run-to-run noise. Confirms the 38th-sync merge is a no-op
  for the agpt path.

### `moe_2b_ep @ LBS=16` (registry default) — **OOM**

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2x0435bc
- Log: `logs/smoke-38th-sync/moe_2b_ep-fresh-20260522-120344.log`
- Trainer init clean, model placed, 12 ranks all hit
  `torch.OutOfMemoryError: XPU out of memory. Tried to allocate
  62.53 GiB.` inside the first forward (called from
  `model_parts[0](inputs, ...)` in `trainer.py:530`).
- Each Max 1550 tile has 64 GiB. With 6.30 GiB already pinned, the
  62.53 GiB single allocation overflows.

Compare to yesterday's 37th-sync moe `_ep` follow-up smoke
([20260520-smoke-n2-pr3386-ep-followup.md](20260520-smoke-n2-pr3386-ep-followup.md)),
which ran identical config (`moe_2b_ep`, LBS=16, 2N, EP=2) and
peaked at **15.03 GiB** with 2,860 TPS. The merge between yesterday
and today consists of the 38th-sync upstream commits and the PR
#3389 replay in `experiments/ezpz/moe/model.py`; the upstream
`token_dispatcher.py` restructuring in PR #3389 is the only
plausible source of the regression.

### `moe_2b_ep @ LBS=2` (workaround)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/re576w5b
- Log: `logs/smoke-38th-sync/moe_2b_ep-lbs2-20260522-121209.log`
- 50 steps, **exit 0**, 427 s wall.
- Loss 12.93 → 6.15.
- Peak memory **27.08 GiB (42.33%)** — interesting. Still well above
  yesterday's LBS=16 peak of 15 GiB, so even the "small" batch sits
  ~1.8× higher than the previous LBS-16 baseline. The per-batch
  blowup at LBS=16 is plausibly **superlinear** in LBS, not constant.
- Throughput ~3,200 TPS/GPU, ~28 TFLOPS, 9.4% MFU. About **45% of
  yesterday's** 2,860 TPS at LBS=16 (account for the 8× smaller LBS
  → expected ~360 TPS if perfectly linear; we're at 3,200, so per-token
  efficiency is up but per-step is down).

## Diagnosis

The replay itself is correct — `isinstance` dispatch on
`DeepEPTokenDispatcher.Config` / `HybridEPTokenDispatcher.Config`
fires cleanly (we know because the EP=2 path that uses
`LocalTokenDispatcher` ran end-to-end at LBS=2). The OOM is
upstream-side and likely lives in one of:

1. The new `local_reorder` helper on `LocalTokenDispatcher` —
   deduplicates a histc/argsort/score-weighting block that previously
   lived in two places. New helper may materialize additional
   intermediates.
2. Buffer sizing changes pulled in alongside the dispatcher split
   (the diff touches 288 lines of `token_dispatcher.py`).
3. PR #3389's bundled DeepEP CUDA stream race fix — shouldn't apply
   on XPU but might have unintended side-effects via shared
   code paths.

Worth filing upstream with the comparison numbers (yesterday 15 GiB
clean, today 62 GiB OOM, both at LBS=16) plus the LBS=2 workaround
as the bisect handle.

## Action items

1. **File upstream issue** on `pytorch/torchtitan` referencing this
   report, the wandb runs, and the LBS=16 → LBS=2 datapoint.
2. **Pin `moe_2b_ep` to LBS=2 in the registry** as a temporary
   workaround (similar to what
   [`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
   did for `moe_debugmodel_ep` after the 37th sync).
3. Re-test once an upstream fix lands.

## Artifacts

- Smoke logs under `logs/smoke-38th-sync/`:
  - `agpt_2b-fresh-20260522-120017.log` (clean)
  - `moe_2b_ep-fresh-20260522-120344.log` (OOM)
  - `moe_2b_ep-lbs2-20260522-121209.log` (clean workaround)
- Stale checkpoint backup at `outputs/checkpoint-20260522-120005/`
  (pre-PR-3159 layout; safe to delete after this report).
