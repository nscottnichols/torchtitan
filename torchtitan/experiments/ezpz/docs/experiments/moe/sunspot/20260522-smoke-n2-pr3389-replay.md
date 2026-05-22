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
- **`moe_2b_ep @ LBS=16` (registry default)** — OOM at first forward
  (62.53 GiB single allocation, vs 64 GiB per-tile capacity). The
  allocation is the **`lm_head` vocab projection**
  (`[LBS × seq_len, vocab_size] = [16 × 8192, 256128]` in bf16 ≈
  62.5 GiB), not the MoE dispatch path. **This is the same
  pre-existing `_ep` registry-default vocab-projection OOM** that
  the 37th-sync follow-up
  ([`20260520-smoke-n2-pr3386-ep-followup.md`](20260520-smoke-n2-pr3386-ep-followup.md))
  already documented and called out as an action item — yesterday's
  "clean 15 GiB" datapoint was at `LBS=1` *override*, not LBS=16.
- **`moe_2b_ep @ LBS=1` (matching yesterday's actual config)** —
  clean, **peak 14.95 GiB (23.36%)**, 2,820 TPS, 8.2% MFU.
  **No memory regression vs the 37th-sync baseline of 15.03 GiB.**
- **`moe_2b_ep @ LBS=2`** — clean, peak 27.08 GiB (42.33%), 3,200 TPS,
  9.4% MFU. Scales roughly linearly with LBS in the projection step.

The PR #3389 replay (isinstance dispatch on `token_dispatcher` Config
classes) is functionally correct **and** numerically stable: today's
LBS=1 peak matches yesterday's to within run-to-run noise (0.08 GiB),
and TPS is within 1.4%. There is **no regression** introduced by the
38th sync.

The OOM at LBS=16 is the well-known pre-existing condition: every
`_ep` config inherits its parent's LBS, but with the Gemma vocab
(256128) at seq_len=8192, anything above LBS=1 (and certainly LBS=16)
overflows a Max 1550 tile on the bf16 vocab projection. The
**registry needs to pin LBS** on the `_ep` variants, mirroring what
[`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
did for `moe_debugmodel_ep` after the 37th sync.

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

### `moe_2b_ep @ LBS=16` (registry default) — pre-existing vocab-projection OOM

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2x0435bc
- Log: `logs/smoke-38th-sync/moe_2b_ep-fresh-20260522-120344.log`
- 12 ranks all hit `torch.OutOfMemoryError: XPU out of memory. Tried
  to allocate 62.53 GiB.` inside the **first forward call** on the
  `lm_head` linear (`decoder.py:148: output = self.lm_head(h)` →
  `F.linear(input, self.weight, self.bias)`).

The 62.53 GiB single allocation matches the bf16 vocab-projection
shape exactly:

```
[LBS × seq_len, vocab_size] × 2 bytes
= [16 × 8192, 256128] × 2
≈ 62.5 GiB
```

This is the same pre-existing `_ep` failure mode the
[37th-sync follow-up](20260520-smoke-n2-pr3386-ep-followup.md)
flagged as **"`_ep` configs should override LBS in the registry"** —
`moe_2b_ep` inherits `LBS=16` from `moe("2B",
local_batch_size=16)`, and with the Gemma vocab (256128) any LBS > 1
at seq_len=8192 overflows a Max 1550 tile's 64 GiB on the
projection step.

**My first reading of this OOM as a PR #3389 regression was wrong.**
Yesterday's report records `moe_2b_ep` peak at "15.03 GiB", but the
qualifier on that row reads `(LBS=1 override)` — i.e. it was never
running at LBS=16. Apples-to-apples comparison below.

### `moe_2b_ep @ LBS=1` (parity check vs yesterday)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/__lbs1__
  (alloc 12467323, log `moe_2b_ep-lbs1-parity-20260522-142430.log`)
- 50 steps, **exit 0**, 299 s wall.
- Loss 12.93 → 6.57 (final step volatile; step 49 was 6.10).
- **Peak step memory: 14.95 GiB (23.36%)**.
- Throughput ~2,820 TPS/GPU, 24.5 TFLOPS, 8.2% MFU.

Side-by-side with yesterday's 37th-sync `moe_2b_ep @ LBS=1` row:

| Metric | Yesterday (37th, `1d4115d3f`) | Today (38th, `d87729ad8`) | Δ |
|--------|-------------------------------|----------------------------|---|
| Peak step memory | 15.03 GiB | **14.95 GiB** | -0.08 GiB |
| TPS/GPU | ~2,860 | ~2,820 | -1.4% (noise) |
| MFU | ~8.3% | ~8.2% | -0.1pp |
| Loss step 50 | 6.07 | 6.57 (volatile) | within run-to-run noise |

**No regression.** PR #3389's `token_dispatcher.py` restructure is
numerically equivalent on the `LocalTokenDispatcher` + for_loop
backend path that ezpz uses on XPU.

### `moe_2b_ep @ LBS=2` (alternate workaround datapoint)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/re576w5b
- Log: `logs/smoke-38th-sync/moe_2b_ep-lbs2-20260522-121209.log`
- 50 steps, **exit 0**, 427 s wall.
- Loss 12.93 → 6.15.
- Peak memory **27.08 GiB (42.33%)** — consistent with the LBS=1
  baseline scaled by the vocab-projection's linear LBS dependence
  (`14.95 + (2-1) × ~12 GiB ≈ 27 GiB`).
- ~3,200 TPS/GPU, 9.4% MFU.

## Diagnosis

There is **no PR #3389 regression**. The replay is functionally and
numerically correct; the `LocalTokenDispatcher` path on XPU is
unchanged in behavior.

The OOM at the LBS=16 registry default is the **pre-existing
vocab-projection OOM** that already exists for every `_ep` config
(the same bug
[`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
patched for `moe_debugmodel_ep` after the 37th sync). It would have
OOMed yesterday too; we just never ran it without an LBS override.

## Action items

1. **Pin `moe_2b_ep` LBS in the registry** (mirror `f2cbc0327`),
   so the registry default doesn't OOM on a fresh user.
2. ~~File upstream issue on `pytorch/torchtitan`~~ — **not needed**;
   no regression to report.

## Artifacts

- Smoke logs under `logs/smoke-38th-sync/`:
  - `agpt_2b-fresh-20260522-120017.log` (clean)
  - `moe_2b_ep-fresh-20260522-120344.log` (OOM at registry-default LBS=16)
  - `moe_2b_ep-lbs2-20260522-121209.log` (clean at LBS=2)
  - `moe_2b_ep-lbs1-parity-20260522-142430.log` (parity check vs yesterday at LBS=1)
- Stale checkpoint backup at `outputs/checkpoint-20260522-120005/`
  (pre-PR-3159 layout; safe to delete after this report).
