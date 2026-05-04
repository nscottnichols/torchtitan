# AuroraGPT Sync — Meeting Notes

> Sam Foreman. Most recent first.

---

## 2026-05-04

### TP loss-reporting bug — every TP > 1 dashboard is wrong since 2026-04-27

- Upstream `_dist_reduce` change ([`pytorch/torchtitan@1786292d`](https://github.com/pytorch/torchtitan/commit/1786292d))
  silently drops the cross-batch `all_reduce` when `loss` is a DTensor
  on a mesh orthogonal to `loss_mesh`.
- **All TP > 1 W&B loss curves are under-reported by `dp_world_size`.**
  For our 80B production at 256N TP=2 that means the logged loss is
  **1536× smaller** than the truth.
- Gradients and optimizer steps are unaffected — the actual model
  training is fine, only the printed/logged number is wrong.
- **Fix filed:**
  [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
  — mergeable, all 3 Meta CI checks passing, 8-GPU CI workflow
  awaiting maintainer approval.
- **Local workaround landed in ezpz:**
  [`a24ed2e1`](https://github.com/saforem2/torchtitan/commit/a24ed2e1)
  (trainer-side) and
  [`a0b9b13d`](https://github.com/saforem2/torchtitan/commit/a0b9b13d)
  (new `EzpzValidator` subclass + blendcorpus validation wiring).
  Production runs continuing forward get correct numbers from the
  next checkpoint.
- Full diagnosis:
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../guides/loss-reporting-tp-dist-reduce.md).
  Upstream issue notes:
  [`docs/upstream-issues/dist_reduce_dtensor_skip.md`](../upstream-issues/dist_reduce_dtensor_skip.md).

### Production status

- **2B 512N canonical chain** (`8460301 → 8463626 → 8463627`):
  step **5,073**, loss **2.97**, **510B tokens / 10.9% of 4.67T
  target**. Continuation 8463627 queued, follow-up 8466847 held on
  `afterany:8463627`. Trajectory page:
  [`docs/production/agpt/2b/n512/`](../production/agpt/2b/n512/README.md).
- **20B 512N canonical chain** (`8460302 → 8463628`): step **~862**,
  loss **~3.47**, MFU ~17.8%. 8463628 currently running near
  walltime; 8466848 held on `afterany:8463628`. Trajectory page:
  [`docs/production/agpt/20b/n512/`](../production/agpt/20b/n512/README.md).
- **20B 256N (8463659):** ran 9h walltime then **NODE_FAIL after
  step 364** (loss 4.61, 18.3B tokens). `shepherd died from signal 9`
  on `x4406c6s7b0n0`, PBS exit -20 — same recurring Aurora bad-node
  failure mode as 8459818 / 8460301. **step-300 ckpt saved cleanly,
  resumable.** Throughput on this run was bouncing 21-410 TPS
  depending on flare contention (1-20% MFU). Trajectory page:
  [`docs/production/agpt/20b/n256/`](../production/agpt/20b/n256/README.md).
  Are these recurring `signal 9` crashes being tracked anywhere?
  They've now killed three long-walltime jobs across three different
  nodes — worth raising with ALCF support if not.
- **v1 vs v2 smoking gun (validates the bf16-master RMSNorm-freeze
  fix):**
  - 2B ARC-Easy climbed **0.277 → 0.429** over 100B tokens on v2,
    vs v1's flat ~0.27 across 450B tokens.
  - **+19.8pp ARC-Easy / +15.4pp HellaSwag at 503B tokens** vs v1's
    flat baseline.
  - 2B eval comparison:
    [`docs/evals/agpt/2b/`](../evals/agpt/2b/README.md);
    20B eval comparison:
    [`docs/evals/agpt/20b/`](../evals/agpt/20b/README.md).
  - Restart-from-scratch was the right call. See
    [`docs/guides/training-dtype-bf16-norm-freeze.md`](../guides/training-dtype-bf16-norm-freeze.md)
    for the cross-linked evidence (training-loss overlays + lm-eval
    figures inline).

### 80B blocker

- 80B `compile + AC + TP=2` still hits the
  `tensors_saved_for_backwards_with_vc_check_slice` AOT autograd
  assertion. Toy minimal repro doesn't fire — bug needs the real
  `Module.parallelize` + `LocalMapConfig` path that torchtitan uses.
- **New finding from today:** added `agpt_50b_wide` (dim=9216,
  **48 layers**, ~48B params) as a smaller bisect target — commit
  [`98a02d04`](https://github.com/saforem2/torchtitan/commit/98a02d04).
  Smoke ran cleanly compile + AC + TP=2 at 95.94% memory, MFU 15%.
  **Bug does NOT reproduce at 48 layers — only at 84 (the 80B
  config).** So the upstream bug is **depth-sensitive**, not
  width/head-sensitive. Narrows the minimal-repro scope significantly.
- Initial repro attempt (toy version that does NOT fire — needs the
  new sharding API):
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../upstream-issues/repro_devicemesh_in_saved_tensors.py).
- Workaround in the meantime: `compile=OFF` for 80B (eats throughput).

### Open work I'm holding

- **Validation loss wiring:** blendcorpus's existing val split (5%
  slice) is now plumbed through `EzpzValidator` (subclass that also
  fixes the TP loss-reporting bug). Default `enable=False` so
  production isn't disturbed. Smoke test not yet done. Do we want
  held-out NLL on production runs, or are downstream lm-eval scores
  at checkpoint cadence the right signal?
- **80B production** — still has open issues from prior sessions
  (LR=1e-6 stable but bad-node Gloo timeout crash at step 51).
  Worth flagging if 80B production is on the agenda.

### Action items

- [ ] (Sam) Smoke-test `EzpzValidator` end-to-end on `agpt_2b` once
      consensus on whether to enable val.
- [ ] (Sam, blocked on review) Push for review on
      [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).
- [ ] (?) Volunteer to build LocalMapConfig-based minimal repro for
      the 80B compile + AC + TP=2 crash.
- [ ] (?) Decide cadence for held-out validation loss on production.
- [ ] (?) Decide whether to file an ALCF support ticket for the
      recurring `shepherd died from signal 9` NODE_FAIL pattern
      (jobs 8459818, 8460301, 8463659 — three crashes, three
      different nodes). Resume 8463659 from step-300 in the meantime?
