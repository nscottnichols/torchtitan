# Production Training Runs — Aurora

> **Living document** — updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-05-27

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
| 2B  | 512 | **27,100+** (persisted) | **2.72** | **2.73T** (58.4%) | [`8508753`](agpt/2b/n512/README.md) R (sync-mode) | **🏁 Sync-mode workaround holding cleanly.** Chain has advanced step 13,300 → 27,100 (+13,800) since 2026-05-24, ~107 ckpts persisted across 8506221 + 8507196 (pals-RPC infra failure, separate issue) + 8507199 + 8508753. |
| 20B | 512 | **3,270** (persisted) | **2.65** | **329B** (7.0%) | [`8508214`](agpt/20b/n512/README.md) Q (sync-mode) | **🏁 20B 512N sync now beats 2B 256N async on every benchmark per token.** Chain has advanced step 800 → 3,270 (+2,470) since 2026-05-24 across 8505258 + 8505259 + 8507197 + 8507200. ARC-Easy 0.665, HellaSwag norm 0.574 at step-3,200. |
| 80B | 256 | — | — | — | [`8505222`](agpt/80b/n512/README.md) F (5 retries exhausted) | All dispatches since 2026-05-11 fail. Two distinct failure modes documented: [80B SIGSEGV cascade](../experiments/agpt/aurora/20260524-80b-256n-sigsegv-cascade-8505222.md) (production scale) + [blendcorpus EOFError race](../guides/known-bugs/blendcorpus-eoferror-race.md) (8N smoke). Mitigations identified, not yet retested. |

> **Failover wrapper production-validated 2026-05-23**: [`8505298`](agpt/2b/n256/README.md) (2B 8N smoke) caught a real silent hang at step 37, watchdog tripped, blind-swapped the bad node, attempt-2 recovered cleanly + persisted DCP checkpoints. **First end-to-end real-world validation of the swap-and-retry path on a true silent-hang failure.** See [incident report](../experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md).

### Active 256N trajectories

| Model | Nodes | Cumulative steps | Loss | Tokens | Latest job | Status |
|-------|------:|-----------------:|-----:|-------:|------------|--------|
| 2B  | 256 | **49,900+** (persisted) | **2.68** | **2.50T** (53.5%) | [`8508020`](agpt/2b/n256/README.md) R | Async-mode stable at 256N across 9 dispatches since 2026-05-23. Chain has advanced step 25,500 → 49,900 (+24,400 in 4 days). Plateau in eval scores around ARC-Easy 0.645 / HellaSwag norm 0.547 — model has saturated on this LR/data mix. |
| 20B | 256 | **1,125** (persisted) | **3.28** | **113B** (2.4%) | [`8505255`](agpt/20b/n256/README.md) F (12h walltime) | Sync-mode 12h dispatch reached step 1,125 cleanly. No continuation queued (256N is per-token comparator; canonical chain is 512N). |

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
7. **Async checkpoint save is being killed mid-write by bad-node
   crashes** (discovered 2026-05-22 during eval refresh). Every recent
   20B run *logs* progress past the latest persisted ckpt (e.g. 8481645
   logged step 1000 but step-900 ckpt dir is empty; 8481646 logged
   step 500 but no ckpt past step-300 has finalized in 3 weeks). The
   `--checkpoint.async-mode=async` flag lets training continue while
   the save streams to flare in the background; if the bad-node crash
   fires during that window, the partially-written `step-N00/` dir
   stays in place but lacks `.metadata` and `__*_0.distcp` shards,
   making it unloadable. **Mitigation:** consider switching to
   `--checkpoint.async-mode=sync` for at least one save per chain
   continuation, OR detect and `mv` the orphaned ckpt dir before next
   training start (the resume code falls back to the previous step).
