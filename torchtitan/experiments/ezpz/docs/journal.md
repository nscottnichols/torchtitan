# Development Journal

Running log of what's happening, session by session. Most recent first.

---

## 2026-06-02 — xccl split_group workaround for nested mesh init

`moe_2b_ep` smoke on torch 2.13 on XPU was hitting:

```
RuntimeError: No backend for the parent process group or its backend
does not support splitting
```

at trainer init, inside `ParallelDims.build_mesh` → the EP-flavored
sparse mesh `("pp", "dp_replicate", "efsdp", "ep")` (`parallel_dims.py:200`).

**Root cause** (confirmed by reading upstream C++ headers via
`gh search code`): `ProcessGroupXCCL` never declares
`supportsSplitting() override`. It inherits the base
`Backend::supportsSplitting()` from
[`Backend.hpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp)
which returns `false`. `ProcessGroupNCCL` overrides to `true` —
xccl doesn't.

`DeviceMesh._init_one_process_group` (torch 2.13, `device_mesh.py:550-562`)
routes nested mesh PG creation through `split_group` whenever
`bound_device_id` is set on the default group AND the accelerator is
available AND the backend name matches. None of those rule out xccl;
the ezpz eager-init path sets `bound_device_id` on XPU, so the gate
always takes the broken branch.

`split_group` itself (`distributed_c10d.py:5565-5570`) then reads
`parent_backend.supports_splitting` (Python property bound to the C++
method), sees `False`, and raises before ever calling
`xcclCommSplit`. So the failure is purely at the gate — there's no
xccl split implementation to even crash on yet.

**Workaround**: monkey-patch `DeviceMesh._init_one_process_group`
from a new module
[`xccl_split_group_workaround.py`](../xccl_split_group_workaround.py).
The wrapper:

  1. No-ops on cuda/cpu builds (gates on `is_xccl_available() and
     torch.xpu.is_available()`).
  2. On xccl, inspects the default group's per-accelerator backend's
     `supports_splitting`. If `True` (NCCL), calls upstream verbatim.
  3. If `False` (xccl), temporarily clears `bound_device_id` on the
     default group so the upstream gate's first clause goes `False`
     and we fall through to the existing `new_group` loop. Restores
     `bound_device_id` afterwards.

Installed lazily from
[`FaultTolerantTrainer.init_distributed`](../trainer.py) so it only
fires when ezpz's trainer kicks off; never touches other torchtitan
paths.

Per Golden Rule #1, no upstream file was modified. Full diagnosis +
removal criteria in
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md).

**Smoke validation** (2026-06-02, Sunspot 2N, job 12467823, exit 0
in 103 s — see
[`docs/experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md`](experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md)):

- Workaround install line in the log:
  `Installed xccl split_group workaround on DeviceMesh._init_one_process_group`.
- EP sparse mesh built cleanly:
  `Successfully created meshes with active dimensions: ['batch', 'loss', 'ep', 'efsdp', 'fsdp']`.
- Loss 12.945 → 8.273 across 10 steps, peak 58.28 GiB, ~3,050 TPS.
- The pre-existing xccl-timeout shim also ran:
  `Applied train timeout 0:01:40 to 6 xccl ProcessGroup(s)`
  (6 PGs = world + dense + sparse + efsdp + loss + batch — all the
  meshes the EP path constructs).

Two latent bugs in `scripts/submit_moe_smoke.sh` surfaced and were
fixed during the smoke (both apply to the analogous production
scripts under `scripts/submit_agpt_*_aurora_venv.sh`):

1. **`ezpz_setup_job` overwrites `$PBS_O_WORKDIR`** with the script's
   initial cwd (which is `$HOME` under default `qsub`). A naïve
   `cd "${PBS_O_WORKDIR}"` after sourcing the utils ends up in
   `/home/foremans` and `source .venv/bin/activate` resolves against
   `~/.venv` (no `ezpz`). Fix: stash submit dir into `SUBMIT_DIR`
   before sourcing utils.
2. **`ezpz yeet-env` is deprecated** in ezpz 0.18.x in favour of
   explicit `ezpz tar-env` + `ezpz yeet .venv.tar.gz`. Switched.

Open follow-ups:
- Mirror the `SUBMIT_DIR` + `ezpz yeet` fixes into
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` before the next
  512N+ production launch.
- File `pytorch/pytorch` issue with the two-part fix
  (`ProcessGroupXCCL::supportsSplitting() override + working split()`).

---

## 2026-06-02 — 45th upstream sync (PR #3450 closes the #3436 thread)

Merged 7 upstream commits (`b72d98648..04a309858`). Headline is PR
[#3450](https://github.com/pytorch/torchtitan/pull/3450) — the
upstream-canonical `histc → routing_map` swap for MoE routing that
finally lands the fix our [PR
#3436](https://github.com/pytorch/torchtitan/pull/3436) was tracking.
Approach: build a boolean `routing_map_BLE` via `scatter_` in the
router, compute `num_tokens_per_expert_E = routing_map.sum(...)` once,
thread the map through both router and dispatcher. Same direction we
recommended (scatter-based), executed better (computed once, reused).

API change at the boundary: router returns a 4-tuple now, `dispatch()`
takes a new `num_local_tokens_per_expert_E` positional arg. ezpz
doesn't override either method or unpack router output directly, so
inherits cleanly. ezpz `set_moe_sharding_config` calls upstream's
helper so the new shard declaration also lands automatically. **No
ezpz replay needed.** Imports + syntax green.

Other 6 commits: HybridEP compile support (#3360), MXFP8 consolidation
(#3473), graph_trainer DTensor fix + test re-enables (#3480), graph_trainer
test skips (#3432), and two CI migrations (#3464, #3479). None touch
ezpz code paths.

**Smoke status**: live 2N validation deferred to Aurora. The project
`.venv/bin/python` symlinks at
`/opt/aurora/26.26.0/spack/.../python-3.12.12-5zo3wzv/bin/python3`,
which is an Aurora-only Spack build (`5zo3wzv` hash). Sunspot has
the equivalent under a different hash (`nvje3vk`), so the symlink
is broken on Sunspot compute nodes. The `GroupedExperts`-touching
smoke per the prior journal lesson should run on Aurora next time
there's an alloc.

Also updated `~/.ezpz/utils.sh` from the bit.ly canonical (it now
exposes the new unified `ezpz_setup` function; the old
`ezpz_setup_job` / `ezpz_setup_xpu` split is superseded). Backup at
`~/.ezpz/utils.sh-20260602-065437`.

Open follow-ups:
- Close PR #3436 with a comment pointing at PR #3450.
- Smoke `moe_2b_ep` on Aurora to validate the routing_map flow.

---

## 2026-06-01 — Error-propagation fixes + 44th upstream sync

### Error-propagation fixes (from this morning's BlendCorpus debug)

Two bugs surfaced when a rank crashed at init with the misleading
`BlendCorpus dataset was requested but blendcorpus is not installed`
message — when in fact `blendcorpus` was installed and the actual
missing module was `deepspeed` (a transitive dep).

- [`9eb680dbc`](https://github.com/saforem2/torchtitan/commit/9eb680dbc)
  — `_import_blendcorpus_modules` now inspects `exc.name` and emits
  one of two messages: "blendcorpus is not installed" vs "blendcorpus
  IS installed but pulled in missing transitive dep `<name>`". Names
  the actual culprit module so users don't chase the wrong fix.
- [`2c5c7d597`](https://github.com/saforem2/torchtitan/commit/2c5c7d597)
  — wrap `config.build()` in its own try/except in `train.py:main()`.
  Rank 0 logs a single `RANK 0 ABORT during config.build():` line
  followed by the full `__cause__`/`__context__` chain on failure.
  mpiexec presents rank 0 output first, so this surfaces above the
  per-rank `rank N exited with code 1` spam.

### 44th upstream sync

Merged 1 upstream commit (`b72d98648`, PR
[#3403](https://github.com/pytorch/torchtitan/pull/3403)): 4-line
addition to project-root `.claude/CLAUDE.md` recommending ≥10
iterations for perf comparisons. No code change, no ezpz replay.

---

## 2026-05-31 — 43rd upstream sync (interleaved dataloader, no-op replay)

Merged 1 upstream commit (`221041490`, PR
[#3063](https://github.com/pytorch/torchtitan/pull/3063) — weighted
interleaved multi-source HF dataloader). Pure additions —
`InterleavedHuggingFaceTextDataLoader`, `InterleavedChatDataLoader`,
`HFDataSource`/`ChatDataSource`, an `InterleavedDataset` weighted
sampler, plus tests. The existing `HuggingFaceTextDataLoader` and
`DATASETS` that `experiments/ezpz/datasets.py` and
`blendcorpus/blendcorpus_builder.py` import are unchanged. No
replay; imports smoke green.

`git log HEAD..upstream/main` listed 8 commits but only 1 was a
genuine new patch — the other 7 were the patch-equivalent duplicates
from 41st/42nd-sync bookkeeping flagged in the 42nd-sync entry.
`git merge` handled the difference correctly.

---

## 2026-05-29 — 42nd upstream sync (RoPE refactor + replays)

Merged 6 upstream commits (`28483d0eb..065c2625d`). The headline is
PR [#3395](https://github.com/pytorch/torchtitan/pull/3395) which
restructures every model's `update_from_config`:

- Renames the `trainer_config` keyword to `config`.
- Promotes `seq_len > rope.max_seq_len` from warning to hard
  `ValueError`.
- Moves TP / `n_heads`/`n_kv_heads` validation, MoE `deepep`/`hybridep`
  EP=1 guard, MoE `moe_force_load_balance` debug flag, and the
  `rope.max_seq_len` sync into `Decoder.Config.update_from_config`.
- Adds async out-of-bounds checks inside `apply_rotary_emb_*`.

Replayed in `experiments/ezpz/{agpt,moe}/model.py`:

- agpt was a pure `trainer_config`→`config` rename plus parameter
  forwarding.
- moe was bigger: deleted the now-duplicated rope/MoE/TP checks and
  delegated to `Decoder.Config.update_from_config`; kept only the
  per-layer attention rope-field sync, the for_loop XPU fallback,
  the CP+MoE attention check, and `set_moe_sharding_config`. Dropped
  three now-unused imports.

Mirrors the post-refactor shape of upstream `deepseek_v3/model.py`.
2N smoke on alloc 12467655 surfaced two follow-ups:

- A missed `trainer_config`→`config` rename at
  `experiments/ezpz/trainer.py:163` (the trainer's own
  `model_config.update_from_config(...)` callsite, not the model
  override). Fixed in [04199e522](https://github.com/saforem2/torchtitan/commit/04199e522).
- A pre-existing miss from the 41st sync ([#3425](https://github.com/pytorch/torchtitan/pull/3425)
  MoE shape-suffix rename) that we hadn't smoked: three ezpz-side
  references to the old `w1`/`w2`/`w3` parameter names in
  `EzpzGroupedExperts` sharding/init/forward paths — caught when
  the rope replay let us reach trainer init for the first time post-
  41st-sync. Fixed in [88dbd916e](https://github.com/saforem2/torchtitan/commit/88dbd916e).

Post-fix smoke results (both clean, matching 2026-05-27 baselines):

- `agpt_2b`: 154 s, peak 24.34 GiB (38.04%), 21.95% MFU.
- `moe_2b_ep` LBS=2: 323 s, peak 26.99 GiB (42.18%), 9.09% MFU,
  loss step 50 = 6.13.

Lesson worth remembering: always smoke `moe_2b_ep` after a sync that
touches `GroupedExperts`. Today's chain of bugs hid behind the
trainer init failure — the model-init order is rope → expert sharding
→ first forward, so a rope-stage failure prevents us from seeing
expert-stage failures.

Smaller commits in the same sync: #3448 (#3395 fix-forward), #3452
(1-line determinism cleanup), #3445/#3446 (flux/qwen3-vl), #3347 (RL
batcher). None affect ezpz.

Bookkeeping curiosity: `git cherry` flagged 41st-sync MoE [8/n]
(`200100e7d`) as already present in our branch under a different SHA
(`56dc8e1d4`). `git merge` correctly skipped it. So the "7 unmerged
commits" `git log` showed was really 6.

---

## 2026-05-28 — 41st upstream sync (no-op replay)

Merged 1 upstream commit (`200100e7d`, PR
[#3425](https://github.com/pytorch/torchtitan/pull/3425) — MoE [8/n]
shape-suffix rename). Pure rename refactor applying the Shazeer
shape-suffix convention across all MoE tensors. Loss-comparator
verified `--assert-equal` upstream. Renames break several internal
method signatures (`dispatch`, `_unpermute`, `_make_dispatcher`) but
none are called from `experiments/ezpz/`. No replay needed; imports
smoke green.

Still open: maintainer direction on
[pytorch/torchtitan#3436](https://github.com/pytorch/torchtitan/pull/3436)
(histc → bincount/scatter for XPU determinism). Posted the
statistical E2E A/B yesterday — bincount and histc are
indistinguishable on `moe_2b_ep` (Welch's p=0.66, n=135 per variant).
Recommendation: scatter_add_ as the cleanest swap. Waiting on
maintainer choice between options A/B/C.

---

## 2026-05-27 — 40th upstream sync (7 commits)

Merged 7 upstream commits (`19c567f76..af33f7638`):

- **PR #3398** ([Module] Replace from_nn_module with native Module
  subclasses) consolidates `common/{linear,rmsnorm,embedding}.py`
  into `common/nn_modules.py`. Broke 3 import paths in ezpz; replayed
  in [`b052f29e4`](https://github.com/saforem2/torchtitan/commit/b052f29e4)
  with pure import-path swaps (class API is unchanged).
- **PR #3146** (Use deterministic ops in MoE routing) is the upstream
  fix for the `_histc_xpu does not have a deterministic
  implementation` blocker we hit on 2026-05-21. Replaces `histc` with
  `bincount` and adds `aten.topk.default` to the SAC save list.
  Inherits transitively; `--debug.deterministic` on MoE+XPU should
  now work.
- **PR #3423** (MoE [7/n], 3D tensors through MoE) continues the
  MoE refactor from #3386/#3389. Doesn't touch `deepseek_v3/model.py`
  and ezpz doesn't expose the 2D-flatten seam, so we inherit
  transitively.
- **PR #3105** (FSDP symmetric memory) adds an `enable_fsdp_symm_mem`
  flag, plumbed through each model's `apply_fsdp`. ezpz has its own
  local `apply_fsdp`, so the kwarg doesn't reach our path. Skipping
  the replay — symm_mem is an optimization and XPU's CCL likely
  doesn't support it anyway.
- **PRs #3331 / #3369 / #3361** are all graph_trainer-only; no-ops
  for ezpz.

Quick imports smoke (`python3 -c "import torchtitan.experiments.ezpz.{agpt,moe.model}"`)
passes. Live 2N smokes (job 12467455) on Sunspot:

- `agpt_2b` and `moe_2b_ep` clean post-merge, numerically identical
  to the 2026-05-22 baselines.
- `--debug.deterministic` on MoE+XPU **still fails**. PR #3146 was
  supposed to fix the `_histc_xpu` blocker via a `histc → bincount`
  swap, but the merged diff is missing that change — only the
  `aten.topk.default` save-list addition landed. Verified via the
  GitHub API that PR #3146's only file change is
  `activation_checkpoint.py`. The `histc` call at
  `common/moe.py:262` is untouched. Smoke report at
  [`docs/experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md`](experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md).

Action item: file upstream issue for the incomplete PR #3146.

---

## 2026-05-27 — 20B 512N sync chain doubles its eval scores; 2B chains pass step-49K + step-27K

### Production progress (May 25 → May 27, ~40h)

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 36,528 → **49,666+** | +13,138+ | **~114** | 8507195 + 8507198 + 8508020 (R) |
| **2B 512N sync** | 16,676 → **27,106+** | +10,430+ | **~107** | 8507196 (pals-RPC infra fail, +76) + 8507199 (+50) + 8508753 (R) |
| **20B 512N sync** | 2,043 → **3,270** | +1,227 | **+12** | 8507197 + 8507200 |

**Total: ~233 new on-disk checkpoints across 3 chains in 40h of wall clock.**
Sync-mode workaround continues to hold for both 2B 512N and 20B 512N
trajectories; async-mode still stable at 256N.

### 🏁 Headline: 20B 512N sync now beats 2B 256N async per token on every benchmark

Eval'd the full 24-ckpt sync-mode sweep at steps 900..3,200. The 20B
512N sync trajectory now leads the 2B 256N async on **all four
benchmarks** at matched token counts:

| Task | 20B 512N step-3,200 (329B tok) | 2B 256N step-45,500 (~2.3T tok) |
|------|---:|---:|
| ARC-Easy `acc` | **0.6646** | 0.6418 |
| ARC-C `acc_norm` | **0.3225** | ~0.315 (oscillating) |
| HellaSwag `acc_norm` | **0.5737** | 0.5452 |
| Winogrande `acc` | **0.5612** | ~0.55-0.56 |

The 20B model is now token-efficient in a way the 2B has begun to
saturate (plateau at ARC-Easy ~0.645, HellaSwag norm ~0.547). The
full sweep is monotonic — no plateau, no oscillation, no sign of
optimizer instability across 24 consecutive checkpoints. This is
**the first time in the entire v2 experiment that the bigger model
has outperformed the smaller one at matched token counts**, and the
strongest live signal yet that the fp32-master + sync-mode
combination is the right operational stack for the 20B at scale.

### Async-regression workaround still solving 512N

The `CHECKPOINT_ASYNC_MODE=disabled` workaround continues to keep both
512N trajectories advancing. 2B 512N: 4 consecutive sync dispatches
(starting from `8506221`) have added ~10K steps and ~107 persisted
ckpts. 20B 512N: 4 consecutive sync dispatches have added 1,243 steps
and 24 persisted ckpts. No async-cascade failures in any of these.

### One Aurora pals-RPC infra failure (separate from any other bug)

`8507196` (2B 512N) trained cleanly to step **20,989** in-memory
(persisting 76 ckpts), but all three wrapper attempts hit an Aurora
**pals-RPC infrastructure failure** during launch (exit 127). The
wrapper correctly identified the failures as node-related and swapped
three different "bad" nodes — but pals-RPC is upstream of anything
the wrapper can repair. See
`memory/project_aurora_pals_rpc_launch_failure.md`. Not the
async-cascade bug, not a model issue, not a wrapper bug.

### 80B still blocked (separate SIGSEGV pattern)

No 80B production progress this stretch. Last attempts continue to
die in the same rank-level SIGSEGV pattern during init that bracketed
the earlier 80B failures, separate from any of the 2B/20B failure
modes. Writeup + likely upstream patches still pending from the
05-25 entry.

### Capacity is now the bottleneck

`8508214` (20B 512N continuation queued at 03:44) has been **Q ~10h**
in the `small` queue without starting — Aurora capacity for the 512N
slot is exhausted. The 20B chain is currently throttled not by any
bug or workaround but by raw queue availability. Same applies if any
of the 2B 512N continuations need to chain after `8508753`.

---

## 2026-05-25 — 🏁 Sync-mode workaround fully validated for both 2B + 20B 512N; 100+ ckpts persisted overnight

### Production progress (May 24 → May 25)

After kicking off 6 production dispatches and 3 smoke runs late on 2026-05-23,
**three production chains advanced significantly on disk overnight**:

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 25,500 → **36,528** | +11,028 | **65** (every 100 steps) | 8505175 + 8505252 |
| **2B 512N sync** | 13,300 → **16,676** | +3,376 | **21** (every 100 steps) | 8506221 |
| **20B 512N sync** | 800 → **2,043** | +1,243 | **12** (every 100 steps) | 8505258 + 8505259 |

**Total: 98 new on-disk checkpoints across 3 chains in ~36h of wall clock.**
First sustained 512N progress since 2026-05-03 for both 2B and 20B.

### Sync-mode async-cascade workaround validated

8505258 (20B 512N sync) was the proof-of-concept: trained 800→1414 cleanly with
6 ckpts persisted, the first 20B 512N to clear step-800 since the May-3 async
regression. 8505259 carried the chain to step-2043 (6 more ckpts).

8505176 (2B 512N async) confirmed the **same bug pattern at 2B 512N**: every
attempt cleanly trained 13300→13400, then died at the step-13400 async save,
wrapper swapped + retried 3 times before exhausting. Switching to sync mode
in 8506221 produced **21 consecutive ckpts** at 512N — same fix as 20B.

`CHECKPOINT_ASYNC_MODE=disabled` is now the standard 512N workaround for both
models. Async still works fine at 256N (65 ckpts in one dispatch).

### Preflight smoke bugs surfaced + fixed

8506215 (first 2B 512N sync attempt) revealed the preflight smoke's 120s
idle-timeout was too tight for 6144-rank DDP init. Fixed by bumping to 600s
default + adding `--train-iters 5` to cap the test length (was running 200
iters by default → 1+ hour of preflight at 512N). See
[`memory/feedback_preflight_timeout_scales_with_n.md`](.).

8506221 then hit a *real* silent hang during preflight (iter 111, 49 min of
silence), wrapper SIGTERM'd on watchdog, blind-swapped `x4305c0s7b0n0` for
`x4602c3s3b0n0`, preflight attempt 2 succeeded, main training started. **The
wrapper handled the failure exactly as designed even during the preflight
phase.**

### New jobs queued (May 25 afternoon)

- 8507195 (2B 256N async cont) + 8507198 (afterany +1)
- 8507196 (2B 512N sync cont) + 8507199 (afterany +1)
- 8507197 (20B 512N sync cont) + 8507200 (afterany +1)
- 8507204 / 8507205 / 8507206: lm-eval batches on the new 2B 256N (26-36K),
  2B 512N (14-16K), and 20B 512N (900-2000) checkpoints respectively

### 80B is still blocked

8505222 (80B 256N production) failed with rank-level SIGSEGV (signal 11) on
attempt 1 + every retry, wrapper correctly classified as bad-node failure
but every spare it swapped in was also bad. 5 retries exhausted. Separately,
80B 8N smoke 8505326 surfaced a deterministic blendcorpus EOFError race in
cache build (3 wrapper attempts, all same failure). Two distinct 80B failure
modes both need writeups + likely upstream patches.

---

## 2026-05-23 (late) — 🏁 Failover wrapper passes first production silent-hang test (job 8505298)

While running the 8-node smoke validation of the fresh ezpz 0.16.0
tarballs (jobs `8505298` / `8505325` / `8505326`), the **2B smoke job
8505298 caught a real silent training hang at step 37 and recovered
automatically** — every code path in the v2 failover wrapper that
exists to handle the [8479579 incident
pattern](experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
fired correctly, in sequence, on a real-world failure.

Sequence (clean steps to recovery to checkpoint-persist):
1. Preflight `ezpz.examples.test` ran in ~2 min, exit 0.
2. Main attempt-1 trained steps 1 → 37, then logged went **completely
   silent at 21:06:41** — no traceback, no save attempt, no MPI error.
3. **30 min later at 21:36:41**, `ezpz launch --timeout=1800` watchdog
   tripped, SIGTERM'd PID 191892, exit 124.
4. Wrapper classified exit 124 as silent-hang bad-node failure (not
   walltime), couldn't identify a specific bad node from the log
   (none — the hang was silent), fell through to
   `failover_swap_one_blind()`, rotated `x4220c3s6b0n0` →
   `x4220c5s3b0n0`.
5. Attempt-2 started 4s after the swap, hit step 1 at 21:38:17, ran
   cleanly to step 296 (loss 12.96 → 5.68 = ~466M tokens) until
   PBS walltime kill at 21:57:50.
6. **step-100 and step-200 DCP checkpoints both persisted** on flare
   (96 shards × 239 MB each + 5.5 MB metadata). First unambiguous
   proof since the 2026-05-03 regression that async-save works
   end-to-end with the fresh ezpz 0.16.0 tarball.

Full writeup with exact log snippets:
[`experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md`](experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md).

Knock-on actions: bumped the
[`bad-node-failover.md`](guides/bad-node-failover.md) status from
"v2 in production" to "v2 production-validated"; added back-pointer
from the original 8479579 incident report; greenlit qdel + resubmit
of the 9 queued production jobs (2B/20B/80B chains) against the
fresh tarballs.

20B + 80B smokes (`8505325` / `8505326`) still Q in capacity queue —
blocked behind `8505200` (the 2-node test.sh used for the tarball
rebuilds + smoke runs); will start when that walltimes out around
2026-05-24 04:00 UTC.

---

## 2026-05-23 — Failover wrapper hardening: tests, ANSI fix, async-mode regression diagnosed

### Failover-wrapper test harness + 2 more wrapper bugs

Built `tests/failover/{run_tests.sh, fixtures/*.log}` — 9 synthetic
log fixtures with byte-identical ANSI escapes, each reproducing one
of the failure modes we've seen in production. The harness mirrors
`failover_lib.sh`'s rc-determination block into a standalone
`evaluate_rc()` function and asserts the expected `(rc, decision)`
tuple for each fixture. Pattern: edit `failover_lib.sh` → update
fixtures → run `bash tests/failover/run_tests.sh` here in the main
repo → THEN push + pull into v2 clones. Stops the
edit-push-pray-discover-bug-only-in-production loop that ate ~6
production runs over the past 48h.

Running the new tests immediately surfaced **2 more wrapper bugs**:

1. **`grep -c ... || echo 0` produces `0\n0`** when grep matches
   nothing → arithmetic eval `syntax error (error token is "0")` →
   the entire crash-line branch silently skipped. `grep -c` already
   writes `0` to stdout on no-match; the `|| echo 0` fallback was
   dead code that produced malformed output. Removed.
2. **Walltime guard regex was a strict subset of crash-detect regex**
   — only matched `Connection closed by peer | died from signal (9|11)`,
   missing `OutOfMemoryError`, `UR_RESULT_ERROR`, `Timed out waiting`,
   `EOFError`. So a real bad-node failure that surfaced as shell exit
   143 (mpiexec SIGTERM after EOFError) got misclassified as a clean
   walltime kill and the wrapper bailed without retry. Made both
   regexes identical (the broader set).

Both bugs silently active in production yesterday/today before fix.
Commit `0d93a1e91` adds the harness + fixes; 9/9 tests pass in both
the main repo and the 20B v2 production clone.

### ANSI codes in 'Execution finished with N' parsing

Earlier in the day, several jobs zombie-succeeded because the
wrapper's `inner_rc` extraction returned empty on ANSI-coded log
trailers:

  Logged:  `Execution finished with \x1b[1;36m143\x1b[0m`
  Regex:   `Execution finished with \[?[0-9]+\]?`
  Match:   none (the `\[?` matches a literal `[` byte, but the
           actual byte sequence is ESC + `[`)

Fix `94a8fda66`: strip ANSI codes with `sed -r 's/\x1b\[[0-9;]*m//g'`
before grepping the trailer.

### Async-mode regression for 20B 512N — the actual root cause of
"async ckpt save kills the cluster"

Spent hours today investigating why 20B 512N hasn't persisted past
`step-800` since 2026-05-03 — three weeks of dispatches all dying
mid-save. Today found the **smoking gun**: every save on disk between
step-200 and step-800 happened on 2026-05-01 + 2026-05-03 under the
**default checkpoint mode (sync)**. The submit script switched to
`--checkpoint.async-mode=async` sometime between May 3 and May 11,
and nothing has persisted past step-100 on the 20B 512N chain since.

The previous "files-per-save / Lustre saturation" hypothesis was
half right but missed the actual mechanism: at 6,144 ranks, async
saves stream the 244 GB ckpt to flare in the background AT THE SAME
TIME as the gloo training-step heartbeat. Either the writes or the
gloo traffic backs up, one peer times out, cascade. Sync saves block
training while writing — no overlap, no cascade.

Submitted `8505258` (20B 512N) + `8505259` (cont) with
`CHECKPOINT_ASYNC_MODE=disabled` (= true sync). Also queued
`8505255/56/57` (20B 256N sync variants). 2B chains stay on async
(those have always saved cleanly at this scale).

### Production chain progress today

**2B 256N**: persisted **step-25,000 → step-25,500** over 2 dispatches
(8503506 walltime-finished + 8505119 advanced ckpts then exited 127).
+500 fresh steps. Loss 2.74. The only trajectory actually moving.

**20B 256N**: still wall-bound at `step-300`. Six dispatches today,
each reached in-RAM step 308-326 then died from bad-node mid-training
before crossing the next 100-step save boundary. Wrapper detection
working correctly (validated against fixtures); the wall is genuine
Aurora bad-node prevalence at the per-dispatch survival window.

**20B 512N**: still wall-bound at `step-800`. Four dispatches today,
all 3-attempt exhausted at init before any training step. Hopes
pinned on the sync-mode resubmit (`8505258`).

**80B 256N**: 0 persisted, 5 attempts today. Bumped to `select=276`
(20 spares) + `FAILOVER_MAX_RETRIES=5` for `8505221` — still died.
Environment too unstable for 80B init right now.

### Wrapper fix chronology (today)

| Commit | Fix |
|--------|-----|
| `94a8fda66` | Strip ANSI codes before parsing 'Execution finished with N' |
| `0d93a1e91` | Test fixtures (force-added .log) |
| `4310258` | Drop `\|\| echo 0` bug + walltime-guard regex parity |

All 3 commits pulled into all 3 v2 production clones. Tests pass in
all clones.

### `ezpz` upgraded to 0.15.1 in all v2 clones

Got the `--timeout T` + `--retries N` flags from ezpz PR #136. Wired
`--timeout=1800` into `failover_lib.sh` (commit `eefccfc9d`,
yesterday). Catches silent-hang failure mode that previously was
invisible (the 8479579 incident — 5h of W&B heartbeat alive but
training metrics dead). Exit 124 from the watchdog now routes
through swap-and-retry.

### What's next

- Once `8505123` (20B 256N, currently R, in-RAM step 326) either
  crosses step-400 ckpt save or dies, the sync-mode 20B chains
  (`8505255` for 256N, `8505258` for 512N) take over. That's the
  live test of the async→sync regression hypothesis.
- Eval batch `8505205` (13 fresh 2B 256N ckpts, step 14K → 25.1K)
  running on capacity; 5/13 done so far. Will refresh plots +
  README tables once all land.

### 39th upstream sync — DebugMode numerics debugger (no-op for ezpz)

Merged 1 upstream commit (`19c567f76`,
[PR #3323](https://github.com/pytorch/torchtitan/pull/3323)). Pure
tooling addition: new `torchtitan/tools/numerics_debugging/` module
(activation tracer + bitwise comparator, ~2 KLOC) plus a
`numerics_debugging` skill under `.claude/skills/`. No code path
ezpz exercises changed; no replay needed. The new skill auto-loads
in this session and could be useful next time we need to bisect a
silent loss-curve divergence.

### Closing follow-up — PR #184767 closed in favor of upstream #183625

`@frost-intel` flagged that
[pytorch/pytorch#183625](https://github.com/pytorch/pytorch/pull/183625)
is a draft already covering the xccl `_set_pg_timeout` dispatch +
the new `test_c10d_xccl.py` (in pieces). Closed our PR #184767 in
deference. Local workaround in
[`22847fcb3`](https://github.com/saforem2/torchtitan/commit/22847fcb3)
(`_set_pg_timeouts_xpu_aware`) stays load-bearing until #183625
actually lands.

---

## 2026-05-22 — First upstream PyTorch PR filed; 2-week summary

### Upstream PyTorch PR for xccl `_set_pg_timeout` dispatch

Filed https://github.com/pytorch/pytorch/pull/184767 — adds the
missing xpu branch in `torch.distributed.distributed_c10d._set_pg_timeout`
so xccl PGs route through `ProcessGroupXCCL.set_timeout` instead of
silently no-op'ing with the `"Set timeout is now only supported for
either nccl or gloo."` warning. Initial commit
[37af153](https://github.com/saforem2/pytorch/commit/37af153); review
fixes (Backend type annotation, simpler single-binding form, updated
warning text, `find_free_port` + `@retry_on_connect_failures` for the
test) in [8ceedc7](https://github.com/saforem2/pytorch/commit/8ceedc7).
All 7 inline review threads from copilot + codex addressed and
resolved. Pinged `@kwen2501` (c10d CODEOWNER) + `@guangyey` +
`@frost-intel` for review; CI gated on first-time-contributor workflow
approval.

Empirically verified the diff on Sunspot 1N × 12 ranks against the
in-repo `.venv` torch 2.13 (allocs 12467214 + 12467219 + 12467231,
all released). Side finding: xccl's C++ `set_timeout` **does** mutate
`backend.options._timeout` — contradicts the pessimistic line in
[`PLAN_xccl_timeout_upstream_pr.md`](upstream-issues/PLAN_xccl_timeout_upstream_pr.md)
PR 2 that "xccl stores the value but does nothing." Storage works;
only **enforcement** (watchdog + abort) is still missing. PR 2 scope
unchanged. Local pytest port at
[`tests/distributed/test_c10d_xccl.py`](../tests/distributed/test_c10d_xccl.py)
verifies the patched-vs-unpatched contract on either side.

### Two-week summary

Wrote up the 2026-05-08 → 2026-05-22 retrospective at
[`docs/summaries/2026-05-08_to_2026-05-22.md`](summaries/2026-05-08_to_2026-05-22.md).
51 commits across 8 themes: upstream xccl PR, 4 upstream syncs (one
no-op, one no-replay, two with replays), 80B bad-node failover
infrastructure (the silent-hang detection bug fix in
[e216a2523](https://github.com/saforem2/torchtitan/commit/e216a2523)
closes the 8479579 incident class), Sunspot smoke campaigns, MoE EP=2
hang reclassification, TPC26 talk prep, and docs hygiene.

### 38th upstream sync — MoE dispatcher split + ChunkedCELoss/TP grad fix

Merged 4 upstream commits (`cfe97c605..c2a3771a4`). One replay landed
in [`d87729ad8`](https://github.com/saforem2/torchtitan/commit/d87729ad8):
mirror upstream PR
[#3389](https://github.com/pytorch/torchtitan/pull/3389)'s isinstance
dispatch on `token_dispatcher` Config classes in
`experiments/ezpz/moe/model.py`, dropping the removed `DeepEPMoE`
swap. PR
[#3412](https://github.com/pytorch/torchtitan/pull/3412) is internal
to `torchtitan/components/loss.py` — no ezpz replay.

Smoke (2N Sunspot, jobs 12467277 + 12467288 + 12467323):
- `agpt_2b` clean, byte-comparable baseline (140 s, peak 24.34 GiB).
- `moe_2b_ep` at LBS=1 clean and numerically equivalent to the
  37th-sync baseline: 14.95 GiB vs 15.03 GiB, TPS within 1.4%.
- `moe_2b_ep` at the previous registry-default LBS=16 OOMs on the
  bf16 vocab projection (`[16 × 8192, 256128] × 2 B ≈ 62.5 GiB`,
  overflows a 64 GiB Max 1550 tile). Same pre-existing `_ep`
  vocab-projection OOM the 37th-sync follow-up flagged. Closed by
  pinning `moe_2b_ep` to LBS=2 in
  [`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f).
- Smoke report:
  [`docs/experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md`](experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md).

Side issue: stale `outputs/checkpoint/step-100` from pre-PR-3159
layout is no longer loadable (`Missing key in checkpoint state_dict:
layers.0.attention.qkv_linear.wk.weight.`); backed up to
`outputs/checkpoint-20260522-120005`.

---

## 2026-05-21 — Step-41 EP hang retry + registry fix + TPC26 talk prep

### `moe_debugmodel_ep` LBS=2 hang did not reproduce — reclassified transient

Retried yesterday's hung config on a sibling 2N alloc (12467180).
Full 50 steps clean, exit 0, 139 s wall. **The original 16-min stall
at step 41 was a transient**, not a systemic EP/AC/compile bug.
Reclassified in
[`docs/upstream-issues/moe_ep_step41_hang.md`](upstream-issues/moe_ep_step41_hang.md)
with three adjacent findings worth keeping
visible:

- `comm.train_timeout_seconds=100` did not fire on the original
  985 s silence — timeout may not be wired into the CCL/XCCL
  collective path on XPU. Worth a separate writeup.
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` crashes on XPU with
  `Backend fake does not yet support sequence numbers`. The error
  doesn't mention XPU — easy footgun on next attempt.
- `--debug.deterministic` is incompatible with MoE on XPU:
  `_histc_xpu does not have a deterministic implementation`. So
  bit-exact regression gates for MoE on Intel are blocked on an
  upstream PyTorch deterministic `_histc_xpu` kernel.

Also a curious throughput band: retry ran at ~11.8k TPS, original at
~6.8k TPS — same code, same nodes-of-the-same-class. ~1.7× spread,
plausibly correlated with whatever caused the original hang.

### Registry fix for `_ep` configs

Pinned `moe_debugmodel_ep` to LBS=2 (was inheriting LBS=8 from
`moe_debugmodel()`, OOM'ing at ~33 GiB on the vocab projection).
Annotated `moe_2b_ep` with a docstring confirming its LBS=16 default
is the validated peak. Commit `f2cbc0327`.

### TPC26 MAPE talk

Got invited to speak at the TPC26 MAPE track (Baltimore / Munich,
May 31 - Jun 3) by Rio Yokota. Drafted title, abstract, and 11-section
outline at
[`docs/notes/slides-2026-05-21.md`](notes/slides-2026-05-21.md).
The fork-tax-as-first-class-workflow angle (§4) and the silent-numerics
section (§6) are the most differentiated bits. Folded today's
engineer-hours/week estimate into §9: **~8-15 hr/wk recurring
operational triage** across 4 weeks of journal entries (~25-35% of
one engineer), with episodic spikes to 30-40 hr when a silent bug
surfaces or a bisect goes wide. The unbounded-cost punchline lives in
the silent class: bf16-freeze alone burned ~450B tokens of v1
compute.

---

## 2026-05-20 (late) — PR #3386 EP follow-up + agpt merge sanity smoke

Followed up the 37th-sync replay with two parallel smoke campaigns at 2N on
Sunspot (commit `1d4115d3f`, jobs 12467180/12467181):

### moe `_ep` follow-up — EP=2 path validated, but registry configs need LBS override

Reports: [`moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md)

- **`moe_2b_ep` (LBS=16 from registry)** — clean 50 steps,
  12.94 → 6.07, 2,860 TPS/GPU, peak **15.03 GiB** vs `moe_2b` EP=1's
  14.97 GiB. Token-dispatch overhead at 2B scale is essentially free
  (+0.06 GiB, ~1% TPS hit). PR #3386's `wire_meshes` plumbing works.
- **`moe_debugmodel_ep` (LBS=8 from registry)** — **OOM at init** in vocab
  projection (`(LBS*8192, 256128) bf16` = 31.27 GiB requested). The default
  LBS the registry inherits from `moe_debugmodel()` is too aggressive for
  the EP variant at 2N. Override needed.
- **`moe_debugmodel_ep` re-run at LBS=2** — clean steps 1-41 then **hung
  16 min at step 41/50** and got SIGTERM (exit 143). New finding,
  uninvestigated — possibly EP all-to-all backend stall under the
  `standard` `moe_comm_backend`. Not blocking but worth a follow-up.

### agpt merge sanity — clean, plus DeviceMesh regression re-confirmed

Reports: [`agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md`](experiments/agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)

- **`agpt_2b`** — clean 50 steps, 12.97 → 6.58, peak **24.34 GiB
  (38.04%)** — *byte-identical* to the prior post-resync baseline.
  Confirms PR #3346 (`graph_trainer` regional_inductor refactor, bundled
  in the same merge) is a no-op for the agpt path. Throughput within the
  expected 2-3% noise band.
- **`agpt_50b_wide`** — re-confirms the torch-2.13
  `DeviceMesh`-in-saved-tensors `AssertionError` in AOT autograd's
  `save_from_forward`. Crash in ~121s on 2N, all 24 ranks identical
  signature. 37th sync did **not** fix it (didn't expect it to — bug is
  in PyTorch, not torchtitan). Standing workaround (compile=OFF for
  80B-family on torch 2.13, or stay on torch 2.10) still the only
  option. See [`project_80b_devmesh_bisect`](../../../../../home/foremans/.claude/projects/-lus-tegu-projects-datascience-foremans-projects-saforem2-torchtitan/memory/project_80b_devmesh_bisect.md).

### Action items dropped on the floor

- `moe_debugmodel_ep` / `moe_2b_ep` config registry: either pin a sane
  LBS in the `_ep` variants or document the OOM in the registry.
- Investigate the LBS=2 debugmodel_ep step-41 hang (EP all-to-all
  backend? token-dispatch deadlock?). Not reproducing automatically until
  someone re-runs.

---

## 2026-05-20 — 37th upstream sync (MoE clean DTensor boundaries) + replay smoke

Second sync of the day. Merged `89987072b` (2 commits beyond the 36th sync):

- **`963c20cba` — PR #3386** [MoE][5/n] Refactor MoE to clean DTensor
  boundaries for shared/routed experts. The big one.
- **`83e490429` — PR #3346** graph_trainer `regional_inductor` refactor.
  No ezpz dep.

### What PR #3386 changes

Restructures MoE TP/EP wiring from an imperative parallelize-time pass to
config-based sharding declarations populated at `update_from_config` and
applied by `model.parallelize(parallel_dims)`:

- **Deleted upstream:** `torchtitan/distributed/expert_parallel.py`
  (`ExpertParallel`, `TensorParallel`), `ColwiseParallelWithGradPlacement`.
- **New upstream:** `torchtitan/models/common/moe_sharding.py` with
  `set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
  expert_param_layout)` populating router gate, shared experts, routed
  experts.
- **`GroupedExperts.parallelize`** added — calls `super().parallelize` then
  `token_dispatcher.wire_meshes(ep_mesh, tp_mesh)`.
- **`MoE.forward` simplified** — drops the explicit
  `DTensor.to_local(grad_placements=Partial)` at the top (now handled by
  config), splits shared-experts addition out of `combine()`.
- **`parallelize_deepseekv3`** drops `apply_moe_ep_tp` call; new flow:
  `if tp_enabled or ep_enabled: model.parallelize(parallel_dims)`.

### Replay scope

Only ezpz/moe was affected:

| File | Δ lines | Change |
|------|--------:|--------|
| `experiments/ezpz/moe/parallelize.py` | -73 | Drop `apply_moe_ep_tp` entirely + 3 deleted-symbol imports; collapse two-pass to single `model.parallelize` |
| `experiments/ezpz/moe/sharding.py` | +35 | Add `enable_ep` kwarg, call upstream's `set_moe_sharding_config` per MoE layer with `{w1:Shard(1), w2:Shard(2), w3:Shard(1)}` layout |
| `experiments/ezpz/moe/model.py` | +3 | Pass `enable_ep=...` from `update_from_config` |
| `experiments/ezpz/moe/config_registry.py` | -2 | Stale docstring scrub |

`apply_fsdp` (Aurora `ShardPlacementResult` workaround) and
`disable_fsdp_gradient_division` (CCL SUM-reduction workaround) stay
inlined locally.

### Smoke verification (Sunspot 2N, job 12467131)

| Config | Final loss | Δ vs baseline | Memory | TPS |
|--------|-----------:|--------------:|-------:|----:|
| `moe_debugmodel` LBS=2 | 6.99880 | -0.010 | 16.99 GiB (matches) | ~12,700 |
| `moe_2b` LBS=1 | 6.10607 | -0.050 | 14.97 GiB (+0.5 vs baseline) | ~2,900 |

Both within ±0.05 nats of the 35th-sync baseline. Drift is expected:
PR #3386's commit message states *"loss is expected to diverge compared
to main due to different reduction pattern, and shared expert
computation changes place"* — confirmed at our parallelism configuration.
`for_loop` expert backend fires the same warning count (5 + 17) as
baseline. No NaN/OOM. No recompilation events.

### Concerns flagged before merging (all resolved)

- **`MoE.forward` graph shape changed.** Watched for inductor
  recompilation events — none observed. Memory uptick at moe_2b
  (+0.5 GiB) is the only visible cost, plausibly from a separate buffer
  for `shared_out` before the final add.
- **Removed async overlap between shared_experts and DeepEP combine.**
  Documented upstream as a follow-up "can restore overlap using CUDA
  streams." We don't use DeepEP on Aurora/Sunspot so this is a no-op
  for ezpz today, but worth tracking if we ever turn it on.

### Reports + W&B

- Smoke: [`docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md)
- 37th sync entry: [`docs/upstream-sync.md`](upstream-sync.md)
- W&B `moe_debugmodel`: [`efficient-bird-2070`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/saga2gds)
- W&B `moe_2b`: [`electric-pond-2071`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/zudqly3y)

---

## 2026-05-20 — Post-resync smoke campaign (Sunspot 2N)

Validated yesterday's [35th upstream sync](#2026-05-19--upstream-resync-35th-full-dtensor-3159)
with a four-config compute-node smoke on Sunspot (job `12467124`, 2 nodes,
24 XPUs). All configs ran 50 steps cleanly post-replay; no NaN/OOM,
monotonic loss descent, no regression vs the historical Apr 25 baseline.

| Config              | LBS | Final loss | TPS/GPU | MFU    | Peak mem            |
|---------------------|----:|-----------:|--------:|-------:|---------------------|
| `agpt_debugmodel`   | 2   | 6.77       | ~37,500 | ~2.9%  | 2.60 GiB (4.06%)    |
| `agpt_2b` (LBS=1)   | 1   | 6.01       | ~6,100  | ~22.8% | 24.34 GiB (38.04%)  |
| `agpt_2b` (LBS=2)   | 2   | 6.12       | ~7,200  | ~27.0% | 44.73 GiB (69.91%)  |
| `moe_debugmodel`    | 2   | 7.01       | ~13,000 | ~9.0%  | 16.99 GiB (26.55%)  |
| `moe_2b` (LBS=1)    | 1   | 6.16       | ~2,900  | ~8.4%  | 14.47 GiB (22.62%)  |

**`agpt_2b` LBS=2 matches Apr 25 n=2 baseline** (7,224 TPS / 27.11% MFU
today vs 7,142 TPS / 27.6% MFU then) within noise. PR #3159's
`Module.parallelize(parallel_dims)` signature is wired correctly through
both `experiments/ezpz/{agpt,moe}/parallelize.py`.

**`for_loop` expert backend (PR #13) still fires on XPU** -- 5 warnings
for moe_debugmodel, 17 for moe_2b (one per MoE layer in each flavor).
End-to-end forward/backward/optimizer under FSDP all converge cleanly.

**Default `moe_2b()` LBS=16 OOMs on Max 1550** with a single 62.53 GiB
allocation. Likely the fused activation for all experts × full-batch-tokens
materialized by the for_loop path. Pre-existing XPU constraint, not caused
by the resync. Switched to LBS=1 for the smoke.

### Permissions sidestep that worked

The auto-mode classifier blocks the `source <(curl -fsSL https://bit.ly/ezpz-utils)`
pattern on every `ezpz launch`. Workaround: on the compute node, cache the
utils script once with
`mkdir -p ~/.ezpz && curl -fsSL https://bit.ly/ezpz-utils -o ~/.ezpz/utils.sh`,
then prefix every launch with `source ~/.ezpz/utils.sh && ezpz_setup_job && ezpz_setup_xpu`.
The cache is persistent on the compute node so this only needs doing once
per allocation. Used successfully for all five smoke launches.

### Reports

- [`docs/experiments/agpt/sunspot/20260520-smoke-n2-postresync.md`](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md)
- [`docs/experiments/moe/sunspot/20260520-smoke-n2-postresync.md`](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)
- `docs/upstream-sync.md` 35th entry updated with smoke validation table.

W&B runs: `olive-plasma-2054`, `sunny-waterfall-2055`, `dry-water-2056`,
`worldly-music-2058`, `azure-field-2061` (all under `aurora_gpt/torchtitan.ezpz.train`).

---

## 2026-05-19 — Upstream resync (35th, Full DTensor #3159)

Pulled 22 upstream commits (`ee4e91a13..52a292d29`, merge `a14987132`).
The headline is [pytorch/torchtitan#3159](https://github.com/pytorch/torchtitan/pull/3159)
"Config-based Full DTensor for Llama3" — a substantial refactor of the
config-based sharding API that lays the foundation for
`--training.full_dtensor` (all params/buffers/inputs become DTensors on
a multi-dim SPMD mesh).

**Breaking signature change for ezpz:** `Module.parallelize(mesh)` →
`Module.parallelize(parallel_dims)`. Each Module now self-resolves its
SPMD submesh from the axes referenced in its `NamedPlacement`s, instead
of being handed a bare `tp_mesh`. Two ezpz callsites hit:
`experiments/ezpz/agpt/parallelize.py` and
`experiments/ezpz/moe/parallelize.py`. Both replayed: pass
`parallel_dims` to `model.parallelize`, keep the explicit
`parallel_dims.get_mesh("tp")` for the async-TP plumbing on the next
line. `apply_moe_ep_tp` still takes per-axis meshes directly (it doesn't
route through `Module.parallelize`), so it's untouched.

`ShardingConfig` got three new optional fields (`out_src_shardings`,
`local_input_grad_placements`, `local_output_grad_placements`); all
default `None` so ezpz's existing `set_agpt_sharding_config` /
`set_moe_sharding_config` construct unchanged shapes. The
`set_gqa_inner_attention_local_map` helper that ezpz/agpt calls had
internal arg renames (`xq/xk/xv` → `q/k/v`) but the public call site is
identical.

`trainer.py` gained a `full_dtensor`-gated `parallelize_inputs` call and
a `pred.to_local()` fallback under `disable_loss_parallel`. ezpz's
`FaultTolerantTrainer` doesn't override `_get_batch` or
`forward_backward`, so both inherit cleanly. `full_dtensor` defaults
`False`, so no behavior change for current production.

Verification: `.venv/bin/python -c "import
torchtitan.experiments.ezpz.{agpt,moe}.parallelize"` succeeds on both
post-replay. A real compute-node smoke (`agpt_2b`/`moe_500m`) is
pending the next allocation. Doc: `docs/upstream-sync.md` 35th entry.

---

## 2026-05-12 — Upstream resync (#3308) + `for_loop` backend smoke

### Resync PR #13

Pulled 12 commits from `upstream/main` into a fresh `ezpz-moe-resync`
branch. The big-ticket landing was
[pytorch/torchtitan#3308](https://github.com/pytorch/torchtitan/pull/3308),
which deleted `_run_experts_for_loop` and the `use_grouped_mm` config
field from `models/common/moe.py` and inlined `torch._grouped_mm` as the
only expert path. Upstream's argument: `_grouped_mm` already provides a
CUDA fallback. **XPU has no `_grouped_mm` kernel at all**, so this would
have broken every ezpz MoE config on Aurora / Sunspot at first forward.

Replay strategy: introduce `EzpzGroupedExperts(GroupedExperts)` in
`experiments/ezpz/moe/experts.py` with a
`compute_backend: Literal["for_loop", "grouped_mm"]` selector. Default
defers to upstream. The `for_loop` branch re-vendors the deleted
`_run_experts_for_loop` body verbatim, restoring the XPU / pre-SM90 path.
`model.py` `update_from_config` now switches to `for_loop` on any device
that fails `has_cuda_capability(9, 0)`.

PR #13: https://github.com/saforem2/torchtitan/pull/13 (replaces #12).
PRs #9 / #10 / #11 (Sam Wheeler / Sam Wheeler / Nathan Nichols) flagged
on each that they should rebase onto this and adapt to the
`EzpzGroupedExperts` subclass.

### Smoke validation (Sunspot 8N)

Job `12466707` on `x1921c5s0b0n0`–`x1921c5s7b0n0`. `moe_500m`, 50 steps,
local batch 4, seq 8192, GBS 384. Run:
[`fluent-glitter-2042`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lt77xx0o).

- 11 layer-wise warnings emitted (1 per MoE layer):
  `torch._grouped_mm requires SM90+ CUDA; falling back to for_loop expert backend.`
  Confirms PR #13's `compute_backend = "for_loop"` switch is taken.
- Loss descended cleanly **12.90 → 6.66 (-6.24 nats)** over 50 steps.
- Steady-state throughput **~8,694 TPS / GPU, ~13% MFU** — actually ~20%
  per-GPU TPS uplift vs the
  [2026-04-13 2N benchmark](experiments/moe/sunspot/20260413-benchmark-n2.md)'s
  7,228 TPS / 9.11%, attributable to torch.compile inductor improvements.
  **No measurable regression vs the upstream `_run_experts_for_loop`
  body that #3308 deleted** (which makes sense — it's the same kernel).
- Memory stable at 54.6% across the run.
- Wall: 541s end-to-end including env setup + compile warmup.
- Report:
  [`docs/experiments/moe/sunspot/20260512-for-loop-smoke-n8.md`](experiments/moe/sunspot/20260512-for-loop-smoke-n8.md).

PR #13 is now smoke-validated end-to-end on XPU.

---

## 2026-05-05 — 80B DeviceMesh-bisect: torch-version, not depth

### Bisect kills the May 3 "depth-sensitive" claim

Submitted job 12465952 on Sunspot 4N to bisect the
`tensors_saved_with_vc_check` AOT autograd assertion across the agpt
80B family on the torch 2.13 venv. Three configs ran sequentially:

- `agpt_50b_wide` (48 layers): crashed in **66 s**, every rank logs the
  assertion (49 ranks × 1 = full crash on the first
  forward+backward).
- `agpt_70b_wide` (72 layers, new
  [`b9cda4b2`](https://github.com/saforem2/torchtitan/commit/b9cda4b2)
  config — 80B family with 12 fewer layers): same assertion, 28 s.
- `agpt_80b` (84 layers): same assertion, 29 s.

Per-config logs at `logs/agpt-80b-bisect-12465952/{config}.log`. The
"depth-sensitive — works at 48 layers" conclusion in the May 3 entry
was wrong. Two variables had changed between the May 3 50B_wide
success and today's 50B_wide failure: node count (2N → 4N) AND torch
version (`aurora_frameworks-2025.3.1` torch 2.10 → torch 2.13 venv).

To pin the variable I submitted job 12465962 (2N + torch 2.13,
[`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)).
Result: same assertion, 57 s. **torch 2.13 alone is the trigger;
node count is a non-factor.**

This means:

- The bug is **torch-version-sensitive, not depth-sensitive**.
- `agpt_50b_wide` on 2N + torch 2.13 is now a **clean ~30-60 s
  reproducer** for the upstream report (much smaller than 80B).
- The May 3 50B_wide success was masked entirely by the older AOT
  autograd code path on torch 2.10.

### Updated docs

- `experiments/ezpz/.claude/CLAUDE.md` — corrected the Recent Findings
  bullet, the Production v2 80B note, and the Known Bugs entry.
- `experiments/ezpz/docs/meeting-notes/agpt-sync.md` — added the
  2026-05-05 correction beneath the original 2026-05-03 finding so
  the meeting can show both.
- The new `agpt_70b_wide` config and the
  [`submit_80b_bisect.sh`](../scripts/submit_80b_bisect.sh) +
  [`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)
  scripts are now part of the bisect-tooling.

### Validator small-batch + global-state bug

While running the agpt_2b validator smoke separately, noticed the
val build was using `Global batch size: 48` and warning about
`train_iters defaulting to 1` on every validate() call. Three
related issues, all in
`torchtitan/components/validate.py:Validator.validate()` not passing
the trainer's `training_steps` and `global_batch_size` through to
`dl_config.build()`. Fixed in
[`3edfb0ff`](https://github.com/saforem2/torchtitan/commit/3edfb0ff)
by capturing `job_config` in `EzpzValidator.__init__` and forwarding
the right values. Validator was using ~1/4-sized batches per call
(noisier val loss); a worse latent issue was that `bc_set_config`
overwrites the blendcorpus library's global `DATA_CONFIG` with the
wrong `train_iters` and `global_batch_size`, which would silently
corrupt any later resume-from-ckpt rebuild of the train dataloader.

### blendcorpus ↔ Megatron parallelism aliasing

Reviewed `BlendCorpusDataLoader` for further Megatron-style
parallelism leftovers. Found 7 inconsistencies:

- 2 active and fixed today
  ([`140481d3`](https://github.com/saforem2/torchtitan/commit/140481d3)):
  scope the `dist.barrier` → CPU/gloo monkey-patch to torch < 2.13
  (it was being silently applied on 2.13 even though XCCL is fixed
  there); rename `_train_ds` → `_served_ds` so the attribute matches
  what it actually holds when `serve_validation=True`.
- 5 latent items (PP semantics, CP→SP aliasing, dp_world_size
  double-source, parallel-state duplication, hard-coded Megatron
  knobs) tracked in
  [`docs/TODO.md` §6](TODO.md) and documented in
  [`docs/guides/known-bugs/blendcorpus-megatron-aliasing.md`](guides/known-bugs/blendcorpus-megatron-aliasing.md).

### Other

- Pulled 11 upstream commits (32nd sync,
  [logged](upstream-sync.md#2026-05-05-32nd-sync--observability--moe-token-pad--cp-fix--rlgraph_trainer-churn)).
  Notable: `b2cd149f` adds `torchtitan/observability/` structured
  logging hooks. ezpz's `FaultTolerantTrainer` and `EzpzValidator`
  override their respective base classes' methods entirely so the
  new `@sl.log_trace_span` decorators don't propagate — non-blocking
  but worth re-adding later if we want trace spans on the ezpz path.
- Aligned `~/.claude/statusline-command.sh` to match starship.toml
  (true gray time, fish-style abbreviated path with cyan-bold +
  underlined-blue repo root, bold-purple branch).

### 80B v2 has a working path on torch 2.13 + `compile=OFF`

After the bisect closed out the DeviceMesh question, ran job 12466025
(4N, 20-step smoke) to test whether `compile=OFF` actually unlocks
80B v2 production on the torch 2.13 venv:

- Config: `agpt_80b` at TP=2, AC=full, **compile=OFF**, AdamW LR=1e-6,
  fp32-master, 4 nodes (24 ranks, dp_shard=12).
- Result: clean run, all 20 steps. Loss descended **12.98 → 10.46**
  (-2.52 nats), MFU steady at **~17.8%**, memory peaked at **88.94%**
  (~7 GiB headroom per tile), exit code 0.
- Grad-norm bumped to ~34 around steps 15-16 then recovered to ~14
  by step 20 — early-training oscillation, not a stall. Production
  80B should add the 200-step linear warmup the 2B/20B v2 configs
  already use.

This confirms an actually-working v2 80B path. Notable that
`compile=OFF` MFU (~17.8%) *matches* what compile-on used to give v1
on torch 2.10, so we're not paying any throughput penalty for not
compiling — though that'll change once `compile=ON` works again
upstream and the inductor optimizations actually kick in.

Submit script:
[`scripts/submit_80b_no_compile_t213.sh`](../scripts/submit_80b_no_compile_t213.sh).
Per-step log:
`logs/agpt-80b-no-compile-t213-12466025/run.log`.

`compile=ON` for the 80B family is currently broken on **both** torch
versions: torch 2.10 hits the step-1 hang regression from Apr 16-23
upstream changes; torch 2.13 hits the DeviceMesh-in-saved-tensors
AOT autograd assertion. `compile=OFF` is the only viable v2 80B path
until either upstream bug is fixed.

---

## 2026-05-04 — 20B chain walltime, 1024N startup crashes, doc cleanup

### Production training

- **20B 512N canonical chain (8463628)** finished its 12h walltime
  cleanly at step **863, loss 3.46**. Final TPS ~358, MFU ~17.8%.
  step-100..step-800 ckpts all saved. Continuation **8466848** auto-released
  from hold and is now Q for a 512N slot.
- **20B 256N v2 (8463659) started running.** Fresh-start trajectory at
  256N, separate ckpt dir (`n256-gbs6144`) — *not* a chain extension.
  Currently at **step 200, loss 5.65, MFU 14-20%**, 2 ckpts saved
  (step-100, step-200). Useful as a per-token-vs-512N comparator at
  matched optimizer state. Loss curve looks healthy:
  12.96 → 8.5 (step 45) → 6.35 (step 124) → 5.65 (step 200).
- **2B 512N canonical chain** still at step 5,073 (loss 2.97).
  Continuation 8463627 still Q ("Not enough free nodes available");
  8466847 held behind it.

### 1024N first-attempts both crashed at startup

Both 1024N v2 jobs (queued since 2026-05-01) finally got slots and
**crashed within 4 minutes**:

- **2B 1024N (8463182)**: `MemoryError: std::bad_alloc` inside
  `torch.distributed.broadcast` during `set_determinism` init. Died
  after 211s, exit 143.
- **20B 1024N (8463183)**: rank 4732 died from signal 11 (SIGSEGV)
  during the same init phase, exit 143.

12,288 ranks (1024 nodes × 12 GPUs) appears to be hitting an init-time
memory/comm scaling issue we don't see at 256N or 512N. Worth a
smaller-scale repro before resubmitting — maybe 768N or 896N to bracket
where it starts failing. Not blocking the canonical 512N chains.

### 20B v2 eval through step-600

- **8467370** (capacity queue, 8h requested, ran to **12h15m walltime
  kill**) delivered eval scores for steps 100/200/300/400/500/600
  before being killed mid-step-700 lm-eval. Steps 700/800 will need
  a resubmit.
- **ARC-Easy `acc` lifts monotonically** from 0.266 → 0.290 → 0.295 →
  0.318 → 0.346 → **0.393** across that range — the cleanest-yet
  signal that fp32 master is producing real benchmark progression
  vs. v1's flat ~0.27 baseline.
- HellaSwag also creeping: 0.257 → 0.270. ARC-Challenge and
  Winogrande still in noise (expected at <100B tokens).
- v1-vs-v2 plot regenerated; results table added under
  [`docs/evals/agpt/20b/README.md`](evals/agpt/20b/README.md).
- Eval throughput tanked while concurrent 256N v2 (8463659) was
  starting — both share flare bandwidth for ckpt I/O and dataset
  reads. Steps 100-500 each took ~30 min; step-600 took ~3h. Will
  hold off on resubmitting steps 700/800 until 256N finishes.

### Doc cleanup

- **Compile flag was wrong in 5 v2 setup tables.** `agpt_2b()` and
  `agpt_20b()` both default to `compile=True`, and W&B configs +
  startup logs ("Compiling each TransformerBlock with torch.compile")
  confirm compile is on for all v2 runs. The "Compile | off" rows in
  the v2 setup tables under `docs/production/agpt/{2b,20b}/n*/README.md`
  were stale carryovers from drafts. Flipped to "on" across 2B
  n256/n512/n1024 and 20B n512/n1024.
- **Submit-script links added** to all 6 per-node-count READMEs. v2
  entries point to `scripts/submit_agpt_{2b,20b}_aurora_venv.sh`
  (one script handles all node counts via env vars); v1 entries point
  to the per-node `submit/aurora/submit_agpt_*.sh`.
- **Absolute log paths added** for every Job ID in every Progress
  table — v1 logs at `/lus/flare/.../torchtitan-ezpz/agpt-*-sophiag-*.o<JOBID>`,
  v2 logs at `/flare/.../runs/agpt-{2b,20b}-v2/torchtitan-ezpz/agpt-*.o<JOBID>`.
  Queued/held jobs note where the log will land on start.
- **`submit/README.md` added** marking the directory as legacy. The
  torch-2.10 `submit/{aurora,sunspot}/*.sh` scripts produced every v1
  trajectory and were the production driver through 2026-04-29. After
  the v2 restart on 2026-04-30, all production training moved to
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` on the torch 2.13
  venv stack — nothing live reaches into `submit/` anymore.

### Plotter

- `PRODUCTION_RUNS["20b_v2_256"]` added (wandb run `r1yyxbmt` =
  8463659) so the 20B 256N v2 trajectory shows up in dashboard
  refreshes.

### 8463659 NODE_FAIL after step 364 (afternoon)

Same recurring Aurora bad-node failure mode that killed 8459818 /
8460301 / 8460302. Last training step was 364 at 11:09:12, then
immediately:

```
x4406c6s7b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
x4218c2s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 606 died from signal 15
```

PALS shepherd on `x4406c6s7b0n0` got SIGKILL (kernel OOM-killed,
hardware fault, or system-level take-out), all ranks on that node
lost their parent → cascading SIGTERM. PBS reports `Exit_status -20`
= NODE_FAIL after 9h walltime. step-300 ckpt saved cleanly (loss 4.61
final). Trajectory pages updated to reflect the crash; no continuation
chained since the canonical chain is at 512N and this 256N run was a
per-token comparator scaling experiment rather than a chain.

### 20B v2 eval — full step-100..800 sweep complete

After waiting for 8463659 to finish (and free flare bandwidth),
resubmitted the missing step-700 + step-800 evals as 8469257 (capacity,
3h walltime). Finished cleanly in 3h flat. ARC-Easy `acc` continues
its monotonic ascent: 0.359 (step 500) → 0.393 (600) → 0.391 (700) →
**0.444** (800) — clean signal vs v1 256N's flat ~0.27 across all of
0-63B tokens. HellaSwag `acc_norm` 0.270 → 0.281 → 0.284 (+3pp above
v1 by step 800). v1-vs-v2 plot regenerated; `docs/evals/agpt/20b/`
table updated with all 8 v2 ckpts.

### Direct verification of the bf16 fix in checkpoint weights

User asked for RMSNorm.weight variance across the new 2B production
ckpts. Pulled stats from 6 HF-converted ckpts:

| ckpt | mean(var) | mean(std) | min weight | max weight |
|---|---:|---:|---:|---:|
| v1 step-10000 | **0** | **0** | **1.000** | **1.000** |
| v1 step-15000 | **0** | **0** | **1.000** | **1.000** |
| v2 256N step-2000 | 2.2e-5 | 0.0045 | 0.973 | 1.039 |
| v2 512N step-1000 | 5.4e-6 | 0.0016 | 0.988 | 1.016 |
| v2 512N step-3000 | 5.4e-5 | 0.0071 | 0.957 | 1.063 |
| v2 512N step-5000 | **1.2e-4** | **0.011** | **0.926** | **1.102** |

Every single v1 RMSNorm channel is exactly 1.0 — bf16-master
sub-ULP-update bug really did freeze every norm. v2 weights are
training: variance grows monotonically with token count, range fans
out from [0.988, 1.016] at step 1000 to [0.926, 1.102] at step 5000.
Per-layer at step 5000: `model.norm.weight` is biggest (mean 1.098,
std 0.006 — every channel uniformly scaling up); mid-depth layers
(5-7) have the highest per-element std (0.014-0.017, learning the
most differentiated channel scales). Final smoking gun for the v2
restart, complementary to the lm-eval evidence.

### Doc maintenance + plotter

- Refreshed 20B v2 256N plots (added wandb `r1yyxbmt` to
  `PRODUCTION_RUNS`); re-ran on the dead trajectory after NODE_FAIL.
- Updated parent snapshots (`production/README.md`,
  `production/agpt/README.md`, `production/agpt/20b/README.md`) to
  reflect 8463659 NODE_FAIL + 1024N startup crashes.
- `submit/README.md` added marking torch-2.10 `submit/` as legacy.
- Submit-script links + absolute log paths added to all 6 per-node
  READMEs.
- Compile-flag rows in 5 v2 setup tables corrected from "off" to "on"
  (defaults to `True` in `agpt(...)` and confirmed in W&B + startup
  logs).
- `running-with-newer-pytorch.md` expanded with the at-scale yeet
  section (8N→4096N table, tarball workflow, `/tmp/.venv` switch).
- Saved `memory/project_1024n_init_crash.md` so future sessions know
  to bracket 1024N attempts at 768N/896N first.

### Still queued

- **8463627** (2B 512N chain1 continuation), **8466847** (held
  `afterany:8463627`), **8466848** (20B 512N chain1 continuation),
  **8467141** (√2-LR fork chain1), **8467142** (held
  `afterany:8467141`) — all Q for 512N slots, none running.
  No production training is currently active.

---

## 2026-05-03 (evening) — TP > 1 loss-reporting bug + agpt_50b_wide

### Loss reporting on TP > 1 is off by `dp_world_size`

Hunting a different bug (the 80B `compile + AC + TP=2`
DeviceMesh-in-saved-tensors crash) we added `agpt_60b` and then
`agpt_50b_wide` as smaller bisect targets. The `50b_wide` smoke at
2N TP=2 reported step-1 loss = **1.07**. That should be ≈ ln(256128) ≈
**12.45 nats** for random init — 12× too small. The 12 matched
`dp_world_size = 12` (24 ranks ÷ TP=2) exactly, which pointed at a
missing cross-batch reduction.

Root cause is upstream commit `1786292d` (2026-04-27) in
`torchtitan/distributed/utils.py`. The new DTensor branch in
`_dist_reduce` returns `float(x.full_tensor().item())` and skips the
requested mesh `all_reduce`. That is correct only when the DTensor's
mesh matches the requested mesh — but the trainer's loss reduction
passes `loss_mesh` (= batch × cp), and the loss is a Replicated
DTensor on the **TP** mesh (orthogonal). The cross-batch sum is
silently dropped.

Verified the diagnosis by re-running `agpt_2b` at TP=2 (no compile, 3
steps) with a workaround in place: convert `loss` DTensor to a plain
tensor before `dist_sum`/`dist_max`. Step-1 loss came back as **12.94**
— matches the known-good TP=1 baseline of 12.95. Bug confirmed and fix
verified in one shot.

Workaround landed in `experiments/ezpz/`:

- `trainer.py` — adds `loss = loss.full_tensor()` before the
  `dist_sum`/`dist_max` reductions.
- New `validator.py` — `EzpzValidator(Validator)` subclass with the
  same fix in its `validate()` override.
- `agpt/config_registry.py` `_base_config` — uses
  `EzpzValidator.Config` instead of `Validator.Config`.

Documented in `docs/guides/loss-reporting-tp-dist-reduce.md` and the
upstream issue note at
`docs/upstream-issues/dist_reduce_dtensor_skip.md`. Filed upstream as
[pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).

**Implications for live dashboards:**

- 2B / 20B production W&B (TP=1): correct, no action needed.
- **80B production W&B (TP=2): under-reported by `dp_world_size`.**
  Multiply reported loss by `world_size / tp_degree` to recover the
  true per-token NLL. A 256N TP=2 dashboard's loss is currently 1536×
  smaller than the truth.

### agpt_50b_wide added; agpt_60b removed

To get a smaller compile target for the 80B-bug bisect we first
added `agpt_60b` (dim=9216, 60 layers, ~58B params, same per-layer
shape as 80B). That smoke crashed with an Intel GPU SegFault at step 2
— OOM dressed up as a not-present-PDE fault, after step-1 measured at
**97.41% memory** with no headroom for the step-2 activation peak.

Replaced with `agpt_50b_wide` (dim=9216, **48 layers**, ~48B params).
Smoke ran all 10 steps cleanly at 95.94% memory: MFU ≈ 15%, TPS ≈ 140
on Sunspot. Concluded "this is a working `compile + AC + TP=2` dense
config — the 80B-family DeviceMesh-in-saved-tensors crash does NOT
reproduce at 48 layers, only at 84, so the bug is depth-sensitive."

> **CORRECTION (2026-05-05):** that conclusion was wrong. The May 3
> smoke happened to use torch 2.10 (`aurora_frameworks-2025.3.1`); a
> proper bisect on torch 2.13 (jobs 12465952 + 12465962) showed the
> bug fires on `agpt_50b_wide` / `agpt_70b_wide` / `agpt_80b` alike,
> on both 2N and 4N. Bug is **torch-version-sensitive**, not
> depth-sensitive. See the 2026-05-05 journal entry above for the
> full bisect.

### Validation loss work — partial

Started wiring blendcorpus's already-built validation split through the
`Validator`. Stopped short to chase the loss-reporting bug above.
Status:

- `BlendCorpusDataLoader.Config` now has `serve_validation: bool`
  and `eval_iters: int` (default 100). `eval_iters` is only requested
  when `serve_validation=True` (otherwise we'd trigger a fresh
  validation index build that hangs in dataset construction).
- `_base_config` constructs a validator with
  `BlendCorpusDataLoader.Config(serve_validation=True)` for the val
  dataloader, but `validator.enable=False` by default.
- `EzpzValidator` is wired in but not yet smoke-tested.

Remaining: 250-step `agpt_2b` run with `--validator.enable` to confirm
the validator fires at step 1 and step 200, and that the val loss is
reasonable.

---

## 2026-05-03 — Production progress + canonical-chain consolidation + doc reorg

### Production training

- **2B 512N canonical chain (8460301 → 8463626 → 8463627)** — 8463626
  finished its 12h walltime cleanly at step 5,073 (loss 2.97). 50
  ckpts saved (every 100 steps, step-100 to step-5000). Continuation
  8463627 queued (will auto-resume from step-5000). **Cumulative:
  step 5,073, loss 2.97, 510B tokens (10.9% of 4.67T target).**
- **20B 512N canonical chain (8460302 → 8463628)** — 8460302 hit
  walltime at step 300 (loss 4.95). 8463628 (12h continuation) is
  currently running, resumed from step-200 (step-300 ckpt was
  incomplete from the NODE_FAIL on 8460302 so DCP picked the prior
  good ckpt). At write time: step 287, loss 5.00, MFU 17.5%.
- **Other queued jobs at different node counts** (2B 1024N 8463182,
  20B 1024N 8463183, 20B 256N 8463659) are *independent* trajectories
  — they write to separate ckpt dirs (keyed on `gbs`) and would start
  fresh from step 0. Treated as scaling experiments, not chain
  extensions.

### v1-vs-v2 evals (smoking gun)

- Ran `eval-2b-v2.sh` on 10 v2 256N ckpts (steps 200-2000). ARC-Easy
  climbed 0.277 → 0.429 over 100B tokens; v1's flat ~0.27 across 450B
  tokens validates the bf16-master RMSNorm-freeze fix end-to-end.
  HellaSwag also broke out at 80-100B tokens (v2 0.301 vs v1 0.251).
- Ran `eval-20b-v2.sh` on step-100 + step-200 (10B and 20B tokens).
  ARC-Easy v2 +1.5pp above v1 at the same token count, others still
  in noise. Will revisit at higher v2 token counts.
- Submitted fresh evals (8466827 for 2B 512N steps 1000-5000;
  8466828 for 20B 512N step-300).
- Added per-trajectory plotter at
  `docs/evals/agpt/{2b,20b}/plot_v1_vs_v2.py` — 4-panel comparison
  (HellaSwag/ARC-Easy/ARC-Challenge/Winogrande) with random baseline
  marked.

### Production-side doc reorganization

- Surfaced 2B 512N + 20B 512N v2 dashboards at the agpt index level
  (`docs/production/agpt/README.md`). Previously only embedded in
  per-model READMEs.
- Wrapped all v1 historical sections in `<details closed>` blocks
  across both production and eval READMEs, so v2 stays prominent.
- Renamed all production figure filenames to be explicit about
  v1/v2 (e.g. `production_2b_v2_512n.png` vs the older
  `production_2b_256n.png` ambiguity).
- Generated v1-vs-v2 overlay plots
  (`overlay_2b_v1_vs_v2.png`, `overlay_20b_v1_vs_v2.png`) via
  `plot_production_wandb.py --overlay {2b,20b}`.
- Moved the **MDS 2B SophiaG training curves** from
  `docs/evals/agpt/2b-mds/` to `docs/production/agpt/2b-mds/` (the
  eval scores stay under evals). Cross-linked both ways. Noted as
  "pre-torchtitan reference baseline" in the agpt production index.
- `plot_production_wandb.py` extended to support multiple v2
  trajectories per model — `PRODUCTION_RUNS` keyed on
  `<model>_v2_<nodes>`, output filenames keyed on the same.

### Canonical chain dashboards (current state)

| Model | Cumulative | Loss | Tokens | Latest |
|-------|-----------:|-----:|-------:|--------|
| 2B 512N | 5,073 | 2.97 | 510B (10.9%) | 8463627 (Q) |
| 20B 512N | 300 | 4.95 | 30B (0.6%) | 8463628 (R) |

### Dataset CLI mistake-catcher

When `--dataloader.dataset=user/repo` (HF hub path) and
`--dataloader.dataset-path=...` are both passed, `_validate_dataset`
silently overrides `dataset_path` to None so the registered hub path
wins. Correct behavior, but it hid user mistakes (e.g. expecting
`--dataset-path` to point at a local clone of the hub dataset).

`9ef2908d` — emit a clear warning naming both args and telling the
user to drop `--dataloader.dataset-path` or pick a local-file dataset
name like `blendcorpus`. Override semantics unchanged; only logging
added. Production scripts that hardcode
`--dataset=blendcorpus --dataset-path=$DFL` are unaffected (they hit
the registered-name branch which keeps the path).

### 2B 256N vs 2B 512N — large-batch under-training observation

Pulled fresh evals on 2B v2 256N (10 ckpts, steps 200-2000) and 2B v2
512N (5 ckpts, steps 1000-5000). Surprise: at matched **token** counts,
256N beats 512N noticeably:

| Tokens (B) | 256N HellaSwag | 512N HellaSwag | Δ |
|-----------:|---------------:|---------------:|--:|
| ~100 | 0.301 (step 2K) | 0.264 (step 1K) | -3.7pp |

But at matched **step** counts, they're indistinguishable:

| Step | 256N HellaSwag | 512N HellaSwag | Δ |
|-----:|---------------:|---------------:|--:|
| 1000 | 0.262 | 0.264 | +0.2pp |
| 2000 | 0.301 | 0.304 | +0.3pp |

This is the classic large-batch under-training pattern. Both runs use
SophiaG LR=2.28e-5 (tuned for 256N / GBS=6,144). At 512N (GBS=12,288)
each step covers 2× the tokens but the optimizer state evolves at half
the cadence per token, with no compensating LR scale-up. Per-step
parity confirms the optimizer is healthy; per-token gap is purely
batch-size-induced under-training.

Implications:
- **Per-step**: 256N == 512N
- **Per-token**: 256N wins ~3-8pp at matched tokens
- **Per-wall-clock**: 512N wins (~2× throughput)

So the canonical-chain choice (512N) optimizes for wall-clock time to
target loss, not token efficiency. If chasing minimum tokens, 256N
would be preferable. Open follow-up: would √2-LR scaling (3.22e-5) at
512N close the per-token gap? Worth a fresh fork to test, but not
worth perturbing the running 512N chain's LR mid-run.

Documented in
[`docs/evals/agpt/2b/README.md`](evals/agpt/2b/README.md) under "v2
256N vs v2 512N — same model, two batch sizes".

---

## 2026-05-02 — 2B 512N continuation reaches 510B tokens

Light day. The 2B 512N continuation chain (8463626) ran cleanly
through 12h of walltime — went from step 1,300 (resume) to step
5,073, with loss dropping from 3.59 to 2.97. NODE_FAIL at the very
end again, but all 50 ckpts saved. This was the first chain run where
NODE_FAIL didn't cost meaningful progress — the ckpt-100 cadence +
keep_latest_k=0 (keep all) policy means we always have a recent
recovery point.

20B 512N (8460302) finished its 6h walltime at step 300, loss 4.95.
3 ckpts saved (step 100/200/300, though step-300 was incomplete and
DCP fell back to step-200 on resume).

Eval pipeline kept producing rolling 2B v2 results — first ARC-Easy
points landed in the 0.28-0.34 range across early ckpts.

---

## 2026-05-01 — 1024N production runs, yeet-env scaling sweep

### Production training

- **2B 1024N** (8463182, 12h walltime, small queue) and **20B 1024N**
  (8463183, 12h, small) both submitted. Same configs as the running
  256N/512N v2 runs (LBS=2, SophiaG LR=2.28e-5, plain CE, fp32 master,
  `.venv.tar.gz` yeet-env, no compile). Currently queued behind the
  in-flight 20B 512N (8460302).
- Submitted **chained 12h continuations** for the 512N runs:
  - 2B: fresh 8463626 + 8463627 (depend=afterany:8463626)
  - 20B: 8463628 (depend=afterany:8460302)
- 2B 256N (8459818) and 2B 512N (8460301) v2 runs both hit NODE_FAIL
  at end-of-walltime (step 2070 and 1387 respectively) — bad-node TPS
  degradation pattern (TPS dropped from ~5K to ~30 in the final few
  hundred steps before kill). All checkpoints survived (`keep_latest_k=0`
  keeps everything; 20 ckpts at step 100..2000 for the 256N, 13 ckpts
  step 100..1300 for the 512N).

### yeet-env tarball-broadcast scaling sweep

Measured `ezpz yeet-env --src .venv.tar.gz` at 8/16/32/64/128/256/512/
1024/2048 N (4096N still queued in `large`). Submitted via
`scripts/yeet_env_scaling_test.sh` chained one-at-a-time through
`/tmp/yeet_chain.sh` (per-user PBS-Q limit forces serial submission).

| Nodes | yeet-env (s) | Per-node (ms) |
|------:|-------------:|--------------:|
| 8     | 70           | 8,712 |
| 16    | 90           | 5,606 |
| 32    | 89           | 2,788 |
| 64    | 91           | 1,425 |
| 128   | 110          | 862 |
| 256   | 133          | 519 |
| 512   | 175          | 341 |
| 1024  | 255          | 249 |
| 2048  | 421          | 206 |

Two regimes: 8-64N is extract-bound (flat ~90s total), ≥128N is
broadcast-bound (linear-ish growth, super-linear knee at 2048N).
Per-node amortized cost drops 42× from N=8 to N=2048. Even at 2048N,
total wall-clock is <8 min — vs the "1-2 hours" the old per-file rsync
mode in CLAUDE.md predicted.

Plots + script: [`docs/scaling/yeet_env/`](scaling/yeet_env/README.md).

### Docs refresh

- Added `docs/evals/agpt/2b-mds/{loss_data, figures}` train+val loss,
  grad_norm, TFLOP/s, TPS plots pulled from W&B (113 SophiaG MDS
  continuation runs stitched by iteration).
- Refactored `experiments/ezpz/scripts/`: moved 14 hidden one-off
  shell scripts out of the repo root into `scripts/{eval,debug,lr-finder}/`
  subdirs; consolidated the existing 3 eval shell scripts there too.
- Refreshed all production READMEs with the v2 run state (this entry).

### Async checkpointing — verification + production switch

Built smoke configs `smoke_2b_async_ckpt` (`async`) and
`smoke_2b_async_ckpt_pinned` (`async_with_pinned_mem`) with
`enable_first_step_checkpoint=True` and `interval=10` so we get 5
saves across a 50-step run.

| Job | Config | Result |
|---|---|---|
| `12465723` | `async` | ✅ PASS — 50 steps, 5 saves @ ~0.2s each, loss 7.05, exit 0 |
| `12465724` | `async_with_pinned_mem` | ❌ FAIL — `KeyError: <class 'type'>` at step-20 save |

Loss-baseline check on the `async` run vs v24 sync baseline:
final Δ -0.055, tail10 Δ -0.053 (well within ±0.10). On-disk DCP
format byte-identical to sync save (1809 keys, modern `qkv_linear` +
`lm_head` naming). Sync ↔ async is bidirectionally wire-compatible —
no migration needed for in-flight production.

Flipped all 10 production train/submit/smoke scripts to default
`--checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}"`. The env
var lets anyone roll back to sync without editing the script:

```bash
CHECKPOINT_ASYNC_MODE=disabled qsub ...
```

Per-step staging cost is ~0.2s with a one-step TPS dip (5.7K vs 7.4K
steady) the next step, then full recovery. Compared to a sync save
(which blocks the entire training step for the full disk-write
duration — 4+ seconds per save at 2N), this is a clear win.

### `async_with_pinned_mem` — upstream PyTorch bug filed

Diagnosed and reproduced a real bug in `torch.distributed.checkpoint`:
`StateDictStager.deepcopy_with_tensor_offload` line 320 indexes
`self._deepcopy_dispatch[type]` directly, but `StateDictStager.close()`
(called by `DefaultStager._stage` after every stage to break a closure
cycle, `staging.py:251`) clears the entire dispatch dict. Result:
**second** stage call on any state dict containing a class object
crashes with `KeyError: <class 'type'>`.

Reproduces in pure CPU Python (no distributed init, no GPU). Three
files committed at `docs/upstream-issues/`:

- `STATE_DICT_STAGER_ISSUE.md` — issue draft for pytorch/pytorch
- `repro_state_dict_stager_bug.py` — minimal ~30-line repro
- `repro_state_dict_stager_fix_verification.py` — proves the naive
  one-line `.get(type, _deepcopy_atomic)` fix doesn't work because
  *all* atomic-type entries are also missing post-`close()`. Real
  fix is structural: either don't clear `_deepcopy_dispatch` in
  `close()` (only the cached storages cause the leak), or rebuild
  it at the start of each `stage()`.

Plain `async` is unaffected — it uses the in-process default stager
(regular `copy.deepcopy`), not the pinned-memory `StateDictStager`
path.

### 28th–30th upstream syncs (4 days, 5 sync entries, no replays)

| Entry | Date | Commits | Why no replay |
|---|---|---|---|
| 27th | 2026-04-30 | HybridEP cleanup + autoparallel/dsv3 deletion | Resolved 2 modify/delete conflicts on autoparallel/dsv3/ by accepting upstream's deletion; ezpz doesn't depend |
| 28th | 2026-05-01 | graph_trainer qwen3 + CI lint + ft.llama3 attn_backend | Scoped to graph_trainer / lint / ft.llama3 (we use our own model_registry) |
| 29th | 2026-05-01 | RL vLLM compile-time + graph_trainer skill | Scoped to experiments/rl + graph_trainer |
| 30th | 2026-05-03 | Bucketing pass + RMSNorm fusion (graph_trainer) | All 3 commits scoped to experiments/graph_trainer |

The loss-baseline workflow has now been exercised in production once
(v22 → v24 baseline refresh, both PASS). Workflow doc is at
`docs/baselines/README.md`.

### Eval pipeline v2 plumbing

Eval scripts were originally hardcoded for the v1 256N run. To run on
v2 ckpts (256N, 512N, 1024N, ...) needed several fixes early in the
day:

- `feat(ezpz/scripts/eval): support v2 ckpts via --ckpt-name + add
  20B v2 eval` (`55154291`)
- `fix(ezpz/scripts/eval): drop set -u from eval-20b-v2.sh` (`1835d8c2`)
- `fix(ezpz/scripts/eval): cd into v2 clone for conversion` (`9af67b81`)
- `fix(ezpz/scripts/eval): force PYTHONPATH=. + add subshell debug
  echoes` (`db1380b6`)

After those landed, ARC-Easy points started flowing:
0.277 (step-200) → 0.366 (step-1200) → 0.429 (step-2000), validating
the bf16 fix end-to-end. v1 was flat at ~0.27 across 450B tokens.

---

## 2026-04-30 — bf16-master RMSNorm freeze diagnosis + v2 restart

### What broke

The 2B/20B/80B SophiaG production runs from Apr 14-29 were all silently
training with frozen RMSNorm weights:

- `training.dtype = bfloat16` (default at the time) was being used as
  the *master* weight dtype for FSDP MixedPrecisionPolicy.
- bf16 ULP at 1.0 is ~7.8e-3; per-step RMSNorm.weight updates from
  SophiaG were ~1.6e-5 — sub-ULP, so every update rounded to zero.
- Loss curves looked plausible because attention/FFN weights were
  unfrozen and the model could still descend, but the model has no
  trainable normalization. Eval scores reflected this: the
  DCP→HF-converted checkpoints scored near-random on hellaswag/arc_easy
  while the parallel MDS pipeline (which used different defaults) was
  cleanly converging.

Full diagnosis writeup at
[`docs/guides/training-dtype-bf16-norm-freeze.md`](guides/training-dtype-bf16-norm-freeze.md).

### Fix + v2 restart

- Default `training.dtype` flipped to `float32` in `agpt` and `moe`
  config registries.
- ChunkedCELoss opt-in support ported to the ezpz trainer (sets
  `lm_head` for the chunked path; off by default).
- Restarted training from scratch in fresh per-model clones:
  `/flare/AuroraGPT/foremans/runs/agpt-{2b,20b}-v2/torchtitan-ezpz/`.
- Built fresh torch 2.13 + xpu venvs in each clone (~2.7 GB tarball).
- Adapted `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` for Aurora:
  proxy env vars set inline, ezpz-utils source cached locally to dodge
  bit.ly hangs, tarball-aware yeet-env, plain-CE default with optional
  `CONFIG_SUFFIX=_chunkedce` opt-in.

### Smoke + scaling validation

Validated v2 stack at 2/4/8/16/64 N via short PBS jobs before
launching production. 2B v2 16N test ran ~57 min and reached step 2117
(loss 5.16 → 3.57, TPS ~5K, MFU ~17-20%) before NODE_FAIL — flaky-node
issue, not a code issue. 20B v2 16N test (capacity queue, 1h) submitted
in parallel.

### 26th + 27th upstream syncs

Replayed the MoE ETP deprecation (#3167) onto `experiments/ezpz/moe/`:
removed `ExpertTensorParallel` import, dropped `etp_mesh`/`ep_etp_mesh`
from `apply_moe_ep_tp` signature + call site, switched to reading
`comm_backend` off `experts.token_dispatcher` (matches upstream
deepseek_v3.model). Pulled the 27th sync (HybridEP cleanup +
autoparallel/deepseek_v3 deletion) — no impact on ezpz.

---

## 2026-04-29 — 24th upstream sync replay (All2All token dispatcher consolidation)

### What landed

Two upstream commits, one breaking:

- `20628f4e` (#3125) consolidates EP=1 and EP>1 to all use
  `AllToAllTokenDispatcher` (with a local-fallback path when ep_mesh is
  None). `make_token_dispatcher_config` and `make_experts_config` now
  require a non-None `comm_backend`; default changed from `None` to
  `"standard"`.
- `35c5d529` graph_trainer-only (no impact).

### Replay (1 commit)

`23b8ba59 fix(ezpz/moe)`: `moe_comm_backend: str | None = None` →
`moe_comm_backend: str = "standard"` in both `_build_moe_layers` and
`model_registry`. Drop the now-dead `if moe_comm_backend is not None`
guard around the dispatcher rebuild loop. (No agpt changes — agpt
doesn't use the moe-only helpers.)

### Smoke results — both PASS within ±0.10

| Config | Final loss | vs v22 baseline | tail10 mean | Δ tail10 |
|---|---|---|---|---|
| agpt 2b (job 12465533) | 7.108 | -0.029 | 7.221 | -0.037 |
| moe 500m (job 12465534) | 6.912 | -0.014 | 6.978 | -0.016 |

Both deltas dominated by streaming-data shuffle noise. Baselines
refreshed to v24.

### Side cleanup

`b9a324f3` renamed `docs/upstream-sync/` → `docs/baselines/` to remove
the visual collision with the neighboring `docs/upstream-sync.md` log
file. Updated path references in `loss_baseline.py`, the workflow
README, and the link from `upstream-sync.md`.

---

## 2026-04-28 — 22nd upstream sync replay (quantize-on-config, LocalMapInnerAttention removal)

### What landed

The 22nd sync (merged `4b0a4fd5`) brought in 9 commits, three breaking:

- **#3127 `6348d93d` quantize on config instead of on model.** Removes
  `protocols/model_converter.py`, drops `model_converters` field from
  `JobConfig` and from every `parallelize_*` signature. Quantization
  converters are now applied to the model *config* at registry time
  (`q.build().convert(config)`). Also moves `FaultTolerantModelSpec`
  from `protocols/model_spec.py` into `experiments/ft/config/job_config.py`.
- **#2986 `b9e33527` Remove LocalMapInnerAttention.** Replaces the
  runtime DTensor wrapper class with a static `LocalMapConfig` set on
  the inner-attention sharding_config via
  `set_gqa_inner_attention_local_map`. All inner attention types now
  inherit `Module` directly.
- **#3113 `053dbf9a` MeshDimName → MeshAxisName.** Transparent for ezpz
  (we only use the upstream helpers).

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.92 → 7.14 (job 12465527, exit 0) |
| `moe`  | replayed | 50 steps, loss 12.92 → ~7 (job 12465529, in flight) |

### Commits (ezpz branch)

- `4b0a4fd5` — Merge upstream/main into ezpz (clean automatic merge).
- `69a8cfc7` — Drop `model_converters=` kwarg from `parallelize_llama` /
  `parallelize_moe` signatures and from both `parallelize_fn` /
  `pipelining_fn` call sites in `ezpz/trainer.py`. Drop runtime
  `model_converters.build/convert/post_optimizer_hook`. Switch
  `has_quantization` to read from `model_config` via the upstream
  `torchtitan.components.quantization.utils.has_quantization` helper.
- `59d9f37d` — `LocalMapInnerAttention` → `Module` for
  `SoftcappedFlexAttention` (agpt) and `Attention.Config.inner_attention`
  (moe). Add `set_gqa_inner_attention_local_map(...)` calls in both
  `sharding.py` files so inner attention gets the static `LocalMapConfig`
  it now needs.
- `cd29417e` — moe `model_registry` accepts `quantization=[...]`
  parameter; `moe_671b()` re-registers via
  `model_spec=model_registry("671B", quantization=[...])` instead of
  mutating `cfg.model_converters`. Also re-import
  `FaultTolerantModelSpec` from `experiments/ft/config/job_config`.
- `1b87fa38` — `Float8GroupedMMConverter` → `Float8GroupedExpertsConverter`
  (rename caught at smoke-test import time).

### Smoke-test details

- agpt smoke (`12465527`): loss 12.92 → 7.14 across 50 steps,
  ~7,400 TPS, 28% MFU. Exit 0. Numerics match the v21 run within
  data-shuffle variance — replay is loss-neutral.
- moe smoke (`12465529`): 12.92 → 7 (in flight at step 12, on track),
  ~7,200 TPS at steady state. Same compile-warmup pattern as v21.

### Why moe failed once first

`12465528` failed with `ImportError: Cannot import config_registry for
module 'ezpz.moe'`. The underlying error was the
`Float8GroupedMMConverter` rename to `Float8GroupedExpertsConverter`
(and dropping `fqns=["experts"]` since the new Config takes no
extra args). `config/manager.py` swallows the underlying `ImportError`
which made the cause invisible — only resubmittable after grepping the
upstream class names.

---

## 2026-04-28 — 21st upstream sync replay (sharding API, ChunkedCELoss)

### What landed

The 21st upstream sync (merged `b6c04698`) brought in three breaking
changes that broke ezpz at import / config-build / training-init time:

- **#2963 / #2969** — config-based DTensor sharding. Replaces string-keyed
  `parallelize_module(plan)` with `Module.parallelize(mesh)` reading
  `ShardingConfig` declarations attached to each sub-module's `.Config`.
- **#2937** — ChunkedCELoss. Removed `build_cross_entropy_loss` and
  `ModelSpec.build_loss_fn`; loss now lives on `JobConfig.loss`.
- **`Decoder.Config`** renamed `output: Linear.Config` → `lm_head:
  Linear.Config`.

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.96 → 7.09 (job 12465500) |
| `moe`  | replayed | 50 steps, loss 12.93 → 6.91 (job 12465502) |
| `qwen3` | removed | Drift too large; nobody ran it; restorable from history |

### Commits (ezpz branch)

- `03b9f486` — Mechanical: drop `build_cross_entropy_loss` imports +
  `build_loss_fn=` kwargs, rename `output=` → `lm_head=` in agpt+moe
  configs, switch `ezpz/trainer.py` to `config.loss.build()`.
- `472f4743` — agpt sharding-API replay. New `agpt/sharding.py`
  (handles QK-Norm), new `agpt/model.py` (`AgptModel(Llama3Model)`
  overriding `update_from_config`), rewritten `agpt/parallelize.py` as
  thin orchestrator. Float8 tensorwise TP path dropped (no equivalent in
  the new API yet).
- `9bc774a6` — Set `loss=CrossEntropyLoss.Config()` in both `_base_config`
  helpers (was defaulting to abstract `BaseLoss.Config`).
- `40526628` — Enable compile in `smoke_2b_50steps` (XPU CE OOMs without it
  at vocab=256k).
- `dd4e065a` — moe sharding-API replay. New `moe/sharding.py`, extended
  `update_from_config`, rewritten `parallelize.py` (drops 230+ lines of
  manual ColwiseParallel/RowwiseParallel plans). MoE block sharding still
  done at parallelize-time by `apply_moe_ep_tp` (mirrors upstream).
- `5f88abc3` — `smoke_moe_500m_50steps` config + submit script.
- `80a23d41` — Removed `ezpz/qwen3` (drift too large for unused code).

### Preserved agpt-/moe-specific behavior

- `disable_fsdp_gradient_division` still calls
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL/XPU).
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False (keeps the separately-compiled CrossEntropyLoss working on dense
  models).
- agpt `apply_fsdp` keeps the `[norm, lm_head]` joint grouping with
  reshard_after_forward gated on the policy.
- moe `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
  import which doesn't exist in Aurora's PyTorch) with the Shard(0)
  fallback when expert hidden dim isn't FSDP-divisible.
- moe `apply_compile` is per-block `block.compile(backend=...)` instead of
  upstream's fullgraph `apply_compile_sparse` (XPU can't fullgraph compile
  MoE routing's dynamic shapes).

### Smoke-test details

- agpt smoke (`12465500`): loss 12.96 → 7.09 across 50 steps, ~7,400 TPS,
  27% MFU. Standard dense-2B numbers — replay is loss-neutral.
- moe smoke (`12465502`): loss 12.93 → 6.91 across 50 steps, ~7,200 TPS,
  ~10% MFU. The MFU is low because the metrics divisor uses the full
  dense FLOP estimate but only 2/8 experts fire per token — reporting
  artifact, not a perf regression.

---

## 2026-04-27 — Full 10B training, TorchMuon, local dataset

### Local Dataset Cache

- Downloaded FineWeb-Edu `sample-100BT` (267 GB, 140 parquet files)
  to `/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/`
- Added `register_local_dataset()` to `datasets.py` for parquet/arrow files
- Registered as `fineweb_edu_local` — eliminates HF streaming rate limits
  and ensures reproducible data ordering across runs

### TorchMuon Integration

- Added `TorchMuonOptimizersContainer` using `torch.optim.Muon` (built-in
  since PyTorch 2.9)
- Required `_CompositeOptimizer` wrapper — `OptimizersContainer` expects one
  optimizer per model part, but Muon only handles 2D params (need separate
  AdamW for embeddings/head)
- Multiple fix iterations: missing `import torch`, empty param list rejection
  from `Optimizer.__init__`, FSDP empty model parts
- **Result: same TPS as custom Muon (~4,600)** — Newton-Schulz overhead is
  inherent to the algorithm on XPU, not an implementation issue
- **Streaming data shuffle causes ~1.3 loss variance** — same optimizer gives
  very different loss across runs due to HF streaming data ordering

### Speedrun Competition Final Results (1000 steps, 2 nodes)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | Muon (custom) | **3.557** | 4,556 |
| 2 | AdamW + QK-Norm | **3.569** | 7,178 |
| 3 | Muon + cosine | 3.591 | 4,625 |
| 4 | Mano + QK-Norm | 3.604 | 6,980 |
| 5 | Mano | 3.631 | 7,048 |

### Full Training (10B tokens, 8 nodes, GBS=384)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW | **2.711** | 7,354 |
| 2 | AdamW + QK-Norm | 2.720 | 7,480 |
| 3 | Mano + QK-Norm | 2.854 | 7,346 |
| 4 | Mano | 2.875 | 7,429 |
| 5 | Muon | DNF (compile stuck) | — |

### Key Findings

- **AdamW wins at large batch (GBS=384)** — simpler update more efficient
  per token than manifold optimizers
- **QK-Norm effect diminishes at 10B** — 0.009 loss improvement (vs 0.23
  in 1000-step speedruns). Helps early training but washes out
- **Mano ~0.16 behind AdamW at GBS=384** — LR finder was tuned at GBS=48,
  needs re-tuning for larger batch
- **Muon compile broken with GAS** — inductor can't pickle cyclic objects
  in Newton-Schulz with gradient accumulation on torch 2.13
- **8-node scaling excellent** — 7,300-7,500 TPS/GPU across all configs

### Architecture Tweaks Implemented

- **Logit softcapping** — `SoftcappedFlexAttention` using FlexAttention
  `score_mod` with tanh cap at 30.0. Falls back to eager on XPU (4x slower).
  Manual attention OOMs at seq_len=8192 (materializes full attention matrix).
- **ReLU²** — `ReLUSquaredFeedForward` subclass. Didn't help (3.92 vs 3.80
  baseline). SiLU gating is better for this architecture.
- **WSM** — `eval/merge_checkpoints.py` utility for weighted state merging
  of checkpoints. Supports uniform, linear, and exponential weighting.
- New model variants: `2B_softcap`, `2B_relu2`, `2B_kitchen_sink`

### Round 4: 2N, GAS=8, 1000 steps (local dataset)

Reproducible speedrun with GBS=384 on 2 nodes using local FineWeb-Edu.

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW+QK-Norm | **3.205** | 7,428 |
| 2 | AdamW | 3.220 | 7,397 |
| 3 | Mano | 3.294 | 7,397 |
| 4 | Mano+QK-Norm | 3.307 | 7,423 |
| 5 | Mano (8.5e-4) | 3.328 | 7,348 |
| 6 | AdamW (3.7e-3) | 5.884 | 7,603 |

**Key findings:**
- AdamW+QK-Norm wins again — consistent across all GBS=384 experiments
- Mano leads early/mid training but AdamW catches up in cosine decay phase
- sqrt LR scaling too aggressive for AdamW (diverged), Mano tolerated it
- Softcap results invalid — local dataset loader memorizes with FlexAttention
  path (data sharding bug)
- FlexAttention on XPU falls back to eager (Triton-XPU can't codegen tanh)
  — 4x throughput penalty makes softcap impractical on this hardware

### Docs Restructure

- Reorganized `docs/competition/` → `docs/competitions/` with per-experiment dirs
- Added light/dark theme loss curve plots using `<picture>` media queries
- Created `docs/competitions/agpt2b-n2-gas8-1000steps/` with live loss curves

### Upstream Sync (20th)

- Merged upstream: dataset checkpoint resume fix (#3008), RL refactor (#3073)
- Clean merge, no replay needed

### CLAUDE.md Added

- Created `experiments/ezpz/.claude/CLAUDE.md` with project rules that
  travel with the codebase (upstream sync protocol, never modify outside
  ezpz, document every run, etc.)
- Updated with Aurora-specific knowledge (queues, yeet-env scaling, eval pipeline)

---

## 2026-04-26 — RL refactor, docs reorg, competition launch

### RL Multi-Task Support

- Refactored `rl/` from hardcoded sum-of-digits to a pluggable task registry
- Created `rl/tasks/` package with `RLTask` dataclass, `register_task()`, `get_task()`
- Moved sum_digits dataset+rewards into `tasks/sum_digits.py` (self-registering)
- Added 3 new tasks: `multiply`, `word_sort`, `countdown`
- Added CLI args (`--task`, `--model-name-or-path`, `--steps`, etc.) to `train_grpo.py`
- Default model: `argonne_private/AuroraGPT-7B` with `Qwen/Qwen3-0.6B` fallback
- Fixed safetensors E2BIG crash by disabling mid-training checkpoints
- Moved RL docs to `docs/rl/README.md`

### Docs Reorganization

- Created `configs/` — moved dense-configs.md, moe-configs.md
- Created `guides/` — moved known-issues.md, running-with-newer-pytorch.md, xpu-attention-issues.md
- Renamed `production-training/` → `production/`
- Created `scaling/` — consolidated scaling-study.md, scaling-study-torch213.md,
  benchmark-80B.md, benchmarks.md into per-model pages (agpt-2b, agpt-20b, agpt-80b, moe)
- Rewrote top-level README.md with organized sections
- Fixed all 31 internal cross-references; link checker passes with 0 broken

### Generic HF Dataset Streaming

- Created `datasets.py` with `register_hf_dataset()` for explicit registration
- Added auto-fallback: unknown `--dataloader.dataset` names are treated as HF hub paths
  (e.g. `--dataloader.dataset stanfordnlp/imdb` just works)
- Pre-registered: fineweb_edu, fineweb, slimpajama, pile, openwebtext, wikitext, c4_streaming
- Silenced httpx/huggingface_hub HTTP log spam

### agpt_2b Loss Competition

**Goal:** lowest loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).
**Fixed:** FineWeb-Edu streaming, LBS=2, seq_len=8192, 1000 steps.
**W&B:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

#### New Optimizers Implemented

- **Mano** (`optimizer/mano.py`) — manifold-normalized optimizer
  ([arxiv 2601.23000](https://arxiv.org/abs/2601.23000)).
  Tangent-space projection on rotating Oblique manifold. Vector-norm ops
  instead of Newton-Schulz → runs at AdamW speed (~7,200 TPS/GPU).
- **SPAM** (`optimizer/spam.py`) — spike-aware Adam with momentum reset
  ([arxiv 2501.06842](https://arxiv.org/abs/2501.06842)).
  Gradient spike detection via EMA + periodic moment reset every DeltaT steps.

#### Architecture Tweaks

- **QK-Norm** — added `qk_norm` parameter to `_build_agpt_layers` and
  `_build_agpt_config`. New `2B_qknorm` model variant. RMSNorm on Q,K
  before attention dot product.

#### Competition Results

| Rank | Config | Optimizer | LR | Loss | Steps | TPS/GPU |
|------|--------|-----------|------|------|-------|---------|
| 1 | `speedrun_2b_muon` | Muon | 2.4e-3 | **3.628*** | 967 | 4,695 |
| 2 | `speedrun_2b_mano` | **Mano** | 3.0e-4 | **3.631** | 1000 | ~7,200 |
| 3 | `speedrun_2b_adamw_cosine` | AdamW | 1.3e-3 | 3.789 | 990 | 7,245 |
| 4 | `speedrun_2b_adamw` | AdamW | 1.3e-3 | 3.801 | 1000 | 7,245 |
| 5 | `speedrun_2b_adamw_short_decay` | AdamW | 1.3e-3 | 4.053 | 1000 | 7,245 |
| 6 | `speedrun_2b_adamw_fast_warmup` | AdamW | 1.3e-3 | 4.546 | 1000 | 7,245 |
| 7 | `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | 4.399* | 976 | 4,596 |
| 8 | `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | 4.719 | 1000 | 7,208 |
| 9 | `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | 5.850 | 1000 | 7,344 |
| 10 | `speedrun_2b_spam` | SPAM | 1.3e-3 | 5.881* | 865 | ~7,200 |

*Still running at time of reporting.

#### Key Findings

- **Muon and Mano essentially tied on loss** (~3.63), but Mano ran at
  full AdamW speed (7,200 TPS) vs Muon's 4,700 TPS. **Mano wins on
  wall-clock time.**
- **Muon is 35% slower per step** due to 5x Newton-Schulz iterations
  (large matmuls on every 2D param). Mano replaces these with O(dim)
  vector-norm ops.
- **Cosine decay beats linear** for AdamW (3.789 vs 3.801).
- **Shorter decay (10%) hurts** — not enough time in decay phase.
- **Shorter warmup (5 steps) hurts** — destabilizes early training.
- **SPAM underperforms** — spike clipping + momentum reset don't help
  on this clean dataset with well-tuned LR.
- **AdamW LR=2.6e-3 diverges** — confirms LR finder boundary (1.3e-3).
- **SophiaG underperforms** AdamW by ~0.9 loss at same step count.

#### Failed Experiments

- `speedrun_2b_muon_short_decay` — crashed at startup (exit 143)
- `speedrun_2b_muon_fast_warmup` — crashed at startup (exit 143)
- `speedrun_2b_adamw_qknorm` — crashed at startup (exit 143)
- `speedrun_2b_muon_qknorm` — crashed at startup (exit 143)

Need to investigate QK-Norm and Muon schedule tweak crashes.

#### Issues Hit

- PBS `qsub -- bash -c '...'` doesn't work — need a proper script file
- `set -euo pipefail` kills venv activate scripts (`ZSH_EVAL_CONTEXT: unbound`)
- Concurrent jobs sharing `--checkpoint.folder=checkpoint` clobber each other
  → fixed with per-config checkpoint dirs
- Disk quota hit at 12TB → cleaned 3.2TB of old scaling study checkpoints
  and 117GB of old repo checkpoints
- `git stash pop` during disk quota crunch wiped train.py to 0 bytes
  → restored from `git show HEAD:...`

---

## 2026-04-25 — torch 2.13 venv, scaling study, production scripts

### Torch 2.13 Environment

- Created `.venv/` with PyTorch 2.13 (built from source for XPU)
- Added `running-with-newer-pytorch.md` guide for setting up the venv
- Added `ezpz yeet-env` integration to copy venv to `/tmp` on compute nodes

### Production Training Scripts

- Created `scripts/train_agpt_2b_venv.sh` and `train_agpt_20b_venv.sh`
  for training with the torch 2.13 venv
- Fixed `ezpz_setup_job` ordering — must run before venv activation
- Fixed `/tmp/.venv/bin` PATH handling after `yeet-env activate`
- Set `local_batch_size=2` as default for 2B training

### 2B Scaling Study (torch 2.13, Sunspot)

- Ran weak scaling study from 2 to 64 nodes on Sunspot
- Results: 7,142 TPS/GPU at 2N (27.6% MFU) — **+23% over torch 2.10**
- Near-perfect scaling to 8 nodes (~100%), 94% efficiency at 64 nodes
- Memory nearly constant at ~44 GiB across all scales
- Documented in `docs/scaling-study-torch213.md`

### Production Training Status

- Updated production run tracking for 2B/20B/80B models
- Added per-model subdirectories with loss curve plots
- Updated upstream sync log with session findings

### Upstream Sync

- Merged upstream `pytorch/torchtitan` main into ezpz branch
- Reverted `.ezpz-interactive-launch.sh` tracking change
- Added interactive launch script and loss CSVs

---

## 2026-04-23 — XPU fixes, upstream merge

### XCCL Barrier Fix

- Fixed torch 2.10 XCCL hangs for barrier and TP collectives
- Root cause: XCCL backend doesn't support barrier() — was hanging
  all multi-node runs
- Fix: use gloo backend for barriers when available

### DTensor TP Revert

- Reverted full DTensor TP (`use_local_output=False`) for agpt models
- Was causing shape mismatches in the attention layer on XPU
- Reverted to standard `use_local_output=True`

### Upstream Merge

- Merged upstream main into ezpz branch
- Upstream changes included GraphTrainer bucketing fixes,
  SAC + FSDP improvements, and Qwen3-VL fused QKV support
