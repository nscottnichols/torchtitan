# Upstream Sync Log

Tracks changes merged from `upstream/main` (pytorch/torchtitan) into the `ezpz`
branch, and any modifications required to keep `experiments/ezpz/{agpt,moe,qwen3}`
compatible.

## How to use this document

After each `git merge upstream/main`, check if the incoming commits touch:

1. **`models/llama3/`** — replay changes onto `experiments/ezpz/agpt/`
2. **`models/deepseek_v3/`** — replay changes onto `experiments/ezpz/moe/`
3. **`models/qwen3/`** — replay changes onto `experiments/ezpz/qwen3/`
4. **`models/common/`** — check if ezpz models depend on changed APIs
5. **`distributed/`** — check if ezpz trainer or parallelize files use removed/renamed APIs
6. **`trainer.py`** — check if ezpz trainer mirrors the same code path

Add an entry below with the date, upstream commits, what changed, and what
was required in ezpz.

After replaying, verify convergence didn't break by running both smoke
tests and checking against the saved baselines — see
[`baselines/README.md`](baselines/README.md).

---

## 2026-05-22 (38th sync — MoE [6/n] dispatcher split + ChunkedCELoss/TP grad fix)

Upstream merged in 4 commits (`cfe97c605..c2a3771a4`).

**Upstream commits (4):**

- **[`da38566d3` — \[MoE\]\[6/n\] Extract local_reorder, split DeepEP/HybridEP
  dispatchers (#3389)](https://github.com/pytorch/torchtitan/pull/3389).**
  Direct continuation of PR #3386 (37th sync).
  - **Extract `local_reorder` helper** on `LocalTokenDispatcher` —
    deduplicates the histc/argsort/score-weighting block shared between
    `LocalTokenDispatcher.dispatch()` and
    `AllToAllTokenDispatcher.dispatch()`.
  - **Split `DeepEPTokenDispatcher` into `DeepEPTokenDispatcher` +
    `HybridEPTokenDispatcher`** — eliminates `comm_backend` string
    branching; config fields (`non_blocking_capacity_factor`,
    `pad_multiple`) are now class-specific. Callsites use `isinstance`
    instead of `getattr(..., "comm_backend", ...)`. The old `DeepEPMoE`
    wrapper class is gone entirely; the dispatcher classes now own that
    logic.
  - **Bundled DeepEP CUDA stream race fix** (#3413) — `torch.cuda.synchronize()`
    before `get_dispatch_layout`; XPU-irrelevant.

- **[`c2a3771a4` — \[loss\] Fix ChunkedCELoss + TP gradient placement
  mismatch (#3412)](https://github.com/pytorch/torchtitan/pull/3412).**
  Purely internal to `torchtitan/components/loss.py`. `GradAccumulator`
  used to wrap its buffer with the reference activation's placement
  (Replicate), but the buffer held chunk gradients which can be
  `Partial(sum)` under TP/ColwiseParallel lm_head. The Replicate label
  made downstream autograd treat Partial values as already-summed,
  poisoning decoder backward and lifting TP-loss above single-GPU
  baseline. Fix: capture `_placements` from the first added chunk
  instead of from the reference activation.
  - **No ezpz replay needed.** Fix is fully internal.
  - **Worth noting**: ezpz's separate TP-loss-reporting workaround in
    `experiments/ezpz/trainer.py` + `validator.py` addresses a
    different bug (the reported scalar, not gradients). The two are
    independent. This upstream fix means any historical TP>1 ezpz
    *training* (not just reporting) was likely also affected — but no
    current production runs use TP>1, so no live dashboards/checkpoints
    are wrong.

- **[`b5852826b` — Fix imports for latest DeepEP (#3414)](https://github.com/pytorch/torchtitan/pull/3414).**
  Two-line change in `torchtitan/distributed/deepep/deepep.py`. XPU
  doesn't use DeepEP. **No-op for ezpz.**

- **[`3f721b4b5` — \[graph_trainer\] Fix fsdp_passes compat with
  BitsetAncestors from pytorch passes (#3416)](https://github.com/pytorch/torchtitan/pull/3416).**
  graph_trainer is a Meta-internal experiment that ezpz doesn't use.
  **No-op for ezpz.**

### Replays in ezpz

[**`d87729ad8` — fix(ezpz/moe): replay PR #3389 — isinstance dispatch on
token_dispatcher Config**](https://github.com/saforem2/torchtitan/commit/d87729ad8).
Mirror upstream's pattern in `experiments/ezpz/moe/model.py`:
- Import `DeepEPTokenDispatcher` and `HybridEPTokenDispatcher` from
  `torchtitan.models.common.token_dispatcher`.
- Replace `getattr(..., "comm_backend", "standard") in ("deepep", "hybridep")`
  with `isinstance(token_dispatcher_cfg, (DeepEPTokenDispatcher.Config,
  HybridEPTokenDispatcher.Config))`.
- Drop the dead `MoE → DeepEPMoE.Config` swap; `DeepEPMoE` no longer
  exists upstream.

ezpz doesn't actually exercise the deepep/hybridep paths on XPU (no
DeepEP kernels), but we keep the EP=1 guard so any CUDA-side ezpz
user who flips the dispatcher config to deepep gets the same error
semantics as upstream `deepseek_v3`.

### Smoke test results

Reports:
[`docs/experiments/moe/sunspot/20260522-smoke-n2-pr3389-replay.md`](experiments/moe/sunspot/20260522-smoke-n2-pr3389-replay.md).

- **`agpt_2b` (2N, LBS=1, GBS=24)** — clean, 50 steps in 140 s, peak
  24.34 GiB, byte-comparable to the prior post-resync baseline.
  W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/c6vff0te.
- **`moe_2b_ep` at registry default (LBS=16, GBS=384, EP=2)** — **OOM
  regression**: 62.53 GiB allocation request inside first forward,
  vs yesterday's 15.03 GiB peak with identical config. New finding,
  almost certainly upstream PR #3389's responsibility (buffer sizing
  in the new `local_reorder` helper or token-dispatcher restructure).
  W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2x0435bc.
- **`moe_2b_ep` at LBS=2 (GBS=48, EP=2)** — clean, 50 steps in 427 s,
  peak 27.08 GiB, loss 12.93 → 6.15. Workaround validated.
  W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/re576w5b.

### Side issue surfaced

Stale `outputs/checkpoint/step-100` from a pre-PR-3159 (35th sync)
revision is no longer loadable post-merge (`Missing key in
checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight.` —
Llama3 `qkv_linear` weight layout changed in PR #3159). Old checkpoint
backed up to `outputs/checkpoint-20260522-120005`; safe to delete.

### Action items

1. **File upstream issue** on `pytorch/torchtitan` referencing the
   `moe_2b_ep` LBS=16 → 62 GiB regression (yesterday 15 GiB clean,
   today 62 GiB OOM, LBS=2 still clean).
2. **Pin `moe_2b_ep` to LBS=2 in the registry** as a workaround
   until the upstream fix lands (mirroring the
   [`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
   pattern for `moe_debugmodel_ep` after the 37th sync).

---

## 2026-05-20 (37th sync — MoE clean DTensor boundaries + graph_trainer regional_inductor)

Upstream merged in `89987072b` (2 commits, `cfe97c605..963c20cba`).

**Upstream commits (2):**

- **`963c20cba` — [MoE][5/n] Refactor MoE to clean DTensor boundaries for
  shared/routed experts (#3386).** Substantial restructuring of how MoE
  TP/EP sharding is wired.
  - **Deleted:** `torchtitan/distributed/expert_parallel.py` (`ExpertParallel`,
    `TensorParallel` ParallelStyles), `ColwiseParallelWithGradPlacement`
    from `torchtitan/distributed/tensor_parallel.py`.
  - **New:** `torchtitan/models/common/moe_sharding.py` providing
    `set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
    expert_param_layout)` which populates `sharding_config` declarations
    on the MoE wrapper, router gate, shared experts (when present), and
    routed `GroupedExperts` — replacing the old
    `apply_moe_ep_tp(model, tp_mesh, ep_mesh)` parallelize-time pass.
  - **`GroupedExperts.parallelize`** is a new override that calls
    `super().parallelize(parallel_dims)` then
    `self.token_dispatcher.wire_meshes(ep_mesh=..., tp_mesh=...)` —
    keeping dispatch/combine mesh-aware at runtime under CooR precompile
    without an explicit pass.
  - **`MoE.forward` simplified.** Drops the explicit `isinstance(x,
    DTensor): x.to_local(grad_placements=Partial)` block at the
    top — replaced by the config-driven enter/exit redistribution
    declared on the MoE wrapper's `sharding_config`. Also splits the
    shared-experts addition out of `combine()`: shared experts run after
    `experts()` (or in parallel with DeepEP's async combine, then
    `sync_combine()` waits). Same math; different float reduction order.
  - **`GroupedExperts.forward`** drops the `shared_experts` kwarg
    accordingly.
  - **`parallelize_deepseekv3` simplified.** Old flow:
    ```
    if tp_enabled: model.parallelize(parallel_dims)
    if tp_enabled or ep_enabled: apply_moe_ep_tp(...)
    ```
    New flow:
    ```
    if tp_enabled or ep_enabled: model.parallelize(parallel_dims)
    ```
    No separate MoE pass — the config-based sharding handles both dense
    and MoE submodules in one call.
  - **`set_deepseek_v3_sharding_config`** gains `enable_ep: bool` and
    populates MoE submodule sharding configs unconditionally
    (`resolve_mesh` filters out disabled axes at runtime).
- **`83e490429` — [graph_trainer] Rework `full_inductor_compilation_pass`
  via `regional_inductor` + CPU attr migration (#3346).** Refactor of the
  graph_trainer inductor compilation pass. Touches
  `experiments/graph_trainer/` exclusively. No ezpz dependency.

**Replayed onto ezpz (PR #3386):**

| File | Change |
|------|--------|
| `experiments/ezpz/moe/parallelize.py` | Drop `apply_moe_ep_tp` function entirely + its imports (`ExpertParallel`, `TensorParallel`, `ColwiseParallelWithGradPlacement`). Replace the old "TP via configs / EP+TP via `apply_moe_ep_tp`" two-pass flow with the new single-pass `if parallel_dims.tp_enabled or parallel_dims.ep_enabled: model.parallelize(parallel_dims)`. Keep our local `apply_fsdp` (Aurora `ShardPlacementResult` workaround) and `disable_fsdp_gradient_division` (CCL backend) unchanged. |
| `experiments/ezpz/moe/sharding.py` | Add `enable_ep` kwarg to `set_moe_sharding_config` (avoids name collision with upstream's identically-named helper by importing the upstream one as `_set_moe_block_sharding_config`). Call upstream's helper on every MoE-enabled layer with `expert_param_layout = {"w1": Shard(1), "w2": Shard(2), "w3": Shard(1)}` (matches `EzpzGroupedExperts.{w1,w2,w3}` and upstream's `_GROUPED_EXPERTS_PARAM_LAYOUT` for deepseek_v3). |
| `experiments/ezpz/moe/model.py` | Pass `enable_ep=parallelism.expert_parallel_degree > 1` into our `set_moe_sharding_config` call from `update_from_config`. Update the surrounding comment. |
| `experiments/ezpz/moe/config_registry.py` | Stale-docstring scrub: drop the `apply_moe_ep_tp` mention from the 500M smoke-config docstring. |

**Replayed onto ezpz (PR #3346):** none. `experiments/graph_trainer/` only.

**Verification:** Smoke validated on Sunspot 2N, job 12467131:
- `moe_debugmodel` LBS=2 / 50 steps: loss 12.92 → 7.00 (Δ -0.01 vs 35th-sync
  baseline 7.01). Memory exactly matches baseline (16.99 GiB / 26.55%).
  TPS ~12,700.
- `moe_2b` LBS=1 / 50 steps: loss 12.95 → 6.11 (Δ -0.05 vs 35th-sync
  baseline 6.16). Memory 14.97 GiB (+0.5 GiB vs baseline 14.47 GiB,
  expected from new graph shape). TPS ~2,900.
- Both within ±0.05 nats of baseline at step 50 — fully consistent with
  PR #3386's documented behavior (graph reordering moves `shared_experts`
  add point, changing FP reduction order). `for_loop` expert backend
  fires the same warning count as baseline (5 + 17). No NaN/OOM. No
  recompilation events.

Full smoke report:
[`docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md).

---

## 2026-05-20 (36th sync — RL CI fixes only, no replay)

Upstream merged in `8b14712a8` (2 commits, `52a292d29..cfe97c605`).
Both touch `experiments/rl/` exclusively; ezpz does not subclass the
upstream RL actors.

**Upstream commits (2):**

- `cfe97c605` — [rl] use wandb=False in CI (#3408). README + integration
  test update to use `MetricsProcessor.Config(enable_wandb=True)` after
  the API change in #3391. Touches `experiments/rl/README.md` (+2) and
  `experiments/rl/tests/integration_tests.py` (+3). No ezpz dependency.
- `2057b5621` — [rl] fix CI timeout: vllm engine V2 teardown (#3365).
  Touches `experiments/rl/actors/generator.py` only (+5 −8). No ezpz
  dependency.

**Replayed onto ezpz:** none. Zero ezpz/{agpt,moe,qwen3} files touched
by these commits.

**Verification:** No code change required; no smoke needed. The build
state is identical to the 35th sync verification (which validated
agpt + moe end-to-end on Sunspot 2N on 2026-05-20).

---

## 2026-05-19 (35th sync — Full DTensor for Llama3 + graph_trainer churn + RL observability)

Upstream merged in `a14987132` (22 commits, `ee4e91a13..52a292d29`).

**Upstream commits touching ezpz-relevant core paths (2):**

- **`d64eabcce` — [Full DTensor] Config-based Full DTensor for Llama3
  (#3159).** Large refactor of the config-based sharding API.
  - **Breaking signature change:** `Module.parallelize(mesh)` →
    `Module.parallelize(parallel_dims)`. Each Module now self-resolves
    its SPMD submesh via `parallel_dims.get_module_mesh(axes)` from the
    axes referenced by its `NamedPlacement`s, instead of receiving a
    bare `tp_mesh` from the caller. Required for the new
    `--training.full_dtensor` mode where DP/CP/TP all participate in the
    Module's mesh.
  - `ShardingConfig` got new optional fields (`out_src_shardings`,
    `local_input_grad_placements`, `local_output_grad_placements`) —
    all default `None`, so existing callsites still construct the same
    shape.
  - `set_gqa_inner_attention_local_map` (which ezpz/agpt calls) is
    unaffected at the call site: arg-name changes (`xq/xk/xv` → `q/k/v`,
    `in_placements/out_placements` → `in_dst_shardings/out_src_shardings`)
    are internal to the helper.
  - `trainer.py` gained a `full_dtensor`-gated `parallelize_inputs` call
    and a `pred.to_local()` fallback for the `disable_loss_parallel`
    path. ezpz's `FaultTolerantTrainer` overrides `train_step` /
    checkpoint plumbing but does not override `_get_batch` or
    `forward_backward`, so it picks up both changes for free.
  - `apply_fsdp` gained an optional `dp_mesh_dims: DataParallelMeshDims`
    kwarg, only used under `full_dtensor`. ezpz/agpt's local `apply_fsdp`
    does not take the kwarg; that's fine since `full_dtensor=False` is
    the default and we don't enable it.
- **`a2a0d99e3` — RL: observability spans across trainer/generator/
  controller (#3234).** Adds `@sl.log_trace_span` decorators around RL
  actor methods. ezpz/rl uses its own `GRPOTask` registry and doesn't
  subclass the upstream RL actors, so this is a no-op for ezpz.

**Replayed onto ezpz:**

`experiments/ezpz/agpt/parallelize.py`:

- `model.parallelize(tp_mesh)` → `model.parallelize(parallel_dims)`.
  Async-TP plumbing still takes `parallel_dims.get_mesh("tp")` at the
  callsite. Module docstring updated.

`experiments/ezpz/moe/parallelize.py`:

- Same `model.parallelize(tp_mesh)` → `model.parallelize(parallel_dims)`
  swap on the dense-path TP application. `apply_moe_ep_tp` still takes
  the per-axis meshes directly (it doesn't go through
  `Module.parallelize`). Module docstring updated.

`experiments/ezpz/moe/model.py`: docstring reference to
`Module.parallelize(tp_mesh)` updated to `(parallel_dims)`.

**Other upstream commits in this batch (no ezpz impact):**

- `52a292d29` — [graph_trainer] Gate overlap_fsdp_ag_rs_pass behind a config flag (#3241).
- `8aa96acb2` — [graph_trainer] Fix eager SAC policy mm counter to reset at layer boundaries (#3397).
- `cf3c4312e` — [graph_trainer] Re-enable FlexAttention tests after upstream fix (#3394).
- `ebfceebe1` — [rl] Add TITO generator and gen metrics (#3391).
- `4238c3575` — Re-enable compile for gpt-oss integration tests (#3373).
- `6a1b334e8` — Fix optimizer state and module state coupling (#3356).
- `013890bf9` — [graph_trainer] Defer cudagraph compatibility check (#3355).
- `c9126af8a` — [graph_trainer] Add bucketing ops to precompile serialization filter (#3354).
- `9f9ae4d81` — [graph_trainer] Fix H100 CI failure from DeepEP compilation break (#3390).
- `a866d04a0` — Fix memory snapshot pickle protocol for memory visualizer compat (#3375).
- `cb7bf09ab` — ci: declare workflow-level `contents: read` on 2 workflows (#3367).
- `a670699ca` — [graph_trainer] Skip dense numerics tests due to upstream DTensor regression (#3372).
- `b64292ba1` — [graph_trainer] Remove fsdp_reshard_after_fwd_pass (#3370).
- `f0795f0d5` — [RL] - Enable experiment metrics (#3237).
- `8cdfdc236` — [rl] Better initial weight loading (#3318).
- `3781d1d2d` — Back out [graph_trainer] SAC remat fresh FakeTensor storage (#3358).
- `4614e3022` — [graph_trainer] Support multiple FSDP PGs in overlap_fsdp_ag_rs_pass (#3351).
- `d4a3c6349` — [graph_trainer] Generalize minimal_fx_tracer to module + optimizer roots (#3164).
- `cee49826b` — [graph_trainer] Fix SAC remat to produce fresh FakeTensor storage (#3343).
- `1690e0aea` — [cpu-offloading] Encode last consumer as dep arg in ao.wait (#3333).

**Verification:** `agpt/parallelize.py` and `moe/parallelize.py` import
cleanly via `.venv/bin/python -c "import ...parallelize"` post-replay;
both files round-trip the new `Module.parallelize(parallel_dims)`
signature.

**Compute-node smoke (2026-05-20, Sunspot 2N, job 12467124):** all four
configs ran 50 steps cleanly; no NaN/OOM, monotonic loss descent.

| Config              | LBS | Final loss | TPS/GPU | MFU    | Peak mem            | Report |
|---------------------|----:|-----------:|--------:|-------:|---------------------|--------|
| `agpt_debugmodel`   | 2   | 6.77       | ~37,500 | ~2.9%  | 2.60 GiB (4.06%)    | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `agpt_2b` (LBS=1)   | 1   | 6.01       | ~6,100  | ~22.8% | 24.34 GiB (38.04%)  | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `agpt_2b` (LBS=2)   | 2   | 6.12       | ~7,200  | ~27.0% | 44.73 GiB (69.91%)  | [link](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md) |
| `moe_debugmodel`    | 2   | 7.01       | ~13,000 | ~9.0%  | 16.99 GiB (26.55%)  | [link](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)  |
| `moe_2b` (LBS=1)    | 1   | 6.16       | ~2,900  | ~8.4%  | 14.47 GiB (22.62%)  | [link](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)  |

`agpt_2b` at LBS=2 matches the historical [2026-04-25 n=2 baseline](experiments/agpt/sunspot/20260425-scaling-2b-venv-torch213.md)
(7,142 TPS / 27.6% MFU) within noise -- the resync did not perturb
steady-state throughput. The `for_loop` MoE expert backend (PR #13)
still fires correctly on XPU under the new parallelize signature.

Note: default `moe_2b()` is LBS=16, which OOMs on a 64 GiB Max 1550 tile
with a single 62.53 GiB allocation; LBS=1 is the safe per-GPU batch for
the for_loop path. This is a pre-existing XPU constraint, not caused by
the resync.

---

## 2026-05-13 (34th sync — RL vLLM v2 + repeat_interleave revert + graph_trainer AOT removal)

**Upstream commits (3 in batch):**

- `6f2fa2f9a` — [rl] switch to vllm v2 engine (#3330). Touches
  `experiments/rl/actors/generator.py`. ezpz/rl is unaffected — we
  don't subclass the upstream RL actors directly.
- `7b418ab30` — Revert "Avoid repeat_interleave output-size sync (#3274)"
  (#3335). Restores prior behavior in
  `models/common/token_dispatcher.py` after the optimization broke
  deepseek_v3 at TP=1 + PP=4 + EP=32 + AC=full. Pure revert; no new
  logic. ezpz/moe defers to upstream `LocalTokenDispatcher` so this
  silently restores correctness on any high-EP config.
- `1a22c2da1` — [graph_trainer] Remove deprecated AOT compile mode
  (#3327). Touches `experiments/graph_trainer/` only; no ezpz dependency.

**Replayed onto ezpz:** none. Zero ezpz/{agpt,moe,qwen3} files
touched by these commits.

**Verification:** `agpt_{debugmodel,2b,2b_real,20b,80b,80b_real}` and
`moe_{debugmodel,500m,10B_2B_sdpa}` all build cleanly post-merge.

---

## 2026-05-12 (33rd sync — `_grouped_mm` only path + graph_trainer churn)

**Upstream commits (12 in batch):**

- `b301dfa0` — **[MoE] Remove expert for-loop fallback (#3308).** Deletes
  `_run_experts_for_loop` and the `use_grouped_mm` config field from
  `models/common/moe.py`. `GroupedExperts._experts_forward` now always
  calls `torch._grouped_mm`. Upstream's argument is that
  `torch._grouped_mm` already provides a CUDA fallback path on pre-SM90
  hardware. **This breaks `experiments/ezpz/moe/model.py` which used
  `use_grouped_mm = False` as the XPU fallback** (XPU has no
  `_grouped_mm` kernel at all).
- `d57df092` — Make ChunkedCELoss support `torch.autograd.grad` (#3249).
- `5ca23a5d` — [GraphTrainer] Add Context Parallel support (#3305).
- `1a0fe3e3` — [graph_trainer] Refactor passes.py into focused modules (#3319).
- `e9dbff63` — [graph_trainer] Refactor selective activation remat to in-place (#3270).
- `2ceff82b` — [graph_trainer] Add log_timer utility for tracing step timing (#3311).
- `0fadde3b` — [graph_trainer] Fix AutoParallel input_fn to include positions tensor (#3315).
- `34801c00` — [graph_trainer] Improve SAC tagging and CPU offload pass metadata (#3321).
- `0b5e8998` — Fix precompile tests (#3316).
- `ca4c7f22` — [rl] Register customized config parser to vllm + less vllm config dependency (#3242).
- `7f602b98` — Add AGENTS.md symlinks for Codex usage (#3326).
- `7f070c93` — Enhance Lychee Link Checker (Resiliency & Performance) (#3203).

**Replayed onto ezpz:**

`experiments/ezpz/moe/`:

- New `experts.py` defining `EzpzGroupedExperts(GroupedExperts)` with a
  `compute_backend: Literal["for_loop", "grouped_mm"]` config field.
  Default `"grouped_mm"` defers to upstream; `"for_loop"` re-vendors
  the `_run_experts_for_loop` body that #3308 deleted, restoring the
  XPU / pre-SM90 path.
- New `make_ezpz_experts_config(...)` wrapper in `__init__.py` that
  calls upstream's `make_experts_config(...)` then re-wraps the result
  as `EzpzGroupedExperts.Config`. `_build_moe_layers` now threads a
  `compute_backend` kwarg (default `"grouped_mm"`) through to it.
- `model.py` `update_from_config` previously mutated
  `experts.use_grouped_mm = False` on pre-SM90 devices; that field no
  longer exists. Replaced with `experts_cfg.compute_backend = "for_loop"`
  guarded by `getattr(..., "compute_backend", "grouped_mm")` so the
  block is robust to future config-shape changes.

`experiments/ezpz/agpt/`: no replay needed; #3308's deletion was
MoE-only, and none of the other upstream commits in this batch touch
`models/llama3/` in a way ezpz/agpt depends on.

**Notes for downstream PRs:**

- Open PR #9 (Sam Wheeler — HSDP fix), #10 (Sam Wheeler —
  `batched_mm_padded` backend), and #11 (Nathan Nichols — MoE
  optimizations) all assume the pre-#3308 `GroupedExperts` shape
  (`use_grouped_mm` config field, `_run_experts_for_loop` importable
  from `models/common/moe.py`). They will need to rebase onto the
  resync'd `ezpz` and adapt to the `EzpzGroupedExperts` subclass.
  PR #10's `compute_backend` selector becomes a third option in
  `ExpertComputeBackend`; PR #11's expert-side optimizations layer
  onto the for-loop method here.

---

## 2026-05-05 (32nd sync — observability + MoE token-pad + CP fix + RL/graph_trainer churn)

**Upstream commits (11 in batch):**

- `b2cd149f` — Observability: structured logging + training instrumentation (#3176).
  Adds `torchtitan/observability/` module + `init_structured_logger()` in
  `Configurable.build()` + `@sl.log_trace_span(...)` decorators on hot
  paths in `trainer.py` and `validate.py`. Optional jsonl/database
  output for per-step timing spans.
- `d3414079` — [MoE] Pad token count to a multiple of `sp_size` in
  `AllToAllTokenDispatcher` (#3193). Internal correctness fix.
- `179d9e10` — CP AllGather on the wrong dimension (#3206). FlexAttention
  + Context Parallelism gather was on the wrong axis. Bug fix.
- `d3c96e80` — [mxfp8] Fix `MXFP8GroupedExpertsConverter` to actually swap
  `GroupedExperts` params (#3199). Quantization plumbing.
- `af8d2430` — [GraphTrainer] Skip identity-slice rewrite when start/end/step
  are dynamic Nodes (#3195).
- `706ed8d8` — [GraphTrainer] Annotate generated FX code with user source
  lines (#3194).
- `080c1d4c` — [GraphTrainer] Annotate loss region with `module_fqn` (#3207).
- `0b6a29e6` — [rl] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  for monarch RDMA (#3221).
- `2ae13405` — [rl] add `import torch` to provisioner bootstrap (#3220).
- `0b148a2e` — [CI] Run integration tests in parallel (#3144).
- `f522db0e` — Fix commands in DSV3 readme (#3116).

**Impact on ezpz:**

- **Observability (#3176)** does NOT affect us at runtime — our
  `FaultTolerantTrainer.__init__` inlines the upstream `Trainer.__init__`
  body (FT-required) rather than calling `super().__init__()`, so the
  new `init_structured_logger()` and `@sl.log_trace_span` decorators
  on upstream `Trainer` methods don't propagate. Likewise, our
  `EzpzValidator.validate()` overrides `Validator.validate()` entirely,
  so the new `@sl.log_trace_span("eval")` decorator on the parent
  doesn't apply to us. Imports verified to still work after merge.
  If we want trace spans on the ezpz path we'd need to add the
  decorators ourselves; not blocking.
- **MoE token-pad (#3193)** — affects `AllToAllTokenDispatcher` only;
  our `moe_10b_2b_sdpa_ep` uses `comm_backend="standard"` →
  `All2AllTokenDispatcher` (different code path). Safe to take.
- **CP AllGather fix (#3206)** — ezpz never enables CP
  (`cp_degree` always 1 in our configs). Safe to take.
- All other commits scoped to `experiments/{rl,graph_trainer}/`,
  quantization, CI, or docs. ezpz doesn't depend.

**Replay into ezpz `agpt/` or `moe/`:** None needed. No
`models/llama3/`, `models/deepseek_v3/`, or `models/common/` changes
in this batch beyond the internal token_dispatcher fix.

Merge commit: `0690e67e`.

---

## 2026-05-04 (31st sync — precompile revert + PP refactor + CI baseline)

**Upstream commits:**

- `60650551` — Revert precompile PRs (#3107, #3178) to fix CI (#3212).
  Backs out the bucketing-pass + FlexAttention precompile bitwise
  deterministic-tests changes that were merged in the 2026-05-03 sync;
  the upstream PyTorch fix (pytorch/pytorch#181529) those changes
  depended on still hadn't landed on `viable/strict`.
- `db5da4b6` — Refactor pipeline parallel helpers for graph PP reuse
  (#2724). Extracts pipeline metadata + module splitting + PP
  rank-to-stage mapping out of `pipeline_llm` so graph PP can reuse
  it. Renames `build_pipeline_schedule`,
  `generate_llm_fqn_per_model_part`, and `pipeline_module_split` to
  private (underscore prefix) — they are no longer public API.
- `95ae6a00` — `ci: regenerate qwen3_moe_rocm_mi350x.txt baseline with
  actual MI350 losses (#3196)`. Loss-baseline file regen for ROCm
  MI350 CI; no production impact.

**Impact on ezpz:** None.

- The PP refactor (#2724) only renamed helpers used internally by
  `pipeline_llm`; ezpz doesn't import any of `build_pipeline_schedule`,
  `generate_llm_fqn_per_model_part`, or `pipeline_module_split`
  directly (verified via grep across `experiments/ezpz/`).
- The revert (#3212) only touches
  `experiments/graph_trainer/precompile_main.py`,
  `experiments/graph_trainer/tests/test_bitwise_deterministic.py`,
  and `experiments/transformers_modeling_backend/pipeline.py` — none
  of which ezpz depends on.
- The baseline regen is test data only.

Merge commit: `fc10014d`.

---

## 2026-05-03 (30th sync — graph_trainer only)

**Upstream commits:**

- `627126fe` — Enable bucketing pass in precompile path (#3107)
- `d27d0c36` — [GraphTrainer] Fix FlexAttention precompile bitwise
  deterministic tests (#3178)
- `14f6a75e` — [GraphTrainer][AutoDev] Fuse RMSNorm kernels via regional
  Inductor compilation (#3132). Adds `performance_passes.py` and a new
  test file.

**Impact on ezpz:** None. All 3 commits scoped to
`experiments/graph_trainer/`. ezpz doesn't depend.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed.

---

## 2026-05-01 (29th sync — RL vLLM compile-time + graph_trainer skill)

**Upstream commits:**

- `2e5f1371` — Improve compilation time (~50s → ~15s for vLLM) (#3145).
  Scoped to `experiments/rl/{generate,grpo,models}` + RL tests.
- `37fe579a` — [GraphTrainer] Add weekly report generator skill (#3200).
  One new file: `experiments/graph_trainer/.claude/weekly_report.md`.

**Impact on ezpz:** None.
- ezpz doesn't depend on `experiments/rl/` or `experiments/graph_trainer/`.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed.

---

## 2026-05-01 (28th sync — graph_trainer qwen3 + CI lint + FT llama3 attn_backend)

**Upstream commits:**

- `2b935fa3` — [GraphTrainer] Add Qwen3 MoE bitwise deterministic tests
  and fix weight-tying gradient bug (#3174). graph_trainer + qwen3 only.
- `70340f4e` — [CI] Use replace-imports-with-any (#3180). Removes
  `# pyrefly: ignore[missing-module-attribute]` comments now redundant
  with the new pyrefly config; lint-only.
- `9732db4a` — [ft] Forward attn_backend to llama3 config functions
  (#3182). 2-line change to `experiments/ft/llama3/__init__.py` —
  adds `attn_backend="sdpa"` parameter to that registry. We don't use
  `ft.llama3` (we have our own `ezpz.agpt.model_registry`).

**Impact on ezpz:** None. All three commits scoped to graph_trainer,
linter cleanups, or ft.llama3 (which we don't use).

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-30 (27th sync — HybridEP comm_backend cleanup + autoparallel/deepseek_v3 deletion)

**Upstream commits (truly new — most others were patch-equivalent
duplicates of the ETP deprecation already replayed at `97e44a5a`):**

- `f7940d5b` — Remove stale `experiments/autoparallel/deepseek_v3/`
  (#2271). 5 files / 534 lines deleted.
- `0138bde8` — [HybridEP] Read comm_backend from model config instead
  of ParallelismConfig (#3177). Adds optional `non_blocking_capacity_factor`
  kwarg to deepseek_v3 model factory functions.
- `115f4c9c` — [HybridEP] Enable HybridEP with graph_trainer (#3007).
  graph_trainer + 1-line deepseek_v3/__init__.py change.

**Impact on ezpz:** None.
- ezpz/moe doesn't use `non_blocking_capacity_factor` (no HybridEP code path)
- We don't depend on `experiments/autoparallel/deepseek_v3/`

**Merge conflicts:** Two modify/delete conflicts on
`experiments/autoparallel/deepseek_v3/{config_registry,parallelize_deepseekv3}.py`
— ezpz had local mods to those files from a previous ETP-removal patch
sequence. Accepted upstream's deletion (we don't use the dir).

**Replay status:** Clean otherwise. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-30 (26th sync — ETP deprecation + varlen window + CI)

**Upstream commits:**

- `c4b409af` — [MoE][4/n] deprecate expert tensor parallel (ETP) (#3167).
  Removes `ExpertTensorParallel` class, the `etp` mesh axis, the
  `expert_tensor_parallel_degree` and `expert_parallel_comm_backend`
  config fields, and all ETP references across models and experiments.
  `comm_backend` now lives on `moe.experts.token_dispatcher`, not on
  `parallelism`.
- `a2b5ee66` — varlen window size configurable (#3173). Adds
  `attention.window_size` plumbing for varlen attention; additive only.
- `47d3045f` — [CI] Move lychee link checker to nightly workflow (#3158).
- `efebe6de` — [GraphTrainer] Remove SAC peak-memory test from H100 CI
  (#3170).
- `4e48ebec` — Add MoE loss comparison guard to CI workflow (#3081).
- `6495ce57` — [GraphTrainer] Clean up precompile bitwise deterministic
  tests (#3169).

**Impact on ezpz:**
- `ezpz/moe/parallelize.py` imported `ExpertTensorParallel` (now deleted)
  and called `apply_moe_ep_tp` with `etp_mesh` / `ep_etp_mesh` kwargs that
  the upstream signature no longer accepts.
- `ezpz/moe/model.py` referenced `parallelism.expert_parallel_comm_backend`
  to decide between `MoE` and `DeepEPMoE`. That config field is gone;
  `comm_backend` now lives on the per-expert token_dispatcher config.
- `ezpz/agpt/`, `ezpz/qwen3/`, and the trainer don't touch ETP — no
  changes needed.
- `experiments/ezpz/moe_runs/*.json` snapshots still contain
  `"expert_tensor_parallel_degree": 1`. These are historical run records;
  not stripping. Re-running them through the current config parser would
  fail and require dropping that key.

**Replay (mirrors `c4b409af` deepseek_v3 changes):**
- `ezpz/moe/parallelize.py`:
  - Drop `ExpertTensorParallel` from the `expert_parallel` import.
  - `apply_moe_ep_tp(...)` signature: drop `etp_mesh` / `ep_etp_mesh`.
  - Drop the `etp_mesh`/`ep_etp_mesh` kwargs from the call inside
    `parallelize_moe`.
  - Collapse the `experts_mesh` / `experts_plan` selection to the two
    surviving cases (EP disabled → TP-shard experts; EP enabled →
    `ExpertParallel()`).
- `ezpz/moe/model.py`:
  - Read `comm_backend` from
    `layer_cfg.moe.experts.token_dispatcher` (matches upstream
    `deepseek_v3.model`).
  - Raise `ValueError` if `comm_backend in ("deepep","hybridep")` and
    `expert_parallel_degree == 1` (was previously implicit — now
    explicit, matching upstream).
  - Keep the `MoE → DeepEPMoE.Config` swap inside the same
    `comm_backend in ("deepep","hybridep")` branch.

**Verified:** Both modified files parse with `ast.parse`. No baseline
re-check needed — ezpz/moe doesn't run on Aurora MoE production right
now (MoE SIGABRT regression is still open from the 00b7f569 sync), so
there's nothing live to break. Will smoke when MoE work resumes.

---

## 2026-04-29 (25th sync — graph_trainer + VLM deletion + minor)

**Upstream commits:**

- `01a8b068` — [GraphTrainer] log_activation_memory_policy (#3062)
- `533795ee` — [graph_trainer] simple_fsdp unconditional for NGPU=1 (#3148)
- `a890192e` — [graph_trainer] Fix test_trace_module backends (#3155)
- `ef8e2820` — [graph_trainer] Async TP graph pass (#3129)
- `0ad6772a` — [VLM] deprecate vlm experiment (#3151) — deletes 18
  files / 2,248 lines from `experiments/vlm/`
- `a3a01604` — fix(hf_datasets): ChatDataset shuffle before split (#3131)
- `719085ae` — Use CrossEntropyLoss for torchcomms 3D compile tests (#3157)

**Impact on ezpz:** None.
- 4 commits scoped to `experiments/graph_trainer/`
- 1 commit deletes `experiments/vlm/` (ezpz doesn't depend on it)
- 1 commit touches `hf_datasets/text_datasets.py` for `ChatDataset`
  only — ezpz uses BlendCorpus or HuggingFaceFW streaming via
  `datasets.py`, not `ChatDataset`
- 1 commit is a single line in CI test config

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-29 (24th sync — All2All token dispatcher consolidation)

**Upstream commits:**

- `20628f4e` — [MoE][3/n] consolidate EP=1 and EP>1 to all use
  `All2AllTokenDispatcher` (#3125). `AllToAllTokenDispatcher` now falls
  back to `LocalTokenDispatcher` behavior when `ep_mesh is None`.
  Default `comm_backend` changed from `None` to `"standard"`; the
  `None` → `LocalTokenDispatcher` path was removed entirely.
  `make_token_dispatcher_config` and `make_experts_config` now require
  a non-None `comm_backend`.
- `35c5d529` — [graph_trainer] Remove `apply_graph_ac` (#3147). No
  impact on ezpz (graph_trainer experiment only).

**Impact on ezpz:** `moe.model_registry()` and `_build_moe_layers()` both
defaulted `moe_comm_backend` to `None`, which is no longer valid. At
build time `make_experts_config(comm_backend=None)` raises
`ValueError: Unknown comm_backend: 'None'`.

**Replay:**
- `moe/__init__.py`: change both `moe_comm_backend: str | None = None`
  defaults to `moe_comm_backend: str = "standard"` (mirrors upstream).
  Drop the `if moe_comm_backend is not None` guard around the
  token-dispatcher rebuild loop in `model_registry` — the dispatcher is
  now always rebuilt with the user's chosen backend (or "standard" by
  default), and EP=1 is handled by the dispatcher's local-fallback path
  rather than by skipping the rebuild.

**Verified:** Both smoke configs build cleanly post-fix. Existing
`moe_debugmodel_ep` / `moe_7b_ep` configs that pass
`moe_comm_backend="standard"` explicitly continue to work (kwarg is
redundant but not wrong).

---

## 2026-04-29 (post-21st-sync regression: legacy `output.weight` checkpoint load)

**Symptom:** Both 20B continuation jobs (`8453664`, `8453665`) crashed at
checkpoint load with:

    RuntimeError: Missing key in checkpoint state_dict: lm_head.weight

The 2B continuation (`8453662`) didn't reach checkpoint load — segfaulted
on a bad node during `set_determinism()` — but would have hit the same
error.

**Root cause:** The 21st upstream sync (`b6c04698`) renamed
`Decoder.Config.output` to `Decoder.Config.lm_head`, replayed in ezpz
`03b9f486`. Production checkpoints saved before the rename
(2B step 33,700+, 20B step 4,100+) still have `output.*` keys on disk;
new code's state_dict has `lm_head.*`; DCP's strict matching crashes.

**Fix:** `experiments/ezpz/checkpoint_compat.py` monkey-patches
`CheckpointManager.dcp_load` to bridge the rename on the fly:
state_dict keys go `lm_head.*` -> `output.*` before `dcp.load`, then
back to `lm_head.*` before `model.load_state_dict`. Applied from
`train.py`, idempotent.

Verified by `utils/verify_checkpoint_compat.py` round-tripping the 20B
step-4100 `output.weight` ([256128, 5120] bf16): non-zero, finite,
sensible values.

**Lesson:** Field renames in `Decoder.Config` (and any upstream `Module`
attribute rename) need an explicit DCP backwards-compat plan. Going
forward, when replaying upstream renames, also save a fresh checkpoint
with the new naming on the next opportunity so the shim can be removed.

**Removal (2026-04-29):** Shim and verifier deleted. The bf16-tainted
production checkpoints were renamed to `*.bf16-norm-bug-20260429` and
will not be resumed from. The shim was unconditionally renaming
`lm_head` -> `output` in the load path, which BROKE auto-resume from
new-style checkpoints (the new restart writes `lm_head.*` keys; the
shim rewrote the load state_dict to ask for `output.*`, missed the
on-disk metadata). Files removed: `experiments/ezpz/checkpoint_compat.py`,
`experiments/ezpz/utils/verify_checkpoint_compat.py`. The
`patch_checkpoint_manager()` call removed from `train.py`.

---

## 2026-04-28 (23rd sync — graph_trainer experiment + ROCm CI only)

**Upstream commits:**

- `a364b4b4` — [GraphTrainer] Add full inductor compilation pass (#3141)
- `9ed1a028` — [graph_trainer] Joint graph bucketing + prefetching
  composes with SAC (#3056)
- `69761ca8` — [ROCm][CI] Re-disable experimental workflows for ROCm (#3140)

**Impact on ezpz:** None. All three commits are scoped to
`experiments/graph_trainer/` and `.github/workflows/`. No changes to
`models/`, `protocols/`, `distributed/`, `trainer.py`, or `components/`.

**Replay status:** Clean fast-forward merge. ezpz imports verified.
No baseline re-check needed (no code path that affects agpt/moe changed).

---

## 2026-04-28 (22nd sync — quantize-on-config, LocalMapInnerAttention removal, MeshAxisName rename)

**Upstream commits:**

- `6348d93d` — quantize on config instead of on model (#3127). Removes
  `protocols/model_converter.py`, drops `model_converters` from
  `JobConfig`, drops `model_converters` argument from all `parallelize_*`
  signatures. New pattern: pass `quantization=[Float8LinearConverter.Config(...)]`
  to `model_registry()`, which applies the converter to the model
  config at registry time. Also removes `FaultTolerantModelSpec` from
  `protocols/model_spec` (moved to `experiments/ft/config/job_config`).
- `b9e33527` — [Module] Remove LocalMapInnerAttention, use static
  LocalMapSpec (#2986). Replaces runtime DTensor detection in
  `LocalMapInnerAttention` with a static `LocalMapConfig` set on the
  inner-attention sharding_config via
  `set_gqa_inner_attention_local_map`. All inner attention types
  (SDPA, FlexAttention, Varlen) now inherit `Module` directly.
- `053dbf9a` — [Module] Rename MeshDimName → MeshAxisName (#3113).
  Transparent for ezpz: our sharding files use the helpers from
  `decoder_sharding`, which were updated upstream.
- + 6 minor (linter fixes, MATH backend revert, qwen3 RL cleanup,
  GraphTrainer CI, claude.md updates).

**Impact on ezpz:** Three breaking changes hit at import time:

- `from torchtitan.protocols.model_converter import ModelConvertersContainer` → gone
- `from torchtitan.models.common.attention import LocalMapInnerAttention` → gone
- `from torchtitan.protocols.model_spec import FaultTolerantModelSpec` → gone

**Replay:**

- `agpt/parallelize.py` — drop `model_converters` parameter and import.
- `moe/parallelize.py` — drop `model_converters` parameter and import.
- `agpt/__init__.py` — `LocalMapInnerAttention` → `Module` for the
  `SoftcappedFlexAttention` base class and `_ezpz_get_attention_config`
  return-type annotation. Re-import `FaultTolerantModelSpec` from its
  new location.
- `moe/model.py` — `LocalMapInnerAttention.Config` →
  `Module.Config` for `Attention.Config.inner_attention`.
- `agpt/sharding.py`, `moe/sharding.py` — add
  `set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)`
  call so inner attention gets the static `LocalMapConfig` it now
  needs (replaces the runtime DTensor wrapper).
- `moe/__init__.py` — extend `model_registry` with `quantization`
  parameter; apply each converter to the config via `q.build().convert(config)`.
- `moe/config_registry.py` — `moe_671b()` re-registers via
  `model_spec=model_registry("671B", quantization=[...])` instead of
  setting `cfg.model_converters`.
- `ezpz/trainer.py` — drop the runtime model_converters build/convert,
  the `register_step_post_hook` post_optimizer_hook, the
  `model_converters=` kwarg from both `parallelize_fn` and `pipelining_fn`
  call sites, the `QuantizationConverter` import. Switch
  `has_quantization` to read from `model_config` via
  `torchtitan.components.quantization.utils.has_quantization`.

**Verified:** agpt + moe import cleanly, `model_registry()` returns
valid spec, `update_from_config` populates sharding_config including
the new `LocalMapConfig` on inner_attention.

---

## 2026-04-27 (21st sync — config-based DTensor sharding)

**Upstream commits:**

- `1786292d` — [Module][Full DTensor] Config-based sharding infrastructure with
  llama3 adoption (#2963). Replaces string-keyed plan dicts with `ShardingSpec`
  attached to `Module.Config`. New `protocols/sharding.py` and `protocols/module.py`.
- `8e820844` — [Module][Full DTensor] Config-based sharding for Qwen3, Llama4,
  DeepSeek V3, GPT-OSS (#2969). All models now use `Module.parallelize()`.
- `786e26f8` — ChunkedCELoss (#2937)
- `1ea5d511` — [AutoParallel] Use autoparallel_backend() for torch.compile (#3114)
- + 5 more (GraphTrainer, Qwen3VL, vLLM cache reset)

**Impact on ezpz:**

- **llama3/parallelize.py REWRITTEN** — old `parallelize_module()` API replaced
  with `Module.parallelize()`. The ezpz `agpt/parallelize.py` still uses the old
  API. **Needs replay** but deferred to avoid breaking running experiments.
- **New files:** `llama3/sharding.py`, `deepseek_v3/sharding.py`,
  `protocols/sharding.py`, `protocols/module.py`, `common/decoder_sharding.py`
- **trainer.py** — 53 lines changed (model initialization flow updated)

**Replay status (2026-04-28):**

- **agpt: DONE.** Three commits land the replay:
  - `03b9f486 fix(ezpz): unbreak imports after upstream loss/lm_head refactors`
    — drops `build_cross_entropy_loss` imports / `build_loss_fn=` kwargs
    (#2937 removed both); renames `output=Linear.Config(...)` to
    `lm_head=Linear.Config(...)` (Decoder.Config field rename); ezpz/trainer.py
    switches to `config.loss.build(compile_config=...)`.
  - `472f4743 refactor(ezpz/agpt): replay config-based DTensor sharding`
    — adds `agpt/sharding.py` and `agpt/model.py`. Rewrites
    `agpt/parallelize.py` as a thin orchestrator that calls
    `model.parallelize(tp_mesh)` instead of the old `parallelize_module()` plan.
    Preserves agpt-specific `disable_fsdp_gradient_division`
    (force_sum_reduction for CCL/XPU), the `capture_scalar_outputs=False`
    reset after compile, and the `[norm, lm_head]` joint FSDP grouping.
  - Smoke test: 2N debug-scaling pending.
- **moe: DONE.** Same shape of replay against deepseek_v3:
  - New `moe/sharding.py` mirrors `models/deepseek_v3/sharding.py` but
    binds against `experiments.ezpz.moe.model.Attention` (our MLA Attention
    is a separate class from upstream's, even though structurally identical).
  - `moe/model.py`: extends `moeModel.Config.update_from_config` to call
    `set_moe_sharding_config` after the existing rope/MoE sync logic.
  - `moe/parallelize.py`: rewritten as a thin orchestrator. The non-MoE TP
    plumbing (`apply_non_moe_tp` with manual ColwiseParallel/RowwiseParallel/
    SequenceParallel/PrepareModuleInput plans for attention + norms +
    dense FFN) is gone — replaced by `model.parallelize(tp_mesh)`.
    `apply_moe_ep_tp` is kept (mirrors upstream — MoE blocks are still
    parallelized at parallelize-time, not via sharding_config).
    `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
    import which doesn't exist in Aurora's PyTorch) and includes the
    Shard(0) fallback for when expert hidden dim isn't FSDP-divisible.
    Per-block `block.compile(backend=...)` workaround kept (XPU can't do
    fullgraph=True for MoE routing).
  - Smoke build verified locally: imports + `update_from_config` +
    `Module.parallelize(1D mesh)` + meta `cfg.build()` all work on the
    moe `debugmodel` flavor.
- **qwen3: REMOVED.** ezpz/qwen3 had drifted significantly from upstream
  (single `layer:` template vs upstream's `layers: list`, custom
  `__init__(config, *, layer_id, dim, n_layers)` block constructor,
  `self.output` references, no `lm_head` slot in qwen3_configs entries).
  README marked it "still under development" and no production jobs ever
  used it. Removed via `git rm -r` rather than carry the replay debt
  for unused code. History retained — restore with
  `git checkout <parent-sha> -- torchtitan/experiments/ezpz/qwen3` if
  ever needed.

**Float8 tensorwise TP:** dropped from `agpt/parallelize.py` during the
replay. The new sharding API doesn't expose an equivalent yet
(`Float8ColwiseParallel` etc. were tied to the old plan API). We weren't
using it in production. Revisit when float8 lands in the new API upstream.

---

## 2026-04-27 (20th sync)

**Upstream commits:**

- `cca3be50` — Fix reproducible training resume across epoch boundaries for
  map and streaming datasets (#3008). Fixes two bugs in `HuggingFaceTextDataset`
  and `ChatDataset` related to checkpoint resume after epoch re-loop.
- `a7205469` — [rl] Env rollout based + controller refactor (#3073). Refactors
  upstream RL experiment (not our ezpz/rl).

**Changes required in ezpz:** None. Clean merge. Verified `_validate_dataset`
still exists and is compatible with our `datasets.py` monkey-patch.

**No replay needed:** Neither commit touches `llama3/`, `deepseek_v3/`, or
`models/common/`.

---

## 2026-04-25 (19th sync)

**Upstream commits:**

- `bfc2914b` — Use `current_accelerator` for device in AutoParallel calls to enable XPU (#3092)
- `eb518a1d` — Add fused QKV support to Qwen3-VL state_dict_adapter (#3102)
- `42b73643` — Fix SAC test compatibility with PyTorch indexed storage (#3098)
- + 11 more (GraphTrainer CPU offload, CI, ROCm)

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-24 (18th sync — torch 2.10 XCCL fixes + DTensor TP revert)

**Context:** Investigating and fixing training hangs on torch 2.10
(`aurora_frameworks-2025.3.1`) with the XCCL backend. Also fixing
80B TP=2 regression on both torch 2.10 and 2.13.

**Root causes identified:**

1. **Blendcorpus barrier hang (torch 2.10):** The XCCL C++ backend
   ignores `opts.device` for `barrier()` operations, defaulting all
   ranks to device 0. This causes hangs during blendcorpus dataset
   building which uses `barrier(group=mpu.get_data_parallel_group())`.

2. **TP=2 forward pass hang (torch 2.10):** Removing the IPEX import
   (`import intel_extension_for_pytorch`) in the 17th sync broke TP
   collectives on torch ≤2.10, because IPEX provides XPU operator
   overrides that the XCCL backend relies on.

3. **80B AC + compile crash (torch 2.13):** The `use_local_output=False`
   DTensor TP change (`fc3880c5`) causes `DeviceMesh` objects to leak
   into the AOT autograd saved state, triggering
   `AssertionError: expected all tensors_saved_with_vc_check to be Tensors`.
   This is a PyTorch bug — AC's version check doesn't handle non-Tensor
   objects from DTensor-parallelized modules.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `blendcorpus/blendcorpus_builder.py` | Gloo barrier workaround: temporarily replace `dist.barrier` with a CPU-side gloo barrier during dataset building when `bound_device_id` is None | `312045b3` |
| `train.py` | Re-enable IPEX import gated on `torch.__version__ < "2.11"` | `312045b3` |
| `agpt/parallelize.py` | **Revert** `fc3880c5`: restore `use_local_output=enable_sp` (pre-DTensor-TP default). The full DTensor TP requires a newer torch that handles DeviceMesh in AC autograd context. | `8e9ebc23` |

**Verified on Sunspot (2 nodes, 24 XPU tiles):**

| Config | torch 2.10 | torch 2.13 |
|--------|-----------|-----------|
| agpt_2b (TP=1) | PASS | PASS |
| agpt_20b (TP=1) | PASS | PASS |
| agpt_80b (TP=2, compile) | PASS (22.5 tflops) | PASS w/o compile (49 tflops) |
| moe_7b | PASS | PASS |

**80B status by torch version:**

| Torch | Compile | AC | Result |
|-------|---------|-----|--------|
| 2.10 | ON | full | PASS (with IPEX, 22.5 tflops, 7.5% MFU) |
| 2.13 | OFF | full | PASS (49 tflops, 16.4% MFU) |
| 2.13 | ON | full | FAIL (DeviceMesh in AOT autograd — upstream bug) |

---

## 2026-04-23 (17th sync)

**Upstream commits:**

- `703b72ef` — [full DTensor] Use all DTensor for Qwen3 and llama4 at TP region (#2149)
- `95a25420` — fix(hf_datasets): shuffle HuggingFaceTextDataset on re-loop (#3023)
- `d52d2475` — [rl] Generator refactor (#3001)
- `a676793a` — [rl] Rename inference example (#3045)
- `0de35f96` — [profiler] Suppress Callable field from tyro CLI parsing (#3038)
- + 10 more (GraphTrainer, CI, ROCm)

**Breaking changes in `models/llama3/parallelize.py`:**

- `use_local_output=True` → `use_local_output=False` for embed_plan,
  norm_plan, and rowwise_output_plan. TP now keeps tensors as DTensors
  instead of converting to plain tensors.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Replay `use_local_output=False` for embed, norm, rowwise plans | `fc3880c5` |
| `agpt/parallelize.py` | **Reverted in 18th sync** — causes hangs on torch 2.10 and AC crash on 2.13 | `8e9ebc23` |

---

## 2026-04-20 (16th sync)

**Upstream commits:**

- `dd0cbe65` — [GraphTrainer] Remove compile_with_inductor annotation from qwen3 FlexAttention (#3019)
- `ac154e60` — [GraphTrainer] Remove unused standalone GraphTrainerConfig dataclass (#3018)
- `526c7de2` — [graph_trainer] Fix test_bitwise_deterministic AttributeError (#3028)
- `4a61bd87` — Update .md files to use --profiler (#3026)
- `a91f57bd` — [graph_trainer] Add MANIFESTO.md (#3014)
- `fdc71a30` — [Typing] Remove unused pyrefly ignores in token_dispatcher (#3027)

**Files changed:** `experiments/graph_trainer/`, docs, `models/common/token_dispatcher.py` (typing only).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-20 (15th sync)

**Upstream commits:**

- `08da451f` — [Typing] Pyrefly: --remove-unused-ignores during pre-commit (#3010)

**Files changed:** 35 files across models/ and distributed/ — removing unused
`# pyrefly: ignore` comments. No functional changes.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-18 (14th sync)

**Upstream commits:**

- `7cec1660` — [MoE][2/n] Move EP setup from trainer to config registry and model_registry params (#2960)
- `ca46729b` — Update models/README.md (#3012)
- `518d708f` — Todo debt tracker updates (#3011)
- `114fedc9` — fix CP n_tokens_seen overcounting (#2990)
- `de7fc7bd` — Feature/rfc 2408 compute comms overlap (#3020)
- `ab87f584` — ci: install torchvision in transformers backend workflow (#3002)
- `09ea7d8e` — [MoE][1/n] Introduce token dispatcher and replace token reorderer (#2842)
- `c88cfb03` — Add Qwen3-14B GRPO config and 14B_varlen model registry entry (#2941)
- `d2adfd99` — [graph_trainer] Add debug_graph_passes config (#3003)
- `ae4b0a7a` — Unbreak CI (#2999)
- `ad7a8653` — Fused QKV GQAttention implementation (#2878)
- `aa0214b8` — [ROCm][CI] Turn off ROCm CI on experimental workflows (#2874)
- `2359c2f5` — [rl] Remove test_bitwise_identity.py (#2997)
- `bdbd6582` — [graph_trainer] Copy forward metadata to backward subgraphs (#2875)
- `132683a2` — [graph_trainer] Add torch.no_grad() and graph-based SAC (#2766)
- `5e46f947` — [GraphTrainer] Enable FlexAttention bitwise deterministic tests (#2989)
- `d3918650` — [ez][graph_trainer] fix comment (#2996)
- `1c4e18b9` — [graph_trainer] Add CooR precompile support (#2975)
- `ea6a1cdf` — [GraphTrainer] Fix custom_codegen_pass hang (#2993)
- `dafd87a9` — [GraphTrainer] Disable custom_codegen_pass and clean up (#2992)
- `8b7a85cc` — Fix memory snapshot hang with aot_fx_trace codegen (#2991)
- `be09084b` — [GraphTrainer] Add custom codegen pass with dual-path profiling (#2955)

**Major breaking changes:**

1. **Attention backend abstraction:** `inner_attention`/`mask_type` params replaced
   with `attn_backend: str` + `get_attention_config()` helper
2. **Fused QKV:** `GQAttention.Config` now uses `qkv_linear: BaseQKVLinear.Config`
   instead of separate `wq`/`wkv`; `make_gqa_config()` takes `fuse_qkv` param
3. **Token dispatcher:** `TokenReorderer` removed from `moe.py`, replaced by
   `LocalTokenDispatcher` and subclasses in new `token_dispatcher.py`
4. **Profiler migration:** `torchtitan.tools.profiling` deleted, replaced by
   `torchtitan.tools.profiler` with `Profiler` class
5. **EP config refactor:** `score_before_experts` moved from `MoE.Config` to token
   dispatcher; `make_experts_config()` now requires `top_k` param

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `trainer.py` | Migrate `maybe_enable_profiling`/`maybe_enable_memory_snapshot` to `Profiler.build()` | `e9022648` |
| `agpt/__init__.py` | Replace `inner_attention`/`mask_type` with `attn_backend`/`fuse_qkv`; add `_ezpz_get_attention_config()` XPU-aware wrapper; remove deepcopy overlay functions | `8f6d5cad` |
| `agpt/parallelize.py` | Add `FusedQKVLinear` import and detection; branching TP plan for fused vs unfused QKV | `8f6d5cad` |
| `moe/__init__.py` | Replace `inner_attention`/`mask_type` with `attn_backend`; update `make_moe_config` (remove `score_before_experts`); update `make_experts_config` (add `top_k`, `comm_backend`) | `9d3f3823` |
| `moe/parallelize.py` | Remove `DeepEPExpertParallel`/`ReordererSequenceParallel` imports; remove `hybridep_non_blocking_expert_capacity_factor` reference; EP now handled by token dispatcher | `9d3f3823` |

---

## 2026-04-15 (13th sync)

**Upstream commits:**

- `98ec7b4f` — [rl] Add batched RL training with varlen sequence packing (#2906)

**Files changed:** `experiments/rl/` only (actors, types, config_registry).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-15 (12th sync)

**Upstream commits:**

- `0610235b` — Add bf16 optimizer state support via step pre-hook (#2732)

**Files changed:** `components/optimizer.py`, `docs/bf16_optimizer_states.md`, tests.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-15 (11th sync)

**Upstream commits:**

- `5242bdf7` — Enable per-layer compile with or without MoE (#2741)
- `74485ea3` — Update pytorch nightly to cu13 (#2945)
- `42170d8a` — Revert FlexAttn max_autotune default to True (#2964)

**Breaking changes:**

- `distributed/compile.py` — consolidated `apply_compile_dense` and
  `apply_compile_sparse` into single `apply_compile`. Unconditionally sets
  `torch._dynamo.config.capture_scalar_outputs = True`.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Update import from `apply_compile_dense` to `apply_compile`; reset `capture_scalar_outputs = False` after apply for dense models (prevents unbacked symbol crash in compiled loss with TP + loss_parallel) | `d09d9708`, `e8cbb8ef` |

---

## 2026-04-14 (10th sync)

**Upstream commits:**

- `b35ca339` — [GraphTrainer][AutoDev] Extract remove-noop graph passes into dedicated module (#2952)
- `6d8c7e90` — [GraphTrainer] Update bitwise hash (#2962)

**Files changed:** `experiments/graph_trainer/` only (pass refactoring).

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-14 (9th sync)

**Upstream commits:**

- `6f7a6a79` — [CI] Fix torchvision::nms error in RL integration tests (#2959)
- `4f73d027` — [AutoDev] Restrict agent to only read actionable board items (#2953)
- `f5ecda7e` — [GraphTrainer] Enable regional_inductor for GraphTrainer (#2869)
- `b245fcaa` — [RL] Two small fixes in `inference_example.py` (#2944)
- `c630d30f` — [rl] Add torchcomms installation to setup instructions and remove xformers (#2943)

**Files changed:** `experiments/graph_trainer/`, `experiments/rl/`, CI workflows only.

No changes to `models/`, `distributed/`, or `trainer.py`.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-13 (8th sync)

**Upstream commits:**

- `878041cb` — [Bugfix] Reenable llvm with triton pin update (#2873) <!-- codespell:ignore-line -->

**Files changed:** `models/common/attention.py` — removed `DISABLE_LLVM_OPT=1`
env var workaround. The upstream Triton pin (pytorch/pytorch#179586) fixed
the LLVM change that caused FlexAttention failures.

**Changes required in ezpz:** None. We don't use FlexAttention on XPU.

---

## 2026-04-13 (7th sync)

**Upstream commits:**

- `5c02bb54` — Enable 2 tier compilation with flex attention (#2929)
- `9f055b43` — [AutoDev] Default FlexAttn max_autotune to False (#2935)

**Files changed in `models/common/`:**

- `models/common/attention.py` — `FlexAttention.inductor_configs` changed:
  `wrap_inductor_compiled_regions` True (was False),
  `max_autotune` False (was True),
  `coordinate_descent_tuning` False (was True).

**Changes required in ezpz:** None. Our agpt configs use
`XPUScaledDotProductAttention` (not FlexAttention), and MoE configs use
`ScaledDotProductAttention` via the `_sdpa` variants. FlexAttention is only
used by `moe_debugmodel_flex_attn` and `moe_10b_2b` (non-SDPA), which are
not run on XPU.

---

## 2026-04-12 (6th sync)

**Upstream commits:**

- `dba7154b` — [GraphTrainer][AutoDev] Add remove_identity_slice_pass graph pass (#2920)
- `b9e8d1d1` — [GraphTrainer][AutoDev] Add remove_identity_view_pass graph pass (#2919)
- `314577bc` — [GraphTrainer][AutoDev] Add remove_detach_pass graph pass (#2917)

**Files changed:** `experiments/graph_trainer/passes.py`, `experiments/graph_trainer/tests/test_passes.py` only.

No changes to `models/`, `distributed/`, or `trainer.py`.

**Changes required in ezpz:** None. Clean merge.

---

## 2026-04-11 (5th sync)

**Upstream commits:**

- `12979c66` — [Qwen3VL] add qwen3 vl (#2409)
- `ab598d9c` — Skip parallelize_fn for seed checkpoint creation (#2928)
- `a5a60cc2` — [DeepEP] hide buffer init from sac (#2794)
- `dd33cde0` — TODO Debt Tracker (#2887)
- `1f9c677a` — [GraphTrainer] Nightly scout hands off (#2933)
- `98ff51e0` — [rl] Update monarch version in README (#2930)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `gradient_divide_factor` param from
  `apply_fsdp`, `disable_fsdp_gradient_division` moved to `distributed/fsdp.py`
- `models/deepseek_v3/parallelize.py` — same: removed `gradient_divide_factor`

**Breaking changes in `distributed/`:**

- `distributed/parallel_dims.py` — removed `fsdp_gradient_divide_factor` property
- `distributed/fsdp.py` — added `disable_fsdp_gradient_division` function
  (previously each model had its own copy)

**Changes required in ezpz:**

| File | Change | Status |
|------|--------|--------|
| `moe/parallelize.py` | Remove `gradient_divide_factor` from `apply_fsdp` call | Done |
| `moe/parallelize.py` | Replace `apply_compile_sparse` with per-block compile without `fullgraph=True` — MoE routing dynamic shapes cause `FailOnRecompileLimitHit` after upstream `00b7f569` removed `maybe_enable_amp` context boundary | Done |
| `moe/__init__.py` | Add `_10b_2b_sdpa()` variant using SDPA instead of FlexAttention — FlexAttention triggers `torch.autocast(dtype=torch.float32)` in MoE router which XPU doesn't support | Done |
| `moe/config_registry.py` | Add `moe_10b_2b_sdpa()` config entry | Done |
| `agpt/parallelize.py` | No changes needed — doesn't use `gradient_divide_factor` | OK |

**Note on MoE compile breakage:**

Upstream `00b7f569` (merged in 4th sync) removed `maybe_enable_amp` from
`trainer.py`'s `forward_backward_step`. This changed how `torch._dynamo`
traces MoE models — the context manager previously provided a graph boundary
that prevented recompilation. Without it, `fullgraph=True` in
`apply_compile_sparse` hits the recompilation limit on MoE routing's dynamic
expert dispatch. Fixed by applying compile per-block without `fullgraph=True`
in the ezpz MOE parallelize.

**Note on FlexAttention + MoE on XPU:**

The `10B_2B` config uses `FlexAttention` which calls
`torch.autocast(device_type="xpu", dtype=torch.float32)` in the MoE router
(`torchtitan/models/common/moe.py:231`). XPU autocast only supports bf16/fp16,
not fp32, causing a crash. Added `10B_2B_sdpa` variant that uses SDPA with
causal masking instead.

**Verified working (2026-04-11):**

| Config | compile | Status |
|--------|---------|--------|
| `moe_debugmodel` | enabled | OK |
| `moe_debugmodel` | disabled | OK |
| `moe_10b_2b` | enabled | FAIL — FlexAttention fp32 autocast on XPU |
| `moe_10b_2b_sdpa` | enabled | Testing |

---

## 2026-04-10 (4th sync)

**Upstream commits:**

- `00b7f569` — [FSDP2] replace amp and replicate with fully_shard (#2900)
- `7ef10559` — add CITATION.cff file (#2925)
- `55658e93` — [RL] Make reference model and KL penalty optional (#2750)
- `6930593b` — [GraphTrainer] Add bitwise deterministic test to H100 CI (#2921)
- `e24e465c` — [GraphTrainer] Changed tagging flex_attn via graph_pass (#2924)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `apply_replicate`, removed
  `fsdp_enabled` / `dp_replicate_enabled` branching, always use `fully_shard`
- `models/deepseek_v3/parallelize.py` — same refactor as llama3

**Breaking changes in `distributed/` and `trainer.py`:**

- `distributed/utils.py` — removed `maybe_enable_amp()` function. AMP is now
  handled internally by `fully_shard`'s `MixedPrecisionPolicy`.
- `distributed/parallel_dims.py` — `_mesh_exist` now returns `True` for `"fsdp"`
  always, so FSDP mesh exists even at degree 1.
- `trainer.py` — removed `self.maybe_enable_amp` from `forward_backward_step`,
  breaking MoE compile with `fullgraph=True` (see note above).

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Removed `apply_replicate`, always use `fully_shard` | `5930b392` |
| `moe/parallelize.py` | Same: removed `apply_replicate` import and usage | `5930b392` |
| `trainer.py` | Removed `maybe_enable_amp` references (2 locations) | `6a2a09ec` |

---

## 2026-04-10 (3rd sync)

**Upstream commits:**

- `00b7f569` — [FSDP2] replace amp and replicate with fully_shard (#2900)
- `7ef10559` — add CITATION.cff file (#2925)
- `55658e93` — [RL] Make reference model and KL penalty optional (#2750)
- `6930593b` — [GraphTrainer] Add bitwise deterministic test to H100 CI (#2921)
- `e24e465c` — [GraphTrainer] Changed tagging flex_attn via graph_pass (#2924)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- `models/llama3/parallelize.py` — removed `apply_replicate`, removed
  `fsdp_enabled` / `dp_replicate_enabled` branching, always use `fully_shard`
- `models/deepseek_v3/parallelize.py` — same refactor as llama3

**Breaking changes in `distributed/`:**

- `distributed/utils.py` — removed `maybe_enable_amp()` function. AMP is now
  handled internally by `fully_shard`'s `MixedPrecisionPolicy`.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `agpt/parallelize.py` | Removed `apply_replicate`, removed `fsdp_enabled` branching, always use `fully_shard` | `5930b392` |
| `moe/parallelize.py` | Same: removed `apply_replicate` import and usage, always use `fully_shard` | `5930b392` |
| `trainer.py` | Removed `maybe_enable_amp` references (2 locations) | `6a2a09ec` |

---

## 2026-04-10 (2nd sync)

**Upstream commits:**

- `5470cc7e` — GQAttention: Combine `q_norm` and `k_norm` into `qk_norm` (#2872)
- `4b45999f` — [GraphTrainer] Add FlexAttention bitwise deterministic tests (#2903)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- Neither `models/llama3/` nor `models/deepseek_v3/` changed.

**Breaking changes in `models/common/`:**

- `models/common/attention.py` — `GQAttention.Config` renamed `q_norm` and
  `k_norm` fields to a single `qk_norm` field. Removed the `__post_init__`
  validation that checked both were set together.

**Changes required in ezpz:**

| File | Change | Commit |
|------|--------|--------|
| `qwen3/__init__.py` | Updated 12 config entries from `q_norm=..., k_norm=...` to `qk_norm=...` | `4fe7aa48` |

`agpt` and `moe` were unaffected — `agpt` doesn't use QK norms, and `moe` has
its own `q_norm` field on a custom `DeepSeekAttention` config (not `GQAttention`).

---

## 2026-04-10 (1st sync)

**Upstream commits:**

- `8328876d` — [RL] Fix RL h100 workflow + `enable_gqa` flag in attention (#2891)
- `3c811045` — [GraphTrainer] Document aot_fx_trace compilation mode (#2912)

**Files changed in `models/llama3/` and `models/deepseek_v3/`:**

- Neither changed.

**Changes in `models/common/`:**

- `models/common/attention.py` — added `enable_gqa` parameter to
  `FlexAttention.forward()` and `ScaledDotProductAttention.forward()`.
  The ezpz branch already had the `VarlenAttention` forwarding for
  `enable_gqa`, which matched the upstream addition — no conflict.

**Changes required in ezpz:**

- None. Clean merge.
