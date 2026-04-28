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

- `878041cb` — [Bugfix] Reenable llvm with triton pin update (#2873)

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
