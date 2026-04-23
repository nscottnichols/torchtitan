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
