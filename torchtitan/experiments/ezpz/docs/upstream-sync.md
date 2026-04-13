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
