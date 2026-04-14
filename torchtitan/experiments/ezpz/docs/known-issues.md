# Known Issues and Operational Notes

Troubleshooting reference for running ezpz experiments across ALCF machines.

## torch.compile + SDPA `set_priority=True` (PyTorch 2.11)

**Symptoms:** `RuntimeError('Invalid backend')` during `torch.compile` tracing
of `F.scaled_dot_product_attention`. Crashes on all ranks at model init, before
any training steps run.

**Root cause:** `torch._dynamo` bug — `SDPAKernelVariable.enter()` in
`ctx_manager.py` creates FX graph nodes that pass proxy objects to `int()`
when `set_priority=True`, producing invalid backend IDs for
`_set_sdp_priority_order()`. Only triggers in the full model context
(activation checkpointing + DTensor + distributed), not in standalone tests.

**Fix:** `EzpzScaledDotProductAttention` in `experiments/ezpz/agpt/__init__.py`
overrides `forward()` to call `sdpa_kernel(self.sdpa_backends)` without
`set_priority=True`. Use `_default_inner_attention()` in configs — it
auto-selects the right subclass based on device (XPU vs CUDA).

**Affects:** All model configs using SDPA with `torch.compile` enabled.

## Blendcorpus data cache race condition

**Symptoms:** `FileNotFoundError` or `EOFError` on
`.cache/blendcorpus/*_index.npy` when launching with a new combination of
`seq_len` and `global_batch_size`.

**Root cause:** The blendcorpus data loader caches index files keyed by a hash
of (seq_len, global_batch_size, dataset). On the first run with a new
combination, rank 0 builds the cache while other ranks race to read it.
The hash changes with `global_batch_size`, which depends on parallelism
settings (e.g. TP=4 FSDP=2 LBS=1 gives GBS=2, different from FSDP=8 LBS=2
which gives GBS=16).

**Workaround:** Just retry — rank 0 builds the cache on the first attempt, so
the second run will find the files. Kill stale processes first:

```bash
ssh <node> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
```

## CXI resource leak after killing processes (Polaris)

**Symptoms:** `mpiexec` fails instantly with:

```
CXI alloc failed on cxi0: request exceeds PTEs, TXQs, TGQs, EQs, CTs, LEs, ACs limits
```

**Root cause:** When training processes are killed (via SIGTERM, TaskStop, or
SSH disconnect), the Slingshot CXI network resources are not released by the
palsd daemon.

**Fix:**

1. Kill ALL related processes on ALL nodes in the allocation
2. Wait a few seconds for CXI cleanup
3. If still failing, submit a new PBS job (fresh nodes)

```bash
# Kill on both nodes
ssh <node1> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
ssh <node2> 'ps aux | grep torchtitan | grep -v grep | awk "{print \$2}" | xargs -r kill'
```

## Memory limits by machine

### Polaris (8x NVIDIA A100-SXM4-40GB per 2-node job)

| Config | Status | Memory | Parallelism |
|--------|--------|--------|-------------|
| agpt debugmodel | PASS | 2.23 GiB (6%) | FSDP=8 |
| agpt 2B | PASS | 24.98 GiB (63%) | FSDP=8 |
| agpt 7B | PASS | 12.33 GiB (31%) | FSDP=8 |
| agpt 20B | PASS | 35.07 GiB (89%) | FSDP=8, LBS=1, seq=4096 |
| agpt 50B | OOM | 36/39.5 GiB | TP=2, FSDP=4, LBS=1, seq=4096 |
| agpt 80B | OOM | 38.2/39.5 GiB | TP=4, FSDP=2, LBS=1, seq=4096 |
| moe debugmodel-4B | PASS | <54% | FSDP=8 |
| moe 7B | PASS | 35.09 GiB (89%) | FSDP=8, AC=none |
| moe 10B_2B_sdpa | PASS | 30.22 GiB (77%) | FSDP=8, AC=none, LBS=1 |

50B+ agpt models need 4+ Polaris nodes or A100-80GB GPUs.

### Aurora / Sunspot (24x Intel Max 1550 per 2-node job)

- 64 GiB per tile, 12 tiles per node
- 80B fits at TP=2 on older Aurora nodes (`x1921c1s*`) but OOMs on newer nodes (`x4216c5s*`)
- 80B variants (alt, wide, deep) run reliably at TP=3+ on both Aurora and Sunspot
- Best 80B throughput: 80B_wide TP=4 compile = 68 TPS, 11.79% MFU (Aurora)

## Tokenizer compatibility

The `agpt_8B` config uses `vocab_size=128256` (Llama 3 architecture) which
requires a Llama 3 tokenizer. The available tokenizer on ALCF systems is
Gemma (`vocab_size=32000`), causing a CUDA assert on embedding index out
of bounds.

**Affected:** `agpt_8B` only. All other agpt configs use `vocab_size=32000`
or `vocab_size=256128` (compatible with Gemma tokenizer).

## MoE activation checkpointing incompatibility

**Symptoms:** `CheckpointError` during AC recomputation for MoE configs 7B+.

**Root cause:** MoE routing is non-deterministic (expert dispatch varies
between forward passes). Activation checkpointing recomputes the forward
pass during backward, but the routing decisions differ from the original
forward, causing mismatched tensor shapes.

**Fix:** Add `--activation_checkpoint.mode none` for MoE configs 7B and larger
(7B, 10B_2B, 10B_2B_sdpa, 16B, small). Smaller configs (debugmodel, 500M,
2B, 4B) are not affected in short runs.

**Trade-off:** Disabling AC increases memory usage significantly. The moe 7B
config uses 35 GiB (89%) with AC=none on Polaris A100-40GB.

## MoE + Tensor Parallelism (TP > 1)

**Symptoms:** `AssertionError: q, k, v must have the same placements, but got
q=(Shard(dim=2),), k=(Shard(dim=2),), v=(Replicate())`

**Affected:** All MoE configs with `--parallelism.tensor_parallel_degree > 1`.

**Root cause:** The MoE MLA (Multi-head Latent Attention) implementation uses
LoRA-based KV projection with asymmetric sharding. When TP shards q and k
across heads, the v tensor from `wkv_b` remains replicated because its
projection shape doesn't match the TP sharding pattern.

**Workaround:** Use TP=1 for MoE models. Expert parallelism (EP>1) is also
blocked on `aurora_frameworks-2025.3.1` (missing `ShardPlacementResult`).
MoE scaling requires a newer PyTorch version.

## Context Parallelism (CP > 1) on agpt

**Symptoms:** With compile: `TorchRuntimeError: Dynamo failed to run FX node
with fake tensors: call_function scaled_dot_product_attention`. Without
compile: `RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor`.

**Affected:** All agpt configs with `--parallelism.context_parallel_degree > 1`.

**Root cause:** The agpt attention implementation doesn't convert all tensors
(e.g. RoPE embeddings) to DTensors on the CP mesh. When CP shards Q/K/V
along the sequence dimension, the non-sharded tensors remain as regular
`torch.Tensor`, causing DTensor/Tensor mixing errors.

**Workaround:** Use CP=1 (default). For longer sequences, increase TP instead.

## 80B TP=2 on Aurora (regression since ~2026-04-04)

**Symptoms:** `torch.OutOfMemoryError` or `UR_RESULT_ERROR_OUT_OF_RESOURCES`
on step 2. Step 1 completes at 60.75 GiB (94.94%) but step 2 needs 5.25 GiB
with only 5.19 GiB free (missed by 60 MiB). Memory numbers are identical
across all tested nodes.

**Affected:** All 80B variants at TP=2 on Aurora, tested 2026-04-12/13 on
4 different node pairs (`x4216c5s*`, `x4704c1s*`, `x4219c2s*`, `x4310c3s*`).

**Previously worked:** 80B TP=2 ran successfully on Aurora on 2026-04-04
(nodes `x4201c1s1b0n0`, 89 TPS with compile=on). Also works on Sunspot
(85 TPS, 2026-03-30).

**Likely cause:** An undiagnosed regression between 2026-04-04 and 2026-04-12,
likely at the system level (framework libraries, Level Zero driver, or XPU
runtime) rather than in torchtitan code. Evidence:
- 4 different node pairs all fail with identical memory numbers
  (42.82 GiB allocated, 12.36 GiB reserved, 5.19 GiB free) — rules out
  node-specific hardware issues
- Same torchtitan code at the benchmark commit (c6ff706) also OOMs —
  rules out a code regression
- The `aurora_frameworks-2025.3.1` module version string is unchanged but
  the underlying libraries may have been updated in place

**Tried and failed:**
- `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`
- `--training.gc_freq 1`
- `--metrics.no-enable_wandb`
- `--training.enable_cpu_offload`
- Same code at benchmark commit (c6ff706)

**Workaround:** Use TP=4 with 80B_wide (68 TPS, 11.79% MFU). Or run on Sunspot.

**To investigate:** Compare the exact library versions (Level Zero, PyTorch
internals, IPEX) between the April 4 working environment and the current one.
Check if `aurora_frameworks-2025.3.1` was patched in place between those dates.
