# TODO

## 3. MoE Throughput Optimization

### Problem
MoE models currently get 4-11% MFU on Aurora. torch.compile hurts MoE (-35%
for 500m). The 7b+ configs can't use activation checkpointing (routing
non-determinism causes CheckpointError).

### Knobs to try
- **Expert parallelism (EP)**: Split experts across ranks. Try
  `--parallelism.expert_parallel_degree 12` (one expert group per tile).
  Reduces per-rank expert count, may improve memory and compute balance.
- **Tensor parallelism for MoE**: Try TP=2 or TP=4 for the larger MoE
  configs (7b, 10b). Currently all MoE runs use TP=1 (FSDP only).
- **Context parallelism (CP)**: CP=2 or CP=4 to split the sequence
  dimension. Halves activation memory per rank.
- **Per-block compile**: The MoE parallelize.py already does per-block
  compile instead of fullgraph. Test if `compile.components=["loss"]`
  alone helps (skip model compile, just compile loss).
- **DeepEP / HybridEP backends**: Try
  `--parallelism.expert_parallel_comm_backend deepep` or `hybridep`
  for optimized expert dispatch communication.
- **Float8 quantization**: `Float8LinearConverter` or
  `Float8GroupedMMConverter` for expert layers — cuts compute/memory.
- **Larger batch sizes**: MoE models use less memory than dense.
  Try LBS=2 or LBS=4 for the smaller configs (500m, 2b).

### Experiments to run
```
# EP sweep for 10b
for ep in 1 2 4 12; do
  ezpz launch ... --config moe_10b_2b_sdpa \
    --parallelism.expert_parallel_degree $ep \
    --activation_checkpoint.mode none --compile.no-enable
done

# TP sweep for 7b
for tp in 1 2 4; do
  ezpz launch ... --config moe_7b \
    --parallelism.tensor_parallel_degree $tp \
    --activation_checkpoint.mode none --compile.no-enable
done

# LBS sweep for 2b
for lbs in 1 2 4 8; do
  ezpz launch ... --config moe_2b \
    --training.local_batch_size $lbs --compile.no-enable
done
```

## 4. Aurora Scaling Study

### Goal
Reproduce the Sunspot scaling study (2-128 nodes) on Aurora to measure
weak and strong scaling efficiency.

### Plan
1. **Configs**: Use the best configs from throughput benchmarks:
   - agpt_20b: TP=2, compile=on
   - agpt_80b_wide: TP=4, compile=on
   - moe_2b: no-compile
   - moe_10b_2b_sdpa: no-compile, AC=none

2. **Node counts**: 2, 4, 8, 16, 32, 64, 128 nodes

3. **Metrics**: TPS, TFLOPS, MFU, memory per node, weak scaling
   efficiency (TPS * N_nodes / TPS_1node)

4. **Script**: Extend `scripts/run_scaling_study.sh` or create a new
   one that submits jobs at each node count and collects results.

5. **Output**: Scaling curves (TPS vs nodes), efficiency table,
   report in `docs/experiments/agpt/aurora/` and `moe/aurora/`

### Challenges
- Need large allocations (64-128 nodes on capacity queue)
- May need to adjust TP/DP ratios at different scales
- MoE expert parallelism becomes relevant at higher node counts

## 5. Production Multi-Stage Training Plan

### Goal
Design configs for a production training run combining optimal LRs
(from LR finder) with optimal throughput configs (from benchmarks).

### Recommended configs

**agpt_20b production:**
```
Model:      agpt_20b
TP:         2
Compile:    on
seq_len:    8192
LBS:        1
AC:         full
LR:         4e-4 (AdamW) / 4e-5 (Muon)
Warmup:     200 steps
Decay:      cosine, min_lr_factor=0.1
Optimizer:  AdamW (most stable) or Muon (if LR tuned carefully)
Expected:   355 TPS per 2-node, ~17.7% MFU
```

**agpt_80b_wide production:**
```
Model:      agpt_80b_wide
TP:         4
Compile:    on
seq_len:    8192
LBS:        1
AC:         full
LR:         ~1e-4 (AdamW, extrapolated)
Warmup:     200 steps
Decay:      cosine, min_lr_factor=0.1
Optimizer:  AdamW
Expected:   68 TPS per 2-node, ~11.8% MFU
```

**moe_2b production:**
```
Model:      moe_2b
TP:         1 (FSDP only)
Compile:    off
seq_len:    4096
LBS:        2
AC:         full
LR:         8e-4
Expected:   6,600 TPS per 2-node, ~11.1% MFU
```

### Validation plan
1. Run each config for 1000 steps with the recommended LR
2. Verify loss convergence (should decrease monotonically after warmup)
3. Compare final loss with baseline (default LR 8e-4)
4. Run for 10,000 steps to verify stability
5. Enable checkpointing every 500 steps for fault tolerance

### Multi-stage training
For large-scale runs, consider:
- Stage 1: Short context (2048) for fast initial convergence
- Stage 2: Full context (8192) for long-range capability
- Stage 3: Cooldown with reduced LR for final quality

## 6. Debug 80B TP=2 Aurora OOM

### Problem
80B at TP=2 OOMs on Aurora with `aurora_frameworks-2025.3.1` at model
init or step 2. The same config worked on Sunspot (93.49% memory,
85 TPS). The error is `UR_RESULT_ERROR_OUT_OF_RESOURCES` from the
Level Zero driver, not a PyTorch OOM.

### Evidence
- All 80B variants (80B, 80B_alt, 80B_wide) OOM at TP=2
- 80B_wide gets through step 1 at 94.94% (60.75 GiB) but OOMs on
  step 2 (needs 5.25 GiB, only 5.18 GiB free — missed by 70 MiB)
- Even with cpu_offload, the driver OOMs
- Same code at same commit (c6ff706) OOMs — not a code regression
- Benchmark was run on Sunspot nodes, not Aurora nodes

### Debugging plan

1. **Verify on Sunspot**: Run 80B TP=2 on Sunspot to confirm it still
   works there. If it does, the issue is Aurora-specific.

2. **Check driver versions**: Compare Level Zero / GPU driver versions
   between Aurora and Sunspot. The memory allocator behavior may differ.

3. **Try `PYTORCH_XPU_ALLOC_CONF`**: Test memory allocator tuning:
   ```bash
   export PYTORCH_XPU_ALLOC_CONF="max_split_size_mb:512"
   export PYTORCH_XPU_ALLOC_CONF="expandable_segments:True"
   ```

4. **Memory profiling**: Enable `--profiling.enable_memory_snapshot`
   to get a detailed allocation trace. Compare Aurora vs Sunspot
   allocation patterns.

5. **Reduce memory by 70 MiB**: The 80B_wide TP=2 missed by 70 MiB.
   Try reducing overhead:
   - Disable W&B logging (`--metrics.no-enable_wandb`)
   - Reduce gradient clipping buffer
   - Try `torch.xpu.empty_cache()` before step 2

6. **File bug report**: If driver-level, file with Intel/ALCF support
   team with reproduction steps:
   - Node type, driver version, framework version
   - Exact command that fails
   - Memory allocation trace showing the 70 MiB gap
