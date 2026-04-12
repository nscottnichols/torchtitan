# Benchmarks

We provide the [torchtitan/experiments/ezpz/run_benchmarks.sh](../run_benchmarks.sh)
which will:

1. Run through each of the AuroraGPT model configs:
   - AuroraGPT-2B
   - AuroraGPT-20B
   - AuroraGPT-MoE-debug
   - AuroraGPT-MoE-10B-2B

   for 10 training steps and record:

   - Throughput:
       - TPS (tokens per second)
       - TFLOPS
   - Hardware utilization (MFU %)
   - Wall Time (s) (end-to-end)
   - Exit status

The full output is included below for reference:

## Benchmarks on Sunspot

<details closed><summary><code>run_benchmarks.log</code></summary>

```bash
#[aurora_frameworks-2025.3.1](torchtitan-aurora_frameworks-2025.3.1)
#[/t/d/f/p/s/torchtitan][ezpz][$!?⇡] [ ]
#[04/04/26 @ 11:34:30][x1921c1s0b0n0]
; bash torchtitan/experiments/ezpz/run_benchmarks.sh --training.local-batch-size 1
[2026-04-04-113435][I][/dev/fd/63:2850] Detected PBS scheduler environment.
[2026-04-04-113435][I][/dev/fd/63:2582] [ezpz_setup_env]...
[2026-04-04-113435][I][/dev/fd/63:1363] [PYTHON]
[2026-04-04-113436][I][/dev/fd/63:1407]   - Found both conda_prefix and virtual_env in environment.
[2026-04-04-113436][I][/dev/fd/63:1408]   - Using conda from: /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
[2026-04-04-113436][I][/dev/fd/63:1409]   - Using venv from: /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1
[2026-04-04-113436][I][/dev/fd/63:1418]   - Using python from: /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1/bin/python3
[2026-04-04-113436][I][/dev/fd/63:2424] [JOB]
[2026-04-04-113436][I][/dev/fd/63:2425]   - Parsing job env for foremans
[2026-04-04-113436][I][/dev/fd/63:2426]   - Detected pbs scheduler
[2026-04-04-113436][I][/dev/fd/63:2427]   - Machine: sunspot
[2026-04-04-113436][I][/dev/fd/63:2428]   - Hostname: x1921c1s0b0n0
[2026-04-04-113436][I][/dev/fd/63:2338]   - PBS_JOBID=12464028.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
    to calculate:
      - num_hosts: 2
      - num_cores_per_host: 208
      - num_cpus_per_host: 104
      - num_gpus_per_host: 12
      - depth: 8
      - num_gpus: 24
[2026-04-04-113436][I][/dev/fd/63:1844] [HOSTS]
[2026-04-04-113436][I][/dev/fd/63:1846]   - Detected PBS Scheduler
[2026-04-04-113436][I][/dev/fd/63:1864]   - HOSTFILE=/var/spool/pbs/aux/12464028.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2026-04-04-113436][I][/dev/fd/63:1865]   - NHOSTS=2
[2026-04-04-113436][I][/dev/fd/63:1866]   - HOSTS:
[2026-04-04-113436][I][/dev/fd/63:1869]     - [host:0] - x1921c1s0b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov
[2026-04-04-113436][I][/dev/fd/63:1869]     - [host:1] - x1921c1s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov
[2026-04-04-113436][I][/dev/fd/63:2030] [DIST_INFO]
[2026-04-04-113436][I][/dev/fd/63:2031]   - HOSTFILE=/var/spool/pbs/aux/12464028.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2026-04-04-113436][I][/dev/fd/63:2032]   - NHOSTS=2
[2026-04-04-113436][I][/dev/fd/63:2033]   - NGPU_PER_HOST=12
[2026-04-04-113436][I][/dev/fd/63:2034]   - NGPUS=24
[2026-04-04-113436][I][/dev/fd/63:2606] [✓] Finished [ezpz_setup_env]
============================================================
 ezpz benchmarks — 20260404_113436
 steps=10  gpus=24  nodes=2
============================================================

--- [agpt_2b] running (module=ezpz.agpt config=agpt_2b) ---
    started @ 2026-04-04-113436
    logfile=outputs/benchmarks/20260404_113436/agpt_2b.log

    status=OK  wall=79s  tps=4707  tflops=52.66  mfu=17.66

--- [agpt_20b] running (module=ezpz.agpt config=agpt_20b) ---
    started @ 2026-04-04-113555
    logfile=outputs/benchmarks/20260404_113436/agpt_20b.log

    status=OK  wall=308s  tps=349  tflops=51.98  mfu=17.43

--- [moe_debugmodel] running (module=ezpz.moe config=moe_debugmodel) ---
    started @ 2026-04-04-114104
    logfile=outputs/benchmarks/20260404_113436/moe_debugmodel.log

    status=OK  wall=32s  tps=8385  tflops=4.94  mfu=1.66

--- [moe_10b_2b] running (module=ezpz.moe config=moe_10b_2b) ---
    started @ 2026-04-04-114136
    logfile=outputs/benchmarks/20260404_113436/moe_10b_2b.log

    status=OK  wall=2042s  tps=271  tflops=3.81  mfu=1.28

============================================================
# ezpz Benchmark Report

| Field   | Value                                                  |
|---------|--------------------------------------------------------|
| Date    | 2026-04-04T11:34:36-05:00                              |
| Commit  | 421b6d33                                               |
| Machine | x1921c1s0b0n0                                          |
| Job ID  | 12464028.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov |
| Nodes   | 2                                                      |
| GPUs    | 24                                                     |
| Steps   | 10                                                     |

## Results

| Config         | Steps |     TPS |   TFLOPS |    MFU | Wall Time (s) | Status |
|----------------|-------|---------|----------|--------|---------------|--------|
| agpt_2b        |    10 |    4707 |    52.66 |  17.66 |            79 |     OK |
| agpt_20b       |    10 |     349 |    51.98 |  17.43 |           308 |     OK |
| moe_debugmodel |    10 |    8385 |     4.94 |   1.66 |            32 |     OK |
| moe_10b_2b     |    10 |     271 |     3.81 |   1.28 |          2042 |     OK |

Logs: `outputs/benchmarks/20260404_113436/`
============================================================

Report saved to: outputs/benchmarks/20260404_113436/report.md
Logs saved to:   outputs/benchmarks/20260404_113436/
```

</details>

## Benchmarks on Aurora

<details closed><summary><code>run_benchmarks.log</code></summary>

```bash
#[aurora_frameworks-2025.3.1](torchtitan-ezpz-aurora_frameworks-2025.3.1)[5s]
#[04/04/26,11:30:40][x4209c2s4b0n0][/f/A/f/p/s/torchtitan-ezpz][ezpz][!?⇡]
; bash torchtitan/experiments/ezpz/run_benchmarks.sh --training.local-batch-size 1
[2026-04-04-113203][I][/dev/fd/63:2850] Detected PBS scheduler environment.
[2026-04-04-113203][I][/dev/fd/63:2582] [ezpz_setup_env]...
[2026-04-04-113203][I][/dev/fd/63:1363] [PYTHON]
[2026-04-04-113203][I][/dev/fd/63:1407]   - Found both conda_prefix and virtual_env in environment.
[2026-04-04-113203][I][/dev/fd/63:1408]   - Using conda from: /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
[2026-04-04-113203][I][/dev/fd/63:1409]   - Using venv from: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/venvs/aurora/torchtitan-ezpz-aurora_frameworks-2025.3.1
[2026-04-04-113203][I][/dev/fd/63:1418]   - Using python from: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/venvs/aurora/torchtitan-ezpz-aurora_frameworks-2025.3.1/bin/python3
[2026-04-04-113203][I][/dev/fd/63:2424] [JOB]
[2026-04-04-113203][I][/dev/fd/63:2425]   - Parsing job env for foremans
[2026-04-04-113203][I][/dev/fd/63:2426]   - Detected pbs scheduler
[2026-04-04-113203][I][/dev/fd/63:2427]   - Machine: aurora
[2026-04-04-113203][I][/dev/fd/63:2428]   - Hostname: x4209c2s4b0n0
[2026-04-04-113205][I][/dev/fd/63:2338]   - PBS_JOBID=8421140.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    to calculate:
      - num_hosts: 2
      - num_cores_per_host: 208
      - num_cpus_per_host: 104
      - num_gpus_per_host: 12
      - depth: 8
      - num_gpus: 24
[2026-04-04-113205][I][/dev/fd/63:1844] [HOSTS]
[2026-04-04-113205][I][/dev/fd/63:1846]   - Detected PBS Scheduler
[2026-04-04-113205][I][/dev/fd/63:1864]   - HOSTFILE=/var/spool/pbs/aux/8421140.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
[2026-04-04-113205][I][/dev/fd/63:1865]   - NHOSTS=2
[2026-04-04-113205][I][/dev/fd/63:1866]   - HOSTS:
[2026-04-04-113205][I][/dev/fd/63:1869]     - [host:0] - x4209c2s4b0n0.hsn.cm.aurora.alcf.anl.gov
[2026-04-04-113205][I][/dev/fd/63:1869]     - [host:1] - x4209c2s6b0n0.hsn.cm.aurora.alcf.anl.gov
[2026-04-04-113205][I][/dev/fd/63:2030] [DIST_INFO]
[2026-04-04-113205][I][/dev/fd/63:2031]   - HOSTFILE=/var/spool/pbs/aux/8421140.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
[2026-04-04-113205][I][/dev/fd/63:2032]   - NHOSTS=2
[2026-04-04-113205][I][/dev/fd/63:2033]   - NGPU_PER_HOST=12
[2026-04-04-113205][I][/dev/fd/63:2034]   - NGPUS=24
[2026-04-04-113205][I][/dev/fd/63:2606] [✓] Finished [ezpz_setup_env]
============================================================
 ezpz benchmarks — 20260404_113205
 steps=10  gpus=24  nodes=2
============================================================

--- [agpt_2b] running (module=ezpz.agpt config=agpt_2b) ---
    started @ 2026-04-04-113205
    logfile=outputs/benchmarks/20260404_113205/agpt_2b.log

    status=OK  wall=478s  tps=4396  tflops=49.18  mfu=16.49

--- [agpt_20b] running (module=ezpz.agpt config=agpt_20b) ---
    started @ 2026-04-04-114003
    logfile=outputs/benchmarks/20260404_113205/agpt_20b.log

    status=OK  wall=317s  tps=356  tflops=52.91  mfu=17.74

--- [moe_debugmodel] running (module=ezpz.moe config=moe_debugmodel) ---
    started @ 2026-04-04-114520
    logfile=outputs/benchmarks/20260404_113205/moe_debugmodel.log

    status=OK  wall=108s  tps=8877  tflops=5.23  mfu=1.75

--- [moe_10b_2b] running (module=ezpz.moe config=moe_10b_2b) ---
    started @ 2026-04-04-114708
    logfile=outputs/benchmarks/20260404_113205/moe_10b_2b.log

    status=OK  wall=2037s  tps=2  tflops=0.03  mfu=0.01

============================================================
# ezpz Benchmark Report

| Field   | Value                                                   |
|---------|---------------------------------------------------------|
| Date    | 2026-04-04T11:32:05-05:00                               |
| Commit  | eeb4cce0                                                |
| Machine | x4209c2s4b0n0                                           |
| Job ID  | 8421140.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes   | 2                                                       |
| GPUs    | 24                                                      |
| Steps   | 10                                                      |

## Results

| Config         | Steps |     TPS |   TFLOPS |    MFU | Wall Time (s) | Status |
|----------------|-------|---------|----------|--------|---------------|--------|
| agpt_2b        |    10 |    4396 |    49.18 |  16.49 |           478 |     OK |
| agpt_20b       |    10 |     356 |    52.91 |  17.74 |           317 |     OK |
| moe_debugmodel |    10 |    8877 |     5.23 |   1.75 |           108 |     OK |
| moe_10b_2b     |    10 |       2 |     0.03 |   0.01 |          2037 |     OK |

Logs: `outputs/benchmarks/20260404_113205/`
============================================================

Report saved to: outputs/benchmarks/20260404_113205/report.md
Logs saved to:   outputs/benchmarks/20260404_113205/
took: 49m 3s
```

</details>

## AGPT 2B Throughput Optimization (2026-04-10)

**Platform:** Sunspot (Intel Data Center GPU Max 1550, 64GB HBM per tile)
**Nodes:** 2 (24 XPU tiles, 12 per node)
**Model:** agpt_2b (1.99B params, 12 layers, 16 heads, 4 KV heads, dim=2048)
**Sequence length:** 8192 | **Precision:** bfloat16 | **Optimizer:** SophiaG

Starting from a well-tuned baseline (compile + FSDP), we explored parallelism
strategies, activation checkpointing modes, batch sizes, and attention backends.

### Best Config

```
LBS=1, GAS=1, compile=True, AC=full, fsdp_reshard_after_forward=never
```

**~5,690 tps / 63.6 TFLOPS / 21.3% MFU** at 78% memory — **+3.4%** over baseline.

### Eager (no compile) Results

All runs with `--activation-checkpoint.mode=full`.

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| LBS=1 GAS=1 | 3,670 | 41 | 13.8% | 55% | Eager baseline |
| LBS=1 GAS=2 | 3,740 | 42 | 14.0% | 55% | Marginal GAS benefit |
| LBS=1 GAS=4 | 3,700 | 41 | 13.9% | 55% | No gain over GAS=2 |
| LBS=1 AC=none | OOM | — | — | — | |
| LBS=1 AC=selective | OOM | — | — | — | |
| LBS=2 AC=full | OOM | — | — | — | |

### Compiled Results

All runs with `--compile.enable --activation-checkpoint.mode=full` unless noted.

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| **Baseline** (LBS=1) | 5,500 | 61 | 20.6% | 73% | Default config |
| LBS=2 | 5,030 | 56 | 19.0% | 43% | |
| LBS=4 | 5,340 | 60 | 20.0% | 80% | Near memory limit |
| LBS=5 | OOM | — | — | — | |

### Parallelism

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| FSDP default reshard | 5,500 | 61 | 20.6% | 73% | Baseline |
| **FSDP reshard=never** | **5,690** | **63.6** | **21.3%** | **78%** | **+3.4%** |
| HSDP (replicate=2 shard=12) | 5,620 | 63 | 21.1% | 79% | Marginal at 2 nodes |
| TP=2 | 3,780 | 42 | 14.2% | 60% | Model too small for TP |
| CP=2 (compile) | FAIL | — | — | — | Dynamo can't trace SDPA with CP DTensors |
| CP=2 (no compile) | FAIL | — | — | — | Mixed Tensor/DTensor in residual add |

### Activation Checkpointing

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| AC=full (baseline) | 5,500 | 61 | 20.6% | 73% | |
| AC=memory_budget(0.85) LBS=1 | 5,000 | 56 | 18.8% | 41% | Solver too conservative |
| AC=memory_budget(0.85) LBS=2 | 5,550 | 62 | 20.8% | 69% | Matches baseline |

### Other

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| + num_workers=4 | 5,550 | 62 | 20.8% | 73% | Within noise |
| + disable_loss_parallel | 5,700 | 63.7 | 21.4% | 78% | Marginal |
| FlexAttention | FAIL | — | — | — | Triton XPU compile error |

### Attention Backend

Default SDPA dispatches to `MATH` (naive attention) on XPU. `OVERRIDEABLE` is
Intel's fused attention kernel.

| Backend | Microbench (ms/iter) | E2E no-compile | E2E compile |
|---------|---------------------|----------------|-------------|
| MATH | 65.89 | 3,670 tps | 5,500 tps |
| OVERRIDEABLE | 2.87 (**23x**) | 3,800 tps | 5,640 tps |

`torch.compile` absorbs most of the kernel-level difference. OVERRIDEABLE is
still the correct default (and will matter more without compile or at larger
scale).

### Key Findings

1. **Compile is the biggest lever:** +50% throughput (3,670 → 5,500 tps), 2x memory reduction
2. **`reshard=never` is the only tuning win:** +3.4% by eliminating an allgather per layer
3. **LBS=1 is optimal** with compile — larger batches don't improve throughput
4. **Full AC is required** at seq_len=8192; selective and memory_budget modes either OOM or underperform
5. **TP hurts at 2B scale** — communication overhead exceeds compute savings
6. **HSDP is irrelevant at 2 nodes** — cross-node traffic isn't the bottleneck
7. **FlexAttention is blocked** by Triton XPU's incomplete IR support
8. **OVERRIDEABLE attention is 23x faster** in isolation but compile masks the difference
9. **CP is blocked on XPU** — compile can't trace SDPA with CP's sequence-sharded DTensors, and eager mode hits mixed Tensor/DTensor errors in residual connections

## AGPT 20B Throughput Optimization (2026-04-10)

**Platform:** Sunspot (Intel Data Center GPU Max 1550, 64GB HBM per tile)
**Nodes:** 2 (24 XPU tiles, 12 per node)
**Model:** agpt_20b (21.5B params, 64 layers, 40 heads, 8 KV heads, dim=5120)
**Sequence length:** 8192 | **Precision:** bfloat16 | **Optimizer:** SophiaG

### Best Config

```
LBS=1, GAS=1, compile=True, AC=full, fsdp_reshard_after_forward=default
```

**~352 tps / 52.2 TFLOPS / 17.5% MFU** at 70% memory.

### Parallelism

All runs with compile + AC=full + LBS=1.

### Parallelism

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| **FSDP (reshard=default)** | **352** | **52.2** | **17.5%** | **70%** | **Baseline** |
| FSDP reshard=never | OOM | — | — | — | Unsharded params exceed 64GB |
| FSDP reshard=always | 354 | 52.6 | 17.6% | 68% | Neutral |
| HSDP (replicate=2 shard=12) | 348 | 51.7 | 17.3% | 85% | Higher memory, no throughput gain |
| TP=2 | 273 | 41.0 | 13.7% | 54% | TP allreduce overhead per layer too high |
| TP=2 + reshard=never | 303 | 45.0 | 15.1% | 79% | Better than TP=2 alone, still worse than baseline |

### Gradient Accumulation

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| GAS=1 (baseline) | 352 | 52.2 | 17.5% | 70% | |
| GAS=2 | 357 | 53.1 | 17.8% | 70% | +1.4% |
| **GAS=2 + workers=4 + no loss_parallel + gc=1000** | **358** | **53.3** | **17.9%** | **70%** | **Best: +1.7%** |

### Activation Checkpointing

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| AC=full (baseline) | 352 | 52.2 | 17.5% | 70% | |
| AC=memory_budget(0.85) | OOM | — | — | — | Solver doesn't respect budget |
| AC=memory_budget(0.65) | OOM | — | — | — | Same: always allocates 54.68 GiB |

### Key Findings

1. **GAS=2 is the only measurable win** (+1.7%) — amortizes optimizer step overhead
2. **`reshard=never` OOMs** on the 20B model — keeping 21B params unsharded exceeds 64GB
3. **`reshard=always`** saves 2% memory but no throughput gain
4. **TP=2 hurts** even more than on 2B (273 vs 352 tps) — 64 layers of TP allreduces are expensive
5. **HSDP is neutral** at 2 nodes — cross-node FSDP traffic isn't the bottleneck
6. **`memory_budget` AC is broken** for 20B on XPU — solver ignores the budget target and always allocates the same 54.68 GiB, causing OOM at any budget setting
7. **The 20B model needs more nodes** to improve MFU — at 2 nodes, per-device memory is the binding constraint

## AGPT 80B Throughput Optimization (2026-04-11)

**Platform:** Sunspot (Intel Data Center GPU Max 1550, 64GB HBM per tile)
**Nodes:** 16 (192 XPU tiles, 12 per node)
**Model:** agpt_80b (80.8B params, 84 layers, 72 heads, 12 KV heads, dim=9216)
**Sequence length:** 8192 | **Precision:** bfloat16
**Dataset:** c4_test (blendcorpus blocked by Lustre cache race at 192 ranks)

### Key Constraint: SDPA MATH Backend

The SDPA `MATH` backend materializes the full `N × N` attention score matrix.
For the 80B model with 72 heads: `72 × 8192 × 8192 × 2 bytes = 9 GiB` per tile.
This single allocation exceeds available memory regardless of node count.
**TP is mandatory** to split heads and reduce this allocation.

### Best Config

```
TP=2, LBS=1, compile=True, AC=full, fsdp_reshard_after_forward=default
```

**~108 tps / 59.6 TFLOPS / 20.0% MFU** at 75% memory.

### Feasibility

| Nodes | TP | Status | Notes |
|-------|-----|--------|-------|
| 2 (24 tiles) | 1 | OOM | SIGTERM during first step |
| 8 (96 tiles) | 1 | OOM | 9 GiB attention matrix allocation |
| 16 (192 tiles) | 1 | OOM | Same 9 GiB allocation (per-tile, not total) |
| 16 (192 tiles) | 4 | OK | 2.25 GiB attention matrix fits |
| 16 (192 tiles) | 2 | OK | 4.5 GiB attention matrix fits |

### Throughput (16 nodes / 192 tiles)

All runs with AC=full + LBS=1.

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| TP=4 no compile | 47 | 25.4 | 8.7% | 44% | Eager baseline |
| TP=4 compile | 54 | 29.7 | 10.0% | 46% | TP overhead dominates |
| **TP=2 compile** | **108** | **59.6** | **20.0%** | **75%** | **Best: 2x faster than TP=4** |
| TP=2 LBS=2 compile | OOM | — | — | — | 75% too tight for LBS=2 |

### Blockers

1. **Blendcorpus Lustre race** — **FIXED** by reinstalling from `deps/blendcorpus`
   which has `_load_with_retry` (30 retries × 2s). The installed `site-packages`
   version lacked retry logic entirely.
2. **SDPA MATH backend inside FSDP** — the `OVERRIDEABLE` (XPU fused) backend
   works in isolation (0.15 GB vs 9 GiB) but is **not used inside FSDP**. The
   `sdpa_kernel` context manager and `torch.backends.cuda.enable_math_sdp(False)`
   are both ignored by the XPU dispatch inside FSDP-wrapped modules. This is a
   **PyTorch XPU bug**. TP remains the only workaround.
3. **TP=6 seq_len incompatibility** — 8192 % 6 != 0, so TP=6 fails the
   `seq_len_divisor` assertion.

### OVERRIDEABLE Backend Investigation (2026-04-11)

Extensive investigation into why `OVERRIDEABLE` isn't used inside FSDP:

| Approach | Result |
|----------|--------|
| `XPUScaledDotProductAttention` class variable | Config ownership bug — `build()` created parent class |
| Added explicit `Config` class | Correct class created, OVERRIDEABLE in backend list |
| `sdpa_kernel(OVERRIDEABLE)` in forward | Ignored inside FSDP |
| `sdpa_kernel(OVERRIDEABLE)` only, no fallback | Still 9 GiB allocation |
| `torch.backends.cuda.enable_math_sdp(False)` | Only affects CUDA, not XPU |
| Override `forward()` entirely | Same 9 GiB — XPU dispatch ignores all hints |
| Microbenchmark (no FSDP) | Works: 0.15 GB, 23x faster |

**Conclusion:** XPU's FSDP dispatch path bypasses the `sdpa_kernel` context
manager entirely. The MATH backend is hardcoded in the FSDP code path.
This requires an upstream PyTorch XPU fix.

### Key Findings

1. **TP=2 is optimal** — halving TP degree from 4→2 doubles throughput by halving allreduces per step (84 layers × 2 allreduces vs × 4)
2. **Compile helps less with TP** — only +15% vs +50% for pure FSDP, because TP communication can't be compiled away
3. **80B requires TP** due to the MATH attention matrix bug — no amount of nodes/FSDP sharding fixes a per-tile allocation
4. **Blendcorpus fixed** — reinstalling from local `deps/blendcorpus` provides retry logic for Lustre metadata propagation

## MoE Throughput Optimization (2026-04-12)

**Platform:** Sunspot (Intel Data Center GPU Max 1550, 64GB HBM per tile)
**Nodes:** 2 (24 XPU tiles, 12 per node)
**Configs:** MoE variants from debugmodel to 10B_2B

### Model Configs

| Config | Total | Active | Ratio | Layers | Experts | TopK | Heads | Dim |
|--------|-------|--------|-------|--------|---------|------|-------|-----|
| debugmodel | 0.05B | 0.04B | 89.8% | 6 | 8 | 3 | 16 | 256 |
| 500M | 0.25B | 0.14B | 55.4% | 12 | 16 | 3 | 16 | 512 |
| 2B | 1.61B | 0.49B | 30.3% | 18 | 24 | 3 | 16 | 1024 |
| 4B | 2.89B | 0.81B | 27.9% | 22 | 24 | 3 | 12 | 1536 |
| 7B | 7.54B | 1.57B | 20.8% | 24 | 36 | 3 | 24 | 2048 |
| 10B_2B | 9.41B | 1.98B | 21.1% | 27 | 36 | 3 | 16 | 2048 |

### LBS Scaling (compile enabled, seq_len=4096)

| Config | LBS | TPS | TFLOPS | MFU | Memory | vs default |
|--------|-----|-----|--------|-----|--------|------------|
| 2B | 2 (old default) | 3,860 | 19.3 | 6.5% | 5% | — |
| 2B | 8 | 6,441 | 32.2 | 10.8% | 13% | +67% |
| **2B** | **16 (new default)** | **7,012** | **35.1** | **11.8%** | **24%** | **+82%** |
| 2B | 32 | 7,227 | 36.1 | 12.1% | 47% | +87% |
| 2B | 64 | 7,402 | 37.0 | 12.4% | 93% | +92% |
| 4B | 1 (old default) | 1,470 | 9.8 | 3.3% | 6% | — |
| 4B | 4 | 3,734 | 24.7 | 8.3% | 11% | +154% |
| **4B** | **16 (new default)** | **5,236** | **34.7** | **11.6%** | **32%** | **+256%** |
| 7B | 1 AC=none | 660 | 9.0 | 3.0% | 32% | — |
| 10B_2B_sdpa | 1 AC=none | 572 | 8.0 | 2.7% | 34% | — |
| **10B_2B_sdpa** | **2 AC=none (new default)** | **979** | **13.8** | **4.6%** | **56%** | **+71%** |

### Key Findings

1. **LBS is the biggest lever** — default LBS=1-2 wastes 90%+ of memory.
   Increasing LBS gives 67-256% throughput gains across all MoE configs.
2. **Sweet spots:** 2B→LBS=16, 4B→LBS=16, 10B→LBS=2
3. **AC=full is incompatible with MoE** — routing produces different-shaped
   expert tensors on recomputation (e.g. 2048x206 vs 207x1280). Only AC=none
   works. `determinism-check=none` suppresses the check but causes RuntimeError
   from shape mismatch.
4. **Per-block compile without fullgraph** works but has long warmup (10-20 min
   for 7B/10B). Upstream `fullgraph=True` in `apply_compile_sparse` is
   incompatible with MoE dynamic routing after `00b7f569` removed
   `maybe_enable_amp`.
5. **FlexAttention crashes on XPU** for MoE models due to `torch.autocast(
   dtype=torch.float32)` in the MoE router. Use SDPA variants instead.
6. **Reported MFU is misleadingly low** for MoE — it's computed against total
   params but only top_k experts are active. Corrected for active params, the
   2B at LBS=16 achieves ~39% active-MFU, comparable to dense models.
