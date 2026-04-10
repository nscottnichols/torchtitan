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

| Config | TPS | TFLOPS | MFU | Memory | Notes |
|--------|-----|--------|-----|--------|-------|
| **FSDP (reshard=default)** | **352** | **52.2** | **17.5%** | **70%** | **Baseline / best** |
| FSDP reshard=never | OOM | — | — | — | Unsharded params exceed 64GB |
| HSDP (replicate=2 shard=12) | 348 | 51.7 | 17.3% | 85% | Higher memory, no throughput gain |
| TP=2 | 273 | 41.0 | 13.7% | 54% | TP allreduce overhead per layer too high |
| TP=2 + reshard=never | 303 | 45.0 | 15.1% | 79% | Better than TP=2 alone, still worse than baseline |

### Key Findings

1. **The baseline is already optimal** for 2 nodes — FSDP with default reshard is the best config
2. **`reshard=never` OOMs** on the 20B model (works for 2B) — keeping 21B params unsharded exceeds 64GB
3. **TP=2 hurts** even more than on 2B (273 vs 352 tps) — 64 layers of TP allreduces are expensive
4. **HSDP is neutral** at 2 nodes — cross-node FSDP traffic isn't the bottleneck at this scale
5. **The 20B model needs more nodes** to improve MFU — at 2 nodes, per-device memory is the binding constraint
