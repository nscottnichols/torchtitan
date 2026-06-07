# Scaling Results

Consolidated scaling data across all AGPT models (2B, 20B, 80B), MoE
variants (2B, 7B), machines (Sunspot, Aurora), and PyTorch versions
(2.10, 2.13). All numbers are weak-scaling unless otherwise noted.

Per-run raw data lives under `outputs/scaling_study_aurora/<TS>/n<N>/<group>/results.json`
(plus `report.md` and per-config `*.log`). Aggregate with
`utils/aggregate_scaling.py`.

---

## Quick Reference

| Model | Best TPS/GPU | Peak MFU | Validated Max N | Stack |
|-------|-------------|----------|-----------------|-------|
| agpt_2b | 7,344 (4N) | 27.6% | 256 (clean), 512+ blocked | Aurora torch 2.13 |
| agpt_20b | 453 (64N) | 22.6% | 128 | Aurora torch 2.13 |
| agpt_80b | 100 (8N) | 18.3% | 32 | Sunspot torch 2.10 |
| moe_2b | 7,121 (1N) | 11.9% | 64 | Sunspot torch 2.10 |
| moe_7b | 1,841 (1N) | 8.4% | 16 | Sunspot torch 2.10 |

---

## agpt_2b

### Aurora torch 2.13 (current production stack)

`FSDP-only (TP=1), compile=on, AC=full, seq_len=8192, LBS=1, dataset=eliplutchok/fineweb-small-sample, 20-step bench`

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS | MFU | Memory | Status | Job |
|-------|------|-----|---------|--------|-----|--------|--------|-----|
| 4 | 48 | 96 | 7,344 | — | 27.55% | — | OK | 2026-05-29 sweep |
| 8 | 96 | 192 | 7,291 | — | 27.36% | — | OK | 2026-05-29 sweep |
| 16 | 192 | 384 | 6,803 | — | 25.53% | — | OK | 2026-05-29 sweep |
| 32 | 384 | 768 | 6,984 | — | 26.20% | — | OK | 2026-05-29 sweep |
| **64** | **768** | **768** | **5,062** | **56.64** | **18.99%** | **19.79 GiB (31%)** | **OK** | [8528805](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/cids97r8) (2026-06-06) |
| **128** | **1,536** | **1,536** | **4,300** | **48.10** | **16.13%** | **19.82 GiB (31%)** | **OK** | [8528834](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/sy6gomvy) (2026-06-06) |
| 256 | 3,072 | 6,144 | 5,002 | — | 18.77% | — | OK | 2026-05-29 sweep |
| 512 | 6,144 | 12,288 | — | — | — | — | Pending | requires prod queue |
| 1,024 | 12,288 | 24,576 | — | — | — | — | Blocked | `set_determinism` init crash, see [memory/project_1024n_init_crash](.). |
| 2,048 | 24,576 | 49,152 | — | — | — | — | Blocked | same as 1,024 |
| 4,096 | 49,152 | 98,304 | — | — | — | — | Blocked | same as 1,024 |

**Note on 64N + 128N MFU drop:** the 2026-05-29 small-N sweep ran at
`LBS=2 GBS=N×24`; the new 64N/128N points ran at `LBS=1 GBS=N×12`
(half the per-GPU work per step), hence lower per-GPU TFLOPS. Re-run
with `LBS=2` to restore apples-to-apples comparison.

**Historical: the n=64/128 CRASH era (2026-05-29 → 2026-06-03)** was a
five-bug stack, fully resolved by 2026-06-06:
1. `.venv.tar.gz` rebuild lost `.venv/bin/` (empty in tarball even though
   present in source venv).
2. Wrapper script's trailing `"$@"` leaked PBS `-v` CLI args into the
   inner `python3 -m torchtitan.experiments.ezpz.train` invocation,
   causing instant arg-parse failure.
3. blendcorpus segfault at ≥768 ranks in `blendcorpus_builder.py:275
   __init__` — bypassed by switching the scaling sweep to
   `SCALING_DATASET=eliplutchok/fineweb-small-sample` (HF streaming).
4. `qsub -- /bin/bash -c "..."` swallowed the script's `#!/bin/bash
   --login` shebang, leaving `module` undefined on the PBS-spawned
   shell → `module load oneapi/release/2025.3.1` silently failed →
   oneAPI MPI binaries (incl. `mpiexec`, `qstat`) not on PATH →
   `ezpz launch` → `get_active_jobid()` → `from sh import qstat`
   → ImportError. Direct `qsub <script>` (no `bash -c`) respects the
   shebang and fixes this.
5. The historical sh.qstat fix attempt sourced `/etc/bash.bashrc.local`
   from inside the script — necessary but not sufficient until (4) was
   also fixed.

Raw failure logs preserved under `outputs/scaling_2b_aurora/20260529_*/n{64,128}/light/results.json.bak-*`
and `outputs/scaling_study_aurora/{20260603_*,20260606_215454}/`.

### Sunspot torch 2.10 (historical reference)

`FSDP only (TP=1), compile=on, AC=full, seq_len=4096, LBS=1`

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 6,457 ± 38 | 77,490 ± 466 | 24.2% | 47.29 GiB (74%) | 100.0% |
| 2 | 24 | 5,529 ± 71 | 132,708 ± 1,714 | 20.8% | 46.80 GiB (73%) | 85.6% |
| 4 | 48 | 5,580 ± 81 | 267,864 ± 3,903 | 20.9% | 46.66 GiB (73%) | 86.4% |
| 8 | 96 | 5,384 ± 173 | 516,912 ± 16,631 | 20.2% | 46.58 GiB (73%) | 83.4% |
| 16 | 192 | 5,032 ± 44 | 966,240 ± 8,553 | 18.9% | 46.42 GiB (73%) | 77.9% |
| 32 | 384 | 5,091 ± 118 | 1,954,944 ± 45,616 | 19.1% | 46.41 GiB (73%) | 78.8% |
| 64 | 768 | 4,783 | 3,673,344 | 17.9% | 46.38 GiB (72%) | 74.1% |

Results: `outputs/scaling_study/20260412_091635/`

---

## agpt_20b

### Aurora torch 2.13

`FSDP only (TP=1), compile=on, AC=full, seq_len=8192, LBS=1, dataset=eliplutchok/fineweb-small-sample`

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS | MFU | Memory | Status | Job |
|-------|------|-----|---------|--------|-----|--------|--------|-----|
| 2 | 24 | 24 | 440 | 65.3 | 22.0% | 32.07 GiB (50%) | OK | 2026-04 |
| 32 | 384 | 768 | 4,500 | 50.3 | 16.9% | 43.96 GiB (69%) | OK | 2026-04 |
| **64** | **768** | **1,536** | **453 (447)** | **67.34 (66.56)** | **22.58% (22.32%)** | **28.86 GiB (45%)** | **OK** | [8521698](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3pm3admv) / [8528805](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i0gra8by) |
| **128** | **1,536** | **3,072** | **417** | **62.12** | **20.83%** | **28.73 GiB (45%)** | **OK** | [8528834](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/5kpt6al1) (2026-06-06) |
| 256–4096 | — | — | — | — | — | — | Pending | |

**Improvement over torch 2.10:** 440 TPS/GPU (2.13) vs 358 (2.10) at 2N = **+23%**.

### Sunspot torch 2.10 (historical)

`FSDP only (TP=1), compile=on, AC=full, seq_len=8192, LBS=1`

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 406 ± 2 | 4,872 ± 33 | 20.3% | 48.52 GiB (76%) | 100.0% |
| 2 | 24 | 358 ± 4 | 8,592 ± 101 | 17.9% | 44.64 GiB (70%) | 88.2% |
| 4 | 48 | 368 ± 1 | 17,664 ± 67 | 18.4% | 42.62 GiB (67%) | 90.6% |
| 8 | 96 | 366 ± 2 | 35,184 ± 203 | 18.3% | 41.85 GiB (65%) | 90.3% |
| 16 | 192 | 360 ± 4 | 69,120 ± 814 | 18.0% | 41.52 GiB (65%) | 88.7% |
| 32 | 384 | 366 ± 0 | 140,736 ± 271 | 18.3% | 41.12 GiB (64%) | 90.3% |
| 64 | 768 | 353 | 271,104 | 17.6% | 40.93 GiB (64%) | 86.9% |

---

## agpt_80b

### Model variants

| Variant | HIDDEN | NLAYERS | HEADS | KV_HEADS | FFN_HIDDEN |
|---------|--------|---------|-------|----------|------------|
| 80B | 9216 | 84 | 72 | 12 | 25600 |
| 80B_alt | 9216 | 84 | 72 | 12 | 25596 |
| 80B_wide | 10752 | 48 | 84 | 12 | 39936 |
| 80B_deep | 7680 | 96 | 60 | 12 | 28672 |
| 80B_deep_alt | 7680 | 96 | 60 | 12 | 28668 |

### Sunspot weak scaling, torch 2.10

`TP=2, compile=on, AC=full, seq_len=8192, LBS=1`

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory |
|-------|------|---------|-----------|-----|--------|
| 1 | 12 | CRASH | — | — | — |
| 2 | 24 | CRASH | — | — | — |
| 4 | 48 | 90 | 4,320 | 16.5% | 53.03 GiB (83%) |
| 8 | 96 | 100 | 9,600 | 18.3% | 49.55 GiB (77%) |
| 16 | 192 | 96 | 18,432 | 17.6% | 48.06 GiB (75%) |
| 32 | 384 | 96.5 ± 0.7 | 37,056 ± 271 | 17.7% | 46.98 GiB (73%) |
| 64 | 768 | CRASH | — | — | — |

### Aurora throughput sweep (2N, torch 2.13, compile=on)

| Variant | TP | DP | GBS | Memory | TPS | TFLOPS | MFU |
|---------|----|----|-----|--------|-----|--------|-----|
| 80B | 2 | 12 | 12 | 59.82 GiB (93%) | 89 | 48.44 | 16.24% |
| 80B_alt | 2 | 12 | 12 | 59.82 GiB (93%) | 84 | 45.75 | 15.34% |
| 80B_deep | 2 | 12 | 12 | 58.84 GiB (92%) | 86 | 47.26 | 15.85% |
| 80B_deep_alt | 2 | 12 | 12 | 58.95 GiB (92%) | 82 | 44.78 | 15.02% |
| 80B_wide | 2 | 12 | 12 | 60.75 GiB (95%) | 44 | 22.62 | 7.58% |

**80B TP=2 Aurora regression note:** broken 2026-04-12 → 2026-04-17 (OOM by ~60 MiB). Resolved 2026-04-18 by removing `import intel_extension_for_pytorch` (IPEX allocator overhead). See [restoration report](../experiments/agpt/aurora/20260418-80b-tp2-restored.md).

### Sunspot TP sweep (2N, torch 2.10)

| Variant | TP | DP | GBS | Memory | TPS | TFLOPS | MFU | Status |
|---------|----|----|-----|--------|-----|--------|-----|--------|
| 80B | 2 | 12 | 12 | 59.82 GiB (93%) | 85 | 46.31 | 15.53 | OK |
| 80B_alt | 2 | 12 | 12 | 59.82 GiB (93%) | 81 | 44.34 | 14.87 | OK |
| 80B_alt | 3 | 8 | 8 | 51.61 GiB (81%) | 23 | 12.40 | 4.16 | OK |
| 80B_alt | 6 | 4 | 4 | 39.54 GiB (62%) | 43 | 23.30 | 7.82 | OK |
| 80B_alt | 12 | 2 | 2 | — | — | — | — | OOM |
| 80B_wide | 2 | 12 | 12 | 60.75 GiB (95%) | 43 | 22.23 | 7.45 | Slow |
| 80B_wide | 3 | 8 | 8 | 52.75 GiB (82%) | 29 | 15.16 | 5.08 | OK |
| 80B_wide | 6 | 4 | 4 | 40.47 GiB (63%) | 51 | 26.35 | 8.84 | OK |
| 80B_wide | 12 | 2 | 2 | — | — | — | — | OOM |
| 80B_deep | 2 | 12 | 12 | 58.84 GiB (92%) | 83 | 45.58 | 15.29 | OK |
| 80B_deep_alt | 2 | 12 | 12 | 58.95 GiB (92%) | 77 | 41.87 | 14.04 | OK |
| 80B_deep_alt | 3 | 8 | 8 | 47.43 GiB (74%) | 23 | 12.56 | 4.21 | OK |
| 80B_deep_alt | 6 | 4 | 4 | 41.63 GiB (65%) | 41 | 22.43 | 7.52 | OK |
| 80B_deep_alt | 12 | 2 | 2 | 35.48 GiB (55%) | 28 | 15.40 | 5.16 | OK |

**Best config:** 80B TP=2 — 85–89 TPS, ~16% MFU. TP=3/6/12 all
drop sharply; only useful as memory-relief paths for the wider variants.
Cross-reference: [80B throughput leaderboard](../experiments/agpt/aurora/80b-throughput-leaderboard.md).

### 80B raw results dirs

- Sunspot weak scaling: `outputs/scaling_study/20260412_091635/`
- Sunspot TP sweep: `/tegu/datascience/foremans/projects/torchtitan/outputs/benchmarks/80b_20260330_065453/`
- Aurora 2N variants sweep: `outputs/benchmarks/20260404_212401/`

---

## MoE

### Sunspot weak scaling, torch 2.10

`no-compile, AC=none, FSDP only (TP=1), seq_len=4096, LBS=1`

#### moe_2b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 7,121 ± 77 | 85,458 ± 924 | 11.9% | 15.65 GiB (24%) | 100.0% |
| 2 | 24 | 6,970 ± 7 | 167,280 ± 169 | 11.7% | 15.34 GiB (24%) | 97.9% |
| 4 | 48 | 6,784 ± 76 | 325,632 ± 3,665 | 11.4% | 15.92 GiB (25%) | 95.3% |
| 8 | 96 | 6,392 ± 45 | 613,680 ± 4,412 | 10.7% | 16.84 GiB (26%) | 89.8% |
| 16 | 192 | 4,959 ± 49 | 952,128 ± 9,503 | 8.3% | 19.09 GiB (30%) | 69.6% |
| 32 | 384 | 5,170 ± 12 | 1,985,472 ± 4,615 | 8.7% | 26.27 GiB (41%) | 72.6% |
| 64 | 768 | 3,376 | 2,592,768 | 5.7% | 37.54 GiB (59%) | 47.4% |

#### moe_7b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 1,841 ± 13 | 22,098 ± 161 | 8.4% | 33.47 GiB (52%) | 100.0% |
| 2 | 24 | 1,197 ± 18 | 28,728 ± 441 | 5.4% | 33.31 GiB (52%) | 65.0% |
| 4 | 48 | 1,417 ± 28 | 68,016 ± 1,357 | 6.4% | 31.09 GiB (49%) | 76.9% |
| 8 | 96 | 1,111 ± 0 | 106,704 ± 67 | 5.0% | 35.87 GiB (56%) | 60.4% |
| 16 | 192 | 421 ± 0 | 80,928 ± 135 | 1.9% | 40.78 GiB (64%) | 22.9% |
| 32 | 384 | OOM | — | — | — | — |
| 64 | 768 | OOM | — | — | — | — |

MoE scaling efficiency degrades sharply past 8 nodes (all-to-all communication buffers). `moe_7b` OOMs at 32+ nodes. Open work tracked in [TODO #3](../TODO.md#3-moe-throughput-optimization).

### Aurora torch 2.13

| Nodes | TPS/GPU | Status | Notes |
|-------|---------|--------|-------|
| 64 | — | NO_OUTPUT (8528805) | likely upstream `edp_mesh=None` regression, see CLAUDE.md "MoE SIGABRT" |
| 128 | — | CRASH (8528834) | same as above |

Needs separate diagnosis ([task #144]).

---

## Aurora queue mapping (for resubmits)

| N | Queue | Walltime |
|---|-------|----------|
| 1–2 | debug | 1h |
| 3–256 | debug-scaling | 1h |
| 256–1,024 | prod → small | 12h |
| 1,025–1,919 | prod → medium | 12h |
| 1,920+ | prod → large | 12h |

## Reproducing

```bash
# Submit single-N scaling sweep on Aurora
qsub -A AuroraGPT -q debug-scaling -l walltime=01:00:00 -l select=<N> \
    -l filesystems=home:flare -N 2b-scale-n<N> -k doe -j oe \
    -v SCALING_GROUP=light,SCALING_DATASET=eliplutchok/fineweb-small-sample \
    torchtitan/experiments/ezpz/scripts/run_scaling_study_aurora.sh

# Aggregate results across runs
python3 torchtitan/experiments/ezpz/utils/aggregate_scaling.py \
    outputs/scaling_study_aurora/<TS1> outputs/scaling_study_aurora/<TS2> ...
```

## See Also

- [Per-run experiment reports](../experiments/agpt/) and [MoE per-run](../experiments/moe/)
- [80B throughput leaderboard](../experiments/agpt/aurora/80b-throughput-leaderboard.md)
- [80B TP=2 Aurora restoration](../experiments/agpt/aurora/20260418-80b-tp2-restored.md)
- [Production training: 2B](../production/agpt/2b/), [20B](../production/agpt/20b/), [80B](../production/agpt/80b/)
- [MoE configs](../configs/moe.md), [MoE throughput TODO](../TODO.md#3-moe-throughput-optimization)
- [yeet-env scaling](yeet_env/) — venv broadcast wall-clock by N
- [Known issues](../guides/known-issues.md), [1024N init crash memory](memory/project_1024n_init_crash.md)
