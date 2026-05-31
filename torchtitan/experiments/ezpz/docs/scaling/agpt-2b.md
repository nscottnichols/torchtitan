# AuroraGPT-2B Scaling

## Aurora Weak Scaling (torch 2.13 + `ezpz yeet-env` tarball, 2026-05-29)

> Single sweep across 4 → 256 nodes with `ezpz yeet-env` tarball
> broadcast (the production env-setup pattern). All entries are 20-step
> bench runs with `BENCH_STEPS=20 SCALING_GROUP=light LBS=2`.
> **Config:** FSDP-only (TP=1), `compile=on`, AC=full, seq_len=8192,
> LBS=2 (GBS = N × 12 × 2 = production-matched).

| Nodes | GPUs | GBS    | TPS/GPU | Total TPS  | MFU    | Efficiency vs 4N |
|-------|------|--------|---------|------------|--------|------------------|
| 4     | 48   | 96     | 7,344   | 352,512    | 27.55% | 100.0%           |
| 8     | 96   | 192    | 7,291   | 699,936    | 27.36% | 99.3%            |
| 16    | 192  | 384    | 6,803   | 1,306,176  | 25.53% | 92.6%            |
| 32    | 384  | 768    | 6,984   | 2,681,856  | 26.20% | 95.1%            |
| 64    | 768  | 1,536  | CRASH   | —          | —      | —                |
| 128   | 1,536| 3,072  | CRASH   | —          | —      | —                |
| 256   | 3,072| 6,144  | 5,002   | 15,366,144 | 18.77% | 68.1%            |
| 512   | 6,144| 12,288 | NO_OUTPUT| —         | —      | —                |
| 1,024 | 12,288| 24,576| NO_OUTPUT| —         | —      | —                |
| 2,048 | 24,576| 49,152| NO_OUTPUT| —         | —      | —                |
| 4,096 | 49,152| 98,304| NO_OUTPUT| —         | —      | —                |

**Headline:** With the production `ezpz yeet-env` setup applied to the
scaling sweep, small-N (4–32) lands cleanly at **~27% MFU** (matching
the torch-2.13 Sunspot target and well above the torch-2.10 Sunspot
baseline below). At **N=256** with one successful sweep, throughput
dropped to 18.77% MFU (≈68% efficiency vs 4N).

**Known failure modes blocking the missing N:**

- `n=64` and `n=128` retries all CRASH with `from sh import qstat`
  ImportError. Root cause: when the scaling job is launched via
  `qsub -- /bin/bash -c '...'` (non-login subshell), `module` is
  undefined → `module load oneapi/release/2025.3.1` silently fails
  → oneAPI MPI binaries (incl. `mpiexec`, `qstat`) are not on PATH
  → `ezpz launch` → `get_active_jobid()` → `from sh import qstat`
  → ImportError. Fixed across multiple commits ending at `2b0073170`
  (source `/etc/bash.bashrc.local` to define `module` + `MODULEPATH`
  the same way `bash --login` does). Pending validation on a fresh
  submission.
- `n=512`+ NO_OUTPUT: distinct from sh.qstat. At 6,144 ranks the
  failure mode is XCCL communicator init segfault (or
  `set_determinism std::bad_alloc` at 12,288 + 49,152 ranks); see
  [`memory/project_1024n_init_crash.md`](.). The yeet helped at
  N=256 but doesn't fully fix the larger-N init scaling wall.

**Raw sweep dirs aggregated** (for `aggregate_scaling.py`):
- `outputs/scaling_2b_aurora/20260529_081759/` (n=4/8/16/32 OK, n=64
  most-recent CRASH; n=128/256 retries CRASH from sh.qstat bug)
- `outputs/scaling_2b_aurora/20260529_075349/` (n=256 OK, the one
  clean N=256 result this week)

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes, April 2026)

Historical reference — pre-`yeet-env` torch-2.10 baseline.

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 6,457+/-38 | 77,490+/-466 | 24.2% | 47.29 GiB (74%) | 100.0% |
| 2 | 24 | 5,529+/-71 | 132,708+/-1714 | 20.8% | 46.80 GiB (73%) | 85.6% |
| 4 | 48 | 5,580+/-81 | 267,864+/-3903 | 20.9% | 46.66 GiB (73%) | 86.4% |
| 8 | 96 | 5,384+/-173 | 516,912+/-16631 | 20.2% | 46.58 GiB (73%) | 83.4% |
| 16 | 192 | 5,032+/-44 | 966,240+/-8553 | 18.9% | 46.42 GiB (73%) | 77.9% |
| 32 | 384 | 5,091+/-118 | 1,954,944+/-45616 | 19.1% | 46.41 GiB (73%) | 78.8% |
| 64 | 768 | 4,783 | 3,673,344 | 17.9% | 46.38 GiB (72%) | 74.1% |

**Config:** FSDP only (TP=1), compile=on, AC=full, seq_len=4096, LBS=1

**Results directory:** `outputs/scaling_study/20260412_091635/`

## See Also

- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/2b/) — live training status
