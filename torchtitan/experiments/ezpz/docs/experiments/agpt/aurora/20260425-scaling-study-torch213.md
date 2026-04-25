# Scaling Study — agpt_20b on Aurora (PyTorch 2.13)

> Sam Foreman
> 2026-04-25
> **Living document** — updated as jobs complete.

## Overview

Weak scaling study for agpt_20b across 2–4096 Aurora nodes using the new
PyTorch 2.13 venv (`.venv/`). Testing scaling behavior with SophiaG
optimizer (LR=2.28e-5) on olmo-mix-1124 data.

## Configuration

| Field | Value |
|-------|-------|
| Model | agpt_20b (20.7B params) |
| Script | `torchtitan/experiments/ezpz/scripts/train_agpt_20b.sh` |
| PyTorch | 2.13.0.dev (from `.venv/`) |
| Optimizer | SophiaG, LR=2.28e-5 |
| Seq len | 8192 |
| Steps | 20 (per node count) |
| Dataset | olmo-mix-1124 (blendcorpus) |

## Results

| Nodes | GPUs | GBS | Job ID | TPS/GPU | TFLOPS | MFU | Memory | Loss (final) | Compile | Status |
|-------|------|-----|--------|---------|--------|-----|--------|--------------|---------|--------|
| 2 | 24 | 24 | 8451183 | 440 | 65.3 | 22.0% | 32.07 GiB (50%) | 9.43 | on | ✅ Complete |
| 4 | 48 | — | — | — | — | — | — | — | — | Pending |
| 8 | 96 | — | — | — | — | — | — | — | — | Pending |
| 16 | 192 | — | — | — | — | — | — | — | — | Pending |
| 32 | 384 | — | — | — | — | — | — | — | — | Pending |
| 64 | 768 | — | — | — | — | — | — | — | — | Pending |
| 128 | 1536 | — | — | — | — | — | — | — | — | Pending |
| 256 | 3072 | — | — | — | — | — | — | — | — | Pending |
| 512 | 6144 | — | — | — | — | — | — | — | — | Pending |
| 1024 | 12288 | — | — | — | — | — | — | — | — | Pending |
| 2048 | 24576 | — | — | — | — | — | — | — | — | Pending |
| 4096 | 49152 | — | — | — | — | — | — | — | — | Pending |

## Scaling Efficiency

_To be filled after results are collected._

| Nodes | Per-GPU TPS | Total TPS | Efficiency (vs 2N) |
|-------|-------------|-----------|-------------------|
| 2 | — | — | 100% (baseline) |
| ... | | | |

## Queue Mapping

| Node count | Queue | Walltime | Notes |
|------------|-------|----------|-------|
| 2 | debug | 1h | Quick test |
| 4–256 | debug-scaling | 1h | |
| 256–1024 | prod → small | 6h | |
| 1025–1919 | prod → medium | 6h | |
| 1920+ | prod → large | 6h | |

## Job Submission Plan

1. **Test phase**: 2-node job to verify script works with torch 2.13
2. **Small scale**: Submit 4–128 via `debug-scaling` (parallel jobs)
3. **Large scale**: Submit 256–4096 via `prod` (sequential due to queue limits)

## Notes

- Uses `ezpz yeet-env` to copy venv to `/tmp` for faster imports
- Script sources `.venv/` directly (PyTorch 2.13), not the frameworks module
- `EXTRA_ARGS` env var can be used to pass additional flags
