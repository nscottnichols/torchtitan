# Production Training Runs — Aurora

> **Living document** — updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-04-26

## Scaling Performance

See [scaling-performance.md](scaling-performance.md) for the detailed
experiment log from Apr 18-21 (compile scaling, 80B at 4-512N, interactive
workflow validation).

## Overview

Full-scale production training of AuroraGPT models on the
[olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) dataset
(4.67T tokens) across Aurora compute nodes.

## Loss Curves

![Production Training Loss](../experiments/agpt/aurora/figures/production_training_loss.png)

## Runs

### Dense (agpt)

| Run | Model | Nodes | Optimizer | LR | Compile | Steps Done | Loss | Status |
|-----|-------|-------|-----------|------|---------|------------|------|--------|
| [2B-256N](agpt/2b/) | 2B | 256 | SophiaG | 2.28e-5 | on | 12435+ | 5.76 | **Running** |
| [2B-512N](agpt/2b/) | 2B | 512 | SophiaG | 2.28e-5 | off | 0 | — | Segfault |
| [20B-256N](agpt/20b/) | 20B | 256 | SophiaG | 2.28e-5 | on | 1739+ | 5.17 | **Running** |
| [20B-512N](agpt/20b/) | 20B | 512 | SophiaG | 2.28e-5 | on | 458 | 7.09 | Segfault |
| [80B-256N](agpt/80b/) | 80B | 256 | AdamW | 1.1e-5 | on | 777 | NaN | NaN@138 (killed) |
| [80B-256N](agpt/80b/) | 80B | 256 | AdamW | 1e-6 | on | 51 | 12.91 | Stable (walltime) |
| [80B-512N](agpt/80b/) | 80B | 512 | AdamW | 1.1e-5 | off | 495 | NaN | NaN@15 (killed) |

### MoE

| Run | Model | Nodes | Status |
|-----|-------|-------|--------|
| [10B_2B EP=12](moe/10b_2b_sdpa_ep/) | 10B_2B_sdpa | TBD | Planned |

## Known Issues

1. **torch.compile OOM at 512N** — 2B OOMs on GPU, 80B OOMs on CPU. Use
   `--compile.no-enable` for 512N jobs.
2. **SophiaG/Muon broken at 80B** — bf16 overflow in Hessian/Newton-Schulz.
   Use AdamW only.
3. **80B AdamW LR=1.1e-5 → NaN** — loss diverges at step 138 (256N) and
   step 15 (512N). LR from LR finder (2-node, GBS=12) may be too high for
   production GBS (1536-3072). Need to re-run LR finder at production GBS
   or reduce LR to ~1e-6.
4. **Transient segfaults** — single bad nodes crash the whole job. Retry
   usually works. Both 512N jobs (2B and 20B) segfaulted on their first
   continuation attempt.
5. **Compile time at 256N** — ~7-15 min depending on model size. Eats into
   the 12h walltime.
