# Production Training Runs — Aurora

> **Living document** — updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-04-23

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
| [2B-256N](agpt/2b/) | 2B | 256 | SophiaG | 2.28e-5 | on | 1431 | 6.13 | Continuing |
| [2B-512N](agpt/2b/) | 2B | 512 | SophiaG | 2.28e-5 | off | 0 | — | Resubmit queued |
| [20B-256N](agpt/20b/) | 20B | 256 | SophiaG | 2.28e-5 | on | 581 | 7.35 | Continuing |
| [20B-512N](agpt/20b/) | 20B | 512 | SophiaG | 2.28e-5 | on | 458 | 7.09 | Continuing |
| [80B-256N](agpt/80b/) | 80B | 256 | AdamW | 1.1e-5 | on | 0 | — | Retry queued |
| [80B-512N](agpt/80b/) | 80B | 512 | AdamW | 1.1e-5 | off | 0 | — | Resubmit queued |

### MoE

| Run | Model | Nodes | Status |
|-----|-------|-------|--------|
| [10B_2B EP=12](moe/10b_2b_sdpa_ep/) | 10B_2B_sdpa | TBD | Planned |

## Known Issues

1. **torch.compile OOM at 512N** — 2B OOMs on GPU, 80B OOMs on CPU. Use
   `--compile.no-enable` for 512N jobs.
2. **SophiaG/Muon broken at 80B** — bf16 overflow in Hessian/Newton-Schulz.
   Use AdamW with LR=1.1e-5.
3. **Transient segfaults** — single bad nodes crash the whole job. Retry
   usually works.
4. **Compile time at 256N** — ~7-15 min depending on model size. Eats into
   the 12h walltime.
