---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Sections
> ordered by importance (live → reference → outbound). Within each
> section, rows are sorted newest first by last-commit date.

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Modified | Notes |
|------|---------:|-------|
| [Production Index](./production/README.md) | 2026-05-05 | Top-level snapshot of every active trajectory |
| [Dense (agpt) Production](./production/agpt/README.md) | 2026-05-05 | 2B / 20B / 80B chains, v1-vs-v2 overlays |
| [agpt 20B](./production/agpt/20b/README.md) | 2026-05-05 | All 20B trajectories + v1-vs-v2 overlay |
| [20B 256N](./production/agpt/20b/n256/README.md) | 2026-05-05 | 8463659 NODE_FAIL → 8470102/8470103 256N continuation chain |
| [2B 256N](./production/agpt/2b/n256/README.md) | 2026-05-05 | 8470100/8470101 256N continuation chain (resume from step-2000) |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | 2026-05-04 | First attempt (8463183) crashed at startup |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | 2026-05-04 | First attempt (8463182) crashed at startup |
| [agpt 2B](./production/agpt/2b/README.md) | 2026-05-03 | All 2B trajectories + v1-vs-v2 overlay |
| [20B 512N](./production/agpt/20b/n512/README.md) | 2026-05-03 | **Canonical 20B chain** (step 863, loss 3.46) |
| [2B 512N](./production/agpt/2b/n512/README.md) | 2026-05-03 | **Canonical 2B chain** (step 5,073, loss 2.97) |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | 2026-05-03 | Pre-torchtitan Megatron-DeepSpeed reference baseline |
| [Production Scaling Report](./production/scaling-performance.md) | 2026-04-26 | Apr 18-21 experiments (historical) |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Modified | Notes |
|------|---------:|-------|
| [agpt 20B evals](./evals/agpt/20b/README.md) | 2026-05-05 | v1 vs v2, steps 100-800 (ARC-Easy 0.27 → 0.44) |
| [agpt 2B evals](./evals/agpt/2b/README.md) | 2026-05-03 | v1 vs v2 + 256N-vs-512N per-batch comparison |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | 2026-05-03 | Pre-torchtitan reference scores |
| [Eval Index](./evals/README.md) | 2026-04-30 | Top-level eval landing page |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Modified | Notes |
|------|---------:|-------|
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | 2026-05-03 | Root cause of v1 → v2 restart; `dtype=float32` is now default |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | 2026-05-03 | Upstream regression since 2026-04-27. Fix filed as pytorch/torchtitan#3204 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | 2026-04-29 | Catch-all for live workarounds |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | 2026-04-26 | SDPA, FlexAttention, Triton on Intel Max 1550 |

## Day-by-day Work

| Page | Modified | Notes |
|------|---------:|-------|
| [Development Journal](./journal.md) | 2026-05-05 | Session-by-session log of what happened, with findings and incidents |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | 2026-05-04 | Recurring agendas + action items |
| [Meeting Notes Index](./meeting-notes/README.md) | 2026-05-04 | Top-level meeting index |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md) | 2026-05-04 | 2-week retrospective |
| [Periodic Summaries Index](./summaries/README.md) | 2026-05-04 | Index of 2-week / monthly retros |

## Setup & Reference

| Page | Modified | Notes |
|------|---------:|-------|
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | 2026-05-04 | torch 2.13 venv setup + at-scale yeet (8N → 4096N) |
| [Reference Baselines](./baselines/README.md) | 2026-04-29 | Training curves and benchmarks |
| [Dense Model Configs](./configs/dense.md) | 2026-04-26 | 2B / 20B / 50B / 80B |
| [MoE Variants](./configs/moe.md) | 2026-04-26 | 500M-10B |

## Scaling Studies

| Page | Modified | Notes |
|------|---------:|-------|
| [Scaling Index](./scaling/README.md) | 2026-04-26 | Top-level scaling landing page |
| [agpt 2B scaling](./scaling/agpt-2b.md) | 2026-04-26 | Per-N TPS / MFU |
| [agpt 20B scaling](./scaling/agpt-20b.md) | 2026-04-26 | Per-N TPS / MFU |
| [agpt 80B scaling](./scaling/agpt-80b.md) | 2026-04-26 | Per-N TPS / MFU |
| [MoE scaling](./scaling/moe.md) | 2026-04-26 | Per-N TPS / MFU |
| [Per-run Experiment Reports](./experiments/README.md) | 2026-04-12 | Raw smoke tests, LR-finder sweeps, benchmark logs |

## Sandboxes / Side-channels

| Page | Modified | Notes |
|------|---------:|-------|
| [Optimizer Speedrun Competitions](./competitions/README.md) | 2026-04-28 | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) |
| [RL (GRPO) Experiment](./rl/README.md) | 2026-04-26 | TRL-based GRPO on XPU (experimental) |

## Outbound (upstream)

| Page | Modified | Notes |
|------|---------:|-------|
| [Upstream Sync Log](./upstream-sync.md) | 2026-05-05 | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | 2026-05-03 | TP loss-reporting bug repro + patch |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | 2026-05-01 | Repro for upstream filing |

## Planning

| Page | Modified | Notes |
|------|---------:|-------|
| [TODO](./TODO.md) | 2026-05-05 | Open work items |
