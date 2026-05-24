---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Sections
> ordered by importance (live → reference → outbound). Within each
> section, rows are sorted newest first by last-commit date.
>
> **Looking for something specific?** See [`TREE.md`](./TREE.md)
> for a single-page annotated tree of every directory and file
> under `docs/`, with descriptions of what goes where.

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-05-23 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-05-23 |
| [2B 256N](./production/agpt/2b/n256/README.md) | **🚀 only trajectory making consistent on-disk progress.** step **25,500** persisted (1.28T tokens, 27.4% of target). +12,500 steps in 30 hours. | 2026-05-23 |
| [2B 512N](./production/agpt/2b/n512/README.md) | Step **13,200** persisted. Hits step-13400 wall every dispatch (async-save cluster cascade). Same root cause as 20B 512N. | 2026-05-23 |
| [20B 512N](./production/agpt/20b/n512/README.md) | **Canonical 20B chain** — step **800** persisted (81B tokens). Three weeks of dispatches lost to async-save regression; sync-mode resubmits live-testing 2026-05-23 evening. | 2026-05-23 |
| [20B 256N](./production/agpt/20b/n256/README.md) | Step **300** persisted. Same async wall as 512N; sync-mode (`8505255`) Q to test fix. | 2026-05-23 |
| [agpt 80B](./production/agpt/80b/README.md) | 11 attempts since 2026-05-11; zero persisted. Wrapper detects failures correctly, but Aurora bad-node prevalence at init is too high for current spare count. | 2026-05-23 |
| [80B 256N](./production/agpt/80b/n512/README.md) | Latest config: AdamW LR=1e-6, TP=2, AC=full, compile=OFF, fp32-master. Most recent attempt `8505221` used 20 spares + 5 retries, still died in init. | 2026-05-23 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-05-04 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-05-04 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-05-23 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-05-03 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-04-26 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). 8505205 R adding steps 14K-25K. | 2026-05-23 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator at matched steps | 2026-05-22 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-05-03 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-04-30 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **v2 in production 2026-05-23.** Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Now also handles silent hangs via `ezpz launch --timeout=1800` watchdog. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-05-23 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-05-23 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-05-03 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | Upstream regression since 2026-04-27. Fix filed as pytorch/torchtitan#3204 | 2026-05-03 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-04-26 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-05-05 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-05-05 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md) | 2-week retrospective | 2026-05-04 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-05-04 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-05-04 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-04-29 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-04-26 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-04-26 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-04-26 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-04-26 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-04-26 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-04-26 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-04-26 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-04-12 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-04-28 |
| [RL (GRPO) Experiment](./rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-04-26 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe | 2026-05-05 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | TP loss-reporting bug repro + patch | 2026-05-03 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items | 2026-05-05 |
