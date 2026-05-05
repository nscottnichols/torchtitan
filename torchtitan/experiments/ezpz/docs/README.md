---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Most recently
> changed pages are at the top of each section. Dates are last-commit
> dates, not last-edited-on-disk.

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

- [Production Index](./production/README.md) — top-level snapshot of
  every active trajectory _(modified: 2026-05-05)_
- [Dense (agpt) Production](./production/agpt/README.md) — 2B / 20B /
  80B chains, v1-vs-v2 overlays _(modified: 2026-05-05)_
- [agpt 20B](./production/agpt/20b/README.md) — 256N + 512N + 1024N
  trajectories, v1-vs-v2 overlay _(modified: 2026-05-05)_
  - [20B 256N](./production/agpt/20b/n256/README.md) _(modified: 2026-05-05)_
  - [20B 512N](./production/agpt/20b/n512/README.md) — **canonical chain** _(modified: 2026-05-03)_
  - [20B 1024N](./production/agpt/20b/n1024/README.md) — first attempt crashed at startup _(modified: 2026-05-04)_
- [agpt 2B](./production/agpt/2b/README.md) _(modified: 2026-05-03)_
  - [2B 256N](./production/agpt/2b/n256/README.md) _(modified: 2026-05-05)_
  - [2B 512N](./production/agpt/2b/n512/README.md) — **canonical chain** _(modified: 2026-05-03)_
  - [2B 1024N](./production/agpt/2b/n1024/README.md) — first attempt crashed at startup _(modified: 2026-05-04)_
- [agpt 2B (MDS)](./production/agpt/2b-mds/README.md) — pre-torchtitan
  Megatron-DeepSpeed reference baseline _(modified: 2026-05-03)_
- [Production Scaling Report (Apr 18-21)](./production/scaling-performance.md) _(modified: 2026-04-26)_

## Evaluation (lm-eval results)

The smoking-gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

- [Eval Index](./evals/README.md) _(modified: 2026-04-30)_
- [agpt 20B evals](./evals/agpt/20b/README.md) — v1 vs v2, steps
  100-800 (ARC-Easy 0.27 → 0.44) _(modified: 2026-05-05)_
- [agpt 2B evals](./evals/agpt/2b/README.md) — v1 vs v2 + 256N-vs-512N
  per-batch comparison _(modified: 2026-05-03)_
- [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) — pre-torchtitan
  reference scores _(modified: 2026-05-03)_

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

- [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md)
  — root cause of the v1 → v2 restart; `dtype=float32` is now the
  default _(modified: 2026-05-03)_
- [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md)
  — upstream regression since 2026-04-27, local workaround in
  `EzpzValidator` + `trainer.py`, fix filed as
  pytorch/torchtitan#3204 _(modified: 2026-05-03)_
- [Known Issues / Operational Notes](./guides/known-issues.md)
  _(modified: 2026-04-29)_
- [XPU Attention Issues](./guides/xpu-attention-issues.md) — SDPA,
  FlexAttention, Triton on Intel Max 1550 _(modified: 2026-04-26)_

## Day-by-day Work

- [Development Journal](./journal.md) — session-by-session log of
  what happened, with findings and incidents _(modified: 2026-05-05)_
- [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) — recurring
  agendas + action items _(modified: 2026-05-04)_
  - [Meeting Notes Index](./meeting-notes/README.md) _(modified: 2026-05-04)_
- [Periodic Summaries](./summaries/README.md) — 2-week / monthly retrospectives
  _(modified: 2026-05-04)_
  - [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md)
    _(modified: 2026-05-04)_

## Setup & Reference

- [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md)
  — torch 2.13 venv setup + at-scale yeet (8N → 4096N) _(modified: 2026-05-04)_
- [Dense Model Configs (2B, 20B, 50B, 80B)](./configs/dense.md)
  _(modified: 2026-04-26)_
- [MoE Variants (500M-10B)](./configs/moe.md) _(modified: 2026-04-26)_
- [Reference Baselines (training curves and benchmarks)](./baselines/README.md)
  _(modified: 2026-04-29)_

## Scaling Studies

- [Scaling Index](./scaling/README.md) _(modified: 2026-04-26)_
- [agpt 2B scaling](./scaling/agpt-2b.md) _(modified: 2026-04-26)_
- [agpt 20B scaling](./scaling/agpt-20b.md) _(modified: 2026-04-26)_
- [agpt 80B scaling](./scaling/agpt-80b.md) _(modified: 2026-04-26)_
- [MoE scaling](./scaling/moe.md) _(modified: 2026-04-26)_
- [Per-run Experiment Reports](./experiments/README.md) — raw smoke
  tests, LR-finder sweeps, benchmark logs _(modified: 2026-04-12)_

## Sandboxes / Side-channels

- [Optimizer Speedrun Competitions](./competitions/README.md)
  ([W&B](https://api.wandb.ai/links/aurora_gpt/hda3milo))
  _(modified: 2026-04-28)_
- [RL (GRPO) Experiment](./rl/README.md) — TRL-based GRPO on XPU
  (experimental) _(modified: 2026-04-26)_

## Outbound (upstream)

- [Upstream Sync Log](./upstream-sync.md) — what we pulled from
  pytorch/torchtitan and what we replayed onto agpt/moe
  _(modified: 2026-05-05)_
- [Upstream Issue Drafts](./upstream-issues/) — repros + patches we're
  filing back
  - [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md)
    _(modified: 2026-05-03)_
  - [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md)
    _(modified: 2026-05-01)_

## Planning

- [TODO](./TODO.md) _(modified: 2026-05-05)_
