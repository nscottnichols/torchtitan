---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

## Model Configs

- [Dense Models (2B, 20B)](./configs/dense.md)
- [MoE Variants (500M–10B)](./configs/moe.md)

## Scaling & Benchmarks

- [Scaling Results](./scaling/) — consolidated per-model scaling data
- [Experiment Reports](./experiments/) — raw per-run benchmark logs, smoke tests, LR finder

## Production Training

- [Production Runs](./production/) — live status of ongoing training
- [Scaling & Production Report](./production/scaling-performance.md) — Apr 18–21 experiments

## Guides & Reference

- [Known Issues and Operational Notes](./guides/known-issues.md)
- [XPU Attention Issues](./guides/xpu-attention-issues.md) (SDPA, FlexAttention, Triton)
- [Running with Newer PyTorch](./guides/running-with-newer-pytorch.md)
- [Upstream Sync Log](./upstream-sync.md)

## Experiments

- [agpt_2b Loss Competition](./competition/README.md) — optimizer/schedule/architecture speedrun ([W&B](https://api.wandb.ai/links/aurora_gpt/hda3milo))
- [RL (GRPO) Experiment](./rl/README.md) — TRL-based GRPO on XPU (experimental)

## Planning

- [TODO](./TODO.md)
