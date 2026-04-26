---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

- Dense Models: [AuroraGPT-{2,20}B](./configs/dense.md)
- MoE Variants: [AuroraGPT-10B-2B](./configs/moe.md)
- [Known Issues and Operational Notes](./guides/known-issues.md)
- [XPU Attention Issues](./guides/xpu-attention-issues.md) (SDPA, FlexAttention, Triton)
- [Upstream Sync Log](./upstream-sync.md)
- [Experiment Reports](./experiments/) (benchmarks, smoke tests, LR finder, production runs)
- [Scaling Results](./scaling/) — consolidated scaling data across all models and machines
- [Scaling & Production Runs Report](./production/scaling-performance.md) — Apr 18–21 experiments
- [Production Training Runs](./production/) — live status of ongoing training
- [RL (GRPO) Experiment](./rl/README.md) — TRL-based GRPO on XPU (experimental)
- [TODO](./TODO.md)
