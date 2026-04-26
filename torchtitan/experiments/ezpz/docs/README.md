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
- [Scaling Study — PyTorch 2.13](./scaling-study-torch213.md) — weak scaling 2–4096 nodes (in progress)
- [Scaling & Production Runs Report](./production-training/scaling-performance.md) — Apr 18–21 experiments
- [Production Training Runs](./production-training/) — live status of ongoing training
- [RL (GRPO) Experiment](./rl/README.md) — TRL-based GRPO on XPU (experimental)
- [TODO](./TODO.md)
