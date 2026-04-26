# Scaling Results

Consolidated scaling data across all models, machines, and PyTorch versions.

## Quick Reference

| Model | Best TPS/GPU (2N) | Peak MFU | Max Nodes | Details |
|-------|-------------------|----------|-----------|---------|
| agpt_2b | 5,529 | 24.2% | 64 | [agpt-2b.md](agpt-2b.md) |
| agpt_20b | 440 (torch 2.13) | 22.0% | 4096 (in progress) | [agpt-20b.md](agpt-20b.md) |
| agpt_80b | 89 | 16.2% | 32 | [agpt-80b.md](agpt-80b.md) |
| moe_2b | 6,970 | 11.9% | 64 | [moe.md](moe.md) |
| moe_7b | 1,197 | 8.4% | 8 | [moe.md](moe.md) |

## Per-Model Pages

- [agpt-2b.md](agpt-2b.md) — 2B dense model scaling (Sunspot, 1–64 nodes)
- [agpt-20b.md](agpt-20b.md) — 20B dense model scaling (Sunspot + Aurora torch 2.13)
- [agpt-80b.md](agpt-80b.md) — 80B dense model variants and TP sweeps
- [moe.md](moe.md) — MoE model scaling (2B and 7B)

## Experiment Reports

Raw per-run benchmark logs are in [experiments/](../experiments/).
