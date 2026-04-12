# MoE (Mixture of Experts) Benchmarks

MoE training benchmarks using DeepSeek-style MLA + MoE architecture across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-002800-smoke-n2.md) | 500M, 2B, 4B, 7B, 10B_2B_sdpa | 2 | All pass; AC incompatible with 7B+ routing |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-033047-smoke-n2.md) | debugmodel | 2 | OK; 25.8k TPS, 5.11% MFU |

### Sunspot

*No reports yet.*

### Polaris

*No reports yet.*
