# MoE (Mixture of Experts) Benchmarks

MoE training benchmarks using DeepSeek-style MLA + MoE architecture across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-002800-smoke-n2.md) | 500M, 2B, 4B, 7B, 10B_2B_sdpa | 2 | All pass; AC incompatible with 7B+ routing |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-033047-smoke-n2.md) | debugmodel | 2 | OK; 25.8k TPS, 5.11% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035648-smoke-n2.md) | debugmodel, 2b | 2 | OK; 2b at 11.4% MFU, 34 TFLOPS |
| 2026-04-13 | [Throughput (n2)](aurora/20260413-143400-throughput-n2.md) | 500m-10b sweep | 2 | 2b best MFU (11.1%); compile hurts MoE |

### Sunspot

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Scaling Study](../../scaling-study.md) | moe_2b, moe_7b at 1-64N | 1-64 | 2b 47% efficiency at 64N; 7b OOM at 32N+ |
| 2026-04-13 | [Benchmark (n2)](sunspot/20260413-benchmark-n2.md) | All 7 MoE configs | 2 | 9.7% MFU (debugmodel); compile hurts MoE |
| 2026-04-15 | [Full Benchmark (n2)](../agpt/sunspot/20260415-benchmark-n2.md) | All 7 MoE configs | 2 | 6/7 pass; moe_10b_2b Inductor crash |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-185037-smoke-n2.md) | debugmodel, 500M, 2b, 4b, 7b, 10b_2b_sdpa | 2 | All pass; 2b best at 22.5% MFU; 7B+ memory-bound |
