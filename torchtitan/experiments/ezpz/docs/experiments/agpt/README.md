# agpt (Dense AuroraGPT) Benchmarks

Dense transformer training benchmarks across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../benchmark-results-80B.md) | 80B, 80B_alt, 80B_wide, 80B_deep | 2 | TP=2 best at 15.5% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035148-smoke-n2.md) | debugmodel, 2b | 2 | 2b at 20% MFU, 59.6 TFLOPS |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/aurora/20260412-144400-lr-finder-n2.md) | 2b, 20b x {AdamW,Muon,SophiaG} | 2 | AdamW most tolerant; SophiaG needs 10x lower LR |
| 2026-04-12 | [80B Throughput (n2)](aurora/20260412-193100-throughput-80b-n2.md) | 80B_alt TP={3,6} x compile | 2 | TP=6+compile best: 45 TPS, 8.23% MFU |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-160749-smoke-n2.md) | debugmodel, 2b | 2 | PASS: debugmodel 8% MFU, 2b 7.4% MFU |

### Sunspot

See [80B results](../../benchmark-results-80B.md) (Sunspot section).
