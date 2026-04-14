# agpt (Dense AuroraGPT) Benchmarks

Dense transformer training benchmarks across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../benchmark-80B.md) | 80B, 80B_alt, 80B_wide, 80B_deep | 2 | TP=2 best at 15.5% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035148-smoke-n2.md) | debugmodel, 2b | 2 | 2b at 20% MFU, 59.6 TFLOPS |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/aurora/20260412-144400-lr-finder-n2.md) | 2b, 20b x {AdamW,Muon,SophiaG} | 2 | AdamW most tolerant; SophiaG needs 10x lower LR |
| 2026-04-12 | [80B Throughput (n2)](aurora/20260412-193100-throughput-80b-n2.md) | 80B_alt TP={3,6} x compile | 2 | TP=6+compile best: 45 TPS, 8.23% MFU |
| 2026-04-13 | [80B Leaderboard](aurora/80b-throughput-leaderboard.md) | 80B_wide/alt/deep x TP={2-12} | 2 | 80B_wide TP=4 compile: 68 TPS, 11.79% MFU |
| 2026-04-13 | [20B Throughput (n2)](aurora/20260413-143800-throughput-20b-n2.md) | 20B TP={1,2,4} x compile | 2 | TP=1 compile: 357 TPS, 17.82% MFU |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-160749-smoke-n2.md) | debugmodel, 2b, 7b, 8b, 20b, 50b, 80b | 2 | 7b 8.8% MFU; 50b/80b OOM; 8b vocab mismatch |

### Sunspot

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../benchmark-80B.md) | 80B variants x TP={2,3,6,12} | 2 | TP=2 best: 85 TPS, 15.5% MFU |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/sunspot/20260412-lr-finder-n2.md) | 2B, 20B x {AdamW,Muon,SophiaG} | 2 | All 6 sweeps; 20B SophiaG suggested LR=1.5e-5 |
| 2026-04-12 | [Scaling Study](../../scaling-study.md) | 2b, 20b, 80b at 1-64N | 1-64 | 20b 87% efficiency at 64N; 80b OK at 4-32N |
| 2026-04-13 | [Benchmark (n2)](sunspot/20260413-benchmark-n2.md) | All 11 agpt configs | 2 | 80b_deep best 80B variant: 83 TPS, 15.2% MFU |
