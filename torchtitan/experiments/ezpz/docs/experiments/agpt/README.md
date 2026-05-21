# agpt (Dense AuroraGPT) Benchmarks

Dense transformer training benchmarks across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../scaling/agpt-80b.md) | 80B, 80B_alt, 80B_wide, 80B_deep | 2 | TP=2 best at 15.5% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035148-smoke-n2.md) | debugmodel, 2b | 2 | 2b at 20% MFU, 59.6 TFLOPS |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/aurora/20260412-144400-lr-finder-n2.md) | 2b, 20b x {AdamW,Muon,SophiaG} | 2 | AdamW most tolerant; SophiaG needs 10x lower LR |
| 2026-04-12 | [80B Throughput (n2)](aurora/20260412-193100-throughput-80b-n2.md) | 80B_alt TP={3,6} x compile | 2 | TP=6+compile best: 45 TPS, 8.23% MFU |
| 2026-04-13 | [80B Leaderboard](aurora/80b-throughput-leaderboard.md) | 80B_wide/alt/deep x TP={2-12} | 2 | 80B_wide TP=4 compile: 68 TPS, 11.79% MFU |
| 2026-04-13 | [20B Throughput (n2)](aurora/20260413-143800-throughput-20b-n2.md) | 20B TP={1,2,4} x compile | 2 | TP=1 compile: 357 TPS, 17.82% MFU |
| 2026-04-14 | [20B Production (n512)](aurora/20260414-production-20b-n512.md) | 20B SophiaG LR=2.28e-5 | 512 | Verify: loss 12.92->10.50 in 146 steps |
| 2026-04-18 | [80B TP=2 Restored](aurora/20260418-80b-tp2-restored.md) | 80B TP=2 compile | 2 | 88 TPS, 16% MFU — regression fixed |
| 2026-04-18 | [Scaling & Production](../../production/scaling-performance.md) | 2B, 20B, 80B | 4-512 | 80B scales perfectly to 128N; compile wall at 512N |
| 2026-04-25 | [Scaling Study (torch 2.13)](../../scaling/agpt-20b.md) | 20B | 2-4096 | 440 TPS @ 2N (+23% vs torch 2.10); in progress |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-160749-smoke-n2.md) | debugmodel, 2b, 7b, 8b, 20b, 50b, 80b | 2 | 7b 8.8% MFU; 50b/80b OOM; 8b vocab mismatch |

### Sunspot

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../scaling/agpt-80b.md) | 80B variants x TP={2,3,6,12} | 2 | TP=2 best: 85 TPS, 15.5% MFU |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/sunspot/20260412-lr-finder-n2.md) | 2B, 20B x {AdamW,Muon,SophiaG} | 2 | All 6 sweeps; 20B SophiaG suggested LR=1.5e-5 |
| 2026-04-12 | [Scaling Study](../../scaling/) | 2b, 20b, 80b at 1-64N | 1-64 | 20b 87% efficiency at 64N; 80b OK at 4-32N |
| 2026-04-13 | [Benchmark (n2)](sunspot/20260413-benchmark-n2.md) | All 11 agpt configs | 2 | 80b_deep best 80B variant: 83 TPS, 15.2% MFU |
| 2026-04-15 | [Full Benchmark (n2)](sunspot/20260415-benchmark-n2.md) | All 18 configs (agpt+MoE) | 2 | 80B compile regression found; fix in e8cbb8ef |
| 2026-04-18 | [Torch 2.12 Benchmark (n2)](sunspot/20260418-torch212-benchmark-n2.md) | 8 configs (agpt+MoE) | 2 | 2b +11% TPS, -49% mem; 20b +29% TPS; 80b AC regression |
| 2026-04-21 | [LR Finder 80B + GAS (n2)](../lr-finder/agpt/sunspot/20260421-lr-finder-80b-n2.md) | 80B x 3 opts, 2B/20B GAS sweep | 2 | 80B AdamW LR=1.1e-5; Muon/SophiaG broken at 80B |
| 2026-04-27 | [10B Optimizer Sweep (n8)](../../competitions/agpt2b-n8-10BT/) | 2B x {AdamW, AdamW+QKNorm, Mano, Mano+QKNorm} | 8 | AdamW wins at GBS=384; loss 2.711 |
| 2026-05-20 | [Post-resync smoke (n2)](sunspot/20260520-smoke-n2-postresync.md) | debugmodel, 2b (LBS=1, LBS=2) | 2 | PR #3159 replay verified: 2b LBS=2 matches Apr 25 baseline (7,224 TPS / 27.1% MFU) |
| 2026-05-20 | [PR #3386 merge follow-up smoke (n2)](sunspot/20260520-smoke-n2-pr3386-merge-followup.md) | 2b, 50b_wide | 2 | agpt_2b byte-identical to baseline (24.34 GiB / 38.04%); agpt_50b_wide re-confirms torch-2.13 `DeviceMesh`-in-saved-tensors crash (~121s to repro) |
