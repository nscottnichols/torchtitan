# 80B Throughput Leaderboard -- Aurora

Best throughput results for ~80B AGPT models on Aurora (2 nodes, 24 XPUs).

## Leaderboard

```
TPS (tokens/sec)
 50 |                                              *
 48 |                                           *
 45 |                                        *  .
 44 |                                     *  .  .
 41 |                                  *  .  .  .
 40 |                               *  .  .  .  .
 23 |            *  *               .  .  .  .  .
 20 |         *  .  .               .  .  .  .  .
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+-
       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                      Experiment #
```

## All Results (sorted by TPS)

| Rank | TPS | TFLOPS | MFU   | Variant      | TP | Compile | seq_len | LBS | AC   | Other | Memory | Status |
|------|-----|--------|-------|--------------|-----|---------|---------|-----|------|-------|--------|--------|
| 1    | **68** | **35.15** | **11.79%** | **80B_wide** | **4** | **on** | **8192** | 1 | full | -- | 44.2GiB (69.1%) | **BEST** |
| 2    | 63  | 32.21  | 10.80%| 80B_wide     | 4   | off     | 8192    | 1   | full | --    | 49.5GiB (77.3%) | OK |
| 3    | 53  | 27.28  | 9.18% | 80B_wide     | 6   | on      | 8184    | 1   | full | --    | 40.5GiB (63.3%) | OK |
| 2    | 48  | 24.34  | 8.16% | 80B_alt      | 6   | on      | 4092    | 2   | full | --    | 43.2GiB (67.6%) | OK |
| 3    | 50  | 25.58  | 8.58% | 80B_wide     | 6   | off     | 8184    | 1   | full | --    | 41.4GiB (64.7%) | OK |
| 4    | 45  | 24.53  | 8.23% | 80B_alt      | 6   | on      | 8184    | 1   | full | --    | 39.5GiB (61.8%) | OK |
| 3    | 45  | 24.34  | 8.16% | 80B_alt      | 6   | on      | 8184    | 1   | full | async TP | 39.5GiB (61.8%) | no gain |
| 4    | 44  | 23.91  | 8.02% | 80B_deep_alt | 6   | on      | 8184    | 1   | full | --    | 41.6GiB (65.0%) | OK |
| 5    | 41  | 22.52  | 7.55% | 80B_alt      | 6   | off     | 8184    | 1   | full | --    | 44.2GiB (69.0%) | OK |
| 6    | 40  | 21.85  | 7.33% | 80B_deep_alt | 6   | off     | 8184    | 1   | full | --    | 41.2GiB (64.4%) | OK |
| 7    | 23  | 12.46  | 4.18% | 80B_alt      | 3   | on      | 8190    | 1   | full | --    | 51.6GiB (80.7%) | OK |
| 8    | 31  | 16.68  | 5.59% | 80B_deep_alt | 12  | on      | 8184    | 1   | full | --    | 35.5GiB (55.5%) | OK |
| 9    | 20  | 11.25  | 3.77% | 80B_alt      | 3   | off     | 8190    | 1   | full | --    | 52.4GiB (81.9%) | OK |
| --   | --  | --     | --    | 80B          | 2   | off     | 8192    | 1   | full | --    | OOM | driver OOM |
| --   | --  | --     | --    | 80B          | 2   | on      | 8192    | 1   | full | --    | OOM | driver OOM |
| --   | --  | --     | --    | 80B          | 2   | off     | 8192    | 1   | full | cpu_offload | OOM | driver OOM |
| --   | --  | --     | --    | 80B_alt      | 6   | on      | 8184    | 1   | selective | -- | 61.6GiB (96.2%) | OOM |
| --   | --  | --     | --    | 80B_alt      | 6   | on      | 8184    | 1   | full | reshard=never | 56.1GiB (87.6%) | OOM |
| --   | --  | --     | --    | 80B_alt      | 6   | on      | 8184    | 1   | memory_budget(0.3) | -- | -- | graph break |

## Key Insights

1. **Best production config (seq~8k):** 80B_alt TP=6 + compile = **45 TPS, 8.23% MFU**
2. **torch.compile always helps:** +10-15% TPS, often reduces memory
3. **80B_alt > 80B_deep_alt:** Wider is faster than deeper at same TP
4. **TP=2 blocked on Aurora:** Level Zero driver OOM, not a PyTorch issue
5. **TP=6 >> TP=3:** 2x throughput despite 2x more TP communication

## Pending Experiments

- TP=4 (80B_alt, 80B_deep_alt) -- unexplored TP degree
- TP=12 (80B_deep_alt) -- maximum TP, minimum per-rank memory
- 80B_wide TP=6 -- widest variant
- Gradient accumulation (LBS=1 GAS=2 vs LBS=2 GAS=1)
- Pipeline parallelism (PP=2 TP=3 if PP bug is fixed)
