# AuroraGPT-2B Scaling

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes)

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 6,457+/-38 | 77,490+/-466 | 24.2% | 47.29 GiB (74%) | 100.0% |
| 2 | 24 | 5,529+/-71 | 132,708+/-1714 | 20.8% | 46.80 GiB (73%) | 85.6% |
| 4 | 48 | 5,580+/-81 | 267,864+/-3903 | 20.9% | 46.66 GiB (73%) | 86.4% |
| 8 | 96 | 5,384+/-173 | 516,912+/-16631 | 20.2% | 46.58 GiB (73%) | 83.4% |
| 16 | 192 | 5,032+/-44 | 966,240+/-8553 | 18.9% | 46.42 GiB (73%) | 77.9% |
| 32 | 384 | 5,091+/-118 | 1,954,944+/-45616 | 19.1% | 46.41 GiB (73%) | 78.8% |
| 64 | 768 | 4,783 | 3,673,344 | 17.9% | 46.38 GiB (72%) | 74.1% |

**Config:** FSDP only (TP=1), compile=on, AC=full, seq_len=4096, LBS=1

**Results directory:** `outputs/scaling_study/20260412_091635/`

## See Also

- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/2b/) — live training status
