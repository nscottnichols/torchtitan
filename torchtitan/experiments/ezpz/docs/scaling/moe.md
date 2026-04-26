# MoE Model Scaling

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes)

### moe_2b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 7,121+/-77 | 85,458+/-924 | 11.9% | 15.65 GiB (24%) | 100.0% |
| 2 | 24 | 6,970+/-7 | 167,280+/-169 | 11.7% | 15.34 GiB (24%) | 97.9% |
| 4 | 48 | 6,784+/-76 | 325,632+/-3665 | 11.4% | 15.92 GiB (25%) | 95.3% |
| 8 | 96 | 6,392+/-45 | 613,680+/-4412 | 10.7% | 16.84 GiB (26%) | 89.8% |
| 16 | 192 | 4,959+/-49 | 952,128+/-9503 | 8.3% | 19.09 GiB (30%) | 69.6% |
| 32 | 384 | 5,170+/-12 | 1,985,472+/-4615 | 8.7% | 26.27 GiB (41%) | 72.6% |
| 64 | 768 | 3,376 | 2,592,768 | 5.7% | 37.54 GiB (59%) | 47.4% |

### moe_7b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 1,841+/-13 | 22,098+/-161 | 8.4% | 33.47 GiB (52%) | 100.0% |
| 2 | 24 | 1,197+/-18 | 28,728+/-441 | 5.4% | 33.31 GiB (52%) | 65.0% |
| 4 | 48 | 1,417+/-28 | 68,016+/-1357 | 6.4% | 31.09 GiB (49%) | 76.9% |
| 8 | 96 | 1,111+/-0 | 106,704+/-67 | 5.0% | 35.87 GiB (56%) | 60.4% |
| 16 | 192 | 421+/-0 | 80,928+/-135 | 1.9% | 40.78 GiB (64%) | 22.9% |
| 32 | 384 | OOM | — | — | — | — |
| 64 | 768 | OOM | — | — | — | — |

**Config:** no-compile, AC=none, FSDP only (TP=1), seq_len=4096, LBS=1

**Note:** MoE scaling efficiency degrades significantly beyond 8 nodes.
Memory usage grows with node count (all-to-all communication buffers).
moe_7b OOMs at 32+ nodes. See [TODO — MoE throughput optimization](../TODO.md#3-moe-throughput-optimization)
for planned experiments (EP, TP, float8).

## Results Directory

- Sunspot: `outputs/scaling_study/20260412_091635/`

## See Also

- [MoE configs](../configs/moe.md) — model architecture details
- [Experiment reports](../experiments/moe/) — per-run benchmark logs
