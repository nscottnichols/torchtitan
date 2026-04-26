# AuroraGPT-20B Scaling

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes)

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 406+/-2 | 4,872+/-33 | 20.3% | 48.52 GiB (76%) | 100.0% |
| 2 | 24 | 358+/-4 | 8,592+/-101 | 17.9% | 44.64 GiB (70%) | 88.2% |
| 4 | 48 | 368+/-1 | 17,664+/-67 | 18.4% | 42.62 GiB (67%) | 90.6% |
| 8 | 96 | 366+/-2 | 35,184+/-203 | 18.3% | 41.85 GiB (65%) | 90.3% |
| 16 | 192 | 360+/-4 | 69,120+/-814 | 18.0% | 41.52 GiB (65%) | 88.7% |
| 32 | 384 | 366+/-0 | 140,736+/-271 | 18.3% | 41.12 GiB (64%) | 90.3% |
| 64 | 768 | 353 | 271,104 | 17.6% | 40.93 GiB (64%) | 86.9% |

**Config:** FSDP only (TP=1), compile=on, AC=full, seq_len=8192, LBS=1

## Aurora Weak Scaling (torch 2.13, 2–4096 nodes) — In Progress

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS | MFU | Memory | Loss (final) | Status |
|-------|------|-----|---------|--------|-----|--------|--------------|--------|
| 2 | 24 | 24 | 440 | 65.3 | 22.0% | 32.07 GiB (50%) | 9.43 | Complete |
| 4 | 48 | — | — | — | — | — | — | Pending |
| 8 | 96 | — | — | — | — | — | — | Pending |
| 16 | 192 | — | — | — | — | — | — | Pending |
| 32 | 384 | 768 | 4,500 | 50.3 | 16.9% | 43.96 GiB (69%) | 12.91 | Complete |
| 64–4096 | — | — | — | — | — | — | — | Pending |

**Config:** SophiaG LR=2.28e-5, compile=on, seq_len=8192, olmo-mix-1124

**Improvement over torch 2.10:** 440 TPS/GPU (2.13) vs 358 (2.10) at 2N = **+23%**

### Queue Mapping

| Node count | Queue | Walltime |
|------------|-------|----------|
| 2 | debug | 1h |
| 4–256 | debug-scaling | 1h |
| 256–1024 | prod → small | 6h |
| 1025–1919 | prod → medium | 6h |
| 1920+ | prod → large | 6h |

## Results Directory

- Sunspot: `outputs/scaling_study/20260412_091635/`
- Aurora (torch 2.13): per-job, see job IDs above

## See Also

- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/20b/) — live training status
