# Production Training — Dense (agpt) Models

> Last updated: 2026-05-03
>
> **Restarted in v2 clones on 2026-04-30** after the bf16-master
> RMSNorm-freeze regression. All current production training is on
> `dtype=float32` master weights. Historical bf16-tainted runs are
> retained inside each per-model README under "Historical".

## Single canonical chain per model

Per-model production training is consolidated on **one** chain to
avoid divergent trajectories. Other queued jobs at different node
counts share the model but write to *different* checkpoint dirs (keyed
on `gbs`), so they're independent trajectories — they're not joining
the canonical chain. Treat them as scaling experiments, not as
extensions.

### 2B canonical chain (512N)

| Job ID | Walltime | Steps | Loss | Status |
|--------|---------:|------:|-----:|--------|
| 8460301 | 6h | 1–1387 | 12.65 → 3.59 | Done (NODE_FAIL @ end) |
| 8463626 | 12h | 1300–5073 | 3.59 → **2.97** | Done (NODE_FAIL @ end). 50 ckpts saved. |
| 8463627 | 12h | 5000+ | (resuming) | **Queued** (auto-resumes from step-5000) |

**Latest cumulative**: step **5,073** · loss **2.97** · **510B tokens** (10.9% of 4.67T target).

### 20B canonical chain (512N)

| Job ID | Walltime | Steps | Loss | Status |
|--------|---------:|------:|-----:|--------|
| 8460302 | 6h | 1–300 | 12.94 → 4.95 | Done (NODE_FAIL @ end). 3 ckpts saved. |
| 8463628 | 12h | 300+ | (resuming) | **Running** (auto-resumes from step-300) |

**Latest cumulative**: step **300** · loss **4.95** · **30B tokens** (0.6% of 4.67T target).

## Other queued jobs (independent ckpt trajectories, NOT canonical chain)

| Job ID | Model | Nodes | Walltime | Status | Notes |
|--------|-------|------:|---------:|--------|-------|
| 8463182 | 2B | 1024 | 12h | Queued | Fresh start, separate ckpt dir (`n1024-gbs24576`) |
| 8463183 | 20B | 1024 | 12h | Queued | Fresh start, separate ckpt dir (`n1024-gbs24576`) |
| 8463659 | 20B | 256 | 12h | Queued | Fresh start, separate ckpt dir (`n256-gbs6144`) |

**80B**: not yet restarted in v2.

## v1 (bf16-master, broken) vs v2 (fp32-master, current)

The original 2026-04-{14..29} runs were trained with
`--training.dtype=bfloat16`, which silently froze every RMSNorm.weight
at 1.0 (sub-ULP master-weight updates). v2 is the clean restart on
`--training.dtype=float32`. See
[`guides/training-dtype-bf16-norm-freeze.md`](../../guides/training-dtype-bf16-norm-freeze.md)
for the full diagnosis. Per-model overlays:

| 2B (v1 256N vs v2 256N + 512N) | 20B (v1 256N vs v2 512N) |
|--------------------------------|--------------------------|
| ![2B overlay](2b/figures/overlay_2b_v1_vs_v2.png) | ![20B overlay](20b/figures/overlay_20b_v1_vs_v2.png) |

Reproduce:
```bash
python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py --overlay 2b
python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py --overlay 20b
```

## Diagnostics

### 2B v2 256N

![2B v2 256N Diagnostics](2b/figures/training_diagnostics_2b_v2_256n.png)

### 20B v2 512N

![20B v2 512N Diagnostics](20b/figures/training_diagnostics_20b_v2_512n.png)
