# Learning Rate Finder Experiments

LR finder sweeps across model sizes, optimizers, and machines. Uses exponential
LR sweep from 1e-6 to 1.0 with EMA-smoothed loss tracking
([Smith 2015](https://arxiv.org/abs/1506.01186)).

## Quick Reference: Recommended Learning Rates

Derived from blow-up point / 10 across all completed sweeps.

| Model | AdamW   | Muon    | SophiaG |
|-------|---------|---------|---------|
| 2B    | 2e-3    | 8e-4    | 3e-4    |
| 20B   | 4e-4    | 4e-5    | 1e-5    |
| 80B   | ~1e-4*  | ~1e-5*  | ~3e-6*  |

*Extrapolated from scaling trend; not empirically verified (80B OOM on 2 nodes).

## Key Findings

1. **Optimizer sensitivity:** AdamW (most tolerant) > Muon > SophiaG (most sensitive)
2. **Model scaling:** Larger models need lower LRs; Muon/SophiaG scale more aggressively
3. **Blow-up severity:** SophiaG diverges catastrophically vs gradual for AdamW/Muon

## Reports

### agpt (Dense)

| Date | Report | Models | Optimizers | Nodes | Machine | Key Result |
|------|--------|--------|-----------|-------|---------|------------|
| 2026-04-12 | [LR Finder](agpt/aurora/20260412-144400-lr-finder-n2.md) | 2B, 20B | AdamW, Muon, SophiaG | 2 | Aurora | AdamW most tolerant; SophiaG 10x lower LR |

### moe (Sparse)

*No LR finder runs yet.*
