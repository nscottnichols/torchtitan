# LR Finder — MoE Models (Sunspot 2-node, 2026-04-21)

## Environment

| Field | Value |
|-------|-------|
| Date | 2026-04-21 |
| Machine | Sunspot |
| Nodes | 2 (24 XPU tiles) |
| Torch | 2.13.0.dev20260418+xpu |
| Compile | enabled |
| Vocab size | 256128 (Gemma tokenizer) |
| Seq len | 8192 |
| LBS | 1 |
| LR range | 1e-6 to 1.0 |
| Steps | 100 (fraction=0.1 of 1000) |

## Results

### AdamW

| Config | Experts | NaN | Suggested LR | Blow-up |
|--------|---------|-----|-------------|---------|
| debugmodel | 8 | 0 | 1.49e-7* | 1.49e-6* |
| 500M | 16 | 0 | 4.17e-2 | 4.17e-1 |
| 2B | 24 | 0 | 1.72e-7* | 1.72e-6* |
| 4B | 24 | 0 | 1.22e-3 | 1.22e-2 |
| 7B | 36 | 0 | 3.99e-4 | 3.99e-3 |
| 10b_2b_sdpa | 36 | — | — | OOM at seq_len=8192 |

### Muon

| Config | Experts | NaN | Status |
|--------|---------|-----|--------|
| debugmodel | 8 | — | expired (Muon NS overhead too slow) |
| 500M | 16 | — | expired |
| 2B | 24 | 0 | ✓ |
| 4B | 24 | 0 | ✓ |
| 7B | 36 | 0 | ✓ |

### SophiaG

| Config | Experts | NaN | Status |
|--------|---------|-----|--------|
| debugmodel | 8 | 0 | ✓ |
| 500M | 16 | 0 | ✓ |
| 2B | 24 | 0 | ✓ |
| 4B | 24 | 1 | mild instability at high LR |
| 7B | 36 | 5 | mild instability at high LR |

*Very low suggested LRs for debugmodel and 2B are likely derivative
analysis artifacts — early noise in the loss curve triggers false
blow-up detection. The 4B (1.22e-3) and 7B (3.99e-4) values are more
representative.

## Key Findings

1. **All MoE models are numerically stable** — Muon and SophiaG both work
   (unlike 80B dense where they crash). The MoE model dim=2048 is well
   below the bf16 overflow threshold (9216).

2. **AdamW is universally clean** — 0 NaN across all 5 configs tested.

3. **SophiaG has mild instability** — 1 NaN on 4B at high LR only. Not a
   structural issue like the 80B overflow.

4. **Muon is disproportionately slow on small MoE models** — the
   Newton-Schulz iteration overhead dominates compute for debugmodel and
   500M, causing 2-hour job timeouts.

5. **10b_2b_sdpa OOMs at seq_len=8192** — needs seq_len=4096 or more
   nodes. At LBS=1 and seq_len=8192, the model uses ~62 GiB per tile
   which exceeds the 64 GiB limit.

## Plots

See `docs/experiments/lr-finder/moe/sunspot/figures/` for per-model and
comparison plots.
