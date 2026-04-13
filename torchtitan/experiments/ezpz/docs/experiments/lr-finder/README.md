# Learning Rate Finder

## How It Works

The learning rate finder identifies the optimal learning rate for a given
model and optimizer by running a short sweep of exponentially increasing
learning rates while monitoring training loss.

### Algorithm

Based on [Smith 2015](https://arxiv.org/abs/1506.01186) and
[Gugger's implementation](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html),
ported from [argonne-lcf/Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed).

1. Set `N = fraction * training.steps` (default: 10% of total steps)
2. Compute multiplicative factor: `mult = (max_lr / init_lr) ^ (1/N)`
3. For each step `i` in `1..N`:
   - Run one full training step (forward + backward + optimizer)
   - Record the loss
   - Compute EMA-smoothed loss: `avg = beta * avg + (1 - beta) * loss`
   - Apply bias correction: `smoothed = avg / (1 - beta^i)`
   - Update learning rate: `lr *= mult`
4. Save results and exit (no further training)

### The LR-vs-Loss Curve

The sweep produces a characteristic curve with three regions:

```
Loss
  |  ___________
  | /           \        <- flat (LR too small to affect loss)
  |/             \       <- descent (optimal region)
  |               \___   <- blow-up (LR too large, divergence)
  +-----|-----|--------> LR (log scale)
     init_lr  optimal  max_lr
```

### Selecting the Optimal LR

Two common heuristics:

1. **Blow-up point / 10**: Find the LR where loss starts increasing,
   divide by 10. Conservative, safe for production.
2. **Steepest descent**: Pick the LR at the steepest part of the
   loss curve. More aggressive, potentially faster convergence.

### Knobs to Tune

| Parameter | Effect | When to change |
|-----------|--------|---------------|
| `init_lr` | Starting LR | Lower if the flat region is too short |
| `max_lr` | Ending LR | Raise if blow-up isn't reached |
| `fraction` | Sweep length | Increase for smoother curves (more steps) |
| `beta` | EMA smoothing | Lower (0.9) for noisier curves, higher (0.99) for smoother |
| `training.steps` | Base for fraction | Set to 1000+ for 100+ finder steps |

### Important Notes

- The finder **bypasses the normal LR scheduler** — it directly sets
  `param_group["lr"]` on all optimizer param groups after each step
- The model is **not saved** after the sweep — it's meant for LR
  selection, not training
- Results are **distributed-aware** — loss is reduced across all ranks
  before recording
- The sweep runs the **full training step** including gradient clipping,
  so the curve reflects realistic training dynamics

## Quick Reference: Recommended Learning Rates

Derived from blow-up point / 10 across all completed sweeps.

| Model | AdamW   | Muon    | SophiaG |
|-------|---------|---------|---------|
| 2B    | 2e-3    | 8e-4    | 3e-4    |
| 20B   | 4e-4    | 4e-5    | 1e-5    |
| 80B   | ~1e-4*  | ~1e-5*  | ~3e-6*  |

*Extrapolated from scaling trend; not empirically verified (80B OOM on Aurora 2 nodes).

## Key Findings

1. **Optimizer sensitivity:** AdamW (most tolerant) > Muon > SophiaG (most sensitive)
2. **Model scaling:** Larger models need lower LRs; Muon/SophiaG scale more aggressively
3. **Blow-up severity:** SophiaG diverges catastrophically vs gradual for AdamW/Muon

## Usage

### Running the LR finder

Add `--lr_finder.enable` to any training command:

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_2b \
  --training.steps 1000 \
  --checkpoint.no_enable \
  --lr_finder.enable
```

The finder runs for 10% of `training.steps` (e.g. 100 steps for 1000), sweeping
LR exponentially from `init_lr` to `max_lr`.

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--lr_finder.enable` | false | Run LR finder instead of training |
| `--lr_finder.init_lr` | 1e-6 | Starting learning rate |
| `--lr_finder.max_lr` | 1.0 | Maximum learning rate |
| `--lr_finder.fraction` | 0.1 | Fraction of training.steps to sweep |
| `--lr_finder.beta` | 0.98 | EMA smoothing factor |

### With different optimizers

```bash
# AdamW (default)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable

# Muon
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable \
  --optimizer muon

# SophiaG
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_20b \
  --training.steps 1000 --checkpoint.no_enable --lr_finder.enable \
  --optimizer sophiag
```

### Output

Results are saved to `outputs/lr_finder/ezpz/<module>/<flavor>/<optimizer>/`:

- `lr_finder_data.csv` — two columns: learning_rate, loss
- `lr_finder_data.npz` — numpy arrays for programmatic analysis
- `lr_vs_loss.png` — log-scale plot with minimum marked

### Interpreting results

The optimal learning rate is estimated as:
- **Blow-up point / 10**: find where loss starts increasing, divide that LR by 10
- **Steepest descent**: pick the LR in the steepest part of the loss curve

### Reproducing the experiments

Use the script at `torchtitan/experiments/ezpz/scripts/run_lr_finder_sweep.sh`:

```bash
bash torchtitan/experiments/ezpz/scripts/run_lr_finder_sweep.sh
```

See the script for configuration options (models, optimizers, steps).

## References

- Smith, L.N. (2015). [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- Gugger, S. [How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
- [Megatron-DeepSpeed LR Finder](https://github.com/saforem2/Megatron-DeepSpeed/blob/updates-and-model-card/ALCF/notes/large_batch_optimizers_settings.md#learning-rate) (argonne-lcf)

## Reports

### agpt (Dense)

| Date | Report | Models | Optimizers | Nodes | Machine | Key Result |
|------|--------|--------|-----------|-------|---------|------------|
| 2026-04-12 | [LR Finder](agpt/aurora/20260412-144400-lr-finder-n2.md) | 2B, 20B | AdamW, Muon, SophiaG | 2 | Aurora | AdamW most tolerant; SophiaG 10x lower LR |
| 2026-04-13 | [LR Finder](agpt/polaris/20260413-015510-lr-finder-n2.md) | 2B, 20B | AdamW, Muon, SophiaG | 2 | Polaris | Reproduces Aurora; cross-hardware LR consistency confirmed |
| 2026-04-12 | [LR Finder](agpt/sunspot/20260412-lr-finder-n2.md) | 2B | AdamW, Muon, SophiaG | 2 | Sunspot | Reproduces Aurora; SophiaG blow-up more violent (910 vs 295) |

### moe (Sparse)

*No LR finder runs yet.*
