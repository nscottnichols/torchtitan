# yeet-env Tarball Broadcast Scaling

Measures `ezpz yeet-env --src .venv.tar.gz` wall-clock time across node
counts on Aurora. Replaces the per-file rsync mode previously documented
in CLAUDE.md (which predicted hours at 256+ nodes).

## Setup

Per-test job:
- 30-min walltime (1h for ≥1024N)
- 8/16/32/64/128/256 → debug-scaling
- 512/1024 → small (route prod)
- 2048/4096 → prod-large (route prod)
- Each job: time yeet-env, then 10 training steps of agpt_2b to verify
  the broadcast venv is functional.

Submit script: `torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh`
Results CSV: `.yeet-env-scaling-results.csv` (in repo root).
Plots: `figures/yeet_env_seconds.png`, `figures/yeet_env_per_node.png`.

## Reproducing

```bash
# Submit each scale, one at a time (PBS per-user-Q limit blocks
# concurrent submission; chain via qsub depend or a polling wrapper).
for N in 8 16 32 64 128 256; do
    qsub -q debug-scaling -l select=$N -l walltime=00:30:00 \
        -N yeet-n$N \
        torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh
done

# 512/1024 → prod (auto-routes to small)
for N in 512 1024; do
    qsub -q prod -l select=$N -l walltime=01:00:00 \
        -N yeet-n$N \
        torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh
done

# 2048/4096 → prod (auto-routes to prod-large)
for N in 2048 4096; do
    qsub -q prod -l select=$N -l walltime=01:00:00 \
        -N yeet-n$N \
        torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh
done

# Plot results once ≥3 points have landed:
python3 torchtitan/experiments/ezpz/docs/scaling/yeet_env/plot_yeet_env_scaling.py
```

## Results

(populated as jobs complete — see `figures/`)
