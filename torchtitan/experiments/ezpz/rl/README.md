# RL (GRPO) Experiment

**Status: Experimental** — verified on Sunspot XPU, not production-ready.

Reinforcement Learning via Group Relative Policy Optimization (GRPO) using
HuggingFace TRL's `GRPOTrainer`. Alternative to upstream torchtitan's RL
experiment which requires CUDA-only dependencies (vLLM, torchmonarch,
flash-attn).

## Architecture

- **TRL GRPOTrainer** — handles the GRPO training loop (generate → score →
  compute advantages → update policy)
- **ezpz** — distributed launch, device setup, wandb tracking
- **No vLLM** — uses HF transformers `.generate()` for completions (slower
  but works on XPU)

## Task: Sum-of-Digits

Simple arithmetic task for verification: "What is 3 + 7 + 2?" → "12"

**Reward functions:**
- `sum_digits_reward` — 1.0 if extracted answer matches ground truth, 0.0
  otherwise
- `sum_digits_format_reward` — 0.5 bonus if completion shows addition
  expression (e.g. "3 + 7 + 2 = 12")

## Quick Start

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo
```

With custom settings:

```bash
GRPO_MODEL=Qwen/Qwen3-0.6B \
GRPO_STEPS=50 \
GRPO_NUM_SAMPLES=1000 \
GRPO_BATCH_SIZE=1 \
GRPO_GENERATIONS=2 \
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo
```

## Dependencies

- `trl` — install with `uv pip install --no-deps --no-cache --link-mode=copy trl`
  (use `--no-deps` to avoid pulling CUDA torch)
- `transformers`, `datasets` — usually already installed
- Model checkpoint downloaded from HF Hub (requires network access on
  compute nodes via proxy)

## Verified Results (Sunspot, 2026-04-15)

**Config:** Qwen3-0.6B, 24 XPU tiles (2 nodes), 10 steps, batch=1,
2 generations per prompt, 100 training samples.

| Step | Accuracy Reward | Format Reward |
|------|----------------|---------------|
| 1    | 4.2%           | 10.4%         |
| 5    | 33.3%          | 50.0%         |
| 8    | 62.5%          | 50.0%         |
| 9    | **87.5%**      | 50.0%         |
| 10   | 62.5%          | 50.0%         |

Training time: 41.3s, 5.8 samples/sec.

## Files

| File | Description |
|------|-------------|
| `train_grpo.py` | Main entry point — ezpz setup + TRL GRPOTrainer |
| `tasks.py` | Reward functions (accuracy + format) |
| `data.py` | Sum-of-digits dataset builder |

## Limitations

- **No vLLM** — generation is slow (HF `.generate()` on each rank)
- **All ranks generate** — no separate generator/trainer split like upstream
- **No weight sync** — single model instance, no Monarch actor framework
- **Sum-of-digits only** — task is hardcoded; swap `build_sum_digits_dataset`
  and reward functions for other tasks

## Upstream Comparison

The upstream `torchtitan/experiments/rl/` experiment uses:
- Monarch actors for separate generator/trainer GPU meshes
- vLLM for fast inference (4 GPUs for generation, 2 for training)
- TorchStore for weight synchronization via GPU-to-GPU RDMA
- Requires CUDA, flash-attn, torchmonarch

This ezpz alternative trades performance for portability — runs on any
device backend that TRL/Accelerate supports (XPU, CUDA, CPU).
