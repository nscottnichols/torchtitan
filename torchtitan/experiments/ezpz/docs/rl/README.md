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

## Tasks

Tasks are pluggable via a registry. Use `--task <name>` to select:

| Task | Description | Difficulty |
|------|-------------|------------|
| `sum_digits` (default) | Addition: "What is 3 + 7 + 2?" → "12" | Easy |
| `multiply` | Multiplication: "What is 7 × 8?" → "56" | Easy |
| `word_sort` | Sort words alphabetically (partial credit) | Medium |
| `countdown` | Reach target using arithmetic on given numbers | Hard |

### Adding a new task

Create a module in `rl/tasks/`, define a dataset builder + reward functions,
and call `register_task()`. See `tasks/sum_digits.py` for the pattern.

## Quick Start

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo
```

With CLI args:

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --task sum_digits \
    --steps 50 \
    --batch-size 1 \
    --num-generations 2
```

Environment variables (`GRPO_MODEL`, `GRPO_TASK`, `GRPO_STEPS`, etc.) are
also supported as fallbacks when CLI args are not provided.

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
| [`train_grpo.py`](../../rl/train_grpo.py) | Main entry point — task-agnostic GRPO loop |
| [`tasks/__init__.py`](../../rl/tasks/__init__.py) | Task registry (`RLTask`, `register_task`, `get_task`) |
| [`tasks/common.py`](../../rl/tasks/common.py) | Shared helpers (answer extraction, completion text) |
| [`tasks/sum_digits.py`](../../rl/tasks/sum_digits.py) | Sum-of-digits task (dataset + rewards) |
| [`tasks/multiply.py`](../../rl/tasks/multiply.py) | Multiplication task |
| [`tasks/word_sort.py`](../../rl/tasks/word_sort.py) | Word sorting task (partial credit) |
| [`tasks/countdown.py`](../../rl/tasks/countdown.py) | Countdown arithmetic reasoning task |

## Limitations

- **No vLLM** — generation is slow (HF `.generate()` on each rank)
- **All ranks generate** — no separate generator/trainer split like upstream
- **No weight sync** — single model instance, no Monarch actor framework

## Upstream Comparison

The upstream `torchtitan/experiments/rl/` experiment uses:
- Monarch actors for separate generator/trainer GPU meshes
- vLLM for fast inference (4 GPUs for generation, 2 for training)
- TorchStore for weight synchronization via GPU-to-GPU RDMA
- Requires CUDA, flash-attn, torchmonarch

This ezpz alternative trades performance for portability — runs on any
device backend that TRL/Accelerate supports (XPU, CUDA, CPU).
