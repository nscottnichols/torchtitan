# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# GRPO training with pluggable tasks using TRL + ezpz.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --model-name-or-path Qwen/Qwen3-0.6B --task sum_digits --steps 20
#
# CLI args take precedence over environment variables.

import argparse
import logging
import os

import ezpz
import ezpz.distributed

from torchtitan.experiments.ezpz.rl.tasks import get_task

log = logging.getLogger(__name__)

DEFAULT_MODEL = "argonne_private/AuroraGPT-7B"
FALLBACK_MODEL = "Qwen/Qwen3-0.6B"


def _resolve_model(name: str) -> str:
    """Try to load tokenizer for *name*; fall back if it doesn't exist."""
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(name)
        return name
    except Exception:
        log.warning(f"Model {name!r} not available, falling back to {FALLBACK_MODEL!r}")
        return FALLBACK_MODEL


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with TRL + ezpz")
    p.add_argument(
        "--model-name-or-path",
        default=os.environ.get("GRPO_MODEL", ""),
        help=f"HuggingFace model name/path (default: {DEFAULT_MODEL}, "
        f"fallback: {FALLBACK_MODEL})",
    )
    p.add_argument(
        "--task",
        default=os.environ.get("GRPO_TASK", "sum_digits"),
        help="Task name from the registry (default: sum_digits)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("GRPO_STEPS", "50")),
        help="Number of training steps (default: 50)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=int(os.environ.get("GRPO_NUM_SAMPLES", "1000")),
        help="Number of training prompts (default: 1000)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("GRPO_LR", "1e-5")),
        help="Learning rate (default: 1e-5)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GRPO_BATCH_SIZE", "2")),
        help="Per-device batch size (default: 2)",
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=int(os.environ.get("GRPO_GENERATIONS", "4")),
        help="Completions per prompt (default: 4)",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("GRPO_OUTPUT_DIR", ""),
        help="Output directory (default: outputs/rl/grpo-{task})",
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=int(os.environ.get("GRPO_MAX_COMPLETION_LENGTH", "64")),
        help="Max tokens for generation (default: 64)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("GRPO_TEMPERATURE", "0.7")),
        help="Sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip saving the model after training",
    )
    return p.parse_args()


def main() -> None:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer

    # Silence noisy HTTP request logs from huggingface_hub / httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    args = _parse_args()

    rank = ezpz.distributed.get_rank()
    device_type = ezpz.distributed.get_torch_device_type()

    # Resolve model: CLI/env override → default with fallback
    if args.model_name_or_path:
        model_name = args.model_name_or_path
    else:
        model_name = _resolve_model(DEFAULT_MODEL)

    # Resolve task from registry
    task = get_task(args.task)

    # Default output dir includes task name
    output_dir = args.output_dir or f"outputs/rl/grpo-{args.task}"

    log.info(
        f"[rank {rank}] GRPO config: model={model_name} task={args.task} "
        f"steps={args.steps} lr={args.lr} batch_size={args.batch_size} "
        f"generations={args.num_generations} device={device_type}"
    )

    # W&B tracking via ezpz (rank 0 only)
    if rank == 0:
        ezpz.distributed.setup_wandb(
            project_name="torchtitan.ezpz.rl",
            config={
                "model": model_name,
                "task": args.task,
                "num_steps": args.steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_generations": args.num_generations,
                "device_type": device_type,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = task.build_dataset(num_samples=args.num_samples)
    log.info(f"[rank {rank}] Built dataset with {len(dataset)} samples")

    # Disable mid-training checkpoints to avoid safetensors E2BIG errors
    # on filesystems with path length limits.
    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=args.steps + 1,
        save_strategy="no",
        use_vllm=False,
        bf16=True,
        gradient_accumulation_steps=1,
        report_to="wandb" if rank == 0 else "none",
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=task.reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info(f"[rank {rank}] Starting GRPO training...")
    trainer.train()
    log.info(f"[rank {rank}] Training complete.")

    if rank == 0 and not args.no_save:
        save_path = os.path.join(output_dir, "final")
        try:
            trainer.save_model(save_path)
            log.info(f"Model saved to {save_path}")
        except Exception as e:
            log.warning(f"Failed to save model to {save_path}: {e}")


if __name__ == "__main__":
    ezpz.distributed.setup_torch()
    main()
