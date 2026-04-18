# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# GRPO training for sum-of-digits task using TRL + ezpz.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo
#
# Environment variables:
#   GRPO_MODEL       — HuggingFace model name/path (default: Qwen/Qwen3-0.6B)
#   GRPO_STEPS       — Number of training steps (default: 50)
#   GRPO_NUM_SAMPLES — Number of training prompts (default: 1000)
#   GRPO_LR          — Learning rate (default: 1e-5)
#   GRPO_BATCH_SIZE  — Per-device batch size (default: 2)
#   GRPO_GENERATIONS — Completions per prompt (default: 4)
#   GRPO_OUTPUT_DIR  — Output directory (default: outputs/rl/grpo-sum-digits)

import logging
import os

import ezpz
import ezpz.distributed

from torchtitan.experiments.ezpz.rl.data import build_sum_digits_dataset
from torchtitan.experiments.ezpz.rl.tasks import (
    sum_digits_format_reward,
    sum_digits_reward,
)

log = logging.getLogger(__name__)


def main() -> None:
    # Lazy imports so ezpz setup happens first
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer

    rank = ezpz.distributed.get_rank()
    device_type = ezpz.distributed.get_torch_device_type()

    model_name = os.environ.get("GRPO_MODEL", "Qwen/Qwen3-0.6B")
    num_steps = int(os.environ.get("GRPO_STEPS", "50"))
    num_samples = int(os.environ.get("GRPO_NUM_SAMPLES", "1000"))
    lr = float(os.environ.get("GRPO_LR", "1e-5"))
    batch_size = int(os.environ.get("GRPO_BATCH_SIZE", "2"))
    num_generations = int(os.environ.get("GRPO_GENERATIONS", "4"))
    output_dir = os.environ.get("GRPO_OUTPUT_DIR", "outputs/rl/grpo-sum-digits")

    log.info(
        f"[rank {rank}] GRPO config: model={model_name} steps={num_steps} "
        f"lr={lr} batch_size={batch_size} generations={num_generations} "
        f"device={device_type}"
    )

    # W&B tracking via ezpz (rank 0 only)
    if rank == 0:
        ezpz.distributed.setup_wandb(
            project_name="torchtitan.ezpz.rl",
            config={
                "model": model_name,
                "task": "sum_digits",
                "num_steps": num_steps,
                "lr": lr,
                "batch_size": batch_size,
                "num_generations": num_generations,
                "device_type": device_type,
            },
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = build_sum_digits_dataset(num_samples=num_samples)
    log.info(f"[rank {rank}] Built dataset with {len(dataset)} samples")

    # TRL config
    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        max_completion_length=64,
        temperature=0.7,
        learning_rate=lr,
        logging_steps=1,
        save_steps=num_steps,
        use_vllm=False,
        bf16=True,
        gradient_accumulation_steps=1,
        report_to="wandb" if rank == 0 else "none",
    )

    # Trainer with accuracy + format rewards
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[sum_digits_reward, sum_digits_format_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info(f"[rank {rank}] Starting GRPO training...")
    trainer.train()
    log.info(f"[rank {rank}] Training complete.")

    if rank == 0:
        trainer.save_model(os.path.join(output_dir, "final"))
        log.info(f"Model saved to {output_dir}/final")


if __name__ == "__main__":
    ezpz.distributed.setup_torch()
    main()
