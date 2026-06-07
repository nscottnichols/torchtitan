# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# GRPO training with pluggable tasks using TRL + ezpz.
#
# Every TRL ``GRPOConfig`` field (and every HF ``TrainingArguments``
# field it inherits) is exposed as a CLI flag via ``HfArgumentParser``.
# Run with ``--help`` to see the full list — the curated defaults below
# only override values where the ezpz/XPU defaults differ from upstream
# TRL.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --task multiply --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
#       --max_steps 20 --per_device_train_batch_size 1
#
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
#       --task sum_digits --beta 0.04 --num_generations 8 \
#       --use_vllm --vllm_gpu_memory_utilization 0.5
#
# Note: HfArgumentParser uses ``snake_case`` flags (e.g.
# ``--per_device_train_batch_size``), not the hyphenated form.

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import ezpz
import ezpz.distributed

from torchtitan.experiments.ezpz.rl.tasks import get_task

log = logging.getLogger(__name__)

DEFAULT_MODEL = "argonne_private/AuroraGPT-7B"
FALLBACK_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class EzpzGRPOArgs:
    """ezpz-side CLI args that are not part of GRPOConfig."""

    task: str = field(
        default="sum_digits",
        metadata={"help": "Task name from torchtitan.experiments.ezpz.rl.tasks registry."},
    )
    model_name_or_path: str = field(
        default="",
        metadata={
            "help": (
                f"HuggingFace model name or local path. If empty, resolves "
                f"{DEFAULT_MODEL!r} with fallback to {FALLBACK_MODEL!r}."
            )
        },
    )
    num_samples: int = field(
        default=1000,
        metadata={"help": "Number of training prompts to materialize from the task."},
    )
    no_save: bool = field(
        default=False,
        metadata={"help": "Skip the final trainer.save_model() call."},
    )


def _ezpz_grpo_config_cls():
    """Build EzpzGRPOConfig lazily so importing this module doesn't drag
    in TRL/transformers (which pull in torch). Tasks/tests that just want
    the dataclass shape can stay light."""

    from trl import GRPOConfig

    @dataclass
    class EzpzGRPOConfig(GRPOConfig):
        # --- output / cadence -------------------------------------------------
        # output_dir is required by HF TrainingArguments. We give it a sentinel
        # so the user can leave it unset and we'll fill it in from --task below.
        output_dir: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "Output directory. Defaults to outputs/rl/grpo-{task} if unset."
                )
            },
        )
        max_steps: int = 50
        logging_steps: float = 1
        # Disable mid-training checkpoints to avoid safetensors E2BIG errors
        # on filesystems with path-length limits.
        save_strategy: str = "no"

        # --- precision / memory ------------------------------------------------
        bf16: bool = True
        gradient_accumulation_steps: int = 1
        gradient_checkpointing: bool = True
        # Empties XPU cache every N optimizer steps to fight allocator
        # fragmentation. XPU's allocator is more fragmentation-prone than CUDA.
        torch_empty_cache_steps: Optional[int] = 1

        # --- GRPO objective ----------------------------------------------------
        # beta=0.0 disables KL-against-reference. With beta=0 TRL doesn't
        # allocate or load the frozen reference model at all (saves ~bf16
        # model size per rank, e.g. ~4 GB for a 2B llama). Override to e.g.
        # 0.04 if you want the KL term back.
        beta: float = 0.0

        # --- generation --------------------------------------------------------
        num_generations: int = 4
        max_completion_length: int = 64
        temperature: float = 0.7

        # --- vLLM (off by default — XPU vLLM is fragile) -----------------------
        use_vllm: bool = False

    return EzpzGRPOConfig


def _resolve_model(name: str) -> str:
    """Try to load tokenizer for *name*; fall back if it doesn't exist."""
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(name)
        return name
    except Exception:
        log.warning(f"Model {name!r} not available, falling back to {FALLBACK_MODEL!r}")
        return FALLBACK_MODEL


def main() -> None:
    from trl import GRPOTrainer
    from transformers import AutoTokenizer, HfArgumentParser

    # Silence noisy HTTP request logs from huggingface_hub / httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    EzpzGRPOConfig = _ezpz_grpo_config_cls()
    parser = HfArgumentParser((EzpzGRPOArgs, EzpzGRPOConfig))
    ezpz_args, config = parser.parse_args_into_dataclasses()

    rank = ezpz.distributed.get_rank()
    device_type = ezpz.distributed.get_torch_device_type()

    # Resolve model: CLI override → default with fallback
    model_name = ezpz_args.model_name_or_path or _resolve_model(DEFAULT_MODEL)

    # Fill in output_dir sentinel from task
    if config.output_dir is None:
        config.output_dir = f"outputs/rl/grpo-{ezpz_args.task}"

    # report_to to wandb only on rank 0 (avoids 48 ranks all writing).
    # If the user passed --report_to explicitly we respect it on rank 0
    # but force "none" on every other rank.
    if rank != 0:
        config.report_to = []

    # save_steps must be > max_steps so the in-train save never fires.
    # (Belt-and-suspenders alongside save_strategy="no".)
    if config.save_steps and config.save_steps <= config.max_steps:
        config.save_steps = config.max_steps + 1

    # Resolve task from registry
    task = get_task(ezpz_args.task)

    log.info(
        f"[rank {rank}] GRPO config: model={model_name} task={ezpz_args.task} "
        f"max_steps={config.max_steps} lr={config.learning_rate} "
        f"bsz={config.per_device_train_batch_size} "
        f"num_gens={config.num_generations} beta={config.beta} "
        f"grad_ckpt={config.gradient_checkpointing} use_vllm={config.use_vllm} "
        f"device={device_type}"
    )

    # W&B tracking via ezpz (rank 0 only)
    if rank == 0:
        ezpz.distributed.setup_wandb(
            project_name="torchtitan.ezpz.rl",
            config={
                "model": model_name,
                "task": ezpz_args.task,
                "num_steps": config.max_steps,
                "lr": config.learning_rate,
                "batch_size": config.per_device_train_batch_size,
                "num_generations": config.num_generations,
                "beta": config.beta,
                "gradient_checkpointing": config.gradient_checkpointing,
                "use_vllm": config.use_vllm,
                "device_type": device_type,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = task.build_dataset(num_samples=ezpz_args.num_samples)
    log.info(f"[rank {rank}] Built dataset with {len(dataset)} samples")

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

    if rank == 0 and not ezpz_args.no_save:
        save_path = os.path.join(config.output_dir, "final")
        try:
            trainer.save_model(save_path)
            log.info(f"Model saved to {save_path}")
        except Exception as e:
            log.warning(f"Failed to save model to {save_path}: {e}")


if __name__ == "__main__":
    ezpz.distributed.setup_torch()
    main()
