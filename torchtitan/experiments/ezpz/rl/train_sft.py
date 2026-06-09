# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SFT companion to train_grpo.py. Same shape: HfArgumentParser CLI,
# FSDP env bootstrap, vocab-aware chat-template fallback, rank-0
# model prefetch + broadcast, wandb auto-config. Difference: uses
# TRL's SFTTrainer + a separate SFT dataset registry.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
#       --sft_dataset gsm8k \
#       --model_name_or_path /flare/AuroraGPT/.../global_step138650 \
#       --per_device_train_batch_size 1 --bf16 \
#       --fsdp full_shard --num_train_epochs 3
#
# After training, the SFT'd checkpoint can be passed as
# --model_name_or_path to train_grpo for a stronger RL starting point.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import ezpz
import ezpz.distributed

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    SFT_REGISTRY,
    _is_mix_spec,
    _parse_mix_spec,
    get_sft_dataset,
)
# Reuse the helpers from train_grpo so we don't drift between the two
# training entry points.
from torchtitan.experiments.ezpz.rl.train_grpo import (
    _autodetect_wrap_cls,
    _bootstrap_fsdp_env,
    _build_wandb_config,
    _pick_chat_template,
    _prefetch_and_broadcast_model,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
)

log = logging.getLogger(__name__)


def _sft_dataset_help() -> str:
    if not SFT_REGISTRY:
        return "SFT dataset name (registry is empty — no datasets imported?)."
    rows = "; ".join(f"{n}: {d.description}" for n, d in sorted(SFT_REGISTRY.items()))
    return (
        f"SFT dataset. Either a registered name from "
        f"torchtitan.experiments.ezpz.rl.datasets_sft (choices below) "
        f"OR a mix-spec like 'tulu-3-sft-mixture:0.5,gsm8k:0.5' that "
        f"interleaves multiple registered datasets at the given (auto-"
        f"renormalized) weights. Choices: {rows}"
    )


@dataclass
class EzpzSFTArgs:
    """ezpz-side CLI args that aren't part of SFTConfig."""

    # NOTE: no `choices=` because we also accept a mix-spec string
    # (e.g. 'tulu-3-sft-mixture:0.5,gsm8k:0.5'). Validation moved into
    # __post_init__ below — bare names must be in SFT_REGISTRY, mix
    # specs must parse cleanly and reference only registered names.
    sft_dataset: str = field(
        default="gsm8k",
        metadata={"help": _sft_dataset_help()},
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
    no_save: bool = field(
        default=False,
        metadata={"help": "Skip the final trainer.save_model() call."},
    )
    fsdp_transformer_layer_cls_to_wrap: str = field(
        default="LlamaDecoderLayer",
        metadata={
            "help": (
                "Transformer block class name to auto-wrap for FSDP. "
                "Auto-detected at runtime via model_type; override only "
                "if your model has a non-standard block class name."
            )
        },
    )
    fsdp_cpu_ram_efficient_loading: bool = field(
        default=False,
        metadata={
            "help": (
                "Load model on rank 0 only and broadcast shards. Slower "
                "init but avoids OOM during model loading on small-memory "
                "tiles. Has no effect unless --fsdp is set."
            )
        },
    )

    def __post_init__(self) -> None:
        # Validate sft_dataset early so a typo doesn't cost a model
        # load + dataset prefetch before failing. Two valid shapes:
        # bare name (must be in SFT_REGISTRY) or mix-spec (must parse
        # and reference only registered names).
        name = self.sft_dataset
        if _is_mix_spec(name):
            try:
                component_names, _weights = _parse_mix_spec(name)
            except ValueError as e:
                raise ValueError(f"Bad --sft_dataset {name!r}: {e}") from e
            missing = [n for n in component_names if n not in SFT_REGISTRY]
            if missing:
                available = ", ".join(sorted(SFT_REGISTRY))
                raise ValueError(
                    f"--sft_dataset mix references unknown component(s) "
                    f"{missing}. Available: {available}"
                )
        elif name not in SFT_REGISTRY:
            available = ", ".join(sorted(SFT_REGISTRY))
            raise ValueError(
                f"--sft_dataset {name!r} is not a registered name and not "
                f"a mix-spec (mix-specs contain ':'). Available registered "
                f"names: {available}"
            )


def _ezpz_sft_config_cls():
    """Build EzpzSFTConfig lazily so importing this module doesn't drag
    in TRL/transformers (which pull in torch)."""

    from trl import SFTConfig

    @dataclass
    class EzpzSFTConfig(SFTConfig):
        # --- output / cadence -------------------------------------------------
        output_dir: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "Output directory. Defaults to outputs/sft/{sft_dataset} "
                    "if unset."
                )
            },
        )
        # SFTConfig convention: max_steps=-1 means "use num_train_epochs"
        max_steps: int = -1
        num_train_epochs: float = 3.0
        logging_steps: float = 10
        save_strategy: str = "no"

        # --- precision / memory ------------------------------------------------
        bf16: bool = True
        gradient_accumulation_steps: int = 1
        gradient_checkpointing: bool = True
        torch_empty_cache_steps: Optional[int] = 1

        # --- SFT specifics -----------------------------------------------------
        # Compute loss only on the assistant turn(s), not on the user
        # prompt. Standard SFT setup — training on the prompt would just
        # make the model better at predicting the user's question.
        assistant_only_loss: bool = True
        # 1024 is usually enough for math CoT. Bump if your dataset has
        # longer responses.
        max_length: int = 1024
        # Packing concatenates multiple short examples into a single
        # sequence up to max_length, improving throughput. Compatible
        # with assistant_only_loss.
        packing: bool = True

        def __post_init__(self):
            # Same FSDP + gradient_checkpointing migration as
            # EzpzGRPOConfig: when --fsdp is set, migrate
            # gradient_checkpointing into fsdp_config so transformers'
            # "redundant AllGather" warning never fires.
            if self.fsdp and self.gradient_checkpointing:
                fsdp_cfg = self.fsdp_config
                if fsdp_cfg is None:
                    fsdp_cfg = {}
                elif isinstance(fsdp_cfg, str):
                    import json
                    with open(fsdp_cfg, encoding="utf-8") as f:
                        fsdp_cfg = json.load(f)
                else:
                    fsdp_cfg = dict(fsdp_cfg)
                fsdp_cfg.setdefault("activation_checkpointing", True)
                self.fsdp_config = fsdp_cfg
                self.gradient_checkpointing = False
            # Override TRL's device_map="auto" — see train_grpo.py
            # __post_init__ for the full diagnosis. With FSDP on, we want
            # the model to land on CPU first so HF Trainer puts it on
            # the per-rank accelerator.device.
            if self.fsdp:
                mik = self.model_init_kwargs
                if mik is None:
                    mik = {}
                elif isinstance(mik, str):
                    import json
                    mik = json.loads(mik)
                else:
                    mik = dict(mik)
                if "device_map" not in mik:
                    mik["device_map"] = None
                self.model_init_kwargs = mik
            super().__post_init__()

    return EzpzSFTConfig


def main() -> None:
    from trl import SFTTrainer
    from transformers import AutoTokenizer, HfArgumentParser

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    EzpzSFTConfig = _ezpz_sft_config_cls()
    parser = HfArgumentParser((EzpzSFTArgs, EzpzSFTConfig))
    ezpz_args, config = parser.parse_args_into_dataclasses()

    rank = ezpz.distributed.get_rank()
    device_type = ezpz.distributed.get_torch_device_type()

    model_name = _prefetch_and_broadcast_model(ezpz_args.model_name_or_path, rank)

    # Auto-detect FSDP wrap class from model_type
    wrap_cls = _autodetect_wrap_cls(
        model_name, ezpz_args.fsdp_transformer_layer_cls_to_wrap
    )
    ezpz_args.fsdp_transformer_layer_cls_to_wrap = wrap_cls

    _bootstrap_fsdp_env(
        config.fsdp,
        transformer_layer_cls_to_wrap=wrap_cls,
        cpu_ram_efficient_loading=ezpz_args.fsdp_cpu_ram_efficient_loading,
        bf16=config.bf16,
    )

    if config.output_dir is None:
        config.output_dir = f"outputs/sft/{ezpz_args.sft_dataset}"

    # Rank 0 defaults to wandb so the run is always observable; worker
    # ranks always get silenced so we don't double-log. To opt out,
    # pass --report_to none explicitly.
    #
    # Why we override the TRL default: TRL ships `report_to="none"` by
    # default, which silently means runs without an explicit
    # --report_to wandb don't show up in our wandb project at all.
    # Defaulting to wandb here is "boringly-correct" given that every
    # production ezpz run wants wandb anyway.
    if rank == 0:
        # Detect "user didn't pass --report_to" by checking the HF/TRL
        # default. The HF Trainer sentinel is the string "none" or the
        # list ["none"] depending on parser version; either means "not
        # explicitly set". An empty list means the same thing.
        not_explicitly_set = (
            config.report_to == "none"
            or config.report_to == ["none"]
            or not config.report_to
        )
        if not_explicitly_set:
            config.report_to = ["wandb"]
    else:
        config.report_to = []

    sft_ds = get_sft_dataset(ezpz_args.sft_dataset)

    log.info(
        f"[rank {rank}] SFT config: model={model_name} "
        f"dataset={ezpz_args.sft_dataset} epochs={config.num_train_epochs} "
        f"max_steps={config.max_steps} lr={config.learning_rate} "
        f"bsz={config.per_device_train_batch_size} "
        f"packing={config.packing} max_length={config.max_length} "
        f"fsdp={config.fsdp or 'off'} device={device_type}"
    )

    if rank == 0:
        ezpz.distributed.setup_wandb(
            project_name="torchtitan.ezpz.sft",
            config=_build_wandb_config(
                ezpz_args, config, model_name, device_type, rank,
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        kind, tokenizer.chat_template = _pick_chat_template(tokenizer)
        log.info(
            f"[rank {rank}] tokenizer has no chat_template; injected "
            f"{kind!r} fallback"
        )

    # Same shape as _prefetch_and_broadcast_model: rank 0 builds the
    # dataset first (which triggers the HF Hub download into the local
    # cache), then we barrier so worker ranks load from the warm cache
    # instead of all 384 ranks hammering HF Hub with concurrent xet-read
    # requests (job 12468217 died this way with HTTP 429 storms +
    # ".incomplete/dataset_info.json not found" cascade failures).
    import torch.distributed as dist
    from datetime import timedelta
    import threading
    import time

    def _rank0_progress_beacon(stop_event, label):
        """Print a heartbeat every 30s so it's obvious rank 0 is still
        alive during a long dataset build. Without this, builds that
        take 5-15 min (e.g. interleave_datasets with OpenMathInstruct-2's
        14M rows) look identical to a hang from the outside — log
        stays silent for ~10 min while ps shows 100% CPU.
        """
        t0 = time.monotonic()
        while not stop_event.wait(timeout=30):
            log.info(f"[prefetch] rank 0 still building {label} ({time.monotonic()-t0:.0f}s elapsed)")

    if rank == 0:
        log.info(f"[prefetch] rank 0 building SFT dataset (warms HF cache)...")
        beacon_stop = threading.Event()
        beacon = threading.Thread(
            target=_rank0_progress_beacon,
            args=(beacon_stop, ezpz_args.sft_dataset),
            daemon=True,
        )
        beacon.start()
        try:
            dataset = sft_ds.build()
        finally:
            beacon_stop.set()
            beacon.join(timeout=5)
        log.info(f"[prefetch] rank 0 cache warm for {ezpz_args.sft_dataset!r}")
    if dist.is_initialized():
        # Long timeout — rank 0 may take many minutes to build a large
        # interleaved mix (OpenMathInstruct-2 is 14M rows; even with
        # the HF .map() cache warm, instantiation + interleave setup
        # is ~20s+; cold first run is ~10 min). The PyTorch default
        # ProcessGroup timeout is 10 min, but on Sunspot the underlying
        # oneCCL barrier errors out much sooner (~30s) with
        # `atl_comm->wait fails with status: 1` — caught this in
        # job 12468348 where rank 0 was still mid-build when workers
        # hit the default-timeout barrier and crashed.
        dist.barrier(timeout=timedelta(minutes=30))
    if rank != 0:
        dataset = sft_ds.build()
    log.info(f"[rank {rank}] Built SFT dataset: {len(dataset)} samples")

    trainer = SFTTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info(f"[rank {rank}] Starting SFT training...")
    # Pass resume_from_checkpoint EXPLICITLY. HF Trainer's documented
    # behavior is to pick it up from args automatically when train() is
    # called with no arg, but with FSDP-sharded checkpoints the auto-
    # detect can silently fall through to fresh-from-scratch training
    # without raising. Explicit pass-through forces the FSDP-sharded
    # checkpoint load path (12468222 lost its resume to this exact bug).
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint or None)
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