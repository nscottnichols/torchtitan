# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import is_dataclass
from typing import Any

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan.trainer import Trainer

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(base_fn) -> FaultTolerantTrainer.Config:
    cfg = base_fn()
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


# cfg.training.dtype = "bfloat16"
# cfg.dataloader.dataset = "blendcorpus"
# cfg.metrics.log_freq = 1
# cfg.metrics.enable_wandb = True
# cfg.compile = CompileConfig(enable=True)
# cfg.checkpoint.enable = True
# cfg.checkpoint.interval = 50
def moe_debugmodel() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        debug=DebugConfig(print_config=True),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=model_registry("debugmodel"),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        dataloader=BlendCorpusDataLoader.Config(dataset="blendcorpus"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def moe_debugmodel_hf() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        debug=DebugConfig(print_config=True),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def moe_debugmodel_flex_attn() -> FaultTolerantTrainer.Config:
    config = moe_debugmodel()
    config.model_spec = model_registry("debugmodel_flex_attn")
    return config


def moe_debugmodel_flex_attn_hf() -> FaultTolerantTrainer.Config:
    config = moe_debugmodel_hf()
    config.model_spec = model_registry("debugmodel_flex_attn_hf")
    return config


def moe_small() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        debug=DebugConfig(print_config=True),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=model_registry("small"),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        dataloader=BlendCorpusDataLoader.Config(dataset="blendcorpus"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def moe_small_hf() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/gemma-7b",
        debug=DebugConfig(print_config=True),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=model_registry("small"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def moe_16b() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry("16B"),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        # dataloader=HuggingFaceTextDataLoader.Config(
        #     dataset="c4",
        # ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def moe_671b() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry("671B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ],
        ),
    )


def moe_10b_2b() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/gemma-7b",
        debug=DebugConfig(print_config=True),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=model_registry("10B_2B"),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        dataloader=BlendCorpusDataLoader.Config(dataset="blendcorpus"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def moe_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_debugmodel)


def moe_small_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_small)


def moe_16b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_16b)


def moe_10b_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b)


def moe_671b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_671b)
