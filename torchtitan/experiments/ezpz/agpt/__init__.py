# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torch.nn.attention import SDPBackend

from torchtitan.models.common.attention import (
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)


class XPUScaledDotProductAttention(ScaledDotProductAttention):
    """SDPA with OVERRIDEABLE backend for XPU-optimized fused attention."""

    sdpa_backends = [
        SDPBackend.OVERRIDEABLE,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import FaultTolerantModelSpec

__all__ = [
    "model_registry",
    "parallelize_llama",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_agpt_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    inner_attention=None,
    mask_type: str = "causal",
    rope_backend: Literal["complex", "cos_sin"] = "complex",
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=(
                        inner_attention
                        if inner_attention is not None
                        else XPUScaledDotProductAttention.Config()
                    ),
                    mask_type=mask_type,
                    rope_backend=rope_backend,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _build_agpt_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
    rope_backend: Literal["complex", "cos_sin"] = "complex",
    scaling: Literal["none", "llama", "yarn"] = "none",
    max_seq_len: int = 131072,
) -> Llama3Model.Config:
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            backend=rope_backend,
            scaling=scaling,
        ),
        layers=_build_agpt_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            rope_backend=rope_backend,
        ),
    )


agpt_configs = {
    "debugmodel": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    "debugmodel_flex_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    "debugmodel_varlen_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    "2B": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
    ),
    "2B_flex_attn": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
    ),
    "7B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000,
        vocab_size=32000,
        hidden_dim=11008,
    ),
    "8B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(
            4096, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    "20B": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "20B_flex_attn": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "50B": _build_agpt_config(
        dim=8192,
        n_layers=56,
        n_heads=64,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(
            8192, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    # Aurora-native ~80B configs: n_kv_heads=12, n_heads divisible by 12
    # so TP can be any factor of 12 (2, 3, 4, 6, 12).
    #
    # ~80.8B: Balanced width/depth. PP divides 84: {1,2,3,4,6,12}.
    "80B": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # ~80.0B: Wider (dim=10752), shallower (48 layers).
    # PP divides 48: {1,2,3,4,6,8,12,16,24}.
    "80B_wide": _build_agpt_config(
        dim=10752,
        n_layers=48,
        n_heads=84,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=39936,
    ),
    # ~80.9B: Narrower (dim=7680), deeper (96 layers).
    # PP divides 96: {1,2,3,4,6,8,12,16,24}.
    "80B_deep": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28672,
    ),
    # Variants with hidden_dim divisible by 12, for clean TP sharding
    # across all factors of 12 (2, 3, 4, 6, 12).
    "80B_alt": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25596,  # 25600 -> 25596 (multiple of 12)
    ),
    "80B_deep_alt": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28668,  # 28672 -> 28668 (multiple of 12)
    ),
}


# Apply flex_attn overlay to the relevant configs
def _apply_flex_attn(config: Llama3Model.Config) -> Llama3Model.Config:
    flex_cfg = FlexAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


def _apply_varlen_attn(config: Llama3Model.Config) -> Llama3Model.Config:
    varlen_cfg = VarlenAttention.Config()
    layers = []
    for layer_cfg in config.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = varlen_cfg
        layer_cfg.attention.mask_type = "block_causal"
        layers.append(layer_cfg)
    config.layers = layers
    return config


_apply_flex_attn(agpt_configs["debugmodel_flex_attn"])
_apply_varlen_attn(agpt_configs["debugmodel_varlen_attn"])
_apply_flex_attn(agpt_configs["2B_flex_attn"])
_apply_flex_attn(agpt_configs["20B_flex_attn"])

# Case-insensitive aliases
agpt_configs["2b"] = agpt_configs["2B"]
agpt_configs["2b_flex_attn"] = agpt_configs["2B_flex_attn"]
agpt_configs["7b"] = agpt_configs["7B"]
agpt_configs["8b"] = agpt_configs["8B"]
agpt_configs["20b"] = agpt_configs["20B"]
agpt_configs["20b_flex_attn"] = agpt_configs["20B_flex_attn"]
agpt_configs["50b"] = agpt_configs["50B"]
agpt_configs["80b"] = agpt_configs["80B"]
agpt_configs["80b_wide"] = agpt_configs["80B_wide"]
agpt_configs["80b_deep"] = agpt_configs["80B_deep"]
agpt_configs["80b_alt"] = agpt_configs["80B_alt"]
agpt_configs["80b_deep_alt"] = agpt_configs["80B_deep_alt"]


def model_registry(flavor: str) -> FaultTolerantModelSpec:
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.experiments.ft.diloco import fragment_llm

    config = agpt_configs[flavor]

    return FaultTolerantModelSpec(
        name="ezpz.agpt",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
