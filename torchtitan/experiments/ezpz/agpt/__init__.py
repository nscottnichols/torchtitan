# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ft.diloco import fragment_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    FeedForward,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import FaultTolerantModelSpec

__all__ = [
    "model_registry",
    "parallelize_llama",
]

# backend: Literal['complex', 'cos_sin'] = "complex",
#    scaling: Literal['none', 'llama', 'yarn'] = "none",


def _build_llama3_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
    attn_backend: str = "sdpa",
    rope_backend: Literal["complex", "cos_sin"] = "complex",
    scaling: Literal["none", "llama", "yarn"] = "none",
    max_seq_len: int = 131072,
) -> Llama3Model.Config:
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(hidden_dim=hidden_dim),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                attn_backend=attn_backend,
                rope_backend=rope_backend,
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            backend=rope_backend,
            scaling=scaling,
        ),
    )


agpt_configs = {
    # "debugmodel": Llama3Model.Config(
    #     dim=256,
    #     n_layers=6,
    #     vocab_size=2048,
    #     layer=Llama3TransformerBlock.Config(
    #         feed_forward=FeedForward.Config(
    #             hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
    #         ),
    #         attention=GQAttention.Config(
    #             n_heads=16, attn_backend="sdpa", rope_backend="complex"
    #         ),
    #     ),
    #     rope=RoPE.Config(
    #         # TODO: find better ways to enforce dim = decoder dim // n_heads, for all models
    #         dim=256 // 16,
    #         max_seq_len=131072,
    #         theta=500000,
    #         backend="complex",
    #         scaling="llama",
    #     ),
    # ),
    "debugmodel": _build_llama3_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    "debugmodel_flex_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "debugmodel_varlen_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="varlen",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "2B": _build_llama3_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
    ),
    "2B_flex_attn": Llama3Model.Config(
        dim=2048,
        n_layers=12,
        vocab_size=256128,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(2048, multiple_of=1024)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=2048 // 16,
            max_seq_len=256128,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "7B": _build_llama3_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000,
        vocab_size=32000,
        hidden_dim=11008,
    ),
    "8B": _build_llama3_config(
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
    "20B": _build_llama3_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "50B": _build_llama3_config(
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
    "80B": _build_llama3_config(
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
    "80B_wide": _build_llama3_config(
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
    "80B_deep": _build_llama3_config(
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
    "80B_alt": _build_llama3_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25596,  # 25600 -> 25596 (multiple of 12)
    ),
    "80B_deep_alt": _build_llama3_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28668,  # 28672 -> 28668 (multiple of 12)
    ),
}
# agpt_configs["debugmodel"] = agpt_configs["debug"]
agpt_configs["2b"] = agpt_configs["2B"]
agpt_configs["2b_flex_attn"] = agpt_configs["2B_flex_attn"]
agpt_configs["7b"] = agpt_configs["7B"]
agpt_configs["8b"] = agpt_configs["8B"]
agpt_configs["20b"] = agpt_configs["20B"]
agpt_configs["50b"] = agpt_configs["50B"]
agpt_configs["80b"] = agpt_configs["80B"]
agpt_configs["80b_wide"] = agpt_configs["80B_wide"]
agpt_configs["80b_deep"] = agpt_configs["80B_deep"]
agpt_configs["80b_alt"] = agpt_configs["80B_alt"]
agpt_configs["80b_deep_alt"] = agpt_configs["80B_deep_alt"]


def model_registry(flavor: str) -> FaultTolerantModelSpec:
    return FaultTolerantModelSpec(
        name="ezpz.agpt",
        flavor=flavor,
        model=agpt_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
