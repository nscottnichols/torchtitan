# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.experiments.blendcorpus.dataset.blendcorpus_builder import (
    build_blendcorpus_dataloader,
)
from torchtitan.experiments.blendcorpus.dataset.build_tokenizer import build_tokenizer

from torchtitan.experiments.blendcorpus.infra.parallelize import parallelize_llama
from torchtitan.experiments.blendcorpus.infra.pipeline import pipeline_llama
from torchtitan.experiments.blendcorpus.model.args import TransformerModelArgs
from torchtitan.experiments.blendcorpus.model.model import Transformer
from torchtitan.experiments.blendcorpus.model.state_dict_adapter import (
    Llama3StateDictAdapter,
)
from torchtitan.experiments.blendcorpus.validate import build_blendcorpus_validator
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "model_configs",
]


# _enforced: str = "This field is used to enforce al",
#  dim: int = 4096,
#  n_layers: int = 32,
#  n_heads: int = 32,
#  n_kv_heads: int | None = None,
#  vocab_size: int = 128256,
#  multiple_of: int = 256,
#  ffn_dim_multiplier: float | None = None,
#  norm_eps: float = 0.00001,
#  rope_theta: float = 10000,
#  max_seq_len: int = 131072,
#  depth_init: bool = True,
#  use_flex_attn: bool = False,
#  attn_mask_type: str = "causal",
#  eos_id: int = 0


def compute_intermediate_size(
    dim: int,
    ffn_dim_multiplier: int = 1,
    multiple_of: int = 256,
):
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * dim / 3)) + multiple_of - 1) // multiple_of
    )


def compute_ffn_dim_multiplier(
    dim: int,
    intermediate_size: int,
    multiple_of: int = 256,
):
    return 1 + intermediate_size / multiple_of - multiple_of / (8 * dim / 3)


model_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=32000, rope_theta=500000
    ),
    "AuroraGPT-7B": TransformerModelArgs(
        dim=4096,
        n_kv_heads=8,
        # ffn_dim_multiplier=11008,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        rope_theta=10000,
    ),
    "AuroraGPT-2B": TransformerModelArgs(
        dim=2048,
        n_kv_heads=4,
        n_layers=12,
        n_heads=16,
        # ffn_dim_multiplier=compute_ffn,
        vocab_size=256128,
        rope_theta=50000,
    ),
    "Llama-2-7b": TransformerModelArgs(
        dim=4096, n_layers=6, n_heads=32, vocab_size=32000, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2000,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "Llama3-8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "Llama3-70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=256,
        rope_theta=500000,
    ),
    # "70B": TransformerModelArgs(
    #     dim=8192,
    #     n_layers=80,
    #     n_heads=64,
    #     n_kv_heads=8,
    #     ffn_dim_multiplier=1.3,
    #     multiple_of=4096,
    #     rope_theta=500000,
    # ),
    "Llama-3-405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="blendcorpus",
        model_cls=Transformer,
        model_args=model_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_blendcorpus_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_blendcorpus_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)
