# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama

from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import Transformer
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
]


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


# # AuroraGPT-2B
# export HEADS=16
# export NLAYERS=12
# export HIDDEN=2048
# export NUM_KV_HEAD=4
# export FFN_HIDDEN_SIZE=11008
# export SEQ=8192
# export MODEL_ARCH="AuroraGPT-2B"
#
#
# ffn_dim_multiplier_2b = compute_ffn_dim_multiplier(
#     dim=2048, intermediate_size=11008, multiple_of=1024
# )


flavors = {
    "debug": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
    "2b": TransformerModelArgs(
        dim=2048,
        n_kv_heads=4,
        n_layers=12,
        n_heads=16,
        # 11008 is the intermediate size, so compute the multiplier
        # ffn_dim_multiplier=(11008 / 2048),
        # ffn_dim_multiplier=compute_ffn_dim_multiplier(
        #     dim=2048, intermediate_size=11008, multiple_of=1024
        # ),
        ffn_dim_multiplier=1.3,
        vocab_size=256128,
        rope_theta=50000,
    ),
    "7b": TransformerModelArgs(
        dim=4096,
        n_kv_heads=8,
        # ffn_dim_multiplier=11008,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        rope_theta=10000,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=flavors,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
