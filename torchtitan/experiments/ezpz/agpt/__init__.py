from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ft.diloco import fragment_llm
from torchtitan.models.common import (
    FeedForward,
    GQAttention,
    RoPE,
    compute_ffn_hidden_dim,
)
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import FaultTolerantModelSpec

__all__ = [
    "model_registry",
    "parallelize_llama",
]


def _build_llama3_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
) -> Llama3Model.Config:
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(hidden_dim=hidden_dim),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=rope_theta,
            backend="complex",
            scaling="llama",
        ),
    )


agpt_configs = {
    "llama3_debugmodel": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            # TODO: find better ways to enforce dim = decoder dim // n_heads, for all models
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "llama3_debugmodel_flex_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
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
    "llama3_debugmodel_varlen_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
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
    "debug": _build_llama3_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=2048,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
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
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "50B": _build_llama3_config(
        dim=8192,
        n_layers=56,
        n_heads=64,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(
            8192, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
}
agpt_configs["debugmodel"] = agpt_configs["debug"]
agpt_configs["2b"] = agpt_configs["2B"]
agpt_configs["7b"] = agpt_configs["7B"]
agpt_configs["8b"] = agpt_configs["8B"]
agpt_configs["20b"] = agpt_configs["20B"]
agpt_configs["50b"] = agpt_configs["50B"]


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
