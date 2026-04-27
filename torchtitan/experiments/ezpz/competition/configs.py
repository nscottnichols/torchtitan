# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Speedrun configs for the agpt_2b loss competition.
#
# Goal: lowest loss in 1000 steps on 2 nodes (24 XPU tiles).
# Fixed: dataset=HuggingFaceFW/fineweb-edu, LBS=2, seq_len=8192.
#
# Usage:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
#       --module ezpz.agpt --config speedrun_2b_muon

import torchtitan.experiments.ezpz.datasets  # noqa: F401 — register HF datasets

from torchtitan.experiments.ezpz.agpt.config_registry import agpt
from torchtitan.experiments.ezpz.optimizer import (
    ManoOptimizersContainer,
    MuonOptimizersContainer,
    SPAMOptimizersContainer,
    SophiaGOptimizersContainer,
    TorchMuonOptimizersContainer,
)

# Fixed competition parameters — same for all configs
DATASET = "HuggingFaceFW/fineweb-edu"
LOCAL_BATCH_SIZE = 2
SEQ_LEN = 8192
STEPS = 1000


def _speedrun_base():
    """Base speedrun config: 1000 steps, WSD schedule, fineweb-edu streaming.

    Fixed parameters (not tunable for fairness):
    - Dataset: HuggingFaceFW/fineweb-edu (streaming)
    - LBS: 2 (GBS=48 on 24 tiles)
    - seq_len: 8192
    - Steps: 1000

    Tunable: optimizer, LR, LR schedule, gradient clipping, etc.
    """
    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )

    # Dataset: stream from HF (no local data needed)
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None

    # Training: 1000 steps
    cfg.training.steps = STEPS

    # No checkpointing for speedruns
    cfg.checkpoint.enable = False

    # LR schedule: WSD (warmup 20, stable to 800, decay last 200)
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0

    return cfg


# ---- Competition configs ----


def speedrun_2b_adamw():
    """AdamW baseline — LR from LR finder (1.3e-3)."""
    cfg = _speedrun_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw"
    return cfg


def speedrun_2b_muon():
    """Muon — best NanoGPT speedrun optimizer. LR from LR finder (2.4e-3)."""
    cfg = _speedrun_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon"
    return cfg


def speedrun_2b_sophiag():
    """SophiaG — second-order Hessian approx. LR from LR finder (3.1e-4)."""
    cfg = _speedrun_base()
    cfg.optimizer = SophiaGOptimizersContainer.Config(lr=3.1e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_sophiag"
    return cfg


def speedrun_2b_muon_aggressive():
    """Muon with 2x LR — pushing convergence speed."""
    cfg = _speedrun_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=4.8e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_aggressive"
    return cfg


def speedrun_2b_adamw_high_lr():
    """AdamW with 2x LR — testing upper bound."""
    cfg = _speedrun_base()
    cfg.optimizer.lr = 2.6e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_high_lr"
    return cfg


# ---- Hparam tweak configs ----


def speedrun_2b_adamw_short_decay():
    """AdamW with 10% decay (vs 20%) — more time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_short_decay"
    return cfg


def speedrun_2b_adamw_cosine():
    """AdamW with cosine decay instead of linear."""
    cfg = _speedrun_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_cosine"
    return cfg


def speedrun_2b_adamw_fast_warmup():
    """AdamW with 5-step warmup + 10% decay — max time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_fast_warmup"
    return cfg


def speedrun_2b_muon_short_decay():
    """Muon with 10% decay — more time at peak LR."""
    cfg = _speedrun_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_short_decay"
    return cfg


def speedrun_2b_muon_fast_warmup():
    """Muon with 5-step warmup + 10% decay."""
    cfg = _speedrun_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.1
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_fast_warmup"
    return cfg


# ---- New optimizer configs ----


def speedrun_2b_mano():
    """Mano — manifold-normalized optimizer, 1.75x faster than Muon."""
    cfg = _speedrun_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano"
    return cfg


def speedrun_2b_spam():
    """SPAM — spike-aware Adam with momentum reset."""
    cfg = _speedrun_base()
    cfg.optimizer = SPAMOptimizersContainer.Config(lr=1.3e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_spam"
    return cfg


# ---- Architecture tweak configs ----


def _speedrun_qknorm_base():
    """Base config with QK-Norm enabled."""
    cfg = agpt(
        "2b_qknorm",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )
    cfg.dataloader.dataset = DATASET
    cfg.dataloader.dataset_path = None
    cfg.training.steps = STEPS
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0
    return cfg


def speedrun_2b_adamw_qknorm():
    """AdamW + QK-Norm — stabilizes early attention training."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_adamw_qknorm"
    return cfg


def speedrun_2b_muon_qknorm():
    """Muon + QK-Norm — best optimizer + attention stabilization."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_qknorm"
    return cfg


# ---- Round 2: combo configs based on round 1 findings ----


def speedrun_2b_mano_high_lr():
    """Mano with higher LR (6e-4) — try to close gap to Muon."""
    cfg = _speedrun_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=6.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_high_lr"
    return cfg


def speedrun_2b_mano_1e3():
    """Mano with LR=1e-3 — aggressive push."""
    cfg = _speedrun_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=1.0e-3)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_1e3"
    return cfg


def speedrun_2b_muon_cosine():
    """Muon + cosine decay — combine best optimizer with best schedule."""
    cfg = _speedrun_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_muon_cosine"
    return cfg


def speedrun_2b_mano_cosine():
    """Mano + cosine decay — fast optimizer with best schedule."""
    cfg = _speedrun_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_cosine"
    return cfg


def speedrun_2b_mano_qknorm():
    """Mano + QK-Norm — fast manifold optimizer with attention stabilization."""
    cfg = _speedrun_qknorm_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/speedrun_2b_mano_qknorm"
    return cfg


# ---- torch.optim.Muon (built-in, optimized) ----


def speedrun_2b_torchmuon():
    """torch.optim.Muon — official PyTorch implementation, much faster per-step."""
    cfg = _speedrun_base()
    cfg.optimizer = TorchMuonOptimizersContainer.Config(lr=2.4e-3)
    return cfg


def speedrun_2b_torchmuon_cosine():
    """torch.optim.Muon + cosine decay."""
    cfg = _speedrun_base()
    cfg.optimizer = TorchMuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.lr_scheduler.decay_type = "cosine"
    return cfg


# ---- Architecture tweak speedruns (round 3) ----


def _speedrun_variant_base(variant: str):
    """Base speedrun using a model variant (softcap, relu2, kitchen_sink)."""
    cfg = agpt(
        variant,
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=STEPS,
    )
    cfg.dataloader.dataset = DATASET_LOCAL
    cfg.dataloader.dataset_path = None
    cfg.training.steps = STEPS
    cfg.checkpoint.enable = False
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.0
    return cfg


def speedrun_2b_softcap():
    """AdamW + logit softcapping at 30.0 (Gemma 2 style)."""
    cfg = _speedrun_variant_base("2b_softcap")
    cfg.optimizer.lr = 1.3e-3
    return cfg


def speedrun_2b_relu2():
    """AdamW + ReLU-squared activation in FFN (NanoGPT speedrun)."""
    cfg = _speedrun_variant_base("2b_relu2")
    cfg.optimizer.lr = 1.3e-3
    return cfg


def speedrun_2b_kitchen_sink():
    """QK-Norm + logit softcap + ReLU² — everything combined."""
    cfg = _speedrun_variant_base("2b_kitchen_sink")
    cfg.optimizer.lr = 1.3e-3
    return cfg


def speedrun_2b_mano_kitchen_sink():
    """Mano + QK-Norm + logit softcap + ReLU² — best optimizer + all tweaks."""
    cfg = _speedrun_variant_base("2b_kitchen_sink")
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    return cfg


# ---- Full training configs (10B tokens, 8 nodes, local dataset) ----
#
# 8 nodes = 96 tiles, LBS=2, GAS=2 → GBS=384
# 10B tokens / (384 * 8192) = ~3,180 steps
# ~4 hours at AdamW/Mano speed (~7,200 TPS/GPU)

DATASET_LOCAL = "fineweb_edu_local"
TOKENS_10B = 10_000_000_000
TILES_8N = 96
GAS = 2


def _full_train_base():
    """Base config for 10B token training on 8 nodes.

    Uses locally cached FineWeb-Edu for reproducibility.
    GBS = 96 tiles × LBS=2 × GAS=2 = 384.
    ~3,180 steps, ~4 hours at AdamW speed.
    """
    gbs = TILES_8N * LOCAL_BATCH_SIZE * GAS
    tokens_per_step = gbs * SEQ_LEN
    steps = TOKENS_10B // tokens_per_step

    cfg = agpt(
        "2b",
        local_batch_size=LOCAL_BATCH_SIZE,
        activation_checkpoint_mode="none",
        seq_len=SEQ_LEN,
        compile=True,
        checkpoint_interval=500,
    )

    cfg.dataloader.dataset = DATASET_LOCAL
    cfg.dataloader.dataset_path = None
    cfg.training.steps = steps
    cfg.training.global_batch_size = gbs

    # WSD: warmup 2%, stable, cosine decay last 20%
    warmup = max(steps // 50, 10)
    cfg.lr_scheduler.warmup_steps = warmup
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.0

    cfg.checkpoint.enable = True

    return cfg


def full_2b_adamw():
    """AdamW baseline, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer.lr = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/full_2b_adamw"
    return cfg


def full_2b_adamw_qknorm():
    """AdamW + QK-Norm — wall-clock champion, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.model_spec = agpt("2b_qknorm").model_spec
    cfg.optimizer.lr = 1.3e-3
    cfg.checkpoint.folder = "checkpoints/full_2b_adamw_qknorm"
    return cfg


def full_2b_muon():
    """Muon — best loss optimizer, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer = MuonOptimizersContainer.Config(lr=2.4e-3)
    cfg.checkpoint.folder = "checkpoints/full_2b_muon"
    return cfg


def full_2b_mano():
    """Mano — fast manifold optimizer, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/full_2b_mano"
    return cfg


def full_2b_mano_qknorm():
    """Mano + QK-Norm — best combo, 10B tokens, 8 nodes."""
    cfg = _full_train_base()
    cfg.model_spec = agpt("2b_qknorm").model_spec
    cfg.optimizer = ManoOptimizersContainer.Config(lr=3.0e-4)
    cfg.checkpoint.folder = "checkpoints/full_2b_mano_qknorm"
    return cfg
