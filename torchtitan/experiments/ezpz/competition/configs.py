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

# Fixed competition parameters — same for all configs
DATASET = "HuggingFaceFW/fineweb-edu"
LOCAL_BATCH_SIZE = 2
SEQ_LEN = 8192
STEPS = 1000


def _speedrun_base(
    optimizer: str = "adamw",
    lr: float = 1.3e-3,
    **optimizer_kwargs,
):
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

    # LR schedule: WSD (warmup 20, stable to 800, decay last 200)
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.2
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0

    # Optimizer
    cfg.optimizer.name = optimizer
    cfg.optimizer.lr = lr
    for k, v in optimizer_kwargs.items():
        setattr(cfg.optimizer, k, v)

    return cfg


# ---- Competition configs ----


def speedrun_2b_adamw():
    """AdamW baseline — LR from LR finder (1.3e-3)."""
    return _speedrun_base(optimizer="adamw", lr=1.3e-3)


def speedrun_2b_muon():
    """Muon — best NanoGPT speedrun optimizer. LR from LR finder (2.4e-3)."""
    return _speedrun_base(optimizer="muon", lr=2.4e-3)


def speedrun_2b_sophiag():
    """SophiaG — second-order Hessian approx. LR from LR finder (3.1e-4)."""
    return _speedrun_base(optimizer="sophiag", lr=3.1e-4)


def speedrun_2b_muon_aggressive():
    """Muon with 2x LR — pushing convergence speed."""
    return _speedrun_base(optimizer="muon", lr=4.8e-3)


def speedrun_2b_adamw_high_lr():
    """AdamW with 2x LR — testing upper bound."""
    return _speedrun_base(optimizer="adamw", lr=2.6e-3)
