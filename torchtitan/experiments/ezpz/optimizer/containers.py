from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.ezpz.optimizer.adopt import ADOPT
from torchtitan.experiments.ezpz.optimizer.mano import Mano
from torchtitan.experiments.ezpz.optimizer.muon import Muon, MuonClip, QKInputRecorder
from torchtitan.experiments.ezpz.optimizer.sophia import SophiaG
from torchtitan.experiments.ezpz.optimizer.spam import SPAM

__all__ = [
    "ADOPTOptimizersContainer",
    "ManoOptimizersContainer",
    "MuonClipOptimizersContainer",
    "MuonOptimizersContainer",
    "SPAMOptimizersContainer",
    "SophiaGOptimizersContainer",
]


class ADOPTOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "ADOPT"
        clip_lambda_power: float = 0.25
        decouple: bool = False

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return ADOPT

    @staticmethod
    def _build_optimizer_kwargs(config: ADOPTOptimizersContainer.Config) -> dict[str, Any]:
        if config.implementation == "fused":
            raise ValueError(
                "ADOPT does not support fused implementation. "
                "Use 'foreach' or 'for-loop' instead."
            )
        clip_lambda: Callable[[int], float] | None = None
        if config.clip_lambda_power > 0:
            power = config.clip_lambda_power

            def clip_lambda(step: int, _power: float = power) -> float:
                return step**_power

        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "decouple": config.decouple,
            "clip_lambda": clip_lambda,
            "foreach": config.implementation == "foreach",
        }


class SophiaGOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "SophiaG"
        rho: float = 0.04

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return SophiaG

    @staticmethod
    def _build_optimizer_kwargs(config: SophiaGOptimizersContainer.Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "rho": config.rho,
            "weight_decay": config.weight_decay,
        }

    def update_hessian(self) -> None:
        """Delegate hessian update to each inner SophiaG optimizer."""
        for optimizer in self.optimizers:
            optimizer.update_hessian()


class MuonOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "Muon"
        beta1: float = 0.95
        beta2: float = 0.95
        momentum: float = 0.95
        nesterov: bool = True
        ns_steps: int = 5
        adamw_eps: float = 1e-8

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return Muon

    @staticmethod
    def _build_optimizer_kwargs(config: MuonOptimizersContainer.Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "wd": config.weight_decay,
            "momentum": config.momentum,
            "nesterov": config.nesterov,
            "ns_steps": config.ns_steps,
            "adamw_betas": (config.beta1, config.beta2),
            "adamw_eps": config.adamw_eps,
        }


class MuonClipOptimizersContainer(MuonOptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(MuonOptimizersContainer.Config):
        name: str = "MuonClip"
        clip_t: float = 100.0
        clip_alpha: float = 0.5
        use_sqrt_d: bool = True

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return MuonClip

    @staticmethod
    def _build_optimizer_kwargs(
        config: MuonClipOptimizersContainer.Config,
    ) -> dict[str, Any]:
        base = MuonOptimizersContainer._build_optimizer_kwargs(config)
        base.update(
            {
                "qk_clip": True,
                "clip_t": config.clip_t,
                "alpha": config.clip_alpha,
                "use_sqrt_d": config.use_sqrt_d,
            }
        )
        return base


class ManoOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "Mano"
        momentum: float = 0.95

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return Mano

    @staticmethod
    def _build_optimizer_kwargs(config: ManoOptimizersContainer.Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "adamw_betas": (config.beta1, config.beta2),
            "adamw_eps": config.eps,
        }


class SPAMOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "SPAM"
        spike_threshold: float = 2.0
        delta_t: int = 100
        ema_beta: float = 0.999

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return SPAM

    @staticmethod
    def _build_optimizer_kwargs(config: SPAMOptimizersContainer.Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "spike_threshold": config.spike_threshold,
            "delta_t": config.delta_t,
            "ema_beta": config.ema_beta,
        }


def register_muonclip_qk_pairs(
    optimizer: MuonClipOptimizersContainer,
    model_parts: list[nn.Module],
    *,
    q_attr: str = "wq",
    k_attr: str = "wk",
    d_head: int | None = None,
) -> list[QKInputRecorder]:
    """Register QK pairs for MuonClip post-step clipping.

    Call this as a ``post_optimizer_build_fn`` in model specs. Iterates over
    all attention modules in ``model_parts`` and registers Q/K weight pairs
    with the MuonClip optimizers.

    Returns the list of QKInputRecorder instances (keep them alive).
    """
    recorders: list[QKInputRecorder] = []
    for model in model_parts:
        for inner_opt in optimizer.optimizers:
            if not isinstance(inner_opt, MuonClip):
                continue
            for module in model.modules():
                if hasattr(module, q_attr) and hasattr(module, k_attr):
                    recorder = inner_opt.attach_to_attention(
                        module,
                        q_attr=q_attr,
                        k_attr=k_attr,
                        d_head=d_head,
                    )
                    recorders.append(recorder)
    return recorders
