from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.ezpz.optimizer.adopt import ADOPT
from torchtitan.experiments.ezpz.optimizer.mano import Mano
from torchtitan.experiments.ezpz.optimizer.muon import Muon, MuonClip, QKInputRecorder
from torchtitan.experiments.ezpz.optimizer.sophia import SophiaG
from torchtitan.experiments.ezpz.optimizer.schedule_free import AdamWScheduleFree
from torchtitan.experiments.ezpz.optimizer.spam import SPAM

__all__ = [
    "ADOPTOptimizersContainer",
    "ManoOptimizersContainer",
    "MuonClipOptimizersContainer",
    "MuonOptimizersContainer",
    "SPAMOptimizersContainer",
    "ScheduleFreeOptimizersContainer",
    "SophiaGOptimizersContainer",
    "TorchMuonOptimizersContainer",
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


class _CompositeOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers into a single Optimizer interface.

    Needed because OptimizersContainer expects one optimizer per model part,
    but torch.optim.Muon only handles 2D params (need a separate AdamW for
    embeddings/head).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        # Skip Optimizer.__init__ — it rejects empty param lists.
        # We manage param_groups and state via the inner optimizers.
        self._optimizers = optimizers
        self.defaults = {}
        self.state = {}
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
            self.state.update(opt.state)

    def step(self, closure=None):
        loss = None
        for opt in self._optimizers:
            result = opt.step(closure)
            if result is not None:
                loss = result
        return loss

    def zero_grad(self, *args, **kwargs):
        for opt in self._optimizers:
            opt.zero_grad(*args, **kwargs)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self._optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self._optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


class TorchMuonOptimizersContainer(OptimizersContainer):
    """Uses torch.optim.Muon (built-in, optimized) for 2D hidden layers
    and torch.optim.AdamW for embeddings/head/1D params.

    Much faster than the custom Muon implementation — benefits from
    PyTorch's fused kernels and Gram Newton-Schulz optimizations.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "TorchMuon"
        momentum: float = 0.95
        nesterov: bool = True
        ns_steps: int = 5
        adamw_lr_factor: float = 1.0

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        import torch.optim

        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        for model in model_parts:
            muon_params = []
            adamw_params = []
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2 and max(p.shape) <= 10000:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            inner_opts = []
            if muon_params:
                inner_opts.append(torch.optim.Muon(
                    muon_params,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    momentum=config.momentum,
                    nesterov=config.nesterov,
                    ns_steps=config.ns_steps,
                ))
            if adamw_params:
                inner_opts.append(torch.optim.AdamW(
                    adamw_params,
                    lr=config.lr * config.adamw_lr_factor,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2),
                    eps=config.eps,
                ))

            if not inner_opts:
                # Empty model part (FSDP sharding) — use a no-op AdamW
                inner_opts.append(torch.optim.AdamW([{"params": []}], lr=config.lr))

            self.optimizers.append(_CompositeOptimizer(inner_opts))
            all_params.extend(muon_params)
            all_params.extend(adamw_params)

        optimizer_kwargs = {"lr": config.lr, "weight_decay": config.weight_decay}
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


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


class ScheduleFreeOptimizersContainer(OptimizersContainer):
    """Schedule-Free AdamW — no LR schedule needed.

    Requires .train() before training and .eval() before evaluation/checkpointing.
    Based on: https://github.com/facebookresearch/schedule_free
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "ScheduleFree"
        warmup_steps: int = 200
        r: float = 0.0
        weight_lr_power: float = 2.0

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return AdamWScheduleFree

    @staticmethod
    def _build_optimizer_kwargs(config: ScheduleFreeOptimizersContainer.Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "r": config.r,
            "weight_lr_power": config.weight_lr_power,
        }

    def train_mode(self) -> None:
        """Switch all optimizers to train mode."""
        for optimizer in self.optimizers:
            optimizer.train()

    def eval_mode(self) -> None:
        """Switch all optimizers to eval mode."""
        for optimizer in self.optimizers:
            optimizer.eval()


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
