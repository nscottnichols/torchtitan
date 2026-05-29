# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz expert compute backends for MoE.

Subclasses upstream `GroupedExperts` to add a `compute_backend` selector
without modifying core. Two backends are supported here:

- ``"grouped_mm"`` (default): defer to upstream's ``torch._grouped_mm``
  path. Requires SM90+ on CUDA; on XPU there is no grouped-mm fallback.
- ``"for_loop"``: per-expert ``matmul`` loop. Slower but works on every
  device. Re-vendored from upstream's ``_run_experts_for_loop`` which
  was deleted in pytorch/torchtitan#3308.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts


ExpertComputeBackend = Literal["for_loop", "grouped_mm"]


def _empty_expert_output(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Zero-token expert call. Preserve a zero-gradient path through w1/w2/w3."""
    out = x.new_empty((0, w2.shape[1]))
    return out + (w1.sum() + w2.sum() + w3.sum() + x.sum()) * 0


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 0:
        return _empty_expert_output(w1, w2, w3, x)

    # NOTE: this incurs a device-host sync.
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    x_splits = torch.split(
        x,
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    return torch.cat(out_experts_splits, dim=0)


class EzpzGroupedExperts(GroupedExperts):
    """GroupedExperts variant that selects between expert compute backends.

    Defers to upstream's grouped-mm path by default. Set ``compute_backend``
    to ``"for_loop"`` on devices without grouped-mm support (e.g. XPU,
    pre-SM90 CUDA).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend: ExpertComputeBackend = config.compute_backend

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.compute_backend == "grouped_mm":
            return super()._experts_forward(x, num_tokens_per_expert)

        # Param names use Shazeer shape-suffix style post upstream PR #3425
        # (41st sync): w1_EFD, w2_EDF, w3_EFD.
        if isinstance(self.w1_EFD, DTensor):
            w1 = self.w1_EFD.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2_EDF.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3_EFD.to_local()
        else:
            w1 = self.w1_EFD
            w2 = self.w2_EDF
            w3 = self.w3_EFD

        if self.compute_backend == "for_loop":
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend!r}")
