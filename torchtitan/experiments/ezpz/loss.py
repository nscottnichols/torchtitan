# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.components.loss import (
    ChunkedCELoss,
    GradAccumulator,
)
from torchtitan.config import CompileConfig


class EzpzChunkedCELoss(ChunkedCELoss):
    """Chunked CE loss with ezpz MoE low-memory knobs."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        empty_cache_between_chunks: bool = False
        """Release cached device memory between chunks for tight-memory runs."""
        keep_lm_head_unsharded_between_chunks: bool = True
        """Keep FSDP lm_head unsharded across chunks to avoid repeated all-gathers."""
        sync_replicated_lm_head_grad: bool = False
        """All-reduce lm_head grads when lm_head is intentionally not FSDP-wrapped."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        super().__init__(config, compile_config=compile_config)
        self.empty_cache_between_chunks = config.empty_cache_between_chunks
        self.keep_lm_head_unsharded_between_chunks = (
            config.keep_lm_head_unsharded_between_chunks
        )
        self.sync_replicated_lm_head_grad = config.sync_replicated_lm_head_grad

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from torch.distributed._composable.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, Replicate

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            if mesh.mesh_dim_names is not None and "tp" in mesh.mesh_dim_names:
                tp_dim = mesh.mesh_dim_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_dim], Replicate):
                    placements[tp_dim] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        requires_grad = hidden_states.requires_grad

        h_detached = hidden_states.detach().requires_grad_(requires_grad)
        h_chunks = [
            c.contiguous().detach().requires_grad_(requires_grad)
            for c in torch.chunk(h_detached, num_chunks, dim=1)
        ]
        label_chunks = torch.chunk(labels, num_chunks, dim=1)

        grad_accumulator = GradAccumulator(
            h_detached,
            num_chunks=num_chunks,
            dtype=torch.float32,
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        keep_lm_head_unsharded = (
            fsdp_enabled and self.keep_lm_head_unsharded_between_chunks
        )
        if keep_lm_head_unsharded:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)
        elif fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)

        last_idx = len(h_chunks) - 1
        for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
            if keep_lm_head_unsharded and i == last_idx:
                lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                    True, recurse=False
                )

            h_chunk_for_lm_head = h_chunk
            if fsdp_enabled and not isinstance(h_chunk_for_lm_head, DTensor):
                lm_head_mesh = None
                try:
                    lm_head_state = lm_head._get_fsdp_state()
                    lm_head_param_group = lm_head_state._fsdp_param_group
                    if lm_head_param_group is not None:
                        lm_head_mesh = lm_head_param_group.mesh_info.mesh
                except Exception:
                    lm_head_mesh = None

                lm_head_param = getattr(lm_head, "weight", None)
                if not isinstance(lm_head_param, DTensor):
                    try:
                        lm_head_param = next(lm_head.parameters())
                    except StopIteration:
                        lm_head_param = None
                if isinstance(lm_head_param, DTensor):
                    lm_head_mesh = lm_head_param.device_mesh
                    placements = tuple(Replicate() for _ in lm_head_param.placements)
                elif lm_head_mesh is not None:
                    placements = tuple(Replicate() for _ in range(lm_head_mesh.ndim))
                else:
                    placements = None

                if lm_head_mesh is not None and placements is not None:
                    h_chunk_for_lm_head = DTensor.from_local(
                        h_chunk,
                        lm_head_mesh,
                        placements,
                        run_check=False,
                    )

            logits = lm_head(h_chunk_for_lm_head)
            if isinstance(logits, DTensor) and not isinstance(label_chunk, DTensor):
                label_chunk = DTensor.from_local(
                    label_chunk,
                    logits.device_mesh,
                    tuple(Replicate() for _ in logits.placements),
                    run_check=False,
                )

            chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()

            if requires_grad:
                chunk_loss.backward()
                h_chunk_grad = h_chunk.grad
                if h_chunk_grad is None and isinstance(h_chunk_for_lm_head, DTensor):
                    h_chunk_grad = h_chunk_for_lm_head.grad
                assert h_chunk_grad is not None
                grad_accumulator.add(h_chunk_grad)
                h_chunk.grad = None
                if isinstance(h_chunk_for_lm_head, DTensor):
                    h_chunk_for_lm_head.grad = None

            if self.empty_cache_between_chunks:
                del logits, chunk_loss
                if hidden_states.device.type == "xpu":
                    torch.xpu.empty_cache()
                elif hidden_states.device.type == "cuda":
                    torch.cuda.empty_cache()

        if keep_lm_head_unsharded:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()
        elif fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
        elif self.sync_replicated_lm_head_grad and requires_grad:
            self._sync_lm_head_grad(lm_head)
        if not requires_grad:
            return total_loss

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        return self._gradient_backprop(
            hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled
        )

    @staticmethod
    def _sync_lm_head_grad(lm_head: nn.Module) -> None:
        from torch.distributed.tensor import DTensor

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        for param in lm_head.parameters():
            if param.grad is None or isinstance(param.grad, DTensor):
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
