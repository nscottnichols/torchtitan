# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the ezpz/moe model.

This is the moe mirror of `torchtitan.models.deepseek_v3.parallelize`. It
uses the new config-based DTensor sharding API for the non-MoE path: TP
on attention/norms/dense-FFN is applied via `model.parallelize(tp_mesh)`,
which reads `sharding_config` declarations filled in by
`moeModel.Config.update_from_config`.

MoE expert/router TP and EP are still applied at parallelize-time by
`apply_moe_ep_tp` — that mirrors upstream deepseek_v3, where
`set_deepseek_v3_sharding_config` also leaves the MoE block alone.

Differences vs upstream `parallelize_deepseekv3`:

- `disable_fsdp_gradient_division` enables
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends
  (CCL on XPU). Upstream's version only sets the divide factor.
- `apply_compile`: upstream uses fullgraph=True via `apply_compile_sparse`,
  which fails on XPU (MoE routing's dynamic shapes). We compile each
  block with `block.compile(backend=...)` (no fullgraph).
- `apply_fsdp` is inlined locally to avoid importing
  `ShardPlacementResult`, which doesn't exist in Aurora's PyTorch. Also
  adds a Shard(0) fallback when expert hidden dim isn't divisible by the
  FSDP world size.
"""

import os
from collections.abc import Sequence
from functools import partial
from typing import Any

import ezpz
import ezpz.distributed
import torch
import torch.distributed
import torch.nn as nn
from ezpz.models import summarize_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import (
    distribute_module,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInputOutput,
    RowwiseParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.expert_parallel import TensorParallel
from torchtitan.experiments.ezpz.moe.expert_parallel import ExpertParallel
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.experiments.ezpz.moe import moeModel
from torchtitan.tools.logging import logger


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


class ColwiseParallelWithGradPlacement(ColwiseParallel):
    """ColwiseParallel with fallback for wheels lacking grad_placements."""

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
        local_input_grad_placements: Sequence[Placement] | None = None,
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )
        self.local_input_grad_placements = local_input_grad_placements

    @staticmethod
    # pyrefly: ignore [bad-param-name-override]
    def _prepare_input_fn(
        input_layouts,
        desired_input_layouts,
        local_input_grad_placements,
        mod,
        inputs,
        device_mesh,
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert local_input_grad_placements is not None, (
                "local_input_grad_placements must be specified when input is a "
                "plain tensor."
            )
            try:
                input_tensor = DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    input_layouts,
                    run_check=False,
                    grad_placements=local_input_grad_placements,
                )
            except TypeError:
                input_tensor = DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    input_layouts,
                    run_check=False,
                )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallelWithGradPlacement currently only supports "
                "nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn,  # pyrefly: ignore [bad-argument-type]
                self.input_layouts,
                self.desired_input_layouts,
                self.local_input_grad_placements,
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


def maybe_disable_fsdp_backward_prefetch(model: nn.Module) -> None:
    """Disable FSDP2's default reverse-order backward prefetch when requested.

    FSDP2 enables the default backward prefetch when no explicit prefetch list is
    configured. Passing each FSDP module itself as the explicit target suppresses
    the default next-module prefetch; after the module's own pre-backward unshard
    has completed, that explicit self-prefetch is a no-op.
    """
    if not _env_flag_enabled("TT_MOE_FSDP_DISABLE_BACKWARD_PREFETCH"):
        return

    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        logger.warning(
            "TT_MOE_FSDP_DISABLE_BACKWARD_PREFETCH=1 ignored: FSDPModule unavailable"
        )
        return

    updated = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_modules_to_backward_prefetch([module])
            updated += 1

    logger.info(
        "Configured FSDP backward prefetch self-target no-op for %d modules",
        updated,
    )


def maybe_checkpoint_attention_only(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> None:
    """Checkpoint only attention modules when requested.

    Full-block checkpointing is unsafe for normal MoE routing because recompute
    can produce dynamic routing tensors with different metadata. Attention-only
    checkpointing avoids MoE dispatch/collectives while reducing the largest
    dense activation footprint in the 8192-sequence full trainer.
    """
    if not _env_flag_enabled("TT_MOE_CHECKPOINT_ATTENTION_ONLY"):
        return

    updated = 0
    for block in getattr(model, "layers", {}).values():
        attention = getattr(block, "attention", None)
        if attention is None:
            continue
        block.attention = checkpoint_wrapper(
            attention,
            preserve_rng_state=ac_config.preserve_rng_state,
        )
        updated += 1

    logger.info("Applied attention-only activation checkpointing to %d modules", updated)


def maybe_checkpoint_moe_only(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> None:
    """Checkpoint only MoE modules when requested.

    This targets the grad-enabled normal-routing MoE stack without also wrapping
    attention, FSDP, lm_head, or optimizer behavior. It is intentionally env
    gated while the XPU/XCCL activation-growth failure is being isolated.
    """
    if not _env_flag_enabled("TT_MOE_CHECKPOINT_MOE_ONLY"):
        return

    checkpoint_impl = (
        CheckpointImpl.REENTRANT
        if _env_flag_enabled("TT_MOE_CHECKPOINT_MOE_REENTRANT")
        else CheckpointImpl.NO_REENTRANT
    )
    updated = 0
    for block in getattr(model, "layers", {}).values():
        if not getattr(block, "moe_enabled", False):
            continue
        moe = getattr(block, "moe", None)
        if moe is None:
            continue
        block.moe = checkpoint_wrapper(
            moe,
            checkpoint_impl=checkpoint_impl,
            preserve_rng_state=ac_config.preserve_rng_state,
        )
        updated += 1

    logger.info(
        "Applied MoE-only activation checkpointing to %d modules (impl=%s)",
        updated,
        checkpoint_impl.name,
    )


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division and (on XPU/CCL) force
    sum reduction for cross-rank gradient comms.

    On NCCL the default reduce-mean works correctly. On CCL (XPU) we need
    SUM and divide ourselves to avoid losing precision.
    """
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = ezpz.distributed.get_torch_backend() or str(
            torch.distributed.get_backend()
        )
        if backend and "nccl" not in str(backend).lower():
            force_sum_reduction = True

    fsdp_modules_updated = 0
    for module in model.modules():
        # Be resilient to FSDPModule class location changes across PyTorch
        # releases by going through the public method.
        set_divide_factor = getattr(module, "set_gradient_divide_factor", None)
        if callable(set_divide_factor):
            set_divide_factor(1.0)
            fsdp_modules_updated += 1
            if force_sum_reduction:
                set_force_sum = getattr(
                    module, "set_force_sum_reduction_for_comms", None
                )
                if callable(set_force_sum):
                    set_force_sum(True)

    logger.info(
        "Configured FSDP gradient division for %d modules (force_sum_reduction=%s)",
        fsdp_modules_updated,
        force_sum_reduction,
    )


def parallelize_moe(
    model: moeModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply CP + TP + EP + AC + compile + FSDP to the moe model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Context Parallel with Tensor Parallel is not yet supported "
                "for DeepSeek-V3. "
                "See https://github.com/pytorch/torchtitan/issues/2446"
            )
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # TP via the config-based sharding API. The model's sharding_config
    # declarations were filled in by update_from_config (see model.py).
    # MoE blocks are intentionally not handled here — apply_moe_ep_tp
    # below does that (mirrors upstream deepseek_v3).
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # EP/TP for MoE blocks.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
    else:
        maybe_checkpoint_attention_only(model, ac_config)
        maybe_checkpoint_moe_only(model, ac_config)

    if model_compile_enabled:
        # Upstream apply_compile_sparse uses fullgraph=True which fails on
        # XPU after 00b7f569 removed maybe_enable_amp — MoE routing's
        # dynamic shapes cause recompilation that fullgraph=True forbids.
        # Apply compile per-block without fullgraph instead.
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        for layer_id, block in model.layers.named_children():
            block.compile(backend=compile_config.backend)
            model.layers.register_module(layer_id, block)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    logger.info(f"\n+{summarize_model(model)}")

    return model


# ---------------------------------------------------------------------------
# Inlined from torchtitan.models.llama4.parallelize to avoid importing
# ShardPlacementResult, which doesn't exist in Aurora's PyTorch framework
# release. Also adds a Shard(0) fallback when the expert hidden dim isn't
# divisible by the FSDP world size — upstream's version assumes divisibility
# and crashes otherwise.
# ---------------------------------------------------------------------------


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    replicate_lm_head = os.environ.get("TT_MOE_REPLICATE_LM_HEAD", "0") == "1"
    if model.norm is not None and model.lm_head is not None:
        if replicate_lm_head:
            logger.info(
                "Leaving lm_head replicated because TT_MOE_REPLICATE_LM_HEAD=1"
            )
            fully_shard(
                model.norm,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )
        else:
            fully_shard(
                [model.norm, model.lm_head],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )
    elif model.norm is not None:
        fully_shard(
            model.norm,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    for layer_id, transformer_block in model.layers.items():
        if transformer_block.moe_enabled:
            assert hasattr(transformer_block, "moe")
            expert_params = set(transformer_block.moe.experts.parameters())
            num_experts = transformer_block.moe.experts.num_experts

            if ep_degree > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            else:
                efsdp_ep_size = fsdp_config["mesh"].size()

            # Shard(1) shards the hidden dim instead of expert dim when
            # there are more FSDP ranks than experts. But this requires
            # the hidden dim to be evenly divisible by the world size.
            # Fall back to Shard(0) if not (avoids uneven sharding error).
            if efsdp_ep_size > num_experts:
                expert_w = next(iter(transformer_block.moe.experts.parameters()))
                if expert_w.shape[1] % efsdp_ep_size == 0:
                    expert_shard_placement = Shard(1)
                else:
                    expert_shard_placement = Shard(0)
            else:
                expert_shard_placement = Shard(0)

            if ep_degree == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_degree == 1:

                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_degree > 1: prefer the per-param mesh API when this
                # PyTorch build provides it. Some XPU wheels only accept a
                # Shard return from shard_placement_fn; for those, shard the
                # experts bottom-up on edp_mesh, then shard the remaining
                # block parameters on dp_mesh.
                assert edp_mesh is not None

                try:
                    from torch.distributed.fsdp._fully_shard._fsdp_common import (
                        FSDPMeshInfo,
                        ShardPlacementResult,
                    )
                except ImportError:

                    def _expert_shard_placement_fn(
                        param: nn.Parameter,
                        _expert_placement: Shard = expert_shard_placement,
                    ) -> Shard:
                        return _expert_placement

                    expert_fsdp_config = {**fsdp_config, "mesh": edp_mesh}
                    fully_shard(
                        transformer_block.moe.experts,
                        **expert_fsdp_config,
                        reshard_after_forward=reshard_after_forward,
                        shard_placement_fn=_expert_shard_placement_fn,
                    )
                    fully_shard(
                        transformer_block,
                        **fsdp_config,
                        reshard_after_forward=reshard_after_forward,
                    )
                else:
                    edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
                    dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

                    def _shard_placement_fn(
                        param: nn.Parameter,
                        _expert_params: set = expert_params,
                        _expert_placement: Shard = expert_shard_placement,
                        _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                        _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                    ) -> ShardPlacementResult:
                        if param in _expert_params:
                            return ShardPlacementResult(
                                placement=_expert_placement, mesh_info=_edp_mesh_info
                            )
                        return ShardPlacementResult(
                            placement=Shard(0), mesh_info=_dp_mesh_info
                        )

                    fully_shard(
                        transformer_block,
                        **fsdp_config,
                        reshard_after_forward=reshard_after_forward,
                        shard_placement_fn=_shard_placement_fn,
                    )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)
    maybe_disable_fsdp_backward_prefetch(model)
    disable_fsdp_gradient_division(model)


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
):
    """Apply MoE expert/tensor parallelism plans to MoE-enabled blocks.

    Same plan structure as upstream `llama4.parallelize.apply_moe_ep_tp`,
    minus the DeepEP/HybridEP token-dispatcher plumbing (we don't use those
    backends on Aurora). Token dispatching for the standard backend is
    handled internally by the LocalTokenDispatcher class at model build.
    """
    assert ep_mesh is not None or tp_mesh is not None

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                    use_local_output=False,
                ),
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            if transformer_block.moe.shared_experts is not None:
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial(),
                        ),
                        "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        # EP disabled: shard routed expert weights across TP mesh.
        # EP enabled: shard across EP mesh (ETP deprecated upstream — see #3167).
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
