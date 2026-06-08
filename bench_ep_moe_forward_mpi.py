#!/usr/bin/env python3
"""
MPI-launched EP microbenchmark for torchtitan.experiments.ezpz.moe.MoE.forward.

Target default: 10B_2B_sdpa-shaped MoE path on XPU/XCCL with system MPI.

Forward timing uses torch.no_grad(), not torch.inference_mode().
Forward+backward timing runs with autograd enabled.
"""

from __future__ import annotations

import argparse
import gc
import os
import socket
import statistics
import time
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.experiments.ezpz.moe.expert_parallel import ExpertParallel
from torchtitan.experiments.ezpz.moe.experts import EzpzGroupedExperts
from torchtitan.experiments.ezpz.moe.moe import MoE, TokenChoiceTopKRouter
from torchtitan.experiments.ezpz.moe.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    deterministic_scatter_add_,
    _print_moe_fastpath_counters,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    local_size: int
    device_type: str
    device: torch.device
    backend: str


def first_int_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return None


def get_mpi_env() -> tuple[int, int, int]:
    rank = first_int_env(
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PALS_RANKID",
        "SLURM_PROCID",
    )
    world_size = first_int_env(
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PALS_WORLD_SIZE",
        "SLURM_NTASKS",
    )
    local_size = first_int_env(
        "LOCAL_WORLD_SIZE",
        "CCL_LOCAL_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "I_MPI_LOCAL_SIZE",
        "PMI_LOCAL_SIZE",
        "PALS_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
    )
    return rank or 0, world_size or 1, local_size or 1


def get_local_rank(rank: int, local_size: int) -> int:
    local_rank = first_int_env(
        "LOCAL_RANK",
        "CCL_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "I_MPI_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "PALS_LOCAL_RANKID",
        "SLURM_LOCALID",
    )
    return local_rank if local_rank is not None else rank % max(1, local_size)


def resolve_backend(device_type: str, backend: str | None) -> str:
    if backend is not None:
        return backend
    return "nccl" if device_type == "cuda" else "xccl"


def initialize_mpi_kvs_if_needed(backend: str, rank: int) -> None:
    if backend != "xccl":
        return
    if os.environ.get("CCL_KVS_MODE", "").lower() != "mpi":
        return
    try:
        from mpi4py import MPI
    except ImportError as exc:
        if rank == 0:
            print(
                f"[rank0] warning: CCL_KVS_MODE=mpi but mpi4py import failed: {exc}",
                flush=True,
            )
        return

    # On this stack oneCCL's MPI KVS path calls MPI_Comm_rank during
    # init_process_group; importing mpi4py initializes the same system MPICH
    # runtime used by the PALS mpiexec launch.
    if not MPI.Is_initialized():
        MPI.Init_thread()


def set_device(local_rank: int, device_type: str) -> torch.device:
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if device_type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Requested --device xpu but torch.xpu is not available.")
        torch.xpu.set_device(local_rank)
        return torch.device("xpu", local_rank)
    raise ValueError(f"Unknown device_type={device_type}")


def setup_distributed(args: argparse.Namespace) -> DistInfo:
    rank, world_size, local_size = get_mpi_env()
    local_rank = get_local_rank(rank, local_size)
    device = set_device(local_rank, args.device)
    backend = resolve_backend(device.type, args.backend)
    initialize_mpi_kvs_if_needed(backend, rank)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Make the process also look torchrun-like for libraries that inspect env.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_size)

    if args.print_ccl_env and rank == 0:
        print("[rank0] CCL env before init:")
        for key in [
            "CCL_KVS_MODE",
            "CCL_ATL_TRANSPORT",
            "CCL_PROCESS_LAUNCHER",
            "CCL_LOCAL_RANK",
            "CCL_LOCAL_SIZE",
            "CCL_WORKER_AFFINITY",
            "FI_MR_CACHE_MONITOR",
            "ZE_FLAT_DEVICE_HIERARCHY",
        ]:
            print(f"[rank0]   {key}={os.environ.get(key)}")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    return DistInfo(rank, local_rank, world_size, local_size, device.type, device, backend)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    dist.barrier(device_ids=[device.index])


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def dtype_from_arg(dtype: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MPI-launched EP benchmark for torchtitan MoE.forward")

    p.add_argument("--model-flavor", default="10B_2B_sdpa")
    p.add_argument("--device", choices=["xpu", "cuda"], default="xpu")
    p.add_argument("--backend", choices=["xccl", "nccl", "gloo"], default=None)

    # 10B_2B / 10B_2B_sdpa MoE-layer defaults.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=1408)
    p.add_argument("--num-experts", type=int, default=36)
    p.add_argument("--num-shared-experts", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--ep-degree", type=int, default=12)

    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--score-func", choices=["softmax", "sigmoid"], default="softmax")
    p.add_argument("--score-before-experts", action="store_true", default=False)
    p.add_argument("--score-after-experts", dest="score_before_experts", action="store_false")
    p.add_argument("--route-norm", action="store_true")
    p.add_argument("--route-scale", type=float, default=1.0)
    p.add_argument("--force-load-balance", action="store_true")

    p.add_argument("--use-grouped-mm", action="store_true")
    p.add_argument("--no-shared-experts", action="store_true")
    p.add_argument("--include-ffn-norm", action="store_true")

    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--suppress-xpu-autocast-warning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-ccl-env", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timing-breakdown", action="store_true")
    p.add_argument("--timing-breakdown-iters", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def init_float_params(module: nn.Module, std: float = 0.02) -> None:
    for param in module.parameters():
        if param.is_floating_point():
            torch.nn.init.normal_(param, mean=0.0, std=std)


def maybe_shared_experts(args: argparse.Namespace) -> FeedForward.Config | None:
    if args.no_shared_experts or args.num_shared_experts <= 0:
        return None

    shared_hidden_dim = args.hidden_dim * args.num_shared_experts
    return FeedForward.Config(
        w1=Linear.Config(in_features=args.dim, out_features=shared_hidden_dim, bias=False),
        w2=Linear.Config(in_features=shared_hidden_dim, out_features=args.dim, bias=False),
        w3=Linear.Config(in_features=args.dim, out_features=shared_hidden_dim, bias=False),
    )


def build_mesh(args: argparse.Namespace, info: DistInfo) -> tuple[DeviceMesh, DeviceMesh, int, int]:
    world_size = info.world_size
    ep_degree = args.ep_degree

    if ep_degree < 1:
        raise ValueError("--ep-degree must be >= 1")
    if world_size % ep_degree != 0:
        raise ValueError(f"world_size={world_size} must be divisible by ep_degree={ep_degree}")
    if args.num_experts % ep_degree != 0:
        raise ValueError(f"num_experts={args.num_experts} must be divisible by ep_degree={ep_degree}")

    dp_degree = world_size // ep_degree
    if dp_degree == 1:
        mesh = init_device_mesh(
            device_type=info.device_type,
            mesh_shape=(ep_degree,),
            mesh_dim_names=("ep",),
        )
        ep_mesh = mesh
    else:
        mesh = init_device_mesh(
            device_type=info.device_type,
            mesh_shape=(dp_degree, ep_degree),
            mesh_dim_names=("dp", "ep"),
        )
        ep_mesh = mesh["ep"]

    return mesh, ep_mesh, dp_degree, ep_degree


def build_moe(args: argparse.Namespace, info: DistInfo, ep_mesh: DeviceMesh, dtype: torch.dtype) -> MoE:
    moe_cfg = MoE.Config(
        num_experts=args.num_experts,
        experts=EzpzGroupedExperts.Config(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.num_experts,
            compute_backend="grouped_mm" if args.use_grouped_mm else "for_loop",
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=args.num_experts,
                top_k=args.top_k,
                score_before_experts=args.score_before_experts,
                force_load_balance=args.force_load_balance,
            ),
        ),
        router=TokenChoiceTopKRouter.Config(
            num_experts=args.num_experts,
            gate=Linear.Config(in_features=args.dim, out_features=args.num_experts, bias=False),
            top_k=args.top_k,
            score_func=args.score_func,
            route_norm=args.route_norm,
            route_scale=args.route_scale,
            _debug_force_load_balance=args.force_load_balance,
        ),
        load_balance_coeff=None,
        shared_experts=maybe_shared_experts(args),
    )

    moe = moe_cfg.build().to(device=info.device, dtype=dtype)
    init_float_params(moe)

    # Shard GroupedExperts over the expert dimension and install ep_mesh on the dispatcher.
    moe.experts = ExpertParallel()._apply(moe.experts, ep_mesh)
    moe.train(args.backward)
    return moe


def benchmark_forward(fn, args: argparse.Namespace, info: DistInfo) -> list[float]:
    # IMPORTANT: use no_grad, not inference_mode. On XPU, all_to_all_single_autograd
    # passes in no_grad but can fail under inference_mode.
    with torch.no_grad():
        for _ in range(args.warmup):
            fn()

        sync_device(info.device)

        samples_ms = []
        for _ in range(args.iters):
            sync_device(info.device)
            t0 = time.perf_counter()
            fn()
            sync_device(info.device)
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1000.0)

    return samples_ms


def benchmark_forward_backward(
    fn,
    x: torch.Tensor,
    module: nn.Module,
    args: argparse.Namespace,
    info: DistInfo,
) -> list[float]:
    def step():
        y = fn()
        loss = y.float().square().mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    for _ in range(args.warmup):
        step()

    sync_device(info.device)

    samples_ms = []
    for _ in range(args.iters):
        sync_device(info.device)
        t0 = time.perf_counter()
        step()
        sync_device(info.device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    return samples_ms


def timed_phase(
    name: str,
    samples: dict[str, list[float]],
    info: DistInfo,
    fn,
):
    sync_device(info.device)
    t0 = time.perf_counter()
    result = fn()
    sync_device(info.device)
    t1 = time.perf_counter()
    samples.setdefault(name, []).append((t1 - t0) * 1000.0)
    return result


@torch.no_grad()
def timing_breakdown(
    moe: MoE,
    norm: nn.Module | None,
    x: torch.Tensor,
    args: argparse.Namespace,
    info: DistInfo,
) -> None:
    if args.backward or args.compile:
        raise RuntimeError("--timing-breakdown only supports forward eager mode")

    samples: dict[str, list[float]] = {}
    iters = args.timing_breakdown_iters

    for _ in range(iters):
        if norm is None:
            x_norm = x
        else:
            x_norm = timed_phase("norm", samples, info, lambda: norm(x))

        bs, slen, dim = x_norm.shape
        x_flat = x_norm.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = timed_phase(
            "router",
            samples,
            info,
            lambda: moe.router(x_flat, moe.expert_bias),
        )

        timed_phase(
            "token_count_accum",
            samples,
            info,
            lambda: moe.tokens_per_expert.add_(num_tokens_per_expert),
        )

        routed_input, num_tokens_local, metadata = timed_phase(
            "dispatch",
            samples,
            info,
            lambda: moe.experts.token_dispatcher.dispatch(
                x_flat,
                top_scores,
                selected_experts_indices,
                num_tokens_per_expert,
            ),
        )

        routed_output = timed_phase(
            "experts",
            samples,
            info,
            lambda: moe.experts._experts_forward(
                routed_input,
                getattr(metadata, "num_tokens_per_expert_list", None)
                or num_tokens_local,
            ),
        )

        dispatcher = moe.experts.token_dispatcher
        if (
            isinstance(metadata, AllToAllDispatchMetadata)
            and dispatcher.ep_mesh is not None
        ):
            def combine_a2a_for_timing():
                combine_input = routed_unpermuted
                input_splits = metadata.output_splits
                output_splits = metadata.input_splits
                equal_split_size = metadata.equal_a2a_split_size
                if equal_split_size is not None:
                    combine_input = dispatcher._pad_to_equal_splits(
                        combine_input,
                        metadata.output_splits,
                        equal_split_size,
                    )
                    input_splits = None
                    output_splits = None
                combined = torch.ops._c10d_functional.wait_tensor(
                    all_to_all_single_autograd(
                        combine_input,
                        output_splits,
                        input_splits,
                        dispatcher.ep_mesh,
                    )
                )
                if equal_split_size is not None:
                    combined = dispatcher._compact_equal_splits(
                        combined,
                        metadata.input_splits,
                        equal_split_size,
                    )
                return combined

            routed_unpermuted = timed_phase(
                "combine_unpermute",
                samples,
                info,
                lambda: dispatcher._unpermute(
                    routed_output,
                    metadata.input_shape,
                    metadata.permuted_indices,
                    metadata.rank_major_shape,
                ),
            )
            routed_combined = timed_phase(
                "combine_a2a",
                samples,
                info,
                combine_a2a_for_timing,
            )
            shared_out = timed_phase(
                "shared_experts",
                samples,
                info,
                lambda: moe.shared_experts(x_flat)
                if moe.shared_experts is not None
                else torch.zeros_like(x_flat),
            )
            if dispatcher.score_before_experts:
                scored_output = routed_combined
            else:
                scored_output = timed_phase(
                    "score_apply",
                    samples,
                    info,
                    lambda: (
                        routed_combined
                        * metadata.top_scores_experts_sorted.to(
                            routed_combined.dtype
                        ).reshape(-1, 1)
                    ),
                )
            out_flat = timed_phase(
                "scatter_add",
                samples,
                info,
                lambda: deterministic_scatter_add_(
                    shared_out,
                    metadata.token_indices_experts_sorted.reshape(-1, 1).expand(
                        -1, x_flat.shape[-1]
                    ),
                    scored_output,
                ),
            )
        else:
            out_flat = timed_phase(
                "combine",
                samples,
                info,
                lambda: dispatcher.combine(
                    routed_output,
                    metadata,
                    x_flat,
                    moe.shared_experts,
                ),
            )
        out = out_flat.reshape(bs, slen, dim)
        if tuple(out.shape) != tuple(x.shape):
            raise RuntimeError(
                f"bad timing output shape: got {tuple(out.shape)}, expected {tuple(x.shape)}"
            )

    phase_names = [
        "norm",
        "router",
        "token_count_accum",
        "dispatch",
        "experts",
        "combine_unpermute",
        "combine_a2a",
        "shared_experts",
        "score_apply",
        "scatter_add",
        "combine",
    ]
    local_means = torch.tensor(
        [statistics.mean(samples.get(name, [0.0])) for name in phase_names],
        device=info.device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local_means) for _ in range(info.world_size)]
    dist.all_gather(gathered, local_means)

    if info.rank != 0:
        return

    stats = torch.stack(gathered).cpu()
    print("")
    print(f"timing_breakdown_iters={iters}")
    for idx, name in enumerate(phase_names):
        values = stats[:, idx]
        print(
            f"timing_breakdown_phase={name} "
            f"mean_rank0_ms={values[0].item():.3f} "
            f"max_rank_mean_ms={values.max().item():.3f} "
            f"min_rank_mean_ms={values.min().item():.3f}"
        )


def summarize(
    args: argparse.Namespace,
    info: DistInfo,
    dp_degree: int,
    ep_degree: int,
    mode: str,
    samples_ms: list[float],
) -> None:
    mean_ms = statistics.mean(samples_ms)
    p50_ms = statistics.median(samples_ms)
    p90_ms = statistics.quantiles(samples_ms, n=10)[8]
    p99_ms = statistics.quantiles(samples_ms, n=100)[98] if len(samples_ms) >= 100 else max(samples_ms)

    local_stats = torch.tensor(
        [mean_ms, p50_ms, p90_ms, p99_ms],
        device=info.device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local_stats) for _ in range(info.world_size)]
    dist.all_gather(gathered, local_stats)

    if info.rank != 0:
        return

    stats = torch.stack(gathered).cpu()
    slowest_mean_ms = stats[:, 0].max().item()
    local_tokens = args.batch_size * args.seq_len
    global_tokens = local_tokens * info.world_size

    print("benchmark=torchtitan.experiments.ezpz.moe.MoE.forward")
    print(f"model_flavor={args.model_flavor}")
    print(f"host={socket.gethostname()}")
    print(f"mode={mode}")
    print(f"device={info.device_type}")
    print(f"backend={info.backend}")
    print(f"world_size={info.world_size}")
    print(f"local_size={info.local_size}")
    print(f"dp_degree={dp_degree}")
    print(f"ep_degree={ep_degree}")
    print(f"batch_size_per_rank={args.batch_size}")
    print(f"seq_len={args.seq_len}")
    print(f"dim={args.dim}")
    print(f"hidden_dim={args.hidden_dim}")
    print(f"num_experts_global={args.num_experts}")
    print(f"num_experts_per_ep_rank={args.num_experts // ep_degree}")
    print(f"num_shared_experts={0 if args.no_shared_experts else args.num_shared_experts}")
    print(f"top_k={args.top_k}")
    print(f"dtype={args.dtype}")
    print(f"score_func={args.score_func}")
    print(f"score_before_experts={args.score_before_experts}")
    print(f"force_load_balance={args.force_load_balance}")
    print(f"use_grouped_mm={args.use_grouped_mm}")
    print(f"include_ffn_norm={args.include_ffn_norm}")
    print(f"compile={args.compile}")
    print("patch_xpu_a2a=False")
    print("autograd_a2a_patch=False")
    if args.backward:
        print("forward_context=grad_enabled")
        print("loss=y.float().square().mean()")
        print("optimizer_step=False")
    else:
        print("forward_context=torch.no_grad")
    print("")

    print("per-rank latency ms:")
    for r in range(info.world_size):
        print(
            f"  rank={r} "
            f"mean={stats[r, 0]:.3f} "
            f"p50={stats[r, 1]:.3f} "
            f"p90={stats[r, 2]:.3f} "
            f"p99={stats[r, 3]:.3f}"
        )

    print("")
    print(f"slowest_rank_mean_ms={slowest_mean_ms:.3f}")
    print(f"tokens_per_sec_per_rank={local_tokens / (slowest_mean_ms / 1000.0):,.0f}")
    print(f"global_tokens_per_sec={global_tokens / (slowest_mean_ms / 1000.0):,.0f}")


def main() -> None:
    args = parse_args()

    if args.suppress_xpu_autocast_warning:
        warnings.filterwarnings(
            "ignore",
            message=".*XPU Autocast only supports dtypes.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*In XPU autocast, but the target dtype is not supported.*",
            category=UserWarning,
        )

    info = setup_distributed(args)

    if info.rank == 0:
        print(f"[rank0] host={socket.gethostname()}")
        print(f"[rank0] MASTER_ADDR={os.environ.get('MASTER_ADDR')}")
        print(f"[rank0] MASTER_PORT={os.environ.get('MASTER_PORT')}")
        print(f"[rank0] backend={info.backend} device={info.device}")
        print(f"[rank0] world_size={info.world_size} local_size={info.local_size}")
        print(f"[rank0] args={vars(args)}")

    dtype = dtype_from_arg(args.dtype)
    torch.manual_seed(1234 + info.rank)

    _, ep_mesh, dp_degree, ep_degree = build_mesh(args, info)
    moe = build_moe(args, info, ep_mesh, dtype)

    norm = None
    if args.include_ffn_norm:
        norm = torch.nn.RMSNorm(args.dim).to(device=info.device, dtype=dtype)
        norm.train(args.backward)

    modules = nn.ModuleList([m for m in [norm, moe] if m is not None])

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.dim,
        device=info.device,
        dtype=dtype,
        requires_grad=args.backward,
    )

    if norm is None:
        def fn():
            return moe(x)
    else:
        def fn():
            return moe(norm(x))

    # Smoke check in no_grad for forward-only, grad-enabled for backward.
    if args.backward:
        y = fn()
    else:
        with torch.no_grad():
            y = fn()
    expected_shape = (args.batch_size, args.seq_len, args.dim)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(f"bad output shape: got {tuple(y.shape)}, expected {expected_shape}")

    if args.compile:
        fn = torch.compile(fn, fullgraph=False, dynamic=False)

    if args.backward:
        samples_ms = benchmark_forward_backward(fn, x, modules, args, info)
        mode = "fwd+bwd"
    else:
        moe.eval()
        if norm is not None:
            norm.eval()
        samples_ms = benchmark_forward(fn, args, info)
        mode = "fwd"

    summarize(args, info, dp_degree, ep_degree, mode, samples_ms)
    if args.timing_breakdown:
        timing_breakdown(moe, norm, x, args, info)

    needs_explicit_exit = args.compile and info.device_type == "xpu"

    if env_flag("TT_MOE_BENCH_FAST_EXIT"):
        if info.rank == 0:
            _print_moe_fastpath_counters()
        sync_device(info.device)
        os._exit(0)

    if needs_explicit_exit or env_flag("TT_MOE_BENCH_EXIT_AFTER_DESTROY"):
        if info.rank == 0:
            _print_moe_fastpath_counters()

    # torch.compile can leave XPU/XCCL work and graph-owned references alive
    # until interpreter teardown. Release them while the process group is still
    # valid so benchmark shutdown does not race distributed finalization.
    sync_device(info.device)
    del samples_ms
    del y
    del fn
    del x
    del modules
    del norm
    del moe
    gc.collect()
    if info.device_type == "cuda":
        torch.cuda.empty_cache()
    elif info.device_type == "xpu":
        torch.xpu.empty_cache()
    sync_device(info.device)
    dist.destroy_process_group()

    if needs_explicit_exit or env_flag("TT_MOE_BENCH_EXIT_AFTER_DESTROY"):
        os._exit(0)


if __name__ == "__main__":
    main()
