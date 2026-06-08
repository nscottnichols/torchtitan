#!/usr/bin/env bash
set -euo pipefail

# Single-node 12-rank XPU/XCCL benchmark for the 10B_2B_sdpa MoE path.
# Assumes scripts/install_uv_env.sh has already been run.

VENV="${VENV:-/tmp/tt-moe-bench-venv}"
REPO="${REPO:-/tmp/torchtitan-ezpz}"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$PKG_DIR/bench_ep_moe_forward_mpi.py}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO"

# Ensure venv-provided impi-rt launch/runtime files do not shadow system MPI.
"$PKG_DIR/scripts/quarantine_vendored_mpi.sh" >/dev/null
hash -r 2>/dev/null || true

export CPU_BIND="${CPU_BIND:-list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96}"
export CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY:-5,13,21,29,37,45,57,65,73,81,89,97}"

# System MPI / oneCCL MPI transport path.
export CCL_PROCESS_LAUNCHER="${CCL_PROCESS_LAUNCHER:-pmix}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_KVS_MODE="${CCL_KVS_MODE:-mpi}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
export CCL_OP_SYNC="${CCL_OP_SYNC_OVERRIDE:-0}"
# The 12-rank benchmark requires FLAT hierarchy so each tile is visible as a
# device. Inheriting COMPOSITE exposes fewer devices and fails ranks >= 6.
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY_OVERRIDE:-FLAT}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-2345}"

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE

NPROC="${NPROC:-12}"
PPN="${PPN:-12}"
EP_DEGREE="${EP_DEGREE:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
FORCE_LOAD_BALANCE="${FORCE_LOAD_BALANCE:-1}"

EXTRA_BENCH_ARGS=()
if [[ "${TT_MOE_TIMING:-0}" == "1" ]]; then
  EXTRA_BENCH_ARGS+=(--timing-breakdown)
  EXTRA_BENCH_ARGS+=(--timing-breakdown-iters "${TT_MOE_TIMING_ITERS:-5}")
fi
if [[ "${FORCE_LOAD_BALANCE}" == "1" ]]; then
  EXTRA_BENCH_ARGS+=(--force-load-balance)
fi

LAUNCH_CMD=(python -u "$BENCH")
if [[ "${USE_CCL_LOCAL_WRAP:-0}" == "1" ]]; then
  LAUNCH_CMD=("$PKG_DIR/scripts/ccl_local_wrap.sh" python -u "$BENCH")
fi

which mpiexec
mpiexec --cpu-bind "${CPU_BIND}" -n "${NPROC}" -ppn "${PPN}" -env PALS_WORLD_SIZE="${NPROC}" -- \
  "${LAUNCH_CMD[@]}" \
    --model-flavor 10B_2B_sdpa \
    --device xpu \
    --backend xccl \
    --ep-degree "${EP_DEGREE}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --dim 2048 \
    --hidden-dim 1408 \
    --num-experts 36 \
    --num-shared-experts 2 \
    --top-k 3 \
    --dtype bf16 \
    --score-func softmax \
    --score-after-experts \
    --include-ffn-norm \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    "${EXTRA_BENCH_ARGS[@]}" \
    "$@"
