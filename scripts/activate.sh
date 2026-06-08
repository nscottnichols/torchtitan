#!/usr/bin/env bash
# Source this file, do not execute it:
#   source scripts/activate.sh

VENV="${VENV:-/tmp/tt-moe-bench-venv}"
REPO="${REPO:-/tmp/torchtitan-ezpz}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV" >&2
  echo "Run scripts/install_uv_env.sh first, or set VENV=/path/to/venv." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Use system MPI/PMIx transport for oneCCL/XCCL.
export CCL_PROCESS_LAUNCHER="${CCL_PROCESS_LAUNCHER:-pmix}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_KVS_MODE="${CCL_KVS_MODE:-mpi}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
export CCL_OP_SYNC="${CCL_OP_SYNC:-1}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"

# Do not let stale torchrun-style env override MPI/PALS rank discovery.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE

if [[ -d "$REPO" ]]; then
  cd "$REPO" || return 1 2>/dev/null || exit 1
fi

echo "Activated $VENV"
echo "Repo: ${REPO}"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "mpiexec: $(command -v mpiexec || true)"
