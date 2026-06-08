#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/tmp/tt-moe-bench-venv}"
QUARANTINE="$VENV/quarantine-system-mpi-conflicts"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: VENV does not exist: $VENV" >&2
  exit 1
fi

mkdir -p "$QUARANTINE"

# These are installed by impi-rt in the venv and can shadow system MPI/libfabric.
# Move them aside rather than uninstalling impi-rt, since torch metadata may list
# impi-rt as a dependency.
shopt -s nullglob
items=(
  "$VENV"/bin/mpiexec*
  "$VENV"/bin/mpirun*
  "$VENV"/bin/fi_info
  "$VENV"/bin/fi_pingpong
  "$VENV"/lib/libmpi.so*
  "$VENV"/lib/libfabric.so*
)

if (( ${#items[@]} > 0 )); then
  mv "${items[@]}" "$QUARANTINE"/ 2>/dev/null || true
fi

hash -r 2>/dev/null || true

echo "Quarantined venv MPI/libfabric conflicts under: $QUARANTINE"
find "$QUARANTINE" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null | sort || true
