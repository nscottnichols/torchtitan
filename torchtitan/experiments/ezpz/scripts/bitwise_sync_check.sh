#!/bin/bash --login
#PBS -A datascience
#PBS -N bitwise-sync-check
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# Bitwise-equivalence smoke for upstream-sync verification.
#
# Per the root CLAUDE.md project rule:
#   "Non-computation changes (e.g. activation checkpointing, refactoring)
#    must produce identical loss before vs. after with --debug.seed=42
#    and --debug.deterministic."
#
# Runs the same N-step agpt_2b smoke TWICE — once on the current HEAD
# (post-merge), once on a user-specified pre-merge commit — both with
# --debug.seed=42 --debug.deterministic so loss + grad_norm should be
# bit-identical for any pure-refactor / import-rename merge.
#
# Usage (from a login node, with this script as the qsub argument):
#
#   qsub -A datascience -q workq -l select=2 -l walltime=01:00:00 \
#     -l filesystems=tegu:home \
#     -v PRE_MERGE_COMMIT=8b610e911,STEPS=20 \
#     torchtitan/experiments/ezpz/scripts/bitwise_sync_check.sh
#
# Env knobs:
#   PRE_MERGE_COMMIT  required — sha to compare HEAD against (defaults
#                      to HEAD's first parent if omitted)
#   STEPS             default: 20  (cheap; enough to surface drift)
#   MODEL             default: 2b
#   SEED              default: 42  (matches CLAUDE.md convention)
#
# Output: logs/bitwise-sync-check-${PBS_JOBID}/
#   head.log         post-merge run output
#   pre.log          pre-merge run output
#   head.metrics     extracted (step, loss, grad_norm) from head.log
#   pre.metrics      same from pre.log
#   diff.txt         diff between the two .metrics files
#   verdict          "IDENTICAL" if diff is empty, "DRIFT" otherwise
#
# IMPORTANT: uses git stash + checkout to switch between HEAD and
# PRE_MERGE_COMMIT. Any uncommitted working-tree changes are stashed
# BEFORE the run and popped on exit. Don't run with half-finished
# edits you can't afford to have stashed.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

HEAD_COMMIT="$(git rev-parse HEAD)"
PRE_MERGE_COMMIT="${PRE_MERGE_COMMIT:-$(git rev-parse HEAD^1 2>/dev/null)}"
if [[ -z "$PRE_MERGE_COMMIT" ]] || ! git rev-parse "$PRE_MERGE_COMMIT" >/dev/null 2>&1; then
    echo "FATAL: PRE_MERGE_COMMIT='$PRE_MERGE_COMMIT' is not a valid git ref"
    exit 1
fi
PRE_MERGE_SHA="$(git rev-parse "$PRE_MERGE_COMMIT")"
HEAD_SHORT="${HEAD_COMMIT:0:9}"
PRE_SHORT="${PRE_MERGE_SHA:0:9}"

STEPS="${STEPS:-20}"
MODEL="${MODEL:-2b}"
SEED="${SEED:-42}"
NNODES="$(wc -l < "${PBS_NODEFILE}")"
LOG_DIR="logs/bitwise-sync-check-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

ORIG_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo "$HEAD_COMMIT")"
STASH_MARKER="bitwise-sync-check-${PBS_JOBID%%.*}"
if [[ -n "$(git status --porcelain -uno)" ]]; then
    git stash push -u -m "$STASH_MARKER" 2>&1 | tee -a "${LOG_DIR}/run.log"
    STASHED=1
else
    STASHED=0
fi

cleanup() {
    git checkout "$ORIG_BRANCH" 2>&1 | tee -a "${LOG_DIR}/run.log"
    if [[ "$STASHED" == "1" ]]; then
        git stash list | grep -q "$STASH_MARKER" && \
            git stash pop 2>&1 | tee -a "${LOG_DIR}/run.log"
    fi
}
trap cleanup EXIT

source .venv/bin/activate
python3 -c "import torch; print('torch', torch.__version__)" \
    | tee -a "${LOG_DIR}/run.log"

run_one() {
    local label="$1"
    local commit="$2"
    local outlog="${LOG_DIR}/${label}.log"
    local ckptdir="checkpoints/bitwise-sync-check-${PBS_JOBID%%.*}-${label}"

    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "phase: ${label}  commit: ${commit:0:9}" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"

    git checkout "$commit" 2>&1 | tee -a "${LOG_DIR}/run.log"

    local NGPUS_LOCAL="${NHOSTS:-$NNODES}"
    local GBS=$(( NGPUS_LOCAL * 12 ))

    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module=ezpz.agpt \
        --config="agpt_${MODEL}" \
        --compile.no-enable \
        --checkpoint.no-enable \
        --dataloader.dataset=blendcorpus \
        --dataloader.dataset-path="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt" \
        --dataloader.data-cache-path="${ckptdir}/.cache/books/index-cache" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --training.local-batch-size=1 \
        --training.global-batch-size="${GBS}" \
        --training.seq-len=8192 \
        --training.steps="${STEPS}" \
        --optimizer=sophiag \
        --optimizer.lr=2.28e-5 \
        2>&1 | tee "${outlog}"
}

run_one "head" "$HEAD_COMMIT"
run_one "pre" "$PRE_MERGE_COMMIT"

extract_metrics() {
    local in="$1" out="$2"
    sed 's/\x1b\[[0-9;]*m//g' "$in" \
        | grep -oE 'step:\s+[0-9]+\s+loss:\s+[0-9.]+\s+grad_norm:\s+[0-9.]+' \
        > "$out"
}
extract_metrics "${LOG_DIR}/head.log" "${LOG_DIR}/head.metrics"
extract_metrics "${LOG_DIR}/pre.log"  "${LOG_DIR}/pre.metrics"

diff -u "${LOG_DIR}/pre.metrics" "${LOG_DIR}/head.metrics" > "${LOG_DIR}/diff.txt"
DIFF_RC=$?

if [[ "$DIFF_RC" == "0" ]] && [[ -s "${LOG_DIR}/head.metrics" ]] && [[ -s "${LOG_DIR}/pre.metrics" ]]; then
    echo "IDENTICAL" > "${LOG_DIR}/verdict"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "VERDICT: IDENTICAL — loss + grad_norm match bit-for-bit" \
        | tee -a "${LOG_DIR}/run.log"
    echo "    pre:  ${PRE_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    head: ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    exit 0
else
    echo "DRIFT" > "${LOG_DIR}/verdict"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "VERDICT: DRIFT — see ${LOG_DIR}/diff.txt" | tee -a "${LOG_DIR}/run.log"
    echo "    pre:  ${PRE_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    head: ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    diff (first 40 lines):" | tee -a "${LOG_DIR}/run.log"
    head -40 "${LOG_DIR}/diff.txt" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    exit 1
fi
