#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Benchmark/smoke-test script for ezpz experiment configs.
# Runs a few training iterations for each config, parses metrics, and
# generates a summary report with timing/throughput numbers.
#
# Usage:
#   BENCH_STEPS=5 bash torchtitan/experiments/ezpz/run_benchmarks.sh
#
# Extra CLI args are forwarded to every run:
#   bash torchtitan/experiments/ezpz/run_benchmarks.sh --parallelism.tp_degree 2

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup (same pattern as run_train.sh)
# ---------------------------------------------------------------------------
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENCH_STEPS="${BENCH_STEPS:-10}"
NGPU="${NGPU:-${NGPUS:-${WORLD_SIZE:-4}}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/benchmarks/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

# Benchmark configs: parallel arrays of (label, module, config)
LABELS=(    "agpt_2b"       "agpt_20b"      "moe_debugmodel"  "moe_10b_2b")
MODULES=(   "ezpz.agpt"    "ezpz.agpt"     "ezpz.moe"        "ezpz.moe")
CONFIGS=(   "agpt_2b"      "agpt_20b"       "moe_debugmodel"  "moe_10b_2b")

NUM_CONFIGS="${#LABELS[@]}"

# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
RUN_DATE="$(date -Iseconds)"
MACHINE_NAME="$(hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
declare -a WALL_TIMES
declare -a STATUSES
declare -a TPS_VALUES
declare -a TFLOPS_VALUES
declare -a MFU_VALUES

echo "============================================================"
echo " ezpz benchmarks — ${TIMESTAMP}"
echo " steps=${BENCH_STEPS}  gpus=${NGPU}  nodes=${NUM_NODES}"
echo "============================================================"
echo ""

for ((i = 0; i < NUM_CONFIGS; i++)); do
    label="${LABELS[$i]}"
    module="${MODULES[$i]}"
    config="${CONFIGS[$i]}"
    logfile="${OUTDIR}/${label}.log"

    echo "--- [${label}] running (module=${module} config=${config}) ---"
    echo "    started @ $(tstamp)"
    echo "    logfile=${logfile}"
    echo ""

    start_seconds=$SECONDS

    if NGPU="${NGPU}" \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module "${module}" \
            --config "${config}" \
            --training.steps "${BENCH_STEPS}" \
            --metrics.log-freq 1 \
            --checkpoint.no-enable \
            --dataloader.dataset blendcorpus \
            --dataloader.dataset_path "${DATASET_PATH}" \
            "$@" \
        > "${logfile}" 2>&1; then
        STATUSES[$i]="OK"
    else
        STATUSES[$i]="FAILED"
    fi

    elapsed=$(( SECONDS - start_seconds ))
    WALL_TIMES[$i]="${elapsed}"

    # Parse metrics from the last line containing "tps:"
    last_metrics_line="$(egrep 'loss:' "${logfile}" | tail -1 || true)"

    if [[ -n "${last_metrics_line}" ]]; then
        # Extract tps: value (integer with commas)
        tps="$(echo "${last_metrics_line}" | sed -n 's/.*tps: \([0-9,]*\).*/\1/p' | tr -d ',')"
        # Extract tflops: value (decimal with commas)
        tflops="$(echo "${last_metrics_line}" | sed -n 's/.*tflops: \([0-9,.]*\).*/\1/p' | tr -d ',')"
        # Extract mfu: value (decimal with %)
        mfu="$(echo "${last_metrics_line}" | sed -n 's/.*mfu: \([0-9.]*\)%.*/\1/p')"
    else
        tps=""
        tflops=""
        mfu=""
    fi

    TPS_VALUES[$i]="${tps:-N/A}"
    TFLOPS_VALUES[$i]="${tflops:-N/A}"
    MFU_VALUES[$i]="${mfu:-N/A}"

    echo "    status=${STATUSES[$i]}  wall=${elapsed}s  tps=${TPS_VALUES[$i]}  tflops=${TFLOPS_VALUES[$i]}  mfu=${MFU_VALUES[$i]}"
    echo ""
done

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

# Find the widest value across all metadata fields for column alignment
_meta_keys=(Date Commit Machine "Job ID" Nodes GPUs Steps)
_meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPU}" "${BENCH_STEPS}")
_vw=5
for _v in "${_meta_vals[@]}"; do
    (( ${#_v} > _vw )) && _vw=${#_v}
done

{
    echo "# ezpz Benchmark Report"
    echo ""
    printf "| %-7s | %-${_vw}s |\n" "Field" "Value"
    printf "|-%s-|-%s-|\n" "$(printf '%0.s-' $(seq 1 7))" "$(printf '%0.s-' $(seq 1 "${_vw}"))"
    for ((_j = 0; _j < ${#_meta_keys[@]}; _j++)); do
        printf "| %-7s | %-${_vw}s |\n" "${_meta_keys[$_j]}" "${_meta_vals[$_j]}"
    done
    echo ""
    echo "## Results"
    echo ""
    printf "| %-14s | %5s | %7s | %8s | %6s | %13s | %6s |\n" \
        "Config" "Steps" "TPS" "TFLOPS" "MFU" "Wall Time (s)" "Status"
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|\n" \
        "$(printf '%0.s-' $(seq 1 14))" \
        "$(printf '%0.s-' $(seq 1 5))" \
        "$(printf '%0.s-' $(seq 1 7))" \
        "$(printf '%0.s-' $(seq 1 8))" \
        "$(printf '%0.s-' $(seq 1 6))" \
        "$(printf '%0.s-' $(seq 1 13))" \
        "$(printf '%0.s-' $(seq 1 6))"
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        printf "| %-14s | %5s | %7s | %8s | %6s | %13s | %6s |\n" \
            "${LABELS[$i]}" \
            "${BENCH_STEPS}" \
            "${TPS_VALUES[$i]}" \
            "${TFLOPS_VALUES[$i]}" \
            "${MFU_VALUES[$i]}" \
            "${WALL_TIMES[$i]}" \
            "${STATUSES[$i]}"
    done
    echo ""
    echo "Logs: \`${OUTDIR}/\`"
} > "${REPORT}"

# Print report to stdout
echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Logs saved to:   ${OUTDIR}/"
