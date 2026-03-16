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

    echo "--- [${label}] starting (module=${module} config=${config}) ---"

    start_seconds=$SECONDS

    if NGPU="${NGPU}" \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module "${module}" \
            --config "${config}" \
            --training.steps "${BENCH_STEPS}" \
            --metrics.log-freq 1 \
            --checkpoint.no-enable \
            --metrics.no-enable-wandb \
            "$@" \
        > "${logfile}" 2>&1; then
        STATUSES[$i]="OK"
    else
        STATUSES[$i]="FAILED"
    fi

    elapsed=$(( SECONDS - start_seconds ))
    WALL_TIMES[$i]="${elapsed}"

    # Parse metrics from the last line containing "tps:"
    last_metrics_line="$(grep 'tps:' "${logfile}" | tail -1 || true)"

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

cat > "${REPORT}" <<EOF
# ezpz Benchmark Report

| Field     | Value                          |
|-----------|--------------------------------|
| Date      | ${RUN_DATE}                    |
| Commit    | ${GIT_COMMIT}                  |
| Machine   | ${MACHINE_NAME}                |
| Job ID    | ${JOB_ID}                      |
| Nodes     | ${NUM_NODES}                   |
| GPUs      | ${NGPU}                        |
| Steps     | ${BENCH_STEPS}                 |

## Results

| Config | Steps | TPS | TFLOPS | MFU | Wall Time (s) | Status |
|--------|-------|-----|--------|-----|---------------|--------|
EOF

for ((i = 0; i < NUM_CONFIGS; i++)); do
    printf "| %-14s | %5s | %7s | %10s | %6s | %13s | %6s |\n" \
        "${LABELS[$i]}" \
        "${BENCH_STEPS}" \
        "${TPS_VALUES[$i]}" \
        "${TFLOPS_VALUES[$i]}" \
        "${MFU_VALUES[$i]}" \
        "${WALL_TIMES[$i]}" \
        "${STATUSES[$i]}" \
        >> "${REPORT}"
done

echo "" >> "${REPORT}"
echo "Logs: \`${OUTDIR}/\`" >> "${REPORT}"

# Print report to stdout
echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Logs saved to:   ${OUTDIR}/"
