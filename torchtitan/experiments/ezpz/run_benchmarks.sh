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
#
# Environment variables:
#   BENCH_STEPS     — training iterations per run (default: 10)
#   BENCH_SEQ_LEN   — sequence length (default: 8192)
#   BENCH_LOCAL_BS  — local batch size (default: 2)
#   BENCH_GAS       — gradient accumulation steps (default: 1)
#   BENCH_TIMEOUT   — per-run timeout in seconds (default: 1800)
#   FILTER_NONZERO_RANKS — set to 1 to suppress output from non-rank-0 (default: 0)
#   NO_COMPILE      — set to 1 to disable torch.compile (default: 0)

set -uo pipefail

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
BENCH_SEQ_LEN="${BENCH_SEQ_LEN:-8192}"
BENCH_LOCAL_BS="${BENCH_LOCAL_BS:-2}"
BENCH_GAS="${BENCH_GAS:-1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
FILTER_NONZERO_RANKS="${FILTER_NONZERO_RANKS:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
NGPU="${NGPU:-${NGPUS:-${WORLD_SIZE:-4}}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/benchmarks/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

# Benchmark configs: parallel arrays of (label, module, config)
LABELS=(    "agpt_2b"    "agpt_20b"   "agpt_80b"   "agpt_80b_alt"   "agpt_80b_wide"   "agpt_80b_deep"   "agpt_80b_deep_alt"   "moe_debugmodel"  "moe_10b_2b")
MODULES=(   "ezpz.agpt"  "ezpz.agpt"  "ezpz.agpt"  "ezpz.agpt"      "ezpz.agpt"       "ezpz.agpt"       "ezpz.agpt"           "ezpz.moe"        "ezpz.moe")
CONFIGS=(   "agpt_2b"    "agpt_20b"   "agpt_80b"   "agpt_80b_alt"   "agpt_80b_wide"   "agpt_80b_deep"   "agpt_80b_deep_alt"   "moe_debugmodel"  "moe_10b_2b")

NUM_CONFIGS="${#LABELS[@]}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Kill stale python processes from previous runs
# ---------------------------------------------------------------------------
echo "--- Cleaning up stale processes and cache ---"
pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true
rm -rf .cache/blendcorpus/*.npy 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
RUN_DATE="$(date -Iseconds)"
MACHINE_NAME="$(hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
DEVICES_PER_NODE=$(( NGPU / NUM_NODES ))

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
declare -a WALL_TIMES STATUSES TPS_VALUES TFLOPS_VALUES MFU_VALUES MEMORY_VALUES WANDB_VALUES

echo "============================================================"
echo " ezpz benchmarks — ${TIMESTAMP}"
echo " steps=${BENCH_STEPS}  devices=${NGPU}  nodes=${NUM_NODES}"
echo "============================================================"
echo ""

for ((i = 0; i < NUM_CONFIGS; i++)); do
    label="${LABELS[$i]}"
    module="${MODULES[$i]}"
    config="${CONFIGS[$i]}"
    logfile="${OUTDIR}/${label}.log"

    # Clear stale index cache between runs
    rm -rf .cache/blendcorpus/*.npy 2>/dev/null || true

    echo "--- [${label}] running (module=${module} config=${config}) ---"
    echo "    started @ $(tstamp)"
    echo "    logfile: ${logfile}"
    echo ""

    start_seconds=$SECONDS

    compile_args=()
    if (( NO_COMPILE )); then
        compile_args=("--compile.no-enable")
    fi

    timeout "${BENCH_TIMEOUT}" \
        stdbuf -oL -eL \
        env NGPU="${NGPU}" PYTHONUNBUFFERED=1 \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module "${module}" \
            --config "${config}" \
            --training.steps "${BENCH_STEPS}" \
            --training.local_batch_size "${BENCH_LOCAL_BS}" \
            --training.gradient_accumulation_steps "${BENCH_GAS}" \
            --training.seq_len "${BENCH_SEQ_LEN}" \
            --metrics.log_freq 1 \
            --checkpoint.no-enable \
            --dataloader.dataset blendcorpus \
            --dataloader.dataset_path "${DATASET_PATH}" \
            "${compile_args[@]}" \
            "$@" \
        2>&1 | if (( FILTER_NONZERO_RANKS )); then grep -v '^\[rank[1-9][0-9]*\]:'; else cat; fi > "${logfile}" || true
    exit_code=${PIPESTATUS[0]}

    # Kill any leftover processes from this run
    pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null || true
    sleep 2

    # Determine run status
    if (( exit_code == 124 )); then
        STATUSES[$i]="TIMEOUT"
    elif (( exit_code != 0 )); then
        STATUSES[$i]="FAIL(rc=${exit_code})"
    elif grep -q 'OUT_OF_RESOURCES\|out of memory\|OOM' "${logfile}"; then
        STATUSES[$i]="OOM"
    elif grep -q 'Traceback\|Error\|Exception' "${logfile}" && ! grep -q 'loss:' "${logfile}"; then
        STATUSES[$i]="CRASH"
    elif ! grep -q 'loss:' "${logfile}"; then
        STATUSES[$i]="NO_OUTPUT"
    else
        STATUSES[$i]="OK"
    fi

    elapsed=$(( SECONDS - start_seconds ))
    WALL_TIMES[$i]="${elapsed}"

    # Parse metrics from last log line containing "loss:"
    last_line="$(grep 'loss:' "${logfile}" | tail -1 || true)"

    if [[ -n "${last_line}" ]]; then
        TPS_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*tps: \([0-9,]*\).*/\1/p' | tr -d ',')"
        TFLOPS_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*tflops: \([0-9,.]*\).*/\1/p' | tr -d ',')"
        MFU_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*mfu: \([0-9.]*\)%.*/\1/p')"
        MEMORY_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*memory: \([0-9.]*GiB([0-9.]*%)\).*/\1/p')"
    fi

    # Parse wandb run URL
    WANDB_VALUES[$i]="$(grep -m1 'View run at' "${logfile}" | sed -n 's/.*View run at \(https:[^ ]*\).*/\1/p' || true)"

    # Fill in defaults for missing values
    TPS_VALUES[$i]="${TPS_VALUES[$i]:-N/A}"
    TFLOPS_VALUES[$i]="${TFLOPS_VALUES[$i]:-N/A}"
    MFU_VALUES[$i]="${MFU_VALUES[$i]:-N/A}"
    MEMORY_VALUES[$i]="${MEMORY_VALUES[$i]:-N/A}"
    WANDB_VALUES[$i]="${WANDB_VALUES[$i]:-}"

    echo "    status=${STATUSES[$i]}  wall=${elapsed}s  memory=${MEMORY_VALUES[$i]}  tps=${TPS_VALUES[$i]}  tflops=${TFLOPS_VALUES[$i]}  mfu=${MFU_VALUES[$i]}"
    echo ""
done

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# ezpz Benchmark Report"
    echo ""

    # Metadata table
    _meta_keys=("Date" "Commit" "Machine" "Job ID" "Nodes" "Devices" "Devices/Node" "Steps")
    _meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPU}" "${DEVICES_PER_NODE}" "${BENCH_STEPS}")
    _vw=5
    for _v in "${_meta_vals[@]}"; do
        (( ${#_v} > _vw )) && _vw=${#_v}
    done

    printf "| %-12s | %-${_vw}s |\n" "Field" "Value"
    printf "|-%s-|-%s-|\n" "$(printf '%0.s-' $(seq 1 12))" "$(printf '%0.s-' $(seq 1 "${_vw}"))"
    for ((_j = 0; _j < ${#_meta_keys[@]}; _j++)); do
        printf "| %-12s | %-${_vw}s |\n" "${_meta_keys[$_j]}" "${_meta_vals[$_j]}"
    done
    echo ""

    echo "## Results"
    echo ""
    printf "| %-14s | %5s | %20s | %7s | %8s | %6s | %10s |" \
        "Config" "Steps" "Memory" "TPS" "TFLOPS" "MFU" "Wall (s)"
    # Add wandb column only if any run has a URL
    has_wandb=false
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        if [[ -n "${WANDB_VALUES[$i]}" ]]; then
            has_wandb=true
            break
        fi
    done
    if $has_wandb; then
        printf " %-6s |" "W&B"
    fi
    printf " %-6s |\n" "Status"

    # Separator
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|" \
        "$(printf '%0.s-' $(seq 1 14))" \
        "$(printf '%0.s-' $(seq 1 5))" \
        "$(printf '%0.s-' $(seq 1 20))" \
        "$(printf '%0.s-' $(seq 1 7))" \
        "$(printf '%0.s-' $(seq 1 8))" \
        "$(printf '%0.s-' $(seq 1 6))" \
        "$(printf '%0.s-' $(seq 1 10))"
    if $has_wandb; then
        printf -- "-%s-|" "$(printf '%0.s-' $(seq 1 6))"
    fi
    printf -- "-%s-|\n" "$(printf '%0.s-' $(seq 1 6))"

    for ((i = 0; i < NUM_CONFIGS; i++)); do
        printf "| %-14s | %5s | %20s | %7s | %8s | %6s | %10s |" \
            "${LABELS[$i]}" \
            "${BENCH_STEPS}" \
            "${MEMORY_VALUES[$i]}" \
            "${TPS_VALUES[$i]}" \
            "${TFLOPS_VALUES[$i]}" \
            "${MFU_VALUES[$i]}" \
            "${WALL_TIMES[$i]}"
        if $has_wandb; then
            if [[ -n "${WANDB_VALUES[$i]}" ]]; then
                printf " [link](%s) |" "${WANDB_VALUES[$i]}"
            else
                printf " %-6s |" ""
            fi
        fi
        printf " %-6s |\n" "${STATUSES[$i]}"
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
