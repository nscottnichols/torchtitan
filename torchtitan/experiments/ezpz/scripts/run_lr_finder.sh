#!/bin/bash --login
# LR Finder sweep for agpt models across optimizers.
# Runs LR finder for each (model, optimizer) combination.
#
# Usage (inside a PBS allocation):
#   bash torchtitan/experiments/ezpz/scripts/run_lr_finder.sh
#
# Environment variables:
#   LRF_MODELS      — space-separated model flavors (default: "2b 20b")
#   LRF_OPTIMIZERS  — space-separated optimizers (default: "adamw muon sophiag")
#   LRF_STEPS       — total training steps for fraction calc (default: 1000)
#   LRF_FRACTION    — fraction of steps to sweep (default: 0.1)
#   LRF_INIT_LR     — starting LR (default: 1e-6)
#   LRF_MAX_LR      — max LR (default: 1.0)
#   LRF_TIMEOUT     — per-run timeout in seconds (default: 1800)
#   LRF_LBS         — local batch size per device (default: 1). At production
#                     N + LBS to measure the optimal LR at the actual GBS the
#                     production chain will run at (e.g. LBS=2 for 2b 256N
#                     gives GBS=6144 matching submit_agpt_2b_aurora_venv.sh).

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup — production torch 2.13 .venv + ezpz yeet-env (matches
# scripts/submit_agpt_2b_aurora_venv.sh so the LR-finder runs in the SAME
# stack as the production chain it's calibrating).
#
# When this script is invoked via `qsub -- /bin/bash -c 'bash <this>'`, the
# inner bash is NOT a login shell and `module` is undefined. Source
# Aurora's lmod init explicitly. (Sourcing /etc/profile.d/*.sh alone
# does NOT define `module` on Aurora — lmod lives at /usr/share/lmod/.)
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
    if [[ -r /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    else
        for f in /etc/profile.d/*.sh; do
            [[ -r "$f" ]] && source "$f" >/dev/null 2>&1 || true
        done
    fi
fi
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
set -u

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "lr-finder: yeet-env via tarball (.venv.tar.gz)"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "lr-finder: yeet-env via per-file rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LRF_MODELS="${LRF_MODELS:-2b 20b}"
LRF_OPTIMIZERS="${LRF_OPTIMIZERS:-adamw muon sophiag}"
LRF_STEPS="${LRF_STEPS:-1000}"
LRF_FRACTION="${LRF_FRACTION:-0.1}"
LRF_INIT_LR="${LRF_INIT_LR:-1e-6}"
LRF_MAX_LR="${LRF_MAX_LR:-1.0}"
LRF_TIMEOUT="${LRF_TIMEOUT:-1800}"
LRF_LBS="${LRF_LBS:-1}"

read -ra MODELS <<< "${LRF_MODELS}"
read -ra OPTIMIZERS <<< "${LRF_OPTIMIZERS}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
OUTDIR="outputs/lr_finder/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Kill stale processes
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
MACHINE_NAME="$(ezpz_get_machine_name 2>/dev/null || hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"
DEVICES_PER_NODE=$(( NGPUS / NUM_NODES ))
FINDER_STEPS=$(python3 -c "print(max(1, int(${LRF_STEPS} * ${LRF_FRACTION})))")

echo "============================================================"
echo " LR Finder Sweep — ${TIMESTAMP}"
echo " devices=${NGPUS}  nodes=${NUM_NODES}"
echo " models: ${LRF_MODELS}"
echo " optimizers: ${LRF_OPTIMIZERS}"
echo " LR range: ${LRF_INIT_LR} → ${LRF_MAX_LR}"
echo " finder steps: ${FINDER_STEPS} (${LRF_FRACTION} × ${LRF_STEPS})"
echo "============================================================"
echo ""

declare -a R_MODEL R_OPT R_STATUS R_WALL R_MIN_LOSS R_LR_AT_MIN
RUN_IDX=0

for model in "${MODELS[@]}"; do
    config="agpt_${model}"
    # 80B needs TP=2
    tp_args=()
    if [[ "${model}" == "80b" || "${model}" == "80B" ]]; then
        tp_args=("--parallelism.tensor_parallel_degree" "2")
    fi

    for opt in "${OPTIMIZERS[@]}"; do
        label="${model}_${opt}"
        logfile="${OUTDIR}/${label}.log"

        echo "--- [${label}] LR finder (config=${config} optimizer=${opt}) ---"
        echo "    started @ $(date +%Y%m%d-%H%M%S)"
        echo "    logfile: ${logfile}"
        echo ""

        R_MODEL[$RUN_IDX]="${model}"
        R_OPT[$RUN_IDX]="${opt}"

        start_seconds=$SECONDS

        timeout "${LRF_TIMEOUT}" \
            stdbuf -oL -eL \
            env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
            ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module ezpz.agpt \
            --config "${config}" \
            --optimizer "${opt}" \
            --training.steps "${LRF_STEPS}" \
            --training.local_batch_size "${LRF_LBS}" \
            --training.seq_len 8192 \
            --metrics.log_freq 1 \
            --checkpoint.no-enable \
            --dataloader.dataset blendcorpus \
            --dataloader.dataset_path "${DATASET_PATH}" \
            --lr_finder.enable \
            --lr_finder.init_lr "${LRF_INIT_LR}" \
            --lr_finder.max_lr "${LRF_MAX_LR}" \
            --lr_finder.fraction "${LRF_FRACTION}" \
            "${tp_args[@]}" \
            "$@" \
            >"${logfile}" 2>&1 || true
        exit_code=$?

        # Kill any leftover processes
        pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null || true
        sleep 2

        elapsed=$(( SECONDS - start_seconds ))
        R_WALL[$RUN_IDX]="${elapsed}"

        # Determine status
        if ((exit_code == 124)); then
            R_STATUS[$RUN_IDX]="TIMEOUT"
        elif grep -q 'OUT_OF_RESOURCES\|out of memory\|OOM' "${logfile}"; then
            R_STATUS[$RUN_IDX]="OOM"
        elif grep -q 'LR Finder complete' "${logfile}"; then
            R_STATUS[$RUN_IDX]="OK"
        elif grep -q 'Traceback\|Error\|Exception' "${logfile}"; then
            R_STATUS[$RUN_IDX]="CRASH"
        else
            R_STATUS[$RUN_IDX]="UNKNOWN"
        fi

        # Parse min loss and LR from finder output
        min_line="$(grep 'Min loss @' "${logfile}" 2>/dev/null | tail -1 || true)"
        if [[ -n "${min_line}" ]]; then
            R_LR_AT_MIN[$RUN_IDX]="$(echo "${min_line}" | sed -n 's/.*lr=\([0-9.eE+-]*\).*/\1/p')"
            R_MIN_LOSS[$RUN_IDX]="$(echo "${min_line}" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')"
        else
            # Parse from CSV if available
            csv_dir="$(grep 'saved CSV to' "${logfile}" 2>/dev/null | sed -n 's/.*saved CSV to \(.*\)/\1/p' | head -1 || true)"
            if [[ -n "${csv_dir}" && -f "${csv_dir}" ]]; then
                # Find min loss row
                min_row="$(tail -n +2 "${csv_dir}" | sort -t, -k2 -n | head -1 || true)"
                R_LR_AT_MIN[$RUN_IDX]="$(echo "${min_row}" | cut -d, -f1)"
                R_MIN_LOSS[$RUN_IDX]="$(echo "${min_row}" | cut -d, -f2)"
            else
                R_LR_AT_MIN[$RUN_IDX]="N/A"
                R_MIN_LOSS[$RUN_IDX]="N/A"
            fi
        fi

        R_LR_AT_MIN[$RUN_IDX]="${R_LR_AT_MIN[$RUN_IDX]:-N/A}"
        R_MIN_LOSS[$RUN_IDX]="${R_MIN_LOSS[$RUN_IDX]:-N/A}"

        echo "    status=${R_STATUS[$RUN_IDX]}  wall=${elapsed}s  min_loss=${R_MIN_LOSS[$RUN_IDX]}  lr@min=${R_LR_AT_MIN[$RUN_IDX]}"
        echo ""

        RUN_IDX=$(( RUN_IDX + 1 ))
    done
done

NUM_RUNS="${RUN_IDX}"

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# LR Finder Report"
    echo ""

    _meta_keys=("Date" "Commit" "Machine" "Job ID" "Nodes" "Devices" "Finder Steps" "LR Range")
    _meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPUS}" "${FINDER_STEPS}" "${LRF_INIT_LR} → ${LRF_MAX_LR}")
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
    printf "| %-6s | %-10s | %10s | %12s | %10s | %-8s |\n" \
        "Model" "Optimizer" "Min Loss" "LR @ Min" "Wall (s)" "Status"
    _sep() { printf '%0.s-' $(seq 1 "$1"); }
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|\n" \
        "$(_sep 6)" "$(_sep 10)" "$(_sep 10)" "$(_sep 12)" "$(_sep 10)" "$(_sep 8)"

    for ((i = 0; i < NUM_RUNS; i++)); do
        printf "| %-6s | %-10s | %10s | %12s | %10s | %-8s |\n" \
            "${R_MODEL[$i]}" \
            "${R_OPT[$i]}" \
            "${R_MIN_LOSS[$i]}" \
            "${R_LR_AT_MIN[$i]}" \
            "${R_WALL[$i]}" \
            "${R_STATUS[$i]}"
    done

    echo ""
    echo "Logs: \`${OUTDIR}/\`"
    echo ""
    echo "LR finder curves saved to \`outputs/lr_finder/ezpz/\`"
} >"${REPORT}"

echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Logs saved to:   ${OUTDIR}/"
