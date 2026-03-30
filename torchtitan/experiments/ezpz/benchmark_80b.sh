#!/usr/bin/bash
# Benchmark script for ~80B AGPT model configs across parallelism settings.
#
# Runs 10 training iterations for each (model, TP, PP) combination, parses
# throughput metrics, and generates a markdown summary table.
#
# Usage:
#   # Inside a PBS job or interactive session with ezpz available:
#   bash torchtitan/experiments/ezpz/benchmark_80b.sh
#
#   # Override defaults:
#   BENCH_STEPS=5 BENCH_TP="4 8" BENCH_PP="1 2 4" \
#       bash torchtitan/experiments/ezpz/benchmark_80b.sh
#
# Environment variables:
#   BENCH_STEPS   — training iterations per run (default: 10)
#   BENCH_TP      — space-separated TP degrees to sweep (default: "4 8")
#   BENCH_PP      — space-separated PP degrees to sweep (default: "1 2 4")
#   BENCH_MODELS  — space-separated model flavors (default: "80B 80B_wide 80B_deep")
#   NGPU / WORLD_SIZE — total number of XPU devices

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
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
NGPU="${NGPU:-${NGPUS:-${WORLD_SIZE:-48}}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/benchmarks/80b_${TIMESTAMP}"
mkdir -p "${OUTDIR}"

# Model configs to benchmark
BENCH_MODELS="${BENCH_MODELS:-80B 80B_wide 80B_deep}"
read -ra MODELS <<< "${BENCH_MODELS}"

# Parallelism degrees to sweep (factors of 12 for Aurora's 12 tiles/node)
# NOTE: PP > 1 is disabled by default on XPU — Intel's SDPA kernel segfaults
# with GQA (n_heads != n_kv_heads) during pipeline parallel forward passes.
BENCH_TP="${BENCH_TP:-2 3 6 12}"
BENCH_PP="${BENCH_PP:-1}"
read -ra TP_DEGREES <<< "${BENCH_TP}"
read -ra PP_DEGREES <<< "${BENCH_PP}"

# Layer counts per model (must match __init__.py definitions)
declare -A MODEL_LAYERS=(
    ["80B"]=84
    ["80B_wide"]=48
    ["80B_deep"]=96
)

# n_heads / n_kv_heads per model (for TP divisibility check)
declare -A MODEL_NHEADS=(
    ["80B"]=72
    ["80B_wide"]=84
    ["80B_deep"]=60
)
declare -A MODEL_NKVHEADS=(
    ["80B"]=12
    ["80B_wide"]=12
    ["80B_deep"]=12
)

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Kill stale python processes from previous runs
# ---------------------------------------------------------------------------
echo "--- Cleaning up stale processes ---"
pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true

# ---------------------------------------------------------------------------
# Pre-cache dataset indices (avoids Lustre race condition on multi-node)
# ---------------------------------------------------------------------------
# blendcorpus builds index files on rank 0; on parallel filesystems the
# writes may not be visible to other nodes before they try to read them.
# Build the indices on a single process for each model config before
# launching multi-rank benchmarks.
echo "--- Pre-caching dataset indices ---"
for _precache_model in "${MODELS[@]}"; do
    PRECACHE_LOG="${OUTDIR}/_precache_${_precache_model}.log"
    echo -n "    ${_precache_model}... "
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        python3 -c "
import os
os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
                  MASTER_ADDR='localhost', MASTER_PORT='29500')
import torch
torch.distributed.init_process_group(backend='gloo', world_size=1, rank=0)
from blendcorpus.data.config import set_config, get_config
from blendcorpus.data.gpt_dataset import build_gpt_datasets
from blendcorpus import parallel_state as mpu
from types import SimpleNamespace

# Subclass so any attribute the blendcorpus package adds in the future
# falls back to a sensible default instead of raising AttributeError.
class _Cfg(SimpleNamespace):
    _DEFAULTS = {
        'mmap_warmup': False, 'data_impl': 'mmap', 'seed': 42,
        'eval_iters': 0, 'gate_bias': False, 'num_workers': 0,
    }
    def __getattr__(self, name):
        if name in self._DEFAULTS:
            return self._DEFAULTS[name]
        return None

mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel_size=1,
)

cfg = _Cfg(
    data_file_list='${DATASET_PATH}',
    seq_length=${BENCH_SEQ_LEN},
    train_iters=10,
    micro_batch_size=1,
    global_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel_size=1,
    split='100,0,0',
    dataloader_type='single',
    shuffle=True,
    shuffle_sample_in_corpus=True,
    blend_sample_in_corpus=False,
    append_eod=True,
    provide_attention_mask=False,
    eod_token_id=None,
    data_cache_path='$(pwd)/.cache/blendcorpus',
)
set_config(cfg)
build_gpt_datasets(cfg)
torch.distributed.destroy_process_group()
print('OK')
" > "${PRECACHE_LOG}" 2>&1 && echo "OK" || echo "WARN (see ${PRECACHE_LOG})"
done
# Give Lustre time to propagate index files across nodes
sync && sleep 10
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
# Collect runs
# ---------------------------------------------------------------------------
declare -a R_MODEL R_TP R_PP R_DP R_STATUS R_WALL R_MEMORY R_TPS R_TFLOPS R_MFU R_WANDB
RUN_IDX=0

echo "============================================================"
echo " 80B Benchmark Suite — ${TIMESTAMP}"
echo " steps=${BENCH_STEPS}  devices=${NGPU}  nodes=${NUM_NODES}"
echo " models: ${BENCH_MODELS}"
echo " TP sweep: ${BENCH_TP}"
echo " PP sweep: ${BENCH_PP}"
echo "============================================================"
echo ""

for model in "${MODELS[@]}"; do
    n_layers="${MODEL_LAYERS[$model]}"
    n_heads="${MODEL_NHEADS[$model]}"
    n_kv_heads="${MODEL_NKVHEADS[$model]}"

    for tp in "${TP_DEGREES[@]}"; do
        # Check TP divides n_heads and n_kv_heads
        if (( n_heads % tp != 0 || n_kv_heads % tp != 0 )); then
            echo "--- [${model}] TP=${tp} skipped (doesn't divide heads=${n_heads}/kv=${n_kv_heads}) ---"
            echo ""
            continue
        fi

        # Adjust seq_len to be divisible by TP (required by parallelize_llama)
        seq_len="${BENCH_SEQ_LEN}"
        if (( seq_len % tp != 0 )); then
            seq_len=$(( seq_len - (seq_len % tp) ))
            echo "    [${model}] TP=${tp}: adjusted seq_len to ${seq_len} (must be divisible by TP)"
        fi

        for pp in "${PP_DEGREES[@]}"; do
            # Check PP divides n_layers
            if (( n_layers % pp != 0 )); then
                echo "--- [${model}] TP=${tp} PP=${pp} skipped (PP doesn't divide layers=${n_layers}) ---"
                echo ""
                continue
            fi

            # Check TP*PP divides world size
            tp_pp=$(( tp * pp ))
            if (( NGPU % tp_pp != 0 )); then
                echo "--- [${model}] TP=${tp} PP=${pp} skipped (TP*PP=${tp_pp} doesn't divide NGPU=${NGPU}) ---"
                echo ""
                continue
            fi

            dp=$(( NGPU / tp_pp ))
            label="${model}_tp${tp}_pp${pp}_dp${dp}"
            logfile="${OUTDIR}/${label}.log"

            echo "--- [${label}] running (module=ezpz.agpt config=agpt_${model,,}) ---"
            echo "    model=${model}  TP=${tp}  PP=${pp}  DP=${dp}  layers/stage=$(( n_layers / pp ))"
            echo "    started @ $(tstamp)"
            echo "    logfile=${logfile}"
            echo ""

            R_MODEL[$RUN_IDX]="${model}"
            R_TP[$RUN_IDX]="${tp}"
            R_PP[$RUN_IDX]="${pp}"
            R_DP[$RUN_IDX]="${dp}"

            start_seconds=$SECONDS

            # Disable torch.compile when PP > 1: XPU SDPA with GQA fails
            # during pipeline shape inference with FakeTensors.
            compile_args=()
            if (( pp > 1 )); then
                compile_args=("--compile.no-enable")
            fi

            stdbuf -oL -eL \
                env NGPU="${NGPU}" PYTHONUNBUFFERED=1 \
                ezpz launch python3 -m torchtitan.experiments.ezpz.train \
                    --module ezpz.agpt \
                    --config "agpt_${model,,}" \
                    --training.steps "${BENCH_STEPS}" \
                    --training.local_batch_size "${pp}" \
                    --training.seq_len "${seq_len}" \
                    --activation_checkpoint.mode full \
                    --metrics.log_freq 1 \
                    --checkpoint.no-enable \
                    --parallelism.tensor_parallel_degree "${tp}" \
                    --parallelism.pipeline_parallel_degree "${pp}" \
                    --dataloader.dataset blendcorpus \
                    --dataloader.dataset_path "${DATASET_PATH}" \
                    "${compile_args[@]}" \
                > "${logfile}" 2>&1
            exit_code=$?

            # Check both exit code and presence of training output
            # (mpiexec can return 0 even when child ranks crash)
            if (( exit_code != 0 )); then
                R_STATUS[$RUN_IDX]="FAIL(rc=${exit_code})"
            elif grep -q 'Traceback\|Error\|Exception' "${logfile}" && ! grep -q 'loss:' "${logfile}"; then
                R_STATUS[$RUN_IDX]="CRASH"
            elif ! grep -q 'loss:' "${logfile}"; then
                R_STATUS[$RUN_IDX]="NO_OUTPUT"
            else
                R_STATUS[$RUN_IDX]="OK"
            fi

            elapsed=$(( SECONDS - start_seconds ))
            R_WALL[$RUN_IDX]="${elapsed}"

            # Parse metrics from last log line containing "loss:"
            last_line="$(grep 'loss:' "${logfile}" | tail -1 || true)"

            if [[ -n "${last_line}" ]]; then
                R_TPS[$RUN_IDX]="$(echo "${last_line}" | sed -n 's/.*tps: \([0-9,]*\).*/\1/p' | tr -d ',')"
                R_TFLOPS[$RUN_IDX]="$(echo "${last_line}" | sed -n 's/.*tflops: \([0-9,.]*\).*/\1/p' | tr -d ',')"
                R_MFU[$RUN_IDX]="$(echo "${last_line}" | sed -n 's/.*mfu: \([0-9.]*\)%.*/\1/p')"
                R_MEMORY[$RUN_IDX]="$(echo "${last_line}" | sed -n 's/.*memory: \([0-9.]*GiB([0-9.]*%)\).*/\1/p')"
            fi

            # Parse wandb run URL
            R_WANDB[$RUN_IDX]="$(grep -m1 'View run at' "${logfile}" | sed -n 's/.*View run at \(https:[^ ]*\).*/\1/p' || true)"

            # Fill in defaults for missing values
            R_TPS[$RUN_IDX]="${R_TPS[$RUN_IDX]:-N/A}"
            R_TFLOPS[$RUN_IDX]="${R_TFLOPS[$RUN_IDX]:-N/A}"
            R_MFU[$RUN_IDX]="${R_MFU[$RUN_IDX]:-N/A}"
            R_MEMORY[$RUN_IDX]="${R_MEMORY[$RUN_IDX]:-N/A}"
            R_WANDB[$RUN_IDX]="${R_WANDB[$RUN_IDX]:-}"

            echo "    status=${R_STATUS[$RUN_IDX]}  wall=${elapsed}s  memory=${R_MEMORY[$RUN_IDX]}  tps=${R_TPS[$RUN_IDX]}  tflops=${R_TFLOPS[$RUN_IDX]}  mfu=${R_MFU[$RUN_IDX]}"
            echo ""

            RUN_IDX=$(( RUN_IDX + 1 ))
        done
    done
done

NUM_RUNS="${RUN_IDX}"

# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# 80B Benchmark Report"
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
    printf "| %-10s | %3s | %3s | %5s | %20s | %9s | %8s | %6s | %10s |"  \
        "Model" "TP" "PP" "DP" "Memory" "TPS" "TFLOPS" "MFU" "Wall (s)"
    # Add wandb column only if any run has a URL
    has_wandb=false
    for ((i = 0; i < NUM_RUNS; i++)); do
        if [[ -n "${R_WANDB[$i]}" ]]; then
            has_wandb=true
            break
        fi
    done
    if $has_wandb; then
        printf " %-6s |" "W&B"
    fi
    printf " %-6s |\n" "Status"

    # Separator
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|" \
        "$(printf '%0.s-' $(seq 1 10))" \
        "$(printf '%0.s-' $(seq 1 3))" \
        "$(printf '%0.s-' $(seq 1 3))" \
        "$(printf '%0.s-' $(seq 1 5))" \
        "$(printf '%0.s-' $(seq 1 20))" \
        "$(printf '%0.s-' $(seq 1 9))" \
        "$(printf '%0.s-' $(seq 1 8))" \
        "$(printf '%0.s-' $(seq 1 6))" \
        "$(printf '%0.s-' $(seq 1 10))"
    if $has_wandb; then
        printf "-%s-|" "$(printf '%0.s-' $(seq 1 6))"
    fi
    printf "-%s-|\n" "$(printf '%0.s-' $(seq 1 6))"

    for ((i = 0; i < NUM_RUNS; i++)); do
        printf "| %-10s | %3s | %3s | %5s | %20s | %9s | %8s | %6s | %10s |" \
            "${R_MODEL[$i]}" \
            "${R_TP[$i]}" \
            "${R_PP[$i]}" \
            "${R_DP[$i]}" \
            "${R_MEMORY[$i]}" \
            "${R_TPS[$i]}" \
            "${R_TFLOPS[$i]}" \
            "${R_MFU[$i]}" \
            "${R_WALL[$i]}"
        if $has_wandb; then
            if [[ -n "${R_WANDB[$i]}" ]]; then
                printf " [link](%s) |" "${R_WANDB[$i]}"
            else
                printf " %-6s |" ""
            fi
        fi
        printf " %-6s |\n" "${R_STATUS[$i]}"
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
