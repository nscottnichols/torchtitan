#!/bin/bash --login
#PBS -N agpt-20b-sophiag-olmo-mix-n1024
#PBS -l select=1024
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q prod
#PBS -k doe
#PBS -j oe

# ---- Configuration ----
MODEL="20b"
NNODES=1024
SEQ_LEN=8192
GBS=12288

# 4,673,780,159,710 tokens / (12288 * 8192) = 46,429 steps
TRAINING_STEPS=46429

DFL="torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt"
OPTIMIZER="sophiag"
LR="2.28e-5"

CKPT_DIR="checkpoints/agpt-${MODEL}-${OPTIMIZER}-olmo-mix-1124-n${NNODES}-gbs${GBS}"

# ---- Environment ----
cd "${PBS_O_WORKDIR}" || exit 1

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config "agpt_${MODEL}" \
    --debug.print-config \
    --training.local-batch-size 1 \
    --activation_checkpoint.mode full \
    --training.global-batch-size "${GBS}" \
    --training.seq-len "${SEQ_LEN}" \
    --training.steps "${TRAINING_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --optimizer.lr "${LR}" \
    --dataloader.dataset blendcorpus \
    --dataloader.dataset-path "${DFL}" \
    --checkpoint.enable \
    --checkpoint.folder "${CKPT_DIR}" \
    --checkpoint.interval 100 \
    --checkpoint.keep-latest-k 0 \
    --checkpoint.no-last-save-model-only \
    "$@"
