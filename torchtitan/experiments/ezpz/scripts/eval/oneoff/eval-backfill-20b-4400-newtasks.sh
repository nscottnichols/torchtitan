#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -j oe

# One-off backfill: re-run 20B step-4400 eval with the new 7-task set
# (existing results.json was moved aside to .results.4task.json).

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_CPP_LOG_LEVEL=ERROR

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate

bash torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh \
    --model 20b \
    --step 4400 \
    --tasks "hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
    --eval-only \
    --ckpt-name agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
    --label 512n
