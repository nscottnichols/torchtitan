#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=06:00:00
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
ezpz yeet-env
deactivate
source /tmp/.venv/bin/activate
source <(curl -fsSL https://bit.ly/ezpz-utils)
MACHINE="$(ezpz_get_machine_name)"

DATASET_PATH="${DATASET_PATH:-torchtitan/experiments/ezpz/data-lists/${MACHINE}/olmo-mix-1124.txt}"
STEPS="${STEPS:-1000}"

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_2b \
    --optimizer=sophiag \
    --optimizer.lr=2.28e-5 \
    --training.local-batch-size=2 \
    --training.steps="${STEPS}" \
    --dataloader.dataset-path="${DATASET_PATH}" \
    ${EXTRA_ARGS:-}
