#!/bin/bash --login

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>/dev/null
module load frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz

echo "Python: $(which python3)"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')"

TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq"

run_eval() {
    local MODEL=$1 STEP=$2
    local HF_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/hf"
    local RESULTS_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/results"

    [[ ! -f "${HF_DIR}/config.json" ]] && echo "[SKIP] ${MODEL} step-${STEP} — no config" && return
    find "${RESULTS_DIR}" -name "results_*.json" 2>/dev/null | grep -q . && echo "[SKIP] ${MODEL} step-${STEP} — done" && return

    echo "======== ${MODEL} @ step ${STEP} ========"
    mkdir -p "${RESULTS_DIR}"
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${HF_DIR}" \
        --tasks "${TASKS}" --batch_size 8 --num_fewshot 0 \
        --output_path "${RESULTS_DIR}" --device xpu 2>&1
    echo "Done ${MODEL} step ${STEP}."
}

echo "=== Eval Sweep (frameworks only, no user venv) ==="
echo "Start: $(date)"

for step in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000; do
    run_eval "2b" "$step"
done

for step in 100 500 1000 1500 2000 2500; do
    run_eval "20b" "$step"
done

echo "=== Complete: $(date) ==="
