#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-sweep
#PBS -j oe

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

TASKS="hellaswag,arc_easy,arc_challenge,winogrande"

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
        --output_path "${RESULTS_DIR}" 2>&1
    echo "Done ${MODEL} step ${STEP}."
}

echo "=== Eval Sweep (ezpz_setup_env, no dtype arg) ==="
echo "Python: $(which python3)"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
echo "Start: $(date)"

echo "--- 2B ---"
for step in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000; do
    run_eval "2b" "$step"
done

echo "--- 20B ---"
for step in 100 500 1000 1500 2000 2500; do
    run_eval "20b" "$step"
done

echo "=== Complete: $(date) ==="
