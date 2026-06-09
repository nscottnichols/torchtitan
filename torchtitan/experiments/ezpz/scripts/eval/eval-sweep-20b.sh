#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-20b-sweep
#PBS -j oe

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>/dev/null
module load frameworks/2025.3.1 2>/dev/null

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
export PYTHONPATH="$(pwd)"

MODEL="20b"
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq"
CSV_FILE="outputs/evals/eval_results_20b.csv"
echo "model,step,task,metric,value,stderr" > "$CSV_FILE"

run_eval() {
    local STEP=$1
    local CKPT_DIR="outputs/checkpoints/agpt-${MODEL}-sophiag-olmo-mix-1124-n256-gbs3072/step-${STEP}"
    local HF_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/hf"
    local RESULTS_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/results"
    local CONFIG="torchtitan/experiments/ezpz/eval/configs/agpt_${MODEL}_config.json"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[SKIP] step-${STEP} — checkpoint not found"
        return 1
    fi

    # Skip if results already exist
    if find "${RESULTS_DIR}" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] step-${STEP} — results already exist, extracting to CSV"
        _extract_to_csv "$STEP" "$RESULTS_DIR"
        return 0
    fi

    echo ""
    echo "======== agpt-${MODEL} @ step ${STEP} ========"

    # Convert DCP → HF
    if [[ ! -f "${HF_DIR}/model-00001-of-00001.safetensors" ]]; then
        echo "Converting DCP → HF..."
        mkdir -p "${HF_DIR}"
        python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
            "${CKPT_DIR}" "${HF_DIR}" \
            --model_name experiments.ezpz.agpt --model_flavor "${MODEL}" \
            --hf_assets_path assets/hf/gemma-7b --export_dtype bfloat16 2>&1
        cp "${CONFIG}" "${HF_DIR}/config.json"
        cp assets/hf/gemma-7b/tokenizer.json assets/hf/gemma-7b/tokenizer.model \
           assets/hf/gemma-7b/tokenizer_config.json assets/hf/gemma-7b/special_tokens_map.json \
           "${HF_DIR}/"
    fi

    # Run lm-eval — 20B is ~40GB bf16, should fit on one XPU tile (48GB HBM)
    echo "Running lm-eval..."
    mkdir -p "${RESULTS_DIR}"
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${HF_DIR},dtype=bfloat16" \
        --tasks "${TASKS}" --batch_size 4 --num_fewshot 0 \
        --output_path "${RESULTS_DIR}" --device xpu 2>&1

    _extract_to_csv "$STEP" "$RESULTS_DIR"
    echo "Done step ${STEP}."
}

_extract_to_csv() {
    local STEP=$1
    local RESULTS_DIR=$2
    local RESULT_FILE=$(find "${RESULTS_DIR}" -name "results_*.json" 2>/dev/null | head -1)
    [[ -z "$RESULT_FILE" ]] && return
    python3 -c "
import json
with open('${RESULT_FILE}') as f:
    data = json.load(f)
for task, metrics in data.get('results', {}).items():
    for key, val in metrics.items():
        if key == 'alias' or 'stderr' in key: continue
        metric = key.replace(',none', '')
        stderr_key = key.replace(',none', '_stderr,none')
        stderr = metrics.get(stderr_key, '')
        print(f'${MODEL},{STEP},{task},{metric},{val},{stderr}')
" >> "${CSV_FILE}"
}

echo "=== 20B Eval Sweep ==="
echo "Start: $(date)"

for step in 100 500 1000 1500 2000 2500; do
    run_eval "$step"
done

echo ""
echo "=== 20B Sweep Complete: $(date) ==="
cat "$CSV_FILE"
