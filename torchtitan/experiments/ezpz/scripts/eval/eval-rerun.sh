#!/bin/bash --login

export PBS_JOBID="${PBS_JOBID}"
export PBS_NODEFILE="${PBS_NODEFILE}"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

echo "Python: $(which python3)"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')"

TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq"
CSV_FILE="outputs/evals/eval_results_2b.csv"
echo "model,step,task,metric,value,stderr" > "$CSV_FILE"

run_eval() {
    local MODEL=$1
    local STEP=$2
    local HF_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/hf"
    local RESULTS_DIR="outputs/evals/agpt-${MODEL}/step-${STEP}/results"
    local CSV=$3

    # Skip if no HF checkpoint
    if [[ ! -f "${HF_DIR}/config.json" ]]; then
        echo "[SKIP] ${MODEL} step-${STEP} — no config.json"
        return 1
    fi

    # Skip if results already exist
    if find "${RESULTS_DIR}" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] ${MODEL} step-${STEP} — results exist"
        _extract "$MODEL" "$STEP" "$RESULTS_DIR" "$CSV"
        return 0
    fi

    echo ""
    echo "======== ${MODEL} @ step ${STEP} ========"
    mkdir -p "${RESULTS_DIR}"

    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${HF_DIR},dtype=bfloat16" \
        --tasks "${TASKS}" --batch_size 8 --num_fewshot 0 \
        --output_path "${RESULTS_DIR}" --device xpu 2>&1

    _extract "$MODEL" "$STEP" "$RESULTS_DIR" "$CSV"
    echo "Done ${MODEL} step ${STEP}."
}

_extract() {
    local MODEL=$1 STEP=$2 RESULTS_DIR=$3 CSV=$4
    local F=$(find "${RESULTS_DIR}" -name "results_*.json" 2>/dev/null | head -1)
    [[ -z "$F" ]] && return
    python3 -c "
import json
with open('${F}') as f:
    data = json.load(f)
for task, metrics in data.get('results', {}).items():
    for key, val in metrics.items():
        if key == 'alias' or 'stderr' in key: continue
        metric = key.replace(',none', '')
        stderr_key = key.replace(',none', '_stderr,none')
        stderr = metrics.get(stderr_key, '')
        print(f'${MODEL},${STEP},{task},{metric},{val},{stderr}')
" >> "${CSV}"
}

echo "=== 2B Eval Rerun (lm-eval only, checkpoints already converted) ==="
echo "Start: $(date)"

for step in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000; do
    run_eval "2b" "$step" "$CSV_FILE"
done

echo ""
echo "=== 2B Complete: $(date) ==="
echo ""
echo "=== Results ==="
cat "$CSV_FILE"

echo ""
echo "=== Now running 20B ==="
CSV_20B="outputs/evals/eval_results_20b.csv"
echo "model,step,task,metric,value,stderr" > "$CSV_20B"

for step in 100 500 1000 1500 2000 2500; do
    run_eval "20b" "$step" "$CSV_20B"
done

echo ""
echo "=== 20B Complete: $(date) ==="
cat "$CSV_20B"
