#!/bin/bash --login
export PBS_JOBID="${PBS_JOBID}"
export PBS_NODEFILE="${PBS_NODEFILE}"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

echo "Python: $(which python3)"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')"

echo ""
echo "=== Quick test: hellaswag on step-1000, limit=100 ==="
python3 -m lm_eval \
    --model hf \
    --model_args pretrained=outputs/evals/agpt-2b/step-1000/hf \
    --tasks hellaswag \
    --batch_size 4 \
    --num_fewshot 0 \
    --limit 100 \
    2>&1 | tail -15
