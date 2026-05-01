#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-20b-v2
#PBS -j oe
#
# Convert + eval the v2 20B SophiaG checkpoints (fp32 master) and
# produce results that can be directly compared against the v1
# bf16-tainted eval table in docs/evals/agpt/20b/README.md.
#
# v2 ckpt path:
#   /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/
#     outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/step-{N}
#
# Output (in this clone, under outputs/evals/agpt-20b-v2/):
#   step-100/hf/      HF safetensors + tokenizer
#   step-100/results/ lm-eval JSON
#
# Steps to evaluate are passed via STEPS env var (space-separated). At
# write time only step-100 + step-200 are saved; later runs of this
# script will pick up newer ckpts as they land.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md — venv
# activate has unbound vars and would trigger on first source.
set -o pipefail

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1
echo "PWD: $(pwd)"
echo "Modules loaded."

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# This eval pipeline runs against the bare frameworks/2025.3.1 module
# stack (NOT the user venv) per CLAUDE.md — user venv has transformers
# 5.6.2 which breaks lm-eval's HF backend.
#
# But the convert_to_hf step needs torchtitan + the v2 model registry —
# so source the v2 venv for the conversion, then deactivate before
# the lm-eval step.
V2_REPO="/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz"
V2_CKPT_NAME="agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"

STEPS="${STEPS:-100 200}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande}"

for step in $STEPS; do
    DCP_DIR="${V2_REPO}/outputs/checkpoints/${V2_CKPT_NAME}/step-${step}"
    HF_DIR="outputs/evals/agpt-20b-v2/step-${step}/hf"
    RESULTS_DIR="outputs/evals/agpt-20b-v2/step-${step}/results"

    if [[ ! -d "$DCP_DIR" ]]; then
        echo "[SKIP] 20b-v2 step-${step}: no DCP at ${DCP_DIR}"
        continue
    fi
    if [[ -f "${RESULTS_DIR}/results.json" ]]; then
        echo "[SKIP] 20b-v2 step-${step}: results.json already exists"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "20B v2 step-${step}"
    echo "============================================================"

    # ---- Step 1: DCP -> HF (uses v2 venv) ----
    if [[ ! -f "${HF_DIR}/config.json" ]]; then
        echo "[1/2] Converting DCP -> HF (via v2 venv)..."
        source "${V2_REPO}/.venv/bin/activate"
        mkdir -p "${HF_DIR}"
        python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
            "${DCP_DIR}" \
            "${HF_DIR}" \
            --model_name "experiments.ezpz.agpt" \
            --model_flavor "20b" \
            --export_dtype "bfloat16"
        cp torchtitan/experiments/ezpz/eval/configs/agpt_20b_config.json \
            "${HF_DIR}/config.json"
        cp assets/hf/gemma-7b/tokenizer.{json,model} "${HF_DIR}/"
        cp assets/hf/gemma-7b/tokenizer_config.json "${HF_DIR}/"
        cp assets/hf/gemma-7b/special_tokens_map.json "${HF_DIR}/"
        deactivate
        echo "[1/2] Conversion done."
    else
        echo "[1/2] HF already converted, skipping."
    fi

    # ---- Step 2: lm-eval (bare frameworks venv + tt-lm-eval overlay) ----
    echo "[2/2] Running lm-eval..."
    source venvs/aurora/tt-lm-eval/bin/activate
    mkdir -p "${RESULTS_DIR}"
    python3 << PYEOF
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
import json, os
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained=${HF_DIR}",
    tasks="${TASKS}".split(","),
    batch_size=2,
    num_fewshot=0,
    device="xpu:0",
)
with open("${RESULTS_DIR}/results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
for task, metrics in results["results"].items():
    acc = metrics.get("acc_norm,none") or metrics.get("acc,none", "?")
    print(f"  {task}: {acc:.4f}")
PYEOF
    deactivate
    echo "[2/2] Eval done. results: ${RESULTS_DIR}/results.json"
done

echo ""
echo "=== 20B v2 eval complete ==="
