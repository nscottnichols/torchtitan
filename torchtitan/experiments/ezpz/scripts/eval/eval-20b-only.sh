#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-20b-fixed
#PBS -j oe

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

python3 << 'PYEOF'
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None

from lm_eval import evaluator
import json, os

tasks = "hellaswag,arc_easy,arc_challenge,winogrande".split(",")

for step in [100, 500, 1000, 1500, 2000, 2500]:
    hf_dir = f"outputs/evals/agpt-20b/step-{step}/hf"
    results_dir = f"outputs/evals/agpt-20b/step-{step}/results"
    if not os.path.exists(f"{hf_dir}/config.json"):
        print(f"[SKIP] 20b step-{step} — no config")
        continue
    if os.path.exists(f"{results_dir}/results.json"):
        print(f"[SKIP] 20b step-{step} — done")
        continue

    print(f"======== 20b @ step {step} ========")
    os.makedirs(results_dir, exist_ok=True)
    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={hf_dir}",
            tasks=tasks,
            batch_size=2,
            num_fewshot=0,
            device="xpu:0",
        )
        with open(f"{results_dir}/results.json", "w") as f:
            json.dump(results["results"], f, indent=2)
        for task, metrics in results["results"].items():
            acc = metrics.get("acc_norm,none") or metrics.get("acc,none", "?")
            print(f"  {task}: {acc:.4f}")
        print(f"Done 20b step {step}.")
    except Exception as e:
        print(f"ERROR 20b step-{step}: {e}")

print("=== 20B Complete ===")
PYEOF
