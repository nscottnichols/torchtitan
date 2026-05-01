#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-lmeval
#PBS -j oe

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# Use the clean eval venv (system-site-packages from frameworks)
source venvs/aurora/tt-lm-eval/bin/activate
# Remove any bad local transformers installs
pip uninstall -y transformers 2>/dev/null

echo "Python: $(which python3)"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import lm_eval; print(f'lm_eval: {lm_eval.__version__}')"

TASKS="hellaswag,arc_easy,arc_challenge,winogrande"

# Run evals using Python API with monkey-patched CUDA warmup
python3 << 'PYEOF'
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None

from lm_eval import evaluator
import json, os

tasks = "hellaswag,arc_easy,arc_challenge,winogrande".split(",")
base = "outputs/evals"

def run_eval(model, step):
    hf_dir = f"{base}/agpt-{model}/step-{step}/hf"
    results_dir = f"{base}/agpt-{model}/step-{step}/results"

    if not os.path.exists(f"{hf_dir}/config.json"):
        print(f"[SKIP] {model} step-{step} — no config")
        return

    # Check for existing results
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                print(f"[SKIP] {model} step-{step} — done")
                return

    print(f"======== {model} @ step {step} ========")
    os.makedirs(results_dir, exist_ok=True)

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={hf_dir}",
            tasks=tasks,
            batch_size=4,
            num_fewshot=0,
            device="xpu:0",
        )

        # Save results
        out_file = f"{results_dir}/results.json"
        with open(out_file, "w") as f:
            json.dump(results["results"], f, indent=2)

        for task, metrics in results["results"].items():
            acc = metrics.get("acc_norm,none") or metrics.get("acc,none", "?")
            print(f"  {task}: {acc:.4f}")

        print(f"Done {model} step {step}.")
    except Exception as e:
        print(f"ERROR {model} step-{step}: {e}")

# 2B
for step in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000]:
    run_eval("2b", step)

# 20B
for step in [100, 500, 1000, 1500, 2000, 2500]:
    run_eval("20b", step)

print("=== Complete ===")
PYEOF
