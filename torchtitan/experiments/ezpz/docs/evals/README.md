# Evaluation Results

Benchmark evaluations of AuroraGPT production checkpoints using
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Models

| Model | Source | Steps Evaluated | Status |
|-------|--------|-----------------|--------|
| [agpt 2B](agpt/2b/) | torchtitan DCP | steps 1K–18K | Done (suspect embedding bug) |
| [agpt 20B](agpt/20b/) | torchtitan DCP | steps 100–2,500 | Done (same suspect bug) |
| [agpt 2B (MDS)](agpt/2b-mds/) | Megatron-DeepSpeed SophiaG | steps 5K–140K × 3 stages (84 evals) | Done — clean training signal |

## Pipelines

```
# torchtitan DCP runs
DCP checkpoint → eval/convert_to_hf.py → HF safetensors → lm-eval (HF backend, XPU)

# Megatron-DeepSpeed runs
mp_rank_00_model_states.pt → eval/mds_to_hf.py → HF safetensors → lm-eval (HF backend, XPU)
```

See `eval/convert_and_eval.sh` (DCP) and `eval/eval_mds_sweep.sh` (MDS)
for the end-to-end scripts. Aggregate with
`eval/aggregate_evals.py --model {2b,20b,2b-mds}`.

## Environment

- **Module:** `frameworks/2025.3.1` (bare, no user venv)
- **Must set:** `HF_HUB_ENABLE_HF_TRANSFER=0`
- **Device:** `--device xpu`
- **Do NOT** use `ezpz_setup_env` — the user venv has transformers 5.6.2
  which breaks lm-eval's HF backend
