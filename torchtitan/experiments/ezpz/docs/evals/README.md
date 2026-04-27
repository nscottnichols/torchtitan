# Evaluation Results

Benchmark evaluations of AuroraGPT production checkpoints using
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Models

| Model | Steps Evaluated | Status |
|-------|-----------------|--------|
| [agpt 2B](agpt/2b/) | 1K–5K done, 6K–18K pending | In progress |
| [agpt 20B](agpt/20b/) | All pending | Queued |

## Pipeline

```
DCP checkpoint → eval/convert_to_hf.py → HF safetensors → lm-eval (HF backend, XPU)
```

See `eval/convert_and_eval.sh` for the end-to-end script.

## Environment

- **Module:** `frameworks/2025.3.1` (bare, no user venv)
- **Must set:** `HF_HUB_ENABLE_HF_TRANSFER=0`
- **Device:** `--device xpu`
- **Do NOT** use `ezpz_setup_env` — the user venv has transformers 5.6.2
  which breaks lm-eval's HF backend
