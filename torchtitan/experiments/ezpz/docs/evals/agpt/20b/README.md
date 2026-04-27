# Evaluation Results — agpt 20B

> **Living document** — updated as new eval results come in.
>
> Last updated: 2026-04-27

## Setup

| Field | Value |
|-------|-------|
| Model | agpt_20b (20.7B params) |
| Tokenizer | google/gemma-7b (vocab_size=256128) |
| Training data | olmo-mix-1124 (4.67T tokens) |
| Training config | 256N, GBS=3072, SophiaG LR=2.28e-5 |
| Eval tasks | hellaswag, arc_easy, arc_challenge, winogrande |
| Eval backend | lm-eval 0.4.10, HF backend, XPU, dtype=bfloat16 |
| Checkpoints | DCP → HF safetensors via `eval/convert_to_hf.py` |

## Benchmark Accuracy vs Training Step

![20B Eval Results](figures/eval_20b.png)

## Results

| Step | Tokens | Loss | HellaSwag | ARC-Easy | ARC-Challenge | Winogrande |
|------|--------|------|-----------|----------|---------------|------------|
| 100 | 25B | 11.84 | *pending* | | | |
| 500 | 13B | 7.35 | *pending* | | | |
| 1,000 | 25B | 5.73 | *pending* | | | |
| 1,500 | 38B | 5.25 | *pending* | | | |
| 2,000 | 50B | 5.03 | *pending* | | | |
| 2,500 | 63B | 4.85 | *pending* | | | |

**Notes:**
- All HF checkpoints converted (6 total)
- Eval job submitted (8452582), results incoming
- 20B model trains at ~2 steps/min on 256N — fewer checkpoints but each
  represents more effective compute than the 2B equivalents
- Tokens = step × GBS(3072) × seq_len(8192)
