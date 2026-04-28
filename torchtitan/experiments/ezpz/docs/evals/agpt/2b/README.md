# Evaluation Results — agpt 2B

> **Living document** — updated as new eval results come in.
>
> Last updated: 2026-04-27

## Setup

| Field | Value |
|-------|-------|
| Model | agpt_2b (1.99B params) |
| Tokenizer | google/gemma-7b (vocab_size=256128) |
| Training data | olmo-mix-1124 (4.67T tokens) |
| Training config | 256N, GBS=3072, SophiaG LR=2.28e-5 |
| Eval tasks | hellaswag, arc_easy, arc_challenge, winogrande |
| Eval backend | lm-eval 0.4.10, HF backend, XPU, transformers 4.57.6 |
| Checkpoints | DCP → HF safetensors via `eval/convert_to_hf.py` |

## Benchmark Accuracy vs Training Step

![2B Eval Results](figures/eval_2b.png)

## Results

| Step | Tokens | Loss | HellaSwag | ARC-Easy | ARC-Chall | Winogrande |
|------|--------|------|-----------|----------|-----------|------------|
| 1,000 | 25B | 6.13 | 25.35 | 27.44 | 23.38 | 47.83 |
| 2,000 | 50B | 5.84 | 25.26 | 27.19 | 24.23 | 49.57 |
| 3,000 | 75B | 5.76 | 25.25 | 26.89 | 23.55 | 51.07 |
| 4,000 | 101B | 5.72 | 25.12 | 26.98 | 24.06 | 48.46 |
| 5,000 | 126B | 5.69 | 25.12 | 26.97 | 24.06 | 49.02 |
| 6,000 | 151B | 5.67 | 25.12 | 27.48 | 24.23 | 48.46 |
| 7,000 | 176B | 5.65 | 25.37 | 27.19 | 23.55 | 49.88 |
| 8,000 | 201B | 5.64 | 25.37 | 27.02 | 24.49 | 51.07 |
| 9,000 | 226B | 5.63 | 25.19 | 27.23 | 24.15 | 47.75 |
| 10,000 | 252B | 5.61 | 25.12 | 27.78 | 23.81 | 48.94 |
| 11,000 | 277B | — | *pending* | | | |
| 12,000 | 302B | — | *pending* | | | |
| 13,000 | 327B | — | *pending* | | | |
| 14,000 | 352B | — | *pending* | | | |
| 15,000 | 378B | — | *pending* | | | |
| 16,000 | 403B | — | *pending* | | | |
| 17,000 | 428B | — | *pending* | | | |
| 18,000 | 453B | — | *pending* | | | |

**Notes:**
- Scores near random baseline through step 10,000 (~252B tokens, 5.4% of target)
- Slight signal in winogrande (51.1% at step 3K and 8K vs 50% random)
- ARC-challenge fluctuating 23–25% (random = 25%)
- Expect meaningful improvement after ~500B+ tokens
- Note: embeddings may not be loading correctly from DCP conversion (under investigation)
