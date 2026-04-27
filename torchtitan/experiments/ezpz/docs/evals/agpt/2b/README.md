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
| Eval backend | lm-eval 0.4.10, HF backend, XPU, dtype=bfloat16 |
| Checkpoints | DCP → HF safetensors via `eval/convert_to_hf.py` |

## Benchmark Accuracy vs Training Step

![2B Eval Results](figures/eval_2b.png)

## Results

| Step | Tokens | Loss | HellaSwag | ARC-Easy | ARC-Challenge | Winogrande |
|------|--------|------|-----------|----------|---------------|------------|
| 1,000 | 25B | 6.13 | 25.04 | 25.08 | 22.70 | 49.57 |
| 2,000 | 50B | 5.84 | 25.04 | 25.08 | 22.70 | 49.57 |
| 3,000 | 75B | 5.76 | 25.04 | 25.08 | 22.70 | 49.57 |
| 4,000 | 101B | 5.72 | 25.04 | 25.08 | 22.70 | 49.57 |
| 5,000 | 126B | 5.69 | 25.04 | — | — | — |
| 6,000 | 151B | 5.67 | *pending* | | | |
| 7,000 | 176B | 5.65 | *pending* | | | |
| 8,000 | 201B | 5.64 | *pending* | | | |
| 9,000 | 226B | 5.63 | *pending* | | | |
| 10,000 | 252B | 5.61 | *pending* | | | |
| 11,000 | 277B | 5.60 | *pending* | | | |
| 12,000 | 302B | 5.59 | *pending* | | | |
| 13,000 | 327B | 5.58 | *pending* | | | |
| 14,000 | 352B | 5.57 | *pending* | | | |
| 15,000 | 378B | 5.56 | *pending* | | | |
| 16,000 | 403B | 5.55 | *pending* | | | |
| 17,000 | 428B | 5.54 | *pending* | | | |
| 18,000 | 453B | 5.53 | *pending* | | | |

**Notes:**
- All benchmarks at random baseline through step 5,000 (~126B tokens, 2.7% of target)
- This is expected — benchmarks typically don't move above random until 100B–500B+ tokens
- Eval jobs for steps 6K–18K submitted (8452595), results incoming
- Tokens = step × GBS(3072) × seq_len(8192)
