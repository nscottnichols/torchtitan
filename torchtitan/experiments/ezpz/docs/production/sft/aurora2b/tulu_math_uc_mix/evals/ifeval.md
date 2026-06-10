# IFEval — AuroraGPT-2B-sophiag (baseline) vs SFT-step729

**Status:** pending. Eval script staged at
[`rl/scripts/sft/eval_ifeval_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/eval_ifeval_sft_vs_baseline.sh);
queued on PBS job `12468471` (the dedicated 4N eval allocation).
This page will be filled in once both models finish.

## Why this eval matters for this recipe

The 7-task base-LM eval ([`README.md`](README.md)) showed no
meaningful difference between baseline and SFT-step729 — expected,
because those benchmarks measure things SFT doesn't change.

IFEval is the matched metric: 541 generative prompts that test
*verifiable structural instruction-following*. Examples:

> Write a 200-word essay on X. Do not use the word "the". Include at
> least 3 questions. Each sentence must start with a capitalized
> verb.

A base LM ignores those constraints; an SFT'd LM should obey most
of them. The score is the fraction of prompts where all listed
constraints are met (`prompt_level_strict_acc`), with a looser
variant that scores individual constraints (`inst_level_strict_acc`).

## Expected result

The pretraining-derived `AuroraGPT-2B-sophiag-gs138650` baseline
likely lands in the `0.05–0.15` range on `prompt_level_strict_acc`
(near chance, since it tends to ignore prompt structure). The
SFT'd `checkpoint-729-hf` should be substantially higher —
`0.20–0.35` for a 2B-class model with 4.5B SFT tokens, based on
published numbers for similarly-sized post-SFT models on this
benchmark.

If the delta is small (<5pp), the mix isn't pushing instruction
adherence hard enough — would suggest more tulu-3 weight in a
follow-up recipe.

## Setup

| Field | Value |
|-------|-------|
| Eval | `lm-eval --tasks ifeval` |
| Tasks | 541 prompts, generative (`generate_until`) |
| Backend | lm-eval 0.4.10 + transformers 4.50.1 + dtype-shim |
| Hardware | 1 XPU tile per model, sequential on same node for fairness |
| Wall time estimate | ~1 h per model |
