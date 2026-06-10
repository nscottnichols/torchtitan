# GRPO smoke — SFT-step729 vs baseline as RL starting point

**Status:** pending. Smoke script staged at
[`rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh);
queued on PBS job `12468471` alongside IFEval. This page will be
filled in once both 50-step GRPO runs land.

## Why this eval matters for this recipe

The 32N SFT push was justified as "GRPO needs a stronger
initialization than a raw pretrained checkpoint." That's a claim
about training dynamics, not about a static benchmark — so neither
the 7-task lm-eval ([`README.md`](README.md)) nor IFEval
([`ifeval.md`](ifeval.md)) can validate it. The matched test is:
spin up GRPO with the same task / reward / hyperparameters from
each starting point and compare what happens.

## What the smoke does

- 4 nodes × 12 ranks = 48 ranks, FSDP `full_shard`
- `--task sum_digits` (the simplest GRPO task in the registry —
  "what is 3 + 7 + 2?" → "12", with a numeric-match reward)
- 50 GRPO steps per model
- Both models run back-to-back on the same allocation; the only
  difference between them is the `--model_name_or_path` value

We compare:

1. **Reward trajectory** — how fast does the reward signal climb
   from its initial value? An SFT'd model that's already in the
   chat format should respond to the reward signal sooner; a raw
   base model spends the first ~20 steps just learning what a
   completion looks like.
2. **Completion quality at step 50** — sample a few completions
   from each model after the smoke, see whether the SFT'd model
   produces well-formed answers vs the baseline producing
   open-ended garbage.

## Expected result

If the SFT helped the way we hope, the SFT model's reward should
climb faster (~5–10× faster in early steps) and converge to a
higher plateau by step 50. If both models look identical, the SFT
recipe didn't add what we need for RL initialization — possible
signal that the chat template / `assistant_only_loss` weren't
configured the way they need to be for downstream GRPO.

If the SFT model is *worse* (the early steps show reward
*degradation*), that would suggest the model's distribution is too
narrowed by SFT and the RL signal can't move it. Wouldn't be
unprecedented — over-SFT'd models can be hard to RL. The remedy
would be a longer mix with more pretraining-style data, or fewer
SFT epochs.

## Setup

| Field | Value |
|-------|-------|
| Trainer | TRL `GRPOTrainer` (HF Trainer base) |
| Task | `sum_digits` (numeric arithmetic, exact-match reward) |
| Scale | 4 nodes × 12 ranks = 48 ranks, FSDP `full_shard` |
| Steps | 50 per model |
| Hyperparameters | `beta=0.0` (KL term off), bf16 |
| Hardware | 4 XPU nodes from the eval allocation |
| Wall time estimate | ~10 min per model |
