# `training.dtype = bfloat16` silently freezes RMSNorm weights

> **Severity:** all completed agpt production training runs are
> affected. Loss curves are real but every RMSNorm parameter is
> stuck at its 1.0 init.
>
> Discovered: 2026-04-29

## Summary

When `training.dtype = "bfloat16"` (the default in
`agpt/config_registry.py` and `moe/config_registry.py` until commit
`<TBD>`), the master parameter copy is bf16 (no fp32 master). For
`RMSNorm.weight` initialized to `1.0`, the bf16 ULP at scale 1.0 is
`2^-7 ≈ 7.8e-3`, but the per-step optimizer update for those
parameters is much smaller (`lr * exp_avg / sqrt(hessian) ≈ 1.6e-5`
for the 2B SophiaG runs). Every update rounds to zero in bf16, so
`norm.weight` never moves from `1.0` for the entire training run.

This applies to all 25 RMSNorm parameters in the 2B model
(`norm.weight`, `layers.{i}.attention_norm.weight`,
`layers.{i}.ffn_norm.weight` for `i = 0..11`). All are exactly `1.0`
in every position from step 100 through step 17,400 of the
`agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072` checkpoints.

The 20B and any other agpt/moe runs with the same default dtype
have the same bug.

## How to confirm on a checkpoint

```python
import torch, torch.distributed.checkpoint as dcp

ckpt = "outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072/step-1000"
sd = {"norm.weight": torch.zeros(2048)}
dcp.load(sd, checkpoint_id=ckpt)
print(sd["norm.weight"][:5])     # -> tensor([1., 1., 1., 1., 1.])
print(torch.allclose(sd["norm.weight"], torch.ones(2048)))  # -> True
```

Even at step 17,400 the same tensor is uniform 1.0. By contrast,
`layers.0.attention.qkv_linear.wq.weight` shows real changes
(max abs diff ~1.6e-2 = 2 bf16 ULPs at scale ~0.001).

## Why other parameters update fine

Linear layers initialize at small scales (`std ≈ 0.02` for q/k/v,
`std ≈ 0.005` for o). A bf16 ULP at scale `0.005` is `~3.8e-5`,
roughly the same magnitude as the per-step update — so updates can
register, especially after a few steps of accumulation. RMSNorm
weights are 100x larger (1.0 vs 0.01), so their ULP is 100x coarser
relative to the same update size.

## Why optimizer state shows non-zero exp_avg / hessian

The optimizer DOES receive gradients and DOES accumulate exp_avg /
hessian. The update is computed correctly. The failure is in the
final write to the parameter: the ~1.6e-5 update rounds to zero
when added to a bf16 1.0.

We confirmed `optimizer.state.norm.weight.exp_avg ≈ 2.78e-04` and
`hessian ≈ 1.52e-07` at step 1000 — both larger than the
`qkv_linear.wq.weight` equivalents (`5.15e-06` and `2.45e-10`).

## Fix

`agpt/config_registry.py` and `moe/config_registry.py` now default
`dtype = "float32"`. With FSDP `MixedPrecisionPolicy(param_dtype=bf16,
reduce_dtype=fp32)`:

- Master params: fp32 (sub-ulp accumulation works)
- Forward / backward all-gather: bf16 (memory-efficient)
- Gradient reduce: fp32 (numerical stability)
- Optimizer step: on fp32 master

Memory cost: ~1 GB extra master at 2B, ~10 GB at 20B. Well under
budget at production scale.

## Existing checkpoints

The 2B step-33,700 and 20B step-4,100 checkpoints inherit the
problem — every RMSNorm.weight is exactly 1.0. Resuming with the
new default doesn't retroactively fix them; the optimizer state
still has stale `exp_avg` / `hessian`. Two paths forward:

1. **Continue training from existing checkpoint with `dtype=float32`** —
   norm weights start at 1.0 (already-trained value, same as init),
   then begin accumulating real updates from this point on. Loss
   trajectory should slowly diverge from what we'd see resuming with
   `dtype=bf16`.

2. **Restart from scratch** — only justified if we conclude that
   17K (resp. 4K) steps of stuck-norm training is so degenerate
   that resuming makes no sense.

We chose path 1 for the 2B/20B production runs.

## Related

- The "DCP -> HF converter has an embedding bug" diagnosis in earlier
  versions of `docs/evals/agpt/2b/README.md` was wrong. The converter
  is innocent; the original observation (`tok_embeddings.weight[0]`
  identical across checkpoints) was a frozen padding-token row, not
  a generic stuck-embedding issue. Embeddings DO update for
  non-padding tokens.

- bf16 master weights in distributed training is a recognized
  general issue; see torchtitan #600 and the FSDP2 docs on
  `MixedPrecisionPolicy`.
