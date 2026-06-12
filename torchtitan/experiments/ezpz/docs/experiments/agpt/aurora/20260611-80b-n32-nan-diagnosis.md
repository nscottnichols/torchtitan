# 80B n=32 NaN diagnosis — bf16 forward overflow at GBS≥96

**Date**: 2026-06-11
**Status**: Diagnosis complete, production-fix candidates in flight

## TL;DR

The 80B training stack NaNs deterministically at small node counts on
Aurora. **n=16 (GBS=96) is the largest configuration that trains clean
for 20 steps.** At n=32 (GBS=192) loss goes NaN at step 6; at
n=256 (8530891) loss went NaN at step 2.

A 7-job factorial at n=32 isolated the cause as a **bf16 forward-path
overflow**:

- Not LR (LR=1e-7 NaN'd at the same step as LR=1e-6)
- Not data (different seed NaN'd at the same step)
- Not the master dtype (`training.dtype=float32` doesn't change activations)
- Not the optimizer/grad clip (post-clip grads ≤1.0 by default; NaN
  happens upstream in the forward pass)
- **It IS bf16 activations**: with `training.mixed_precision_param=float32`
  the model survives 20 steps including grad_norm spikes of 21K–79K
  that would have produced inf activations in bf16.

The grad_norm spikes themselves are present at every config including
the clean n=16 baseline — they happen around steps 15–17 with the
specific weight state Adam reaches. At GBS=96 bf16's dynamic range
just barely absorbs the resulting activations and the model recovers;
at GBS=192 it doesn't.

## What we know after the factorial

| Test | n | TP | LBS | GBS | dtype/clip | Result | NaN onset |
|------|--:|---:|---:|----:|------------|--------|-----------|
| n=16 baseline | 16 |  2 | 1 |  96 | bf16, clip 1.0 | ✓ 20 clean steps, loss 12.93 → 10.41 | — |
| n=32 baseline | 32 |  2 | 1 | 192 | bf16, clip 1.0 | NaN step 6 (grad_norm `inf` step 2, recovered, then nan) | step 6 |
| n=32 LR=1e-7  | 32 |  2 | 1 | 192 | bf16, clip 1.0, LR=1e-7 | NaN step 6 (same shape) | step 6 |
| n=32 seed=12345 | 32 |  2 | 1 | 192 | bf16, clip 1.0, seed=12345 | NaN step 6 (different per-step losses, same onset) | step 6 |
| n=32 dtype=fp32 (wrong flag) | 32 |  2 | 1 | 192 | bf16 acts, fp32 master, clip 1.0 | NaN step 2 (`training.dtype` only sets master) | step 2 |
| n=32 fp32 acts | 32 |  2 | 1 | 192 | **fp32**, clip 1.0 | **OOM at model init** (TP=2 + fp32 acts ≈ 80 GiB > 64 GiB tile cap) | — |
| n=32 TP=4 GBS=96 | 32 |  4 | 1 |  96 | bf16, clip 1.0 | NaN step 18 (grad_norm explosion 5→50, didn't recover) | step 18 |
| n=32 TP=4 GBS=192 | 32 |  4 | 2 | 192 | bf16, clip 1.0 | NaN step 2 (`loss = -inf`) | step 2 |
| **n=32 TP=4 fp32 acts GBS=96** | 32 |  4 | 1 |  96 | **fp32**, clip 1.0 | **✓ 20 clean steps**, loss 12.96 → 10.93, grad_norm spikes 21K, 79K, 55K, 15K, 6K | — |

n=16 baseline and the TP=4 + fp32-activations run are the only two
configs that survive 20 steps. They share GBS=96, and the fp32 run
proves the bf16 forward path is the failure point: under fp32 the
model absorbs grad_norm spikes of up to 79K and keeps training; under
bf16 those same spikes drive activations into inf.

## Per-step grad_norm — the smoking gun

Side-by-side at GBS=96 (n=16 baseline vs n=32 TP=4 same GBS, both bf16):

| Step | n=16 TP=2 grad_norm | n=32 TP=4 grad_norm | n=32 TP=4 fp32-acts grad_norm |
|-----:|--------------------:|--------------------:|------------------------------:|
| 1 |   4.79 |   4.86 |   4.87 |
| 2 |   4.87 |   4.94 |   4.95 |
| 3 |   4.92 |   5.00 |   5.00 |
| 4 |   4.90 |   4.97 | **21,309** |
| 5 |   4.89 |   4.97 | **79,284** |
| 6 |   5.09 |   5.16 |   5.10 |
| 7 |   5.18 |   5.27 |   5.14 |
| 8 |   5.37 |   5.47 |   5.27 |
| 9 |   5.62 |   5.72 | **54,873** |
| 10 |   6.14 |   6.26 |   5.84 |
| 11 |   6.53 |   6.83 |   6.26 |
| 12 |   7.12 |   7.43 | **15,028** |
| 13 |   7.92 |   8.00 | **6,452** |
| 14 |   7.42 |   7.45 |   8.20 |
| 15 |  16.80 |  17.23 |   8.23 |
| 16 |  40.50 |  42.13 |   8.37 |
| 17 |  49.46 |  51.83 |  71.07 |
| 18 |  45.90 | **nan** |  91.35 |
| 19 |  27.05 |   nan  |  60.99 |
| 20 |  23.07 |   nan  |  64.05 |

Two things stand out:

1. **The bf16 runs at GBS=96 produce nearly identical grad_norms** (n=16
   TP=2 and n=32 TP=4). Both spike to ~50 at steps 15–17. n=16 survives
   the spike, n=32 doesn't. The difference is presumably small numerical
   drift in the loss-reduction path or the TP collectives.

2. **The fp32-activations run produces TOTALLY DIFFERENT grad_norms** —
   massive spikes (21K, 79K, 55K, 15K, 6K) at steps 4–13 that the bf16
   versions never see. Those spikes are the *real* gradient magnitudes;
   bf16 silently clips them via overflow, producing the "tame" 5–7
   range we see in bf16 logs. By the time `max_norm` clipping runs,
   the gradient buffer is already partial garbage.

The implication: bf16 has been masking enormous gradient activity
for the first ~13 steps even in the working configs. The model only
NaNs once that masking flips from "lucky" to "inf-producing" — which
in turn depends on the exact weight state. GBS=192 puts you over the
edge fast; GBS=96 stays just under it.

## Why `training.dtype=float32` isn't the fix

torchtitan has two dtype configs:

- `training.dtype` — master weight dtype (default `float32` in v2)
- `training.mixed_precision_param` — FSDP MixedPrecisionPolicy `param_dtype`,
  i.e. the dtype activations and parameter copies use during forward/backward
  (default `bfloat16`)

The first attempt at the bf16-overflow test (`n32-fp32-dtype.sh`,
8536657) only set `training.dtype=float32`, which is already the v2
default. Memory stayed at 41.68 GiB confirming activations were still
bf16. The fix is `training.mixed_precision_param=float32`, which doubles
activation memory.

## Why fp32 isn't free

`training.mixed_precision_param=float32` doubles activation memory:

- TP=2 LBS=1: ~80 GiB peak (exceeds the 64 GiB per-tile cap → OOM at
  model init)
- TP=4 LBS=1: ~42 GiB peak (fits; this is the validated working config)

It also halves throughput on Aurora's XPU (no native bf16 matmul speedup
when activations are fp32). The TP=4 fp32-act run at 8.3 TFLOPs/GPU vs
the bf16 baseline's 26 TFLOPs/GPU — **3.1× slower**. At the same TP
the gap is closer to 5×.

## Production-fix candidates

Ranked by cost-if-it-works:

| Candidate | Hypothesis | Throughput cost | Status |
|-----------|------------|-----------------|--------|
| Tighter `max_norm` (1.0 → 0.1) | Smaller weight updates → weights stay near init → bf16 forward doesn't overflow | ~0% | **❌ Refuted by 8539593** (NaN'd at step 4 — earlier than baseline, since clipping is post-backward and can't prevent forward-pass overflow) |
| Lower LR (1e-6 → 1e-8) | Same mechanism as tighter clip | ~0% (slower convergence in nat/token but not in throughput/sec) | Untested |
| Logit softcap (Gemma-2 style) | Caps logits pre-softmax → fewer paths to inf | ~0% via FlexAttention | **Not available on XPU** (FlexAttention unsupported, would need custom impl) |
| `mixed_precision_param=float32` at TP=4 | Forces all-fp32 forward/backward | **~3-5× slower** | **Validated** (8537349 cleanly trained 20 steps) |

**Update (2026-06-12)**: tighter `max_norm` was refuted by 8539593 —
NaN at step 4 instead of step 6. This makes sense in hindsight:
gradient clipping runs AFTER backward, so by the time `max_norm`
would constrain the gradient, the bf16 forward has already
overflowed and the gradient tensor already contains nan. Clipping
nan→nan doesn't help.

**The only known production-viable fix is `mixed_precision_param=float32`
at TP=4**, with the 3-5× throughput cost.

## Implications for production

- **256N (GBS=1536) likely needs the fp32-activations fix** unless
  tighter clipping works. The NaN onset gets earlier as GBS increases
  (step 6 at GBS=192, step 2 at GBS=1536), which is consistent with
  the bf16-overflow story: more aggressive gradient signal per step
  drives weights into the overflow regime faster.
- Until we have a production-viable fix, **256N 80B production stays
  blocked**.
- The 4N validation run is unaffected (GBS=24, well below the
  failure threshold). 8N and 16N also fine (GBS=48 / 96).

## Test jobs

| Job ID | Script | Wandb |
|--------|--------|-------|
| 8536194 | n16-smoke.sh | https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/nivubyxr |
| 8536249 | n32-smoke.sh (baseline, NaN'd) | (job 8536249) |
| 8536417 | n32-smoke-lr1e-7.sh | (job 8536417) |
| 8536916 | n32-seed-shift.sh | (job 8536916) |
| 8536657 | n32-fp32-dtype.sh (wrong flag) | (job 8536657) |
| 8536967 | n32-fp32-mixed-precision.sh (OOM) | (job 8536967) |
| 8537029 | n32-tp4-lbs1.sh | (job 8537029) |
| 8537168 | n32-tp4-lbs2.sh | (job 8537168) |
| 8537349 | n32-tp4-fp32-mp.sh (✓ 20 steps) | (job 8537349) |
| 8539568 | n32-tight-clip.sh (PBS protocol flake — preflight failed) | — |
| 8539593 | n32-tight-clip.sh (resubmit) — NaN'd step 4, refuting tight-clip hypothesis | (job 8539593) |

All logs in `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/80b-*.o*`.

## Related

- The `--training.mixed-precision-param=float32` workaround mirrors
  what was needed for 70B+ models in the [Llama 3 405B paper](https://arxiv.org/abs/2407.21783)
  (their fp32-residual fix solved a similar bf16-overflow problem at
  scale).
- See also `docs/guides/training-dtype-bf16-norm-freeze.md` for the
  prior bf16-master regression (different problem: master weights stuck
  at init due to bf16 ULP > optimizer step). That fix doesn't help here
  because the failure is in the forward path, not the master copy.
