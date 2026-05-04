# Loss reporting is off by `dp_world_size` on TP > 1

**Affects:** All torchtitan runs with `tensor_parallel_degree > 1` since
upstream commit `1786292d` (2026-04-27). Both training and validation
loss reporting are wrong by a factor of `dp_world_size`. **Gradients
and optimizer steps are unaffected** — only the `loss:` field that
appears in stdout / W&B is wrong.

## Symptom

Step-1 loss for our `agpt_*` configs at vocab=256128 should be roughly
`ln(256128) ≈ 12.45 nats` (uniform softmax over the vocabulary). On
TP=1 runs this matches:

| Run | TP | Step 1 loss |
|---|---|---|
| `agpt_2b` (Apr 25) | 1 | **12.95** ✓ |

After the upstream change, TP > 1 runs show much smaller values:

| Run | TP | dp_size | Step 1 loss | True loss |
|---|---|---|---|---|
| `agpt_50b_wide` (May 3) | 2 | 12 | **1.07** ✗ | 12.84 (= 1.07 × 12) |
| `agpt_2b` TP=2 (May 3, no fix) | 2 | 12 | (would be) 1.08 | 12.94 |
| `agpt_2b` TP=2 (May 3, with fix) | 2 | 12 | **12.94** ✓ | 12.94 |

In all cases reported `loss = true_loss / dp_world_size`.

## Root cause

`torchtitan/distributed/utils.py:47-64` has a DTensor short-circuit in
`_dist_reduce`:

```python
if isinstance(x, DTensor):
    # DTensor path: ``full_tensor()`` already performs the mesh reduction
    # for Partial placements and is a no-op for Replicate. Skipping the
    # subsequent mesh all-reduce is required to avoid double-counting.
    ...
    return float(x.full_tensor().item())
```

This is correct **only if the DTensor's mesh equals the requested
reduction mesh**. But every call site of `dist_sum`/`dist_max` passes
`batch_mesh` or `loss_mesh` (= batch × cp), and the loss is a Replicated
DTensor on the **TP mesh** — orthogonal to the requested mesh.

`full_tensor()` on a Replicated DTensor is a no-op (no reduction across
TP ranks because Replicated already has the full value on every rank);
returning here without doing the requested reduction across `batch_mesh`
silently drops the cross-batch sum. Result: every batch rank reports
the same local per-token NLL without summing across batches.

## Where the loss enters this code path

The loss comes back as a Replicated DTensor because:

1. The trainer enables `loss_parallel()` whenever `tp_enabled` is True
   (`torchtitan/distributed/utils.py:317`).
2. `cross_entropy_loss(pred, labels)` inside the `loss_parallel()`
   context returns a Replicated DTensor on the TP mesh.
3. `loss / global_valid_tokens` (a Python float) keeps DTensor-ness.
4. The trainer's `dist_sum(loss, loss_mesh, ft_pg)` (`trainer.py:780`,
   `experiments/ezpz/trainer.py:520`) and the validator's
   `dist_sum(loss, parallel_dims.get_optional_mesh("loss"))`
   (`components/validate.py:320`) both hit the buggy DTensor branch.

## Workaround (local to ezpz)

Convert the loss DTensor to a plain tensor before calling
`dist_sum`/`dist_max`. The plain-tensor path of `_dist_reduce` does the
correct mesh all_reduce.

- **Training path:** `torchtitan/experiments/ezpz/trainer.py:503-516`
  unconditionally `loss = loss.full_tensor()` before the reduction.
- **Validation path:** `torchtitan/experiments/ezpz/validator.py`
  defines `EzpzValidator(Validator)` which overrides `validate()` with
  the same `full_tensor()` conversion. `_base_config` in
  `agpt/config_registry.py` builds the validator from
  `EzpzValidator.Config` instead of `Validator.Config`.

## Upstream

Filed as
[pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
(2026-05-03). The PR detects mesh axis overlap via `mesh_dim_names`:
skip the explicit reduction only when the DTensor's mesh shares an
axis with the requested mesh; otherwise materialize via
`full_tensor()` and fall through to the regular plain-tensor
reduction. Includes a regression test covering both the orthogonal
and same-mesh cases.

Once #3204 lands, the ezpz workaround can be removed.

## Implications for production dashboards

- **2B / 20B production W&B dashboards: correct.** Those configs use
  `tensor_parallel_degree=1`, so loss never enters the DTensor branch.
- **80B production W&B dashboards: under-reported by a factor of
  `dp_world_size`.** A 256N run (`world=3072`, TP=2) would report loss
  values 1536× too small. Multiply reported losses by
  `dp_world_size = world_size / tp_degree` to recover true NLL.

## Verification

The 2026-05-03 2B-TP=2-no-compile smoke test in
`logs/agpt-2b-tp2-nocompile-loss-test-v2/run.log` was run with the
ezpz workaround in place and shows step-1 loss = 12.94, matching the
known-good 2B-TP=1 baseline of 12.95.
