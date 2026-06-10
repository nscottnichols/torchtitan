# `ShardedTensor.device` fallback hardcodes CUDA, breaking FSDP resume on XPU

**Status:** Local workaround in
`torchtitan/experiments/ezpz/rl/train_sft.py`
(`_patch_sharded_tensor_device_for_xpu`). Not yet filed upstream.

**Affects:** torch 2.13 (likely older too — the bad code is unchanged
back to at least torch 2.5). Bites every XPU host that calls into HF
Trainer + accelerate's `load_fsdp_model` for checkpoint resume on a
sharded state dict.

## Summary

`torch.distributed._shard.sharded_tensor._ops.tensor_ops.tensor_device`
(`tensor_ops.py:54`) hardcodes `torch.device(torch.cuda.current_device())`
as the fallback device when a `ShardedTensor` has no local shards and
the process group backend is not gloo:

```python
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    dev: torch.device
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    elif pg and pg._get_backend_name() == "gloo":
        dev = torch.device("cpu")
    else:
        dev = torch.device(torch.cuda.current_device())   # <-- BUG
    return dev
```

On XPU, where torch is built without CUDA support, the fallback fires
with:

```
AssertionError: Torch not compiled with CUDA enabled
  File ".../torch/cuda/__init__.py", line 1201, in current_device
    _lazy_init()
  File ".../torch/cuda/__init__.py", line 518, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
```

## Reproducer (production trace, job 12468408 on Sunspot)

```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/.../train_sft.py", line 441, in main
[rank0]:     trainer.train(resume_from_checkpoint=rfc or None)
[rank0]:   File ".../transformers/trainer.py", line 1466, in _inner_training_loop
[rank0]:     model, train_dataloader = self._prepare_for_training(...)
[rank0]:   File ".../transformers/trainer.py", line 1644, in _prepare_for_training
[rank0]:     self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
[rank0]:   File ".../transformers/trainer.py", line 3362, in _load_from_checkpoint
[rank0]:     load_fsdp_model(...)
[rank0]:   File ".../accelerate/utils/fsdp_utils.py", line 221, in load_fsdp_model
[rank0]:     dist_cp.load(state_dict=..., planner=DefaultLoadPlanner(), ...)
[rank0]:   File ".../torch/distributed/checkpoint/state_dict_loader.py", line 266, in local_step
[rank0]:     planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
[rank0]:   File ".../torch/distributed/checkpoint/default_planner.py", line 323, in set_up_planner
[rank0]:     _init_state_dict(state_dict)
[rank0]:   File ".../torch/distributed/checkpoint/planner_helpers.py", line 509, in _init_state_dict
[rank0]:     _iterate_state_dict(state_dict, dtensor_func, sharded_tensor_func, tensor_func)
[rank0]:   File ".../torch/distributed/checkpoint/planner_helpers.py", line 489, in sharded_tensor_func
[rank0]:     device = getattr(value, "device", None)
[rank0]:   File ".../torch/distributed/_shard/sharded_tensor/api.py", line 1218, in dispatch
[rank0]:     return _SHARDED_OPS[func](types, args, kwargs, st._process_group)
[rank0]:   File ".../torch/distributed/_shard/sharded_tensor/_ops/tensor_ops.py", line 54, in tensor_device
[rank0]:     dev = torch.device(torch.cuda.current_device())
[rank0]: AssertionError: Torch not compiled with CUDA enabled
```

All 384 ranks of the 32N production SFT run flooded the same traceback
simultaneously, killing the resume from `checkpoint-100/`.

## Why the fallback fires here

`_init_state_dict` calls `getattr(value, "device", None)` on each
entry in the state dict. For ShardedTensors whose local shards are
metadata-only (a common case in HF Trainer's FSDP resume path — the
state dict is built lazily before sharded data is read in), the
`_local_shards` branch is taken with an empty list and falls through
to the CUDA fallback.

`tensor_func` and `dtensor_func` in the same file already do the
right thing — they go through
`dist.distributed_c10d._get_pg_default_device().type` →
`_get_device_module(device_type).current_device()`, which resolves
to `xpu` on XPU systems via the registered backend. Only the
`ShardedTensor` dispatch hardcodes CUDA.

## Proposed upstream fix

Replace the hardcoded fallback with a device-agnostic resolution that
mirrors what `planner_helpers.py:_init_state_dict` already does:

```python
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    if self_st._local_shards:
        return self_st._local_shards[0].tensor.device
    if pg and pg._get_backend_name() == "gloo":
        return torch.device("cpu")
    # Device-agnostic fallback (works on cuda/xpu/hpu/mps)
    from torch.distributed.distributed_c10d import _get_pg_default_device
    from torch.distributed._functional_collectives import _get_device_module
    device_type = _get_pg_default_device(pg).type
    return torch.device(_get_device_module(device_type).current_device())
```

(or, on torch >= 2.5, use the `torch.accelerator` namespace directly.)

## Local workaround (ezpz)

`torchtitan/experiments/ezpz/rl/train_sft.py` registers a replacement
`tensor_device` impl at startup via `_sharded_op_impl`, which
overwrites the entry in `_SHARDED_OPS`. The replacement uses
`torch.accelerator.current_device_index()` (torch >= 2.5) and falls
back to whichever accelerator namespace is available
(`torch.xpu`/`torch.cuda`/`torch.hpu`/`torch.mps`).

See `_patch_sharded_tensor_device_for_xpu` in `train_sft.py`. The
patch is called once at the top of `main()` before any HF Trainer
code runs.

## Impact

- **Without the patch:** FSDP checkpoint resume on XPU is broken in any
  pipeline that calls into `accelerate.utils.fsdp_utils.load_fsdp_model`
  (HF Trainer, accelerate native, anything that follows the same path).
  Every ezpz `auto-retry` restart loses all progress between
  checkpoints.
- **With the patch:** Resume works; the 32N tulu-mix SFT can survive
  a `ccl::v1::exception` mid-training and pick up from the most
  recent checkpoint instead of restarting at step 0.
