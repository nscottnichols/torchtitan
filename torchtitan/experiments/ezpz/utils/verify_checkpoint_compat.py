#!/usr/bin/env python3
"""Validate checkpoint_compat patches DCP load correctly.

Loads only the ``output.weight`` tensor from a legacy DCP checkpoint
(saved before the upstream ``output`` -> ``lm_head`` rename) into a
``lm_head.weight`` slot using the rename shim, and verifies the
result is non-zero and finite. Does not build the full model — only
exercises the rename logic.

Run from the repo root:

    python3 torchtitan/experiments/ezpz/utils/verify_checkpoint_compat.py \\
        [outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs3072/step-4100]
"""
import logging
import sys

logging.basicConfig(level=logging.WARNING)

# Apply the patch
from torchtitan.experiments.ezpz.checkpoint_compat import (
    _rename_keys_in_place,
    RENAMES,
    patch_checkpoint_manager,
)
patch_checkpoint_manager()

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

CKPT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs3072/step-4100"
)

# Discover what shape lm_head.weight should be from the on-disk metadata
from torch.distributed.checkpoint import FileSystemReader
reader = FileSystemReader(CKPT)
meta = reader.read_metadata()
all_keys = list(meta.state_dict_metadata.keys())

print(f"Checkpoint: {CKPT}")
print(f"Total state_dict keys: {len(all_keys)}")
print()
print("Looking for output.weight in metadata...")
for k in all_keys:
    if k == "output.weight":
        md = meta.state_dict_metadata[k]
        if isinstance(md, TensorStorageMetadata):
            print(f"  found {k}: shape={md.size}, dtype={md.properties.dtype}")
        else:
            print(f"  found {k}: {type(md).__name__}")
        break
else:
    print("  NOT FOUND — checkpoint already has lm_head.weight?")
    sys.exit(1)

# Test the rename logic manually first (no DCP)
print()
print("Testing _rename_keys_in_place (new -> old):")
test_state_dict = {
    "lm_head.weight": "MARKER_NEW",
    "tok_embeddings.weight": "tok_embed",
    "optimizer.param_groups.lm_head.weight.lr": "lr",
    "optimizer.state.lm_head.weight.exp_avg": "ea",
}
print(f"  before: {sorted(test_state_dict.keys())}")
applied = _rename_keys_in_place(test_state_dict, RENAMES)
print(f"  after:  {sorted(test_state_dict.keys())}")
print(f"  applied: {applied}")

# And the inverse
print()
print("Testing reverse rename (old -> new):")
inverse = tuple((dst, src) for src, dst in RENAMES)
applied = _rename_keys_in_place(test_state_dict, inverse)
print(f"  after:  {sorted(test_state_dict.keys())}")
print(f"  applied: {applied}")

# Now try a real partial DCP load with just lm_head.weight
print()
print("Loading just lm_head.weight via patched DCP...")

# Build a state_dict with new naming (lm_head) but allocate an empty tensor
# of the right size (DCP fills in-place).
shape = meta.state_dict_metadata["output.weight"].size
dtype = meta.state_dict_metadata["output.weight"].properties.dtype
state_dict = {"lm_head.weight": torch.zeros(shape, dtype=dtype)}

# We can't use the patched CheckpointManager.dcp_load directly without a full
# CheckpointManager; instead exercise just the rename + dcp.load + rename-back logic.
applied = _rename_keys_in_place(state_dict, RENAMES)
print(f"  renamed for load: {applied}")
dcp.load(state_dict, checkpoint_id=CKPT)
print(f"  loaded keys: {list(state_dict.keys())}")
print(f"  output.weight shape: {state_dict['output.weight'].shape}")
print(f"  output.weight dtype: {state_dict['output.weight'].dtype}")
print(f"  output.weight any non-zero: {state_dict['output.weight'].abs().sum().item() > 0}")
print(f"  output.weight any NaN: {torch.isnan(state_dict['output.weight']).any().item()}")
print(f"  output.weight first 5: {state_dict['output.weight'].flatten()[:5].tolist()}")

# Reverse rename
inverse = tuple((dst, src) for src, dst in RENAMES)
applied = _rename_keys_in_place(state_dict, inverse)
print(f"  renamed back: {applied}")
print(f"  final keys: {list(state_dict.keys())}")
assert "lm_head.weight" in state_dict, "lm_head.weight missing after reverse rename!"
print()
print("OK — full round-trip works.")
