"""Backwards-compat shim for loading pre-rename DCP checkpoints.

Background: upstream commit (replayed in ezpz `03b9f486`) renamed
``Decoder.Config.output`` to ``Decoder.Config.lm_head``. This means
the model now exposes ``lm_head.weight`` where it previously had
``output.weight``. Existing production-training DCP checkpoints
(2B step 33,700+ and 20B step 4,100+) were saved before that rename,
so they have ``output.weight`` (and ``optimizer.param_groups.output.*``)
on disk.

DCP's load path matches state_dict keys against the on-disk metadata
strictly, so resuming from these checkpoints crashes with
``RuntimeError: Missing key in checkpoint state_dict: lm_head.weight``.

This module patches `CheckpointManager.dcp_load` to bridge the rename
during load:
  1. Rename `lm_head.*` -> `output.*` in the user's state_dict (so it
     matches the on-disk metadata).
  2. Call `dcp.load`, which fills the renamed entries from disk.
  3. Rename `output.*` back to `lm_head.*` so `model.load_state_dict`
     sees the keys it expects.

Apply once from the ezpz trainer entry point (idempotent). Once all
checkpoints have been resaved with the new naming, this shim can be
removed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpoint import MODEL, CheckpointManager

logger = logging.getLogger(__name__)

# Mapping: new (post-rename) prefix -> old (pre-rename) prefix.
# Renames apply to both the model state ('lm_head.weight') and the
# optimizer state ('optimizer.param_groups.lm_head.*',
# 'optimizer.state.lm_head.*').
RENAMES: tuple[tuple[str, str], ...] = (
    ("lm_head.weight", "output.weight"),
    ("optimizer.param_groups.lm_head.weight", "optimizer.param_groups.output.weight"),
    ("optimizer.state.lm_head.weight", "optimizer.state.output.weight"),
)

_PATCHED = False


def _rename_keys_in_place(
    d: dict[str, Any],
    pairs: tuple[tuple[str, str], ...],
) -> list[tuple[str, str]]:
    """Rename top-level keys in `d` matching any (src, dst) prefix.

    Matches `src` exactly, or any `src.<suffix>` form. Returns the
    list of (old_key, new_key) renames actually performed.
    """
    applied = []
    for src, dst in pairs:
        keys_to_rename = [k for k in d if k == src or k.startswith(src + ".")]
        for k in keys_to_rename:
            new_k = dst + k[len(src):]
            d[new_k] = d.pop(k)
            applied.append((k, new_k))
    return applied


def patch_checkpoint_manager() -> None:
    """Wrap `CheckpointManager.dcp_load` so legacy `output.*` keys load.

    Idempotent: subsequent calls are no-ops.
    """
    global _PATCHED
    if _PATCHED:
        return

    original_dcp_load = CheckpointManager.dcp_load

    def dcp_load_with_legacy_rename(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        # HF format goes through the StateDictAdapter, which already handles
        # `lm_head` <-> HF naming. Nothing to do here.
        if from_hf:
            return original_dcp_load(
                self, state_dict, checkpoint_id, from_hf, from_quantized
            )

        # Step 1: rename new -> old so DCP finds matching keys on disk.
        renames = _rename_keys_in_place(state_dict, RENAMES)
        if renames:
            logger.warning(
                "checkpoint_compat: renaming %d key(s) for legacy DCP load (%s)",
                len(renames),
                ", ".join(f"{a} -> {b}" for a, b in renames),
            )

        # Step 2: read from disk into the renamed slots.
        dcp.load(state_dict, checkpoint_id=checkpoint_id)

        # Step 3: rename back so model/optimizer load_state_dict sees current names.
        if renames:
            inverse_pairs = tuple((dst, src) for src, dst in RENAMES)
            _rename_keys_in_place(state_dict, inverse_pairs)

        # Step 4: drive model.load_state_dict explicitly, mirroring the
        # original dcp_load behavior.
        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)

    CheckpointManager.dcp_load = dcp_load_with_legacy_rename
    _PATCHED = True
    logger.info(
        "checkpoint_compat: patched CheckpointManager.dcp_load with legacy lm_head/output rename"
    )
