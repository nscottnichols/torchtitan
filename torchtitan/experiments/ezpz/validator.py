# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Subclass of upstream Validator that fixes loss reporting on TP > 1.

Upstream's `_dist_reduce` (torchtitan/distributed/utils.py) short-circuits
DTensor inputs by returning `float(x.full_tensor().item())` and skips the
requested mesh all_reduce. That is correct only when the DTensor's mesh
equals the reduction mesh. The validator's loss reduction passes the
loss_mesh (= batch x cp), but the loss is a Replicated DTensor on the TP
mesh — orthogonal to loss_mesh — so the cross-batch reduction is silently
dropped and reported val loss is off by a factor of `dp_world_size`.

This subclass overrides `validate()` to convert the loss DTensor to a
plain tensor before calling `dist_sum`, so the regular all_reduce path
runs and val loss is correct on TP > 1.

See docs/guides/known-bugs/loss-reporting-tp-dist-reduce.md.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.tools import utils


class EzpzValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        pass

    def _get_validation_dataloader(self):
        """Cache the validation dataloader on the instance.

        Upstream `Validator.validate()` rebuilds it on every call. For the
        default `c4_validation` HF stream that's cheap, but for our
        blendcorpus-backed loader it re-runs `build_gpt_datasets()` every
        validation pass (cached index file load + dataset wrapper
        construction + a no-op `bc_mpu` re-init guarded by our patch),
        which adds visible noise to the log and ~few hundred ms of
        wasted work per call.

        `BlendCorpusDataLoader.__iter__` yields a fresh iterator each
        call by re-instantiating the underlying torch DataLoader, so it
        is safe to cache the wrapper and re-iterate it per validation.
        """
        if getattr(self, "_cached_dataloader", None) is None:
            self._cached_dataloader = self.dl_config.build(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                local_batch_size=self.local_batch_size,
                parallel_dims=self.parallel_dims,
            )
        return self._cached_dataloader

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        validation_dataloader = self._get_validation_dataloader()

        for input_dict, labels in validation_dataloader:
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            self.metrics_processor.ntokens_since_last_log += labels.numel()
            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels, model_parts
            )

            local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=device_type)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()

            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh, None
                )
            else:
                global_valid_tokens = local_valid_tokens.float()

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None
                with self.validation_context():
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )

                if self.pp_has_last_stage:
                    assert losses is not None
                    loss_sum = torch.sum(torch.stack(losses)).to(device_type)
                else:
                    loss_sum = torch.tensor([-1.0], device=device_type)
            else:
                with self.validation_context():
                    assert len(model_parts) == 1
                    predictions = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    loss_sum = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss_sum.detach() / global_valid_tokens)
            num_steps += 1

        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        # Workaround for upstream `_dist_reduce` skipping the mesh all_reduce
        # when the input is a DTensor. The loss is a Replicated DTensor on
        # the TP mesh; we want a sum across batch_mesh (orthogonal). Convert
        # to a plain tensor first so the regular reduction runs.
        if isinstance(loss, DTensor):
            loss = loss.full_tensor()
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_sum(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = float(loss.item())

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        for model in model_parts:
            model.train()
