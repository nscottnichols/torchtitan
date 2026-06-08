# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, distribute_module, distribute_tensor
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ParallelStyle

from .token_dispatcher import AllToAllTokenDispatcher, DeepEPTokenDispatcher


class ExpertParallel(ParallelStyle):
    """ExpertParallel variant that accepts ezpz-owned token dispatchers."""

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

        assert hasattr(
            mod, "token_dispatcher"
        ), f"{type(mod)} missing token_dispatcher attribute"
        assert isinstance(
            mod.token_dispatcher,
            (AllToAllTokenDispatcher, DeepEPTokenDispatcher),
        ), (
            "Expected ezpz AllToAllTokenDispatcher or DeepEPTokenDispatcher, "
            f"got {type(mod.token_dispatcher)}"
        )
        mod.token_dispatcher.ep_mesh = device_mesh

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )
