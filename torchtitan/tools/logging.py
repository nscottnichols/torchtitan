# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import ezpz


logger = ezpz.get_logger(__name__)


def get_logger(name: str) -> logging.Logger:
    return ezpz.get_logger(name)


def init_logger(name: str | None = None):
    level = logging.INFO if ezpz.get_rank() == 0 else logging.CRITICAL
    logger = ezpz.get_logger(name or __name__)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
