# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Generic HuggingFace dataset support for torchtitan experiments.
#
# Import this module to enable arbitrary HF datasets:
#
#   import torchtitan.experiments.ezpz.datasets  # noqa: F401
#
# Then any HF dataset path works directly:
#
#   --dataloader.dataset stanfordnlp/imdb
#   --dataloader.dataset HuggingFaceFW/fineweb-edu
#   --dataloader.dataset eliplutchok/fineweb-small-sample
#
# Or register one explicitly for custom text columns / configs:
#
#   from torchtitan.experiments.ezpz.datasets import register_hf_dataset
#   register_hf_dataset("my_data", "my-org/my-data", text_column="content")

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from datasets import load_dataset

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets import text_datasets
from torchtitan.hf_datasets.text_datasets import DATASETS

log = logging.getLogger(__name__)

# Silence noisy HTTP request logs from huggingface_hub / httpx / urllib3
for _noisy in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _make_loader(
    *,
    config_name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> Callable:
    """Build a dataset loader callable for the DATASETS registry.

    Returns a callable that accepts a dataset path (from the registry or
    CLI override) and returns a HF dataset object.
    """

    def _load(dataset_path: str) -> Any:
        kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        if config_name is not None:
            kwargs["name"] = config_name
        return load_dataset(dataset_path, **kwargs)

    return _load


def _make_text_processor(text_column: str = "text") -> Callable:
    """Build a sample processor that extracts text from a given column."""

    def _process(sample: dict[str, Any]) -> str:
        return sample[text_column]

    return _process


def register_hf_dataset(
    name: str,
    path: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> DatasetConfig:
    """Register a HuggingFace dataset for use with torchtitan's data pipeline.

    The registered dataset can be used with the existing
    HuggingFaceTextDataset/HuggingFaceTextDataLoader infrastructure,
    which handles streaming, distributed sharding, tokenization,
    checkpointing, and infinite looping.

    Args:
        name: Registry key (used as --dataloader.dataset <name>).
        path: HuggingFace dataset path (e.g. "HuggingFaceFW/fineweb-edu").
        config_name: Dataset config/subset (e.g. "default", "en").
        split: Dataset split (default "train").
        text_column: Column containing the text to train on.
        streaming: Use HF streaming mode (default True).
        trust_remote_code: Allow running dataset scripts from the hub.

    Returns:
        The registered DatasetConfig.

    Example:
        >>> register_hf_dataset("fineweb_edu", "HuggingFaceFW/fineweb-edu")
        >>> # Then in config: --dataloader.dataset fineweb_edu
    """
    config = DatasetConfig(
        path=path,
        loader=_make_loader(
            config_name=config_name,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        ),
        sample_processor=_make_text_processor(text_column),
    )
    DATASETS[name] = config
    log.debug(f"Registered HF dataset {name!r} -> {path}")
    return config


# ---------------------------------------------------------------------------
# Auto-registration fallback: treat unknown dataset names as HF hub paths
# ---------------------------------------------------------------------------

# Save the original validator so we can call it for known datasets
_original_validate_dataset = text_datasets._validate_dataset


def _validate_dataset_with_fallback(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset, auto-registering unknown names as HF hub datasets.

    If dataset_name is in the registry, behaves identically to the original.
    Otherwise, treats dataset_name as a HuggingFace hub path (e.g.
    "stanfordnlp/imdb", "eliplutchok/fineweb-small-sample") and auto-registers
    it as a streaming dataset with text_column="text".
    """
    if dataset_name in DATASETS:
        return _original_validate_dataset(dataset_name, dataset_path)

    # Auto-register: treat the dataset name as a HF hub path.
    # Force dataset_path=None so the registered hub path is used
    # instead of any stale local path from the config (e.g. books.txt).
    log.info(
        f"Dataset {dataset_name!r} not in registry — "
        f"auto-registering as streaming HF dataset"
    )
    register_hf_dataset(
        name=dataset_name,
        path=dataset_name,
        streaming=True,
    )
    return _original_validate_dataset(dataset_name, dataset_path=None)


# Patch the validator so all code paths (HuggingFaceTextDataset,
# HuggingFaceTextDataLoader, BlendCorpusDataLoader) benefit.
text_datasets._validate_dataset = _validate_dataset_with_fallback


# ---------------------------------------------------------------------------
# Pre-registered common datasets
# ---------------------------------------------------------------------------

register_hf_dataset(
    "fineweb_edu",
    "HuggingFaceFW/fineweb-edu",
    config_name="default",
)

register_hf_dataset(
    "fineweb",
    "HuggingFaceFW/fineweb",
    config_name="default",
)

register_hf_dataset(
    "slimpajama",
    "cerebras/SlimPajama-627B",
)

register_hf_dataset(
    "pile",
    "monology/pile-uncopyrighted",
)

register_hf_dataset(
    "openwebtext",
    "Skylion007/openwebtext",
    streaming=False,
)

register_hf_dataset(
    "wikitext",
    "wikitext",
    config_name="wikitext-103-raw-v1",
    streaming=False,
)

register_hf_dataset(
    "c4_streaming",
    "allenai/c4",
    config_name="en",
)
