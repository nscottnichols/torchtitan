# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SFT dataset registry. Each entry is a (load_fn, description) pair;
# load_fn takes no args and returns an HF Dataset with columns:
#   - prompt: list[dict] of chat messages (user turn(s))
#   - completion: list[dict] of chat messages (assistant turn(s))
# SFTTrainer with assistant_only_loss=True will only compute loss on
# the assistant turn(s), which is the conventional SFT setup.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset


@dataclass
class SFTDataset:
    name: str
    build: Callable[..., Dataset]
    description: str = ""


SFT_REGISTRY: dict[str, SFTDataset] = {}


def register_sft_dataset(ds: SFTDataset) -> SFTDataset:
    SFT_REGISTRY[ds.name] = ds
    return ds


def get_sft_dataset(name: str) -> SFTDataset:
    if name not in SFT_REGISTRY:
        available = ", ".join(sorted(SFT_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown SFT dataset {name!r}. Available: {available}")
    return SFT_REGISTRY[name]


# ---------------------------------------------------------------------------
# gsm8k — grade-school math word problems with step-by-step solutions
# ---------------------------------------------------------------------------


def _build_gsm8k() -> Dataset:
    """Load gsm8k 'main' split (7473 train examples) and format as chat.

    Each gsm8k row has {question, answer} where answer is the chain-of-
    thought ending with '#### N'. We format the prompt as a user turn
    asking the question and the completion as an assistant turn
    containing the full CoT solution including the '#### N' answer
    marker. extract_answer() in tasks.common already matches this format.
    """
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["question"]}],
            "completion": [{"role": "assistant", "content": ex["answer"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="gsm8k",
        build=_build_gsm8k,
        description=(
            "Grade-school math word problems with step-by-step CoT "
            "solutions (openai/gsm8k 'main' split, 7473 examples). "
            "Answers end with '#### N' format already handled by "
            "tasks.common.extract_answer."
        ),
    )
)


# ---------------------------------------------------------------------------
# metamathqa — augmented + rephrased GSM8K+MATH, ~400k examples
# ---------------------------------------------------------------------------


def _build_metamathqa() -> Dataset:
    """Load MetaMathQA (~395k augmented GSM8K+MATH solutions).

    Columns: query, response, type, original_question. We use query +
    response and ignore the rest. Much bigger than gsm8k — useful when
    you have the compute and want stronger generalization.
    """
    from datasets import load_dataset

    raw = load_dataset("meta-math/MetaMathQA", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["query"]}],
            "completion": [{"role": "assistant", "content": ex["response"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="metamathqa",
        build=_build_metamathqa,
        description=(
            "Augmented + rephrased GSM8K+MATH solutions "
            "(meta-math/MetaMathQA, ~395k examples). Stronger "
            "generalization than gsm8k at higher cost."
        ),
    )
)
