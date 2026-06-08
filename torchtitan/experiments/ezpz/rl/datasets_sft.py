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


# ---------------------------------------------------------------------------
# alpaca — broad instruction-following (52k examples)
# ---------------------------------------------------------------------------


def _build_alpaca() -> Dataset:
    """Load tatsu-lab/alpaca (52k instruction-following examples).

    Each row has {instruction, input, output}. We concatenate
    instruction+input into the user turn (with a blank line between
    them when input is non-empty) and use output as the assistant
    turn. Broader-purpose than math-only datasets — useful for
    teaching the base AuroraGPT-2B model to follow instructions
    other than just "do the math".
    """
    from datasets import load_dataset

    raw = load_dataset("tatsu-lab/alpaca", split="train")

    def _format(ex):
        instruction = ex["instruction"]
        if ex.get("input"):
            user_content = f"{instruction}\n\n{ex['input']}"
        else:
            user_content = instruction
        return {
            "prompt": [{"role": "user", "content": user_content}],
            "completion": [{"role": "assistant", "content": ex["output"]}],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="alpaca",
        build=_build_alpaca,
        description=(
            "Broad instruction-following (tatsu-lab/alpaca, ~52k "
            "examples). Use to teach instruction-following beyond "
            "math; mix with gsm8k/metamathqa via the 'math_alpaca_mix' "
            "entry."
        ),
    )
)


# ---------------------------------------------------------------------------
# math_alpaca_mix — interleaves metamathqa + gsm8k + alpaca by weight
# ---------------------------------------------------------------------------


def _build_math_alpaca_mix(
    weights=(0.6, 0.1, 0.3),
    seed: int = 42,
) -> Dataset:
    """Interleave metamathqa + gsm8k + alpaca with given probabilities.

    Default weight 0.6/0.1/0.3 keeps the math signal dominant (where
    we have the reward functions to exercise it) while exposing the
    model to ~30% general instruction-following data — useful for
    tasks like word_sort that aren't pure arithmetic.

    Uses ``interleave_datasets(stopping_strategy='all_exhausted')`` so
    smaller datasets cycle until the largest is exhausted; total
    yielded examples will be roughly ``max_size / max_weight`` so the
    sampled proportions actually match the requested weights.
    """
    from datasets import interleave_datasets

    if len(weights) != 3:
        raise ValueError(
            f"math_alpaca_mix weights must be (w_metamath, w_gsm8k, w_alpaca); "
            f"got {weights!r}"
        )
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0; got {sum(weights)}")

    return interleave_datasets(
        [_build_metamathqa(), _build_gsm8k(), _build_alpaca()],
        probabilities=list(weights),
        seed=seed,
        stopping_strategy="all_exhausted",
    )


register_sft_dataset(
    SFTDataset(
        name="math_alpaca_mix",
        build=_build_math_alpaca_mix,
        description=(
            # %% — argparse %-formats help strings so literal % must be
            # escaped or it crashes with "unsupported format character"
            # (we hit this in job 12468232 — '60% m' parses as %m).
            "Weighted mix: 60%% metamathqa + 10%% gsm8k + 30%% alpaca. "
            "Broader than math-only; better for downstream tasks "
            "that aren't pure arithmetic (e.g. word_sort)."
        ),
    )
)
