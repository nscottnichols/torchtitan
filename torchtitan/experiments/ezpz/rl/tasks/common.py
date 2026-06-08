# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Shared helpers for reward functions across tasks.

import re


def extract_answer(text: str) -> str | None:
    """Extract a numeric answer from model completion.

    Tries several patterns in order:
    1. \\boxed{...} (LaTeX-style)
    2. "answer is X" or "= X"
    3. Last standalone number in the text
    """
    # \\boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # "answer is X" or "the answer is X"
    match = re.search(r"(?:the\s+)?answer\s+is\s+(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # "= X" at end of expression
    match = re.search(r"=\s*(\d+)", text)
    if match:
        return match.group(1).strip()

    # Last standalone number
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[-1]

    return None


def get_completion_text(completion) -> str:
    """Extract text from a completion (str or chat-format list[dict])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


_DEFAULT_LARGE_POOL = 100_000


def build_streaming_or_finite(sample_fn, num_samples: int):
    """Wrap a per-sample generator function into an HF Dataset.

    The convention: ``num_samples == 0`` means "as close to streaming
    as TRL supports" — materializes a large finite pool of
    ``_DEFAULT_LARGE_POOL`` (100,000) randomly-generated samples so
    a typical training run never reuses the same prompt. Any positive
    integer materializes that many samples.

    Why not a true ``IterableDataset``: TRL's ``GRPOTrainer`` rejects
    iterable datasets at __init__ (see trl#3213,
    ``trl/trainer/grpo_trainer.py:602``). A 100k pool at e.g.
    GBS=48 / max_steps=1000 means each prompt is seen at most ~2× on
    average rather than ~48× with the old default of 1000.

    ``sample_fn`` is a zero-arg callable that returns one dict per
    call (must contain at least ``prompt`` and ``answer`` keys).
    The caller is responsible for seeding the RNG inside sample_fn
    so reproducibility behaves correctly.
    """
    from datasets import Dataset

    effective_n = _DEFAULT_LARGE_POOL if num_samples == 0 else num_samples
    return Dataset.from_list([sample_fn() for _ in range(effective_n)])
