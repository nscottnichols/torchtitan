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


def build_streaming_or_finite(sample_fn, num_samples: int):
    """Wrap a per-sample generator function into an HF Dataset.

    The convention: ``num_samples == 0`` means stream forever
    (returns an ``IterableDataset`` — every training step sees a
    fresh prompt, so the model can't memorize a fixed pool). Any
    positive integer materializes a finite ``Dataset`` of that size,
    which is what older training code expects.

    ``sample_fn`` is a zero-arg callable that returns one dict per
    call (must contain at least ``prompt`` and ``answer`` keys).
    The caller is responsible for seeding the RNG inside sample_fn
    so reproducibility behaves correctly.
    """
    from datasets import Dataset, IterableDataset

    if num_samples == 0:
        def _stream():
            while True:
                yield sample_fn()
        return IterableDataset.from_generator(_stream)

    return Dataset.from_list([sample_fn() for _ in range(num_samples)])
