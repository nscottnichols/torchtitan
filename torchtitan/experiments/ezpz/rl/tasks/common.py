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
