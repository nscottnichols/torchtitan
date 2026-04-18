# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Sum-of-digits task and reward functions for GRPO training.

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


def _get_text(completion) -> str:
    """Extract text from a completion (str or chat-format list[dict])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


def sum_digits_reward(completions, answer, **kwargs) -> list[float]:
    """Reward function for sum-of-digits task.

    Returns 1.0 if the extracted answer matches the expected answer, 0.0 otherwise.

    Args:
        completions: Model-generated completions (str or chat-format).
        answer: Ground truth answers (as strings).
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        text = _get_text(completion)
        extracted = extract_answer(text)
        if extracted is not None and extracted == str(expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def sum_digits_format_reward(completions, **kwargs) -> list[float]:
    """Bonus reward for completions that show work (e.g. 'X + Y + Z = N').

    Returns 0.5 if the completion contains an addition expression, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if re.search(r"\d+\s*\+\s*\d+", text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
