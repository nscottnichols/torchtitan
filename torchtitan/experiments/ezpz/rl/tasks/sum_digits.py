# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Sum-of-digits task for GRPO training.
#
# Simple arithmetic verification task: "What is 3 + 7 + 2?" → "12"

import random
import re

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import (
    extract_answer,
    get_completion_text,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(
    num_samples: int = 1000,
    min_addends: int = 2,
    max_addends: int = 5,
    max_digit: int = 9,
    seed: int = 42,
) -> Dataset:
    """Generate sum-of-digits prompts with ground truth answers.

    Each sample is a prompt like "What is 3 + 7 + 2?" with answer "12".

    Returns:
        HuggingFace Dataset with columns: prompt (list[dict]), answer (str).
    """
    rng = random.Random(seed)
    prompts = []
    answers = []

    for _ in range(num_samples):
        num_addends = rng.randint(min_addends, max_addends)
        digits = [rng.randint(0, max_digit) for _ in range(num_addends)]
        total = sum(digits)

        expression = " + ".join(str(d) for d in digits)
        prompt = [
            {
                "role": "user",
                "content": f"What is {expression}? Reply with just the number.",
            }
        ]

        prompts.append(prompt)
        answers.append(str(total))

    return Dataset.from_dict({"prompt": prompts, "answer": answers})


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = extract_answer(text)
        if extracted is not None and extracted == str(expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.5 bonus if completion shows work (e.g. 'X + Y + Z = N')."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        if re.search(r"\d+\s*\+\s*\d+", text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="sum_digits",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, format_reward],
        description="Sum-of-digits arithmetic verification",
    )
)
