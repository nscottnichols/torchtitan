# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Multiplication task for GRPO training.
#
# "What is 7 × 8?" → "56"
# Harder than sum_digits: requires multiplication, larger answer space.

import random

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
    num_factors: int = 2,
    max_factor: int = 12,
    seed: int = 42,
) -> Dataset:
    """Generate multiplication prompts with ground truth answers.

    Args:
        num_samples: Number of samples to generate.
        num_factors: Number of factors per problem (default 2).
        max_factor: Maximum value for each factor (inclusive).
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset with columns: prompt (list[dict]), answer (str).
    """
    rng = random.Random(seed)
    prompts = []
    answers = []

    for _ in range(num_samples):
        factors = [rng.randint(2, max_factor) for _ in range(num_factors)]
        product = 1
        for f in factors:
            product *= f

        expression = " × ".join(str(f) for f in factors)
        prompt = [
            {
                "role": "user",
                "content": f"What is {expression}? Reply with just the number.",
            }
        ]

        prompts.append(prompt)
        answers.append(str(product))

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
    """0.5 bonus if completion shows the multiplication expression."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        # Match patterns like "7 * 8", "7 × 8", "7 x 8"
        import re

        if re.search(r"\d+\s*[×x\*]\s*\d+", text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="multiply",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, format_reward],
        description="Multiplication problems (e.g. 7 × 8 = 56)",
    )
)
