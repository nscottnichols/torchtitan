# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Dataset builder for sum-of-digits GRPO training.

import random

from datasets import Dataset


def build_sum_digits_dataset(
    num_samples: int = 1000,
    min_addends: int = 2,
    max_addends: int = 5,
    max_digit: int = 9,
    seed: int = 42,
) -> Dataset:
    """Generate a dataset of sum-of-digits prompts with ground truth answers.

    Each sample is a prompt like "What is 3 + 7 + 2?" with the expected
    answer "12".

    Args:
        num_samples: Number of samples to generate.
        min_addends: Minimum number of digits to sum.
        max_addends: Maximum number of digits to sum.
        max_digit: Maximum value for each digit (inclusive).
        seed: Random seed for reproducibility.

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
