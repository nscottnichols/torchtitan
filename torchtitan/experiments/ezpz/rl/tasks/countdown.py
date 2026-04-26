# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Countdown task for GRPO training.
#
# Given a set of numbers and a target, find an arithmetic expression that
# equals the target using +, -, *, /. Inspired by the TV game show and
# used in DeepSeek-R1 and other GRPO papers as a reasoning benchmark.
#
# "Using 3, 7, 2, make 17" → "3 * 7 - 2 * 2 = 17" or "7 * 2 + 3 = 17"

import random
import re
from itertools import permutations

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import get_completion_text


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------

_OPS = ["+", "-", "*"]  # skip division to keep targets integer-valued


def _evaluate(nums: list[int], ops: list[str]) -> int | None:
    """Evaluate a left-to-right expression (no precedence, just sequential)."""
    result = nums[0]
    for op, n in zip(ops, nums[1:]):
        if op == "+":
            result += n
        elif op == "-":
            result -= n
        elif op == "*":
            result *= n
    return result


def _find_target(
    numbers: list[int],
    rng: random.Random,
) -> int | None:
    """Find an achievable target by trying random operator combos.

    Returns a target reachable with some permutation of the numbers and
    operators, or None if nothing interesting found.
    """
    # Try random combos, pick one that gives a "nice" positive target
    ops_needed = len(numbers) - 1
    candidates = set()
    for perm in permutations(numbers):
        for _ in range(20):
            ops = [rng.choice(_OPS) for _ in range(ops_needed)]
            val = _evaluate(list(perm), ops)
            if val is not None and 1 <= val <= 999:
                candidates.add(val)
    if not candidates:
        return None
    return rng.choice(list(candidates))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(
    num_samples: int = 1000,
    min_numbers: int = 3,
    max_numbers: int = 5,
    max_value: int = 25,
    seed: int = 42,
) -> Dataset:
    """Generate countdown problems with achievable targets.

    Args:
        num_samples: Number of samples to generate.
        min_numbers: Minimum count of given numbers.
        max_numbers: Maximum count of given numbers.
        max_value: Maximum value for each given number.
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset with columns: prompt (list[dict]),
        answer (str), numbers (str).
    """
    rng = random.Random(seed)
    prompts = []
    answers = []
    all_numbers = []

    attempts = 0
    while len(prompts) < num_samples and attempts < num_samples * 10:
        attempts += 1
        n = rng.randint(min_numbers, max_numbers)
        numbers = [rng.randint(1, max_value) for _ in range(n)]
        target = _find_target(numbers, rng)
        if target is None:
            continue

        nums_str = ", ".join(str(x) for x in numbers)
        prompt = [
            {
                "role": "user",
                "content": (
                    f"Using the numbers {nums_str}, create an arithmetic "
                    f"expression using +, -, * that equals {target}. "
                    f"Show your expression and result."
                ),
            }
        ]

        prompts.append(prompt)
        answers.append(str(target))
        all_numbers.append(nums_str)

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "numbers": all_numbers,
    })


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _extract_result(text: str) -> int | None:
    """Try to extract the final numeric result from a completion."""
    # Look for "= <number>" pattern
    matches = re.findall(r"=\s*(-?\d+)", text)
    if matches:
        return int(matches[-1])
    # Fall back to last number
    numbers = re.findall(r"\b(-?\d+)\b", text)
    if numbers:
        return int(numbers[-1])
    return None


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if the expression evaluates to the target, 0.0 otherwise."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        result = _extract_result(text)
        if result is not None and result == int(expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.5 bonus if the completion contains a valid arithmetic expression."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        # Check for expression pattern: number op number (op number)*
        if re.search(r"\d+\s*[+\-\*]\s*\d+", text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def uses_given_numbers_reward(completions, numbers, **kwargs) -> list[float]:
    """0.5 bonus if the completion only uses numbers from the given set."""
    rewards = []
    for completion, nums_str in zip(completions, numbers):
        text = get_completion_text(completion)
        given = [int(x.strip()) for x in nums_str.split(",")]
        # Find all numbers used in arithmetic expressions
        # (ignore the target number after =)
        expr_match = re.search(r"([\d\s+\-\*]+)=", text)
        if expr_match:
            used = [int(x) for x in re.findall(r"\d+", expr_match.group(1))]
            given_sorted = sorted(given)
            used_sorted = sorted(used)
            # Check used numbers are a subset of given numbers
            if all(
                used_sorted.count(n) <= given_sorted.count(n)
                for n in set(used_sorted)
            ):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="countdown",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, format_reward, uses_given_numbers_reward],
        description="Reach a target number using arithmetic on given numbers",
    )
)
