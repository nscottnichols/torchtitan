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
    """Resolve an SFT dataset by name OR by a CLI mix-spec string.

    Two accepted shapes:
      - A registered dataset name (e.g. 'gsm8k', 'tulu_math_uc_mix') —
        looked up in SFT_REGISTRY.
      - A mix spec like 'tulu-3-sft-mixture:0.5,gsm8k:0.5' — parsed
        and synthesized into an ad-hoc SFTDataset that interleaves
        the components. Weights are renormalized to sum to 1.

    The mix-spec form lets you A/B different ratios from the launch
    command without touching code.
    """
    if _is_mix_spec(name):
        return SFTDataset(
            name=name,
            build=lambda: _build_ad_hoc_mix(name),
            description=f"Ad-hoc mix from CLI spec: {name}",
        )
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


# ---------------------------------------------------------------------------
# Helpers for the multi-turn / chat-format datasets below
# ---------------------------------------------------------------------------


def _messages_to_prompt_completion(messages: list[dict]) -> dict:
    """Split a chat-format `messages` list into SFTTrainer's
    {prompt, completion} shape.

    Convention used by tulu-3-sft-mixture, ultrachat_200k, and most
    HF chat datasets: `messages` is an ordered list of
    `{role, content}` dicts ending with an assistant turn that's the
    target for SFT loss. We put everything except the last assistant
    turn in `prompt`, and the last assistant turn in `completion`.

    For datasets where the last turn isn't `assistant`, fall back to
    "everything is the prompt, completion is empty" (which `interleave`
    will tolerate; SFTTrainer skips empty completions).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return {"prompt": messages, "completion": []}
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


# ---------------------------------------------------------------------------
# tulu-3-sft-mixture — allenai's curated SFT mix, ~940k examples
# ---------------------------------------------------------------------------


def _build_tulu3_sft_mixture() -> Dataset:
    """Load allenai/tulu-3-sft-mixture (~939k chat-format examples).

    The canonical recipe behind Tulu-3 — blends math + reasoning + IF +
    code + safety at validated ratios. Single best "drop-in" SFT
    dataset for a small instruct model in 2026. Each row has
    {id, messages, source}; we split messages into prompt/completion.
    """
    from datasets import load_dataset

    raw = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    def _format(ex):
        return _messages_to_prompt_completion(ex["messages"])

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="tulu-3-sft-mixture",
        build=_build_tulu3_sft_mixture,
        description=(
            "AllenAI's curated SFT mix from Tulu-3 (~939k examples). "
            "Math + reasoning + IF + code + safety at validated ratios. "
            "Single best drop-in choice for general SFT in 2026."
        ),
    )
)


# ---------------------------------------------------------------------------
# OpenMathInstruct-2 — NVIDIA's massive augmented math dataset
# ---------------------------------------------------------------------------


def _build_openmath_instruct2() -> Dataset:
    """Load nvidia/OpenMathInstruct-2 (~14M math instruction examples).

    NVIDIA's superset of GSM8K + MATH augmentations. Each row has
    {problem, generated_solution, expected_answer, problem_source}.
    Use when you want to push math capability hard; 14M is a lot of
    data — consider mixing with broader data unless you specifically
    want math depth.
    """
    from datasets import load_dataset

    raw = load_dataset("nvidia/OpenMathInstruct-2", split="train")

    def _format(ex):
        return {
            "prompt": [{"role": "user", "content": ex["problem"]}],
            "completion": [
                {"role": "assistant", "content": ex["generated_solution"]}
            ],
        }

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="OpenMathInstruct-2",
        build=_build_openmath_instruct2,
        description=(
            "NVIDIA OpenMathInstruct-2 (~14M math examples, superset of "
            "GSM8K + MATH augmentations). Use for math depth; mix with "
            "broader datasets unless you specifically want math-only."
        ),
    )
)


# ---------------------------------------------------------------------------
# ultrachat-200k — high-quality GPT-distilled multi-turn conversations
# ---------------------------------------------------------------------------


def _build_ultrachat_200k() -> Dataset:
    """Load HuggingFaceH4/ultrachat_200k (~208k multi-turn chats).

    Note: this dataset's train split is named 'train_sft' (not
    'train'). Common gotcha. Each row has {prompt, prompt_id,
    messages} where messages includes both the user prompt and
    assistant response(s); we use messages directly.
    """
    from datasets import load_dataset

    raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def _format(ex):
        return _messages_to_prompt_completion(ex["messages"])

    return raw.map(_format, remove_columns=raw.column_names)


register_sft_dataset(
    SFTDataset(
        name="ultrachat-200k",
        build=_build_ultrachat_200k,
        description=(
            "HuggingFaceH4/ultrachat_200k (~208k high-quality GPT-distilled "
            "multi-turn conversations). Good IF/conversational complement "
            "to math-heavy SFT data."
        ),
    )
)


# ---------------------------------------------------------------------------
# tulu_math_uc_mix — canonical "blessed" 3-way mix for AuroraGPT-2B SFT
# ---------------------------------------------------------------------------


def _build_tulu_math_uc_mix(
    weights=(0.65, 0.15, 0.20),
    seed: int = 42,
) -> Dataset:
    """Interleave tulu-3-sft-mixture + OpenMathInstruct-2 + ultrachat-200k.

    Default 0.65 / 0.15 / 0.20 ratio comes from a 0.55 / 0.15 / 0.20
    plan with the dropped 0.10 olmo-mix replay absorbed into tulu
    (tulu-3 already contains some pretrain-flavored data like FLAN and
    OpenAssistant, so this partially approximates the replay effect).

    Uses 'all_exhausted' stopping so smaller datasets cycle until the
    largest is fully consumed — OpenMathInstruct-2 is ~14M, so the
    effective size is dominated by it at ~93M after scaling.
    """
    from datasets import interleave_datasets

    if len(weights) != 3:
        raise ValueError(
            f"tulu_math_uc_mix weights must be (w_tulu, w_math, w_uc); "
            f"got {weights!r}"
        )
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0; got {sum(weights)}")

    return interleave_datasets(
        [
            _build_tulu3_sft_mixture(),
            _build_openmath_instruct2(),
            _build_ultrachat_200k(),
        ],
        probabilities=list(weights),
        seed=seed,
        stopping_strategy="all_exhausted",
    )


register_sft_dataset(
    SFTDataset(
        name="tulu_math_uc_mix",
        build=_build_tulu_math_uc_mix,
        description=(
            # %% escapes for argparse — see note above on math_alpaca_mix.
            "Canonical SFT recipe: 65%% tulu-3-sft-mixture + "
            "15%% OpenMathInstruct-2 + 20%% ultrachat-200k. Tulu's "
            "validated broad mix + extra math depth + multi-turn IF."
        ),
    )
)


# ---------------------------------------------------------------------------
# Generic mix-spec parser — `--sft_dataset 'a:0.5,b:0.3,c:0.2'`
# ---------------------------------------------------------------------------


def _is_mix_spec(name: str) -> bool:
    """Detect a CLI mix-spec string like 'tulu-3-sft-mixture:0.5,gsm8k:0.5'.

    A mix spec contains at least one ':' and at least one ',' (or a
    single entry like 'gsm8k:1.0' is also legal). A bare dataset name
    won't match because it has no ':'.
    """
    return ":" in name


def _parse_mix_spec(spec: str) -> tuple[list[str], list[float]]:
    """Parse 'a:0.5,b:0.3,c:0.2' into (['a','b','c'], [0.5,0.3,0.2])."""
    pairs = [p.strip() for p in spec.split(",") if p.strip()]
    names: list[str] = []
    weights: list[float] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(
                f"Mix-spec entry {pair!r} missing ':<weight>'. "
                f"Expected 'name:0.5' format; got {spec!r}"
            )
        n, w = pair.rsplit(":", 1)
        names.append(n.strip())
        try:
            wval = float(w.strip())
        except ValueError as e:
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) isn't a valid float"
            ) from e
        # float() accepts 'NaN' and 'inf'; reject both — they'd silently
        # poison interleave_datasets' probabilities.
        if not (wval == wval) or wval == float("inf") or wval == float("-inf"):
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) isn't finite"
            )
        if wval < 0:
            raise ValueError(
                f"Mix-spec weight {w!r} (in {pair!r}) is negative"
            )
        weights.append(wval)
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Mix-spec weights sum to {total}; need > 0")
    # Renormalize so the user doesn't have to make them sum to 1
    weights = [w / total for w in weights]
    return names, weights


def _build_ad_hoc_mix(spec: str, seed: int = 42) -> Dataset:
    """Build a one-off interleaved mix from a CLI spec string.

    Each component name must already be in SFT_REGISTRY. Components
    are interleaved with their (renormalized) weights via
    `interleave_datasets(stopping_strategy='all_exhausted')`, same
    as the canonical hardcoded mixes.
    """
    from datasets import interleave_datasets

    names, weights = _parse_mix_spec(spec)
    for n in names:
        if n not in SFT_REGISTRY:
            available = ", ".join(sorted(SFT_REGISTRY))
            raise ValueError(
                f"Mix-spec references unknown dataset {n!r}. "
                f"Available: {available}"
            )
    components = [SFT_REGISTRY[n].build() for n in names]
    return interleave_datasets(
        components,
        probabilities=weights,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
