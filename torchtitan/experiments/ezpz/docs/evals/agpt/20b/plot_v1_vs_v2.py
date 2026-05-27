#!/usr/bin/env python3
"""Overlay v1 (bf16-master, broken) vs v2 (fp32-master, current) 20B
eval scores on a single figure per task.

The v1 scores are hardcoded from the eval table in this directory's
README (frozen-RMSNorm regime, all hover near random). The v2 scores
are loaded fresh from outputs/evals/agpt-20b-v2-512n/step-{N}/results/results.json
each run, so this script can be re-run as new v2 ckpts get evaluated.

If v2 even moves a few percentage points above v1's flat line, the
fp32-master fix is validated end-to-end.

Usage:
    python3 plot_v1_vs_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import ambivalent  # noqa: F401

    plt.style.use(ambivalent.STYLES["ambivalent"])
except ImportError:
    pass

plt.rcParams["font.family"] = "monospace"

REPO_ROOT = Path(__file__).resolve().parents[7]
V2_RESULTS_BASE = REPO_ROOT / "outputs" / "evals" / "agpt-20b-v2-512n"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# v1 (bf16-tainted) scores from the eval README. Steps × GBS=3072 × seq=8192.
V1_RESULTS = {
    100:   {"hellaswag": 0.2650, "arc_easy": 0.2571, "arc_challenge": 0.2560, "winogrande": 0.4957},
    500:   {"hellaswag": 0.2562, "arc_easy": 0.2712, "arc_challenge": 0.2355, "winogrande": 0.4886},
    1000:  {"hellaswag": 0.2549, "arc_easy": 0.2647, "arc_challenge": 0.2440, "winogrande": 0.5178},
    1500:  {"hellaswag": 0.2505, "arc_easy": 0.2681, "arc_challenge": 0.2398, "winogrande": 0.4807},
    2000:  {"hellaswag": 0.2480, "arc_easy": 0.2740, "arc_challenge": 0.2543, "winogrande": 0.5020},
    2500:  {"hellaswag": 0.2462, "arc_easy": 0.2736, "arc_challenge": 0.2483, "winogrande": 0.5193},
}
V1_GBS = 3072  # LBS=1
V1_SEQ = 8192

# v2 token math: GBS=12288 (LBS=2 × 512 nodes × 12 GPUs), seq=8192.
V2_GBS = 12288
V2_SEQ = 8192

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande"]
TASK_TITLES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "Winogrande",
}
RANDOM_BASELINE = {
    "hellaswag": 0.25,       # 4-way
    "arc_easy": 0.25,        # 4-way
    "arc_challenge": 0.25,   # 4-way
    "winogrande": 0.50,      # 2-way
}


def _acc(metrics: dict) -> float | None:
    """lm-eval reports `acc_norm,none` for HellaSwag/ARC; `acc,none` for Winogrande."""
    for k in ("acc_norm,none", "acc,none"):
        if k in metrics:
            return float(metrics[k])
    return None


def load_v2() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not V2_RESULTS_BASE.exists():
        print(f"no v2 results at {V2_RESULTS_BASE}")
        return out
    for step_dir in sorted(V2_RESULTS_BASE.glob("step-*")):
        results_file = step_dir / "results" / "results.json"
        if not results_file.exists():
            continue
        try:
            step = int(step_dir.name.split("-")[1])
        except ValueError:
            continue
        with results_file.open() as f:
            payload = json.load(f)
        scores: dict[str, float] = {}
        # lm-eval JSON either has {task: {metric: value, ...}} or wraps in
        # {"results": {task: ...}}. Handle both.
        results = payload.get("results", payload)
        for task in TASKS:
            if task in results and (acc := _acc(results[task])) is not None:
                scores[task] = acc
        if scores:
            out[step] = scores
    return out


def _tokens_for(step: int, gbs: int, seq: int) -> float:
    return step * gbs * seq / 1e9  # billions


def plot_per_task(
    v1: dict[int, dict[str, float]],
    v2: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "AuroraGPT 20B  —  v1 (bf16-master, broken) vs v2 (fp32-master, current)",
        fontsize=13,
        fontweight="bold",
    )
    for ax, task in zip(axes.flat, TASKS):
        v1_steps = sorted(v1)
        v1_tokens = [_tokens_for(s, V1_GBS, V1_SEQ) for s in v1_steps]
        v1_y = [v1[s][task] for s in v1_steps]

        v2_steps = sorted(v2)
        v2_tokens = [_tokens_for(s, V2_GBS, V2_SEQ) for s in v2_steps]
        v2_y = [v2[s].get(task) for s in v2_steps]

        ax.axhline(
            RANDOM_BASELINE[task], color="#888", lw=1, ls=":", label="random",
        )
        ax.plot(
            v1_tokens, v1_y, marker="o", ms=4, lw=1.4,
            color="#94a3b8", alpha=0.85, label=f"v1 256N (n={len(v1_steps)})",
        )
        if v2_y and any(v is not None for v in v2_y):
            v2_pts = [(t, y) for t, y in zip(v2_tokens, v2_y) if y is not None]
            xs, ys = zip(*v2_pts)
            ax.plot(
                xs, ys, marker="s", ms=6, lw=1.8,
                color="#dc2626", alpha=1.0, label=f"v2 512N (n={len(xs)})",
            )

        ax.set_xlabel("Tokens consumed (B)")
        ax.set_ylabel("Accuracy")
        ax.set_title(TASK_TITLES[task])
        ax.set_ylim(0.20, max(0.55, max(v1_y) * 1.15))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def print_table(
    v1: dict[int, dict[str, float]],
    v2: dict[int, dict[str, float]],
) -> None:
    print("\n## v1 vs v2 — by training step\n")
    print("| Run | Step | Tokens (B) | HellaSwag | ARC-Easy | ARC-Chall | Winogrande |")
    print("|-----|-----:|-----------:|----------:|---------:|----------:|-----------:|")
    for step in sorted(v1):
        s = v1[step]
        tok = _tokens_for(step, V1_GBS, V1_SEQ)
        print(
            f"| v1 256N | {step:,} | {tok:5.1f} | "
            f"{s['hellaswag']:.4f} | {s['arc_easy']:.4f} | "
            f"{s['arc_challenge']:.4f} | {s['winogrande']:.4f} |"
        )
    for step in sorted(v2):
        s = v2[step]
        tok = _tokens_for(step, V2_GBS, V2_SEQ)
        cells = []
        for t in TASKS:
            cells.append(f"{s[t]:.4f}" if t in s else "—")
        # Use {tok:.1f} (no width) so we don't get leading-space-in-bold
        # rendering like `** 10.1**`.
        print(
            f"| **v2 512N** | **{step:,}** | **{tok:.1f}** | "
            + " | ".join(f"**{c}**" for c in cells)
            + " |"
        )


def main() -> None:
    v2 = load_v2()
    print(f"loaded v1: {len(V1_RESULTS)} steps, v2: {len(v2)} steps")
    plot_per_task(V1_RESULTS, v2, FIG_DIR / "v1_vs_v2.svg")
    print_table(V1_RESULTS, v2)


if __name__ == "__main__":
    main()
