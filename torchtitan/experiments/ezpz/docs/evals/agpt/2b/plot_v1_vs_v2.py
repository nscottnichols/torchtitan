#!/usr/bin/env python3
"""Overlay v1 (bf16-master, broken) vs v2 (fp32-master, current) 2B
eval scores on a single figure per task.

Mirrors docs/evals/agpt/20b/plot_v1_vs_v2.py — see that file for the
design rationale (v1 scores hardcoded from the eval table; v2 scores
loaded fresh from outputs/evals/agpt-2b-v2/).

Usage:
    python3 plot_v1_vs_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required — silent fallback gave us months of charts with
# the wrong style + opaque white background. If it fails to load, surface
# loudly so we install the missing dep instead of shipping bad plots.
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])
plt.rcParams["font.family"] = "monospace"

REPO_ROOT = Path(__file__).resolve().parents[7]
# Per-trajectory v2 eval roots; gbs differs per node count, so they
# can't be conflated.
V2_TRAJECTORIES = {
    # node_count -> (results dir, GBS)
    256: (REPO_ROOT / "outputs" / "evals" / "agpt-2b-v2-256n", 6_144),   # LBS=2 × 256N × 12 GPUs
    512: (REPO_ROOT / "outputs" / "evals" / "agpt-2b-v2-512n", 12_288),  # LBS=2 × 512N × 12 GPUs
}
# 2B-MDS reference baseline (pre-torchtitan SophiaG, 140K steps / 7.77T tokens).
# Three stage dirs all symlink to the same physical ckpt dir — same step
# can appear up to 3 times; we average across replicates.
MDS_RESULTS_BASE = REPO_ROOT / "outputs" / "evals" / "agpt-2b-mds"
MDS_TOKENS_PER_STEP = 7_770e9 / 140_000  # ~55.5M tokens/step
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# v1 (bf16-tainted) scores from the eval README.
# Steps × GBS=3072 (LBS=1) × seq=8192. Scores are reported as percentages
# in the README (25.35 = 0.2535).
V1_RESULTS = {
    1000:  {"hellaswag": 0.2535, "arc_easy": 0.2744, "arc_challenge": 0.2338, "winogrande": 0.4783},
    2000:  {"hellaswag": 0.2526, "arc_easy": 0.2719, "arc_challenge": 0.2423, "winogrande": 0.4957},
    3000:  {"hellaswag": 0.2525, "arc_easy": 0.2689, "arc_challenge": 0.2355, "winogrande": 0.5107},
    4000:  {"hellaswag": 0.2512, "arc_easy": 0.2698, "arc_challenge": 0.2406, "winogrande": 0.4846},
    5000:  {"hellaswag": 0.2509, "arc_easy": 0.2698, "arc_challenge": 0.2406, "winogrande": 0.4901},
    6000:  {"hellaswag": 0.2510, "arc_easy": 0.2748, "arc_challenge": 0.2423, "winogrande": 0.4854},
    7000:  {"hellaswag": 0.2539, "arc_easy": 0.2719, "arc_challenge": 0.2355, "winogrande": 0.4988},
    8000:  {"hellaswag": 0.2535, "arc_easy": 0.2698, "arc_challenge": 0.2449, "winogrande": 0.5107},
    9000:  {"hellaswag": 0.2517, "arc_easy": 0.2723, "arc_challenge": 0.2406, "winogrande": 0.4767},
    10000: {"hellaswag": 0.2507, "arc_easy": 0.2782, "arc_challenge": 0.2381, "winogrande": 0.4886},
    11000: {"hellaswag": 0.2536, "arc_easy": 0.2668, "arc_challenge": 0.2415, "winogrande": 0.4799},
    12000: {"hellaswag": 0.2504, "arc_easy": 0.2731, "arc_challenge": 0.2406, "winogrande": 0.4909},
    13000: {"hellaswag": 0.2511, "arc_easy": 0.2731, "arc_challenge": 0.2415, "winogrande": 0.4901},
    14000: {"hellaswag": 0.2500, "arc_easy": 0.2748, "arc_challenge": 0.2457, "winogrande": 0.4988},
    15000: {"hellaswag": 0.2517, "arc_easy": 0.2778, "arc_challenge": 0.2415, "winogrande": 0.5099},
    16000: {"hellaswag": 0.2527, "arc_easy": 0.2744, "arc_challenge": 0.2415, "winogrande": 0.4996},
    17000: {"hellaswag": 0.2516, "arc_easy": 0.2673, "arc_challenge": 0.2560, "winogrande": 0.4980},
    18000: {"hellaswag": 0.2522, "arc_easy": 0.2727, "arc_challenge": 0.2500, "winogrande": 0.4949},
}
V1_GBS = 3072
V1_SEQ = 8192

V2_SEQ = 8192

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande"]
TASK_TITLES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "Winogrande",
}
RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.50,
}


def _acc(metrics: dict) -> float | None:
    for k in ("acc_norm,none", "acc,none"):
        if k in metrics:
            return float(metrics[k])
    return None


def load_v2_trajectory(results_base: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not results_base.exists():
        return out
    for step_dir in sorted(results_base.glob("step-*")):
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
        results = payload.get("results", payload)
        for task in TASKS:
            if task in results and (acc := _acc(results[task])) is not None:
                scores[task] = acc
        if scores:
            out[step] = scores
    return out


def load_mds() -> dict[int, dict[str, float]]:
    """MDS layout: <stage>/step-<N>/results/results.json with 3 stages all
    symlinked to the same physical ckpt dir for SophiaG. Average across
    replicates (XPU lm-eval is not bitwise-deterministic)."""
    if not MDS_RESULTS_BASE.exists():
        return {}
    by_step: dict[int, dict[str, list[float]]] = {}
    for p in sorted(MDS_RESULTS_BASE.glob("*/step-*/results/results.json")):
        try:
            step = int(p.parent.parent.name.split("-")[1])
        except ValueError:
            continue
        with p.open() as f:
            payload = json.load(f)
        results = payload.get("results", payload)
        for task in TASKS:
            if task in results and (acc := _acc(results[task])) is not None:
                by_step.setdefault(step, {}).setdefault(task, []).append(acc)
    return {
        step: {task: sum(vs) / len(vs) for task, vs in d.items()}
        for step, d in sorted(by_step.items())
    }


def _tokens_for(step: int, gbs: int, seq: int) -> float:
    return step * gbs * seq / 1e9


# Per-trajectory plot styles (one entry per V2_TRAJECTORIES key).
V2_STYLE = {
    256: {"color": "#1e88e5", "marker": "s", "label_prefix": "v2 256N"},
    512: {"color": "#0d47a1", "marker": "D", "label_prefix": "v2 512N"},
}


def plot_per_task(
    v1: dict[int, dict[str, float]],
    v2_by_nodes: dict[int, dict[int, dict[str, float]]],
    v2_gbs: dict[int, int],
    mds: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "AuroraGPT 2B  —  v1 (bf16-master, broken) vs v2 (fp32-master, current) vs MDS (SophiaG reference)",
        fontsize=12,
        fontweight="bold",
    )
    for ax, task in zip(axes.flat, TASKS):
        v1_steps = sorted(v1)
        v1_tokens = [_tokens_for(s, V1_GBS, V1_SEQ) for s in v1_steps]
        v1_y = [v1[s][task] for s in v1_steps]

        ax.axhline(
            RANDOM_BASELINE[task], color="#888", lw=1, ls=":", label="random",
        )
        ax.plot(
            v1_tokens, v1_y, marker="o", ms=4, lw=1.4,
            color="#94a3b8", alpha=0.85, label=f"v1 256N (n={len(v1_steps)})",
        )

        # MDS reference (pre-torchtitan SophiaG, 7.77T tokens).
        mds_steps = sorted(s for s in mds if task in mds[s])
        if mds_steps:
            mds_tokens = [s * MDS_TOKENS_PER_STEP / 1e9 for s in mds_steps]
            mds_y = [mds[s][task] for s in mds_steps]
            ax.plot(
                mds_tokens, mds_y, marker="x", ms=5, lw=1.4,
                color="#7B1FA2", alpha=0.8, linestyle="--",
                label=f"MDS SophiaG ref (n={len(mds_steps)})",
            )

        all_y_max = max(v1_y) if v1_y else 0.55
        if mds_steps:
            all_y_max = max(all_y_max, max(mds[s][task] for s in mds_steps))
        for nodes, v2_traj in v2_by_nodes.items():
            if not v2_traj:
                continue
            style = V2_STYLE.get(nodes, {"color": "#666", "marker": "x", "label_prefix": f"v2 {nodes}N"})
            gbs = v2_gbs[nodes]
            v2_steps = sorted(v2_traj)
            v2_tokens = [_tokens_for(s, gbs, V2_SEQ) for s in v2_steps]
            v2_y = [v2_traj[s].get(task) for s in v2_steps]
            v2_pts = [(t, y) for t, y in zip(v2_tokens, v2_y) if y is not None]
            if not v2_pts:
                continue
            xs, ys = zip(*v2_pts)
            ax.plot(
                xs, ys,
                marker=style["marker"], ms=6, lw=1.8,
                color=style["color"], alpha=1.0,
                label=f"{style['label_prefix']} (n={len(xs)})",
            )
            all_y_max = max(all_y_max, max(ys))

        ax.set_xlabel("Tokens consumed (B)")
        ax.set_ylabel("Accuracy")
        ax.set_title(TASK_TITLES[task])
        ax.set_ylim(0.20, max(0.55, all_y_max * 1.10))
        ax.grid(alpha=0.25)

    # Single shared legend above the grid — per-axis legends were landing
    # directly over the climbing v2 trajectories.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 4),
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_table(
    v1: dict[int, dict[str, float]],
    v2_by_nodes: dict[int, dict[int, dict[str, float]]],
    v2_gbs: dict[int, int],
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
    for nodes in sorted(v2_by_nodes):
        v2 = v2_by_nodes[nodes]
        gbs = v2_gbs[nodes]
        for step in sorted(v2):
            s = v2[step]
            tok = _tokens_for(step, gbs, V2_SEQ)
            cells = [f"{s[t]:.4f}" if t in s else "—" for t in TASKS]
            # Use {tok:.1f} (no width) so we don't get leading-space-in-bold
            # rendering like `** 10.1**`.
            print(
                f"| **v2 {nodes}N** | **{step:,}** | **{tok:.1f}** | "
                + " | ".join(f"**{c}**" for c in cells)
                + " |"
            )


def main() -> None:
    v2_by_nodes = {}
    v2_gbs = {}
    for nodes, (path, gbs) in V2_TRAJECTORIES.items():
        traj = load_v2_trajectory(path)
        v2_by_nodes[nodes] = traj
        v2_gbs[nodes] = gbs
        print(f"loaded v2 {nodes}N: {len(traj)} steps from {path}")
    mds = load_mds()
    print(f"loaded MDS reference: {len(mds)} steps from {MDS_RESULTS_BASE}")
    print(f"loaded v1: {len(V1_RESULTS)} steps")
    plot_per_task(V1_RESULTS, v2_by_nodes, v2_gbs, mds, FIG_DIR / "v1_vs_v2.svg")
    print_table(V1_RESULTS, v2_by_nodes, v2_gbs)


if __name__ == "__main__":
    main()
