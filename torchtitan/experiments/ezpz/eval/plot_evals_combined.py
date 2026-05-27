#!/usr/bin/env python3
"""Combined-overlay eval chart across all 5 production trajectories.

One figure, 4 subplot panels (HellaSwag acc_norm, ARC-Easy acc, ARC-C acc_norm,
Winogrande acc), with 5 curves per panel — one per production trajectory —
plotted against **tokens consumed** so trajectories with different GBS can
be compared directly:

    - 2B-MDS (Megatron-DeepSpeed reference, ~7.77T tokens)
    - 2B 256N async (current production comparator)
    - 2B 512N sync (canonical 2B chain)
    - 20B 256N (per-token comparator)
    - 20B 512N sync (canonical 20B chain)

Writes to docs/evals/figures/all_production_evals.svg (single artifact
referenced from docs/evals/README.md as the landing-page chart).

Run:
    python3 -m torchtitan.experiments.ezpz.eval.plot_evals_combined
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

try:
    import ambivalent
    import matplotlib.pyplot as plt

    plt.style.use(ambivalent.STYLES["ambivalent"])
except ImportError:
    import matplotlib.pyplot as plt  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALS_DIR = REPO_ROOT / "outputs" / "evals"
OUT_PATH = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/evals/figures/all_production_evals.svg"
)

# Tokens-per-step for each trajectory (computed from GBS × SEQ_LEN where
# SEQ_LEN=8192 across the board).
TRAJECTORIES: list[dict] = [
    {
        "label": "2B-MDS (SophiaG, n256)",
        "eval_subdir": "agpt-2b-mds",
        "layout": "mds",
        "tokens_per_step": 7_770e9 / 140_000,
        "color": "#808080",
        "linestyle": ":",
        "marker": "x",
    },
    {
        "label": "2B 256N async (GBS=6144)",
        "eval_subdir": "agpt-2b-v2-256n",
        "layout": "dcp",
        "tokens_per_step": 6144 * 8192,
        "color": "#1E88E5",
        "linestyle": "-",
        "marker": "o",
    },
    {
        "label": "2B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-2b-v2-512n",
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": "#43A047",
        "linestyle": "-",
        "marker": "s",
    },
    {
        "label": "20B 256N (GBS=3072)",
        "eval_subdir": "agpt-20b-v2-256n",
        "layout": "dcp",
        "tokens_per_step": 3072 * 8192,
        "color": "#FB8C00",
        "linestyle": "--",
        "marker": "^",
    },
    {
        "label": "20B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-20b-v2-512n",
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": "#D32F2F",
        "linestyle": "-",
        "marker": "D",
    },
]

PANELS: list[tuple[str, str, str]] = [
    ("hellaswag", "acc_norm,none", "HellaSwag (acc_norm)"),
    ("arc_easy", "acc,none", "ARC-Easy (acc)"),
    ("arc_challenge", "acc_norm,none", "ARC-Challenge (acc_norm)"),
    ("winogrande", "acc,none", "Winogrande (acc)"),
]

RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.5,
}


def _read_metric(path: Path, task: str, metric: str) -> float | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    t = d.get(task)
    if not isinstance(t, dict):
        return None
    return t.get(metric)


def load_dcp(subdir: str, task: str, metric: str) -> list[tuple[int, float]]:
    base = EVALS_DIR / subdir
    out: list[tuple[int, float]] = []
    for p in sorted(base.glob("step-*/results/results.json")):
        step = int(p.parent.parent.name.split("-")[1])
        val = _read_metric(p, task, metric)
        if val is not None:
            out.append((step, val))
    return sorted(out)


def load_mds(subdir: str, task: str, metric: str) -> list[tuple[int, float]]:
    """MDS layout: 3 stage dirs containing step-{N}/results/results.json.

    The three stages all symlink to the same physical ckpt dir for the
    SophiaG sweep — same step appears up to 3 times. Average across
    replicates (XPU lm-eval isn't bit-deterministic).
    """
    base = EVALS_DIR / subdir
    by_step: dict[int, list[float]] = {}
    for p in sorted(base.glob("*/step-*/results/results.json")):
        step = int(p.parent.parent.name.split("-")[1])
        val = _read_metric(p, task, metric)
        if val is not None:
            by_step.setdefault(step, []).append(val)
    return sorted((s, sum(v) / len(v)) for s, v in by_step.items())


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, (task, metric, title) in zip(axes, PANELS):
        for traj in TRAJECTORIES:
            loader = load_mds if traj["layout"] == "mds" else load_dcp
            pts = loader(traj["eval_subdir"], task, metric)
            if not pts:
                print(f"  [{title}] no data for {traj['label']}")
                continue
            steps, accs = zip(*pts)
            tokens_b = [s * traj["tokens_per_step"] / 1e9 for s in steps]
            ax.plot(
                tokens_b,
                accs,
                marker=traj["marker"],
                color=traj["color"],
                linestyle=traj["linestyle"],
                label=traj["label"],
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            print(
                f"  [{title}] {traj['label']}: "
                f"{len(pts)} pts, tokens {tokens_b[0]:.1f}B → {tokens_b[-1]:.1f}B, "
                f"acc {accs[0]:.3f} → {accs[-1]:.3f}"
            )

        ax.axhline(
            y=RANDOM_BASELINE[task],
            color="black",
            linestyle=":",
            alpha=0.4,
            linewidth=1,
            label="random",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (B, log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "AuroraGPT v2 — Eval Benchmarks vs Training Tokens (all production trajectories)",
        y=1.06,
        fontsize=14,
    )
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
