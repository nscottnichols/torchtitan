#!/usr/bin/env python3
"""Aggregate lm-eval results across training steps and generate plots/tables.

Reads results from `outputs/evals/agpt-{model}/step-{N}/results/results.json`.
Generates per-model plots in `docs/evals/agpt/{model}/figures/eval_{model}.png`
and prints a markdown table of accuracies.

Usage:
    python aggregate_evals.py --model 2b
    python aggregate_evals.py --model 20b
    python aggregate_evals.py --model both
    python aggregate_evals.py --model 2b --csv outputs/evals/eval_results_2b.csv
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_COLORS = {
    "hellaswag": "#e74c3c",
    "arc_easy": "#2ecc71",
    "arc_challenge": "#3498db",
    "winogrande": "#f39c12",
}

RANDOM_BASELINES = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.5,
}


def load_results(model: str, evals_dir: Path) -> dict[int, dict[str, float]]:
    """Load all eval results for a model, keyed by training step."""
    data: dict[int, dict[str, float]] = {}
    base = evals_dir / f"agpt-{model}"
    for step_dir in sorted(base.glob("step-*/results/results.json")):
        step = int(step_dir.parent.parent.name.split("-")[1])
        with open(step_dir) as f:
            d = json.load(f)
        scores: dict[str, float] = {}
        for task, m in d.items():
            if not isinstance(m, dict):
                continue
            acc = m.get("acc_norm,none") or m.get("acc,none")
            if acc is not None:
                scores[task] = acc
        if scores:
            data[step] = scores
    return data


def make_plot(data: dict, model: str, outpath: Path) -> None:
    """Generate accuracy-vs-step plot for a model."""
    if not data:
        print(f"[skip] no data for {model}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for task, color in TASK_COLORS.items():
        steps = sorted([s for s in data if task in data[s]])
        accs = [data[s][task] for s in steps]
        if steps:
            ax.plot(
                steps,
                accs,
                "o-",
                color=color,
                label=task,
                markersize=6,
                linewidth=2,
            )
            ax.axhline(
                y=RANDOM_BASELINES[task],
                color=color,
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

    ax.set_xlabel("Training Step", color="white", fontsize=12)
    ax.set_ylabel("Accuracy", color="white", fontsize=12)
    ax.set_title(
        f"agpt_{model} — Benchmark Accuracy vs Training Step ({len(data)} checkpoints)",
        color="white",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {outpath}")


def print_table(data: dict, model: str) -> None:
    """Print a markdown table of results."""
    if not data:
        print(f"\n## agpt_{model}: no results")
        return

    tasks = sorted({t for s in data for t in data[s]})
    header = "| Step | " + " | ".join(tasks) + " |"
    sep = "|------|" + "|".join(["------"] * len(tasks)) + "|"
    print(f"\n## agpt_{model} — {len(data)} checkpoints\n")
    print(header)
    print(sep)
    for step in sorted(data):
        row = f"| {step:>5} |"
        for task in tasks:
            v = data[step].get(task)
            row += f" {v:.4f} |" if v is not None else " — |"
        print(row)


def write_csv(data: dict, model: str, outpath: Path) -> None:
    """Write results to CSV for downstream analysis."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "task", "acc"])
        for step in sorted(data):
            for task, acc in sorted(data[step].items()):
                w.writerow([model, step, task, acc])
    print(f"Wrote CSV: {outpath}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description="Aggregate lm-eval results")
    parser.add_argument(
        "--model",
        choices=["2b", "20b", "both"],
        default="both",
        help="Which model to process",
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=repo_root / "outputs/evals",
        help="Base directory containing per-step eval results",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=repo_root / "torchtitan/experiments/ezpz/docs/evals/agpt",
        help="Base docs directory for plot output",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (default: skip)",
    )
    args = parser.parse_args()

    models = ["2b", "20b"] if args.model == "both" else [args.model]

    for model in models:
        data = load_results(model, args.evals_dir)
        print_table(data, model)
        plot_path = args.docs_dir / model / "figures" / f"eval_{model}.png"
        make_plot(data, model, plot_path)
        if args.csv:
            csv_path = (
                args.csv
                if args.model != "both"
                else args.csv.with_stem(f"{args.csv.stem}_{model}")
            )
            write_csv(data, model, csv_path)


if __name__ == "__main__":
    main()
