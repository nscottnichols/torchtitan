#!/usr/bin/env python3
"""Aggregate lm-eval results across training steps and generate plots.

Usage:
    python aggregate_evals.py --model 2b
    python aggregate_evals.py --model 20b --output-dir docs/production/agpt/2b/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(evals_dir: Path) -> list[dict]:
    """Load all lm-eval results for a model, sorted by step."""
    results = []
    if not evals_dir.exists():
        return results

    for step_dir in sorted(evals_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step-"):
            continue

        step = int(step_dir.name.split("-")[1])
        results_dir = step_dir / "results"

        # lm-eval saves results in a subdirectory with timestamp
        result_files = list(results_dir.glob("**/results.json"))
        if not result_files:
            continue

        with open(result_files[0]) as f:
            data = json.load(f)

        entry = {"step": step, "tasks": {}}
        for task_name, task_data in data.get("results", {}).items():
            # Extract accuracy metric (acc or acc_norm)
            acc = task_data.get("acc,none") or task_data.get("acc_norm,none")
            acc_stderr = task_data.get("acc_stderr,none") or task_data.get(
                "acc_norm_stderr,none"
            )
            if acc is not None:
                entry["tasks"][task_name] = {
                    "acc": acc,
                    "acc_stderr": acc_stderr,
                }

        if entry["tasks"]:
            results.append(entry)

    return sorted(results, key=lambda x: x["step"])


def print_table(results: list[dict], model: str):
    """Print markdown table of results."""
    if not results:
        print("No results found.")
        return

    # Collect all task names
    all_tasks = set()
    for r in results:
        all_tasks.update(r["tasks"].keys())
    tasks = sorted(all_tasks)

    # Header
    header = f"| Step | " + " | ".join(tasks) + " |"
    separator = "|------|" + "|".join(["------"] * len(tasks)) + "|"
    print(f"\n## Evaluation Results — agpt_{model}\n")
    print(header)
    print(separator)

    # Rows
    for r in results:
        row = f"| {r['step']:>5} |"
        for task in tasks:
            if task in r["tasks"]:
                acc = r["tasks"][task]["acc"]
                row += f" {acc:.4f} |"
            else:
                row += " — |"
        print(row)


def plot_results(results: list[dict], model: str, output_path: Path):
    """Generate benchmark accuracy vs training step plot."""
    if not results:
        return

    all_tasks = set()
    for r in results:
        all_tasks.update(r["tasks"].keys())
    tasks = sorted(all_tasks)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = plt.cm.Set2(range(len(tasks)))

    for task, color in zip(tasks, colors):
        steps = []
        accs = []
        errs = []
        for r in results:
            if task in r["tasks"]:
                steps.append(r["step"])
                accs.append(r["tasks"][task]["acc"])
                errs.append(r["tasks"][task].get("acc_stderr") or 0)

        if steps:
            ax.errorbar(
                steps,
                accs,
                yerr=errs,
                marker="o",
                markersize=5,
                label=task,
                color=color,
                linewidth=2,
                capsize=3,
            )

    ax.set_xlabel("Training Step", color="white", fontsize=12)
    ax.set_ylabel("Accuracy", color="white", fontsize=12)
    ax.set_title(
        f"agpt_{model} — Benchmark Accuracy vs Training Step",
        color="white",
        fontsize=14,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate lm-eval results")
    parser.add_argument(
        "--model", type=str, required=True, choices=["2b", "20b", "80b"]
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=None,
        help="Base evals directory (default: outputs/evals/agpt-{model}/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    evals_dir = args.evals_dir or repo_root / f"outputs/evals/agpt-{args.model}"
    output_dir = (
        args.output_dir
        or repo_root
        / f"torchtitan/experiments/ezpz/docs/production/agpt/{args.model}/figures"
    )

    results = load_results(evals_dir)
    print_table(results, args.model)

    if results:
        plot_path = output_dir / f"eval_agpt_{args.model}.png"
        plot_results(results, args.model, plot_path)
    else:
        print(f"\nNo eval results found in {evals_dir}")
        print("Run convert_and_eval.sh first to generate results.")


if __name__ == "__main__":
    main()
