#!/usr/bin/env python3
"""Plot production training metrics pulled from W&B.

W&B is the single source of truth for production training history,
since PBS log files only exist for jobs that have already exited and
the active runs span multiple resumes. Generates three figures per
model size:

  - ``production_<model>_<num_nodes>n.png``: loss / tps-per-gpu / mfu
    vs step (overwrites the same filename produced by the legacy
    ``plot_production.py``).
  - ``training_diagnostics_<model>_<num_nodes>n.png``: grad_norm, lr,
    and max_loss vs step.
  - ``tokens_vs_time_<model>_<num_nodes>n.png``: cumulative
    ``n_tokens_seen`` vs wall-clock datetime.

Run from the repo root:

    python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import ambivalent

    plt.style.use(ambivalent.STYLES["ambivalent"])
except ImportError as e:
    import warnings

    warnings.warn(
        f"ambivalent style unavailable, using matplotlib defaults: {e}",
        stacklevel=2,
    )

import wandb  # noqa: E402

PROJECT = "aurora_gpt/torchtitan.ezpz.train"

# Production runs identified by step ranges (cross-checked with PBS logs).
# Listed oldest first so concatenation matches resume order.
PRODUCTION_RUNS: dict[str, dict] = {
    "2b": {
        "run_ids": [
            "v5ytgu0o",
            "pjanidnw",
            "4u9w23p9",
            "tahlsmy9",
            "iy1xbv0t",
            "11jzfnno",
            "hqwaw075",
            "6ictshbs",
            "wviyqysc",
        ],
        "num_nodes": 256,
    },
    "20b": {
        "run_ids": [
            "q9oq5huj",
            "pnkaurba",
            "lrlv3xsc",
            "pigwfqkg",
            "lvyzlocg",
            "e2anhgt2",
            "he01jr7f",
            "t0ja3dl4",
        ],
        "num_nodes": 256,
    },
}

MODEL_COLORS = {
    "2b": "#1E88E5",
    "20b": "#D32F2F",
    "80b": "#388E3C",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DOCS_BASE = REPO_ROOT / "torchtitan" / "experiments" / "ezpz" / "docs"

METRIC_KEYS = (
    "_step",
    "_timestamp",
    "grad_norm",
    "lr",
    "loss_metrics/global_avg_loss",
    "loss_metrics/global_max_loss",
    "n_tokens_seen",
    "throughput(tps)",
    "mfu(%)",
)


def fetch_run(api: wandb.Api, run_id: str) -> dict[str, np.ndarray]:
    """Pull the full history for the given metric keys from one wandb run.

    Uses ``scan_history`` so we get every logged step, not a downsample.
    """
    run = api.run(f"{PROJECT}/{run_id}")
    columns: dict[str, list] = {k: [] for k in METRIC_KEYS}
    for row in run.scan_history(keys=list(METRIC_KEYS)):
        for k in METRIC_KEYS:
            columns[k].append(row.get(k))
    return {k: np.array(v) for k, v in columns.items()}


def concat_runs(api: wandb.Api, run_ids: list[str]) -> dict[str, np.ndarray]:
    """Fetch and concatenate multiple wandb runs by ascending _step."""
    parts = []
    for rid in run_ids:
        data = fetch_run(api, rid)
        if len(data["_step"]) == 0:
            print(f"  {rid}: no rows, skipping")
            continue
        print(f"  {rid}: {len(data['_step'])} rows, steps [{data['_step'][0]}, {data['_step'][-1]}]")
        parts.append(data)

    # For each step, keep the record from the latest run (resume semantics).
    by_step: dict[int, dict] = {}
    for data in parts:
        for i, s in enumerate(data["_step"]):
            if s is None:
                continue
            by_step[int(s)] = {k: data[k][i] for k in METRIC_KEYS}

    sorted_steps = sorted(by_step)
    out = {k: np.array([by_step[s][k] for s in sorted_steps]) for k in METRIC_KEYS}
    return out


def smooth(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Centered moving average; falls back to identity for short series."""
    values = np.asarray(values, dtype=float)
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def plot_dashboard(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: loss, tps/gpu, mfu vs step.

    Replaces the log-parsing version in ``plot_production.py``: pulls
    from W&B so it reflects in-progress runs whose PBS log files
    haven't been written yet.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    num_gpus = num_nodes * 12
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    # `throughput(tps)` is already logged per-rank (per-GPU) by torchtitan,
    # not aggregated — values for the 2B run sit around 1k-3k, matching
    # the per-GPU numbers in the PBS-log dashboards. Don't divide.
    tps_per_gpu = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)

    valid = ~np.isnan(steps)
    steps = steps[valid]
    loss = loss[valid]
    tps_per_gpu = tps_per_gpu[valid]
    mfu = mfu[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Production Training  |  "
        f"{num_nodes} nodes ({num_gpus} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, loss, color=color, alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(loss), color=color, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5)
    ax.plot(
        steps,
        smooth(tps_per_gpu),
        color="#43A047",
        linewidth=1.8,
        label="TPS/GPU (smoothed)",
    )
    ax.set_ylabel("Tokens/sec/GPU")
    ax.set_title("Throughput per GPU")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(mfu), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_diagnostics(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: grad_norm, lr, max_loss vs step."""
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    steps = data["_step"].astype(float)
    grad_norm = data["grad_norm"].astype(float)
    lr = data["lr"].astype(float)
    max_loss = data["loss_metrics/global_max_loss"].astype(float)

    # Filter out NaN-only rows that some logging configs emit
    valid = ~np.isnan(steps)
    steps = steps[valid]
    grad_norm = grad_norm[valid]
    lr = lr[valid]
    max_loss = max_loss[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Diagnostics  |  "
        f"{num_nodes} nodes ({num_nodes * 12} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, grad_norm, color=color, alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(grad_norm), color=color, linewidth=1.8, label="grad_norm (smoothed)")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient Norm")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lr, color="#FF9800", linewidth=1.8, label="lr")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, max_loss, color="#7B1FA2", alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(max_loss), color="#7B1FA2", linewidth=1.8, label="max_loss (smoothed)")
    ax.set_ylabel("global_max_loss")
    ax.set_xlabel("Training Step")
    ax.set_title("Per-Step Max Loss (across DP ranks)")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_tokens_vs_time(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """Cumulative tokens vs wall-clock datetime.

    Sorts by timestamp (not step) so resumed runs read as a single
    monotonic curve. When restarts re-tokenize early steps after a
    checkpoint, those rows are filtered out via a running max so the
    cumulative curve never goes backwards.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    ts = data["_timestamp"].astype(float)
    tokens = data["n_tokens_seen"].astype(float)

    valid = ~np.isnan(ts) & ~np.isnan(tokens)
    ts = ts[valid]
    tokens = tokens[valid]

    # Sort by wall-clock time so the curve reflects actual execution order.
    order = np.argsort(ts)
    ts = ts[order]
    tokens = tokens[order]

    # Restarts that wandb logged before the resume's `n_tokens_seen` caught
    # up to the previous best produce dips. Take the running max so the
    # curve stays monotonic non-decreasing.
    tokens = np.maximum.accumulate(tokens)

    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, tokens / 1e9, color=color, linewidth=1.8)

    final_tokens_b = tokens[-1] / 1e9
    target_b = 4670  # 4.67T tokens
    pct = 100 * final_tokens_b / target_b

    ax.set_xlabel("Date")
    ax.set_ylabel("Tokens consumed (billions)")
    ax.set_title(
        f"AuroraGPT {model_name.upper()} — Tokens vs Wall Clock  |  "
        f"{num_nodes} nodes  |  "
        f"{final_tokens_b:,.1f}B tokens ({pct:.1f}% of 4.67T target)",
        fontsize=14,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="2b, 20b, or omit for all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: docs/production/agpt/<model>/figures/)",
    )
    args = parser.parse_args()

    api = wandb.Api()

    models = [args.model] if args.model else list(PRODUCTION_RUNS)
    for model_name in models:
        cfg = PRODUCTION_RUNS[model_name]
        out_dir = args.output_dir or (
            DOCS_BASE / "production" / "agpt" / model_name / "figures"
        )

        print(f"\n=== Pulling {model_name.upper()} ({len(cfg['run_ids'])} runs) ===")
        data = concat_runs(api, cfg["run_ids"])
        print(f"  Concatenated: {len(data['_step'])} unique steps")

        plot_dashboard(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"production_{model_name}_{cfg['num_nodes']}n.png",
        )
        plot_diagnostics(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"training_diagnostics_{model_name}_{cfg['num_nodes']}n.png",
        )
        plot_tokens_vs_time(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"tokens_vs_time_{model_name}_{cfg['num_nodes']}n.png",
        )


if __name__ == "__main__":
    main()
