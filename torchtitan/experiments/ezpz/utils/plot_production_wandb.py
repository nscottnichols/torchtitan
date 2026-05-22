#!/usr/bin/env python3
"""Plot production training metrics pulled from W&B.

W&B is the single source of truth for production training history,
since PBS log files only exist for jobs that have already exited and
the active runs span multiple resumes. Generates three figures per
model size:

  - ``production_<model>_<num_nodes>n.svg``: loss / tps-per-gpu / mfu
    vs step.
  - ``training_diagnostics_<model>_<num_nodes>n.svg``: grad_norm, lr,
    and max_loss vs step.
  - ``tokens_vs_time_<model>_<num_nodes>n.svg``: cumulative
    ``n_tokens_seen`` vs wall-clock datetime.

The dense raw per-step traces (13K+ points) use ``rasterized=True`` so
matplotlib embeds them as a small PNG inside the otherwise-vector SVG.
Axes, labels, ticks, smoothed curves, and the legend all stay vector;
``savefig(..., dpi=200)`` controls the resolution of the rasterized
region. Net file size: ~50-150 KB SVG vs ~300-500 KB PNG.

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
# Each key here drives the figure filename + output dir:
#   docs/production/agpt/<model>/n<num_nodes>/figures/<scope>_<key>n.svg
# So the key MUST encode model + version + node count, e.g. "2b_v1_256",
# "20b_v2_512". Don't include the trailing "n" — the template adds it.
#
# v1 = original 2026-04-{14..29} runs (torch 2.10, LBS=1) trained with
#      `--training.dtype=bfloat16`. Sub-ULP master-weight updates froze
#      every RMSNorm.weight at its 1.0 init; loss curves are real but
#      the model has no trainable normalization. Tainted, superseded
#      by v2. Kept here for the historical record. See
#      docs/guides/training-dtype-bf16-norm-freeze.md.
# v2 = fresh restarts on 2026-04-30 from /flare/AuroraGPT/foremans/runs/
#      agpt-{2b,20b}-v2/ (torch 2.13 venv, LBS=2,
#      `--training.dtype=float32`, plain CrossEntropyLoss). These are
#      the current production runs.
PRODUCTION_RUNS: dict[str, dict] = {
    "2b_v1_256": {
        "run_ids": [
            "v5ytgu0o", "pjanidnw", "4u9w23p9", "tahlsmy9", "iy1xbv0t",
            "11jzfnno", "hqwaw075", "6ictshbs", "wviyqysc",
        ],
        "num_nodes": 256,
        "model": "2b",
    },
    "20b_v1_256": {
        "run_ids": [
            "q9oq5huj", "pnkaurba", "lrlv3xsc", "pigwfqkg", "lvyzlocg",
            "e2anhgt2", "he01jr7f", "t0ja3dl4",
        ],
        "num_nodes": 256,
        "model": "20b",
    },
    "2b_v2_256": {
        # lytjeegk = 8459818 (6h initial, step 0->2070)
        # 0t4h0kuw = 8470100 (12h chain1, resumed step 2000)
        # j7bz39tj = 8470101 (12h chain2)
        # 0qpf3hnc = 8481320 (failover wrapper, 2026-05-22)
        "run_ids": ["lytjeegk", "0t4h0kuw", "j7bz39tj", "0qpf3hnc"],
        "num_nodes": 256,
        "model": "2b",
    },
    "2b_v2_512": {
        # i252kps9 = 8460301 (initial 6h, step 0->1387)
        # d4hlr8qe = 8463626 (12h chain1, resumed step 1300, ended step 5073)
        # 1va7zfki = 8463627 (12h chain2, resumed step 5000, ended step 6955)
        # 6op7ozfh = 8466847 (12h chain3)
        # y70rh76h = 8479989 (12h chain5, advanced step 13282 -> 13400)
        # logai2xn = 8485509 (1h19m chain6, pinned at step-13400)
        # 2qqhpcrm = 8485511 (1h23m chain7, also pinned at step-13400)
        "run_ids": ["i252kps9", "d4hlr8qe", "1va7zfki", "6op7ozfh",
                    "y70rh76h", "logai2xn", "2qqhpcrm"],
        "num_nodes": 512,
        "model": "2b",
    },
    "20b_v2_512": {
        # 9tsyx5us = 8460302 (initial 6h, step 0->300)
        # ej3zy5cq = 8463628 (12h chain1, resumed step 200, ended step 863)
        # 8466848 (chain2 resubmit) crashed at startup with std::bad_alloc, no wandb run
        # s6b159xk = 8479579 (12h chain2 retry, silent hang killed by qdel @ 5h56m)
        # gkzl19dg = 8481645 (failover wrapper, 200 fresh steps; step-900 ckpt
        #            never persisted — log shows step 800->1000 in-memory)
        # 10vf1mqr = 8481647 (failover continuation, 2026-05-22, currently running)
        "run_ids": ["9tsyx5us", "ej3zy5cq", "s6b159xk", "gkzl19dg", "10vf1mqr"],
        "num_nodes": 512,
        "model": "20b",
    },
    "20b_v2_256": {
        # r1yyxbmt = 8463659 (12h initial, step 0->363, NODE_FAIL)
        # 72airpph = 8470102 (3h, resumed step 300, gloo TCP timeout @ ~3h)
        # m9c5wx2e = 8470103 (3h, chain2, also gloo TCP timeout @ ~3h)
        # 6eocrnxs = 8479581 (12h chain3 retry, resumed step 400, gloo TCP @ 3h39m)
        # 5481v99b = 8479582 (chain4 continuation, 2h40m)
        # yrq1s1ac = 8481646 (failover wrapper validated, swap+retry; no new ckpt persisted)
        # xt03uvp6 = 8481648 (failover continuation, 2026-05-22, currently running)
        "run_ids": ["r1yyxbmt", "72airpph", "m9c5wx2e", "6eocrnxs",
                    "5481v99b", "yrq1s1ac", "xt03uvp6"],
        "num_nodes": 256,
        "model": "20b",
    },
    "2b_v2_512_lr3.22e-5": {
        # 8edrii5e = 8467141 (4h, sqrt(2)-LR fork — separate ckpt dir gbs12288-lr3.22e-5)
        # oujzdxri = 8467142 (1h53m, chain2)
        # Tests whether scaling LR by sqrt(2) closes per-token gap to 256N at GBS=12288
        "run_ids": ["8edrii5e", "oujzdxri"],
        "num_nodes": 512,
        "model": "2b",
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

    # rasterized=True on the dense raw lines keeps them as a small embedded
    # PNG inside the SVG (13K+ points would otherwise be 13K SVG path nodes).
    # Smoothed lines + axes + legend stay vector.
    ax = axes[0]
    ax.plot(steps, loss, color=color, alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(loss), color=color, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5,
            rasterized=True)
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
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(mfu), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # dpi here sets resolution of the rasterized region in the SVG.
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
    ax.plot(steps, grad_norm, color=color, alpha=0.25, linewidth=0.5,
            rasterized=True)
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
    ax.plot(steps, max_loss, color="#7B1FA2", alpha=0.25, linewidth=0.5,
            rasterized=True)
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


def plot_overlay(
    series: list[dict],
    model_name: str,
    output_path: Path,
) -> Path:
    """Overlay multiple PRODUCTION_RUNS entries on a 3-panel loss/TPS/MFU
    dashboard.

    Each ``series`` entry is a dict with:
        - ``key``    : PRODUCTION_RUNS key (used in the legend)
        - ``data``   : already-fetched + concatenated W&B history
        - ``color``  : matplotlib color
        - ``alpha``  : line alpha (raw curve uses 0.4× this)

    The point is to make the v1 (bf16-tainted) vs v2 (fp32) contrast
    visually unmissable: the loss curves descend together but their
    *eval-time* behavior diverges because v1 has frozen RMSNorm
    weights. See docs/guides/training-dtype-bf16-norm-freeze.md.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} — v1 (bf16-master) vs v2 (fp32-master)",
        fontsize=14,
        fontweight="bold",
    )

    last_steps = []
    for s in series:
        data = s["data"]
        steps = data["_step"].astype(float)
        loss = data["loss_metrics/global_avg_loss"].astype(float)
        tps = data["throughput(tps)"].astype(float)
        mfu = data["mfu(%)"].astype(float)
        valid = ~np.isnan(steps)
        steps = steps[valid]
        loss = loss[valid]
        tps = tps[valid]
        mfu = mfu[valid]
        if len(steps) == 0:
            continue
        last_steps.append(int(steps[-1]))

        color = s["color"]
        alpha = s["alpha"]
        label = s["key"]

        axes[0].plot(steps, loss, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[0].plot(steps, smooth(loss), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[1].plot(steps, tps, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[1].plot(steps, smooth(tps), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[2].plot(steps, mfu, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[2].plot(steps, smooth(mfu), color=color, alpha=alpha, linewidth=1.8, label=label)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("Tokens/sec/GPU")
    axes[1].set_title("Throughput per GPU")
    axes[1].legend(loc="lower right")
    axes[2].set_ylabel("MFU (%)")
    axes[2].set_xlabel("Training Step")
    axes[2].set_title("Model FLOPs Utilization")
    axes[2].legend(loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# Color/alpha presets per PRODUCTION_RUNS key. Lower alpha for the
# v1 (bf16-tainted) entries so v2 reads as the "real" curve.
OVERLAY_STYLE: dict[str, dict] = {
    "2b_v1_256":  {"color": "#94a3b8", "alpha": 0.55},  # slate, faded
    "2b_v2_256":  {"color": "#1E88E5", "alpha": 1.00},
    "2b_v2_512":  {"color": "#0d47a1", "alpha": 1.00},
    "20b_v1_256": {"color": "#94a3b8", "alpha": 0.55},
    "20b_v2_512": {"color": "#D32F2F", "alpha": 1.00},
}


def overlay_keys_for_model(model: str) -> list[str]:
    """Return PRODUCTION_RUNS keys whose `model` field matches `model`."""
    return [k for k, v in PRODUCTION_RUNS.items() if v.get("model", k) == model]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Run key from PRODUCTION_RUNS to plot (e.g. 2b_v1_256, "
            "2b_v2_256, 2b_v2_512, 20b_v1_256, 20b_v2_512), or omit "
            "to plot all."
        ),
    )
    parser.add_argument(
        "--overlay", type=str, default=None, choices=["2b", "20b"],
        help=(
            "Generate a v1-vs-v2 overlay dashboard for the given model, "
            "instead of (or in addition to) per-run dashboards. The "
            "figure goes to docs/production/agpt/<model>/figures/"
            "overlay_<model>_v1_vs_v2.svg."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: docs/production/agpt/<model>/n<num_nodes>/figures/ for per-trajectory dashboards; docs/production/agpt/<model>/figures/ for overlays)",
    )
    args = parser.parse_args()

    api = wandb.Api()

    if args.overlay is not None:
        keys_to_overlay = overlay_keys_for_model(args.overlay)
        if not keys_to_overlay:
            raise SystemExit(
                f"no PRODUCTION_RUNS entries found for model={args.overlay!r}"
            )
        out_dir = args.output_dir or (
            DOCS_BASE / "production" / "agpt" / args.overlay / "figures"
        )
        series = []
        for key in keys_to_overlay:
            cfg = PRODUCTION_RUNS[key]
            print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
            data = concat_runs(api, cfg["run_ids"])
            print(f"  Concatenated: {len(data['_step'])} unique steps")
            if len(data["_step"]) == 0:
                print(f"  no data, skipping {key}")
                continue
            style = OVERLAY_STYLE.get(
                key, {"color": MODEL_COLORS.get(args.overlay, "#666"), "alpha": 1.0}
            )
            series.append({"key": key, "data": data, **style})
        if not series:
            raise SystemExit("no series with data — nothing to overlay")
        plot_overlay(
            series,
            args.overlay,
            out_dir / f"overlay_{args.overlay}_v1_vs_v2.svg",
        )
        return

    keys = [args.model] if args.model else list(PRODUCTION_RUNS)
    for key in keys:
        cfg = PRODUCTION_RUNS[key]
        # `model` (for color + figure title) defaults to the dict key.
        model_name = cfg.get("model", key)
        # Output dir is keyed on (model, node count) — per-trajectory
        # figures land at production/agpt/<model>/n<nodes>/figures/.
        # The model-level overlay (handled in the --overlay branch
        # above) stays in production/agpt/<model>/figures/.
        out_dir = args.output_dir or (
            DOCS_BASE
            / "production"
            / "agpt"
            / model_name
            / f"n{cfg['num_nodes']}"
            / "figures"
        )

        print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
        data = concat_runs(api, cfg["run_ids"])
        print(f"  Concatenated: {len(data['_step'])} unique steps")
        if len(data["_step"]) == 0:
            print(f"  no data, skipping {key}")
            continue

        plot_dashboard(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"production_{key}n.svg",
        )
        plot_diagnostics(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"training_diagnostics_{key}n.svg",
        )
        plot_tokens_vs_time(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"tokens_vs_time_{key}n.svg",
        )


if __name__ == "__main__":
    main()
