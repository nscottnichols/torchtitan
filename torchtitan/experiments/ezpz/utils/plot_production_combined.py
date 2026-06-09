#!/usr/bin/env python3
"""Combined-overlay production chart across production trajectories.

One figure, 3 subplot panels (Loss / TPS-per-GPU / MFU), with one curve
per trajectory plotted against **tokens consumed (log scale)** so
trajectories with different GBS can be compared directly. Used in 4
places, controlled by `--model`:

    --model all  (default)
        AuroraGPT-2B:  MDS-ref + TT v2 256N + TT v2 512N
        AuroraGPT-20B: TT v2 256N + TT v2 512N
        Writes: docs/production/figures/all_production_training.svg
        Embedded in: docs/production/README.md (landing page) + the
        dir-level docs/production/agpt/README.md rollup.

    --model 2b
        2B trajectories only: MDS-ref + TT v2 256N + TT v2 512N
        + sqrt(2)-LR fork.
        Writes: docs/production/agpt/2b/figures/all_2b_chains.svg
        Embedded in: docs/production/agpt/2b/README.md

    --model 20b
        20B trajectories only: TT v2 256N + TT v2 512N
        Writes: docs/production/agpt/20b/figures/all_20b_chains.svg
        Embedded in: docs/production/agpt/20b/README.md

MDS has no MFU column so it's only on the Loss + TPS panels.

Run:
    python3 -m torchtitan.experiments.ezpz.utils.plot_production_combined
    python3 -m torchtitan.experiments.ezpz.utils.plot_production_combined --model 2b
    python3 -m torchtitan.experiments.ezpz.utils.plot_production_combined --model 20b
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ambivalent is required — silent fallback hides style regressions.
import ambivalent  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(ambivalent.STYLES["ambivalent"])
plt.rcParams["font.family"] = "monospace"

# Reuse the W&B fetch + .o-log fallback already proven in
# plot_production_wandb.py — same data path that feeds the per-trajectory
# dashboards, so this chart can't diverge from those.
from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: E402
    PRODUCTION_RUNS,
    concat_runs,
)

import wandb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]

# Per-model output paths. --model all writes the original cross-model overview.
OUT_PATHS = {
    "all": REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/figures/all_production_training.svg",
    "2b":  REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/agpt/2b/figures/all_2b_chains.svg",
    "20b": REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/agpt/20b/figures/all_20b_chains.svg",
}

# MDS data lives as a CSV pulled separately from the MDS W&B project.
MDS_CSV = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/production/agpt/2b-mds/loss_data/train_metrics.csv"
)
MDS_TOKENS_PER_STEP = 7_770e9 / 140_000  # ~55.5M tokens/step at GBS=3072, seq=8192

# Canonical per-trajectory palette — keep in sync with
# eval/plot_evals_combined.py and per-model {2b,20b}/plot_v1_vs_v2.py.
COLOR_2B_MDS         = "C0"       # matplotlib C0 (ambivalent palette first color — reads well on both light + dark bg)
COLOR_2B_TT_256N     = "#ef5350"  # salmon-red
COLOR_2B_TT_512N     = "#b71c1c"  # dark red
COLOR_2B_TT_512N_FORK = "#8e24aa"  # purple — sqrt(2)-LR fork (separate ckpt dir)
COLOR_20B_TT_256N    = "#fb8c00"  # orange (20B 256N: one-off NODE_FAIL run; no canonical color)
COLOR_20B_TT_512N    = "#1b8a3a"  # green

# Trajectory registry keyed by model. The "all" view = 2b + 20b minus the
# sqrt(2)-LR fork (which is too small/divergent to share an x-axis cleanly).
TRAJECTORIES_BY_MODEL: dict[str, list[dict]] = {
    "2b": [
        {
            "label": "2B-MDS (n256, SophiaG ref)",
            "source": "mds",
            "csv_path": str(MDS_CSV),
            "tokens_per_step": MDS_TOKENS_PER_STEP,
            "color": COLOR_2B_MDS,
            "linestyle": "--",
            "marker": None,
        },
        {
            "label": "2B-TT v2 (n256, async)",
            "source": "wandb",
            "key": "2b_v2_256",
            "tokens_per_step": 6144 * 8192,
            "color": COLOR_2B_TT_256N,
            "linestyle": "-",
            "marker": None,
        },
        {
            "label": "2B-TT v2 (n512, sync)",
            "source": "wandb",
            "key": "2b_v2_512",
            "tokens_per_step": 12288 * 8192,
            "color": COLOR_2B_TT_512N,
            "linestyle": "-",
            "marker": None,
        },
        {
            "label": "2B-TT v2 (n512, sqrt2-LR fork)",
            "source": "wandb",
            "key": "2b_v2_512_lr3.22e-5",
            "tokens_per_step": 12288 * 8192,
            "color": COLOR_2B_TT_512N_FORK,
            "linestyle": ":",
            "marker": None,
        },
    ],
    "20b": [
        {
            "label": "20B-TT v2 (n256)",
            "source": "wandb",
            "key": "20b_v2_256",
            "tokens_per_step": 3072 * 8192,
            "color": COLOR_20B_TT_256N,
            "linestyle": "--",
            "marker": None,
        },
        {
            "label": "20B-TT v2 (n512, sync)",
            "source": "wandb",
            "key": "20b_v2_512",
            "tokens_per_step": 12288 * 8192,
            "color": COLOR_20B_TT_512N,
            "linestyle": "-",
            "marker": None,
        },
    ],
}

# "all" = 2b canonical (MDS, n256, n512) + 20b (n256, n512). Skip the LR
# fork since it's a tiny diagnostic run that crowds the legend.
TRAJECTORIES_BY_MODEL["all"] = [
    t for t in TRAJECTORIES_BY_MODEL["2b"]
    if t.get("key") != "2b_v2_512_lr3.22e-5"
] + TRAJECTORIES_BY_MODEL["20b"]


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


def load_wandb_trajectory(
    api: wandb.Api, key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull a trajectory from W&B via the existing concat_runs path.
    Returns (steps, loss, tps_per_gpu, mfu) arrays with NaN-rows filtered.
    """
    cfg = PRODUCTION_RUNS[key]
    data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    tps = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)
    valid = ~np.isnan(steps) & ~np.isnan(loss)
    return steps[valid], loss[valid], tps[valid], mfu[valid]


def load_mds_trajectory(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse train_metrics.csv (iteration, lm_loss, grad_norm, tflops,
    tps_per_gpu, run_id). Returns (iteration, loss, tps_per_gpu). MDS
    doesn't log MFU so the MFU panel just skips this trajectory.
    """
    iters, losses, tps_list = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                it = int(row["iteration"])
                lo = float(row["lm_loss"])
                tp = float(row["tps_per_gpu"])
            except (ValueError, KeyError):
                continue
            iters.append(it)
            losses.append(lo)
            tps_list.append(tp)
    return np.array(iters), np.array(losses), np.array(tps_list)


TITLE_BY_MODEL = {
    "all": "AuroraGPT production training — all canonical chains overlaid",
    "2b":  "AuroraGPT-2B production training — all chains overlaid",
    "20b": "AuroraGPT-20B production training — all chains overlaid",
}


def render(model: str) -> None:
    """Render one combined-overlay chart for the given model filter
    ('all', '2b', '20b') and write to OUT_PATHS[model]."""
    api = wandb.Api()

    trajectories = TRAJECTORIES_BY_MODEL[model]
    out_path = OUT_PATHS[model]

    # Pull all trajectories
    series: list[dict] = []
    for traj in trajectories:
        print(f"\n=== loading {traj['label']} ===")
        if traj["source"] == "wandb":
            steps, loss, tps, mfu = load_wandb_trajectory(api, traj["key"])
            tokens_b = steps * traj["tokens_per_step"] / 1e9
            print(f"  {len(steps)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "tokens_b": tokens_b, "loss": loss, "tps": tps, "mfu": mfu})
        else:  # mds
            iters, loss, tps = load_mds_trajectory(traj["csv_path"])
            tokens_b = iters * traj["tokens_per_step"] / 1e9
            print(f"  {len(iters)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "tokens_b": tokens_b, "loss": loss, "tps": tps, "mfu": None})

    # 3 panels: Loss, TPS/GPU, MFU
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(TITLE_BY_MODEL[model], fontsize=15, fontweight="bold")

    # Panel 1: Loss vs tokens (all 5)
    ax = axes[0]
    for s in series:
        ax.plot(
            s["tokens_b"], s["loss"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["loss"], window=min(100, max(2, len(s["loss"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}  (n={len(s['loss'])})",
        )
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    # Panel 2: TPS/GPU vs tokens (all 5)
    ax = axes[1]
    for s in series:
        ax.plot(
            s["tokens_b"], s["tps"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["tps"], window=min(100, max(2, len(s["tps"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title("Throughput per GPU")
    ax.grid(alpha=0.25)

    # Panel 3: MFU vs tokens (TT only — MDS has no MFU)
    ax = axes[2]
    for s in series:
        if s["mfu"] is None:
            continue
        ax.plot(
            s["tokens_b"], s["mfu"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["mfu"], window=min(100, max(2, len(s["mfu"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_title("Model FLOPs Utilization (TT only — MDS does not log MFU)")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # SVG keeps axes/labels/legend vector + dense raw lines rasterized
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
        if __doc__ else "Render combined-overlay production chart."
    )
    parser.add_argument(
        "--model", choices=("all", "2b", "20b", "every"), default="every",
        help="Which model filter to render. 'every' (default) renders all 3 "
             "(all, 2b, 20b) in sequence — used by update_all_charts.sh."
    )
    args = parser.parse_args()
    if args.model == "every":
        for m in ("all", "2b", "20b"):
            render(m)
    else:
        render(args.model)


if __name__ == "__main__":
    main()
