#!/usr/bin/env python3
"""Plot train + val loss for the MDS AuroraGPT-2B SophiaG continuation.

Reads {train,val}_loss.csv produced by ``pull_wandb_loss.py`` and writes
three PNGs into ../figures/:

    train_loss.png       train loss vs iteration (EMA-smoothed)
    val_loss.png         val loss vs iteration
    train_val_loss.png   both series on shared axes

The MDS run was a single SophiaG continuation that the W&B project split
across ~191 restart-runs; the puller already collapsed those into one
sequence keyed on the global iteration counter. The data-mix transitions
listed in the README correspond to these iteration boundaries (per the
schedule):

    iter ~95,000  : 4673B → 7064B mix shift
    iter ~134,000 : 7064B → 7770B mix shift

Stage transitions are dotted vertical lines on the plots.

Usage:
    python3 plot_loss.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import ambivalent  # noqa: F401

    plt.style.use(ambivalent.STYLES["ambivalent"])
except ImportError:
    pass

DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Stage transitions in the 3-stage SophiaG continuation. Iteration values
# come from the README's data-slice token counts divided by global batch
# size (6144) × seq_len (8192).
STAGE_BOUNDARIES = [
    (95_000, "ntok4673B → ntok7064B"),
    (134_000, "ntok7064B → ntok7770B"),
]


def load_csv(path: Path, value_col: str) -> tuple[list[int], list[float]]:
    iters: list[int] = []
    vals: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iteration"]))
            vals.append(float(row[value_col]))
    return iters, vals


def ema(vals: list[float], alpha: float = 0.05) -> list[float]:
    if not vals:
        return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def plot_train_loss(
    train_iters: list[int],
    train_vals: list[float],
    out_path: Path,
) -> None:
    smooth = ema(train_vals, alpha=0.02)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train_iters, train_vals, alpha=0.18, lw=0.6, label="raw", color="#3b82f6")
    ax.plot(train_iters, smooth, lw=1.6, label="EMA(α=0.02)", color="#1e3a8a")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Train Loss")
    ax.set_title("AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Training Loss")
    for x, label in STAGE_BOUNDARIES:
        if train_iters and x <= train_iters[-1]:
            ax.axvline(x, color="#888", ls=":", lw=1)
            ax.text(
                x, ax.get_ylim()[1] * 0.95, label,
                rotation=90, fontsize=8, va="top", ha="right", color="#666",
            )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(train_iters)} points)")


def plot_val_loss(
    val_iters: list[int],
    val_vals: list[float],
    out_path: Path,
) -> None:
    # Drop iter=0 random-init outlier (~9000) so the converging range is readable.
    pairs = [(it, v) for it, v in zip(val_iters, val_vals) if it > 0]
    iters = [it for it, _ in pairs]
    vals = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, marker="o", ms=3, lw=1.2, color="#dc2626")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Validation Loss")
    ax.set_title("AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Validation Loss")
    for x, label in STAGE_BOUNDARIES:
        if iters and x <= iters[-1]:
            ax.axvline(x, color="#888", ls=":", lw=1)
            ax.text(
                x, ax.get_ylim()[1] * 0.95, label,
                rotation=90, fontsize=8, va="top", ha="right", color="#666",
            )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points, dropped iter=0 outlier)")


def plot_train_val(
    train_iters: list[int],
    train_vals: list[float],
    val_iters: list[int],
    val_vals: list[float],
    out_path: Path,
) -> None:
    smooth = ema(train_vals, alpha=0.02)
    # Drop iter=0 val outlier — it's the random-init starting point and
    # would otherwise compress the y-axis range.
    val_pairs = [(it, v) for it, v in zip(val_iters, val_vals) if it > 0]
    v_iters = [it for it, _ in val_pairs]
    v_vals = [v for _, v in val_pairs]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        train_iters, train_vals, alpha=0.15, lw=0.6,
        label="train (raw)", color="#3b82f6",
    )
    ax.plot(
        train_iters, smooth, lw=1.4,
        label="train (EMA α=0.02)", color="#1e3a8a",
    )
    ax.plot(
        v_iters, v_vals, marker="o", ms=3, lw=1.0,
        label="validation", color="#dc2626",
    )
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Loss")
    ax.set_title(
        "AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Training + Validation Loss"
    )
    for x, label in STAGE_BOUNDARIES:
        last_iter = max(
            train_iters[-1] if train_iters else 0,
            v_iters[-1] if v_iters else 0,
        )
        if x <= last_iter:
            ax.axvline(x, color="#888", ls=":", lw=1)
            ax.text(
                x, ax.get_ylim()[1] * 0.95, label,
                rotation=90, fontsize=8, va="top", ha="right", color="#666",
            )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    train_iters, train_vals = load_csv(DATA_DIR / "train_loss.csv", "lm_loss")
    val_iters, val_vals = load_csv(DATA_DIR / "val_loss.csv", "val_loss")
    print(
        f"loaded: train={len(train_iters)} pts "
        f"(iter {train_iters[0] if train_iters else '?'}..{train_iters[-1] if train_iters else '?'}); "
        f"val={len(val_iters)} pts "
        f"(iter {val_iters[0] if val_iters else '?'}..{val_iters[-1] if val_iters else '?'})"
    )

    plot_train_loss(train_iters, train_vals, FIG_DIR / "train_loss.png")
    plot_val_loss(val_iters, val_vals, FIG_DIR / "val_loss.png")
    plot_train_val(
        train_iters, train_vals, val_iters, val_vals,
        FIG_DIR / "train_val_loss.png",
    )


if __name__ == "__main__":
    main()
