import json
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
TRAINER_STATE = ROOT / "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729/trainer_state.json"
OUT_DIR = ROOT / "torchtitan/experiments/ezpz/docs/production/sft/aurora2b/tulu_math_uc_mix/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOB_SPANS = [
    ("12468404", 1,  10, 140, "#fde0dc"),
    ("12468404", 2,  10, 100, "#fdcec5"),
    ("12468409", 1, 110, 200, "#dde9fb"),
    ("12468409", 2, 210, 300, "#c7dcfa"),
    ("12468437", 1, 310, 400, "#d8f0d8"),
    ("12468437", 2, 410, 500, "#c5e9c5"),
    ("12468437", 3, 510, 600, "#b1e2b1"),
    ("12468437", 4, 610, 729, "#9bdb9b"),
]


def add_job_shading(ax, smin, smax):
    for job_id, attempt, lo, hi, color in JOB_SPANS:
        if hi < smin or lo > smax:
            continue
        ax.axvspan(lo, hi, alpha=0.35, color=color, zorder=0)


def render_curves():
    state = json.load(open(TRAINER_STATE))
    log_history = state["log_history"]
    steps = np.array([e["step"] for e in log_history])
    loss = np.array([e["loss"] for e in log_history])
    grad_norm = np.array([e["grad_norm"] for e in log_history])
    lr = np.array([e["learning_rate"] for e in log_history])
    acc = np.array([e["mean_token_accuracy"] for e in log_history])
    entropy = np.array([e["entropy"] for e in log_history])
    tokens_b = np.array([e["num_tokens"] for e in log_history]) / 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    fig.suptitle(
        f"AuroraGPT-2B x tulu_math_uc_mix SFT trajectory  (global_step {steps[-1]}, 3 epochs, GBS=6144)",
        fontsize=13, y=0.995,
    )
    smin, smax = int(steps.min()), int(steps.max())
    panels = [
        (axes[0, 0], loss,       "loss",                 "loss",                "tab:blue"),
        (axes[0, 1], grad_norm,  "grad_norm",            "grad_norm",           "tab:red"),
        (axes[0, 2], lr * 1e5,   "learning rate",        "lr (x 1e-5)",         "tab:orange"),
        (axes[1, 0], acc,        "mean token accuracy",  "mean_token_accuracy", "tab:green"),
        (axes[1, 1], entropy,    "entropy",              "entropy (nats)",      "tab:purple"),
        (axes[1, 2], tokens_b,   "tokens seen",          "tokens (B)",          "tab:gray"),
    ]
    for ax, y, title, ylabel, color in panels:
        add_job_shading(ax, smin, smax)
        ax.plot(steps, y, color=color, lw=1.5)
        ax.scatter(steps, y, s=10, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("global step")
    # Legend for the job shading
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=f"{jid} attempt {a}")
        for jid, a, _, _, color in JOB_SPANS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = OUT_DIR / "sft-curves.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def render_throughput():
    """Aggregate per-wandb-run throughput + tokens/sec."""
    # Look at recent SFT wandb runs (2026-06-10 between 10:00 and 14:00)
    wandb_glob = ROOT / "wandb/run-20260610_1[0123]*"
    runs = sorted(glob(str(wandb_glob)))
    data = []
    for rd in runs:
        meta_p = Path(rd) / "files" / "wandb-metadata.json"
        hist_p = Path(rd) / "files" / "wandb-history.jsonl"
        sum_p = Path(rd) / "files" / "wandb-summary.json"
        if not meta_p.exists():
            continue
        meta = json.load(open(meta_p))
        args = " ".join(meta.get("args", []))
        if "sft_dataset" not in args or "tulu" not in args:
            continue
        if not hist_p.exists():
            continue
        steps, tps, runtime = [], [], []
        with open(hist_p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                # TRL writes 'train/loss', 'train/global_step', 'train/learning_rate' etc.
                # throughput keys: 'train/train_tokens_per_second' or similar; also 'train_runtime'
                if "train/loss" in e and "train/global_step" in e:
                    steps.append(e["train/global_step"])
                # final summary contains throughput
        summary = json.load(open(sum_p)) if sum_p.exists() else {}
        data.append({
            "run_id": Path(rd).name.split("-")[-1],
            "start": Path(rd).name.split("-")[1],
            "summary": summary,
            "n_log": len(steps),
        })
    if not data:
        print("no throughput data found")
        return None
    # Just print a summary; not all runs have throughput so I'll show what we have
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title("SFT per-run throughput (from wandb summaries)")
    keys_of_interest = ["train_steps_per_second", "train_samples_per_second", "train_runtime"]
    xs = list(range(len(data)))
    for k in keys_of_interest:
        ys = [d["summary"].get(k, np.nan) for d in data]
        if any(not np.isnan(y) for y in ys):
            ax.plot(xs, ys, marker="o", label=k)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d['start'][8:]}\n{d['run_id'][:6]}" for d in data], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("value (mixed units)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "sft-throughput.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    render_curves()
    render_throughput()
