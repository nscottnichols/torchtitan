#!/usr/bin/env python3
"""Pull train+val loss history for the MDS AuroraGPT-2B SophiaG run.

The Megatron-DeepSpeed run was restarted ~191 times — every PBS job became
its own W&B run under aurora_gpt/AuroraGPT — so we identify the slice via
config.args.optimizer == "sophiag" + hidden_size == 2048 and stitch
all per-run histories together by the `iteration` axis (deduping on
collision; later starts win).

Output:
    train_loss.csv   columns: iteration, lm_loss, run_id
    val_loss.csv     columns: iteration, val_loss, run_id

Usage:
    python3 pull_wandb_loss.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import wandb

ENTITY = "aurora_gpt"
PROJECT = "AuroraGPT"

TRAIN_KEY = "loss/lm loss"
TRAIN_ITER_KEY = "loss/iteration"
VAL_KEY = "val/lm loss"
VAL_ITER_KEY = "val/iteration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on number of runs (debug)."
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent,
        help="Directory to write CSVs.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    # Filter to the *production* AuroraGPT-2B SophiaG continuation:
    # nl=12, hs=2048, seq=8192, gbs=6144. Loosening any of these brings
    # in unrelated 24-layer Sophia experiments at smaller seq lengths
    # whose iteration counters collide with this run's timeline and
    # produce spurious sub-2 loss dips.
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "config.args.optimizer": "sophiag",
            "config.args.hidden_size": 2048,
            "config.args.num_layers": 12,
            "config.args.seq_length": 8192,
            "config.args.global_batch_size": 6144,
        },
        per_page=100,
        order="+created_at",
    )

    train_rows: dict[int, tuple[float, str]] = {}
    val_rows: dict[int, tuple[float, str]] = {}
    n = 0
    for r in runs:
        n += 1
        if args.limit and n > args.limit:
            break
        # Pull train history (loss/lm loss + loss/iteration)
        added_train = 0
        try:
            for row in r.scan_history(keys=[TRAIN_KEY, TRAIN_ITER_KEY]):
                it = row.get(TRAIN_ITER_KEY)
                tl = row.get(TRAIN_KEY)
                if it is not None and tl is not None:
                    train_rows[int(it)] = (float(tl), r.id)
                    added_train += 1
        except Exception as e:
            print(f"  [{n}] {r.name} ({r.id}): train scan err {e}")
        # Pull val history (val/lm loss + val/iteration)
        added_val = 0
        try:
            for row in r.scan_history(keys=[VAL_KEY, VAL_ITER_KEY]):
                it = row.get(VAL_ITER_KEY)
                vl = row.get(VAL_KEY)
                if it is not None and vl is not None:
                    val_rows[int(it)] = (float(vl), r.id)
                    added_val += 1
        except Exception as e:
            print(f"  [{n}] {r.name} ({r.id}): val scan err {e}")
        print(
            f"  [{n}] {r.name} ({r.id}) {str(r.created_at)[:10]} "
            f"+{added_train} train, +{added_val} val "
            f"-> totals {len(train_rows)} train / {len(val_rows)} val"
        )

    # Write CSVs sorted by iteration
    train_csv = args.out_dir / "train_loss.csv"
    with train_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "lm_loss", "run_id"])
        for it in sorted(train_rows):
            tl, rid = train_rows[it]
            w.writerow([it, tl, rid])
    print(f"wrote {train_csv} ({len(train_rows)} rows)")

    val_csv = args.out_dir / "val_loss.csv"
    with val_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "val_loss", "run_id"])
        for it in sorted(val_rows):
            vl, rid = val_rows[it]
            w.writerow([it, vl, rid])
    print(f"wrote {val_csv} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
