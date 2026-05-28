# Production Training — agpt 2B

> Last updated: 2026-05-28
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

## Snapshot

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 256N (async)**](n256/README.md) | 8508977 done (pals-RPC exit 127); 8510692 Q for slot | **55,026** | **2.67** | **~2.77T (59.3%)** |
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | 8508753 done (walltime); 8509042 crashed (std::bad_alloc); 8510693 Q for slot | **30,484** | **2.71** | **~3.07T (65.7%)** |
| [v2 1024N](n1024/README.md) | Crashed at startup, not retried | — | — | — |

**Headlines (2026-05-28):**

- **256N async chain**: 8505175 → 8505252 → 8507195 → 8507198 → 8508020 (walltime) → **8508977 (pals-RPC exit 127)** at step **55,026** (loss 2.67, ~2.77T tokens, 59.3% of 4.67T target). 8510692 cont5 Q for slot. Async-mode stable; eval plateau in HSn 0.547-0.553 / ARC-E 0.59-0.64 range since step 47K.
- **🏁 512N sync-mode workaround holding** (`CHECKPOINT_ASYNC_MODE=disabled`): chain advanced 8506221 (+21) → 8507196 (pals-RPC, +76) → 8507199 (+50) → 8508753 (walltime, +80) to step **30,484**. Next continuation 8509042 hit the documented intermittent `std::bad_alloc` in `set_determinism` at 6,144 ranks (same failure mode as 8466848); 8510693 cont5 Q for slot. Loss **2.71**, ~3.07T tokens, **65.7%** of target.

## Per-trajectory detail

- [n256/](n256/README.md) — v2 256N async (running)
- [n512/](n512/README.md) — **canonical v2 512N sync chain**
- [n1024/](n1024/README.md) — v2 1024N (queued, no data)

## Eval scores

See [`docs/evals/agpt/2b/`](../../../evals/agpt/2b/README.md) for the
current 2B lm-eval tables (HellaSwag / ARC-Easy / ARC-Challenge /
Winogrande). Plateau: ARC-Easy ~0.645, HellaSwag acc_norm ~0.547.
**Note:** as of 2026-05-27 the 20B 512N sync chain at step 3,270 now
beats this 2B 256N plateau on every benchmark per token — see
[`evals/agpt/20b/`](../../../evals/agpt/20b/README.md).
