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
| [**v2 256N (async)**](n256/README.md) | **Running** (8508020 R, +chained) | **49,900+** | **2.68** | **~2.50T (53.5%)** |
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | **Running** (8508753 R, +2 held) | **27,100+** | **2.72** | **~2.73T (58.4%)** |
| [v2 1024N](n1024/README.md) | Crashed at startup, not retried | — | — | — |

**Headlines:**

- **256N async chain**: 8505175 → 8505252 → 8507195 → 8507198 → **8508020 R** at step **49,900+** (loss 2.68, ~2.50T tokens, 53.5% of 4.67T target). Async-mode stable; eval plateau ARC-Easy ~0.645, HellaSwag norm ~0.547.
- **🏁 512N sync-mode workaround validated** (`CHECKPOINT_ASYNC_MODE=disabled`): since 8506221 on 2026-05-24 broke through the May-3 async-cascade regression wall, the sync workaround has been holding cleanly. Recent chain: 8506221 (+21) → 8507196 (pals-RPC fail, +76 ckpts persisted) → 8507199 (+50) → **8508753 R** at step **27,100+** (loss 2.72, ~2.73T tokens, 58.4% of 4.67T target). One quirk: 8507196 hit a separate Aurora pals-RPC launcher infra bug (exit 127) — distinct from the async-cascade.

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
