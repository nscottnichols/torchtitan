# Production Training — agpt 20B

> Last updated: 2026-05-28
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

## 🏁 Headline (2026-05-27)

**20B 512N sync chain now beats 2B 256N async on every benchmark per token.** Eval'd 16 ckpts step-900 → step-3200:

- ARC-Easy `acc` 0.463 → **0.665** (+20pp)
- HellaSwag `acc_norm` 0.296 → **0.574** (+28pp)
- ARC-C `acc_norm` 0.224 → **0.322** (+10pp)

Monotonic lift — no plateau, no oscillation, 24+ consecutive ckpts trending up. See
[`evals/agpt/20b/`](../../../evals/agpt/20b/README.md) for the full tables.

## Snapshot

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | 8507200 done (12h walltime end 2026-05-27 03:43); 8508214 Q ~10h (capacity-blocked in `small`); 8509393 H | **3,270** | **2.65** | **~329B (7.0%)** |
| [v2 256N](n256/README.md) (8505255 final) | Done — 12h walltime end 2026-05-26 20:35 at step **1,125**. No chain continuation queued (256N is per-token comparator; canonical 20B chain is 512N). | 1,125 | — | — |
| [v2 1024N](n1024/README.md) | First attempt 8463183 crashed at startup (SIGSEGV at 12,288 ranks); not retried | — | — | — |

**Canonical 512N chain (sync-mode)**: 8505258 (🏁 sync-mode breakthrough, +6) → 8505259 (+6) → 8507197 (+6) → 8507200 (+6, walltime end at step 3,270) → 8508214 Q → 8509393 H.

## Per-trajectory detail

- [n512/](n512/README.md) — **canonical v2 512N sync chain**
- [n256/](n256/README.md) — v2 256N (ended step 1,125; no continuation)
- [n1024/](n1024/README.md) — v2 1024N (8463183 crashed at startup, std::bad_alloc / SIGSEGV — needs investigation)

## Eval scores

See [`docs/evals/agpt/20b/`](../../../evals/agpt/20b/README.md) for the
current 20B lm-eval tables and the **🏁 headline** finding above. At
step 3,200 (~329B tokens) v2 512N sync now reaches ARC-Easy **0.665**
and HellaSwag `acc_norm` **0.574** — beating the 2B 256N async plateau
on every benchmark per token. Monotonic across 24+ consecutive ckpts.
