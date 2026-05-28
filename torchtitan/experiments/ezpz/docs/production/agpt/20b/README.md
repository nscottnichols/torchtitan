# Production Training — agpt 20B

> Last updated: 2026-05-28
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

## 🏁 Headline (2026-05-28)

**20B 512N sync chain now beats 2B 256N async on every benchmark per token.** Eval'd 30 ckpts step-100 → step-3800:

- ARC-Easy `acc` 0.266 → **0.678** (+41pp)
- HellaSwag `acc_norm` 0.254 → **0.611** (+36pp)
- ARC-C `acc_norm` 0.232 → **0.355** (+12pp)
- Winogrande `acc` 0.492 → **0.582** (+9pp)

Monotonic lift — no plateau, no oscillation, 30+ consecutive ckpts trending up. See
[`evals/agpt/20b/`](../../../evals/agpt/20b/README.md) for the full tables.

## Snapshot

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | 8508214 done (12h walltime end 2026-05-28 09:26); 8509393 Q ~16h (capacity-blocked in `small`); 8510696 H | **3,806** | **2.60** | **~382B (8.2%)** |
| [v2 256N](n256/README.md) (8505255 final) | Done — 12h walltime end 2026-05-26 20:35 at step **1,125**. No chain continuation queued (256N is per-token comparator; canonical 20B chain is 512N). | 1,125 | — | — |
| [v2 1024N](n1024/README.md) | First attempt 8463183 crashed at startup (SIGSEGV at 12,288 ranks); not retried | — | — | — |

**Canonical 512N chain (sync-mode)**: 8505258 (🏁 sync-mode breakthrough, +6) → 8505259 (+6) → 8507197 (+6) → 8507200 (+6, walltime, ended step 3,270) → 8508214 (walltime, +5 ckpts, ended step 3,806) → 8509393 Q → 8510696 H.

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
