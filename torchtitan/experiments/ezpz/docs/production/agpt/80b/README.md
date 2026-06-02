# Production Training — agpt 80B

> Last updated: 2026-06-02

## v2 status — STILL BLOCKED (11+ failed dispatches since 2026-05-11)

**Zero ckpts persisted** for 80B production since the v2 effort began on 2026-05-11. The original 8480361/8480362
dispatch (and 9 follow-up retries) all failed before saving a single checkpoint. Two distinct failure modes:

| Mode | Where | Status |
|------|-------|--------|
| **SIGSEGV cascade** | 80B 256N production (every dispatch) | Documented in [`20260524-80b-256n-sigsegv-cascade-8505222.md`](../../../experiments/agpt/aurora/20260524-80b-256n-sigsegv-cascade-8505222.md) |
| **`blendcorpus` EOFError race** | 80B 8N smoke 8505326 | Documented in [`blendcorpus-eoferror-race.md`](../../../guides/known-bugs/blendcorpus-eoferror-race.md) |

**Latest dispatch `8505222` (2026-05-24)** failed after **5 wrapper retries** — every attempt hit SIGSEGV on a
different bad node, 3 of them from the **x4101c5/c6 rack cluster**. The failover wrapper rotates spares correctly,
but the bad-node hit-rate at 256N is high enough that 10 spares is not enough headroom.

The 4N smoke 12466025 from 2026-05-05 (Aurora) was the **first successful 80B v2 training**
(20 steps, loss 12.98 → 10.46). Replicated on Sunspot 4N as job 12467825 on
2026-06-02 with the xccl_split_group workaround now in place (commit `8031d1d3a`):
20 steps, loss 12.94 → 10.39, ~17.8% MFU, 88.94% memory — numerically equivalent
to the May 5 baseline. The xccl workaround does NOT regress the working config.
See [`20260602-smoke-n4-80b-tp2-xccl-workaround.md`](../../../experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md).

### Next steps (from SIGSEGV writeup)

1. **Try `select=296`** (40 spares vs current 10) to absorb more bad nodes per cascade.
2. **File ALCF ticket** for the x4101c5/c6 rack — 3 of the 5 retry failures landed there.
3. **Try 64N / 128N** to characterize whether the SIGSEGV rate scales with node count or is rack-specific.

### Working config (smoke-only)

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| 4N smoke 12466025 (Aurora, 2026-05-05) | Done (20 steps) | 20 | 12.98 → 10.46 | — |
| 4N smoke 12467825 (Sunspot, 2026-06-02, xccl workaround in place) | Done (20 steps) | 20 | 12.94 → 10.39 | — |
| [**v2 256N**](n512/README.md) (canonical chain attempt) | 11+ failed dispatches | 0 | — | — |
| v2 512N | Not yet attempted | — | — | — |

Working config (proven in 4N smoke): AdamW LR=1e-6, TP=2, AC=full, compile=OFF, fp32-master. Loss descended
cleanly 12.98 → 10.46 over 20 steps. Submit script:
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../../scripts/submit_agpt_80b_aurora_venv_failover.sh).

Production clone: `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/`.

Historical v1 (NaN'd) runs: see [`../historical/v1-bf16/`](../historical/v1-bf16/README.md).
