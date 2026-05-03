# Production Training — agpt 20B @ 1024 nodes

> **Status: queued, no data yet.**

## v2 — 20B @ 1024N — SophiaG LR=2.28e-5 (fp32 master)

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/` |
| Stack | torch 2.13 venv (yeet-env tarball mode) |
| Compile | off |
| GBS | 24,576 (LBS=2 × 1024 nodes × 12 GPUs) |
| Total tokens | 4.67T target |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n1024-gbs24576` (will create on first save) |

### Progress

| Job ID | Walltime | Steps | Status |
|--------|---------:|------:|--------|
| 8463183 | 12h | — | **Queued** (waiting for 1024-node slot) |

This is an *independent* trajectory from the canonical 512N chain
([n512/](../n512/README.md)) — it writes to a different ckpt dir
(`gbs24576` vs `gbs12288`) and would start fresh from step 0. Useful
as a scaling experiment, not as a chain extension.

There are no v1 figures for 1024N — v1 never ran at this scale.
