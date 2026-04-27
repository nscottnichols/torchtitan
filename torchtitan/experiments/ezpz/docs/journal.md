# Development Journal

Running log of what's happening, session by session. Most recent first.

---

## 2026-04-27 — Full 10B training, TorchMuon, local dataset

### Local Dataset Cache

- Downloaded FineWeb-Edu `sample-100BT` (267 GB, 140 parquet files)
  to `/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/`
- Added `register_local_dataset()` to `datasets.py` for parquet/arrow files
- Registered as `fineweb_edu_local` — eliminates HF streaming rate limits
  and ensures reproducible data ordering across runs

### TorchMuon Integration

- Added `TorchMuonOptimizersContainer` using `torch.optim.Muon` (built-in
  since PyTorch 2.9)
- Required `_CompositeOptimizer` wrapper — `OptimizersContainer` expects one
  optimizer per model part, but Muon only handles 2D params (need separate
  AdamW for embeddings/head)
- Multiple fix iterations: missing `import torch`, empty param list rejection
  from `Optimizer.__init__`, FSDP empty model parts
- **Result: same TPS as custom Muon (~4,600)** — Newton-Schulz overhead is
  inherent to the algorithm on XPU, not an implementation issue
- **Streaming data shuffle causes ~1.3 loss variance** — same optimizer gives
  very different loss across runs due to HF streaming data ordering

### Speedrun Competition Final Results (1000 steps, 2 nodes)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | Muon (custom) | **3.557** | 4,556 |
| 2 | AdamW + QK-Norm | **3.569** | 7,178 |
| 3 | Muon + cosine | 3.591 | 4,625 |
| 4 | Mano + QK-Norm | 3.604 | 6,980 |
| 5 | Mano | 3.631 | 7,048 |

### Full Training (10B tokens, 8 nodes, GBS=384)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW | **2.711** | 7,354 |
| 2 | AdamW + QK-Norm | 2.720 | 7,480 |
| 3 | Mano + QK-Norm | 2.854 | 7,346 |
| 4 | Mano | 2.875 | 7,429 |
| 5 | Muon | DNF (compile stuck) | — |

### Key Findings

- **AdamW wins at large batch (GBS=384)** — simpler update more efficient
  per token than manifold optimizers
- **QK-Norm effect diminishes at 10B** — 0.009 loss improvement (vs 0.23
  in 1000-step speedruns). Helps early training but washes out
- **Mano ~0.16 behind AdamW at GBS=384** — LR finder was tuned at GBS=48,
  needs re-tuning for larger batch
- **Muon compile broken with GAS** — inductor can't pickle cyclic objects
  in Newton-Schulz with gradient accumulation on torch 2.13
- **8-node scaling excellent** — 7,300-7,500 TPS/GPU across all configs

### CLAUDE.md Added

- Created `experiments/ezpz/.claude/CLAUDE.md` with project rules that
  travel with the codebase (upstream sync protocol, never modify outside
  ezpz, document every run, etc.)

---

## 2026-04-26 — RL refactor, docs reorg, competition launch

### RL Multi-Task Support

- Refactored `rl/` from hardcoded sum-of-digits to a pluggable task registry
- Created `rl/tasks/` package with `RLTask` dataclass, `register_task()`, `get_task()`
- Moved sum_digits dataset+rewards into `tasks/sum_digits.py` (self-registering)
- Added 3 new tasks: `multiply`, `word_sort`, `countdown`
- Added CLI args (`--task`, `--model-name-or-path`, `--steps`, etc.) to `train_grpo.py`
- Default model: `argonne_private/AuroraGPT-7B` with `Qwen/Qwen3-0.6B` fallback
- Fixed safetensors E2BIG crash by disabling mid-training checkpoints
- Moved RL docs to `docs/rl/README.md`

### Docs Reorganization

- Created `configs/` — moved dense-configs.md, moe-configs.md
- Created `guides/` — moved known-issues.md, running-with-newer-pytorch.md, xpu-attention-issues.md
- Renamed `production-training/` → `production/`
- Created `scaling/` — consolidated scaling-study.md, scaling-study-torch213.md,
  benchmark-80B.md, benchmarks.md into per-model pages (agpt-2b, agpt-20b, agpt-80b, moe)
- Rewrote top-level README.md with organized sections
- Fixed all 31 internal cross-references; link checker passes with 0 broken

### Generic HF Dataset Streaming

- Created `datasets.py` with `register_hf_dataset()` for explicit registration
- Added auto-fallback: unknown `--dataloader.dataset` names are treated as HF hub paths
  (e.g. `--dataloader.dataset stanfordnlp/imdb` just works)
- Pre-registered: fineweb_edu, fineweb, slimpajama, pile, openwebtext, wikitext, c4_streaming
- Silenced httpx/huggingface_hub HTTP log spam

### agpt_2b Loss Competition

**Goal:** lowest loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).
**Fixed:** FineWeb-Edu streaming, LBS=2, seq_len=8192, 1000 steps.
**W&B:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

#### New Optimizers Implemented

- **Mano** (`optimizer/mano.py`) — manifold-normalized optimizer
  ([arxiv 2601.23000](https://arxiv.org/abs/2601.23000)).
  Tangent-space projection on rotating Oblique manifold. Vector-norm ops
  instead of Newton-Schulz → runs at AdamW speed (~7,200 TPS/GPU).
- **SPAM** (`optimizer/spam.py`) — spike-aware Adam with momentum reset
  ([arxiv 2501.06842](https://arxiv.org/abs/2501.06842)).
  Gradient spike detection via EMA + periodic moment reset every DeltaT steps.

#### Architecture Tweaks

- **QK-Norm** — added `qk_norm` parameter to `_build_agpt_layers` and
  `_build_agpt_config`. New `2B_qknorm` model variant. RMSNorm on Q,K
  before attention dot product.

#### Competition Results

| Rank | Config | Optimizer | LR | Loss | Steps | TPS/GPU |
|------|--------|-----------|------|------|-------|---------|
| 1 | `speedrun_2b_muon` | Muon | 2.4e-3 | **3.628*** | 967 | 4,695 |
| 2 | `speedrun_2b_mano` | **Mano** | 3.0e-4 | **3.631** | 1000 | ~7,200 |
| 3 | `speedrun_2b_adamw_cosine` | AdamW | 1.3e-3 | 3.789 | 990 | 7,245 |
| 4 | `speedrun_2b_adamw` | AdamW | 1.3e-3 | 3.801 | 1000 | 7,245 |
| 5 | `speedrun_2b_adamw_short_decay` | AdamW | 1.3e-3 | 4.053 | 1000 | 7,245 |
| 6 | `speedrun_2b_adamw_fast_warmup` | AdamW | 1.3e-3 | 4.546 | 1000 | 7,245 |
| 7 | `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | 4.399* | 976 | 4,596 |
| 8 | `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | 4.719 | 1000 | 7,208 |
| 9 | `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | 5.850 | 1000 | 7,344 |
| 10 | `speedrun_2b_spam` | SPAM | 1.3e-3 | 5.881* | 865 | ~7,200 |

*Still running at time of reporting.

#### Key Findings

- **Muon and Mano essentially tied on loss** (~3.63), but Mano ran at
  full AdamW speed (7,200 TPS) vs Muon's 4,700 TPS. **Mano wins on
  wall-clock time.**
- **Muon is 35% slower per step** due to 5x Newton-Schulz iterations
  (large matmuls on every 2D param). Mano replaces these with O(dim)
  vector-norm ops.
- **Cosine decay beats linear** for AdamW (3.789 vs 3.801).
- **Shorter decay (10%) hurts** — not enough time in decay phase.
- **Shorter warmup (5 steps) hurts** — destabilizes early training.
- **SPAM underperforms** — spike clipping + momentum reset don't help
  on this clean dataset with well-tuned LR.
- **AdamW LR=2.6e-3 diverges** — confirms LR finder boundary (1.3e-3).
- **SophiaG underperforms** AdamW by ~0.9 loss at same step count.

#### Failed Experiments

- `speedrun_2b_muon_short_decay` — crashed at startup (exit 143)
- `speedrun_2b_muon_fast_warmup` — crashed at startup (exit 143)
- `speedrun_2b_adamw_qknorm` — crashed at startup (exit 143)
- `speedrun_2b_muon_qknorm` — crashed at startup (exit 143)

Need to investigate QK-Norm and Muon schedule tweak crashes.

#### Issues Hit

- PBS `qsub -- bash -c '...'` doesn't work — need a proper script file
- `set -euo pipefail` kills venv activate scripts (`ZSH_EVAL_CONTEXT: unbound`)
- Concurrent jobs sharing `--checkpoint.folder=checkpoint` clobber each other
  → fixed with per-config checkpoint dirs
- Disk quota hit at 12TB → cleaned 3.2TB of old scaling study checkpoints
  and 117GB of old repo checkpoints
- `git stash pop` during disk quota crunch wiped train.py to 0 bytes
  → restored from `git show HEAD:...`

---

## 2026-04-25 — torch 2.13 venv, scaling study, production scripts

### Torch 2.13 Environment

- Created `.venv/` with PyTorch 2.13 (built from source for XPU)
- Added `running-with-newer-pytorch.md` guide for setting up the venv
- Added `ezpz yeet-env` integration to copy venv to `/tmp` on compute nodes

### Production Training Scripts

- Created `scripts/train_agpt_2b_venv.sh` and `train_agpt_20b_venv.sh`
  for training with the torch 2.13 venv
- Fixed `ezpz_setup_job` ordering — must run before venv activation
- Fixed `/tmp/.venv/bin` PATH handling after `yeet-env activate`
- Set `local_batch_size=2` as default for 2B training

### 2B Scaling Study (torch 2.13, Sunspot)

- Ran weak scaling study from 2 to 64 nodes on Sunspot
- Results: 7,142 TPS/GPU at 2N (27.6% MFU) — **+23% over torch 2.10**
- Near-perfect scaling to 8 nodes (~100%), 94% efficiency at 64 nodes
- Memory nearly constant at ~44 GiB across all scales
- Documented in `docs/scaling-study-torch213.md`

### Production Training Status

- Updated production run tracking for 2B/20B/80B models
- Added per-model subdirectories with loss curve plots
- Updated upstream sync log with session findings

### Upstream Sync

- Merged upstream `pytorch/torchtitan` main into ezpz branch
- Reverted `.ezpz-interactive-launch.sh` tracking change
- Added interactive launch script and loss CSVs

---

## 2026-04-23 — XPU fixes, upstream merge

### XCCL Barrier Fix

- Fixed torch 2.10 XCCL hangs for barrier and TP collectives
- Root cause: XCCL backend doesn't support barrier() — was hanging
  all multi-node runs
- Fix: use gloo backend for barriers when available

### DTensor TP Revert

- Reverted full DTensor TP (`use_local_output=False`) for agpt models
- Was causing shape mismatches in the attention layer on XPU
- Reverted to standard `use_local_output=True`

### Upstream Merge

- Merged upstream main into ezpz branch
- Upstream changes included GraphTrainer bucketing fixes,
  SAC + FSDP improvements, and Qwen3-VL fused QKV support
