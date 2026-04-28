# Development Journal

Running log of what's happening, session by session. Most recent first.

---

## 2026-04-28 — 21st upstream sync replay (sharding API, ChunkedCELoss)

### What landed

The 21st upstream sync (merged `b6c04698`) brought in three breaking
changes that broke ezpz at import / config-build / training-init time:

- **#2963 / #2969** — config-based DTensor sharding. Replaces string-keyed
  `parallelize_module(plan)` with `Module.parallelize(mesh)` reading
  `ShardingConfig` declarations attached to each sub-module's `.Config`.
- **#2937** — ChunkedCELoss. Removed `build_cross_entropy_loss` and
  `ModelSpec.build_loss_fn`; loss now lives on `JobConfig.loss`.
- **`Decoder.Config`** renamed `output: Linear.Config` → `lm_head:
  Linear.Config`.

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.96 → 7.09 (job 12465500) |
| `moe`  | replayed | 50 steps, loss 12.93 → 6.91 (job 12465502) |
| `qwen3` | removed | Drift too large; nobody ran it; restorable from history |

### Commits (ezpz branch)

- `03b9f486` — Mechanical: drop `build_cross_entropy_loss` imports +
  `build_loss_fn=` kwargs, rename `output=` → `lm_head=` in agpt+moe
  configs, switch `ezpz/trainer.py` to `config.loss.build()`.
- `472f4743` — agpt sharding-API replay. New `agpt/sharding.py`
  (handles QK-Norm), new `agpt/model.py` (`AgptModel(Llama3Model)`
  overriding `update_from_config`), rewritten `agpt/parallelize.py` as
  thin orchestrator. Float8 tensorwise TP path dropped (no equivalent in
  the new API yet).
- `9bc774a6` — Set `loss=CrossEntropyLoss.Config()` in both `_base_config`
  helpers (was defaulting to abstract `BaseLoss.Config`).
- `40526628` — Enable compile in `smoke_2b_50steps` (XPU CE OOMs without it
  at vocab=256k).
- `dd4e065a` — moe sharding-API replay. New `moe/sharding.py`, extended
  `update_from_config`, rewritten `parallelize.py` (drops 230+ lines of
  manual ColwiseParallel/RowwiseParallel plans). MoE block sharding still
  done at parallelize-time by `apply_moe_ep_tp` (mirrors upstream).
- `5f88abc3` — `smoke_moe_500m_50steps` config + submit script.
- `80a23d41` — Removed `ezpz/qwen3` (drift too large for unused code).

### Preserved agpt-/moe-specific behavior

- `disable_fsdp_gradient_division` still calls
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL/XPU).
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False (keeps the separately-compiled CrossEntropyLoss working on dense
  models).
- agpt `apply_fsdp` keeps the `[norm, lm_head]` joint grouping with
  reshard_after_forward gated on the policy.
- moe `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
  import which doesn't exist in Aurora's PyTorch) with the Shard(0)
  fallback when expert hidden dim isn't FSDP-divisible.
- moe `apply_compile` is per-block `block.compile(backend=...)` instead of
  upstream's fullgraph `apply_compile_sparse` (XPU can't fullgraph compile
  MoE routing's dynamic shapes).

### Smoke-test details

- agpt smoke (`12465500`): loss 12.96 → 7.09 across 50 steps, ~7,400 TPS,
  27% MFU. Standard dense-2B numbers — replay is loss-neutral.
- moe smoke (`12465502`): loss 12.93 → 6.91 across 50 steps, ~7,200 TPS,
  ~10% MFU. The MFU is low because the metrics divisor uses the full
  dense FLOP estimate but only 2/8 experts fire per token — reporting
  artifact, not a perf regression.

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

### Architecture Tweaks Implemented

- **Logit softcapping** — `SoftcappedFlexAttention` using FlexAttention
  `score_mod` with tanh cap at 30.0. Falls back to eager on XPU (4x slower).
  Manual attention OOMs at seq_len=8192 (materializes full attention matrix).
- **ReLU²** — `ReLUSquaredFeedForward` subclass. Didn't help (3.92 vs 3.80
  baseline). SiLU gating is better for this architecture.
- **WSM** — `eval/merge_checkpoints.py` utility for weighted state merging
  of checkpoints. Supports uniform, linear, and exponential weighting.
- New model variants: `2B_softcap`, `2B_relu2`, `2B_kitchen_sink`

### Round 4: 2N, GAS=8, 1000 steps (local dataset)

Reproducible speedrun with GBS=384 on 2 nodes using local FineWeb-Edu.

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW+QK-Norm | **3.205** | 7,428 |
| 2 | AdamW | 3.220 | 7,397 |
| 3 | Mano | 3.294 | 7,397 |
| 4 | Mano+QK-Norm | 3.307 | 7,423 |
| 5 | Mano (8.5e-4) | 3.328 | 7,348 |
| 6 | AdamW (3.7e-3) | 5.884 | 7,603 |

**Key findings:**
- AdamW+QK-Norm wins again — consistent across all GBS=384 experiments
- Mano leads early/mid training but AdamW catches up in cosine decay phase
- sqrt LR scaling too aggressive for AdamW (diverged), Mano tolerated it
- Softcap results invalid — local dataset loader memorizes with FlexAttention
  path (data sharding bug)
- FlexAttention on XPU falls back to eager (Triton-XPU can't codegen tanh)
  — 4x throughput penalty makes softcap impractical on this hardware

### Docs Restructure

- Reorganized `docs/competition/` → `docs/competitions/` with per-experiment dirs
- Added light/dark theme loss curve plots using `<picture>` media queries
- Created `docs/competitions/agpt2b-n2-gas8-1000steps/` with live loss curves

### Upstream Sync (20th)

- Merged upstream: dataset checkpoint resume fix (#3008), RL refactor (#3073)
- Clean merge, no replay needed

### CLAUDE.md Added

- Created `experiments/ezpz/.claude/CLAUDE.md` with project rules that
  travel with the codebase (upstream sync protocol, never modify outside
  ezpz, document every run, etc.)
- Updated with Aurora-specific knowledge (queues, yeet-env scaling, eval pipeline)

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
