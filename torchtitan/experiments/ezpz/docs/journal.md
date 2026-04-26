# Development Journal

Running log of what's happening, session by session. Most recent first.

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

### agpt_2b Loss Competition (started)

**Goal:** lowest loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).
**Fixed:** FineWeb-Edu streaming, LBS=2, seq_len=8192, 1000 steps.
**W&B:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

#### New Optimizers Implemented

- **Mano** (`optimizer/mano.py`) — manifold-normalized optimizer
  ([arxiv 2601.23000](https://arxiv.org/abs/2601.23000)).
  Tangent-space projection on rotating Oblique manifold. Vector-norm ops
  instead of Newton-Schulz → should be ~1.75x faster than Muon wall-clock.
- **SPAM** (`optimizer/spam.py`) — spike-aware Adam with momentum reset
  ([arxiv 2501.06842](https://arxiv.org/abs/2501.06842)).
  Gradient spike detection via EMA + periodic moment reset every DeltaT steps.

#### Architecture Tweaks

- **QK-Norm** — added `qk_norm` parameter to `_build_agpt_layers` and
  `_build_agpt_config`. New `2B_qknorm` model variant. RMSNorm on Q,K
  before attention dot product.

#### Results So Far

| Config | Optimizer | LR | Loss (step 1000) | TPS/GPU |
|--------|-----------|------|------------------|---------|
| `speedrun_2b_adamw` | AdamW | 1.3e-3 | **3.801** | 7,245 |
| `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | 4.719 | 7,208 |
| `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | 5.850 | 7,344 |
| `speedrun_2b_muon` | Muon | 2.4e-3 | 3.915 (step 785)* | 4,695 |

*Muon timed out at 1h — resubmitted with 3h walltime, currently running.

#### Findings

- **Muon is 35% slower per step** than AdamW due to 5x Newton-Schulz
  iterations (large matmuls on every 2D param). 4,700 vs 7,200 TPS/GPU.
- **AdamW LR=2.6e-3 diverges** — confirms LR finder boundary (1.3e-3).
- **SophiaG underperforms** AdamW by ~0.9 loss at same step count.

#### Jobs In Flight

- 2 Muon reruns (3h walltime)
- 5 hparam tweaks (short decay, cosine, fast warmup variants)
- 2 new optimizers (Mano, SPAM)
- 2 QK-Norm variants (AdamW + QK-Norm, Muon + QK-Norm)

#### Issues Hit

- PBS `qsub -- bash -c '...'` doesn't work — need a proper script file
- `set -euo pipefail` kills venv activate scripts (`ZSH_EVAL_CONTEXT: unbound`)
- Concurrent jobs sharing `--checkpoint.folder=checkpoint` clobber each other
  → fixed with per-config checkpoint dirs
- Disk quota hit at 12TB → cleaned 3.2TB of old scaling study checkpoints
  and 117GB of old repo checkpoints
