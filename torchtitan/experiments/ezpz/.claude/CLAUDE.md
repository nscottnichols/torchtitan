# ezpz Experiment Guide

## Golden Rules

1. **Never modify code outside `experiments/ezpz/`.** All changes stay here.
   If upstream core doesn't support something, work around it in ezpz.

2. **Every experiment must be tracked.** Any job submitted or run executed
   must have a clear purpose and be documented in an appropriate markdown
   file under `docs/`. Link reports from parent READMEs.

3. **Upstream sync protocol:**
   - When pulling changes from upstream `main`, replay any changes to
     `llama3/` onto `ezpz/agpt/` and `deepseek_v3/` onto `ezpz/moe/`.
   - Always update `docs/upstream-sync.md` with what changed and what
     was replayed.

4. **Use `backup` instead of `rm`.** The `~/.local/bin/backup` command
   renames with a timestamp instead of deleting.

5. **Never `pip install` torch or its deps.** Can silently replace the
   Intel XPU build with CUDA. Use `--no-deps --no-cache --link-mode=copy`
   for any package that touches torch. Verify with
   `python3 -c "import torch; print(torch.xpu.is_available())"` after.

6. **Never run pre-commit hooks** as a verification step. User handles
   linting separately.

7. **Commit style:** Break changes into logically grouped commits with
   descriptive messages. Never batch unrelated changes.

## Hardware & Platform

- **Machines:** Sunspot (Intel Max 1550 XPU), Aurora (Intel Max 1550 XPU), Polaris (A100)
- **Scheduler:** PBS (`qsub`, `qstat`, `qdel`)
- **XPU limitations:** No flash attention, no Triton backend, selective AC may not work.
  Always check XPU compatibility before suggesting optimization strategies.
- **torch.optim.Muon** is available since PyTorch 2.9 — prefer it over the custom
  Newton-Schulz implementation. The custom `optimizer/muon.py` is 35% slower.

## Environment Setup

- **torch 2.13 venv:** `.venv/` in repo root, copy to compute with `ezpz yeet-env`
- **PBS scripts must NOT use `set -euo pipefail`** — venv activate has unbound vars

## Key Paths

```
torchtitan/experiments/ezpz/
├── agpt/                    # AuroraGPT model configs and registry
├── competition/             # Loss speedrun competition
├── datasets.py              # Generic HF dataset streaming
├── docs/                    # All documentation
│   ├── journal.md           # Day-by-day development log
│   ├── upstream-sync.md     # Upstream merge tracking
│   ├── competition/         # Competition leaderboard + tracking
│   ├── configs/             # Model config docs
│   ├── guides/              # Known issues, pytorch setup, XPU attention
│   ├── scaling/             # Per-model scaling results
│   └── production/          # Live training run tracking
├── optimizer/               # Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
├── rl/                      # GRPO reinforcement learning (task registry)
├── scripts/                 # Training and benchmark scripts
└── train.py                 # Main training entry point
```

## Running Training

```bash
# Interactive (from compute node with allocation)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config ezpz_agpt_2b

# PBS submission (from login node)
qsub -l select=2 -N my_job -v CONFIG=speedrun_2b_muon \
    torchtitan/experiments/ezpz/competition/submit_run.sh
```

## Available Optimizers

| CLI name | Container | Implementation |
|----------|-----------|----------------|
| `adamw` | `OptimizersContainer` | `torch.optim.AdamW` |
| `muon` | `MuonOptimizersContainer` | Custom Newton-Schulz (slow) |
| `torchmuon` | `TorchMuonOptimizersContainer` | `torch.optim.Muon` (fast) |
| `mano` | `ManoOptimizersContainer` | Manifold-normalized (arxiv 2601.23000) |
| `spam` | `SPAMOptimizersContainer` | Spike-aware Adam (arxiv 2501.06842) |
| `sophiag` | `SophiaGOptimizersContainer` | Second-order Hessian |
| `adopt` | `ADOPTOptimizersContainer` | Adaptive clipping |

## HF Dataset Streaming

Any HF dataset works as `--dataloader.dataset <path>`:
```bash
--dataloader.dataset HuggingFaceFW/fineweb-edu
--dataloader.dataset stanfordnlp/imdb
```

Auto-registered at runtime via `datasets.py`. No core changes needed.

**Caution:** Submitting many streaming jobs simultaneously hits HF rate limits
(1000 API requests / 5 min). Stagger submissions or cache locally.

## Competition (agpt_2b speedrun)

**Goal:** Lowest loss in 1000 steps on 2 nodes.
**Fixed:** FineWeb-Edu, LBS=2, seq_len=8192.
**Tracking:** `docs/competition/README.md`
**W&B:** https://api.wandb.ai/links/aurora_gpt/hda3milo

Current leader: Muon (custom) at 3.557, Mano at 3.631.
TorchMuon integration in progress.

## Common Pitfalls

- **Optimizer name casing:** Config objects use `"AdamW"` (capital), CLI uses
  `"adamw"` (lower). Use container Config classes in programmatic configs.
- **QK-Norm RMSNorm:** Needs `param_init=_NORM_INIT`, and lowercase alias
  (`2b_qknorm`) in `agpt_configs`.
- **Checkpoint conflicts:** Concurrent jobs must use different `checkpoint.folder`.
- **FSDP empty model parts:** Custom optimizer containers must handle model parts
  with zero parameters after sharding.
- **HF rate limits:** 24 ranks × multiple jobs = hundreds of API requests.
  Stagger PBS submissions by 5+ minutes.

## Known Bugs / Open Issues

- **80B TP=2 regression on torch 2.10:** Hangs at step 1 since upstream changes
  April 16-23. Works on torch 2.13. See `project_80b_bisect` memory.
- **MoE SIGABRT:** MOE models crash after upstream `00b7f569` refactor —
  `edp_mesh=None` when EP disabled. See `project_moe_sigabrt` memory.
- **TorchMuon integration:** `_CompositeOptimizer` wrapper needed for
  `OptimizersContainer`'s one-optimizer-per-model-part constraint. Still
  hitting HF rate limit errors on multi-job submissions.
