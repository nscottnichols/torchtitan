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
- **torch 2.10 conda env:** `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env`
  Loads `frameworks/2025.3.1` module + user venv overlay. Has lm_eval, vllm, transformers.
- **PBS scripts must NOT use `set -euo pipefail`** — venv activate has unbound vars
- **Compute nodes need proxy:** `export http_proxy=http://proxy.alcf.anl.gov:3128`
  (also `https_proxy`, `ftp_proxy`). Required before any `curl`, `pip`, or HF download.

## Aurora-Specific

### Queue Structure

| Queue | Nodes | Walltime | Notes |
|-------|-------|----------|-------|
| debug | 1–2 | 1h | Fast turnaround |
| debug-scaling | 1–256 | 1h | Best for small-scale tests |
| capacity | 1–16 | 168h | Long-running, good for evals |
| prod → small | 256–1024 | 12h | Production training |

### Job Submission

- **Production training (torch 2.10):** Use `submit/aurora/submit_agpt_{2b,20b}.sh`
  These use `ezpz_setup_env` (conda). LBS=1, 256 nodes.
- **Production training (torch 2.13):** Use `scripts/train_agpt_{2b,20b,80b}_venv.sh`
  These use `.venv/` + `ezpz yeet-env`. LBS=2.
- **Never submit multiple 512N venv jobs simultaneously.** The `yeet-env` rsync
  of 8.6GB to 512+ nodes saturates the flare filesystem for hours, killing
  TPS on all co-running jobs. Stagger or use DAOS instead.
- Always chain continuations: `qsub -W depend=afterany:<jobid> <script>`

### yeet-env Scaling

Copies `.venv/` (8.6GB) from flare to `/tmp` on every compute node.
- 2–16 nodes: ~5–10 min
- 64 nodes: ~15–30 min
- 256 nodes: ~30–60 min (solo), hours if concurrent
- 512+ nodes: 1–2+ hours (solo), DO NOT run concurrently

### Optimizer Constraints at Scale

- **SophiaG/Muon broken at 80B:** bf16 overflow in Hessian/Newton-Schulz
  at dim=9216. Use AdamW only for 80B.
- **80B AdamW LR=1.1e-5 → NaN** at production GBS (1536–3072). Use LR=1e-6.
- **torch.compile OOM at 512N:** 2B OOMs on GPU, 80B OOMs on CPU.
  Use `--compile.no-enable` for 512N jobs.
- **torch.compile time:** ~7–15 min at 256N depending on model size.
  4+ hours at TP=4 for 80B.

### Evaluation Pipeline

- **Checkpoint conversion:** `eval/convert_to_hf.py` converts DCP → HF safetensors.
  Use `--model_name experiments.ezpz.agpt --model_flavor 2b`.
  2B takes ~4 min, 20B takes ~20–30 min.
- **HF config:** Must copy `eval/configs/agpt_{2b,20b}_config.json` + tokenizer
  files from `assets/hf/gemma-7b/` into the HF checkpoint dir.
- **lm-eval:** Use bare `module load frameworks/2025.3.1` (NOT the user venv).
  The user venv has transformers 5.6.2 which breaks lm-eval's HF backend
  with `TypeError: LlamaForCausalLM.__init__() got unexpected kwarg 'dtype'`.
  Must also set `HF_HUB_ENABLE_HF_TRANSFER=0`.
- **Device:** `--device xpu` for lm-eval HF backend. 2B fits on 1 tile.
  20B (~40GB bf16) may need multi-tile.
- **Tokenizer:** google/gemma-7b, vocab_size=256128, bos=2, eos=1.

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

## Production Training Status (Aurora, 256N)

Tracking in `docs/production/`. Uses `submit/aurora/` scripts (torch 2.10).

### 2B @ 256N — SophiaG LR=2.28e-5
- **Cumulative steps:** 26,000+ (as of 2026-04-27)
- **Loss:** 5.69 (still converging)
- **Tokens consumed:** ~655B (14% of 4.67T target)
- **Throughput:** ~2,500 TPS/GPU, 9–10% MFU with compile
- **Checkpoint dir:** `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072/`
- **Checkpoint interval:** every 100 steps, keep_latest_k=0 (all kept)

### 20B @ 256N — SophiaG LR=2.28e-5
- **Cumulative steps:** 3,300+ (as of 2026-04-27)
- **Loss:** 4.68 (still converging)
- **Tokens consumed:** ~83B (1.8% of 4.67T target)
- **Throughput:** ~285 TPS/GPU, 14% MFU with compile
- **Checkpoint dir:** `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs3072/`

### 80B — Status
- AdamW LR=1.1e-5: NaN at step 138 (256N) and step 15 (512N)
- AdamW LR=1e-6: Stable for 51 steps, then crashed (bad node Gloo timeout)
- SophiaG/Muon: NaN immediately (bf16 overflow at dim=9216)
- Needs resubmit with LR=1e-6

## Scaling Study Results (Aurora)

### 2B Scaling (torch 2.10, compiled)
| Nodes | TPS/GPU | MFU |
|-------|---------|-----|
| 2 | 5,500 | 22% |
| 4 | 5,380 | 20% |
| 16 | 5,553 | 21% |
| 32 | 4,500 | 18% |
| 128 | 3,950 | 15% |
| 256 | 2,500 | 9.5% |

### 2B Scaling (torch 2.13, compiled, Sunspot)
| Nodes | TPS/GPU | MFU |
|-------|---------|-----|
| 2 | 7,142 | 27.6% |
| 4 | 7,068 | 27.3% |
| 16 | 6,995 | 27.0% |
| 64 | 6,702 | 25.9% |

## Known Bugs / Open Issues

- **80B TP=2 regression on torch 2.10:** Hangs at step 1 since upstream changes
  April 16-23. Works on torch 2.13. See `project_80b_bisect` memory.
- **MoE SIGABRT:** MOE models crash after upstream `00b7f569` refactor —
  `edp_mesh=None` when EP disabled. See `project_moe_sigabrt` memory.
- **TorchMuon integration:** `_CompositeOptimizer` wrapper needed for
  `OptimizersContainer`'s one-optimizer-per-model-part constraint. Still
  hitting HF rate limit errors on multi-job submissions.
- **IPEX import steals 60 MiB/rank:** Removing `import intel_extension_for_pytorch`
  fixed the 80B TP=2 OOM regression. Don't re-add it.
- **tyro Callable crash:** `trace_post_processor: Function.Config | None` in
  Profiler.Config. Fixed with `tyro.conf.Suppress`. PR pytorch/torchtitan#3038.
- **User venv transformers conflict:** The user venv at
  `venvs/aurora/torchtitan-ezpz-aurora_frameworks-2025.3.1/` has transformers
  5.6.2 which breaks lm-eval's HF backend. Use bare `module load frameworks`
  for evals.
- **Stale checkpoint resume crash:** 20B 512N tried loading incompatible
  checkpoint from a different parallelism config. Fix: rename old checkpoint
  dir to `.bak-YYYYMMDD`.

## User Preferences (Aurora sessions)

- **Never kill running jobs** without explicit user confirmation.
- **Never discard local changes** — always stash-pull-pop, never reset/checkout.
- **Always document experiments** — every run needs a markdown report.
- **Don't modify project-level .gitignore.**
- **Don't add `.ezpz-interactive-launch.sh`** to git — use `scripts/interactive-launch.sh`.
- **Terse responses** — don't summarize what you just did at the end.
- **Append-only tables** — production training progress tables should append, not replace.
- **Use `submit/aurora/` scripts** for continuing torch 2.10 production runs,
  `scripts/*_venv.sh` for new torch 2.13 runs.
