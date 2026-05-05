# ezpz Experiment Guide

## Golden Rules

1. **Never modify code outside `experiments/ezpz/`.** All changes stay here.
   If upstream core doesn't support something, work around it in ezpz.

2. **Every experiment must be tracked.** Any job submitted or run executed
   must have a clear purpose and be documented in an appropriate markdown
   file under `docs/`. Link reports from parent READMEs. End-of-session,
   update `docs/journal.md` with what happened.

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

8. **Never kill running jobs** without explicit user confirmation.

9. **Never discard local changes** — always stash-pull-pop, never
   reset/checkout to throw away work.

10. **Terse responses** — don't summarize what you just did at the end
    of every reply. The user can read the diff.

## Hardware & Platform

- **Machines:** Sunspot (Intel Max 1550 XPU), Aurora (Intel Max 1550 XPU), Polaris (A100)
- **Scheduler:** PBS (`qsub`, `qstat`, `qdel`)
- **XPU limitations:** No flash attention, no Triton backend, selective AC may not work.
  Always check XPU compatibility before suggesting optimization strategies.
- **torch.optim.Muon** is available since PyTorch 2.9 — prefer it over the custom
  Newton-Schulz implementation. The custom `optimizer/muon.py` is 35% slower.
- **HSDP (`dp_replicate × dp_shard > 1`) is untested for ezpz models** as of
  2026-05-04 — see "Recent Findings" below for the `aten.normal_.default`
  failure mode. Stick with pure FSDP (`dp_replicate=1`) until that's fixed.

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
│   ├── competitions/        # Competition leaderboard + tracking
│   ├── configs/             # Model config docs
│   ├── evals/               # lm-eval results per model (v1 vs v2)
│   ├── experiments/         # Per-machine smoke / benchmark / LR-finder reports
│   ├── guides/              # Known issues, pytorch setup, XPU attention,
│   │                        # bf16 norm freeze, TP loss-reporting bug
│   ├── meeting-notes/       # AuroraGPT sync agendas + action items
│   ├── production/          # Live production training tracking (per-model, per-N)
│   ├── scaling/             # Per-model scaling study results
│   ├── summaries/           # Periodic 2-week / monthly retrospectives
│   └── upstream-issues/     # Upstream PR drafts + repro scripts
├── moe/                     # MoE model (DeepSeek-style) + parallelize
├── optimizer/               # Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
├── rl/                      # GRPO reinforcement learning (task registry)
├── scripts/                 # Training, benchmark, LR-finder scripts
├── submit/aurora/           # Production submission scripts (torch 2.10 conda)
├── trainer.py               # FaultTolerantTrainer (overrides Trainer)
├── validator.py             # EzpzValidator (overrides Validator with TP loss fix)
├── blendcorpus/             # Blendcorpus dataloader (with serve_validation flag)
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

## Competitions

**Tracking:** `docs/competitions/`
**W&B:** https://api.wandb.ai/links/aurora_gpt/hda3milo

| Competition | Winner | Loss |
|-------------|--------|------|
| 1000 steps, 2N, GBS=48 | Muon 3.557 / AdamW+QK-Norm 3.569 | 3.557 |
| 10B tokens, 8N, GBS=384 | AdamW 2.711 | 2.711 |
| 1000 steps, 2N, GAS=8, GBS=384 | AdamW+QK-Norm 3.205 | 3.205 |

**Key pattern:** Mano/Muon win short runs, AdamW wins in cosine decay phase.

## Recent Findings (2026-04-29 → 2026-05-04)

Big-ticket items from the last week. Each links to the canonical writeup
under `docs/`. Always check the relevant guide before suggesting work
that touches one of these areas.

- **bf16-master RMSNorm-freeze (2026-04-29).** All v1 production runs
  (2B / 20B / 80B) had `training.dtype = bfloat16`, keeping the master
  param copy in bf16. RMSNorm.weight starts at 1.0; the bf16 ULP at 1.0
  is ~7.8e-3 and per-step optimizer updates are ~1.6e-5 — every update
  rounds to zero. **All v1 RMSNorm.weights stayed at 1.0 for the whole
  run.** Fixed by flipping the default to `float32`. v2 production was
  restarted from scratch (path 2). Smoking gun: 2B v2 ARC-Easy hit 0.429
  at 100B tokens; v1 hovered at 0.272 across 450B tokens (+19.8pp
  ARC-Easy / +15.4pp HellaSwag). See
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../docs/guides/training-dtype-bf16-norm-freeze.md).

- **TP > 1 loss reporting is off by `dp_world_size`** (since upstream
  2026-04-27, commit `1786292d`). `_dist_reduce` short-circuits DTensor
  inputs with `full_tensor()` and skips the requested mesh `all_reduce`
  when the meshes are orthogonal. Loss is a Replicated DTensor on the
  TP mesh; reductions are requested across `loss_mesh = batch × cp`.
  Result: reported loss = `true / dp_world_size`. Gradients/optimizer
  steps unaffected. Fix filed as
  [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).
  Local workaround: `loss.full_tensor()` before `dist_sum` in
  `trainer.py`, plus a new `EzpzValidator(Validator)` subclass in
  `validator.py` that does the same in `validate()`. **No current
  production runs use TP > 1**, so no live dashboard is wrong — but
  historical 80B v1 W&B traces show `loss / 1536`. See
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../docs/guides/loss-reporting-tp-dist-reduce.md).

- **80B compile + AC + TP=2 still crashes** with
  `tensors_saved_for_backwards_with_vc_check_slice` AOT autograd
  assertion (DeviceMesh leaks into saved-for-backward tensors). Toy
  repro using legacy `parallelize_module` does NOT fire — needs the
  new `Module.parallelize` + `LocalMapConfig` path. Added
  `agpt_50b_wide` (dim=9216, **48 layers**, ~48B params) as a smaller
  bisect target. **Bug does NOT reproduce at 48 layers, only at 84
  (the 80B config) — so the bug is depth-sensitive, not
  width/head-sensitive.** Workaround: `compile=OFF` for 80B. Toy repro:
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../docs/upstream-issues/repro_devicemesh_in_saved_tensors.py).

- **HSDP (`dp_replicate × dp_shard > 1`) hits an `aten.normal_.default`
  failure during init** (2026-05-04). Param init goes through
  `nn.init.trunc_normal_(param)` → `tensor.normal_(mean, std)` in-place,
  but DTensor sharding-prop has no strategy for in-place `normal_` on
  HSDP-style 2D meshes with mixed Replicate/Shard placements:
  `RuntimeError: aten.normal_.default: in-place operations that require
  placement changes are not supported`. Workaround until the
  materialize-then-init-then-redistribute fix lands: just use pure FSDP
  (`--data-parallel-shard-degree=<world_size>` and
  `--data-parallel-replicate-degree=1`).

- **Recurring `signal 9` Aurora NODE_FAIL pattern.** Three production
  jobs (8459818, 8460301, 8463659) killed mid-run with
  `shepherd died from signal 9` on three different nodes. step-N
  ckpts saved cleanly so trajectories are resumable. Open question
  whether to file an ALCF support ticket — see
  [`docs/meeting-notes/agpt-sync.md`](../docs/meeting-notes/agpt-sync.md).

- **`EzpzValidator` + blendcorpus validation split wired in but not
  smoke-tested yet.** `BlendCorpusDataLoader.Config` has
  `serve_validation: bool` and `eval_iters: int`; `_base_config`
  builds the validator with a blendcorpus loader pointed at the
  validation slice. Default `enable=False` so production isn't
  disturbed. End-to-end smoke with `--validator.enable` is pending.

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
- **HSDP with `dp_replicate > 1` will crash at init** — see "Recent
  Findings" above. Use pure FSDP.
- **Don't trust `loss:` from TP > 1 runs** without the `EzpzValidator`/
  `trainer.py` workaround in place — multiply by `dp_world_size` to
  recover the true value. See "Recent Findings" above.
- **`--dataloader.dataset=user/repo` (HF hub path) silently overrides
  `--dataloader.dataset-path`.** A warning is now emitted; if you want
  a local file dataset use `--dataloader.dataset blendcorpus
  --dataloader.dataset-path <txt-file>` instead.

## Production Training Status (Aurora)

Tracking in `docs/production/`. **All v2 (post-bf16-fix) runs use
torch 2.10 + `submit/aurora/` scripts**. Default dtype is now `float32`
(see Recent Findings).

### v2 — 2B 512N canonical chain (`8460301 → 8463626 → 8463627`)

- **Cumulative steps:** 5,073 (as of 2026-05-04)
- **Loss:** 2.97
- **Tokens consumed:** 510B (10.9% of 4.67T target)
- **Throughput:** ~2,700 TPS/GPU, ~10% MFU
- **Checkpoint dir:** `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/`
- **Trajectory page:** [`docs/production/agpt/2b/n512/`](../docs/production/agpt/2b/n512/README.md)

### v2 — 20B 512N canonical chain (`8460302 → 8463628`)

- **Cumulative steps:** ~862 (as of 2026-05-04)
- **Loss:** ~3.47
- **Tokens consumed:** ~87B (~1.9% of 4.67T target)
- **Throughput:** ~355 TPS/GPU, ~17.8% MFU
- **Trajectory page:** [`docs/production/agpt/20b/n512/`](../docs/production/agpt/20b/n512/README.md)

### v2 — 20B 256N (`8463659`, NODE_FAIL after step 364)

- step 364, loss 4.61, 18.3B tokens. Killed by recurring `signal 9`
  Aurora NODE_FAIL — same pattern as 8459818 / 8460301.
- step-300 ckpt saved cleanly; resumable.
- **Trajectory page:** [`docs/production/agpt/20b/n256/`](../docs/production/agpt/20b/n256/README.md)

### v2 — 80B

Not yet restarted post-bf16-fix. Open work; needs LR=1e-6 and either
(a) `compile=OFF` to avoid the depth-sensitive AOT autograd crash or
(b) a smaller variant like `agpt_50b_wide` (works compile + AC + TP=2).

### v1 (bf16-tainted, historical)

All v1 trajectories are kept under each per-trajectory page in
`docs/production/agpt/{2b,20b,80b}/n*/` for v1-vs-v2 comparison
purposes. Don't add tokens to v1 chains — they're frozen reference
points.

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

Active issues with workarounds in place. For the full diagnosis +
empirical evidence, follow the doc link.

- **TP > 1 loss reporting off by `dp_world_size`** (since upstream
  2026-04-27). Fix filed as `pytorch/torchtitan#3204`. Local workaround
  in `trainer.py` + `validator.py`. See
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../docs/guides/loss-reporting-tp-dist-reduce.md).

- **HSDP init crashes** with `aten.normal_.default: in-place operations
  that require placement changes are not supported`. Workaround: pure
  FSDP only.

- **80B compile + AC + TP=2 crashes** with
  `tensors_saved_for_backwards_with_vc_check_slice` AOT autograd
  assertion. Depth-sensitive — does NOT reproduce at 48 layers
  (`agpt_50b_wide`), only at 84 (the 80B config). Workaround:
  `compile=OFF` for 80B. Toy repro at
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../docs/upstream-issues/repro_devicemesh_in_saved_tensors.py).

- **80B TP=2 regression on torch 2.10:** Hangs at step 1 since upstream
  changes April 16-23. Works on torch 2.13. See `project_80b_bisect`
  memory.

- **MoE SIGABRT:** MOE models crash after upstream `00b7f569` refactor —
  `edp_mesh=None` when EP disabled. See `project_moe_sigabrt` memory.

- **TorchMuon integration:** `_CompositeOptimizer` wrapper needed for
  `OptimizersContainer`'s one-optimizer-per-model-part constraint.

- **IPEX import steals 60 MiB/rank:** Removing
  `import intel_extension_for_pytorch` fixed the 80B TP=2 OOM
  regression. Don't re-add it.

- **tyro Callable crash:** `trace_post_processor: Function.Config | None`
  in Profiler.Config. Fixed with `tyro.conf.Suppress`. PR
  pytorch/torchtitan#3038.

- **User venv transformers conflict:** The user venv at
  `venvs/aurora/torchtitan-ezpz-aurora_frameworks-2025.3.1/` has
  transformers 5.6.2 which breaks lm-eval's HF backend. Use bare
  `module load frameworks` for evals.

- **Stale checkpoint resume crash:** Trying to resume from a checkpoint
  built with a different parallelism config crashes silently. Fix:
  rename old checkpoint dir to `.bak-YYYYMMDD`.

- **Recurring Aurora `signal 9` NODE_FAIL** on long-walltime jobs (3
  jobs killed across 3 different nodes). Cause unclear; resume from
  most recent ckpt is the operational workaround. Possibly worth an
  ALCF support ticket.

## User Preferences (operational)

These are *preferences* (style/formatting), not rules. Hard rules
moved up to "Golden Rules".

- **Always document experiments** — every run needs a markdown report
  under `docs/experiments/<module>/<machine>/<date>-<purpose>.md`,
  linked from a parent README.
- **Don't modify project-level `.gitignore`.**
- **Don't add `.ezpz-interactive-launch.sh`** to git — use
  `scripts/interactive-launch.sh`.
- **Append-only tables** — production training progress tables should
  append, not replace.
- **Use `submit/aurora/` scripts** for continuing torch 2.10 production
  runs, `scripts/*_venv.sh` for new torch 2.13 runs.
- **Date filenames as `YYYY-MM-DD`** for any per-day artifacts.
  Per-recurring-meeting docs use a stable filename with `## YYYY-MM-DD`
  sections inside (see `docs/meeting-notes/agpt-sync.md`).
- **Cross-link related docs.** Production READMEs link to eval READMEs
  and vice versa; the bf16-norm-freeze guide links to both training
  overlays and lm-eval figures.
