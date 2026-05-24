# Bad-node failover for production training

> Status: **v2 in production** as of 2026-05-23.
>
> v2 (2026-05-13 → 2026-05-23) added: silent-hang detection via
> `ezpz launch --timeout` watchdog (PR `eefccfc9d`), ANSI-aware
> 'Execution finished with N' parsing (`94a8fda66`), unified
> walltime + crash regex (`0d93a1e91`), and a fixture-based test
> harness under [`tests/failover/`](../../tests/failover/) that
> verifies the rc-determination logic against 9 synthetic log
> fixtures (one per known failure mode). Before any future edit to
> `failover_lib.sh`, run `bash tests/failover/run_tests.sh` to
> verify behavior — all 9 fixtures must pass.

## Why this exists

Recurring Aurora bad-node failures have killed at least 6 production
jobs in the past 2 weeks:

| Job ID | Trajectory | Failure mode |
|---|---|---|
| 8459818 | 2B 256N v2 | `shepherd died from signal 9` after step 2070 |
| 8460301 | 2B 512N v2 | `shepherd died from signal 9` after step 1387 |
| 8460302 | 20B 512N v2 | `shepherd died from signal 9` at end of walltime |
| 8463659 | 20B 256N v2 | `shepherd died from signal 9` after step 364 |
| 8470102 | 20B 256N v2 | gloo TCP `Connection closed by peer` after ~3h |
| 8470103 | 20B 256N v2 | gloo TCP timeout after ~3h |
| 8466848 | 20B 512N v2 | `set_determinism` `MemoryError: std::bad_alloc` at startup |
| 8479581 | 20B 256N v2 | gloo TCP timeout during DCP ckpt save at step 500 |
| 8479579 | 20B 512N v2 | **silent hang** at step 803 (no exit, killed manually) |

The pattern: PBS gives us 256/512 nodes, one of them is bad, training
either crashes or hangs after N hours of compute, we lose all
post-checkpoint progress, and the walltime is then up.

This wrapper requests N+spare nodes from PBS, splits the allocation
into an active training subset and a spare pool, and on a bad-node
crash swaps the offending node out for a spare and retries.

## Files

| Path | Purpose |
|---|---|
| [`scripts/failover_lib.sh`](../../scripts/failover_lib.sh) | Bash library: `failover_init`, `failover_yeet_all`, `failover_swap_in`, `failover_swap_one_blind`, `failover_run`. |
| [`scripts/scrape_bad_nodes.py`](../../scripts/scrape_bad_nodes.py) | Extracts bad-node hostnames from a training log. |
| [`scripts/submit_agpt_2b_aurora_venv_failover.sh`](../../scripts/submit_agpt_2b_aurora_venv_failover.sh) | 2B production submit script with failover. |
| [`scripts/submit_agpt_20b_aurora_venv_failover.sh`](../../scripts/submit_agpt_20b_aurora_venv_failover.sh) | 20B production submit script with failover. |
| [`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../scripts/submit_agpt_80b_aurora_venv_failover.sh) | 80B production submit script with failover (AdamW LR=1e-6, TP=2, AC=full, compile=OFF). |
| [`scripts/FAILOVER.md`](../../scripts/FAILOVER.md) | Brief code-adjacent quickref. **This page is the canonical reference.** |

## Usage

Submit with `qsub -l select=N+spare -v NHOSTS_TRAIN=N`:

```bash
# 2B canonical chain at 512 active + 10 spare nodes
qsub -q prod \
    -l select=522 \
    -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=512 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_2b_aurora_venv_failover.sh

# 20B canonical chain at 512 active + 10 spare
qsub -q prod \
    -l select=522 \
    -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=512 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_20b_aurora_venv_failover.sh

# 20B 256N continuation with smaller spare pool (256 active + 4 spare)
qsub -q prod \
    -l select=260 \
    -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=256,FAILOVER_MAX_RETRIES=4 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_20b_aurora_venv_failover.sh

# 80B at 512 active + 10 spare with reduced retry budget (each 80B
# retry pays a 5-15 min model-build/dataloader-init cost — don't
# burn a 12h walltime on retries)
qsub -q prod \
    -l select=522 \
    -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=2 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
```

All other env vars (LBS, GAS, OPTIMIZER, LR, CKPT_DIR, etc.) work
exactly as in the baseline scripts — the failover wrapper is
otherwise transparent.

### How many spares?

Rule of thumb: **~2% spare**, minimum 4. For 512N use 10 spares
(522 total); for 1024N use 20 (1044 total). At 2% the queue penalty
is small and you can survive 3-4 distinct bad-node hits in one
walltime window.

The scheduler treats your job as `select=N+spare`, so wider
allocations queue with the wider class — make sure your account has
the relevant queue/limit headroom before bumping spare counts.

## How it works

### On startup (`failover_init`)

1. Read `PBS_NODEFILE` (which has N+spare lines).
2. Split into `active.hostfile` (first N) + `spare.hostfile` (rest)
   under `$FAILOVER_LOG_DIR` (default `$(pwd)/logs/failover-<JOBID>/`).
3. Override `PBS_NODEFILE` → `active.hostfile` so `ezpz_setup_job`
   and downstream `ezpz launch` only see the training subset. NHOSTS,
   GBS, etc. all auto-derive correctly.
4. Truncate `bad_nodes.txt` (the canonical record of which nodes
   misbehaved during this allocation).

### After yeet (`failover_yeet_all`)

5. Temporarily set `PBS_NODEFILE` → `all.hostfile` (active + spare).
6. Run `ezpz yeet-env --src .venv.tar.gz` — the venv lands on **all**
   nodes including the spare pool, so swapping a spare into the
   active set is a no-op from a filesystem perspective.
7. Restore `PBS_NODEFILE` → `active.hostfile`.

### Per attempt (`failover_run`)

8. Run the training command, redirecting stdout+stderr to
   `$FAILOVER_LOG_DIR/attempt-N.log` (also tee'd to the PBS .o file).
9. On exit, check the return code:
   - **0** → success, return.
   - **143** → SIGTERM / walltime. Not a bad-node failure; do NOT retry.
   - **other** → run `scrape_bad_nodes.py` against the attempt log:
     - If specific bad node(s) identified, `failover_swap_in HOSTS...`
       (`sed`-replace each bad host in `active.hostfile` with the next
       spare popped from `spare.hostfile`; append bad host to
       `bad_nodes.txt`).
     - If no specific bad host found (e.g. `set_determinism`
       `std::bad_alloc` init crash), `failover_swap_one_blind` rotates
       the first active host out for a spare.
10. Sanity-check that the active count still equals `NHOSTS_TRAIN`,
    then loop back to step 8.
11. Up to `FAILOVER_MAX_RETRIES` (default 3) retries; bail if exhausted.

## Detected failure modes

`scrape_bad_nodes.py` is empirically tuned against 5 production crash
logs from late April / early May 2026:

| Pattern | Detection | Production examples |
|---|---|---|
| `<host>: shepherd died from signal 9` | regex match → emit hostname | 8459818, 8460301, 8460302, 8463659 |
| `RuntimeError: ... Connection closed by peer [IP]:port` | regex match → reverse-resolve via `getent hosts` → normalize to `.hsn.cm.aurora.alcf.anl.gov` | 8470102, 8470103, 8479581 |
| `set_determinism` `std::bad_alloc` init crash | not matched — falls back to blind rotation | 8463182, 8463183, 8466848 |

### Why we DON'T match `signal {11,15}`

`rank N died from signal 11` (SIGSEGV) and `rank N died from signal 15`
(SIGTERM) are almost always **cascading** deaths from a primary kill
on a different node. Matching them would falsely tag innocent nodes.
Verified against 8466848: that crash was caused by `std::bad_alloc`
on rank 2413, but 8 other ranks died from signal 11/15 as a
downstream effect — none of them were the actual bad node.

### Hostname normalization

PBS hostfile entries use the `.hsn.cm.aurora.alcf.anl.gov` form.
`getent hosts <ip>` may return multiple aliases (`.hostmgmtNNNN.cm.aurora`,
etc.). The detector always emits the canonical `.hsn.cm.aurora` form
so `failover_swap_in` can match against `active.hostfile` cleanly.

## Limitations

### Silent hangs — handled by `--timeout` watchdog (2026-05-23)

The wrapper now injects `--timeout=$FAILOVER_IDLE_TIMEOUT` (default
1800s = 30min) into `ezpz launch` invocations. If the launched
process emits no output for that long, `ezpz launch` sends SIGTERM
to the inner mpiexec and exits 124. The wrapper treats exit 124 as
a bad-node failure: scrape (typically no specific host since the
hang IS the silence), `failover_swap_one_blind`, retry.

Requires `ezpz >= 0.15.1` in the venv (all 3 v2 production clones
upgraded 2026-05-23).

Override the timeout via env: `FAILOVER_IDLE_TIMEOUT=0` disables
the watchdog; bump to e.g. `3600` for longer legitimate quiet
windows (long ckpt saves, etc.). 1800s was chosen because the
longest observed legitimate quiet period was a 19-min async ckpt
save at 20B 512N — 30 min leaves comfortable headroom.

Reference incident:
[`8479579`](../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
(2026-05-11) trained 3 steps after resume, then stopped logging
for 5 hours in PBS `R` state. That kind of failure now exits 124
within `FAILOVER_IDLE_TIMEOUT` seconds and triggers retry.

### Walltime hits don't retry

Exit code 143 (SIGTERM, which PBS sends on walltime) is treated as
"normal" and not retried. Continue to chain `qsub -W depend=afterany`
continuations as before; the failover wrapper composes correctly
with chained continuations.

### Compile cost amplifies on retry

Each retry re-runs `torch.compile` at start. For 2B/20B at 256N
this is 7-15 min; for 80B at TP=4 it's 4+ hours. Set
`FAILOVER_MAX_RETRIES=1` (or 2) for 80B so a single bad-node hit
doesn't burn the whole walltime on recompiles.

### NaN / genuine bugs

Any non-zero exit triggers retry, but if the bug is deterministic
(NaN, bad config, code bug) the retries will all hit it. The
`FAILOVER_MAX_RETRIES` cap is the circuit breaker. After exhaustion
the job exits with the original failure code, surfacing the bug.

### Async checkpointing helps

Gloo failures during a sync DCP save (which is what killed 8479581
at step 500) abort the save. With async checkpointing
(`--checkpoint.async-mode=async`, the production default since
2026-05-01) the failover retry resumes from the last
**successfully-saved** ckpt, losing < 100 steps. All current
production submit scripts include this flag.

## Postmortem files

After a job runs (success or failure), `$FAILOVER_LOG_DIR` contains:

- `active.hostfile` — final active node set (after any swaps)
- `spare.hostfile` — remaining spares
- `all.hostfile` — full PBS allocation (constant)
- `bad_nodes.txt` — list of nodes that misbehaved during this
  allocation (one per swap event). **This is the canonical record
  of bad nodes — copy to a long-lived bad-node tracking file
  before the job's allocation expires** if you want to feed it to
  an ALCF support ticket.
- `attempt-1.log`, `attempt-2.log`, ... — per-attempt training
  output.

## See also

- [`docs/experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md`](../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
  — silent-hang incident report (the failure mode the wrapper does
  NOT handle)
- [`scripts/FAILOVER.md`](../../scripts/FAILOVER.md) — code-adjacent
  quickref
