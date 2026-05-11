# Bad-node failover for production training

> Recurring Aurora bad-node failures (`shepherd died from signal 9`,
> gloo TCP timeouts) have killed at least 6 production jobs in the
> past 2 weeks. This wrapper requests N+spare nodes from PBS, splits
> the allocation into an active training subset and a spare pool,
> and on a bad-node crash swaps the offending node out for a spare
> and retries.

## Files

- **`failover_lib.sh`** — bash library: `failover_init`,
  `failover_yeet_all`, `failover_swap_in`, `failover_swap_one_blind`,
  `failover_run`. Source from a submit script.
- **`scrape_bad_nodes.py`** — extracts bad-node hostnames from a
  training log. Used by `failover_run` to decide which node to swap
  out on a non-zero exit.
- **`submit_agpt_2b_aurora_venv_failover.sh`** — 2B submit script
  with failover (otherwise identical to `submit_agpt_2b_aurora_venv.sh`).
- **`submit_agpt_20b_aurora_venv_failover.sh`** — 20B submit script
  with failover.

## Usage

Submit with `qsub -l select=N+spare -v NHOSTS_TRAIN=N`:

```bash
# 2B canonical chain at 512 active + 10 spare nodes (= 522 PBS allocation)
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

## What the failover wrapper does

On startup:

1. Read PBS_NODEFILE (which has N+spare lines).
2. Split into `active.hostfile` (first N) + `spare.hostfile` (rest).
3. Override `PBS_NODEFILE` → `active.hostfile` so `ezpz launch` only
   sees the training subset.
4. yeet venv to **all** nodes (active + spare) so a spare can swap
   in without re-broadcast.
5. Run the training command, capturing output to
   `logs/failover-<JOBID>/attempt-1.log`.

On a non-zero exit (other than walltime / SIGTERM = 143):

1. Run `scrape_bad_nodes.py` against the attempt log:
   - Pattern 1: `<host>: shepherd died from signal 9` → emits the host.
   - Pattern 2: `Connection closed by peer [IP]:port` → reverse-resolves
     the IP via `getent hosts`, emits the canonical Aurora hostname.
2. If bad nodes were identified, swap each one out for a fresh spare
   (popping from `spare.hostfile`).
3. If no specific bad host was identifiable (e.g. the
   `set_determinism` `std::bad_alloc` init crash), rotate one spare
   in blindly.
4. Retry up to `FAILOVER_MAX_RETRIES` times (default 3).
5. Walltime exits (143) are NOT treated as failures — no retry.

All bad nodes are appended to `logs/failover-<JOBID>/bad_nodes.txt`
for postmortem. After the job finishes, that file is the canonical
record of which nodes misbehaved during this allocation.

## Detected failure modes

The detector is empirically tuned against 5 production crashes from
late April / early May 2026. See `scrape_bad_nodes.py` for the regex
patterns and the conscious decision to **not** match
`rank N died from signal {11,15}` (those are usually downstream of a
primary kill on a different node).

| Pattern | Observed in | Detection method |
|---|---|---|
| `shepherd died from signal 9` | 8459818, 8460301, 8460302, 8463659 | regex match → emit hostname |
| gloo TCP `Connection closed by peer` / `Timed out waiting Nms` | 8470102, 8470103, 8479581 | regex match → reverse-resolve IP |
| `set_determinism` `std::bad_alloc` init crash | 8463182, 8463183, 8466848 | not matched — falls back to blind rotation |

## Caveats

- **Doesn't help with NaN explosions or genuine bugs in user code** —
  any non-zero exit triggers retry, but if the bug is deterministic
  the retries will all hit it. `FAILOVER_MAX_RETRIES=3` is meant as
  a circuit breaker.
- **Doesn't help with walltime hits** (those are exit 143, not retried).
  Continue to chain `qsub -W depend=afterany` continuations as before.
- **Tarball broadcast covers active + spare** which uses ~2% extra
  yeet time. Worth it: spare swap-in is then a no-op from a
  filesystem perspective.
- **Compile-time matters**: each retry re-runs `torch.compile` at
  start, costing 7-15 min for 2B/20B at 256N (4+ hours for 80B at
  TP=4). For 80B you may want `FAILOVER_MAX_RETRIES=1` so a single
  bad-node hit doesn't burn the whole walltime on recompiles.
- **Async checkpointing helps**: gloo failures during a sync DCP save
  (which is what killed 8479581 at step 500) abort the save —
  the failover retry will resume from the last successfully-saved
  ckpt, losing < 100 steps. Worth keeping
  `--checkpoint.async-mode=async` in the launch command.
