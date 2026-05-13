#!/bin/bash
# failover_lib.sh — bad-node failover support for ezpz/torchtitan training.
#
# Source this file from a submit script that has requested
# NHOSTS_TRAIN + NHOSTS_SPARE nodes via PBS. Then call:
#
#   failover_init NHOSTS_TRAIN
#   failover_run <command...>
#
# `failover_init` splits PBS_NODEFILE → active.hostfile (first N lines)
# + spare.hostfile (rest), exports PBS_NODEFILE → active.hostfile so
# downstream `ezpz launch` only sees the training subset, and yeets the
# venv to ALL nodes (active + spare) so any spare can swap in cleanly.
#
# `failover_run` runs the command, scrapes the output for known
# bad-node failure modes, swaps the offending node out for a spare,
# and retries (up to FAILOVER_MAX_RETRIES, default 3).
#
# Detected failure modes (see scrape_bad_nodes.py):
#  - `<hostname>: rank N died from signal {9,11,15}` (PALS shepherd kill)
#  - `RuntimeError: ... Connection closed by peer [IP]:port` (gloo TCP)
#  - `RuntimeError: ... Timed out waiting Nms for recv` (gloo timeout)
#  - `MemoryError: std::bad_alloc` in `torch.distributed.broadcast` (init OOM —
#    no specific node, rotates 1 spare in blindly)
#
# Env vars (all optional):
#   FAILOVER_MAX_RETRIES   default 3
#   FAILOVER_LOG_DIR       default $(pwd)/logs/failover-${PBS_JOBID%%.*}
#   FAILOVER_KEEP_BAD      default 1 — append bad node to bad_nodes.txt for inspection

set -o pipefail  # don't `set -euo pipefail` here — venv activate has unbound vars

_failover_log() { echo "[failover] $*" >&2; }

# ---------------------------------------------------------------------------
# failover_init NHOSTS_TRAIN
#
# Splits PBS_NODEFILE into active + spare files in $FAILOVER_LOG_DIR.
# Exports PBS_NODEFILE → active.hostfile.
# Yeets venv to ALL nodes (so spare swap-in is instantaneous later).
# ---------------------------------------------------------------------------
failover_init() {
    local nhosts_train="$1"
    [[ -n "$nhosts_train" ]] || { _failover_log "ERROR: failover_init needs NHOSTS_TRAIN arg"; return 1; }
    [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] || { _failover_log "ERROR: PBS_NODEFILE not set or missing"; return 1; }

    local total
    total=$(wc -l < "$PBS_NODEFILE")
    if (( total < nhosts_train )); then
        _failover_log "ERROR: PBS gave us $total nodes, training needs $nhosts_train"
        return 1
    fi

    export FAILOVER_LOG_DIR="${FAILOVER_LOG_DIR:-$(pwd)/logs/failover-${PBS_JOBID%%.*}}"
    mkdir -p "$FAILOVER_LOG_DIR"

    export FAILOVER_PBS_NODEFILE_ORIG="$PBS_NODEFILE"
    export FAILOVER_ACTIVE="$FAILOVER_LOG_DIR/active.hostfile"
    export FAILOVER_SPARE="$FAILOVER_LOG_DIR/spare.hostfile"
    export FAILOVER_BAD="$FAILOVER_LOG_DIR/bad_nodes.txt"

    head -n "$nhosts_train"   "$PBS_NODEFILE" > "$FAILOVER_ACTIVE"
    tail -n +"$((nhosts_train + 1))" "$PBS_NODEFILE" > "$FAILOVER_SPARE"
    : > "$FAILOVER_BAD"  # truncate

    export PBS_NODEFILE="$FAILOVER_ACTIVE"
    export NHOSTS="$nhosts_train"

    _failover_log "PBS gave us $total nodes ($(wc -l < "$FAILOVER_ACTIVE") active, $(wc -l < "$FAILOVER_SPARE") spare)"
    _failover_log "active hostfile: $FAILOVER_ACTIVE"
    _failover_log "spare hostfile:  $FAILOVER_SPARE"
    _failover_log "bad-node log:    $FAILOVER_BAD"
}

# ---------------------------------------------------------------------------
# failover_yeet_all
#
# Run `ezpz yeet-env --src .venv.tar.gz` against the FULL nodelist
# (active + spare), so a spare can swap in without re-broadcast.
# Restores PBS_NODEFILE → active afterwards.
# ---------------------------------------------------------------------------
failover_yeet_all() {
    local full_nodefile="$FAILOVER_LOG_DIR/all.hostfile"
    cat "$FAILOVER_ACTIVE" "$FAILOVER_SPARE" > "$full_nodefile"
    local saved_pbs_nodefile="$PBS_NODEFILE"
    local saved_nhosts="$NHOSTS"
    export PBS_NODEFILE="$full_nodefile"
    export NHOSTS=$(wc -l < "$full_nodefile")
    _failover_log "yeet-env to ALL $NHOSTS nodes (active + spare)"
    if [[ -f .venv.tar.gz ]]; then
        ezpz yeet-env --src .venv.tar.gz
    else
        ezpz yeet-env
    fi
    local rc=$?
    export PBS_NODEFILE="$saved_pbs_nodefile"
    export NHOSTS="$saved_nhosts"
    return $rc
}

# ---------------------------------------------------------------------------
# failover_swap_in BAD_HOSTNAMES...
#
# For each bad hostname, remove it from active.hostfile and replace it
# with the next available spare (popping from spare.hostfile). Logs
# all bad hostnames to bad_nodes.txt for postmortem.
# Returns 0 if all swaps succeeded, 1 if we ran out of spares.
# ---------------------------------------------------------------------------
failover_swap_in() {
    local bad
    local swapped=0
    for bad in "$@"; do
        # Bad node must actually be in active set
        if ! grep -qxF "$bad" "$FAILOVER_ACTIVE" 2>/dev/null; then
            _failover_log "skip: $bad not in active hostfile"
            continue
        fi
        # Need a spare
        local spare
        spare=$(head -n 1 "$FAILOVER_SPARE")
        if [[ -z "$spare" ]]; then
            _failover_log "ERROR: out of spares — cannot replace $bad"
            return 1
        fi
        # Pop spare, swap in
        tail -n +2 "$FAILOVER_SPARE" > "$FAILOVER_SPARE.tmp" && mv "$FAILOVER_SPARE.tmp" "$FAILOVER_SPARE"
        sed -i "s|^$bad\$|$spare|" "$FAILOVER_ACTIVE"
        echo "$bad" >> "$FAILOVER_BAD"
        _failover_log "swapped: $bad -> $spare"
        swapped=$((swapped + 1))
    done
    _failover_log "swap summary: $swapped bad nodes replaced; $(wc -l < "$FAILOVER_SPARE") spares remaining"
    return 0
}

# ---------------------------------------------------------------------------
# failover_swap_one_blind
#
# Generic init-time crash with no specific bad hostname (e.g. the
# `set_determinism` `std::bad_alloc`). Just rotate ONE spare into the
# active set — pop the first active host, push to spare, pop spare,
# push to active.
# Returns 0 if success, 1 if no spares left.
# ---------------------------------------------------------------------------
failover_swap_one_blind() {
    local first_active
    first_active=$(head -n 1 "$FAILOVER_ACTIVE")
    [[ -n "$first_active" ]] || return 1
    failover_swap_in "$first_active"
}

# ---------------------------------------------------------------------------
# failover_run COMMAND...
#
# Run COMMAND, capturing its output to $FAILOVER_LOG_DIR/attempt-N.log.
# On non-zero exit, scrape the log for bad-node signatures and retry
# with swap-ins, up to $FAILOVER_MAX_RETRIES times.
# ---------------------------------------------------------------------------
failover_run() {
    local max=${FAILOVER_MAX_RETRIES:-3}
    local attempt=1
    local rc

    # If the command is `ezpz launch ...`, inject explicit topology args so
    # ezpz launch doesn't re-derive nhosts/ngpus from the original PBS aux
    # file (which still has 260/522 nodes — the spares we excluded). Without
    # this, _infer_topology computes ngpus=N_full*12 then trips
    # "ngpus must be > 0 and <= N_active*12, got N_full*12".
    local cmd=("$@")
    if [[ "${cmd[0]}" == "ezpz" && "${cmd[1]}" == "launch" ]]; then
        local ppn="${NGPU_PER_HOST:-12}"
        local nproc=$(( NHOSTS * ppn ))
        cmd=(
            "${cmd[@]:0:2}"
            "--hostfile=$FAILOVER_ACTIVE"
            "--nhosts=$NHOSTS"
            "--nproc-per-node=$ppn"
            "--nproc=$nproc"
            "${cmd[@]:2}"
        )
    fi

    while (( attempt <= max + 1 )); do
        local logf="$FAILOVER_LOG_DIR/attempt-${attempt}.log"
        _failover_log "attempt ${attempt}/${max} — active=$(wc -l < "$FAILOVER_ACTIVE") nodes, spare=$(wc -l < "$FAILOVER_SPARE") nodes"
        _failover_log "logging to $logf"

        # Use stdbuf to keep tee'd output unbuffered, redirect both stderr and stdout.
        "${cmd[@]}" 2>&1 | tee "$logf"
        rc=${PIPESTATUS[0]}

        if (( rc == 0 )); then
            _failover_log "attempt ${attempt} succeeded (exit 0)"
            return 0
        fi

        # Walltime kills are NOT bad-node failures — exit -29 means walltime hit.
        # PBS reports exit -29 as bash exit 143 (128+15). Don't retry on those.
        if (( rc == 143 )); then
            _failover_log "attempt ${attempt} exited 143 (walltime / SIGTERM) — not a bad-node failure, no retry"
            return $rc
        fi

        _failover_log "attempt ${attempt} failed (exit $rc) — scraping for bad nodes"
        local bad_nodes
        bad_nodes=$(python3 "$(dirname "${BASH_SOURCE[0]}")/scrape_bad_nodes.py" "$logf" 2>/dev/null || true)

        if (( attempt > max )); then
            _failover_log "ERROR: max retries ($max) exhausted; giving up"
            return $rc
        fi

        if [[ -n "$bad_nodes" ]]; then
            _failover_log "bad nodes detected: $bad_nodes"
            failover_swap_in $bad_nodes || { _failover_log "swap failed; giving up"; return $rc; }
        else
            _failover_log "no specific bad node identified — rotating one spare in blindly"
            failover_swap_one_blind || { _failover_log "no spares left; giving up"; return $rc; }
        fi

        # Sanity check: still have nhosts_train active nodes
        local active_count
        active_count=$(wc -l < "$FAILOVER_ACTIVE")
        if (( active_count != NHOSTS )); then
            _failover_log "ERROR: active count drifted ($active_count != $NHOSTS); giving up"
            return $rc
        fi

        attempt=$((attempt + 1))
    done
}
