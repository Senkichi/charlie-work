# Fleet and Supervision Design

## No-daemon principle

The orchestrator is a **foreground loop of unchanged level-triggered passes**,
not a background daemon service.

What this means in practice:

- `charlie bash-rats` is an explicit foreground process.  The operator starts
  it; it runs; the operator stops it with Ctrl+C.  There is no auto-start,
  no restart-on-crash, no PID file, no systemd unit.
- Each `loop()` pass is a complete re-derivation of the system state from
  ground truth (GitHub labels + sidecar files).  The supervisor layer only
  decides *when* to run the next pass — it never accumulates local state
  between passes or bypasses the ground-truth read.
- The supervised loop is **not async** and not threaded.  It is a single-
  threaded poll-sleep-pass cycle.

This matters for the following invariants:

| Invariant | Why it holds |
|-----------|-------------|
| No stale in-memory state | Each `loop()` re-reads GitHub and state.json from scratch |
| No double-dispatch | Supervisor lock (`supervisor.lock`) blocks concurrent invocations |
| Crash-safe | Process death releases the OS file lock; the next manual invocation starts clean |
| Atomic state writes | Unchanged — `tmp.replace(path)` pattern still applies to every JSON write |

## Fleet architecture

`charlie fleet bash-rats` is a **one-shot multi-repo sweep**:

1. Walk the fleet registry oldest-`last_seen`-first (or `--repos` explicit list).
2. For each repo: build `OrchestratorApp`, call `app.loop()` once, aggregate
   results into an attention digest.
3. Exit.

The fleet command does **not** use the supervised loop.  It is designed for
scheduled invocation by Task Scheduler / cron — the scheduler provides the
repeat cadence, and the lock in each repo's `supervisor.lock` prevents
concurrent single-repo `bash-rats` runs from overlapping with a fleet sweep.

## Supervisor design details

The supervised infill loop lives in `src/charlie_work/supervise.py`:

- `LocalSnapshot` — ephemeral in-process observation of sidecar mtimes and
  verdict file mtimes.  Never persisted.
- `has_delta()` — pure equality check; no network.
- `should_exit()` — pure predicate; exits when `live_count == 0`,
  `dispatched == 0`, `rework == 0`, `merged == 0`, and `open_tracked_prs == 0`.
  The `open_tracked_prs` field keeps the loop alive while PRs await operator
  verdicts — the verdict file's mtime change triggers the delta that fires the
  merge-lane pass.
- `try_acquire_supervisor_lock()` — non-blocking `msvcrt.LK_NBLCK` (Windows)
  / `fcntl.LOCK_EX|LOCK_NB` (POSIX) on `.var/charlie-work/supervisor.lock`.
  Returns `None` on contention (second invocation exits cleanly, no error).
- `run_supervised()` — the main loop.  Injected `sleep`/`clock` callables
  enable unit testing without real wall-clock delays.

## Same-head packet skip

`OrchestratorApp.loop()` includes an optimization for undecided PRs: if a
`prs/pr-N/pr.json` packet already exists with the same `headRefOid` as the
live PR, `review()` is skipped for that PR on this pass.  This prevents
repeated supervised passes from regenerating review packets or re-firing
`review_started` label transitions every 5 minutes while the operator is
reading the packet.

When the operator writes a verdict (`review-decision.json`), its mtime change
appears in the next `take_snapshot()`, triggering `has_delta()` → full pass →
merge lane fires.  The skip guard lives only in `loop()`, not in `review()`
itself — direct `charlie why-charlie-hate` invocations are unaffected.
