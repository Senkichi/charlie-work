"""Supervised infill loop for ``charlie bash-rats``.

``run_supervised`` drives repeated ``OrchestratorApp.loop()`` passes in a
foreground loop, using cheap local-signal delta detection to decide when a
pass is warranted.  The loop is single-threaded and injected-clock/sleep
testable (same pattern as cross_family.py).

Design notes:
- No threads, no asyncio, no daemon process.
- Each pass is a complete level-triggered re-derivation from ground truth
  (the same ``loop()`` call that ``--once`` makes); only the cadence changes.
- ``LocalSnapshot`` is ephemeral in-process observation — not persisted state.
- The supervisor lock is a SEPARATE non-blocking lock from state_lock; it
  prevents concurrent ``bash-rats`` invocations (including Task Scheduler
  pileup) from double-dispatching through the loop's governor read→launch
  window.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .worker import iter_workers

if TYPE_CHECKING:
    from .workflow import CommandResult, OrchestratorApp


# ---------------------------------------------------------------------------
# Local snapshot (zero network)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSnapshot:
    """Cheap local-filesystem observation used to detect deltas between polls.

    All fields are hashable so ``has_delta`` is a plain equality check.
    """

    live_count: int
    sidecar_mtimes: frozenset[tuple[str, float]]  # sessions/*.json name+mtime
    verdict_mtimes: frozenset[tuple[str, float]]  # prs/*/review-decision.json (pr-dir-name, mtime)


def take_snapshot(sessions_dir: Path, prs_dir: Path) -> LocalSnapshot:
    """Capture a fresh ``LocalSnapshot`` from the filesystem (never raises)."""
    # Live session count via sidecar files (adapter-agnostic)
    sidecar_mtimes: set[tuple[str, float]] = set()
    if sessions_dir.exists():
        for path in sessions_dir.glob("*.json"):
            try:
                sidecar_mtimes.add((path.name, path.stat().st_mtime))
            except OSError:
                pass

    # Verdict files: prs/pr-N/review-decision.json — key on the PR-unique parent
    # directory name ("pr-N"), not path.name (always the constant string
    # "review-decision.json"). Keying on path.name collides across every PR,
    # so a rewritten verdict for PR A can produce a frozenset identical to one
    # from PR B and has_delta() would miss the change.
    verdict_mtimes: set[tuple[str, float]] = set()
    if prs_dir.exists():
        for path in prs_dir.glob("*/review-decision.json"):
            try:
                verdict_mtimes.add((path.parent.name, path.stat().st_mtime))
            except OSError:
                pass

    # Count actual live workers, not just sidecar files. This correctly
    # excludes terminal launch-failure sidecars (pid=None, error set) and
    # dead workers while still using sidecar mtimes for delta detection.
    live_count = 0
    if sessions_dir.exists():
        try:
            live_count = sum(
                1 for w in iter_workers(sessions_dir) if w.error is None and w.is_alive()
            )
        except Exception:
            live_count = 0

    return LocalSnapshot(
        live_count=live_count,
        sidecar_mtimes=frozenset(sidecar_mtimes),
        verdict_mtimes=frozenset(verdict_mtimes),
    )


def has_delta(before: LocalSnapshot, after: LocalSnapshot) -> bool:
    """Return True if any local signal changed between the two snapshots."""
    return (
        before.live_count != after.live_count
        or before.sidecar_mtimes != after.sidecar_mtimes
        or before.verdict_mtimes != after.verdict_mtimes
    )


# ---------------------------------------------------------------------------
# Exit predicate (pure, unit-testable)
# ---------------------------------------------------------------------------


def should_exit(pass_result: CommandResult, live_count: int) -> bool:
    """Return True when the system is fully drained and nothing is actionable.

    Keeps the loop alive while:
    - any workers are live (they may open PRs or complete);
    - any fresh or rework dispatches occurred this pass (slots just filled);
    - any merges actually SUCCEEDED (base may have shifted for remaining PRs) --
      a failed merge attempt (can_merge=False) does not itself indicate
      drained-ness; the PR it failed on is still open, which the
      ``open_tracked_prs`` check below already covers;
    - any open tracked PRs await operator verdicts (verdict → merge on next pass);
    - dispatch was deferred due to provider throttling or a held fleet lock —
      queued issues are still waiting to be dispatched once the cooldown clears
      or the lock becomes available, so "nothing happened this pass" must not be
      read as "drained".
    """
    data = pass_result.data
    dispatch_data = data.get("dispatch", {})
    rework_data = data.get("dispatch_rework", {})
    dispatched = dispatch_data.get("selected_count", 0)
    rework = rework_data.get("selected_count", 0)
    # loop() appends merge_ready(...).data for every approved PR regardless of
    # outcome (each entry carries a "merged" bool) -- count only entries that
    # actually merged. A failed attempt is not "activity" in its own right.
    merged = sum(1 for entry in data.get("merges", []) if entry.get("merged"))
    open_prs = data.get("open_tracked_prs", 0)
    # Any deferred_reason (provider throttling, fleet lock held) means the loop
    # should stay alive and retry instead of reporting "drained".
    if dispatch_data.get("deferred_reason") or rework_data.get("deferred_reason"):
        return False
    return live_count == 0 and dispatched == 0 and rework == 0 and merged == 0 and open_prs == 0


# ---------------------------------------------------------------------------
# Supervisor lock (non-blocking, separate from state_lock)
# ---------------------------------------------------------------------------


class _SupervisorLock:
    """Holds an OS-level non-blocking lock on ``supervisor.lock`` for the run.

    Released in ``__del__`` and explicit ``release()``; also released on process
    death (OS closes all file handles).
    """

    def __init__(self, path: Path, handle: object) -> None:
        self._path = path
        self._handle = handle
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # Unlock and close are independent failure modes: an msvcrt/fcntl
        # unlock raising OSError must not skip closing the handle (that would
        # leak the file descriptor and, on Windows, keep the lock file
        # undeletable/unlockable by anyone else).
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass
        try:
            self._handle.close()  # type: ignore[attr-defined]
        except OSError:
            pass

    def __del__(self) -> None:
        self.release()


def try_acquire_supervisor_lock(lock_path: Path) -> _SupervisorLock | None:
    """Try to acquire the supervisor lock non-blocking.

    Returns a ``_SupervisorLock`` if acquired; ``None`` if another process holds
    it (second-instance rejection).  Never raises.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Write 1 byte on creation so msvcrt.locking(... 1) has a byte to lock.
        # touch() creates an empty (0-byte) file; msvcrt locks specific byte ranges
        # and will raise EACCES on a 0-byte file even for a non-blocking attempt.
        if not lock_path.exists():
            lock_path.write_bytes(b"\x00")
        handle = lock_path.open("r+b")
        if sys.platform == "win32":
            import msvcrt

            # Guard against a pre-existing 0-byte lock file (e.g. left over
            # from an older touch()-based implementation, or a race with
            # another process's creation) — the write above only fires when
            # the file doesn't exist yet, so a stale empty file would still
            # make msvcrt.locking raise EACCES.
            if handle.seek(0, 2) == 0:
                handle.write(b"\x00")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                return None
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, BlockingIOError):
                handle.close()
                return None
        return _SupervisorLock(lock_path, handle)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Main supervised loop
# ---------------------------------------------------------------------------


def run_supervised(
    app: OrchestratorApp,
    *,
    limit: int | None = None,
    merge: bool | None = None,
    poll_interval_override: int | None = None,
    max_runtime_override: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    max_passes: int | None = None,
) -> CommandResult:
    """Run a supervised infill loop of ``app.loop()`` passes.

    Polls cheap local signals (sidecar mtime, verdict file mtime) and runs a
    full pass when something actionable changes.  Falls back to a full pass
    after ``full_pass_interval_seconds`` even with no local delta (catches
    GitHub-side changes).

    ``poll_interval_override`` and ``max_runtime_override`` apply CLI arguments
    on top of ``app.config.supervisor``.  ``sleep`` and ``clock`` are injected
    for testing.  ``max_passes`` is a test escape-hatch (unlimited in production).

    Returns ``CommandResult(ok=True)`` on clean drain or KeyboardInterrupt.
    Returns ``CommandResult(ok=False, ...)`` when the supervisor lock is held.
    """
    # Import here to avoid circular imports (supervise ← workflow ← supervise)
    from .workflow import CommandResult

    # Single-instance guard
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    if lock is None:
        return CommandResult(
            False,
            "supervisor already running (supervisor.lock held)",
            {},
        )

    # Apply CLI overrides on top of the configured supervisor section as a
    # single ``dataclasses.replace`` — one config object instead of parallel
    # locals that can drift out of sync with each other or with future fields.
    overrides: dict[str, int] = {}
    if poll_interval_override is not None:
        overrides["poll_interval_seconds"] = poll_interval_override
    if max_runtime_override is not None:
        overrides["max_runtime_minutes"] = max_runtime_override
    cfg = dataclasses.replace(app.config.supervisor, **overrides)

    poll_interval = cfg.poll_interval_seconds
    max_runtime_minutes = cfg.max_runtime_minutes
    full_pass_interval = cfg.full_pass_interval_seconds
    active_cooldown = cfg.active_cooldown_seconds

    sessions_dir = app._resolve(app.config.devin.sessions_dir)
    prs_dir = app.paths.prs

    pass_number = 0
    total_dispatched = 0
    total_merged = 0
    start_time = clock()
    # Prime: subtract full_pass_interval so the first iteration fires immediately
    last_full_pass_at = start_time - full_pass_interval

    snapshot = take_snapshot(sessions_dir, prs_dir)

    try:
        while True:
            now = clock()

            # Max-runtime check
            if max_runtime_minutes is not None and max_runtime_minutes > 0:
                elapsed_minutes = (now - start_time) / 60.0
                if elapsed_minutes >= max_runtime_minutes:
                    break

            # Max-passes check (test escape-hatch)
            if max_passes is not None and pass_number >= max_passes:
                break

            # Delta + fallback check
            new_snapshot = take_snapshot(sessions_dir, prs_dir)
            fallback_due = (now - last_full_pass_at) >= full_pass_interval
            run_pass = has_delta(snapshot, new_snapshot) or fallback_due

            if run_pass:
                pass_number += 1
                last_full_pass_at = now

                pass_result = app.loop(limit, merge=merge)

                # Snapshot AFTER the pass becomes the baseline for the next
                # iteration's delta check (and supplies live_count). Using the
                # pre-pass ``new_snapshot`` as the baseline instead would (a)
                # race the next iteration's own take_snapshot call, and (b)
                # guarantee a spurious extra pass whenever this pass's own
                # side effects (e.g. a fresh dispatch writing a new sidecar
                # file) show up as a "delta" on the very next poll.
                snapshot = take_snapshot(sessions_dir, prs_dir)
                live_count = snapshot.live_count

                # Accumulate totals
                data = pass_result.data
                dispatched = data.get("dispatch", {}).get("selected_count", 0) + data.get(
                    "dispatch_rework", {}
                ).get("selected_count", 0)
                # merge_ready(...) appends one entry per approved PR regardless
                # of outcome ("merged": bool) -- count only the ones that
                # actually merged, not every attempt (finding: a pass with 3
                # failed can_merge=False attempts previously reported "merged
                # 3" despite zero real merges).
                merges = data.get("merges", [])
                merge_attempts = len(merges)
                merged = sum(1 for entry in merges if entry.get("merged"))
                total_dispatched += dispatched
                total_merged += merged

                # One-line summary
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                errors_count = len(data.get("errors", []))
                warnings_count = len(data.get("warnings", []))
                open_prs = data.get("open_tracked_prs", 0)
                # Dispatch: fresh+rework
                fresh = data.get("dispatch", {}).get("selected_count", 0)
                rework = data.get("dispatch_rework", {}).get("selected_count", 0)
                reviewed = len(data.get("reviews", []))
                skipped = data.get("skipped_reviews", 0)
                # Compact by default ("merged N"); surface failed attempts as
                # "merged N/M" only when they diverge from successes, so the
                # common (all-succeeded or no-attempts) case stays stable.
                merged_str = (
                    f"{merged}/{merge_attempts}" if merge_attempts > merged else str(merged)
                )
                # Prefer the dispatch-scoped governor's live count; it is the same
                # number the governor used for its clamp decision this pass.
                # Fall back to the local snapshot count only when the governor did
                # not emit a live count.
                summary_live_count = live_count
                for source in (data.get("dispatch", {}), data.get("dispatch_rework", {}), data):
                    for key in ("fleet_live_session_count", "live_session_count"):
                        if key in source:
                            summary_live_count = source[key]
                            break
                    else:
                        continue
                    break
                print(
                    f"[{now_str}] pass {pass_number}: dispatched {fresh}+{rework},"
                    f" merged {merged_str}, reviewed {reviewed}(+{skipped} skipped),"
                    f" live ~{summary_live_count}, prs-open {open_prs},"
                    f" errors {errors_count}, warnings {warnings_count}",
                    flush=True,
                )

                # Exit when drained
                if should_exit(pass_result, live_count):
                    break

                # Cooldown sleep: shorter after action, longer after idle pass
                active = dispatched > 0 or merged > 0
                sleep(active_cooldown if active else float(poll_interval))
            else:
                snapshot = new_snapshot
                sleep(float(poll_interval))

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        # Errors-as-values invariant: a raw exception from app.loop() (or
        # anything else in the loop body) must not propagate past the
        # supervisor — callers expect CommandResult(ok=False, ...), not a
        # traceback. The lock is still released via `finally` below.
        elapsed_s = clock() - start_time
        return CommandResult(
            False,
            f"supervisor aborted on pass {pass_number}: {exc}",
            {
                "passes": pass_number,
                "total_dispatched": total_dispatched,
                "total_merged": total_merged,
                "elapsed_seconds": elapsed_s,
            },
        )

    finally:
        lock.release()

    elapsed_s = clock() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    return CommandResult(
        True,
        f"supervised loop complete: {pass_number} pass(es) in {elapsed_str},"
        f" dispatched {total_dispatched}, merged {total_merged}",
        {
            "passes": pass_number,
            "total_dispatched": total_dispatched,
            "total_merged": total_merged,
            "elapsed_seconds": elapsed_s,
        },
    )
