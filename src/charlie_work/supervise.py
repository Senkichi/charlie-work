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
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import fleet_registry, worktree
from .file_lock import ByteRangeFileLock, try_acquire_byte_range_lock
from .subprocess_runner import RunResult, run_captured
from .worker import iter_workers

logger = logging.getLogger(__name__)

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


_SupervisorLock = ByteRangeFileLock


def try_acquire_supervisor_lock(lock_path: Path) -> _SupervisorLock | None:
    """Try to acquire the supervisor lock non-blocking.

    Returns a ``_SupervisorLock`` if acquired; ``None`` if another process holds
    it (second-instance rejection).  Never raises.
    """
    return try_acquire_byte_range_lock(lock_path)


# ---------------------------------------------------------------------------
# Self-deploy (FF-pull origin/main + uv sync on dependency changes)
# ---------------------------------------------------------------------------


_DEP_LOCK_FILES: frozenset[str] = frozenset({"pyproject.toml", "uv.lock"})

#: Orchestrator source tree root (directory containing ``pyproject.toml``).
#: Derived once at import so every consumer sees the same path.
_ORCHESTRATOR_ROOT: Path = Path(__file__).resolve().parents[2]


def orchestrator_root() -> Path:
    """Return the orchestrator source tree root.

    This is the directory that contains ``pyproject.toml`` and is the target
    for ``self_deploy``'s ``git pull`` / ``uv sync``.  It is derived from this
    module's location (``src/charlie_work/supervise.py``) instead of being
    recomputed at every call site so file moves cannot silently break one copy.
    """
    return _ORCHESTRATOR_ROOT


def _pending_sync_marker_path(repo_root: Path) -> Path:
    """Return the path to the pending-sync marker for this orchestrator tree."""
    return repo_root / ".var" / "charlie-work" / "pending-sync.json"


def _write_marker(path: Path, from_sha: str, to_sha: str) -> None:
    """Persist the pending-sync marker atomically (temp-file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"from_sha": from_sha, "to_sha": to_sha}, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _read_marker(path: Path) -> dict[str, str]:
    """Read the marker, returning an empty dict on any read/parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _clear_marker(path: Path) -> None:
    """Remove the pending-sync marker, if it exists."""
    path.unlink(missing_ok=True)


@dataclass(frozen=True)
class SelfDeployResult:
    """Result of a self-deploy attempt.

    ``pulled`` is True when ``git pull`` reported success (including the
    already-up-to-date case).  ``changed`` is True when HEAD moved.  ``synced``
    is True when ``uv sync`` ran and succeeded.  ``ok`` is False whenever any
    step reported an error; callers must treat this as non-fatal and continue
    the pass.

    ``venv_repaired`` is True when the orchestrator venv's editable ``.pth``
    was detected pointing outside ``repo_root/src`` and was atomically rewritten
    to the correct path.
    """

    ok: bool
    pulled: bool
    changed: bool
    synced: bool
    from_sha: str | None = None
    to_sha: str | None = None
    message: str = ""
    error: str | None = None
    venv_repaired: bool = False


def _is_venv(path: Path) -> bool:
    """Return True when ``path`` looks like a virtual environment directory."""
    if not path.is_dir():
        return False
    return (
        (path / "pyvenv.cfg").is_file()
        or (path / "Lib" / "site-packages").is_dir()
        or (path / "lib").is_dir()
    )


def _find_venv_path(repo_root: Path) -> Path | None:
    """Locate the orchestrator virtual environment to verify/self-heal.

    Prefers ``repo_root/.venv`` when it exists, then accepts ``VIRTUAL_ENV``
    only when it points inside ``repo_root`` (or exactly at ``repo_root/.venv``
    after resolving).  This prevents a leaked ``VIRTUAL_ENV`` from a different
    checkout from being used for self-heal, and keeps tests that pass a fake
    ``repo_root`` from touching the real orchestrator venv.
    """
    repo_venv = (repo_root / ".venv").resolve()
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = Path(virtual_env).resolve()
        if _is_venv(candidate) and (
            candidate == repo_venv or candidate.is_relative_to(repo_root.resolve())
        ):
            return candidate
    if _is_venv(repo_venv):
        return repo_venv
    return None


def _repair_venv_pth(repo_root: Path, venv_path: Path) -> tuple[bool, str]:
    """Atomically rewrite the project editable ``.pth`` to point at ``repo_root/src``.

    Scans ``site-packages`` for ``.pth`` files whose names contain a top-level
    package name from ``repo_root/src``.  Every path line in those files that
    resolves anywhere other than ``repo_root/src`` is rewritten to the correct
    absolute path, then the file is replaced atomically via temp-file +
    ``.replace()``.
    """
    site_packages = worktree._site_packages_dir(venv_path)
    if not site_packages:
        return False, "could not locate site-packages in shared venv"
    main_src = (repo_root / "src").resolve()
    package_names = worktree._top_level_package_names(repo_root)
    project_pth_files = [
        pth
        for pth in site_packages.glob("*.pth")
        if any(name in pth.name for name in package_names)
    ]
    if not project_pth_files:
        return False, "no editable .pth found for project packages"

    repaired_any = False
    for pth in project_pth_files:
        original = pth.read_text(encoding="utf-8")
        lines = original.splitlines()
        new_lines: list[str] = []
        changed = False
        for raw_line in lines:
            target = worktree._resolve_pth_line(site_packages, raw_line)
            if target != Path() and target != main_src:
                new_lines.append(str(main_src))
                changed = True
            else:
                new_lines.append(raw_line)
        if not changed:
            continue

        new_content = "\n".join(new_lines)
        if original.endswith("\n"):
            new_content += "\n"
        try:
            tmp = pth.with_suffix(pth.suffix + ".tmp")
            tmp.write_text(new_content, encoding="utf-8")
            tmp.replace(pth)
        except OSError as exc:
            return False, f"failed to rewrite {pth.name}: {exc}"
        repaired_any = True

    if not repaired_any:
        return False, "editable .pth did not require rewriting"
    return True, "rewrote editable .pth to point at main checkout src"


def _check_venv(repo_root: Path) -> SelfDeployResult:
    """Verify the orchestrator venv's editable ``.pth`` and repair on mismatch.

    Reads the venv's editable ``.pth`` via ``worktree.verify_shared_venv`` and
    checks that it resolves to ``repo_root/src``.  On mismatch, logs loudly and
    atomically rewrites the ``.pth`` text to the correct target.  All errors
    are returned as values.
    """
    venv_path = _find_venv_path(repo_root)
    if venv_path is None:
        return SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message="no orchestrator venv found; pth check skipped",
        )

    venv_ok, venv_message = worktree.verify_shared_venv(repo_root, venv_path)
    if venv_ok:
        return SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message=venv_message,
        )

    logger.error("ORCHESTRATOR VENV PTH MISMATCH: %s", venv_message)

    repair_ok, repair_message = _repair_venv_pth(repo_root, venv_path)
    if not repair_ok:
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error=f"venv pth repair failed: {repair_message}",
        )

    venv_ok, venv_message = worktree.verify_shared_venv(repo_root, venv_path)
    if not venv_ok:
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error=f"venv pth repair did not fix the mismatch: {venv_message}",
        )

    return SelfDeployResult(
        ok=True,
        pulled=False,
        changed=False,
        synced=False,
        venv_repaired=True,
        message=f"venv editable target repaired: {venv_message}",
    )


def self_deploy(
    repo_root: Path,
    *,
    fleet_dir_override: str | None = None,
    run_command: Callable[..., RunResult] = run_captured,
    pull_timeout: int = 60,
    sync_timeout: int = 300,
) -> SelfDeployResult:
    """FF-pull ``origin/main`` and run ``uv sync`` when dependency files changed.

    Before pulling, verifies the orchestrator venv's editable ``.pth`` points at
    ``repo_root/src`` and self-heals by atomically rewriting the ``.pth`` text
    when it does not.  The .pth rewrite is not exe-locked and can run even while
    the current process image is in use.

    Uses ``git diff --name-only <from>..<to>`` to detect whether the pull
    touched ``pyproject.toml`` or ``uv.lock``.  Before running ``uv sync`` the
    fleet registry is consulted for live worker sessions; if any are active the
    sync is deferred and a pending-sync marker is written atomically.  The
    marker is checked on every subsequent pass, so the sync retries even when
    the next ``git pull`` finds no new commits.

    All subprocess errors are returned as values (non-fatal); the function
    never raises.
    """
    marker_path = _pending_sync_marker_path(repo_root)
    try:
        venv_check = _check_venv(repo_root)
        if not venv_check.ok:
            return venv_check
        venv_repaired = venv_check.venv_repaired

        before_res = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            timeout_seconds=pull_timeout,
        )
        if not before_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                venv_repaired=venv_repaired,
                error=before_res.error or before_res.stderr or "failed to read HEAD",
            )
        before_sha = before_res.stdout.strip()

        pull_res = run_command(
            ["git", "pull", "--ff-only", "origin", "main"],
            cwd=repo_root,
            timeout_seconds=pull_timeout,
        )
        if not pull_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                from_sha=before_sha,
                venv_repaired=venv_repaired,
                error=pull_res.error or pull_res.stderr or "pull failed",
            )

        after_res = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            timeout_seconds=pull_timeout,
        )
        if not after_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=True,
                changed=False,
                synced=False,
                from_sha=before_sha,
                venv_repaired=venv_repaired,
                error=after_res.error or after_res.stderr or "failed to read HEAD after pull",
            )
        after_sha = after_res.stdout.strip()

        marker = _read_marker(marker_path) if marker_path.exists() else None
        marker_from = marker.get("from_sha") if marker else None

        if before_sha == after_sha and not marker:
            return SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha=before_sha,
                to_sha=after_sha,
                venv_repaired=venv_repaired,
                message="already up to date",
            )

        head_changed = before_sha != after_sha
        changed = head_changed or marker is not None

        changed_files: set[str] = set()
        if head_changed:
            diff_res = run_command(
                ["git", "diff", "--name-only", f"{before_sha}..{after_sha}"],
                cwd=repo_root,
                timeout_seconds=pull_timeout,
            )
            if not diff_res.ok:
                return SelfDeployResult(
                    ok=False,
                    pulled=True,
                    changed=changed,
                    synced=False,
                    from_sha=before_sha,
                    to_sha=after_sha,
                    venv_repaired=venv_repaired,
                    error=diff_res.error or diff_res.stderr or "diff failed",
                )
            changed_files = {line.strip() for line in diff_res.stdout.splitlines() if line.strip()}

        dep_files_changed = bool(changed_files & _DEP_LOCK_FILES)
        if not dep_files_changed and not marker:
            return SelfDeployResult(
                ok=True,
                pulled=True,
                changed=changed,
                synced=False,
                from_sha=before_sha,
                to_sha=after_sha,
                venv_repaired=venv_repaired,
                message=f"code-only update: {after_sha}",
            )

        from_sha = marker_from or before_sha
        to_sha = after_sha

        live_count, _ = fleet_registry.count_fleet_live_sessions(fleet_dir_override)
        if live_count > 0:
            _write_marker(marker_path, from_sha, to_sha)
            runner_word = "runner" if live_count == 1 else "runners"
            if marker is not None:
                print(
                    f"WARNING: pending dependency sync still deferred: {live_count} "
                    f"{runner_word} active (marker {from_sha}..{to_sha})",
                    flush=True,
                )
            return SelfDeployResult(
                ok=True,
                pulled=True,
                changed=changed,
                synced=False,
                from_sha=from_sha,
                to_sha=to_sha,
                venv_repaired=venv_repaired,
                message=f"sync deferred: {live_count} {runner_word} active",
            )

        # Persist marker before attempting sync so a crash between the pull and
        # the successful sync is retried on the next pass.
        _write_marker(marker_path, from_sha, to_sha)
        sync_res = run_command(
            ["uv", "sync"],
            cwd=repo_root,
            timeout_seconds=sync_timeout,
        )
        if not sync_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=True,
                changed=changed,
                synced=False,
                from_sha=from_sha,
                to_sha=to_sha,
                venv_repaired=venv_repaired,
                error=sync_res.error or sync_res.stderr or "uv sync failed",
            )

        _clear_marker(marker_path)
        return SelfDeployResult(
            ok=True,
            pulled=True,
            changed=changed,
            synced=True,
            from_sha=from_sha,
            to_sha=to_sha,
            venv_repaired=venv_repaired,
            message=f"updated and synced: {to_sha}",
        )
    except Exception as exc:
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            venv_repaired=False,
            error=f"self_deploy crashed: {exc}",
        )


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
