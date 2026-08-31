"""Supervised infill loop for ``charlie bash-rats``.

``run_supervised`` drives repeated ``OrchestratorApp.loop()`` passes in a
foreground loop, using cheap local-signal delta detection to decide when a
pass is warranted.  The loop is single-threaded and injected-clock/sleep
testable.

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
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Sequence

from . import fleet_registry, git_pull_blockers, layout, worktree
from .file_lock import ByteRangeFileLock, try_acquire_byte_range_lock
from .instrumentation import log_event
from .safe_path import contains
from .state import state_lock
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


def read_head_sha(
    repo_root: Path,
    *,
    run_command: Callable[..., RunResult] = run_captured,
    timeout_seconds: int = 30,
) -> str | None:
    """Return the current ``HEAD`` SHA of ``repo_root``, or ``None`` on error.

    Used by the supervisor to detect HEAD drift caused by an external actor
    (operator pull, another process) between passes — ``self_deploy`` only
    reports ``from_sha``/``to_sha`` for pulls *it* performed, so a HEAD moved
    out-of-band shows as "already up to date" and the daemon silently runs
    stale code forever.
    """
    res = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    if not res.ok:
        return None
    return res.stdout.strip() or None


def _pending_sync_marker_path(state_root: Path) -> Path:
    """Return the path to the deferred-``uv sync`` marker under ``state_root``."""
    return layout.pending_sync_path(state_root)


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
    already-up-to-date case).  ``changed`` is True when HEAD moved *or* a
    pending-sync marker exists (i.e. some dependency change is still
    outstanding, even if this pass's own pull was a no-op) -- callers that
    need to know whether *this pass* moved HEAD must use ``head_changed``
    instead, not ``changed``.  ``synced`` is True when ``uv sync`` ran and
    succeeded.  ``ok`` is False whenever any step reported an error; callers
    must treat this as non-fatal and continue the pass.

    ``head_changed`` is True only when ``git pull`` actually advanced HEAD on
    *this* attempt (``before_sha != after_sha``).  This is the correct signal
    for "the running process now has stale code loaded and must exit for a
    watchdog restart" -- ``from_sha``/``to_sha`` are not, because on a
    deferred-sync pass they report the marker's original range even when
    HEAD did not move on this attempt, which previously caused a restart loop
    (the supervisor kept exiting and relaunching without ever reaching zero
    live workers to complete the deferred sync).

    ``venv_repaired`` is True when the orchestrator venv's editable ``.pth``
    was detected pointing outside ``repo_root/src`` and was atomically rewritten
    to the correct path.

    ``deferred`` is True only on the one return path where a pending dependency
    sync was actually postponed because a fleet worker was live (issue #858).
    It is *not* derivable from ``ok and not synced``: both the "already up to
    date" and "code-only update" paths are also ``ok=True, synced=False`` but
    have nothing pending to sync at all, so treating that predicate as
    "deferred" made a genuine, resolved pull success indistinguishable from an
    unresolved deferral -- and made the failure-streak counter refuse to reset
    after real recovery (caught by
    ``test_self_deploy_failure_streak_fires_alarm_once_at_threshold``). Same
    lesson as ``head_changed`` one field up: callers that need to know about
    a specific event on *this* attempt need a field describing that event,
    not an inference from the shape of other fields.
    """

    ok: bool
    pulled: bool
    changed: bool
    synced: bool
    head_changed: bool = False
    from_sha: str | None = None
    to_sha: str | None = None
    message: str = ""
    error: str | None = None
    venv_repaired: bool = False
    previewed: bool = False
    deferred: bool = False

    @property
    def alertable(self) -> bool:
        """True when this failure warrants a durable, operator-facing alert.

        A preview never qualifies, and callers must consult this rather than
        ``ok`` before emitting a digest.  ``emit_digest``'s file sink mkdirs and
        *appends* to ``digest.jsonl``, so alerting on a previewed failure is a
        real persistent write performed by ``--dry-run`` -- and the entry it
        writes is indistinguishable from a genuine self-deploy failure, so it
        misleads the operator as well as mutating state.

        Read failures are reachable under preview precisely because the preview
        deliberately does not fetch: a stale or absent ``origin/main`` tracking
        ref makes ``git rev-parse origin/main`` fail on a checkout where the real
        self-deploy would have succeeded.
        """
        return not self.ok and not self.previewed

    """True when this was a ``dry_run`` preview and nothing was touched.

    Callers print ``message`` on a notable outcome, and the existing conditions for
    "notable" are ``synced`` and ``venv_repaired`` -- both False for a preview. Without
    a flag of its own the preview would run silently, which is worse than useless: the
    operator would see no output and conclude the deploy step did nothing at all.
    """


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
        if _is_venv(candidate) and contains(repo_root, candidate):
            return candidate
    if _is_venv(repo_venv):
        return repo_venv
    return None


def _repair_venv_pth(repo_root: Path, venv_path: Path) -> tuple[bool, str, list[str]]:
    """Atomically rewrite poisoned editable ``.pth`` lines to their configured roots.

    Scans every ``.pth`` in site-packages (no filename filter -- see
    :func:`worktree.verify_shared_venv`).  The expected root for each ``.pth``
    is inferred from its filename via :func:`worktree._match_pth_to_root`
    against the package map built from
    :func:`worktree._configured_editable_roots`.  When a specific root is
    derivable, a path-bearing line is poisoned when it resolves outside THAT
    root -- including the cross-root false-green case where it lands in a
    *different* configured root (fast-follow #1180).  When no specific root is
    derivable (unknown package), a line resolving outside all configured roots
    is the poisoned case and the file is reported unrepairable.

    The per-package root is the fix for issue #969 gap 1: the former repair
    rewrote *every* poisoned line to a single ``main_src`` constant
    (``repo_root/src``), which is correct for this repo's own packages but is a
    hard ``ImportError`` for a foreign editable like ``ci_fleet`` -- there is
    no ``ci_fleet`` package under ``charlie-work/src``.  Widening the
    verification filter without fixing this first would have auto-written that
    error as the first statement of every self-deploy attempt.

    When the correct root for a poisoned ``.pth`` cannot be determined (the
    filename matches no known package, or matches several ambiguously), the
    file is left untouched and reported as unrepairable.  Refusing to write a
    wrong root is strictly safer than guessing: a missed repair surfaces as a
    verification mismatch on re-check, while a wrong repair surfaces as a
    silent ``ImportError`` that verifies clean because the line now equals a
    configured root.

    Returns ``(ok, message, repaired_files)``.  ``repaired_files`` lists the
    ``.pth`` filenames that were successfully rewritten, even when ``ok`` is
    ``False`` because other files were unrepairable.  This closes the
    partial-repair-looks-like-no-op gap surfaced in PR #1176 review: a mixed
    poisoned/unmatchable scenario previously returned ``False`` with a message
    naming only the unrepairable entries, leaving the files that *were*
    rewritten indistinguishable from a no-op failure in ``events.db``.
    """
    site_packages = worktree._site_packages_dir(venv_path)
    if not site_packages:
        return False, "could not locate site-packages in shared venv", []
    roots = worktree._configured_editable_roots(repo_root)
    if not roots:
        return False, "could not derive configured editable roots from repo", []
    # package_name -> src_root, for matching a .pth filename to its target.
    package_to_root: dict[str, Path] = {}
    for src_root, package_names in roots:
        for name in package_names:
            package_to_root[name] = src_root

    repaired_files: list[str] = []
    unrepairable: list[str] = []
    for pth in site_packages.glob("*.pth"):
        original = pth.read_text(encoding="utf-8")
        lines = original.splitlines()
        # Infer this .pth's expected root from its filename.  When a specific
        # root is derivable, a line resolving into a *different* configured
        # root is the cross-root false-green case (fast-follow #1180) and must
        # be treated as poisoned -- not accepted because it happens to land in
        # *some* owned tree.  When no specific root is derivable (unknown
        # package), the "any configured root" check is the correct fallback: a
        # line outside all roots is still poisoned.
        expected_root = worktree._match_pth_to_root(pth, package_to_root)
        poisoned = False
        for raw_line in lines:
            target = worktree._resolve_pth_line(site_packages, raw_line)
            if target == Path():
                continue
            if expected_root is not None:
                if contains(expected_root, target):
                    continue
                poisoned = True
                break
            if any(contains(root, target) for root, _ in roots):
                continue
            poisoned = True
            break
        if not poisoned:
            continue

        correct_root = expected_root
        if correct_root is None:
            unrepairable.append(pth.name)
            continue

        new_lines: list[str] = []
        changed = False
        for raw_line in lines:
            target = worktree._resolve_pth_line(site_packages, raw_line)
            if target != Path() and not contains(correct_root, target):
                new_lines.append(str(correct_root))
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
            return False, f"failed to rewrite {pth.name}: {exc}", repaired_files
        repaired_files.append(pth.name)

    if unrepairable:
        return (
            False,
            (
                "could not determine correct root for editable .pth: "
                + ", ".join(sorted(unrepairable))
            ),
            repaired_files,
        )
    if not repaired_files:
        return False, "editable .pth did not require rewriting", []
    return True, "rewrote editable .pth targets to configured checkouts", repaired_files


def _check_venv(repo_root: Path) -> SelfDeployResult:
    """Verify the orchestrator venv's editable ``.pth`` and repair on mismatch.

    Reads the venv's editable ``.pth`` via ``worktree.verify_shared_venv`` and
    checks that every path line resolves into a configured checkout.  On
    mismatch, logs loudly, emits a ``venv_pth_mismatch`` event, and atomically
    rewrites the ``.pth`` text to the correct per-package target.  A successful
    repair emits ``venv_pth_repaired``; a failed repair or re-verification
    emits ``venv_pth_repair_failed``.  All errors are returned as values.

    The events close the observability gap from issue #969 gap 3: across
    20,727 rows in ``events.db`` zero matched any venv- or pth-related kind,
    so a silent auto-repair was indistinguishable from never having had a
    problem -- and a false green (the old filter reporting "repaired" over a
    still-poisoned foreign editable) left no trace at all.
    """
    state_path = _self_deploy_state_path(repo_root)
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
    log_event(
        state_path,
        "venv_pth_mismatch",
        {"venv_path": str(venv_path), "detail": venv_message},
    )

    repair_ok, repair_message, repaired_files = _repair_venv_pth(repo_root, venv_path)
    if not repair_ok:
        failed_payload: dict[str, Any] = {
            "venv_path": str(venv_path),
            "detail": repair_message,
        }
        # Record which .pth files WERE successfully rewritten even though the
        # overall call reports failure (unrepairable entries, or a mid-loop
        # OSError on a later file).  Without this a partial repair is
        # indistinguishable from a no-op failure in events.db -- the gap
        # surfaced in PR #1176 review.
        if repaired_files:
            failed_payload["repaired_files"] = repaired_files
        log_event(state_path, "venv_pth_repair_failed", failed_payload)
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error=f"venv pth repair failed: {repair_message}",
        )

    venv_ok, venv_message = worktree.verify_shared_venv(repo_root, venv_path)
    if not venv_ok:
        reverify_payload: dict[str, Any] = {
            "venv_path": str(venv_path),
            "detail": f"re-verification failed after repair: {venv_message}",
        }
        if repaired_files:
            reverify_payload["repaired_files"] = repaired_files
        log_event(state_path, "venv_pth_repair_failed", reverify_payload)
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error=f"venv pth repair did not fix the mismatch: {venv_message}",
        )

    repaired_payload: dict[str, Any] = {
        "venv_path": str(venv_path),
        "detail": repair_message,
    }
    if repaired_files:
        repaired_payload["repaired_files"] = repaired_files
    log_event(state_path, "venv_pth_repaired", repaired_payload)
    return SelfDeployResult(
        ok=True,
        pulled=False,
        changed=False,
        synced=False,
        venv_repaired=True,
        message=f"venv editable target repaired: {venv_message}",
    )


def _command_failure_message(command: Sequence[str], result: RunResult, fallback: str) -> str:
    """Build a diagnostic failure message that prefers the specific stderr.

    ``run_captured`` (the real production runner) always populates
    ``RunResult.error`` on any non-zero exit with a generic
    ``"command exited {code}"``. Under the old ``result.error or result.stderr``
    fallback chain that generic string was always truthy and permanently
    shadowed ``.stderr`` in production, even when stderr carried the actual
    diagnostic -- e.g. ``git pull --ff-only`` names the exact colliding paths
    on a dirty-tree collision, and that name was unreachable (issue #817 item
    3). Prefer ``.stderr``; fall back to ``.error``, then to ``fallback``, only
    when stderr is empty. The failing argv is always included so the message
    is actionable without cross-referencing a log.
    """
    detail = (result.stderr or "").strip() or (result.error or "").strip() or fallback
    return f"{' '.join(command)}: {detail}"


def _self_deploy_preview(
    repo_root: Path,
    marker_path: Path,
    *,
    run_command: Callable[..., RunResult],
    timeout: int,
) -> SelfDeployResult:
    """Report what :func:`self_deploy` would do, touching nothing.

    Read-only by construction: no ``git pull``, no ``uv sync``, and no venv
    ``.pth`` repair.  ``origin/main`` is read from the existing remote-tracking
    ref rather than fetched, so the answer is only as fresh as the last fetch --
    that staleness is the deliberate price of not mutating the ref store.  A
    preview that fetched would be more accurate and less honest.
    """
    head_cmd = ["git", "rev-parse", "HEAD"]
    head_res = run_command(head_cmd, cwd=repo_root, timeout_seconds=timeout)
    if not head_res.ok:
        # previewed=True on the failure paths too: the flag means "this result came
        # from the preview", not "the preview succeeded". Leaving it False here let a
        # dry run fall into the callers' `if not deploy.ok:` alert arm and append a
        # real ERROR entry to the notify sink -- see SelfDeployResult.alertable.
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error=_command_failure_message(head_cmd, head_res, "failed to read HEAD"),
            previewed=True,
        )
    head_sha = head_res.stdout.strip()

    target_cmd = ["git", "rev-parse", "origin/main"]
    target_res = run_command(target_cmd, cwd=repo_root, timeout_seconds=timeout)
    if not target_res.ok:
        return SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            from_sha=head_sha,
            error=_command_failure_message(target_cmd, target_res, "failed to read origin/main"),
            previewed=True,
        )
    target_sha = target_res.stdout.strip()

    notes: list[str] = []
    if head_sha == target_sha:
        notes.append("no fast-forward pending (HEAD is at last-known origin/main)")
    else:
        notes.append(f"would fast-forward {head_sha[:12]}..{target_sha[:12]}")
        diff_res = run_command(
            ["git", "diff", "--name-only", f"{head_sha}..{target_sha}"],
            cwd=repo_root,
            timeout_seconds=timeout,
        )
        if not diff_res.ok:
            notes.append("could not diff for dependency changes")
        elif {
            line.strip() for line in diff_res.stdout.splitlines() if line.strip()
        } & _DEP_LOCK_FILES:
            notes.append("would run `uv sync` (dependency files change)")

    if marker_path.exists():
        notes.append("a pending-sync marker exists, so a deferred `uv sync` would be retried")

    return SelfDeployResult(
        ok=True,
        pulled=False,
        changed=False,
        synced=False,
        from_sha=head_sha,
        to_sha=target_sha,
        message="dry run, nothing touched: " + "; ".join(notes),
        previewed=True,
    )


#: Default consecutive-failure count at which ``self_deploy`` fires a
#: ``self_deploy_alarm`` events.db entry (issue #817 item 5). Mirrors
#: ``config.AutoMergeConfig.failed_attempt_alarm``'s default and its "0
#: disables the alarm" convention.
DEFAULT_SELF_DEPLOY_FAILURE_ALARM = 3


def _self_deploy_state_path(repo_root: Path) -> Path:
    """Return the ``state.json`` path used to derive ``events.db`` for self-deploy.

    ``self_deploy`` operates on the orchestrator's own checkout (``repo_root``
    is ``orchestrator_root()`` at every real call site), not a per-repo fleet
    target, so it logs against the *default* state root. The pending-sync
    marker, by contrast, resolves from the ``state_root`` passed to
    :func:`self_deploy`.
    """
    return layout.state_file_path(layout.default_state_root(repo_root))


def _self_deploy_failure_counter_path(repo_root: Path) -> Path:
    """Return the consecutive-failure counter sidecar path for ``repo_root``."""
    return layout.self_deploy_failure_state_path(layout.default_state_root(repo_root))


def _self_deploy_event_kind(result: SelfDeployResult) -> str:
    """Classify a :class:`SelfDeployResult` into an events.db event kind (item 4).

    Three outcomes, matching the issue's explicit wording ("success ... skip,
    and failure"): any ``not ok`` result is a failure; an ``ok`` result that
    neither moved HEAD nor repaired the venv did nothing this pass (a skip);
    everything else -- a code-only update, a full update-and-sync, a deferred
    sync, or a venv repair -- actually changed something (a success).
    """
    if not result.ok:
        return "self_deploy_failed"
    if result.changed or result.venv_repaired:
        return "self_deploy_succeeded"
    return "self_deploy_skipped"


def _log_self_deploy_outcome(repo_root: Path, result: SelfDeployResult) -> None:
    """Record every real (non-preview) self_deploy outcome to events.db (item 4).

    Best-effort via :func:`log_event`: a logging failure never fails the
    deploy itself. Previews are deliberately excluded by the caller
    (:func:`self_deploy`) -- a ``--dry-run`` touches nothing, so logging it as
    a real outcome would misrepresent the deploy history.
    """
    log_event(
        _self_deploy_state_path(repo_root),
        _self_deploy_event_kind(result),
        {
            "ok": result.ok,
            "from_sha": result.from_sha,
            "to_sha": result.to_sha,
            "changed": result.changed,
            "synced": result.synced,
            "venv_repaired": result.venv_repaired,
            "message": result.message,
            "error": result.error,
        },
    )


def _read_failure_streak(path: Path) -> int:
    """Read the persisted consecutive-failure count, defaulting to 0 on any error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    count = data.get("consecutive_failures", 0)
    return count if isinstance(count, int) else 0


def _write_failure_streak(path: Path, count: int) -> None:
    """Persist the consecutive-failure count atomically (temp-file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"consecutive_failures": count}, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _record_self_deploy_failure_streak(
    repo_root: Path, result: SelfDeployResult, *, threshold: int
) -> None:
    """Track consecutive self_deploy failures and escalate at ``threshold`` (item 5).

    Three outcomes, not two (issue #858): ``result.deferred`` is a genuine
    deferral -- a dependency sync was actually pending but postponed because a
    fleet worker was live -- and carries no information about whether that
    sync would have succeeded, so it leaves the streak untouched. Any other
    ``ok`` result (including "already up to date" and "code-only update",
    both ``ok=True, synced=False`` but with nothing pending to sync at all) is
    a genuine success and resets the streak to 0. ``not ok`` is a genuine
    failure and increments it.

    Deliberately *not* keyed on ``ok and not synced``: that predicate cannot
    tell a real deferral apart from a resolved pull with nothing to sync, so
    it also refuses to reset the streak after genuine recovery from a run of
    pull failures -- see ``SelfDeployResult.deferred``'s docstring and
    ``test_self_deploy_failure_streak_fires_alarm_once_at_threshold``, which
    caught exactly that regression.

    Before this fix, any ``ok`` result -- including a deferral, which is
    reported as ``ok=True, synced=False`` precisely because declining to
    mutate the shared venv while a worker is live is the *correct* choice,
    not a failure -- reset the counter to 0. That let a permanently failing
    sync interleaved with occasional deferrals (``fail, fail, defer, fail,
    fail, defer, ...``) hold the streak at or below ``threshold - 1``
    forever, silencing the exact alarm this counter exists to raise.

    Mirrors ``workflow._merge_deferred_stale_base_result``'s alarm pattern
    otherwise: a distinct ``self_deploy_alarm`` event fires exactly once --
    the pass the counter *reaches* ``threshold``, not on every subsequent
    pass past it -- so a long-running outage produces one escalation, not
    one per pass (the digest's own dedup already handles "don't repeat every
    pass" for the plain ERROR entry; this is a second, coarser signal for
    "this has now gone on long enough to be a distinct incident").
    ``threshold <= 0`` disables the alarm entirely, matching
    ``failed_attempt_alarm``'s "0 disables" convention.
    """
    counter_path = _self_deploy_failure_counter_path(repo_root)
    # Ensure the parent directory exists before state_lock tries to create the
    # sibling .lock file (advisory_file_lock touches it directly) — mirrors the
    # mkdir in _filter_fleet_health_transitions. Without this, a cold state dir
    # (no prior events.db/state.json write in this pass) raises out of
    # self_deploy(), which violates its documented never-raises contract.
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(counter_path):
        count = _read_failure_streak(counter_path)
        if result.deferred:
            # A pending sync was genuinely postponed: no new information about
            # whether it would have succeeded. Leave the streak exactly as it
            # is -- do not reset, do not increment.
            return
        if result.ok:
            if count:
                _write_failure_streak(counter_path, 0)
            return
        count += 1
        _write_failure_streak(counter_path, count)
        if threshold > 0 and count == threshold:
            log_event(
                _self_deploy_state_path(repo_root),
                "self_deploy_alarm",
                {
                    "consecutive_failures": count,
                    "threshold": threshold,
                    "error": result.error,
                },
            )


#: Default consecutive-zero-repo-pass-cycle count at which the fleet
#: supervisor fires a ``supervisor_zero_pass_alarm`` events.db entry (issue
#: #855, the general shape behind #851). Mirrors
#: ``DEFAULT_SELF_DEPLOY_FAILURE_ALARM``'s default and "0 disables" convention.
DEFAULT_ZERO_PASS_ALARM = 3


def _zero_pass_streak_counter_path(repo_root: Path) -> Path:
    """Return the consecutive-zero-repo-pass-cycle counter sidecar path for ``repo_root``."""
    return layout.zero_pass_streak_state_path(layout.default_state_root(repo_root))


def _read_zero_pass_streak(path: Path) -> int:
    """Read the persisted consecutive-zero-repo-pass-cycle count, defaulting to 0 on any error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    count = data.get("consecutive_zero_pass_cycles", 0)
    return count if isinstance(count, int) else 0


def _write_zero_pass_streak(path: Path, count: int) -> None:
    """Persist the consecutive-zero-repo-pass-cycle count atomically (temp-file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"consecutive_zero_pass_cycles": count}, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def record_zero_pass_streak(
    repo_root: Path,
    *,
    repo_passes: int,
    repos_configured: bool,
    threshold: int = DEFAULT_ZERO_PASS_ALARM,
) -> None:
    """Track consecutive zero-repo-pass fleet-supervisor cycles; escalate at ``threshold``.

    Mirrors :func:`_record_self_deploy_failure_streak` (issue #855, the
    general shape behind #851): a persisted counter increments on every
    cycle that completed with ``repo_passes == 0`` and resets to 0 on any
    cycle with ``repo_passes > 0``. A distinct ``supervisor_zero_pass_alarm``
    event fires exactly once -- the cycle the counter *reaches* ``threshold``,
    not on every subsequent cycle past it -- so a long-running outage (the
    #851 shape: every watchdog restart exits before ``fleet_loop`` ever runs,
    exit code 0 every time) produces one escalation, not one per cycle.
    ``threshold <= 0`` disables the alarm entirely, matching
    ``self_deploy_failure_alarm``'s "0 disables" convention.

    ``repos_configured`` must be computed by the caller from the fleet
    registry, independent of whether this cycle actually reached
    ``fleet_loop`` -- the #851 exit shape this alarm targets breaks out of
    the supervisor loop before ``fleet_loop`` is ever called, so
    ``repo_passes`` alone cannot distinguish "no repos configured" from "repos
    configured but the loop never got to them". When ``repos_configured`` is
    False, this function is a complete no-op: it must not move the counter in
    either direction, so a fleet with zero configured repos never fires this
    alarm regardless of how long it runs (acceptance criterion 4).
    """
    if not repos_configured:
        return
    counter_path = _zero_pass_streak_counter_path(repo_root)
    # Ensure the parent directory exists before state_lock tries to create the
    # sibling .lock file (advisory_file_lock touches it directly) -- mirrors
    # _record_self_deploy_failure_streak. Without this, a cold state dir (no
    # prior events.db/state.json write yet this process) raises here, which
    # would propagate out of run_fleet_supervise's post-loop bookkeeping.
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(counter_path):
        count = _read_zero_pass_streak(counter_path)
        if repo_passes > 0:
            if count:
                _write_zero_pass_streak(counter_path, 0)
            return
        count += 1
        _write_zero_pass_streak(counter_path, count)
        if threshold > 0 and count == threshold:
            log_event(
                _self_deploy_state_path(repo_root),
                "supervisor_zero_pass_alarm",
                {
                    "consecutive_zero_pass_cycles": count,
                    "threshold": threshold,
                },
            )


def self_deploy(
    repo_root: Path,
    *,
    state_root: Path | None = None,
    fleet_dir_override: str | None = None,
    run_command: Callable[..., RunResult] = run_captured,
    pull_timeout: int = 60,
    sync_timeout: int = 300,
    dry_run: bool = False,
    failure_alarm_threshold: int = DEFAULT_SELF_DEPLOY_FAILURE_ALARM,
    pull_ci_fleet: bool = False,
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

    ``state_root`` is the resolved orchestrator state directory; when omitted,
    the pending-sync marker falls back to the default state root under
    ``repo_root``.  Other self-deploy state files (``events.db`` and the
    consecutive-failure counter) continue to resolve from the default state
    root regardless of ``state_root``.

    ``dry_run`` reports what would happen and touches nothing.  The gate sits
    above *every* mutating step, including ``_check_venv`` -- which repairs the
    editable ``.pth`` as a side effect of checking it, and so is not safe to run
    under a preview either.  Callers reached by the global ``--dry-run`` flag
    must pass it: this function moves the deployed checkout's HEAD, and a HEAD
    move terminates a running ``fleet supervise`` by design (drift exit), so an
    ungated preview can end the fleet rather than describe it (issue #613).

    Every *real* (non-preview) call is recorded to ``events.db`` and rolled
    into a persisted consecutive-failure counter that escalates via a
    ``self_deploy_alarm`` event at ``failure_alarm_threshold`` (issue #817
    items 4-5). A preview is excluded from both: it is read-only by
    construction and reports nothing that actually happened.
    """
    marker_path = _pending_sync_marker_path(
        state_root if state_root is not None else layout.default_state_root(repo_root)
    )
    if dry_run:
        return _self_deploy_preview(
            repo_root, marker_path, run_command=run_command, timeout=pull_timeout
        )
    result = _self_deploy_attempt(
        repo_root,
        marker_path,
        fleet_dir_override=fleet_dir_override,
        run_command=run_command,
        pull_timeout=pull_timeout,
        sync_timeout=sync_timeout,
        pull_ci_fleet=pull_ci_fleet,
    )
    _log_self_deploy_outcome(repo_root, result)
    _record_self_deploy_failure_streak(repo_root, result, threshold=failure_alarm_threshold)
    return result


_BLOCKER_NAMES_IN_MESSAGE = 10


def _format_blocker_names(paths: tuple[str, ...]) -> str:
    """Render blocker paths for an operator-facing message, bounded.

    The caller always prints ``len(paths)`` alongside this, so truncating the
    names loses no information about scale, and ``git status`` has the rest.
    Unbounded it would be a slow leak rather than a crash: a wedged checkout
    can hold dozens of blockers (19 untracked files in the deployed checkout
    at the time of writing), and the resulting string lands in
    ``SelfDeployResult.error`` -- which is re-rendered into a
    ``self_deploy_failed`` payload once per retry, roughly every five minutes,
    for as long as the wedge lasts. The readability of that message is the
    real reason for the cap; an operator scanning a failure line wants the
    first few names and the count, not a paragraph.
    """
    if len(paths) <= _BLOCKER_NAMES_IN_MESSAGE:
        return ", ".join(paths)
    shown = ", ".join(paths[:_BLOCKER_NAMES_IN_MESSAGE])
    return f"{shown}, +{len(paths) - _BLOCKER_NAMES_IN_MESSAGE} more"


@dataclass(frozen=True)
class PullBlockerRepair:
    """What one lossless-blocker sweep cleared before a self-deploy retry.

    ``retained`` names the blockers deliberately left in place -- either
    because their content genuinely differs from the incoming commit (so
    discarding them would lose work) or because they sit somewhere repair must
    never touch. An operator reading the ``self_deploy_failed`` event needs
    that list to know what to adjudicate, so it is carried on the result rather
    than only logged.
    """

    cleared: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()

    @property
    def acted(self) -> bool:
        """True when disk changed, i.e. when a retry could plausibly differ."""
        return bool(self.cleared)

    def describe(self) -> str:
        parts: list[str] = []
        if self.cleared:
            names = _format_blocker_names(self.cleared)
            parts.append(f"auto-cleared {len(self.cleared)} lossless blocker(s) [{names}]")
        if self.retained:
            names = _format_blocker_names(self.retained)
            parts.append(f"{len(self.retained)} blocker(s) still need a human [{names}]")
        return "; ".join(parts)


def _repairable_blocker_path(repo_root: Path, path: str) -> bool:
    """Whether blocker repair is allowed to touch ``path`` at all.

    Independent of whether removing it would lose content: some paths are off
    limits regardless. ``.venv`` is the orchestrator's own interpreter, and
    containment is enforced on the *resolved* path so a reparse point under
    ``repo_root`` cannot hand back a tree somewhere else.
    """
    if PurePosixPath(path).parts[:1] == (".venv",):
        return False
    return contains(repo_root, repo_root / path)


def _repair_lossless_pull_blockers(
    repo_root: Path,
    *,
    run_command: Callable[..., RunResult],
    timeout: int,
) -> PullBlockerRepair:
    """Clear only the pull blockers that provably cost nothing to clear.

    ``git pull --ff-only`` refuses to *start* when the incoming tree would
    clobber local work, in either of the two shapes
    :mod:`charlie_work.git_pull_blockers` describes. On the outage below it
    named both in a *single* refusal -- unlike the merge path, which revealed
    the second class only after the first was cleared -- which is exactly why
    both sets are derived here rather than read off the message.
    Without remediation the orchestrator simply stops deploying:
    on 2026-08-06 it ran stale code for 2h41m -- essentially the whole
    observation window to that point -- because four files sat in the way and
    nothing here could move them.

    The bar for touching a file is deliberately high: its worktree content must
    hash **identical to the blob the pull is about to write there**. Under that
    proof, clearing it is not a discard -- the pull immediately restores the
    same bytes. Anything else is left alone and named, because resolving it
    needs someone who knows which version supersedes which. On the incident
    above that split 3 auto-clearable against 1 (a stale draft) that genuinely
    required judgement, so this narrows the manual surface rather than
    eliminating it.

    Partial clearing is deliberate, and differs from the union gate in
    ``worktree._repair_declared_scaffolding_blockers`` because the proof is
    stronger here. There, "declared scaffolding" means *the orchestrator can
    regenerate this*, so a repair that failed to unblock the merge would have
    destroyed files for no benefit. Byte-identity is not a claim about
    regeneration -- there is nothing to lose, so clearing what is provably free
    is worth doing even when a sibling blocker keeps the pull red.
    """

    def run_git(argv: list[str]) -> RunResult:
        return run_command(argv, cwd=repo_root, timeout_seconds=timeout)

    # Only a *pending fast-forward* is ours to unblock. When the fetch half of
    # `git pull` is what failed, origin/main is stale -- usually equal to HEAD
    # -- and this returns empty, so a network failure is left alone instead of
    # provoking a repair of a tree that was never in the way. This is also the
    # discriminator that avoids parsing git's localized refusal text.
    head_res = run_git(["git", "rev-parse", "HEAD"])
    target_res = run_git(["git", "rev-parse", "origin/main"])
    if not head_res.ok or not target_res.ok:
        return PullBlockerRepair()
    head_sha = head_res.stdout.strip()
    ref = target_res.stdout.strip()
    if not head_sha or not ref or head_sha == ref:
        return PullBlockerRepair()
    if not run_git(["git", "merge-base", "--is-ancestor", head_sha, ref]).ok:
        # Diverged, not fast-forwardable. Rewriting local history is never this
        # function's call.
        return PullBlockerRepair()

    untracked = git_pull_blockers.untracked_paths_shadowing_ref(run_git, ref)
    modified = git_pull_blockers.modified_paths_overwritten_by_ref(run_git, ref)
    if not untracked and not modified:
        return PullBlockerRepair()

    def is_lossless(path: str) -> bool:
        if not _repairable_blocker_path(repo_root, path):
            return False
        incoming = run_git(["git", "rev-parse", f"{ref}:{path}"])
        if not incoming.ok:
            return False
        # Filters ON -- `git hash-object`'s default, NOT --no-filters. This
        # checkout runs core.autocrlf=true, so worktree bytes are CRLF while
        # the blob is LF; hashing raw reports every file as differing and makes
        # this sweep silently inert rather than wrong-looking.
        local = run_git(["git", "hash-object", "--", path])
        if not local.ok:
            return False
        local_sha = local.stdout.strip()
        return bool(local_sha) and local_sha == incoming.stdout.strip()

    def nothing_staged(path: str) -> bool:
        """No staged change for ``path``, so the index holds nothing to lose.

        ``is_lossless`` hashes the *worktree* file, which says nothing about the
        index. Without this, a path staged with unique content and then edited
        to match origin would read as lossless, and ``git checkout HEAD --``
        (which resets index *and* worktree) would destroy the staged version --
        the one thing here that exists nowhere else.

        Costless in practice: git does not block a fast-forward when the staged
        content already equals the incoming content, so a path that is both
        staged and genuinely lossless never reaches this code as a blocker.
        """
        return run_git(["git", "diff", "--cached", "--quiet", "HEAD", "--", path]).ok

    lossless_modified = tuple(
        path for path in modified if is_lossless(path) and nothing_staged(path)
    )
    # Untracked paths have no index entry, so there is no staged version to lose.
    lossless_untracked = tuple(path for path in untracked if is_lossless(path))
    retained = set(untracked) | set(modified)
    retained -= set(lossless_modified) | set(lossless_untracked)

    cleared: list[str] = []
    if lossless_modified:
        # `git checkout HEAD --`, never bare `git checkout --`: the latter
        # restores from the *index*, so a staged copy survives and blocks the
        # retry a second time. Same trap as in worktree.py.
        checkout = run_git(["git", "checkout", "HEAD", "--", *lossless_modified])
        if checkout.ok:
            cleared.extend(lossless_modified)
        else:
            retained |= set(lossless_modified)
    for path in lossless_untracked:
        try:
            (repo_root / path).unlink()
        except OSError:
            retained.add(path)
        else:
            cleared.append(path)

    return PullBlockerRepair(cleared=tuple(sorted(cleared)), retained=tuple(sorted(retained)))


def _pull_ci_fleet_sibling(
    repo_root: Path,
    *,
    run_command: Callable[..., RunResult],
    timeout: int,
) -> None:
    """FF-pull ``origin/main`` in the declared ``ci-fleet`` sibling checkout.

    Deploy-clone half of issue #552: ``self_deploy`` only ever pulls the
    orchestrator checkout, so a dedicated daemon layout's editable ``ci_fleet``
    sibling would otherwise freeze at clone time, silently. Gated by
    ``supervisor.self_deploy_pull_ci_fleet`` (default false -- in a dev layout
    the sibling is a working repo whose HEAD must never be moved out from
    under a session).

    Fail-safe preconditions, checked here rather than trusted from config: the
    sibling must be on ``main`` with a clean tree, else the pull is skipped
    and the skip reason recorded. Every outcome -- pulled, unchanged, skipped,
    failed -- lands in events.db as ``self_deploy_ci_fleet_pull`` so sibling
    staleness is observable instead of silent; failures additionally log at
    WARNING. Deliberately excluded from the ``self_deploy_alarm`` failure
    streak: a sibling wedge bounds staleness but does not block orchestrator
    deploys, and conflating the two would page at the wrong severity.

    Never raises; the caller's deploy result is already decided.
    """
    payload: dict[str, object] = {"ok": False}
    try:
        from .ci_fleet_anchor import declared_ci_fleet_root

        declared_src = declared_ci_fleet_root()
        if declared_src is None:
            payload["skipped_reason"] = "no declared ci-fleet path source to pull"
        else:
            sibling = declared_src.parent
            branch_res = run_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=sibling, timeout_seconds=timeout
            )
            status_res = run_command(
                ["git", "status", "--porcelain"], cwd=sibling, timeout_seconds=timeout
            )
            payload["sibling"] = str(sibling)
            if not branch_res.ok or not status_res.ok:
                payload["skipped_reason"] = (
                    f"could not inspect sibling: {branch_res.error or status_res.error}"
                )
            elif branch_res.stdout.strip() != "main":
                payload["skipped_reason"] = (
                    f"sibling on branch {branch_res.stdout.strip()!r}, not main"
                )
            elif status_res.stdout.strip():
                payload["skipped_reason"] = "sibling tree is dirty"
            else:
                before = run_command(
                    ["git", "rev-parse", "HEAD"], cwd=sibling, timeout_seconds=timeout
                )
                pull_res = run_command(
                    ["git", "pull", "--ff-only", "origin", "main"],
                    cwd=sibling,
                    timeout_seconds=timeout,
                )
                after = run_command(
                    ["git", "rev-parse", "HEAD"], cwd=sibling, timeout_seconds=timeout
                )
                payload["from_sha"] = before.stdout.strip() if before.ok else None
                payload["to_sha"] = after.stdout.strip() if after.ok else None
                if pull_res.ok:
                    payload["ok"] = True
                    payload["changed"] = (
                        before.ok and after.ok and before.stdout.strip() != after.stdout.strip()
                    )
                else:
                    payload["error"] = _command_failure_message(
                        ["git", "pull", "--ff-only", "origin", "main"],
                        pull_res,
                        "ci-fleet sibling pull failed",
                    )
    except Exception as exc:  # noqa: BLE001 -- must never fail the deploy that succeeded
        payload["error"] = f"{type(exc).__name__}: {exc}"
    if not payload["ok"]:
        logger.warning("ci-fleet sibling pull did not run: %s", payload)
    log_event(_self_deploy_state_path(repo_root), "self_deploy_ci_fleet_pull", payload)


def _self_deploy_attempt(
    repo_root: Path,
    marker_path: Path,
    *,
    fleet_dir_override: str | None,
    run_command: Callable[..., RunResult],
    pull_timeout: int,
    sync_timeout: int,
    pull_ci_fleet: bool = False,
) -> SelfDeployResult:
    """Perform the real (non-preview) pull/diff/sync attempt.

    Split out from :func:`self_deploy` so the public entry point can wrap
    every return path once with events.db instrumentation and failure-streak
    tracking (issue #817 items 4-5), instead of duplicating that wrapping at
    each of this function's several return statements.
    """
    try:
        venv_check = _check_venv(repo_root)
        if not venv_check.ok:
            return venv_check
        venv_repaired = venv_check.venv_repaired

        before_cmd = ["git", "rev-parse", "HEAD"]
        before_res = run_command(before_cmd, cwd=repo_root, timeout_seconds=pull_timeout)
        if not before_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                venv_repaired=venv_repaired,
                error=_command_failure_message(before_cmd, before_res, "failed to read HEAD"),
            )
        before_sha = before_res.stdout.strip()

        pull_cmd = ["git", "pull", "--ff-only", "origin", "main"]
        pull_res = run_command(pull_cmd, cwd=repo_root, timeout_seconds=pull_timeout)
        if not pull_res.ok:
            repair = _repair_lossless_pull_blockers(
                repo_root, run_command=run_command, timeout=pull_timeout
            )
            if repair.acted:
                # Exactly one retry, never a loop: the blocker set is recomputed
                # from scratch on the next pass anyway, so looping here only
                # hammers a tree that is genuinely stuck.
                pull_res = run_command(pull_cmd, cwd=repo_root, timeout_seconds=pull_timeout)
                # A repair that WORKS is otherwise completely invisible: the
                # wedge simply stops happening and the pass logs an ordinary
                # success. That would make this fix a mask -- whatever keeps
                # depositing these files (a script landing untracked, a
                # half-applied cherry-pick) would go on doing so unobserved,
                # and the one place it used to be visible was the outage. So
                # record it on the success path too, not just in the failure
                # message. Best-effort, like every other log_event here: a
                # logging failure must never fail a deploy that worked.
                log_event(
                    _self_deploy_state_path(repo_root),
                    "self_deploy_blockers_cleared",
                    {
                        "cleared": list(repair.cleared),
                        "retained": list(repair.retained),
                        "pull_ok_after_retry": pull_res.ok,
                    },
                )
            if not pull_res.ok:
                detail = repair.describe()
                return SelfDeployResult(
                    ok=False,
                    pulled=False,
                    changed=False,
                    synced=False,
                    from_sha=before_sha,
                    venv_repaired=venv_repaired,
                    error=_command_failure_message(pull_cmd, pull_res, "pull failed")
                    + (f" -- {detail}" if detail else ""),
                )

        after_cmd = ["git", "rev-parse", "HEAD"]
        after_res = run_command(after_cmd, cwd=repo_root, timeout_seconds=pull_timeout)
        if not after_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=True,
                changed=False,
                synced=False,
                from_sha=before_sha,
                venv_repaired=venv_repaired,
                error=_command_failure_message(
                    after_cmd, after_res, "failed to read HEAD after pull"
                ),
            )
        after_sha = after_res.stdout.strip()

        if pull_ci_fleet:
            # Best-effort and deliberately outside the deploy's ok/error flow:
            # the orchestrator deploy above already succeeded, and a sibling
            # that cannot be pulled is bounded staleness, not a failed deploy.
            _pull_ci_fleet_sibling(repo_root, run_command=run_command, timeout=pull_timeout)

        marker = _read_marker(marker_path) if marker_path.exists() else None
        marker_from = marker.get("from_sha") if marker else None

        if before_sha == after_sha and not marker:
            return SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                head_changed=False,
                from_sha=before_sha,
                to_sha=after_sha,
                venv_repaired=venv_repaired,
                message="already up to date",
            )

        head_changed = before_sha != after_sha
        changed = head_changed or marker is not None

        changed_files: set[str] = set()
        if head_changed:
            diff_cmd = ["git", "diff", "--name-only", f"{before_sha}..{after_sha}"]
            diff_res = run_command(diff_cmd, cwd=repo_root, timeout_seconds=pull_timeout)
            if not diff_res.ok:
                return SelfDeployResult(
                    ok=False,
                    pulled=True,
                    changed=changed,
                    synced=False,
                    head_changed=head_changed,
                    from_sha=before_sha,
                    to_sha=after_sha,
                    venv_repaired=venv_repaired,
                    error=_command_failure_message(diff_cmd, diff_res, "diff failed"),
                )
            changed_files = {line.strip() for line in diff_res.stdout.splitlines() if line.strip()}

        dep_files_changed = bool(changed_files & _DEP_LOCK_FILES)
        if not dep_files_changed and not marker:
            return SelfDeployResult(
                ok=True,
                pulled=True,
                changed=changed,
                synced=False,
                head_changed=head_changed,
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
                head_changed=head_changed,
                from_sha=from_sha,
                to_sha=to_sha,
                venv_repaired=venv_repaired,
                message=f"sync deferred: {live_count} {runner_word} active",
                deferred=True,
            )

        # Persist marker before attempting sync so a crash between the pull and
        # the successful sync is retried on the next pass.
        _write_marker(marker_path, from_sha, to_sha)
        sync_cmd = ["uv", "sync"]
        sync_res = run_command(sync_cmd, cwd=repo_root, timeout_seconds=sync_timeout)
        if not sync_res.ok:
            return SelfDeployResult(
                ok=False,
                pulled=True,
                changed=changed,
                synced=False,
                head_changed=head_changed,
                from_sha=from_sha,
                to_sha=to_sha,
                venv_repaired=venv_repaired,
                error=_command_failure_message(sync_cmd, sync_res, "uv sync failed"),
            )

        _clear_marker(marker_path)
        return SelfDeployResult(
            ok=True,
            pulled=True,
            changed=changed,
            synced=True,
            head_changed=head_changed,
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

    # Same interpreter-anchored refusal as run_fleet_supervise: a repointed
    # editable means everything below runs unreviewed code (issue #974).
    from .venv_anchor import verify_interpreter_anchored_editables

    anchor = verify_interpreter_anchored_editables()
    if not anchor.ok:
        logger.error("VENV EDITABLE ANCHOR VIOLATION: %s", anchor.detail)
        log_event(
            app.paths.state_file,
            "venv_editable_anchor_violation",
            {"detail": anchor.detail},
        )
        return CommandResult(False, f"refusing to supervise: {anchor.detail}", {})
    logger.info("Venv editable anchor: %s", anchor.detail)

    # Record where ci_fleet was actually imported from plus the sibling
    # repo's HEAD and dirty-state (issue #954). Same instrumentation as
    # run_fleet_supervise: the editable .pth means the supervisor runs
    # whatever is saved in the sibling working tree, committed or not.
    # This records, it does not prevent. Best-effort, never raises.
    from .ci_fleet_anchor import ci_fleet_provenance_payload, ci_fleet_provenance_snapshot

    provenance = ci_fleet_provenance_snapshot()
    log_event(
        app.paths.state_file,
        "ci_fleet_provenance",
        ci_fleet_provenance_payload(provenance),
    )

    # Single-instance guard
    lock_path = layout.supervisor_lock_path(app.paths.root)
    lock = try_acquire_supervisor_lock(lock_path)
    if lock is None:
        return CommandResult(
            False,
            "supervisor already running (supervisor.lock held)",
            {},
        )

    # Issue #1339: ensure every LabelConfig-derived label exists on the repo
    # once at supervisor startup, so a new LabelConfig field converges to its
    # label with no operator action. ``ensure_labels`` is idempotent, records
    # failures as events (not exceptions), and never blocks the loop. Guarded
    # so a stand-in app without the method (tests) is tolerated.
    ensure = getattr(app, "ensure_labels", None)
    if callable(ensure):
        try:
            ensure()
        except Exception as exc:  # noqa: BLE001 — never block supervisor startup
            logger.warning("startup label ensure failed: %s", exc)

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

    sessions_dir = app.layout.sessions_dir
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
