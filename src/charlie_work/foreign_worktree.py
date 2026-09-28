"""Worktree ownership markers and foreign-checkout adoption.

Two halves of one question — "who owns the checkout serving this branch":

- The ``.charlie-writer.json`` marker family (``write_worktree_marker`` /
  ``read_worktree_marker`` / ``remove_worktree_marker`` plus the operator-claim
  sentinel constants) is how a checkout declares its writer. The orchestrator
  writes a worker marker after spawn; the operator CLI writes an
  ``operator-claim`` sentinel for ``charlie claim``.
- Foreign-checkout adoption (issue #1476): ``check_foreign_adoption`` decides
  whether a rework session may borrow a worktree the orchestrator did not
  create, ``find_branch_worktree`` answers where a branch is checked out in
  git's worktree registry, ``WorktreeForeignWriterError`` is the refusal
  signal, and ``log_foreign_adopted`` makes a borrowed dispatch observable.

This module is deliberately dependency-light — it must never import
``worktree`` (which imports it back for the adoption gate and marker users):
the guard it delegates to is injected as ``marker_guard``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from .config import WRITER_MARKER_FILENAME


# Sentinel values for operator claim markers. The operator marker intentionally
# does NOT record a real pid; liveness is derived from the ``operator_claimed_at``
# field in state.json.
OPERATOR_MARKER_SESSION_ID = "operator-claim"
OPERATOR_MARKER_KIND = "operator"


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically using a temp file + rename (issue #400)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def write_worktree_marker(
    worktree_path: Path,
    pid: int,
    session_id: str,
    kind: str = "worker",
    *,
    process_start_time: float | None = None,
) -> None:
    """Write a ``.charlie-writer.json`` marker into the worktree root.

    Records the process id and a session identifier so the orchestrator can
    detect a live foreign writer before dispatching a second one into the
    same worktree. ``kind`` distinguishes long-lived operator claim markers
    (``pid`` is a sentinel) from ordinary worker session markers.

    Issue #1423: ``process_start_time`` is the OS process creation timestamp
    captured immediately after spawn (same fingerprint the session sidecars
    store). It is read back by ``_reap_idle_foreign_writer`` and passed to
    ``kill_process_tree`` so the kill path re-verifies process identity
    immediately before terminating — the same PID-recycling defense every
    other ``kill_process_tree`` call site in this codebase uses. A marker
    without it (legacy marker written before this field existed, or an
    operator sentinel marker with ``pid == 0``) cannot be safely reaped and
    falls back to the block/escalate path instead.
    """
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    marker: dict[str, Any] = {
        "pid": pid,
        "session_id": session_id,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "kind": kind,
    }
    if process_start_time is not None:
        marker["process_start_time"] = process_start_time
    _write_json_atomic(marker_path, marker)


def read_worktree_marker(worktree_path: Path) -> dict[str, Any] | None:
    """Read the writer marker for ``worktree_path``, if any."""
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    if not marker_path.exists():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def remove_worktree_marker(worktree_path: Path, session_id: str | None = None) -> bool:
    """Remove the writer marker for ``worktree_path``.

    If ``session_id`` is provided, only removes the marker when its
    ``session_id`` matches, preventing an operator-claim marker from being
    wiped by a worker reap.
    """
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    if not marker_path.exists():
        return False
    if session_id is not None:
        marker = read_worktree_marker(worktree_path)
        if marker is None or marker.get("session_id") != session_id:
            return False
    try:
        marker_path.unlink()
        return True
    except OSError:
        return False


class WorktreeForeignWriterError(RuntimeError):
    """Raised when ``create_worktree`` is about to use a worktree that has a
    live writer marker belonging to a session the orchestrator does not own
    (e.g. an operator's editor or an out-of-band agent), OR when the target
    branch is already checked out in a worktree at a foreign path the
    orchestrator did not create (issue #1118). The launch shim surfaces this
    as ``failure_kind="worktree_foreign_writer"`` so the issue stays queued
    and the dispatch event log records the conflict.
    """

    def __init__(
        self,
        *,
        worktree_path: Path,
        pid: int | None,
        session_id: str | None,
        detail: str | None = None,
    ) -> None:
        self.worktree_path = worktree_path
        self.pid = pid
        self.session_id = session_id
        if pid is None and session_id is None:
            # Issue #1118: the branch is checked out in a worktree at a path
            # the orchestrator did not create — no writer marker to inspect.
            # Issue #1476: ``detail`` names the specific blocker when the
            # refusal came from the adoption gate rather than the blanket
            # path check, so the escalation payload says why this checkout
            # cannot be borrowed.
            message = (
                f"worktree {worktree_path} is a foreign checkout the "
                f"orchestrator did not create; refusing to adopt it"
            )
            if detail:
                message = f"{message} ({detail})"
            super().__init__(message)
        else:
            super().__init__(
                f"worktree {worktree_path} has a live foreign writer "
                f"(pid={pid}, session_id={session_id})"
            )


def find_branch_worktree(existing_worktrees: list[dict], branch: str) -> dict | None:
    """The registered worktree whose checked-out branch is ``branch``, if any.

    Single matcher for "where is this branch checked out": porcelain
    ``branch`` values carry a ``refs/heads/`` prefix, normalized away here so
    both forms compare equal. The match must be exact — a suffix match like
    ``endswith(f"/{branch}")`` would alias ``team/agent/issue-5-x`` onto a
    request for ``agent/issue-5-x``, and that was only ever safe while the
    answer drove a refusal (issue #1118, fail-closed). Now that it also gates
    adoption (issue #1476), a false positive would dispatch a worker into a
    checkout holding a different branch — so every site that answers this
    question must agree on exact equality.
    """
    return next(
        (
            wt
            for wt in existing_worktrees
            if wt.get("branch", "").removeprefix("refs/heads/") == branch
        ),
        None,
    )


def check_foreign_adoption(
    foreign_path: Path,
    *,
    managed_path: Path,
    repo_root: Path,
    registered: list[dict],
    recovery: bool,
    marker_guard: Callable[[Path], None] | None = None,
) -> bool:
    """Issue #1476 gate: may a rework session run in this foreign checkout?

    Returns ``True`` when the checkout at ``foreign_path`` is safe to borrow;
    ``False`` when it merely *looks* foreign (e.g. a path canonicalization
    difference on the managed worktree). Raises ``WorktreeForeignWriterError``
    — the same blocked-environment signal the unconditional #1118 refusal
    produced — when the checkout is unsafe, with ``detail`` naming the
    specific blocker.

    ``marker_guard`` is the caller's writer-marker guard (the
    ``_check_worktree_writer_marker`` family in ``worktree.py``), injected so
    this module stays import-cycle-free; ``None`` disables the live-writer /
    operator-claim check exactly as a missing ``sessions_dir`` does on the
    managed path. In ``recovery`` mode the guard's stale-marker cleanup is
    deferred, not skipped: the marker is the adoption's proof of prior
    orchestrator ownership, so it is restored when the guard removes it and
    only the post-launch ``write_worktree_marker`` overwrite replaces it —
    otherwise any post-gate failure (a failed fetch, a failed launch whose
    teardown skips borrowed checkouts) strands the next recovery dispatch
    markerless and refused, re-creating the #1476 loop. ``registered`` is the
    porcelain worktree list — its first entry is always the repo's main
    worktree, the anchor for the "never dispatch into the checkout hosting
    the orchestrator" refusal.

    The dispatching-into-it checks live here; the shared rework flow adds the
    touching-its-contents checks (dirt, non-FF divergence), so nothing about a
    borrowed checkout is ever mutated before every veto has had its say.
    """

    def _refuse(detail: str) -> NoReturn:
        raise WorktreeForeignWriterError(
            worktree_path=foreign_path, pid=None, session_id=None, detail=detail
        )

    resolved_path = foreign_path.resolve()
    # Never dispatch a worker into the checkout that hosts the orchestrator —
    # that is the repo's own main worktree (always the first ``git worktree
    # list`` entry) or, when the caller itself runs inside a linked worktree,
    # ``repo_root`` too.
    main_paths = {repo_root.resolve()}
    if registered:
        main_paths.add(Path(registered[0]["worktree"]).resolve())
    if resolved_path in main_paths:
        _refuse("the branch is checked out in the repo's main checkout")
    if resolved_path == managed_path.resolve():
        # Same directory as the managed target, reached via a non-canonical
        # path (e.g. an 8.3 short-name component): this is our own worktree,
        # not a foreign one.
        return False
    if not foreign_path.is_dir():
        # Registered but gone (git would call it prunable): it cannot host a
        # session. Only the operator/prune can resolve this.
        _refuse("the registered worktree directory is missing (prunable)")
    marker = read_worktree_marker(foreign_path)
    if recovery:
        # Recovery semantics presume the leftover worktree is ours — its dirt
        # is a dead worker's partial work to continue from, and the non-FF
        # reset below is unconditional. Only a worker-kind marker (real pid +
        # session id, never an operator sentinel/claim) proves a prior
        # orchestrator session ran here; anything else leaves the checkout
        # foreign.
        is_worker_marker = (
            marker is not None
            and marker.get("kind", "worker") == "worker"
            and isinstance(marker.get("session_id"), str)
            and not marker["session_id"].startswith("operator-")
            and isinstance(marker.get("pid"), int)
            and marker["pid"] > 0
        )
        if not is_worker_marker:
            _refuse("recovery cannot prove prior orchestrator ownership (no worker writer marker)")
    if marker_guard is not None:
        # The ordinary writer-marker guard, at the foreign path: a live
        # foreign writer or an active operator claim still refuses; a stale
        # marker is cleaned; an idle fleet writer may be reaped exactly as on
        # the managed path.
        marker_guard(foreign_path)
        if recovery and marker is not None and read_worktree_marker(foreign_path) is None:
            # The guard's stale-marker cleanup (or an idle-writer reap) just
            # deleted the worker marker that IS the recovery ownership proof.
            # Every post-gate step can still fail — the rework fetch, the
            # non-FF reset, or the launch itself, whose teardown skips
            # borrowed checkouts — and a markerless retry is refused above as
            # "cannot prove prior orchestrator ownership": the #1476 stuck
            # loop again. Cleanup is deferred, not skipped: the worker's
            # post-spawn ``write_worktree_marker`` overwrites this file on a
            # successful launch, so restoring the proof here leaves nothing
            # stale behind.
            _write_json_atomic(foreign_path / WRITER_MARKER_FILENAME, marker)
    return True


def log_foreign_adopted(
    state_file: Path,
    *,
    issue_number: int | None,
    worktree_path: Path,
    branch: str,
    recovery: bool,
) -> None:
    """Emit the ``worktree_foreign_adopted`` instrumentation event.

    Best-effort: adoption is a borrowed-checkout dispatch — the teardown /
    ownership semantics differ from a managed worktree, so it must be
    observable, but a failure to emit must never fail the adoption.
    """
    try:
        from .instrumentation import log_event

        log_event(
            state_file,
            "worktree_foreign_adopted",
            {
                "issue_number": issue_number,
                "worktree_path": str(worktree_path),
                "branch": branch,
                "recovery": recovery,
            },
        )
    except Exception:  # noqa: BLE001 — instrumentation is best-effort
        pass
