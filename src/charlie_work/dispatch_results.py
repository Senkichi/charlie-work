"""Worker-dispatch result values and the adapter lanes' launch-failure seam.

Extracted from ``adapters.py`` (issue #2229 rework): the worker-launch
errors-as-values boundary had pushed that module past the 800-line file-size
ratchet cap. This leaf holds the dispatch result value object
(``SessionDispatchResult``) and every site that constructs one — ``_result``
(the shared constructor), ``_record_result`` (launcher record -> result),
``_launch_exc_result`` (exception -> failed result + event) — plus
``_emit_launch_failed``, the ``launch_failed`` event emit for dispatch paths
that produce an error value without a record coming back from a
record-returning launch function (issue #2246).

``adapters.py`` re-exports every name here, so all existing
``charlie_work.adapters.<name>`` import paths and monkeypatch targets keep
resolving unchanged. This module is a leaf: it imports ``launch_events`` at
top and ``adapters`` under ``TYPE_CHECKING`` only (for ``SessionRequest`` /
``AdapterSettings`` annotations) — never at runtime, so no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import launch_events

if TYPE_CHECKING:
    from .adapters import AdapterSettings, SessionRequest


@dataclass(frozen=True)
class SessionDispatchResult:
    issue_number: int
    issue_title: str
    prompt_path: str
    branch_name: str
    adapter: str
    ok: bool
    command: str | list[str] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None
    pid: int | None = None  # Worker process PID for state-based liveness detection
    process_start_time: float | None = None  # Process creation time for PID recycling protection
    failure_kind: str | None = None  # stable machine-readable classification of a failure
    # Issue #1423: the worktree path the launch resolved to. Carried so the
    # blocked-environment reap path (``_try_reap_blocked_foreign_writer``) can
    # read the writer marker without re-deriving the path from the branch name.
    # Empty for adapters that never create a worktree (command/manual/dry-run).
    worktree_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_number": self.issue_number,
            "issue_title": self.issue_title,
            "prompt_path": self.prompt_path,
            "branch_name": self.branch_name,
            "adapter": self.adapter,
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "reclaimed": self.reclaimed,
            "pid": self.pid,
            "process_start_time": self.process_start_time,
            "failure_kind": self.failure_kind,
            "worktree_path": self.worktree_path,
        }


def _emit_launch_failed(
    repo_root: Path,
    settings: AdapterSettings,
    request: SessionRequest,
    *,
    harness: str,
    error_class: str,
    error: str,
    model: str = "",
) -> None:
    """Issue #2246: emit one ``launch_failed`` event for a dispatch path that
    produces an error value *without* a record coming back from a
    record-returning launch function (prompt reads, missing config, render
    failures, the command adapter's blocking run, unsupported harnesses, and
    an exception raised -- not recorded -- inside the launch function).
    Paths that reach ``launch_devin_session``/``launch_claude_worker``/
    ``launch_api_worker`` and get a record back must not call this -- the
    error-record seam already emitted.

    Every ``dispatch_sessions`` result is a worker launch (rescue-tier
    dispatches included), so ``role`` is fixed "worker" here.
    """
    launch_events.emit_launch_failed(
        launch_events.state_path_for(repo_root, settings.config),
        role="worker",
        harness=harness,
        model=model,
        issue_number=request.issue_number,
        error_class=error_class,
        error=error,
    )


def _record_result(request: SessionRequest, adapter: str, record: Any) -> SessionDispatchResult:
    """Map a launcher record into a ``SessionDispatchResult`` -- the same
    shape for every record-returning adapter lane (errors arrive as values on
    the record; ``ok`` is failure unless a pid came back)."""
    ok = record.error is None and record.pid is not None
    return _result(
        request,
        adapter=adapter,
        ok=ok,
        command=list(record.command),
        error=record.error if not ok else None,
        reclaimed=record.reclaimed,
        pid=record.pid,
        process_start_time=record.process_start_time,
        failure_kind=record.failure_kind,
        worktree_path=record.worktree_path,
    )


def _launch_exc_result(
    repo_root: Path,
    settings: AdapterSettings,
    request: SessionRequest,
    *,
    adapter: str,
    model: str,
    exc: Exception,
) -> SessionDispatchResult:
    """The adapter lanes' errors-as-values boundary (issue #2229).

    An unexpected raise inside a record-returning launch function comes back
    as a failed result plus exactly one ``launch_failed`` event -- never a
    raise (CLAUDE.md invariant: errors from external processes come back as
    values).
    """
    error = f"launch failed: {exc}"
    _emit_launch_failed(
        repo_root,
        settings,
        request,
        harness=adapter,
        model=model,
        error_class=launch_events.LAUNCH_ERR_INTERNAL,
        error=error,
    )
    return _result(request, adapter=adapter, ok=False, error=error)


def _result(
    request: SessionRequest,
    *,
    adapter: str,
    ok: bool,
    command: str | list[str] | None = None,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    reclaimed: str | None = None,
    pid: int | None = None,
    process_start_time: float | None = None,
    failure_kind: str | None = None,
    worktree_path: str = "",
) -> SessionDispatchResult:
    return SessionDispatchResult(
        issue_number=request.issue_number,
        issue_title=request.issue_title,
        prompt_path=str(request.prompt_path),
        branch_name=request.branch_name,
        adapter=adapter,
        ok=ok,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
        reclaimed=reclaimed,
        pid=pid,
        process_start_time=process_start_time,
        failure_kind=failure_kind,
        worktree_path=worktree_path,
    )
