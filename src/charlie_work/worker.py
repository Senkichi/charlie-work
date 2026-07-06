"""Adapter-agnostic worker abstraction for unified fleet supervision.

This module provides a unified view of worker sessions across all adapters
(devin-shell, claude-code, and future adapters). It collapses the duplicated
adapter-iteration loops in workflow.py into a single abstraction point.
"""

from dataclasses import dataclass
from os import stat_result
from pathlib import Path

from .claude_code import (
    ClaudeWorkerRecord,
    _sidecar_path as claude_sidecar_path,
    is_worker_alive,
    read_worker_records,
)
from .devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    is_session_alive,
    read_session_records,
)


@dataclass(frozen=True)
class WorkerView:
    """Adapter-agnostic view of a worker session.

    This frozen dataclass provides a unified shape for worker records across
    all adapters, enabling single-loop iteration over the entire fleet without
    duplicating devin-shell/claude-code-specific code.
    """

    adapter_kind: str  # "devin" | "claude-code"
    issue_number: int
    repo_key: (
        str  # required — cross-repo disambiguation (fleet work); "" for single-repo callers today
    )
    pid: int | None
    started_at: str
    process_start_time: float | None
    log_path: str
    error: str | None
    failure_kind: str | None
    reclaimed: str | None

    def is_alive(self) -> bool:
        """Check whether the process behind this worker is still running.

        Dispatches to the adapter-specific liveness probe based on adapter_kind.
        Preserves the existing PID + process_start_time recycling-safe check.
        """
        if self.adapter_kind == "devin":
            # Reconstruct a minimal SessionRecord for the liveness probe
            record = SessionRecord(
                issue_number=self.issue_number,
                branch="",  # Not used by is_session_alive
                worktree_path="",  # Not used by is_session_alive
                prompt_path="",  # Not used by is_session_alive
                command=(),  # Not used by is_session_alive
                pid=self.pid,
                started_at=self.started_at,
                log_path=self.log_path,
                error=self.error,
                failure_kind=self.failure_kind,
                process_start_time=self.process_start_time,
                reclaimed=self.reclaimed,
            )
            return is_session_alive(record)
        elif self.adapter_kind == "claude-code":
            # Reconstruct a minimal ClaudeWorkerRecord for the liveness probe
            record = ClaudeWorkerRecord(
                issue_number=self.issue_number,
                branch="",  # Not used by is_worker_alive
                worktree_path="",  # Not used by is_worker_alive
                prompt_path="",  # Not used by is_worker_alive
                command=(),  # Not used by is_worker_alive
                pid=self.pid,
                started_at=self.started_at,
                log_path=self.log_path,
                error=self.error,
                failure_kind=self.failure_kind,
                process_start_time=self.process_start_time,
                reclaimed=self.reclaimed,
            )
            return is_worker_alive(record)
        else:
            # Unknown adapter kind - conservatively treat as dead
            return False

    def log_stat(self) -> stat_result | None:
        """Stat() the log_path, swallow OSError -> None.

        Returns the os.stat_result for the log file if it exists and is accessible,
        None otherwise. This is useful for checking log file mtime for stall detection.
        """
        try:
            return Path(self.log_path).stat()
        except OSError:
            return None

    def reap_sidecar(self, sessions_dir: Path) -> None:
        """Delete the sidecar file for this worker to prevent phantom sessions.

        Dispatches to the adapter-specific sidecar path function based on adapter_kind.
        Best-effort cleanup: OSError is swallowed to avoid failing the entire dead-session
        classification loop if a single unlink fails (the sidecar will be reaped on the
        next cycle).

        This is called after a session is detected as dead and classified to prevent
        phantom sessions from PID recycling (issue #113).
        """
        if self.adapter_kind == "devin":
            sidecar_path = devin_sidecar_path(sessions_dir, self.issue_number)
        elif self.adapter_kind == "claude-code":
            sidecar_path = claude_sidecar_path(sessions_dir, self.issue_number)
        else:
            # Unknown adapter kind - nothing to reap
            return

        try:
            sidecar_path.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup - don't fail if unlink fails
            pass


def _from_session_record(record: SessionRecord, repo_key: str) -> WorkerView:
    """Convert a SessionRecord to a WorkerView."""
    return WorkerView(
        adapter_kind="devin",
        issue_number=record.issue_number,
        repo_key=repo_key,
        pid=record.pid,
        started_at=record.started_at,
        process_start_time=record.process_start_time,
        log_path=record.log_path,
        error=record.error,
        failure_kind=record.failure_kind,
        reclaimed=record.reclaimed,
    )


def _from_claude_record(record: ClaudeWorkerRecord, repo_key: str) -> WorkerView:
    """Convert a ClaudeWorkerRecord to a WorkerView."""
    return WorkerView(
        adapter_kind="claude-code",
        issue_number=record.issue_number,
        repo_key=repo_key,
        pid=record.pid,
        started_at=record.started_at,
        process_start_time=record.process_start_time,
        log_path=record.log_path,
        error=record.error,
        failure_kind=record.failure_kind,
        reclaimed=record.reclaimed,
    )


def iter_workers(sessions_dir: Path, *, repo_key: str = "") -> list[WorkerView]:
    """Read every devin-shell + claude-code sidecar in sessions_dir and return
    a unified, adapter-tagged list of WorkerView.

    Malformed sidecars are skipped (matches the existing read_session_records/
    read_worker_records contract) — never raises.

    Args:
        sessions_dir: Directory containing session sidecar files
        repo_key: Cross-repo disambiguation key (empty string for single-repo)

    Returns:
        List of WorkerView objects, one per valid sidecar file
    """
    workers: list[WorkerView] = []

    # Read devin-shell sidecars
    for record in read_session_records(sessions_dir):
        workers.append(_from_session_record(record, repo_key))

    # Read claude-code sidecars
    for record in read_worker_records(sessions_dir):
        workers.append(_from_claude_record(record, repo_key))

    return workers


__all__ = [
    "WorkerView",
    "iter_workers",
]
