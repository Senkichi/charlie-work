"""Shared helpers for the ``test_worker.py`` family of test modules.

Hoisted out of ``tests/test_worker.py`` under the #1574 Track-1 shoulder split:
``_wg`` is consumed by tests in both ``test_worker.py`` and the
``test_worker_stalled_sessions.py``/``test_worker_rate_limit_deferral.py``
siblings, and test modules may not import from each other, so the shared
``WriteGate`` builder lives here. ``_make_stalled_devin_session`` and
``_stale_devin_probe`` are shared by the stalled-session and rate-limit
deferral siblings, so they live here too.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    _write_json,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _make_stalled_devin_session(
    tmp_path: Path,
    issue_number: int,
    log_text: str,
    *,
    rate_limit_defer_until: str | None = None,
) -> tuple[Path, Path, Path]:
    """Create a sessions directory, sidecar, and stale log for a live worker."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    # Set mtime to 30 minutes ago so the log looks stalled at the default
    # 20-minute stall threshold.
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("devin", "--prompt-file", str(tmp_path / "prompt.md")),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=31)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at=old_time.isoformat().replace("+00:00", "Z"),
        log_bytes=log_path.stat().st_size,
        rate_limit_defer_until=rate_limit_defer_until,
    )
    _write_json(sidecar_path, record.to_dict())
    state_file = tmp_path / "state.json"
    return sessions_dir, state_file, log_path


def _stale_devin_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
    """Return a probe that is stale (not fresh) and not all-errored.

    Issue #307: a worker with a stale sidecar log and a stale real-session
    activity signal must still be classified as STALLED. Tests that exercise
    the rate-limit defer path must not be tripped up by an all-errored probe,
    which now defers to avoid the fail-open bug in Signal 3.
    """
    now = datetime.now(UTC)
    timestamp = now - timedelta(minutes=30)
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=timestamp,
                staleness_seconds=(now - timestamp).total_seconds(),
                error=None,
            ),
        )
    )
