"""A terminal record proving a clean, completed exit suppresses log-tail throttle
classification (the #656 class, recurred 2026-09-29 on issue #2002).

The stall-reap lane calls ``update_worker_record_with_failure_classification``
without ``session_completed``; a worker whose issue is *about* rate limits ends
its log with completion prose quoting "rate limit", which armed a 76-minute
repo throttle. The helper now derives completion from the worker's own
terminal record -- but only when that record is for the same pid, exited 0,
and carries a worker outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.claude_code import update_worker_record_with_failure_classification
from charlie_work.process_utils import (
    terminal_record_proves_completion,
    worker_terminal_status_path,
)

PID = 1234
# A tail the classifier genuinely matches as rate_limited.
RATE_LIMIT_TAIL = (
    "Summary: fixed the orphan sweep.\n"
    "Error: Reached overall message rate limit. Please try again later. "
    "Your limit will reset in 10 minutes.\n"
)


def _setup(tmp_path: Path, terminal: dict[str, Any] | None) -> tuple[Path, Path]:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = sessions_dir / "issue-42-rework.claude.log"
    log_path.write_text(RATE_LIMIT_TAIL, encoding="utf-8")
    sidecar = sessions_dir / "issue-42.claude.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": PID,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(log_path),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    if terminal is not None:
        worker_terminal_status_path(sessions_dir, 42, "claude").write_text(
            json.dumps(terminal), encoding="utf-8"
        )
    return sessions_dir, sidecar


def _terminal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pid": PID,
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:16:00Z",
        "duration_seconds": 960.0,
        "worker_outcome": {"head_sha": "a" * 40, "pr_body": "## Summary"},
    }
    return {**base, **overrides}


def test_clean_completed_exit_is_not_rate_limited(tmp_path: Path) -> None:
    sessions_dir, sidecar = _setup(tmp_path, _terminal())

    kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert kind == "stalled"
    assert throttled_until is None
    assert json.loads(sidecar.read_text(encoding="utf-8"))["failure_kind"] == "stalled"


@pytest.mark.parametrize(
    "terminal",
    [
        None,  # no terminal record: legacy behavior
        _terminal(worker_outcome=None),  # exit 0 without a handoff is not proof
        _terminal(exit_code=1),  # non-zero exit
        _terminal(exit_code=None),  # exit status unknown
        _terminal(pid=PID + 1),  # stale record from an earlier attempt
    ],
    ids=["no-record", "no-outcome", "exit-1", "exit-unknown", "other-pid"],
)
def test_without_completion_proof_real_throttle_still_classified(
    tmp_path: Path, terminal: dict[str, Any] | None
) -> None:
    sessions_dir, _ = _setup(tmp_path, terminal)

    kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert kind == "rate_limited"
    assert throttled_until is not None


def test_proof_helper_never_raises_on_malformed_record(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    path = worker_terminal_status_path(sessions_dir, 42, "claude")
    path.write_text("{not json", encoding="utf-8")
    assert terminal_record_proves_completion(sessions_dir, 42, "claude", PID) is False
    path.write_text(json.dumps(_terminal(pid="garbage")), encoding="utf-8")
    assert terminal_record_proves_completion(sessions_dir, 42, "claude", PID) is False
    path.write_text(json.dumps(_terminal()), encoding="utf-8")
    assert terminal_record_proves_completion(sessions_dir, 42, "claude", None) is False
    assert terminal_record_proves_completion(sessions_dir, 42, "claude", PID) is True
