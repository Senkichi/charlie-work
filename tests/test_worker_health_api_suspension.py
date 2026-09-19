"""API provider-suspension tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
the Signal 2.5 provider-suspended liveness override -- suspension
marks an api worker dead, never applies to claude-code, and a
genuine 429 or quoted/prose suspension phrase is not a kill.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import OrchestratorConfig
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
)


def test_classify_worker_health_api_provider_suspended_is_dead(tmp_path: Path) -> None:
    """Issue #1342: an api worker whose log tail carries a provider
    account-suspension signature is classified DEAD immediately (within one
    supervision pass), not left in the multi-minute stall backoff. The
    suspension is a terminal billing failure — the CLI is alive but stuck
    retrying a permanently-failing endpoint."""
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # The suspension message is NOT the last line — the CLI logs retry
    # attempts during backoff after it. Signal 2.5 scans the tail, not just
    # the last line, so it must still fire.
    log_file.write_text(
        "Working...\n"
        "Error: suspended due to insufficient balance, please recharge your "
        "account or check your plan and billing details.\n"
        "Retrying in 60s...\n",
        encoding="utf-8",
    )
    recent_start = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    view = WorkerView(
        adapter_kind="api",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider="example",
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_provider_suspended_not_for_claude_code(
    tmp_path: Path,
) -> None:
    """Issue #1342: the suspension DEAD signal is api-only — a claude-code
    worker with the same log is not killed (it does not route through a
    billable provider endpoint)."""
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "Working...\nError: suspended due to insufficient balance, please recharge.\n",
        encoding="utf-8",
    )
    recent_start = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Not DEAD from the suspension signal (api-only). A claude-code worker
        # with a fresh log is HEALTHY.
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_genuine_429_not_dead(tmp_path: Path) -> None:
    """Issue #1342 acceptance criterion 4: a genuine transient 429 rate-limit
    in an api worker's log must NOT trip the terminal DEAD signal — the
    existing stall/backoff behavior is preserved."""
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "Working...\nError: Reached overall message rate limit. resets in 5 minutes.\n",
        encoding="utf-8",
    )
    recent_start = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    view = WorkerView(
        adapter_kind="api",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider="example",
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # A fresh log with a transient rate limit is HEALTHY (not DEAD).
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_suspended_phrase_quoted_prose_not_killed(
    tmp_path: Path,
) -> None:
    """PR #1426 round-2 review: a LIVE api worker whose log tail contains the
    suspension phrase only as quoted/reviewed prose (not the session's actual
    terminal error) must NOT be killed as DEAD by Signal 2.5. The structural
    anchor requires the billing phrase to co-occur on the same log line with
    an HTTP 402 status or a CLI ``Error:`` line prefix; prose/code that merely
    quotes the trigger phrase does not start with ``Error:``. This is the
    self-inflicted-kill guard: a worker reviewing this very fix must not be
    killed mid-session."""
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # The suspension trigger appears only inside a code-string quote and a
    # prose sentence — neither line starts with ``Error:`` nor carries a 402.
    # The session is making normal forward progress (no terminal error).
    log_file.write_text(
        "Reviewing PR #1426...\n"
        '    log_path.write_text("Error: suspended due to insufficient '
        'balance, please recharge your account")\n'
        "The regex matches `suspended due to insufficient balance` and "
        "`recharge your account` phrases.\n"
        "Continuing review...\n",
        encoding="utf-8",
    )
    recent_start = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    view = WorkerView(
        adapter_kind="api",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider="example",
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Not DEAD — the quoted/prose lines have no structural anchor on the
        # same line, so Signal 2.5 does not fire. A fresh, progressing log is
        # HEALTHY.
        assert health == WorkerHealth.HEALTHY
