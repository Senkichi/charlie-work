"""``fleet status`` workers section: session listing, Claude Code rework layout, and real-activity-probe corroboration.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from _fakes_github import FakeGitHub
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import DevinConfig, OrchestratorConfig, PostMortemConfig, WatchdogConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_status_includes_workers_section(tmp_path: Path) -> None:
    """Issue #167: status (roll-call) should include workers section with health classification."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubWithWorkers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithWorkers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake live session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with recent mtime (not stalled)
    log_file = sessions_dir / "issue-167.log"
    log_file.write_text("working on issue\n", encoding="utf-8")

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-167.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 167,
                "branch": "agent/issue-167",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 12345,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive to return True for PID 12345
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.status()

    # Check that workers section is present
    assert "workers" in result.data
    assert isinstance(result.data["workers"], list)
    assert len(result.data["workers"]) == 1

    # Check worker entry has required fields
    worker = result.data["workers"][0]
    assert worker["issue"] == 167
    assert worker["adapter"] == "devin"
    assert worker["repo"] == "test-owner/test-repo"
    assert "health" in worker
    assert "runtime_seconds" in worker
    assert "last_activity_at" in worker
    assert worker["tool_calls"] is None  # Devin has no structured stream
    assert worker["tokens"] is None
    assert worker["cost_usd"] is None


def test_status_workers_section_claude_code_rework_layout(tmp_path: Path) -> None:
    """Issue #329 (F1): status()'s _summarize_worker must use the canonical
    events.jsonl derivation for rework-layout claude-code sessions too.

    A rework claude-code session logs to ``issue-<n>-rework.claude.log``, with
    its structured events at ``issue-<n>-rework.events.jsonl`` -- not
    ``issue-<n>.events.jsonl``, which the old rework=False-only
    ``_events_path(sessions_dir, issue_number)`` derivation would read instead
    (a stale prior attempt's tool_calls/tokens/cost_usd, or nothing). This
    test plants both a stale non-rework events.jsonl and the real rework
    sibling, and asserts the workers section reports the rework sibling's
    usage, not the stale one's.
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithWorkers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithWorkers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    issue_number = 329
    log_file = sessions_dir / f"issue-{issue_number}-rework.claude.log"
    log_file.write_text("reworking issue\n", encoding="utf-8")

    # Stale events.jsonl from a prior (non-rework) attempt: different usage.
    stale_events_file = sessions_dir / f"issue-{issue_number}.events.jsonl"
    stale_events_file.write_text(
        '{"type": "tool_call", "tokens": 111, "cost_usd": 0.11}\n',
        encoding="utf-8",
    )

    # The real rework events.jsonl sibling: the usage that must be reported.
    events_file = sessions_dir / f"issue-{issue_number}-rework.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 987654, "cost_usd": 12.34}\n{"type": "tool_call"}\n',
        encoding="utf-8",
    )

    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path="/fake/path",
        prompt_path="/fake/prompt",
        command=("claude", "-p"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
    )
    sidecar = sessions_dir / f"issue-{issue_number}-rework.claude.json"
    sidecar.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    assert len(result.data["workers"]) == 1
    worker = result.data["workers"][0]
    assert worker["issue"] == issue_number
    assert worker["adapter"] == "claude-code"
    assert worker["tool_calls"] == 2
    assert worker["tokens"] == 987654
    assert worker["cost_usd"] == 12.34


@pytest.mark.real_activity_probe_live
def test_status_workers_not_killed_when_real_activity_probe_fresh(tmp_path: Path) -> None:
    """Issue #301 status()-path wiring: a claude-code worker whose sidecar log is
    frozen but whose events.jsonl sibling carries fresh activity must be
    reported healthy (not stalled) in the status() workers section (~1616).

    A future edit that drops the ``probe`` argument from the
    classify_worker_health call inside status(), or that neuters the
    claude_events_jsonl Source-3 construction in
    post_mortem.real_activity_for_worker, must make this test fail (the
    worker reports health="stalled") rather than silently reverting to
    mtime-only classification.

    Marked ``real_activity_probe_live`` so the autouse
    ``_stub_real_activity_probe_for_stalled_tests`` fixture leaves
    ``real_activity_probe_for`` unstubbed for this test only (rename-safe
    opt-out; issue #307 non-blocking cleanup).
    """
    from datetime import timedelta

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 303

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_path.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_path, (time.time(), fresh_time.timestamp()))

    sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": str(tmp_path / "worktree"),
                "prompt_path": str(tmp_path / "prompt.md"),
                "command": ["claude", "prompt.md"],
                "pid": 77777,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                "log_path": str(log_path),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    workers = [w for w in result.data["workers"] if w["issue"] == issue_number]
    assert len(workers) == 1
    assert workers[0]["health"] == "healthy"


@pytest.mark.real_activity_probe_live
def test_status_workers_surfaces_corroboration_alive_but_polling(tmp_path: Path) -> None:
    """Issue #1346: an alive-but-polling worker (stale sidecar log mtime, fresh
    events.jsonl corroboration) must be visibly distinguishable from a genuinely
    stalled one (stale log AND stale corroboration) in the status()/roll-call
    workers section.

    Both workers share the same stale sidecar log mtime -- the only signal a
    log-mtime monitor sees -- so pre-#1346 they printed identically. After
    #1346 the workers section carries the watchdog's corroboration probe
    verdict (``last_corroborated_activity_at`` / ``last_corroborated_activity_source``
    / ``corroboration_fresh``) alongside ``health``, all derived from the same
    ``real_activity_probe_for`` + ``classify_worker_health`` code path the
    watchdog uses. The alive-but-polling worker reports ``health="healthy"`` +
    ``corroboration_fresh=True``; the stalled worker reports
    ``health="stalled"`` + ``corroboration_fresh=False``.

    Marked ``real_activity_probe_live`` so the autouse stale-probe stub fixture
    leaves ``real_activity_probe_for`` unstubbed and the real events.jsonl
    source is exercised.
    """
    from datetime import timedelta

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)

    def _plant_claude_worker(issue_number: int, *, fresh_corroboration: bool) -> None:
        log_path = sessions_dir / f"issue-{issue_number}.claude.log"
        log_path.write_text("Working on task...\nLast line", encoding="utf-8")
        # Stale sidecar log mtime for BOTH workers -- this is the signal that
        # log-mtime monitors cannot disambiguate.
        os.utime(log_path, (time.time(), old_time.timestamp()))

        events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
        ts = fresh_time if fresh_corroboration else old_time
        events_path.write_text(
            f'{{"type": "tool_call", "timestamp": "{ts.isoformat()}"}}\n',
            encoding="utf-8",
        )
        os.utime(events_path, (time.time(), ts.timestamp()))

        sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": f"agent/issue-{issue_number}",
                    "worktree_path": str(tmp_path / f"worktree-{issue_number}"),
                    "prompt_path": str(tmp_path / "prompt.md"),
                    "command": ["claude", "prompt.md"],
                    "pid": 70000 + issue_number,
                    "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                    "log_path": str(log_path),
                    "error": None,
                    "failure_kind": None,
                    "process_start_time": 1710000000.0,
                    "reclaimed": None,
                    # last_activity_at is the sidecar's stored log mtime -- the
                    # stale signal a log-mtime monitor reads. Both workers
                    # carry the same stale value so the only disambiguator is
                    # the corroboration probe.
                    "last_activity_at": old_time.isoformat(),
                    "log_bytes": len("Working on task...\nLast line"),
                }
            ),
            encoding="utf-8",
        )

    # Worker 1346: alive-but-polling (stale log, fresh corroboration).
    _plant_claude_worker(1346, fresh_corroboration=True)
    # Worker 1347: genuinely stalled (stale log, stale corroboration).
    _plant_claude_worker(1347, fresh_corroboration=False)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    by_issue = {w["issue"]: w for w in result.data["workers"]}
    assert set(by_issue) == {1346, 1347}

    alive_polling = by_issue[1346]
    # Acceptance criterion 1: visibly distinguishable. The stale log mtime is
    # identical to the stalled worker's, but the corroboration fields differ.
    assert alive_polling["health"] == "healthy"
    assert alive_polling["corroboration_fresh"] is True
    assert alive_polling["last_corroborated_activity_at"] is not None
    assert alive_polling["last_corroborated_activity_source"] is not None

    stalled = by_issue[1347]
    # Acceptance criterion 3: stale-both classifies stalled.
    assert stalled["health"] == "stalled"
    assert stalled["corroboration_fresh"] is False

    # The distinguishing signal: same stale last_activity_at (log mtime), but
    # the corroboration fields diverge -- exactly the gap #1346 closes.
    assert alive_polling["last_activity_at"] is not None
    assert stalled["last_activity_at"] is not None
    assert alive_polling["corroboration_fresh"] is not stalled["corroboration_fresh"]


def test_status_workers_empty_when_no_live_sessions(tmp_path: Path) -> None:
    """Issue #167: workers section should be empty list when no live sessions exist."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.status()

    # Check that workers section is present but empty
    assert "workers" in result.data
    assert isinstance(result.data["workers"], list)
    assert len(result.data["workers"]) == 0
