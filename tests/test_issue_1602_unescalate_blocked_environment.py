"""Issue #1602: ``charlie unescalate`` must clear stale ``blocked_environment_at``.

``dispatch_blocked_environment`` escalations record each pre-launch
environment block (e.g. a foreign worktree) in the issue entry's
``blocked_environment_at`` list. The rework pre-check in
``_dispatch_rework_impl`` escalates without dispatching once the in-window
count reaches ``max_auto_redispatch``. Before this fix, ``charlie unescalate``
released the escalation (status + labels) but left the timestamps in place,
so the very next ``rework_requested`` pass re-escalated on the stale count --
one operator release bought zero fresh dispatch attempts.

The fix treats an operator unescalate as the statement "the environment is
fixed": ``blocked_environment_at`` is popped on re-arm (mirroring what a
successful foreign-writer reap already does at the guard sites) and the
``unescalated`` event payload records what was cleared.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _unescalate_fixtures import _events
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.dispatch_selection import _windowed_blocked_environment_at
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _in_window_timestamps(count: int, *, window_minutes: int = 240) -> list[str]:
    """``count`` ISO timestamps strictly inside ``window_minutes`` of now.

    Derived from ``datetime.now(UTC)`` so the windowed filter in
    ``_windowed_blocked_environment_at`` cannot rot as the real clock moves
    (test-hygiene rule: never hardcode calendar dates in seed fixtures).
    """
    now = datetime.now(UTC)
    return [
        (now - timedelta(minutes=window_minutes - 10 - i * 5)).isoformat().replace("+00:00", "Z")
        for i in range(count)
    ]


def _rework_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=3, redispatch_window_minutes=240),
    )


def _rework_app(tmp_path: Path, fake_gh: FakeGitHub) -> OrchestratorApp:
    config = _rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _write_rework_prompt(tmp_path: Path) -> None:
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")


class _ReworkGitHub(FakeGitHub):
    def __init__(self, config: OrchestratorConfig) -> None:
        super().__init__()
        self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]


def _seed_escalated_blocked_environment(
    app: OrchestratorApp, blocked_timestamps: list[str]
) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "escalated",
            "escalation_reason": "dispatch_blocked_environment",
            "reason_class": "mechanical",
            "blocked_environment_at": list(blocked_timestamps),
        }
        save_state(app.paths.state_file, state)


# --- the fix: unescalate clears blocked_environment_at ---


def test_unescalate_clears_blocked_environment_at_and_records_it_in_event(
    tmp_path: Path,
) -> None:
    """An operator re-arm pops ``blocked_environment_at`` and the
    ``unescalated`` event payload carries the reset so the audit trail
    explains why the next rework pass dispatched instead of re-escalating."""
    config = _rework_config()
    app = _rework_app(tmp_path, _ReworkGitHub(config))
    timestamps = _in_window_timestamps(3)
    _seed_escalated_blocked_environment(app, timestamps)

    result = app.unescalate(issue_number=123)

    assert result.ok is True
    assert result.data["changed"] is True
    state = load_state(app.paths.state_file)
    issue_entry = state["issues"]["123"]
    # Popped on re-arm -- the windowed view the rework pre-check reads is empty.
    assert "blocked_environment_at" not in issue_entry
    assert (
        _windowed_blocked_environment_at(
            issue_entry, window_minutes=config.watchdog.redispatch_window_minutes
        )
        == []
    )
    # The event payload records the reset.
    events = _events(state, "unescalate")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["blocked_environment_at_reset"] is True
    assert payload["blocked_environment_at_prior_count"] == 3


def test_unescalate_dry_run_reports_blocked_environment_reset_without_clearing(
    tmp_path: Path,
) -> None:
    """``--dry-run`` reports the would-be reset without touching state, so an
    operator can see the stale count before committing to the re-arm."""
    config = _rework_config()
    app = _rework_app(tmp_path, _ReworkGitHub(config))
    timestamps = _in_window_timestamps(3)
    _seed_escalated_blocked_environment(app, timestamps)

    result = app.unescalate(issue_number=123, dry_run=True)

    assert result.ok is True
    assert result.data["changed"] is False
    assert result.data["blocked_environment_at_reset"] is True
    assert result.data["blocked_environment_at_prior_count"] == 3
    # State untouched.
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["blocked_environment_at"] == timestamps


# --- the fix buys a fresh dispatch attempt ---


def test_unescalate_buys_fresh_rework_dispatch_after_blocked_environment_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an operator release, the next ``rework_requested`` pass must
    attempt a dispatch instead of re-escalating on the stale count. This is
    the core acceptance bar from issue #1602: one operator release must buy
    at least one fresh dispatch attempt."""
    config = _rework_config()
    fake_gh = _ReworkGitHub(config)
    app = _rework_app(tmp_path, fake_gh)
    _seed_escalated_blocked_environment(app, _in_window_timestamps(3))
    _write_rework_prompt(tmp_path)

    # Operator release: clears the stale blocked_environment_at.
    release = app.unescalate(issue_number=123)
    assert release.ok and release.data["changed"]

    # The release returns the issue to the passive pr-open state. A subsequent
    # legitimate request_changes review flips it back to rework_requested --
    # the exact sequence in the #1585 incident (unescalate at 20:50, review
    # request_changes at 21:09, re-escalation at 21:16). Simulate that flip.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"]["status"] = "rework_requested"
        save_state(app.paths.state_file, state)

    dispatch_calls: list[int] = []

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        dispatch_calls.extend(request.issue_number for request in requests)
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                command=(sys.executable,),
                pid=4321,
                process_start_time=1.0,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework()

    # A dispatch was attempted (not escalated on the stale count).
    assert dispatch_calls == [123]
    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert state["issues"]["123"].get("escalation_reason") != "dispatch_blocked_environment"
    assert "blocked_environment_at" not in state["issues"]["123"]


# --- positive control: without the unescalate, the stale count still escalates ---


def test_positive_control_stale_blocked_environment_at_still_escalates_without_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current behaviour preserved: an issue in ``rework_requested`` with
    ``max_auto_redispatch`` in-window ``blocked_environment_at`` timestamps
    escalates with ``dispatch_blocked_environment`` on the next rework pass
    WITHOUT attempting a dispatch. This is the positive control for the
    fix above -- the same fixture still escalates when no release ran."""
    config = _rework_config()
    fake_gh = _ReworkGitHub(config)
    app = _rework_app(tmp_path, fake_gh)
    _write_rework_prompt(tmp_path)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "blocked_environment_at": _in_window_timestamps(3),
        }
        save_state(app.paths.state_file, state)

    dispatch_calls: list[int] = []

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        dispatch_calls.extend(request.issue_number for request in requests)
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                command=(sys.executable,),
                pid=4321,
                process_start_time=1.0,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework()

    # The pre-check escalated without dispatching -- no launch attempted.
    assert dispatch_calls == []
    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
