"""Review-dispatch caps, reviewer quota, deferral, and turn limits.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the gating half of the ``test_dispatch_reviews_*`` seam -- per-pass review
caps, reviewer-quota probe set/clear/rollback, quota deferral digests, and
turn-limit summary comments. Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _write_review_packet,
)
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import (
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dispatch_reviews_respects_local_process_cap(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: max_local_review_processes caps concurrent reviewer launches."""
    prs = [
        {
            "number": i,
            "title": f"Fix #{i}",
            "url": f"https://example.test/pull/{i}",
            "headRefName": f"agent/issue-{i}-fix",
            "baseRefName": "main",
            "headRefOid": f"sha-{i}",
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{i}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
        for i in range(300, 303)
    ]
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_local_review_processes=2),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    for pr in prs:
        _write_review_packet(tmp_path, pr["number"], pr["headRefOid"])

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 2
    assert result.data["selected_count"] == 2
    assert result.data["skipped_count"] == 1
    assert result.data["max_local_review_processes"] == 2
    assert len(launched) == 2


def test_dispatch_reviews_respects_max_concurrent_reviews(monkeypatch, tmp_path: Path) -> None:
    """max_concurrent_reviews caps the number of reviewers launched in a single pass."""
    prs = [
        {
            "number": i,
            "title": f"Fix #{i}",
            "url": f"https://example.test/pull/{i}",
            "headRefName": f"agent/issue-{i}-fix",
            "baseRefName": "main",
            "headRefOid": f"sha-{i}",
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{i}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
        for i in range(300, 303)
    ]
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            enabled=True,
            max_local_review_processes=0,
            max_concurrent_reviews=2,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    for pr in prs:
        _write_review_packet(tmp_path, pr["number"], pr["headRefOid"])

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *args, **kwargs: _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        ),
    )

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 2


def test_dispatch_reviews_defers_when_reviewer_quota_exhausted(tmp_path: Path) -> None:
    """When reviewer quota is exhausted and the probe window has not passed, defer without dispatching."""
    from datetime import UTC, datetime, timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["reviewer_quota"] = {"throttled_until": future, "probe_after": future}
        save_state(app.paths.state_file, state)

    result = app.dispatch_reviews()

    assert result.data["selected_count"] == 0
    assert result.data["launched_count"] == 0
    assert result.data.get("deferred_reason") == "reviewer_quota_probe_backoff"


def test_dispatch_reviews_quota_deferral_emits_one_shot_digest(
    monkeypatch, tmp_path: Path
) -> None:
    """The first deferred pass of a quota-exhaustion episode emits one
    REVIEWER_QUOTA_EXHAUSTED digest and persists reviewer_quota.alerted_at;
    subsequent deferred passes in the same episode emit nothing."""
    from charlie_work.config import NotifyConfig

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        notify=NotifyConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(tmp_path, 100, "sha-100")

    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["reviewer_quota"] = {"throttled_until": future, "probe_after": future}
        save_state(app.paths.state_file, state)

    captured: list[Any] = []
    monkeypatch.setattr(
        "charlie_work.workflow.emit_digest",
        lambda notify_config, digest: captured.append(digest),
    )

    result = app.dispatch_reviews()

    assert result.data.get("deferred_reason") == "reviewer_quota_probe_backoff"
    assert len(captured) == 1
    assert captured[0].transitions[0].health == "REVIEWER_QUOTA_EXHAUSTED"
    state_after = load_state(app.paths.state_file)
    assert state_after["reviewer_quota"].get("alerted_at") is not None

    # Second deferred pass in the same episode: no new digest.
    result2 = app.dispatch_reviews()

    assert result2.data.get("deferred_reason") == "reviewer_quota_probe_backoff"
    assert len(captured) == 1


def test_dispatch_reviews_probe_success_clears_reviewer_quota(monkeypatch, tmp_path: Path) -> None:
    """A successful reviewer verdict from a probe clears the global reviewer quota.

    Fix 1.1: the quota is cleared only when a verdict is actually recorded from
    a dead reviewer, not when a probe process merely starts. A session-limited
    reviewer starts successfully but dies seconds later — clearing on start
    caused a hot redispatch loop.
    """
    from datetime import UTC, datetime, timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    future_throttle = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    past_probe = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["reviewer_quota"] = {
            "throttled_until": future_throttle,
            "probe_after": past_probe,
        }
        # Simulate a prior dispatched reviewer that has died with a verdict in
        # its log — _reap_review_verdicts will record this verdict before the
        # dispatch pass, and the recorded verdict is what clears the quota.
        state["prs"]["100"] = {
            **state["prs"].get("100", {}),
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
        }
        save_state(app.paths.state_file, state)

    # Write a sidecar + log with a verdict so _reap_review_verdicts finds it.
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        'Review complete.\n```json\n{"decision": "approved", "summary": "LGTM"}\n```',
        encoding="utf-8",
    )
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt"),
                "command": ["claude"],
                "pid": 0,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "log_path": str(log_path),
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *args, **kwargs: _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        ),
    )

    app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    # The verdict was recorded, which clears the quota. The reaped PR is now
    # completed, so the queue is empty and dispatch returns early — but the
    # quota clearing happens before the queue check (fix 1.1).
    assert state.get("reviewer_quota", {}).get("throttled_until") is None
    # The verdict-reap recovery path must stamp the same recovery marker a
    # green flat probe would, so later dead-reviewer sweeps can suppress
    # backoff from throttle signatures that predate the recovery (issue #662).
    assert state.get("reviewer_quota", {}).get("last_probe_cleared_at") is not None


def test_dispatch_reviews_probe_failure_sets_reviewer_quota_and_rolls_back(
    monkeypatch, tmp_path: Path
) -> None:
    """A reviewer launch failure matching a usage-limit signature sets the global quota and rolls back the PR claim."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=kwargs.get("issue_number") or args[0],
            branch=kwargs.get("branch") or args[1],
            worktree_path="/fake/worktree",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p", "--permission-mode", "plan"),
            pid=None,
            started_at="2026-07-20T12:00:00Z",
            log_path="/fake/log.log",
            error="usage limit exceeded",
            process_start_time=1.0,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    assert result.data["launched_count"] == 0
    assert result.data.get("quota_hit") is True
    assert state["reviewer_quota"]["throttled_until"] is not None
    assert state["reviewer_quota"]["probe_after"] is not None
    # The PR should not be marked as a normal failed dispatch; its claim is rolled back.
    assert state["prs"]["100"].get("review_dispatch_status") is None


def test_dispatch_reviews_turn_limit_posts_summary_comment(monkeypatch, tmp_path: Path) -> None:
    """When a reviewer dies without a verdict but has analysis in events.jsonl,
    a summary PR comment is posted so the work is not lost."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            **state["prs"].get("100", {}),
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
        }
        save_state(app.paths.state_file, state)

    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)

    # Write a log with NO verdict block (simulates turn-limit exhaustion).
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "I reviewed the diff and checked the tests.\n"
        "The changes look reasonable but I need more turns to verify.\n",
        encoding="utf-8",
    )

    # Write events.jsonl with assistant messages and turn metrics. The session
    # must actually reach review_max_turns: since issue #588 a session short of
    # the cap is classified died_mid_session, and one with no turns at all is a
    # launch failure, so a turn-limit test has to seed a real turn-limit run.
    events_path = reviews_dir / "issue-100-review.events.jsonl"
    max_turns = app.config.review_dispatch.review_max_turns
    events_lines = [
        '{"type": "tool_call", "tokens": 300}',
        '{"type": "assistant_message", "content": "Let me read the diff first.", "tokens": 200}',
    ]
    # Pad to the turn cap, leaving the final two messages as the analysis the
    # summary comment is expected to surface.
    while len(events_lines) < max_turns - 1:
        events_lines.append(
            '{"type": "assistant_message", "content": "Checking another file.", "tokens": 200}'
        )
    events_lines.append(
        '{"type": "assistant_message", "content": "The changes look reasonable but I '
        'need more turns to verify the edge cases.", "tokens": 400}'
    )
    events_lines.append(
        '{"type": "assistant_message", "content": "I was unable to complete the '
        'review within the turn limit.", "tokens": 500}'
    )
    events_path.write_text("\n".join(events_lines) + "\n", encoding="utf-8")

    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt"),
                "command": ["claude"],
                "pid": 0,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "log_path": str(log_path),
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    # Capture PR comments.
    captured_comments: list[tuple[int, str]] = []

    class _CapturingGitHub(type(app.gh)):  # type: ignore[misc]
        def pr_comment(self, number: int, body_file: Path) -> None:
            captured_comments.append((number, body_file.read_text(encoding="utf-8")))

    app.gh = _CapturingGitHub()

    result = app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    # A summary comment should have been posted.
    assert len(captured_comments) == 1
    assert captured_comments[0][0] == 100
    comment_body = captured_comments[0][1]
    assert "no verdict produced" in comment_body
    assert "Recent analysis" in comment_body

    # The state flag should be set to prevent duplicate comments.
    assert state["prs"]["100"].get("review_turn_limit_summary_posted") is True

    # The missed list should include the turn_limit_summary_posted reason.
    missed = result.data.get("missed_verdicts", [])
    assert any(m.get("reason") == "turn_limit_summary_posted" for m in missed)


def test_dispatch_reviews_turn_limit_summary_not_posted_twice(monkeypatch, tmp_path: Path) -> None:
    """The turn-limit summary is only posted once per dispatch lifecycle."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            **state["prs"].get("100", {}),
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
            "review_turn_limit_summary_posted": True,
        }
        save_state(app.paths.state_file, state)

    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text("Some analysis without a verdict.\n", encoding="utf-8")
    events_path = reviews_dir / "issue-100-review.events.jsonl"
    events_path.write_text(
        '{"type": "assistant_message", "content": "Some analysis.", "tokens": 100}\n',
        encoding="utf-8",
    )
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt"),
                "command": ["claude"],
                "pid": 0,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "log_path": str(log_path),
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    captured_comments: list[tuple[int, str]] = []

    class _CapturingGitHub(type(app.gh)):  # type: ignore[misc]
        def pr_comment(self, number: int, body_file: Path) -> None:
            captured_comments.append((number, body_file.read_text(encoding="utf-8")))

    app.gh = _CapturingGitHub()

    app.dispatch_reviews()

    # No comment should be posted — the flag was already set.
    assert len(captured_comments) == 0
