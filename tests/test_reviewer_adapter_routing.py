"""Tests for reviewer adapter routing through non-claude-code harnesses
(issue #1513).

Extracted from ``tests/test_charlie_work.py`` as part of the attachment-contracts
ratchet remedy (issue #1616): ``test_charlie_work.py`` exceeded its baselined
ceiling by two members, and the over-ceiling tests are the #1535 reviewer-harness
routing tests -- the launch-side routing test and the read-side verdict-reap
test -- which belong with the reviewer-adapter-routing subject rather than the
monolithic ``test_charlie_work.py`` module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _review_fixtures import (
    _dispatch_reviews_app,
    _set_review_dispatched_state,
    _write_review_packet,
)

from charlie_work.devin_shell import SessionRecord
from charlie_work.state import load_state


def test_dispatch_reviews_routes_through_devin_shell_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1513: reviewer.harness="devin-shell" must actually route the
    reviewer launch through launch_devin_session (with review=True and the
    PR's head_sha) instead of the previously-hardcoded launch_claude_worker.
    Before this issue, reviewer.harness could only ever be "claude-code" --
    setting anything else was rejected by config validation, so there was no
    way to even reach this call site with a different harness. This pins the
    launch-side half of the fix: config accepting the value is not enough if
    dispatch never routes through it (the "wired, not just loosened"
    requirement)."""
    from dataclasses import replace

    from charlie_work.config import ReviewerRoleConfig

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
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    app.config = replace(app.config, reviewer=ReviewerRoleConfig(harness="devin-shell"))
    _write_review_packet(tmp_path, 100, "sha-100")

    captured: list[dict[str, Any]] = []

    def fake_devin_launch(*args: Any, **kwargs: Any) -> SessionRecord:
        captured.append(kwargs)
        return SessionRecord(
            issue_number=args[0],
            branch=args[1],
            worktree_path="/fake/review-checkout",
            prompt_path=str(args[2]),
            command=("devin", "--prompt-file", "{prompt_path}", "--print"),
            pid=12345,
            started_at="2026-07-06T12:00:00Z",
            log_path="/fake/log.log",
            error=None,
            process_start_time=1.0,
        )

    def fail_if_claude_launched(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("launch_claude_worker must not be called for a devin-shell reviewer")

    monkeypatch.setattr("charlie_work.workflow.launch_devin_session", fake_devin_launch)
    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fail_if_claude_launched)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert len(captured) == 1
    launch_kwargs = captured[0]
    assert launch_kwargs.get("review") is True
    assert launch_kwargs.get("head_sha") == "sha-100"

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_reap_review_verdicts_records_dead_devin_reviewer_verdict(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1513: the reap sweep that reads a dead reviewer's log for a
    fenced-JSON verdict must not be hardcoded to claude-code sidecars -- a
    devin-shell reviewer's verdict must be found and recorded too. This is
    the read-side counterpart to the launch-side routing test above: wiring
    the launch call alone is not enough if the verdict it eventually
    produces can never be reaped back (workflow._reap_review_verdicts used
    to `continue` past every non-claude-code WorkerView unconditionally)."""
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
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T11:50:00Z")

    # A devin-shell sidecar/log: no "-review"/".claude"/".api" dotted infix --
    # devin_shell's sidecar and log naming is identical for worker and review
    # sessions (reviews_dir is already a directory distinct from the worker
    # sessions_dir, so no collision is possible), and read_session_records'
    # stem regex (`^issue-\d+$`) only matches this exact shape.
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "issue-100.log"
    log_path.write_text(
        'Review complete.\n```json\n{"decision": "approved", "summary": "LGTM"}\n```',
        encoding="utf-8",
    )
    sidecar_path = reviews_dir / "issue-100.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "review-checkout"),
                "prompt_path": str(tmp_path / "review-prompt.md"),
                "command": ["devin", "--prompt-file", "{prompt_path}", "--print"],
                "pid": 0,
                "started_at": "2026-07-06T11:50:00Z",
                "log_path": str(log_path),
                "process_start_time": None,
            }
        ),
        encoding="utf-8",
    )

    # No new reviewer should be launched this pass -- the PR is already
    # claimed dispatched, and reaping its dead reviewer's verdict (not
    # launching a fresh one) is exactly what this test is asserting.
    def fail_if_launched(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no reviewer should be (re)launched in this pass")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fail_if_launched)
    monkeypatch.setattr("charlie_work.workflow.launch_devin_session", fail_if_launched)

    app.dispatch_reviews()

    decision_path = app.paths.prs / "pr-100" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision.get("decision") == "approved"

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_completed"
