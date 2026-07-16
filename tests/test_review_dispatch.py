"""Tests for the auto review-dispatch stage (#370)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import CommandResult, OrchestratorApp


class FakeGitHub:
    """Minimal GitHub stub for review-dispatch tests."""

    def __init__(self, prs: list[dict[str, Any]] | None = None) -> None:
        self.prs = prs or []

    def pr_list(self) -> list[dict[str, Any]]:
        return list(self.prs)


def _app(tmp_path: Path, prs: list[dict[str, Any]] | None = None) -> OrchestratorApp:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "issues": {}, "prs": {}, "events": []}
    paths.state_file.write_text(json.dumps(state), encoding="utf-8")
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub(prs), dry_run=False)


def _write_review_packet(
    tmp_path: Path,
    pr_number: int,
    head_ref_name: str,
    packet_head_sha: str,
    decision: dict[str, Any] | None = None,
) -> Path:
    """Create a review packet fixture for a PR."""
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    pr = {
        "number": pr_number,
        "headRefName": head_ref_name,
        "headRefOid": packet_head_sha,
    }
    (pr_dir / "pr.json").write_text(json.dumps(pr), encoding="utf-8")
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}",
        encoding="utf-8",
    )
    if decision is not None:
        (pr_dir / "review-decision.json").write_text(
            json.dumps(decision),
            encoding="utf-8",
        )
    return pr_dir


def _fake_record(issue_number: int, pr_number: int) -> ClaudeWorkerRecord:
    return ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}-fix",
        worktree_path="/tmp/review",
        prompt_path="/tmp/review/prompt.md",
        command=("claude", "-p"),
        pid=12345,
        started_at="2026-01-01T00:00:00Z",
        log_path="/tmp/review.log",
        process_start_time=time.time(),
    )


def _pr(number: int, issue: int, head_sha: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{issue}",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue}-fix",
        "baseRefName": "main",
        "headRefOid": head_sha,
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def test_dispatch_reviews_disabled_returns_empty(tmp_path: Path) -> None:
    """When review_dispatch is disabled, dispatch_reviews is a no-op."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)
    result = app.dispatch_reviews()
    assert result.ok is True
    assert result.data["selected_count"] == 0


def test_dispatch_reviews_launches_for_queued_pr(tmp_path: Path, monkeypatch) -> None:
    """A PR with a current packet and no verdict gets a reviewer launched."""
    prs = [_pr(100, 10, "sha-100")]
    app = _app(tmp_path, prs)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")

    launched: list[tuple[int, str, dict[str, Any]]] = []

    def fake_launch(
        issue_number: int,
        branch: str,
        prompt_text: str,
        *,
        review_mode: bool = False,
        **kwargs: Any,
    ) -> ClaudeWorkerRecord:
        launched.append((issue_number, branch, kwargs))
        assert review_mode is True
        return _fake_record(issue_number, 100)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_worker_alive", lambda _r: True)

    result = app.dispatch_reviews()
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert len(launched) == 1
    assert launched[0][0] == 10
    assert launched[0][1] == "agent/issue-10-fix"
    assert launched[0][2]["repo_root"] == tmp_path

    state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatched"


def test_dispatch_reviews_no_double_dispatch(tmp_path: Path, monkeypatch) -> None:
    """A PR already dispatched is not launched again on the next pass."""
    prs = [_pr(100, 10, "sha-100")]
    app = _app(tmp_path, prs)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")

    call_count = 0

    def fake_launch(issue_number: int, branch: str, prompt_text: str, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        return _fake_record(issue_number, 100)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_worker_alive", lambda _r: True)

    app.dispatch_reviews()
    assert call_count == 1
    app.dispatch_reviews()
    assert call_count == 1


def test_dispatch_reviews_launches_all_queued_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    """Without a local cap, every queued PR gets a reviewer."""
    prs = [_pr(100, 10, "sha-100"), _pr(200, 20, "sha-200")]
    app = _app(tmp_path, prs)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")
    _write_review_packet(tmp_path, 200, "agent/issue-20-fix", "sha-200")

    launched = []

    def fake_launch(issue_number: int, branch: str, prompt_text: str, **kwargs: Any):
        launched.append(issue_number)
        return _fake_record(issue_number, 100 if issue_number == 10 else 200)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_worker_alive", lambda _r: True)

    result = app.dispatch_reviews()
    assert result.data["selected_count"] == 2
    assert set(launched) == {10, 20}


def test_dispatch_reviews_respects_local_cap(tmp_path: Path, monkeypatch) -> None:
    """max_local_review_processes caps concurrent reviewer launches."""
    prs = [_pr(100, 10, "sha-100"), _pr(200, 20, "sha-200")]
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_local_review_processes=1),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(prs), dry_run=True)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")
    _write_review_packet(tmp_path, 200, "agent/issue-20-fix", "sha-200")

    launched = []

    def fake_launch(issue_number: int, branch: str, prompt_text: str, **kwargs: Any):
        launched.append(issue_number)
        return _fake_record(issue_number, 100 if issue_number == 10 else 200)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_worker_alive", lambda _r: True)

    result = app.dispatch_reviews()
    assert result.data["selected_count"] == 1
    assert len(launched) == 1


def test_dispatch_reviews_records_failed_launch(tmp_path: Path, monkeypatch) -> None:
    """A failed reviewer launch is recorded so the claim is not re-taken immediately."""
    prs = [_pr(100, 10, "sha-100")]
    app = _app(tmp_path, prs)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")

    def fake_launch(**kwargs: Any) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=10,
            branch="agent/issue-10-fix",
            worktree_path="",
            prompt_path="",
            command=(),
            pid=None,
            started_at="2026-01-01T00:00:00Z",
            log_path="",
            error="claude not found",
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()
    assert result.ok is False
    assert result.data["failed_count"] == 1
    state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"


def test_dispatch_reviews_prompt_includes_repo_and_verdict_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """The reviewer prompt is appended with the repo root and verdict command."""
    prs = [_pr(100, 10, "sha-100")]
    app = _app(tmp_path, prs)
    _write_review_packet(tmp_path, 100, "agent/issue-10-fix", "sha-100")

    captured_prompt = ""

    def fake_launch(
        issue_number: int,
        branch: str,
        prompt_text: str,
        **kwargs: Any,
    ) -> ClaudeWorkerRecord:
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return _fake_record(issue_number, 100)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_worker_alive", lambda _r: True)

    app.dispatch_reviews()
    assert "Auto-dispatched reviewer output" in captured_prompt
    assert str(tmp_path) in captured_prompt
    assert "verdict --pr 100" in captured_prompt


def test_loop_calls_dispatch_reviews(tmp_path: Path, monkeypatch) -> None:
    """loop() invokes dispatch_reviews and includes its result in the payload."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    monkeypatch.setattr(
        app, "intake", lambda: CommandResult(True, "", {"processed": 0})
    )
    monkeypatch.setattr(
        app,
        "dispatch",
        lambda *a, **k: CommandResult(True, "", {"selected_count": 0}),
    )
    monkeypatch.setattr(
        app,
        "dispatch_rework",
        lambda *a, **k: CommandResult(True, "", {"selected_count": 0}),
    )
    dispatch_reviews_mock = MagicMock(
        return_value=CommandResult(True, "reviews dispatched", {"selected_count": 0})
    )
    monkeypatch.setattr(app, "dispatch_reviews", dispatch_reviews_mock)

    result = app.loop()
    assert result.ok is True
    assert "dispatch_reviews" in result.data
    dispatch_reviews_mock.assert_called_once_with()
