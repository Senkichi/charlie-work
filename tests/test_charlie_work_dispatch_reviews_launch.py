"""Review-dispatch launch mechanics: launches, claims, and reviewer plumbing.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the launch half of the ``test_dispatch_reviews_*`` seam -- launching reviews
for queued PRs, claim release/restore, double-dispatch prevention, config and
model plumbing, and effort-arm recording. Sibling seams:
``test_charlie_work_dispatch_reviews_quota.py`` (caps/quota/deferral) and
``test_charlie_work_dispatch_reviews_stalled.py`` (stalled claims, empty
diffs). Shared fakes and helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
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
    ReviewerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dispatch_reviews_launches_for_all_queued_prs(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: dispatch_reviews launches a Claude reviewer for every queued PR."""
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
        {
            "number": 200,
            "title": "Fix #20",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    _write_review_packet(tmp_path, 200, "sha-200")

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 2
    assert result.data["failed_count"] == 0
    assert result.data["selected_count"] == 2
    assert len(launched) == 2
    for _args, kwargs in launched:
        assert kwargs.get("review") is True
        assert "reviews" in str(kwargs.get("sessions_dir", ""))

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert state["prs"]["100"]["reviewer_pid"] == 12345
    assert state["prs"]["200"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_dispatch_reviews_forwards_orchestrator_config_to_launch(
    monkeypatch, tmp_path: Path
) -> None:
    """dispatch_reviews() must pass the live OrchestratorConfig into
    launch_claude_worker so review-only pins (review_effort, review_max_turns,
    the review_effort experiment) actually take effect. Without this, every
    reviewer launch resolves effort/max-turns from a bare default
    OrchestratorConfig() inside launch_claude_worker itself, silently
    discarding whatever the operator configured."""
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

    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert len(captured) == 1
    assert captured[0].get("config") is app.config


def test_dispatch_reviews_threads_reviewer_model_as_model_override(
    monkeypatch, tmp_path: Path
) -> None:
    """dispatch_reviews() must pass config.reviewer.model as model_override so
    a split worker/reviewer model configuration actually changes which model
    the reviewer launches with -- launch_claude_worker's own fallback
    (resolved_config.worker.model) is claimed by the WORKER when
    worker.harness == 'claude-code' (worker.harness is read directly from
    config, with no resolver involved), so without an explicit override the
    reviewer would silently launch with the worker's model whenever the two
    are configured to differ."""
    from dataclasses import replace

    from charlie_work.config import ReviewerRoleConfig, WorkerRoleConfig

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
    app.config = replace(
        app.config,
        worker=WorkerRoleConfig(harness="claude-code", model="claude-opus-4-1"),
        reviewer=ReviewerRoleConfig(harness="claude-code", model="claude-sonnet-5"),
    )
    _write_review_packet(tmp_path, 100, "sha-100")

    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert len(captured) == 1
    assert captured[0].get("model_override") == "claude-sonnet-5"


def test_dispatch_reviews_records_review_effort_arm_on_state_and_event(
    monkeypatch, tmp_path: Path
) -> None:
    """The review_effort experiment's per-PR arm/effort assignment must be
    recorded on the PR's state entry and in the review_dispatch_claim event
    at claim time (so it's analyzable even if the launch itself later fails)."""
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
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        reviewer=ReviewerRoleConfig(effort="high", effort_experiment_fraction=1.0),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(tmp_path, 100, "sha-100")

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *a, **kw: _fake_claude_worker_record(
            kw.get("issue_number") or a[0], kw.get("branch") or a[1]
        ),
    )

    result = app.dispatch_reviews()
    assert result.ok is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_effort_arm"] == "treatment"
    assert state["prs"]["100"]["review_effort_used"] == "high"

    claim_events = [e for e in state["events"] if e.get("kind") == "review_dispatch_claim"]
    assert len(claim_events) == 1
    assignments = claim_events[0]["payload"]["review_effort_assignments"]
    # Issue #1439: the structure-aware turn cap is resolved alongside the
    # review_effort arm and mirrored into the same assignment record.
    assert assignments == [
        {
            "pr_number": 100,
            "review_effort_arm": "treatment",
            "review_effort_used": "high",
            "review_turn_cap": config.review_dispatch.review_max_turns,
        }
    ]


def test_dispatch_reviews_experiment_disabled_records_no_arm(monkeypatch, tmp_path: Path) -> None:
    """fraction=0.0 (default): review_effort still applies to all PRs, but
    since there's no experiment, no arm is recorded."""
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
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        reviewer=ReviewerRoleConfig(effort="high"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(tmp_path, 100, "sha-100")

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *a, **kw: _fake_claude_worker_record(
            kw.get("issue_number") or a[0], kw.get("branch") or a[1]
        ),
    )

    result = app.dispatch_reviews()
    assert result.ok is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_effort_arm"] is None
    assert state["prs"]["100"]["review_effort_used"] == "high"


def test_dispatch_reviews_launch_failure_releases_claim(monkeypatch, tmp_path: Path) -> None:
    """Issue #487: a failed reviewer launch (e.g. WinError 2 from an
    unresolved npm ``.CMD`` shim) must not strand the PR at
    ``review_dispatch_pending`` forever. ``dispatch_reviews`` claims the PR
    as pending before launching and must upgrade a failed launch to
    ``review_dispatch_failed`` with the error recorded, freeing it for
    redispatch after the stale-claim timeout rather than silently holding
    the claim."""
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

    launch_error = (
        "failed to launch claude: [WinError 2] The system cannot find the file specified"
    )

    def fake_launch_failure(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=kwargs.get("issue_number") or args[0],
            branch=kwargs.get("branch") or args[1],
            worktree_path="/fake/worktree",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p", "--permission-mode", "plan"),
            pid=None,
            started_at="2026-07-20T12:00:00Z",
            log_path="/fake/log.log",
            error=launch_error,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch_failure)

    result = app.dispatch_reviews()

    assert result.ok is False
    assert result.data["launched_count"] == 0
    assert result.data["failed_count"] == 1
    assert result.data["failed"] == [{"pr": 100, "error": launch_error}]

    state = load_state(app.paths.state_file)
    pr_state = state["prs"]["100"]
    # The pending claim written before launch must be cleared, not left
    # dangling — otherwise `_is_review_dispatchable` would never see this PR
    # as re-claimable and it would sit stuck forever, same failure shape as
    # issue #487's "never claimed at all" gap.
    assert pr_state["review_dispatch_status"] == "review_dispatch_failed"
    assert pr_state["review_dispatch_pending_at"] is None
    assert pr_state["review_dispatched_at"] is None
    assert pr_state["reviewer_pid"] is None
    assert pr_state["review_dispatch_error"] == launch_error
    assert pr_state["review_dispatch_failed_at"] is not None


def test_dispatch_reviews_launch_success_clears_stale_review_dispatch_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A successful launch must clear a stale review_dispatch_error left over
    from an earlier failed attempt; otherwise the last error string is
    carried forward verbatim by the **pr_state spread forever."""
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
            "number": 100,
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_error": "old boom",
        }
        save_state(app.paths.state_file, state)

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *args, **kwargs: _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        ),
    )

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 1
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_error"] is None


def test_dispatch_reviews_prevents_double_dispatch(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: a live reviewer blocks re-dispatch of the same PR."""
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

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    def fake_is_pid_alive(pid: int, *_args: Any, **_kwargs: Any) -> bool:
        # Pretend the fake reviewer PID is still alive.
        return pid == 12345

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", fake_is_pid_alive)

    first = app.dispatch_reviews()
    assert first.data["launched_count"] == 1
    assert len(launched) == 1

    second = app.dispatch_reviews()
    assert second.data["launched_count"] == 0
    assert len(launched) == 1
    assert second.data["deferred_count"] == 1
