"""Review-claim bookkeeping: stale-claim recovery skips, throttled-review rollback, decision cache.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import os
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from _helpers import _init_git_repo
from _review_fixtures import (
    _approved_automerge,
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_packet,
)
from _rework_dispatch_fixtures import _wg
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import (
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.review_decision import ReviewDecision
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import (
    OrchestratorApp,
    _detect_and_handle_stalled_reviews,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_stale_claim_recovery_skipped_logs_when_prompt_path_missing_from_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #708: a reviewing PR whose prompt_path is missing from state must
    emit a review_stale_claim_recovery_skipped event instead of silently moving
    on, so a stuck-PR investigation can distinguish "recovery gave up" from
    "recovery was not needed"."""
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

    # reviewing PR with NO review_dispatch_status and NO prompt_path -- the
    # stale-claim recovery path's first skip branch.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
        }
        save_state(app.paths.state_file, state)

    # No launch should happen: recovery gave up before reaching dispatch.
    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    state = load_state(app.paths.state_file)
    # The PR was not reaped -- it stays in its stuck reviewing state with no
    # dispatch claim, exactly as before #708. The fix is observability, not a
    # behavior change to the recovery decision itself.
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(app.paths.state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "prompt_path missing from state"
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_prompt_path_file_gone(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #708: a reviewing PR whose prompt_path points at a file that no
    longer exists on disk must emit a review_stale_claim_recovery_skipped event
    (with the path) instead of silently moving on."""
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

    gone_prompt = tmp_path / "deleted-review-prompt.md"

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(gone_prompt),
        }
        save_state(app.paths.state_file, state)

    assert not gone_prompt.exists()

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(app.paths.state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "prompt_path file does not exist on disk"
    assert payload["prompt_path"] == str(gone_prompt)
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_decision_already_recorded(
    tmp_path: Path,
) -> None:
    """Issue #734: a reviewing PR whose decision_path already holds a verdict
    (e.g. ``request_changes``) is silently passed over by stale-claim recovery
    on every pass -- the verdict was never acted upon, but without an event
    nobody can tell recovery considered the PR and declined. This is the second
    of the three silent skip paths identified in #734."""
    from datetime import timedelta

    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    # Create a valid prompt_path on disk and a decision file with a verdict.
    pr_dir = tmp_path / "prs" / "pr-100"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "review-prompt.md"
    prompt_path.write_text("review prompt", encoding="utf-8")
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(json.dumps({"decision": "request_changes"}), encoding="utf-8")

    # Age the packet so the stale-claim timeout is satisfied -- the skip must
    # come from the decision gate, not from the packet-age gate.
    old_mtime = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(prompt_path, (old_mtime, old_mtime))

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(decision_path),
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    # The PR was not reaped -- recovery declined because a verdict exists.
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "decision_already_recorded"
    assert payload["decision"] == "request_changes"
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_packet_not_stale(
    tmp_path: Path,
) -> None:
    """Issue #734: a reviewing PR whose packet is not yet past the stale-claim
    timeout is silently skipped on every pass until it becomes stale. This is
    the third of the three silent skip paths identified in #734. The event is
    info-level (not warning) because this is expected flow control -- the
    packet simply is not old enough yet -- unlike the other two skips which
    indicate a PR recovery cannot help."""
    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    # Create a valid prompt_path on disk with NO decision file (decision_value
    # defaults to "missing", which passes the decision gate). The packet is
    # fresh -- not aged -- so the stale-claim timeout is not met.
    pr_dir = tmp_path / "prs" / "pr-100"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "review-prompt.md"
    prompt_path.write_text("review prompt", encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(pr_dir / "review-decision.json"),
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "packet_not_stale"
    assert "packet_age" in payload
    assert skip_events[0]["level"] == "info"


def test_corrupt_review_decision_treated_as_not_approved(tmp_path: Path) -> None:
    """A corrupt review-decision.json must not crash merge_ready/loop; it must
    be treated as a non-approval so the PR waits for a real review."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text("{truncated", encoding="utf-8")

    result = app.merge_ready(456)

    # Issue #1362 Stage 1: a corrupt flat file with no round-archive fallback
    # now resolves to {"decision": "missing"} rather than the old "invalid"
    # sentinel (review_decision.resolve_decision_payload) -- both are
    # equally non-terminal, so the fail-safe outcome below is unchanged.
    assert result.data["review_decision"] == {"decision": "missing"}
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []


def test_refresh_pr_decision_cache_updates_disagreeing_tracked_pr(tmp_path: Path) -> None:
    """Issue #1362 Stage 3: a tracked PR whose cache disagrees with the
    file-first decision gets its three cache fields overwritten."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {
        "status": "reviewing",
        "issue_number": 123,
        "decision": "pending",
        "reviewed_head_sha": "stale-sha",
        "decision_path": "stale-path",
    }
    save_state(paths.state_file, seed)

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )
    app._refresh_pr_decision_cache(456, decision, decision_path)

    refreshed = load_state(paths.state_file)["prs"]["456"]
    assert refreshed["decision"] == "approved"
    assert refreshed["reviewed_head_sha"] == "fresh-sha"
    assert refreshed["decision_path"] == str(decision_path)
    # Non-decision fields (status, issue_number) must survive the mirror
    # write untouched -- the refresh must never clobber the rest of the entry.
    assert refreshed["status"] == "reviewing"
    assert refreshed["issue_number"] == 123


def test_refresh_pr_decision_cache_no_op_when_cache_already_agrees(tmp_path: Path) -> None:
    """Issue #1362 Stage 3: when the cache already agrees with the file, the
    refresh must not write state.json at all -- the docstring's promised
    short-circuit for the common (no verdict activity) case."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {
        "status": "reviewing",
        "decision": "approved",
        "reviewed_head_sha": "fresh-sha",
        "decision_path": str(decision_path),
    }
    save_state(paths.state_file, seed)
    mtime_before = paths.state_file.stat().st_mtime_ns

    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )

    # Positive signal, not just an absence: wrap the gated writer so a call
    # that happens but happens to leave the mtime unchanged (e.g. a
    # sub-resolution clock or a write of byte-identical content) cannot
    # read as "no write occurred". An mtime check alone would pass even if
    # the short-circuit above it were deleted, as long as the resulting
    # write raced under the OS's mtime granularity.
    #
    # ``write_gate`` is a frozen dataclass (its instances reject attribute
    # assignment), so the wrap patches the *class* method rather than the
    # instance attribute.
    save_state_calls: list[dict] = []
    write_gate_cls = type(app.write_gate)
    original_save_state = write_gate_cls.save_state

    def _tracking_save_state(self: object, state: dict) -> None:
        save_state_calls.append(state)
        original_save_state(self, state)

    write_gate_cls.save_state = _tracking_save_state  # type: ignore[method-assign]
    try:
        app._refresh_pr_decision_cache(456, decision, decision_path)
    finally:
        write_gate_cls.save_state = original_save_state  # type: ignore[method-assign]

    assert save_state_calls == []
    assert paths.state_file.stat().st_mtime_ns == mtime_before


def test_refresh_pr_decision_cache_skips_pr_not_yet_in_state(tmp_path: Path) -> None:
    """Issue #1362 Stage 3 (review finding F3): a PR not yet tracked in
    state["prs"] must be left untouched by the refresh rather than
    materializing a decision-only partial entry with no status/counters."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    decision_path = paths.prs / "pr-999" / "review-decision.json"
    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )
    app._refresh_pr_decision_cache(999, decision, decision_path)

    state = load_state(paths.state_file)
    assert "999" not in state["prs"]


def test_stalled_review_throttled_rolls_back_attempt_count(monkeypatch, tmp_path: Path) -> None:
    """A throttled reviewer death in _detect_and_handle_stalled_reviews must
    not consume the per-PR review_dispatch_attempt_count. The reviewer hit a
    provider limit, not a PR-specific failure.
    """
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
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
    reviews_dir = app._layout.reviews_dir

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(
        reviews_dir, 100, "Error: usage limit exceeded; please try again later"
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    # Set attempt_count to 1 to simulate a prior dispatch
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"]["review_dispatch_attempt_count"] = 1
        save_state(app.paths.state_file, state)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir,
        app.paths.state_file,
        app.config,
        repo_root,
        write_gate=_wg(app.paths.state_file),
    )
    assert any(entry.get("pr") == 100 for entry in stalled)

    state = load_state(app.paths.state_file)
    # Attempt count should be 0 (rolled back from 1)
    assert state["prs"]["100"].get("review_dispatch_attempt_count", 0) == 0
    # Claim should be cleared (rolled back, not failed)
    assert state["prs"]["100"].get("review_dispatch_status") is None
