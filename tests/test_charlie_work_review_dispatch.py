"""Review dispatch: label transitions, escalated short circuits, quota accounting.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the ``app.review()`` / review-started dispatch seam -- label transitions, escalated-issue short circuits, stale checks preservation, and review-dispatch quota/noise accounting. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub, FakeGitHubWithChecks
from _review_fixtures import _dispatch_reviews_app, _required_checks_config, _write_review_packet
from _rework_dispatch_fixtures import _wg
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import append_event, empty_state, load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp, _append_sweep_events
from charlie_work.claude_code import ClaudeWorkerRecord
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_checks_unavailable_blocks_and_preserves_labels(tmp_path: Path) -> None:
    """gh pr checks command failure must block review, leave labels unchanged, and surface checks_unavailable."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data["checks_unavailable"] is True
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"


def test_review_preserves_recorded_decision_in_state(tmp_path: Path) -> None:
    from charlie_work.state import load_state as _load
    from charlie_work.state import save_state as _save

    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state = _load(paths.state_file)
    state["prs"]["456"] = {"decision": "approved", "custom": "kept"}
    _save(paths.state_file, state)

    app.review(456)

    after = _load(paths.state_file)
    assert after["prs"]["456"]["decision"] == "approved"  # was clobbered pre-fix
    assert after["prs"]["456"]["custom"] == "kept"
    assert after["prs"]["456"]["status"] == "reviewing"


def test_review_label_transition_failure_persists_packet(tmp_path: Path) -> None:
    """Issue #135: A PARTIAL_FAILURE during review_started label transition must
    leave the review packet persisted in state and report structured label_error."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    reviewing_label = config.labels.reviewing

    class LabelFailReviewGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            if label == reviewing_label:
                # Return False to simulate add failure (error-as-value)
                return False
            return super().add_issue_label(number, label)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailReviewGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_started"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    # PR #456 is linked to issue #123 in FakeGitHub
    assert (123, reviewing_label) in label_error["add_failures"]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert state["prs"]["456"]["label_error"]["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert Path(state["prs"]["456"]["decision_path"]).exists()


def test_review_started_clears_needs_rework() -> None:
    # Re-review after a rework must not stack reviewing on top of needs-rework.
    from charlie_work.labels import transition, TransitionOutcome

    fake_gh = FakeGitHub()
    result = transition(fake_gh, OrchestratorConfig().labels, 123, "review_started")

    assert result.outcome == TransitionOutcome.APPLIED
    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert (123, "agent:needs-rework") in fake_gh.labels_removed


def test_review_started_skip_when_head_unchanged_after_request_changes(tmp_path: Path) -> None:
    """Janitor blocks review when head hasn't changed after request_changes (no-op rework).

    This prevents pointless packet churn and preserves the needs_rework label on
    budget-deferred rework candidates. The janitor now blocks before review_started
    can fire.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision with a specific head SHA
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": 456,
                "title": "PR for issue 123",
                "url": "https://example.test/pr/456",
                "headRefOid": "sha-abc123",
                "isCrossRepository": False,
                "headRefName": "agent/issue-123",
                "baseRefName": "main",
            }
        ),
        encoding="utf-8",
    )

    # Set initial diff
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )

    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Verify the decision was recorded
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["prs"]["456"]["decision"] == "request_changes"
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
        # Verify patch-id was calculated
        assert "reviewed_patch_id" in state["prs"]["456"]

    # Clear label tracking to isolate the review() call
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Call review again with the same head SHA and same diff (no-op rework)
    result = app.review(456)

    # The janitor should block the PR because the diff is unchanged (no-op rework)
    assert result.ok is False
    # With patch-id comparison, the message should mention patch-id
    assert "PR diff unchanged since request_changes verdict" in result.message
    # review_started transition should not fire (janitor blocks before it)
    assert (123, "agent:pr-open") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_review_started_fires_when_head_advanced_after_request_changes(tmp_path: Path) -> None:
    """Review_started transition should fire when head has advanced after request_changes."""
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision with a specific head SHA
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": 456,
                "title": "PR for issue 123",
                "url": "https://example.test/pr/456",
                "headRefOid": "sha-abc123",
                "isCrossRepository": False,
                "headRefName": "agent/issue-123",
                "baseRefName": "main",
            }
        ),
        encoding="utf-8",
    )

    # Set initial diff
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )

    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Advance the PR head and change the diff (simulating actual content changes)
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+changed"
    )

    # Call review again with the advanced head
    result = app.review(456)

    # The review_started transition should fire (adds pr_open and reviewing)
    assert result.ok is True
    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added


def test_review_does_not_clobber_escalated_label_on_head_advance(tmp_path: Path) -> None:
    """Issue #384: an escalated issue must stay terminal on re-review.

    After record_review escalates an issue to agent:operator-queue (issue
    #1266: max_rework_cycles_exceeded is mechanical), a later review() pass
    (e.g., from loop()) that sees a newly-advanced head must not regenerate a
    packet or fire review_started, which would strip the escalation label
    and put the PR back into an active-automation state.
    """
    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes (count = 1)
    fake_gh.pr_head_shas[456] = "1111111111111111111111111111111111111111"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 1"
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Second request_changes (count = 2)
    fake_gh.pr_head_shas[456] = "2222222222222222222222222222222222222222"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 2"
    app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )

    # Third request_changes (escalated)
    fake_gh.pr_head_shas[456] = "3333333333333333333333333333333333333333"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 3"
    app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    # Issue #1266: max_rework_cycles_exceeded is mechanical, so it lands
    # agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    # Clear label tracking to isolate the review() call
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Simulate a worker pushing after escalation: new head and new diff
    fake_gh.pr_head_shas[456] = "4444444444444444444444444444444444444444"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change new"

    result = app.review(456)

    # review() must short-circuit and must not touch the escalation label
    assert result.ok is True
    assert (123, config.labels.operator_queue) not in fake_gh.labels_removed
    assert (123, config.labels.pr_open) not in fake_gh.labels_added
    assert (123, config.labels.reviewing) not in fake_gh.labels_added

    # State must stay escalated and not be overwritten back to "reviewing"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"


def test_review_short_circuits_escalated_issue_less_pr(tmp_path: Path) -> None:
    """Issue #384: PR-level escalation is terminal even without a linked issue.

    Cross-repo PRs (or same-repo branches that don't match the configured
    prefix) fail closed: ``linked_issue_number`` returns ``None``.
    ``record_review`` still sets the PR's own state status to ``"escalated"``
    after ``max_rework_cycles``. A later ``review()`` pass must short-circuit on
    that PR-level status and must not fall through to the janitor gate, which
    would overwrite ``status`` with ``"janitor_blocked"``.
    """
    config = OrchestratorConfig()  # max_rework_cycles = 2, require_issue_link = True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Cross-repo PRs never resolve to a linked issue for lifecycle purposes.
    fake_gh.prs[0]["isCrossRepository"] = True

    # First request_changes (count = 1)
    fake_gh.pr_head_shas[456] = "1111111111111111111111111111111111111111"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 1"
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Second request_changes (count = 2)
    fake_gh.pr_head_shas[456] = "2222222222222222222222222222222222222222"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 2"
    app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )

    # Third request_changes (escalated)
    fake_gh.pr_head_shas[456] = "3333333333333333333333333333333333333333"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 3"
    result = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert result.data["escalated"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    # No issue entry should exist for an issue-less PR.
    assert "123" not in state.get("issues", {})
    # Label transitions are gated on issue_number, so no labels should fire.
    assert not fake_gh.labels_added
    assert not fake_gh.labels_removed

    # Clear label tracking to isolate the review() call.
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Simulate a later loop/review pass with a new head and new diff.
    fake_gh.pr_head_shas[456] = "4444444444444444444444444444444444444444"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+new change"
    review_result = app.review(456)

    # review() must short-circuit and must not touch anything.
    assert review_result.ok is True
    assert review_result.data.get("pass_skipped") is True
    assert not fake_gh.labels_added
    assert not fake_gh.labels_removed

    # No review packet/decision should be (re)written on the short-circuited call.
    pr_dir = paths.prs / "pr-456"
    assert not (pr_dir / "review-prompt.md").exists()

    # The PR state must remain escalated, not be clobbered to janitor_blocked.
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"


def test_review_refreshes_janitor_diagnostics_while_issue_escalated(tmp_path: Path) -> None:
    """job-cannon #1397/#1443 (2026-07-27): janitor_failures must not go stale
    while the linked issue is escalated for an UNRELATED reason (e.g. a dead
    rework-worker session), even though status/labels/routing stay frozen.

    Before this fix, review()'s escalation short-circuit returned before ever
    calling run_janitor again, so a PR whose merge conflict cleared (or whose
    CI went green) kept reporting the stale pre-escalation failure for as long
    as the issue stayed escalated -- sometimes many hours, until an operator
    ran unescalate(). This asserts janitor_ok/janitor_failures track reality
    on every review() call, while status/labels/attempt counters stay inert.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.pr_head_shas[456] = "sha-1"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 1"
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"

    # Seed a PR record with a stale-but-then-true merge-conflict failure, as
    # the janitor gate itself would have written it before escalation.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["PR has merge conflicts (mergeable=CONFLICTING)"],
        }
        # Escalate the linked ISSUE only (mirrors a dead rework-worker
        # escalation) -- the PR's own status is deliberately left at
        # "janitor_blocked", matching the real #1397/#1443 state shape.
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    # The conflict clears (e.g. a branch update landed) while still escalated.
    fake_gh.prs[0]["mergeable"] = "MERGEABLE"
    fake_gh.prs[0]["mergeStateStatus"] = "CLEAN"

    result = app.review(456)

    assert result.ok is True
    assert result.data.get("pass_skipped") is True
    assert not fake_gh.labels_added
    assert not fake_gh.labels_removed

    state = load_state(paths.state_file)
    # Frozen: escalation stays terminal for status -- only unescalate() may
    # move this.
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["issues"]["123"]["status"] == "escalated"
    # Refreshed: the stale conflict failure must be gone.
    assert state["prs"]["456"]["janitor_ok"] is True
    assert state["prs"]["456"]["janitor_failures"] == []

    # A second call with nothing changed must not re-log a duplicate event
    # (cost-spirals.md Finding 2 dedup applies here too).
    events_before = len(state.get("events", []))
    app.review(456)
    state = load_state(paths.state_file)
    janitor_gate_events = [e for e in state.get("events", []) if e.get("kind") == "janitor_gate"]
    assert len(janitor_gate_events) == 1
    assert len(state.get("events", [])) == events_before


def test_review_started_fires_when_no_recorded_verdict(tmp_path: Path) -> None:
    """Review_started transition should fire when there's no prior verdict."""
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create PR directory without any prior review decision
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": 456,
                "title": "PR for issue 123",
                "url": "https://example.test/pr/456",
                "headRefOid": "sha-abc123",
                "isCrossRepository": False,
                "headRefName": "agent/issue-123",
            }
        ),
        encoding="utf-8",
    )

    # Call review without any prior verdict
    result = app.review(456)

    # The review_started transition should fire (adds pr_open and reviewing)
    assert result.ok is True
    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added


def test_review_started_does_not_fire_when_review_dispatch_disabled(tmp_path: Path) -> None:
    """Issue #868: review() must not stamp reviewing/agent:reviewing when
    review_dispatch is disabled.

    Nothing services the "reviewing" state when dispatch is off --
    dispatch_reviews()'s launch+reap machinery is gated off by the same
    flag -- so stamping it here just strands the PR under a label its own
    reaper never runs to clear. Byte-identical scenario to
    test_review_started_fires_when_no_recorded_verdict above, minus the
    enabled flag.
    """
    config = OrchestratorConfig()  # review_dispatch.enabled defaults to False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": 456,
                "title": "PR for issue 123",
                "url": "https://example.test/pr/456",
                "headRefOid": "sha-abc123",
                "isCrossRepository": False,
                "headRefName": "agent/issue-123",
            }
        ),
        encoding="utf-8",
    )

    result = app.review(456)

    assert result.ok is True
    assert (123, "agent:reviewing") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("status") != "reviewing"
    # The packet itself is still generated -- only the state/label stamp is gated.
    assert state["prs"]["456"]["prompt_path"] == result.data["prompt_path"]


def test_review_dispatch_noise_loop_aggregation_preserves_history(tmp_path: Path) -> None:
    """Issue #525: a repeating per-pass noise loop cannot evict unrelated events.

    Simulates 5 ghost reviewer sessions x 2 events per pass for 250 passes.
    Without aggregation the events array would hold 2501 entries and evict the
    diagnostic event; with per-kind aggregation it stays at 501.

    Issue #1264 (W6 PR3): this test predates the ``write_gate`` requirement
    and operates on an in-memory ``state`` dict with no ``state_file``
    fixture. ``_append_sweep_events`` now requires a real ``WriteGate``, so a
    ``tmp_path``-scoped one (``dry_run=False``, matching the pre-conversion
    always-write behavior this test exercises) is threaded through purely to
    satisfy that contract -- it is never read from disk.
    """
    state = empty_state()
    state = append_event(state, "diagnostic_event", {"note": "keep me"}, max_size=2000)

    prs = list(range(1, 6))
    passes = 250
    for _ in range(passes):
        sweep_events = [
            ("review_dispatch_stalled", {"pr_number": pr, "status": "dispatched"}) for pr in prs
        ] + [
            (
                "review_dispatch_lifecycle_reaped",
                {"pr_number": pr, "github_state": "merged"},
            )
            for pr in prs
        ]
        state = _append_sweep_events(
            state, sweep_events, max_size=2000, write_gate=_wg(tmp_path / "state.json")
        )

    diagnostic = [e for e in state["events"] if e.get("kind") == "diagnostic_event"]
    assert len(diagnostic) == 1
    assert len(state["events"]) == 1 + (passes * 2)
    stalled_sweeps = [
        e for e in state["events"] if e.get("kind") == "review_dispatch_stalled_sweep"
    ]
    assert len(stalled_sweeps) == passes
    assert all(e["payload"]["count"] == len(prs) for e in stalled_sweeps)


def test_review_dispatch_quota_failure_rolls_back_attempt_count(
    monkeypatch, tmp_path: Path
) -> None:
    """A quota failure during review dispatch must not consume the per-PR
    review_dispatch_attempt_count. Without this fix, 3 quota hits (where no
    reviewer actually ran) would escalate a PR that was never reviewed.
    """
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
    # Claim should be rolled back (not failed)
    assert state["prs"]["100"].get("review_dispatch_status") is None
    # Attempt count should still be 0 (incremented at claim, then rolled back)
    assert state["prs"]["100"].get("review_dispatch_attempt_count", 0) == 0
