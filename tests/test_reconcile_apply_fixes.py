"""``reconcile.apply_fixes`` tests for the GitHub-state drift kinds.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
label transitions, state finalization, reconcile-event payloads, and
the generic fix machinery. The session/throttle/salvage lanes live in
``tests/test_reconcile_drift_sessions.py``; the mergequeue and
aviator lanes in their own subject modules.
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
from _reconcile_fixtures import (
    FakeGitHub,
    _issue,
    _pr,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import read_event_log
from charlie_work.paths import resolved_layout
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    empty_state,
)


def test_apply_fixes_closed_unmerged_pr_removes_active_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_active_labels"
    ]

    apply_fixes(gh, state, drift, config)

    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed


def test_apply_fixes_issue_active_label_no_open_pr_adds_ready_label(tmp_path: Path) -> None:
    """Issue #417 AC(b): mop-up --fix must add the ready label back, not only
    remove the stale active one, so the issue actually becomes dispatchable
    again instead of being left with no state-machine label at all.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(30, [config.labels.in_progress])])
    state = empty_state()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)
    matches = [item for item in drift if item.kind == "issue_active_label_no_open_pr"]
    assert matches

    new_state = apply_fixes(gh, state, matches, config)

    assert (30, config.labels.in_progress) in gh.labels_removed
    assert (30, config.labels.ready) in gh.labels_added
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert all(
        "label_write_failed" not in a
        for e in reconcile_events
        for a in e["payload"]["fix_actions"]
    )


def test_apply_fixes_issue_active_label_with_open_pr() -> None:
    """Issue #515: the --fix path must repair labels and update state status."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["30"] = {
        "number": 30,
        # Issue #1092: was "rework_requested", which now corroborates the
        # needs_rework label and correctly suppresses the rule. The status is
        # incidental scaffolding for this test -- its subject is the --fix
        # APPLY path, not which statuses qualify -- so it is retargeted to a
        # genuinely-stale status rather than the assertion being relaxed.
        # test_detect_drift_skips_issue_whose_status_corroborates_the_label
        # pins the behaviour that displaced it.
        "status": "dispatch_failed",
        "worker_pid": 12345,
    }

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_active_label_with_open_pr"
    ]
    assert drift

    new_state = apply_fixes(gh, state, drift, config)

    assert (30, config.labels.needs_rework) in gh.labels_removed
    assert (30, config.labels.pr_open) in gh.labels_added
    # Issue #515 (generalized): PASSIVE_OPEN_STATUS -- not "approved" -- is
    # the status this repair mirrors, instead of implying a review verdict
    # was recorded. Distinct from the active "reviewing" review() writes
    # (#955).
    assert new_state["issues"]["30"]["status"] == PASSIVE_OPEN_STATUS
    assert "worker_pid" not in new_state["issues"]["30"]


def test_apply_fixes_terminal_state_stale_emits_reconcile_event_with_content() -> None:
    """The generic unfixable-kind fallback in `apply_fixes` (precedented by
    `snapshot_truncated`/`escalated_labels_converged`) must emit a
    `"reconcile"` event whose payload carries the specific kind and the
    parked issue's number -- asserting on content, not just that some event
    fired."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="terminal_state_stale",
            issue_number=894,
            pr_number=None,
            detail="issue #894 has been parked in 'agent:human-needed' for 5.0 day(s)",
            fix_actions=(),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state["events"] if e.get("kind") == "reconcile"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["kind"] == "terminal_state_stale"
    assert payload["issue_number"] == 894
    assert "5.0 day" in payload["detail"]
    # No GitHub label mutation for an alert-only kind.
    assert gh.labels_added == []
    assert gh.labels_removed == []


def test_apply_fixes_returns_new_state_without_mutating_original() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    original_state = empty_state()
    original_state["prs"]["1"] = {"status": "reviewing"}
    original_snapshot = {
        "issues": dict(original_state["issues"]),
        "prs": {k: dict(v) for k, v in original_state["prs"].items()},
        "events": list(original_state["events"]),
    }
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    new_state = apply_fixes(gh, original_state, drift, config)

    assert original_state["prs"]["1"] == original_snapshot["prs"]["1"]
    assert original_state["events"] == original_snapshot["events"]
    assert new_state is not original_state
    assert new_state["prs"]["1"]["status"] == "merged"


def test_apply_fixes_merged_outside_orchestrator_transitions_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert (10, config.labels.done) in gh.labels_added
    # Issue #215: merged transition removes ALL other workflow labels, not just active
    for label in sorted(config.labels.workflow_labels - {config.labels.done}):
        assert (10, label) in gh.labels_removed
    assert new_state["prs"]["1"]["status"] == "merged"


def test_apply_fixes_merged_outside_orchestrator_stamps_and_preserves_merged_at() -> None:
    """Issue #747: ``apply_fixes`` must stamp ``merged_at`` when a PR transitions
    non-merged -> merged via the ``merged_outside_orchestrator`` drift fix, and
    must preserve an existing ``merged_at`` unchanged when the PR is already
    recorded as merged. The latter is the issue-still-active re-run path, where
    ``detect_drift`` re-emits the drift item even though state status is already
    ``'merged'`` (see reconcile.detect_drift: ``state_status != "merged" or
    issue_still_active``); the ``merged_at`` guard stops that re-run from
    back-dating the original observation time."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'",),
        )
    ]

    # Genuine transition: prior status is not 'merged' -> merged_at is stamped.
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}
    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["1"]["status"] == "merged"
    stamped = new_state["prs"]["1"].get("merged_at")
    assert stamped is not None
    assert stamped  # non-empty ISO 8601 timestamp
    # The stamp is a real 'Z'-suffixed utc_now() value, not a stale literal.
    assert stamped.endswith("Z")

    # Re-run where the PR is already recorded as merged (the issue-still-active
    # path, where detect_drift re-emits the drift despite status == 'merged'):
    # the original merged_at must be preserved unchanged, never back-dated.
    # Derived from the real clock (one day in the past) so no date-window
    # filter can ever rot this seed.
    original_merged_at = (
        (datetime.now(UTC) - timedelta(days=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state2 = empty_state()
    state2["prs"]["1"] = {"status": "merged", "merged_at": original_merged_at}
    new_state2 = apply_fixes(gh, state2, drift, config)
    assert new_state2["prs"]["1"]["status"] == "merged"
    assert new_state2["prs"]["1"]["merged_at"] == original_merged_at


def test_apply_fixes_contradiction_removes_active_labels_directly() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="done_label_with_active_labels",
            issue_number=40,
            pr_number=None,
            detail="issue #40 has done + reviewing",
            fix_actions=(f"remove label '{config.labels.reviewing}' from issue #40",),
            remove_labels=(config.labels.reviewing,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.labels_removed == [(40, config.labels.reviewing)]
    assert gh.labels_added == []


def test_apply_fixes_state_pr_missing_on_github_drops_entry() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}
    drift = [
        DriftItem(
            kind="state_pr_missing_on_github",
            issue_number=5,
            pr_number=999,
            detail="state has prs[999] but gh reports no such PR",
            fix_actions=("drop prs[999] from state",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert "999" not in new_state["prs"]
    assert "999" in state["prs"]


def test_apply_fixes_state_active_status_issue_closed_finalizes_state_and_labels() -> None:
    """Issue #259: apply_fixes sets status closed and removes active labels."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.in_progress], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)
    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["issues"]["259"]["status"] == "closed"
    assert (259, config.labels.in_progress) in gh.labels_removed
    assert state["issues"]["259"]["status"] == "dispatched"


def test_apply_fixes_state_active_status_issue_closed_idempotent() -> None:
    """Issue #259: re-running reconcile on a finalized issue is a no-op."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "closed"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] == []


def test_apply_fixes_appends_reconcile_event() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="issue_active_label_no_open_pr",
            issue_number=30,
            pr_number=None,
            detail="issue #30 stale",
            fix_actions=(f"remove label '{config.labels.in_progress}' from issue #30",),
            remove_labels=(config.labels.in_progress,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["issue_number"] == 30
    assert state["events"] == []


def test_apply_fixes_handles_quote_containing_label() -> None:
    """Structured remove_labels means quote characters in label names don't parse ambiguously."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    quote_label = "agent:has'quote"
    drift = [
        DriftItem(
            kind="done_label_with_active_labels",
            issue_number=50,
            pr_number=None,
            detail="issue #50 has done + quote label",
            fix_actions=(f"remove label '{quote_label}' from issue #50",),
            remove_labels=(quote_label,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.labels_removed == [(50, quote_label)]


def test_apply_fixes_clears_stale_dispatch_pending_claims() -> None:
    """apply_fixes must clear stale dispatch_pending claims from state."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["issues"]["123"] = {
        "number": 123,
        "status": "dispatch_pending",
        "dispatch_pending_at": "2020-01-01T00:00:00+00:00",
    }

    drift = [
        DriftItem(
            kind="stale_dispatch_pending_claim",
            issue_number=123,
            pr_number=None,
            detail="issue #123 has a stale dispatch_pending claim",
            fix_actions=("clear dispatch_pending claim for issue #123",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # The stale claim should be removed from state
    assert "123" not in new_state["issues"]
    assert "123" in state["issues"]  # Original state unchanged


def test_apply_fixes_snapshot_truncated_emits_reconcile_event() -> None:
    """Issue #259 review: a truncated snapshot produces a warning event."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="snapshot_truncated",
            issue_number=None,
            pr_number=None,
            detail="snapshot truncated",
            fix_actions=("skip completeness-dependent sweeps",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "snapshot_truncated"


def test_apply_fixes_multi_item_with_one_failed_label_write() -> None:
    """Issue #125: apply_fixes should record failure when one label write fails."""
    config = OrchestratorConfig()
    # Simulate a failed remove for one label
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_remove_labels={(20, config.labels.pr_open)},
    )
    state = empty_state()

    # Create multiple drift items
    drift = [
        DriftItem(
            kind="closed_unmerged_pr_active_labels",
            issue_number=20,
            pr_number=2,
            detail="PR #2 closed without merging",
            fix_actions=(
                f"remove label '{config.labels.pr_open}' from issue #20",
                f"remove label '{config.labels.reviewing}' from issue #20",
            ),
            remove_labels=(config.labels.pr_open, config.labels.reviewing),
        ),
        DriftItem(
            kind="issue_active_label_no_open_pr",
            issue_number=30,
            pr_number=None,
            detail="issue #30 has active label but no PR",
            fix_actions=(f"remove label '{config.labels.in_progress}' from issue #30",),
            remove_labels=(config.labels.in_progress,),
        ),
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Both items should have been processed
    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed
    assert (30, config.labels.in_progress) in gh.labels_removed

    # Check that the failure was recorded in the event
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 2

    # The first event should have label_write_failed recorded
    first_event = reconcile_events[0]
    assert first_event["payload"]["kind"] == "closed_unmerged_pr_active_labels"
    assert "label_write_failed: true" in first_event["payload"]["fix_actions"]

    # The second event should not have label_write_failed (it succeeded)
    second_event = reconcile_events[1]
    assert second_event["payload"]["kind"] == "issue_active_label_no_open_pr"
    assert "label_write_failed" not in " ".join(second_event["payload"]["fix_actions"])


def test_apply_fixes_transition_failure_recorded_in_event() -> None:
    """Issue #125: apply_fixes should record transition outcome when it fails."""
    config = OrchestratorConfig()
    # Simulate a failed add during transition
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_add_labels={(10, config.labels.done)},
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Check that the transition outcome was recorded in the event
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1

    event = reconcile_events[0]
    assert event["payload"]["kind"] == "merged_outside_orchestrator"
    assert "transition outcome" in " ".join(event["payload"]["fix_actions"])
    assert "partial_failure" in " ".join(event["payload"]["fix_actions"])
    assert "add_failures" in " ".join(event["payload"]["fix_actions"])


def test_apply_fixes_merged_pr_reaps_review_checkout_and_clears_dispatch_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #494: a PR merged externally while a reviewer dispatch is still
    in-flight must have its isolated review checkout removed and its
    review-dispatch state cleared during mop-up --fix.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["1"] = {
        "number": 1,
        "issue_number": 10,
        "status": "reviewing",
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }

    removed_calls: list[tuple[Any, int, Any]] = []

    def fake_remove_review_checkout(
        repo_root: Path, pr_number: int, *, reviews_dir: Any = None
    ) -> bool:
        removed_calls.append((repo_root, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.reconcile.remove_review_checkout", fake_remove_review_checkout
    )

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "merged_outside_orchestrator"
    ]
    assert drift

    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    repo_root, pr_number, reviews_dir = removed_calls[0]
    assert repo_root == tmp_path
    assert pr_number == 1
    assert reviews_dir == tmp_path / ".var" / "charlie-work" / "dispatches" / "reviews"

    assert new_state["prs"]["1"]["status"] == "merged"
    assert new_state["prs"]["1"]["review_dispatch_status"] is None
    assert new_state["prs"]["1"]["review_dispatched_at"] is None
    assert new_state["prs"]["1"]["reviewer_pid"] is None
    assert new_state["prs"]["1"]["reviewer_process_start_time"] is None

    # Original state is never mutated in place.
    assert state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_apply_fixes_merged_pr_defers_reap_while_reviewer_alive(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #504: a PR merged externally while its reviewer is alive must not
    have its review checkout removed, dispatch claim cleared, or issue labels
    transitioned until the reviewer exits.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["1"] = {
        "number": 1,
        "issue_number": 10,
        "status": "reviewing",
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 1,
        "branch": "agent/issue-10-x",
        "worktree_path": str(reviews_dir / "pr-1"),
        "prompt_path": str(reviews_dir / "pr-1" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 12345,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-1.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-1.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    removed_calls: list[tuple[Any, int, Any]] = []

    def fake_remove_review_checkout(
        repo_root: Path, pr_number: int, *, reviews_dir: Any = None
    ) -> bool:
        removed_calls.append((repo_root, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.reconcile.remove_review_checkout", fake_remove_review_checkout
    )

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "merged_outside_orchestrator"
    ]
    assert drift

    # Live reviewer: defer the reap.
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert removed_calls == []
    assert gh.labels_added == []
    assert gh.labels_removed == []
    assert new_state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert new_state["prs"]["1"]["status"] == "reviewing"

    # Dead reviewer: proceed with the reap.
    removed_calls.clear()
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    assert removed_calls[0][1] == 1
    assert new_state["prs"]["1"]["status"] == "merged"
    assert new_state["prs"]["1"]["review_dispatch_status"] is None


def test_apply_fixes_closed_unmerged_pr_defers_reap_while_reviewer_alive(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #504: a PR closed without merging while its reviewer is alive
    must not have its review checkout removed, dispatch claim cleared, or
    active labels stripped until the reviewer exits.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["2"] = {
        "number": 2,
        "issue_number": 20,
        "status": "reviewing",
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 2,
        "branch": "agent/issue-20-x",
        "worktree_path": str(reviews_dir / "pr-2"),
        "prompt_path": str(reviews_dir / "pr-2" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 12345,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-2.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-2.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    removed_calls: list[tuple[Any, int, Any]] = []

    def fake_remove_review_checkout(
        repo_root: Path, pr_number: int, *, reviews_dir: Any = None
    ) -> bool:
        removed_calls.append((repo_root, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.reconcile.remove_review_checkout", fake_remove_review_checkout
    )

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_active_labels"
    ]
    assert drift

    # Live reviewer: defer the reap.
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert removed_calls == []
    assert gh.labels_removed == []
    assert new_state["prs"]["2"]["review_dispatch_status"] == "review_dispatch_dispatched"

    # Dead reviewer: proceed with the reap.
    removed_calls.clear()
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    assert removed_calls[0][1] == 2
    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed
    assert new_state["prs"]["2"]["review_dispatch_status"] is None


def test_apply_fixes_dual_writes_reconcile_event_to_events_db(tmp_path: Path) -> None:
    """``apply_fixes`` must pass ``state_path`` through to ``append_event`` --
    without it, every reconcile fix (including aviator_stale_blocked re-queues)
    is invisible to events.db and only survives in the capped 200-entry ring
    in state.json (found via job-cannon audit: 39 merged_outside_orchestrator
    fixes recorded in state.json, zero in events.db)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="aviator_stale_blocked",
            issue_number=None,
            pr_number=1400,
            detail="PR #1400 has a stale Aviator 'blocked' label",
            fix_actions=("remove label 'blocked' from PR #1400",),
            remove_labels=("blocked",),
            add_labels=(),
        )
    ]
    state_path = tmp_path / "state.json"

    apply_fixes(gh, state, drift, config, state_path=state_path)

    events = read_event_log(state_path)
    reconcile_events = [e for e in events if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "aviator_stale_blocked"
    assert reconcile_events[0]["payload"]["pr_number"] == 1400


def test_apply_fixes_has_no_dry_run_parameter() -> None:
    """Issue #1051: ``apply_fixes`` must NOT accept a ``dry_run`` parameter.

    The dry-run invariant for ``mop-up --fix`` is enforced at a single point --
    the ``if fix and not dry_run and drift:`` gate in ``_reconcile_locked``
    (workflow.py), which short-circuits before ``apply_fixes`` is ever called.
    Adding a ``dry_run`` parameter to ``apply_fixes`` would create unreachable
    dead code: the caller guarantees ``dry_run`` is False on every code path
    that reaches ``apply_fixes``, so any ``dry_run``-conditional branch inside
    it can never fire from a real CLI invocation. This test locks in the
    single-point-of-enforcement design so the dead-code pattern is not
    reintroduced (e.g. by a PR that threads ``dry_run`` into ``apply_fixes``
    without also restructuring the caller gate to let it through).
    """
    import inspect

    sig = inspect.signature(apply_fixes)
    assert "dry_run" not in sig.parameters, (
        "apply_fixes must not accept a dry_run parameter (issue #1051): "
        "the caller-level `not dry_run` gate in _reconcile_locked is the "
        "single enforcement point; a dry_run param here would be unreachable "
        "dead code."
    )
