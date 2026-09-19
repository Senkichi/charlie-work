"""GitHub-state drift tests for ``reconcile.detect_drift``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
merged/closed-PR convergence, issue active-label drift, and terminal-state
staleness. The session/worktree drift tests live in
``tests/test_reconcile_drift_sessions.py``; the snapshot
pagination/normalization and guard tests live in
``tests/test_reconcile_drift_snapshot.py``.
"""

from __future__ import annotations

import logging
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _reconcile_fixtures import (
    FakeGitHub,
    _issue,
    _pr,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import (
    detect_drift,
    _LIST_LIMIT as reconcile_list_limit,
)
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    empty_state,
    is_claim_stale,
)


def test_detect_drift_makes_zero_mutating_calls() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
    )
    state = empty_state()

    detect_drift(gh, state, config)

    assert gh.labels_added == []
    assert gh.labels_removed == []
    # PRs and issues are both fetched via paginated REST snapshots.
    assert any(call[0] == "api" and "pulls?state=all" in call[1] for call in gh.run_calls)
    assert any(call[0] == "api" and "issues?state=all" in call[1] for call in gh.run_calls)


def test_detect_drift_finds_merged_outside_orchestrator() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    item = matches[0]
    assert item.pr_number == 1
    assert item.issue_number == 10
    assert any("merged" in action for action in item.fix_actions)


def test_detect_drift_merged_but_state_already_correct_and_labels_clean_is_not_drift() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.done])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "merged"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "merged_outside_orchestrator"] == []


def test_detect_drift_finds_closed_unmerged_pr_with_active_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "closed_unmerged_pr_active_labels"]
    assert len(matches) == 1
    assert matches[0].issue_number == 20
    assert matches[0].pr_number == 2
    assert set(matches[0].fix_actions) == {
        f"remove label '{config.labels.pr_open}' from issue #20",
        f"remove label '{config.labels.reviewing}' from issue #20",
    }


def test_detect_drift_finds_state_pr_missing_on_github() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_pr_missing_on_github"]
    assert len(matches) == 1
    assert matches[0].pr_number == 999
    assert matches[0].issue_number == 5


def test_detect_drift_pr_list_paginated_finds_state_pr_missing_on_github(
    caplog,
) -> None:
    """Issue #762: the PR list is fetched via paginated REST and is complete,
    so state_pr_missing_on_github is no longer gated on an artificial
    ``_LIST_LIMIT`` snapshot cap. A tracked PR that is genuinely absent from the
    complete snapshot must still be flagged.
    """
    caplog.set_level(logging.WARNING)
    config = OrchestratorConfig()
    # More than _LIST_LIMIT PRs, none numbered 999. With a single 500-item page
    # the snapshot would be provably truncated, but the paginated fetch must now
    # return all of them, so PR #999 (tracked in state, missing on GitHub) is
    # unambiguously missing and must be reported.
    prs = [_pr(i, "OPEN") for i in range(1, reconcile_list_limit + 101)]
    gh = FakeGitHub(prs=prs, issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}

    drift = detect_drift(gh, state, config)

    missing = [item for item in drift if item.kind == "state_pr_missing_on_github"]
    assert len(missing) == 1
    assert missing[0].pr_number == 999
    assert not any(item.kind == "snapshot_truncated" and "PR" in item.detail for item in drift)
    assert "incomplete" not in caplog.text.lower()


def test_detect_drift_issue_list_paginated_finalizes_out_of_window_closed_issue(
    caplog,
) -> None:
    """Issue #762: the issue list is fetched via paginated REST and is complete,
    so state_active_status_issue_closed is no longer gated on an artificial
    ``_LIST_LIMIT`` snapshot cap. A closed issue beyond the old cap must still be
    finalized.
    """
    caplog.set_level(logging.WARNING)
    config = OrchestratorConfig()
    # More than _LIST_LIMIT issues, with the closed one at the end of the list.
    closed_issue_number = reconcile_list_limit + 100
    issues = [_issue(i, [config.labels.ready]) for i in range(1, closed_issue_number)]
    issues.append(_issue(closed_issue_number, [config.labels.in_progress], state="CLOSED"))
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(closed_issue_number)] = {
        "number": closed_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    closed_items = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(closed_items) == 1
    assert closed_items[0].issue_number == closed_issue_number
    assert not any(item.kind == "snapshot_truncated" and "Issue" in item.detail for item in drift)
    assert "truncated" not in caplog.text.lower()


def test_detect_drift_finds_issue_active_label_no_open_pr(tmp_path: Path) -> None:
    """Issue #417: this fix path must also add the ready label back, not just
    remove the stale active one -- otherwise a --fix run leaves the issue with
    no dispatch-eligible label at all (mirrors the sibling
    session_failed_relabeled kind).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(30, [config.labels.in_progress])])
    state = empty_state()

    # Ensure no sessions directory exists (to avoid picking up session drift from other tests)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)  # No repo_root, so session detection shouldn't run

    matches = [item for item in drift if item.kind == "issue_active_label_no_open_pr"]
    assert len(matches) >= 1  # May be multiple if both adapters read the same issue
    assert matches[0].issue_number == 30
    assert matches[0].fix_actions == (
        f"remove label '{config.labels.in_progress}' from issue #30",
        f"add label '{config.labels.ready}' to issue #30",
    )
    assert matches[0].remove_labels == (config.labels.in_progress,)
    assert matches[0].add_labels == (config.labels.ready,)


def test_detect_drift_issue_active_label_with_open_pr_is_not_drift() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.pr_open])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "issue_active_label_no_open_pr"] == []
    assert [item for item in drift if item.kind == "issue_active_label_with_open_pr"] == []


def test_detect_drift_finds_issue_active_label_with_open_pr() -> None:
    """Issue #515: an issue stuck on needs_rework while an open PR exists is drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "issue_active_label_with_open_pr"]
    assert len(matches) == 1
    assert matches[0].issue_number == 30
    assert matches[0].pr_number == 3
    assert matches[0].remove_labels == (config.labels.needs_rework,)
    assert matches[0].add_labels == (config.labels.pr_open,)
    assert matches[0].fix_actions == (
        f"remove label '{config.labels.needs_rework}' from issue #30",
        f"add label '{config.labels.pr_open}' to issue #30",
        f"set state issues[30].status = {PASSIVE_OPEN_STATUS!r}",
    )


def test_detect_drift_skips_issue_whose_status_corroborates_the_label() -> None:
    """Issue #1092: reconcile must not revert a lane the orchestrator is holding.

    This fixture is the exact shape of the production deadlock (job-cannon issue
    1487): an issue queued behind the dispatch concurrency cap carries
    ``agent:needs-rework`` with ``status == "rework_requested"`` and an open PR,
    and has no live worker session because it has not been dispatched yet.

    "Active label + open PR + no live worker" -- the rule's only pre-existing
    false-positive guard -- is satisfied by that state, so the rule fired and
    reset the status to PASSIVE_OPEN_STATUS. Since ``_dispatch_rework_impl``
    selects on ``status == "rework_requested"``, that removed the issue from the
    dispatch scan; the stranded-PR router then re-set it, and the two alternated
    every reconcile pass with no forward progress.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["30"] = {"number": 30, "status": "rework_requested"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "issue_active_label_with_open_pr"] == []


def test_detect_drift_still_heals_stale_active_label_after_failed_dispatch() -> None:
    """Issue #1092: ``dispatch_failed`` stays outside the exemption.

    A rework loop that failed to dispatch is precisely the "left over from a
    failed rework loop" case the rule was written for (#515). Nothing is holding
    the issue, so nothing will re-set the label, and the self-heal must still run.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["30"] = {"number": 30, "status": "dispatch_failed"}

    matches = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_active_label_with_open_pr"
    ]

    assert len(matches) == 1
    assert matches[0].issue_number == 30
    assert matches[0].new_status == PASSIVE_OPEN_STATUS


def test_detect_drift_finds_done_label_with_active_labels(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(40, [config.labels.done, config.labels.reviewing])],
    )
    state = empty_state()

    # Ensure no sessions directory exists (to avoid picking up session drift from other tests)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)  # No repo_root, so session detection shouldn't run

    matches = [item for item in drift if item.kind == "done_label_with_active_labels"]
    assert len(matches) >= 1  # May be multiple if both adapters read the same issue
    assert matches[0].issue_number == 40
    assert matches[0].fix_actions == (f"remove label '{config.labels.reviewing}' from issue #40",)
    assert matches[0].remove_labels == (config.labels.reviewing,)


# --- issue #947: agent:human-needed silently invisible past a configurable age ---
def test_detect_drift_finds_terminal_state_stale_via_terminal_since(tmp_path: Path) -> None:
    """A `terminal_since` stamp (written by `_escalate_issue` since #947) past
    the configured threshold fires `terminal_state_stale` with the parked
    issue's number and a numeric age, not merely "some event fired"."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(894, [config.labels.human_needed])])
    state = empty_state()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    state["issues"]["894"] = {
        "number": 894,
        "status": "escalated",
        "terminal_since": "2026-01-05T00:00:00Z",  # 5 days before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert "5.0 day" in matches[0].detail
    assert matches[0].fix_actions == ()


def test_detect_drift_finds_terminal_state_stale_for_operator_queue(tmp_path: Path) -> None:
    """Issue #1266 counterpart of the test above: a mechanical escalation
    parks on `agent:operator-queue` instead of `agent:human-needed`, and the
    #947 staleness alert must watch that label too -- otherwise an issue
    whose de-escalation sweep stopped clearing it (e.g. sweep itself broken,
    not merely mid-retry) would sit in the sink forever with no alert at
    all, silently reintroducing the exact invisibility #947 fixed for the
    judgment-escalation case. The detail message must name the label that is
    actually present (`operator-queue`), not hardcode `human-needed`."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(894, [config.labels.operator_queue])])
    state = empty_state()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    state["issues"]["894"] = {
        "number": 894,
        "status": "escalated",
        "reason_class": "mechanical",
        "terminal_since": "2026-01-05T00:00:00Z",  # 5 days before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert "5.0 day" in matches[0].detail
    assert config.labels.operator_queue in matches[0].detail
    assert config.labels.human_needed not in matches[0].detail
    assert matches[0].fix_actions == ()


def test_detect_drift_terminal_state_stale_not_yet_due(tmp_path: Path) -> None:
    """A fresh escalation (age below the configured threshold) must not fire
    -- this is the negative control for the positive case above."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(895, [config.labels.human_needed])])
    state = empty_state()
    now = datetime(2026, 1, 10, 1, 0, 0, tzinfo=UTC)
    state["issues"]["895"] = {
        "number": 895,
        "status": "escalated",
        "terminal_since": "2026-01-10T00:00:00Z",  # 1 hour before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    assert [item for item in drift if item.kind == "terminal_state_stale"] == []


def test_detect_drift_terminal_state_stale_legacy_escalation_via_events_db(
    tmp_path: Path,
) -> None:
    """Issue #894 shape: an issue escalated BEFORE #947 shipped carries no
    `terminal_since` field at all. The detector must still report a real
    numeric age (not "never observed") by falling back to the most recent
    escalation-transition event in events.db -- the same CI-verified kind
    registry `_backfill_missing_reason_classes` already relies on. Without
    this fallback tier the original design silently degraded #894 itself to
    the "never observed" bucket, which is the exact bug this PR fixes."""
    from charlie_work.instrumentation import log_event

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(894, [config.labels.human_needed])])
    state = empty_state()
    # Deliberately no terminal_since / merged_pr_mention_flagged_at: this is
    # the legacy (pre-#947) shape.
    state["issues"]["894"] = {"number": 894, "status": "escalated"}

    state_path = tmp_path / ".var" / "charlie-work" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_event(state_path, "session_failed_escalated", {"issue_number": 894}, repo="test-repo")

    now = datetime.now(UTC) + timedelta(days=5)
    drift = detect_drift(gh, state, config, state_path=state_path, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert "never observed" not in matches[0].detail
    assert "4." in matches[0].detail or "5." in matches[0].detail


def test_detect_drift_terminal_state_stale_events_db_ignores_non_escalation_kinds(
    tmp_path: Path,
) -> None:
    """A non-escalation event for the issue (e.g. a routine dispatch record)
    must NOT be mistaken for an escalation-transition timestamp -- proves the
    events.db fallback filters by the CI-verified escalation-kind registry,
    not "any event for this issue_number"."""
    from charlie_work.instrumentation import log_event

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(898, [config.labels.human_needed])])
    state = empty_state()
    state["issues"]["898"] = {"number": 898, "status": "escalated"}

    state_path = tmp_path / ".var" / "charlie-work" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # "dispatch" is not an escalation-transition kind: it neither ends in
    # "_escalated" nor is registered in ESCALATION_REASON_CLASS_BY_EVENT_KIND
    # / DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS.
    log_event(state_path, "dispatch", {"issue_number": 898}, repo="test-repo")

    drift = detect_drift(gh, state, config, state_path=state_path)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 898
    assert "never observed" in matches[0].detail


def test_detect_drift_terminal_state_stale_never_observed_without_any_timestamp(
    tmp_path: Path,
) -> None:
    """No `terminal_since`, no `merged_pr_mention_flagged_at`, and no
    matching events.db row (or no `state_path` at all): the detector must
    still surface the issue immediately, distinctly labeled "never observed"
    rather than silently defaulting an unknown age to "healthy" -- mirroring
    `classify_backlog_reachability`'s `observed: False` precedent."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(896, [config.labels.human_needed])])
    state = empty_state()
    state["issues"]["896"] = {"number": 896, "status": "escalated"}

    drift = detect_drift(gh, state, config)  # No state_path passed at all.

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 896
    assert "never observed" in matches[0].detail


def test_detect_drift_terminal_state_stale_ignores_done_label() -> None:
    """`agent:done` is a normal, expected terminal state (issue closed via
    the ordinary lifecycle) -- it must never be treated as a stuck
    human-needed issue, even though both are members of `labels.terminal`."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(897, [config.labels.done])])
    state = empty_state()

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "terminal_state_stale"] == []


def test_detect_drift_finds_state_active_status_issue_closed() -> None:
    """Issue #259: a closed issue with an active state-machine status is drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(matches) == 1
    assert matches[0].issue_number == 259
    assert matches[0].fix_actions == ("set state issues[259].status = 'closed'",)


def test_detect_drift_state_active_status_issue_closed_removes_active_labels() -> None:
    """Issue #259: lingering active labels are stripped from the closed issue."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.in_progress], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(matches) == 1
    assert matches[0].remove_labels == (config.labels.in_progress,)
    assert f"remove label '{config.labels.in_progress}' from issue #259" in matches[0].fix_actions


def test_detect_drift_surfaces_stale_dispatch_pending_claims() -> None:
    """Stale dispatch_pending claims must be detected as drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Mock is_claim_stale to return True for our test timestamp
    original_is_claim_stale = is_claim_stale

    def _mock_is_claim_stale(claim_timestamp: str | None) -> bool:
        if claim_timestamp == "2020-01-01T00:00:00+00:00":
            return True  # Treat this specific timestamp as stale
        return original_is_claim_stale(claim_timestamp)

    # Temporarily replace is_claim_stale in the reconcile module
    import charlie_work.reconcile as reconcile_module

    original_reconcile_is_claim_stale = reconcile_module.is_claim_stale
    reconcile_module.is_claim_stale = _mock_is_claim_stale

    try:
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatch_pending",
            "dispatch_pending_at": "2020-01-01T00:00:00+00:00",  # Stale timestamp
        }

        drift = detect_drift(gh, state, config)

        stale_claim_drift = [d for d in drift if d.kind == "stale_dispatch_pending_claim"]
        assert len(stale_claim_drift) == 1
        assert stale_claim_drift[0].issue_number == 123
        assert "stale dispatch_pending claim" in stale_claim_drift[0].detail
    finally:
        # Restore original function
        reconcile_module.is_claim_stale = original_reconcile_is_claim_stale
