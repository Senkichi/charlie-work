"""Snapshot and guard drift tests for ``reconcile.detect_drift``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1): issue
snapshot pagination, status normalization under incomplete PR snapshots,
fork-PR binding, and the GraphQL-budget / unreadable-snapshot guards.
"""

from __future__ import annotations

import json
import logging
import pytest
from _reconcile_fixtures import (
    FakeGitHub,
    _EmptyStdoutGitHub,
    _issue,
    _pr,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.github import (
    GitHubError,
    GraphQLBudgetError,
)
from charlie_work.reconcile import (
    detect_drift,
    _LIST_LIMIT as reconcile_list_limit,
)
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    empty_state,
)


def test_detect_drift_issue_snapshot_paginated_finalizes_in_window_and_skips_missing(
    caplog,
) -> None:
    """Issue #762/#857: the issue list is now fetched via paginated REST, so the
    snapshot is complete by construction. A closed issue that IS in the snapshot
    is still finalized, while a state-tracked issue whose number does not appear
    anywhere in the snapshot is still skipped silently because its absence is
    unanswerable -- not because the snapshot as a whole is truncated.
    """
    caplog.set_level(logging.WARNING)
    config = OrchestratorConfig()
    closed_issue_number = reconcile_list_limit
    issues = [_issue(i, [config.labels.ready]) for i in range(1, reconcile_list_limit)]
    issues.append(_issue(closed_issue_number, [config.labels.in_progress], state="CLOSED"))
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(closed_issue_number)] = {
        "number": closed_issue_number,
        "status": "dispatched",
    }
    out_of_window_issue_number = reconcile_list_limit + 1000
    state["issues"][str(out_of_window_issue_number)] = {
        "number": out_of_window_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    closed_items = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(closed_items) == 1
    assert closed_items[0].issue_number == closed_issue_number
    assert [item for item in drift if item.kind == "issue_status_normalized"] == []
    assert not any(item.kind == "snapshot_truncated" for item in drift)
    assert "truncated" not in caplog.text.lower()


def test_detect_drift_issue_snapshot_paginated_skips_genuinely_missing_issue() -> None:
    """Issue #762: a state-tracked issue that is simply not in the (now
    complete) issue snapshot is skipped silently, with no truncation warning.
    """
    config = OrchestratorConfig()
    out_of_window_issue_number = reconcile_list_limit + 1000
    issues = [_issue(i, [config.labels.ready]) for i in range(1, reconcile_list_limit + 1)]
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(out_of_window_issue_number)] = {
        "number": out_of_window_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] == []
    assert [item for item in drift if item.kind == "issue_status_normalized"] == []
    assert not any(item.kind == "snapshot_truncated" for item in drift)


def test_detect_drift_snapshot_not_truncated_finalizes_closed_issues() -> None:
    """Issue #259 review: a below-limit snapshot is complete; sweep works."""
    config = OrchestratorConfig()
    closed_issue_number = reconcile_list_limit - 1
    issues = [_issue(i, [config.labels.ready]) for i in range(1, closed_issue_number)]
    issues.append(_issue(closed_issue_number, [config.labels.in_progress], state="CLOSED"))
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(closed_issue_number)] = {
        "number": closed_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "snapshot_truncated"] == []
    closed_items = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(closed_items) == 1
    assert closed_items[0].issue_number == closed_issue_number


def test_detect_drift_issue_status_normalized_skips_under_incomplete_pr_snapshot() -> None:
    """Issue #859: PR-side counterpart of issue #789 (a few lines above it in
    detect_drift) -- reworked per PR #972 review to a PER-ITEM condition.

    ``open_prs_by_issue`` is built from the same ``prs`` snapshot that can be
    provably incomplete once total PR count hits ``_LIST_LIMIT``. If an
    issue's genuinely open PR fell off that page, ``open_prs_by_issue.get(...)``
    returns nothing even though a PR exists, and the pre-#859 code fell
    through to ``target_status = None`` -- silently normalizing a real, live
    issue's status to the untracked baseline with no warning.

    The fix does NOT gate on the global ``pr_snapshot_incomplete`` flag (an
    earlier version of this PR did, and was rejected in review: under
    ``--state all`` that flag is monotonic, so once the repo permanently
    crosses ``_LIST_LIMIT`` PRs it would disable this sweep repo-wide,
    forever -- reproducing the exact #857/#860 failure mode this repo already
    fixed once). Instead it uses issue-specific evidence: state.json's own PR
    record (``state["prs"]``) says this issue has a still-open PR (status not
    yet "closed"/"merged") whose PR number is entirely absent from this
    pass's ``prs`` snapshot. That is this test's setup below. This is the
    regression test for the original bug: it must fail on the pre-#859 code
    and pass on the fix.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    tracked_pr_number = 999999  # deliberately outside the PR snapshot below
    # The issue itself is well inside the issue snapshot and genuinely OPEN;
    # only the PR snapshot is the one under test.
    issues = [_issue(target_issue_number, [])]
    # Exactly _LIST_LIMIT PRs, none linked to target_issue_number and none
    # numbered tracked_pr_number: the PR snapshot is provably incomplete, and
    # this issue's real open PR (which state.json tracks and which exists on
    # GitHub) simply isn't on this page.
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        # "closed" is the reachable value outside ORCHESTRATOR_OWNED_ISSUE_STATUSES
        # (a stale value from a GitHub reopen, per issue #859's own example).
        "status": "closed",
    }
    # state.json's own record: this issue has a still-open tracked PR that
    # happens to be absent from the `prs` snapshot fetched above. This is the
    # issue-specific evidence the per-item guard requires.
    state["prs"][str(tracked_pr_number)] = {
        "number": tracked_pr_number,
        "issue_number": target_issue_number,
        "status": "reviewing",
    }

    drift = detect_drift(gh, state, config)

    bad = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert bad == []

    # Requirement from PR #972 review comment 4: the deferral must be named
    # in the drift log, not silent.
    deferred = [
        item
        for item in drift
        if item.kind == "snapshot_truncated" and "issue_status_normalized deferred" in item.detail
    ]
    assert len(deferred) == 1
    assert str(target_issue_number) in deferred[0].detail


def test_detect_drift_issue_status_normalized_none_still_fires_with_no_tracked_pr_anywhere() -> (
    None
):
    """Issue #859 review comment 2/3: proves the fix is NOT a repo-wide kill
    switch once the PR snapshot is incomplete.

    Same incomplete-PR-snapshot setup as the regression test above, but this
    issue has no PR anywhere -- not in the GitHub snapshot, and not tracked in
    state.json either. There is no issue-specific evidence to distrust the
    negative answer, so ``target_status = None`` must still fire exactly as
    it did before #859, even while the global snapshot is provably
    incomplete. Without this test, a broad guard keyed on the global
    ``pr_snapshot_incomplete`` flag (the shape rejected in review) would pass
    every other test in this module while silently disabling
    ``issue_status_normalized`` for the entire repo once PR count crosses
    ``_LIST_LIMIT`` -- exactly the #857/#860 regression this rework exists to
    avoid reintroducing.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [])]
    # Exactly _LIST_LIMIT PRs, none linked to target_issue_number: the PR
    # snapshot is provably incomplete (same global condition as the
    # regression test), but state.json tracks NO PR for this issue at all.
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }
    # No state["prs"] entry for this issue at all -- state.json has zero
    # opinion about a PR existing for it.

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert len(matches) == 1

    # And the per-item deferral warning must NOT claim this issue was
    # deferred, since it wasn't.
    deferred = [
        item
        for item in drift
        if item.kind == "snapshot_truncated" and "issue_status_normalized deferred" in item.detail
    ]
    assert deferred == []


def test_detect_drift_issue_status_normalized_none_still_fires_when_pr_snapshot_complete() -> None:
    """Discriminator: under a COMPLETE PR snapshot, a genuinely absent open PR
    still normalizes the stale status to None.

    Proves the #859 guard is not an over-broad kill switch on
    ``issue_status_normalized``'s None outcome -- it only defers the
    normalization when the snapshot can't support the "no open PR"
    conclusion, not always.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [])]
    # Well under _LIST_LIMIT: the PR snapshot is complete.
    prs = [_pr(2, "OPEN", head_ref="agent/issue-99999-x")]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert len(matches) == 1


def test_detect_drift_issue_status_normalized_closed_wins_despite_incomplete_pr_snapshot() -> None:
    """CLOSED-on-GitHub still wins first, even under an incomplete PR snapshot.

    ``target_status = "closed"`` is derived from ``_issue_state(issue)``, not
    from the PR snapshot at all, so the #859 guard (which only applies to the
    would-be-None branch) must never intercept it.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [], state="CLOSED")]
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        # Not in VALID_ISSUE_STATUSES at all, and not "closed" either, so a
        # real transition to "closed" is expected regardless of PR truncation.
        "status": "garbage-value",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized" and item.issue_number == target_issue_number
    ]
    assert len(matches) == 1
    assert matches[0].new_status == "closed"


def test_detect_drift_issue_status_normalized_open_pr_wins_despite_incomplete_pr_snapshot() -> (
    None
):
    """A positively-observed open PR still normalizes to PASSIVE_OPEN_STATUS
    even under an incomplete PR snapshot.

    The #859 guard only intercepts the would-be-None branch; a PR that IS
    present in the (still-incomplete) snapshot is a positive observation, not
    an absence, so it must keep winning.

    The issue already carries the ``pr_open`` label (not an active label
    outside {pr_open, reviewing}, and not missing pr_open either) so the
    earlier ``issue_active_label_with_open_pr`` self-heal sweep -- which also
    reacts to an issue with an open PR -- does not fire first and mark this
    issue as already repaired; this test is isolating the later
    ``issue_status_normalized`` sweep specifically.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [config.labels.pr_open])]
    prs = [_pr(1, "OPEN", head_ref=f"agent/issue-{target_issue_number}-x")]
    prs += [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(2, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized" and item.issue_number == target_issue_number
    ]
    assert len(matches) == 1
    assert matches[0].new_status == PASSIVE_OPEN_STATUS


def test_detect_drift_fork_pr_branch_name_does_not_bind() -> None:
    """Issue #9: Fork PRs must not bind via branch name (attacker-controlled)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1,
                "MERGED",
                head_ref="agent/issue-42-fix",
                is_cross_repository=True,
            )
        ],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    # The fork PR should NOT bind to issue 42 via branch name, so drift
    # should be detected for the PR status but NOT for the issue labels.
    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    # The drift item should have issue_number=None because the fork PR
    # didn't bind to issue 42.
    assert matches[0].issue_number is None
    assert matches[0].pr_number == 1


def test_detect_drift_fork_pr_closing_keyword_does_not_bind() -> None:
    """Issue #9: Fork PRs must NOT bind via closing keywords for lifecycle purposes."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1,
                "MERGED",
                head_ref="attacker-branch",
                body="Closes #42",
                is_cross_repository=True,
            )
        ],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    # The fork PR should NOT bind to issue 42 via closing keyword, so drift
    # should be detected for the PR status but NOT for the issue labels.
    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    # The drift item should have issue_number=None because the fork PR
    # didn't bind to issue 42.
    assert matches[0].issue_number is None
    assert matches[0].pr_number == 1


def test_detect_drift_defers_when_graphql_rate_limit_below_threshold() -> None:
    """Issue #398: detect_drift must refuse to start a quota-heavy sweep when the
    GraphQL budget is below the configured threshold.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
        rate_limit_sufficient=False,
        rate_limit_remaining=100,
        rate_limit_reset=1234567890,
    )

    with pytest.raises(GraphQLBudgetError) as exc_info:
        detect_drift(gh, empty_state(), config)

    assert exc_info.value.remaining == 100
    assert exc_info.value.reset_at == 1234567890
    assert exc_info.value.threshold == config.runtime.graphql_rate_limit_threshold


def test_detect_drift_leaves_state_untouched_when_snapshot_unreadable() -> None:
    """End-to-end property: a failed read aborts the pass instead of mutating.

    This is what makes the downstream sweep correct *by construction* -- once
    an unreadable snapshot can no longer arrive as ``[]``, an empty ``prs``
    genuinely means "GitHub has zero PRs" and no "suspiciously empty"
    heuristic is needed to second-guess it.
    """
    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["999"] = {
        "issue_number": 5,
        "status": "reviewing",
        "decision": "approved",
    }
    before = json.dumps(state, sort_keys=True)

    with pytest.raises(GitHubError):
        detect_drift(_EmptyStdoutGitHub(), state, config)  # type: ignore[arg-type]

    assert json.dumps(state, sort_keys=True) == before
    assert state["prs"]["999"]["decision"] == "approved"
