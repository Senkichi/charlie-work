"""Closed-unmerged PR convergence tests (issue #558 family, plus #1066/#1398).

Split out of ``test_fix_reconcile.py`` in the Track-1 shoulder split campaign
(issue #1575). Every test in this module exercises the ``closed_unmerged_*``
drift kinds: the PR-side ``closed_unmerged_pr_state_converged`` rule, the
issue-side ``closed_unmerged_pr_issue_state_converged`` dormant-status rule,
their idempotency / field-preservation / independence contracts, and the
issue #1398 guard that keeps a stale closed PR from stripping the labels and
status of a session that postdates the PR close.

Reuses the lightweight ``FakeGitHub``/``_pr``/``_issue`` fixtures defined in
``test_reconcile.py``'s shared fixture module ``_reconcile_fixtures.py``
(pytest's rootless import mode makes this a plain top-level import, the same
pattern ``test_fix_reconcile.py`` itself uses).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import apply_fixes, detect_drift
from charlie_work.state import empty_state

from _reconcile_fixtures import FakeGitHub, _issue, _pr


# ---------------------------------------------------------------------------
# Issue #558: closed-unmerged PR state entry convergence
# ---------------------------------------------------------------------------


def test_closed_unmerged_pr_with_janitor_blocked_status_converges_to_closed() -> None:
    """A state PR entry stuck in 'janitor_blocked' while GitHub reports the
    PR CLOSED (unmerged) must converge to 'closed' so the janitor stops
    re-fetching it every pass.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(528, "CLOSED", head_ref="agent/issue-100-x")],
        issues=[_issue(100, [])],
    )
    state = empty_state()
    state["prs"]["528"] = {"number": 528, "issue_number": 100, "status": "janitor_blocked"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1
    assert drift[0].pr_number == 528
    assert drift[0].new_status == "closed"

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["528"]["status"] == "closed"
    assert new_state["prs"]["528"]["issue_number"] == 100


def test_closed_unmerged_pr_with_rework_requested_status_converges_to_closed() -> None:
    """A state PR entry stuck in 'rework_requested' while GitHub reports the
    PR CLOSED (unmerged) must converge to 'closed'.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x")],
        issues=[_issue(495, [])],
    )
    state = empty_state()
    state["prs"]["500"] = {"number": 500, "issue_number": 495, "status": "rework_requested"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1
    assert drift[0].new_status == "closed"

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["500"]["status"] == "closed"


def test_closed_unmerged_pr_with_reviewing_status_converges_to_closed() -> None:
    """A state PR entry in 'reviewing' (the passive open-PR placeholder) while
    GitHub reports the PR CLOSED (unmerged) must converge to 'closed'.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(700, "CLOSED", head_ref="agent/issue-200-x")],
        issues=[_issue(200, [])],
    )
    state = empty_state()
    state["prs"]["700"] = {"number": 700, "issue_number": 200, "status": "reviewing"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["700"]["status"] == "closed"


def test_closed_unmerged_pr_with_escalated_status_converges_to_closed() -> None:
    """A state PR entry stuck in 'escalated' while GitHub reports the PR
    CLOSED (unmerged) must converge to 'closed'.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(800, "CLOSED", head_ref="agent/issue-300-x")],
        issues=[_issue(300, [])],
    )
    state = empty_state()
    state["prs"]["800"] = {"number": 800, "issue_number": 300, "status": "escalated"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["800"]["status"] == "closed"


def test_merged_pr_is_not_touched_by_closed_unmerged_state_converged() -> None:
    """A MERGED PR must never fire closed_unmerged_pr_state_converged -- that
    is merged_outside_orchestrator's job.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(900, "MERGED", head_ref="agent/issue-400-x")],
        issues=[_issue(400, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["900"] = {"number": 900, "issue_number": 400, "status": "reviewing"}

    drift = detect_drift(gh, state, config)
    assert [item for item in drift if item.kind == "closed_unmerged_pr_state_converged"] == []
    # merged_outside_orchestrator should fire instead
    assert [item for item in drift if item.kind == "merged_outside_orchestrator"] != []


def test_closed_pr_already_closed_status_is_no_op_idempotent() -> None:
    """A state PR entry that is already 'closed' while GitHub reports the PR
    CLOSED must NOT fire closed_unmerged_pr_state_converged -- it is already
    converged (idempotent).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(548, "CLOSED", head_ref="agent/issue-500-x")],
        issues=[_issue(500, [])],
    )
    state = empty_state()
    state["prs"]["548"] = {"number": 548, "issue_number": 500, "status": "closed"}

    drift = detect_drift(gh, state, config)
    assert [item for item in drift if item.kind == "closed_unmerged_pr_state_converged"] == []


def test_closed_unmerged_pr_state_converged_preserves_other_fields() -> None:
    """apply_fixes must preserve unrelated fields on the PR entry (e.g.
    decision, reviewed_head_sha) while setting status to 'closed'.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(528, "CLOSED", head_ref="agent/issue-100-x")],
        issues=[_issue(100, [])],
    )
    state = empty_state()
    state["prs"]["528"] = {
        "number": 528,
        "issue_number": 100,
        "status": "janitor_blocked",
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["528"]["status"] == "closed"
    assert new_state["prs"]["528"]["decision"] == "request_changes"
    assert new_state["prs"]["528"]["reviewed_head_sha"] == "abc123"


def test_closed_unmerged_pr_state_converged_fires_alongside_active_labels() -> None:
    """Both closed_unmerged_pr_active_labels (issue-side) and
    closed_unmerged_pr_state_converged (PR-side) may fire for the same PR;
    they are independent and must not suppress each other.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(528, "CLOSED", head_ref="agent/issue-100-x")],
        issues=[_issue(100, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["528"] = {"number": 528, "issue_number": 100, "status": "janitor_blocked"}

    drift = detect_drift(gh, state, config)
    kinds = {item.kind for item in drift}
    assert "closed_unmerged_pr_active_labels" in kinds
    assert "closed_unmerged_pr_state_converged" in kinds


def test_closed_unmerged_pr_state_converged_second_pass_is_idempotent() -> None:
    """After apply_fixes converges the status to 'closed', a second
    detect_drift pass must not re-emit the drift item.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(528, "CLOSED", head_ref="agent/issue-100-x")],
        issues=[_issue(100, [])],
    )
    state = empty_state()
    state["prs"]["528"] = {"number": 528, "issue_number": 100, "status": "janitor_blocked"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["528"]["status"] == "closed"

    # Second pass: no drift for this kind
    second_drift = [
        item
        for item in detect_drift(gh, new_state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert second_drift == []


# ---------------------------------------------------------------------------
# Issue #558 (rework): issue-side status convergence. The PR-side
# closed_unmerged_pr_state_converged rule converges the PR entry but defers
# the linked issue's disposition to the existing closed-unmerged issue-side
# handling. That handling (closed_unmerged_pr_active_labels) only strips
# GitHub labels and never touches state["issues"][n]["status"], and
# state_active_status_issue_closed only fires when the GitHub issue itself is
# CLOSED. So an OPEN issue stuck in an ACTIVE_STATE_STATUS (e.g.
# "rework_requested") whose PR closed-unmerged is invisible to both -- and
# dispatch_rework's state-driven candidate scan calls gh.issue_view() on it
# every loop pass forever. The new closed_unmerged_pr_issue_state_converged
# kind closes that gap by dropping the issue's status key (dormant baseline).
# ---------------------------------------------------------------------------


def test_closed_unmerged_pr_issue_rework_requested_status_converges_to_dormant() -> None:
    """An OPEN issue stuck in 'rework_requested' whose PR is CLOSED-unmerged
    must have its state status dropped (dormant baseline) so
    dispatch_rework's state-driven candidate scan stops selecting it -- and
    stops calling gh.issue_view() on it every pass.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x")],
        issues=[_issue(495, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["495"] = {"number": 495, "status": "rework_requested"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    assert len(drift) == 1
    assert drift[0].issue_number == 495
    assert drift[0].pr_number == 500
    assert drift[0].new_status is None

    new_state = apply_fixes(gh, state, drift, config)
    # The status key is dropped (dormant baseline), not set to a placeholder.
    assert "status" not in new_state["issues"]["495"]
    # Other fields are preserved.
    assert new_state["issues"]["495"]["number"] == 495


@pytest.mark.parametrize("stuck_status", ["reviewing", "dispatched"])
def test_closed_unmerged_pr_issue_other_active_statuses_converge_to_dormant(
    stuck_status: str,
) -> None:
    """Every ACTIVE_STATE_STATUS on an OPEN issue whose PR closed-unmerged
    must converge to dormant -- not just 'rework_requested'. "escalated" is
    the deliberate exception, covered separately below (issue #1066).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(700, "CLOSED", head_ref="agent/issue-200-x")],
        issues=[_issue(200, [])],
    )
    state = empty_state()
    state["issues"]["200"] = {"number": 200, "status": stuck_status}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    assert len(drift) == 1
    new_state = apply_fixes(gh, state, drift, config)
    assert "status" not in new_state["issues"]["200"]


def test_closed_unmerged_pr_issue_escalated_status_is_not_converged_to_dormant() -> None:
    """Issue #1066: an OPEN issue whose status is 'escalated' must NOT have
    its status dropped when its linked PR closes unmerged, unlike every other
    ACTIVE_STATE_STATUSES member. 'escalated' is a human-owned terminal
    disposition (agent:human-needed stays live regardless of what happens to
    this PR) -- dropping the status key detaches the state entry from that
    still-live label with no automated repair path back into the human
    queue. This mirrors the ORCHESTRATOR_OWNED_ISSUE_STATUSES guard the
    sibling issue_status_normalized sweep already applies, and matches a
    real production divergence (issue #894 via PR #948).

    Without the fix this asserts `len(drift) == 0`, which is exactly what
    the pre-#1066 code violates -- reverting the source change alone (fix
    stays applied) must fail this test.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(948, "CLOSED", head_ref="agent/issue-894-x")],
        issues=[_issue(894, [config.labels.human_needed])],
    )
    state = empty_state()
    state["issues"]["894"] = {"number": 894, "status": "escalated"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    assert drift == []

    # apply_fixes on the (empty) drift list must leave the escalated status
    # entry completely untouched.
    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["894"]["status"] == "escalated"


def test_closed_unmerged_pr_issue_escalated_status_stable_across_passes() -> None:
    """A second detect_drift pass over an already-escalated issue whose PR is
    closed-unmerged must also emit nothing -- this is not a transient
    no-drift result, it is a stable exclusion.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(700, "CLOSED", head_ref="agent/issue-200-x")],
        issues=[_issue(200, [config.labels.human_needed])],
    )
    state = empty_state()
    state["issues"]["200"] = {"number": 200, "status": "escalated"}

    for _ in range(2):
        drift = [
            item
            for item in detect_drift(gh, state, config)
            if item.kind == "closed_unmerged_pr_issue_state_converged"
        ]
        assert drift == []
        state = apply_fixes(gh, state, drift, config)
        assert state["issues"]["200"]["status"] == "escalated"


def test_closed_unmerged_pr_issue_state_converged_skips_closed_github_issue() -> None:
    """A CLOSED GitHub issue with an active status is owned by
    state_active_status_issue_closed (which sets status to 'closed'), NOT by
    the new issue-side convergence rule. The new rule must only fire for
    OPEN issues whose PR closed-unmerged.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(800, "CLOSED", head_ref="agent/issue-300-x")],
        issues=[_issue(300, [], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["300"] = {"number": 300, "status": "rework_requested"}

    drift = detect_drift(gh, state, config)
    assert [
        item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"
    ] == []
    # state_active_status_issue_closed owns this shape instead.
    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] != []


def test_closed_unmerged_pr_issue_state_converged_idempotent() -> None:
    """After apply_fixes drops the status key, a second detect_drift pass
    must not re-emit the issue-side convergence drift.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x")],
        issues=[_issue(495, [])],
    )
    state = empty_state()
    state["issues"]["495"] = {"number": 495, "status": "rework_requested"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    new_state = apply_fixes(gh, state, drift, config)
    assert "status" not in new_state["issues"]["495"]

    second_drift = [
        item
        for item in detect_drift(gh, new_state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    assert second_drift == []


def test_closed_unmerged_pr_issue_state_converged_fires_alongside_active_labels() -> None:
    """closed_unmerged_pr_active_labels (label strip), the PR-side
    closed_unmerged_pr_state_converged, and the new issue-side
    closed_unmerged_pr_issue_state_converged may all fire for the same PR;
    they are independent and must not suppress each other.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(528, "CLOSED", head_ref="agent/issue-100-x")],
        issues=[_issue(100, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["528"] = {"number": 528, "issue_number": 100, "status": "janitor_blocked"}
    state["issues"]["100"] = {"number": 100, "status": "reviewing"}

    kinds = {item.kind for item in detect_drift(gh, state, config)}
    assert "closed_unmerged_pr_active_labels" in kinds
    assert "closed_unmerged_pr_state_converged" in kinds
    assert "closed_unmerged_pr_issue_state_converged" in kinds


def test_closed_unmerged_pr_issue_state_converged_preserves_other_fields() -> None:
    """apply_fixes must preserve unrelated fields on the issue entry (e.g.
    title, url) while dropping the status key.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x")],
        issues=[_issue(495, [])],
    )
    state = empty_state()
    state["issues"]["495"] = {
        "number": 495,
        "status": "rework_requested",
        "title": "cached title",
        "url": "https://example.test/issues/495",
    }

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_issue_state_converged"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert "status" not in new_state["issues"]["495"]
    assert new_state["issues"]["495"]["title"] == "cached title"
    assert new_state["issues"]["495"]["url"] == "https://example.test/issues/495"


def test_open_issue_with_open_pr_and_rework_requested_is_not_converged() -> None:
    """Baseline guard: an OPEN issue with an OPEN PR and 'rework_requested'
    status is a legitimate in-flight rework candidate -- the issue-side
    convergence rule must NOT fire (the PR is not closed-unmerged).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(456, "OPEN", head_ref="agent/issue-123-x")],
        issues=[_issue(123, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["123"] = {"number": 123, "status": "rework_requested"}

    drift = detect_drift(gh, state, config)
    assert [
        item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"
    ] == []


# ---------------------------------------------------------------------------
# Issue #558 (Minor 2): symmetric None-status handling. The sibling OPEN-PR
# repair branch (pr_status_normalized) normalizes a tracked PR with no status
# key to the passive placeholder. The CLOSED-unmerged branch must symmetrically
# converge a tracked PR with no status key to 'closed' instead of skipping it.
# ---------------------------------------------------------------------------


def test_closed_unmerged_tracked_pr_with_none_status_converges_to_closed() -> None:
    """A tracked PR (state entry exists) that is CLOSED-unmerged on GitHub
    but has no 'status' key must converge to 'closed' -- mirroring the
    OPEN-PR branch's handling of a tracked PR with no status. Previously
    the CLOSED branch gated on `state_status is not None` and skipped it.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(529, "CLOSED", head_ref="agent/issue-110-x")],
        issues=[_issue(110, [])],
    )
    state = empty_state()
    # Tracked PR with no status key at all.
    state["prs"]["529"] = {"number": 529, "issue_number": 110}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1
    assert drift[0].new_status == "closed"

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["529"]["status"] == "closed"
    assert new_state["prs"]["529"]["issue_number"] == 110


def test_closed_unmerged_untracked_pr_is_not_invented() -> None:
    """An untracked closed-unmerged PR (no state entry) must NOT get an
    entry invented by the convergence rule -- the same boundary the OPEN
    branch respects (it only normalizes tracked PRs).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(999, "CLOSED", head_ref="agent/issue-120-x")],
        issues=[_issue(120, [])],
    )
    state = empty_state()
    # No state["prs"]["999"] entry at all.

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert drift == []


# ---------------------------------------------------------------------------
# Issue #1398: a CLOSED-unmerged PR left linked to an issue after an
# un-escalate + re-dispatch must NOT cause the closed-unmerged convergence
# rules to strip the NEW session's labels / status. The rules used to key
# only on "issue OPEN + active status/label" and "PR CLOSED unmerged" and
# never asked whether the issue's *current* active session postdates the PR
# close -- so every reconcile pass detached a still-live worker from the
# orchestrator's tracking and re-selected the issue as a fresh dispatch
# candidate, burning a concurrency-governor slot each pass.
#
# Regression contract from the issue: "close PR A at t0, dispatch issue at
# t1>t0, reconcile must produce zero drift for the issue."
# ---------------------------------------------------------------------------


def test_closed_unmerged_pr_rules_skip_issue_when_dispatch_postdates_close() -> None:
    """The exact regression from issue #1398: PR A closed at t0, issue
    re-dispatched at t1>t0 (status 'dispatched', dispatched_at=t1, live
    agent:in-progress label). Both issue-side closed-unmerged rules must
    produce ZERO drift for the issue -- the closed PR is stale, the active
    session is the redispatch the un-gate sweep intended.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    dispatched_at = (now - timedelta(minutes=22)).isoformat().replace("+00:00", "Z")
    assert dispatched_at > closed_at  # sanity: session postdates PR close

    gh = FakeGitHub(
        prs=[_pr(1214, "CLOSED", head_ref="agent/issue-1068-x", closed_at=closed_at)],
        issues=[_issue(1068, [config.labels.in_progress])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "dispatched",
        "dispatched_at": dispatched_at,
        "worker_pid": 29512,
    }

    drift = detect_drift(gh, state, config)
    assert [item for item in drift if item.kind == "closed_unmerged_pr_active_labels"] == []
    assert [
        item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"
    ] == []

    # apply_fixes on the (empty) issue-side drift must leave the live
    # session's state entirely untouched.
    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["status"] == "dispatched"
    assert new_state["issues"]["1068"]["dispatched_at"] == dispatched_at
    assert (1068, config.labels.in_progress) not in gh.labels_removed


def test_closed_unmerged_pr_state_converged_still_fires_for_stale_pr_entry() -> None:
    """The PR-side ``closed_unmerged_pr_state_converged`` rule is NOT gated
    by the #1398 guard -- the PR genuinely IS closed, so its own state
    entry must still converge to 'closed' regardless of whether the linked
    issue has a newer session. The guard is scoped to the two issue-side
    rules only.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    dispatched_at = (now - timedelta(minutes=22)).isoformat().replace("+00:00", "Z")

    gh = FakeGitHub(
        prs=[_pr(1214, "CLOSED", head_ref="agent/issue-1068-x", closed_at=closed_at)],
        issues=[_issue(1068, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["1214"] = {"number": 1214, "issue_number": 1068, "status": "reviewing"}
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "dispatched",
        "dispatched_at": dispatched_at,
    }

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_state_converged"
    ]
    assert len(drift) == 1
    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["prs"]["1214"]["status"] == "closed"


def test_closed_unmerged_pr_rules_skip_issue_when_pr_number_points_elsewhere() -> None:
    """The second #1398 signal: the issue's recorded ``pr_number`` points to
    a *different* (newer) PR than the closed one, AND that newer PR actually
    appears among the issue's linked/fetched PRs. The issue has moved on to a
    real, newer PR; the stale closed PR must not strip the new session's
    labels/status even when dispatched_at is absent (e.g. the worker already
    opened the new PR and dispatched_at was cleared).

    The corroboration requirement (the referenced pr_number must appear in the
    fetched snapshot) is what separates this from a stale reference -- see
    ``test_closed_unmerged_pr_rules_fire_when_pr_number_is_stale_dangling``.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")

    gh = FakeGitHub(
        prs=[
            _pr(1348, "CLOSED", head_ref="agent/issue-1342-x", closed_at=closed_at),
            # The newer PR the issue moved on to -- present in the fetched
            # snapshot, so the pr_number mismatch is corroborated.
            _pr(1399, "OPEN", head_ref="agent/issue-1342-y"),
        ],
        issues=[_issue(1342, [config.labels.in_progress])],
    )
    state = empty_state()
    # The issue's current PR is a newer one (#1399), not the stale #1348.
    state["issues"]["1342"] = {
        "number": 1342,
        "status": "reviewing",
        "pr_number": 1399,
    }

    drift = detect_drift(gh, state, config)
    assert [item for item in drift if item.kind == "closed_unmerged_pr_active_labels"] == []
    assert [
        item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"
    ] == []


def test_closed_unmerged_pr_rules_fire_when_pr_number_is_stale_dangling() -> None:
    """Issue #1398 rework regression: when ``state.json``'s ``pr_number``
    names a PR that is ABSENT from the fetched GitHub PR snapshot entirely
    (a stale/dangling reference from a botched salvage, a hand edit, or a
    race -- not a legitimate newer PR), and there is no ``dispatched_at``
    corroboration, the pr_number mismatch must NOT be treated as proof of a
    newer session. Both issue-side closed-unmerged convergence rules must
    still fire so the issue reaches a terminal state instead of being
    permanently skipped -- the #558/#1066 permanent-stuck failure class.

    This is the hardening the round-1 review required: a bare mismatch with
    no corroborating real PR and no dispatched_at signal lets convergence
    proceed rather than silently suppressing it forever.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")

    gh = FakeGitHub(
        # Only the stale closed PR links the issue; the referenced #1399 does
        # NOT exist in the fetched snapshot at all.
        prs=[_pr(1348, "CLOSED", head_ref="agent/issue-1342-x", closed_at=closed_at)],
        issues=[_issue(1342, [config.labels.in_progress])],
    )
    state = empty_state()
    # Stale dangling reference: pr_number points to a PR that does not exist
    # on GitHub, and no dispatched_at to corroborate a newer session.
    state["issues"]["1342"] = {
        "number": 1342,
        "status": "reviewing",
        "pr_number": 1399,
    }

    drift = detect_drift(gh, state, config)
    assert len([item for item in drift if item.kind == "closed_unmerged_pr_active_labels"]) == 1
    assert (
        len([item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"])
        == 1
    )


def test_closed_unmerged_pr_rules_fire_when_session_predates_close() -> None:
    """Baseline guard for #1398: when the issue's active session does NOT
    postdate the PR close (dispatched_at is older than closedAt, the closed
    PR genuinely is the issue's current dead PR), both issue-side rules
    must fire exactly as before. The guard never weakens the existing
    convergence for the case #558/#1066 exist to handle.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    dispatched_at = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    closed_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    assert dispatched_at < closed_at  # sanity: session predates PR close

    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x", closed_at=closed_at)],
        issues=[_issue(495, [config.labels.in_progress])],
    )
    state = empty_state()
    state["issues"]["495"] = {
        "number": 495,
        "status": "dispatched",
        "dispatched_at": dispatched_at,
    }

    drift = detect_drift(gh, state, config)
    assert len([item for item in drift if item.kind == "closed_unmerged_pr_active_labels"]) == 1
    assert (
        len([item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"])
        == 1
    )


def test_closed_unmerged_pr_rules_fire_when_no_session_timestamp() -> None:
    """Baseline guard for #1398: when the issue has an active status but no
    dispatched_at and no pr_number (the pre-#558 shape, e.g.
    'rework_requested' with dispatched_at cleared), there is no positive
    evidence of a newer session, so both issue-side rules must fire as
    before. The guard only skips on positive evidence; it never over-skips.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    gh = FakeGitHub(
        prs=[_pr(500, "CLOSED", head_ref="agent/issue-495-x", closed_at=closed_at)],
        issues=[_issue(495, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["495"] = {"number": 495, "status": "rework_requested"}

    drift = detect_drift(gh, state, config)
    assert len([item for item in drift if item.kind == "closed_unmerged_pr_active_labels"]) == 1
    assert (
        len([item for item in drift if item.kind == "closed_unmerged_pr_issue_state_converged"])
        == 1
    )


# ---------------------------------------------------------------------------
# Issue #1498: the #1398 guard's uncovered variant. ``state["issues"][N]
# ["pr_number"]`` is only written by the orchestrator's own PR-opening paths
# (dispatch salvage / orphaned-branch recovery). A PR that links itself to
# an issue via its own closing keyword -- opened outside those paths --
# never updates the cache, so it can keep pointing at a dead closed-unmerged
# PR while a live open PR also links the issue. That staleness defeats
# ``_closed_pr_superseded_by_newer_session``'s signal 1 (cached ==
# closed PR being evaluated, so "issue moved on" cannot be proven) and the
# two rules then flap ``agent:pr-open`` off/on every other pass (issue
# #1068: 23 add/remove transitions in 16h). The ``stale_issue_pr_number``
# drift kind repoints the cache at the issue's current open PR -- the same
# ``min(open PR number)`` pick ``issue_active_label_with_open_pr`` uses --
# so the supersession guard sees the live PR on the next pass and both
# rules converge.
# ---------------------------------------------------------------------------


def test_stale_issue_pr_number_repoints_cache_to_open_linked_pr() -> None:
    """Detection: cached ``pr_number`` names a CLOSED-unmerged PR while a
    different OPEN PR links the issue via its own closing keyword. The drift
    item must carry the open PR's number and ``apply_fixes`` must rewrite
    the cached pointer, preserving the entry's other fields. A second pass
    over the corrected state must emit nothing (self-heals)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            # Closing-keyword bodies -- not branch names -- are what binds
            # both PRs to issue #1068, mirroring the real incident's
            # closingIssuesReferences linkage.
            _pr(1214, "CLOSED", head_ref="fix/dead-attempt", body="Closes #1068\n\nstale"),
            _pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068\n\nlive"),
        ],
        # Labels already converged (agent:pr-open present) so the sibling
        # label repair does not fire and the pointer repair is isolated.
        issues=[_issue(1068, [config.labels.ready, config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1214,
        "title": "cached title",
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1
    assert drift[0].issue_number == 1068
    assert drift[0].pr_number == 1405

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405
    # Other fields are preserved by the overlay write.
    assert new_state["issues"]["1068"]["status"] == "open_passive"
    assert new_state["issues"]["1068"]["title"] == "cached title"

    second = [
        item
        for item in detect_drift(gh, new_state, config)
        if item.kind == "stale_issue_pr_number"
    ]
    assert second == []


def test_stale_issue_pr_number_picks_lowest_open_linked_pr() -> None:
    """Tie-break: when several open PRs link the same issue, the cache
    repoints to the LOWEST PR number -- the same
    ``min(int(pr["number"]) for pr in open_prs)`` pick
    ``issue_active_label_with_open_pr`` reports."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(1501, "OPEN", head_ref="fix/b", body="Closes #1068"),
            _pr(1405, "OPEN", head_ref="fix/a", body="Closes #1068"),
        ],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1501,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1
    assert drift[0].pr_number == 1405

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405


def test_stale_issue_pr_number_populates_missing_key() -> None:
    """An issue state entry with no ``pr_number`` key at all (e.g. the
    externally-linked PR was never recorded) still gets the pointer
    written -- ``None`` counts as differing from the open PR's number."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {"number": 1068, "status": "open_passive"}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405


def test_stale_issue_pr_number_does_not_fire_when_cache_matches_open_pr() -> None:
    """Idempotency guard: cached ``pr_number`` already naming the open
    linked PR is not drift -- the kind must not re-fire every pass."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1405,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_does_not_fire_without_open_linked_pr() -> None:
    """No open linked PR means there is nothing truthful to repoint to --
    a closed-unmerged-only linkage is the closed-unmerged rules' shape,
    not this one's."""
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    gh = FakeGitHub(
        prs=[
            _pr(
                1214,
                "CLOSED",
                head_ref="fix/dead-attempt",
                body="Closes #1068",
                closed_at=closed_at,
            )
        ],
        issues=[_issue(1068, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["1068"] = {"number": 1068, "pr_number": 1214}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_does_not_fire_for_closed_issue() -> None:
    """A CLOSED GitHub issue's ``pr_number`` is a historical record of
    which PR resolved it (``_merged_issue_fields`` preserves it verbatim),
    not a live pointer to repair -- an open PR linking it later must not
    overwrite that record."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "closed",
        "pr_number": 1214,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_self_heal_stops_pr_open_label_flap() -> None:
    """The #1068 regression shape, end to end: cached ``pr_number`` names
    the closed-unmerged PR #1214 while the open PR #1405 also links issue
    #1068, and the issue carries ``agent:pr-open``.

    Without the repoint, ``closed_unmerged_pr_active_labels`` (strips
    ``agent:pr-open``) and ``issue_active_label_with_open_pr`` (re-adds it)
    alternate forever -- 23 transitions over 16h in production. Once the
    cache heals to #1405, the #1398 supersession guard sees the live PR on
    the next pass and the label state reaches a fixed point: two
    consecutive reconcile passes must leave the label set untouched.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(days=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(1068, [config.labels.ready, config.labels.pr_open])
    gh = FakeGitHub(
        prs=[
            _pr(
                1214,
                "CLOSED",
                head_ref="fix/dead-attempt",
                body="Closes #1068\n\nsuperseded attempt",
                closed_at=closed_at,
            ),
            _pr(
                1405,
                "OPEN",
                head_ref="fix/janitor-cross-pr-revert",
                body="Closes #1068\n\nfix(janitor): detect_cross_pr_revert fails closed",
            ),
        ],
        issues=[issue],
    )
    state = empty_state()
    # The live #1068 entry shape: pr_number still names the dead PR, and
    # there is no ``dispatched_at`` for the guard's signal 2 (only
    # dispatch_pending_at / terminal_since from the long-terminated
    # pending session).
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1214,
        "dispatch_pending_at": (now - timedelta(days=6)).isoformat().replace("+00:00", "Z"),
        "terminal_since": (now - timedelta(days=6)).isoformat().replace("+00:00", "Z"),
    }

    flap_kinds = {
        "closed_unmerged_pr_active_labels",
        "closed_unmerged_pr_issue_state_converged",
        "issue_active_label_with_open_pr",
        "stale_issue_pr_number",
    }
    label_history: list[frozenset[str]] = []
    kinds_by_pass: list[set[str]] = []
    for _ in range(4):
        drift = detect_drift(gh, state, config)
        state = apply_fixes(gh, state, drift, config)
        # Mirror the applied label writes onto the fake's served payload the
        # way the real API's next ``issue list`` reflects prior mutations --
        # FakeGitHub only records the calls.
        names = {entry["name"] for entry in issue["labels"]}
        names.difference_update(label for n, label in gh.labels_removed if n == 1068)
        names.update(label for n, label in gh.labels_added if n == 1068)
        issue["labels"] = [{"name": name} for name in sorted(names)]
        gh.labels_added.clear()
        gh.labels_removed.clear()
        label_history.append(frozenset(names))
        kinds_by_pass.append({item.kind for item in drift if item.issue_number == 1068})

    # The cache repointed to the live open PR on the first pass and the
    # drift kind fired exactly once -- never again.
    assert state["issues"]["1068"]["pr_number"] == 1405
    assert "stale_issue_pr_number" in kinds_by_pass[0]
    assert all("stale_issue_pr_number" not in kinds for kinds in kinds_by_pass[1:])

    # Convergence: the label set is a fixed point across the last two
    # consecutive passes (the remove/add oscillation is over).
    assert (
        label_history[-1]
        == label_history[-2]
        == frozenset({config.labels.ready, config.labels.pr_open})
    )

    # The final pass emitted no flap-class drift for the issue at all:
    # the #1398 supersession guard now sees the live PR and both
    # closed-unmerged rules skip it.
    assert kinds_by_pass[-1] & flap_kinds == set()
