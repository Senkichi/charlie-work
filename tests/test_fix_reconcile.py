"""Tests for TASK W3: generalized pr_open self-heal + issue/PR status normalization.

Covers the confirmed drift class where an issue carries NO active agent label
at all (bare ``automated-ready``) while an open PR already links to it -- the
original ``issue_active_label_with_open_pr`` gate only fired when a *wrong*
active label was present (``stale_active`` truthy), so an issue with zero
active labels was invisible to the self-heal even though it needed the same
repair. Also covers the new issue/PR status-normalization sweep that recovers
records whose ``status`` is missing or holds a value no code path in the
orchestrator ever assigns (e.g. ``"ready"``, which is only ever a label
default, never a status).

Reuses the lightweight ``FakeGitHub``/``_pr``/``_issue`` fixtures already
defined in ``test_reconcile.py`` (pytest's rootless import mode makes this a
plain top-level import, same pattern ``test_reconcile.py`` itself uses for
``_sessions_db_fixtures``).
"""

from __future__ import annotations

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import apply_fixes, detect_drift
from charlie_work.state import empty_state

from test_reconcile import FakeGitHub, _issue, _pr


def test_orphan_no_active_label_with_open_pr_is_drift() -> None:
    """A bare 'automated-ready' label + an open tracked PR is the orphan shape
    the original `stale_active` gate missed (empty set is falsy).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "ready"}

    full_drift = detect_drift(gh, state, config)
    issue_40_items = [item for item in full_drift if item.issue_number == 40]

    # Exactly one drift item for issue 40 this pass -- the label/status repair
    # must not also double-fire the separate status-normalization sweep for
    # the same issue in the same pass.
    assert len(issue_40_items) == 1
    item = issue_40_items[0]
    assert item.kind == "issue_active_label_with_open_pr"
    assert item.pr_number == 50
    assert item.remove_labels == ()
    assert item.add_labels == (config.labels.pr_open,)
    assert item.new_status == "reviewing"


def test_orphan_no_active_label_with_open_pr_gets_label_and_status_repaired() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "ready"}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_active_label_with_open_pr"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)

    assert (40, config.labels.pr_open) in gh.labels_added
    assert gh.labels_removed == []  # nothing to remove: no stale active label was present
    # "reviewing" -- not "ready" and not "approved" -- is the passive status
    # the normal dispatch->pr-open flow writes once a PR is open and no
    # verdict has landed.
    assert new_state["issues"]["40"]["status"] == "reviewing"


def test_corrupt_stub_status_normalized_on_closed_issue() -> None:
    """Issue #361: a record that is just {'merge_alert': 'OK'} (no status, no
    number) on an issue closed on GitHub must be normalized to 'closed'.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(361, [], state="CLOSED")])
    state = empty_state()
    state["issues"]["361"] = {"merge_alert": "OK"}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "issue_status_normalized"
    ]
    assert len(drift) == 1
    assert drift[0].issue_number == 361
    assert drift[0].new_status == "closed"

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["issues"]["361"]["status"] == "closed"
    assert new_state["issues"]["361"]["merge_alert"] == "OK"  # other fields preserved


def test_closed_issue_with_invalid_status_gets_closed() -> None:
    """A record with an explicit but never-written status value (e.g. 'ready',
    which is only ever a label default, never a status) on a GH-closed issue
    is recomputed to 'closed', with unrelated cached fields preserved.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(374, [], state="CLOSED")])
    state = empty_state()
    state["issues"]["374"] = {
        "number": 374,
        "status": "ready",
        "title": "some cached title",
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "issue_status_normalized"
    ]
    assert len(drift) == 1
    assert drift[0].new_status == "closed"

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["issues"]["374"]["status"] == "closed"
    assert new_state["issues"]["374"]["title"] == "some cached title"


def test_pr_status_normalized_when_tracked_pr_missing_status() -> None:
    """A PR record the orchestrator is already tracking (has a state entry)
    but that never got a status written is normalized to the same passive
    placeholder issue records get.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(60, "OPEN", head_ref="agent/issue-45-x")],
        issues=[_issue(45, [config.labels.pr_open])],
    )
    state = empty_state()
    state["prs"]["60"] = {"issue_number": 45}  # no "status" key at all

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "pr_status_normalized"
    ]
    assert len(drift) == 1
    assert drift[0].pr_number == 60
    assert drift[0].new_status == "reviewing"

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["prs"]["60"]["status"] == "reviewing"
    assert new_state["prs"]["60"]["issue_number"] == 45


def test_valid_status_and_healthy_labels_produce_no_normalization_drift() -> None:
    """Baseline: an issue/PR pair that is already fully correct must never be
    flagged -- guards against false positives in the new sweeps.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(70, "OPEN", head_ref="agent/issue-50-x")],
        issues=[_issue(50, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    state["issues"]["50"] = {"number": 50, "status": "reviewing"}
    state["prs"]["70"] = {"issue_number": 50, "status": "reviewing"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "issue_active_label_with_open_pr"] == []
    assert [item for item in drift if item.kind == "issue_status_normalized"] == []
    assert [item for item in drift if item.kind == "pr_status_normalized"] == []


def test_second_pass_after_orphan_repair_is_idempotent() -> None:
    """Running detect_drift again after apply_fixes with the post-fix labels
    and state must produce zero drift/events for the same issue -- the
    self-heal + status normalization must never loop forever.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "ready"}

    first_drift = detect_drift(gh, state, config)
    matches = [item for item in first_drift if item.kind == "issue_active_label_with_open_pr"]
    assert len(matches) == 1

    new_state = apply_fixes(gh, state, matches, config)

    # FakeGitHub is a dumb recorder -- it does not mutate its own `_issues`
    # label snapshot when add_issue_label/remove_issue_label are called.
    # Rebuild a second fake reflecting the labels exactly as they would look
    # after a real `gh` accepted the same add/remove calls.
    gh2 = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready, config.labels.pr_open])],
    )
    second_drift = detect_drift(gh2, new_state, config)

    assert [item for item in second_drift if item.issue_number == 40] == []


def test_escalated_status_zero_labels_converges_instead_of_self_heal() -> None:
    """An escalated issue whose human_needed label write failed (crash or
    label-API error between the status write and the label transition) has
    NO workflow labels at all -- exactly the zero-label shape the widened
    open-PR self-heal would otherwise silently re-arm as pr_open/reviewing,
    undoing a terminal-until-human escalation. Reconcile must treat the
    tracked status as ground truth and re-apply the label instead.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "escalated"}

    full_drift = detect_drift(gh, state, config)
    issue_40_items = [item for item in full_drift if item.issue_number == 40]

    assert len(issue_40_items) == 1
    item = issue_40_items[0]
    assert item.kind == "escalated_labels_converged"
    assert item.add_labels == (config.labels.human_needed,)
    assert item.remove_labels == ()
    # Self-heal must NOT fire for this issue.
    assert not any(
        i.kind == "issue_active_label_with_open_pr" for i in full_drift if i.issue_number == 40
    )

    new_state = apply_fixes(gh, state, issue_40_items, config)
    assert (40, config.labels.human_needed) in gh.labels_added
    # Status stays escalated -- convergence repairs labels, never status.
    assert new_state["issues"]["40"]["status"] == "escalated"


def test_escalated_status_with_stale_active_label_strips_it() -> None:
    """Escalated in state, human_needed landed, but a stale active label
    survived a partial transition: converge to the escalated label set
    (human_needed only) rather than letting any repair path act on the
    active label.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[
            _issue(
                40,
                [config.labels.ready, config.labels.human_needed, config.labels.needs_rework],
            )
        ],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "escalated"}

    full_drift = detect_drift(gh, state, config)
    issue_40_items = [item for item in full_drift if item.issue_number == 40]

    assert len(issue_40_items) == 1
    item = issue_40_items[0]
    assert item.kind == "escalated_labels_converged"
    assert item.add_labels == ()
    assert item.remove_labels == (config.labels.needs_rework,)


def test_escalated_status_with_correct_labels_is_no_drift() -> None:
    """An escalated issue whose labels already reflect it (human_needed, no
    actives) must produce zero drift -- the convergence check is idempotent.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready, config.labels.human_needed])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "escalated"}

    full_drift = detect_drift(gh, state, config)
    assert [item for item in full_drift if item.issue_number == 40] == []


def test_no_open_pr_repair_skips_terminal_labeled_issue() -> None:
    """The no-open-PR relabel repair must never make a terminal-labeled
    (human_needed/done) issue dispatchable again by re-adding `ready` --
    only the terminal+active contradiction repair may touch it.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[
            _issue(
                41,
                [config.labels.ready, config.labels.human_needed, config.labels.in_progress],
            )
        ],
    )
    state = empty_state()

    full_drift = detect_drift(gh, state, config)
    kinds = {item.kind for item in full_drift if item.issue_number == 41}
    assert "issue_active_label_no_open_pr" not in kinds
    assert kinds == {"done_label_with_active_labels"}


def test_valid_issue_statuses_covers_every_assigned_status_literal() -> None:
    """Tripwire: reconcile's normalization sweep strips any issue status
    outside VALID_ISSUE_STATUSES, so a status some workflow code path
    assigns but the frozenset omits is silently erased by the next repair
    pass -- which is how ``manifest_written``/``dispatch_failed`` records
    would have lost their in-flight/dispatch-cap bookkeeping. Scan the
    package source for every status literal written anywhere and force each
    new one to be explicitly classified as an issue status or a
    PR/event-only status.
    """
    import re
    from pathlib import Path

    import charlie_work
    from charlie_work.state import VALID_ISSUE_STATUSES

    pattern = re.compile(
        r'["\']status["\']\s*(?::|\]\s*=)\s*["\']([a-z_]+)["\']'  # dict key / subscript
        r"|\bstatus\s*=\s*[\"']([a-z_]+)[\"']"  # bare local later stored under "status"
    )
    found: set[str] = set()
    for py in Path(charlie_work.__file__).parent.glob("*.py"):
        for match in pattern.finditer(py.read_text(encoding="utf-8")):
            found.add(match.group(1) or match.group(2))

    # Statuses that only ever appear on PR records or event payloads, never
    # on issues[n]["status"]. A new literal landing in neither set fails
    # this test and forces explicit classification by the author.
    pr_or_event_only = {
        "merged",
        "janitor_blocked",
        "mergequeue",
        "pending",
        "unclaimed",
    }

    unclassified = found - VALID_ISSUE_STATUSES - pr_or_event_only
    assert unclassified == set(), (
        f"status literals {sorted(unclassified)} are assigned in src/charlie_work "
        "but are classified neither as issue statuses (state.VALID_ISSUE_STATUSES) "
        "nor as PR/event-only statuses (this test's allowlist); classify them or "
        "reconcile's normalization sweep will strip them from live records"
    )
    # And the scan itself must keep seeing the two statuses whose omission
    # motivated it -- if the regex rots, this fails loudly instead of the
    # tripwire silently going blind.
    assert {"manifest_written", "dispatch_failed"} <= found


def test_closed_issue_with_dispatch_failed_status_is_finalized() -> None:
    """Adding dispatch_failed/manifest_written to VALID_ISSUE_STATUSES made
    the normalization sweep skip them -- so ACTIVE_STATE_STATUSES must
    include them, or a closed-on-GitHub issue stuck in either status is
    invisible to BOTH finalization sweeps and never reaches 'closed'.
    """
    config = OrchestratorConfig()
    for stuck_status in ("dispatch_failed", "manifest_written"):
        gh = FakeGitHub(prs=[], issues=[_issue(90, [], state="CLOSED")])
        state = empty_state()
        state["issues"]["90"] = {"number": 90, "status": stuck_status}

        drift = [
            item
            for item in detect_drift(gh, state, config)
            if item.kind == "state_active_status_issue_closed"
        ]
        assert len(drift) == 1, f"no finalization drift for status {stuck_status!r}"
        new_state = apply_fixes(gh, state, drift, config)
        assert new_state["issues"]["90"]["status"] == "closed"


def test_active_state_statuses_is_valid_minus_deliberate_exclusions() -> None:
    """Structural invariant: the normalization sweep skips everything in
    VALID_ISSUE_STATUSES, so any valid status missing from
    ACTIVE_STATE_STATUSES (beyond the deliberate exclusions) is invisible to
    both sweeps -- a new status added to one set but not the other recreates
    the dead zone this pins down.
    """
    from charlie_work.reconcile import ACTIVE_STATE_STATUSES
    from charlie_work.state import VALID_ISSUE_STATUSES

    deliberate_exclusions = {"closed", "approved", "blocked"}
    assert ACTIVE_STATE_STATUSES == VALID_ISSUE_STATUSES - deliberate_exclusions
