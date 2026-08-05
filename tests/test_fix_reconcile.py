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

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import apply_fixes, detect_drift
from charlie_work.state import ORCHESTRATOR_OWNED_ISSUE_STATUSES, empty_state

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


def test_reopened_issue_with_closed_state_status_and_open_pr_is_rederived() -> None:
    """Issue #789 / AC-1 / AC-5: a GitHub reopen must un-strand a state entry
    that was correctly normalized to "closed" at the time but has since gone
    stale. Reproduces the confirmed live instance (issue #649 / PR #693): the
    issue carries its normal `pr_open` label (so the separate
    issue_active_label_with_open_pr self-heal has nothing to fix) but
    state.json's status is still the pre-reopen "closed" -- before this fix
    that was self-validating and the entry was stranded forever.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(693, "OPEN", head_ref="agent/issue-649-x")],
        issues=[_issue(649, [config.labels.pr_open], state="OPEN")],
    )
    state = empty_state()
    state["issues"]["649"] = {"number": 649, "status": "closed"}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "issue_status_normalized"
    ]
    assert len(drift) == 1
    assert drift[0].issue_number == 649
    assert drift[0].new_status == "reviewing"  # PASSIVE_OPEN_STATUS

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["issues"]["649"]["status"] == "reviewing"


def test_both_closed_issue_is_unchanged_and_costs_no_extra_github_call() -> None:
    """Issue #789 / AC-2 / AC-5: the common case -- state status "closed" and
    the GitHub issue is genuinely still closed -- must be a no-op: no drift
    item (so no spurious event/log noise every pass) and, since re-examining
    "closed" entries now runs a GitHub-state lookup that used to be skipped
    entirely, no *additional* `gh.run` call beyond the two upfront list
    queries `detect_drift` always issues. `issues_by_number` is populated from
    that same `--state all` issue-list snapshot, so the lookup is a dict
    `.get()`, not a network call.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(900, [], state="CLOSED")])
    state = empty_state()
    state["issues"]["900"] = {"number": 900, "status": "closed"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.issue_number == 900] == []
    # Exactly the two unconditional snapshot queries (`pr list`, `issue list`)
    # -- no per-issue call was added for the closed entry.
    assert len(gh.run_calls) == 2


def test_closed_status_with_issue_absent_from_snapshot_emits_no_drift() -> None:
    """Issue #789 follow-up: a state entry with status "closed" whose issue
    number is absent from the GitHub issue-list snapshot entirely must NOT be
    treated as evidence the issue is gone. Before the explicit `issue is
    None` guard in the normalization loop, this fell through to
    `target_status = None` and `apply_fixes` would strip the "closed" status
    -- harmless today only by the accident that the repo's issue count sits
    under `_LIST_LIMIT`, so `issues_by_number` never actually drops a real
    closed issue. Once the repo passes that limit, every closed entry whose
    issue fell off the `--state all` page would get its status silently
    wiped on the very next reconcile pass, with no signal in the drift log
    to explain it. This test pins the guard directly -- it must fail against
    the unguarded version of the fix (verified via mutation check).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])  # issue 950 absent from the snapshot entirely
    state = empty_state()
    state["issues"]["950"] = {"number": 950, "status": "closed"}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "issue_status_normalized"
    ]
    assert drift == []


@pytest.mark.parametrize("owned_status", sorted(ORCHESTRATOR_OWNED_ISSUE_STATUSES))
def test_orchestrator_owned_status_keeps_skip_behavior_on_closed_issue(
    owned_status: str,
) -> None:
    """Issue #789 / AC-3: every orchestrator-owned status (VALID_ISSUE_STATUSES
    minus the externally-derived "closed") must keep its exact pre-#789 skip
    behavior in the status-normalization sweep -- reconcile must not start
    overwriting escalated/dispatched/etc. Uses a GitHub issue that is CLOSED
    with no tracked PR: the one shape that *would* trigger normalization for
    a status outside the skip set, which is what makes this discriminating.
    A fix that accidentally narrowed the skip set (e.g. a typo'd literal at
    the call site instead of deriving from the frozenset) would surface here
    as a spurious issue_status_normalized drift item.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(950, [], state="CLOSED")])
    state = empty_state()
    state["issues"]["950"] = {"number": 950, "status": owned_status}

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_status_normalized" and item.issue_number == 950
    ]
    assert drift == [], f"status {owned_status!r} was incorrectly re-normalized"


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

    # Issue #947: any OPEN issue carrying `human_needed` also gets a
    # `terminal_state_stale` alert (age unknown -> "never observed" here,
    # since no state_path/timestamp is supplied) alongside whatever
    # label-convergence drift this test is actually pinning. Assert the
    # convergence item specifically rather than the raw count.
    convergence_items = [
        item for item in issue_40_items if item.kind == "escalated_labels_converged"
    ]
    assert len(convergence_items) == 1
    item = convergence_items[0]
    assert item.add_labels == ()
    assert item.remove_labels == (config.labels.needs_rework,)
    assert {item.kind for item in issue_40_items} == {
        "escalated_labels_converged",
        "terminal_state_stale",
    }


def test_escalated_status_with_correct_labels_is_no_drift() -> None:
    """An escalated issue whose labels already reflect it (human_needed, no
    actives) must produce zero *label-convergence* drift -- the convergence
    check is idempotent. Issue #947's orthogonal `terminal_state_stale` alert
    still fires (age never observed, no timestamp seeded here) -- that is
    the intended new behavior, not label-convergence drift re-appearing.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(50, "OPEN", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [config.labels.ready, config.labels.human_needed])],
    )
    state = empty_state()
    state["issues"]["40"] = {"number": 40, "status": "escalated"}

    full_drift = detect_drift(gh, state, config)
    issue_40_kinds = {item.kind for item in full_drift if item.issue_number == 40}
    assert issue_40_kinds == {"terminal_state_stale"}


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
    # Issue #947: terminal_state_stale also fires alongside the
    # terminal+active contradiction repair -- both are legitimate, orthogonal
    # findings for the same issue.
    assert kinds == {"done_label_with_active_labels", "terminal_state_stale"}


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


@pytest.mark.parametrize("stuck_status", ["reviewing", "escalated", "dispatched"])
def test_closed_unmerged_pr_issue_other_active_statuses_converge_to_dormant(
    stuck_status: str,
) -> None:
    """Every ACTIVE_STATE_STATUS on an OPEN issue whose PR closed-unmerged
    must converge to dormant -- not just 'rework_requested'.
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
