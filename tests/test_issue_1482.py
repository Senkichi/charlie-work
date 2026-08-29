"""Regression tests for issue #1482: auto-deescalation sweep clears issue but
never resets linked PR's stale escalated status, permanently blocking packet
regen.

``_deescalate_mechanical_issue`` reset the issue's ``status`` to
``PASSIVE_OPEN_STATUS`` and mirror-cleared the linked PR's escalation-reason
fields via ``clear_escalation_on_issue_prs``, but never reset the PR record's
own ``status`` field.  ``_escalate_issue(..., pr_number=...)`` sets
``pr.status = "escalated"`` on every PR-side escalation, and
``clear_escalation_on_issue_prs`` is scoped to escalation-reason fields by
design, so ``pr.status == "escalated"`` survived the auto-clear unchanged.

That split state (issue ``open_passive``, PR ``escalated``, no GitHub label)
silently excluded the PR from packet regeneration forever: ``loop()``'s
per-pass regen check skips regenerating an escalated PR's packet, so the
stored ``headRefOid`` went stale the moment the PR's head next moved and
``review_queue()``'s stale-packet guard permanently excluded the PR from the
review queue.

These tests were extracted here from ``tests/test_deescalation.py`` and
``tests/test_fix_unescalate.py`` to avoid growing those already-over-cap
files past their file-size ratchet marks (issue #1442).  The
``_reset_linked_pr_status_to_passive_open`` helper itself lives in
``charlie_work.escalation`` and is re-exported through ``workflow.py``'s
facade import block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock

from _unescalate_fixtures import _app


def test_sweep_resets_linked_pr_status_to_passive_open(tmp_path: Path) -> None:
    """Issue #1482: ``_deescalate_mechanical_issue`` resets the issue's
    ``status`` to ``PASSIVE_OPEN_STATUS`` and mirror-clears the linked PR
    record's escalation-reason fields via ``clear_escalation_on_issue_prs``,
    but it must ALSO reset the PR record's own ``status`` field.

    ``_escalate_issue(..., pr_number=...)`` (the escalation write path) sets
    ``pr.status = "escalated"`` whenever a PR-side escalation is recorded
    (e.g. ``dead_dispatched_worker_reap``, ``dispatch_blocked_environment``,
    ``worker_death_loop``).  ``clear_escalation_on_issue_prs`` is scoped to
    escalation-reason fields (``escalation_reason`` / ``reason_class`` /
    ``escalation_reasons_seen``) by design and docstring -- it never touches
    ``status``.  Before this fix the PR was left at
    ``pr.status == "escalated"`` with no GitHub label reflecting it, which
    silently excluded it from packet regeneration forever: ``loop()``'s
    per-pass regen check deliberately skips regenerating an escalated PR's
    packet, so the stored ``headRefOid`` went stale the moment the PR's head
    next moved and ``review_queue()``'s stale-packet guard permanently
    excluded the PR from the review queue.

    The sweep must reset the linked PR's ``status`` to the same
    ``PASSIVE_OPEN_STATUS`` target the operator ``unescalate`` door uses for
    a live OPEN PR (``_apply_pr_reset``), so the two doors cannot diverge on
    the PR-side status reset.  It must NOT clear the full
    ``UNESCALATE_PR_RESET_FIELDS`` set -- that is the operator door's
    broader re-arm, and resetting those counters here would violate the
    unbounded paid-session loop guard (issue #783 hazard (b)); only the
    per-mechanism rework counter is reset (already covered by the
    ``_REWORK_BUDGET_RESET_BY_ESCALATION_REASON`` tests in
    ``test_deescalation.py``).
    """
    app = _app(tmp_path)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            # PR-side status was set to "escalated" by _escalate_issue's
            # pr_number= path (e.g. dead_dispatched_worker_reap).
            "status": "escalated",
            "escalation_reason": "dead_dispatched_worker_reap",
            # A counter the sweep must NOT touch (hazard (b) guard): only the
            # per-mechanism counter gating the CLEARED reason is reset, and
            # dead_dispatched_worker_reap has no rework-budget entry, so this
            # unrelated review-lane counter survives the clear.
            "request_changes_count": 2,
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "dead_dispatched_worker_reap",
            "reason_class": "mechanical",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    pr_456 = state["prs"]["456"]
    # The PR-side status is reset to the same target the operator door uses
    # for a live OPEN PR -- the core of issue #1482.
    assert pr_456["status"] == PASSIVE_OPEN_STATUS
    # The escalation-reason fields are still mirror-cleared (regression guard
    # for the existing clear_escalation_on_issue_prs behavior).
    assert "escalation_reason" not in pr_456
    assert "escalation_reasons_seen" not in pr_456
    # The unrelated review-lane counter survives -- the sweep does NOT apply
    # the operator door's full UNESCALATE_PR_RESET_FIELDS (hazard (b)).
    assert pr_456["request_changes_count"] == 2
    # The issue side is cleared as before.
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS


@pytest.mark.parametrize("terminal_status", ["merged", "closed"])
def test_sweep_does_not_revert_pr_that_became_terminal_in_pre_lock_window(
    tmp_path: Path, terminal_status: str
) -> None:
    """Review on #1482: the new ``pr.status = PASSIVE_OPEN_STATUS`` write is
    guarded against a PR that a concurrent writer advanced to a terminal
    status ("merged"/"closed") in state.json during the pre-lock
    GitHub-fetch window.

    ``_deescalate_mechanical_issue`` fetches ``pr_state_str`` from
    ``self.gh.pr_view`` *before* acquiring the state lock, and gates the
    clear on ``pr_state_str == "OPEN"`` using that stale pre-lock value.
    The in-lock fresh state load can then observe a PR a concurrent writer
    (reconcile, another loop lane, an operator ``unescalate``) already
    advanced to ``"merged"``/``"closed"``.  Without the guard the
    unconditional ``status = PASSIVE_OPEN_STATUS`` write would revert that
    terminal PR to ``open_passive``, creating a split state (PR
    merged/closed on GitHub, ``open_passive`` in state.json) that
    reconcile must self-heal -- the exact race the review flagged.

    The race window is simulated by wrapping ``app.gh.pr_view`` so that,
    when the sweep fetches the PR (after the pre-lock state load selected
    this PR on its ``"escalated"`` status, but before the in-lock fresh
    load), a concurrent writer advances the PR's state.json ``status`` to
    the terminal value.  ``pr_view`` itself still returns ``state == "OPEN"``
    so the ``pr_state_str == "OPEN"`` gate passes -- isolating the guard to
    the in-lock fresh-status check, not the pre-lock GitHub gate.

    Mirroring the operator door's ``_apply_pr_reset`` (which branches on
    live PR state and never writes ``PASSIVE_OPEN_STATUS`` over a
    merged/closed PR) and this function's own PR-selection skip
    (``status not in ("merged", "closed")``), the sweep must leave the
    terminal ``status`` untouched.
    """
    app = _app(tmp_path)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        # Pre-lock state: PR is "escalated" so the sweep's PR-selection
        # (``status not in ("merged", "closed")``) picks it up.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            "reason_class": "mechanical",
        }
        save_state(app.paths.state_file, state)

    original_pr_view = app.gh.pr_view

    def _racing_pr_view(number: int) -> dict:
        # Simulate a concurrent writer advancing the PR to a terminal
        # status in state.json during the pre-lock GitHub-fetch window --
        # after the pre-lock state load selected this PR on "escalated",
        # before the in-lock fresh load.  pr_view still reports OPEN so the
        # ``pr_state_str == "OPEN"`` gate passes and the guard is the only
        # thing standing between the write and the revert.
        with state_lock(app.paths.state_file):
            racing_state = load_state(app.paths.state_file)
            racing_pr = racing_state["prs"].get(str(number))
            if isinstance(racing_pr, dict):
                racing_pr["status"] = terminal_status
            save_state(app.paths.state_file, racing_state)
        return original_pr_view(number)

    app.gh.pr_view = _racing_pr_view

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    pr_456 = state["prs"]["456"]
    # The terminal status is NOT reverted to PASSIVE_OPEN_STATUS -- the
    # core of the review finding.
    assert pr_456["status"] == terminal_status
    assert pr_456["status"] != PASSIVE_OPEN_STATUS
    # The escalation-reason fields are still mirror-cleared (the guard is
    # scoped to ``status`` only; clear_escalation_on_issue_prs still runs).
    assert "escalation_reason" not in pr_456
    assert "escalation_reasons_seen" not in pr_456


def test_operator_and_automated_doors_agree_on_pr_status_reset_target(
    tmp_path: Path,
) -> None:
    """Issue #1482: the operator ``unescalate`` door and the automated
    ``_deescalate_mechanical_issue`` sweep must both reset a live OPEN PR's
    ``status`` to ``PASSIVE_OPEN_STATUS``.  Before this fix the automated
    door reset the issue's status but left ``pr.status == "escalated"``,
    producing a split state with no GitHub label that silently excluded the
    PR from packet regeneration forever.

    This is the cross-door agreement assertion the issue asks for: both doors
    are exercised on equivalent escalated-PR setups (issue + PR both
    ``escalated``, PR OPEN on GitHub) and must converge on the same
    PR-side ``status``.  A future change to one door's PR-status target that
    forgets the other fails here instead of reintroducing the split state.
    """
    app = _app(tmp_path)

    def _seed() -> None:
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["456"] = {
                "number": 456,
                "issue_number": 123,
                "status": "escalated",
                "escalation_reason": "session_failed_escalated",
            }
            state["issues"]["123"] = {
                "number": 123,
                "status": "escalated",
                "escalation_reason": "session_failed_escalated",
                "reason_class": "mechanical",
            }
            save_state(app.paths.state_file, state)

    # Operator door.
    _seed()
    op_result = app.unescalate(pr_number=456)
    assert op_result.ok and op_result.data["changed"] is True
    operator_pr_status = load_state(app.paths.state_file)["prs"]["456"]["status"]

    # Automated door -- re-seed the identical escalated state and run the
    # sweep instead of the operator command.
    _seed()
    app._maybe_deescalate_mechanical()
    automated_pr_status = load_state(app.paths.state_file)["prs"]["456"]["status"]

    assert operator_pr_status == PASSIVE_OPEN_STATUS
    assert automated_pr_status == PASSIVE_OPEN_STATUS
    assert operator_pr_status == automated_pr_status
