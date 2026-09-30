"""Decision-table tests for ``worker_fate`` (architecture-deepening candidate 1).

Every ``resolve_fate`` test builds a ``FateEvidence`` literal by hand and
asserts the returned ``WorkerFate`` variant -- no fakes, no ``tmp_path``, no
git, per the design doc's Shape A rationale (``wf-design.md`` §1). Tests are
grouped by which of the nine named flip rules (plan §"Candidate decisions")
they pin, plus the freshness step (rules 1/7) they all sit behind, plus the
``is_alive``.

Nothing here is wired to a consumer yet (that is Group B in the plan); these
tests exercise the module's own public interface only.
"""

from __future__ import annotations


from datetime import timedelta

import pytest

from charlie_work.worker_fate import (
    Blocked,
    Completed,
    Crashed,
    EvidenceSource,
    FailureEvidence,
    Live,
    PushedWithoutPr,
    StaleReason,
    Stranded,
    Throttled,
    resolve_fate,
)
from charlie_work.worker import WorkerHealth
from _worker_fate_fixtures import (
    AFTER,
    BEFORE,
    DISPATCHED,
    NOW,
    _branch,
    _evidence,
    _outcome,
    _terminal,
)


# --------------------------------------------------------------------------
# Row 1 / rule 2: a fresh "blocked" outcome beats everything, alive or dead.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alive", [True, False])
def test_row1_blocked_beats_push_flags_regardless_of_liveness(alive: bool) -> None:
    outcome = _outcome(outcome="blocked", push_succeeded=True, head_sha="abc")
    evidence = _evidence(
        pid_alive=alive,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="abc", open_pr_number=9),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.basis.rule.startswith("R2")


def test_row1_blocked_reason_surfaces_from_raw() -> None:
    outcome = _outcome(outcome="blocked", raw={"reason": "needs human input"})
    fate = resolve_fate(_evidence(worktree_outcome=outcome), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.reason == "needs human input"


def test_row1_blocked_reason_none_when_absent() -> None:
    fate = resolve_fate(_evidence(worktree_outcome=_outcome(outcome="blocked")), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.reason is None


# --------------------------------------------------------------------------
# Rows 2-3 / rules 4,5,8: a fresh, remote-confirmed push -> Completed or
# PushedWithoutPr by PR existence.
# --------------------------------------------------------------------------


def test_row2_confirmed_push_with_open_pr_is_completed() -> None:
    outcome = _outcome(push_succeeded=True, head_sha="abc")
    evidence = _evidence(
        pid_alive=False,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="abc", open_pr_number=42, pr_known=True),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Completed)
    assert fate.pr_number == 42
    assert fate.head_sha == "abc"


def test_row3_confirmed_push_no_open_pr_is_pushed_without_pr() -> None:
    outcome = _outcome(push_succeeded=True, head_sha="abc")
    evidence = _evidence(
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="abc", open_pr_number=None, pr_known=True),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, PushedWithoutPr)
    assert fate.head_sha == "abc"


def test_row2_pr_created_claim_is_ignored_still_pushed_without_pr() -> None:
    """Rule 8: ``pr_created`` is a claim only (workers have no gh token,
    #1771) -- PR existence is decided from ``branch.open_pr_number``, never
    from the outcome's own claim.
    """
    outcome = _outcome(push_succeeded=True, pr_created=True, head_sha="abc")
    evidence = _evidence(
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="abc", open_pr_number=None, pr_known=True),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, PushedWithoutPr)


def test_row2_push_confirmed_via_remote_ahead_when_head_unknown() -> None:
    """``head_sha == remote_head_sha`` OR head unknown and ``remote_ahead >
    0``: the second disjunct.
    """
    outcome = _outcome(push_succeeded=True, head_sha=None)
    evidence = _evidence(
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha=None, remote_ahead=2, open_pr_number=7),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Completed)
    assert fate.pr_number == 7


@pytest.mark.parametrize("alive", [True, False])
def test_pr_existence_unknown_falls_through_to_liveness(alive: bool) -> None:
    """Confirmed push but ``pr_known`` is False: neither Completed nor
    PushedWithoutPr can be proven, so rows 2-4 no-op. Not one of the nine
    named rules -- a documented fallback, pinned so a future change to this
    edge doesn't happen by accident.
    """
    outcome = _outcome(push_succeeded=True, head_sha="abc")
    evidence = _evidence(
        pid_alive=alive,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="abc", open_pr_number=None, pr_known=False),
    )
    fate = resolve_fate(evidence, now=NOW)
    if alive:
        assert isinstance(fate, Live)
    else:
        assert isinstance(fate, Crashed)


# --------------------------------------------------------------------------
# Row 4 / rules 3,9: declared push the remote doesn't show, with local
# unpushed commits -> Stranded, even while the PID is still alive.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alive", [True, False])
def test_row4_unconfirmed_push_with_unpushed_commits_is_stranded(alive: bool) -> None:
    """The outcome claims a push but reports no ``head_sha`` (so rule 1's
    head check can't invalidate it) and neither confirmation path -- a head
    match or ``remote_ahead > 0`` -- proves the push landed.
    """
    outcome = _outcome(push_succeeded=True, head_sha=None)
    evidence = _evidence(
        pid_alive=alive,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha=None, remote_ahead=0, unpushed=2),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.unpushed == 2
    assert fate.basis.rule.startswith("R3+R9")


def test_row4_known_head_mismatch_is_stale_not_unconfirmed_push() -> None:
    """A KNOWN ``head_sha`` that mismatches the live remote head is
    discarded as stale evidence under rule 1, not treated as row 4's
    "declared push, remote doesn't show it" case -- it never reaches row 4
    at all. This is the property the design doc's rows-2-4-vs-6 precedence
    note (§3) relies on: a salvage push moving the remote head is exactly
    what makes a stale outcome's head claim stop matching. While the PID is
    (still) alive this resolves to ``Live``, not ``Stranded`` -- row 4 only
    ever sees a push claim rule 1 didn't already discard.
    """
    outcome = _outcome(push_succeeded=True, head_sha="claimed-sha")
    evidence = _evidence(
        pid_alive=True,
        health=WorkerHealth.HEALTHY,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="different-sha", unpushed=2),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Live)
    assert any(s.reason == StaleReason.HEAD_MISMATCH for s in fate.basis.stale)


def test_row4_known_head_mismatch_dead_still_stranded_via_row6() -> None:
    """Same stale-outcome input as above, but dead: row 4 still never
    fires (the outcome is discarded), but row 6 (dead + unpushed > 0, no
    outcome required) reaches the same ``Stranded`` verdict independently.
    """
    outcome = _outcome(push_succeeded=True, head_sha="claimed-sha")
    evidence = _evidence(
        pid_alive=False,
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha="different-sha", unpushed=2),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.basis.outcome is None
    assert any(s.reason == StaleReason.HEAD_MISMATCH for s in fate.basis.stale)


def test_row4_no_remote_repo_is_still_stranded() -> None:
    outcome = _outcome(push_succeeded=True, head_sha="claimed-sha")
    evidence = _evidence(
        worktree_outcome=outcome,
        branch=_branch(remote_head_sha=None, remote_ahead=0, unpushed=1),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)


# --------------------------------------------------------------------------
# Row 5 / rule 5: live PID, no earlier row decided it -> Live(health).
# --------------------------------------------------------------------------


def test_row5_live_pid_with_no_decisive_evidence() -> None:
    evidence = _evidence(pid_alive=True, health=WorkerHealth.HEALTHY)
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Live)
    assert fate.health is WorkerHealth.HEALTHY
    assert fate.basis.pid_alive is True


def test_row5_live_pid_with_unconfirmed_and_unpushed_evidence_still_live() -> None:
    """Rows 2-4 require either a confirmed push or unpushed commits; with
    neither, a live PID reaches row 5 even if an outcome exists.
    """
    outcome = _outcome(push_succeeded=False)
    evidence = _evidence(pid_alive=True, worktree_outcome=outcome, health=WorkerHealth.STALLED)
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Live)
    assert fate.health is WorkerHealth.STALLED


# --------------------------------------------------------------------------
# Row 6 / rules 3,9: dead with local-only commits -> Stranded, carrying any
# failure classification (throttle rearm still applies post-salvage).
# --------------------------------------------------------------------------


def test_row6_dead_with_unpushed_commits_is_stranded() -> None:
    evidence = _evidence(pid_alive=False, branch=_branch(unpushed=3))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.unpushed == 3


def test_row6_remote_unknown_local_ahead_is_stranded_with_remote_unknown_rule() -> None:
    """N5: no remote read (``remote_ahead``/``remote_head_sha``/``unpushed`` all
    unknown) but local commits exist -> Stranded under a distinct rule."""
    evidence = _evidence(pid_alive=False, branch=_branch(local_ahead=2))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.unpushed == 2
    assert fate.basis.rule == "R3+R9-remote-unknown"


def test_row6_local_ahead_ignored_when_remote_known() -> None:
    """``local_ahead`` says nothing about the remote: once the remote is known
    (here ``remote_ahead == 0``) it must not strand anything."""
    evidence = _evidence(pid_alive=False, branch=_branch(local_ahead=2, remote_ahead=0))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Crashed)
    # A known remote head alone also rules the remote-unknown rule out.
    evidence = _evidence(pid_alive=False, branch=_branch(local_ahead=2, remote_head_sha="abc"))
    assert isinstance(resolve_fate(evidence, now=NOW), Crashed)
    # And a proven ``unpushed`` count wins under the ordinary row-6 rule.
    evidence = _evidence(pid_alive=False, branch=_branch(local_ahead=5, unpushed=1))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.unpushed == 1
    assert fate.basis.rule == "R3+R9"


def test_row6_stranded_carries_throttle_failure_ahead_of_row7() -> None:
    failure = FailureEvidence(
        kind="rate_limited", throttled_until=NOW + timedelta(minutes=10), fresh=True
    )
    evidence = _evidence(pid_alive=False, branch=_branch(unpushed=1), failure=failure)
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Stranded)
    assert fate.failure is failure


# --------------------------------------------------------------------------
# Row 7 / rule 6: dead, no fresh outcome, a provider-throttle classification.
# --------------------------------------------------------------------------


def test_row7_dead_no_outcome_throttled() -> None:
    failure = FailureEvidence(
        kind="rate_limited", throttled_until=NOW + timedelta(minutes=5), fresh=False
    )
    evidence = _evidence(pid_alive=False, failure=failure)
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Throttled)
    assert fate.failure is failure


def test_row7_provider_suspended_is_not_a_throttle_kind() -> None:
    """``provider_suspended`` is deliberately outside
    ``PROVIDER_THROTTLE_FAILURE_KINDS`` (terminal billing failure, escalates
    instead of cooling down) -- falls through to Crashed (row 10).
    """
    failure = FailureEvidence(kind="provider_suspended", throttled_until=None, fresh=True)
    evidence = _evidence(pid_alive=False, failure=failure)
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Crashed)
    assert fate.failure is failure


@pytest.mark.parametrize("kind", ["quota_exhausted", "rate_limited", "provider_auth"])
def test_row7_every_throttle_kind_routes_to_throttled(kind: str) -> None:
    failure = FailureEvidence(kind=kind, throttled_until=NOW, fresh=True)
    fate = resolve_fate(_evidence(pid_alive=False, failure=failure), now=NOW)
    assert isinstance(fate, Throttled)


# --------------------------------------------------------------------------
# Row 8 / rule 4: dead, no fresh outcome, a fresh clean exit with published
# commits -- the exit code decides only because no outcome file exists.
# --------------------------------------------------------------------------


def test_row8_clean_exit_with_open_pr_is_completed() -> None:
    terminal = _terminal(ended_at=AFTER, exit_code=0)
    evidence = _evidence(
        pid_alive=False,
        terminal=terminal,
        branch=_branch(remote_head_sha="head", open_pr_number=3),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Completed)
    assert fate.pr_number == 3
    assert fate.head_sha == "head"


def test_row8_clean_exit_with_ahead_commits_no_pr_is_pushed_without_pr() -> None:
    terminal = _terminal(ended_at=AFTER, exit_code=0)
    evidence = _evidence(pid_alive=False, terminal=terminal, branch=_branch(remote_ahead=1))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, PushedWithoutPr)


def test_row8_stale_exit_code_is_ignored_not_just_the_rule() -> None:
    """The terminal record's own ``ended_at`` must be newer than
    ``dispatched_at`` before its exit code counts -- otherwise it is stale
    too (design doc §3), not merely "row 8 doesn't apply."
    """
    terminal = _terminal(ended_at=BEFORE, exit_code=0)
    evidence = _evidence(pid_alive=False, terminal=terminal, branch=_branch())
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Crashed)
    assert fate.basis.exit_code is None
    assert any(s.reason == StaleReason.OLDER_THAN_DISPATCH for s in fate.basis.stale)


# --------------------------------------------------------------------------
# Row 9 / rule 3: dead, no outcome at all, remote shows pushed commits --
# any exit code (including none).
# --------------------------------------------------------------------------


def test_row9_no_terminal_no_outcome_remote_ahead_is_pushed_without_pr() -> None:
    evidence = _evidence(pid_alive=False, terminal=None, branch=_branch(remote_ahead=1))
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, PushedWithoutPr)


def test_row9_nonzero_exit_with_remote_ahead_still_credited() -> None:
    terminal = _terminal(ended_at=AFTER, exit_code=1)
    evidence = _evidence(
        pid_alive=False, terminal=terminal, branch=_branch(remote_ahead=2, open_pr_number=11)
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Completed)
    assert fate.pr_number == 11


def test_row9_non_confirming_outcome_still_defers_to_remote_evidence() -> None:
    """B1 (wf-review-opus.md): a fresh outcome that exists but never
    confirms a push (a rework-shaped outcome, ``push_succeeded`` False)
    must not shadow row 9's remote-confirmed evidence. Legacy behaviour was
    ``reported_push or ahead_count > 0`` -- gating row 9 on bare
    ``outcome is None`` silently dropped real pushed work (including a
    #1248 salvage push landing after this outcome was written) whenever any
    non-push-claiming outcome happened to exist.
    """
    outcome = _outcome(outcome="rework_requested", push_succeeded=False)
    evidence = _evidence(
        pid_alive=False,
        terminal=None,
        worktree_outcome=outcome,
        branch=_branch(remote_ahead=1),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, PushedWithoutPr)


def test_row9_outcome_missing_push_field_still_defers_to_remote_evidence() -> None:
    """Same as above, but ``push_succeeded`` is simply absent (``None``)
    rather than explicitly ``False`` -- the common shape for an outcome
    that never claimed anything about the push.
    """
    outcome = _outcome(outcome="completed")
    evidence = _evidence(
        pid_alive=False,
        terminal=None,
        worktree_outcome=outcome,
        branch=_branch(remote_ahead=1, open_pr_number=7),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Completed)
    assert fate.pr_number == 7


def test_row9_self_reported_push_with_no_evidence_still_defers_not_credited() -> None:
    """The companion, negative case: an outcome that DOES confirm a push
    (``push_succeeded is True``) but cannot be verified here (no branch
    evidence at all) must NOT fall through to row 9 -- that self-report-
    with-no-evidence case stays exactly as before the B1 fix (see
    ``tests/test_issue_1006.py``, deferred as an architecture-owner call).
    """
    outcome = _outcome(outcome="completed", push_succeeded=True, head_sha=None)
    evidence = _evidence(
        pid_alive=False,
        terminal=None,
        worktree_outcome=outcome,
        branch=_branch(remote_ahead=None, remote_head_sha=None),
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Crashed)


# --------------------------------------------------------------------------
# Row 10: otherwise -> Crashed.
# --------------------------------------------------------------------------


def test_row10_dead_with_nothing_to_credit_is_crashed() -> None:
    fate = resolve_fate(_evidence(pid_alive=False), now=NOW)
    assert isinstance(fate, Crashed)
    assert fate.failure is None


def test_row10_carries_non_throttle_failure() -> None:
    failure = FailureEvidence(kind="launch_failed", throttled_until=None, fresh=True)
    fate = resolve_fate(_evidence(pid_alive=False, failure=failure), now=NOW)
    assert isinstance(fate, Crashed)
    assert fate.failure is failure


# --------------------------------------------------------------------------
# Freshness (rules 1 and 7): step 0 ahead of the rule table.
# --------------------------------------------------------------------------


def test_freshness_stale_outcome_ignored_next_candidate_used() -> None:
    stale_terminal_outcome = _outcome(
        source=EvidenceSource.TERMINAL, written_at=BEFORE, outcome="blocked"
    )
    terminal = _terminal(ended_at=AFTER, exit_code=None, outcome=stale_terminal_outcome)
    fresh_worktree_outcome = _outcome(
        source=EvidenceSource.WORKTREE, written_at=AFTER, outcome="blocked"
    )
    fate = resolve_fate(
        _evidence(terminal=terminal, worktree_outcome=fresh_worktree_outcome), now=NOW
    )
    assert isinstance(fate, Blocked)
    assert fate.basis.outcome is fresh_worktree_outcome
    assert any(
        s.source == EvidenceSource.TERMINAL and s.reason == StaleReason.OLDER_THAN_DISPATCH
        for s in fate.basis.stale
    )


def test_freshness_stale_outcome_ignored_nothing_left_falls_through() -> None:
    stale_outcome = _outcome(written_at=BEFORE, outcome="blocked")
    evidence = _evidence(
        worktree_outcome=stale_outcome, pid_alive=True, health=WorkerHealth.HEALTHY
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Live)


def test_freshness_head_mismatch_marks_stale() -> None:
    outcome = _outcome(written_at=AFTER, outcome="blocked", head_sha="stale-head")
    evidence = _evidence(
        worktree_outcome=outcome,
        pid_alive=True,
        health=WorkerHealth.HEALTHY,
        branch=_branch(remote_head_sha="live-head"),
    )
    fate = resolve_fate(evidence, now=NOW)
    # blocked is discarded as stale (head mismatch), so liveness decides instead.
    assert isinstance(fate, Live)
    assert any(s.reason == StaleReason.HEAD_MISMATCH for s in fate.basis.stale)


def test_freshness_terminal_wins_over_worktree_when_both_fresh() -> None:
    terminal_outcome = _outcome(
        source=EvidenceSource.TERMINAL, written_at=AFTER, outcome="blocked"
    )
    terminal = _terminal(ended_at=AFTER, outcome=terminal_outcome)
    worktree_outcome = _outcome(source=EvidenceSource.WORKTREE, written_at=AFTER, outcome=None)
    fate = resolve_fate(_evidence(terminal=terminal, worktree_outcome=worktree_outcome), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.basis.outcome is terminal_outcome


def test_freshness_empty_terminal_claim_is_not_decisive_falls_through_to_worktree() -> None:
    """Rule 7: an empty terminal claim (``TerminalEvidence.outcome is
    None``) is "no claim", not "stale evidence" -- it is skipped, not
    reported in ``basis.stale``, and the worktree candidate is used.
    """
    terminal = _terminal(ended_at=AFTER, exit_code=0, outcome=None)
    worktree_outcome = _outcome(written_at=AFTER, outcome="blocked")
    fate = resolve_fate(_evidence(terminal=terminal, worktree_outcome=worktree_outcome), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.basis.outcome is worktree_outcome
    assert fate.basis.stale == ()


def test_freshness_empty_outcome_evidence_object_is_not_decisive_falls_through() -> None:
    """N1 (wf-review-opus.md): a *present* ``OutcomeEvidence`` object with
    every claim field ``None`` -- the exact shape ``_no_pr_outcome_evidence({})``
    and ``rework_outcome._outcome_evidence({})`` build for an empty
    ``.worker-outcome.json`` dict -- must not out-rank a worktree candidate
    that carries a real claim, even though it is timestamp-fresh.

    This is a different shape from the sibling
    ``test_freshness_empty_terminal_claim_is_not_decisive_falls_through_to_worktree``
    test above, which passes ``TerminalEvidence.outcome=None`` (no
    ``OutcomeEvidence`` object at all -- always-correctly skipped). No
    production consumer ever produces that shape for ``{}``; they all
    produce a real, content-empty object, which is what this test builds.
    """
    empty_terminal_outcome = _outcome(source=EvidenceSource.TERMINAL, written_at=AFTER)
    terminal = _terminal(ended_at=AFTER, exit_code=0, outcome=empty_terminal_outcome)
    worktree_outcome = _outcome(written_at=AFTER, outcome="blocked")
    fate = resolve_fate(_evidence(terminal=terminal, worktree_outcome=worktree_outcome), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.basis.outcome is worktree_outcome
    assert fate.basis.stale == ()


def test_freshness_legacy_mode_accepts_and_tags_rule_suffix() -> None:
    outcome = _outcome(written_at=None, outcome="blocked")
    fate = resolve_fate(_evidence(dispatched_at=None, worktree_outcome=outcome), now=NOW)
    assert isinstance(fate, Blocked)
    assert fate.basis.rule.endswith("-nodispatch")


def test_freshness_written_at_unknown_with_dispatch_known_is_stale() -> None:
    outcome = _outcome(written_at=None, outcome="blocked")
    evidence = _evidence(
        dispatched_at=DISPATCHED,
        worktree_outcome=outcome,
        pid_alive=True,
        health=WorkerHealth.HEALTHY,
    )
    fate = resolve_fate(evidence, now=NOW)
    assert isinstance(fate, Live)
    assert any(s.reason == StaleReason.OLDER_THAN_DISPATCH for s in fate.basis.stale)
