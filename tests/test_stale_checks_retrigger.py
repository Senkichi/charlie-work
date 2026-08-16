"""Tests for issue #1274 (W17): stale-check-suite auto-remediation, the
RETRIGGER half wired into ``review()``'s main (non-escalated) janitor-gate
path (part 2 of the lane). Detection itself (``_detect_ci_run_never_created``)
is pre-existing and out of scope here -- see ``test_fix_event_dedup.py``'s
``test_ci_run_never_created_*`` tests for its characterization coverage.

Covers:
* AC3 -- gating correctness: a co-occurring janitor failure (shaped like the
  real #1186/#1192/#1214 population -- missing checks AND a merge conflict)
  does NOT suppress the retrigger; an ordinary janitor failure alone (no
  ``ci_run_never_created_head``) never triggers it.
* AC4 -- the escalated-visibility branch never reaches the retrigger action
  (0 close/reopen/empty-commit calls), even though it still calls
  ``_detect_ci_run_never_created`` for visibility.
* AC5 -- the attempt counter increments by exactly 1 on a successful
  retrigger, a second pass inside ``stale_checks_grace_minutes`` makes ZERO
  mechanical calls and leaves the counter unchanged, and a third pass after
  the grace window elapses (simulated by backdating the persisted timestamp,
  never a real sleep) retriggers again.
* AC6 -- close/reopen failing falls back to the empty-commit push, consuming
  the SAME counter; a fully exhausted PR makes EXACTLY ZERO mechanical calls
  on a subsequent pass.
* AC7 -- exhaustion -> escalation (part 3 of the lane): once
  ``stale_checks_retrigger_attempts >= stale_checks_max_retriggers`` AND the
  check suite is still missing, the linked issue is escalated via
  ``_escalate_issue`` (``reason="stale_checks_retrigger_exhausted"``,
  ``reason_class="mechanical"``) and the SAME ``agent:human-needed`` label
  edge every other cap-exhaustion escalation in this file relies on actually
  fires (asserted on ``fake_gh.labels_added``, not just the state.json
  write). Dedup is proven two ways: a second ``review()`` pass with the
  issue/PR already escalated does not re-fire (the structural
  escalated-visibility early return), and the escalation method's own
  internal guard clause is exercised directly, independent of that
  structural path.
* AC8 -- both ``ci_retriggered_stale_checks`` and
  ``stale_checks_retrigger_exhausted`` are registered in
  instrumentation.py's kind registry (``test_event_kind_registry_exhaustive``
  in ``tests/test_instrumentation.py`` covers this structurally) and are
  actually emitted at their call sites -- proven functionally here (the
  event appears in state.json with the right payload after the triggering
  action), which is strictly stronger evidence than a source grep.
* AC11 -- an end-to-end integration smoke test
  (``test_full_chain_detection_through_retrigger_to_exhaustion_escalation``)
  drives a single #1186/#1192/#1214-shaped fixture through detection, two
  gated retrigger attempts, and the final exhaustion escalation across a
  sequence of simulated passes, asserting the full chain composes correctly
  in one coherent state.json rather than re-testing each link in isolation.

AC1/AC2 (the detection-predicate characterization and the
no-second-detection-window structural fence) live in
``test_stale_checks_detection_regression.py`` instead of here, since they
pin ``_detect_ci_run_never_created`` itself rather than this file's
``review()``-level retrigger/escalation policy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.config import AutoMergeConfig, OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub
from test_fix_event_dedup import FakeGitHubWithMissingRequiredAndRuns

_STALE_HEAD = "abc123abc123"


def _stale_checks_config(
    *,
    ci_run_never_created_grace_minutes: int = 5,
    stale_checks_grace_minutes: int = 15,
    stale_checks_max_retriggers: int = 3,
) -> OrchestratorConfig:
    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
        enabled=True,
        ci_run_never_created_grace_minutes=ci_run_never_created_grace_minutes,
    )
    review = ReviewConfig(
        stale_checks_grace_minutes=stale_checks_grace_minutes,
        stale_checks_max_retriggers=stale_checks_max_retriggers,
    )
    return OrchestratorConfig(auto_merge=auto_merge, review=review)


class FakeGitHubMissingChecksAndConflict(FakeGitHubWithMissingRequiredAndRuns):
    """Models #1186/#1192/#1214's real shape: required checks were never
    created for the head AND the PR independently has a merge conflict --
    the co-occurring-failure population issue #1274's binding comment item 5
    says the retrigger must still fire against.
    """

    def __init__(self, runs: list[dict[str, Any]] | None) -> None:
        super().__init__(runs=runs)
        self.prs[0]["mergeable"] = "CONFLICTING"


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _stale_updated_at() -> str:
    return (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")


def _app_with_conflict_and_missing_checks(
    tmp_path: Path,
    *,
    runs: list[dict[str, Any]] | None = None,
    issue_status: str = "dispatched",
    head_sha: str = _STALE_HEAD,
    **config_kwargs: Any,
) -> OrchestratorApp:
    """A PR shaped like #1186/#1192/#1214: missing required checks (so
    ``_detect_ci_run_never_created`` can fire) AND a merge conflict (so
    ``_route_janitor_gate_failure_to_rework`` runs first). Verified against
    all three PRs on 2026-08-16: each has ``mergeable=CONFLICTING`` and an
    empty ``statusCheckRollup`` -- confirming the missing-checks-AND-conflict
    co-occurrence this fixture models is real, not hypothetical.

    ``issue_status`` defaults to "dispatched" purely as a reachability
    choice for the fixture: it is the status value that makes
    ``_route_janitor_gate_failure_to_rework`` return None (rework already
    pending) and fall through to the main janitor-gate path this test
    exercises. It is NOT a claim about the linked issues' real status.

    As of 2026-08-16, all three linked issues (#807/#763/#1068) actually
    carry ``agent:human-needed`` on GitHub -- i.e. they are escalated, so
    ``review()`` currently takes the *escalated* early-return branch for
    these PRs, not the main gate wired here (AC4 forbids this fixture from
    touching that branch, and item 7/part 3 owns exhaustion->escalation
    routing). Why they're escalated is not determinable from here: the
    escalation reason lives in the live ``.var/`` state.json this agent is
    fenced out of, and `gh` doesn't expose it. Flag for part 3: decide
    whether exhaustion->escalation alone is sufficient, or whether the
    stale-checks retrigger lane also needs an entry point on the escalated
    path to reach this population.
    """
    config = _stale_checks_config(**config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubMissingChecksAndConflict(runs=[] if runs is None else runs)
    fake_gh.prs[0]["headRefOid"] = head_sha
    fake_gh.prs[0]["updatedAt"] = _stale_updated_at()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": issue_status}
        save_state(app.paths.state_file, state)
    return app


def _stale_checks_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return _events(state, "ci_retriggered_stale_checks")


# ---------------------------------------------------------------------------
# AC3: gating correctness
# ---------------------------------------------------------------------------


def test_retrigger_fires_despite_co_occurring_merge_conflict_failure(tmp_path: Path) -> None:
    """The gate that matters is "is the check-suite run missing", not "is it
    the ONLY janitor failure" (binding comment item 5 on issue #1274 --
    deliberately reverses a strict ``is_missing_checks_only_block`` gate).
    A fixture whose PR ALSO has a merge conflict (mergeable=CONFLICTING)
    still gets a retrigger attempt.
    """
    app = _app_with_conflict_and_missing_checks(tmp_path)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("stale_checks_retriggered") is True
    assert app.gh.pr_close_calls == [456]
    assert app.gh.pr_reopen_calls == [456]
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1
    assert state["prs"]["456"]["is_missing_checks_only_block"] is False
    events = _stale_checks_events(state)
    assert len(events) == 1
    assert events[0]["payload"]["method"] == "close_reopen"
    assert events[0]["payload"]["attempt"] == 1


def test_retrigger_not_triggered_without_ci_run_never_created_head(tmp_path: Path) -> None:
    """An ordinary janitor failure alone (merge conflict, with required
    checks reported normally -- not "never created") must never call
    pr_close/pr_reopen/push_empty_commit. This isolates
    ``ci_run_never_created_head`` as the discriminator, not merely "the PR
    is janitor_blocked" or "the PR has a merge conflict".
    """
    config = _stale_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "dispatched"}
        save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("stale_checks_retriggered") is None
    assert app.gh.pr_close_calls == []
    assert app.gh.pr_reopen_calls == []
    assert app.gh.push_empty_commit_calls == []
    state = load_state(app.paths.state_file)
    assert "stale_checks_retrigger_attempts" not in state["prs"]["456"]
    # Reachability control (not just a vacuous zero-calls count): confirm
    # execution actually reached the terminal janitor-gate-blocked return
    # AFTER passing through the gate -- not that it never got there because
    # an earlier branch (e.g. the merge-conflict router) returned first.
    assert result.message.startswith("janitor gate blocked PR #456")


def test_retrigger_not_triggered_when_runs_pending_not_missing(tmp_path: Path) -> None:
    """Checks are "missing" per the janitor gate, but Actions HAS already
    created run(s) for the head (pending, not never-created) --
    ``_detect_ci_run_never_created`` legitimately returns None, and with no
    persisted marker either, the retrigger must not fire.
    """
    app = _app_with_conflict_and_missing_checks(tmp_path, runs=[{"id": 1, "status": "queued"}])

    result = app.review(456)

    assert result.ok is False
    assert app.gh.pr_close_calls == []
    assert app.gh.pr_reopen_calls == []
    assert app.gh.push_empty_commit_calls == []
    # Reachability control: same rationale as the sibling test above -- prove
    # the zero-calls count comes from a gate that declined, not a branch that
    # short-circuited before the gate was ever reached.
    assert result.message.startswith("janitor gate blocked PR #456")


# ---------------------------------------------------------------------------
# AC4: escalated-branch exclusion
# ---------------------------------------------------------------------------


def test_escalated_branch_never_calls_retrigger_action(tmp_path: Path) -> None:
    """The escalated-visibility branch (``review()``'s early return for an
    escalated PR/issue) keeps calling ``_detect_ci_run_never_created`` for
    visibility/bookkeeping only -- it must NEVER reach pr_close/pr_reopen/
    push_empty_commit. Count-based assertion (0 calls), not merely "no
    error" -- mirrors ``test_ci_run_never_created_fires_for_an_escalated_pr``'s
    fixture shape (issue escalated) so the detector's own visibility
    behavior is proven unchanged in the same breath.
    """
    config = _stale_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[])
    fake_gh.prs[0]["headRefOid"] = _STALE_HEAD
    fake_gh.prs[0]["updatedAt"] = _stale_updated_at()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": [],
        }
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.review(456)

    assert result.ok is True
    assert result.data.get("pass_skipped") is True
    # The detector itself still fires for visibility (pre-existing behavior,
    # unchanged by this diff).
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["ci_run_never_created_head"] == _STALE_HEAD
    # But the retrigger action is never reachable from this branch.
    assert app.gh.pr_close_calls == []
    assert app.gh.pr_reopen_calls == []
    assert app.gh.push_empty_commit_calls == []
    assert "stale_checks_retrigger_attempts" not in state["prs"]["456"]
    assert len(_stale_checks_events(state)) == 0


# ---------------------------------------------------------------------------
# AC5: attempt counter + grace-after-retrigger wait
# ---------------------------------------------------------------------------


def test_retrigger_attempt_counter_and_grace_wait_sequence(tmp_path: Path) -> None:
    app = _app_with_conflict_and_missing_checks(
        tmp_path, stale_checks_grace_minutes=15, stale_checks_max_retriggers=3
    )

    # Pass 1: in scope, no prior attempt -> retriggers, attempts -> 1.
    result1 = app.review(456)
    assert result1.data.get("stale_checks_retriggered") is True
    assert app.gh.pr_close_calls == [456]
    assert app.gh.pr_reopen_calls == [456]
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1
    first_retrigger_at = state["prs"]["456"]["stale_checks_last_retrigger_at"]
    assert first_retrigger_at

    # Pass 2: still inside the grace window (just retriggered) -> makes
    # ZERO mechanical calls, attempt count UNCHANGED, no new event.
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data.get("stale_checks_retriggered") is None
    assert app.gh.pr_close_calls == [456]  # unchanged
    assert app.gh.pr_reopen_calls == [456]  # unchanged
    assert app.gh.push_empty_commit_calls == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1
    assert state["prs"]["456"]["stale_checks_last_retrigger_at"] == first_retrigger_at
    assert len(_stale_checks_events(state)) == 1

    # Backdate the persisted retrigger timestamp to simulate the grace
    # window elapsing -- never a real sleep.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        backdated = (datetime.now(UTC) - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
        state["prs"]["456"]["stale_checks_last_retrigger_at"] = backdated
        save_state(app.paths.state_file, state)

    # Pass 3: grace window elapsed, still missing -> retriggers again,
    # attempts -> 2.
    result3 = app.review(456)
    assert result3.data.get("stale_checks_retriggered") is True
    assert app.gh.pr_close_calls == [456, 456]
    assert app.gh.pr_reopen_calls == [456, 456]
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 2
    assert state["prs"]["456"]["stale_checks_last_retrigger_at"] != backdated
    assert len(_stale_checks_events(state)) == 2


# ---------------------------------------------------------------------------
# AC6 (partial): fallback to empty-commit push, and the bound
# ---------------------------------------------------------------------------


def test_close_reopen_failure_falls_back_to_empty_commit_same_counter(tmp_path: Path) -> None:
    app = _app_with_conflict_and_missing_checks(tmp_path)
    app.gh.pr_close_ok = False

    result = app.review(456)

    assert result.data.get("stale_checks_retriggered") is True
    assert result.data.get("stale_checks_retrigger_method") == "empty_commit"
    assert app.gh.pr_close_calls == [456]
    assert app.gh.pr_reopen_calls == []  # never called once close failed
    assert app.gh.push_empty_commit_calls == ["agent/issue-123-fix-search"]
    state = load_state(app.paths.state_file)
    # Same shared counter, not a separate one.
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1
    events = _stale_checks_events(state)
    assert len(events) == 1
    assert events[0]["payload"]["method"] == "empty_commit"


def test_reopen_failure_falls_back_to_empty_commit(tmp_path: Path) -> None:
    app = _app_with_conflict_and_missing_checks(tmp_path)
    app.gh.pr_reopen_ok = False

    result = app.review(456)

    assert result.data.get("stale_checks_retriggered") is True
    assert result.data.get("stale_checks_retrigger_method") == "empty_commit"
    assert app.gh.pr_close_calls == [456]
    assert app.gh.pr_reopen_calls == [456]
    assert app.gh.push_empty_commit_calls == ["agent/issue-123-fix-search"]
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1


def test_total_mechanical_failure_does_not_consume_attempt(tmp_path: Path) -> None:
    """A transient gh API error on every path (close, reopen never reached
    since close already failed, AND the empty-commit fallback) must not
    burn the bounded retrigger budget -- mirrors the flake-rerun block's
    "record the error but do not consume the attempt" convention.
    """
    app = _app_with_conflict_and_missing_checks(tmp_path)
    app.gh.pr_close_ok = False
    app.gh.push_empty_commit_ok = False

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("stale_checks_retriggered") is None
    state = load_state(app.paths.state_file)
    assert "stale_checks_retrigger_attempts" not in state["prs"]["456"]
    assert len(_stale_checks_events(state)) == 0


def test_exhausted_bound_makes_zero_mechanical_calls(tmp_path: Path) -> None:
    """Once ``stale_checks_retrigger_attempts >= stale_checks_max_retriggers``,
    a subsequent pass must make EXACTLY ZERO close/reopen/empty-commit calls
    (call-count assertion, not merely "no error"). This same pass is also
    where exhaustion -> escalation (AC7) fires -- see the dedicated
    escalation assertions below, and ``test_exhausted_bound_escalates_to_human``
    for the fuller AC7 fixture (label edge + event payload + dedup).
    """
    app = _app_with_conflict_and_missing_checks(tmp_path, stale_checks_max_retriggers=2)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": [],
            "ci_run_never_created_head": _STALE_HEAD,
            "stale_checks_retrigger_attempts": 2,
            "stale_checks_last_retrigger_at": (
                (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            ),
        }
        save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("stale_checks_retriggered") is None
    assert app.gh.pr_close_calls == []
    assert app.gh.pr_reopen_calls == []
    assert app.gh.push_empty_commit_calls == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 2
    # The same pass that stops attempting also escalates (AC7): the counter
    # is exhausted AND the check suite is still missing this pass (that's
    # `stale_checks_retrigger_in_scope`'s own precondition for even calling
    # into the retrigger method), so this is exactly item 7's trigger.
    assert result.data.get("stale_checks_retrigger_exhausted") is True
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "stale_checks_retrigger_exhausted"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"


# ---------------------------------------------------------------------------
# AC7: exhaustion -> escalation (part 3 of the lane)
# ---------------------------------------------------------------------------


def test_exhausted_bound_escalates_to_human(tmp_path: Path) -> None:
    """Once exhausted AND the check suite is still missing, the linked issue
    is escalated via ``_escalate_issue`` with a distinct reason and
    ``reason_class="mechanical"`` -- and the SAME ``_escalate_issue`` +
    ``transition(...)``-backed ``agent:human-needed`` label edge every other
    cap-exhaustion escalation in this file relies on actually fires. Assert
    the label transition call landed (``fake_gh.labels_added``), not just the
    state.json write -- mirrors the concern
    ``_route_janitor_gate_failure_to_rework``'s docstring raises about a past
    regression here (a write with no label is functionally invisible: the
    issue never reaches a human's queue).
    """
    app = _app_with_conflict_and_missing_checks(tmp_path, stale_checks_max_retriggers=1)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": [],
            "ci_run_never_created_head": _STALE_HEAD,
            "stale_checks_retrigger_attempts": 1,
            "stale_checks_last_retrigger_at": (
                (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            ),
        }
        save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("stale_checks_retrigger_exhausted") is True
    assert result.data.get("label_error") is None
    assert (123, app.config.labels.human_needed) in app.gh.labels_added

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "stale_checks_retrigger_exhausted"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["prs"]["456"]["escalation_reason"] == "stale_checks_retrigger_exhausted"

    events = _events(state, "stale_checks_retrigger_exhausted")
    assert len(events) == 1
    assert events[0]["payload"]["pr_number"] == 456
    assert events[0]["payload"]["issue_number"] == 123
    assert events[0]["payload"]["head_sha"] == _STALE_HEAD
    assert events[0]["payload"]["attempts"] == 1
    assert events[0]["payload"]["max_retriggers"] == 1
    # AC6: mechanical calls still stay at zero -- exhaustion routes to
    # escalation, never to another retrigger attempt.
    assert app.gh.pr_close_calls == []
    assert app.gh.pr_reopen_calls == []
    assert app.gh.push_empty_commit_calls == []


def test_exhaustion_guard_skips_when_already_escalated_for_this_reason(tmp_path: Path) -> None:
    """Direct unit coverage for ``_escalate_stale_checks_exhaustion``'s own
    dedup guard (``existing_pr_state.get("escalation_reason") ==
    exhaustion_reason``), calling the method directly rather than through
    ``review()``. The method's own docstring calls this guard
    belt-and-suspenders relative to ``review()``'s structural
    escalated-visibility early return (``test_exhausted_escalation_does_not_refire_on_a_later_pass``
    below proves that structural path); this test proves the guard clause
    itself trips independent of that early return -- the one case the
    docstring names as the reason it exists (something resets ``status``
    while ``escalation_reason`` survives). No ``state_lock`` should even be
    entered: zero label calls, zero events, and the pre-existing issue state
    left untouched.
    """
    config = _stale_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "stale_checks_retrigger_exhausted",
            "reason_class": "mechanical",
        }
        save_state(app.paths.state_file, state)

    result = app._escalate_stale_checks_exhaustion(
        pr_number=456,
        issue_number=123,
        head_sha=_STALE_HEAD,
        attempts=3,
        max_retriggers=3,
        existing_pr_state={"escalation_reason": "stale_checks_retrigger_exhausted"},
    )

    assert result is None
    assert app.gh.labels_added == []

    state = load_state(app.paths.state_file)
    assert _events(state, "stale_checks_retrigger_exhausted") == []
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "stale_checks_retrigger_exhausted"


def test_exhausted_escalation_does_not_refire_on_a_later_pass(tmp_path: Path) -> None:
    """A second pass, with the PR/issue already escalated for this exact
    reason, must not re-escalate: no duplicate
    ``stale_checks_retrigger_exhausted`` event, no duplicate label call, and
    the escalation fields are left exactly as they were. In normal operation
    this is structurally guaranteed by ``review()``'s own top-of-function
    escalated-visibility early return (``status == "escalated"`` routes away
    from the janitor-gate path entirely, the same way it already excludes
    the retrigger action -- scope fence item 3/b) -- this test exercises
    that real path (calling ``review()`` again, not the escalation method
    directly).

    Deliberately uses a fixture with NO co-occurring merge conflict (unlike
    the AC3/AC6 fixtures above): issue #776's pre-existing "escalation is
    terminal only for the lane that caused it" escape hatch means a PR that
    ALSO has an independently-capped merge conflict legitimately gets routed
    to conflict rework again on pass 2, from inside the escalated-visibility
    branch itself, even while escalated for this (unrelated) reason -- that
    is correct, intentional behavior for that population, not something this
    test should fight. Isolating the plain "no other co-occurring janitor
    failure" case here proves the specific structural claim about THIS
    lane's own dedup cleanly, without #776's unrelated escape hatch
    confounding the assertion.
    """
    config = _stale_checks_config(stale_checks_max_retriggers=1)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[])
    fake_gh.prs[0]["headRefOid"] = _STALE_HEAD
    fake_gh.prs[0]["updatedAt"] = _stale_updated_at()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "dispatched"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": [],
            "ci_run_never_created_head": _STALE_HEAD,
            "stale_checks_retrigger_attempts": 1,
            "stale_checks_last_retrigger_at": (
                (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            ),
        }
        save_state(app.paths.state_file, state)

    result1 = app.review(456)
    assert result1.data.get("stale_checks_retrigger_exhausted") is True
    state = load_state(app.paths.state_file)
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 1
    first_labels_added = list(app.gh.labels_added)

    # Second pass: issue/PR status is now "escalated" -> review() takes the
    # escalated-visibility early return before ever reaching the janitor-gate
    # path this lane's retrigger/escalation logic lives in. No co-occurring
    # merge conflict here, so nothing else re-enters remediation either.
    result2 = app.review(456)
    assert result2.ok is True
    assert result2.data.get("pass_skipped") is True

    state = load_state(app.paths.state_file)
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 1
    assert app.gh.labels_added == first_labels_added
    assert state["issues"]["123"]["escalation_reason"] == "stale_checks_retrigger_exhausted"


# ---------------------------------------------------------------------------
# AC11: end-to-end integration smoke test on a #1186/#1192/#1214-shaped
# fixture -- detection -> gated retrigger -> event -> (follow-up simulated
# passes) exhaustion -> escalation, asserting the full chain lands correctly
# in state.json for a SINGLE PR across a single ``review()`` pass sequence.
# The individual links are each already unit-tested above (AC3/AC5/AC6/AC7);
# this test's job is only to prove they compose end-to-end without a step
# in the middle silently dropping state the next step depends on.
# ---------------------------------------------------------------------------


def test_full_chain_detection_through_retrigger_to_exhaustion_escalation(
    tmp_path: Path,
) -> None:
    """Fixture shaped like the real #1186/#1192/#1214 population: required
    checks never created for the head AND an independent merge conflict.
    Drives three simulated passes (grace window elapsed via backdating the
    persisted timestamp between passes, never a real sleep) over the SAME
    PR/head and asserts the full chain -- detection, two retrigger attempts,
    and the final exhaustion escalation -- all land correctly in one
    coherent state.json, in the order the real janitor-gate loop would
    produce them.
    """
    app = _app_with_conflict_and_missing_checks(
        tmp_path, stale_checks_grace_minutes=15, stale_checks_max_retriggers=2
    )

    # Pass 1: fresh head, never seen before -> _detect_ci_run_never_created
    # fires for the first time (ci_run_never_created event) AND the retrigger
    # action is in scope this same pass (persisted-or-live marker) -> attempt 1.
    result1 = app.review(456)
    assert result1.ok is False
    assert result1.data.get("stale_checks_retriggered") is True
    assert result1.data.get("stale_checks_retrigger_method") == "close_reopen"
    assert result1.data.get("stale_checks_retrigger_attempts") == 1

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["ci_run_never_created_head"] == _STALE_HEAD
    assert state["prs"]["456"]["is_missing_checks_only_block"] is False
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 1
    assert len(_events(state, "ci_run_never_created")) == 1
    assert len(_stale_checks_events(state)) == 1
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 0
    assert state["prs"]["456"]["status"] != "escalated"

    # Elapse the grace window (backdate, never sleep) and take pass 2: the
    # SAME head is still marked never-created, checks are still missing ->
    # retriggers again, attempts -> 2 == max_retriggers.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        backdated = (datetime.now(UTC) - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
        state["prs"]["456"]["stale_checks_last_retrigger_at"] = backdated
        save_state(app.paths.state_file, state)

    result2 = app.review(456)
    assert result2.data.get("stale_checks_retriggered") is True
    assert result2.data.get("stale_checks_retrigger_attempts") == 2

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 2
    assert len(_stale_checks_events(state)) == 2
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 0
    assert app.gh.pr_close_calls == [456, 456]
    assert app.gh.pr_reopen_calls == [456, 456]

    # Elapse the grace window once more and take pass 3: attempts (2) is now
    # >= max_retriggers (2) on entry, AND the check suite is still missing
    # this pass -> exhaustion routes to escalation instead of a third
    # mechanical attempt.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        backdated2 = (datetime.now(UTC) - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
        state["prs"]["456"]["stale_checks_last_retrigger_at"] = backdated2
        save_state(app.paths.state_file, state)

    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data.get("stale_checks_retrigger_exhausted") is True
    assert result3.data.get("label_error") is None

    # Zero additional mechanical calls on the exhausting pass (AC6b), and the
    # counter is left exactly where it was (AC7 does not bump attempts).
    assert app.gh.pr_close_calls == [456, 456]
    assert app.gh.pr_reopen_calls == [456, 456]
    assert app.gh.push_empty_commit_calls == []

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_checks_retrigger_attempts"] == 2
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "stale_checks_retrigger_exhausted"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, app.config.labels.human_needed) in app.gh.labels_added

    # The full event trail, in order, is exactly what the three passes
    # should have produced: one detection, two retriggers, one exhaustion.
    assert len(_events(state, "ci_run_never_created")) == 1
    assert len(_stale_checks_events(state)) == 2
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 1

    # Pass 4: now escalated for THIS reason -> the stale-checks lane never
    # re-fires (zero additional close/reopen/empty-commit calls, no
    # duplicate exhaustion event -- see AC7's dedicated dedup test,
    # ``test_exhausted_escalation_does_not_refire_on_a_later_pass``, for the
    # structural "escalated-visibility early return" proof on a fixture
    # without a co-occurring conflict). This fixture's own independent
    # merge-conflict rework lane is a separate, unrelated escape hatch
    # (issue #776) that may still route pass 4 to conflict rework -- that is
    # correct, intentional behavior for this population, not a claim this
    # test makes about THIS lane's chain, so only the stale-checks-specific
    # counters are asserted here.
    result4 = app.review(456)
    assert result4.data.get("stale_checks_retrigger_exhausted") is not True
    state = load_state(app.paths.state_file)
    assert app.gh.pr_close_calls == [456, 456]
    assert app.gh.pr_reopen_calls == [456, 456]
    assert app.gh.push_empty_commit_calls == []
    assert len(_events(state, "stale_checks_retrigger_exhausted")) == 1
    assert len(_stale_checks_events(state)) == 2
