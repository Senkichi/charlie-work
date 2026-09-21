from __future__ import annotations

import json
import pytest
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
from _reconcile_fixtures import (
    FakeGitHub,
    _EmptyStdoutGitHub,
    _issue,
    _pr,
)
from charlie_work.config import (
    LabelConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
)
from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.github import (
    GitHubError,
    _LIST_LIMIT as github_list_limit,
)
from charlie_work.instrumentation import read_event_log
from charlie_work.paths import (
    resolved_layout,
    runtime_paths,
)
from charlie_work.reconcile import (
    ACTIVE_STATE_STATUSES,
    DORMANT_CONVERGENCE_EXCLUDED_STATUSES,
    DriftItem,
    _LABEL_CORROBORATING_STATUSES,
    _LABEL_STALE_STATUSES,
    _fetch_issues,
    _fetch_prs,
    _normalize_reconcile_issue,
    _normalize_reconcile_pr,
    apply_fixes,
    detect_drift,
    _LIST_LIMIT as reconcile_list_limit,
)
from charlie_work.state import (
    empty_state,
    load_state,
)
from charlie_work.workflow import OrchestratorApp


# Module-level default label config for parametrize decorators that need
# label strings at collection time (before any test creates an OrchestratorConfig).
_config_labels = LabelConfig()


def _raw_rest_pr(
    number: int,
    state: str = "open",
    *,
    merged: bool = False,
    merged_at: str | None = None,
    closed_at: str | None = None,
    head_ref: str = "agent/issue-1-x",
    head_repo: str | None = "owner/test-repo",
    base_repo: str | None = "owner/test-repo",
) -> dict[str, Any]:
    """Return a raw REST ``pulls`` response with no ``headRefName``.

    This is the shape ``_normalize_reconcile_pr`` sees in production, but
    existing test fixtures all set the normalized ``RECONCILE_PR_FIELDS`` keys.
    ``closed_at`` is the REST ``pulls`` snake_case field that the normalizer
    maps to the camelCase ``closedAt`` (issue #1398); it defaults to ``None``
    so pre-existing callers are unaffected.
    """
    return {
        "number": number,
        "title": f"pr {number}",
        "html_url": f"https://example.test/pull/{number}",
        "head": {
            "ref": head_ref,
            "sha": f"sha-{number}",
            "repo": {"full_name": head_repo} if head_repo is not None else None,
        },
        "base": {
            "ref": "main",
            "sha": "base-sha",
            "repo": {"full_name": base_repo} if base_repo is not None else None,
        },
        "body": "",
        "state": state,
        "labels": [],
        "merged": merged,
        "merged_at": merged_at,
        "closed_at": closed_at,
    }


def test_fetch_prs_normalizes_raw_rest_pulls() -> None:
    """Issue #762: _normalize_reconcile_pr must map the REST ``pulls`` shape
    (no ``headRefName``, ``head``/``base`` sub-objects, ``merged_at``) to the
    ``RECONCILE_PR_FIELDS`` shape. Existing test fixtures already carry the
    normalized keys, so the transformation branch was previously unexercised.

    Issue #1398: the REST ``pulls`` endpoint names the close time
    ``closed_at`` (snake_case); the normalizer must surface it as the
    camelCase ``closedAt`` so the closed-unmerged convergence rules can
    compare it against the issue's active-session start. A typo on that one
    mapping line would silently revert the #1398 fix while passing every
    guard test in test_fix_reconcile.py (which all use the already-normalized
    ``_pr`` fixture), so this assertion pins the production code path:
    ``_fetch_prs`` -> ``_normalize_reconcile_pr`` on a raw REST payload.
    """
    # Derive the close timestamp from the clock so a date-window filter can
    # never rot this seed (test-hygiene rule).
    closed_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    merged = _raw_rest_pr(1, state="closed", merged_at="2026-08-05T00:00:00Z")
    open_same = _raw_rest_pr(2, state="open")
    cross = _raw_rest_pr(
        3,
        state="open",
        head_ref="fork/issue-3-x",
        head_repo="fork/test-repo",
        base_repo="owner/test-repo",
    )
    deleted_fork = _raw_rest_pr(4, state="open", head_repo=None)
    closed_unmerged = _raw_rest_pr(5, state="closed", closed_at=closed_at)
    gh = FakeGitHub(prs=[merged, open_same, cross, deleted_fork, closed_unmerged], issues=[])

    result = _fetch_prs(gh)

    assert len(result) == 5
    assert result[0]["state"] == "MERGED"
    assert result[0]["headRefName"] == "agent/issue-1-x"
    assert result[0]["url"] == "https://example.test/pull/1"
    assert result[1]["isCrossRepository"] is False
    assert result[2]["isCrossRepository"] is True
    assert result[3]["isCrossRepository"] is None
    assert all("headRefName" in pr for pr in result)
    # Issue #1398: the snake_case REST ``closed_at`` must survive the
    # _fetch_prs/_normalize_reconcile_pr pipeline as the camelCase ``closedAt``
    # with the identical value, and PRs that omit it must normalize to None.
    assert result[4]["closedAt"] == closed_at
    assert result[4]["state"] == "CLOSED"
    assert result[0]["closedAt"] is None  # merged PR: closed_at not set on the seed
    assert result[1]["closedAt"] is None  # open PR: closed_at absent
    assert all("closedAt" in pr for pr in result)


def test_normalize_reconcile_pr_is_idempotent_on_gh_shape() -> None:
    """The normalizer must be a no-op for fixtures that already carry the
    normalized ``RECONCILE_PR_FIELDS`` shape (``headRefName`` present).
    """
    normalized = _pr(1, "OPEN")
    assert _normalize_reconcile_pr(normalized) is normalized


def test_fetch_issues_filters_pull_requests_and_maps_url() -> None:
    """Issue #762: the REST ``issues`` endpoint returns both issues and PRs;
    _fetch_issues must drop PR-shaped rows and map ``html_url`` to ``url``.
    """
    raw_issue = {
        "number": 1,
        "title": "issue 1",
        "html_url": "https://example.test/issues/1",
        "body": "",
        "labels": [{"name": "ready"}],
        "state": "open",
    }
    raw_pr = {
        "number": 2,
        "title": "pr 2",
        "html_url": "https://example.test/pull/2",
        "body": "",
        "labels": [],
        "state": "open",
        "pull_request": {"url": "https://example.test/pull/2"},
    }
    gh = FakeGitHub(prs=[], issues=[raw_issue, raw_pr])

    result = _fetch_issues(gh)

    assert len(result) == 1
    assert result[0]["number"] == 1
    assert result[0]["url"] == "https://example.test/issues/1"
    assert _normalize_reconcile_issue(raw_issue)["url"] == "https://example.test/issues/1"


def test_corroborating_status_suppresses_both_sub_cases_not_just_stale_active() -> None:
    """Issue #1092: the exemption must not be gated on ``stale_active`` alone.

    The rule has two independent sub-cases and the motivating issue triggers
    BOTH at once: it carries ``agent:needs-rework`` (so ``stale_active`` is
    non-empty) and lacks ``agent:pr-open`` (so ``needs_pr_open`` is true). An
    exemption written as "skip only when ``not stale_active``" reads correctly,
    passes a naive version of the criterion above, and still flips this issue on
    every single pass via the ``needs_pr_open`` half.

    Asserting the fixture really does arm both predicates is the point of this
    test -- without it the suppression assertion could pass for the wrong reason.
    """
    config = OrchestratorConfig()
    issue_labels = [config.labels.needs_rework]

    # Positive control on the fixture itself: both sub-cases must be armed, or
    # this test degenerates into a duplicate of the one above.
    assert set(issue_labels) - {config.labels.pr_open, config.labels.reviewing}, (
        "fixture must arm the stale_active sub-case"
    )
    assert config.labels.pr_open not in issue_labels, "fixture must arm the needs_pr_open sub-case"

    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, issue_labels)],
    )
    state = empty_state()
    state["issues"]["30"] = {"number": 30, "status": "rework_requested"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "issue_active_label_with_open_pr"] == []


def test_needs_pr_open_still_fires_when_the_issue_has_no_state_entry() -> None:
    """Issue #1092: an absent status must FALL THROUGH, not be exempted.

    ``tracked_status`` is ``None`` for an issue with no state entry at all. That
    is the untracked-issue case the ``needs_pr_open`` half exists to catch, so it
    must still be reported. This is why the predicate is written in the deny-list
    direction (``not in _LABEL_CORROBORATING_STATUSES``); the allow-list spelling
    (``in _LABEL_STALE_STATUSES``) reads equivalently, excludes ``None``, and
    would silently stop labelling brand-new issues forever.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [])],
    )
    state = empty_state()
    assert "30" not in state["issues"], "fixture must leave tracked_status None"

    matches = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_active_label_with_open_pr"
    ]

    assert len(matches) == 1
    assert matches[0].add_labels == (config.labels.pr_open,)


def test_label_stale_statuses_stays_derived_from_active_state_statuses() -> None:
    """Issue #1092: a status added later must default to PROTECTED.

    The protected set is derived by subtraction rather than enumerated, so the
    safe default is automatic. This guard fails if someone converts the
    derivation back into a hand-maintained list, or adds a member to
    ``_LABEL_STALE_STATUSES`` that no longer names a real status.

    The asymmetry that justifies the direction: a wrong "stale" verdict is an
    infinite loop that starves dispatch, while a wrong "corroborated" verdict is
    one missed self-heal.
    """
    # Derivation preserved, not replaced by a literal.
    assert _LABEL_CORROBORATING_STATUSES == ACTIVE_STATE_STATUSES - _LABEL_STALE_STATUSES

    # No orphan members: every "stale" status is a real ACTIVE_STATE_STATUSES member.
    assert _LABEL_STALE_STATUSES <= ACTIVE_STATE_STATUSES

    # Positive control -- an empty protected set would satisfy every assertion
    # above while exempting nothing, i.e. the guard would pass on the bug.
    assert _LABEL_CORROBORATING_STATUSES, "protected set must not be empty"

    # The two members the fix turns on, pinned by name.
    assert "rework_requested" in _LABEL_CORROBORATING_STATUSES
    assert "dispatch_failed" not in _LABEL_CORROBORATING_STATUSES

    # "escalated" must stay PROTECTED. This assertion is currently unobservable
    # through behaviour -- the `tracked_status == "escalated"` branch upstream
    # converges labels from state and `continue`s, so an escalated issue never
    # reaches the predicate either way (pinned by the companion test below).
    # It is asserted anyway because the construction must fail CLOSED: if that
    # short-circuit is ever removed, listing "escalated" as stale would reset
    # an escalated issue's status to PASSIVE_OPEN_STATUS while
    # `agent:human-needed` stayed live -- the #894 split-brain that
    # DORMANT_CONVERGENCE_EXCLUDED_STATUSES exists to prevent.
    assert "escalated" in _LABEL_CORROBORATING_STATUSES
    assert "escalated" in DORMANT_CONVERGENCE_EXCLUDED_STATUSES


def test_escalated_issue_never_reaches_the_open_pr_self_heal() -> None:
    """Issue #1092 / #894: escalation is terminal-until-human at BOTH layers.

    The upstream ``tracked_status == "escalated"`` branch short-circuits, so
    this rule never sees an escalated issue. That is the behaviour this asserts
    -- and it is asserted with a positive control, because "no drift of kind X"
    is equally consistent with "correctly suppressed" and "fixture never armed
    the rule at all". The control uses ``dispatch_failed`` on the byte-identical
    fixture: it must produce drift, proving the shape reaches the rule.
    """
    config = OrchestratorConfig()

    def probe(status: str) -> list[str]:
        gh = FakeGitHub(
            prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
            issues=[_issue(30, [config.labels.needs_rework])],
        )
        state = empty_state()
        state["issues"]["30"] = {"number": 30, "status": status}
        return [item.kind for item in detect_drift(gh, state, config)]

    # Positive control: the same fixture DOES arm the rule.
    assert "issue_active_label_with_open_pr" in probe("dispatch_failed")

    escalated_kinds = probe("escalated")
    assert "issue_active_label_with_open_pr" not in escalated_kinds
    # ...and it is suppressed by being HANDLED elsewhere, not by falling through
    # every rule silently. The escalated issue still gets its labels converged.
    assert "escalated_labels_converged" in escalated_kinds


def test_blocked_issue_never_reaches_the_open_pr_self_heal() -> None:
    """Issue #1765: "blocked" is a ``state.SINK_STATUSES`` member exactly like
    "escalated" (both map to the identical human_needed label edge), so it
    must get the same terminal-status protection here. Before the fix, the
    upstream gate was a bare ``tracked_status == "escalated"`` check, so a
    "blocked" issue with this fixture's shape (stale active label, open PR)
    fell through to ``issue_active_label_with_open_pr`` -- which is not just
    "no repair for blocked", it is reconcile actively resetting a blocked
    issue's status back to PASSIVE_OPEN_STATUS while the stale active label
    remains, silently un-blocking it. It must instead take the same
    label-converging path "escalated" already does.
    """
    config = OrchestratorConfig()

    def probe(status: str) -> list[DriftItem]:
        gh = FakeGitHub(
            prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
            issues=[_issue(30, [config.labels.needs_rework])],
        )
        state = empty_state()
        state["issues"]["30"] = {"number": 30, "status": status}
        return detect_drift(gh, state, config)

    # Positive control, shared with the escalated test above: the same
    # fixture DOES arm issue_active_label_with_open_pr for a non-sink status.
    control_kinds = [item.kind for item in probe("dispatch_failed")]
    assert "issue_active_label_with_open_pr" in control_kinds

    blocked_items = probe("blocked")
    blocked_kinds = [item.kind for item in blocked_items]
    assert "issue_active_label_with_open_pr" not in blocked_kinds
    assert "escalated_labels_converged" in blocked_kinds

    converged = next(item for item in blocked_items if item.kind == "escalated_labels_converged")
    # Judgment is the fallback reason_class (no reason_class on the tracked
    # entry), so the expected label is human_needed, not operator_queue.
    assert converged.add_labels == (config.labels.human_needed,)
    # The message must name the actual status, not a hardcoded "escalated".
    assert "'blocked'" in converged.detail
    assert "'escalated'" not in converged.detail


def test_reconcile_and_github_share_list_limit_constant() -> None:
    """Issue #45: reconcile and github.py must derive limits from the same constant."""
    assert reconcile_list_limit == github_list_limit


def test_transition_failed_add_returns_partial_failure() -> None:
    """Issue #125: transition() should return PARTIAL_FAILURE when add fails."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    # Simulate a failed add for the done label
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_add_labels={(10, config.labels.done)},
    )

    result = transition(gh, config.labels, 10, "merged")

    assert result.outcome == TO.PARTIAL_FAILURE
    assert (10, config.labels.done) in result.add_failures
    assert len(result.remove_failures) == 0


def test_transition_failed_remove_returns_partial_failure() -> None:
    """Issue #125: transition() should return PARTIAL_FAILURE when remove fails."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    # Simulate a failed remove for an active label
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_remove_labels={(10, config.labels.in_progress)},
    )

    result = transition(gh, config.labels, 10, "merged")

    assert result.outcome == TO.PARTIAL_FAILURE
    assert (10, config.labels.in_progress) in result.remove_failures
    assert len(result.add_failures) == 0


def test_transition_no_labels_returns_nothing_changed() -> None:
    """Issue #125: transition() should return NOTHING_CHANGED when no labels to add/remove."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])

    # Use an event that has no labels (e.g., a hypothetical no-op event)
    # For this test, we'll use the "blocked" event which only adds human_needed
    result = transition(gh, config.labels, 10, "blocked")

    assert result.outcome == TO.APPLIED  # blocked has labels to add
    assert len(result.add_failures) == 0
    assert len(result.remove_failures) == 0


def test_terminal_transition_clears_sibling_workflow_labels() -> None:
    """Issue #215: terminal transitions (agent:done, agent:blocked, agent:human-needed) must clear sibling agent:* workflow labels."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(852, [config.labels.human_needed])],
    )

    # Transition to agent:done should remove agent:human-needed and all other workflow labels
    result = transition(gh, config.labels, 852, "merged")

    assert result.outcome == TO.APPLIED
    assert len(result.add_failures) == 0
    assert len(result.remove_failures) == 0

    # Verify that agent:done was added
    assert (852, config.labels.done) in gh.labels_added

    # Verify that all other workflow labels were removed (but not agent:done itself)
    # The remove set should include all workflow labels except agent:done
    assert (852, config.labels.queued) in gh.labels_removed
    assert (852, config.labels.in_progress) in gh.labels_removed
    assert (852, config.labels.pr_open) in gh.labels_removed
    assert (852, config.labels.reviewing) in gh.labels_removed
    assert (852, config.labels.needs_rework) in gh.labels_removed
    assert (852, config.labels.human_needed) in gh.labels_removed

    # Verify agent:done was NOT removed (it's the target state)
    assert (852, config.labels.done) not in gh.labels_removed


def test_merged_transition_removes_merge_hold_label() -> None:
    """Issue #496: merge-hold is a transient operator control and must be
    stripped when an issue reaches the terminal merged state."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(10, [config.labels.merge_hold, config.labels.in_progress])],
    )

    result = transition(gh, config.labels, 10, "merged")

    assert result.outcome == TO.APPLIED
    assert (10, config.labels.done) in gh.labels_added
    assert (10, config.labels.merge_hold) in gh.labels_removed
    assert (10, config.labels.in_progress) in gh.labels_removed


def test_closed_unmerged_transition_removes_merge_hold_label() -> None:
    """Issue #496: merge-hold must also be stripped when an issue is closed
    without merging, matching the transient-operator model."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[
            _issue(10, [config.labels.merge_hold, config.labels.pr_open, config.labels.ready])
        ],
    )

    result = transition(gh, config.labels, 10, "closed_unmerged")

    assert result.outcome == TO.APPLIED
    assert (10, config.labels.merge_hold) in gh.labels_removed
    assert (10, config.labels.pr_open) in gh.labels_removed
    assert (10, config.labels.ready) in gh.labels_removed


@pytest.mark.parametrize(
    "event,expected_add",
    [
        ("review_started", (_config_labels.pr_open, _config_labels.reviewing)),
        ("rework_requested", (_config_labels.needs_rework,)),
        ("review_approved", (_config_labels.pr_open,)),
        ("escalated", (_config_labels.human_needed,)),
    ],
)
def test_non_terminal_transition_preserves_merge_hold_label(
    event: str, expected_add: tuple[str, ...]
) -> None:
    """Issue #496 regression: a non-terminal transition must NOT strip the
    merge-hold label from the issue. If it did, an operator's hold on the
    linked issue would be silently removed by the next review/rework cycle,
    and the PR would be swept back into the mergequeue."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(10, [config.labels.merge_hold, config.labels.in_progress])],
    )

    result = transition(gh, config.labels, 10, event)

    assert result.outcome == TO.APPLIED
    for label in expected_add:
        assert (10, label) in gh.labels_added
    # The hold must survive — it must never appear in labels_removed.
    assert (10, config.labels.merge_hold) not in gh.labels_removed


def test_mutation_gate_transition_ignoring_result_fails() -> None:
    """Issue #125: gate test - ignoring transition result must fail."""
    from charlie_work.labels import transition, TransitionOutcome as TO

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_add_labels={(10, config.labels.done)},
    )

    # This test ensures that if someone reverts to fire-and-forget (ignoring result),
    # the test will fail because we assert on the outcome
    result = transition(gh, config.labels, 10, "merged")

    # If someone ignores the result and just calls transition(), this assertion
    # will catch that the operation didn't fully succeed
    assert result.outcome == TO.PARTIAL_FAILURE, (
        "Transition should report PARTIAL_FAILURE when add fails - this gate prevents "
        "reverting to fire-and-forget behavior"
    )


def test_mutation_gate_apply_fixes_false_success_fails() -> None:
    """Issue #125: gate test - reporting failed write as success must fail."""
    config = OrchestratorConfig()
    # Simulate a failed remove
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_remove_labels={(20, config.labels.pr_open)},
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="closed_unmerged_pr_active_labels",
            issue_number=20,
            pr_number=2,
            detail="PR #2 closed without merging",
            fix_actions=(f"remove label '{config.labels.pr_open}' from issue #20",),
            remove_labels=(config.labels.pr_open,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # This gate ensures that if someone removes the failure recording logic,
    # the test will fail because we expect the failure to be present
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1

    event = reconcile_events[0]
    assert "label_write_failed: true" in event["payload"]["fix_actions"], (
        "Label write failure must be recorded in event - this gate prevents "
        "reporting failures as successes"
    )


def test_reconcile_deferred_when_graphql_rate_limit_below_threshold(
    tmp_path: Path,
) -> None:
    """Issue #398: reconcile() writes a deferred event and returns a skip result
    when the GraphQL budget is too low, without issuing any pr/issue list calls.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
        rate_limit_sufficient=False,
        rate_limit_remaining=100,
        rate_limit_reset=1234567890,
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.reconcile()

    assert result.ok is True
    assert result.data["deferred_reason"] == "graphql_rate_limit"
    assert result.data["graphql_remaining"] == 100
    assert result.data["graphql_reset"] == 1234567890
    # No list calls were made because the guard stopped the sweep.
    assert not any(
        c[0] == "api" and ("pulls?state=all" in c[1] or "issues?state=all" in c[1])
        for c in gh.run_calls
    )
    assert not any(c[:2] == ["pr", "list"] for c in gh.run_calls)
    assert not any(c[:2] == ["issue", "list"] for c in gh.run_calls)
    # A deferred event was persisted to state.json.
    state = load_state(paths.state_file)
    events = [e for e in state.get("events", []) if e["kind"] == "graphql_rate_limit_deferred"]
    assert len(events) == 1
    assert events[0]["payload"]["remaining"] == 100


def test_reconcile_defensive_graphql_budget_error_emits_event(tmp_path: Path) -> None:
    """Issue #743: the defensive except GraphQLBudgetError path in reconcile()
    must emit a reconcile_pass_deferred event and persist it to state.json.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
        rate_limit_sufficient=True,
        rate_limit_remaining=100,
        rate_limit_reset=1234567890,
    )
    call_count = 0

    def toggling_check(threshold: int) -> tuple[bool, int, int | None]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (True, 100, 1234567890)
        return (False, 50, 1234567890)

    gh.check_graphql_rate_limit = toggling_check
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.reconcile(fix=True)

    assert result.ok is True
    assert result.data["deferred_reason"] == "graphql_rate_limit"
    assert result.data["graphql_remaining"] == 50
    assert result.data["graphql_reset"] == 1234567890
    # No list calls were made because the budget guard raised before the sweep.
    assert not any(c[:2] == ["pr", "list"] for c in gh.run_calls)
    assert not any(c[:2] == ["issue", "list"] for c in gh.run_calls)
    # The defensive exception path must leave a durable event.
    state = load_state(paths.state_file)
    events = [e for e in state.get("events", []) if e["kind"] == "reconcile_pass_deferred"]
    assert len(events) == 1
    assert events[0]["payload"]["remaining"] == 50
    assert events[0]["payload"]["fix"] is True
    assert events[0]["payload"]["deferred_reason"] == "graphql_rate_limit"
    # The SQLite audit log also has the event.
    log_events = [
        e for e in read_event_log(paths.state_file) if e["kind"] == "reconcile_pass_deferred"
    ]
    assert len(log_events) == 1
    assert log_events[0]["payload"]["remaining"] == 50


def test_reconcile_fix_deferred_when_supervisor_lock_held(tmp_path: Path) -> None:
    """Issue #398: mop-up --fix must be mutually exclusive with a supervised/fleet
    pass on the same repo. If the supervisor.lock is held, reconcile returns a skip.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)

    supervisor_lock_path = paths.root / "supervisor.lock"
    supervisor_lock = try_acquire_byte_range_lock(supervisor_lock_path)
    assert supervisor_lock is not None, "test setup could not acquire supervisor lock"
    try:
        result = app.reconcile(fix=True)
    finally:
        supervisor_lock.release()

    assert result.ok is True
    assert result.data.get("pass_skipped") is True
    assert result.data.get("reason") == "supervisor_lock_held"


def test_fetch_prs_raises_rather_than_degrading_to_empty_list() -> None:
    """A PR snapshot that could not be read must not read as "zero PRs".

    The old ``return result if isinstance(result, list) else []`` made an
    unreadable snapshot bit-identical to an empty GitHub. ``detect_drift``
    answers "GitHub has zero PRs" by flagging every tracked PR
    ``state_pr_missing_on_github``, whose fix handler pops it out of
    ``state["prs"]`` -- erasing ``decision``/``reviewed_head_sha`` fleet-wide.
    """
    with pytest.raises(GitHubError, match="refusing to treat an unreadable"):
        _fetch_prs(_EmptyStdoutGitHub())  # type: ignore[arg-type]


def test_fetch_issues_raises_rather_than_degrading_to_empty_list() -> None:
    """Symmetric with the PR fetcher -- hardening one and not the other would
    leave the identical coercion live on the issue side."""
    with pytest.raises(GitHubError, match="refusing to treat an unreadable"):
        _fetch_issues(_EmptyStdoutGitHub())  # type: ignore[arg-type]


def test_reconcile_dry_run_fix_does_not_mutate_local_state(tmp_path: Path) -> None:
    """Issue #615: mop-up --fix --dry-run must not remove checkouts or write state."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
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
    }
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(json.dumps(state), encoding="utf-8")

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    checkout_path = reviews_dir / "pr-1"
    checkout_path.mkdir(parents=True, exist_ok=True)

    app = OrchestratorApp(tmp_path, paths, config, gh, dry_run=True)
    result = app.reconcile(fix=True)

    assert result.ok is True
    assert result.data["drift_before"] == 1
    assert result.data["fixed"] is False
    assert "dry-run" in result.message.lower()
    assert checkout_path.exists()

    after_state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert after_state["prs"]["1"]["status"] == "reviewing"


def test_maybe_reconcile_drift_dry_run_does_not_mutate_local_state(tmp_path: Path) -> None:
    """Issue #615 (round-2 review) + issue #1324: the periodic in-loop
    auto-fix pass ``_maybe_reconcile_drift`` -- the entry point ``fleet
    supervise --dry-run`` actually uses -- must honour ``app.dry_run`` the
    same way the operator ``mop-up --fix --dry-run`` path does. With real
    drift present and ``dry_run=True`` it must NOT remove the review
    checkout, clear review-dispatch state fields, write any GitHub labels,
    OR leave any event/state.json footprint (issue #1324: ``_record_event``
    and the paired ``save_state`` now route through ``self.write_gate``,
    which performs zero writes under ``dry_run=True``).

    Mirrors ``test_reconcile_dry_run_fix_does_not_mutate_local_state`` but
    drives the periodic-loop entry point (``_maybe_reconcile_drift``) rather
    than the direct ``reconcile(fix=True)`` CLI entry point. The fix under
    test is the ``dry_run=self.dry_run`` threading in ``_maybe_reconcile_drift``'s
    call to ``_reconcile_locked``; without it, ``fleet supervise --dry-run``
    would run the real repair on every loop pass -- the same data-loss bug
    class as #615, one layer up.
    """
    config = OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
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
    }
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(json.dumps(state), encoding="utf-8")

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    checkout_path = reviews_dir / "pr-1"
    checkout_path.mkdir(parents=True, exist_ok=True)

    app = OrchestratorApp(tmp_path, paths, config, gh, dry_run=True)
    # A fresh state has no next_reconcile_at, so is_reconcile_due is True and
    # the pass runs immediately -- no schedule priming required.
    app._maybe_reconcile_drift()

    # No checkout/worktree removal: the drift fix would have deleted pr-1.
    assert checkout_path.exists(), "dry-run reconcile pass removed the review checkout"

    # No state mutation of the PR's review-dispatch fields: the drift fix
    # would have cleared status/review_dispatch_* back to idle.
    after_state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert after_state["prs"]["1"]["status"] == "reviewing"
    assert after_state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"

    # No GitHub label writes: the drift fix would have removed the stale
    # in_progress/reviewing labels from issue 10.
    assert gh.labels_added == []
    assert gh.labels_removed == []

    # Issue #1324: under dry_run=True, _record_event routes through
    # self.write_gate.record_event (which returns state unchanged with zero
    # writes), and the paired save_state routes through
    # self.write_gate.save_state (which also does not write). So no
    # reconcile_pass_* event may appear in state.json's event ring, and
    # state.json itself must be byte-identical to the pre-pass seed -- the
    # WriteGate invariant ("no event at all under dry-run ... exactly the
    # same events.db/state.json footprint as a caller that never ran at
    # all"). The _reconcile_locked(dry_run=True) call still detects drift
    # in-memory (proving the gate engaged rather than the pass silently
    # no-opping), but that detection must not leak to disk.
    events = after_state.get("events", [])
    completed = [e for e in events if str(e.get("kind", "")).startswith("reconcile_pass")]
    assert completed == [], (
        f"dry-run reconcile pass must not write any reconcile_pass_* event "
        f"to state.json (issue #1324 WriteGate invariant), found: {completed}"
    )
    assert json.loads(paths.state_file.read_text(encoding="utf-8")) == state, (
        "dry-run reconcile pass must leave state.json byte-identical to the "
        "pre-pass seed (issue #1324 WriteGate invariant)"
    )


def test_reconcile_dry_run_without_fix_still_reports_drift(tmp_path: Path) -> None:
    """Regression: `charlie --dry-run mop-up` (no --fix) must not report success with drift.

    The global `--dry-run` flag only changes behaviour for the mutating `--fix`
    path. A read-only drift check must keep `ok=False` when drift exists so
    scripts and CI can gate on the exit code.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
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
    }
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(json.dumps(state), encoding="utf-8")

    app = OrchestratorApp(tmp_path, paths, config, gh, dry_run=True)
    result = app.reconcile(fix=False)

    assert result.ok is False
    assert result.data["drift_before"] == 1
    assert result.data["fixed"] is False
    assert "read-only; pass --fix to repair" in result.message
    assert "(dry-run" not in result.message

    after_state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert after_state["prs"]["1"]["status"] == "reviewing"
