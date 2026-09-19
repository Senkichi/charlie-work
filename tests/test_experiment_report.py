"""Tests for ``charlie experiment-report`` (issue #1701).

Covers the issue's acceptance criteria: PR-level unit of analysis (never
round-level inflation), arm values derived from the session-metrics key
with conflict detection, the date-window flags, activity/outcome metric
labelling, the documented stopping rule, state.json merge coverage, the
outcome channel's observed-event liveness, and the docs that carry the
stopping rule.  The CLI-driving tests live in
``test_experiment_report_cli.py`` (split under the 800-line cap, issue
#1442); shared fixtures in ``_experiment_report_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _experiment_report_fixtures import KEY, _evt, _round
from charlie_work.experiment_report import (
    CONFIDENCE_Z,
    EQUIVALENCE_MARGIN,
    _outcome_coverage,
    build_report,
    evaluate_stopping_rule,
    metrics_key_for_experiment,
    parse_window_bound,
    render_text,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Unit of analysis: the PR, not the round
# ---------------------------------------------------------------------------


def test_first_round_rate_is_per_pr_not_per_round() -> None:
    """1 PR with 5 request_changes rounds + 4 PRs with 1 approved round each
    must read 1/5 = 0.2 first-round request_changes -- not 5/9 = 0.556,
    which is what a round-level read-out would claim (the bug the issue
    calls out: every kickback spawns another round in the same arm)."""
    events = [
        *[
            _round(10, "request_changes", "deep", ts=f"2026-08-0{i}T00:00:00Z")
            for i in range(1, 6)
        ],
        *[_round(20 + i, "approved", "deep", ts="2026-08-10T00:00:00Z") for i in range(4)],
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    m = report["metrics"]["first_round_request_changes_rate"]["per_arm"]["deep"]
    assert m["n"] == 5 and m["k"] == 1 and m["rate"] == pytest.approx(0.2)
    approved = report["metrics"]["first_round_approved_rate"]["per_arm"]["deep"]
    assert approved["rate"] == pytest.approx(0.8)
    # rounds-per-PR still counts every round (mean 9/5 = 1.8)
    assert report["metrics"]["rounds_per_pr_mean"]["per_arm"]["deep"]["value"] == pytest.approx(
        1.8
    )


def test_arm_assignment_stable_across_rounds_and_conflicts_flagged() -> None:
    """A PR whose recorded arm values disagree is a data-integrity finding
    and is excluded from every arm -- never silently assigned."""
    events = [
        _round(30, "approved", "deep", ts="2026-08-01T00:00:00Z"),
        _round(30, "request_changes", "shallow", ts="2026-08-02T00:00:00Z"),
        _round(31, "approved", "deep", ts="2026-08-01T00:00:00Z"),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    assert report["per_arm"]["deep"]["prs_assigned"] == 1
    assert report["per_arm"]["deep"]["prs_with_rounds"] == 1
    warnings = report["data_integrity_warnings"]
    assert len(warnings) == 1 and warnings[0]["pr"] == 30
    assert sorted(warnings[0]["arms"]) == ["deep", "shallow"]


def test_arm_values_derived_from_claim_assignments_and_rounds() -> None:
    """The arm set is derived from recorded values -- claim-batch
    assignment lists and per-round session metrics both count, and no arm
    name appears anywhere in the implementation."""
    events = [
        _evt(
            "2026-08-01T00:00:00Z",
            "review_dispatch_claim",
            {
                "pr_numbers": [40, 41],
                "review_effort_assignments": [
                    {"pr_number": 40, KEY: "fast"},
                    {"pr_number": 41, KEY: "slow"},
                ],
            },
        ),
        _round(41, "approved", "slow", ts="2026-08-02T00:00:00Z"),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    assert sorted(report["arms"]) == ["fast", "slow"]
    # PR 40 counts as assigned via the claim even with no recorded round.
    assert report["per_arm"]["fast"]["prs_assigned"] == 1
    assert report["per_arm"]["fast"]["prs_with_rounds"] == 0


def test_no_literal_arm_name_in_implementation() -> None:
    """The report derives arm names from the data; the literal experiment
    arm names must never appear in the implementation source. The module
    set is derived (every ``experiment_report*`` module in the package),
    so a new split-out module is covered without this test being edited."""
    import inspect
    import pkgutil

    import charlie_work

    modules = [
        __import__(f"charlie_work.{m.name}", fromlist=["x"])
        for m in pkgutil.iter_modules(charlie_work.__path__)
        if m.name.startswith("experiment_report")
    ]
    assert any(m.__name__.endswith("experiment_report") for m in modules)
    for module in modules:
        source = inspect.getsource(module)
        for forbidden in ("treatment", "control"):
            assert forbidden not in source, (
                f"{forbidden!r} must not appear in {module.__name__} -- "
                "arm names are derived from the data, not hard-coded"
            )


# ---------------------------------------------------------------------------
# Metrics: cost, rounds, escalation, merge share, dispatch-to-merge
# ---------------------------------------------------------------------------


def test_cost_rounds_escalation_merge_and_latency() -> None:
    """Per-PR cost sums all rounds; escalation and merge are per-PR flags;
    dispatch-to-merge is a median over PRs with both timestamps."""
    events = [
        _evt("2026-08-01T00:00:00Z", "dispatch", {"issue_numbers": [1]}),
        _evt("2026-08-01T01:00:00Z", "review_dispatch", {"launched": [50], "failed": []}),
        _round(50, "request_changes", "x", ts="2026-08-02T00:00:00Z", cost=1.5),
        _round(50, "approved", "x", ts="2026-08-03T00:00:00Z", cost=2.0),
        _evt(
            "2026-08-03T12:00:00Z",
            "janitor_rework_escalated",
            {"pr_number": 51, "issue_number": 2},
        ),
        _round(51, "approved", "x", ts="2026-08-03T00:00:00Z", cost=0.5),
        _evt(
            "2026-08-04T00:00:00Z",
            "merge_succeeded",
            {"pr_number": 50, "issue_number": 1, "merged_at": "2026-08-04T00:00:00Z"},
        ),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    arm = report["metrics"]
    cost = arm["review_cost_per_pr_usd_mean"]["per_arm"]["x"]
    assert cost["n"] == 2 and cost["value"] == pytest.approx((3.5 + 0.5) / 2)
    assert cost["rounds_with_cost"] == 3 and cost["rounds_total"] == 3
    esc = arm["escalation_rate"]["per_arm"]["x"]
    assert esc["k"] == 1 and esc["n"] == 2 and esc["rate"] == pytest.approx(0.5)
    merged = arm["merge_rate"]["per_arm"]["x"]
    assert merged["k"] == 1 and merged["n"] == 2
    d2m = arm["dispatch_to_merge_seconds_median"]["per_arm"]["x"]
    # earliest dispatch marker for PR 50 is the issue dispatch at 00:00;
    # merge at 2026-08-04T00:00 -> 72h = 259200s
    assert d2m["n"] == 1 and d2m["value"] == pytest.approx(259200.0)


def test_state_merged_pr_without_merge_event_counts() -> None:
    """Regression (PR #1714 round-2 review): a PR that state.json marks
    merged but that has no merge event must still count in merge_rate and
    dispatch_to_merge.  On the live fleet most merges land via Aviator and
    emit no event, so the event vocabulary alone reported ~43% against
    state.json's ~96%."""
    events = [
        _evt("2026-08-01T00:00:00Z", "dispatch", {"issue_numbers": [1]}),
        _round(50, "approved", "x", ts="2026-08-02T00:00:00Z", issue=1),
        _round(51, "approved", "x", ts="2026-08-02T00:00:00Z", issue=2),
    ]
    state_prs = {
        50: {"status": "merged", "merged_at": "2026-08-04T00:00:00Z"},
        51: {"status": "open"},
    }
    report = build_report(events, KEY, state_prs=state_prs, min_prs_per_arm=1)
    m = report["metrics"]["merge_rate"]["per_arm"]["x"]
    assert m["k"] == 1 and m["n"] == 2
    d2m = report["metrics"]["dispatch_to_merge_seconds_median"]["per_arm"]["x"]
    # dispatch 08-01T00:00 -> state merged_at 08-04T00:00 = 72h
    assert d2m["n"] == 1 and d2m["value"] == pytest.approx(259200.0)
    cov = report["merge_coverage"]
    assert cov["state_prs"] == "available"
    c = cov["per_arm"]["x"]
    assert c["merged_total"] == 1
    assert c["state_merged"] == 1
    assert c["state_merged_event_observed"] == 0
    text = render_text(report)
    assert "merge events observed for 0 of 1 state-merged PRs" in text
    # without the state snapshot the same events show the old undercount,
    # and the coverage line says so explicitly instead of implying a rate
    event_only = build_report(events, KEY, min_prs_per_arm=1)
    assert event_only["metrics"]["merge_rate"]["per_arm"]["x"]["k"] == 0
    assert event_only["merge_coverage"]["state_prs"] == "unavailable"
    assert "merge events only" in render_text(event_only)


def test_event_merged_pr_counts_when_state_agrees_or_not() -> None:
    """An event-observed merge counts even when state.json disagrees (or
    the PR is absent): the merged set is a union, and merge_coverage
    splits state-confirmed from event-only merges."""
    events = [
        _round(52, "approved", "x", ts="2026-08-02T00:00:00Z"),
        _round(53, "approved", "x", ts="2026-08-02T00:00:00Z"),
        _evt(
            "2026-08-04T00:00:00Z",
            "merge_succeeded",
            {"pr_number": 52, "issue_number": 1},
        ),
    ]
    state_prs = {
        52: {"status": "merged", "merged_at": "2026-08-04T00:00:00Z"},
        53: {"status": "merged", "merged_at": "2026-08-05T00:00:00Z"},
    }
    report = build_report(events, KEY, state_prs=state_prs, min_prs_per_arm=1)
    c = report["merge_coverage"]["per_arm"]["x"]
    # 52 merged by both sources; 53 merged by state only (no event)
    assert c["merged_total"] == 2
    assert c["state_merged"] == 2
    assert c["state_merged_event_observed"] == 1
    assert report["metrics"]["merge_rate"]["per_arm"]["x"]["k"] == 2


# ---------------------------------------------------------------------------
# Window filtering
# ---------------------------------------------------------------------------


def test_exclude_window_removes_events_inside_and_keeps_boundary_outside() -> None:
    """An event inside the exclusion window contributes to no figure; an
    event one second outside it does."""
    start = parse_window_bound("2026-08-05T00:00:00Z")
    end = parse_window_bound("2026-08-06T00:00:00Z")
    events = [
        _round(60, "approved", "a", ts="2026-08-05T12:00:00Z"),  # inside -> dropped
        _round(61, "approved", "a", ts="2026-08-06T00:00:01Z"),  # 1s outside -> kept
    ]
    report = build_report(events, KEY, exclude_windows=[(start, end)], min_prs_per_arm=1)
    assert report["per_arm"]["a"]["prs_with_rounds"] == 1
    assert report["events_scanned"] == 1


def test_since_and_until_are_inclusive() -> None:
    events = [
        _round(70, "approved", "a", ts="2026-08-01T00:00:00Z"),
        _round(71, "approved", "a", ts="2026-08-02T00:00:00Z"),
        _round(72, "approved", "a", ts="2026-08-03T00:00:00Z"),
    ]
    report = build_report(
        events,
        KEY,
        since=parse_window_bound("2026-08-01T00:00:00Z"),
        until=parse_window_bound("2026-08-02T00:00:00Z"),
        min_prs_per_arm=1,
    )
    assert report["per_arm"]["a"]["prs_with_rounds"] == 2


# ---------------------------------------------------------------------------
# Empty / thin data and the stopping rule
# ---------------------------------------------------------------------------


def test_no_events_produces_report_that_says_so() -> None:
    report = build_report([], KEY, min_prs_per_arm=1)
    assert report["arms"] == []
    assert report["stopping_rule"]["met"] is False
    assert report["stopping_rule"]["verdict"] == "not_met"
    text = render_text(report)
    assert "NO DATA" in text


def test_single_arm_reports_zero_other_arm_without_crashing() -> None:
    """An arm with zero PRs in a figure must report its emptiness rather
    than divide by zero; the stopping rule reports not met."""
    events = [_round(80, "approved", "only", ts="2026-08-01T00:00:00Z")]
    report = build_report(events, KEY, min_prs_per_arm=1)
    assert report["arms"] == ["only"]
    # one arm -> no pairwise differences, stopping rule cannot be met
    assert report["metrics"]["merge_rate"]["differences"] == []
    assert report["stopping_rule"]["met"] is False
    # a zero-denominator rate renders n/a, never NaN/ZeroDivisionError
    no_kick = report["metrics"]["no_op_kickback_rate"]["per_arm"]["only"]
    assert no_kick["n"] == 0 and no_kick["rate"] is None
    text = render_text(report)
    assert "n/a" in text and "NOT MET" in text


def test_below_minimum_prs_reports_not_met() -> None:
    events = [_round(90 + i, "approved", "a", ts="2026-08-01T00:00:00Z") for i in range(3)] + [
        _round(100 + i, "approved", "b", ts="2026-08-01T00:00:00Z") for i in range(3)
    ]
    report = build_report(events, KEY, min_prs_per_arm=10)
    rule = report["stopping_rule"]
    assert rule["met"] is False and rule["verdict"] == "not_met"
    assert "minimum" in rule["detail"]


def test_stopping_rule_met_on_outcome_difference() -> None:
    """Arm a: 4/6 approvals bounced to rework by a silent base-commit
    revert (the review-correctness signal). Arm b: 0/6. The outcome
    difference interval excludes 0 -> met, difference_detected, naming the
    lower-defect arm."""
    events = []
    for i in range(6):
        pra, prb = 200 + i, 300 + i
        events.append(_round(pra, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(prb, "approved", "b", ts="2026-08-01T00:00:00Z"))
    for i in range(4):
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "cross_pr_revert_rework_requested",
                {"pr_number": 200 + i, "issue_number": 1},
            )
        )
    report = build_report(events, KEY, min_prs_per_arm=6)
    rule = report["stopping_rule"]
    assert rule["met"] is True and rule["verdict"] == "difference_detected"
    assert '"b"' in rule["detail"]
    outcome = report["metrics"]["post_approval_defect_rate"]
    assert outcome["measure"] == "outcome"
    assert outcome["per_arm"]["a"]["k"] == 4 and outcome["per_arm"]["a"]["n"] == 6
    assert outcome["per_arm"]["b"]["k"] == 0


def test_zero_outcome_events_cannot_end_experiment() -> None:
    """Regression: 200 approved PRs per arm and ZERO outcome events used
    to return met/no_detectable_difference off a degenerate [0, 0] Wald
    interval. Zero observed events cannot distinguish equivalent arms from
    a dead outcome channel, so the rule must stay not_met (underpowered)
    and the difference interval must be non-degenerate."""
    events = []
    for i in range(200):
        events.append(_round(400 + i, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(700 + i, "approved", "b", ts="2026-08-01T00:00:00Z"))
    report = build_report(events, KEY, min_prs_per_arm=100)
    rule = report["stopping_rule"]
    assert rule["met"] is False and rule["verdict"] == "not_met"
    assert "underpowered" in rule["detail"]
    diff = report["metrics"]["post_approval_defect_rate"]["differences"]
    assert len(diff) == 1
    lo, hi = diff[0]["ci95"]
    assert lo < 0.0 < hi  # a real interval, never the old [+0.0000, +0.0000]


def test_zero_outcome_events_below_bound_reports_cannot_conclude() -> None:
    """Regression (PR #1714 round-2 review): at min-N-reached but below the
    equivalence bound, an arm with zero observed outcome events must NOT
    get the 'not yet decisive -- keep running' message: zero events cannot
    distinguish equivalent arms from a dead outcome channel, so the rule
    says it cannot conclude until events are observed."""
    events = [
        *[_round(2000 + i, "approved", "a", ts="2026-08-01T00:00:00Z") for i in range(6)],
        *[_round(2100 + i, "approved", "b", ts="2026-08-01T00:00:00Z") for i in range(6)],
        # one observed outcome event in arm a -- proves the channel fires,
        # but arm b still has zero
        _evt(
            "2026-08-05T00:00:00Z",
            "cross_pr_revert_rework_requested",
            {"pr_number": 2000, "issue_number": 1},
        ),
    ]
    report = build_report(events, KEY, min_prs_per_arm=6)
    rule = report["stopping_rule"]
    assert rule["met"] is False and rule["verdict"] == "not_met"
    assert "zero outcome events" in rule["detail"]
    assert "cannot conclude" in rule["detail"]
    assert '"b"' in rule["detail"]  # names the starved arm
    assert "keep running" not in rule["detail"]
    assert rule["outcome_events_per_arm"] == {"a": 1, "b": 0}


def test_outcome_coverage_reports_observed_event_counts() -> None:
    """outcome_coverage must distinguish 'emitter exists' (derivable) from
    'events observed in window' -- raw REVIEW_DEFECT_KINDS event counts
    per arm, plus a total that includes events on unassigned PRs."""
    events = [
        _round(2200, "approved", "a", ts="2026-08-01T00:00:00Z"),
        _round(2300, "approved", "b", ts="2026-08-01T00:00:00Z"),
        _evt(
            "2026-08-05T00:00:00Z",
            "cross_pr_revert_rework_requested",
            {"pr_number": 2200, "issue_number": 1},
        ),
        _evt(
            "2026-08-06T00:00:00Z",
            "cross_pr_revert_rework_requested",
            {"pr_number": 2200, "issue_number": 1},
        ),
        # an outcome event on a PR outside the experiment: still counts in
        # the total (channel liveness), not in either arm
        _evt(
            "2026-08-07T00:00:00Z",
            "cross_pr_revert_rework_requested",
            {"pr_number": 9999, "issue_number": 1},
        ),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    cov = report["outcome_coverage"]
    assert cov["observed_outcome_events_per_arm"] == {"a": 2, "b": 0}
    assert cov["observed_outcome_events_total"] == 3
    assert cov["derivable"] == ["post_approval_defect_rate"]
    # the outcome metric carries the same counts for the stopping rule
    assert report["metrics"]["post_approval_defect_rate"]["observed_events_per_arm"] == {
        "a": 2,
        "b": 0,
    }
    text = render_text(report)
    assert "a=2" in text and "b=0" in text
    # events were observed somewhere -> no dead-channel warning
    assert "may be dead" not in text


def test_render_warns_when_emitter_exists_but_zero_events() -> None:
    """The derivable list alone used to imply a live outcome channel; with
    zero observed events the render must warn that the emitter exists but
    produced nothing -- emitter-exists is not events-observed."""
    events = [
        _round(2400, "approved", "a", ts="2026-08-01T00:00:00Z"),
        _round(2500, "approved", "b", ts="2026-08-01T00:00:00Z"),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    text = render_text(report)
    # the coverage-section warning (distinct from the rule's own detail)
    assert "zero outcome events were observed in this window" in text
    assert "may be dead" in text


def test_equivalence_bound_ends_experiment_with_evidence() -> None:
    """Positive equivalence case: every arm at >= 2x min, each arm with
    >= 1 observed outcome event, and every difference interval inside the
    stated equivalence margin -> met, no_detectable_difference."""
    events = []
    for i in range(200):
        events.append(_round(600 + i, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(800 + i, "approved", "b", ts="2026-08-01T00:00:00Z"))
    for i in range(5):
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "cross_pr_revert_rework_requested",
                {"pr_number": 600 + i, "issue_number": 1},
            )
        )
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "cross_pr_revert_rework_requested",
                {"pr_number": 800 + i, "issue_number": 2},
            )
        )
    report = build_report(events, KEY, min_prs_per_arm=100)
    rule = report["stopping_rule"]
    assert rule["met"] is True and rule["verdict"] == "no_detectable_difference"
    assert rule["equivalence_margin"] == EQUIVALENCE_MARGIN
    diff = report["metrics"]["post_approval_defect_rate"]["differences"][0]
    assert diff["ci95"][0] >= -EQUIVALENCE_MARGIN
    assert diff["ci95"][1] <= EQUIVALENCE_MARGIN


def test_intervals_wider_than_margin_keep_running_at_equivalence_bound() -> None:
    """Every arm >= 2x min with outcome events, but the difference
    interval is wider than the stated equivalence margin without excluding
    0 -- neither difference_detected nor equivalence: not_met."""
    events = []
    for i in range(200):
        events.append(_round(1000 + i, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(1200 + i, "approved", "b", ts="2026-08-01T00:00:00Z"))
    for i in range(8):
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "cross_pr_revert_rework_requested",
                {"pr_number": 1000 + i, "issue_number": 1},
            )
        )
    for i in range(2):
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "cross_pr_revert_rework_requested",
                {"pr_number": 1200 + i, "issue_number": 2},
            )
        )
    report = build_report(events, KEY, min_prs_per_arm=100)
    rule = report["stopping_rule"]
    assert rule["met"] is False and rule["verdict"] == "not_met"
    diff = report["metrics"]["post_approval_defect_rate"]["differences"][0]
    assert diff["ci95"][0] < 0 < diff["ci95"][1]  # not a detected difference
    assert diff["ci95"][1] > EQUIVALENCE_MARGIN  # and too wide for equivalence


# ---------------------------------------------------------------------------
# Activity vs outcome labelling, JSON shape, intervals
# ---------------------------------------------------------------------------


def test_pipeline_signals_are_not_outcome_metrics() -> None:
    """check_failure_rework_requested keeps the approval standing (its
    rework brief says 'do not re-litigate the review'), so it is pipeline
    state, not a review-correctness outcome: it must not count toward
    post_approval_defect_rate nor end the experiment. no_op_kickback_rate
    measures an empty rework CYCLE, not a wrong kickback -- activity too."""
    events = []
    for i in range(6):
        events.append(_round(1400 + i, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(1500 + i, "approved", "b", ts="2026-08-01T00:00:00Z"))
    # every approved PR in arm a later bounces on failing checks
    for i in range(6):
        events.append(
            _evt(
                "2026-08-05T00:00:00Z",
                "check_failure_rework_requested",
                {"pr_number": 1400 + i, "issue_number": 1},
            )
        )
    report = build_report(events, KEY, min_prs_per_arm=6)
    metrics = report["metrics"]
    # the check failures are reported -- as activity, not outcome
    ci = metrics["post_approval_ci_failure_rate"]
    assert ci["measure"] == "activity"
    assert ci["per_arm"]["a"]["k"] == 6 and ci["per_arm"]["a"]["n"] == 6
    # and they contribute nothing to the review-correctness outcome
    defect = metrics["post_approval_defect_rate"]
    assert defect["measure"] == "outcome"
    assert defect["per_arm"]["a"]["k"] == 0 and defect["per_arm"]["b"]["k"] == 0
    # so check failures alone can never end the experiment
    assert report["stopping_rule"]["met"] is False
    # the no-op kickback rate is likewise activity, and its description
    # does not claim the kickback was wrong
    noop = metrics["no_op_kickback_rate"]
    assert noop["measure"] == "activity"
    assert "no content change" in noop["description"]


def test_no_outcome_metric_reports_statement_and_not_met() -> None:
    """Where no outcome metric is derivable the report must say so
    explicitly and the stopping rule must stay not_met -- it cannot be
    ended on activity metrics alone."""
    coverage = _outcome_coverage({"m": {"measure": "activity"}})
    assert coverage["derivable"] == []
    assert coverage["statement"] is not None
    assert "no outcome metric is derivable" in coverage["statement"]
    rule = evaluate_stopping_rule(["a", "b"], {"a": 10, "b": 10}, None, 1, CONFIDENCE_Z)
    assert rule["met"] is False and rule["verdict"] == "not_met"
    assert rule["outcome_metric"] is None
    assert "no outcome metric" in rule["detail"]
    # the renderer prints the implementation's own statement on an empty
    # derivable set -- a minimal report exercises that path without the
    # test mutating a report to inject the text it then asserts on
    minimal = {
        "metrics_key": KEY,
        "window": {"since": None, "until": None, "exclude_windows": []},
        "events_scanned": 0,
        "arms": [],
        "per_arm": {},
        "metrics": {},
        "data_integrity_warnings": [],
        "prs_with_rounds_without_arm_value": 0,
        "outcome_coverage": coverage,
        "stopping_rule": rule,
    }
    assert "no outcome metric is derivable" in render_text(minimal)


def test_every_metric_labelled_and_at_least_one_outcome() -> None:
    events = [
        _round(600, "approved", "a", ts="2026-08-01T00:00:00Z"),
        _round(601, "request_changes", "b", ts="2026-08-01T00:00:00Z"),
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    measures = {m["measure"] for m in report["metrics"].values()}
    assert measures == {"activity", "outcome"}
    assert report["metrics"]["post_approval_defect_rate"]["measure"] == "outcome"
    text = render_text(report)
    assert "[activity]" in text and "[outcome]" in text
    # the report must not imply a conclusion from activity metrics alone:
    # the not-derivable outcome candidates are stated explicitly
    assert "NOT derivable" in text
    assert report["outcome_coverage"]["not_derivable"]


def test_rates_carry_wilson_intervals_and_pairs_carry_differences() -> None:
    events = [
        *[_round(700 + i, "request_changes", "a", ts="2026-08-01T00:00:00Z") for i in range(6)],
        *[_round(800 + i, "approved", "b", ts="2026-08-01T00:00:00Z") for i in range(6)],
    ]
    report = build_report(events, KEY, min_prs_per_arm=1)
    m = report["metrics"]["first_round_request_changes_rate"]
    for arm in ("a", "b"):
        entry = m["per_arm"][arm]
        assert entry["n"] == 6
        assert entry["ci95"] is not None
        lo, hi = entry["ci95"]
        # boundary rates can sit an fp-epsilon outside their own Wilson
        # bound; compare with tolerance rather than strict containment
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= entry["rate"] + 1e-9 and entry["rate"] <= hi + 1e-9
    diffs = m["differences"]
    assert len(diffs) == 1 and diffs[0]["pair"] == ["a", "b"]
    assert diffs[0]["diff"] == pytest.approx(1.0)
    assert diffs[0]["ci95"][0] > 0  # the gap is unambiguous


def test_wilson_interval_edge_cases() -> None:
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(0, 5, CONFIDENCE_Z)
    assert lo == 0.0 and hi < 0.5
    lo, hi = wilson_interval(5, 5, CONFIDENCE_Z)
    assert lo > 0.5 and hi == 1.0


def test_json_output_is_serializable_and_complete() -> None:
    events = [_round(900, "approved", "a", ts="2026-08-01T00:00:00Z")]
    report = build_report(events, KEY, min_prs_per_arm=1)
    blob = json.dumps(report)  # must not raise
    parsed = json.loads(blob)
    assert parsed["metrics_key"] == KEY
    assert parsed["metrics"]["prs_assigned"]["per_arm"]["a"]["n"] == 1


def test_experiment_key_derivation() -> None:
    assert metrics_key_for_experiment("review_effort") == "review_effort_arm"
    assert metrics_key_for_experiment("brief_arm") == "brief_arm"


# ---------------------------------------------------------------------------
# Docs: the stopping rule is written down next to the experiment's config
# ---------------------------------------------------------------------------


def test_stopping_rule_is_documented() -> None:
    doc = Path(__file__).parent.parent / "docs" / "review-effort-experiment.md"
    assert doc.exists(), "the experiment's config/stopping-rule doc is missing"
    text = doc.read_text(encoding="utf-8")
    for needle in (
        "effort_experiment_fraction",
        "effort_experiment_salt",
        "stopping rule",
        "0.0",
        "post_approval_defect_rate",
        "assigned PRs",
        "experiment-report",
        # the stopping rule's evidence guards must be written down too:
        # the stated equivalence margin and the observed-event floor
        "equivalence margin",
        "MIN_EQUIVALENCE_EVENTS_PER_ARM",
        # outcome claims must match what is measured, and the missing
        # post-merge attribution recording must be linked (issue #1717)
        "no content change",
        "#1717",
    ):
        assert needle in text, f"docs missing required content: {needle!r}"
