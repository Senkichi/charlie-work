"""Tests for ``charlie experiment-report`` (issue #1701).

Covers the issue's acceptance criteria: PR-level unit of analysis (never
round-level inflation), arm values derived from the session-metrics key
with conflict detection, the date-window flags, activity/outcome metric
labelling, the documented stopping rule, read-only behaviour (byte-identical
``events.db``/``state.json``), and the docs that carry the stopping rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _cli_fixtures import _FakeGitHub, _make_repo
from charlie_work import cli, instrumentation
from charlie_work.experiment_report import (
    CONFIDENCE_Z,
    build_report,
    metrics_key_for_experiment,
    parse_window_bound,
    render_text,
    wilson_interval,
)

KEY = "review_effort_arm"


def _evt(ts: str, kind: str, payload: dict) -> dict:
    return {
        "ts": ts,
        "kind": kind,
        "payload": payload,
        "pr_number": payload.get("pr_number"),
        "issue_number": payload.get("issue_number"),
        "repo": None,
        "correlation_id": None,
        "level": "info",
    }


def _round(pr: int, decision: str, arm: str, *, ts: str, cost=None, issue: int = 1) -> dict:
    sm: dict = {KEY: arm}
    if cost is not None:
        sm["cost_usd"] = cost
    return _evt(
        ts,
        "record_review",
        {
            "pr_number": pr,
            "issue_number": issue,
            "decision": decision,
            "session_metrics": sm,
        },
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
    arm names must never appear in the implementation source."""
    import inspect

    import charlie_work.experiment_report as er
    import charlie_work.experiment_report_command as erc

    for module in (er, erc):
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
    """Arm a: 4/6 approvals bounced to rework. Arm b: 0/6. The outcome
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
                "check_failure_rework_requested",
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


def test_equivalence_bound_ends_inconclusive_experiment() -> None:
    """Every arm at >= 2x min with all outcome intervals spanning 0 ->
    met, no_detectable_difference."""
    events = []
    for i in range(12):
        events.append(_round(400 + i, "approved", "a", ts="2026-08-01T00:00:00Z"))
        events.append(_round(500 + i, "approved", "b", ts="2026-08-01T00:00:00Z"))
    report = build_report(events, KEY, min_prs_per_arm=6)
    rule = report["stopping_rule"]
    assert rule["met"] is True and rule["verdict"] == "no_detectable_difference"


# ---------------------------------------------------------------------------
# Activity vs outcome labelling, JSON shape, intervals
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Command layer: read-only, --help, missing db, json flag
# ---------------------------------------------------------------------------


def test_help_describes_read_only() -> None:
    text = cli.build_parser().format_help()
    assert "experiment-report" in text
    # the subcommand help line describes the read-only contract
    assert "Read-only" in text or "read-only" in text


def _state_path(repo: Path) -> Path:
    return repo / ".var" / "charlie-work" / "state.json"


def test_missing_events_db_fails_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only means: a missing events.db must not be created by the
    report (instrumentation._get_db would create+schema it on open)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    before = state_path.read_bytes()

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 1
    assert state_path.read_bytes() == before
    assert not (state_path.parent / "events.db").exists()


def test_report_performs_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """events.db and state.json are byte-identical before and after a run."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    instrumentation.log_event(
        state_path,
        "record_review",
        {
            "pr_number": 1,
            "issue_number": 1,
            "decision": "approved",
            "session_metrics": {KEY: "deep", "cost_usd": 1.25},
        },
    )
    db_bytes = (state_path.parent / "events.db").read_bytes()
    state_bytes = state_path.read_bytes()

    rc = cli.main(["--repo", str(repo), "experiment-report", "--experiment", "review_effort"])

    assert rc == 0
    out = capsys.readouterr().out
    assert 'arm "deep"' in out or '"deep"' in out
    assert "stopping rule" in out
    assert (state_path.parent / "events.db").read_bytes() == db_bytes
    assert state_path.read_bytes() == state_bytes


def test_json_flag_emits_structured_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    instrumentation.log_event(
        state_path,
        "record_review",
        {
            "pr_number": 1,
            "issue_number": 1,
            "decision": "request_changes",
            "session_metrics": {KEY: "deep"},
        },
    )

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "experiment-report",
            "--experiment",
            "review_effort",
            "--json",
        ]
    )

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True
    data = parsed["data"]
    assert data["metrics_key"] == KEY
    assert data["arms"] == ["deep"]
    assert data["metrics"]["first_round_request_changes_rate"]["per_arm"]["deep"]["rate"] == 1.0


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
    ):
        assert needle in text, f"docs missing required content: {needle!r}"
