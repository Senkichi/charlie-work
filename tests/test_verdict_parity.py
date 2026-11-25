"""Scenario table through both Verdict entry points (issue #3101).

Verdict handling exists twice: ``record_review`` (``orchestration/
state_record_review.py``, the PR path) and ``record_local_review``
(``orchestration/local_lanes.py``, the Local path). The second was copied from
the first and the two have drifted; nothing asserted they agree. This file is
the safety net for merging them: every scenario below runs through BOTH entry
points with the existing fakes (``FakeGitHub`` for the PR path, the real
``LocalFileGitHub`` over a real scratch git repo for the Local path) and the
two outcomes are compared.

Compared, field by field (``_collect`` in ``_verdict_parity_driver.py``):

* the result (``ok``, ``data`` -- including the refusal ``reason``);
* the issue status entry and the PR record in ``state.json``;
* the label edge chosen (the ``lifecycle_transition`` Event) and the labels
  that end up on the issue;
* every Event emitted by the call: kind, level and payload;
* the decision file (``review-decision.json``) and its rounds archive;
* the verdict comment body (the channel it is posted on is its own field).

Normalised -- transport only: absolute paths, timestamps (``*_at``/``*_ts``/``*_since`` keys
become ``"<set>"``/``None``), the orchestrator comment marker ``record_review``
prepends to a PR comment, and the human-readable ``message`` prose. Everything
else is compared as is.

``DECLARED_DIFFERENCES`` is the list of differences already known and decided
(or deliberately left for later steps of the chain that merges the two entry
points). A difference is a single (path, value-pair) in one scenario; each row
matches only the paths and scenarios it names. A difference no row matches
fails the test, and a row that no scenario ever exercises fails too, so the
list can neither hide a new drift nor go stale.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY

from _verdict_parity_driver import (
    MISSING,
    Diff,
    Scenario,
    _diffs,
    run_both,
)

HUMAN_CALL_FINDING = "This needs a human decision before it can ship."


def _refused(reason: str) -> tuple[tuple[str, Any], ...]:
    return (("result.ok", False), ("result.data.reason", reason), ("events.order", ()))


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "approve",
        "approved",
        summary="Looks good.",
        expect=(
            ("result.ok", True),
            ("issue.status", "approved"),
            ("pr_record.status", "approved"),
            ("labels.added", ("agent:pr-open",)),
            ("events.order", ("record_review", "lifecycle_transition")),
        ),
    ),
    Scenario(
        "approve_no_summary_no_comment",
        "approved",
        summary="",
        post_verdict_comment=False,
        expect=(("result.ok", True), ("verdict_comment", None)),
    ),
    # ``max_rework_cycles`` is 2: a count of 1 is under the cap, 2 is at it.
    Scenario(
        "rework_under_cap",
        "request_changes",
        prior_count=1,
        prior_head="older",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 2),
            ("issue.status", "rework_requested"),
            ("labels.added", ("agent:needs-rework",)),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_first_verdict",
        "request_changes",
        expect=(
            ("result.data.request_changes_count", 1),
            ("issue.status", "rework_requested"),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_at_cap_new_head",
        "request_changes",
        prior_count=2,
        prior_head="older",
        expect=(
            ("result.data.escalated", True),
            ("result.data.request_changes_count", 2),
            ("issue.status", "escalated"),
            ("pr_record.status", "escalated"),
            ("pr_record.escalation_reason", "max_rework_cycles_exceeded"),
            ("labels.added", ("agent:operator-queue",)),
            ("rework_prompt", False),
            ("verdict_comment", None),
        ),
    ),
    Scenario(
        "rework_at_cap_same_head",
        "request_changes",
        prior_count=2,
        prior_head="same",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 2),
            ("issue.status", "rework_requested"),
            ("labels.added", ("agent:needs-rework",)),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_under_cap_same_head",
        "request_changes",
        prior_count=1,
        prior_head="same",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 1),
            ("issue.status", "rework_requested"),
        ),
    ),
    # The two paths part ways here by design (declared: no Rescue tier locally).
    Scenario(
        "rework_at_cap_new_head_rescue_enabled",
        "request_changes",
        prior_count=2,
        prior_head="older",
        rescue_enabled=True,
    ),
    Scenario(
        "blocked",
        "blocked",
        summary="Needs a product call on the API shape.",
        expect=(
            ("issue.status", "blocked"),
            ("pr_record.status", "blocked"),
            ("pr_record.escalation_reason", "review_blocked"),
            ("labels.added", ("agent:human-needed",)),
        ),
    ),
    Scenario(
        "human_call_reclassified",
        "request_changes",
        summary="One finding for the operator.",
        required_changes=(HUMAN_CALL_FINDING,),
        expect=(
            ("result.data.decision", "blocked"),
            ("result.data.reclassified_from", "request_changes"),
            ("issue.status", "blocked"),
            ("labels.added", ("agent:human-needed",)),
            ("rework_prompt", False),
            (
                "events.order",
                ("record_review", "review_decision_reclassified_blocked", "lifecycle_transition"),
            ),
        ),
    ),
    Scenario(
        "rework_structured_required_changes",
        "request_changes",
        summary="Two things to fix.",
        required_changes=("Add a test.", "Rename the helper."),
        expect=(("decision_file.required_changes", ["Add a test.", "Rename the helper."]),),
    ),
    Scenario(
        "rework_vacuous_summary",
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        expect=(("decision_file.findings_channel", "vacuous"),),
    ),
    Scenario(
        "rework_forced_comment",
        "request_changes",
        comment=True,
        post_verdict_comment=False,
        expect=(("result.ok", True),),
    ),
    Scenario(
        "rework_issue_closed",
        "request_changes",
        issue_closed=True,
        expect=(
            ("issue.status", "rework_requested"),
            ("labels.added", ()),
            (
                "events.order",
                (
                    "worker_test_selection_unavailable",
                    "record_review",
                    "rework_label_skipped_issue_closed",
                ),
            ),
        ),
    ),
    Scenario(
        "rework_ci_gate_provenance",
        "request_changes",
        provenance="ci_gate_auto_reject",
        expect=(
            ("issue.no_op_checkpoint_at", MISSING),
            ("decision_file.verdict_provenance", "ci_gate_auto_reject"),
        ),
    ),
    Scenario("rework_external_findings", "request_changes", external_comment=True),
    Scenario(
        "stale_head_packet_behind_live",
        "approved",
        packet="stale",
        expect=_refused("head_moved_during_build"),
    ),
    Scenario(
        "stale_head_reviewed_head_is_packet",
        "approved",
        packet="stale",
        reviewed_head="packet",
        expect=_refused("head_moved_during_build"),
    ),
    Scenario(
        "stale_head_operator_pins_packet",
        "approved",
        packet="stale",
        reviewed_head="packet",
        allow_stale_head=True,
        expect=(("result.ok", True), ("result.data.reviewed_head_source", "packet")),
    ),
    Scenario(
        "stale_head_reviewed_head_matches_nothing",
        "approved",
        reviewed_head="bogus",
        expect=_refused("reviewed_head_mismatch"),
    ),
    Scenario(
        "no_packet_pins_live_head",
        "approved",
        packet="none",
        expect=(("result.ok", True), ("result.data.reviewed_head_source", "live")),
    ),
    Scenario(
        "already_escalated_record",
        "approved",
        pr_status="escalated",
        expect=_refused("pr_escalated"),
    ),
    Scenario(
        "already_escalated_issue",
        "request_changes",
        issue_status="escalated",
        expect=_refused("pr_escalated"),
    ),
    Scenario(
        "already_merged",
        "request_changes",
        pr_status="merged",
        expect=_refused("pr_terminal_state"),
    ),
    Scenario(
        "already_closed",
        "approved",
        pr_status="closed",
        expect=_refused("pr_terminal_state"),
    ),
    Scenario("invalid_decision", "maybe", expect=_refused("invalid_decision")),
    Scenario(
        "invalid_provenance",
        "approved",
        provenance="made_up",
        expect=_refused("invalid_verdict_provenance"),
    ),
    Scenario("missing_summary", "blocked", summary="  ", expect=_refused("missing_summary")),
)


# ---------------------------------------------------------------------------
# Declared differences
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredDifference:
    """One known difference. ``matches`` is narrow: named paths in named scenarios."""

    name: str
    reason: str
    matches: Callable[[Diff], bool]
    # False only for a difference these two entry points cannot show on their own.
    observable: bool = True


_RESCUE_SCENARIO = frozenset({"rework_at_cap_new_head_rescue_enabled"})
_EXTERNAL_SCENARIO = frozenset({"rework_external_findings"})
_COMMENT_RC_HEADER = "### Required changes"


_RESCUE_PATHS = frozenset(
    {
        "decision_file.escalated",
        "events.order",
        "events.record_review#0.payload.escalated",
        "events.rescue_dispatched#0.level",
        "events.rescue_dispatched#0.payload.cause",
        "events.rescue_dispatched#0.payload.issue_number",
        "events.rescue_dispatched#0.payload.pr_number",
        "issue.escalation_reason",
        "issue.escalation_reasons_seen",
        "issue.no_op_checkpoint_at",
        "issue.reason_class",
        "issue.status",
        "issue.terminal_since",
        "label_edge",
        "labels.added",
        "labels.final",
        "pr_record.escalation_reason",
        "pr_record.request_changes_count",
        "pr_record.rescue_attempted",
        "pr_record.rescue_cause",
        "pr_record.rescue_dispatched_at",
        "pr_record.status",
        "result.data.escalated",
        "result.data.request_changes_count",
        "result.data.rework_path",
        "rework_prompt",
        "rounds",
        "verdict_comment",
        "verdict_comment_channel",
    }
)
_RESCUE_PREFIXES = ("events.worker_test_selection_unavailable#0.",)


def _rescue(diff: Diff) -> bool:
    """With the Rescue tier enabled, at the cap on a new head, the PR path routes a
    rescue rework and the Local path escalates. Only the fields that divergence
    moves, only in that scenario; plus the result key the Local path never sets."""
    if diff.path == "result.data.rescue_dispatched":
        return isinstance(diff.remote, bool) and diff.local is MISSING
    if diff.scenario not in _RESCUE_SCENARIO:
        return False
    return (
        diff.path in _RESCUE_PATHS
        or diff.path.startswith(_RESCUE_PREFIXES)
        or re.fullmatch(r"events\.lifecycle_transition#0\.payload\.(cause|to_state)", diff.path)
        is not None
    )


def _verdict_comment_on_issue(diff: Diff) -> bool:
    return diff.path == "verdict_comment_channel" and diff.remote == "pr" and diff.local == "issue"


def _local_comment_repeats_summary(diff: Diff) -> bool:
    """The PR path leaves the Required changes section out when the list is the
    derived copy of the summary; the Local path appends it. Matches only a Local
    body that is the PR body plus exactly that one section, built from the
    scenario's own summary, in a scenario with no structured required changes."""
    if diff.path != "verdict_comment" or not isinstance(diff.remote, str):
        return False
    scenario = next(sc for sc in SCENARIOS if sc.name == diff.scenario)
    if scenario.required_changes:
        return False
    repeated = f"{diff.remote}\n\n### Required changes\n- {scenario.summary.strip()}"
    return diff.local == repeated


DECLARED_DIFFERENCES: tuple[DeclaredDifference, ...] = (
    DeclaredDifference(
        "No Rescue tier on the Local path",
        "stated in a comment in record_local_review; at the cap a local record escalates directly",
        _rescue,
    ),
    DeclaredDifference(
        "No external-findings intake",
        "there are no PR comments to read",
        lambda diff: (
            diff.path
            in {
                "decision_file.external_findings",
                "decision_file.before",
                "events.record_review#0.payload.external_findings_count",
            }
            and diff.local is MISSING
        ),
    ),
    DeclaredDifference(
        "No CI auto-reject",
        "there is no CI; the ci_gate_auto_reject provenance reaches both entry points "
        "identically, so there is nothing for these two functions to differ on",
        lambda diff: False,
        observable=False,
    ),
    DeclaredDifference(
        "Verdict comment posted on the issue",
        "there is no PR to comment on",
        _verdict_comment_on_issue,
    ),
    DeclaredDifference(
        "Local-only decision fields (human_call_reclassified_from, human_call_marker, local: True)",
        "local bookkeeping",
        lambda diff: (
            (
                diff.path in {"decision_file.human_call_reclassified_from", "decision_file.local"}
                or diff.path.startswith("decision_file.human_call_marker.")
            )
            and diff.remote is MISSING
        ),
    ),
    DeclaredDifference(
        "Local verdict comment repeats the summary",
        "drift, tracked separately",
        _local_comment_repeats_summary,
    ),
    DeclaredDifference(
        "reviewed_body_sha256 not recorded locally",
        "drift, tracked separately",
        lambda diff: (
            diff.path in {"decision_file.reviewed_body_sha256", "pr_record.reviewed_body_sha256"}
            and diff.local is MISSING
        ),
    ),
    DeclaredDifference(
        "Local lifecycle transitions omit the record number",
        "drift, tracked separately",
        lambda diff: (
            re.fullmatch(r"events\.lifecycle_transition#\d+\.payload\.pr_number", diff.path)
            is not None
            and diff.local in (MISSING, None)
        ),
    ),
    # Differences the table surfaced that the issue's list does not name. Each is
    # called out in the PR description; none is decided.
    DeclaredDifference(
        "Local result data carries the issue and local keys",
        "undecided",
        lambda diff: (
            diff.path in {"result.data.issue", "result.data.local"} and diff.remote is MISSING
        ),
    ),
    DeclaredDifference(
        "Terminal-state refusal names a different data key",
        "undecided",
        lambda diff: (
            diff.scenario in {"already_merged", "already_closed"}
            and (
                (diff.path == "result.data.terminal" and diff.remote is MISSING)
                or (diff.path == "result.data.terminal_state" and diff.local is MISSING)
            )
        ),
    ),
    DeclaredDifference(
        "record_review Event payload carries local: True on the Local path",
        "undecided",
        lambda diff: (
            re.fullmatch(r"events\.record_review#\d+\.payload\.local", diff.path) is not None
            and diff.remote is MISSING
        ),
    ),
)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


def _outcomes(scenario: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario.name not in _CACHE:
        _CACHE[scenario.name] = run_both(scenario)
    return _CACHE[scenario.name]


def _undeclared(diffs: list[Diff]) -> list[Diff]:
    return [d for d in diffs if not any(row.matches(d) for row in DECLARED_DIFFERENCES)]


def _ids(scenarios: tuple[Scenario, ...]) -> list[str]:
    return [scenario.name for scenario in scenarios]


# The issue's list, at minimum. Names below map to SCENARIOS entries.
_REQUIRED_SCENARIOS = {
    "approve",
    "rework_under_cap",
    "rework_at_cap_new_head",
    "rework_at_cap_same_head",
    "blocked",
    "human_call_reclassified",
    "stale_head_packet_behind_live",
    "already_escalated_record",
    "already_merged",
    "already_closed",
}


def test_scenario_table_covers_the_issue_list() -> None:
    names = {scenario.name for scenario in SCENARIOS}
    assert _REQUIRED_SCENARIOS <= names
    assert len(names) == len(SCENARIOS), "scenario names must be unique"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_scenario_does_what_it_names_on_both_entry_points(scenario: Scenario) -> None:
    """Both outcomes meet the scenario's expectations, so a scenario cannot pass
    by both paths failing for some other reason."""
    remote, local = _outcomes(scenario)
    for side, outcome in (("record_review", remote), ("record_local_review", local)):
        for path, expected in scenario.expect:
            assert outcome.get(path, MISSING) == expected, (
                f"{side}: [{scenario.name}] {path}: {outcome.get(path, MISSING)!r} != {expected!r}"
            )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_both_entry_points_agree_except_where_declared(scenario: Scenario) -> None:
    remote, local = _outcomes(scenario)
    undeclared = _undeclared(_diffs(scenario.name, remote, local))
    assert not undeclared, (
        "record_review and record_local_review disagree on something DECLARED_DIFFERENCES "
        "does not cover. Fix the drift, or add a row (reason 'undecided' if not yet "
        "decided):\n" + "\n".join(f"  {diff}" for diff in undeclared)
    )


def test_every_declared_difference_is_exercised_by_some_scenario() -> None:
    """A row nothing matches is stale (or too narrow to ever fire); delete or fix it."""
    all_diffs = [
        diff for scenario in SCENARIOS for diff in _diffs(scenario.name, *_outcomes(scenario))
    ]
    for row in DECLARED_DIFFERENCES:
        hits = [diff for diff in all_diffs if row.matches(diff)]
        if row.observable:
            assert hits, f"declared difference never observed: {row.name}"
        else:
            assert not hits, f"{row.name} is declared unobservable here but matched {hits[0]}"


def test_declared_differences_are_seeded_from_the_issue() -> None:
    names = [row.name for row in DECLARED_DIFFERENCES]
    for seeded in (
        "No Rescue tier on the Local path",
        "No external-findings intake",
        "No CI auto-reject",
        "Verdict comment posted on the issue",
        "Local-only decision fields (human_call_reclassified_from, human_call_marker, local: True)",
        "Local verdict comment repeats the summary",
        "reviewed_body_sha256 not recorded locally",
        "Local lifecycle transitions omit the record number",
    ):
        assert seeded in names
    assert all(row.reason for row in DECLARED_DIFFERENCES)


def _scenario(name: str) -> Scenario:
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


@pytest.mark.parametrize(
    ("scenario_name", "path", "changed"),
    [
        ("approve", "issue.status", "rework_requested"),
        ("approve", "pr_record.status", "blocked"),
        ("approve", "labels.added", ("agent:needs-rework",)),
        ("approve", "label_edge", "[]"),
        ("approve", "events.order", ("record_review",)),
        ("approve", "decision_file.reviewed_head_sha", "0" * 40),
        ("rework_at_cap_new_head", "result.data.request_changes_count", 3),
        ("rework_at_cap_new_head", "decision_file.escalated", False),
        ("rework_at_cap_same_head", "result.data.escalated", True),
        ("human_call_reclassified", "decision_file.decision", "request_changes"),
        ("already_merged", "result.data.reason", "pr_escalated"),
    ],
)
def test_a_change_on_one_side_only_is_reported(
    scenario_name: str, path: str, changed: Any
) -> None:
    """The comparison can fail: perturb one field of the Local outcome and it shows
    up as an undeclared difference (the normalisation hides nothing it checks)."""
    remote, local = _outcomes(_scenario(scenario_name))
    assert _undeclared(_diffs(scenario_name, remote, local)) == []
    assert local.get(path, MISSING) != changed
    perturbed = {**local, path: changed}
    assert [d.path for d in _undeclared(_diffs(scenario_name, remote, perturbed))] == [path]


@pytest.mark.parametrize("path", ["issue.status", "labels.added", "events.order"])
def test_the_rescue_exemption_is_limited_to_its_scenario(path: str) -> None:
    """Rescue-tier fields are exempt in the rescue scenario only."""
    remote, local = _outcomes(_scenario("approve"))
    perturbed = {**local, path: "different"}
    assert [d.path for d in _undeclared(_diffs("approve", remote, perturbed))] == [path]
