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

* the result (``ok``, ``message``, ``data`` -- including the refusal ``reason``);
* the issue status entry and the PR record in ``state.json``;
* the label edge chosen (the ``lifecycle_transition`` Event) and the labels
  that end up on the issue;
* every Event emitted by the call: kind, level and payload;
* the decision file (``review-decision.json``), its rounds archive and the
  rework dispatch note (the rework prompt itself is rendered from a per-path
  template and is only checked for existence);
* the verdict comment body (the channel it is posted on is its own field).

Normalised -- transport only: absolute paths, timestamps (``*_at``/``*_ts``/``*_since`` keys
become ``"<set>"``/``None``), the orchestrator comment marker ``record_review``
prepends to a PR comment. Everything else is compared as is.

``DECLARED_DIFFERENCES`` (in ``_verdict_parity_table.py``) is the list of differences already known and decided
(or deliberately left for later steps of the chain that merges the two entry
points). A difference is a single (path, value-pair) in one scenario; each row
matches only the paths and scenarios it names. A difference no row matches
fails the test, and a path a row names that no scenario ever exercises fails
too, so the list can neither hide a new drift nor go stale.
"""

from __future__ import annotations

from typing import Any

import pytest

from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import local_pr_dict

from _verdict_parity_driver import MISSING, Diff, Scenario, _diffs, run_both
from _verdict_parity_table import DECLARED_DIFFERENCES, SCENARIOS

_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


def _outcomes(scenario: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario.name not in _CACHE:
        _CACHE[scenario.name] = run_both(scenario)
    return _CACHE[scenario.name]


def _undeclared(diffs: list[Diff]) -> list[Diff]:
    return [d for d in diffs if not any(row.matches(d) for row in DECLARED_DIFFERENCES)]


def _ids(scenarios: tuple[Scenario, ...]) -> list[str]:
    return [scenario.name for scenario in scenarios]


def _scenario(name: str) -> Scenario:
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


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

# The issue's table, verbatim: (difference, reason). Exception: the `reviewed_body_sha256` row's
# reason differs by design (issue #3105 kept the difference and replaced "drift, tracked separately").
_ISSUE_ROWS = (
    (
        "No Rescue tier on the Local path",
        "stated in a comment in `record_local_review`; at the cap a local record escalates directly",
    ),
    ("No external-findings intake", "there are no PR comments to read"),
    ("No CI auto-reject", "there is no CI"),
    ("Verdict comment posted on the issue", "there is no PR to comment on"),
    (
        "Local-only decision fields (`human_call_reclassified_from`, `human_call_marker`, `local: True`)",
        "local bookkeeping",
    ),
    (
        "`reviewed_body_sha256` not recorded locally",
        (
            "the Local path has no PR body to hash (its packet `pr.json` and `local_pr_dict` "
            "carry no `body`; `LocalFileGitHub.pr_view` returns `{}` and `pr_list` returns "
            '`[]`), and every comparison of the stored hash requires `"body" in pr`, so a '
            "recorded hash could never be compared. `_body_rework_escape_warning` does "
            "execute for a Local path record in the mechanical de-escalation sweep, but with "
            "`pr == {}` it fails closed; `_route_to_rework` only copies the stored value"
        ),
    ),
    ("Local lifecycle transitions omit the record number", "drift, tracked separately"),
)


def test_scenario_table_covers_the_issue_list() -> None:
    names = {scenario.name for scenario in SCENARIOS}
    assert _REQUIRED_SCENARIOS <= names
    assert len(names) == len(SCENARIOS), "scenario names must be unique"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_scenario_does_what_it_names_on_both_entry_points(scenario: Scenario) -> None:
    """Both outcomes meet the scenario's expectations (and each side its own pins), so a
    scenario cannot pass by both paths failing for some other reason."""
    remote, local = _outcomes(scenario)
    for side, outcome, pins in (
        ("record_review", remote, scenario.expect + scenario.expect_remote),
        ("record_local_review", local, scenario.expect + scenario.expect_local),
    ):
        for path, expected in pins:
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


def test_every_path_a_declared_difference_names_is_exercised() -> None:
    """A named path nothing exempts is stale (or too narrow to ever fire)."""
    all_diffs = [
        diff for scenario in SCENARIOS for diff in _diffs(scenario.name, *_outcomes(scenario))
    ]
    for row in DECLARED_DIFFERENCES:
        hit = {row.path_hit(diff.path) for diff in all_diffs if row.matches(diff)}
        if not row.observable:
            assert not hit and not row.paths, f"{row.name} is declared unobservable here"
            continue
        assert set(row.paths) <= hit, f"{row.name}: never observed: {set(row.paths) - hit}"


def test_declared_differences_start_with_the_issues_eight_rows_verbatim() -> None:
    declared = [(row.name, row.reason) for row in DECLARED_DIFFERENCES]
    assert declared[: len(_ISSUE_ROWS)] == list(_ISSUE_ROWS)
    assert [row.reason for row in DECLARED_DIFFERENCES[len(_ISSUE_ROWS) :]] == ["undecided"] * (
        len(DECLARED_DIFFERENCES) - len(_ISSUE_ROWS)
    )


def test_local_path_has_no_body_to_hash_so_no_reviewed_body_sha256(tmp_path) -> None:
    """Pins the reason on the `reviewed_body_sha256` declared difference: if the Local path
    ever gains a PR body, this fails and the difference must be revisited (issue #3105)."""
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path)
    assert gh.pr_view(1) == {}
    assert gh.pr_list() == []
    assert "body" not in local_pr_dict({"number": 1, "branch": "b", "headRefOid": "0" * 40})
    local = _outcomes(_scenario("approve"))[1]
    assert local.get("decision_file.reviewed_body_sha256", MISSING) is MISSING
    assert local.get("pr_record.reviewed_body_sha256", MISSING) is MISSING


def test_patch_id_follows_the_pinned_head_not_the_packet() -> None:
    """With a stale packet, pinning the live head takes the live diff's patch-id and pinning
    the packet head takes the packet's: the patch-id source is discriminated."""
    key = "decision_file.reviewed_patch_id"
    live = _outcomes(_scenario("no_packet_pins_live_head"))
    pinned_live = _outcomes(_scenario("stale_head_operator_pins_live"))
    pinned_packet = _outcomes(_scenario("stale_head_operator_pins_packet"))
    for side in (0, 1):
        assert live[side][key] and live[side][key] == pinned_live[side][key]
        assert pinned_packet[side][key] != live[side][key]


@pytest.mark.parametrize(
    ("scenario_name", "path", "changed"),
    [
        ("approve", "issue.status", "rework_requested"),
        ("approve", "pr_record.status", "blocked"),
        ("approve", "labels.added", ("agent:needs-rework",)),
        ("approve", "label_edge", "[]"),
        ("approve", "events.order", ("record_review",)),
        ("approve", "decision_file.reviewed_head_sha", "0" * 40),
        ("approve", "result.message", "review recorded"),
        ("rework_at_cap_new_head", "result.data.request_changes_count", 3),
        ("rework_at_cap_new_head", "decision_file.escalated", False),
        ("rework_at_cap_same_head", "result.data.escalated", True),
        ("rework_at_cap_same_head", "rework_dispatch_note", "empty"),
        ("rework_log_extracted_verdict", "decision_file.provenance_caveat", None),
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
    """Rescue-tier values are exempt in the rescue scenario only, and only as the
    exact values pinned there."""
    scenario = "rework_at_cap_new_head_rescue_enabled"
    remote, local = _outcomes(_scenario(scenario))
    assert _undeclared(_diffs(scenario, remote, local)) == []
    perturbed = {**local, path: "different"}
    assert [d.path for d in _undeclared(_diffs(scenario, remote, perturbed))] == [path]
    remote, local = _outcomes(_scenario("approve"))
    assert [d.path for d in _undeclared(_diffs("approve", remote, {**local, path: "x"}))] == [path]
