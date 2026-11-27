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

import ast
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import _git_templates
from _dashboard_rollup_fixtures import _register
from _src_ast import parsed
from charlie_work import instrumentation
from charlie_work.dashboard import rollup
from charlie_work.dashboard.drill.timeline import pr_drill
from charlie_work.dashboard.metrics_flow import EXACT_KINDS
from charlie_work.local_issue_files import issue_number_from_name
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import local_pr_dict

from _verdict_parity_driver import (
    MISSING,
    Diff,
    Scenario,
    _diffs,
    _fresh_repo,
    _repo_template,
    _Repo,
    run_both,
    warm_repo_template,
)
from _verdict_parity_table import DECLARED_DIFFERENCES, SCENARIOS

_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


@pytest.fixture(scope="module", autouse=True)
def _warm_repo_template() -> None:
    """Charge the per-process repo template build to setup, not a test's call.

    ``run_both`` takes ~0.5 s when it is a worker's first ``_fresh_repo`` caller
    and ~0.2 s after; under ``--dist=load`` and shard splits any test can be
    that caller, and the test ledger samples call time only (issue #3188).
    """
    warm_repo_template()


def _outcomes(scenario: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario.name not in _CACHE:
        _CACHE[scenario.name] = run_both(scenario)
    return _CACHE[scenario.name]


def _undeclared(diffs: list[Diff]) -> list[Diff]:
    return [d for d in diffs if not any(row.matches(d) for row in DECLARED_DIFFERENCES)]


def _ids(scenarios: tuple[Scenario, ...]) -> list[str]:
    return [scenario.name for scenario in scenarios]


_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}

# The scenario names each non-parametrized consumer drives. ``parity_outcomes``
# reads parametrized consumers (``scenario`` / ``scenario_name``) from the
# callspec; a consumer that names its scenarios only in its body needs them
# here or its ``parity_outcomes[...]`` lookup reports a KeyError.
_SCENARIO_DRIVE: dict[str, tuple[str, ...]] = {
    "test_every_path_a_declared_difference_names_is_exercised": tuple(_BY_NAME),
    "test_patch_id_follows_the_pinned_head_not_the_packet": (
        "no_packet_pins_live_head",
        "stale_head_operator_pins_live",
        "stale_head_operator_pins_packet",
    ),
    "test_the_rescue_exemption_is_limited_to_its_scenario": (
        "rework_at_cap_new_head_rescue_enabled",
        "approve",
    ),
    "test_local_path_has_no_body_to_hash_so_no_reviewed_body_sha256": ("approve",),
    "test_local_lifecycle_transition_omits_the_record_number_so_no_pr_drill_collision": (
        "approve",
    ),
}


@pytest.fixture
def parity_outcomes(
    request: pytest.FixtureRequest,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """``run_both`` outcomes keyed by scenario name, computed in fixture setup.

    ``run_both`` drives both Verdict entry points over a real scratch git repo
    (~0.5 s per scenario on a ``_CACHE`` miss): input acquisition, not the
    assertion under test. Paid inside ``call``, the cost lands on whichever
    test a worker's per-process ``_CACHE`` misses the scenario for -- xdist
    placement, not this repo's code -- which the test ledger reads as a
    per-test regression (issue #3187). Charging the drive to setup makes every
    consumer's ``call`` phase assertion-only; the cost itself stays visible in
    the ledger's slow-fixtures section.
    """
    callspec = getattr(request.node, "callspec", None)
    params = getattr(callspec, "params", {})
    names: list[str] = []
    scenario = params.get("scenario")
    if isinstance(scenario, Scenario):
        names.append(scenario.name)
    scenario_name = params.get("scenario_name")
    if scenario_name in _BY_NAME:
        names.append(scenario_name)
    names += _SCENARIO_DRIVE.get(request.node.name.split("[")[0], ())
    return {name: _outcomes(_BY_NAME[name]) for name in dict.fromkeys(names)}


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
# reason differs by design (issue #3105 kept the difference and replaced "drift, tracked separately"),
# and so does the "Local lifecycle transitions omit the record number" row (issue #3106, same).
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
    (
        "Local lifecycle transitions omit the record number",
        (
            "A Local path record's number is its local issue file's number, not a GitHub PR "
            "number, and the dashboard links milestones to an item by (source, pr) alone. "
            "`lifecycle_transition` rows are the exact Stage time source, so in an events.db "
            "that also holds PR path history a same-numbered PR's drill-down would absorb the "
            "Local issue's stages; the rows are already complete keyed by issue number."
        ),
    ),
)


def test_scenario_table_covers_the_issue_list() -> None:
    names = {scenario.name for scenario in SCENARIOS}
    assert _REQUIRED_SCENARIOS <= names
    assert len(names) == len(SCENARIOS), "scenario names must be unique"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_scenario_does_what_it_names_on_both_entry_points(
    scenario: Scenario,
    parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Both outcomes meet the scenario's expectations (and each side its own pins), so a
    scenario cannot pass by both paths failing for some other reason."""
    remote, local = parity_outcomes[scenario.name]
    for side, outcome, pins in (
        ("record_review", remote, scenario.expect + scenario.expect_remote),
        ("record_local_review", local, scenario.expect + scenario.expect_local),
    ):
        for path, expected in pins:
            assert outcome.get(path, MISSING) == expected, (
                f"{side}: [{scenario.name}] {path}: {outcome.get(path, MISSING)!r} != {expected!r}"
            )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_both_entry_points_agree_except_where_declared(
    scenario: Scenario,
    parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    remote, local = parity_outcomes[scenario.name]
    undeclared = _undeclared(_diffs(scenario.name, remote, local))
    assert not undeclared, (
        "record_review and record_local_review disagree on something DECLARED_DIFFERENCES "
        "does not cover. Fix the drift, or add a row (reason 'undecided' if not yet "
        "decided):\n" + "\n".join(f"  {diff}" for diff in undeclared)
    )


def test_every_path_a_declared_difference_names_is_exercised(
    parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """A named path nothing exempts is stale (or too narrow to ever fire)."""
    all_diffs = [
        diff
        for scenario in SCENARIOS
        for diff in _diffs(scenario.name, *parity_outcomes[scenario.name])
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


def test_local_path_has_no_body_to_hash_so_no_reviewed_body_sha256(
    tmp_path, parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]]
) -> None:
    """Pins the reason on the `reviewed_body_sha256` declared difference: if the Local path
    ever gains a PR body, this fails and the difference must be revisited (issue #3105)."""
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=tmp_path)
    assert gh.pr_view(1) == {}
    assert gh.pr_list() == []
    assert "body" not in local_pr_dict({"number": 1, "branch": "b", "headRefOid": "0" * 40})
    local = parity_outcomes["approve"][1]
    assert local.get("decision_file.reviewed_body_sha256", MISSING) is MISSING
    assert local.get("pr_record.reviewed_body_sha256", MISSING) is MISSING


def test_local_lifecycle_transition_omits_the_record_number_so_no_pr_drill_collision(
    tmp_path, monkeypatch, parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]]
) -> None:
    """Pins the reason on the "Local lifecycle transitions omit the record number" declared
    difference (issue #3106). If a fact stops being true the difference must be revisited."""
    # The Local path's Verdict transition carries no number through the real entry point.
    local = parity_outcomes["approve"][1]
    assert local["events.lifecycle_transition#0.payload.pr_number"] is None
    # (i) a Local path record's number is its local issue file's number.
    assert issue_number_from_name("7_some-issue.md") == 7
    # (ii) lifecycle_transition is an exact Stage time kind.
    assert "lifecycle_transition" in EXACT_KINDS
    # (iii) the dashboard keys a milestone's PR link on the number alone: a row tagged with
    # number 7 surfaces in PR 7's drill-down even though it belongs to a different issue.
    slug = "local/guard"
    root = _register(tmp_path / "fleet", tmp_path / "repo", slug)
    state = root / "state.json"

    def emit(ts: str, payload: dict) -> None:
        monkeypatch.setattr(instrumentation, "_now_iso", lambda: ts)
        instrumentation.log_event(state, "lifecycle_transition", payload)

    emit("2026-10-01T00:00:00Z", {"issue_number": 5, "pr_number": 7, "to_state": "pr_open"})
    emit("2026-10-01T01:00:00Z", {"issue_number": 9, "pr_number": 7, "to_state": "needs_rework"})
    now = datetime(2026, 10, 2, tzinfo=UTC)
    srcs = rollup.rollup_sources(str(tmp_path / "fleet"))
    rollup.run_rollup(srcs, now)
    drill = pr_drill(slug, 7, srcs.db_path, None, now)
    assert {e.issue for e in drill.timeline} == {5, 9}


def test_patch_id_follows_the_pinned_head_not_the_packet(
    parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """With a stale packet, pinning the live head takes the live diff's patch-id and pinning
    the packet head takes the packet's: the patch-id source is discriminated."""
    key = "decision_file.reviewed_patch_id"
    live = parity_outcomes["no_packet_pins_live_head"]
    pinned_live = parity_outcomes["stale_head_operator_pins_live"]
    pinned_packet = parity_outcomes["stale_head_operator_pins_packet"]
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
    scenario_name: str,
    path: str,
    changed: Any,
    parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """The comparison can fail: perturb one field of the Local outcome and it shows
    up as an undeclared difference (the normalisation hides nothing it checks)."""
    remote, local = parity_outcomes[scenario_name]
    assert _undeclared(_diffs(scenario_name, remote, local)) == []
    assert local.get(path, MISSING) != changed
    perturbed = {**local, path: changed}
    assert [d.path for d in _undeclared(_diffs(scenario_name, remote, perturbed))] == [path]


@pytest.mark.skipif(
    not _git_templates.reuse_enabled(),
    reason="CI_FLEET_TEST_REUSE=off forces the fresh path; there is no template to pin",
)
def test_fresh_repo_materializes_without_git_spawns() -> None:
    """Issue #3150 pin: ``_fresh_repo`` copies the per-environment template, so
    only the first call in a process pays ``_make_repo``'s ~12-subprocess git
    build. The verdict suites drive it ~150 times; per-call builds pushed
    shard 6/6 ~1.5x over baseline on the Windows CI host."""
    _fresh_repo()  # warm this environment's template
    with mock.patch.object(subprocess, "Popen", wraps=subprocess.Popen) as popen:
        _fresh_repo()
    assert popen.call_count == 0, "the copy phase spawned instead of copying the template"


def test_fresh_repo_hands_each_caller_its_own_tree(repo_template: _Repo) -> None:
    """Issue #3150 pin: the isolation half of the template-copy contract. The Local
    path writes into ``repo.root`` (issue files, ``.var`` state, ``ci-fleet.toml``),
    so ``_fresh_repo`` must hand every caller its own tree and keep the
    per-environment template pristine -- handing out the template itself would let
    one caller's writes leak into every later copy. Holds under
    ``CI_FLEET_TEST_REUSE=off`` too, so unlike the spawn pin this is not skipped."""
    first = _fresh_repo()
    second = _fresh_repo()
    assert first.root != repo_template.root, "_fresh_repo returned the template itself"
    assert second.root != repo_template.root, "_fresh_repo returned the template itself"
    assert first.root != second.root, "two calls share one tree"
    sentinel = first.root / "written-into-first-copy-only"
    sentinel.write_text("x", encoding="utf-8")
    assert not (second.root / sentinel.name).exists(), "a write into one copy reached another"
    assert not (repo_template.root / sentinel.name).exists(), (
        "a write into a copy reached the template"
    )


@pytest.fixture
def repo_template() -> _Repo:
    """The process repo template, read during fixture setup.

    Under ``CI_FLEET_TEST_REUSE=off`` ``_repo_template`` IS ``_make_repo`` -- a
    fresh ~0.5 s build per call -- and a test-body read charges that to
    ``call`` (the issue-#3190 mechanism the ledger measures). The fresh-repo
    pins compare copies against the template, so they take it through this
    fixture.
    """
    return _repo_template()


# Fixture-side functions: the only places an expensive drive may run. Any other
# function (or a module-level statement) that calls one re-charges the ~0.5 s
# drive to that test's ``call`` phase, which the test ledger reads as a
# per-test regression.
_DRIVE_FIXTURE_SIDE = frozenset(
    {"parity_outcomes", "_outcomes", "_warm_repo_template", "repo_template"}
)


def test_the_drives_stay_in_fixture_setup() -> None:
    """Ledger pin (issues #3187, #3188, #3189, #3190): this module's expensive
    drives may only run during fixture ``setup``, never in a test's ``call``
    phase.

    The two drive classes are the outcome drive -- ``run_both``/``_outcomes``,
    ~0.5 s per scenario on a ``_CACHE`` miss -- and the repo-template build --
    ``_repo_template``/``_make_repo``/``warm_repo_template``, ~0.5 s cold and
    per-call under ``CI_FLEET_TEST_REUSE=off``. Four ledger filings on this
    file (one per param of the parametrized parity tests) each traced to a
    drive running inside a test body. ``_fresh_repo`` is deliberately not a
    drive: post-warm it is the cheap ``copytree`` the fresh-repo pins
    exercise in their bodies by design. A new consumer must take the
    ``parity_outcomes`` or ``repo_template`` fixture -- naming itself in
    ``_SCENARIO_DRIVE`` when it is not parametrized by ``scenario`` /
    ``scenario_name`` -- rather than calling a drive in its body.
    """
    tree = parsed(Path(__file__))
    drive_names = frozenset(
        {"run_both", "_outcomes", "_repo_template", "_make_repo", "warm_repo_template"}
    )

    def drive_calls(node: ast.AST) -> list[int]:
        return [
            sub.lineno
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id in drive_names
        ]

    callers = {
        node.name: drive_calls(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module_level = [
        line
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for line in drive_calls(node)
    ]
    offenders = {
        name: lines for name, lines in callers.items() if lines and name not in _DRIVE_FIXTURE_SIDE
    }
    assert not offenders and not module_level, (
        "a test-body drive re-opens the call-phase ledger regression -- take the "
        "parity_outcomes or repo_template fixture instead: "
        f"{offenders} module-level lines {module_level}"
    )
    # Positive control: the sanctioned call sites are still where the scan
    # expects them, so a broken scan cannot pass vacuously.
    assert {name for name, lines in callers.items() if lines} == set(_DRIVE_FIXTURE_SIDE)


@pytest.mark.parametrize("path", ["issue.status", "labels.added", "events.order"])
def test_the_rescue_exemption_is_limited_to_its_scenario(
    path: str, parity_outcomes: dict[str, tuple[dict[str, Any], dict[str, Any]]]
) -> None:
    """Rescue-tier values are exempt in the rescue scenario only, and only as the
    exact values pinned there."""
    scenario = "rework_at_cap_new_head_rescue_enabled"
    remote, local = parity_outcomes[scenario]
    assert _undeclared(_diffs(scenario, remote, local)) == []
    perturbed = {**local, path: "different"}
    assert [d.path for d in _undeclared(_diffs(scenario, remote, perturbed))] == [path]
    remote, local = parity_outcomes["approve"]
    assert [d.path for d in _undeclared(_diffs("approve", remote, {**local, path: "x"}))] == [path]


_FIXTURE_SIDE = frozenset({"parity_outcomes", "_outcomes"})


def test_the_outcome_drive_stays_in_fixture_setup() -> None:
    """Ledger pin (issues #3187, #3188, #3189): ``run_both``/``_outcomes`` may be
    called only from ``parity_outcomes`` -- which runs during setup -- never
    from a test's ``call`` phase.

    ``run_both`` costs ~0.5 s per scenario on a ``_CACHE`` miss; inside a test
    body the test ledger reads it as that test's ``call`` time and files a
    per-test regression (three filings on this file before the drive moved to
    setup). A new consumer must take the ``parity_outcomes`` fixture -- naming
    itself in ``_SCENARIO_DRIVE`` when it is not parametrized by ``scenario`` /
    ``scenario_name`` -- rather than calling ``_outcomes`` in its body.
    """
    tree = parsed(Path(__file__))
    drive_names = frozenset({"run_both", "_outcomes"})

    def drive_calls(node: ast.AST) -> list[int]:
        return [
            sub.lineno
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id in drive_names
        ]

    callers = {
        node.name: drive_calls(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module_level = [
        line
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for line in drive_calls(node)
    ]
    offenders = {
        name: lines for name, lines in callers.items() if lines and name not in _FIXTURE_SIDE
    }
    assert not offenders and not module_level, (
        "a test-body outcome drive re-opens the call-phase ledger regression -- "
        "take the parity_outcomes fixture instead: "
        f"{offenders} module-level lines {module_level}"
    )
    # Positive control: the sanctioned call sites are still where the scan
    # expects them, so a broken scan cannot pass vacuously.
    assert {name for name, lines in callers.items() if lines} == set(_FIXTURE_SIDE)
