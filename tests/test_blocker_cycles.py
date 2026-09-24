"""Blocker-cycle reporting at intake (issue #1848).

A loop of open issues blocking each other (or an issue listing itself) can
never dispatch, but every member still looks armed -- ``intake()`` reports
each distinct cycle once per pass (up to ``MAX_REPORTED_CYCLES``, with an
explicit truncation marker when more exist) as one ``blocker_cycle`` event
plus one logged warning, with no GitHub writes and no dispatch-decision
change.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.blocker_cycles import (
    MAX_REPORTED_CYCLES,
    _strongly_connected_components,
    detect_open_blocker_cycles,
    find_blocker_cycles,
    open_blocker_edges,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp


def _issue(
    number: int,
    body: str = "",
    *,
    state: str = "OPEN",
    labels: tuple[str, ...] = ("automated-ready",),
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": body,
        "labels": [{"name": name} for name in labels],
        "state": state,
    }


class _CycleGitHub(FakeGitHub):
    """FakeGitHub seeded with a fixed issue set and optional native deps."""

    def __init__(
        self,
        issues: list[dict[str, Any]],
        native_deps: dict[int, list[int]] | None = None,
    ) -> None:
        super().__init__()
        self.issues = list(issues)
        self.prs = []
        self._native_deps = dict(native_deps or {})

    def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
        return {n: self._native_deps.get(n, []) for n in issue_numbers}


def _run_intake(tmp_path: Path, fake_gh: FakeGitHub, *, dry_run: bool = False):
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    save_state(
        paths.state_file,
        {
            "issues": {},
            "prs": {},
            "events": [],
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)
    result = app.intake()
    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e.get("kind") == "blocker_cycle"]
    return result, events


# -- pure graph enumeration -------------------------------------------------


def test_find_blocker_cycles_two_issue_cycle() -> None:
    assert find_blocker_cycles({11: [12], 12: [11]}) == [[11, 12]]


def test_find_blocker_cycles_three_issue_cycle() -> None:
    assert find_blocker_cycles({21: [22], 22: [23], 23: [21]}) == [[21, 22, 23]]


def test_find_blocker_cycles_self_reference() -> None:
    assert find_blocker_cycles({31: [31]}) == [[31]]


def test_find_blocker_cycles_acyclic() -> None:
    assert find_blocker_cycles({51: [52], 52: [53], 53: []}) == []


def test_find_blocker_cycles_reports_each_distinct_cycle_once() -> None:
    # Two elementary cycles share the 4 -> 1 edge; both must be found.
    edges = {1: [2, 3], 2: [4], 3: [4], 4: [1]}
    assert find_blocker_cycles(edges) == [[1, 2, 4], [1, 3, 4]]


def test_find_blocker_cycles_disjoint_cycles() -> None:
    edges = {61: [62], 62: [61], 63: [64], 64: [63]}
    assert find_blocker_cycles(edges) == [[61, 62], [63, 64]]


def test_find_blocker_cycles_large_layered_dag_returns_empty() -> None:
    """Regression: the scan's cost must be driven by cyclic structure, not
    path count.

    A width-5 x depth-12 layered DAG where every issue is blocked by all five
    issues in the next (higher-numbered) layer contains ~7.6e7 simple paths.
    The pre-SCC implementation enumerated all of them -- measured multiple
    seconds of pure-Python DFS for zero cycles; with SCC decomposition the
    scan is a single linear pass. If this test ever takes noticeable time,
    the acyclic fast path has regressed.
    """
    width, depth = 5, 12
    edges = {}
    for layer in range(depth):
        for pos in range(width):
            number = 1000 + layer * width + pos
            edges[number] = (
                [1000 + (layer + 1) * width + k for k in range(width)] if layer < depth - 1 else []
            )
    assert find_blocker_cycles(edges) == []


def test_find_blocker_cycles_limit_stops_enumeration() -> None:
    # A complete digraph has far more than 3 elementary cycles; the limit
    # bounds the work, and the sorted output stays deterministic.
    edges = {n: [m for m in range(1, 6) if m != n] for n in range(1, 6)}
    assert len(find_blocker_cycles(edges, limit=3)) == 3


def test_strongly_connected_components_dag_all_singletons() -> None:
    # Every component of an acyclic graph is a single vertex -- this is what
    # lets find_blocker_cycles skip enumeration entirely on DAGs.
    adjacency = {1: [2, 3], 2: [4], 3: [4], 4: [], 5: [1]}
    components = _strongly_connected_components(adjacency)
    assert sorted(map(sorted, components)) == [[1], [2], [3], [4], [5]]


def test_strongly_connected_components_groups_cyclic_members() -> None:
    adjacency = {1: [2], 2: [1, 3], 3: [4], 4: [3]}
    components = _strongly_connected_components(adjacency)
    assert sorted(map(sorted, components)) == [[1, 2], [3, 4]]


# -- edge construction ------------------------------------------------------


def test_open_blocker_edges_drops_closed_targets_keeps_self(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(1, "Blocked by #2"),
            _issue(2, "", state="CLOSED"),
            _issue(3, "Blocked by #3"),
        ]
    )
    edges = open_blocker_edges(gh, gh.issue_list(state="open"))
    assert edges[1] == []  # #2 is closed: the edge cannot join a loop
    assert edges[3] == [3]  # self-reference is a real edge here
    assert 2 not in edges


def test_open_blocker_edges_unions_body_and_native_deps(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [_issue(1, "Blocked by #2"), _issue(2), _issue(3)],
        native_deps={1: [3]},
    )
    edges = open_blocker_edges(gh, gh.issue_list(state="open"))
    assert edges[1] == [2, 3]


# -- intake integration -----------------------------------------------------


def test_intake_two_issue_cycle_emits_one_event(tmp_path: Path) -> None:
    gh = _CycleGitHub([_issue(11, "Blocked by #12"), _issue(12, "Blocked by #11")])
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [11, 12]
    assert result.data["blocker_cycles"] == [[11, 12]]
    # Reporting only -- no labels, comments, or issue mutations.
    assert gh.labels_added == []
    assert gh.closed_issues == []
    assert not getattr(gh, "issue_comments_posted", [])


def test_intake_three_issue_cycle_emits_one_event(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(21, "Blocked by #22"),
            _issue(22, "Blocked by #23"),
            _issue(23, "Blocked by #21"),
        ]
    )
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [21, 22, 23]


def test_intake_self_reference_emits_one_event(tmp_path: Path) -> None:
    gh = _CycleGitHub([_issue(31, "Blocked by #31")])
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [31]


def test_intake_cycle_through_closed_issue_emits_none(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(41, "Blocked by #42"),
            _issue(42, "Blocked by #43", state="CLOSED"),
            _issue(43, "Blocked by #41"),
        ]
    )
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert events == []
    assert result.data["blocker_cycles"] == []


def test_intake_acyclic_graph_emits_none(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(51, "Blocked by #52"),
            _issue(52, "Blocked by #53"),
            _issue(53, "No blockers"),
        ]
    )
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert events == []


def test_intake_reports_native_dependency_cycle(tmp_path: Path) -> None:
    """Blockers also come from GitHub's native dependencies, not just bodies."""
    gh = _CycleGitHub([_issue(71), _issue(72)], native_deps={71: [72], 72: [71]})
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [71, 72]


def test_intake_reports_cycle_among_unarmed_open_issues(tmp_path: Path) -> None:
    """The graph is over open issues, not just armed ones: a cycle whose
    members lack ``automated-ready`` is still a loop that would deadlock the
    moment they are armed, so it is reported."""
    gh = _CycleGitHub(
        [
            _issue(81, "Blocked by #82", labels=()),
            _issue(82, "Blocked by #81", labels=()),
        ]
    )
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [81, 82]


def test_intake_two_disjoint_cycles_emit_two_events(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(61, "Blocked by #62"),
            _issue(62, "Blocked by #61"),
            _issue(63, "Blocked by #64"),
            _issue(64, "Blocked by #63"),
        ]
    )
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert [e["payload"]["issue_numbers"] for e in events] == [[61, 62], [63, 64]]


def test_intake_logs_one_warning_per_cycle(tmp_path: Path, caplog) -> None:
    gh = _CycleGitHub([_issue(11, "Blocked by #12"), _issue(12, "Blocked by #11")])
    with caplog.at_level(logging.WARNING):
        _run_intake(tmp_path, gh)
    warnings = [r for r in caplog.records if "blocker cycle" in r.message]
    assert len(warnings) == 1
    assert "#11" in warnings[0].message and "#12" in warnings[0].message


def test_intake_dry_run_reports_cycles_without_events(tmp_path: Path) -> None:
    gh = _CycleGitHub([_issue(11, "Blocked by #12"), _issue(12, "Blocked by #11")])
    result, events = _run_intake(tmp_path, gh, dry_run=True)
    assert result.ok is True
    assert result.data["blocker_cycles"] == [[11, 12]]
    assert events == []


def test_intake_blocker_cycle_scan_fail_open(tmp_path: Path) -> None:
    """A gh failure inside the scan must not fail intake."""

    class _BoomGitHub(FakeGitHub):
        def issue_list(self, labels: Any = None, state: Any = None):
            if labels is None:
                raise RuntimeError("gh exploded")
            return super().issue_list(labels, state)

    result, events = _run_intake(tmp_path, _BoomGitHub())
    assert result.ok is True
    assert events == []
    assert result.data["blocker_cycles"] == []


def test_intake_blocker_cycle_scan_fail_open_on_edge_building(tmp_path: Path, caplog) -> None:
    """A failure downstream of issue_list -- here the open-state resolution
    during edge building -- must also fail the scan open, and must log the
    failure so it cannot silently read as 'no cycles'."""

    class _OpenStateBoomGitHub(_CycleGitHub):
        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            raise RuntimeError("state query exploded")

    gh = _OpenStateBoomGitHub([_issue(11, "Blocked by #12"), _issue(12, "Blocked by #11")])
    with caplog.at_level(logging.WARNING):
        result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert events == []
    assert result.data["blocker_cycles"] == []
    assert result.data["blocker_cycles_truncated"] is False
    assert any("blocker-cycle scan failed" in r.message for r in caplog.records)


def test_intake_batch_dependency_failure_falls_back_to_per_issue_rest(
    tmp_path: Path,
) -> None:
    """A raising batched ``issue_dependencies`` must fall back to per-issue
    ``gh api .../issues/N/dependencies/blocked_by`` calls -- the same
    fallback ``_prefetch_blocker_data`` uses -- and still report the cycle.
    """

    class _BatchFailGitHub(_CycleGitHub):
        def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
            raise GitHubError("batched dependency query exploded")

        def run(self, args, *, json_output: bool = False, allow_failure: bool = False):
            match = re.search(r"issues/(\d+)/dependencies/blocked_by", " ".join(args))
            if match:
                number = int(match.group(1))
                return [{"number": dep} for dep in self._native_deps.get(number, [])]
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    gh = _BatchFailGitHub([_issue(71), _issue(72)], native_deps={71: [72], 72: [71]})
    result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert [e["payload"]["issue_numbers"] for e in events] == [[71, 72]]


def test_intake_dense_component_report_is_capped_and_marked(tmp_path: Path, caplog) -> None:
    """k=9 issues each natively blocked by all the others: 125,664 distinct
    elementary cycles. The pass must report at most MAX_REPORTED_CYCLES,
    mark the report truncated on the intake summary event, and log the
    truncation -- never flood the log or hold the state lock for 125k
    events."""
    numbers = list(range(100, 109))
    gh = _CycleGitHub(
        [_issue(n) for n in numbers],
        native_deps={n: [m for m in numbers if m != n] for n in numbers},
    )
    with caplog.at_level(logging.WARNING):
        result, events = _run_intake(tmp_path, gh)
    assert result.ok is True
    assert len(result.data["blocker_cycles"]) == MAX_REPORTED_CYCLES
    assert result.data["blocker_cycles_truncated"] is True
    assert len(events) == MAX_REPORTED_CYCLES
    cycle_warnings = [r for r in caplog.records if "blocker cycle detected" in r.message]
    assert len(cycle_warnings) == MAX_REPORTED_CYCLES
    assert any("truncated" in r.message for r in caplog.records)
    intake_events = [
        e
        for e in load_state(
            runtime_paths(tmp_path, OrchestratorConfig().runtime.state_dir).state_file
        )["events"]
        if e.get("kind") == "intake"
    ]
    assert intake_events[-1]["payload"]["blocker_cycles_reported"] == MAX_REPORTED_CYCLES
    assert intake_events[-1]["payload"]["blocker_cycles_truncated"] is True


def test_detect_open_blocker_cycles_returns_sorted_cycles(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(9, "Blocked by #8"),
            _issue(8, "Blocked by #9"),
            _issue(4, "Blocked by #4"),
        ]
    )
    scan = detect_open_blocker_cycles(gh)
    assert scan.cycles == [[4], [8, 9]]
    assert scan.truncated is False
