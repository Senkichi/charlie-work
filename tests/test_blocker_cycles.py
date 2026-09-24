"""Blocker-cycle reporting at intake (issue #1848).

A loop of open issues blocking each other (or an issue listing itself) can
never dispatch, but every member still looks armed -- ``intake()`` reports
each distinct cycle once per pass as one ``blocker_cycle`` event plus one
logged warning, with no GitHub writes and no dispatch-decision change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.blocker_cycles import (
    detect_open_blocker_cycles,
    find_blocker_cycles,
    open_blocker_edges,
)
from charlie_work.config import OrchestratorConfig
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
        {"issues": {}, "prs": {}, "events": [], "generated_at": "2024-01-01T00:00:00Z"},
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


def test_detect_open_blocker_cycles_returns_sorted_cycles(tmp_path: Path) -> None:
    gh = _CycleGitHub(
        [
            _issue(9, "Blocked by #8"),
            _issue(8, "Blocked by #9"),
            _issue(4, "Blocked by #4"),
        ]
    )
    assert detect_open_blocker_cycles(gh) == [[4], [8, 9]]
