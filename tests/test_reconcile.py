from __future__ import annotations

from typing import Any

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import DriftItem, apply_fixes, detect_drift
from charlie_work.state import empty_state


class FakeGitHub:
    """Records every call so tests can assert detect_drift never mutates."""

    def __init__(self, *, prs: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
        self._prs = prs
        self._issues = issues
        self.run_calls: list[list[str]] = []
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        self.run_calls.append(args)
        if args[:2] == ["pr", "list"]:
            return self._prs
        if args[:2] == ["issue", "list"]:
            return self._issues
        raise AssertionError(f"unexpected gh.run call: {args}")

    def add_issue_label(self, number: int, label: str) -> None:
        self.labels_added.append((number, label))

    def remove_issue_label(self, number: int, label: str) -> None:
        self.labels_removed.append((number, label))


def _pr(
    number: int,
    state: str,
    *,
    head_ref: str | None = None,
    body: str = "",
    title: str = "",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "url": f"https://example.test/pull/{number}",
        "headRefName": head_ref or f"agent/issue-{number}-x",
        "baseRefName": "main",
        "body": body,
        "state": state,
        "labels": [],
    }


def _issue(number: int, labels: list[str]) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "",
        "labels": [{"name": label} for label in labels],
    }


def test_detect_drift_makes_zero_mutating_calls() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
    )
    state = empty_state()

    detect_drift(gh, state, config)

    assert gh.labels_added == []
    assert gh.labels_removed == []
    # Exactly one PR list query and one issue list query.
    assert [call[:2] for call in gh.run_calls] == [["pr", "list"], ["issue", "list"]]


def test_detect_drift_finds_merged_outside_orchestrator() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    item = matches[0]
    assert item.pr_number == 1
    assert item.issue_number == 10
    assert any("merged" in action for action in item.fix_actions)


def test_detect_drift_merged_but_state_already_correct_and_labels_clean_is_not_drift() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "MERGED", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.done])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "merged"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "merged_outside_orchestrator"] == []


def test_detect_drift_finds_closed_unmerged_pr_with_active_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "closed_unmerged_pr_active_labels"]
    assert len(matches) == 1
    assert matches[0].issue_number == 20
    assert matches[0].pr_number == 2
    assert set(matches[0].fix_actions) == {
        f"remove label '{config.labels.pr_open}' from issue #20",
        f"remove label '{config.labels.reviewing}' from issue #20",
    }


def test_apply_fixes_closed_unmerged_pr_removes_active_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "closed_unmerged_pr_active_labels"
    ]

    apply_fixes(gh, state, drift, config)

    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed


def test_detect_drift_finds_state_pr_missing_on_github() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_pr_missing_on_github"]
    assert len(matches) == 1
    assert matches[0].pr_number == 999
    assert matches[0].issue_number == 5


def test_detect_drift_finds_issue_active_label_no_open_pr() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(30, [config.labels.in_progress])])
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "issue_active_label_no_open_pr"]
    assert len(matches) == 1
    assert matches[0].issue_number == 30
    assert matches[0].fix_actions == (
        f"remove label '{config.labels.in_progress}' from issue #30",
    )


def test_detect_drift_issue_active_label_with_open_pr_is_not_drift() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.pr_open])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "issue_active_label_no_open_pr"] == []


def test_detect_drift_finds_done_label_with_active_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(40, [config.labels.done, config.labels.reviewing])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "done_label_with_active_labels"]
    assert len(matches) == 1
    assert matches[0].issue_number == 40
    assert matches[0].fix_actions == (f"remove label '{config.labels.reviewing}' from issue #40",)


def test_apply_fixes_returns_new_state_without_mutating_original() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    original_state = empty_state()
    original_state["prs"]["1"] = {"status": "reviewing"}
    original_snapshot = {
        "issues": dict(original_state["issues"]),
        "prs": {k: dict(v) for k, v in original_state["prs"].items()},
        "events": list(original_state["events"]),
    }
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    new_state = apply_fixes(gh, original_state, drift, config)

    assert original_state["prs"]["1"] == original_snapshot["prs"]["1"]
    assert original_state["events"] == original_snapshot["events"]
    assert new_state is not original_state
    assert new_state["prs"]["1"]["status"] == "merged"


def test_apply_fixes_merged_outside_orchestrator_transitions_labels() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert (10, config.labels.done) in gh.labels_added
    for label in sorted(config.labels.active):
        assert (10, label) in gh.labels_removed
    assert new_state["prs"]["1"]["status"] == "merged"


def test_apply_fixes_contradiction_removes_active_labels_directly() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="done_label_with_active_labels",
            issue_number=40,
            pr_number=None,
            detail="issue #40 has done + reviewing",
            fix_actions=(f"remove label '{config.labels.reviewing}' from issue #40",),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.labels_removed == [(40, config.labels.reviewing)]
    assert gh.labels_added == []


def test_apply_fixes_state_pr_missing_on_github_drops_entry() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}
    drift = [
        DriftItem(
            kind="state_pr_missing_on_github",
            issue_number=5,
            pr_number=999,
            detail="state has prs[999] but gh reports no such PR",
            fix_actions=("drop prs[999] from state",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert "999" not in new_state["prs"]
    assert "999" in state["prs"]


def test_apply_fixes_appends_reconcile_event() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="issue_active_label_no_open_pr",
            issue_number=30,
            pr_number=None,
            detail="issue #30 stale",
            fix_actions=(f"remove label '{config.labels.in_progress}' from issue #30",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["issue_number"] == 30
    assert state["events"] == []
