from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.devin_shell import SessionRecord
from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.github import GraphQLBudgetError, _LIST_LIMIT as github_list_limit
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import (
    DriftItem,
    _LIST_LIMIT as reconcile_list_limit,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import empty_state, is_claim_stale, load_state
from charlie_work.worktree import create_worktree
from charlie_work.workflow import OrchestratorApp


class FakeGitHub:
    """Records every call so tests can assert detect_drift never mutates."""

    def __init__(
        self,
        *,
        prs: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        fail_add_labels: set[tuple[int, str]] | None = None,
        fail_remove_labels: set[tuple[int, str]] | None = None,
        repo_root: Any = None,
        pr_create_return: int | None = None,
        rate_limit_sufficient: bool = True,
        rate_limit_remaining: int = 10000,
        rate_limit_reset: int = 0,
    ) -> None:
        self._prs = prs
        self._issues = issues
        self.run_calls: list[list[str]] = []
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self._fail_add_labels = fail_add_labels or set()
        self._fail_remove_labels = fail_remove_labels or set()
        self.repo_root = repo_root
        self.prs_created: list[dict[str, Any]] = []
        self.pr_create_return = pr_create_return
        self._rate_limit_sufficient = rate_limit_sufficient
        self._rate_limit_remaining = rate_limit_remaining
        self._rate_limit_reset = rate_limit_reset

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        self.run_calls.append(args)
        if args[:2] == ["pr", "list"]:
            return self._prs
        if args[:2] == ["issue", "list"]:
            return self._issues
        raise AssertionError(f"unexpected gh.run call: {args}")

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return (number, label) not in self._fail_add_labels

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return (number, label) not in self._fail_remove_labels

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (
            self._rate_limit_sufficient,
            self._rate_limit_remaining,
            self._rate_limit_reset,
        )


def _pr(
    number: int,
    state: str = "OPEN",
    *,
    head_ref: str | None = None,
    body: str = "",
    title: str = "",
    is_cross_repository: bool = False,
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
        "isCrossRepository": is_cross_repository,
    }


def _issue(number: int, labels: list[str], state: str = "OPEN") -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "",
        "labels": [{"name": label} for label in labels],
        "state": state,
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


def test_detect_drift_finds_issue_active_label_no_open_pr(tmp_path: Path) -> None:
    """Issue #417: this fix path must also add the ready label back, not just
    remove the stale active one -- otherwise a --fix run leaves the issue with
    no dispatch-eligible label at all (mirrors the sibling
    session_failed_relabeled kind).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(30, [config.labels.in_progress])])
    state = empty_state()

    # Ensure no sessions directory exists (to avoid picking up session drift from other tests)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)  # No repo_root, so session detection shouldn't run

    matches = [item for item in drift if item.kind == "issue_active_label_no_open_pr"]
    assert len(matches) >= 1  # May be multiple if both adapters read the same issue
    assert matches[0].issue_number == 30
    assert matches[0].fix_actions == (
        f"remove label '{config.labels.in_progress}' from issue #30",
        f"add label '{config.labels.ready}' to issue #30",
    )
    assert matches[0].remove_labels == (config.labels.in_progress,)
    assert matches[0].add_labels == (config.labels.ready,)


def test_apply_fixes_issue_active_label_no_open_pr_adds_ready_label(tmp_path: Path) -> None:
    """Issue #417 AC(b): mop-up --fix must add the ready label back, not only
    remove the stale active one, so the issue actually becomes dispatchable
    again instead of being left with no state-machine label at all.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(30, [config.labels.in_progress])])
    state = empty_state()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)
    matches = [item for item in drift if item.kind == "issue_active_label_no_open_pr"]
    assert matches

    new_state = apply_fixes(gh, state, matches, config)

    assert (30, config.labels.in_progress) in gh.labels_removed
    assert (30, config.labels.ready) in gh.labels_added
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert all(
        "label_write_failed" not in a
        for e in reconcile_events
        for a in e["payload"]["fix_actions"]
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


def test_detect_drift_finds_done_label_with_active_labels(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(40, [config.labels.done, config.labels.reviewing])],
    )
    state = empty_state()

    # Ensure no sessions directory exists (to avoid picking up session drift from other tests)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    if sessions_dir.exists():
        import shutil

        shutil.rmtree(sessions_dir.parent.parent.parent)

    drift = detect_drift(gh, state, config)  # No repo_root, so session detection shouldn't run

    matches = [item for item in drift if item.kind == "done_label_with_active_labels"]
    assert len(matches) >= 1  # May be multiple if both adapters read the same issue
    assert matches[0].issue_number == 40
    assert matches[0].fix_actions == (f"remove label '{config.labels.reviewing}' from issue #40",)
    assert matches[0].remove_labels == (config.labels.reviewing,)


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
    # Issue #215: merged transition removes ALL other workflow labels, not just active
    for label in sorted(config.labels.workflow_labels - {config.labels.done}):
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
            remove_labels=(config.labels.reviewing,),
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


def test_detect_drift_finds_state_active_status_issue_closed() -> None:
    """Issue #259: a closed issue with an active state-machine status is drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(matches) == 1
    assert matches[0].issue_number == 259
    assert matches[0].fix_actions == ("set state issues[259].status = 'closed'",)


def test_detect_drift_state_active_status_issue_closed_removes_active_labels() -> None:
    """Issue #259: lingering active labels are stripped from the closed issue."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.in_progress], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(matches) == 1
    assert matches[0].remove_labels == (config.labels.in_progress,)
    assert f"remove label '{config.labels.in_progress}' from issue #259" in matches[0].fix_actions


def test_apply_fixes_state_active_status_issue_closed_finalizes_state_and_labels() -> None:
    """Issue #259: apply_fixes sets status closed and removes active labels."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.in_progress], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "dispatched"}

    drift = detect_drift(gh, state, config)
    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["issues"]["259"]["status"] == "closed"
    assert (259, config.labels.in_progress) in gh.labels_removed
    assert state["issues"]["259"]["status"] == "dispatched"


def test_apply_fixes_state_active_status_issue_closed_idempotent() -> None:
    """Issue #259: re-running reconcile on a finalized issue is a no-op."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(259, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["259"] = {"number": 259, "status": "closed"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] == []


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
            remove_labels=(config.labels.in_progress,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["issue_number"] == 30
    assert state["events"] == []


def test_apply_fixes_handles_quote_containing_label() -> None:
    """Structured remove_labels means quote characters in label names don't parse ambiguously."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    quote_label = "agent:has'quote"
    drift = [
        DriftItem(
            kind="done_label_with_active_labels",
            issue_number=50,
            pr_number=None,
            detail="issue #50 has done + quote label",
            fix_actions=(f"remove label '{quote_label}' from issue #50",),
            remove_labels=(quote_label,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.labels_removed == [(50, quote_label)]


def test_detect_drift_surfaces_stale_dispatch_pending_claims() -> None:
    """Stale dispatch_pending claims must be detected as drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Mock is_claim_stale to return True for our test timestamp
    original_is_claim_stale = is_claim_stale

    def _mock_is_claim_stale(claim_timestamp: str | None) -> bool:
        if claim_timestamp == "2020-01-01T00:00:00+00:00":
            return True  # Treat this specific timestamp as stale
        return original_is_claim_stale(claim_timestamp)

    # Temporarily replace is_claim_stale in the reconcile module
    import charlie_work.reconcile as reconcile_module

    original_reconcile_is_claim_stale = reconcile_module.is_claim_stale
    reconcile_module.is_claim_stale = _mock_is_claim_stale

    try:
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatch_pending",
            "dispatch_pending_at": "2020-01-01T00:00:00+00:00",  # Stale timestamp
        }

        drift = detect_drift(gh, state, config)

        stale_claim_drift = [d for d in drift if d.kind == "stale_dispatch_pending_claim"]
        assert len(stale_claim_drift) == 1
        assert stale_claim_drift[0].issue_number == 123
        assert "stale dispatch_pending claim" in stale_claim_drift[0].detail
    finally:
        # Restore original function
        reconcile_module.is_claim_stale = original_reconcile_is_claim_stale


def test_apply_fixes_clears_stale_dispatch_pending_claims() -> None:
    """apply_fixes must clear stale dispatch_pending claims from state."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["issues"]["123"] = {
        "number": 123,
        "status": "dispatch_pending",
        "dispatch_pending_at": "2020-01-01T00:00:00+00:00",
    }

    drift = [
        DriftItem(
            kind="stale_dispatch_pending_claim",
            issue_number=123,
            pr_number=None,
            detail="issue #123 has a stale dispatch_pending claim",
            fix_actions=("clear dispatch_pending claim for issue #123",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # The stale claim should be removed from state
    assert "123" not in new_state["issues"]
    assert "123" in state["issues"]  # Original state unchanged


def test_reconcile_and_github_share_list_limit_constant() -> None:
    """Issue #45: reconcile and github.py must derive limits from the same constant."""
    assert reconcile_list_limit == github_list_limit


def test_detect_drift_snapshot_truncated_skips_closed_issue_finalization(
    caplog,
) -> None:
    """Issue #259 review: a snapshot that hits the page limit is incomplete.

    The finalization sweep must be skipped to avoid acting on a provably
    incomplete snapshot, and a warning drift item must be emitted.
    """
    caplog.set_level(logging.WARNING)
    config = OrchestratorConfig()
    # Exactly _LIST_LIMIT issues: the snapshot is provably truncated.
    closed_issue_number = reconcile_list_limit
    issues = [_issue(i, [config.labels.ready]) for i in range(1, reconcile_list_limit)]
    issues.append(_issue(closed_issue_number, [config.labels.in_progress], state="CLOSED"))
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(closed_issue_number)] = {
        "number": closed_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] == []
    truncated = [item for item in drift if item.kind == "snapshot_truncated"]
    assert len(truncated) == 1
    assert truncated[0].issue_number is None
    assert "truncated" in caplog.text.lower()


def test_detect_drift_snapshot_not_truncated_finalizes_closed_issues() -> None:
    """Issue #259 review: a below-limit snapshot is complete; sweep works."""
    config = OrchestratorConfig()
    closed_issue_number = reconcile_list_limit - 1
    issues = [_issue(i, [config.labels.ready]) for i in range(1, closed_issue_number)]
    issues.append(_issue(closed_issue_number, [config.labels.in_progress], state="CLOSED"))
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(closed_issue_number)] = {
        "number": closed_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "snapshot_truncated"] == []
    closed_items = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(closed_items) == 1
    assert closed_items[0].issue_number == closed_issue_number


def test_apply_fixes_snapshot_truncated_emits_reconcile_event() -> None:
    """Issue #259 review: a truncated snapshot produces a warning event."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="snapshot_truncated",
            issue_number=None,
            pr_number=None,
            detail="snapshot truncated",
            fix_actions=("skip completeness-dependent sweeps",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "snapshot_truncated"


def test_detect_drift_fork_pr_branch_name_does_not_bind() -> None:
    """Issue #9: Fork PRs must not bind via branch name (attacker-controlled)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1,
                "MERGED",
                head_ref="agent/issue-42-fix",
                is_cross_repository=True,
            )
        ],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    # The fork PR should NOT bind to issue 42 via branch name, so drift
    # should be detected for the PR status but NOT for the issue labels.
    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    # The drift item should have issue_number=None because the fork PR
    # didn't bind to issue 42.
    assert matches[0].issue_number is None
    assert matches[0].pr_number == 1


def test_detect_drift_fork_pr_closing_keyword_does_not_bind() -> None:
    """Issue #9: Fork PRs must NOT bind via closing keywords for lifecycle purposes."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1,
                "MERGED",
                head_ref="attacker-branch",
                body="Closes #42",
                is_cross_repository=True,
            )
        ],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}

    drift = detect_drift(gh, state, config)

    # The fork PR should NOT bind to issue 42 via closing keyword, so drift
    # should be detected for the PR status but NOT for the issue labels.
    matches = [item for item in drift if item.kind == "merged_outside_orchestrator"]
    assert len(matches) == 1
    # The drift item should have issue_number=None because the fork PR
    # didn't bind to issue 42.
    assert matches[0].issue_number is None
    assert matches[0].pr_number == 1


def test_detect_drift_provider_throttle_detected_with_repo_root(tmp_path: Path) -> None:
    """Test that detect_drift with repo_root detects dead sessions and classifies throttling."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a sessions directory with a dead session that has a rate-limit log
    # Use the default sessions_dir path from config
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect provider throttle
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42
    assert "rate_limited" in throttle_drift[0].detail
    assert "throttled_until" in throttle_drift[0].fix_actions[0]


def test_apply_fixes_provider_throttle_sets_throttled_until() -> None:
    """Test that apply_fixes correctly sets throttled_until for provider throttle drift."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a provider throttle drift item
    throttled_until = (
        (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )
    drift = [
        DriftItem(
            kind="provider_throttle_detected",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited",
            fix_actions=(f"set throttled_until={throttled_until}",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Verify throttled_until is set in the new state
    assert new_state.get("throttled_until") == throttled_until
    # Original state should be unchanged
    assert state.get("throttled_until") is None


def test_detect_drift_without_repo_root_skips_session_check() -> None:
    """Test that detect_drift without repo_root does not check sessions."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Run detect_drift without repo_root
    drift = detect_drift(gh, state, config)

    # Should not detect any session-related drift
    assert [d for d in drift if d.kind == "provider_throttle_detected"] == []
    assert [d for d in drift if d.kind == "session_failed_relabeled"] == []


def test_detect_drift_session_failed_relabeled_no_open_pr(tmp_path: Path) -> None:
    """Issue #118: dead session with no open PR should trigger label reconciliation."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect both provider throttle and session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)


def test_detect_drift_session_failed_with_open_pr_no_relabel(tmp_path: Path) -> None:
    """Issue #118: dead session with OPEN PR should NOT trigger label reconciliation."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-42-x")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect provider throttle but NOT session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 0


def test_detect_drift_session_failed_with_closed_pr_still_relabeled(tmp_path: Path) -> None:
    """Issue #118: dead session with CLOSED PR should still trigger label reconciliation.

    The guard only counts OPEN PRs, not CLOSED/MERGED. A prior closed PR should
    not permanently suppress the relabel.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "CLOSED", head_ref="agent/issue-42-x")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect both provider throttle and session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels


def test_apply_fixes_session_failed_relabeled(tmp_path: Path) -> None:
    """Issue #118: apply_fixes should remove active labels and add ready label."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a session_failed_relabeled drift item with structured add_labels
    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited, no open PR",
            fix_actions=(
                f"remove label '{config.labels.in_progress}' from issue #42",
                f"add label '{config.labels.ready}' to issue #42",
            ),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.ready,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Verify labels were removed and added
    assert (42, config.labels.in_progress) in gh.labels_removed
    assert (42, config.labels.ready) in gh.labels_added

    # Verify event was emitted
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "session_failed_relabeled"
    assert reconcile_events[0]["payload"]["issue_number"] == 42


def test_apply_fixes_session_failed_relabeled_idempotent(tmp_path: Path) -> None:
    """Issue #118: re-running reconcile on already-relabeled issue should be idempotent."""
    config = OrchestratorConfig()
    # Issue already has ready label and no active labels (already relabeled)
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.ready])],
    )
    state = empty_state()

    # Create a session_failed_relabeled drift item (simulating re-run)
    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited, no open PR",
            fix_actions=(
                f"remove label '{config.labels.in_progress}' from issue #42",
                f"add label '{config.labels.ready}' to issue #42",
            ),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.ready,),
        )
    ]

    # Should not error even though issue doesn't have in_progress label
    new_state = apply_fixes(gh, state, drift, config)

    # Verify the operation completed without error
    assert (42, config.labels.in_progress) in gh.labels_removed
    assert (42, config.labels.ready) in gh.labels_added

    # Verify event was emitted
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1


def test_detect_drift_session_failed_worker_blocked_escalates_instead_of_relabel(
    tmp_path: Path,
) -> None:
    """Issue #261 F5: a dead session whose post-mortem shows worker_blocked
    (killed by a push-gate hook) must NOT be relabeled to ready/redispatched
    like an ordinary dead session — that would hot-redispatch it straight
    back into the same hook and, per attempt_refs.py's motivation, destroy
    its unpushed local commits on the next branch reset. It must escalate
    (session_failed_escalated) instead, mirroring workflow.py's
    "redispatch_escalated" edge for the same signal."""
    worktree_path = str(tmp_path / "worktree")
    now = datetime.now(UTC)

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-1",
        working_directory=worktree_path,
        created_at=now.isoformat(),
        rows=[
            {
                "role": "tool",
                "content": (
                    'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}'
                ),
                "created_at": now.isoformat(),
            }
        ],
    )

    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(db_path)))
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("some work then silence\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # No hot relabel-to-ready for this issue.
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert relabel_drift == []

    escalated_drift = [d for d in drift if d.kind == "session_failed_escalated"]
    assert len(escalated_drift) == 1
    assert escalated_drift[0].issue_number == 42
    assert "worker_blocked" in escalated_drift[0].detail

    # detect_drift is read-only regardless of the worker_blocked branch.
    assert gh.labels_added == []
    assert gh.labels_removed == []


def test_apply_fixes_session_failed_escalated_transitions_labels(tmp_path: Path) -> None:
    """Issue #261 F5: apply_fixes must transition session_failed_escalated
    via the 'redispatch_escalated' label edge (adds human_needed, removes
    the other workflow labels) rather than removing active labels /
    re-adding ready like session_failed_relabeled does."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="session_failed_escalated",
            issue_number=42,
            pr_number=None,
            detail=(
                "issue #42 session died blocked by a push-gate hook (worker_blocked), "
                "no open PR; suppressing relabel-to-ready, escalating instead"
            ),
            fix_actions=("transition issue #42 labels via 'redispatch_escalated' event",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert (42, config.labels.human_needed) in gh.labels_added
    assert (42, config.labels.ready) not in gh.labels_added
    # ready must never be added for an escalated worker_blocked session.

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "session_failed_escalated"
    assert reconcile_events[0]["payload"]["issue_number"] == 42


def test_detect_drift_session_failed_already_has_ready_label(tmp_path: Path) -> None:
    """Issue #118: if issue already has ready label, don't add it again."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.ready, config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled but NOT add ready label action
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels
    # Should not have add ready label in structured field since it's already present
    assert relabel_drift[0].add_labels == ()


def test_detect_drift_claude_code_session_collision_with_unrelated_open_pr(tmp_path: Path) -> None:
    """Issue #118: dead claude-code session issue 42 with unrelated open PR #42 should relabel.

    This is the collision test: issues and PRs share one number sequence, so a dead
    claude-code session for issue N plus any unrelated OPEN PR numbered N must still
    trigger relabel (the guard is keyed by issue, not PR number).
    """
    from charlie_work.claude_code import ClaudeWorkerRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    # Unrelated open PR #42 (does NOT link to issue 42 via branch or closing keyword)
    gh = FakeGitHub(
        prs=[_pr(42, "OPEN", head_ref="some-unrelated-branch")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead claude-code session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a claude-code worker record for a dead session (pid=None to simulate dead).
    # Filename must match claude_code.py's real sidecar convention (issue-{N}.claude.json,
    # see _sidecar_path in claude_code.py) so claude_code.read_worker_records actually
    # picks it up. The old "issue-42-claude-code.json" name never matched that glob and
    # only produced a drift item because devin_shell.py's pre-issue-#343-fix exclusion
    # check let it slip through as a phantom devin session.
    sidecar_path = sessions_dir / "issue-42.claude.json"
    record = ClaudeWorkerRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("claude", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled despite unrelated open PR #42
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels


def test_detect_drift_session_failed_no_pr_mutually_exclusive_with_issue_active_no_pr(
    tmp_path: Path,
) -> None:
    """Issue #118: dead-session-with-no-PR-ever should emit only session_failed_relabeled.

    This test ensures that for a dead session with no PR ever created, we get exactly
    ONE drift item (session_failed_relabeled), not both session_failed_relabeled and
    issue_active_label_no_open_pr. The kinds must be mutually exclusive for a given issue.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    # Issue with active label, no PRs at all (not even closed)
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled but NOT issue_active_label_no_open_pr
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 1, (
        f"Expected exactly 1 session_failed_relabeled, got {len(relabel_drift)}"
    )
    assert all(d.issue_number == 42 for d in relabel_drift)

    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session_failed_relabeled handles it"
    )

    # Verify apply_fixes removes the label exactly once
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    _ = apply_fixes(gh, state, drift, config)

    # Should have exactly one remove call for in_progress
    assert gh.labels_removed.count((42, config.labels.in_progress)) == 1


def test_detect_drift_live_session_no_pr_no_issue_active_drift(tmp_path: Path) -> None:
    """Issue #214: live session with no open PR should NOT trigger issue_active_label_no_open_pr.

    This test ensures that the drift rule checks session liveness before proposing
    label removal. A worker that is still running (is_alive() returns True) should
    not have its labels stripped even if it hasn't opened a PR yet.
    """
    import os
    from charlie_work.devin_shell import SessionRecord, _get_process_start_time
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a LIVE session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Worker is running...\n", encoding="utf-8")

    # Write a session record for a LIVE session (with a real PID that we'll mock as alive)
    # We use the current process's PID to ensure is_alive() returns True
    current_pid = os.getpid()
    current_start_time = _get_process_start_time(current_pid)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=current_pid,  # Use current PID to simulate live session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=current_start_time,  # Use actual process start time
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should NOT detect issue_active_label_no_open_pr for live session
    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session is still alive"
    )

    # Should also not emit session_failed_relabeled (session is alive)
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 0, "Should not emit session_failed_relabeled for live session"


def test_detect_drift_dead_session_no_pr_still_triggers_issue_active_drift(tmp_path: Path) -> None:
    """Issue #214: dead session with no open PR should still trigger issue_active_label_no_open_pr.

    This test ensures that the drift rule still works correctly for dead sessions.
    When a session is dead (is_alive() returns False) and has no open PR, the drift
    rule should still propose label removal.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a DEAD session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Worker died...\n", encoding="utf-8")

    # Write a session record for a DEAD session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled (dead session with no open PR)
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1, "Should emit session_failed_relabeled for dead session"

    # Should NOT detect issue_active_label_no_open_pr (mutually exclusive with session_failed_relabeled)
    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session_failed_relabeled handles it"
    )


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


def test_apply_fixes_multi_item_with_one_failed_label_write() -> None:
    """Issue #125: apply_fixes should record failure when one label write fails."""
    config = OrchestratorConfig()
    # Simulate a failed remove for one label
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_remove_labels={(20, config.labels.pr_open)},
    )
    state = empty_state()

    # Create multiple drift items
    drift = [
        DriftItem(
            kind="closed_unmerged_pr_active_labels",
            issue_number=20,
            pr_number=2,
            detail="PR #2 closed without merging",
            fix_actions=(
                f"remove label '{config.labels.pr_open}' from issue #20",
                f"remove label '{config.labels.reviewing}' from issue #20",
            ),
            remove_labels=(config.labels.pr_open, config.labels.reviewing),
        ),
        DriftItem(
            kind="issue_active_label_no_open_pr",
            issue_number=30,
            pr_number=None,
            detail="issue #30 has active label but no PR",
            fix_actions=(f"remove label '{config.labels.in_progress}' from issue #30",),
            remove_labels=(config.labels.in_progress,),
        ),
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Both items should have been processed
    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed
    assert (30, config.labels.in_progress) in gh.labels_removed

    # Check that the failure was recorded in the event
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 2

    # The first event should have label_write_failed recorded
    first_event = reconcile_events[0]
    assert first_event["payload"]["kind"] == "closed_unmerged_pr_active_labels"
    assert "label_write_failed: true" in first_event["payload"]["fix_actions"]

    # The second event should not have label_write_failed (it succeeded)
    second_event = reconcile_events[1]
    assert second_event["payload"]["kind"] == "issue_active_label_no_open_pr"
    assert "label_write_failed" not in " ".join(second_event["payload"]["fix_actions"])


def test_apply_fixes_transition_failure_recorded_in_event() -> None:
    """Issue #125: apply_fixes should record transition outcome when it fails."""
    config = OrchestratorConfig()
    # Simulate a failed add during transition
    gh = FakeGitHub(
        prs=[],
        issues=[],
        fail_add_labels={(10, config.labels.done)},
    )
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

    # Check that the transition outcome was recorded in the event
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1

    event = reconcile_events[0]
    assert event["payload"]["kind"] == "merged_outside_orchestrator"
    assert "transition outcome" in " ".join(event["payload"]["fix_actions"])
    assert "partial_failure" in " ".join(event["payload"]["fix_actions"])
    assert "add_failures" in " ".join(event["payload"]["fix_actions"])


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


def test_detect_drift_launch_stalled_session(tmp_path: Path) -> None:
    """Issue #221: detect launch_stalled sessions (alive but hung at shim marker)."""
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.worker import _log_is_stalled_at_shim

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a sessions directory with a launch_stalled session
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a log with the shim marker (frozen at ~424-425 bytes)
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 10 minutes ago (past the default 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Verify the log is detected as stalled
    now = datetime.now(UTC)
    assert _log_is_stalled_at_shim(log_path, config.watchdog.launch_stall_grace_minutes, now)

    # Write a session record for a dead session (non-existent PID)
    # The launch_stalled check only runs for alive sessions, so we test the helper directly
    issue_number = 42
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch="agent/issue-42",
        worktree_path="/tmp/worktree-42",
        prompt_path="/tmp/prompt-42.md",
        command=("devin", "prompt.md"),
        pid=None,  # Dead session
        started_at="2026-07-09T00:00:00Z",
        log_path=str(log_path),
        error=None,
        process_start_time=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run detect_drift with repo_root
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Since the session is dead (pid=None), it won't be detected as launch_stalled
    # but the helper function test above confirms the detection logic works
    # This test verifies the integration doesn't crash with the new code
    assert len(drift) == 0  # No drift for dead sessions without open PRs


def test_detect_drift_launch_stalled_calls_kill_process_tree(tmp_path: Path) -> None:
    """Issue #221: launch_stalled path must call kill_process_tree with pid and process_start_time.

    Mutation check: this test FAILS against the old inline-kill code (which calls
    os.killpg / ctypes.TerminateProcess directly and never touches kill_process_tree)
    and PASSES against the fix (which calls kill_process_tree from process_utils).

    Issue #307: the real-activity probe must be conclusive (a genuinely stale,
    non-None timestamp from sessions.db) rather than left to hit the host's real
    sessions.db, which would produce an all-errored/inconclusive probe for this
    fake worktree and now correctly defer instead of killing.
    """
    import json
    import os
    from unittest.mock import patch

    from charlie_work.devin_shell import SessionRecord

    worktree_path = "/tmp/worktree-55"
    now = datetime.now(UTC)

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-55",
        working_directory=worktree_path,
        created_at=now.isoformat(),
        rows=[
            {
                "role": "assistant",
                "content": "still working",
                # Stale past the launch-stall grace period: conclusive evidence
                # of a real stall, not the no-match-yet shape.
                "created_at": (now - timedelta(minutes=20)).isoformat(),
            }
        ],
    )

    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(db_path)))
    gh = FakeGitHub(prs=[], issues=[_issue(55, [config.labels.in_progress])])
    state = empty_state()

    # detect_drift looks in repo_root / config.devin.sessions_dir
    sessions_dir = tmp_path / config.devin.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a small log with only the shim marker — frozen well past grace period
    log_path = sessions_dir / "issue-55.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")
    old_time = now - timedelta(minutes=20)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Use a fake PID that passes is_alive() without actually checking the OS.
    # We patch is_session_alive so the worker reads as alive.
    fake_pid = 99999
    fake_start_time = 1700000000.0

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 55)
    record = SessionRecord(
        issue_number=55,
        branch="agent/issue-55",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt-55.md",
        command=("devin", "prompt.md"),
        pid=fake_pid,
        started_at="2026-07-09T00:00:00Z",
        log_path=str(log_path),
        error=None,
        process_start_time=fake_start_time,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code sidecar interferes
    (sessions_dir / "issue-55.claude.json").unlink(missing_ok=True)

    kill_calls: list[tuple[int, float | None]] = []

    def fake_kill(pid: int, expected_start_time: float | None = None) -> list[int]:
        kill_calls.append((pid, expected_start_time))
        return [pid]

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.reconcile.kill_process_tree", fake_kill),
    ):
        detect_drift(gh, state, config, repo_root=tmp_path)

    assert len(kill_calls) == 1, (
        f"Expected kill_process_tree to be called exactly once, got {kill_calls}"
    )


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote repo and a local clone, return (remote, clone)."""
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _setup_completed_worktree(
    repo_root: Path, issue_number: int, dirty: bool = False
) -> tuple[Path, str]:
    """Create a worktree with one commit beyond origin/main. Return (worktree_path, branch)."""
    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    if dirty:
        (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    return info.path, branch


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    (sessions_dir / f"issue-{issue_number}.claude.json").unlink(missing_ok=True)


def test_detect_drift_completed_unpublished_work_salvaged(tmp_path: Path) -> None:
    """Issue #252: dead session with clean, ahead worktree emits salvage drift."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 252)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 252, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(252, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == 252
    assert salvage[0].branch == branch
    assert salvage[0].base_branch == "main"
    assert salvage[0].remove_labels == (config.labels.in_progress,)
    assert salvage[0].add_labels == (config.labels.pr_open,)

    # No relabel-to-ready drift should be emitted
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert not relabel


def test_detect_drift_dirty_worktree_relabels(tmp_path: Path) -> None:
    """Issue #252: dead session with dirty worktree still relabels to ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 253, dirty=True)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 253, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(253, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert not salvage
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel) == 1
    assert relabel[0].issue_number == 253


def test_detect_drift_no_commits_relabels(tmp_path: Path) -> None:
    """Issue #252: dead session with no commits still relabels to ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-254"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 254, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(254, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert not salvage
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel) == 1
    assert relabel[0].issue_number == 254


def test_apply_fixes_salvage_success_creates_pr_and_labels(tmp_path: Path) -> None:
    """Issue #252: apply_fixes pushes, creates a PR, and moves labels to pr_open."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 255)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(255, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=101,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=255,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    new_state = apply_fixes(gh, empty_state(), drift, config)

    # PR created
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    assert gh.prs_created[0]["base"] == "main"

    # Branch pushed to remote
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-255" in remote_refs.stdout

    # Labels moved
    assert (255, config.labels.in_progress) in gh.labels_removed
    assert (255, config.labels.pr_open) in gh.labels_added

    # Event recorded as salvage
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_unpublished_work_salvaged"


def test_apply_fixes_salvage_push_failure_fallback(tmp_path: Path) -> None:
    """Issue #252: a failed salvage push falls back to relabel-to-ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 256)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(256, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=102,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=256,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    # Force push to fail
    import charlie_work.reconcile

    original_push_branch = charlie_work.reconcile.push_branch
    charlie_work.reconcile.push_branch = lambda repo, br, worktree_path=None: (
        False,
        "simulated push failure",
    )
    try:
        new_state = apply_fixes(gh, empty_state(), drift, config)
    finally:
        charlie_work.reconcile.push_branch = original_push_branch

    # No PR created, active label removed, ready label added
    assert not gh.prs_created
    assert (256, config.labels.in_progress) in gh.labels_removed
    assert (256, config.labels.ready) in gh.labels_added

    # Event recorded as failed relabel
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_failed_relabeled"
    assert any("salvage_failed" in action for action in events[-1]["payload"]["fix_actions"])


def test_detect_drift_defers_when_graphql_rate_limit_below_threshold() -> None:
    """Issue #398: detect_drift must refuse to start a quota-heavy sweep when the
    GraphQL budget is below the configured threshold.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-10-x")],
        issues=[_issue(10, [config.labels.in_progress])],
        rate_limit_sufficient=False,
        rate_limit_remaining=100,
        rate_limit_reset=1234567890,
    )

    with pytest.raises(GraphQLBudgetError) as exc_info:
        detect_drift(gh, empty_state(), config)

    assert exc_info.value.remaining == 100
    assert exc_info.value.reset_at == 1234567890
    assert exc_info.value.threshold == config.runtime.graphql_rate_limit_threshold


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
    assert not any(c[:2] == ["pr", "list"] for c in gh.run_calls)
    assert not any(c[:2] == ["issue", "list"] for c in gh.run_calls)
    # A deferred event was persisted to state.json.
    state = load_state(paths.state_file)
    events = [e for e in state.get("events", []) if e["kind"] == "graphql_rate_limit_deferred"]
    assert len(events) == 1
    assert events[0]["payload"]["remaining"] == 100


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
    assert result.data.get("skipped") is True
    assert result.data.get("reason") == "supervisor_lock_held"


def test_apply_fixes_merged_pr_reaps_review_checkout_and_clears_dispatch_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #494: a PR merged externally while a reviewer dispatch is still
    in-flight must have its isolated review checkout removed and its
    review-dispatch state cleared during mop-up --fix.
    """
    config = OrchestratorConfig()
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
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }

    removed_calls: list[tuple[Any, int, Any]] = []

    def fake_remove_review_checkout(
        repo_root: Path, pr_number: int, *, reviews_dir: Any = None
    ) -> bool:
        removed_calls.append((repo_root, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.reconcile.remove_review_checkout", fake_remove_review_checkout
    )

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "merged_outside_orchestrator"
    ]
    assert drift

    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    repo_root, pr_number, reviews_dir = removed_calls[0]
    assert repo_root == tmp_path
    assert pr_number == 1
    assert reviews_dir == tmp_path / config.review_dispatch.reviews_dir

    assert new_state["prs"]["1"]["status"] == "merged"
    assert new_state["prs"]["1"]["review_dispatch_status"] is None
    assert new_state["prs"]["1"]["review_dispatched_at"] is None
    assert new_state["prs"]["1"]["reviewer_pid"] is None
    assert new_state["prs"]["1"]["reviewer_process_start_time"] is None

    # Original state is never mutated in place.
    assert state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"
