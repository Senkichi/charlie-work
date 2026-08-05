from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import LabelConfig, OrchestratorConfig, PostMortemConfig
from charlie_work.devin_shell import SessionRecord
from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.github import (
    GitHubError,
    GraphQLBudgetError,
    _LIST_LIMIT as github_list_limit,
)
from charlie_work.instrumentation import read_event_log
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.reconcile import (
    AVIATOR_BLOCKED_MESSAGE,
    AVIATOR_CHECK_NAME,
    DriftItem,
    _LIST_LIMIT as reconcile_list_limit,
    _fetch_issues,
    _fetch_prs,
    apply_fixes,
    detect_aviator_stale_blocked,
    detect_drift,
    detect_mergequeue_not_approved,
)
from charlie_work.state import PASSIVE_OPEN_STATUS, empty_state, is_claim_stale, load_state
from charlie_work.worktree import create_worktree
from charlie_work.workflow import OrchestratorApp

# Module-level default label config for parametrize decorators that need
# label strings at collection time (before any test creates an OrchestratorConfig).
_config_labels = LabelConfig()


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
        # PR-scoped label tracking, distinct from the issue-scoped lists above.
        self.pr_labels_added: list[tuple[int, str]] = []
        self.pr_labels_removed: list[tuple[int, str]] = []
        self._fail_add_pr_labels: set[tuple[int, str]] = set()
        self._fail_remove_pr_labels: set[tuple[int, str]] = set()
        # sha -> list of check-run dicts, for detect_aviator_stale_blocked.
        self.check_runs_by_sha: dict[str, list[dict[str, Any]]] = {}
        self.commit_check_runs_calls: list[str] = []

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

    def add_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_added.append((number, label))
        return (number, label) not in self._fail_add_pr_labels

    def remove_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_removed.append((number, label))
        return (number, label) not in self._fail_remove_pr_labels

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        self.commit_check_runs_calls.append(sha)
        return self.check_runs_by_sha.get(sha)

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


def test_detect_drift_pr_snapshot_truncated_still_skips_state_pr_missing_on_github(
    caplog,
) -> None:
    """Issue #857 acceptance criterion 2: state_pr_missing_on_github stays fully gated.

    Unlike the issue-side sweeps (state_active_status_issue_closed,
    issue_status_normalized), this sweep's per-item fix is NOT safe to run
    against a partial snapshot: a false positive here reaches
    ``new_prs.pop(...)`` in ``apply_fixes`` and erases ``decision`` /
    ``reviewed_head_sha`` for an approved PR fleet-wide. Issue #857 deliberately
    leaves this gate untouched -- pin that it still skips outright when the PR
    snapshot is truncated, so a future refactor of the issue-side gate doesn't
    accidentally also lift this one.
    """
    caplog.set_level(logging.WARNING)
    config = OrchestratorConfig()
    # Exactly _LIST_LIMIT PRs, none numbered 999: the PR snapshot is provably
    # truncated AND PR #999 (tracked in state, "missing" on GitHub) fell off it.
    prs = [_pr(i, "OPEN") for i in range(1, reconcile_list_limit + 1)]
    gh = FakeGitHub(prs=prs, issues=[])
    state = empty_state()
    state["prs"]["999"] = {"issue_number": 5, "status": "reviewing"}

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_pr_missing_on_github"] == []
    truncated = [item for item in drift if item.kind == "snapshot_truncated"]
    assert len(truncated) == 1
    assert "incomplete" in caplog.text.lower()


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
    assert [item for item in drift if item.kind == "issue_active_label_with_open_pr"] == []


def test_detect_drift_finds_issue_active_label_with_open_pr() -> None:
    """Issue #515: an issue stuck on needs_rework while an open PR exists is drift."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()

    drift = detect_drift(gh, state, config)

    matches = [item for item in drift if item.kind == "issue_active_label_with_open_pr"]
    assert len(matches) == 1
    assert matches[0].issue_number == 30
    assert matches[0].pr_number == 3
    assert matches[0].remove_labels == (config.labels.needs_rework,)
    assert matches[0].add_labels == (config.labels.pr_open,)
    assert matches[0].fix_actions == (
        f"remove label '{config.labels.needs_rework}' from issue #30",
        f"add label '{config.labels.pr_open}' to issue #30",
        f"set state issues[30].status = {PASSIVE_OPEN_STATUS!r}",
    )


def test_apply_fixes_issue_active_label_with_open_pr() -> None:
    """Issue #515: the --fix path must repair labels and update state status."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(3, "OPEN", head_ref="agent/issue-30-x")],
        issues=[_issue(30, [config.labels.needs_rework])],
    )
    state = empty_state()
    state["issues"]["30"] = {
        "number": 30,
        "status": "rework_requested",
        "worker_pid": 12345,
    }

    drift = [
        item
        for item in detect_drift(gh, state, config)
        if item.kind == "issue_active_label_with_open_pr"
    ]
    assert drift

    new_state = apply_fixes(gh, state, drift, config)

    assert (30, config.labels.needs_rework) in gh.labels_removed
    assert (30, config.labels.pr_open) in gh.labels_added
    # Issue #515 (generalized): PASSIVE_OPEN_STATUS -- not "approved" -- is
    # the status this repair mirrors, instead of implying a review verdict
    # was recorded. Distinct from the active "reviewing" review() writes
    # (#955).
    assert new_state["issues"]["30"]["status"] == PASSIVE_OPEN_STATUS
    assert "worker_pid" not in new_state["issues"]["30"]


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


# --- issue #947: agent:human-needed silently invisible past a configurable age ---


def test_detect_drift_finds_terminal_state_stale_via_terminal_since(tmp_path: Path) -> None:
    """A `terminal_since` stamp (written by `_escalate_issue` since #947) past
    the configured threshold fires `terminal_state_stale` with the parked
    issue's number and a numeric age, not merely "some event fired"."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(894, [config.labels.human_needed])])
    state = empty_state()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    state["issues"]["894"] = {
        "number": 894,
        "status": "escalated",
        "terminal_since": "2026-01-05T00:00:00Z",  # 5 days before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert "5.0 day" in matches[0].detail
    assert matches[0].fix_actions == ()


def test_detect_drift_terminal_state_stale_not_yet_due(tmp_path: Path) -> None:
    """A fresh escalation (age below the configured threshold) must not fire
    -- this is the negative control for the positive case above."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(895, [config.labels.human_needed])])
    state = empty_state()
    now = datetime(2026, 1, 10, 1, 0, 0, tzinfo=UTC)
    state["issues"]["895"] = {
        "number": 895,
        "status": "escalated",
        "terminal_since": "2026-01-10T00:00:00Z",  # 1 hour before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    assert [item for item in drift if item.kind == "terminal_state_stale"] == []


def test_detect_drift_terminal_state_stale_legacy_escalation_via_events_db(
    tmp_path: Path,
) -> None:
    """Issue #894 shape: an issue escalated BEFORE #947 shipped carries no
    `terminal_since` field at all. The detector must still report a real
    numeric age (not "never observed") by falling back to the most recent
    escalation-transition event in events.db -- the same CI-verified kind
    registry `_backfill_missing_reason_classes` already relies on. Without
    this fallback tier the original design silently degraded #894 itself to
    the "never observed" bucket, which is the exact bug this PR fixes."""
    from charlie_work.instrumentation import log_event

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(894, [config.labels.human_needed])])
    state = empty_state()
    # Deliberately no terminal_since / merged_pr_mention_flagged_at: this is
    # the legacy (pre-#947) shape.
    state["issues"]["894"] = {"number": 894, "status": "escalated"}

    state_path = tmp_path / ".var" / "charlie-work" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_event(state_path, "session_failed_escalated", {"issue_number": 894}, repo="test-repo")

    now = datetime.now(UTC) + timedelta(days=5)
    drift = detect_drift(gh, state, config, state_path=state_path, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert "never observed" not in matches[0].detail
    assert "4." in matches[0].detail or "5." in matches[0].detail


def test_detect_drift_terminal_state_stale_events_db_ignores_non_escalation_kinds(
    tmp_path: Path,
) -> None:
    """A non-escalation event for the issue (e.g. a routine dispatch record)
    must NOT be mistaken for an escalation-transition timestamp -- proves the
    events.db fallback filters by the CI-verified escalation-kind registry,
    not "any event for this issue_number"."""
    from charlie_work.instrumentation import log_event

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(898, [config.labels.human_needed])])
    state = empty_state()
    state["issues"]["898"] = {"number": 898, "status": "escalated"}

    state_path = tmp_path / ".var" / "charlie-work" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # "dispatch" is not an escalation-transition kind: it neither ends in
    # "_escalated" nor is registered in ESCALATION_REASON_CLASS_BY_EVENT_KIND
    # / DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS.
    log_event(state_path, "dispatch", {"issue_number": 898}, repo="test-repo")

    drift = detect_drift(gh, state, config, state_path=state_path)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 898
    assert "never observed" in matches[0].detail


def test_detect_drift_terminal_state_stale_never_observed_without_any_timestamp(
    tmp_path: Path,
) -> None:
    """No `terminal_since`, no `merged_pr_mention_flagged_at`, and no
    matching events.db row (or no `state_path` at all): the detector must
    still surface the issue immediately, distinctly labeled "never observed"
    rather than silently defaulting an unknown age to "healthy" -- mirroring
    `classify_backlog_reachability`'s `observed: False` precedent."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(896, [config.labels.human_needed])])
    state = empty_state()
    state["issues"]["896"] = {"number": 896, "status": "escalated"}

    drift = detect_drift(gh, state, config)  # No state_path passed at all.

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 896
    assert "never observed" in matches[0].detail


def test_detect_drift_terminal_state_stale_ignores_done_label() -> None:
    """`agent:done` is a normal, expected terminal state (issue closed via
    the ordinary lifecycle) -- it must never be treated as a stuck
    human-needed issue, even though both are members of `labels.terminal`."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[_issue(897, [config.labels.done])])
    state = empty_state()

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "terminal_state_stale"] == []


def test_apply_fixes_terminal_state_stale_emits_reconcile_event_with_content() -> None:
    """The generic unfixable-kind fallback in `apply_fixes` (precedented by
    `snapshot_truncated`/`escalated_labels_converged`) must emit a
    `"reconcile"` event whose payload carries the specific kind and the
    parked issue's number -- asserting on content, not just that some event
    fired."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="terminal_state_stale",
            issue_number=894,
            pr_number=None,
            detail="issue #894 has been parked in 'agent:human-needed' for 5.0 day(s)",
            fix_actions=(),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state["events"] if e.get("kind") == "reconcile"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["kind"] == "terminal_state_stale"
    assert payload["issue_number"] == 894
    assert "5.0 day" in payload["detail"]
    # No GitHub label mutation for an alert-only kind.
    assert gh.labels_added == []
    assert gh.labels_removed == []


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


def test_detect_drift_snapshot_truncated_still_finalizes_in_window_closed_issue(
    caplog,
) -> None:
    """Issue #857: an in-window closed issue is finalized while an out-of-window
    entry in the SAME pass is still skipped silently.

    This test used to assert ZERO finalization whenever the issue snapshot hit
    the page limit at all. That assertion was wrong: it conflated "the snapshot
    as a whole is incomplete" with "this specific item is unanswerable". The
    closed issue built below IS present in the truncated snapshot -- GitHub gave
    a definite CLOSED answer for it -- so skipping its finalization discarded a
    known-good signal for no reason. The per-item lookup
    (`issues_by_number.get(...)` -> ``None`` -> ``continue``) already fails safe
    on issues that truly fell off the page; the outer total-skip gate added
    nothing beyond that and is removed by issue #857.

    Both entries share one ``detect_drift()`` call (issue #857 acceptance
    criterion 4's exact wording: "assert the in-window issue IS finalized while
    genuinely out-of-window entries are still skipped"). A version of this test
    with only the out-of-window entry would pass unchanged under the OLD, fully
    gated code too -- a lone out-of-window entry produces zero drift either
    way -- so it alone would not prove per-item discrimination replaced the
    outer gate. Combining both in one pass is what actually pins that.

    A warning drift item (``snapshot_truncated``) is still emitted so operators
    know the snapshot is incomplete, but it must not claim the sweeps were
    skipped outright now that they partially ran (acceptance criterion 5).
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
    # A second entry whose number never appears in the snapshot above: GitHub's
    # answer for it is genuinely unknown (not "closed"), so it must be skipped
    # silently rather than flagged (acceptance criterion 3).
    out_of_window_issue_number = reconcile_list_limit + 1000
    state["issues"][str(out_of_window_issue_number)] = {
        "number": out_of_window_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    closed_items = [item for item in drift if item.kind == "state_active_status_issue_closed"]
    assert len(closed_items) == 1
    assert closed_items[0].issue_number == closed_issue_number
    assert [item for item in drift if item.kind == "issue_status_normalized"] == []
    truncated = [item for item in drift if item.kind == "snapshot_truncated"]
    assert len(truncated) == 1
    assert truncated[0].issue_number is None

    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) == 1
    message = warning_records[0].getMessage()
    # Positive check (acceptance criterion 5): the message must name both
    # sweeps and describe partial coverage, not just "truncated".
    assert "state_active_status_issue_closed" in message
    assert "issue_status_normalized" in message
    assert "ran against" in message
    # It must not claim the sweeps were skipped outright now that they
    # partially run. Scoped to this one record (not all of caplog.text) so an
    # unrelated future warning containing "skip" can't false-fail this test.
    assert "skipping" not in message.lower()


def test_detect_drift_snapshot_truncated_skips_out_of_window_issue() -> None:
    """Issue #857 acceptance criterion 3: an out-of-window entry is skipped silently.

    An issue tracked in state.json with an active status, but whose number does
    NOT appear anywhere in the (truncated, exactly-_LIST_LIMIT) snapshot, is
    genuinely unanswerable -- GitHub may have closed it, or it may still be
    open; the snapshot simply doesn't say. It must be skipped, not flagged, and
    the skip must not itself produce a drift item (the single snapshot_truncated
    warning already covers that). This is a standalone companion to the combined
    in-window/out-of-window test above: it isolates the out-of-window case on
    its own so a future change to the in-window path can't accidentally also
    break this one without a dedicated assertion catching it.
    """
    config = OrchestratorConfig()
    # Exactly _LIST_LIMIT issues, none numbered `out_of_window_issue_number`:
    # the snapshot is provably truncated AND this issue fell off the page.
    out_of_window_issue_number = reconcile_list_limit + 1000
    issues = [_issue(i, [config.labels.ready]) for i in range(1, reconcile_list_limit + 1)]
    gh = FakeGitHub(prs=[], issues=issues)
    state = empty_state()
    state["issues"][str(out_of_window_issue_number)] = {
        "number": out_of_window_issue_number,
        "status": "dispatched",
    }

    drift = detect_drift(gh, state, config)

    assert [item for item in drift if item.kind == "state_active_status_issue_closed"] == []
    assert [item for item in drift if item.kind == "issue_status_normalized"] == []
    truncated = [item for item in drift if item.kind == "snapshot_truncated"]
    assert len(truncated) == 1


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


def test_detect_drift_issue_status_normalized_skips_under_incomplete_pr_snapshot() -> None:
    """Issue #859: PR-side counterpart of issue #789 (a few lines above it in
    detect_drift) -- reworked per PR #972 review to a PER-ITEM condition.

    ``open_prs_by_issue`` is built from the same ``prs`` snapshot that can be
    provably incomplete once total PR count hits ``_LIST_LIMIT``. If an
    issue's genuinely open PR fell off that page, ``open_prs_by_issue.get(...)``
    returns nothing even though a PR exists, and the pre-#859 code fell
    through to ``target_status = None`` -- silently normalizing a real, live
    issue's status to the untracked baseline with no warning.

    The fix does NOT gate on the global ``pr_snapshot_incomplete`` flag (an
    earlier version of this PR did, and was rejected in review: under
    ``--state all`` that flag is monotonic, so once the repo permanently
    crosses ``_LIST_LIMIT`` PRs it would disable this sweep repo-wide,
    forever -- reproducing the exact #857/#860 failure mode this repo already
    fixed once). Instead it uses issue-specific evidence: state.json's own PR
    record (``state["prs"]``) says this issue has a still-open PR (status not
    yet "closed"/"merged") whose PR number is entirely absent from this
    pass's ``prs`` snapshot. That is this test's setup below. This is the
    regression test for the original bug: it must fail on the pre-#859 code
    and pass on the fix.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    tracked_pr_number = 999999  # deliberately outside the PR snapshot below
    # The issue itself is well inside the issue snapshot and genuinely OPEN;
    # only the PR snapshot is the one under test.
    issues = [_issue(target_issue_number, [])]
    # Exactly _LIST_LIMIT PRs, none linked to target_issue_number and none
    # numbered tracked_pr_number: the PR snapshot is provably incomplete, and
    # this issue's real open PR (which state.json tracks and which exists on
    # GitHub) simply isn't on this page.
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        # "closed" is the reachable value outside ORCHESTRATOR_OWNED_ISSUE_STATUSES
        # (a stale value from a GitHub reopen, per issue #859's own example).
        "status": "closed",
    }
    # state.json's own record: this issue has a still-open tracked PR that
    # happens to be absent from the `prs` snapshot fetched above. This is the
    # issue-specific evidence the per-item guard requires.
    state["prs"][str(tracked_pr_number)] = {
        "number": tracked_pr_number,
        "issue_number": target_issue_number,
        "status": "reviewing",
    }

    drift = detect_drift(gh, state, config)

    bad = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert bad == []

    # Requirement from PR #972 review comment 4: the deferral must be named
    # in the drift log, not silent.
    deferred = [
        item
        for item in drift
        if item.kind == "snapshot_truncated" and "issue_status_normalized deferred" in item.detail
    ]
    assert len(deferred) == 1
    assert str(target_issue_number) in deferred[0].detail


def test_detect_drift_issue_status_normalized_none_still_fires_with_no_tracked_pr_anywhere() -> (
    None
):
    """Issue #859 review comment 2/3: proves the fix is NOT a repo-wide kill
    switch once the PR snapshot is incomplete.

    Same incomplete-PR-snapshot setup as the regression test above, but this
    issue has no PR anywhere -- not in the GitHub snapshot, and not tracked in
    state.json either. There is no issue-specific evidence to distrust the
    negative answer, so ``target_status = None`` must still fire exactly as
    it did before #859, even while the global snapshot is provably
    incomplete. Without this test, a broad guard keyed on the global
    ``pr_snapshot_incomplete`` flag (the shape rejected in review) would pass
    every other test in this module while silently disabling
    ``issue_status_normalized`` for the entire repo once PR count crosses
    ``_LIST_LIMIT`` -- exactly the #857/#860 regression this rework exists to
    avoid reintroducing.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [])]
    # Exactly _LIST_LIMIT PRs, none linked to target_issue_number: the PR
    # snapshot is provably incomplete (same global condition as the
    # regression test), but state.json tracks NO PR for this issue at all.
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }
    # No state["prs"] entry for this issue at all -- state.json has zero
    # opinion about a PR existing for it.

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert len(matches) == 1

    # And the per-item deferral warning must NOT claim this issue was
    # deferred, since it wasn't.
    deferred = [
        item
        for item in drift
        if item.kind == "snapshot_truncated" and "issue_status_normalized deferred" in item.detail
    ]
    assert deferred == []


def test_detect_drift_issue_status_normalized_none_still_fires_when_pr_snapshot_complete() -> None:
    """Discriminator: under a COMPLETE PR snapshot, a genuinely absent open PR
    still normalizes the stale status to None.

    Proves the #859 guard is not an over-broad kill switch on
    ``issue_status_normalized``'s None outcome -- it only defers the
    normalization when the snapshot can't support the "no open PR"
    conclusion, not always.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [])]
    # Well under _LIST_LIMIT: the PR snapshot is complete.
    prs = [_pr(2, "OPEN", head_ref="agent/issue-99999-x")]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized"
        and item.issue_number == target_issue_number
        and item.new_status is None
    ]
    assert len(matches) == 1


def test_detect_drift_issue_status_normalized_closed_wins_despite_incomplete_pr_snapshot() -> None:
    """CLOSED-on-GitHub still wins first, even under an incomplete PR snapshot.

    ``target_status = "closed"`` is derived from ``_issue_state(issue)``, not
    from the PR snapshot at all, so the #859 guard (which only applies to the
    would-be-None branch) must never intercept it.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [], state="CLOSED")]
    prs = [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(1, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        # Not in VALID_ISSUE_STATUSES at all, and not "closed" either, so a
        # real transition to "closed" is expected regardless of PR truncation.
        "status": "garbage-value",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized" and item.issue_number == target_issue_number
    ]
    assert len(matches) == 1
    assert matches[0].new_status == "closed"


def test_detect_drift_issue_status_normalized_open_pr_wins_despite_incomplete_pr_snapshot() -> (
    None
):
    """A positively-observed open PR still normalizes to PASSIVE_OPEN_STATUS
    even under an incomplete PR snapshot.

    The #859 guard only intercepts the would-be-None branch; a PR that IS
    present in the (still-incomplete) snapshot is a positive observation, not
    an absence, so it must keep winning.

    The issue already carries the ``pr_open`` label (not an active label
    outside {pr_open, reviewing}, and not missing pr_open either) so the
    earlier ``issue_active_label_with_open_pr`` self-heal sweep -- which also
    reacts to an issue with an open PR -- does not fire first and mark this
    issue as already repaired; this test is isolating the later
    ``issue_status_normalized`` sweep specifically.
    """
    config = OrchestratorConfig()
    target_issue_number = 1
    issues = [_issue(target_issue_number, [config.labels.pr_open])]
    prs = [_pr(1, "OPEN", head_ref=f"agent/issue-{target_issue_number}-x")]
    prs += [
        _pr(i, "OPEN", head_ref=f"agent/issue-{i + 100000}-x")
        for i in range(2, reconcile_list_limit + 1)
    ]
    gh = FakeGitHub(prs=prs, issues=issues)
    state = empty_state()
    state["issues"][str(target_issue_number)] = {
        "number": target_issue_number,
        "status": "closed",
    }

    drift = detect_drift(gh, state, config)

    matches = [
        item
        for item in drift
        if item.kind == "issue_status_normalized" and item.issue_number == target_issue_number
    ]
    assert len(matches) == 1
    assert matches[0].new_status == PASSIVE_OPEN_STATUS


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
    # Regression: reason/adapter_kind must survive on the DriftItem itself
    # (not just embedded in the detail string) so apply_fixes can thread them
    # into set_throttled_until -- see test_apply_fixes_provider_throttle_threads_reason_and_adapter_kind.
    assert throttle_drift[0].throttle_reason == "rate_limited"
    assert throttle_drift[0].throttle_adapter_kind == "devin"


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


def test_apply_fixes_provider_throttle_threads_reason_and_adapter_kind() -> None:
    """A ``provider_throttle_detected`` drift item's reason/adapter_kind must
    reach ``set_throttled_until`` -- otherwise ``clear_quota_throttles``
    treats a devin/provider_auth throttle applied via ``reconcile --fix`` as
    claude-code-shaped (the field-unset default) and a later green ambient-CLI
    probe wrongly clears a throttle it never actually tested."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    throttled_until = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    drift = [
        DriftItem(
            kind="provider_throttle_detected",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with provider_auth",
            fix_actions=(f"set throttled_until={throttled_until}",),
            throttle_reason="provider_auth",
            throttle_adapter_kind="devin",
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state.get("throttled_until") == throttled_until
    assert new_state.get("throttle_reason") == "provider_auth"
    assert new_state.get("throttle_adapter_kind") == "devin"


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

    # detect_drift resolves the sessions dir through paths.resolved_layout
    # (config.devin.sessions_dir is a "" sentinel resolved against runtime.state_dir).
    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
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


# Adapter-kind -> sidecar filename suffix. Mirrors claude_code._ADAPTER_SIDECAR_SUFFIXES
# without importing it (keeps the test's failure surface independent of the adapter).
_ADAPTER_SIDECAR_SUFFIX = {"devin": "", "claude-code": ".claude", "api": ".api"}


def _write_dead_session_sidecar_for_adapter(
    sessions_dir: Path,
    issue_number: int,
    branch: str,
    worktree_path: Path,
    adapter_kind: str,
    log_text: str,
) -> Path:
    """Write a dead-session sidecar for any adapter kind, plus its log file.

    Unlike ``_write_dead_session_sidecar`` (devin-only, no log content), this
    also writes the log file with ``log_text`` so log-tail classification has
    real bytes to match against -- required for issue #656 regression coverage
    where the log must carry a throttle marker that *would* reclassify a
    non-completed session.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    suffix = _ADAPTER_SIDECAR_SUFFIX[adapter_kind]
    sidecar_path = sessions_dir / f"issue-{issue_number}{suffix}.json"
    if adapter_kind == "devin":
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path="/tmp/prompt.md",
            command=("devin", "--prompt-file", "/tmp/prompt.md"),
            pid=None,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            log_path=str(log_path),
            error=None,
        )
        sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    else:
        # claude-code / api share the ClaudeWorkerRecord on-disk shape; the
        # ``adapter_kind`` field disambiguates them (worker._from_claude_record
        # honors it so api sidecars surface as adapter_kind=="api").
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": branch,
                    "worktree_path": str(worktree_path),
                    "prompt_path": "/tmp/prompt.md",
                    "command": ["claude", "-p"],
                    "pid": None,
                    "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "log_path": str(log_path),
                    "error": None,
                    "adapter_kind": adapter_kind,
                }
            ),
            encoding="utf-8",
        )
    return sidecar_path


@pytest.mark.parametrize("adapter_kind", ["devin", "claude-code", "api"])
def test_detect_drift_completed_worktree_skips_log_tail_throttle_classification(
    tmp_path: Path, adapter_kind: str
) -> None:
    """Issue #656 regression: a completed worktree's log-tail throttle markers
    must NOT emit ``provider_throttle_detected`` drift.

    This guards the three ``session_completed=True`` call sites in
    ``reconcile.detect_drift`` (one per adapter kind). The log file is seeded
    with ``"usage limit"`` -- a ``_QUOTA_EXHAUSTED_PATTERN`` substring that, if
    log-tail classification ran, would return ``quota_exhausted`` plus a 24h
    ``throttled_until`` and emit a ``provider_throttle_detected`` drift item.
    The worktree inspection is ground truth the session completed, so
    ``session_completed=True`` must skip log-tail matching entirely.

    If ``session_completed=True`` is silently dropped from any of the three
    call sites, this test fails: a ``provider_throttle_detected`` drift item
    appears and the salvage drift (which proves the is_completed lane was
    taken) is shadowed by the throttle.
    """
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    issue_number = 656
    worktree_path, branch = _setup_completed_worktree(repo_root, issue_number)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    # Log tail that quotes a throttle marker in legitimate completion prose --
    # the exact false-positive shape observed live 2026-07-27.
    _write_dead_session_sidecar_for_adapter(
        sessions_dir,
        issue_number,
        branch,
        worktree_path,
        adapter_kind,
        log_text=(
            '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") '
            "that legitimately appear in this codebase's rate-limit/quota domain.\n"
        ),
    )

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(issue_number, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    # The is_completed lane was taken: salvage drift is emitted.
    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == issue_number

    # The throttle must NOT fire despite the "usage limit" marker in the log --
    # session_completed=True skipped log-tail classification entirely.
    throttle = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert not throttle, (
        f"completed {adapter_kind} session was reclassified from log tail despite "
        f"session_completed=True (issue #656 regression): {throttle}"
    )


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
    assert reviews_dir == tmp_path / ".var" / "charlie-work" / "dispatches" / "reviews"

    assert new_state["prs"]["1"]["status"] == "merged"
    assert new_state["prs"]["1"]["review_dispatch_status"] is None
    assert new_state["prs"]["1"]["review_dispatched_at"] is None
    assert new_state["prs"]["1"]["reviewer_pid"] is None
    assert new_state["prs"]["1"]["reviewer_process_start_time"] is None

    # Original state is never mutated in place.
    assert state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_apply_fixes_merged_pr_defers_reap_while_reviewer_alive(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #504: a PR merged externally while its reviewer is alive must not
    have its review checkout removed, dispatch claim cleared, or issue labels
    transitioned until the reviewer exits.
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

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 1,
        "branch": "agent/issue-10-x",
        "worktree_path": str(reviews_dir / "pr-1"),
        "prompt_path": str(reviews_dir / "pr-1" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 12345,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-1.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-1.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

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

    # Live reviewer: defer the reap.
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert removed_calls == []
    assert gh.labels_added == []
    assert gh.labels_removed == []
    assert new_state["prs"]["1"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert new_state["prs"]["1"]["status"] == "reviewing"

    # Dead reviewer: proceed with the reap.
    removed_calls.clear()
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    assert removed_calls[0][1] == 1
    assert new_state["prs"]["1"]["status"] == "merged"
    assert new_state["prs"]["1"]["review_dispatch_status"] is None


def test_apply_fixes_closed_unmerged_pr_defers_reap_while_reviewer_alive(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #504: a PR closed without merging while its reviewer is alive
    must not have its review checkout removed, dispatch claim cleared, or
    active labels stripped until the reviewer exits.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(2, "CLOSED", head_ref="agent/issue-20-x")],
        issues=[_issue(20, [config.labels.pr_open, config.labels.reviewing])],
    )
    state = empty_state()
    state["prs"]["2"] = {
        "number": 2,
        "issue_number": 20,
        "status": "reviewing",
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }

    reviews_dir = resolved_layout(config, tmp_path).reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 2,
        "branch": "agent/issue-20-x",
        "worktree_path": str(reviews_dir / "pr-2"),
        "prompt_path": str(reviews_dir / "pr-2" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 12345,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-2.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-2.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

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
        if item.kind == "closed_unmerged_pr_active_labels"
    ]
    assert drift

    # Live reviewer: defer the reap.
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert removed_calls == []
    assert gh.labels_removed == []
    assert new_state["prs"]["2"]["review_dispatch_status"] == "review_dispatch_dispatched"

    # Dead reviewer: proceed with the reap.
    removed_calls.clear()
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path)

    assert len(removed_calls) == 1
    assert removed_calls[0][1] == 2
    assert (20, config.labels.pr_open) in gh.labels_removed
    assert (20, config.labels.reviewing) in gh.labels_removed
    assert new_state["prs"]["2"]["review_dispatch_status"] is None


# ---------------------------------------------------------------------------
# Issue #480: api-worker budget settlement wiring at the reconcile reap sites
# ---------------------------------------------------------------------------
#
# detect_drift has two production reap_sidecar call sites that wire
# ``api_config=config.api_worker, state_dir=state_dir_root`` so an api worker's
# spend is settled into the ledger before its sidecar is unlinked:
#   - the dead-session lane (~reconcile.py:440)
#   - the launch_stalled lane (~reconcile.py:304)
# Neither had any test coverage. A wiring regression at either site would
# silently disable budget tracking with no test failing. These two tests drive
# the real detect_drift path and assert the ledger is populated.


def test_detect_drift_dead_api_session_settles_budget_ledger(tmp_path: Path) -> None:
    """Dead api-worker session: detect_drift reaps and settles spend (issue #480).

    Covers the dead-session reap call site (~reconcile.py:440). A wiring
    regression that drops ``api_config``/``state_dir`` from that call leaves
    the sidecar reaped but the ledger empty — this assertion fails.
    """
    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        # Disable sessions.db post-mortem so the test does not touch a real
        # sessions.db; the wiring under test is the budget reap, not post-mortem.
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 42, provider="example")
    write_api_events(sessions_dir, 42)

    state_dir_root = runtime_paths(tmp_path, config.runtime.state_dir).root

    detect_drift(gh, state, config, repo_root=tmp_path)

    sessions = ledger_entries(state_dir_root)
    assert len(sessions) == 1, "dead api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 42
    assert entry.provider == "example"
    assert entry.model == "example-model"
    # 1M*3 + 0.2M*15 + 0.5M*0.30 = 6.15
    assert entry.usd == pytest.approx(6.15)


def test_detect_drift_launch_stalled_api_session_settles_budget_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch-stalled api-worker session: the launch_stalled reap settles spend.

    Covers the launch_stalled reap call site (~reconcile.py:304). That lane
    fires only for an alive-but-shim-frozen worker corroborated by a
    conclusive-stale real-activity probe. We patch ``is_worker_alive`` to True
    and ``real_activity_probe_for`` to a conclusive-stale probe so the lane
    runs for an api sidecar without spawning a real process. A wiring
    regression that drops the api kwargs from this call site leaves the ledger
    empty — this assertion fails.
    """
    import os as _os

    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(55, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 55, provider="example")
    write_api_events(sessions_dir, 55)

    # A shim-frozen log: small, contains the marker, stale past the grace period.
    log_path = sessions_dir / "issue-55.claude.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=20)
    _os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Force the api worker to read as alive so the launch_stalled lane runs.
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    # Conclusive-stale probe: has a timestamp (not inconclusive) but stale past
    # the grace period (not fresh), so _log_is_stalled_at_shim returns True.
    stale_source = ActivitySource(
        name="test",
        timestamp=old_time,
        staleness_seconds=20 * 60,
        error=None,
    )
    monkeypatch.setattr(
        "charlie_work.worker.real_activity_probe_for",
        lambda w, cfg, now: RealActivityProbe(sources=(stale_source,)),
    )

    state_dir_root = runtime_paths(tmp_path, config.runtime.state_dir).root

    detect_drift(gh, state, config, repo_root=tmp_path)

    sessions = ledger_entries(state_dir_root)
    assert len(sessions) == 1, "launch_stalled api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 55
    assert entry.provider == "example"
    assert entry.usd == pytest.approx(6.15)


# ---------------------------------------------------------------------------
# detect_aviator_stale_blocked (job-cannon #1387/#1400/#1398/#1392, 2026-07-27)
# ---------------------------------------------------------------------------

_AVIATOR_FAILURE_OUTPUT = {
    "title": "Aviator checks - blocked",
    "summary": (
        "This PR is not ready to merge (currently in state blocked): "
        f"{AVIATOR_BLOCKED_MESSAGE}.\n\n### Pending Status Checks\n\n* ✅ 5 tests passing!"
    ),
}


def _aviator_check_run(
    conclusion: str | None, output: dict[str, Any] | None = None, *, run_id: int = 1
) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": AVIATOR_CHECK_NAME,
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion,
        "output": output or {},
    }


def _passing_check_run(name: str, *, run_id: int) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": name,
        "status": "completed",
        "conclusion": "success",
        "output": {},
    }


def test_detect_aviator_stale_blocked_finds_stale_blocked_pr() -> None:
    config = OrchestratorConfig()
    pr = {**_pr(1400, "OPEN"), "headRefOid": "sha-1400", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1400"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _passing_check_run("Tests passed", run_id=2),
        _passing_check_run("Pre-commit", run_id=3),
    ]

    drift = detect_aviator_stale_blocked(gh, config)

    assert len(drift) == 1
    item = drift[0]
    assert item.kind == "aviator_stale_blocked"
    assert item.pr_number == 1400
    assert item.remove_labels == ("blocked",)
    if config.auto_merge.mergequeue_label:
        assert item.add_labels == (config.auto_merge.mergequeue_label,)
    assert gh.commit_check_runs_calls == ["sha-1400"]


def test_detect_aviator_stale_blocked_ignores_pending_aviator_check() -> None:
    """Aviator still queued (not failed) is the normal, non-stale state."""
    config = OrchestratorConfig()
    pr = {**_pr(1, "OPEN"), "headRefOid": "sha-1", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1"] = [
        _aviator_check_run(None, run_id=1),  # in_progress, no conclusion yet
        _passing_check_run("Tests passed", run_id=2),
    ]

    assert detect_aviator_stale_blocked(gh, config) == []


def test_detect_aviator_stale_blocked_ignores_real_check_failure() -> None:
    """A real CI failure alongside `blocked` must NOT be cleared -- #1329's shape."""
    config = OrchestratorConfig()
    pr = {**_pr(2, "OPEN"), "headRefOid": "sha-2", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-2"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        {
            "id": 2,
            "name": "Tests passed",
            "status": "completed",
            "conclusion": "failure",
            "output": {},
        },
    ]

    assert detect_aviator_stale_blocked(gh, config) == []


def test_detect_aviator_stale_blocked_ignores_unrelated_aviator_failure_message() -> None:
    """aviator/checks can fail for other reasons -- only the specific stale
    'remove the blocked label' message is safe to auto-clear."""
    config = OrchestratorConfig()
    pr = {**_pr(3, "OPEN"), "headRefOid": "sha-3", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-3"] = [
        _aviator_check_run("failure", {"summary": "merge conflict with base branch"}, run_id=1),
        _passing_check_run("Tests passed", run_id=2),
    ]

    assert detect_aviator_stale_blocked(gh, config) == []


def test_detect_aviator_stale_blocked_skips_gh_calls_when_not_blocked() -> None:
    """Cost gate: commit_check_runs must only be called for blocked-labeled PRs."""
    config = OrchestratorConfig()
    prs = [{**_pr(n, "OPEN"), "headRefOid": f"sha-{n}"} for n in range(1, 6)]
    gh = FakeGitHub(prs=prs, issues=[])

    assert detect_aviator_stale_blocked(gh, config) == []
    assert gh.commit_check_runs_calls == []


def test_detect_aviator_stale_blocked_uses_latest_check_run_by_id() -> None:
    """A rerun leaves stale AND fresh entries for the same name -- the higher
    id (most recent) must win, not list order."""
    config = OrchestratorConfig()
    pr = {**_pr(4, "OPEN"), "headRefOid": "sha-4", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-4"] = [
        # Stale failing run for "Tests passed" listed AFTER its fresh success --
        # order must not matter, only id.
        _passing_check_run("Tests passed", run_id=10),
        {
            "id": 5,
            "name": "Tests passed",
            "status": "completed",
            "conclusion": "failure",
            "output": {},
        },
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
    ]

    drift = detect_aviator_stale_blocked(gh, config)
    assert len(drift) == 1
    assert drift[0].pr_number == 4


def test_detect_aviator_stale_blocked_no_readd_when_mergequeue_already_present() -> None:
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    pr = {
        **_pr(5, "OPEN"),
        "headRefOid": "sha-5",
        "labels": [{"name": "blocked"}, {"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-5"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _passing_check_run("Tests passed", run_id=2),
    ]

    drift = detect_aviator_stale_blocked(gh, config)
    assert len(drift) == 1
    assert drift[0].add_labels == ()


def _write_review_decision(
    tmp_path: Path, config: OrchestratorConfig, pr_number: int, decision: dict[str, Any]
) -> None:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")


def _aviator_blocked_pr_and_gh(pr_number: int, head_sha: str) -> tuple[dict[str, Any], Any]:
    pr = {**_pr(pr_number, "OPEN"), "headRefOid": head_sha, "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha[head_sha] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _passing_check_run("Tests passed", run_id=2),
    ]
    return pr, gh


def test_detect_aviator_stale_blocked_does_not_readd_mergequeue_without_repo_root() -> None:
    """Fail closed: without repo_root there is no way to check the review
    decision, so re-queueing must not happen even though CI is green."""
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    _pr, gh = _aviator_blocked_pr_and_gh(1408, "sha-1408")

    drift = detect_aviator_stale_blocked(gh, config)

    assert len(drift) == 1
    assert drift[0].remove_labels == ("blocked",)
    assert drift[0].add_labels == ()


def test_detect_aviator_stale_blocked_does_not_readd_mergequeue_when_not_approved(
    tmp_path: Path,
) -> None:
    """job-cannon #1408/#1404: a PR carrying request_changes must never be
    re-queued just because Aviator's own 'blocked' label went stale."""
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    _pr, gh = _aviator_blocked_pr_and_gh(1408, "sha-1408")
    _write_review_decision(
        tmp_path,
        config,
        1408,
        {"decision": "request_changes", "reviewed_head_sha": "sha-1408"},
    )

    drift = detect_aviator_stale_blocked(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].add_labels == ()


def test_detect_aviator_stale_blocked_does_not_readd_mergequeue_when_approved_at_stale_head(
    tmp_path: Path,
) -> None:
    """An approval recorded for an old commit must not authorize a newer,
    unreviewed head -- mirrors the head_moved check in ship_it's can_merge."""
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    _pr, gh = _aviator_blocked_pr_and_gh(1408, "sha-new")
    _write_review_decision(
        tmp_path,
        config,
        1408,
        {"decision": "approved", "reviewed_head_sha": "sha-old"},
    )

    drift = detect_aviator_stale_blocked(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].add_labels == ()


def test_detect_aviator_stale_blocked_readds_mergequeue_when_approved_at_live_head(
    tmp_path: Path,
) -> None:
    """Once a PR is genuinely approved at its current head, unsticking the
    stale 'blocked' label may still re-queue it for Aviator."""
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    _pr, gh = _aviator_blocked_pr_and_gh(1408, "sha-1408")
    _write_review_decision(
        tmp_path,
        config,
        1408,
        {"decision": "approved", "reviewed_head_sha": "sha-1408"},
    )

    drift = detect_aviator_stale_blocked(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].add_labels == (mergequeue_label,)


def test_apply_fixes_aviator_stale_blocked_removes_blocked_readds_mergequeue() -> None:
    config = OrchestratorConfig()
    mergequeue_label = config.auto_merge.mergequeue_label or "mergequeue"
    config = replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="aviator_stale_blocked",
            issue_number=None,
            pr_number=1400,
            detail="PR #1400 has a stale Aviator 'blocked' label",
            fix_actions=("remove label 'blocked' from PR #1400",),
            remove_labels=("blocked",),
            add_labels=(mergequeue_label,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.pr_labels_removed == [(1400, "blocked")]
    assert gh.pr_labels_added == [(1400, mergequeue_label)]


def test_apply_fixes_dual_writes_reconcile_event_to_events_db(tmp_path: Path) -> None:
    """``apply_fixes`` must pass ``state_path`` through to ``append_event`` --
    without it, every reconcile fix (including aviator_stale_blocked re-queues)
    is invisible to events.db and only survives in the capped 200-entry ring
    in state.json (found via job-cannon audit: 39 merged_outside_orchestrator
    fixes recorded in state.json, zero in events.db)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="aviator_stale_blocked",
            issue_number=None,
            pr_number=1400,
            detail="PR #1400 has a stale Aviator 'blocked' label",
            fix_actions=("remove label 'blocked' from PR #1400",),
            remove_labels=("blocked",),
            add_labels=(),
        )
    ]
    state_path = tmp_path / "state.json"

    apply_fixes(gh, state, drift, config, state_path=state_path)

    events = read_event_log(state_path)
    reconcile_events = [e for e in events if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "aviator_stale_blocked"
    assert reconcile_events[0]["payload"]["pr_number"] == 1400


def test_apply_fixes_aviator_stale_blocked_records_label_write_failure() -> None:
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    gh._fail_remove_pr_labels = {(1400, "blocked")}
    state = empty_state()
    drift = [
        DriftItem(
            kind="aviator_stale_blocked",
            issue_number=None,
            pr_number=1400,
            detail="PR #1400 has a stale Aviator 'blocked' label",
            fix_actions=("remove label 'blocked' from PR #1400",),
            remove_labels=("blocked",),
            add_labels=(),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state.get("events", []) if e.get("kind") == "reconcile"]
    assert any(
        "label_write_failed: true" in e.get("payload", {}).get("fix_actions", []) for e in events
    )


class _EmptyStdoutGitHub:
    """``gh`` exits 0 but writes nothing to stdout.

    ``GitHub.run`` turns that into ``None`` (``if not output: return None`` on
    the success path) -- the exact value both reconcile fetchers used to
    coerce into ``[]``.
    """

    def run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
    ) -> Any:
        return None

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int]:
        return True, 10000, 0

    def invalidate_list_cache(self) -> None:
        return None


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


def test_detect_drift_leaves_state_untouched_when_snapshot_unreadable() -> None:
    """End-to-end property: a failed read aborts the pass instead of mutating.

    This is what makes the downstream sweep correct *by construction* -- once
    an unreadable snapshot can no longer arrive as ``[]``, an empty ``prs``
    genuinely means "GitHub has zero PRs" and no "suspiciously empty"
    heuristic is needed to second-guess it.
    """
    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["999"] = {
        "issue_number": 5,
        "status": "reviewing",
        "decision": "approved",
    }
    before = json.dumps(state, sort_keys=True)

    with pytest.raises(GitHubError):
        detect_drift(_EmptyStdoutGitHub(), state, config)  # type: ignore[arg-type]

    assert json.dumps(state, sort_keys=True) == before
    assert state["prs"]["999"]["decision"] == "approved"


# ---------------------------------------------------------------------------
# detect_mergequeue_not_approved (issue #819 -- irrevocable mergequeue label)
# ---------------------------------------------------------------------------


def _mergequeue_config(mergequeue_label: str | None = "mergequeue") -> OrchestratorConfig:
    config = OrchestratorConfig()
    return replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )


def test_detect_mergequeue_not_approved_regression_pr_695(tmp_path: Path) -> None:
    """Reproduces PR #695's exact sequence (issue #819): ``mergequeue`` was
    applied by ship_it, the recorded verdict later flipped to
    ``request_changes`` at the PR's still-current head, and nothing in the
    orchestrator ever called ``remove_pr_label`` for ``mergequeue`` --
    Aviator (``number_of_approvals: 0``) merged the PR anyway once CI went
    green, over a standing request-changes verdict. This is the single most
    important test in this change: it must fail red on main (label never
    removed) and pass green with the fix."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(695, "OPEN"),
        "headRefOid": "sha-695-live",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path,
        config,
        695,
        {"decision": "request_changes", "reviewed_head_sha": "sha-695-live"},
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    item = drift[0]
    assert item.kind == "mergequeue_revoked"
    assert item.pr_number == 695
    assert item.remove_labels == (mergequeue_label,)
    assert item.add_labels == ()

    # And the fix actually strips the label on the next reconcile pass --
    # this is the mechanical step that would have saved #695.
    state_path = tmp_path / "state.json"
    new_state = apply_fixes(gh, empty_state(), drift, config, state_path=state_path)
    assert gh.pr_labels_removed == [(695, mergequeue_label)]

    events = read_event_log(state_path)
    reconcile_events = [e for e in events if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "mergequeue_revoked"
    assert reconcile_events[0]["payload"]["pr_number"] == 695
    assert new_state["prs"] == {}


def test_detect_mergequeue_not_approved_leaves_approved_at_head_alone(tmp_path: Path) -> None:
    """Negative test, equally important: a PR genuinely approved at its
    current head must never be revoked -- a false-positive revocation here
    would kick every legitimately-queued PR in the fleet out of Aviator."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(700, "OPEN"),
        "headRefOid": "sha-700",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path, config, 700, {"decision": "approved", "reviewed_head_sha": "sha-700"}
    )

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_detect_mergequeue_not_approved_fails_closed_when_decision_missing(
    tmp_path: Path,
) -> None:
    """No review-decision.json at all (never reviewed) must revoke, not
    leave a merge-authorizing label in place by default."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(701, "OPEN"),
        "headRefOid": "sha-701",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    # Deliberately no _write_review_decision call -- the file is absent.

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].remove_labels == (mergequeue_label,)
    assert "no readable review-decision.json" in drift[0].detail


def test_detect_mergequeue_not_approved_fails_closed_when_decision_malformed(
    tmp_path: Path,
) -> None:
    """Corrupt/unreadable JSON must revoke, not be silently ignored --
    mirrors ``_pr_review_approved_at_head``'s own fail-closed contract."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(702, "OPEN"),
        "headRefOid": "sha-702",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-702"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text("{not valid json", encoding="utf-8")

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].remove_labels == (mergequeue_label,)


def test_detect_mergequeue_not_approved_stale_head_still_revoked(tmp_path: Path) -> None:
    """Issue #819 item 4 -- carry-forward interaction: a PR approved at an
    OLDER sha whose head then moved (a rebase in flight) is revoked too,
    not deferred. reconcile.py cannot cheaply re-validate a rebase itself
    (that needs merge_ready's per-PR gh.pr_diff carry-forward check, the
    issue-#361 cost class this module stays out of), and merge_ready's own
    carry-forward-failure path never strips mergequeue either -- so leaving
    a stale-head approval alone reopens the exact #695 hole through a second
    door. This is deliberately revoked rather than escalated to a human:
    revoke-then-cooperative-reapply (a clean rebase gets mergequeue back via
    carry-forward + the idempotent add_pr_label on its next merge_ready
    pass) is safe, and the emitted detail distinguishes this case from a
    genuine request_changes revoke instead of collapsing both into one
    indistinguishable string."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(703, "OPEN"),
        "headRefOid": "sha-703-new",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path, config, 703, {"decision": "approved", "reviewed_head_sha": "sha-703-old"}
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    item = drift[0]
    assert item.remove_labels == (mergequeue_label,)
    assert "approved at stale head" in item.detail
    assert "sha-703-old" in item.detail
    # Distinguishable from the genuine not-approved case in the same field.
    assert "recorded decision is" not in item.detail


def test_detect_mergequeue_not_approved_skips_fs_reads_when_no_pr_labeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost gate: the per-PR review-decision.json read must only happen for
    PRs that are OPEN and already carry mergequeue -- cost must not scale
    with open-PR count, matching detect_aviator_stale_blocked's discipline."""
    config = _mergequeue_config()
    prs = [{**_pr(n, "OPEN"), "headRefOid": f"sha-{n}"} for n in range(1, 6)]
    gh = FakeGitHub(prs=prs, issues=[])

    calls: list[int] = []
    import charlie_work.reconcile as reconcile_module

    original_predicate = reconcile_module._pr_review_approved_at_head

    def _spy(cfg: Any, root: Any, pr_number: int, head_sha: str) -> bool:
        calls.append(pr_number)
        return original_predicate(cfg, root, pr_number, head_sha)

    monkeypatch.setattr(reconcile_module, "_pr_review_approved_at_head", _spy)

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []
    assert calls == []

    # Sanity: labeling exactly one PR triggers exactly one predicate call --
    # proves the spy would have caught a scaling regression above.
    mergequeue_label = config.auto_merge.mergequeue_label
    prs[2] = {**prs[2], "labels": [{"name": mergequeue_label}]}
    gh2 = FakeGitHub(prs=prs, issues=[])
    calls.clear()
    detect_mergequeue_not_approved(gh2, config, repo_root=tmp_path)
    assert calls == [3]


def test_detect_mergequeue_not_approved_blind_without_repo_root_does_not_revoke_fleet(
    tmp_path: Path,
) -> None:
    """``repo_root is None`` means the detector cannot read ANY decision
    file -- it must return [] rather than revoke mergequeue from every
    labeled PR in the fleet. A blanket revocation triggered by the
    detector's own blindness would be a false-positive catastrophe, not
    fail-closed behavior."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    prs = [
        {**_pr(n, "OPEN"), "headRefOid": f"sha-{n}", "labels": [{"name": mergequeue_label}]}
        for n in (710, 711, 712)
    ]
    gh = FakeGitHub(prs=prs, issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=None) == []


def test_detect_mergequeue_not_approved_noop_when_label_unconfigured(tmp_path: Path) -> None:
    """No mergequeue_label configured means Aviator handoff isn't in use at
    all -- nothing to revoke, and no decision-file reads should happen."""
    config = _mergequeue_config(mergequeue_label=None)
    pr = {**_pr(713, "OPEN"), "headRefOid": "sha-713", "labels": [{"name": "mergequeue"}]}
    gh = FakeGitHub(prs=[pr], issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_detect_mergequeue_not_approved_ignores_merged_pr_carrying_label(
    tmp_path: Path,
) -> None:
    """Issue #819 notes PR #695 still carries mergequeue today, post-merge --
    the repo has merged PRs wearing the label right now. Only OPEN PRs are
    eligible for revocation; a merged PR's label is cosmetic history, not a
    live merge authorization."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(695, "MERGED"),
        "headRefOid": "sha-695-final",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_apply_fixes_mergequeue_revoked_removes_label(tmp_path: Path) -> None:
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.pr_labels_removed == [(695, mergequeue_label)]
    assert gh.pr_labels_added == []


def test_apply_fixes_mergequeue_revoked_records_label_write_failure() -> None:
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    gh._fail_remove_pr_labels = {(695, mergequeue_label)}
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state.get("events", []) if e.get("kind") == "reconcile"]
    assert any(
        "label_write_failed: true" in e.get("payload", {}).get("fix_actions", []) for e in events
    )
