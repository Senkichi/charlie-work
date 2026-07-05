from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig
from charlie_work.github import _LIST_LIMIT as github_list_limit
from charlie_work.reconcile import (
    DriftItem,
    _LIST_LIMIT as reconcile_list_limit,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import empty_state, is_claim_stale


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
    assert matches[0].remove_labels == (config.labels.in_progress,)


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

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect both provider throttle and session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 1
    assert relabel_drift[0].issue_number == 42
    assert config.labels.in_progress in relabel_drift[0].remove_labels
    assert any(
        f"add label '{config.labels.ready}'" in action for action in relabel_drift[0].fix_actions
    )


def test_detect_drift_session_failed_with_open_pr_no_relabel(tmp_path: Path) -> None:
    """Issue #118: dead session with open PR should NOT trigger label reconciliation."""
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

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect provider throttle but NOT session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 0


def test_apply_fixes_session_failed_relabeled(tmp_path: Path) -> None:
    """Issue #118: apply_fixes should remove active labels and add ready label."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a session_failed_relabeled drift item
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

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled but NOT add ready label action
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 1
    assert relabel_drift[0].issue_number == 42
    assert config.labels.in_progress in relabel_drift[0].remove_labels
    # Should not have add ready label action since it's already present
    assert not any("add label" in action for action in relabel_drift[0].fix_actions)
