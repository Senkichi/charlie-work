"""Tests for issue #1229: issue-less rework episodes keyed by branch-name
number collide with unrelated PRs.

Root cause: ``linked_issue_number`` trusts a branch-name-derived issue number
unconditionally. A branch ``agent/issue-709-…`` left over from a merged
PR #709, reused by an unrelated issue-less PR (e.g. PR #1660), silently
binds the PR to issue 709. When the PR is routed to rework, the episode is
keyed under ``state["issues"]["709"]``, colliding with the unrelated
issue/PR #709's lifecycle.

Fix: ``linked_issue_number`` now accepts an optional
``branch_issue_validator`` callable. When the branch-name path produces a
candidate, the validator is called; if it returns False (the number is not
a real open issue), the binding is rejected and the function falls through
to the closing-keyword path. The rework-routing call sites
(``merge_ready``, ``review``, ``_dispatch_rework_impl``, ``loop``,
``record_review``) pass a validator built from ``issue_list(state="open")``
so a stale branch-name number can never key a rework episode under
``issues[<n>]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import AutoMergeConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _stale_branch_pr() -> list[dict]:
    """A same-repo PR with a stale branch name ``agent/issue-709-…``.

    Issue #709 does not exist in the fake's issue list (it was merged long
    ago), so the branch-name validator must reject the binding. The PR body
    has no closing keyword, so the fall-through path also returns None —
    the PR is correctly treated as issue-less.
    """
    return [
        {
            "number": 1660,
            "title": "docs: update orchestrator docs",
            "url": "https://example.test/pull/1660",
            "headRefName": "agent/issue-709-job-cannon-docs-devin-orchestration",
            "baseRefName": "main",
            "headRefOid": "sha-stale-branch",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Docs maintenance. No issue — issue-less orchestrator rework.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]


def test_merge_ready_stale_branch_name_does_not_key_rework_under_wrong_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not create a rework episode
    under ``state["issues"]["709"]``.

    The PR has branch ``agent/issue-709-…`` but issue #709 is not open (not
    in the fake's issue list). Without the fix, ``linked_issue_number``
    would return 709 and the rework episode would be keyed under
    ``state["issues"]["709"]``, colliding with the unrelated PR #709. With
    the fix, the validator rejects 709, ``linked_issue_number`` returns
    None, and the PR is treated as issue-less — the same path as a fork PR
    with no linked issue.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default FakeGitHub has issue #123 (OPEN) and PR #456. Replace
    # the PR list with our stale-branch PR. Issue #709 is NOT in
    # fake_gh.issues, so the validator will reject 709.
    fake_gh.prs = _stale_branch_pr()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record an approved verdict so merge_ready enters the approved path.
    app.record_review(1660, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(1660, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "no linked issue, cannot route to rework" in warning
    assert result.data["issue"] is None

    # The critical assertion: no rework episode under issues["709"].
    state = load_state(paths.state_file)
    assert "709" not in state.get("issues", {})
    # No rework event was emitted.
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state.get("events", []))
    # No rework prompt was written.
    assert not (paths.prs / "pr-1660" / "rework-prompt.md").exists()


def test_make_branch_issue_validator_rejects_closed_or_nonexistent_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: ``_make_branch_issue_validator`` returns False for issue
    numbers that are not in the open-issue list."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Default issues: [{number: 123, state: OPEN}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    validator = app._make_branch_issue_validator()
    assert validator is not None
    assert validator(123) is True  # real open issue
    assert validator(709) is False  # not in the open-issue list
    assert validator(999) is False  # doesn't exist at all


def test_make_branch_issue_validator_returns_none_on_api_failure(
    tmp_path: Path,
) -> None:
    """Issue #1229: when ``issue_list`` raises ``GitHubError``, the validator
    returns None so callers skip validation (preserve existing behavior)
    rather than blocking all rework routing during a transient outage."""
    from charlie_work.github import GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class BrokenGitHub(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            raise GitHubError("simulated API outage")

    app = OrchestratorApp(tmp_path, paths, config, BrokenGitHub())
    assert app._make_branch_issue_validator() is None


# ---------------------------------------------------------------------------
# Issue #1229 rework: the literal scenario is a *closed/merged* issue number
# (PR #709 merged 2026-07-03), not merely an absent one. The shared
# FakeGitHub.issue_list must honor ``state="open"`` so the validator can
# distinguish "closed" from "open"; without that, a closed issue present in
# ``self.issues`` is wrongly accepted as open and the stale branch name binds.
# ---------------------------------------------------------------------------


def _fake_with_closed_709() -> FakeGitHub:
    """A FakeGitHub seeded with issue #709 in CLOSED state (merged long ago).

    Issue #709 is present in ``self.issues`` (so it is not "absent from the
    list") but carries ``state="CLOSED"``. The branch-issue validator calls
    ``issue_list(state="open")``, which must filter #709 out — only then does
    the validator reject 709 and prevent the stale branch name from binding.
    """
    fake = FakeGitHub()
    fake.issues.append(
        {
            "number": 709,
            "title": "Old work merged long ago",
            "url": "https://example.test/issues/709",
            "body": "shipped",
            "labels": [],
            "state": "CLOSED",
        }
    )
    return fake


def test_make_branch_issue_validator_rejects_closed_issue_present_in_list(
    tmp_path: Path,
) -> None:
    """Issue #1229 literal scenario: a CLOSED issue #709 present in the issue
    list must be rejected by the validator.

    The pre-existing ``test_make_branch_issue_validator_rejects_closed_or_nonexistent_issue``
    only covers an *absent* #709. This test plants #709 as CLOSED and asserts
    the validator still returns False — which requires ``issue_list(state="open")``
    to honor the state filter (the FakeGitHub fix). Against the un-fixed fake,
    ``issue_list(state="open")`` returns the closed #709 too, the validator
    accepts 709, and a stale ``agent/issue-709-…`` branch silently binds.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _fake_with_closed_709()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    validator = app._make_branch_issue_validator()
    assert validator is not None
    assert validator(123) is True  # genuinely open
    assert validator(709) is False  # closed — must be rejected
    assert validator(999) is False  # absent


def test_merge_ready_stale_branch_name_closed_issue_does_not_key_rework(
    tmp_path: Path,
) -> None:
    """Issue #1229 literal scenario: a stale branch ``agent/issue-709-…`` whose
    number matches a CLOSED issue #709 must not key a rework episode under
    ``state["issues"]["709"]``.

    Unlike the sibling ``test_merge_ready_stale_branch_name_does_not_key_rework_under_wrong_issue``
    (which relies on #709 being absent), this test plants #709 as CLOSED. With
    the FakeGitHub ``state`` fix, ``issue_list(state="open")`` excludes #709,
    the validator rejects 709, ``linked_issue_number`` returns None, and the PR
    is treated as issue-less. Against the un-fixed fake, the validator accepts
    709 and the rework episode is keyed under the unrelated closed issue.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _fake_with_closed_709()
    fake_gh.prs = _stale_branch_pr()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record an approved verdict so merge_ready enters the approved path.
    app.record_review(1660, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(1660, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "no linked issue, cannot route to rework" in warning
    assert result.data["issue"] is None

    # The critical assertion: no rework episode under issues["709"].
    state = load_state(paths.state_file)
    assert "709" not in state.get("issues", {})
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state.get("events", []))
    assert not (paths.prs / "pr-1660" / "rework-prompt.md").exists()


# ---------------------------------------------------------------------------
# Issue #1229 rework: the same untrusted-branch-name root cause remained live
# at two module-level sweeps that gate issue-keyed logic from branch names.
# These tests pin the validator threading through both call sites.
# ---------------------------------------------------------------------------


def _stale_branch_open_pr() -> dict:
    """An OPEN, same-repo PR with a stale branch name ``agent/issue-709-…``.

    Non-conflicting mergeable state and a non-empty statusCheckRollup so it is
    NOT a pre-review rework candidate — keeping the orphaned-worker test on the
    drift path rather than the pre-review routing path.
    """
    return {
        "number": 1660,
        "title": "docs: update orchestrator docs",
        "url": "https://example.test/pull/1660",
        "headRefName": "agent/issue-709-job-cannon-docs-devin-orchestration",
        "baseRefName": "main",
        "headRefOid": "sha-stale-branch",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "Tests passed", "status": "COMPLETED"}],
        "body": "Docs maintenance. No issue — issue-less orchestrator rework.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def test_classify_dead_session_stale_branch_does_not_mask_escalation(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not populate
    ``open_prs_by_issue[709]`` and mask the escalation/salvage-skip guard in
    ``_classify_dead_sessions_and_update_throttle_state``.

    Scenario: a launch-failed rework worker is keyed under issue #709 (the
    collision key from the stale branch name). Issue #709 is CLOSED. An
    unrelated issue-less PR #1660 carries the stale branch
    ``agent/issue-709-…``. Without the validator, ``linked_issue_number``
    binds PR #1660 to 709, so ``open_prs_by_issue[709]`` is non-empty and the
    guard ``w.issue_number not in open_prs_by_issue`` is False — escalation is
    skipped for a session that has no real open PR. With the validator, 709 is
    closed so the binding is rejected, the guard fires, and the dead session
    escalates as it should.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = _fake_with_closed_709()
    fake_gh.prs = [_stale_branch_open_pr()]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["709"] = {
            "number": 709,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-709-job-cannon-docs-devin-orchestration",
            "redispatch_at": [],
        }
        save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-709.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-709.json"
    record = SessionRecord(
        issue_number=709,
        branch="agent/issue-709-job-cannon-docs-devin-orchestration",
        worktree_path=str(tmp_path / "worktrees" / "agent-709"),
        prompt_path=str(paths.prs / "pr-1660" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # Launch failure -- process never started
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["709"]
    # The guard fired and escalated the dead session.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worktree_unsafe"
    # Mechanical escalation lands on operator_queue (issue #1266).
    assert (709, config.labels.operator_queue) in fake_gh.labels_added
    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 709]
    assert "session_failed_escalated" in event_kinds


def test_orphaned_worker_stale_branch_does_not_bind_unrelated_pr(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not bind an unrelated PR to a
    closed issue in ``_detect_and_handle_orphaned_workers``'s ``pr_by_issue``.

    Scenario: state.json carries a dead dispatched worker keyed under issue
    #709 (the collision key). Issue #709 is CLOSED. An unrelated issue-less PR
    #1660 carries the stale branch ``agent/issue-709-…``. Without the
    validator, ``pr_by_issue[709] = PR#1660`` and the orphan is routed down
    the PR-linked drift path, emitting an ``orphaned_worker_drift`` event that
    references ``pr_number=1660`` — acting on the wrong subject. With the
    validator, 709 is closed so the binding is rejected, the orphan is treated
    as a no-open-PR orphan, and no event references the unrelated PR.
    """
    from unittest.mock import patch

    from charlie_work.config import WatchdogConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = _fake_with_closed_709()
    fake_gh.prs = [_stale_branch_open_pr()]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["709"] = {
            "number": 709,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
        save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=_wg(paths.state_file),
        )

    state = load_state(paths.state_file)
    # No drift/recovery event for issue 709 may reference the unrelated PR
    # #1660 — that is the wrong-subject binding the validator prevents.
    wrong_subject_events = [
        e
        for e in state.get("events", [])
        if e["payload"].get("issue_number") == 709 and e["payload"].get("pr_number") == 1660
    ]
    assert wrong_subject_events == [], (
        f"stale branch name bound unrelated PR #1660 to closed issue #709: "
        f"{[e['kind'] for e in wrong_subject_events]}"
    )
