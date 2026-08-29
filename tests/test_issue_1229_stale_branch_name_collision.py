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


def test_dispatch_claim_stale_branch_does_not_bind_unrelated_pr_to_closed_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not bind an unrelated PR to a
    closed issue in the dispatch-claim path's ``pr_by_issue`` construction.

    This test exercises ``_dispatch_impl``'s REAL (non-dry-run) dispatch-claim
    branch (workflow.py:5557 onward -- the ``pr_by_issue`` construction at
    ~5598, the claim phase that stamps ``state["issues"][n]`` to
    ``dispatch_pending``, and the post-launch label transition to
    ``agent:queued``), NOT the ``dry_run=True`` early-return branch (~5315,
    which is read-only and never claims an issue or touches a label). A
    dry-run-only test would assert the validator threading via a spy but could
    not prove the real claim/launch path ran -- the dry-run branch returns
    before the claim state write and the label transition, so neither is
    observable. Running with the default (non-dry-run) ``OrchestratorApp``
    makes the state/label transitions the proof that the real branch executed.

    Scenario: issue #709 is CLOSED but carries the ready label, so it appears in
    the dispatch ``issues`` list (``issue_list(state="all")``, issue #427 -- the
    ready label is intentionally not cleaned for closed issues so externally-
    merged PRs can be finalized). An unrelated issue-less PR #1660 carries the
    stale branch ``agent/issue-709-…``. Without the validator, the dispatch-
    claim ``pr_by_issue`` construction binds PR #1660 to 709, polluting
    ``issues_with_open_tracked_prs`` with a phantom closed-issue binding. With
    the validator, 709 is closed so the binding is rejected and ``pr_by_issue``
    stays free of the phantom binding. A real open dispatchable issue (#123) is
    still claimed for dispatch -- proven here by the actual state transition
    (``state["issues"]["123"]`` stamped with a dispatch status) and the
    ``agent:queued`` label transition, neither of which the dry-run branch
    performs.

    Note on the validator's role here: ``_is_dispatchable`` independently
    excludes closed issues, so the validator does not change whether closed
    #709 is dispatched -- its job in this path is to keep ``pr_by_issue`` free
    of stale closed-issue bindings (which feed ``issues_with_open_tracked_prs``
    and the ``live_dispatched`` guard), preserving the single-point-of-
    enforcement invariant shared with the dead-session and orphan sweeps. The
    test asserts the threading directly via a ``linked_issue_number`` spy
    because the dispatch result does not expose ``pr_by_issue``; the state/label
    transitions are the independent proof that the real (non-dry-run) branch --
    the one the spy is threaded through -- actually executed.
    """
    from unittest.mock import patch

    import charlie_work.workflow as workflow_mod
    from charlie_work.github import linked_issue_number as _real_linked

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = _fake_with_closed_709()
    # #709 must carry the ready label to appear in the dispatch issues list
    # (issue_list(state="all") filters by label, issue #427).
    for issue in fake_gh.issues:
        if issue["number"] == 709:
            issue["labels"] = [{"name": config.labels.ready}]
    # Replace the default PR #456 (which binds to #123) with only the stale-
    # branch PR, so the real open issue #123 has no open PR and is dispatchable.
    fake_gh.prs = [_stale_branch_open_pr()]

    # Default adapter is "manual" -- it writes a session manifest and reports
    # ok=True without launching a real subprocess, so the real claim/launch
    # path runs end-to-end in the test environment (same shape as
    # test_dispatch_selects_ready_issue_without_operator_queue_label).
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    calls: list[dict] = []

    def spy(pr, *, is_cross_repository, branch_prefix, branch_issue_validator=None):
        result = _real_linked(
            pr,
            is_cross_repository=is_cross_repository,
            branch_prefix=branch_prefix,
            branch_issue_validator=branch_issue_validator,
        )
        calls.append(
            {
                "pr_number": pr.get("number"),
                "branch_issue_validator": branch_issue_validator,
                "result": result,
            }
        )
        return result

    with patch.object(workflow_mod, "linked_issue_number", spy):
        result = app.dispatch()

    assert result.ok is True
    # The real open dispatchable issue #123 is claimed and dispatched via the
    # REAL (non-dry-run) branch: the dry-run early return never writes a
    # dispatch status to state.json or adds the agent:queued label, so these
    # two transitions are the proof the real dispatch-claim branch ran.
    assert result.data["selected_count"] == 1, (
        f"expected exactly the real open issue #123 selected for dispatch, "
        f"got selected_count={result.data['selected_count']}"
    )
    session_issues = [s["issue_number"] for s in result.data["sessions"]]
    assert 123 in session_issues, (
        f"real open issue #123 was not claimed for dispatch: sessions={session_issues}"
    )
    assert 709 not in session_issues, (
        f"closed issue #709 was wrongly dispatched: sessions={session_issues}"
    )
    # The agent:queued label is added ONLY in the real dispatch path's
    # post-launch transition (~6526); the dry-run branch never touches labels.
    assert (123, config.labels.queued) in fake_gh.labels_added, (
        "agent:queued label was not added for #123 -- the real (non-dry-run) "
        "dispatch-claim branch did not run"
    )
    state = load_state(paths.state_file)
    entry_123 = state.get("issues", {}).get("123")
    assert entry_123 is not None, (
        "state.json has no issues['123'] entry -- the real claim phase "
        "(which stamps dispatch_pending) did not run"
    )
    assert entry_123.get("status") in ("dispatch_pending", "manifest_written", "dispatched"), (
        f"issues['123'] status is {entry_123.get('status')!r}, not a dispatch "
        "claim/launch status -- the real dispatch-claim branch did not run"
    )

    # The dispatch-claim pr_by_issue construction (the REAL branch at ~5598,
    # not the dry-run one at ~5337) threaded the validator and rejected the
    # stale branch-name binding to closed #709.
    stale_calls = [c for c in calls if c["pr_number"] == 1660]
    assert stale_calls, (
        "dispatch path did not call linked_issue_number for the stale-branch PR #1660"
    )
    stale = stale_calls[0]
    assert stale["branch_issue_validator"] is not None, (
        "dispatch-claim pr_by_issue did not thread branch_issue_validator "
        "(stale branch-name binding not validated against the open-issue set)"
    )
    assert stale["result"] is None, (
        f"stale branch name bound unrelated PR #1660 to closed issue #709: "
        f"linked_issue_number returned {stale['result']}"
    )
    # Belt-and-suspenders: the phantom binding must not have dispatched #709.
    assert (709, config.labels.queued) not in fake_gh.labels_added, (
        "closed issue #709 received the agent:queued label -- the stale "
        "branch-name binding leaked into a real dispatch"
    )


def test_review_queue_stale_branch_does_not_bind_unrelated_pr_to_closed_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: ``review_queue`` must thread ``branch_issue_validator``
    through its ``linked_issue_number`` call so a stale branch name cannot
    bind an unrelated PR to a closed issue and route rework at the wrong
    subject.

    ``review_queue``'s ``issue_number`` feeds
    ``_reroute_stranded_request_changes`` (a real rework-routing state
    mutation, issue #784 AC-8 Case 2) and ``_emit_stale_ci_verdict_requeued``
    (a ``state.json`` write) -- the same phantom-binding failure class already
    fixed at the dispatch-claim, dead-session, and orphaned-worker sites. A
    stale ``agent/issue-709-…`` branch on an unrelated issue-less PR #1660,
    where #709 is CLOSED, must therefore be rejected: ``linked_issue_number``
    returns None, the PR is skipped at the ``if issue_number is None`` guard,
    and no rework routing or stale-CI verdict requeue keys off the wrong issue.

    The test asserts the threading directly via a ``linked_issue_number`` spy
    (the queue result does not expose the validator) and asserts the PR is
    absent from the queue -- the observable consequence of the rejected
    binding.
    """
    import json
    from unittest.mock import patch

    import charlie_work.workflow as workflow_mod
    from charlie_work.github import linked_issue_number as _real_linked

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    fake_gh = _fake_with_closed_709()
    fake_gh.prs = [_stale_branch_open_pr()]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    calls: list[dict] = []

    def spy(pr, *, is_cross_repository, branch_prefix, branch_issue_validator=None):
        result = _real_linked(
            pr,
            is_cross_repository=is_cross_repository,
            branch_prefix=branch_prefix,
            branch_issue_validator=branch_issue_validator,
        )
        calls.append(
            {
                "pr_number": pr.get("number"),
                "branch_issue_validator": branch_issue_validator,
                "result": result,
            }
        )
        return result

    with patch.object(workflow_mod, "linked_issue_number", spy):
        result = app.review_queue()

    assert result.ok is True
    # The stale-branch PR #1660 was rejected: the validator was threaded and
    # returned None, so the PR is skipped at the issue_number-is-None guard
    # and never enters the queue.
    stale_calls = [c for c in calls if c["pr_number"] == 1660]
    assert stale_calls, (
        "review_queue did not call linked_issue_number for the stale-branch PR #1660"
    )
    stale = stale_calls[0]
    assert stale["branch_issue_validator"] is not None, (
        "review_queue did not thread branch_issue_validator through "
        "linked_issue_number (stale branch-name binding not validated "
        "against the open-issue set)"
    )
    assert stale["result"] is None, (
        f"stale branch name bound unrelated PR #1660 to closed issue #709 in "
        f"review_queue: linked_issue_number returned {stale['result']}"
    )
    queued_prs = [entry["pr"] for entry in result.data["queue"]]
    assert 1660 not in queued_prs, (
        f"stale-branch PR #1660 entered the review queue despite the rejected "
        f"binding: queue={queued_prs}"
    )


def test_detect_drift_launch_stalled_stale_branch_does_not_mask_relabel(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not populate
    ``open_prs_by_issue`` for a nonexistent issue and emit a spurious
    ``pr_linked_issue_not_in_repo`` drift, nor prevent the
    ``session_failed_relabeled`` drift for the worker's own OPEN issue.

    Mirrors ``test_classify_dead_session_stale_branch_does_not_mask_escalation``
    but exercises ``reconcile.detect_drift``'s alive/hung launch_stalled
    relabel path (the ``if w.issue_number not in open_prs_by_issue`` guard
    at the ``session_failed_relabeled`` drift site).

    Scenario: a launch_stalled worker is keyed under issue #709 (OPEN, with
    an active ``in_progress`` label). An unrelated issue-less PR #1660
    carries a stale branch ``agent/issue-999-…`` where #999 does not exist in
    this repo. Without the validator, ``linked_issue_number`` returns 999,
    the PR is added to ``open_prs_by_issue[999]`` and
    ``prs_linking_issue[999]``, and — because ``issues_by_number.get(999)``
    is None — the PR loop emits a ``pr_linked_issue_not_in_repo`` drift for
    the nonexistent #999. With the validator, 999 is not in the open-issue
    set, the binding is rejected, and no spurious drift fires. The worker's
    own relabel (``session_failed_relabeled`` for #709) fires in both cases
    because the stale PR binds to 999, not 709 — confirming the stale PR
    does not mask the relabel.

    Mutation check: removing ``branch_issue_validator=branch_validator`` from
    either ``linked_issue_number`` call in ``detect_drift`` causes the
    ``pr_linked_issue_not_in_repo`` assertion to fail (the drift fires for
    the nonexistent #999).
    """
    import json
    import os
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub
    from _reconcile_fixtures import _issue, _pr
    from _sessions_db_fixtures import make_sessions_db
    from charlie_work.config import PostMortemConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.paths import resolved_layout
    from charlie_work.reconcile import detect_drift
    from charlie_work.state import empty_state

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "sessions.db"))
    )
    # Issue #709 is OPEN with an active label so the launch_stalled relabel
    # path can fire. The stale PR's branch references #999, which is NOT in
    # the issues list — the validator must reject 999.
    gh = ReconcileFakeGitHub(
        prs=[
            _pr(
                1660,
                state="OPEN",
                head_ref="agent/issue-999-stale-branch-from-merged-pr",
                body="Docs maintenance. No issue — issue-less orchestrator rework.",
            )
        ],
        issues=[_issue(709, [config.labels.in_progress])],
    )
    state = empty_state()

    worktree_path = "/tmp/worktree-709"
    now = datetime.now(UTC)

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-709",
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

    # detect_drift resolves the sessions dir through paths.resolved_layout
    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a log with only the shim marker — frozen well past grace period
    log_path = sessions_dir / "issue-709.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")
    old_time = now - timedelta(minutes=20)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Use a fake PID that passes is_alive() without actually checking the OS.
    fake_pid = 99999
    fake_start_time = 1700000000.0

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 709)
    record = SessionRecord(
        issue_number=709,
        branch="agent/issue-709-real-work",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt-709.md",
        command=("devin", "prompt.md"),
        pid=fake_pid,
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=fake_start_time,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    # Ensure no claude-code sidecar interferes
    (sessions_dir / "issue-709.claude.json").unlink(missing_ok=True)

    kill_calls: list[tuple[int, float | None]] = []

    def fake_kill(pid: int, expected_start_time: float | None = None) -> list[int]:
        kill_calls.append((pid, expected_start_time))
        return [pid]

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.reconcile.kill_process_tree", fake_kill),
    ):
        drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # The launch_stalled session was killed regardless of the open_prs guard
    # (kill happens before the guard).
    assert len(kill_calls) == 1, (
        f"Expected kill_process_tree to be called exactly once, got {kill_calls}"
    )

    # The stale PR #1660 must NOT bind to nonexistent issue #999.
    # Without the validator this drift fires; with it, the binding is rejected.
    not_in_repo = [
        d for d in drift if d.kind == "pr_linked_issue_not_in_repo" and d.pr_number == 1660
    ]
    assert not not_in_repo, (
        f"stale-branch PR #1660 bound to nonexistent issue #999 in detect_drift: "
        f"{[d.detail for d in not_in_repo]}"
    )

    # The worker's own relabel fires: #709 is OPEN with an active label and
    # no open PR (the stale PR binds to 999, not 709). This confirms the
    # stale PR does not mask the relabel.
    relabeled = [
        d
        for d in drift
        if d.kind == "session_failed_relabeled"
        and d.issue_number == 709
        and d.reason == "launch_stalled_no_open_pr"
    ]
    assert relabeled, (
        f"expected session_failed_relabeled drift for issue #709 with "
        f"reason=launch_stalled_no_open_pr; drift kinds were: "
        f"{[(d.kind, d.issue_number, getattr(d, 'reason', None)) for d in drift]}"
    )
    assert relabeled[0].pr_number is None, (
        f"session_failed_relabeled for #709 should have pr_number=None "
        f"(no open PR for #709), got {relabeled[0].pr_number}"
    )
