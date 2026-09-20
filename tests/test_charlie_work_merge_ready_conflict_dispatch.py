"""Merge-ready conflict rework-dispatch bookkeeping.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_conflict_alarm_message_is_honest(tmp_path: Path) -> None:
    """Issue #371: a persistent merge conflict deferral produces an honest alarm."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result1 = app.merge_ready(456, merge=False)
    assert result1.data["merge_conflict"] is True
    assert result1.data["merge_attempt_alarm"] is False
    assert result1.data["merge_attempt_warning"] is None

    result2 = app.merge_ready(456, merge=False)
    assert result2.data["merge_attempt_alarm"] is False

    result3 = app.merge_ready(456, merge=False)
    assert result3.data["merge_attempt_alarm"] is True
    warning = result3.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 3 passes" in warning
    assert "merge conflict" in warning.lower()

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert "merge conflict" in alarm_events[0]["payload"]["message"].lower()


def test_merge_ready_conflict_rework_routes_past_threshold(tmp_path: Path) -> None:
    """Regression: a PR that exceeded the failed_attempt_alarm threshold must
    still be routed to rework, not silently stuck.

    If the initial rework dispatch at the threshold failed to transition the
    issue (e.g. label transition error), the counter keeps incrementing past
    the threshold.  With ``==`` the rework condition never matches again,
    permanently orphaning the PR.  ``>=`` ensures it retries.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Pre-seed the state with attempt count *past* the threshold (simulates
    # a prior rework dispatch that failed to transition the issue).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"]["consecutive_failed_merge_attempts"] = 5
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1


def test_merge_ready_conflict_no_linked_issue_alarm_is_honest(tmp_path: Path) -> None:
    """Issue #379: an approved conflicting PR with no linked issue cannot be routed.

    The alarm must be honest about the inability to dispatch rework, not claim
    a rework worker was dispatched.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Cross-repo fix",
            "url": "https://example.test/pull/456",
            "headRefName": "fork/fix",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Tests: regression coverage added.",
            "labels": [],
            "isCrossRepository": True,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 1 pass" in warning
    assert "merge conflict" in warning.lower()
    assert "no linked issue, cannot route to rework" in warning
    assert result.data["issue"] is None
    assert result.data["label_error"] is None

    state = load_state(paths.state_file)
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists() is False
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_label_failure_is_recorded(tmp_path: Path) -> None:
    """Issue #379: a merge-conflict rework routing label failure is not swallowed.

    The rework label error must be returned in data['label_error'] and reflected
    in the alarm message.
    """
    from charlie_work.config import AutoMergeConfig
    from charlie_work.labels import TransitionOutcome

    class ReworkLabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            return False

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ReworkLabelFailGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "merge conflict" in warning.lower()
    assert "rework dispatch attempted" in warning
    assert "label update failed" in warning

    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "rework_requested"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_rework_dispatch_bounded_by_cap_across_repeated_evaluation(
    tmp_path: Path,
) -> None:
    """Issue #777: merge_ready()'s conflict-rework trigger must never dispatch
    more rework workers than config.review.max_conflict_rework_attempts, no
    matter how many times merge_ready() re-evaluates the same conflicting PR
    -- including across issue-status resets that mimic an external lane
    (e.g. a dead-session reaper) putting the issue back into a re-dispatchable
    state between passes.

    Before this fix, merge_ready()'s dispatch trigger called
    _request_merge_conflict_rework directly with no attempts_key bookkeeping
    at all, so nothing bounded the number of real dispatches across such
    cycles (real corpus: PR #679/issue #602, where the diagnostic-only
    consecutive_failed_merge_attempts counter climbed past 11 while the
    functional cap sat at 0 the entire time).
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig, ReviewConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    def _reset_issue_to_fresh() -> None:
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            state["issues"]["123"] = {**state["issues"]["123"], "status": "approved"}
            save_state(paths.state_file, state)

    dispatch_events_total = 0
    escalated_seen_at: int | None = None
    for pass_number in range(1, 5):
        result = app.merge_ready(456, merge=False)
        assert result.ok is True
        assert result.data["merge_conflict"] is True
        state = load_state(paths.state_file)
        dispatch_events_total = sum(
            1 for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
        )
        if state["issues"]["123"]["status"] == "escalated":
            escalated_seen_at = pass_number
            assert (
                state["issues"]["123"]["escalation_reason"]
                == "conflict_rework_attempts_cap_exceeded"
            )
            # Once escalated for this lane's own exhausted cap, stop
            # artificially re-arming: no real production path resets an
            # escalated issue's status back to "approved" (only
            # `charlie unescalate` does, and it also clears
            # escalation_reason) -- the loop's reset is only a harness
            # device to reach the cap boundary, not a realistic post-
            # escalation event. Break here and verify stability below
            # instead of feeding the wrapper a state no real caller would
            # ever produce.
            break
        assert state["issues"]["123"]["status"] == "rework_requested"
        # Never more real dispatches than the cap, no matter how many passes.
        assert dispatch_events_total <= config.review.max_conflict_rework_attempts
        _reset_issue_to_fresh()

    # The cap was actually reached and enforced, not merely never approached.
    assert escalated_seen_at is not None
    assert dispatch_events_total == config.review.max_conflict_rework_attempts
    escalated_events_after_first = sum(
        1 for e in state["events"] if e["kind"] == "janitor_rework_escalated"
    )
    assert escalated_events_after_first == 1

    # Issue #776: once escalated for THIS lane's own exhausted cap, a FURTHER
    # evaluation of the same still-conflicting PR must not dispatch yet
    # another worker, must not burn the counter past what it already is, and
    # must not re-fire a duplicate escalation event on every pass ("X blocks
    # retry of X") -- it simply leaves the already-escalated pair alone.
    for _ in range(3):
        result = app.merge_ready(456, merge=False)
        assert result.ok is True
        assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "conflict_rework_attempts_cap_exceeded"
    expected_final_attempts = config.review.max_conflict_rework_attempts + 1
    assert state["prs"]["456"]["conflict_rework_attempts"] == expected_final_attempts
    final_dispatch_events = sum(
        1 for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    )
    assert final_dispatch_events == config.review.max_conflict_rework_attempts
    final_escalated_events = sum(
        1 for e in state["events"] if e["kind"] == "janitor_rework_escalated"
    )
    assert final_escalated_events == 1


def test_merge_ready_conflict_carry_forward_resets_counter_before_dispatch(
    tmp_path: Path,
) -> None:
    """Issue #456 rework: a verdict carry-forward resets the failed-attempt
    counter and must not be defeated by the stale dispatch decision.

    A PR whose head moved but whose cumulative diff is unchanged carries the
    approved verdict forward and resets ``consecutive_failed_merge_attempts``.
    If the PR is still CONFLICTING after the head move, the dispatch gate must
    re-read the counter fresh and debounce from the new baseline instead of
    dispatching based on the pre-reset count.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    pr_number = 456
    issue_number = 123

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[pr_number] = diff_text
    fake_gh.prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}: search",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix-search",
            "baseRefName": "main",
            "headRefOid": old_head,
            "mergeStateStatus": "BEHIND",
            "mergeable": "CONFLICTING",
            "body": f"Closes #{issue_number}\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Two CONFLICTING passes with the original head bring the counter to 2.
    for _ in range(2):
        result = app.merge_ready(pr_number, merge=False)
        assert result.data["merge_conflict"] is True
        assert result.data["merge_attempt_alarm"] is False

    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "approved"
    assert state["prs"][str(pr_number)]["consecutive_failed_merge_attempts"] == 2

    # The PR head advances, but the cumulative diff is unchanged, so the
    # approved verdict carries forward and resets the failed-attempt counter.
    fake_gh.prs[0]["headRefOid"] = new_head
    fake_gh.diffs[pr_number] = diff_text

    result = app.merge_ready(pr_number, merge=False)
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["consecutive_failed_merge_attempts"] == 1

    state = load_state(paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "approved"
    assert state["prs"][str(pr_number)]["status"] == "approved"
    assert state["prs"][str(pr_number)]["consecutive_failed_merge_attempts"] == 1
    assert any(e["kind"] == "verdict_carried_forward_clean_rebase" for e in state["events"])
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (issue_number, config.labels.needs_rework) not in fake_gh.labels_added

    decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["reviewed_head_sha"] == new_head
    assert decision["reviewed_patch_id"] == patch_id


def test_merge_ready_conflict_dispatch_rechecks_issue_status_under_lock(
    tmp_path: Path,
) -> None:
    """Issue #456 rework: the conflict-rework dispatch gate re-reads issue
    status under lock after the network-I/O window.

    If a concurrent pass moves the linked issue into an in-flight state while
    this pass is fetching checks/diff/containment, dispatch must bail silently
    and must not clobber the in-flight status or dispatch a duplicate rework.
    """
    from charlie_work.config import AutoMergeConfig

    class RaceGitHub(FakeGitHub):
        def __init__(self, state_file: Path):
            super().__init__()
            self.state_file = state_file

        def pr_checks(self, number: int):
            # Simulate a concurrent pass transitioning the issue to 'dispatched'
            # while this merge_ready pass was blocked on network I/O.
            with state_lock(self.state_file):
                state = load_state(self.state_file)
                issue = state.setdefault("issues", {}).get("123")
                if issue is not None:
                    issue["status"] = "dispatched"
                    save_state(self.state_file, state)
            return super().pr_checks(number)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = RaceGitHub(paths.state_file)
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)

    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    assert result.data["merge_attempt_warning"] is not None
    assert "merge conflict" in (result.data["merge_attempt_warning"] or "").lower()

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
