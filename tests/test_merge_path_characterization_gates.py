"""Gate-side merge_ready characterization: checks, conflict, merge train, revert.

Continuation of ``test_merge_path_characterization.py`` (split to stay under
the 800-line module cap); the harness lives in
``_merge_path_characterization_harness.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub, FakeGitHubWithChecks, FakeGitHubWithMissingRequired
from _fakes_github_rerun import FakeGitHubWithRerunCapture
from charlie_work import workflow as workflow_module
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
    DevinConfig,
)
from charlie_work.cross_pr_revert import CrossPrRevertResult, CrossPrRevertStatus
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _merge_path_characterization_harness import (
    _DRY_FINAL_KEYS,
    _ISSUE,
    _PR,
    _REQUIRED,
    _assert_dry_run_inert,
    _cfg,
    _conflicted_gh,
    _pair,
    _run,
    _seed_issue_entry,
)


# ===========================================================================
# Checks: unavailable, failed, pending, infra, readiness stall
# ===========================================================================


class _ChecksUnavailable(FakeGitHub):
    def pr_checks(self, number: int):  # type: ignore[override]
        return None


def test_checks_unavailable_blocks_merge_and_reports_not_ok_both_paths(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(required=_REQUIRED), _ChecksUnavailable)

    for out in (pair.live, pair.dry):
        assert out.result.ok is False
        assert out.data["checks_unavailable"] is True
        assert out.data["can_merge"] is False
        assert out.data["merged"] is False
        assert out.gh.merged == []
    assert pair.dry.result.message == "dry-run: checks unavailable (gh failure)"
    # Live: an approved, un-mergeable pass counts as a failed attempt.
    assert pair.live.pr_entry["consecutive_failed_merge_attempts"] == 1
    _assert_dry_run_inert(pair.dry)


def _failing_checks_gh() -> FakeGitHub:
    return FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "SUCCESS"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )


def test_failed_required_check_below_threshold_counts_attempt_without_rework(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, _cfg(required=_REQUIRED, alarm=3), _failing_checks_gh)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.result.ok is True
        assert out.data["can_merge"] is False
        assert out.data["summary_ready"] is False
        assert out.data["checks"]["failed"] == ("Tests passed",)
        assert out.gh.merged == []
    assert live.data["consecutive_failed_merge_attempts"] == 1
    assert live.data["merge_attempt_alarm"] is False
    assert live.issue_entry.get("status") != "rework_requested"
    assert "check_failure_rework_requested" not in live.new_event_kinds()
    # Dry-run reports persisted counters unchanged and never alarms.
    assert dry.data["consecutive_failed_merge_attempts"] == 0
    assert dry.data["merge_attempt_alarm"] is False
    assert dry.data["merge_attempt_warning"] is None
    _assert_dry_run_inert(dry)


def test_failed_required_check_at_threshold_routes_rework_live_only(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(required=_REQUIRED, alarm=1), _failing_checks_gh)

    live, dry = pair.live, pair.dry
    assert live.data["merge_attempt_alarm"] is True
    assert live.data["merge_attempt_warning"] is not None
    assert live.data["consecutive_failed_merge_attempts"] == 1
    assert live.issue_entry["status"] == "rework_requested"
    assert live.pr_entry["status"] == "rework_requested"
    assert "check_failure_rework_requested" in live.new_event_kinds()
    assert "merge_failed_attempt_alarm" in live.new_event_kinds()
    assert (_ISSUE, "agent:needs-rework") in live.gh.labels_added

    # DIVERGENCE (recon a.3 #8): dry-run never alarms or routes.
    assert dry.data["merge_attempt_alarm"] is False
    assert dry.data["merge_attempt_warning"] is None
    _assert_dry_run_inert(dry)


def test_pending_only_checks_do_not_count_as_failed_attempts(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        return FakeGitHubWithChecks(
            checks=[
                {"name": "Tests passed", "state": "PENDING"},
                {"name": "Lint & Format", "state": "SUCCESS"},
                {"name": "Pre-commit", "state": "SUCCESS"},
            ]
        )

    live = _run(tmp_path / "live", _cfg(required=_REQUIRED), gh_factory, dry_run=False)

    assert live.data["can_merge"] is False
    assert live.data["consecutive_failed_merge_attempts"] == 0
    assert live.data["merge_attempt_alarm"] is False
    assert live.gh.merged == []
    assert live.issue_entry.get("status") != "rework_requested"


def test_failed_attempt_counter_climbs_then_clamps_and_alarms_once(tmp_path: Path) -> None:
    config = _cfg(required=_REQUIRED, alarm=2)
    root = tmp_path / "live"
    paths = runtime_paths(root, config.runtime.state_dir)
    root.mkdir(parents=True)
    gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "SUCCESS"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    # Bound issue already escalated: no rework routing, so only the counter moves.
    app = OrchestratorApp(root, paths, config, gh)
    app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    _seed_issue_entry(paths, status="escalated", escalation_reason="redispatch_cap_exceeded")

    seen = [app.merge_ready(_PR, merge=True).data for _ in range(4)]

    assert [d["consecutive_failed_merge_attempts"] for d in seen] == [1, 2, 3, 3]
    assert [d["merge_attempt_alarm"] for d in seen] == [False, True, False, False]


def test_readiness_no_ci_stall_requests_rework_live_and_dry_run_skips_it(
    tmp_path: Path,
) -> None:
    stale = "2020-01-01T00:00:00Z"  # fixed, ancient: always past the stall window

    def gh_factory() -> FakeGitHub:
        gh = FakeGitHubWithMissingRequired()
        gh.prs[0]["updatedAt"] = stale
        gh.prs[0]["mergeStateStatus"] = "CLEAN"
        gh.prs[0]["mergeable"] = "MERGEABLE"
        return gh

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=_REQUIRED,
            require_approved_review=True,
            update_branch_strategy="off",
            require_current_base=False,
            failed_attempt_alarm=3,
            readiness_no_ci_minutes=15,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    pair = _pair(tmp_path, config, gh_factory)

    live, dry = pair.live, pair.dry
    assert live.data["readiness_no_ci_stall"] is True
    assert live.data["can_merge"] is False
    assert live.data["merge_attempt_alarm"] is False
    assert live.issue_entry["status"] == "rework_requested"
    assert "readiness_no_ci_rework_requested" in live.new_event_kinds()
    assert (live.paths.prs / "pr-456" / "rework-prompt.md").exists()

    # DIVERGENCE (recon a.3 #2): dry-run skips the stall lane entirely.
    assert "readiness_no_ci_stall" not in dry.data
    assert dry.data["can_merge"] is False
    assert dry.data["summary_ready"] is False
    _assert_dry_run_inert(dry)


_RUN_LINK = "https://github.com/owner/repo/actions/runs/12345/job/67890"
_CANCELLED = [
    {"name": "Tests passed", "state": "CANCELLED", "link": _RUN_LINK},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]


def test_infra_failed_sole_blocker_reruns_live_and_dry_run_skips_remediation(
    tmp_path: Path,
) -> None:
    pair = _pair(
        tmp_path,
        _cfg(required=_REQUIRED),
        lambda: FakeGitHubWithRerunCapture(checks=list(_CANCELLED)),
    )

    live, dry = pair.live, pair.dry
    assert live.data["infra_rerun_run_ids"] == [12345]
    assert live.data["can_merge"] is False
    assert live.data["merged"] is False
    assert live.gh.rerun_calls == [["run", "rerun", "12345"]]  # type: ignore[attr-defined]
    assert "infra_rerun_triggered" in live.new_event_kinds()
    assert live.pr_entry["infra_rerun_attempts"] == {"sha-abc123": {"Tests passed": {"12345": 1}}}

    # DIVERGENCE (recon a.3 #2): dry-run never reruns anything.
    assert "infra_rerun_run_ids" not in dry.data
    assert dry.gh.rerun_calls == []  # type: ignore[attr-defined]
    assert dry.data["can_merge"] is False
    _assert_dry_run_inert(dry)


# ===========================================================================
# Merge conflict
# ===========================================================================


def test_conflict_below_threshold_counts_attempt_and_flags_conflict(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(alarm=3), _conflicted_gh)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.result.ok is True
        assert out.data["merge_conflict"] is True
        assert out.data["can_merge"] is False
        assert out.data["sync_failed"] is True
        assert out.data["merged"] is False
        assert out.gh.merged == []
        assert out.gh.pr_update_branch_calls == []
    assert live.data["consecutive_failed_merge_attempts"] == 1
    assert live.data["merge_attempt_alarm"] is False
    assert live.issue_entry.get("status") != "rework_requested"
    assert dry.result.message == (
        "dry-run: merge readiness evaluated (merge conflict — would route to rework on threshold)"
    )
    assert dry.data["consecutive_failed_merge_attempts"] == 0
    _assert_dry_run_inert(dry)


def test_conflict_at_threshold_routes_rework_live_only(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(alarm=1), _conflicted_gh)

    live, dry = pair.live, pair.dry
    assert live.data["merge_conflict"] is True
    assert live.data["merge_attempt_alarm"] is True
    assert live.data["merge_attempt_warning"] is not None
    assert live.issue_entry["status"] == "rework_requested"
    assert live.pr_entry["status"] == "rework_requested"
    assert "merge_conflict_rework_requested" in live.new_event_kinds()
    assert (_ISSUE, "agent:needs-rework") in live.gh.labels_added
    assert (live.paths.prs / "pr-456" / "rework-prompt.md").exists()
    # The approved verdict survives the rework request.
    assert live.data["review_decision"]["decision"] == "approved"

    assert dry.data["merge_attempt_alarm"] is False
    assert dry.data["merge_conflict"] is True
    _assert_dry_run_inert(dry)


@pytest.mark.parametrize("issue_status", ["dispatched", "dispatch_pending", "manifest_written"])
def test_conflict_with_live_rework_worker_waits_live_while_dry_run_keeps_evaluating(
    tmp_path: Path, issue_status: str
) -> None:
    """DIVERGENCE (recon a.3 #1): live returns the early 'being resolved'
    wait shape; the dry-run has no such early return."""

    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_issue_entry(paths, status=issue_status)

    pair = _pair(tmp_path, _cfg(alarm=1), _conflicted_gh, pre=pre)

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert live.result.message == f"PR #{_PR} merge conflict is being resolved by a rework worker"
    assert set(live.data) == {
        "pr",
        "issue",
        "can_merge",
        "merged",
        "review_decision",
        "merge_conflict",
        "consecutive_failed_merge_attempts",
        "merge_attempt_alarm",
        "merge_attempt_warning",
    }
    assert live.data["can_merge"] is False
    assert live.data["merge_conflict"] is True
    assert live.data["merge_attempt_alarm"] is False
    assert live.issue_entry["status"] == issue_status
    assert live.new_event_kinds() == []

    assert set(dry.data) == _DRY_FINAL_KEYS
    assert dry.data["merge_conflict"] is True
    _assert_dry_run_inert(dry)


def test_conflict_with_blocked_issue_is_not_rerouted_live(tmp_path: Path) -> None:
    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_issue_entry(paths, status="blocked")

    live = _run(tmp_path / "live", _cfg(alarm=1), _conflicted_gh, dry_run=False, pre=pre)

    assert live.result.ok is True
    assert "awaiting human decision" in live.result.message
    assert live.data["can_merge"] is False
    assert live.data["merge_conflict"] is True
    assert live.issue_entry["status"] == "blocked"
    assert "merge_conflict_rework_requested" not in live.new_event_kinds()
    assert live.gh.merged == []


def test_conflict_on_escalated_issue_is_still_routed_to_rework(tmp_path: Path) -> None:
    """#776: the conflict lane deliberately does NOT exclude 'escalated'."""

    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_issue_entry(paths, status="escalated", escalation_reason="redispatch_cap_exceeded")

    live = _run(tmp_path / "live", _cfg(alarm=1), _conflicted_gh, dry_run=False, pre=pre)

    assert live.data["merge_conflict"] is True
    assert live.data["merge_attempt_alarm"] is True
    assert "merge_conflict_rework_requested" in live.new_event_kinds()


# ===========================================================================
# Merge train: front-of-train, stale base
# ===========================================================================

_NOT_HEAD_MSG = f"PR #{_PR} is not the head of the merge-train queue"
_NOT_HEAD_KEYS = {
    "pr",
    "issue",
    "can_merge",
    "auto_merge_enabled",
    "merged",
    "merge_output",
    "branch_deleted",
    "review_decision",
    "checks",
    "checks_unavailable",
    "label_error",
    "update_open_prs_results",
    "cancel_superseded_runs_results",
    "containment_warnings",
    "consecutive_failed_merge_attempts",
    "merge_attempt_alarm",
    "merge_attempt_warning",
    "merge_conflict",
}


def test_front_of_train_param_not_this_pr_is_not_ready_same_shape_both_paths(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, _cfg(strategy="front_of_train"), merge_train_head=999)

    for out in (pair.live, pair.dry):
        assert out.result.ok is True
        assert out.result.message == _NOT_HEAD_MSG
        assert set(out.data) == _NOT_HEAD_KEYS
        assert out.data["can_merge"] is False
        assert out.data["merged"] is False
        assert out.data["merge_conflict"] is False
        assert out.data["consecutive_failed_merge_attempts"] == 0
        assert out.gh.merged == []
        assert out.gh.pr_update_branch_calls == []
        # Neither path stamps dry_run on this shape; neither writes state.
        assert "dry_run" not in out.data
        assert out.state_after == out.state_before


def test_front_of_train_discovers_head_from_pr_list_when_param_absent(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.prs.append(
            {
                "number": 100,
                "title": "Fix #124: earlier",
                "url": "https://example.test/pull/100",
                "headRefName": "agent/issue-124-earlier",
                "baseRefName": "main",
                "headRefOid": "sha-100",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #124",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        )
        gh.issues.append(
            {
                "number": 124,
                "title": "earlier",
                "url": "https://example.test/issues/124",
                "body": "",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
        )
        return gh

    def pre(app: OrchestratorApp, _paths: Any, _gh: FakeGitHub) -> None:
        app.record_review(
            100,
            "approved",
            summary="ok",
            verdict_provenance="fresh_llm_review",
        )

    pair = _pair(tmp_path, _cfg(strategy="front_of_train"), gh_factory, pre=pre)

    for out in (pair.live, pair.dry):
        assert out.result.message == _NOT_HEAD_MSG
        assert out.data["can_merge"] is False
        assert out.gh.merged == []


def _stale_base_gh() -> FakeGitHub:
    gh = FakeGitHub()
    gh.prs[0]["mergeStateStatus"] = "BEHIND"
    return gh


def _stale_cfg() -> OrchestratorConfig:
    return _cfg(strategy="broadcast", require_current_base=True)


def test_stale_base_defers_merge_live_persists_dry_run_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A sync that reports success but does not move the head leaves the PR
    # stale, exercising the freshness gate rather than the sync step.
    monkeypatch.setattr(FakeGitHub, "pr_update_branch", lambda self, n: True)
    pair = _pair(tmp_path, _stale_cfg(), _stale_base_gh)

    live, dry = pair.live, pair.dry
    msg = f"PR #{_PR} base is stale; merge deferred until base is current"
    assert live.result.ok is True
    assert live.result.message == msg
    assert live.data["stale_base"] is True
    assert live.data["can_merge"] is False
    assert live.data["merged"] is False
    assert live.data["consecutive_stale_base_deferrals"] == 1
    assert live.pr_entry["consecutive_stale_base_deferrals"] == 1
    assert "merge_deferred_stale_base" in live.new_event_kinds()
    assert live.gh.merged == []

    assert dry.result.ok is True
    assert dry.result.message == msg
    assert set(dry.data) == {
        "pr",
        "issue",
        "can_merge",
        "auto_merge_enabled",
        "merged",
        "merge_output",
        "branch_deleted",
        "review_decision",
        "checks",
        "checks_unavailable",
        "label_error",
        "update_open_prs_results",
        "cancel_superseded_runs_results",
        "containment_warnings",
        "stale_base",
        "consecutive_failed_merge_attempts",
        "consecutive_stale_base_deferrals",
        "merge_attempt_alarm",
        "merge_attempt_warning",
        "merge_conflict",
        "dry_run",
    }
    assert dry.data["stale_base"] is True
    assert dry.data["consecutive_stale_base_deferrals"] == 0
    assert dry.data["merge_attempt_alarm"] is False
    assert dry.data["dry_run"] is True
    assert dry.gh.merged == []
    assert dry.state_after == dry.state_before


def test_stale_base_is_repaired_by_update_branch_then_merges_live(tmp_path: Path) -> None:
    live = _run(tmp_path / "live", _stale_cfg(), _stale_base_gh, dry_run=False)

    assert live.gh.pr_update_branch_calls == [_PR]
    assert live.data["merged"] is True
    assert live.data["can_merge"] is True
    assert live.gh.merged == [(_PR, "squash")]
    # The sync moved the head: the approval was re-pinned to the new head.
    assert live.data["review_decision"]["reviewed_head_sha"] == "sha-abc123-updated"


def test_update_branch_failure_marks_sync_failed_and_blocks_merge(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        gh = _stale_base_gh()
        gh.update_branch_ok = False
        gh.pr_update_branch = lambda n: False  # type: ignore[method-assign]
        return gh

    live = _run(tmp_path / "live", _stale_cfg(), gh_factory, dry_run=False)

    assert live.data["sync_failed"] is True
    assert live.data["can_merge"] is False
    assert live.data["merged"] is False
    assert live.gh.merged == []


# ===========================================================================
# Cross-PR revert
# ===========================================================================


def _patch_revert(
    monkeypatch: pytest.MonkeyPatch, status: CrossPrRevertStatus, reason: str
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "detect_cross_pr_revert",
        lambda *_a, **_k: CrossPrRevertResult(status, reason),
    )


def test_cross_pr_revert_detected_blocks_and_routes_rework_live_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_revert(monkeypatch, CrossPrRevertStatus.REVERT_DETECTED, "reverts feature C")
    pair = _pair(tmp_path, _cfg())

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.data["can_merge"] is False
        assert out.data["sync_failed"] is True
        assert out.data["cross_pr_revert_detected"] is True
        assert out.data["cross_pr_revert_undetermined"] is False
        assert out.data["cross_pr_revert_reason"] == "reverts feature C"
        assert out.data["merged"] is False
        assert out.gh.merged == []
    assert live.data["cross_pr_revert_routed"] is True
    assert live.issue_entry["status"] == "rework_requested"
    assert "cross_pr_revert_rework_requested" in live.new_event_kinds()
    assert (_ISSUE, "agent:needs-rework") in live.gh.labels_added

    assert dry.data["cross_pr_revert_routed"] is False
    assert dry.result.message == (
        "dry-run: merge readiness evaluated (cross-PR revert: reverts feature C)"
    )
    _assert_dry_run_inert(dry)


def test_cross_pr_revert_undetermined_fails_closed_without_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_revert(monkeypatch, CrossPrRevertStatus.UNDETERMINED, "git fetch failed")
    pair = _pair(tmp_path, _cfg())

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.result.ok is True
        assert out.data["can_merge"] is False
        assert out.data["cross_pr_revert_detected"] is False
        assert out.data["cross_pr_revert_undetermined"] is True
        assert out.data["cross_pr_revert_routed"] is False
        assert out.gh.merged == []
        assert "undetermined" in out.result.message
        assert "fail-closed" in out.result.message
    assert live.issue_entry.get("status") != "rework_requested"
    assert "cross_pr_revert_rework_requested" not in live.new_event_kinds()
    assert dry.result.message == (
        "dry-run: merge readiness evaluated "
        "(cross-PR revert gate undetermined — would hold merge, fail-closed: git fetch failed)"
    )
    _assert_dry_run_inert(dry)


def test_cross_pr_revert_clean_leaves_the_merge_path_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_revert(monkeypatch, CrossPrRevertStatus.CLEAN, "")
    pair = _pair(tmp_path, _cfg())

    assert pair.live.data["merged"] is True
    assert pair.live.data["cross_pr_revert_detected"] is False
    assert pair.live.data["cross_pr_revert_undetermined"] is False
    assert pair.dry.data["can_merge"] is True
    assert "would merge" in pair.dry.result.message


# ===========================================================================
# Self-merge finalization details (live only)
# ===========================================================================


def test_self_merge_honours_delete_branch_false_and_keeps_merge_on_delete_failure(
    tmp_path: Path,
) -> None:
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            update_branch_strategy="off",
            require_current_base=False,
            delete_branch=False,
        )
    )
    live = _run(tmp_path / "live", config, FakeGitHub, dry_run=False)

    assert live.data["merged"] is True
    assert live.gh.deleted_branches == []
    assert live.data["branch_deleted"] in (None, False)
    assert live.pr_entry["status"] == "merged"


def test_merge_ready_replay_after_self_merge_is_the_idempotent_noop(tmp_path: Path) -> None:
    config = _cfg()
    root = tmp_path / "live"
    paths = runtime_paths(root, config.runtime.state_dir)
    root.mkdir(parents=True)
    gh = FakeGitHub()
    app = OrchestratorApp(root, paths, config, gh)
    app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    first = app.merge_ready(_PR, merge=True)
    second = app.merge_ready(_PR, merge=True)

    assert first.data["merged"] is True
    assert second.data == {
        "pr": _PR,
        "issue": _ISSUE,
        "already_merged": True,
        "merged": True,
    }
    assert gh.merged == [(_PR, "squash")]
