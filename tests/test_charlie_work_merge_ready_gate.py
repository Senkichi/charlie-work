"""Merge-ready gate core evaluation.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import (
    _approved_automerge,
    _required_checks_config,
)
from _rework_dispatch_fixtures import _FakeGitHubWithInfraBlockedJob
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_requires_approved_decision_then_merges(tmp_path: Path) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    not_ready = app.merge_ready(456)
    assert not_ready.data["can_merge"] is False
    assert fake_gh.merged == []

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456, merge=True)  # Explicitly request merge

    assert ready.data["can_merge"] is True
    assert ready.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_merge_flags == [()]
    assert (123, "agent:done") in fake_gh.labels_added
    assert fake_gh.deleted_branches == ["agent/issue-123-fix-search"]
    assert ready.data["branch_deleted"] is True


def test_merge_ready_branch_delete_failure_never_blocks_labels(tmp_path: Path) -> None:
    """The empericus failure mode: a branch checked out in a local worktree made
    `gh pr merge --delete-branch` abort the post-merge label update. Deletion is
    now decoupled and best-effort — labels always land."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.delete_branch_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456, merge=True)

    assert ready.data["merged"] is True
    assert ready.data["branch_deleted"] is False
    assert (123, "agent:done") in fake_gh.labels_added


def test_merge_ready_honors_delete_branch_false(tmp_path: Path) -> None:
    config = _required_checks_config(delete_branch=False)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456, merge=True)

    assert ready.data["merged"] is True
    assert fake_gh.deleted_branches == []
    assert ready.data["branch_deleted"] is None


def test_merge_ready_update_open_prs_disabled_returns_none(tmp_path: Path) -> None:
    """Issue #149: when update_open_prs is disabled, update_open_prs_results must be None."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=False,
            require_current_base=False,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456, merge=True)

    assert ready.data["merged"] is True
    assert ready.data["update_open_prs_results"] is None


def test_merge_ready_not_merged_returns_none(tmp_path: Path) -> None:
    """Issue #149: when PR is not merged, update_open_prs_results must be None."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # No approval decision, so PR won't merge
    ready = app.merge_ready(456)

    assert ready.data["merged"] is False
    assert ready.data["update_open_prs_results"] is None


def test_merge_ready_update_open_prs_zero_matching_returns_empty_list(tmp_path: Path) -> None:
    """Issue #149: when sweep runs with zero matching PRs, update_open_prs_results must be []."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456)

    assert ready.data["merged"] is True
    # The sweep ran but found no other open agent PRs to update
    assert ready.data["update_open_prs_results"] == []


def test_merge_ready_checks_unavailable_returns_false(tmp_path: Path) -> None:
    """gh pr checks command failure must be reported as checks unavailable, not merge."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert result.data["checks_unavailable"] is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert fake_gh.merged == []


def test_merge_ready_infra_blocked_failure_blocks_merge_in_blocked_bucket(
    tmp_path: Path,
) -> None:
    """Round-3 review finding: ``merge_ready()`` uses the shared
    ``_enrich_checks_infra_blocked`` helper, which rewrites a zero-step
    FAILURE required check to ``INFRA_BLOCKED`` (not ``INFRA_FAILURE`` as the
    old inline enrichment did). The check must land in
    ``CheckSummary.infra_blocked`` (not ``infra_failed``) and block the merge
    (``can_merge=False``, ``merged=False``). Both buckets block merge via
    ``CheckSummary.ready``, so the merge gate is unchanged -- only the bucket
    differs. This is the merge_ready()-path coverage the round-1/2 tests
    lacked (only review()'s new path was covered)."""
    from charlie_work.workflow import _infra_blocked_window

    _infra_blocked_window.clear()
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubWithInfraBlockedJob(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        jobs_by_check_run_id={
            9001: {"conclusion": "FAILURE", "steps": []},
        },
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Plant an approved review decision so merge_ready reaches the check gate.
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456, merge=True)

    # The zero-step FAILURE check blocks merge via the infra_blocked bucket.
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert fake_gh.merged == []
    checks = result.data["checks"]
    assert checks["infra_blocked"] == ("Tests passed",)
    assert checks["infra_failed"] == ()
    assert checks["failed"] == ()


def test_merge_ready_real_path_return_data_includes_gate_inputs(
    tmp_path: Path,
) -> None:
    """Issue #1060: the real (non-dry-run) ``merge_ready()`` call's returned
    ``.data`` dict must include the four gate inputs
    (``summary_ready``, ``approved``, ``require_approved_review``,
    ``sync_failed``) alongside ``can_merge``, for diagnostic parity with the
    persisted ``merge_ready`` event. The review found that only the persisted
    event and the dry-run return were covered -- the real-path in-memory
    verdict's gate-input spread had no regression test.

    Two scenarios exercise both the ``can_merge=True`` and ``can_merge=False``
    branches so the assertion is not vacuously satisfied by a missing key
    defaulting to a falsy value.
    """
    gate_keys = {"summary_ready", "approved", "require_approved_review", "sync_failed"}

    # --- Scenario 1: approved + green -> can_merge=True -------------------
    config_ok = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            require_approved_review=True,
            require_current_base=False,
        )
    )
    paths_ok = runtime_paths(tmp_path / "ok", config_ok.runtime.state_dir)
    fake_gh_ok = FakeGitHub()
    app_ok = OrchestratorApp(tmp_path / "ok", paths_ok, config_ok, fake_gh_ok)
    app_ok.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    result_ok = app_ok.merge_ready(456, merge=False)

    assert result_ok.data["can_merge"] is True
    assert gate_keys <= set(result_ok.data)
    assert result_ok.data["summary_ready"] is True
    assert result_ok.data["approved"] is True
    assert result_ok.data["require_approved_review"] is True
    assert result_ok.data["sync_failed"] is False

    # --- Scenario 2: no recorded approval -> can_merge=False --------------
    config_noappr = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            require_approved_review=True,
            require_current_base=False,
        )
    )
    paths_noappr = runtime_paths(tmp_path / "noappr", config_noappr.runtime.state_dir)
    fake_gh_noappr = FakeGitHub()
    app_noappr = OrchestratorApp(tmp_path / "noappr", paths_noappr, config_noappr, fake_gh_noappr)
    # Deliberately do NOT record a review -> approved=False.
    result_noappr = app_noappr.merge_ready(456, merge=False)

    assert result_noappr.data["can_merge"] is False
    assert gate_keys <= set(result_noappr.data)
    assert result_noappr.data["summary_ready"] is True
    assert result_noappr.data["approved"] is False
    assert result_noappr.data["require_approved_review"] is True
    assert result_noappr.data["sync_failed"] is False


def test_merge_ready_sets_status_merged(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


def test_merge_ready_emits_merge_succeeded_on_fleet_merge(tmp_path: Path) -> None:
    """Issue #747: the merge lane emitted events for every outcome except
    success, so merge throughput was unobservable from events.db. A successful
    fleet direct-merge must emit exactly one ``merge_succeeded`` event carrying
    ``pr_number``, ``issue_number``, ``actor='fleet'``, the merge method, and a
    ``merged_at`` timestamp, and the state.json prs entry must gain
    ``merged_at`` so merge latency is computable retrospectively."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is True
    state = load_state(paths.state_file)
    pr_entry = state["prs"]["456"]
    assert pr_entry["status"] == "merged"
    assert pr_entry["merged"] is True
    # state.json merged entry gains a merged_at timestamp (issue #747).
    assert "merged_at" in pr_entry
    assert pr_entry["merged_at"]
    # Exactly one terminal success event, carrying the actor attribution.
    success_events = [e for e in state["events"] if e["kind"] == "merge_succeeded"]
    assert len(success_events) == 1
    payload = success_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["actor"] == "fleet"
    assert payload["merge_method"] == config.auto_merge.strategy
    assert payload["merged_at"] == pr_entry["merged_at"]


def test_merge_ready_does_not_emit_merge_succeeded_when_merge_skipped(
    tmp_path: Path,
) -> None:
    """Issue #747 negative control: ``merge_succeeded`` must NOT fire when the
    merge is skipped (no approval -> not mergeable), so the event distinguishes
    'emitted' from 'always emitted'."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # No approval decision -> PR is not merged (skipped).
    result = app.merge_ready(456)

    assert result.data["merged"] is False
    state = load_state(paths.state_file)
    success_events = [e for e in state["events"] if e["kind"] == "merge_succeeded"]
    assert success_events == []


def test_merge_ready_keeps_merged_state_when_label_transition_fails(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during merged transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Write the approved decision directly so the merge gate opens without
    # needing a (failing) label transition first.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "merged"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


def test_merge_ready_evaluation_only_preserves_recorded_merged_fact(tmp_path: Path) -> None:
    """A later evaluation-only run must not overwrite a previously recorded merged fact."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    merge_result = app.merge_ready(456, merge=True)
    assert merge_result.data["merged"] is True
    merged_state = load_state(paths.state_file)["prs"]["456"]
    assert merged_state["status"] == "merged"
    assert merged_state["merged"] is True

    # A subsequent evaluation-only pass short-circuits via the idempotence guard
    # and reports the PR as already merged without re-calling gh pr merge.
    eval_result = app.merge_ready(456, merge=False)
    assert eval_result.ok is True
    assert eval_result.data["already_merged"] is True
    assert eval_result.data["merged"] is True
    # merge_pr must NOT have been called again.
    assert fake_gh.merged == [(456, "squash")]  # only the first merge
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "merged"
    assert persisted["merged"] is True


def test_merge_ready_pr_list_error_during_update_open_prs_is_caught(tmp_path: Path) -> None:
    """Issue #146: GitHubError from pr_list during post-merge sweep must not propagate."""
    from charlie_work.config import AutoMergeConfig
    from charlie_work.github import GitHubError

    class PrListFailGitHub(FakeGitHub):
        def pr_list(self):
            raise GitHubError("API rate limit exceeded")

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # Enable the feature that calls pr_list
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = PrListFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Write the approved decision directly
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456, merge=True)

    # The merge should still succeed despite pr_list failing
    assert result.data["merged"] is True
    # The error should be recorded in update_open_prs_results
    assert result.data["update_open_prs_results"] is not None
    assert len(result.data["update_open_prs_results"]) == 1
    assert "error" in result.data["update_open_prs_results"][0]
    assert "pr_list failed" in result.data["update_open_prs_results"][0]["error"]
    # The merged state should still be recorded
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


def test_merge_ready_already_merged_is_noop(tmp_path: Path) -> None:
    """ship-it on a PR whose state records status='merged' must return ok=True
    without re-attempting `gh pr merge` (which would fail on an already-merged PR)."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Seed state as if a prior merge_ready already merged this PR.
    state = load_state(paths.state_file)
    state["prs"]["456"] = {"number": 456, "issue_number": 123, "status": "merged", "merged": True}
    save_state(paths.state_file, state)

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data["already_merged"] is True
    assert result.data["merged"] is True
    # merge_pr must NOT have been called — the fake would record it.
    assert fake_gh.merged == []


def test_merge_ready_passes_admin_flag_when_configured(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True, admin=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [()]


def test_merge_ready_passes_merge_flags_when_configured(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(), require_approved_review=True, merge_flags=("--admin",)
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [("--admin",)]


def test_merge_ready_merge_flags_takes_precedence_over_admin(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            admin=True,
            merge_flags=("--admin",),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # merge_flags takes precedence, so admin flag should be True (from merge_flags)
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [("--admin",)]


def test_merge_ready_default_merge_flags_preserves_current_behavior(
    tmp_path: Path,
) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # Default empty tuple should not add admin flag
    assert fake_gh.merged_admin_flags == [False]
    assert fake_gh.merged_merge_flags == [()]


def test_merge_ready_two_approved_prs_second_ship_succeeds_after_first_ship(
    tmp_path: Path,
) -> None:
    """End-to-end test for AC2: shipping two approved PRs in sequence should succeed.

    Regression test for issue #89: when two PRs are approved in the same operator pass,
    merging the first should not base-update the second (which would move its head and
    invalidate its approval). This test goes through the full merge_ready() path
    (not just _update_open_agent_prs) to verify the complete ship-it flow.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up two approved PRs
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",  # Live head matches reviewed head
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",  # Live head matches reviewed head
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision files for both PRs (approved state)
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-def456"},
            indent=2,
        ),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Ship the first PR
    result_456 = app.merge_ready(456, merge=True)
    assert result_456.ok is True
    assert result_456.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Verify PR 789's head is UNCHANGED (the skip worked)
    pr_789 = fake_gh.pr_view(789)
    assert pr_789["headRefOid"] == "sha-def456"  # Still the original head

    # Ship the second PR immediately afterward - should succeed without head-moved error
    result_789 = app.merge_ready(789, merge=True)
    assert result_789.ok is True
    assert result_789.data["merged"] is True
    assert result_789.data["can_merge"] is True
    assert result_789.data.get("head_moved") is not True  # Should not trigger head-moved gate
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]
