"""Merge-train candidate updates, front-of-train verdict carry-forward, open-PR backpressure, readiness predicates.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_front_of_train_only_updates_next_candidate(tmp_path: Path) -> None:
    """Issue #404: a single merge step updates only the new front candidate."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="front_of_train",
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
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Exactly one post-merge update-branch, on the new front candidate (789).
    assert fake_gh.pr_update_branch_calls == [789]
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"
    # The third PR stays behind-base until it reaches the front.
    assert fake_gh.prs[2]["headRefOid"] == "sha-ghi789"


def test_front_of_train_carries_forward_approved_verdict_end_to_end(tmp_path: Path) -> None:
    """Issue #404: non-front approved PRs carry their verdict forward when they reach the front.

    After each merge, the front-of-train update rewrites the approved PR's
    reviewed_head_sha while preserving the patch-id-based verdict, so the next
    merge step can proceed without a re-review.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_branch_strategy="front_of_train",
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
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "baseRefName": "main",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    # First merge: PR 456 is current, PR 789 is the new front and gets updated.
    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True
    fake_gh.prs[0]["state"] = "MERGED"

    decision_789 = json.loads((paths.prs / "pr-789" / "review-decision.json").read_text())
    assert decision_789["decision"] == "approved"
    assert decision_789["reviewed_head_sha"] == "sha-def456-updated"

    # Second merge: PR 789's carried-forward verdict lets it merge without re-review.
    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["merged"] is True
    fake_gh.prs[1]["state"] = "MERGED"

    decision_101 = json.loads((paths.prs / "pr-101" / "review-decision.json").read_text())
    assert decision_101["decision"] == "approved"
    assert decision_101["reviewed_head_sha"] == "sha-ghi789-updated"

    # Exactly two update-branch calls: one for each new front candidate.
    assert fake_gh.pr_update_branch_calls == [789, 101]
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]


def test_front_of_train_skips_request_changes_and_blocked(tmp_path: Path) -> None:
    """Issue #404: front-of-train mode skips request_changes/blocked PRs and
    updates the next approved candidate instead."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="front_of_train",
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
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(
        789, "request_changes", summary="needs work", verdict_provenance="fresh_llm_review"
    )
    app.record_review(101, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    # The request_changes PR is not the front; the next approved candidate is updated.
    assert fake_gh.pr_update_branch_calls == [101]


def test_open_pr_backpressure_clamps_dispatch_end_to_end(tmp_path: Path) -> None:
    """Issue #1129: app.dispatch() (not just _apply_concurrency_governor) clamps
    fresh-issue dispatch to zero when open agent PRs meet the cap.

    This exercises the _dispatch_impl wiring -- the ``apply_open_pr_backpressure=True``
    argument on the governor call inside _dispatch_impl. Every other open-PR
    backpressure test calls ``_apply_concurrency_governor`` directly, so a
    regression that dropped or flipped that argument would pass all of them
    undetected. This test mirrors test_fleet_concurrency_governor_clamps_when_fleet_live_at_cap
    for the fleet governor: it goes through the public ``app.dispatch()`` entry
    point and asserts on ``result.data`` fields.
    """

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Default FakeGitHub has issue #123 (automated-ready, OPEN) and one open PR
    # #456 (headRefName "agent/issue-123-fix-search") linked to #123. #123 is
    # therefore excluded from candidates (it already has an open PR). Add a
    # second dispatchable issue #124 with no open PR so there is a genuine
    # candidate to clamp -- selected_count==0 proves the clamp engaged, not an
    # empty backlog.
    fake_gh = FakeGitHub()
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Fix telemetry",
            "url": "https://example.test/issues/124",
            "body": "Telemetry is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # The single open agent PR (#456) fills the max_open_agent_prs=1 cap, so
    # fresh dispatch is clamped to 0 even though issue #124 is dispatchable.
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["open_pr_count"] == 1
    assert result.data["open_pr_max"] == 1


def test_is_pr_updated_at_older_than() -> None:
    """The shared updatedAt threshold helper parses, tz-normalizes, and compares."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.workflow import _is_pr_updated_at_older_than

    now = datetime.now(UTC)
    stale = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

    assert _is_pr_updated_at_older_than({"updatedAt": stale}, now, 15) is True
    assert _is_pr_updated_at_older_than({"updatedAt": fresh}, now, 15) is False
    assert _is_pr_updated_at_older_than({}, now, 15) is False
    assert _is_pr_updated_at_older_than({"updatedAt": "not-a-date"}, now, 15) is False

    # Naive datetimes are normalized to UTC before comparison.
    naive_now = datetime.now()
    naive_updated = (naive_now - timedelta(minutes=20)).replace(microsecond=0)
    assert (
        _is_pr_updated_at_older_than({"updatedAt": naive_updated.isoformat()}, naive_now, 15)
        is True
    )


def test_is_readiness_no_ci_stall() -> None:
    """Issue #474: the readiness no-CI gate escalates only when required checks are missing and updatedAt is stale."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import AutoMergeConfig
    from charlie_work.workflow import _is_readiness_no_ci_stall

    now = datetime.now(UTC)
    required = ("Tests passed", "Lint & Format")
    stale = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    config = AutoMergeConfig(required_checks=required, readiness_no_ci_minutes=15)

    # Missing required checks and stale updatedAt.
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], config, now) is True

    # A required check has appeared.
    assert (
        _is_readiness_no_ci_stall({"updatedAt": stale}, [{"name": "Tests passed"}], config, now)
        is False
    )

    # Missing checks but the PR was updated recently.
    assert _is_readiness_no_ci_stall({"updatedAt": fresh}, [], config, now) is False

    # Gate disabled by zero minutes.
    disabled = AutoMergeConfig(required_checks=required, readiness_no_ci_minutes=0)
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], disabled, now) is False

    # No required checks configured: there is nothing to be missing.
    no_required = AutoMergeConfig(required_checks=(), readiness_no_ci_minutes=15)
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], no_required, now) is False

    # Missing or malformed updatedAt is treated as not stale.
    assert _is_readiness_no_ci_stall({}, [], config, now) is False
