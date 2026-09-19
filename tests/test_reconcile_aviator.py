"""Aviator stale-``blocked``-label tests for ``reconcile``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
``detect_aviator_stale_blocked`` (job-cannon #1387/#1400/#1398/#1392)
and the matching ``apply_fixes`` lanes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from _reconcile_fixtures import (
    FakeGitHub,
    _AVIATOR_FAILURE_OUTPUT,
    _aviator_check_run,
    _passing_check_run,
    _pr,
    _write_review_decision,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_aviator_stale_blocked,
)
from charlie_work.state import empty_state


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
