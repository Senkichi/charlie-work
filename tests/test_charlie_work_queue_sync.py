"""Queue-sync merge-covered classification.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire
from charlie_work import github as github_module
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import _arm_queue_sync_fixture
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_queue_sync_merge_covered_happy_path_suppresses_finding(tmp_path: Path) -> None:
    """A two-parent bot sync-merge whose sync parent is reachable from
    pre-merge main is recognized as approval-covered: no finding, and the
    suppression is audit-logged (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert detected == []
    recorded = query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")
    assert len(recorded) == 1
    assert recorded[0]["pr_number"] == 601


def test_queue_sync_merge_covered_disabled_when_bot_login_unset(tmp_path: Path) -> None:
    """Default config (queue_bot_login unset) leaves #502 behavior byte-for-byte
    unchanged: an otherwise-perfect sync-merge shape still fires (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()  # queue_bot_login defaults to None
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601
    assert query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered") == []


def test_queue_sync_merge_covered_wrong_author_login_fires(tmp_path: Path) -> None:
    """A merge whose author login is not the configured queue bot is not
    recognized -- identity is conjunctive with reachability (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, author_login="someone-else")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_wrong_committer_login_fires(tmp_path: Path) -> None:
    """A merge not committed by GitHub's web-flow bot is not recognized
    (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, committer_login="not-web-flow")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_wrong_committer_name_fires(tmp_path: Path) -> None:
    """A merge whose ``commit.committer.name`` is not literally "GitHub" is
    not recognized (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, committer_name="Not GitHub")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_single_parent_fires(tmp_path: Path) -> None:
    """A live head with only one parent is not a merge commit at all --
    condition 1 fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, live_head_parents=["sha-approved"])
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_three_parents_fires(tmp_path: Path) -> None:
    """A live head with three parents (octopus merge) fails condition 1
    (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(
        fake_gh,
        paths,
        live_head_parents=["sha-approved", "sha-main-tip", "sha-extra-parent"],
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_neither_parent_matches_fires(tmp_path: Path) -> None:
    """A two-parent merge where neither parent is the approved head is not a
    sync of that approval -- condition 2 fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(
        fake_gh, paths, live_head_parents=["sha-unrelated-one", "sha-unrelated-two"]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_both_parents_match_fires(tmp_path: Path) -> None:
    """A degenerate merge where both parents equal the approved head has
    nothing to sync -- condition 2 requires exactly one match (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, live_head_parents=["sha-approved", "sha-approved"])
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_compare_diverged_fires(tmp_path: Path) -> None:
    """A sync parent that has diverged from pre-merge main carries content the
    approval never covered -- condition 3 fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, compare_status="diverged")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_compare_ahead_fires(tmp_path: Path) -> None:
    """A sync parent reported "ahead" of pre-merge main is not covered
    (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, compare_status="ahead")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_compare_none_fires(tmp_path: Path) -> None:
    """gh.compare() returning None (an unanswerable question) keeps the
    finding firing -- fail closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, compare_missing=True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_missing_merge_commit_oid_fires(tmp_path: Path) -> None:
    """A merged PR record with no ``mergeCommitOid`` (e.g. a queue
    fast-forward, or an older GraphQL-sourced record) cannot anchor
    condition 3 -- fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, merge_commit_sha=None)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_landing_commit_fetch_failure_fires(tmp_path: Path) -> None:
    """gh.commit() failing for the landing merge commit (rate limit, 404,
    transport error) is an ambiguous signal -- fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, landing_missing=True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_live_head_fetch_failure_fires(tmp_path: Path) -> None:
    """gh.commit() failing for the live head itself is an ambiguous signal --
    fails closed (issue #1194)."""
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    _arm_queue_sync_fixture(fake_gh, paths, live_head_missing=True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 601


def test_queue_sync_merge_covered_lazy_for_matching_head(tmp_path: Path) -> None:
    """When the approved head already matches the merged head, the queue-sync
    predicate must never be evaluated -- no gh.commit()/gh.compare() calls
    (issue #1194)."""
    from charlie_work.paths import runtime_paths

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.commit_calls = 0
            self.compare_calls = 0

        def commit(self, sha: str) -> github_module.GitHubRunResult:
            self.commit_calls += 1
            return super().commit(sha)

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            self.compare_calls += 1
            return super().compare(base, head)

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = CountingFakeGitHub()
    fake_gh.prs = [
        {
            "number": 602,
            "title": "fix: approved merge, matching head",
            "url": "https://example.test/pull/602",
            "headRefName": "agent/issue-602-fix",
            "baseRefName": "main",
            "headRefOid": "sha-602",
            "mergeCommitOid": "sha-602-landing",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #602",
            "labels": [],
        },
    ]
    pr_dir = paths.prs / "pr-602"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-602"}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert detected == []
    assert fake_gh.commit_calls == 0
    assert fake_gh.compare_calls == 0


def test_queue_sync_merge_covered_lazy_for_unapproved_pr(tmp_path: Path) -> None:
    """An unapproved (missing-decision) merged worker PR must not pay the
    queue-sync predicate's gh.commit()/gh.compare() calls either -- it is
    unconditionally a finding (issue #1194)."""
    from charlie_work.paths import runtime_paths

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.commit_calls = 0
            self.compare_calls = 0

        def commit(self, sha: str) -> github_module.GitHubRunResult:
            self.commit_calls += 1
            return super().commit(sha)

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            self.compare_calls += 1
            return super().compare(base, head)

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = CountingFakeGitHub()
    fake_gh.prs = [
        {
            "number": 603,
            "title": "fix: worker self-merge, no decision",
            "url": "https://example.test/pull/603",
            "headRefName": "agent/issue-603-fix",
            "baseRefName": "main",
            "headRefOid": "sha-603",
            "mergeCommitOid": "sha-603-landing",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #603",
            "labels": [],
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 603
    assert fake_gh.commit_calls == 0
    assert fake_gh.compare_calls == 0
