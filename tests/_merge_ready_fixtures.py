"""Shared merge-ready / merge-queue test fixtures.

Helpers hoisted out of ``tests/test_charlie_work.py`` (issue #1550,
Track-1 wave 4/8) because they are imported by the merge-ready
sibling test modules: the queue-sync armer, the cross-PR-revert repo
builder, the racing-update app builder, the mergequeue automerge
helper, and the stale-base PR fixture.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def _arm_queue_sync_fixture(
    fake_gh: FakeGitHub,
    paths,
    *,
    pr_number: int = 601,
    issue_number: int = 601,
    reviewed_head_sha: str = "sha-approved",
    live_head_sha: str = "sha-syncmerge",
    merge_commit_sha: str | None = "sha-landing",
    pre_merge_base: str = "sha-premerge-base",
    other_parent: str = "sha-main-tip",
    live_head_parents: list[str] | None = None,
    author_login: str | None = "aviator-app[bot]",
    committer_login: str | None = "web-flow",
    committer_name: str | None = "GitHub",
    live_head_missing: bool = False,
    landing_missing: bool = False,
    landing_parents: list[str] | None = None,
    compare_status: str | None = "behind",
    compare_missing: bool = False,
) -> None:
    """Wire up a merged worker PR shaped like an Aviator queue sync-merge (#1194).

    Every knob defaults to the happy-path shape (two-parent bot merge, first
    landing parent reachable from the approved head via ``compare_status``);
    tests flip exactly the one signal they are pinning by passing an override.
    """
    fake_gh.prs = [
        {
            "number": pr_number,
            "title": f"fix: queue sync merge #{pr_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": live_head_sha,
            "mergeCommitOid": merge_commit_sha,
            "state": "MERGED",
            "isCrossRepository": False,
            "body": f"Closes #{issue_number}",
            "labels": [],
        },
    ]

    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": reviewed_head_sha}),
        encoding="utf-8",
    )

    if not live_head_missing:
        actual_parents = (
            live_head_parents
            if live_head_parents is not None
            else [reviewed_head_sha, other_parent]
        )
        commit: dict[str, Any] = {"parents": [{"sha": p} for p in actual_parents]}
        if author_login is not None:
            commit["author"] = {"login": author_login}
        if committer_login is not None:
            commit["committer"] = {"login": committer_login}
        if committer_name is not None:
            commit["commit"] = {"committer": {"name": committer_name}}
        fake_gh.commits[live_head_sha] = commit

    if merge_commit_sha and not landing_missing:
        actual_landing_parents = (
            landing_parents if landing_parents is not None else [pre_merge_base]
        )
        fake_gh.commits[merge_commit_sha] = {
            "parents": [{"sha": p} for p in actual_landing_parents]
        }

    if not compare_missing and compare_status is not None:
        fake_gh.compare_overrides[(pre_merge_base, other_parent)] = {"status": compare_status}


def _mergequeue_automerge(label: str = "mergequeue"):
    from charlie_work.config import AutoMergeConfig

    # No required checks -> the check gate is vacuously satisfied, isolating the
    # approved-decision path for Aviator MergeQueue handoff tests (task #10).
    return AutoMergeConfig(
        required_checks=(), require_approved_review=True, mergequeue_label=label
    )


def _stale_base_prs() -> list[dict[str, Any]]:
    """Two-PR fixture used by the issue #812 protection-derivation tests: PR
    456 merges first (advancing the fake base tip), then PR 789's merge-base
    is still the old tip, making it organically stale for the second call.
    """
    return [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
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
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]


def _init_cross_pr_revert_repo(repo_root: Path) -> tuple[str, str, str]:
    """Set up a git repo where main has C and an agent branch merges C then reverts C.

    Returns ``(base_sha, feature_sha, agent_sha)`` where ``feature_sha`` is the
    commit on main that the agent branch silently reverts.
    """
    remote = repo_root / "remote"
    remote.mkdir()
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main"],
        cwd=remote,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Base commit (before the feature that will be reverted)
    (repo_root / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Feature commit C on main
    (repo_root / "feature.txt").write_text("feature", encoding="utf-8")
    subprocess.run(["git", "add", "feature.txt"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature C"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    feature_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Agent branch merges the feature commit and then reverts it
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-revert", base_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "main", "-m", "Merge main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "revert", "--no-edit", feature_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    agent_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-revert"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Leave main checked out in the orchestrator repo
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    return base_sha, feature_sha, agent_sha


def _make_racing_merge_ready_app(
    tmp_path: Path, racing_commit: dict[str, Any]
) -> tuple[Any, Any, Any]:
    """Build an app whose update-branch races in a crafted merge commit.

    The racing commit's parents deliberately satisfy the structural checks
    (two parents, old head included) so only the committer-identity predicate
    is under test.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubRacingUpdate(FakeGitHub):
        def pr_update_branch(self, pr_number: int) -> bool:
            ok = super().pr_update_branch(pr_number)
            racing = "racing-sha"
            self.pr_head_shas[pr_number] = racing
            self.commits[racing] = racing_commit
            return ok

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubRacingUpdate()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh, paths
