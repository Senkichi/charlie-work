"""Regression tests for issue #1241: salvage lane opens duplicate PRs for
already-merged work.

The salvage lane decided purely from worktree shape (dead worker + local
commits => open a PR) and never checked whether the linked issue was already
CLOSED or whether the commits were already reachable from the base branch.
These tests cover the single enforcement point added in ``_attempt_salvage``
via ``_salvage_is_superseded`` and the ``salvage_head_on_base`` reachability
probe.

The git-backed tests build real repos under ``tmp_path`` (no mocking of git
plumbing) so the fetch-then-ancestry sequence is exercised against actual git
behavior -- in particular the "stale tracking ref" failure mode that makes a
fetch-before-check mandatory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from test_charlie_work import _init_git_repo
from test_worktree import _clone_repo, _git

from charlie_work.config import OrchestratorConfig


def _commit_file(repo_root: Path, path: str, content: str, message: str) -> None:
    target = repo_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo_root, "add", path)
    _git(repo_root, "commit", "-m", message)


def _labels(config: OrchestratorConfig) -> tuple[set[str], set[str]]:
    active = {config.labels.in_progress}
    issue = {config.labels.in_progress}
    return active, issue


def _write_empty_state(state_file: Path) -> None:
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")


class _SalvageGitHub:
    """Fake ``GitHubLike`` for the salvage lane with controllable issue state.

    Extends the ``_SalvageTestGitHub`` shape from test_issue_956.py with an
    ``issue_view`` method so the superseded check can probe issue open/closed
    state.
    """

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        issue_state: str = "OPEN",
        issue_view_raises: bool = False,
        pr_create_return: int | None = 101,
    ) -> None:
        self.repo_root = repo_root
        self.dry_run = False
        self._issue_state = issue_state
        self._issue_view_raises = issue_view_raises
        self.pr_create_return = pr_create_return
        self.prs_created: list[dict[str, Any]] = []
        self.issue_views: list[int] = []

    def issue_view(self, number: int) -> dict[str, Any]:
        self.issue_views.append(number)
        if self._issue_view_raises:
            raise RuntimeError("gh issue view failed")
        return {"number": number, "state": self._issue_state}

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def remove_issue_label(self, number: int, label: str) -> bool:
        return True

    def add_issue_label(self, number: int, label: str) -> bool:
        return True


# --- salvage_head_on_base reachability probe ---


def test_salvage_head_on_base_true_when_head_ancestor_of_base(tmp_path: Path) -> None:
    """A head whose commits already merged to base is superseded (True)."""
    from charlie_work.worktree import salvage_head_on_base

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "feature")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    feature_head = _git(origin, "rev-parse", "HEAD").stdout.strip()
    # Land the work on main via a merge (the operator-salvage / auto-salvage
    # path that produced the duplicate in the issue).
    _git(origin, "checkout", "main")
    _git(origin, "merge", "--no-ff", "feature", "-m", "merge feature")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    # The clone's origin/main is stale (points at the pre-merge tip) until
    # salvage_head_on_base fetches. This is the exact scenario from the issue:
    # the merge landed on the remote minutes earlier.

    assert salvage_head_on_base(repo, feature_head, "main") is True


def test_salvage_head_on_base_false_when_head_has_unmerged_commits(
    tmp_path: Path,
) -> None:
    """A head with commits not yet on base is not superseded (False)."""
    from charlie_work.worktree import salvage_head_on_base

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "feature")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: unmerged work")
    feature_head = _git(origin, "rev-parse", "HEAD").stdout.strip()
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "feature")

    assert salvage_head_on_base(repo, feature_head, "main") is False


def test_salvage_head_on_base_none_when_fetch_fails(tmp_path: Path) -> None:
    """A fetch failure must fail open (None) -- never suppress salvage."""
    from charlie_work.worktree import salvage_head_on_base

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "feature")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    feature_head = _git(origin, "rev-parse", "HEAD").stdout.strip()

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    # Remove the origin remote so the fetch inside the probe cannot succeed.
    _git(repo, "remote", "remove", "origin")

    assert salvage_head_on_base(repo, feature_head, "main") is None


def test_salvage_head_on_base_none_for_invalid_sha(tmp_path: Path) -> None:
    """A malformed head SHA fails open rather than raising."""
    from charlie_work.worktree import salvage_head_on_base

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    repo = tmp_path / "repo"
    _clone_repo(origin, repo)

    assert salvage_head_on_base(repo, "not-a-sha", "main") is None


# --- _salvage_is_superseded ---


def test_salvage_is_superseded_issue_closed(tmp_path: Path) -> None:
    from charlie_work.workflow import _salvage_is_superseded

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    gh = _SalvageGitHub(repo_root=repo, issue_state="CLOSED")
    superseded, reason = _salvage_is_superseded(
        gh=gh,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
    )

    assert superseded is True
    assert reason == "issue_closed"
    assert gh.issue_views == [1241]


def test_salvage_is_superseded_commits_reachable(tmp_path: Path) -> None:
    from charlie_work.workflow import _salvage_is_superseded

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    _git(origin, "checkout", "main")
    _git(origin, "merge", "--no-ff", "agent/issue-1241", "-m", "merge salvage")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    gh = _SalvageGitHub(repo_root=repo, issue_state="OPEN")
    superseded, reason = _salvage_is_superseded(
        gh=gh,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
    )

    assert superseded is True
    assert reason == "commits_reachable_from_base"


def test_salvage_is_superseded_not_superseded(tmp_path: Path) -> None:
    from charlie_work.workflow import _salvage_is_superseded

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: unmerged work")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    gh = _SalvageGitHub(repo_root=repo, issue_state="OPEN")
    superseded, reason = _salvage_is_superseded(
        gh=gh,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
    )

    assert superseded is False
    assert reason is None


def test_salvage_is_superseded_fails_open_on_issue_view_error(
    tmp_path: Path,
) -> None:
    """A failed issue_view must not suppress salvage (fail-open)."""
    from charlie_work.workflow import _salvage_is_superseded

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: unmerged work")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    gh = _SalvageGitHub(repo_root=repo, issue_view_raises=True)
    superseded, reason = _salvage_is_superseded(
        gh=gh,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
    )

    assert superseded is False
    assert reason is None


# --- _attempt_salvage integration ---


def test_attempt_salvage_skips_when_issue_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed issue => no push, no PR, salvage_skipped_superseded event."""
    from charlie_work.state import load_state
    from charlie_work.workflow import _attempt_salvage

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    state_file = tmp_path / "state.json"
    _write_empty_state(state_file)
    config = OrchestratorConfig()
    active, issue = _labels(config)
    gh = _SalvageGitHub(repo_root=repo, issue_state="CLOSED")

    def _no_push(*a: Any, **k: Any) -> Any:
        raise AssertionError("push_branch must not be called when issue is closed")

    monkeypatch.setattr("charlie_work.workflow.push_branch", _no_push)

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
        active_labels=active,
        issue_labels=issue,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="salvage dup",
    )

    assert salvaged is True
    assert error is None
    assert gh.prs_created == []
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "salvage_skipped_superseded"]
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "issue_closed"
    assert events[0]["payload"]["issue_number"] == 1241


def test_attempt_salvage_skips_when_commits_already_on_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commits already merged to base => no push, no PR, skip event."""
    from charlie_work.state import load_state
    from charlie_work.workflow import _attempt_salvage

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "src/x.py", "x = 1\n", "feat: work")
    _git(origin, "checkout", "main")
    _git(origin, "merge", "--no-ff", "agent/issue-1241", "-m", "merge salvage")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    state_file = tmp_path / "state.json"
    _write_empty_state(state_file)
    config = OrchestratorConfig()
    active, issue = _labels(config)
    gh = _SalvageGitHub(repo_root=repo, issue_state="OPEN")

    def _no_push(*a: Any, **k: Any) -> Any:
        raise AssertionError("push_branch must not be called when commits are already on base")

    monkeypatch.setattr("charlie_work.workflow.push_branch", _no_push)

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
        active_labels=active,
        issue_labels=issue,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="salvage dup",
    )

    assert salvaged is True
    assert error is None
    assert gh.prs_created == []
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "salvage_skipped_superseded"]
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "commits_reachable_from_base"


def test_attempt_salvage_proceeds_when_work_not_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue OPEN + commits not on base => normal salvage (existing behavior)."""
    from charlie_work.state import load_state
    from charlie_work.workflow import _attempt_salvage

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "tests/test_x.py", "def test_x():\n    pass\n", "test: cover x")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    state_file = tmp_path / "state.json"
    _write_empty_state(state_file)
    config = OrchestratorConfig()
    active, issue = _labels(config)
    gh = _SalvageGitHub(repo_root=repo, issue_state="OPEN")

    monkeypatch.setattr("charlie_work.workflow.push_branch", lambda *a, **k: (True, None))

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
        active_labels=active,
        issue_labels=issue,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="real salvage",
    )

    assert salvaged is True
    assert error is None
    assert len(gh.prs_created) == 1
    state = load_state(state_file)
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(salvage_events) == 1
    skip_events = [e for e in state["events"] if e["kind"] == "salvage_skipped_superseded"]
    assert skip_events == []


def test_attempt_salvage_fails_open_when_issue_view_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed issue_view must not suppress salvage (fail-open)."""
    from charlie_work.workflow import _attempt_salvage

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "tests/test_x.py", "def test_x():\n    pass\n", "test: cover x")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")

    state_file = tmp_path / "state.json"
    _write_empty_state(state_file)
    config = OrchestratorConfig()
    active, issue = _labels(config)
    gh = _SalvageGitHub(repo_root=repo, issue_view_raises=True)

    monkeypatch.setattr("charlie_work.workflow.push_branch", lambda *a, **k: (True, None))

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
        active_labels=active,
        issue_labels=issue,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="real salvage",
    )

    # Salvage proceeded despite the issue_view failure.
    assert salvaged is True
    assert error is None
    assert len(gh.prs_created) == 1


def test_attempt_salvage_fails_open_when_reachability_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reachability probe must not suppress salvage (fail-open)."""
    from charlie_work.workflow import _attempt_salvage

    origin = tmp_path / "origin"
    _init_git_repo(origin)
    _git(origin, "checkout", "-b", "agent/issue-1241")
    _commit_file(origin, "tests/test_x.py", "def test_x():\n    pass\n", "test: cover x")
    _git(origin, "checkout", "main")

    repo = tmp_path / "repo"
    _clone_repo(origin, repo)
    _git(repo, "fetch", "origin", "agent/issue-1241")
    _git(repo, "checkout", "agent/issue-1241")
    # Break the origin remote so the fetch inside salvage_head_on_base fails,
    # forcing the reachability probe to return None (fail-open).
    _git(repo, "remote", "remove", "origin")

    state_file = tmp_path / "state.json"
    _write_empty_state(state_file)
    config = OrchestratorConfig()
    active, issue = _labels(config)
    gh = _SalvageGitHub(repo_root=repo, issue_state="OPEN")

    monkeypatch.setattr("charlie_work.workflow.push_branch", lambda *a, **k: (True, None))

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo,
        worktree_path=repo,
        branch="agent/issue-1241",
        base_ref="main",
        issue_number=1241,
        active_labels=active,
        issue_labels=issue,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="real salvage",
    )

    # Salvage proceeded despite the reachability probe failure.
    assert salvaged is True
    assert error is None
    assert len(gh.prs_created) == 1
