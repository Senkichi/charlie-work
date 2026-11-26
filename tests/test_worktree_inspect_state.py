"""``inspect_worktree_state``: completed/partial/unknown classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from _worktree_fixtures import _git
from _worktree_shared_fixtures import _init_repo_with_remote
from charlie_work.config import OrchestratorConfig
from charlie_work.worktree import (
    WorktreeState,
    create_worktree,
    inspect_worktree_state,
    remove_worktree,
)


def test_inspect_worktree_state_completed(tmp_path: Path) -> None:
    """A clean worktree with commits beyond the base is completed."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-1", base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.dirty is False
    assert inspection.resolved_base_ref == "origin/main"

    remove_worktree(repo, info.path, branch="agent/issue-1")


def test_inspect_worktree_state_partial_dirty(tmp_path: Path) -> None:
    """A worktree with uncommitted changes is partial, regardless of commits."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-2", base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.PARTIAL
    assert inspection.dirty is True

    remove_worktree(repo, info.path, branch="agent/issue-2")


def test_inspect_worktree_state_completed_ignores_injected_prompts(tmp_path: Path) -> None:
    """Issue #381: a completed worktree is still COMPLETED if the only dirty
    files are orchestrator-injected prompt paths from the frozen config."""
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    # Simulate orchestrator-injected prompt files being modified in the worktree.
    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-381")


def test_inspect_worktree_state_completed_ignores_tracked_injected_prompts(
    tmp_path: Path,
) -> None:
    """Issue #381: a tracked injected prompt file modified in place is not dirty.

    This is the root-cause scenario: the orchestrator writes the prompt file,
    the worker commits it, then rewrites it in place without staging. The
    porcelain parser must not strip the leading status-column space, which would
    shift the path and drop its leading dot.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381-tracked", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    # Track the injected prompt file, then rewrite it in place without staging.
    prompt = info.path / config.dispatch.injected_paths[0]
    prompt.write_text("original prompt", encoding="utf-8")
    _git(info.path, "add", str(prompt))
    _git(info.path, "commit", "-m", "track prompt")
    prompt.write_text("rewritten prompt", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 2
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-381-tracked")


def test_inspect_worktree_state_partial_with_worker_changes_and_injected(tmp_path: Path) -> None:
    """Issue #381: worker-authored uncommitted changes still block COMPLETED."""
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    (info.path / "worker-change.txt").write_text("worker work\n", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.PARTIAL
    assert inspection.dirty is True

    remove_worktree(repo, info.path, branch="agent/issue-381")


def test_inspect_worktree_state_no_commits(tmp_path: Path) -> None:
    """A clean worktree with no commits beyond the base is no_commits."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-3", base_ref="origin/main")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.NO_COMMITS
    assert inspection.ahead_count == 0
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-3")


def test_inspect_worktree_state_unknown_missing_path(tmp_path: Path) -> None:
    """A missing worktree path returns unknown."""
    inspection = inspect_worktree_state(tmp_path / "does-not-exist")
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None


def test_inspect_worktree_state_empty_path_returns_unknown(tmp_path: Path) -> None:
    """Issue #660: an empty worktree_path (api workers set worktree_path="")
    must short-circuit to UNKNOWN instead of probing the caller's cwd.

    Path("") normalizes to Path("."), which is a real directory (the cwd).
    Without the guard, inspect_worktree_state would run real git merge-base /
    rev-list against whatever checkout the caller happens to be in, and could
    return COMPLETED if that checkout has local commits ahead of its base --
    silently misclassifying every dead api-worker session as "unpublished_work".
    """
    # Run from inside tmp_path (a real directory) to prove the guard fires
    # before any filesystem/git probe of cwd, regardless of cwd state.
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        inspection = inspect_worktree_state(Path(""))
    finally:
        os.chdir(old_cwd)
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None
    assert "empty" in inspection.error


def test_inspect_worktree_state_empty_path_short_circuits_before_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage gap in test_inspect_worktree_state_empty_path_returns_unknown:
    that test chdirs into a bare ``tmp_path`` (not a git repo), so removing the
    guard entirely still returns UNKNOWN there -- git merge-base/rev-list fail
    with "not a git repository" regardless of the guard, and only the error
    *string* assertion would catch a regression, not the state assertion.

    The actual danger (issue #660) requires a cwd that IS a real git checkout
    with commits ahead of its own base -- that is what turns the misprobe into
    WorktreeState.COMPLETED (see the merge-ahead branch a few lines below the
    guard). This test reproduces that exact precondition: a real repo, checked
    out with an unpublished commit ahead of origin/main, clean working tree.
    Without the guard this asserts COMPLETED (empirically confirmed by
    temporarily deleting the guard and re-running this exact scenario); with
    the guard it must stay UNKNOWN.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", "extra.txt")
    _git(repo, "commit", "-m", "ahead of origin/main")
    monkeypatch.chdir(repo)

    inspection = inspect_worktree_state(Path(""))
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None
    assert "empty" in inspection.error
