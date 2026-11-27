"""``tree/`` detached worktree: create, sync, containment (issue #3117)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _worktree_fixtures import (
    _git,
    _init_bare_remote_and_clone,
    _init_repo,
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work.foreman import paths
from charlie_work.foreman import tree as tree_mod
from charlie_work.worktree import list_worktrees


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def _head(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def test_ensure_creates_detached_worktree_at_default_tip(tmp_path: Path, wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    result = tree_mod.ensure_tree(fdir, repo)
    assert result.ok, result.error
    tree = fdir / "tree"
    assert tree.is_dir()
    # Detached (abbrev-ref reports "HEAD", not a branch name) at the main tip.
    assert _git(tree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "HEAD"
    assert _head(tree) == _git(repo, "rev-parse", "main").stdout.strip()
    # Registered with the repo.
    assert any(w["worktree"].resolve() == tree.resolve() for w in list_worktrees(repo))


def test_ensure_is_idempotent(tmp_path: Path, wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    first = tree_mod.ensure_tree(fdir, repo)
    second = tree_mod.ensure_tree(fdir, repo)
    assert first.ok and second.ok
    assert first.value == second.value


def test_sync_moves_to_moved_default_tip(tmp_path: Path, wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    assert tree_mod.ensure_tree(fdir, repo).ok
    old_head = _head(fdir / "tree")
    # Advance the default branch in the registered checkout.
    (repo / "new.txt").write_text("new\n")
    _git(repo, "add", "new.txt")
    _git(repo, "commit", "-m", "advance main")
    synced = tree_mod.sync_tree(fdir, repo)
    assert synced.ok, synced.error
    assert _head(fdir / "tree") != old_head
    assert _head(fdir / "tree") == _git(repo, "rev-parse", "main").stdout.strip()


def test_sync_fetches_origin_tip_and_follows_it(tmp_path: Path, wt_scratch: Path) -> None:
    """``origin/<branch>`` is a remote-tracking ref: without a fetch it names a
    stale tip. origin advancing must move ``tree/`` on the next sync."""
    remote, clone = _init_bare_remote_and_clone(wt_scratch)
    fdir = _foreman(tmp_path)
    assert tree_mod.ensure_tree(fdir, clone).ok
    # Advance origin/main from a second clone; ``clone``'s cached ref stays stale.
    other = wt_scratch / "other-clone"
    subprocess.run(
        ["git", "clone", str(remote), str(other)], check=True, capture_output=True, text=True
    )
    _git(other, "config", "user.email", "test@example.test")
    _git(other, "config", "user.name", "Test User")
    (other / "adv.txt").write_text("advanced\n")
    _git(other, "add", "adv.txt")
    _git(other, "commit", "-m", "advance origin main")
    pushed = _git(other, "rev-parse", "HEAD").stdout.strip()
    _git(other, "push", "origin", "main")
    assert _git(clone, "rev-parse", "origin/main").stdout.strip() != pushed  # still stale
    synced = tree_mod.sync_tree(fdir, clone)
    assert synced.ok, synced.error
    assert _head(fdir / "tree") == pushed


def test_sync_without_tree_is_an_error_value(tmp_path: Path, wt_scratch: Path) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    result = tree_mod.sync_tree(fdir, repo)
    assert not result.ok
    assert "not a registered worktree" in result.error


def test_stale_tree_dir_is_an_error_value_not_a_delete(tmp_path: Path, wt_scratch: Path) -> None:
    """A ``tree/`` that is not a worktree cannot be removed by git — the error
    is reported, never worked around with a recursive delete."""
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    stale = fdir / "tree"
    stale.mkdir()
    (stale / "junk.txt").write_text("x")
    result = tree_mod.ensure_tree(fdir, repo)
    assert not result.ok
    assert stale.is_dir()  # still there — nothing swept it away


def test_broken_git_failure_is_an_error_value(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)

    class _Fail:
        ok = False
        error = "boom"
        stderr = ""
        stdout = ""

    monkeypatch.setattr(
        tree_mod,
        "run_captured",
        lambda *a, **k: _Fail(),  # every git call fails
    )
    result = tree_mod.ensure_tree(fdir, repo)
    assert not result.ok


def test_tree_path_containment_blocks_escape(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    try:
        (fdir / "tree").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    result = tree_mod.ensure_tree(fdir, tmp_path)
    assert not result.ok
    assert "escapes" in result.error


def test_registered_checkout_itself_is_untouched(tmp_path: Path, wt_scratch: Path) -> None:
    """The foreman's tree is never the operator's checkout (design §8)."""
    repo = wt_scratch / "repo"
    _init_repo(repo)
    fdir = _foreman(tmp_path)
    assert tree_mod.ensure_tree(fdir, repo).ok
    tree_mod.sync_tree(fdir, repo)
    # The registered repo still has its own checkout on a branch.
    branch = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert branch.returncode == 0
    assert branch.stdout.strip() == "refs/heads/main"
