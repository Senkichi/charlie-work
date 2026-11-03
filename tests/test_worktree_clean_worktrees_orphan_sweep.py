"""``clean_worktrees``: removal of a junctioned worktree preserves the shared venv,
and the orphan sweep (reparse points, live foreign worktrees, ``worktree list``
failure)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from _worktree_fixtures import _FakeGH, _git, _init_repo
from _worktree_shared_fixtures import _make_state
from charlie_work.config import OrchestratorConfig
from charlie_work.worktree import (
    _default_worktrees_dir,
    clean_worktrees,
    create_worktree,
    is_junction,
)


def test_clean_worktrees_removes_junctioned_worktree_and_preserves_shared_venv_contents(
    tmp_path: Path,
) -> None:
    """Non-blocking review item: junction safety through the clean_worktrees
    path itself (not just remove_worktree in isolation). A worktree with a
    live ``.venv`` junction removed via clean_worktrees must delete only the
    reparse point, never the shared venv's real contents.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Mirror this repo's own .gitignore (.venv/ is ignored); otherwise the
    # .venv junction itself shows up as an untracked path under `git status
    # --porcelain` and the dirty check would (incorrectly, for this test's
    # purposes) treat every junctioned worktree as dirty.
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "add .gitignore")
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-12-junction", base_ref="HEAD", venv_source=venv_source
    )
    assert is_junction(info.path / ".venv")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    worktrees_dir = _default_worktrees_dir(repo_root)
    # No devin.venv_source configured here: this test is only about the
    # removal path's junction safety (remove_worktree), not the separate
    # post-removal poisoned-.pth check, which has its own dedicated tests.
    config = OrchestratorConfig()
    state = _make_state(issue_number=12, pr_number=112)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert len(result.data["removed"]) == 1
    assert not info.path.exists()
    # The shared venv itself, and its contents, must survive the junction-safe
    # removal driven through clean_worktrees.
    assert venv_source.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse point regression")
def test_clean_worktrees_orphan_sweep_removes_unregistered_tree_with_reparse_point(
    tmp_path: Path,
) -> None:
    """Issue #462: orphan directories left after a failed git worktree remove are
    detected and removed without following reparse points.
    """
    import _winapi

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "ignore venv")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    worktrees_dir = _default_worktrees_dir(repo_root)
    orphan_dir = worktrees_dir / "agent-issue-462-orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    _winapi.CreateJunction(str(shared_venv), str(orphan_dir / ".venv"))

    config = OrchestratorConfig()
    state = _make_state(issue_number=462, pr_number=462)
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert result.ok is True
    assert not orphan_dir.exists()
    assert shared_venv.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"
    assert any(str(orphan_dir) == r["worktree"] for r in result.data["orphans"]["removed"])


def test_clean_worktrees_orphan_sweep_spares_live_foreign_worktree(
    tmp_path: Path,
) -> None:
    """2026-08-09 incident: the orphan sweep deleted the sibling ci_runners
    worktree (another repo's live checkout inside this repo's worktrees dir,
    provisioned by worker launch shims for the ci-fleet editable) because it
    can never appear in this repo's own ``git worktree list``. A live foreign
    worktree must be spared; a dangling one (admin dir gone) is residue and
    stays sweepable.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    foreign_repo = tmp_path / "ci_runners"
    _init_repo(foreign_repo)
    foreign_sibling = worktrees_dir / "ci_runners"
    _git(foreign_repo, "worktree", "add", "--detach", str(foreign_sibling))
    assert (foreign_sibling / ".git").is_file()

    plain_orphan = worktrees_dir / "agent-issue-999-residue"
    plain_orphan.mkdir()
    (plain_orphan / "stale.txt").write_text("stale\n", encoding="utf-8")

    config = OrchestratorConfig()
    state = _make_state(issue_number=999, pr_number=999)
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert result.ok is True
    assert foreign_sibling.is_dir()
    assert not plain_orphan.exists()
    removed_paths = {r["worktree"] for r in result.data["orphans"]["removed"]}
    assert str(plain_orphan) in removed_paths
    assert str(foreign_sibling) not in removed_paths

    # Dangle the foreign worktree's registration: with the admin dir gone it
    # is residue, and the next sweep must reclaim it.
    shutil.rmtree(foreign_repo / ".git" / "worktrees")
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())
    assert result.ok is True
    assert not foreign_sibling.exists()
    removed_paths = {r["worktree"] for r in result.data["orphans"]["removed"]}
    assert str(foreign_sibling) in removed_paths


def test_clean_worktrees_skips_orphan_sweep_when_worktree_list_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A git worktree list failure must not be read as zero registered worktrees.

    The orphan sweep must be skipped and surfaced as an attention event so a
    transient git hiccup cannot silently destroy live worker state.
    """
    from charlie_work.subprocess_runner import RunResult
    import charlie_work.worktree as worktree_mod

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    info = create_worktree(repo_root, "agent/issue-1-live", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=1, pr_number=101)

    original_run = worktree_mod.run_captured

    def fake_run(args: list[str], **kwargs: Any) -> RunResult:
        if args[:4] == ["git", "worktree", "list", "--porcelain"]:
            return RunResult(
                returncode=1,
                stdout="",
                stderr="simulated git failure",
                error="simulated git failure",
            )
        return original_run(args, **kwargs)

    monkeypatch.setattr(worktree_mod, "run_captured", fake_run)

    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert info.path.exists()
    assert not result.data["orphans"]["removed"]
    assert not result.data["orphans"]["failed"]
    assert any(e["type"] == "worktree_list_failed" for e in result.data["attention_events"])
    assert result.ok is False
