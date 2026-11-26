"""``materialize_dirs`` / ``_materialize_directory`` / ``_is_git_tracked`` and the
injected-prompt cleanliness contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    _default_worktrees_dir,
    create_worktree,
    _is_git_tracked,
    _materialize_directory,
)

# ---------------------------------------------------------------------------
# Tests for materialize_dirs functionality
# ---------------------------------------------------------------------------


def test_is_git_tracked_returns_true_for_tracked_file(tmp_path: Path) -> None:
    """_is_git_tracked should return True for tracked files."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # README.md is tracked by default
    assert _is_git_tracked(repo_root, repo_root / "README.md") is True


def test_is_git_tracked_returns_false_for_untracked_file(tmp_path: Path) -> None:
    """_is_git_tracked should return False for untracked files."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked file
    untracked = repo_root / "untracked.txt"
    untracked.write_text("untracked\n", encoding="utf-8")

    assert _is_git_tracked(repo_root, untracked) is False


def test_is_git_tracked_returns_false_for_nonexistent_path(tmp_path: Path) -> None:
    """_is_git_tracked should return False for nonexistent paths."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    nonexistent = repo_root / "does-not-exist.txt"
    assert _is_git_tracked(repo_root, nonexistent) is False


def test_materialize_directory_copies_untracked_dir(tmp_path: Path) -> None:
    """_materialize_directory should copy untracked directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("config\n", encoding="utf-8")

    # Create a worktree
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Materialize the directory
    _materialize_directory(repo_root, worktree_path, ".devin")

    # Verify the directory was copied
    target_dir = worktree_path / ".devin"
    assert target_dir.exists()
    assert (target_dir / "config.json").read_text(encoding="utf-8") == "config\n"


def test_materialize_directory_preserves_tracked_files_and_copies_untracked_infra(
    tmp_path: Path,
) -> None:
    """_materialize_directory should copy untracked infra but not overwrite
    tracked files that are already identical in the target."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add prompt template")

    # Pre-existing target with the same tracked prompt template
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".devin" / "prompts").mkdir(parents=True)
    (worktree_path / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )

    written = _materialize_directory(repo_root, worktree_path, ".devin")

    # The tracked prompt template should be preserved (not overwritten)
    assert (worktree_path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "committed worker\n"
    # The untracked infra file should be copied
    assert (worktree_path / ".devin" / "config.json").read_text(encoding="utf-8") == "{}\n"
    assert ".devin/config.json" in written
    assert ".devin/prompts/worker.md" not in written


def test_materialize_directory_skips_nonexistent_source(tmp_path: Path) -> None:
    """_materialize_directory should skip nonexistent source directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Try to materialize a nonexistent directory - should not raise
    _materialize_directory(repo_root, worktree_path, "does-not-exist")

    # Worktree should still be empty
    assert not (worktree_path / "does-not-exist").exists()


def test_materialize_directory_merges_into_existing_target(tmp_path: Path) -> None:
    """_materialize_directory should copy missing files and preserve existing ones."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("original\n", encoding="utf-8")

    # Create a worktree with a pre-existing .devin directory
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    target_dir = worktree_path / ".devin"
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("existing\n", encoding="utf-8")

    # Materialize the directory - should merge, not skip wholesale
    written = _materialize_directory(repo_root, worktree_path, ".devin")

    # The existing file should still be there (not overwritten)
    assert (target_dir / "existing.txt").read_text(encoding="utf-8") == "existing\n"
    # The missing source file should have been copied
    assert (target_dir / "config.json").read_text(encoding="utf-8") == "original\n"
    assert ".devin/config.json" in written


def test_materialize_directory_preserves_empty_subdirectories(tmp_path: Path) -> None:
    """Empty source subdirectories must be recreated in the target (copytree did)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "empty_subdir").mkdir(parents=True)
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    _materialize_directory(repo_root, worktree_path, ".devin")

    assert (worktree_path / ".devin" / "empty_subdir").is_dir()
    assert not any((worktree_path / ".devin" / "empty_subdir").iterdir())
    assert (worktree_path / ".devin" / "config.json").read_text(encoding="utf-8") == "{}\n"


def test_materialize_directory_excludes_pycache_and_pyc(tmp_path: Path) -> None:
    """_materialize_directory must not copy __pycache__/ dirs or *.pyc files.

    Compiled bytecode is interpreter-version-specific and regenerable on first
    import.  Propagating stale mixed-version .pyc files (e.g. cpython-312 and
    cpython-313 side by side) into worktrees is a drift signal with no upside
    (issue #711).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Source .devin with a hooks subdir containing real source plus stale
    # mixed-version bytecode — mirroring the job-cannon layout from issue #711.
    hooks = repo_root / ".devin" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "require_ci_clean.py").write_text("# hook\n", encoding="utf-8")

    pycache = hooks / "__pycache__"
    pycache.mkdir()
    (pycache / "require_ci_clean.cpython-312.pyc").write_bytes(b"\x00\x32")
    (pycache / "require_ci_clean.cpython-313.pyc").write_bytes(b"\x00\x33")

    # A loose .pyc outside __pycache__ (some tools emit these) must also be
    # excluded.
    (hooks / "stray.cpython-313.pyc").write_bytes(b"\x00\x13")

    # A normal sibling file must still be copied.
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    written = _materialize_directory(repo_root, worktree_path, ".devin")

    # Real source and config are materialized.
    assert (worktree_path / ".devin" / "hooks" / "require_ci_clean.py").exists()
    assert (worktree_path / ".devin" / "config.json").exists()
    assert ".devin/hooks/require_ci_clean.py" in written
    assert ".devin/config.json" in written

    # __pycache__ directory is not created in the target at all.
    assert not (worktree_path / ".devin" / "hooks" / "__pycache__").exists()

    # No .pyc file appears anywhere in the materialized tree.
    assert not list((worktree_path / ".devin").rglob("*.pyc"))

    # No .pyc path is in the written manifest.
    assert not any(p.endswith(".pyc") for p in written)


def test_create_worktree_with_materialize_dirs(tmp_path: Path) -> None:
    """create_worktree should materialize specified directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "hooks.json").write_text("hooks\n", encoding="utf-8")

    # Create a worktree with materialize_dirs
    info = create_worktree(
        repo_root,
        "agent/issue-1-materialize",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    # Verify the directory was copied to the worktree
    target_dir = info.path / ".devin"
    assert target_dir.exists()
    assert (target_dir / "hooks.json").read_text(encoding="utf-8") == "hooks\n"
    assert ".devin/hooks.json" in info.materialized_paths


def _simulated_require_ci_clean(worktree_path: Path) -> bool:
    """Return True if the worktree has no staged or unstaged tracked changes."""
    staged = subprocess.run(
        ["git", "diff-index", "--cached", "--exit-code", "HEAD"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    unstaged = subprocess.run(
        ["git", "diff", "--exit-code"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    return staged.returncode == 0 and unstaged.returncode == 0


def test_materialize_directory_overwrites_tracked_prompt_and_hides_dirt(
    tmp_path: Path,
) -> None:
    """Issue #469: per-dispatch prompt files injected over tracked templates
    must not leave the worktree dirty in ``git status``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    # Simulate the orchestrator writing per-dispatch prompt content into the
    # source tree (the content that the materializer will copy).
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "per-dispatch worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "per-dispatch rework\n", encoding="utf-8"
    )

    # Create a target worktree that already has the committed templates tracked.
    worktree_path = tmp_path / "wt"
    _git(repo_root, "clone", str(repo_root), str(worktree_path))
    # Re-initialize the clone as a normal repo so git status is meaningful there.
    _git(worktree_path, "config", "user.email", "test@example.test")
    _git(worktree_path, "config", "user.name", "Test User")

    written = _materialize_directory(repo_root, worktree_path, ".devin")

    assert set(written) == {
        ".devin/prompts/worker.md",
        ".devin/prompts/rework.md",
        ".devin/config.json",
    }
    assert (worktree_path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch worker\n"
    assert (worktree_path / ".devin" / "prompts" / "rework.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch rework\n"

    # The copied prompt files should be marked assume-unchanged in the target.
    ls_files = _git(worktree_path, "ls-files", "-v", ".devin/prompts/worker.md")
    assert ls_files.stdout.startswith("h ")

    # No tracked-file dirt visible to git status or a strict CI clean-tree gate.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(worktree_path)


def test_create_worktree_materialize_dirs_injects_prompt_and_keeps_clean_tree(
    tmp_path: Path,
) -> None:
    """create_worktree with materialize_dirs must inject per-dispatch prompt
    content over tracked templates while leaving the worktree clean."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    # Per-dispatch prompt content in the source tree.
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "per-dispatch worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "per-dispatch rework\n", encoding="utf-8"
    )

    info = create_worktree(
        repo_root,
        "agent/issue-469-materialize",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    assert set(info.materialized_paths) == {
        ".devin/prompts/worker.md",
        ".devin/prompts/rework.md",
    }
    assert (info.path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch worker\n"

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(info.path)


def test_create_worktree_materialize_dirs_survives_external_tracked_prompt_write(
    tmp_path: Path,
) -> None:
    """Issue #469: tracked prompt files overwritten by an external shim after
    create_worktree returns must not dirty the worktree.

    The assume-unchanged bit must be applied proactively to every tracked path
    under the configured materialize surface at worktree-creation time, not only
    to paths whose content differed from the source tree during the copy loop.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    info = create_worktree(
        repo_root,
        "agent/issue-469-shim-write",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    # Source content matches HEAD, so the materializer does not rewrite anything.
    assert info.materialized_paths == ()

    # Simulate the external launch shim writing per-dispatch content into the
    # worktree after create_worktree has already returned.
    (info.path / ".devin" / "prompts" / "worker.md").write_text("shim worker\n", encoding="utf-8")

    ls_files = _git(info.path, "ls-files", "-v", ".devin/prompts/worker.md")
    assert ls_files.stdout.startswith("h ")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(info.path)


@pytest.fixture
def _materialize_fail_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Scaffold for ``test_create_worktree_materialize_dirs_error_cleanup``.

    The one-commit repo, the untracked ``.devin`` directory, and the failing
    ``_materialize_directory`` patch are acquisition, not the property under
    test. Housing them here charges the git-template copy to the ``setup``
    phase so the ledger's ``call`` phase holds only ``create_worktree``'s
    error-cleanup path -- the same remedy as #2624/#2634/#2670 (issue #2746).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("config\n", encoding="utf-8")

    def mock_materialize(*args: object, **kwargs: object) -> None:
        raise OSError("Mock materialization failure")

    monkeypatch.setattr(worktree_module, "_materialize_directory", mock_materialize)
    return repo_root


def test_create_worktree_materialize_dirs_error_cleanup(
    _materialize_fail_repo: Path,
) -> None:
    """Materialization failure should clean up the worktree and branch."""
    repo_root = _materialize_fail_repo

    with pytest.raises(RuntimeError, match="Failed to materialize directory"):
        create_worktree(
            repo_root,
            "agent/issue-2-materialize-fail",
            base_ref="HEAD",
            materialize_dirs=(".devin",),
        )

    # Verify the worktree was cleaned up
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktree_path = worktrees_dir / "agent-issue-2-materialize-fail"
    assert not worktree_path.exists()

    # Verify the branch was deleted. A branch ``git worktree add -b`` just
    # created is a loose ref under the common gitdir's refs/heads; ``git
    # branch -D`` removes that file, so its absence is the same verdict
    # ``git branch --list`` reported -- without another git spawn in the
    # measured call phase (issue #2746).
    branch_ref = repo_root / ".git" / "refs" / "heads" / "agent" / "issue-2-materialize-fail"
    assert not branch_ref.exists()
