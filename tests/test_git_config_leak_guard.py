"""Issue #2060 regression coverage: a test-side ``git config`` write must
never escape into the enclosing checkout's shared ``.git/config``.

Layers exercised:

* session env (``pytest_configure`` -> ``install_session_git_isolation``):
  ``GIT_CEILING_DIRECTORIES`` anchored at the realpath'd temp root so repo
  discovery cannot climb out of the sandbox, and ``GIT_CONFIG_GLOBAL`` /
  ``GIT_CONFIG_SYSTEM`` pointed at per-session files;
* the shared ``_git`` runner's repo-root anchor check, which refuses a
  repo-scoped ``config`` write whose cwd is not the repository root;
* ``protective_git_env``, which lets ``GIT_*``-scrubbing helpers keep the
  containment variables.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from _git_leak_guard import (
    GIT_ISOLATION_ENV_VARS,
    ceiling_directories,
    enclosing_repo_config_path,
    protective_git_env,
)
from _worktree_fixtures import _git, _init_repo


def test_shared_helper_refuses_config_write_from_non_repo_nested_in_repo(
    tmp_path: Path,
) -> None:
    """Acceptance: a non-repo dir nested inside a throwaway repo cannot have
    the enclosing repo's config written through it — the shared helper raises
    and the enclosing ``.git/config`` is byte-identical afterward."""
    outer = tmp_path / "outer"
    _init_repo(outer)
    enclosing_config = outer / ".git" / "config"
    before = enclosing_config.read_bytes()
    inner = outer / "inner"  # deliberately never `git init`'d
    inner.mkdir()

    with pytest.raises(AssertionError, match="issue #2060"):
        _git(inner, "config", "user.email", "leak@example.test")

    assert enclosing_config.read_bytes() == before


def test_raw_git_config_from_non_repo_tmp_dir_cannot_discover_a_repo(
    tmp_path: Path,
) -> None:
    """The session ceiling itself: even a bare subprocess (no helper) run in
    a non-repo dir under the temp root fails discovery instead of ascending
    into the enclosing checkout."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    result = subprocess.run(
        ["git", "config", "user.email", "leak@example.test"],
        cwd=plain,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "git config unexpectedly succeeded from a non-repo tmp dir — the "
        "session GIT_CEILING_DIRECTORIES containment is not installed"
    )


def test_shared_helper_still_writes_config_at_repo_root(tmp_path: Path) -> None:
    """Control: the anchor check does not break the normal case."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    _git(repo, "config", "user.email", "other@example.test")

    assert _git(repo, "config", "--get", "user.email").stdout.strip() == ("other@example.test")


def test_shared_helper_allows_config_read_from_repo_subdir(tmp_path: Path) -> None:
    """Reads are not mutations: ``--get`` from a subdir resolves fine."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    subdir = repo / "sub"
    subdir.mkdir()

    result = _git(subdir, "config", "--get", "user.email")

    assert result.stdout.strip() == "test@example.test"


def test_shared_helper_allows_config_write_at_bare_repo_root(tmp_path: Path) -> None:
    """Bare repos have no worktree toplevel; the anchor check must accept the
    bare gitdir itself (``_init_bare_remote_and_clone``'s ``core.longpaths``
    write is the existing call site)."""
    bare = tmp_path / "remote"
    bare.mkdir()
    _git(bare, "init", "--bare", "--initial-branch=main")

    _git(bare, "config", "core.longpaths", "true")

    assert _git(bare, "config", "--get", "core.longpaths").stdout.strip() == "true"


def test_global_config_writes_land_on_session_file_not_operator_file(
    tmp_path: Path,
) -> None:
    """``git config --global`` must write the per-session copy, and the copy
    must carry the operator's real values so reads are unaffected."""
    global_file = Path(os.environ["GIT_CONFIG_GLOBAL"])
    repo = tmp_path / "repo"
    _init_repo(repo)

    _git(repo, "config", "--global", "test.marker", "2060")

    content = global_file.read_text(encoding="utf-8")
    assert "marker" in content and "2060" in content
    # The write went to the throwaway session file under the temp root, never
    # the operator's ~/.gitconfig.
    assert os.path.normcase(os.path.realpath(global_file)) != os.path.normcase(
        os.path.realpath(Path.home() / ".gitconfig")
    )


def test_session_git_isolation_env_is_installed() -> None:
    """pytest_configure must have installed the containment env vars."""
    ceilings = os.environ.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep)
    assert str(Path(os.path.realpath(tempfile.gettempdir()))) in ceilings
    for var in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        value = os.environ.get(var)
        assert value is not None, f"{var} not set"
        assert Path(value).is_file(), f"{var} points at a missing file"


def test_protective_git_env_reexports_installed_vars() -> None:
    env = protective_git_env()
    for var in ("GIT_CEILING_DIRECTORIES", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        assert var in env


def test_enclosing_repo_config_path_resolves_to_existing_gitdir() -> None:
    """The byte-snapshot guard's watched file: the shared config of whatever
    repo encloses the pytest invocation cwd (the main checkout's ``.git`` when
    the suite runs from a linked worktree)."""
    config_path = enclosing_repo_config_path()
    if config_path is None:
        pytest.skip("session was not launched from inside a repository")
    assert config_path.name == "config"
    assert config_path.parent.is_dir()


def test_ceiling_directories_lists_root_and_ancestors(tmp_path: Path) -> None:
    entries = ceiling_directories(tmp_path)
    root = str(Path(os.path.realpath(tmp_path)))
    assert entries[0] == root
    assert str(Path(root).parent) in entries
    # Every ancestor of the root is present, so a cwd anywhere below an
    # ancestor stops at the nearest listed entry.
    assert str(Path(root).parent.parent) in entries


def test_git_isolation_env_vars_names_the_containment_set() -> None:
    assert "GIT_CEILING_DIRECTORIES" in GIT_ISOLATION_ENV_VARS
    assert "GIT_CONFIG_GLOBAL" in GIT_ISOLATION_ENV_VARS
    assert "GIT_CONFIG_SYSTEM" in GIT_ISOLATION_ENV_VARS
