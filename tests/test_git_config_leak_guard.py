"""Issue #2060 regression coverage: a test-side ``git config`` write must
never escape into the enclosing checkout's shared ``.git/config``.

Layers exercised:

* session env (``pytest_configure`` -> ``install_session_git_isolation``):
  ``GIT_CEILING_DIRECTORIES`` anchored at the realpath'd temp root so repo
  discovery cannot climb out of the sandbox, and ``GIT_CONFIG_GLOBAL`` /
  ``GIT_CONFIG_SYSTEM`` pointed at per-session files;
* the per-test ``_shared_repo_config_guard`` fixture, driven here against a
  throwaway watched file end to end — detect-only by design, so the file is
  asserted to carry exactly what the writer left after the failure;
* the shared ``_git`` runner's repo-root anchor check, which refuses a
  repo-scoped ``config`` write whose cwd is not the repository root;
* ``scrubbed_git_env`` and ``protective_git_env``, which let
  ``GIT_*``-scrubbing helpers keep the containment variables.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest
from _git_leak_guard import (
    ceiling_directories,
    enclosing_repo_config_path,
    protective_git_env,
)
from _worktree_fixtures import _git, _init_repo


def _config_guard_gen(
    monkeypatch: pytest.MonkeyPatch, watched: Path | None, nodeid: str
) -> Iterator[None]:
    """Drive the real autouse ``_shared_repo_config_guard`` end to end.

    Points the session-level watched path at a throwaway file, then invokes
    the fixture's underlying generator function directly: ``next(gen)``
    performs setup (the before-snapshot), a second ``next(gen)`` performs
    teardown (the detect-only comparison that raises ``pytest.fail.Exception`` on a
    mutation). Whatever happens between the two calls stands in for a test
    body. ``_fixture_function`` is pytest >= 9's unwrap; pytest < 9 keeps it
    on ``__pytest_wrapped__.obj``.
    """
    monkeypatch.setattr(conftest, "_ENCLOSING_REPO_CONFIG_PATH", watched)
    fixture_def = conftest._shared_repo_config_guard
    raw = getattr(fixture_def, "_fixture_function", None)
    if raw is None:
        raw = fixture_def.__pytest_wrapped__.obj
    return raw(SimpleNamespace(node=SimpleNamespace(nodeid=nodeid)))


def test_shared_repo_config_guard_fails_on_mutation_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a mutation of the watched file during the "test body"
    fails with the nodeid and the diff — and the guard's detect-only policy
    leaves the writer's bytes exactly as they were left."""
    watched = tmp_path / "shared" / "config"
    watched.parent.mkdir()
    watched.write_bytes(b"[user]\n\temail = operator@example.test\n")
    nodeid = "tests/test_leaky.py::test_leaks_user_email"
    gen = _config_guard_gen(monkeypatch, watched, nodeid)
    next(gen)  # fixture setup: before-snapshot

    leaked = b"[user]\n\temail = test@example.test\n"
    watched.write_bytes(leaked)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        next(gen)
    message = str(excinfo.value)
    assert nodeid in message
    # The rendered diff carries the evidence: the removed operator value and
    # the added leak are both visible.
    assert "operator@example.test" in message
    assert "test@example.test" in message
    # Detect-only policy: the guard wrote nothing back — the file still
    # holds exactly the bytes the writer left.
    assert watched.read_bytes() == leaked


def test_shared_repo_config_guard_fails_on_file_created_during_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``before is None`` path: a watched file that did not exist at
    setup and appears during the test is flagged, and the new file is left
    in place (detect-only never deletes either)."""
    watched = tmp_path / "shared" / "config"
    watched.parent.mkdir()
    nodeid = "tests/test_creator.py::test_creates_shared_config"
    gen = _config_guard_gen(monkeypatch, watched, nodeid)
    next(gen)

    created = b"[init]\n\tdefaultBranch = main\n"
    watched.write_bytes(created)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        next(gen)
    message = str(excinfo.value)
    assert nodeid in message
    assert "defaultBranch" in message
    assert watched.read_bytes() == created


def test_shared_repo_config_guard_fails_on_file_deleted_during_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deletion is a mutation too: the after-side becomes absent and the
    teardown must still fail — silently restoring the file would be exactly
    the write the detect-only policy forbids."""
    watched = tmp_path / "shared" / "config"
    watched.parent.mkdir()
    watched.write_bytes(b"[core]\n\tbare = false\n")
    nodeid = "tests/test_deleter.py::test_deletes_shared_config"
    gen = _config_guard_gen(monkeypatch, watched, nodeid)
    next(gen)

    watched.unlink()

    with pytest.raises(pytest.fail.Exception) as excinfo:
        next(gen)
    assert nodeid in str(excinfo.value)
    assert not watched.exists()


def test_shared_repo_config_guard_passes_when_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: an untouched watched file lets teardown complete silently —
    the generator just ends."""
    watched = tmp_path / "shared" / "config"
    watched.parent.mkdir()
    watched.write_bytes(b"[user]\n\temail = operator@example.test\n")
    gen = _config_guard_gen(monkeypatch, watched, "tests/test_clean.py::test_clean")
    next(gen)

    with pytest.raises(StopIteration):
        next(gen)


def test_shared_repo_config_guard_watches_nothing_outside_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``_ENCLOSING_REPO_CONFIG_PATH is None`` path (suite launched
    outside any repository): the fixture yields once and exits."""
    gen = _config_guard_gen(monkeypatch, None, "tests/test_no_repo.py::test_no_repo")
    next(gen)

    with pytest.raises(StopIteration):
        next(gen)


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
    """The *session-level* ceiling specifically, not the per-test fixture's.

    The subprocess env is rebuilt with ONLY the ``GIT_CEILING_DIRECTORIES``
    value captured at ``pytest_configure`` — every other ``GIT_*`` variable,
    including whatever the function-scoped ``_isolate_git_env`` merged on
    top, is stripped — so a discovery stop can only be credited to the
    session install. ``git rev-parse`` is read-only by design: if the
    session layer regresses, this probe discovers a repo but writes nothing.
    """
    session_ceiling = conftest._SESSION_GIT_CEILING
    assert session_ceiling is not None, (
        "pytest_configure installed no GIT_CEILING_DIRECTORIES — the "
        "session-level isolation layer is absent"
    )
    assert str(Path(os.path.realpath(tempfile.gettempdir()))) in session_ceiling.split(
        os.pathsep
    ), "the session ceiling does not cover the temp root"

    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = session_ceiling

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=plain,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "git discovered a repository from a non-repo tmp dir under only the "
        "session-installed ceiling — containment is not what stopped the ascent"
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


# The helpers that rebuild a subprocess env by dropping every GIT_* key.
# Behavior under test: the env each one actually hands subprocess.run keeps
# the session isolation variables while dropping the redirect hazard.
_GIT_ENV_SCRUBBING_HELPERS = (
    ("test_local_issues_loop_gates", "_init_repo"),
    ("test_local_issues_merge_gate", "_git"),
    ("test_local_issues_patch_equiv", "_git"),
    ("test_local_merge_gate_async", "_git"),
)


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    _GIT_ENV_SCRUBBING_HELPERS,
    ids=[module for module, _ in _GIT_ENV_SCRUBBING_HELPERS],
)
def test_git_env_scrubbing_helpers_preserve_isolation_vars(
    module_name: str,
    helper_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive each GIT_*-scrubbing helper and inspect the env it hands git.

    Hazards (``GIT_DIR`` redirect, ``GIT_CONFIG_*`` injection — the same
    mechanism ``_isolate_git_env`` uses for ``core.longpaths``) must be
    dropped; the session isolation set must survive, so a repo-scoped git
    run inside the scrub still cannot escape the sandbox.
    """
    module = importlib.import_module(module_name)
    helper = getattr(module, helper_name)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", "session-ceiling-sentinel")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "redirect-hazard"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.email")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "hazard@example.test")
    captured: list[dict[str, str] | None] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs.get("env")
        captured.append(env if isinstance(env, dict) else None)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    repo_root = tmp_path / "repo"
    if helper_name == "_init_repo":
        helper(repo_root)
    else:
        helper(repo_root, "rev-parse", "--git-dir")

    env = captured[0]
    assert env is not None, f"{module_name}.{helper_name} ran git without an explicit env"
    assert env["GIT_CEILING_DIRECTORIES"] == "session-ceiling-sentinel"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_DIR" not in env
    assert "GIT_CONFIG_COUNT" not in env
