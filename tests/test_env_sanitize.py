"""Tests for env_sanitize's issue #646 dispatch-parallelism cap and issue #649
shared-venv confinement.

Covers the box-wide-saturation fix in isolation: sanitize_env()'s safe
defaults for PYTEST_XDIST_AUTO_NUM_WORKERS/UV_NO_SYNC, the
resolve_pytest_cap()/resolve_uv_no_sync() precedence helpers, and the handling
of .venv reparse points (junctions/symlinks) so a worker cannot rewrite a
shared venv. Adapter-level integration coverage (does
launch_claude_worker/launch_devin_session actually persist the resolved values
onto the sidecar) lives in test_claude_code_adapter.py / test_devin_shell.py
alongside their existing sanitize_env merge-order tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.env_sanitize import (
    DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS,
    PYTEST_XDIST_AUTO_NUM_WORKERS_VAR,
    UV_NO_SYNC_VAR,
    resolve_pytest_cap,
    resolve_uv_no_sync,
    sanitize_env,
)
from charlie_work.worktree import _create_junction_or_symlink, is_junction

# ---------------------------------------------------------------------------
# sanitize_env: safe-default injection
# ---------------------------------------------------------------------------


def test_sanitize_env_injects_default_pytest_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ambient cap set, sanitize_env must inject the safe default."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env[PYTEST_XDIST_AUTO_NUM_WORKERS_VAR] == DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS


def test_sanitize_env_respects_ambient_pytest_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-exported ambient cap must survive — setdefault, not overwrite."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "4")

    env = sanitize_env(worktree_path)

    assert env[PYTEST_XDIST_AUTO_NUM_WORKERS_VAR] == "4"


def test_sanitize_env_sets_uv_no_sync_when_venv_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree with its own real .venv must get UV_NO_SYNC=1 by default.

    UV_NO_SYNC is only ever paired with a confirmed, real local .venv — never
    with a reparse point that points into a shared venv.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env[UV_NO_SYNC_VAR] == "1"


def test_sanitize_env_omits_uv_no_sync_when_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree without its own .venv must NOT get UV_NO_SYNC forced on it.

    UV_NO_SYNC is only ever paired with a confirmed .venv (see sanitize_env's
    docstring) — a worktree that legitimately manages its own venv must still
    be able to run `uv sync` normally.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert UV_NO_SYNC_VAR not in env


def test_sanitize_env_respects_ambient_uv_no_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-exported ambient UV_NO_SYNC must survive even with a .venv present."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    env = sanitize_env(worktree_path)

    assert env[UV_NO_SYNC_VAR] == "0"


# ---------------------------------------------------------------------------
# resolve_pytest_cap: 3-way precedence (config > env > default)
# ---------------------------------------------------------------------------


def test_resolve_pytest_cap_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, None)

    assert (value, source) == (DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS, "default")


def test_resolve_pytest_cap_ambient_env_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "6")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, None)

    assert (value, source) == ("6", "env")


def test_resolve_pytest_cap_config_wins_over_ambient_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config-level worker_env override must beat both the ambient env and the default.

    This is the merge-order invariant documented in config.py: worker_env is
    merged AFTER sanitize_env's output, so an explicit operator override wins.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "6")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, {PYTEST_XDIST_AUTO_NUM_WORKERS_VAR: "1"})

    assert (value, source) == ("1", "config")


# ---------------------------------------------------------------------------
# resolve_uv_no_sync: 3-way precedence + the no-venv case
# ---------------------------------------------------------------------------


def test_resolve_uv_no_sync_no_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == (None, "no-venv")


def test_resolve_uv_no_sync_default_with_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("1", "default")


def test_resolve_uv_no_sync_ambient_env_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("0", "env")


def test_resolve_uv_no_sync_config_wins_over_ambient_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, {UV_NO_SYNC_VAR: "1"})

    assert (value, source) == ("1", "config")


def test_resolve_uv_no_sync_ambient_env_survives_without_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an ambient UV_NO_SYNC must be reported even with no .venv.

    sanitize_env() never *pops* an ambient UV_NO_SYNC -- it only skips the
    setdefault when there's no .venv -- so the value still reaches the child
    process regardless of .venv presence. A prior version of this function
    checked the no-venv case before the ambient-env case and incorrectly
    reported (None, "no-venv") here, understating what the subprocess
    actually inherited.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "1")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("1", "env")


def test_resolve_uv_no_sync_config_can_force_it_without_a_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit worker_env override must win even when no .venv is present."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, {UV_NO_SYNC_VAR: "1"})

    assert (value, source) == ("1", "config")


def test_resolve_uv_no_sync_reparse_point_reports_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .venv that is a reparse point must not be reported as an owned venv."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".git").write_text("gitdir: /fake/path", encoding="utf-8")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    _create_junction_or_symlink(worktree_path / ".venv", shared_venv)
    assert is_junction(worktree_path / ".venv")

    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)
    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == (None, "no-venv")


# ---------------------------------------------------------------------------
# Issue #649: a .venv reparse point must not masquerade as an owned local venv
# ---------------------------------------------------------------------------


def test_sanitize_env_sets_virtual_env_and_uv_no_sync_for_owned_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree with a real local .venv must get VIRTUAL_ENV and UV_NO_SYNC=1."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env.get("VIRTUAL_ENV") == str(worktree_venv)
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT is uv's default .venv path; setting it is a no-op"
    )
    assert env[UV_NO_SYNC_VAR] == "1"


def test_sanitize_env_omits_uv_project_environment_when_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree without its own .venv must NOT get UV_PROJECT_ENVIRONMENT set.

    With no target venv there is nothing to pin, and any ambient value was the
    orchestrator's leak (popped unconditionally on entry). The key must be
    absent so uv is free to manage its own venv.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_sanitize_env_orchestrator_uv_project_environment_never_survives_unchanged_with_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's ambient UV_PROJECT_ENVIRONMENT must never reach the
    worker unchanged — pins issue #117's leak guarantee, which this change must
    not erode.

    When the worktree has a real .venv, the orchestrator's value is popped and
    not re-set. uv's default project environment is .venv in the project root,
    which is the worktree's own venv, so the leaked value is replaced by
    absence rather than passed through. When it has no .venv, the value is
    simply absent.
    """
    orchestrator_venv = "/orchestrator/.venv"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", orchestrator_venv)

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped, not pinned to the worktree venv"
    )
    assert env.get("VIRTUAL_ENV") == str(worktree_venv)
    assert env.get("VIRTUAL_ENV") != orchestrator_venv


def test_sanitize_env_orchestrator_uv_project_environment_never_survives_unchanged_without_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's ambient UV_PROJECT_ENVIRONMENT must be dropped when
    the worktree has no .venv (issue #117 leak guard, undisturbed by #649).
    """
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_sanitize_env_unlinks_junctioned_shared_venv_in_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .venv reparse point in a worktree must be unlinked, not pinned.

    Setting UV_PROJECT_ENVIRONMENT to the .venv path is a no-op over uv's
    default project-environment lookup, and a junctioned .venv still resolves
    to the shared target. sanitize_env must detect the reparse point and remove
    it so uv is forced to create a real local .venv instead.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".git").write_text("gitdir: /fake/path", encoding="utf-8")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared\n", encoding="utf-8")

    _create_junction_or_symlink(worktree_path / ".venv", shared_venv)
    assert is_junction(worktree_path / ".venv")

    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    # The reparse point is gone; the shared venv is untouched.
    assert not is_junction(worktree_path / ".venv")
    assert not (worktree_path / ".venv").exists()
    assert marker.read_text(encoding="utf-8") == "shared\n"

    # No venv hints are passed to the worker.
    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert UV_NO_SYNC_VAR not in env


def test_sanitize_env_does_not_unlink_repo_root_venv_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must not unlink a .venv reparse point at the repo root.

    Repo roots have a .git directory, not a .git file. The sanitizer should
    leave the venv alone there and simply not set any venv variables.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared\n", encoding="utf-8")

    _create_junction_or_symlink(repo_root / ".venv", shared_venv)
    assert is_junction(repo_root / ".venv")

    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(repo_root)

    # Junction at repo root is not touched.
    assert is_junction(repo_root / ".venv")
    assert marker.read_text(encoding="utf-8") == "shared\n"

    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env
