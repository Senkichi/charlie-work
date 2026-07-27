"""Tests for env_sanitize's issue #646 dispatch-parallelism cap.

Covers the box-wide-saturation fix in isolation: sanitize_env()'s safe
defaults for PYTEST_XDIST_AUTO_NUM_WORKERS/UV_NO_SYNC, and the
resolve_pytest_cap()/resolve_uv_no_sync() precedence helpers used by the
adapter launch sites for diagnostic logging. Adapter-level integration
coverage (does launch_claude_worker/launch_devin_session actually persist the
resolved values onto the sidecar) lives in test_claude_code_adapter.py /
test_devin_shell.py alongside their existing sanitize_env merge-order tests.
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
    """A worktree with its own .venv must get UV_NO_SYNC=1 by default.

    Guards the shared/junctioned-venv wipe risk: an uncapped `uv run`/`uv
    sync` inside a worker must not be free to reinstall a venv that sibling
    worktrees may be concurrently running out of.
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
