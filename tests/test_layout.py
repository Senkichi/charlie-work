"""Tests for :mod:`charlie_work.layout`, the single source of truth for path names.

layout.py exists to stop well-known filenames (``supervisor.lock``,
``fleet.json``, the ``.var/charlie-work`` default state dir, ...) from being
re-spelled at each use site -- see the module's own docstring for the live
split-brain bug (worktrees created under one root, cleaned from another) that
duplication caused. These tests pin the composed layout so a future rename of
any one of these paths shows up as a failing test rather than a silent drift
between two call sites that used to agree by coincidence.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from charlie_work import layout

# ---------------------------------------------------------------------------
# state-root helpers: each composes under whatever root it is given
# ---------------------------------------------------------------------------

_STATE_ROOT_HELPERS: list[object] = [
    pytest.param(layout.state_file_path, ("state.json",), id="state_file_path"),
    pytest.param(layout.supervisor_lock_path, ("supervisor.lock",), id="supervisor_lock_path"),
    pytest.param(layout.pending_sync_path, ("pending-sync.json",), id="pending_sync_path"),
    pytest.param(layout.worktrees_dir, ("worktrees",), id="worktrees_dir"),
    pytest.param(layout.dispatches_dir, ("dispatches",), id="dispatches_dir"),
    pytest.param(
        layout.sessions_dir_default, ("dispatches", "sessions"), id="sessions_dir_default"
    ),
    pytest.param(layout.reviews_dir_default, ("dispatches", "reviews"), id="reviews_dir_default"),
    pytest.param(
        layout.notify_digest_default, ("notify", "digest.jsonl"), id="notify_digest_default"
    ),
]


@pytest.mark.parametrize("helper, expected_parts", _STATE_ROOT_HELPERS)
def test_state_root_helper_composes_under_given_root(
    tmp_path: Path,
    helper: Callable[[Path], Path],
    expected_parts: tuple[str, ...],
) -> None:
    """Every ``state_root``-taking helper must build its path under that root.

    Asserts the exact expected relative parts (not just containment) --
    pinning the layout is the point, so a rename of a filename/dirname
    constant must turn into a failing assertion here.
    """
    root = tmp_path / "some-state-root"
    result = helper(root)
    assert result == root.joinpath(*expected_parts)
    assert result.is_relative_to(root)


# ---------------------------------------------------------------------------
# default_state_root
# ---------------------------------------------------------------------------


def test_default_state_root_is_repo_root_slash_var_charlie_work(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    assert layout.default_state_root(repo_root) == repo_root / ".var" / "charlie-work"


def test_default_state_root_consistent_with_DEFAULT_STATE_DIR(tmp_path: Path) -> None:
    """``default_state_root`` must agree with the ``DEFAULT_STATE_DIR`` constant.

    ``RuntimeConfig.state_dir`` reads ``DEFAULT_STATE_DIR`` as its default;
    ``default_state_root`` is the last-resort fallback for callers with no
    config in hand. The two must never diverge.
    """
    repo_root = tmp_path / "repo"
    assert layout.default_state_root(repo_root) == repo_root / Path(layout.DEFAULT_STATE_DIR)


# ---------------------------------------------------------------------------
# historical-path equivalence: the exact paths the pre-refactor code hardcoded
# ---------------------------------------------------------------------------
#
# These pin the layout against the literals that used to be scattered across
# the package (see layout.py's module docstring for the incident that
# motivated centralizing them). A rename here must not silently move state a
# running orchestrator already has on disk.


def test_supervisor_lock_path_matches_historical_literal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert layout.supervisor_lock_path(root) == root / "supervisor.lock"


def test_worktrees_dir_matches_historical_literal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assert (
        layout.worktrees_dir(layout.default_state_root(repo))
        == repo / ".var" / "charlie-work" / "worktrees"
    )


def test_reviews_dir_default_matches_historical_literal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assert (
        layout.reviews_dir_default(layout.default_state_root(repo))
        == repo / ".var" / "charlie-work" / "dispatches" / "reviews"
    )


def test_sessions_dir_default_matches_historical_literal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert layout.sessions_dir_default(root) == root / "dispatches" / "sessions"


def test_pending_sync_path_matches_historical_literal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assert (
        layout.pending_sync_path(layout.default_state_root(repo))
        == repo / ".var" / "charlie-work" / "pending-sync.json"
    )


def test_notify_digest_default_matches_historical_literal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert layout.notify_digest_default(root) == root / "notify" / "digest.jsonl"


def test_gh_config_dir_matches_historical_literal(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    assert layout.gh_config_dir(worktree) == worktree / ".var" / "gh-config"


# ---------------------------------------------------------------------------
# gh_config_dir is keyed on the worktree, not the state dir (deliberate)
# ---------------------------------------------------------------------------


def test_gh_config_dir_is_keyed_on_worktree_not_state_dir(tmp_path: Path) -> None:
    """``gh_config_dir`` must never route through ``charlie-work``'s state dir.

    Deliberate, per its docstring: each worktree gets an isolated ``gh``
    configuration so a worker cannot inherit or clobber ambient host
    credentials. It is keyed on the worktree path, not on
    ``runtime.state_dir`` -- asserting "charlie-work" is absent from the
    result guards against a future change accidentally nesting it under the
    orchestrator state root, which would defeat the isolation.
    """
    worktree = tmp_path / "some-worktree"
    result = layout.gh_config_dir(worktree)
    assert "charlie-work" not in result.parts


# ---------------------------------------------------------------------------
# fleet-dir helpers: honour an explicit override, then CHARLIE_WORK_FLEET_DIR
# ---------------------------------------------------------------------------

_FLEET_DIR_HELPERS: list[object] = [
    pytest.param(layout.global_config_path, "config.yaml", id="global_config_path"),
    pytest.param(layout.fleet_registry_path, "fleet.json", id="fleet_registry_path"),
    pytest.param(layout.fleet_lock_path, "fleet.lock", id="fleet_lock_path"),
    pytest.param(
        layout.fleet_supervisor_lock_path,
        "fleet-supervisor.lock",
        id="fleet_supervisor_lock_path",
    ),
    pytest.param(
        layout.notify_health_state_path,
        "notify_health_state.json",
        id="notify_health_state_path",
    ),
]


@pytest.mark.parametrize("helper, expected_filename", _FLEET_DIR_HELPERS)
def test_fleet_dir_helper_honours_explicit_override(
    tmp_path: Path,
    helper: Callable[..., Path],
    expected_filename: str,
) -> None:
    override_dir = tmp_path / "fleet-override"
    result = helper(override=str(override_dir))
    assert result == override_dir / expected_filename


@pytest.mark.parametrize("helper, expected_filename", _FLEET_DIR_HELPERS)
def test_fleet_dir_helper_honours_env_var_when_no_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper: Callable[..., Path],
    expected_filename: str,
) -> None:
    """When no ``override`` is passed, the fleet dir must follow the env var.

    ``tests/conftest.py`` already sets ``CHARLIE_WORK_FLEET_DIR`` for every
    test (suite-wide isolation from the operator's real fleet dir); this test
    re-sets it to its own tmp path so the assertion is unambiguous rather than
    relying on the exact value the autouse fixture happened to pick.

    Never asserted here: the real platform default (``%LOCALAPPDATA%`` /
    XDG state home). A test that fell through to that would depend on host
    layout and could pass or fail based on machine-specific state.
    """
    env_dir = tmp_path / "fleet-env"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(env_dir))
    result = helper()
    assert result == env_dir / expected_filename


# ---------------------------------------------------------------------------
# regression guard: per-repo vs. host-wide supervisor locks must not conflate
# ---------------------------------------------------------------------------


def test_per_repo_and_fleet_wide_supervisor_locks_are_distinct(tmp_path: Path) -> None:
    """The per-repo and fleet-wide supervisor locks must never collide.

    ``supervisor_lock_path`` serialises loops driving a single repo;
    ``fleet_supervisor_lock_path`` serialises the fleet supervisor across
    every repo on the host. Conflating them would let a single-repo loop and
    the fleet supervisor stomp on each other's lock file.
    """
    assert layout.SUPERVISOR_LOCK_FILENAME != layout.FLEET_SUPERVISOR_LOCK_FILENAME

    shared_dir = tmp_path / "shared-dir"
    per_repo = layout.supervisor_lock_path(shared_dir)
    fleet_wide = layout.fleet_supervisor_lock_path(override=str(shared_dir))
    assert per_repo != fleet_wide
