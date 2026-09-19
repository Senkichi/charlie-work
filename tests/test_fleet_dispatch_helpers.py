"""Leaf-helper tests: repo selection, snapshots, pass-activity, watchdog probe, ci_fleet guard.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _active_fleet_result,
    _drained_fleet_result,
    _make_ci_fleet_git_repo,
    _make_fleet_json,
    _make_repo,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import RuntimeConfig
from charlie_work.fleet_dispatch import (
    FleetLocalSnapshot,
    _fleet_has_configured_repos,
    _has_fleet_delta,
    _is_fleet_pass_active,
    _lane_failure_state_path,
    _select_repos,
    _ci_fleet_worktree_dirty as _real_ci_fleet_worktree_dirty,
    _take_fleet_snapshot,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.fleet_registry import count_fleet_runners
from charlie_work.github import GitHubError


def test_ci_fleet_guard_clean_src_tree_is_inert(tmp_path: Path) -> None:
    """A clean src/ tree must let allocation proceed normally."""
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root == repo
    assert check.dirty_paths == ()


def test_ci_fleet_guard_detects_dirty_src_tree(tmp_path: Path) -> None:
    """Issue #927: the guard must fire when src/ has uncommitted changes.

    This is the positive-control test: a guard whose tests only exercise the
    clean path can pass forever while the dirty path is broken.
    """
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"
    # Uncommitted addition under src/ -- the same shape as an agent editing
    # planner.py in the live ci_fleet tree.
    (repo / "src" / "ci_fleet" / "planner.py").write_text("x = 1", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is True
    assert check.repo_root == repo
    assert check.dirty_paths
    assert any("planner.py" in p for p in check.dirty_paths)


def test_ci_fleet_guard_editable_install_outside_site_packages_still_checked(
    tmp_path: Path,
) -> None:
    """Issue #1511 regression guard: the site-packages short-circuit must not
    suppress the dirty check for a real editable install whose ``__file__``
    is NOT under site-packages.

    An editable install's resolved origin points at the source tree (e.g.
    ``<checkout>/src/ci_fleet/__init__.py``), never inside site-packages.
    The guard must still walk up to ``.git`` and detect uncommitted ``src/``
    changes in that tree -- the original #927 behavior.
    """
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"
    # Dirty an uncommitted file under src/ -- the #927 positive-control shape.
    (repo / "src" / "ci_fleet" / "planner.py").write_text("x = 1", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is True
    assert check.repo_root == repo
    assert any("planner.py" in p for p in check.dirty_paths)


def test_ci_fleet_guard_git_failure_is_inert(tmp_path: Path, monkeypatch: Any) -> None:
    """A git error must make the guard a no-op, not a hard stop."""
    repo = _make_ci_fleet_git_repo(tmp_path)
    module_file = repo / "src" / "ci_fleet" / "__init__.py"

    def _failing_git(*, cwd, timeout_seconds):  # pragma: no cover
        return RunResult(
            returncode=1,
            stdout="",
            stderr="git exploded",
            error="git exploded",
        )

    # Accept the positional `command` argument and ignore it.
    def _failing_run_captured(
        command: list[str], *, cwd: Path | str, timeout_seconds: int
    ) -> RunResult:
        return _failing_git(cwd=cwd, timeout_seconds=timeout_seconds)

    monkeypatch.setattr("charlie_work.fleet_dispatch.run_captured", _failing_run_captured)

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root == repo
    assert "git status failed" in (check.reason or "")


def test_ci_fleet_guard_no_git_is_inert(tmp_path: Path) -> None:
    """A wheel install or missing .git must not block allocation."""
    pkg = tmp_path / "pkg" / "ci_fleet"
    pkg.mkdir(parents=True)
    module_file = pkg / "__init__.py"
    module_file.write_text("# installed wheel", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root is None
    assert check.dirty_paths == ()


def test_ci_fleet_guard_wheel_in_venv_nested_in_worktree_is_inert(
    tmp_path: Path,
) -> None:
    """Issue #1511: a wheel install under ``.venv/site-packages`` nested inside
    a git worktree must not flag the worktree's own uncommitted ``src/`` as
    ci_fleet dirty.

    Reproduces the false positive: ``ci_fleet`` resolves to a wheel inside the
    worktree's ``.venv/Lib/site-packages``, the walk-up from there finds the
    worktree's own ``.git``, and the guard treats the entire active checkout
    as "the ci_fleet dependency tree" -- flagging it dirty for any uncommitted
    ``src/`` file, regardless of whether it touches ci_fleet at all. The fix
    short-circuits to inert before the walk-up when the resolved module path
    is under a ``site-packages``/``dist-packages`` directory.
    """
    # A git worktree (the standard ``git worktree add`` + ``uv venv`` layout)
    # with a clean committed tree, then a dirty unrelated ``src/`` file added.
    worktree = _make_ci_fleet_git_repo(tmp_path)
    unrelated_src = worktree / "src" / "charlie_work" / "config.py"
    unrelated_src.parent.mkdir(parents=True, exist_ok=True)
    unrelated_src.write_text("x = 1", encoding="utf-8")

    # ci_fleet is a wheel install inside the worktree's own .venv/site-packages.
    wheel_pkg = worktree / ".venv" / "Lib" / "site-packages" / "ci_fleet"
    wheel_pkg.mkdir(parents=True)
    module_file = wheel_pkg / "__init__.py"
    module_file.write_text("# wheel install", encoding="utf-8")

    check = _real_ci_fleet_worktree_dirty(module_file)

    assert check.is_dirty is False
    assert check.repo_root is None
    assert check.dirty_paths == ()
    assert "site-packages" in (check.reason or "")


@patch("charlie_work.fleet_registry._load_registry")
@patch("charlie_work.fleet_registry.GitHub")
def test_count_fleet_runners_propagates_runtime_config(
    mock_gh_class: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """count_fleet_runners passes the caller's RuntimeConfig to every GitHub client."""
    repo_root = tmp_path / "repo1"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo_root),
                "config_path": str(repo_root / "orchestrator.config.yaml"),
            }
        }
    }
    mock_load_registry.return_value = registry

    mock_gh = MagicMock()
    mock_gh.run.return_value = {"runners": [{"busy": False}, {"busy": True}]}
    mock_gh_class.return_value = mock_gh

    runtime = RuntimeConfig(gh_max_retries=7, gh_retry_base_seconds=0.5)
    total, busy, skipped = count_fleet_runners(str(tmp_path / "fleet"), runtime=runtime)

    assert total == 2
    assert busy == 1
    assert skipped == []
    mock_gh_class.assert_called_once_with(repo_root=repo_root, runtime=runtime)


@patch("charlie_work.fleet_registry._load_registry")
@patch("charlie_work.fleet_registry.GitHub")
def test_count_fleet_runners_skips_repo_on_unreadable_response(
    mock_gh_class: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #756: a repo whose runner query is unreadable (gh.run raises
    GitHubError, e.g. because github.py's own boundary fix now raises on
    empty-stdout-success) must land in skipped_repos, not silently contribute
    0 to total_runners/total_busy_runners.

    Before the #756 fix, ``GitHub.run()`` could return a bare ``None`` for
    this case, which ``runners_data.get(...) if runners_data else []``
    coerced to "this repo has zero runners" -- feeding decide_autoscale() a
    false reading that looks identical to a genuinely idle repo. The
    surrounding ``except (GitHubError, Exception)`` here already routes any
    raised exception to skipped_repos; this test proves that contract holds
    end to end.
    """
    repo_root = tmp_path / "repo1"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo_root),
                "config_path": str(repo_root / "orchestrator.config.yaml"),
            }
        }
    }
    mock_load_registry.return_value = registry

    mock_gh = MagicMock()
    mock_gh.run.side_effect = GitHubError(
        "gh exited 0 with empty stdout for command: gh api ...; "
        "cannot distinguish an empty result from an unreadable one"
    )
    mock_gh_class.return_value = mock_gh

    total, busy, skipped = count_fleet_runners(str(tmp_path / "fleet"))

    assert total == 0
    assert busy == 0
    assert skipped == ["owner/repo1"]


def test_fleet_has_configured_repos_false_with_empty_registry(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(tmp_path, fleet_dir, {})
    assert _fleet_has_configured_repos(str(fleet_dir), None) is False


def test_fleet_has_configured_repos_false_with_no_registry_file(tmp_path: Path) -> None:
    """A fleet.json that was never written (fresh host, nothing registered
    yet) must read the same as an explicitly empty registry."""
    assert _fleet_has_configured_repos(str(tmp_path / "never-written"), None) is False


def test_fleet_has_configured_repos_true_with_registered_repo(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(tmp_path / "repo")}})
    assert _fleet_has_configured_repos(str(fleet_dir), None) is True


def test_is_fleet_pass_active_false_when_drained() -> None:
    assert _is_fleet_pass_active(_drained_fleet_result()) is False


def test_is_fleet_pass_active_true_on_dispatch() -> None:
    assert _is_fleet_pass_active(_active_fleet_result()) is True


def test_lane_failure_state_path_falls_back_to_default_without_registry_entry(
    tmp_path: Path,
) -> None:
    """Without a recorded state_dir (a repo that has never registered
    successfully), _lane_failure_state_path falls back to the conventional
    default location so the failure is still recorded somewhere findable."""
    repo_root = tmp_path / "repo"
    entry: dict[str, Any] = {"repo_root": str(repo_root)}

    result = _lane_failure_state_path(repo_root, entry)

    assert result == layout.state_file_path(layout.default_state_root(repo_root))


def test_lane_failure_state_path_prefers_registry_state_dir(tmp_path: Path) -> None:
    """_lane_failure_state_path uses the registry's recorded state_dir when
    present — the common case for a repo that previously registered
    successfully and only later started failing (e.g. self-deploy version
    skew, the actual 2026-07-29 shape)."""
    repo_root = tmp_path / "repo"
    recorded_state_dir = tmp_path / "custom-state"
    entry = {"repo_root": str(repo_root), "state_dir": str(recorded_state_dir)}

    result = _lane_failure_state_path(repo_root, entry)

    assert result == layout.state_file_path(recorded_state_dir)


def test_probe_fleet_watchdog_parses_enabled_state() -> None:
    """probe_fleet_watchdog maps the 'Scheduled Task State' field to armed."""
    from charlie_work.fleet_dispatch import probe_fleet_watchdog

    enabled_output = (
        "TaskName:                             \\charlie-fleet-pass\n"
        "Status:                               Ready\n"
        "Scheduled Task State:                 Enabled\n"
        "Last Result:                          0\n"
    )
    disabled_output = (
        "TaskName:                             \\charlie-fleet-pass\n"
        "Status:                               Disabled\n"
        "Scheduled Task State:                 Disabled\n"
    )

    calls = iter([enabled_output, disabled_output])

    def _runner_seq(command, *, cwd, timeout_seconds):
        return RunResult(returncode=0, stdout=next(calls), stderr="")

    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(run_command=_runner_seq)
    assert probe.armed is True
    assert "Enabled" in probe.detail

    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(run_command=_runner_seq)
    assert probe.armed is False
    assert "Disabled" in probe.detail


def test_probe_fleet_watchdog_unknown_on_missing_field_and_non_windows() -> None:
    """An unparseable or non-Windows probe degrades to armed=None, never False."""
    from charlie_work.fleet_dispatch import probe_fleet_watchdog

    # No 'Scheduled Task State' line -> cannot determine.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(
            run_command=lambda command, *, cwd, timeout_seconds: RunResult(
                returncode=0, stdout="TaskName: \\charlie-fleet-pass\n", stderr=""
            )
        )
    assert probe.armed is None

    # schtasks query failure (task missing) -> unknown, not False.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "win32"
        probe = probe_fleet_watchdog(
            run_command=lambda command, *, cwd, timeout_seconds: RunResult(
                returncode=1, stdout="", stderr="task not found", error="command exited 1"
            )
        )
    assert probe.armed is None

    # Non-Windows -> not probed, unknown.
    with patch("charlie_work.fleet_dispatch.sys") as mock_sys:
        mock_sys.platform = "linux"
        probe = probe_fleet_watchdog()
    assert probe.armed is None


def test_select_repos_all_sorted_by_last_seen() -> None:
    """_select_repos returns all repos sorted by oldest last_seen first."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": "/path/to/repo1",
                "last_seen": "2026-07-07T10:00:00Z",
            },
            "owner/repo2": {
                "repo_root": "/path/to/repo2",
                "last_seen": "2026-07-07T09:00:00Z",  # Oldest
            },
            "owner/repo3": {
                "repo_root": "/path/to/repo3",
                "last_seen": "2026-07-07T11:00:00Z",
            },
        }
    }

    selected = _select_repos(registry, None)

    assert len(selected) == 3
    assert selected[0][0] == "owner/repo2"  # Oldest first
    assert selected[1][0] == "owner/repo1"
    assert selected[2][0] == "owner/repo3"


def test_select_repos_empty_registry() -> None:
    """_select_repos with empty registry returns empty list."""
    registry = {"repos": {}}

    selected = _select_repos(registry, None)

    assert selected == []


def test_select_repos_explicit_subset() -> None:
    """_select_repos with explicit repos returns exactly that subset in given order."""
    registry = {
        "repos": {
            "owner/repo1": {"repo_root": "/path/to/repo1", "last_seen": "2026-07-07T10:00:00Z"},
            "owner/repo2": {"repo_root": "/path/to/repo2", "last_seen": "2026-07-07T09:00:00Z"},
            "owner/repo3": {"repo_root": "/path/to/repo3", "last_seen": "2026-07-07T11:00:00Z"},
        }
    }

    selected = _select_repos(registry, ("owner/repo3", "owner/repo1"))

    assert len(selected) == 2
    assert selected[0][0] == "owner/repo3"  # Explicit order
    assert selected[1][0] == "owner/repo1"


def test_select_repos_explicit_subset_skips_missing() -> None:
    """_select_repos skips keys that don't exist in the registry."""
    registry = {
        "repos": {
            "owner/repo1": {"repo_root": "/path/to/repo1", "last_seen": "2026-07-07T10:00:00Z"},
            "owner/repo2": {"repo_root": "/path/to/repo2", "last_seen": "2026-07-07T09:00:00Z"},
        }
    }

    selected = _select_repos(registry, ("owner/repo3", "owner/repo1"))

    assert len(selected) == 1
    assert selected[0][0] == "owner/repo1"  # owner/repo3 doesn't exist, skipped


def test_select_repos_without_last_seen_goes_last() -> None:
    """Repos without last_seen are sorted last (treated as newest)."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": "/path/to/repo1",
                "last_seen": "2026-07-07T10:00:00Z",
            },
            "owner/repo2": {
                "repo_root": "/path/to/repo2",
                # No last_seen
            },
            "owner/repo3": {
                "repo_root": "/path/to/repo3",
                "last_seen": "2026-07-07T09:00:00Z",
            },
        }
    }

    selected = _select_repos(registry, None)

    assert len(selected) == 3
    assert selected[0][0] == "owner/repo3"  # Oldest with last_seen
    assert selected[1][0] == "owner/repo1"
    assert selected[2][0] == "owner/repo2"  # No last_seen, goes last


def test_take_fleet_snapshot_detects_delta_with_devin_sessions_dir_override(
    tmp_path: Path,
) -> None:
    """_take_fleet_snapshot uses the resolved sessions_dir, not the default.

    Regression for the review of issue #707: _repo_state_dirs built the
    sessions dir from layout.sessions_dir_default, so a devin.sessions_dir
    override produced no snapshot signal and no fleet delta.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    # Global fleet layer overrides the sessions dir.
    (fleet_dir / "config.yaml").write_text(
        "devin:\n  sessions_dir: custom-sessions\n",
        encoding="utf-8",
    )

    custom_sessions = repo / "custom-sessions"
    custom_sessions.mkdir(parents=True)
    (custom_sessions / "issue-1.json").write_text(
        _json.dumps({"dummy": "sidecar"}), encoding="utf-8"
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    before = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    (custom_sessions / "issue-2.json").write_text(
        _json.dumps({"dummy": "sidecar"}), encoding="utf-8"
    )

    after = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert _has_fleet_delta(before, after) is True


def test_take_fleet_snapshot_skips_repo_with_malformed_config(
    tmp_path: Path,
) -> None:
    """A repo with an unparseable per-repo config does not crash _take_fleet_snapshot.

    Regression for the review of issue #707: _take_fleet_snapshot's new
    load_layered_config call caught only ConfigError and OSError, so a
    malformed orchestrator.config.yaml (which raises yaml.YAMLError) crashed
    the fleet supervisor at startup.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    # Plant a malformed YAML file that yaml.safe_load cannot parse.
    (repo / "orchestrator.config.yaml").write_text(
        "devin:\n  sessions_dir: [unclosed\n",
        encoding="utf-8",
    )

    repos_map = {
        "owner/repo": {
            "repo_root": str(repo),
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    result = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert isinstance(result, FleetLocalSnapshot)


def test_take_fleet_snapshot_skips_repo_with_null_repo_root(
    tmp_path: Path,
) -> None:
    """A corrupted registry entry with repo_root: null does not crash _take_fleet_snapshot.

    Regression for the review of issue #707: ``Path(entry.get("repo_root", ""))``
    returns ``Path(None)`` when the key is present with a null value (``.get``'s
    default only applies when the key is *absent*), raising TypeError. The same
    bug class existed in compute_api_worker_fleet_report and the autoscale
    prologue; all three now use ``entry.get("repo_root") or ""``.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(tmp_path, "repo", api_worker=None)

    repos_map = {
        "owner/repo": {
            "repo_root": None,
            "config_path": str(repo / "orchestrator.config.yaml"),
            "state_dir": str(repo / ".var" / "charlie-work"),
        }
    }
    _make_fleet_json(tmp_path, fleet_dir, repos_map)

    result = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert isinstance(result, FleetLocalSnapshot)
