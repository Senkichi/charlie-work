"""In-pass deadline enforcement + last_seen rotation for ``fleet_loop`` (issue #1832).

Split out of ``tests/test_fleet_dispatch_loop_pass.py`` to keep that file under
the repo's 800-line file-size cap (issue #1442) -- these tests were net-new
additions, not a relocation of pre-existing coverage, so growing the existing
file past its cap for them (rather than adding a new, appropriately-sized
file) would have been the wrong call.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _make_fleet_json,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.instrumentation import query_events
from charlie_work.workflow import CommandResult


class _StepClock:
    """A deterministic fake monotonic clock for deadline tests.

    Returns ``steps`` in order, then repeats ``after`` forever once
    exhausted. Deliberately NOT tied to the exact number of ``pass_clock()``
    calls a given code path makes internally (e.g. a per-repo lane's own
    elapsed-time logging) -- only the calls a test cares about need an
    explicit, distinct step; everything past that reads a constant, so an
    unrelated extra/missing call elsewhere cannot flip the outcome.
    """

    def __init__(self, steps: list[float], after: float) -> None:
        self._steps = list(steps)
        self._after = after
        self._n = 0

    def __call__(self) -> float:
        value = self._steps[self._n] if self._n < len(self._steps) else self._after
        self._n += 1
        return value


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_deadline_defers_later_repos(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: a pass over its in-pass deadline defers later repos cleanly.

    Three repos are selected explicitly (bypassing registry ordering). The
    fake clock lets repo1's own deadline check pass, then reports the
    deadline exceeded for every call after -- repo2 and repo3's lanes must
    never start (no app.dispatch() call, no per_repo_results entry for
    either), and both must show up in ``data["deferred"]`` instead of being
    counted as failed.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo3": {
                "repo_root": str(tmp_path / "repo3"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    # steps[0]=pass_started_at, steps[1]=repo1's deadline check (1s elapsed,
    # under the 100s deadline -> repo1 proceeds); every call after reads
    # `after` (10000s elapsed -> over the deadline for repo2/repo3).
    clock = _StepClock(steps=[0.0, 1.0], after=10000.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=("owner/repo1", "owner/repo2", "owner/repo3"),
        work_only=True,
        deadline_seconds=100,
        pass_clock=clock,
    )

    assert result.data["repos"].keys() == {"owner/repo1"}
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["deferred"] == ["owner/repo2", "owner/repo3"]
    # A deferred repo is not a failure -- overall ok stays True.
    assert result.ok is True
    mock_app.dispatch.assert_called_once()

    from charlie_work.fleet_paths import fleet_dir

    fleet_state_path = layout.state_file_path(fleet_dir(override=str(tmp_path / "fleet")))
    deferred_events = query_events(fleet_state_path, kind="fleet_pass_deadline_deferred")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["deferred_repo_keys"] == ["owner/repo2", "owner/repo3"]


@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_deadline_rotates_last_seen(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: only repos actually attempted this pass get last_seen bumped.

    _select_repos orders an implicit (no explicit repos=) pass by oldest
    last_seen first. Without bumping last_seen for attempted repos, that
    order never changes pass to pass; a repo deferred every time would keep
    sorting identically to one that always runs. This test proves the fix:
    repo1 (attempted) gets a fresh last_seen; repo2/repo3 (deferred) keep
    their original last_seen, so _select_repos sorts them first next pass.
    """
    from charlie_work.fleet_dispatch import _select_repos
    from charlie_work.fleet_registry import _load_registry

    fleet_dir_path = tmp_path / "fleet"
    old_last_seen = "2020-01-01T00:00:00Z"
    registry_repos = {
        "owner/repo1": {
            "repo_root": str(tmp_path / "repo1"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
        "owner/repo2": {
            "repo_root": str(tmp_path / "repo2"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
        "owner/repo3": {
            "repo_root": str(tmp_path / "repo3"),
            "config_path": "orchestrator.config.yaml",
            "last_seen": old_last_seen,
        },
    }
    _make_fleet_json(tmp_path, fleet_dir_path, registry_repos)
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    clock = _StepClock(steps=[0.0, 1.0], after=10000.0)
    pass_now = datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC)

    fleet_loop(
        fleet_dir_override=str(fleet_dir_path),
        global_config=None,
        repos=None,
        work_only=True,
        deadline_seconds=100,
        pass_clock=clock,
        now=pass_now,
    )

    updated = _load_registry(fleet_dir_path / "fleet.json")
    expected_stamp = pass_now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    assert updated["repos"]["owner/repo1"]["last_seen"] == expected_stamp
    assert updated["repos"]["owner/repo2"]["last_seen"] == old_last_seen
    assert updated["repos"]["owner/repo3"]["last_seen"] == old_last_seen

    # Rotation in practice: next pass's implicit ordering now starts with
    # the two repos that were deferred, not the one that just ran.
    next_order = [key for key, _ in _select_repos(updated, None)]
    assert next_order[0] in {"owner/repo2", "owner/repo3"}
    assert next_order[-1] == "owner/repo1"


@patch("charlie_work.fleet_dispatch._run_fleet_autoscale_prologue")
@patch("charlie_work.fleet_dispatch._run_fleet_allocation_prologue")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_deadline_defers_autoscale_prologue(
    mock_load_registry: MagicMock,
    mock_allocation_prologue: MagicMock,
    mock_autoscale_prologue: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1832: the deadline is also checked between the two prologue "lanes".

    An empty registry means the per-repo loop body never runs, isolating
    this test to the allocation-vs-autoscale prologue boundary. The fake
    clock reports the deadline exceeded immediately after the allocation
    prologue returns, so autoscale must never be called.
    """
    mock_load_registry.return_value = {"repos": {}}
    mock_allocation_prologue.return_value = []

    # steps: pass_started_at=0, allocation_lane_start=0, elapsed-log call=5
    # (>= the 3s deadline) -> autoscale prologue must be skipped.
    clock = _StepClock(steps=[0.0, 0.0, 5.0], after=5.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        work_only=False,
        deadline_seconds=3,
        pass_clock=clock,
    )

    mock_allocation_prologue.assert_called_once()
    mock_autoscale_prologue.assert_not_called()
    assert result.data["deferred_autoscale_prologue"] is True
