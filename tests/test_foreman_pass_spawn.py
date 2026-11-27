"""The supervise loop's periodic ``charlie scheme`` spawn (issue #3122).

``_maybe_start_foreman_pass`` is the loop's only foreman edge — it owns the
argv, never imports the package (ADR-0010), polls the previous child rather
than waiting, and reports spawn failures to the fleet events.db. The unit
cases drive it directly; one integration case drives ``run_fleet_supervise``
to prove the loop wiring calls it on the interval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from _fleet_dispatch_fixtures import (
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)

from charlie_work import layout
from charlie_work.config import ForemanConfig, OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import _maybe_start_foreman_pass, run_fleet_supervise
from charlie_work.fleet_pause import write_fleet_pause
from charlie_work.host.fakes import (
    FakeClock,
    FakeForemanPassChild,
    FakeForemanPassLauncher,
)
from charlie_work.instrumentation import query_events

INTERVAL = 300.0


def _fleet_dir(tmp_path: Path) -> Path:
    fleet_dir = tmp_path / "fleet"
    (fleet_dir / "foremen" / "g1").mkdir(parents=True)
    return fleet_dir


def _spawn(fake: FakeForemanPassLauncher, fleet_dir: Path, **kw) -> tuple:
    args = {
        "now": 1000.0,
        "interval_seconds": INTERVAL,
        "enabled": True,
        "fleet_dir_override": str(fleet_dir),
    }
    args.update(kw)
    return _maybe_start_foreman_pass(None, 0.0, **args)


def test_spawn_fires_when_interval_elapsed(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir)

    assert isinstance(child, FakeForemanPassChild)
    assert anchor == 1000.0
    argv, cwd = launcher.calls[0]
    assert argv[:3] == (sys.executable, "-m", "charlie_work")
    assert argv[-1] == "scheme"
    assert "--fleet-dir" in argv
    assert argv[argv.index("--fleet-dir") + 1] == str(fleet_dir)


def test_spawn_skips_while_previous_child_alive(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)
    alive = FakeForemanPassChild(exit_code=None)

    child, anchor = _maybe_start_foreman_pass(
        alive,
        0.0,
        now=9999.0,
        interval_seconds=INTERVAL,
        enabled=True,
        fleet_dir_override=str(fleet_dir),
    )

    assert child is alive
    assert anchor == 0.0
    assert launcher.calls == []


def test_spawn_respawns_after_child_exits(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)
    exited = FakeForemanPassChild(exit_code=0)

    child, anchor = _maybe_start_foreman_pass(
        exited,
        0.0,
        now=9999.0,
        interval_seconds=INTERVAL,
        enabled=True,
        fleet_dir_override=str(fleet_dir),
    )

    assert child is not exited
    assert len(launcher.calls) == 1


def test_spawn_waits_for_the_interval(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir, now=100.0)
    # anchor arg is 0.0 -> 100s elapsed < 300s interval: no spawn.
    assert child is None and anchor == 0.0
    assert launcher.calls == []


def test_spawn_skipped_when_disabled_keeps_anchor(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir, enabled=False)

    assert child is None
    assert anchor == 0.0  # overdue stays overdue — fires when re-enabled
    assert launcher.calls == []


def test_spawn_skipped_with_no_foremen(tmp_path: Path, fake_host) -> None:
    fleet_dir = tmp_path / "fleet"
    (fleet_dir / "foremen" / "_shared").mkdir(parents=True)  # _shared is not a foreman
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir)

    assert child is None and anchor == 0.0
    assert launcher.calls == []


def test_spawn_skipped_while_fleet_paused(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    write_fleet_pause(str(fleet_dir), reason="operator")
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir)

    assert child is None and anchor == 0.0
    assert launcher.calls == []


def test_spawn_failure_events_and_moves_anchor(tmp_path: Path, fake_host) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher(["spawn blew up"])
    fake_host(foreman_pass_launch=launcher)

    child, anchor = _spawn(launcher, fleet_dir)

    assert child is None
    assert anchor == 1000.0  # retry waits a full interval, not next poll tick
    rows = query_events(layout.state_file_path(fleet_dir), kind="foreman_pass_start_failed")
    assert len(rows) == 1
    assert rows[0]["payload"]["error"] == "spawn blew up"
    assert "scheme" in rows[0]["payload"]["argv"]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_spawns_foreman_pass_on_interval(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    fake_host,
) -> None:
    """The loop spawns `charlie scheme` immediately, then every interval."""
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher([0])  # each child "exits" instantly
    fake_host(foreman_pass_launch=launcher)
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=10,
        ),
        foreman=ForemanConfig(pass_interval_minutes=1),
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = FakeClock(auto_advance=61.0)
    run_fleet_supervise(
        max_passes=3,
        clock=fc.monotonic,
        sleep=fc.sleep,
        fleet_dir_override=str(fleet_dir),
    )

    # Immediate first spawn, then a respawn each time 60s of FakeClock elapsed.
    assert len(launcher.calls) >= 2
    argv, _cwd = launcher.calls[0]
    assert argv[-1] == "scheme"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_no_spawn_when_disabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    fake_host,
) -> None:
    fleet_dir = _fleet_dir(tmp_path)
    launcher = FakeForemanPassLauncher()
    fake_host(foreman_pass_launch=launcher)
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=10,
        ),
        foreman=ForemanConfig(enabled=False),
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = FakeClock(auto_advance=61.0)
    run_fleet_supervise(
        max_passes=3,
        clock=fc.monotonic,
        sleep=fc.sleep,
        fleet_dir_override=str(fleet_dir),
    )

    assert launcher.calls == []
