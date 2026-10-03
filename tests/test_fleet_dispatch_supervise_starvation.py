"""Starvation-drain wiring tests for ``run_fleet_supervise`` (issue #1855).

Sibling of ``tests/test_fleet_dispatch_supervise_self_deploy.py`` -- split to
keep that file under the file-size ratchet cap; shared helpers and the
autouse hermeticity fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _fleet_dispatch_fixtures import (
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import (
    FleetSupervisorConfig,
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.host.fakes import FakeClock
from charlie_work.supervise import SelfDeployResult


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_drains_new_dispatch_while_sync_starved(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A ``starved`` self_deploy result puts fleet_loop in drain posture --
    ``drain=True`` launches nothing new while reap/review/merge lanes keep
    running -- for exactly the passes starvation holds.

    ``head_changed=False`` on the starved result is load-bearing: the pull
    that carried the dependency change already landed (and already triggered
    its own restart on the pass it arrived), so the supervisor must keep
    running drain passes -- not exit-restart -- until live workers finish
    and the deferred ``uv sync`` can land.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        fleet_supervisor=FleetSupervisorConfig(
            dependency_sync_starvation_seconds=7200,
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=False,
            from_sha="abc123",
            to_sha="def456",
            message="sync deferred: 2 runners active",
            deferred=True,
            starved=True,
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.monotonic, sleep=fc.sleep)

    assert result.ok is True
    # The loop keeps running -- a starved deferral is not a restart reason.
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3
    # The configured bound is plumbed into self_deploy.
    assert deploy_mock.call_args.kwargs["starvation_seconds"] == 7200
    # Every pass ran drained: no new dispatches while the sync starves.
    assert all(call.kwargs["drain"] is True for call in mock_fleet_loop.call_args_list)


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_drain_lifts_once_starved_sync_lands(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The starvation drain is derived per-pass from ``deploy.starved``, not
    latched. The pass where ``uv sync`` finally lands exits for a restart so
    the new dependencies load, and the relaunched process dispatches
    normally: nothing carries the drain across the restart."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    # The sync-landed pass exits for a restart, which probes the watchdog
    # task; keep it hermetic (armed=None skips the alert path).
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        side_effect=[
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=True,
                synced=False,
                head_changed=False,
                from_sha="abc123",
                to_sha="def456",
                message="sync deferred: 1 runner active",
                deferred=True,
                starved=True,
            ),
            # The fleet drained to zero this pass; the marker replayed and
            # uv sync landed. head_changed stays False (HEAD was already at
            # the new commit), but the sync replaced imported packages, so
            # this pass exits for a restart before dispatching.
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=True,
                synced=True,
                head_changed=False,
                from_sha="abc123",
                to_sha="def456",
                message="updated and synced: def456",
            ),
        ]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.monotonic, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True
    assert [c.kwargs["drain"] for c in mock_fleet_loop.call_args_list] == [True]

    # The relaunched process: episode fully resolved, an ordinary
    # up-to-date pass. The drain must not survive the restart.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        MagicMock(
            return_value=SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha="def456",
                to_sha="def456",
                message="already up to date",
            )
        ),
    )
    mock_fleet_loop.reset_mock()
    result = run_fleet_supervise(max_passes=2, clock=fc.monotonic, sleep=fc.sleep)

    assert result.data["passes"] == 2
    assert [c.kwargs["drain"] for c in mock_fleet_loop.call_args_list] == [False, False]
