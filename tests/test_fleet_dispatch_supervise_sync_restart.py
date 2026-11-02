"""Restart decision when a deferred dependency sync lands (``run_fleet_supervise``).

Kept apart from ``test_fleet_dispatch_supervise_self_deploy.py``, which is at the
file-size cap (issue #1442). Shared helpers and the autouse hermeticity fixtures
live in ``tests/_fleet_dispatch_fixtures.py``.
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
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.host.fakes import FakeClock
from charlie_work.supervise import SelfDeployResult


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_when_deferred_sync_lands_with_unmoved_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A deferred ``uv sync`` that lands restarts even though HEAD did not move.

    HEAD moved on an earlier, deferred pass, which already restarted, so this
    pass reports ``head_changed=False``. But the sync just replaced installed
    packages that this process imported at startup. Observed 2026-10-01: the
    sync installed ci-fleet 0.4.0 and the supervisor kept running 0.3.0 until
    an unrelated commit restarted it. ``synced=True`` must trigger the same
    restart exit as a HEAD move.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    # Shape copied from the production self_deploy_succeeded event that exposed
    # the bug: changed, synced, not deferred, HEAD not moved on this attempt.
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            head_changed=False,
            from_sha="abc123",
            to_sha="def456",
            message="updated and synced: def456",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.monotonic, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert mock_fleet_loop.call_count == 0
    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True
