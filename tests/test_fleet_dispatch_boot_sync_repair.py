"""Boot-time pending-sync repair wiring tests for ``run_fleet_supervise``.

Issue #2312. Kept out of ``test_fleet_dispatch_supervise_self_deploy.py``
because that module is at the 800-line ratchet cap (issue #1442).
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
from charlie_work.pending_sync import BootSyncRepair


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_heals_pending_sync_before_config_load(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #2312: the pending-sync boot repair runs BEFORE the config load.

    A deploy whose ``uv sync`` was deferred under live workers can leave the
    checkout ahead of the venv; the layered config load below is exactly the
    step that crashes on missing deps, so the marker/probe repair must
    precede it -- a heal wired after ``load_layered_config`` would never
    reach the host it exists to rescue.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    order: list[str] = []
    mock_load_config.side_effect = lambda *a, **k: (order.append("config"), cfg)[1]
    mock_fleet_loop.return_value = _drained_fleet_result()

    heal_mock = MagicMock(
        side_effect=lambda *a, **k: (
            order.append("heal"),
            BootSyncRepair(),
        )[1]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.heal_pending_sync_at_boot", heal_mock)

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        max_passes=1,
        clock=fc.monotonic,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    assert result.ok is True
    heal_mock.assert_called_once()
    assert order[:2] == ["heal", "config"]
