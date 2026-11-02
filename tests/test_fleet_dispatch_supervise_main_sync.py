"""main-sync runs as a pass step in ``run_fleet_supervise`` (docs/specs/main-sync.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _fleet_dispatch_fixtures import (
    _drained_fleet_result,
)
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
)
from _fleet_dispatch_fixtures import (
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)

from charlie_work.config import NotifyConfig, OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.host.fakes import FakeClock
from charlie_work.main_sync import MainSyncPassResult, RepoSyncResult
from charlie_work.supervise import orchestrator_root


def _cfg(**extra: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        **extra,
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_main_sync_runs_once_per_full_pass_and_prints_lines(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    capsys: Any,
    tmp_path: Path,
) -> None:
    mock_load_config.return_value = _cfg()
    mock_fleet_loop.return_value = _drained_fleet_result()
    synced = MainSyncPassResult(
        results=(RepoSyncResult("o/a", "advanced", None, "a" * 40, "b" * 40),)
    )
    pass_mock = MagicMock(return_value=synced)
    monkeypatch.setattr("charlie_work.fleet_dispatch.run_main_sync_pass", pass_mock)

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.monotonic, sleep=fc.sleep)

    assert result.ok is True
    assert pass_mock.call_count == mock_fleet_loop.call_count == 2
    kwargs = pass_mock.call_args.kwargs
    assert kwargs["exclude_root"] == orchestrator_root()
    assert kwargs["dry_run"] is False
    assert "main-sync: o/a advanced aaaaaaaaa..bbbbbbbbb" in capsys.readouterr().out


@patch("charlie_work.fleet_dispatch._emit_fleet_transition")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_main_sync_alarm_reaches_fleet_attention(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_transition: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    mock_load_config.return_value = _cfg(notify=NotifyConfig(enabled=True, sink="file"))
    mock_fleet_loop.return_value = _drained_fleet_result()
    alarm = RepoSyncResult("o/a", "skipped", "unrelated_history", alarm="unrelated_history")
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.run_main_sync_pass",
        MagicMock(return_value=MainSyncPassResult(results=(alarm,))),
    )

    fc = FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.monotonic, sleep=fc.sleep)

    entries = [c.args[1] for c in mock_transition.call_args_list]
    main_sync_entries = [e for e in entries if e.adapter_kind.startswith("main-sync")]
    assert [(e.adapter_kind, e.health) for e in main_sync_entries] == [("main-sync o/a", "ERROR")]
