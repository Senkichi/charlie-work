"""Fleet pause/resume flag (issue #1776).

Covers: the flag primitives (atomic write, fail-closed read), the supervisor
honoring the flag at a pass boundary only, the pinned exit-code constant, the
launcher refusing to start while the flag exists, and the status block.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _fleet_dispatch_fixtures import (
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.fleet_pause import (
    clear_fleet_pause,
    fleet_pause_pending,
    fleet_pause_status,
    read_fleet_pause,
    run_fleet_pause,
    run_fleet_resume,
    write_fleet_pause,
)
from charlie_work.host.fakes import FakeClock
from charlie_work.supervise_loop import (
    EXIT_FLEET_PAUSED,
    EXIT_RESTART_REQUESTED,
    PREFLIGHT_REFUSAL_EXIT_CODE,
)
from charlie_work.supervisor_lifecycle import is_exit_alertable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _args(tmp_path: Path, **extra: Any) -> argparse.Namespace:
    return argparse.Namespace(fleet_dir=str(tmp_path), dry_run=False, **extra)


def test_exit_code_constant_is_pinned_and_distinct() -> None:
    # Cross-version wire contract (like EXIT_RESTART_REQUESTED): never renumber.
    assert EXIT_FLEET_PAUSED == 5
    assert len({EXIT_FLEET_PAUSED, EXIT_RESTART_REQUESTED, PREFLIGHT_REFUSAL_EXIT_CODE, 0, 1}) == 5
    assert is_exit_alertable(EXIT_FLEET_PAUSED) is False


def test_write_is_atomic_temp_then_replace(tmp_path: Path, patch_path_replace: Any) -> None:
    replaced: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def _spy(self: Path, target: Any) -> Path:
        replaced.append((self, Path(target)))
        return real_replace(self, target)

    patch_path_replace(_spy, scope=tmp_path)
    path = write_fleet_pause(str(tmp_path), reason="maintenance")

    assert len(replaced) == 1
    tmp_used, target = replaced[0]
    assert target == path
    # Unique temp name in the destination dir (issue #2265), still matching
    # the ``*.json.tmp`` orphan-sweep glob.
    assert tmp_used.parent == path.parent
    assert tmp_used.name != path.with_suffix(path.suffix + ".tmp").name
    assert tmp_used.name.startswith(f"{path.stem}.")
    assert tmp_used.name.endswith(f"{path.suffix}.tmp")
    assert not list(path.parent.glob("*.tmp"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reason"] == "maintenance"
    assert path == layout.fleet_pause_path(str(tmp_path))


def test_read_fails_closed_on_unreadable_flag(tmp_path: Path) -> None:
    assert read_fleet_pause(str(tmp_path)) is None
    assert fleet_pause_pending(str(tmp_path)) is False
    layout.fleet_pause_path(str(tmp_path)).write_text("{torn", encoding="utf-8")
    # A torn flag must still mean "paused" -- failing open would resume the fleet.
    assert read_fleet_pause(str(tmp_path)) == {}
    assert fleet_pause_pending(str(tmp_path)) is True


def test_pause_then_resume_commands(tmp_path: Path) -> None:
    result = run_fleet_pause(_args(tmp_path, reason="why"))
    assert result.ok and result.data["reason"] == "why"
    assert "parked floor" in result.message  # consequence is printed
    assert fleet_pause_pending(str(tmp_path))

    status = fleet_pause_status(str(tmp_path))
    assert status is not None and status["reason"] == "why"
    assert "parked floor" in status["consequence"]

    resumed = run_fleet_resume(_args(tmp_path))
    assert resumed.data["removed"] is True
    assert not fleet_pause_pending(str(tmp_path))
    assert fleet_pause_status(str(tmp_path)) is None
    assert run_fleet_resume(_args(tmp_path)).data["removed"] is False
    assert clear_fleet_pause(str(tmp_path)) is False


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = run_fleet_pause(
        argparse.Namespace(fleet_dir=str(tmp_path), dry_run=True, reason=None)
    )
    assert result.data["dry_run"] is True
    assert not fleet_pause_pending(str(tmp_path))


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_reads_flag_at_pass_boundary_only(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )

    def _pass_that_gets_paused_midway(**_kwargs: Any) -> Any:
        write_fleet_pause(str(tmp_path), reason="mid-pass")
        return _drained_fleet_result()

    mock_fleet_loop.side_effect = _pass_that_gets_paused_midway

    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.monotonic, sleep=fc.sleep, max_passes=5
    )

    # The flag landed mid-pass 1: that pass ran to completion (never
    # interrupted) and the supervisor stopped at the next boundary.
    assert mock_fleet_loop.call_count == 1
    assert result.data["exit_reason"] == "fleet_paused"
    assert result.data["fleet_paused"] is True
    assert result.data["restart_requested"] is False
    exit_call = _patch_self_deploy_for_fleet_tests["record_supervisor_exit"].call_args.kwargs
    assert exit_call["exit_code"] == EXIT_FLEET_PAUSED
    assert exit_call["reason"] == "fleet_paused"
    # Never consumed by the supervisor: only `resume` removes it.
    assert fleet_pause_pending(str(tmp_path))


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_started_while_paused_runs_no_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    write_fleet_pause(str(tmp_path))
    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.monotonic, sleep=fc.sleep, max_passes=5
    )
    assert mock_fleet_loop.call_count == 0
    assert result.data["exit_reason"] == "fleet_paused"


def test_launcher_filename_matches_layout() -> None:
    script = (REPO_ROOT / "scripts" / "fleet-pass.ps1").read_text(encoding="utf-8")
    assert f"'{layout.FLEET_PAUSE_FILENAME}'" in script


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_launcher_refuses_to_start_while_paused(tmp_path: Path) -> None:
    write_fleet_pause(str(tmp_path))
    log = REPO_ROOT / ".var" / "charlie-work" / "logs" / "fleet-pass.log"
    before = log.read_bytes() if log.exists() else b""
    env = {**__import__("os").environ, "CHARLIE_WORK_FLEET_DIR": str(tmp_path)}
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts" / "fleet-pass.ps1"),
        ],
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    added = log.read_bytes()[len(before) :].decode("utf-8", errors="replace")
    assert "NOT started: fleet paused" in added
    # It must not have reached the launch marker / supervisor.
    assert "fleet supervise-loop start" not in added
