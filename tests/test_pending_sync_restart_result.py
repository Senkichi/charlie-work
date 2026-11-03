"""``boot_sync_restart_result``: the supervisor's exit after a synced boot repair (issue #2919).

The helper was lifted out of ``run_fleet_supervise`` to keep the over-cap
``fleet_dispatch.py`` from growing (file-size ratchet, issue #1442). The end-to-end
behaviour -- real repair, real ``run_fleet_supervise``, the ``fleet_dispatch``
monkeypatches -- is pinned in ``test_fleet_dispatch_boot_sync_restart.py``. These tests
pin the seam itself, which those cannot: the watchdog probe/alert and
``RESTART_EXIT_REASONS`` arrive as arguments, so the helper must *use* what it is
handed (the end-to-end tests would still pass if it hardcoded ``restart_requested``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from charlie_work.command_result import CommandResult
from charlie_work.pending_sync import (
    BOOT_SYNC_EXIT_REASON,
    BootSyncRepair,
    boot_sync_restart_result,
)

_SYNCED = BootSyncRepair(detected_via="probe", synced=True)
_VOCABULARY = frozenset({"self_deploy", "head_drift", BOOT_SYNC_EXIT_REASON})


@dataclass(frozen=True)
class _FakeProbe:
    """Stands in for ``fleet_dispatch.WatchdogProbe`` (which this module cannot import)."""

    armed: bool | None
    detail: str = "task is Disabled"


class _Seams:
    """Records the order the helper touches its collaborators in.

    Doubles as the supervisor lock: ``release`` is the one method the helper calls on it.
    """

    def __init__(self, *, armed: bool | None) -> None:
        self.order: list[str] = []
        self.alerts: list[dict[str, Any]] = []
        self.armed = armed

    def release(self) -> None:
        self.order.append("release")

    def probe(self) -> _FakeProbe:
        self.order.append("probe")
        return _FakeProbe(self.armed)

    def alert(self, probe: Any, **kwargs: Any) -> None:
        self.order.append("alert")
        self.alerts.append({"probe": probe, **kwargs})


def _run(
    seams: _Seams,
    *,
    restart_exit_reasons: frozenset[str] = _VOCABULARY,
    probe: Callable[[], _FakeProbe] | None = None,
) -> CommandResult:
    return boot_sync_restart_result(
        _SYNCED,
        lock=seams,
        fleet_dir_override="fleet-dir",
        restart_exit_reasons=restart_exit_reasons,
        probe_watchdog=probe or seams.probe,
        alert_watchdog_not_armed=seams.alert,
    )


def test_result_reports_a_successful_boot_sync_restart() -> None:
    result = _run(_Seams(armed=None))

    assert result.ok is True
    assert result.message == "boot repair synced via probe; restarting for new deps"
    assert result.data == {"exit_reason": BOOT_SYNC_EXIT_REASON, "restart_requested": True}


@pytest.mark.parametrize(
    ("vocabulary", "expected"),
    [
        pytest.param(_VOCABULARY, True, id="reason-in-vocabulary"),
        pytest.param(frozenset({"self_deploy", "head_drift"}), False, id="reason-absent"),
    ],
)
def test_restart_requested_is_derived_from_the_vocabulary_it_is_given(
    vocabulary: frozenset[str], expected: bool
) -> None:
    """``restart_requested`` follows the caller's ``RESTART_EXIT_REASONS``, never a literal."""
    result = _run(_Seams(armed=None), restart_exit_reasons=vocabulary)

    assert result.data["exit_reason"] == BOOT_SYNC_EXIT_REASON
    assert result.data["restart_requested"] is expected


def test_disarmed_watchdog_is_alerted_before_the_lock_is_released() -> None:
    """Probe, then alert, then release: the alert must land while the lock is still held."""
    seams = _Seams(armed=False)

    _run(seams)

    assert seams.order == ["probe", "alert", "release"]
    (alert,) = seams.alerts
    assert alert["probe"].detail == "task is Disabled"
    assert alert["exit_reason"] == BOOT_SYNC_EXIT_REASON
    assert alert["notify_config"] is None  # config is not loaded yet at boot
    assert alert["notify_enabled"] is False
    assert alert["fleet_dir_override"] == "fleet-dir"
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", alert["now_str"])


@pytest.mark.parametrize("armed", [True, None], ids=["armed", "unknown"])
def test_only_a_confirmed_disarmed_watchdog_alerts(armed: bool | None) -> None:
    """``armed=None`` cannot prove the watchdog is down, so it must not alarm."""
    seams = _Seams(armed=armed)

    _run(seams)

    assert seams.order == ["probe", "release"]
    assert seams.alerts == []


def test_probe_failure_propagates_and_leaves_the_release_to_the_caller() -> None:
    """The helper does not swallow a probe error, nor release a lock it never finished with.

    ``run_fleet_supervise`` owns the lock through its startup ``try``; that handler's
    release on a raise is pinned in ``test_fleet_dispatch_boot_sync_restart.py``.
    """
    seams = _Seams(armed=None)

    def exploding_probe() -> _FakeProbe:
        raise RuntimeError("schtasks exploded")

    with pytest.raises(RuntimeError, match="schtasks exploded"):
        _run(seams, probe=exploding_probe)

    assert seams.order == []
