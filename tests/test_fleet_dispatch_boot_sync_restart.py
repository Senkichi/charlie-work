"""A boot-time dependency repair that synced must restart the supervisor (issue #2919).

``heal_pending_sync_at_boot`` runs after the supervisor has imported everything,
so a sync that replaces installed packages leaves this process on the old
modules. These tests drive the *real* repair (only the ``git``/``uv`` subprocess
seam is faked) through the real ``run_fleet_supervise`` and pin the three
acceptance cases: a repair that synced requests a restart, a repair with nothing
to do does not, and a repair that failed does not (so a persistent sync failure
cannot become a restart loop).

Kept apart from ``test_fleet_dispatch_boot_sync_repair.py``, which pins the
ordering of the repair against the lock and the config load (issue #2312).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from _fleet_dispatch_fixtures import (
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from _supervise_fixtures import _make_fake_runner
from charlie_work import cli, layout
from charlie_work.command_result import CommandResult
from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import (
    RESTART_EXIT_REASONS,
    WatchdogProbe,
    run_fleet_supervise,
)
from charlie_work.host.fakes import FakeClock
from charlie_work.instrumentation import query_events
from charlie_work.pending_sync import (
    BOOT_SYNC_EXIT_REASON,
    BootSyncRepair,
    heal_pending_sync_at_boot,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path

_WATCHDOG_KIND = "supervisor_restart_watchdog_disabled"
_WATCHDOG_UNKNOWN = WatchdogProbe(armed=None, detail="not probed (mocked)")
_WATCHDOG_DISARMED = WatchdogProbe(armed=False, detail="task 'charlie-fleet-pass' is Disabled")

_PROBE_DRIFT = RunResult(1, "", "")  # ``uv sync --locked --check --inexact``: "would change"
_PROBE_CLEAN = RunResult(0, "", "")
_SYNC_OK = RunResult(0, "", "")
_SYNC_FAILED = RunResult(1, "", "error: failed to build a wheel")


class _Run:
    """The seams ``_supervise`` replaced, so a test can assert on what did NOT run."""

    def __init__(self, fleet_dir: Path, watchdog: WatchdogProbe) -> None:
        self.fleet_dir = fleet_dir
        self.lock = MagicMock(name="supervisor_lock")
        self.load_config = MagicMock(
            name="load_layered_config",
            return_value=OrchestratorConfig(
                supervisor=SupervisorConfig(
                    poll_interval_seconds=5,
                    full_pass_interval_seconds=1,
                    active_cooldown_seconds=7,
                )
            ),
        )
        self.fleet_loop = MagicMock(name="fleet_loop", return_value=_drained_fleet_result())
        self.probe = MagicMock(name="probe_fleet_watchdog", return_value=watchdog)
        self.emit = MagicMock(name="_emit_fleet_transition")

    def watchdog_events(self) -> list[dict]:
        return query_events(supervisor_heartbeat_path(str(self.fleet_dir)), kind=_WATCHDOG_KIND)


def _supervise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    heal: Callable[..., BootSyncRepair],
    *,
    watchdog: WatchdogProbe = _WATCHDOG_UNKNOWN,
) -> tuple[CommandResult, _Run]:
    """Run ``run_fleet_supervise`` for one pass with ``heal`` as the boot repair."""
    run = _Run(tmp_path / "fleet", watchdog)
    for name, replacement in (
        ("try_acquire_supervisor_lock", lambda *_a, **_k: run.lock),
        ("load_layered_config", run.load_config),
        ("fleet_loop", run.fleet_loop),
        ("probe_fleet_watchdog", run.probe),
        ("_emit_fleet_transition", run.emit),
        ("heal_pending_sync_at_boot", heal),
    ):
        monkeypatch.setattr(f"charlie_work.fleet_dispatch.{name}", replacement)
    fc = FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        max_passes=1, clock=fc.monotonic, sleep=fc.sleep, fleet_dir_override=str(run.fleet_dir)
    )
    return result, run


def _stub_repair(outcome: BootSyncRepair) -> Callable[..., BootSyncRepair]:
    return lambda *_a, **_k: outcome


def _real_repair(
    tmp_path: Path, responses: list[RunResult]
) -> tuple[Callable[..., BootSyncRepair], list[tuple[list[str], Path, int]]]:
    """The real ``heal_pending_sync_at_boot`` over a fake ``git``/``uv`` runner.

    Pointed at ``tmp_path`` (a checkout with a ``uv.lock`` and no marker, so the
    ``uv sync --check`` probe decides) with its state root under ``tmp_path``
    too, so the repair's events.db writes never reach the host's real state.
    Returns the repair and the runner's *live* call log (``(argv, cwd, timeout)``
    tuples): read it after the run, not at construction.
    """
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    runner, calls = _make_fake_runner(list(responses))

    def heal(
        _repo_root: Path, *, state_root: Path, live_count: Callable[[], int]
    ) -> BootSyncRepair:
        return heal_pending_sync_at_boot(
            tmp_path,
            state_root=layout.default_state_root(tmp_path),
            live_count=live_count,
            run_command=runner,
        )

    return heal, calls


def _exit_code(result: CommandResult, monkeypatch: pytest.MonkeyPatch) -> int:
    """The process exit code ``cli.main`` derives from a ``fleet supervise`` result."""
    monkeypatch.setattr(cli, "run_fleet_supervise", lambda **_kwargs: result)
    monkeypatch.setattr(cli, "raise_supervisor_to_normal", lambda *_a, **_k: None, raising=False)
    return cli.main(["fleet", "supervise"])


def test_boot_sync_is_a_member_of_the_restart_vocabulary() -> None:
    """``restart_requested`` is derived from this set, so the reason must be in it."""
    assert BOOT_SYNC_EXIT_REASON in RESTART_EXIT_REASONS


def test_synced_repair_requests_restart_before_config_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair that really synced exits for a restart, ahead of everything stale.

    The process holds the pre-sync modules, so it must not go on to load config or
    run a pass on them, and it must give back the supervisor lock it took so the
    relaunched process can acquire it. The exit code is the one the
    ``supervise-loop`` wrapper relaunches on.
    """
    heal, calls = _real_repair(tmp_path, [_PROBE_DRIFT, _SYNC_OK])

    result, run = _supervise(tmp_path, monkeypatch, heal)

    assert [c[0] for c in calls] == [
        ["uv", "sync", "--locked", "--check", "--inexact"],
        ["uv", "sync", "--locked", "--inexact"],
    ]
    assert result.ok is True
    assert result.data["exit_reason"] == BOOT_SYNC_EXIT_REASON
    assert result.data["restart_requested"] is True
    run.load_config.assert_not_called()
    run.fleet_loop.assert_not_called()
    run.lock.release.assert_called_once_with()
    assert _exit_code(result, monkeypatch) == cli.EXIT_RESTART_REQUESTED


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param([_PROBE_CLEAN], id="nothing-to-repair"),
        pytest.param([_PROBE_DRIFT, _SYNC_FAILED], id="sync-failed"),
    ],
)
def test_unsynced_real_repair_does_not_request_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responses: list[RunResult]
) -> None:
    """Nothing to do and a failed sync both leave the install as it was.

    A failed sync is the loop hazard: restarting on it would relaunch into the same
    failing repair indefinitely. The supervisor must carry on into its pass instead.
    """
    heal, _argv = _real_repair(tmp_path, responses)

    result, run = _supervise(tmp_path, monkeypatch, heal)

    assert result.data["restart_requested"] is False
    assert result.data["exit_reason"] == "max_passes"
    run.load_config.assert_called_once()
    run.fleet_loop.assert_called_once()
    assert _exit_code(result, monkeypatch) == 0


def test_deferred_repair_does_not_request_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live workers defer the sync: nothing was installed, so nothing to reload."""
    deferred = BootSyncRepair(
        detected_via="marker", deferred=True, detail="2 live fleet runner(s) still active"
    )

    result, run = _supervise(tmp_path, monkeypatch, _stub_repair(deferred))

    assert result.data["restart_requested"] is False
    assert result.data["exit_reason"] == "max_passes"
    run.load_config.assert_called_once()


def test_restart_follows_synced_even_when_detail_is_also_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart is keyed on ``synced`` alone, not on ``detail is None``.

    The previous ``if detail / elif synced`` shape would have let any future
    ``detail`` on a synced outcome silently suppress the restart.
    """
    outcome = BootSyncRepair(detected_via="probe", synced=True, detail="advisory note")

    result, run = _supervise(tmp_path, monkeypatch, _stub_repair(outcome))

    assert result.data["restart_requested"] is True
    run.load_config.assert_not_called()


def test_disarmed_watchdog_is_alerted_on_a_boot_sync_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #604 watchdog-armed check covers this restart site like the other two.

    Config is not loaded yet here, so there is no notify sink: the alert goes to
    the console and the fleet ``events.db`` and never to ``_emit_fleet_transition``.
    """
    outcome = BootSyncRepair(detected_via="probe", synced=True)

    result, run = _supervise(
        tmp_path, monkeypatch, _stub_repair(outcome), watchdog=_WATCHDOG_DISARMED
    )

    assert result.data["restart_requested"] is True
    run.probe.assert_called_once_with()
    events = run.watchdog_events()
    assert len(events) == 1
    assert events[0]["payload"]["exit_reason"] == BOOT_SYNC_EXIT_REASON
    run.emit.assert_not_called()


def test_unknown_watchdog_raises_no_alert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control for the test above: only a confirmed-disarmed watchdog alerts."""
    outcome = BootSyncRepair(detected_via="probe", synced=True)

    result, run = _supervise(tmp_path, monkeypatch, _stub_repair(outcome))

    assert result.data["restart_requested"] is True
    run.probe.assert_called_once_with()
    assert run.watchdog_events() == []
