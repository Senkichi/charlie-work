"""Supervisor health-alert tests: zero-pass streak and repair digests.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _FakeClock,
    _drained_fleet_result,
    _make_fleet_json,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import (
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.instrumentation import query_events
from charlie_work.supervise import SelfDeployResult


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_emits_attention_digest_on_repair_failure(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A failed self_deploy repair emits an attention digest so it is never silent."""
    from charlie_work.config import NotifyConfig

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    lock = MagicMock()
    mock_lock.return_value = lock

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="venv pth repair failed: Access is denied",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    result = run_fleet_supervise(max_passes=1, fleet_dir_override=str(tmp_path / "fleet"))

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert deploy_mock.call_count == 1
    assert mock_emit_digest.called is True
    digest = mock_emit_digest.call_args[0][1]
    assert digest.repo == "fleet"
    assert len(digest.transitions) == 1
    assert digest.transitions[0].issue_number == -1
    assert digest.transitions[0].adapter_kind == "self-deploy"
    assert digest.transitions[0].health == "ERROR"
    assert "Access is denied" in digest.transitions[0].last_log_line


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_emits_attention_digest_on_venv_repaired(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A successful self_deploy venv repair emits an attention digest so it is never silent."""
    from charlie_work.config import NotifyConfig

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        ),
        notify=NotifyConfig(
            enabled=True,
            sink="file",
            file_path=str(tmp_path / "digest.jsonl"),
        ),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    lock = MagicMock()
    mock_lock.return_value = lock

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            venv_repaired=True,
            message="venv editable target repaired: shared venv editable .pth targets all resolve to configured checkouts",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    result = run_fleet_supervise(max_passes=1, fleet_dir_override=str(tmp_path / "fleet"))

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert deploy_mock.call_count == 1
    assert mock_emit_digest.called is True
    digest = mock_emit_digest.call_args[0][1]
    assert digest.repo == "fleet"
    assert len(digest.transitions) == 1
    assert digest.transitions[0].issue_number == -1
    assert digest.transitions[0].adapter_kind == "self-deploy"
    assert digest.transitions[0].health == "REPAIRED"
    assert "venv editable target repaired" in digest.transitions[0].last_log_line


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_never_fires_with_empty_registry(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #855 acceptance criterion 4, exercised end to end: a fleet with
    zero registered repos never fires the alarm, no matter how many
    consecutive zero-repo-pass cycles it runs -- that is a configuration
    state, not an incident.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: every cycle exits via the self-deploy HEAD-moved break,
    # which now probes the watchdog scheduled task. Mock it to ``armed=None``
    # (unknown) so every cycle stays hermetic -- no real ``schtasks``
    # subprocess call, no coupling to the live state of the
    # ``charlie-fleet-pass`` task -- and the alert path (which fires only on
    # a confirmed ``armed=False``) is not exercised here. The dedicated
    # watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    # Deliberately no _make_fleet_json call: the registry is empty.

    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            # run_fleet_supervise's restart gate reads head_changed, NOT
            # from_sha != to_sha (#853). Without this the simulated HEAD move
            # is a no-op, the supervisor never exits for a watchdog restart,
            # and this test stops exercising the #851 shape it is named for.
            head_changed=True,
            from_sha="a" * 12,
            to_sha="b" * 12,
            message="updated and synced: " + "b" * 12,
        ),
    )

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg

    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    for _ in range(6):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            fleet_dir_override=str(fleet_dir),
            max_passes=5,
            clock=fc.now,
            sleep=fc.sleep,
        )
        assert result.ok is True
        assert result.data["total_repo_passes"] == 0

    assert mock_fleet_loop.call_count == 0
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_replays_851_outage_shape(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #855 acceptance criterion 7: replay the #851 outage shape.

    N consecutive supervisor *process* restarts -- each modeled as a
    separate ``run_fleet_supervise()`` call, exactly like the Task Scheduler
    watchdog relaunching the process every cycle -- every one exiting via
    the self-deploy HEAD-moved break before ``fleet_loop`` ever runs: exit
    code 0 every cycle, ``repo_passes == 0`` every cycle (the log line the
    issue's evidence quotes: "1 pass(es) ... 0 repo pass(es)"), despite a
    repo being registered in the fleet. Exactly one
    ``supervisor_zero_pass_alarm`` must fire, at the cycle the persisted
    streak reaches the configured threshold (3) -- not one per restart.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: every cycle exits via the self-deploy HEAD-moved break,
    # which now probes the watchdog scheduled task. Mock it to ``armed=None``
    # (unknown) so every cycle stays hermetic -- no real ``schtasks``
    # subprocess call, no coupling to the live state of the
    # ``charlie-fleet-pass`` task -- and the alert path (which fires only on
    # a confirmed ``armed=False``) is not exercised here. The dedicated
    # watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(repo_root)}})

    # Isolate orchestrator_root() so the streak counter and alarm event
    # land under an ephemeral tmp_path state dir, never the real checkout
    # this test suite runs from.
    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg

    # Every self_deploy call reports a HEAD move -> run_fleet_supervise
    # exits (break) right after pass 1, before ever reaching fleet_loop this
    # process's lifetime.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            # run_fleet_supervise's restart gate reads head_changed, NOT
            # from_sha != to_sha (#853). Without this the simulated HEAD move
            # is a no-op, the supervisor never exits for a watchdog restart,
            # and this test stops exercising the #851 shape it is named for.
            head_changed=True,
            from_sha="a" * 12,
            to_sha="b" * 12,
            message="updated and synced: " + "b" * 12,
        ),
    )

    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    for cycle in range(1, 4):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            fleet_dir_override=str(fleet_dir),
            max_passes=5,
            clock=fc.now,
            sleep=fc.sleep,
        )
        assert result.ok is True
        assert result.data["passes"] == 1
        assert result.data["total_repo_passes"] == 0
        # fleet_loop must never run -- every cycle exits before reaching it.
        assert mock_fleet_loop.call_count == 0

        alarms = query_events(state_path, kind="supervisor_zero_pass_alarm")
        if cycle < 3:
            assert alarms == [], f"alarm fired early at cycle {cycle}"
        else:
            assert len(alarms) == 1, f"expected exactly one alarm by cycle {cycle}"
            assert alarms[0]["payload"]["consecutive_zero_pass_cycles"] == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_zero_pass_streak_resets_after_repo_work(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A cycle that actually reaches fleet_loop and performs repo work resets
    the streak to 0, so a later zero-pass streak has to build back up to the
    threshold instead of alarming immediately off carried-over count.
    """
    from charlie_work.fleet_dispatch import WatchdogProbe

    mock_lock.return_value = MagicMock()
    # Issue #604: the zero-repo-pass cycles exit via the self-deploy
    # HEAD-moved break, which now probes the watchdog scheduled task. Mock
    # it to ``armed=None`` (unknown) so every cycle stays hermetic -- no
    # real ``schtasks`` subprocess call, no coupling to the live state of
    # the ``charlie-fleet-pass`` task -- and the alert path (which fires
    # only on a confirmed ``armed=False``) is not exercised here. The
    # dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    fleet_dir = tmp_path / "fleet"
    isolated_root = tmp_path / "orchestrator-root"
    isolated_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _make_fleet_json(tmp_path, fleet_dir, {"owner/repo": {"repo_root": str(repo_root)}})

    monkeypatch.setattr("charlie_work.fleet_dispatch.orchestrator_root", lambda: isolated_root)

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            zero_pass_alarm=3,
        )
    )
    mock_load_config.return_value = cfg
    state_path = layout.state_file_path(layout.default_state_root(isolated_root))

    head_moved = SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        # run_fleet_supervise's restart gate reads head_changed, NOT
        # from_sha != to_sha (#853). Without this the simulated HEAD move
        # is a no-op, the supervisor never exits for a watchdog restart,
        # and this test stops exercising the #851 shape it is named for.
        head_changed=True,
        from_sha="a" * 12,
        to_sha="b" * 12,
        message="updated and synced: " + "b" * 12,
    )
    no_op = SelfDeployResult(
        ok=True, pulled=True, changed=False, synced=False, message="already up to date"
    )

    # Two zero-repo-pass cycles (streak -> 2, below threshold 3).
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: head_moved
    )
    for _ in range(2):
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(
            fleet_dir_override=str(fleet_dir), max_passes=5, clock=fc.now, sleep=fc.sleep
        )
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []

    # A cycle that actually performs repo work: self_deploy is a no-op, so
    # the loop proceeds to fleet_loop, which reports one repo processed.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: no_op
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(fleet_dir), max_passes=1, clock=fc.now, sleep=fc.sleep
    )
    assert result.data["total_repo_passes"] == 1

    # Two more zero-repo-pass cycles: if the streak had not reset, this
    # would already be 4 (past threshold 3) and would have alarmed already;
    # since it reset to 0, two more cycles land at 2 -- still below 3.
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy", lambda _repo_root, **_kwargs: head_moved
    )
    for _ in range(2):
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(
            fleet_dir_override=str(fleet_dir), max_passes=5, clock=fc.now, sleep=fc.sleep
        )
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []
