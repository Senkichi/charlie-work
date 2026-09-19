"""Supervisor lifecycle instrumentation wiring (issue #627).

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    SUPERVISOR_BEAT_AT,
    SUPERVISOR_STARTED_AT,
    _FakeClock,
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import (
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import (
    _emit_fleet_transition,
    run_fleet_supervise,
)
from charlie_work.notify import AttentionEntry
from charlie_work.supervise import SelfDeployResult


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_abnormal_exit_alerts_when_notify_enabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """An abnormal (non-zero) exit routes to the attention digest when notify is on."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["is_exit_alertable"].return_value = True
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.side_effect = RuntimeError("boom")

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        run_fleet_supervise(max_passes=3)

    # The autouse fixture's self-deploy no-op also emits its own OK transition
    # unconditionally on every successful self-deploy (issue #817 fix, main-side
    # and unrelated to supervisor lifecycle), so two calls are expected here --
    # find the fleet-supervisor one specifically.
    fleet_supervisor_calls = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].adapter_kind == "fleet-supervisor"
    ]
    assert len(fleet_supervisor_calls) == 1
    entry = fleet_supervisor_calls[0].args[1]
    assert entry.adapter_kind == "fleet-supervisor"
    assert entry.health == "ERROR"
    assert fleet_supervisor_calls[0].kwargs.get("persistent") is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_clean_exit_does_not_alert(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A clean (exit_code=0) exit never reaches the attention digest."""
    # is_exit_alertable defaults to False in the fixture.
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.return_value = _drained_fleet_result()

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    # The autouse fixture's self-deploy no-op emits its own OK transition
    # unconditionally on every successful self-deploy (issue #817 fix, main-side
    # and unrelated to supervisor lifecycle) -- that is expected. What this test
    # actually guards is that the clean supervisor exit itself does not
    # additionally alert.
    fleet_supervisor_calls = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].adapter_kind == "fleet-supervisor"
    ]
    assert fleet_supervisor_calls == []


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_detects_and_records_prior_abnormal_exit(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A prior abnormal exit is detected and recorded before the new supervisor starts."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["detect_prior_abnormal_exit"].return_value = {
        "prior_pid": 4242,
        "prior_started_at": SUPERVISOR_STARTED_AT,
        "prior_last_beat_at": SUPERVISOR_BEAT_AT,
        "prior_pass_number": 9,
        "uptime_seconds": 2609.0,
    }
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    mocks["detect_prior_abnormal_exit"].assert_called_once()
    mocks["record_prior_abnormal_exit"].assert_called_once()
    prior_arg = mocks["record_prior_abnormal_exit"].call_args.args[1]
    assert prior_arg["prior_pid"] == 4242
    # The new supervisor's own start is still recorded after the retroactive exit.
    mocks["record_supervisor_started"].assert_called_once()


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_does_not_record_when_lock_held(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A second invocation rejected by the lock records no lifecycle events."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_lock.return_value = None
    mock_load_config.return_value = OrchestratorConfig()

    result = run_fleet_supervise()

    assert result.ok is False
    mocks["record_supervisor_started"].assert_not_called()
    mocks["record_supervisor_exit"].assert_not_called()


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_head_drift_exit_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A HEAD-drift restart records reason=head_drift_restart with exit_code=0."""
    from charlie_work.fleet_dispatch import WatchdogProbe

    mocks = _patch_self_deploy_for_fleet_tests
    # Issue #604: the head-drift restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    # Force the external HEAD-drift branch by making read_head_sha diverge.
    with patch("charlie_work.fleet_dispatch.read_head_sha") as mock_head:
        mock_head.side_effect = ["aaa", "bbb"]  # startup_head, then current_head
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "head_drift_restart"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_keyboard_interrupt_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """Ctrl+C records reason=keyboard_interrupt with exit_code=0."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.side_effect = [_drained_fleet_result(), KeyboardInterrupt]

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "keyboard_interrupt"


def test_supervisor_lifecycle_persistent_errors_are_deduped(tmp_path: Path) -> None:
    """Persistent health transitions still dedup by default."""
    notify_config = MagicMock()
    notify_config.enabled = True
    entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="self-deploy",
        health="ERROR",
        previous_health=None,
        last_log_line="self-deploy failed",
        pid=None,
    )

    fleet_dir = str(tmp_path / "fleet-persist")
    with patch("charlie_work.fleet_dispatch.emit_digest") as mock_emit:
        _emit_fleet_transition(notify_config, entry, fleet_dir)
        _emit_fleet_transition(notify_config, entry, fleet_dir)

    assert mock_emit.call_count == 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_prior_exit_alerts_when_notify_enabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A detected prior abnormal exit routes to the attention digest when notify is on."""
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["detect_prior_abnormal_exit"].return_value = {
        "prior_pid": 4242,
        "prior_started_at": SUPERVISOR_STARTED_AT,
        "prior_last_beat_at": SUPERVISOR_BEAT_AT,
        "prior_pass_number": 9,
        "uptime_seconds": 2609.0,
    }
    notify_config = MagicMock()
    notify_config.enabled = True
    mock_load_config.return_value = OrchestratorConfig(notify=notify_config)
    mock_fleet_loop.return_value = _drained_fleet_result()

    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        run_fleet_supervise(max_passes=1, clock=fc.now, sleep=fc.sleep)

    # The prior-exit ERROR transition is emitted.
    prior_emit = [
        call
        for call in mock_emit.call_args_list
        if call.args[1].last_log_line and "prior supervisor" in call.args[1].last_log_line
    ]
    assert prior_emit, "prior abnormal exit did not reach the attention digest"
    assert prior_emit[0].args[1].health == "ERROR"
    assert prior_emit[0].kwargs.get("persistent") is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_records_nonzero_exit_on_exception(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """An uncaught exception records supervisor_exited with exit_code=1."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig()
    mock_fleet_loop.side_effect = RuntimeError("boom")

    result = run_fleet_supervise(max_passes=3)

    assert result.ok is False
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 1
    assert exit_kwargs["reason"] == "exception"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_records_started_and_clean_exit(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A run stopped by max_passes emits supervisor_started once and supervisor_exited with exit_code=0."""
    mocks = _patch_self_deploy_for_fleet_tests
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_started"].assert_called_once()
    start_kwargs = mocks["record_supervisor_started"].call_args.kwargs
    assert start_kwargs["max_pass_runtime_seconds"] == 1800
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "max_passes"
    # Heartbeat refreshed once per loop iteration.
    assert mocks["update_supervisor_heartbeat"].call_count >= 2


def test_supervisor_lifecycle_repeated_errors_are_not_deduped(tmp_path: Path) -> None:
    """Supervisor-kill alerts are occurrence-style: repeated kills must all fire."""
    notify_config = MagicMock()
    notify_config.enabled = True
    entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="fleet-supervisor",
        health="ERROR",
        previous_health=None,
        last_log_line="prior supervisor terminated without an exit event",
        pid=4242,
    )

    fleet_dir = str(tmp_path / "fleet")
    with patch("charlie_work.fleet_dispatch.emit_digest") as mock_emit:
        _emit_fleet_transition(notify_config, entry, fleet_dir, persistent=False)
        _emit_fleet_transition(notify_config, entry, fleet_dir, persistent=False)

    assert mock_emit.call_count == 2


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_supervisor_lifecycle_self_deploy_head_move_reason(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """A self-deploy HEAD move records reason=self_deploy_head_moved with exit_code=0."""
    from charlie_work.fleet_dispatch import WatchdogProbe

    mocks = _patch_self_deploy_for_fleet_tests
    # Issue #604: the self-deploy restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(poll_interval_seconds=5, full_pass_interval_seconds=1)
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    # self_deploy is patched to a no-op by the autouse fixture; override it to
    # report a successful pull that moved HEAD.
    with patch(
        "charlie_work.fleet_dispatch.self_deploy",
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=True,
            from_sha="aaa",
            to_sha="bbb",
            message="moved",
        ),
    ):
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    mocks["record_supervisor_exit"].assert_called_once()
    exit_kwargs = mocks["record_supervisor_exit"].call_args.kwargs
    assert exit_kwargs["exit_code"] == 0
    assert exit_kwargs["reason"] == "self_deploy_head_moved"
