"""Self-deploy / restart decision tests for ``run_fleet_supervise``.

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
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import (
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import run_fleet_supervise
from charlie_work.instrumentation import query_events
from charlie_work.supervise import SelfDeployResult


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploys_before_each_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The fleet supervisor calls self_deploy before every fleet_loop pass.

    ``from_sha == to_sha`` here deliberately means HEAD did not move on this
    pull (e.g. a pending dependency-sync marker with no new commit) --
    otherwise the supervisor's restart-for-fresh-code exit (see
    test_run_fleet_supervise_restarts_when_self_deploy_moves_head below)
    would legitimately break the loop after pass 1, since a running process
    never picks up newly-pulled source on its own.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            from_sha="abc123",
            to_sha="abc123",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3
    assert deploy_mock.call_count == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_when_self_deploy_moves_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A pull that actually moves HEAD exits the loop instead of continuing.

    This process already imported every charlie_work module at startup;
    git changing files on disk underneath it does not hot-reload those
    modules. Left looping, the supervisor would keep running whatever code
    was live at process start for its entire max-runtime-0 lifetime,
    silently ignoring every fix merged to main afterward (observed
    2026-07-22: the daemon ran ~40 minutes on stale code after several
    fixes had already landed on main, because self_deploy's git pull only
    updates files on disk -- it never made the already-running process
    pick them up). Exiting here hands control back to the scheduled-task
    watchdog, which relaunches a fresh process with the new commit
    actually imported.
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
    # Issue #604: the self-deploy restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test stays
    # hermetic -- no real ``schtasks`` subprocess call, no coupling to the
    # live state of the ``charlie-fleet-pass`` task -- and the alert path
    # (which fires only on a confirmed ``armed=False``) is not exercised
    # here. The dedicated watchdog-alert tests cover that path.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            head_changed=True,
            from_sha="abc123",
            to_sha="def456",
            message="updated and synced: def456",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    # max_passes=5 proves the exit is driven by the head-change detection,
    # not by exhausting the pass budget.
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert deploy_mock.call_count == 1
    # fleet_loop must never run this pass's (now-stale) code path.
    assert mock_fleet_loop.call_count == 0
    # #862: the exit must say *why*, so the launcher can relaunch immediately
    # instead of leaving the fleet unsupervised for a full watchdog interval.
    # ok=True alone is what made this indistinguishable from a clean timeout.
    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_does_not_restart_when_already_up_to_date(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """No new commits (from_sha == to_sha) must not trigger a restart-exit."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="abc123",
            to_sha="abc123",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_does_not_restart_on_deferred_sync_with_unmoved_head(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Regression guard for the total-fleet-outage bug (issue root cause).

    self_deploy reports a differing ``from_sha``/``to_sha`` pair even though
    HEAD did not move on *this* attempt, because those shas are carried
    forward from an earlier deferred-sync marker (see
    ``test_self_deploy_loud_warning_on_repeated_deferral`` in
    test_supervise.py for the producer side of this exact scenario). Gating
    the restart-exit on ``from_sha != to_sha`` instead of ``head_changed``
    made the supervisor exit and relaunch every single pass without ever
    reaching zero live workers to complete the deferred sync -- a total
    fleet outage. ``head_changed=False`` here must keep the loop running.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
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
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert deploy_mock.call_count == 3
    # The pending-sync marker's from_sha != to_sha must not trigger a
    # restart-exit when head_changed is False -- the loop must keep running
    # so live-worker draining can eventually reach zero and complete the
    # deferred sync.
    assert mock_fleet_loop.call_count == 3


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_restarts_on_external_head_drift(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """HEAD moved externally (operator pull, another process) triggers restart.

    self_deploy reports "already up to date" because HEAD was already at the
    new commit when the daemon's own git pull ran. Without an independent
    startup-vs-current HEAD comparison, the daemon would run stale code
    forever (observed 2026-07-23: ~90 minutes of ConfigError crashes).
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
    # Issue #604: the head-drift restart exit now probes the watchdog
    # scheduled task. Mock it to ``armed=None`` (unknown) so the test does
    # not perform a real ``schtasks`` subprocess call and stays hermetic --
    # the same pattern used by the self_deploy/read_head_sha/fleet_loop
    # mocks already applied in this test. The alert path (armed=False) is
    # covered by the dedicated watchdog-alert tests below.
    mock_probe.return_value = WatchdogProbe(armed=None, detail="not probed (mocked)")

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="def456",
            to_sha="def456",
            message="already up to date",
        )
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    # Simulate: startup HEAD is "abc123", then an external actor moved HEAD
    # to "def456" before the first pass. self_deploy sees "already up to date"
    # because its own pull didn't move anything, but the drift check catches it.
    sha_sequence = iter(["abc123", "def456", "def456"])
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.read_head_sha",
        lambda _root: next(sha_sequence),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert deploy_mock.call_count == 1
    # fleet_loop must never run with stale code.
    assert mock_fleet_loop.call_count == 0
    # head_drift is the OTHER half of the restart contract (RESTART_EXIT_REASONS
    # holds exactly self_deploy and head_drift). Only self_deploy was asserted
    # when the field was introduced, so an edit dropping the reason here would
    # have left drift silently non-restarting with every test still green.
    assert result.data["exit_reason"] == "head_drift"
    assert result.data["restart_requested"] is True


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_alerts_when_watchdog_disabled_on_head_drift(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A HEAD-drift restart with the watchdog disabled alerts + records an event.

    This is the #604 regression: on 2026-07-25 the drift exit fired correctly
    but the ``charlie-fleet-pass`` task was ``Enabled=false``, so no relaunch
    came and the fleet went dark silently. The exit must now surface the
    disarmed watchdog through a channel that is not the launcher log.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=False,
            synced=False,
            from_sha="def456",
            to_sha="def456",
            message="already up to date",
        ),
    )
    sha_sequence = iter(["abc123", "def456", "def456"])
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.read_head_sha", lambda _root: next(sha_sequence)
    )
    mock_probe.return_value = WatchdogProbe(
        armed=False, detail="task 'charlie-fleet-pass' is Disabled"
    )

    fleet_dir = tmp_path / "fleet"
    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            max_passes=5, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(fleet_dir)
        )

    assert result.data["exit_reason"] == "head_drift"
    assert result.data["restart_requested"] is True
    mock_probe.assert_called_once()

    # The alert reached the attention digest (a non-log channel).
    watchdog_calls = [
        call for call in mock_emit.call_args_list if call.args[1].adapter_kind == "fleet-watchdog"
    ]
    assert len(watchdog_calls) == 1
    entry = watchdog_calls[0].args[1]
    assert entry.health == "ERROR"
    assert watchdog_calls[0].kwargs.get("persistent") is False
    assert "disabled" in entry.last_log_line

    # And it was durably recorded to the fleet events.db.
    from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path

    events = query_events(
        supervisor_heartbeat_path(str(fleet_dir)), kind="supervisor_restart_watchdog_disabled"
    )
    assert len(events) == 1
    assert events[0]["payload"]["exit_reason"] == "head_drift"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_no_alert_when_watchdog_armed_or_unknown(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An armed or unknown watchdog must not false-alarm on a restart exit.

    ``armed=None`` (non-Windows, schtasks missing, task not found) is not
    proof the watchdog is disarmed, so it must stay quiet -- otherwise every
    non-Windows restart would page.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True, pulled=True, changed=False, synced=False, message="already up to date"
        ),
    )

    for armed, label in ((True, "armed"), (None, "unknown")):
        sha_sequence = iter(["abc123", "def456", "def456"])
        monkeypatch.setattr(
            "charlie_work.fleet_dispatch.read_head_sha", lambda _root: next(sha_sequence)
        )
        mock_probe.return_value = WatchdogProbe(armed=armed, detail=f"task state {label}")
        with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
            fc = _FakeClock(auto_advance=1.0)
            run_fleet_supervise(
                max_passes=5,
                clock=fc.now,
                sleep=fc.sleep,
                fleet_dir_override=str(tmp_path / "fleet"),
            )
        watchdog_calls = [
            call
            for call in mock_emit.call_args_list
            if call.args[1].adapter_kind == "fleet-watchdog"
        ]
        assert watchdog_calls == [], f"unexpected watchdog alert when {label}"


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_alerts_when_watchdog_disabled_on_self_deploy(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The self-deploy head-moved restart site also alerts when the watchdog is off.

    Both members of RESTART_EXIT_REASONS (self_deploy, head_drift) carry the
    same watchdog dependency; the verification must not be wired at only one
    of the two break sites.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import WatchdogProbe

    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5, full_pass_interval_seconds=1, active_cooldown_seconds=7
        ),
        notify=NotifyConfig(enabled=True, sink="file"),
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=True,
            from_sha="aaa111",
            to_sha="bbb222",
            message="fast-forwarded",
        ),
    )
    # startup_head == current_head so the drift branch does not also fire.
    monkeypatch.setattr("charlie_work.fleet_dispatch.read_head_sha", lambda _root: "bbb222")
    mock_probe.return_value = WatchdogProbe(
        armed=False, detail="task 'charlie-fleet-pass' is Disabled"
    )

    fleet_dir = tmp_path / "fleet"
    with patch("charlie_work.fleet_dispatch._emit_fleet_transition") as mock_emit:
        fc = _FakeClock(auto_advance=1.0)
        result = run_fleet_supervise(
            max_passes=5, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(fleet_dir)
        )

    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True
    mock_probe.assert_called_once()
    watchdog_calls = [
        call for call in mock_emit.call_args_list if call.args[1].adapter_kind == "fleet-watchdog"
    ]
    assert len(watchdog_calls) == 1
    assert watchdog_calls[0].args[1].health == "ERROR"
    assert watchdog_calls[0].kwargs.get("persistent") is False
    assert "disabled" in watchdog_calls[0].args[1].last_log_line
    # Durably recorded to the fleet events.db with the self_deploy reason.
    from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path

    sd_events = query_events(
        supervisor_heartbeat_path(str(fleet_dir)), kind="supervisor_restart_watchdog_disabled"
    )
    assert len(sd_events) == 1
    assert sd_events[0]["payload"]["exit_reason"] == "self_deploy"


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_error_dedups_across_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #554: a persistent self-deploy ERROR emits once, not every supervisor pass."""
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
    mock_lock.return_value = MagicMock()

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

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=2,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    # Two passes, same persistent ERROR — emit_digest fires once (first pass).
    assert mock_emit_digest.call_count == 1
    digest = mock_emit_digest.call_args[0][1]
    assert len(digest.transitions) == 1
    assert digest.transitions[0].health == "ERROR"
    assert digest.transitions[0].previous_health is None


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_failure_success_failure_emits_three_transitions(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #817 AC1: failure -> success -> failure must emit three digest
    entries, not one.

    Before this fix, the producer only ever constructed an AttentionEntry
    for a *failed* self_deploy (item 1's defect): the recovery pass built no
    entry at all, so the baseline sidecar stayed latched at ERROR from the
    first failure onward. ``_filter_fleet_health_transitions`` itself was
    already a correct edge-detector -- the second failure would read
    ERROR -> ERROR against that latched baseline and be suppressed as a
    non-transition, even though a real recovery happened in between.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path
    from charlie_work.fleet_dispatch import _load_fleet_health_state as _load_state

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
    mock_lock.return_value = MagicMock()

    deploy_mock = MagicMock(
        side_effect=[
            SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                error="fatal: Not possible to fast-forward, aborting.",
            ),
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha="abc123",
                to_sha="abc123",
                message="already up to date",
            ),
            SelfDeployResult(
                ok=False,
                pulled=False,
                changed=False,
                synced=False,
                error="fatal: Not possible to fast-forward, aborting.",
            ),
        ]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    # from_sha == to_sha on the success pass deliberately: a real HEAD move
    # would trigger the supervisor's separate restart-for-fresh-code exit
    # (see test_run_fleet_supervise_restarts_when_self_deploy_moves_head),
    # which would end the loop after pass 2 and never reach the third
    # failure this test needs to observe.
    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=3,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    assert mock_emit_digest.call_count == 3
    healths = [call.args[1].transitions[0].health for call in mock_emit_digest.call_args_list]
    assert healths == ["ERROR", "OK", "ERROR"]
    previous = [
        call.args[1].transitions[0].previous_health for call in mock_emit_digest.call_args_list
    ]
    assert previous == [None, "ERROR", "OK"]

    # Final persisted baseline reflects the third (failed) pass.
    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    assert _load_state(state_file) == {"self-deploy:-1": "ERROR"}


@patch("charlie_work.fleet_dispatch.emit_digest")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_self_deploy_success_clears_error_baseline(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_emit_digest: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Issue #817 AC2: after a failure -> success sequence, the *persisted*
    baseline sidecar itself reads back the healthy value -- not just the
    digest object returned in-process for that pass -- proving state
    genuinely moved off the ERROR latch. This is the fact AC1's third
    (failure) emission depends on: if the sidecar file did not actually
    change, the in-memory digest assertion alone would not distinguish a
    real fix from one that merely happens to return the right object once.
    """
    from charlie_work.config import NotifyConfig
    from charlie_work.fleet_dispatch import _fleet_health_state_path
    from charlie_work.fleet_dispatch import _load_fleet_health_state as _load_state

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
    mock_lock.return_value = MagicMock()

    deploy_mock = MagicMock(
        side_effect=[
            SelfDeployResult(
                ok=False, pulled=False, changed=False, synced=False, error="pull failed"
            ),
            SelfDeployResult(
                ok=True,
                pulled=True,
                changed=False,
                synced=False,
                from_sha="abc123",
                to_sha="abc123",
                message="already up to date",
            ),
        ]
    )
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=2,
        clock=fc.now,
        sleep=fc.sleep,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    assert _load_state(state_file) == {"self-deploy:-1": "OK"}
