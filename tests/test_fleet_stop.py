"""``charlie fleet stop [--drain]`` marker module and command (issue #1716).

The marker file is the only control surface a hidden scheduled fleet has:
``fleet stop`` writes it, ``fleet supervise`` consumes it between passes, and
``fleet supervise-loop`` consults it before honoring a child's restart
request.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from _fleet_dispatch_fixtures import (
    _FakeClock,
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import cli, fleet_stop, layout
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig, SupervisorConfig
from charlie_work.fleet_dispatch import (
    FleetLocalSnapshot,
    WatchdogProbe,
    fleet_loop,
    run_fleet_supervise,
    run_fleet_supervise_loop,
)
from charlie_work.fleet_stop import (
    clear_fleet_stop_request,
    read_fleet_stop_request,
    write_fleet_stop_request,
)
from charlie_work.instrumentation import query_events
from charlie_work.supervise import LocalSnapshot
from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED
from charlie_work.supervisor_lifecycle import supervisor_heartbeat_path
from charlie_work.workflow import CommandResult


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    """The marker lands at the centralized layout path with the drain flag."""
    path = write_fleet_stop_request(str(tmp_path), drain=True)

    assert path == layout.fleet_stop_request_path(str(tmp_path))
    request = read_fleet_stop_request(str(tmp_path))
    assert request is not None
    assert request["drain"] is True
    assert request["kind"] == "fleet_stop_request"
    assert request["requested_at"]
    assert isinstance(request["requester_pid"], int)


def test_read_returns_none_when_absent(tmp_path: Path) -> None:
    """No marker is the normal steady state — readers must not choke on it."""
    assert read_fleet_stop_request(str(tmp_path)) is None


def test_read_tolerates_a_corrupt_marker(tmp_path: Path) -> None:
    """A malformed marker must not wedge the supervisor; it reads as absent.

    Fail-open here is deliberate: the operator can re-run ``fleet stop``,
    which overwrites atomically, while a reader that crashed on corrupt
    bytes could never be asked to stop at all.
    """
    path = layout.fleet_stop_request_path(str(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert read_fleet_stop_request(str(tmp_path)) is None


def test_clear_removes_the_marker(tmp_path: Path) -> None:
    """The marker is consumed by whoever honors it — exactly once."""
    write_fleet_stop_request(str(tmp_path), drain=False)

    assert clear_fleet_stop_request(str(tmp_path)) is True
    assert read_fleet_stop_request(str(tmp_path)) is None
    # Already-consumed is a reportable state, not an error.
    assert clear_fleet_stop_request(str(tmp_path)) is False


def test_last_write_wins_on_repeat_requests(tmp_path: Path) -> None:
    """``fleet stop`` after ``fleet stop --drain`` de-escalates to plain stop."""
    write_fleet_stop_request(str(tmp_path), drain=True)
    write_fleet_stop_request(str(tmp_path), drain=False)

    request = read_fleet_stop_request(str(tmp_path))
    assert request is not None
    assert request["drain"] is False


def _stop_args(
    tmp_path: Path, *, drain: bool = False, dry_run: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(fleet_dir=str(tmp_path), drain=drain, dry_run=dry_run)


def test_run_fleet_stop_is_reexported_from_cli() -> None:
    """The command implementation lives in fleet_stop; cli is the facade.

    The file-size extraction moved the body verbatim — the dispatch table
    calls ``cli.run_fleet_stop``, which must resolve to the same function.
    """
    assert cli.run_fleet_stop is fleet_stop.run_fleet_stop


def test_run_fleet_stop_writes_the_marker_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command writes the marker and reports liveness/watchdog context."""
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.probe_fleet_watchdog",
        lambda: WatchdogProbe(armed=True, detail="Enabled"),
    )

    result = cli.run_fleet_stop(_stop_args(tmp_path, drain=True))

    assert result.ok is True
    assert result.data["drain"] is True
    request = read_fleet_stop_request(str(tmp_path))
    assert request is not None and request["drain"] is True
    # No supervisor heartbeat under tmp_path: reported, not silently assumed.
    assert result.data["supervisor_live"] is False
    assert "no live supervisor" in result.message
    # An armed watchdog is the relaunch trap — it must be called out.
    assert result.data["watchdog_armed"] is True
    assert "disable the task" in result.message
    # The request is also dual-recorded in the fleet-level events.db for audit.
    rows = query_events(supervisor_heartbeat_path(str(tmp_path)), kind="fleet_stop_requested")
    assert len(rows) == 1
    assert rows[0]["payload"]["drain"] is True


def test_run_fleet_stop_dry_run_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run previews the write without creating the marker."""
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.probe_fleet_watchdog",
        lambda: WatchdogProbe(armed=None, detail="?"),
    )

    result = cli.run_fleet_stop(_stop_args(tmp_path, dry_run=True))

    assert result.ok is True
    assert result.data["dry_run"] is True
    assert read_fleet_stop_request(str(tmp_path)) is None


def test_run_fleet_stop_reports_a_replaced_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second ``fleet stop`` replaces the pending request — and says so."""
    write_fleet_stop_request(str(tmp_path), drain=True)
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.probe_fleet_watchdog",
        lambda: WatchdogProbe(armed=False, detail="Disabled"),
    )

    result = cli.run_fleet_stop(_stop_args(tmp_path, drain=False))

    assert result.ok is True
    assert result.data["replaced_prior_request"] is True
    request = read_fleet_stop_request(str(tmp_path))
    assert request is not None and request["drain"] is False
    # A disabled watchdog means the fleet actually stays down.
    assert result.data["watchdog_armed"] is False
    assert "stays down" in result.message


def test_run_fleet_stop_detects_a_live_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live heartbeat (pid alive, no exited_at) reports supervisor_live."""
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.probe_fleet_watchdog",
        lambda: WatchdogProbe(armed=True, detail="Enabled"),
    )
    monkeypatch.setattr(
        fleet_stop,
        "read_supervisor_heartbeat",
        lambda _o: {"exited_at": None, "pid": 12345},
    )
    monkeypatch.setattr(fleet_stop, "is_pid_alive", lambda pid: pid == 12345)

    result = cli.run_fleet_stop(_stop_args(tmp_path))

    assert result.ok is True
    assert result.data["supervisor_live"] is True
    assert "no live supervisor" not in result.message


def test_fleet_stop_through_cli_main_uses_the_real_probe_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: ``charlie fleet stop`` end-to-end must not NameError.

    The extracted ``run_fleet_stop`` called a bare ``probe_fleet_watchdog()``
    that a PEP 562 module ``__getattr__`` could not serve — module
    ``__getattr__`` covers attribute access, never in-module ``LOAD_GLOBAL``
    lookups — so the real command crashed after writing the marker while
    every test masked it by injecting the name into ``fleet_stop``. This
    drives the command through ``cli.main`` and fakes only the lowest layer:
    the probe on ``fleet_dispatch`` itself.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.probe_fleet_watchdog",
        lambda: WatchdogProbe(armed=True, detail="Enabled"),
    )

    rc = cli.main(["--fleet-dir", str(tmp_path), "fleet", "stop"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "stop request recorded" in out
    # Armed watchdog -> the relaunch-trap hint must reach the operator.
    assert "disable the task" in out
    assert read_fleet_stop_request(str(tmp_path)) is not None


def test_fleet_stop_global_loads_all_resolve() -> None:
    """Structural guard: no fleet_stop function loads an unbound global.

    Same hole as above, checked mechanically: every ``LOAD_GLOBAL`` name in
    every code object reachable from the module (functions, methods, nested
    definitions) must resolve in the module's own ``__dict__`` or builtins.
    ``vars()`` — not ``hasattr()`` — so a future module ``__getattr__``
    cannot fake a binding the bytecode would never find.
    """
    import builtins
    import dis
    import types

    def _global_names(code: types.CodeType) -> set[str]:
        names = {
            instr.argval for instr in dis.get_instructions(code) if instr.opname == "LOAD_GLOBAL"
        }
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                names |= _global_names(const)
        return names

    def _code_objects(obj: Any) -> list[types.CodeType]:
        # Only objects DEFINED in fleet_stop — an imported Path/CommandResult
        # carries globals from its own module, which are not our problem.
        if isinstance(obj, types.FunctionType):
            return [obj.__code__] if obj.__module__ == fleet_stop.__name__ else []
        if isinstance(obj, (staticmethod, classmethod)):
            return _code_objects(obj.__func__)
        if isinstance(obj, property):
            codes = []
            for accessor in (obj.fget, obj.fset, obj.fdel):
                if accessor is not None:
                    codes.extend(_code_objects(accessor))
            return codes
        if isinstance(obj, type) and obj.__module__ == fleet_stop.__name__:
            codes = []
            for member in vars(obj).values():
                codes.extend(_code_objects(member))
            return codes
        return []

    unresolved: set[str] = set()
    module_ns = vars(fleet_stop)
    for obj in list(module_ns.values()):
        for code in _code_objects(obj):
            for name in _global_names(code):
                if name not in module_ns and not hasattr(builtins, name):
                    unresolved.add(name)

    assert not unresolved, f"unbound globals in fleet_stop: {sorted(unresolved)}"


# --- supervisor honoring ----------------------------------------------------


def _fleet_snap(live_count: int) -> FleetLocalSnapshot:
    """A fleet snapshot with one repo reporting ``live_count`` live workers."""
    return FleetLocalSnapshot(
        frozenset(
            {
                (
                    "owner/repo",
                    LocalSnapshot(
                        live_count=live_count,
                        sidecar_mtimes=frozenset(),
                        verdict_mtimes=frozenset(),
                    ),
                )
            }
        )
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_stop_marker_exits_before_any_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """#1716 AC1: a pending plain stop request exits at the first boundary.

    No pass runs, the marker is consumed (so the next scheduled tick starts a
    clean supervisor), and ``supervisor_exited`` records the named reason —
    the lifecycle evidence the acceptance criteria ask for.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )
    write_fleet_stop_request(str(tmp_path), drain=False)

    fc = _FakeClock(auto_advance=1.0)
    # max_passes=1 as a hang guard (round-4 review): pass_number is 0 going
    # into the very first tick regardless of the cap, so this changes nothing
    # about the intended exit path -- it only stops the loop from running
    # forever if a regression made the marker check never fire.
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=1
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop"
    assert result.data["restart_requested"] is False
    assert result.data["passes"] == 0
    mock_fleet_loop.assert_not_called()
    assert read_fleet_stop_request(str(tmp_path)) is None
    mocks = _patch_self_deploy_for_fleet_tests
    mocks["record_supervisor_exit"].assert_called_once()
    assert mocks["record_supervisor_exit"].call_args.kwargs["reason"] == "operator_stop"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_stop_marker_written_mid_run_exits_after_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """#1716: the marker is checked between passes, not only at startup.

    A ``fleet stop`` landing while a pass runs lets that pass finish (a
    mid-pass abort would be a kill by another name), then the next boundary
    honors it — and the pass that already ran dispatched normally.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )

    def _loop_writes_stop_marker(**_kwargs: Any) -> Any:
        write_fleet_stop_request(str(tmp_path), drain=False)
        return _drained_fleet_result()

    mock_fleet_loop.side_effect = _loop_writes_stop_marker

    fc = _FakeClock(auto_advance=1.0)
    # max_passes=2 as a hang guard (round-4 review): the marker is only
    # checked at the *second* tick's boundary here (the first tick's fallback
    # pass is what writes it), so a cap of 1 would cut the loop off with
    # exit_reason="max_passes" before the marker was ever read -- changing
    # the very behavior under test. 2 is the smallest cap that leaves the
    # intended operator_stop exit path reachable while still bounding a
    # regression that stops the marker check from firing at all.
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=2
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop"
    assert result.data["passes"] == 1
    assert mock_fleet_loop.call_count == 1
    # The one pass that ran was a normal pass — the stop had not landed yet.
    assert mock_fleet_loop.call_args.kwargs["drain"] is False
    assert read_fleet_stop_request(str(tmp_path)) is None


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_drain_suppresses_dispatch_and_exits_at_zero(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """#1716 AC2: drain dispatches nothing new and exits when workers reach 0.

    Scripted live-worker counts: 2 → 2 → 1 (post-pass) → 0 → 0. The pass
    triggered by the last worker's death runs first (so its outcome is
    adopted), then the supervisor exits with the drained reason. The drain
    pass must not self-deploy either — a head-moved restart exit under a
    pending marker would split the drain across watchdog ticks.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()
    write_fleet_stop_request(str(tmp_path), drain=True)

    deploy_mock = MagicMock(name="self_deploy")
    monkeypatch.setattr("charlie_work.fleet_dispatch.self_deploy", deploy_mock)
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(
            side_effect=[
                _fleet_snap(2),  # pre-loop baseline
                _fleet_snap(2),  # pass-1 delta check (fallback fires anyway)
                _fleet_snap(1),  # post-pass: one worker finished
                _fleet_snap(0),  # pass-2 delta check: last worker gone
                _fleet_snap(0),  # post-pass: still zero
            ]
        ),
    )

    fc = _FakeClock(auto_advance=1.0)
    # max_passes=2 as a hang guard (round-4 review): this scenario needs
    # exactly the 2 passes asserted below (drain_complete_if_empty's post-pass
    # check is what ends it, on the second pass, before a third tick's top
    # would ever run) -- a cap of 1 would cut it off with exit_reason=
    # "max_passes" after only one pass, before the worker count ever reaches
    # 0. 2 is the smallest cap that leaves the intended drained exit reachable
    # while still bounding a regression that stops the drain-exit check from
    # firing at all.
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=2
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop_drained"
    assert result.data["restart_requested"] is False
    # Two passes ran — one while a worker was still live, one finalizing the
    # last worker's death — both with dispatch suppressed.
    assert mock_fleet_loop.call_count == 2
    for call in mock_fleet_loop.call_args_list:
        assert call.kwargs["drain"] is True
    deploy_mock.assert_not_called()
    assert read_fleet_stop_request(str(tmp_path)) is None
    mocks = _patch_self_deploy_for_fleet_tests
    assert mocks["record_supervisor_exit"].call_args.kwargs["reason"] == "operator_stop_drained"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_drain_exits_pre_pass_when_already_at_zero(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """Round-4 review, required change 1: ``FleetStopState.drain_tick``'s
    pre-pass ``return True`` branch, reached when a drain is already latched
    and the live-worker count is already 0 with no pass due -- as opposed to
    the post-pass ``drain_complete_if_empty`` variant the existing
    ``..._exits_at_zero`` test above exercises.

    The very first supervisor tick always treats a full pass as due (``
    last_full_pass_at = start_time - full_pass_interval`` makes ``now -
    last_full_pass_at == full_pass_interval`` exactly on tick 1, regardless
    of the configured interval -- the existing drain test's own "(fallback
    fires anyway)" comment is this same fact), so a drain marker present
    before ``run_fleet_supervise`` is even called cannot reach the pre-pass
    branch on tick 1: ``drain_tick``'s ``pass_due`` guard would always be
    True there. This drives the marker in mid-run instead (mirroring the
    existing mid-run stop-marker pattern), lets the unavoidable first-tick
    pass land with the fleet still non-empty, and only then has the worker
    count reach 0 with nothing else changed -- so the *second* tick's
    ``drain_tick`` call sees ``pass_due=False`` and must exit right there,
    calling ``fleet_loop`` no further times beyond the one pass that had
    already run before the drain was even requested.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
        )
    )

    call_count = {"n": 0}

    def _first_pass_requests_drain(**_kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            write_fleet_stop_request(str(tmp_path), drain=True)
        return _drained_fleet_result()

    mock_fleet_loop.side_effect = _first_pass_requests_drain

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(
            side_effect=[
                _fleet_snap(1),  # pre-loop baseline
                _fleet_snap(1),  # tick-1 pre-pass delta check (fallback fires anyway)
                _fleet_snap(0),  # tick-1 post-pass: becomes tick-2's delta baseline
                _fleet_snap(0),  # tick-2 pre-pass: identical -> no delta, no fallback due
            ]
        ),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=3
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop_drained"
    assert result.data["restart_requested"] is False
    # Only the one, unavoidable first-tick pass ran; the drain's own exit,
    # once latched, never reached fleet_loop again.
    assert mock_fleet_loop.call_count == 1
    assert mock_fleet_loop.call_args.kwargs["drain"] is False
    assert read_fleet_stop_request(str(tmp_path)) is None
    mocks = _patch_self_deploy_for_fleet_tests
    assert mocks["record_supervisor_exit"].call_args.kwargs["reason"] == "operator_stop_drained"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_suppresses_head_drift_while_draining(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """Round-4 review, required change 2: ``current_head = startup_head if
    drain_state.draining else read_head_sha(...)`` must not let an external
    HEAD move exit the supervisor with ``head_drift`` while a drain is
    latched.

    ``read_head_sha`` is patched to return the *same* SHA for the two calls a
    correct run makes (the one-time ``startup_head`` capture, plus tick 1's
    own drift check while not yet draining) and a *different* SHA for any
    call beyond that -- which only happens if the ``startup_head if draining
    else ...`` guard regresses and calls ``read_head_sha`` again on a tick
    where a drain is already latched. A no-drain control that this same
    check *does* fire head_drift already exists at
    ``test_run_fleet_supervise_restarts_on_external_head_drift``
    (tests/test_fleet_dispatch_supervise_self_deploy.py).
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )

    call_count = {"n": 0}

    def _first_pass_requests_drain(**_kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            write_fleet_stop_request(str(tmp_path), drain=True)
        return _drained_fleet_result()

    mock_fleet_loop.side_effect = _first_pass_requests_drain

    read_head_mock = MagicMock(side_effect=["same-sha", "same-sha", "drifted-sha", "drifted-sha"])
    monkeypatch.setattr("charlie_work.fleet_dispatch.read_head_sha", read_head_mock)

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(
            side_effect=[
                _fleet_snap(1),  # pre-loop baseline
                _fleet_snap(1),  # tick-1 pre-pass delta check (fallback fires anyway)
                _fleet_snap(1),  # tick-1 post-pass: still live, drain not complete
                _fleet_snap(1),  # tick-2 pre-pass: nonzero -> drain_tick does not exit early
                _fleet_snap(0),  # tick-2 post-pass: last worker gone -> drain completes
            ]
        ),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=3
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop_drained"
    assert result.data["restart_requested"] is False
    # startup_head + tick-1's own (not-yet-draining) drift check -- exactly 2
    # real read_head_sha calls. A 3rd call would mean the drain guard stopped
    # suppressing the check on tick 2, which is exactly the regression this
    # test exists to catch (the 3rd/4th queued value is a different SHA that
    # would otherwise trip head_drift).
    assert read_head_mock.call_count == 2
    assert mock_fleet_loop.call_count == 2
    assert mock_fleet_loop.call_args_list[0].kwargs["drain"] is False
    assert mock_fleet_loop.call_args_list[1].kwargs["drain"] is True
    assert read_fleet_stop_request(str(tmp_path)) is None
    mocks = _patch_self_deploy_for_fleet_tests
    assert mocks["record_supervisor_exit"].call_args.kwargs["reason"] == "operator_stop_drained"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_plain_stop_overrides_a_latched_drain(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
    _patch_self_deploy_for_fleet_tests: dict[str, MagicMock],
) -> None:
    """Round-4 review, required change 3: a plain ``fleet stop`` marker
    landing while a drain is already latched must win at the
    ``run_fleet_supervise`` level -- exit with ``operator_stop`` (not
    ``operator_stop_drained``) and consume the marker -- the same "last
    write wins" semantics ``test_last_write_wins_on_repeat_requests`` proves
    at the marker-file primitive level, but never yet driven through the
    actual supervisor loop with a drain already ``latched`` (``FleetStopState
    .poll_stop_marker``'s ``request.get("drain")`` check ignores
    ``self._draining`` entirely, by design -- this is what proves it).
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )
    write_fleet_stop_request(str(tmp_path), drain=True)

    def _pass_overrides_with_plain_stop(**_kwargs: Any) -> Any:
        write_fleet_stop_request(str(tmp_path), drain=False)
        return _drained_fleet_result()

    mock_fleet_loop.side_effect = _pass_overrides_with_plain_stop

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(
            side_effect=[
                _fleet_snap(2),  # pre-loop baseline
                _fleet_snap(2),  # tick-1 pre-pass delta check (fallback fires anyway;
                # drain_tick's pass_due=True short-circuits it regardless of this count)
                _fleet_snap(1),  # tick-1 post-pass: still nonzero, drain not complete
                # -- if it hit 0 here, drain_complete_if_empty would exit the loop
                # before tick 2 ever got a chance to see the plain-stop override.
            ]
        ),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path), clock=fc.now, sleep=fc.sleep, max_passes=3
    )

    assert result.ok is True
    assert result.data["exit_reason"] == "operator_stop"
    assert result.data["restart_requested"] is False
    # Only tick 1's pass ran (while draining was already latched); tick 2
    # exits via the plain-stop marker before ever reaching fleet_loop again.
    assert mock_fleet_loop.call_count == 1
    assert mock_fleet_loop.call_args.kwargs["drain"] is True
    assert read_fleet_stop_request(str(tmp_path)) is None
    mocks = _patch_self_deploy_for_fleet_tests
    assert mocks["record_supervisor_exit"].call_args.kwargs["reason"] == "operator_stop"


def test_run_fleet_supervise_loop_honors_a_pending_stop_marker(tmp_path: Path) -> None:
    """#1716: a stop marker present at a restart request suppresses relaunch.

    The race this covers: the child self-deploys (exit 3) before it ever saw
    the marker the operator just wrote. Relaunching would resurrect a
    supervisor bound to exit on the same marker — the wrapper declines.
    """
    write_fleet_stop_request(str(tmp_path), drain=False)
    spawns: list[int] = []

    def _spawn(launch_number: int) -> int:
        spawns.append(launch_number)
        return EXIT_RESTART_REQUESTED

    result = run_fleet_supervise_loop(
        spawn=_spawn,
        max_relaunches=3,
        on_cap_reached=lambda _r: None,
        fleet_dir_override=str(tmp_path),
    )

    assert result.ok is True
    assert result.data["stop_requested"] is True
    assert result.data["launches"] == 1
    assert spawns == [1]
    # The wrapper is a checker, not the consumer: the marker stays for the
    # next supervise start to honor.
    assert read_fleet_stop_request(str(tmp_path)) is not None


def test_run_fleet_supervise_loop_keyboard_interrupt_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1716: Ctrl+C on the wrapper is one clean line, not a traceback.

    A foreground ``fleet supervise-loop`` can still be interrupted even
    though the scheduled deployment has no console. The interrupt must stop
    the wrapper without relaunching — the detached child is the operator's
    next problem, not a restart-loop trigger.
    """

    def _interrupt(_launch_number: int) -> int:
        raise KeyboardInterrupt

    result = run_fleet_supervise_loop(
        spawn=_interrupt,
        max_relaunches=3,
        fleet_dir_override=str(tmp_path),
    )

    assert result.ok is True
    assert result.data["interrupted"] is True
    assert "interrupted" in result.message
    out = capsys.readouterr().out
    assert "supervise-loop: interrupted; not relaunching" in out
    assert "Traceback" not in out


# --- drain dispatch suppression inside a pass --------------------------------


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_drain_suppresses_all_new_dispatch(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1716: fleet_loop(drain=True) launches nothing new.

    The pass still runs its other lanes (reap/review/merge live inside
    ``app.loop``), but the per-repo dispatch budget is forced to 0 and
    ``review_dispatch.enabled`` is flipped off on the pass-local config copy —
    so a repo whose real config enables review dispatch still launches no
    reviewer while the fleet drains.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()

    # Enabled deliberately: drain must flip it, and a default-False config
    # could not prove that.
    mock_load_layered_config.return_value = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
        drain=True,
    )

    # limit 0 reaches the per-repo loop — dispatch_rework/dispatch slice
    # candidates[:0], so nothing launches no matter how many are ready.
    mock_app.loop.assert_called_once_with(0, merge=True)
    # The config handed to OrchestratorApp is the pass-local copy with
    # review dispatch disabled — dispatch_reviews() early-returns after its
    # reaper sweeps, so no reviewer is launched either.
    config_seen = mock_app_class.call_args.args[2]
    assert config_seen.review_dispatch.enabled is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_no_drain_keeps_limit_and_review_dispatch(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Control for the drain test: drain=False (default) changes nothing."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    mock_app.loop.assert_called_once_with(3, merge=True)
    config_seen = mock_app_class.call_args.args[2]
    assert config_seen.review_dispatch.enabled is True
