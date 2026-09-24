"""Supervisor-level ``fleet stop --drain`` edge cases (issue #1716, round-4 review).

Split out of ``test_fleet_stop.py`` to keep that file under the file-size cap.
Covers the drain paths ``test_fleet_stop.py``'s supervisor tests don't: the
pre-pass drained exit, head-drift suppression while draining, and a plain
stop overriding a latched drain.
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
from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import FleetLocalSnapshot, run_fleet_supervise
from charlie_work.fleet_stop import read_fleet_stop_request, write_fleet_stop_request
from charlie_work.supervise import LocalSnapshot


# Duplicated from test_fleet_stop.py (same convention as
# test_capacity_starvation_escalation.py) rather than importing across test modules.
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
