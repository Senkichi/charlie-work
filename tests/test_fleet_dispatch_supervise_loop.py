"""Loop mechanics for ``run_fleet_supervise``: cadence, locking, provenance, exit edges.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from _fleet_dispatch_fixtures import (
    _FakeClock,
    _active_fleet_result,
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work.config import (
    OrchestratorConfig,
    SupervisorConfig,
)
from charlie_work.fleet_dispatch import (
    run_fleet_supervise,
    run_fleet_supervise_loop,
)
from charlie_work.instrumentation import query_events
from charlie_work.supervise import SelfDeployResult
from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_ensures_labels_on_first_pass_only(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1339: run_fleet_supervise passes ensure_labels=True on the first
    fleet_loop pass only, so the LabelConfig-derived label ensure runs once per
    supervisor startup per repo (not once per pass).
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert mock_fleet_loop.call_count == 3
    ensure_flags = [call.kwargs.get("ensure_labels") for call in mock_fleet_loop.call_args_list]
    # First pass ensures; subsequent passes do not.
    assert ensure_flags == [True, False, False], ensure_flags


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_full_pass_interval_fallback_triggers_pass(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """When no local delta, the full_pass_interval fallback still drives a pass."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=10,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        lambda _before, _after: False,
    )

    fc = _FakeClock(auto_advance=15.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert mock_fleet_loop.call_count == 2
    assert fc.sleep_calls == [5.0, 5.0]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_keyboard_interrupt_returns_ok(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """Ctrl+C is caught and reported as a clean completion."""
    lock = MagicMock()
    mock_lock.return_value = lock
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
        )
    )
    mock_fleet_loop.side_effect = [_drained_fleet_result(), KeyboardInterrupt]

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert "fleet supervisor complete" in result.message
    assert result.data["passes"] >= 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_local_delta_triggers_pass_before_fallback(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A local signal delta triggers the next fleet pass before the fallback interval."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        MagicMock(),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        MagicMock(side_effect=[False, True]),
    )

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert mock_fleet_loop.call_count == 2
    assert fc.sleep_calls == [5.0, 5.0]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_logs_global_config_provenance(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The supervisor must record whether its global config layer was readable.

    This is the half of the #590 diagnostic that has to survive at the default
    log level: the loader's equivalent line is DEBUG, so on a real host this is
    the only place the fact appears. A successfully-loaded config that reports
    ``absent`` here is the silent-{} path in load_layered_config; one that
    reports ``present`` means the section was lost downstream of the read. The
    two demand opposite fixes, so neither reading may be missing.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    def provenance_lines() -> str:
        return "\n".join(
            r.getMessage()
            for r in caplog.records
            if "Fleet supervisor global config" in r.getMessage()
        )

    # No global config on this fleet dir: reported as absent, at INFO.
    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(
            max_passes=1, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(tmp_path)
        )
    absent = provenance_lines()
    assert absent, "the supervisor logged no global-config provenance at all"
    assert str(tmp_path / "config.yaml") in absent, "the path must be named"
    assert "absent" in absent, f"an absent global layer was not reported: {absent!r}"

    # Same call with the layer in place: distinguishable, with its size.
    caplog.clear()
    (tmp_path / "config.yaml").write_text("dispatch: {}\n", encoding="utf-8")
    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.INFO, logger="charlie_work.fleet_dispatch"):
        run_fleet_supervise(
            max_passes=1, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(tmp_path)
        )
    present = provenance_lines()
    assert "present" in present, f"a present global layer was not reported: {present!r}"
    assert "absent" not in present, "a present layer must not read as absent"
    assert "bytes=" in present, "the size distinguishes an empty layer from a populated one"


def test_run_fleet_supervise_loop_distinguishes_a_cap_from_an_abort() -> None:
    """The paired control for the test above -- ok=True must not mask a crash.

    Both conditions stop the wrapper, and the whole argument for ok=True on cap
    is that a crash keeps exit 1 to itself. If that ever stopped being true the
    cap's ok=True would be hiding real failures rather than disambiguating them.
    """
    capped_events: list[object] = []
    aborted_events: list[object] = []
    capped = run_fleet_supervise_loop(
        spawn=lambda _n: EXIT_RESTART_REQUESTED,
        max_relaunches=1,
        on_cap_reached=capped_events.append,
    )
    aborted = run_fleet_supervise_loop(
        spawn=lambda _n: 1, max_relaunches=1, on_cap_reached=aborted_events.append
    )

    assert (capped.ok, capped.data["cap_reached"]) == (True, True)
    assert (aborted.ok, aborted.data["cap_reached"]) == (False, False)
    assert (len(capped_events), len(aborted_events)) == (1, 0)


def test_run_fleet_supervise_loop_does_not_touch_the_real_state_file() -> None:
    """The cap callback must be injectable, not resolved from the live repo.

    Its default writes through ``orchestrator_root()`` to the real ``events.db``
    and ``state.json`` -- the ones the running supervisor owns. A cap test using
    defaults therefore injects fake events into production and contends for the
    live state lock. Asserting the parameter exists is what keeps the next cap
    test from quietly reaching production again.
    """
    import inspect

    signature = inspect.signature(run_fleet_supervise_loop)
    assert "on_cap_reached" in signature.parameters


def test_run_fleet_supervise_loop_propagates_a_child_failure() -> None:
    """An aborted supervisor stays non-ok rather than being masked by the wrapper."""
    result = run_fleet_supervise_loop(
        spawn=lambda _n: 1, max_relaunches=3, on_cap_reached=lambda _r: None
    )

    assert result.ok is False
    assert result.data["last_exit_code"] == 1
    assert result.data["cap_reached"] is False


def test_run_fleet_supervise_loop_reports_ok_on_a_clean_child_exit() -> None:
    """The wrapper is transparent when the supervisor stops deliberately."""
    result = run_fleet_supervise_loop(spawn=lambda _n: 0, max_relaunches=3)

    assert result.ok is True
    assert result.data["launches"] == 1
    assert result.data["cap_reached"] is False


def test_run_fleet_supervise_loop_reports_ok_when_the_cap_is_hit() -> None:
    """Hitting the cap is a clean handoff, not a failure.

    Stopping at the bound is the wrapper doing its job: it returns restart
    authority to the 5-minute trigger instead of spinning. Reporting it as
    ok=False would exit 1, which `except Exception` in `run_fleet_supervise`
    already uses -- collapsing "self-deploy is not converging" and "supervisor
    crashed" into one indistinguishable code. That is #862's own defect shape
    one layer up, so the cap is signalled by the event and log instead.
    """
    recorded: list[object] = []
    result = run_fleet_supervise_loop(
        spawn=lambda _n: EXIT_RESTART_REQUESTED,
        max_relaunches=2,
        on_cap_reached=recorded.append,
    )

    assert result.ok is True
    assert result.data["cap_reached"] is True
    assert result.data["launches"] == 3
    assert result.data["relaunches"] == 2
    # Never exit 3 upward: the wrapper is the thing that consumed the restart
    # request, so re-signalling it would ask the launcher to relaunch too.
    assert "restart_requested" not in result.data
    # ok=True is only defensible because the cap still announces itself.
    assert len(recorded) == 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_loops_until_max_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """run_fleet_supervise runs fleet_loop repeatedly until max_passes is reached."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=3, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 3
    assert mock_fleet_loop.call_count == 3
    assert fc.sleep_calls == [5.0, 5.0, 5.0]
    # #862 AC4: exhausting the pass budget is a deliberate stop. It shares
    # ok=True with the restart-requesting exits, so the launcher distinguishes
    # them on this field alone -- a regression here would relaunch forever.
    assert result.data["exit_reason"] == "max_passes"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_loud_on_absent_global_layer(
    mock_lock: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud, not silent, in the supervisor.

    Before #623, ``load_layered_config`` treated an unreachable global fleet
    config as an empty mapping with no diagnostic, so a fleet supervisor whose
    global layer was missing silently ran on pristine dataclass defaults --
    ``runner_allocation`` off, ``notify`` off, ``labels`` back to built-ins --
    while passes kept reporting success. ``run_fleet_supervise`` now loads with
    ``require_global=True``, so the absent case raises ``ConfigError`` and the
    supervisor's existing handler catches it, warns, and prints -- the same
    loud path a malformed config already took.

    This test drives the *real* ``load_layered_config`` (not a mock) against an
    empty fleet dir so the ``require_global=True`` wiring is actually exercised
    end-to-end. The provenance line below the handler still fires because it
    uses ``describe_config_file`` directly, so the operator sees both the
    warning and the cause.
    """
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_dispatch"):
        result = run_fleet_supervise(
            max_passes=1,
            clock=fc.now,
            sleep=fc.sleep,
            fleet_dir_override=str(tmp_path),
        )

    # The supervisor continues (the daemon must not crash on a missing global
    # layer) but it does so loudly: a WARNING was emitted naming the failure.
    warning_lines = [r.getMessage() for r in caplog.records if "could not load" in r.getMessage()]
    assert warning_lines, "an absent global layer must trigger the loud handler, not silence"
    assert "per-repo config only" in warning_lines[0], (
        f"the warning must name the fallback: {warning_lines[0]!r}"
    )
    # The ConfigError raised by require_global carries the path and the
    # describe_config_file cause, and the handler interpolates it into the
    # warning -- so the operator sees *why* the layer was unreadable, not just
    # that it was.
    assert str(tmp_path / "config.yaml") in warning_lines[0], (
        "the expected global config path must appear in the warning"
    )
    assert "absent" in warning_lines[0], (
        f"an absent layer must read as absent in the warning: {warning_lines[0]!r}"
    )

    # The handler also prints, so the failure is visible on stdout, not only in
    # the log.
    captured = capsys.readouterr()
    assert "config load failed" in captured.out, (
        "the absent-global failure must be printed, not only logged"
    )

    # The supervisor fell back to the per-repo config (NOT discarded to None or
    # pristine defaults) and still ran the pass -- the daemon stays alive, but
    # the operator has been told exactly why. Discarding the per-repo config
    # with the global layer would regress the #623 silent-disable failure.
    assert result.ok is True
    assert mock_fleet_loop.call_count == 1
    assert mock_fleet_loop.call_args.kwargs.get("global_config") is not None, (
        "fleet_loop must NOT receive global_config=None when the global layer "
        "is absent -- the per-repo config must survive the fallback, not be "
        "discarded with the global layer (#623 silent-disable regression)"
    )


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_records_ci_fleet_provenance(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #954: the supervisor records ci_fleet's import location + sibling git state.

    The live supervisor imports ci_fleet from an editable working tree, not a
    commit. This event makes that coupling attributable: it stamps
    ``ci_fleet.__file__``, the sibling repo's HEAD/branch/dirty-state into the
    fleet-level events.db at every supervisor start. The event is recorded
    even when ``declared_ci_fleet_root`` abstains (e.g. from a worktree), so
    the ``ci_fleet_file`` field is the one fact always present.
    """
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    fc = _FakeClock(auto_advance=1.0)
    run_fleet_supervise(
        max_passes=1, clock=fc.now, sleep=fc.sleep, fleet_dir_override=str(tmp_path)
    )

    # The event lands in the fleet-level events.db (sibling of the heartbeat).
    heartbeat = tmp_path / "supervisor-heartbeat.json"
    rows = query_events(heartbeat, kind="ci_fleet_provenance")
    assert rows is not None, "no events.db reader -- the event was not recorded"
    assert len(rows) == 1, f"expected exactly one ci_fleet_provenance event, got {len(rows)}"
    payload = rows[0]["payload"]
    # ci_fleet is importable in this venv, so __file__ is always set.
    assert payload["ci_fleet_file"] is not None
    # All fields are present (None is a valid value for the sibling fields
    # when declared_ci_fleet_root abstains from a worktree).
    for key in (
        "sibling_root",
        "sibling_head",
        "sibling_branch",
        "sibling_dirty",
        "error",
    ):
        assert key in payload, f"missing field {key!r} in ci_fleet_provenance payload"


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_releases_lock_after_exception(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """The fleet supervisor lock is released even if fleet_loop raises."""
    lock = MagicMock()
    mock_lock.return_value = lock
    mock_load_config.return_value = OrchestratorConfig()
    mock_fleet_loop.side_effect = RuntimeError("boom")

    result = run_fleet_supervise(max_passes=3)

    assert result.ok is False
    assert "boom" in result.message
    lock.release.assert_called_once()


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_respects_max_runtime(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """max_runtime_minutes stops the loop after the wall-clock cap expires."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
            max_runtime_minutes=1,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _active_fleet_result()

    fc = _FakeClock(auto_advance=70.0)
    result = run_fleet_supervise(clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 1
    assert mock_fleet_loop.call_count == 1
    assert fc.sleep_calls == [7.0]
    # A runtime budget expiring is a deliberate stop, not a request to be
    # replaced -- it shares ok=True with the restarting exits, so this field is
    # the only thing keeping the wrapper from relaunching forever.
    assert result.data["exit_reason"] == "max_runtime"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_returns_false_when_lock_held(
    mock_lock: MagicMock,
) -> None:
    """A second concurrent invocation is rejected by the fleet supervisor lock."""
    mock_lock.return_value = None

    result = run_fleet_supervise()

    assert result.ok is False
    assert "fleet supervisor already running" in result.message


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_throttles_idle_passes(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Without a local delta, fleet_loop is skipped until the full-pass fallback expires."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=100,
            max_runtime_minutes=1,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.return_value = _drained_fleet_result()

    snapshot = MagicMock()
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._take_fleet_snapshot",
        lambda **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._has_fleet_delta",
        lambda _before, _after: False,
    )

    fc = _FakeClock(start=0.0, auto_advance=2.0)
    result = run_fleet_supervise(clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert mock_fleet_loop.call_count == 1  # only the initial fallback pass
    assert all(s == 5.0 for s in fc.sleep_calls)
    assert len(fc.sleep_calls) > 1


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_uses_active_cooldown_after_activity(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """After an active pass, sleep equals active_cooldown_seconds; idle equals poll_interval."""
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = [_active_fleet_result(), _drained_fleet_result()]

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=2, clock=fc.now, sleep=fc.sleep)

    assert result.ok is True
    assert result.data["passes"] == 2
    assert fc.sleep_calls == [7.0, 5.0]


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_a_mid_loop_crash_reports_aborted_and_does_not_relaunch(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A crash with no restarting reason already set stays non-restarting.

    The control for
    ``test_zero_pass_bookkeeping_failure_cannot_cancel_a_self_deploy_restart``:
    that test proves an already-set reason survives the handler, and this one
    proves the handler did not simply start relaunching on every exception.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = RuntimeError("boom")

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.ok is False
    assert result.data["exit_reason"] == "aborted"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_an_operator_interrupt_never_asks_to_be_relaunched(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Ctrl-C means stop, and a wrapper that relaunched would defeat that.

    ``interrupted`` is a named reason rather than an unset default precisely so
    this intent is stated and testable. Nothing verified it when the vocabulary
    was introduced, which left the one exit an operator triggers by hand relying
    on ``None`` happening to fall outside ``RESTART_EXIT_REASONS``.
    """
    cfg = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            active_cooldown_seconds=7,
        )
    )
    mock_load_config.return_value = cfg
    mock_fleet_loop.side_effect = KeyboardInterrupt()

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.data["exit_reason"] == "interrupted"
    assert result.data["restart_requested"] is False


@patch("charlie_work.fleet_dispatch.probe_fleet_watchdog")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_zero_pass_bookkeeping_failure_cannot_cancel_a_self_deploy_restart(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_probe: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A failure in post-loop bookkeeping must not suppress the restart signal.

    ``record_zero_pass_streak`` runs after the loop and does real file I/O
    (mkdir, state_lock, log_event); its own docstring says it can raise. It used
    to sit bare inside the outer ``try``, whose handler rewrote ``exit_reason``
    to ``aborted`` and ``restart_requested`` to False unconditionally. So a
    self-deploy that pulled new code, followed by a counter write failing on a
    locked state file, produced an exit the wrapper read as "do not relaunch" --
    the #862 outage, reachable through a secondary failure that has nothing to
    do with whether new code is on disk.

    The important assertion is ``restart_requested``, not ``ok``: the run really
    did fail, so ok=False is correct. What must survive is the instruction to
    replace this process.
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
    # Issue #604: mock the watchdog probe so this test does not perform a real
    # ``schtasks`` subprocess call. ``armed=None`` (unknown) does not trigger
    # the alert path, keeping the test focused on the bookkeeping-failure
    # invariant it exists to guard.
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

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("state file locked by another process")

    monkeypatch.setattr("charlie_work.fleet_dispatch.record_zero_pass_streak", _boom)

    fc = _FakeClock(auto_advance=1.0)
    result = run_fleet_supervise(max_passes=5, clock=fc.now, sleep=fc.sleep)

    assert result.data["exit_reason"] == "self_deploy"
    assert result.data["restart_requested"] is True
