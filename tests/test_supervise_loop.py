"""Tests for the bounded supervisor relaunch wrapper (#862)."""

from __future__ import annotations

import pytest

from charlie_work.supervise_loop import (
    EXIT_RESTART_REQUESTED,
    SuperviseLoopResult,
    run_supervise_relaunch_loop,
)


class _ScriptedSpawn:
    """Returns a fixed sequence of exit codes, recording the launch numbers."""

    def __init__(self, exit_codes: list[int]) -> None:
        self._exit_codes = list(exit_codes)
        self.launch_numbers: list[int] = []

    def __call__(self, launch_number: int) -> int:
        self.launch_numbers.append(launch_number)
        if not self._exit_codes:
            raise AssertionError(
                f"spawn called {len(self.launch_numbers)} times but only "
                f"{len(self.launch_numbers) - 1} exit codes were scripted"
            )
        return self._exit_codes.pop(0)


class _AlwaysRestarting:
    """A supervisor that never stops asking to be restarted.

    The ceiling is what makes this test *fail* rather than *hang* against an
    unbounded implementation. A hang is an ambiguous result -- it looks the same
    as a slow suite -- so the fake raises instead, well above any legitimate cap
    under test.
    """

    def __init__(self, ceiling: int = 50) -> None:
        self.ceiling = ceiling
        self.calls = 0

    def __call__(self, launch_number: int) -> int:
        self.calls += 1
        if self.calls > self.ceiling:
            raise AssertionError(
                f"spawn was called {self.calls} times without the loop stopping; "
                "the relaunch bound is not being enforced"
            )
        return EXIT_RESTART_REQUESTED


def test_normal_exit_does_not_relaunch() -> None:
    """AC4: a deliberate stop (max-runtime / max-passes) must not relaunch."""
    spawn = _ScriptedSpawn([0])

    result = run_supervise_relaunch_loop(spawn, max_relaunches=5, log=lambda _: None)

    assert result == SuperviseLoopResult(
        launches=1, relaunches=0, last_exit_code=0, cap_reached=False
    )
    assert spawn.launch_numbers == [1]


def test_failure_exit_does_not_relaunch() -> None:
    """An aborted supervisor (exit 1) is not a restart request either.

    Complements the exit-0 case: relaunching must key on the specific restart
    code, not merely on "the child stopped".
    """
    spawn = _ScriptedSpawn([1])

    result = run_supervise_relaunch_loop(spawn, max_relaunches=5, log=lambda _: None)

    assert result.launches == 1
    assert result.relaunches == 0
    assert result.last_exit_code == 1
    assert result.cap_reached is False


def test_restart_request_relaunches_until_a_normal_exit() -> None:
    """AC2: each restart request is answered immediately, then the loop settles."""
    spawn = _ScriptedSpawn([EXIT_RESTART_REQUESTED, EXIT_RESTART_REQUESTED, 0])

    result = run_supervise_relaunch_loop(spawn, max_relaunches=5, log=lambda _: None)

    assert result.launches == 3
    assert result.relaunches == 2
    assert result.last_exit_code == 0
    assert result.cap_reached is False
    assert spawn.launch_numbers == [1, 2, 3]


def test_always_restarting_supervisor_terminates_at_the_cap() -> None:
    """AC5: the pathological case must stop, not spin.

    Observed failing against an unbounded implementation (the `if relaunches >=
    max_relaunches` branch deleted): `_AlwaysRestarting` raises at 51 calls.
    """
    spawn = _AlwaysRestarting()
    messages: list[str] = []
    cap_events: list[SuperviseLoopResult] = []

    result = run_supervise_relaunch_loop(
        spawn,
        max_relaunches=3,
        log=messages.append,
        on_cap_reached=cap_events.append,
    )

    # 3 relaunches means 4 launches total: the original plus its replacements.
    assert result.launches == 4
    assert result.relaunches == 3
    assert result.cap_reached is True
    assert result.last_exit_code == EXIT_RESTART_REQUESTED
    assert spawn.calls == 4

    # AC3: a distinct line, and exactly one event -- not one per relaunch.
    assert cap_events == [result]
    cap_lines = [line for line in messages if "cap reached" in line]
    assert len(cap_lines) == 1
    assert "3/3" in cap_lines[0]


def test_cap_of_zero_runs_once_and_refuses_to_relaunch() -> None:
    """The boundary: 0 relaunches still runs the supervisor exactly once."""
    spawn = _AlwaysRestarting()
    cap_events: list[SuperviseLoopResult] = []

    result = run_supervise_relaunch_loop(
        spawn, max_relaunches=0, log=lambda _: None, on_cap_reached=cap_events.append
    )

    assert result.launches == 1
    assert result.relaunches == 0
    assert result.cap_reached is True
    assert len(cap_events) == 1


def test_negative_cap_is_rejected() -> None:
    """A negative bound is a config error, not a silently-unbounded loop."""
    with pytest.raises(ValueError, match="max_relaunches must be >= 0"):
        run_supervise_relaunch_loop(_AlwaysRestarting(), max_relaunches=-1)


def test_cap_event_is_not_emitted_on_a_normal_exit() -> None:
    """Positive control for the cap event.

    Without this, an implementation that fired ``on_cap_reached`` unconditionally
    would still pass the cap test above.
    """
    cap_events: list[SuperviseLoopResult] = []

    run_supervise_relaunch_loop(
        _ScriptedSpawn([EXIT_RESTART_REQUESTED, 0]),
        max_relaunches=5,
        log=lambda _: None,
        on_cap_reached=cap_events.append,
    )

    assert cap_events == []


def test_restart_exit_code_is_distinct_from_the_ordinary_ones() -> None:
    """AC1 at the vocabulary level: 0 (stopped) and 1 (aborted) are both taken."""
    assert EXIT_RESTART_REQUESTED not in (0, 1)
