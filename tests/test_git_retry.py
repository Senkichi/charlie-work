"""Behavioral tests for ``git_retry.run_git_with_retry``.

Every test injects ``sleep``/``random_uniform`` no-ops so the suite never
actually sleeps for backoff -- the delay *values* passed to those injected
callables are asserted directly instead, which is a stronger check than
timing the real call would be and keeps the suite fast.
"""

from __future__ import annotations

from pathlib import Path

import charlie_work.subprocess_runner as subprocess_runner_module
from charlie_work.git_retry import (
    DEFAULT_MAX_RETRIES,
    RetryOutcome,
    run_git_with_retry,
)
from charlie_work.subprocess_runner import RunResult

# A real curl/schannel shape this host's own `git fetch` produces against an
# unroutable remote (see transient_errors.py's module docstring and its own
# test suite) -- not the hand-assembled `connectex` hybrid the old fixture
# used, which no tool actually emits and was an inert control for whether the
# classifier matches real git output (issue #1777 finding 2).
_TRANSIENT_STDERR = (
    "fatal: unable to access 'https://github.com/x/y.git/': Failed to connect to "
    "github.com port 443 after 2093 ms: Couldn't connect to server"
)
_TERMINAL_STDERR = "fatal: Not possible to fast-forward, aborting."


def _ok_result() -> RunResult:
    return RunResult(returncode=0, stdout="", stderr="")


def _transient_result() -> RunResult:
    return RunResult(returncode=128, stdout="", stderr=_TRANSIENT_STDERR)


def _terminal_result() -> RunResult:
    return RunResult(returncode=1, stdout="", stderr=_TERMINAL_STDERR)


class _QueueRunner:
    """Injectable ``run_command`` that pops canned results in call order."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, cwd: Path | str, timeout_seconds: int) -> RunResult:
        self.calls.append(command)
        return self._results.pop(0)


def _no_sleep(_seconds: float) -> None:
    return None


def _no_jitter(low: float, high: float) -> float:
    return 0.0


def test_transient_then_success_retries_and_returns_ok(tmp_path: Path) -> None:
    runner = _QueueRunner([_transient_result(), _ok_result()])
    on_retry_calls: list[RetryOutcome] = []

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        on_retry=on_retry_calls.append,
    )

    assert result.ok is True
    assert len(runner.calls) == 2
    # Exactly one event for the retried call, never one per attempt.
    assert on_retry_calls == [RetryOutcome(attempts=2, ok=True, error=None)]


def test_transient_exhausted_returns_error_value_never_raises(tmp_path: Path) -> None:
    # max_retries additional attempts on top of the first => max_retries + 1
    # total calls, all transient, so the loop runs out of retries rather than
    # succeeding.
    results = [_transient_result() for _ in range(DEFAULT_MAX_RETRIES + 1)]
    runner = _QueueRunner(results)
    on_retry_calls: list[RetryOutcome] = []

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        on_retry=on_retry_calls.append,
    )

    assert result.ok is False
    assert result is results[-1]
    assert len(runner.calls) == DEFAULT_MAX_RETRIES + 1
    assert on_retry_calls == [
        RetryOutcome(attempts=DEFAULT_MAX_RETRIES + 1, ok=False, error=result.error)
    ]


def test_non_transient_error_is_not_retried(tmp_path: Path) -> None:
    runner = _QueueRunner([_terminal_result()])
    on_retry_calls: list[RetryOutcome] = []

    result = run_git_with_retry(
        ["git", "pull", "--ff-only", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        on_retry=on_retry_calls.append,
    )

    assert result.ok is False
    assert len(runner.calls) == 1
    # on_retry never fires for a call that failed terminally on attempt 1.
    assert on_retry_calls == []


def test_on_retry_not_called_when_first_attempt_succeeds(tmp_path: Path) -> None:
    runner = _QueueRunner([_ok_result()])
    on_retry_calls: list[RetryOutcome] = []

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        on_retry=on_retry_calls.append,
    )

    assert result.ok is True
    assert len(runner.calls) == 1
    assert on_retry_calls == []


def test_stderr_is_classified_over_generic_error_field(tmp_path: Path) -> None:
    # run_captured always sets .error to a generic "command exited N" string
    # on any non-zero exit, which would shadow the real stderr text if a
    # classifier read .error first. Confirms the retry loop reads .stderr
    # first (issue #817 item 3's fix, mirrored here).
    shadowed = RunResult(
        returncode=128,
        stdout="",
        stderr=_TRANSIENT_STDERR,
        error="command exited 128",
    )
    runner = _QueueRunner([shadowed, _ok_result()])

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
    )

    assert result.ok is True
    assert len(runner.calls) == 2


def test_max_elapsed_seconds_budget_stops_retrying(monkeypatch, tmp_path: Path) -> None:
    # A monotonic clock stub that reports the budget as already exhausted
    # after the first attempt, regardless of max_retries -- proves the wall-
    # clock bound is independent of (and can bind tighter than) the attempt
    # count. monkeypatch restores the real time.monotonic on teardown, so
    # this cannot leak into other tests.
    results = [_transient_result(), _transient_result(), _ok_result()]
    runner = _QueueRunner(results)

    clock = iter([0.0, 100.0, 100.0, 100.0, 100.0])

    import charlie_work.git_retry as git_retry_module

    def fake_monotonic() -> float:
        try:
            return next(clock)
        except StopIteration:
            return 100.0

    monkeypatch.setattr(git_retry_module.time, "monotonic", fake_monotonic)

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        max_elapsed_seconds=20.0,
    )

    # Only the first attempt ran: by the time the loop re-checks elapsed time
    # it already reads 100s (> the 20s budget), so it stops instead of
    # consuming any of the remaining queued responses.
    assert result.ok is False
    assert len(runner.calls) == 1


def test_backoff_delay_and_jitter_bounds_passed_to_injected_callables(tmp_path: Path) -> None:
    results = [_transient_result(), _transient_result(), _ok_result()]
    runner = _QueueRunner(results)
    delays_seen: list[tuple[float, float]] = []

    def recording_uniform(low: float, high: float) -> float:
        delays_seen.append((low, high))
        return 0.0

    run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=recording_uniform,
        base_delay_seconds=1.0,
    )

    # attempt 1 backoff: base_delay * 2**0 = 1.0, +/-25% jitter => (-0.25, 0.25)
    # attempt 2 backoff: base_delay * 2**1 = 2.0, +/-25% jitter => (-0.5, 0.5)
    assert delays_seen == [(-0.25, 0.25), (-0.5, 0.5)]


def test_max_elapsed_seconds_accounts_for_the_imminent_sleep(tmp_path: Path) -> None:
    """Issue #1777 finding 7: the budget check must include the sleep about to
    happen, not just elapsed time so far -- otherwise the loop can commit to a
    sleep that itself blows the budget.

    With ``max_elapsed_seconds=0.5`` and a real (near-zero) clock, elapsed
    time alone is well under budget, but the computed backoff
    (``base_delay_seconds * 2**0 == 1.0``, no jitter) would push elapsed +
    sleep to ~1.0s, over the 0.5s budget. If the check only looked at elapsed
    time before committing to the sleep (the pre-fix behavior), this test
    fails: the loop would sleep, consume the second queued (successful)
    response, and return ok=True with 2 calls recorded.
    """
    runner = _QueueRunner([_transient_result(), _ok_result()])

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        sleep=_no_sleep,
        random_uniform=_no_jitter,
        base_delay_seconds=1.0,
        max_elapsed_seconds=0.5,
    )

    assert result.ok is False
    assert len(runner.calls) == 1


def test_run_command_none_resolves_subprocess_runner_module_attribute_dynamically(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1777 finding 9: omitting ``run_command`` must resolve
    ``subprocess_runner.run_captured`` fresh on each call (a live module-
    attribute lookup), not a reference captured once at import time -- so a
    test or future caller that monkeypatches
    ``charlie_work.subprocess_runner.run_captured`` (the usual seam in this
    repo) actually takes effect. Before the fix, the default was bound at
    def time (``run_command: Callable = run_captured``), so this monkeypatch
    would have no effect and the call would hit the real, unpatched
    ``run_captured`` -- a silent no-op seam.
    """
    calls: list[list[str]] = []

    def fake_run_captured(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        calls.append(command)
        return _ok_result()

    monkeypatch.setattr(subprocess_runner_module, "run_captured", fake_run_captured)

    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
    )

    assert result.ok is True
    assert calls == [["git", "fetch", "origin", "main"]]


def test_sleep_none_resolves_time_module_attribute_dynamically(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1777 finding 8/9 (same seam, different callable): omitting
    ``sleep`` must resolve ``time.sleep`` fresh on each call, so
    monkeypatching ``charlie_work.git_retry.time.sleep`` (the same pattern
    ``test_max_elapsed_seconds_budget_stops_retrying`` already uses for
    ``time.monotonic``) actually takes effect. Before the fix, the default
    was bound at def time (``sleep: Callable = time.sleep``); this
    monkeypatch would have had no effect and the real ``time.sleep`` would
    have run instead.
    """
    import charlie_work.git_retry as git_retry_module

    sleep_calls: list[float] = []
    monkeypatch.setattr(git_retry_module.time, "sleep", sleep_calls.append)

    runner = _QueueRunner([_transient_result(), _ok_result()])
    result = run_git_with_retry(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        timeout_seconds=30,
        run_command=runner,
        random_uniform=_no_jitter,
    )

    assert result.ok is True
    assert sleep_calls == [1.0]
