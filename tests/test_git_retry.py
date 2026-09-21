"""Behavioral tests for ``git_retry.run_git_with_retry``.

Every test injects ``sleep``/``random_uniform`` no-ops so the suite never
actually sleeps for backoff -- the delay *values* passed to those injected
callables are asserted directly instead, which is a stronger check than
timing the real call would be and keeps the suite fast.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.git_retry import (
    DEFAULT_MAX_RETRIES,
    RetryOutcome,
    run_git_with_retry,
)
from charlie_work.subprocess_runner import RunResult

_TRANSIENT_STDERR = "fatal: unable to access 'https://github.com/x/y.git/': connectex"
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
