"""``GitHub.run()``'s per-pass circuit breaker wiring (issue #1833).

The breaker state machine itself (classification, trip, half-open, reset) is
unit-tested in isolation in tests/test_circuit_breaker.py. This file covers
the integration: GitHub.run() gates on the breaker, records outcomes at each
terminal exit, fails fast without spawning gh once open, resets via
GitHub.reset_circuit_breaker() (the per-pass hook), and emits
github_circuit_opened/github_circuit_closed events.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module
from charlie_work.config import GhCircuitBreakerConfig, RuntimeConfig
from charlie_work.github_capabilities.circuit_breaker import CircuitBreakerState
from charlie_work.instrumentation import query_events


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / ".var" / "charlie-work" / "state.json"


def _gh(tmp_path: Path, *, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
    return github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(
            # No internal retries: each gh.run() call maps to exactly one
            # subprocess.run() invocation, so consecutive-failure counting
            # is deterministic and independent of gh_max_retries.
            gh_max_retries=0,
            gh_circuit_breaker=GhCircuitBreakerConfig(
                failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
            ),
        ),
    )


def _transport_failure_run(cmd, *args, **kwargs):
    return subprocess.CompletedProcess(
        args=cmd, returncode=1, stdout="", stderr="connection refused"
    )


def test_breaker_trips_after_threshold_consecutive_transport_failures(
    monkeypatch, tmp_path: Path
) -> None:
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _transport_failure_run(cmd, *args, **kwargs)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = _gh(tmp_path, failure_threshold=3)

    for _ in range(3):
        result = gh.run(["issue", "list"], json_output=True, allow_failure=True)
        assert result.ok is False

    assert call_count == 3
    assert gh._transport._circuit_breaker_state.phase == "open"


def test_open_breaker_fails_fast_without_spawning_gh_and_returns_a_value(
    monkeypatch, tmp_path: Path
) -> None:
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _transport_failure_run(cmd, *args, **kwargs)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = _gh(tmp_path, failure_threshold=2)

    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert call_count == 2

    result = gh.run(["issue", "list"], json_output=True, allow_failure=True)

    # Errors-as-values: no new subprocess, and the caller gets a structured
    # failure rather than an exception.
    assert call_count == 2
    assert result.ok is False
    assert result.returncode == github_module._CIRCUIT_OPEN_RETURNCODE
    assert "circuit breaker open" in (result.error or "").lower()


def test_open_breaker_raises_githuberror_when_allow_failure_false(
    monkeypatch, tmp_path: Path
) -> None:
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _transport_failure_run(cmd, *args, **kwargs)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = _gh(tmp_path, failure_threshold=1)

    with pytest.raises(github_module.GitHubError):
        gh.run(["issue", "list"], json_output=True)  # trips it
    assert call_count == 1

    with pytest.raises(github_module.GitHubError, match="circuit breaker open"):
        gh.run(["issue", "list"], json_output=True)  # fails fast, no new gh spawn
    assert call_count == 1


def test_semantic_failures_never_count_toward_the_breaker(monkeypatch, tmp_path: Path) -> None:
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 422: Validation Failed"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = _gh(tmp_path, failure_threshold=2)

    for _ in range(5):
        result = gh.run(["issue", "view", "1"], json_output=True, allow_failure=True)
        assert result.ok is False

    assert call_count == 5
    assert gh._transport._circuit_breaker_state.phase == "closed"


def test_timeout_and_file_not_found_count_as_transport_failures(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = _gh(tmp_path, failure_threshold=2)

    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.consecutive_failures == 1

    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.phase == "open"


def test_reset_circuit_breaker_rearms_without_waiting_for_cooldown(
    monkeypatch, tmp_path: Path
) -> None:
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _transport_failure_run(cmd, *args, **kwargs)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    # A long cooldown -- reset_circuit_breaker() must not need to wait it out.
    gh = _gh(tmp_path, failure_threshold=1, cooldown_seconds=3600.0)

    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.phase == "open"
    assert call_count == 1

    gh.reset_circuit_breaker()

    assert gh._transport._circuit_breaker_state.phase == "closed"
    result = gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert call_count == 2  # gh was actually spawned again, not fast-failed
    assert result.ok is False  # still fails (same fake), but as a fresh attempt


def test_breaker_trip_emits_github_circuit_opened_event(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(github_module.subprocess, "run", _transport_failure_run)
    gh = _gh(tmp_path, failure_threshold=2, cooldown_seconds=45.0)

    gh.run(["issue", "list"], json_output=True, allow_failure=True)
    gh.run(["issue", "list"], json_output=True, allow_failure=True)  # trips here

    events = query_events(_state_path(tmp_path), kind="github_circuit_opened")

    assert len(events) == 1
    assert events[0]["level"] == "warning"
    assert events[0]["payload"]["consecutive_failures"] == 2
    assert events[0]["payload"]["failure_threshold"] == 2
    assert events[0]["payload"]["cooldown_seconds"] == 45.0


def test_breaker_recovery_emits_github_circuit_closed_event(monkeypatch, tmp_path: Path) -> None:
    def fake_success_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='[{"number": 1}]', stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_success_run)
    gh = _gh(tmp_path, failure_threshold=1, cooldown_seconds=45.0)

    # Force a half-open probe state directly through the breaker's own public
    # API (cooldown_seconds=0.0 means the very next allow_call() -- using the
    # real wall clock, no sleep needed -- immediately elapses the cooldown),
    # then swap it onto the GitHub instance via the same object.__setattr__
    # escape hatch __post_init__ itself uses on this frozen dataclass. This
    # avoids waiting out a real cooldown or injecting a fake clock through
    # RuntimeConfig (which has no such knob, by design -- the breaker's
    # clock is not operator-configurable).
    probe_breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=0.0)
    probe_breaker.record_transport_failure()
    assert probe_breaker.allow_call() is True
    assert probe_breaker.phase == "half_open"
    object.__setattr__(gh, "_circuit_breaker_state", probe_breaker)

    result = gh.run(["issue", "list"], json_output=True)

    assert result == [{"number": 1}]
    assert gh._transport._circuit_breaker_state.phase == "closed"

    events = query_events(_state_path(tmp_path), kind="github_circuit_closed")
    assert len(events) == 1
    assert events[0]["level"] == "info"
