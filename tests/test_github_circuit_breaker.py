"""``GitHub.run()``'s per-pass circuit breaker wiring (issue #1833).

The breaker state machine itself (classification, trip, half-open, reset) is
unit-tested in isolation in tests/test_circuit_breaker.py. This file covers
the integration: GitHub.run() gates on the breaker, records outcomes at each
terminal exit, fails fast without sending a request once open, resets via
GitHub.reset_circuit_breaker() (the per-pass hook), and emits
github_circuit_opened/github_circuit_closed events.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _fake_transport import FakeAdapter, failure, make_github, ok
from charlie_work import github as github_module
from charlie_work.config import GhCircuitBreakerConfig, RuntimeConfig
from charlie_work.github_capabilities.circuit_breaker import CircuitBreakerState
from charlie_work.github_transport import FailureKind
from charlie_work.instrumentation import query_events


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / ".var" / "charlie-work" / "state.json"


_READ = ["api", "repos/{owner}/{repo}/issues"]  # a modelled REST read


def _refused():
    """A transport-class failure: the request went out and no response came back."""
    return failure(FailureKind.SENT_NO_RESPONSE, "connection refused")


def _gh(
    tmp_path: Path,
    *script: object,
    failure_threshold: int = 3,
    cooldown_seconds: float = 60.0,
):
    """A real ``GitHub`` whose HTTP adapter replays *script* (last entry repeats).

    Returns ``(github, http)``; ``http.api_requests`` is every request that
    actually reached the wire, so a fast-failed call is visible as no growth.
    """
    github, http, _gh_adapter = make_github(
        tmp_path,
        http=FakeAdapter("http", list(script)),  # type: ignore[arg-type]
        runtime=RuntimeConfig(
            # No internal retries: each gh.run() call maps to exactly one
            # request, so consecutive-failure counting is deterministic and
            # independent of gh_max_retries.
            gh_max_retries=0,
            gh_circuit_breaker=GhCircuitBreakerConfig(
                failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
            ),
        ),
    )
    return github, http


def test_breaker_trips_after_threshold_consecutive_transport_failures(tmp_path: Path) -> None:
    gh, http = _gh(tmp_path, _refused(), failure_threshold=3)

    for _ in range(3):
        result = gh.run(_READ, json_output=True, allow_failure=True)
        assert result.ok is False

    assert len(http.api_requests) == 3
    assert gh._transport._circuit_breaker_state.phase == "open"


def test_open_breaker_fails_fast_without_spawning_gh_and_returns_a_value(
    tmp_path: Path,
) -> None:
    gh, http = _gh(tmp_path, _refused(), failure_threshold=2)

    gh.run(_READ, json_output=True, allow_failure=True)
    gh.run(_READ, json_output=True, allow_failure=True)
    assert len(http.api_requests) == 2

    result = gh.run(_READ, json_output=True, allow_failure=True)

    # Errors-as-values: no new request, and the caller gets a structured
    # failure rather than an exception.
    assert len(http.api_requests) == 2
    assert result.ok is False
    assert result.returncode == github_module._CIRCUIT_OPEN_RETURNCODE
    assert "circuit breaker open" in (result.error or "").lower()


def test_open_breaker_raises_githuberror_when_allow_failure_false(tmp_path: Path) -> None:
    gh, http = _gh(tmp_path, _refused(), failure_threshold=1)

    with pytest.raises(github_module.GitHubError):
        gh.run(_READ, json_output=True)  # trips it
    assert len(http.api_requests) == 1

    with pytest.raises(github_module.GitHubError, match="circuit breaker open"):
        gh.run(_READ, json_output=True)  # fails fast, nothing sent
    assert len(http.api_requests) == 1


def test_semantic_failures_never_count_toward_the_breaker(tmp_path: Path) -> None:
    gh, http = _gh(tmp_path, ok({"message": "Validation Failed"}, status=422), failure_threshold=2)

    for _ in range(5):
        result = gh.run(_READ, json_output=True, allow_failure=True)
        assert result.ok is False

    assert len(http.api_requests) == 5
    assert gh._transport._circuit_breaker_state.phase == "closed"


def test_timeout_and_file_not_found_count_as_transport_failures(tmp_path: Path) -> None:
    # A timeout, then a missing gh binary: both are "no response reached us".
    gh, _http = _gh(
        tmp_path,
        failure(FailureKind.TIMEOUT, "timed out"),
        failure(FailureKind.CLI_MISSING, "gh not installed"),
        failure_threshold=2,
    )

    gh.run(_READ, json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.consecutive_failures == 1

    gh.run(_READ, json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.phase == "open"


def test_reset_circuit_breaker_rearms_without_waiting_for_cooldown(tmp_path: Path) -> None:
    # A long cooldown -- reset_circuit_breaker() must not need to wait it out.
    gh, http = _gh(tmp_path, _refused(), failure_threshold=1, cooldown_seconds=3600.0)

    gh.run(_READ, json_output=True, allow_failure=True)
    assert gh._transport._circuit_breaker_state.phase == "open"
    assert len(http.api_requests) == 1

    gh.reset_circuit_breaker()

    assert gh._transport._circuit_breaker_state.phase == "closed"
    result = gh.run(_READ, json_output=True, allow_failure=True)
    assert len(http.api_requests) == 2  # sent again, not fast-failed
    assert result.ok is False  # still fails (same fake), but as a fresh attempt


def test_breaker_trip_emits_github_circuit_opened_event(tmp_path: Path) -> None:
    gh, _http = _gh(tmp_path, _refused(), failure_threshold=2, cooldown_seconds=45.0)

    gh.run(_READ, json_output=True, allow_failure=True)
    gh.run(_READ, json_output=True, allow_failure=True)  # trips here

    events = query_events(_state_path(tmp_path), kind="github_circuit_opened")

    assert len(events) == 1
    assert events[0]["level"] == "warning"
    assert events[0]["payload"]["consecutive_failures"] == 2
    assert events[0]["payload"]["failure_threshold"] == 2
    assert events[0]["payload"]["cooldown_seconds"] == 45.0


def test_breaker_recovery_emits_github_circuit_closed_event(tmp_path: Path) -> None:
    gh, _http = _gh(tmp_path, ok([{"number": 1}]), failure_threshold=1, cooldown_seconds=45.0)

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

    result = gh.run(_READ, json_output=True)

    assert result == [{"number": 1}]
    assert gh._transport._circuit_breaker_state.phase == "closed"

    events = query_events(_state_path(tmp_path), kind="github_circuit_closed")
    assert len(events) == 1
    assert events[0]["level"] == "info"
