"""Unit tests for ``github_capabilities.circuit_breaker`` (issue #1833).

Pure, I/O-free state machine and classifier -- see that module's docstring
for why it is deliberately decoupled from config/instrumentation. Covers the
four acceptance-criteria behaviors directly: classification, trip, half-open,
reset. GitHub.run()'s wiring (fast-fail without spawning gh, event emission)
is covered separately in tests/test_github_circuit_breaker.py.
"""

from __future__ import annotations

import pytest

from charlie_work.github_capabilities.circuit_breaker import (
    CircuitBreakerState,
    GhFailureClass,
    classify_gh_failure,
)


class _FakeClock:
    """A deterministic, manually-advanced stand-in for time.monotonic."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- classify_gh_failure -----------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        'Post "https://api.github.com/graphql": net/http: TLS handshake timeout',
        "dial tcp: connection refused",
        "read: connection reset by peer",
        "curl: (56) Recv failure: Connection was reset",
        "gh: could not connect to server",
        "couldn't connect to host",
        "Error connecting to api.github.com",
        "i/o timeout",
        "remote end hung up unexpectedly",
        "empty reply from server",
        "dial tcp: lookup api.github.com: no such host",
        "gh: Could not resolve host: api.github.com",
        "connectex: A connection attempt failed",
    ],
    ids=[
        "tls_handshake_timeout",
        "connection_refused",
        "connection_reset",
        "connection_was_reset",
        "could_not_connect",
        "couldnt_connect",
        "error_connecting_to",
        "io_timeout",
        "remote_end_hung_up",
        "empty_reply",
        "no_such_host",
        "could_not_resolve_host",
        "connectex",
    ],
)
def test_classify_gh_failure_detects_transport_markers(error: str) -> None:
    assert classify_gh_failure(error) is GhFailureClass.TRANSPORT


def test_classify_gh_failure_is_case_insensitive() -> None:
    assert classify_gh_failure("TLS Handshake Timeout") is GhFailureClass.TRANSPORT
    assert classify_gh_failure("CONNECTION REFUSED") is GhFailureClass.TRANSPORT


@pytest.mark.parametrize(
    "error",
    [
        "Bad credentials",
        "Could not resolve to a Issue with the number 999.",
        "HTTP 422: Validation Failed",
        "HTTP 502: Bad Gateway",
        "HTTP 429: rate limit exceeded",
        "Unknown JSON field",
        None,
        "",
    ],
    ids=[
        "bad_credentials",
        "not_found",
        "validation_422",
        "gateway_502",
        "rate_limit_429",
        "unknown_field",
        "none",
        "empty_string",
    ],
)
def test_classify_gh_failure_treats_semantic_and_empty_as_semantic(error: str | None) -> None:
    assert classify_gh_failure(error) is GhFailureClass.SEMANTIC


def test_classify_gh_failure_timed_out_is_always_transport_regardless_of_text() -> None:
    # A hang produces no usable stderr; timed_out=True must win regardless of
    # whatever text (or lack of it) accompanies it.
    assert classify_gh_failure(None, timed_out=True) is GhFailureClass.TRANSPORT
    assert (
        classify_gh_failure("HTTP 422: Validation Failed", timed_out=True)
        is GhFailureClass.TRANSPORT
    )


# --- CircuitBreakerState construction ----------------------------------------


def test_constructor_rejects_invalid_failure_threshold() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreakerState(failure_threshold=0, cooldown_seconds=1.0)


def test_constructor_rejects_negative_cooldown() -> None:
    with pytest.raises(ValueError, match="cooldown_seconds"):
        CircuitBreakerState(failure_threshold=1, cooldown_seconds=-1.0)


def test_starts_closed_and_allows_calls() -> None:
    breaker = CircuitBreakerState(failure_threshold=3, cooldown_seconds=60.0)
    assert breaker.phase == "closed"
    assert breaker.allow_call() is True
    assert breaker.consecutive_failures == 0


# --- trip ---------------------------------------------------------------------


def test_trips_after_exactly_failure_threshold_consecutive_failures() -> None:
    breaker = CircuitBreakerState(failure_threshold=3, cooldown_seconds=60.0)

    assert breaker.record_transport_failure() is None
    assert breaker.phase == "closed"
    assert breaker.record_transport_failure() is None
    assert breaker.phase == "closed"
    assert breaker.record_transport_failure() == "opened"
    assert breaker.phase == "open"
    assert breaker.consecutive_failures == 3


def test_non_transport_outcome_resets_the_consecutive_streak() -> None:
    breaker = CircuitBreakerState(failure_threshold=3, cooldown_seconds=60.0)

    breaker.record_transport_failure()
    breaker.record_transport_failure()
    assert breaker.consecutive_failures == 2

    assert breaker.record_success() is None  # not tripped -> no transition
    assert breaker.phase == "closed"
    assert breaker.consecutive_failures == 0

    # A fresh run of failures still needs the full threshold again.
    breaker.record_transport_failure()
    breaker.record_transport_failure()
    assert breaker.phase == "closed"


# --- fast-fail while open ------------------------------------------------------


def test_open_breaker_refuses_calls_until_cooldown_elapses() -> None:
    clock = _FakeClock(start=100.0)
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=30.0, clock=clock)

    breaker.record_transport_failure()
    assert breaker.phase == "open"
    assert breaker.allow_call() is False

    clock.advance(29.9)
    assert breaker.allow_call() is False
    assert breaker.phase == "open"

    clock.advance(0.2)  # total elapsed now >= 30.0
    assert breaker.allow_call() is True
    assert breaker.phase == "half_open"


# --- half-open probe ------------------------------------------------------------


def test_half_open_probe_success_closes_the_breaker() -> None:
    clock = _FakeClock()
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=10.0, clock=clock)

    breaker.record_transport_failure()
    clock.advance(10.0)
    assert breaker.allow_call() is True
    assert breaker.phase == "half_open"

    assert breaker.record_success() == "closed"
    assert breaker.phase == "closed"
    assert breaker.consecutive_failures == 0


def test_half_open_probe_failure_reopens_and_restarts_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=10.0, clock=clock)

    breaker.record_transport_failure()
    clock.advance(10.0)
    assert breaker.allow_call() is True  # -> half_open

    assert breaker.record_transport_failure() == "opened"
    assert breaker.phase == "open"

    # Cooldown restarted from the failed probe's time, not the original trip.
    assert breaker.allow_call() is False
    clock.advance(9.9)
    assert breaker.allow_call() is False
    clock.advance(0.2)
    assert breaker.allow_call() is True


# --- reset (per-pass rearm) ------------------------------------------------------


def test_reset_rearms_a_tripped_breaker_to_closed() -> None:
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=60.0)

    breaker.record_transport_failure()
    assert breaker.phase == "open"

    breaker.reset()

    assert breaker.phase == "closed"
    assert breaker.consecutive_failures == 0
    assert breaker.allow_call() is True
