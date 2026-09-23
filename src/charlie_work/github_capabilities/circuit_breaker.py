"""Per-pass circuit breaker for ``gh`` transport failures (issue #1833).

Follow-up to #1832 (overnight 2026-09-23 outage): TLS handshake timeouts and
120s ``gh`` subprocess timeouts, retried up to 4x and called serially,
stretched a single fleet pass past 90 minutes. Lowering the per-call timeout
(``transport.py``'s ``_timeout_seconds``/``_long_call_timeout_seconds``)
bounds *one* call's cost, but a degraded network still pays that bounded cost
on *every* call in the pass. This module adds the second half of the fix: a
breaker that, after a run of transport-class failures, stops spawning ``gh``
at all for the rest of the pass and fails those calls fast as values --
matching the errors-as-values invariant (CLAUDE.md) rather than blocking the
orchestrator loop on a network condition it cannot fix by retrying harder.

Deliberately decoupled from instrumentation/config here (no imports of
``charlie_work.config`` or ``charlie_work.instrumentation``): ``config.py``
already imports ``charlie_work.github`` (for ``ORCHESTRATOR_MANAGED_MERGE_FLAGS``),
so anything this module's *caller graph* re-imports at module level would
risk the same ``config`` -> ``github`` -> ``github_capabilities`` cycle
``transport.py``'s module docstring documents for ``validate_field_lists``'s
``ConfigError`` import. Keeping ``CircuitBreakerState`` a pure, I/O-free
state machine (constructed from plain ``int``/``float`` thresholds, not a
config dataclass) sidesteps that entirely and makes the four required
behaviors -- classification, trip, fast-fail, half-open, reset -- testable
without mocking a filesystem or an event sink. ``circuit_breaker_transport.py``
(which already imports ``RuntimeConfig`` only under ``TYPE_CHECKING``) is
responsible for reading ``RuntimeConfig.gh_circuit_breaker`` and for turning
this module's transition signals into ``github_circuit_opened`` /
``github_circuit_closed`` events; ``transport.py`` itself only holds the thin
``Transport`` wrapper methods that delegate to it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

__all__ = [
    "GhFailureClass",
    "classify_gh_failure",
    "CircuitBreakerState",
    "GhCircuitBreakerConfig",
]


class GhFailureClass(Enum):
    """Whether a failed ``gh`` invocation indicates a broken transport.

    ``TRANSPORT``: the request never got a response from GitHub -- a TLS/
    connect-level failure (handshake timeout, connection refused/reset, DNS
    resolution failure) or the subprocess hung past its timeout. These are
    exactly the failure shapes the #1832 outage saw, and they are what the
    breaker counts: a run of these means the network path itself is down,
    and retrying (or trying a *different* ``gh`` call) will not help until it
    recovers.

    ``SEMANTIC``: ``gh`` ran and GitHub answered -- a 4xx/422/5xx HTTP
    status, a rate limit, an auth failure, a "could not resolve to a X"
    404. The transport worked; the *request* was rejected or GitHub itself
    reported trouble. Per the acceptance criteria, semantic failures never
    count toward the breaker -- a real validation error or a legitimate 404
    must not be mistaken for "GitHub is unreachable."
    """

    TRANSPORT = "transport"
    SEMANTIC = "semantic"


# Connect/handshake/DNS marker set. Deliberately narrower than
# ``transient_errors.is_transient_network_error``'s retry allowlist: that
# classifier also treats HTTP 429/502/503/504 and rate limits as retryable
# because those *are* safe to retry, but every one of them means a response
# came back from GitHub's edge -- the opposite of what the breaker needs to
# detect. A pattern here must mean "no response reached the caller at all."
#
# Deliberately does not import that allowlist and subtract the HTTP-status
# entries from it either: the two lists answer different questions (retry
# safety vs. transport health) and conflating them by derivation would make
# an unrelated retry-policy edit silently change breaker classification.
_TRANSPORT_CLASS_MARKERS: tuple[str, ...] = (
    # TLS handshake timeout (Go net/http, the #1832 signature).
    "tls handshake timeout",
    # Connection-level failures shared with transient_errors.py's allowlist
    # (see that module's docstring for the gh-vs-git wording split).
    "connection refused",
    "connection reset",
    "connection was reset",
    "could not connect",
    "couldn't connect",
    "error connecting to",
    "i/o timeout",
    "remote end hung up",
    "empty reply from server",
    # DNS resolution failure. Not in transient_errors.py's gh-side list
    # today (only its git-specific "could not resolve host" pattern is
    # close); Go's net/http/net.DNSError renders as "no such host" on a
    # failed lookup, and gh's Windows backend surfaces the underlying
    # Winsock error via "connectex" (see transient_errors.py).
    "no such host",
    "could not resolve host",
    "connectex",
)

_TRANSPORT_CLASS_RE = re.compile(
    "|".join(re.escape(marker) for marker in _TRANSPORT_CLASS_MARKERS)
)


def classify_gh_failure(error: str | None, *, timed_out: bool = False) -> GhFailureClass:
    """Classify a failed ``gh`` invocation as transport-class or semantic.

    ``timed_out=True`` (the subprocess hit its timeout -- a hung call, the
    other #1832 signature alongside the TLS handshake timeout) is always
    transport-class regardless of ``error``, since a hang produces no usable
    stderr to pattern-match against.

    Anything not provably a connect/handshake/DNS/hang failure is treated as
    semantic -- the safe default for a breaker that must never trip on a
    real 404/422/auth failure (mirrors ``transient_errors.is_transient_network_error``'s
    own "unknown is terminal by default" stance, applied to a different
    question).
    """
    if timed_out:
        return GhFailureClass.TRANSPORT
    if not error:
        return GhFailureClass.SEMANTIC
    if _TRANSPORT_CLASS_RE.search(error.lower()):
        return GhFailureClass.TRANSPORT
    return GhFailureClass.SEMANTIC


_Phase = Literal["closed", "open", "half_open"]

#: Transition signals ``CircuitBreakerState`` returns from ``record_*``, for
#: the caller (``circuit_breaker_transport.note_circuit_breaker_result``) to
#: turn into ``github_circuit_opened`` / ``github_circuit_closed`` events.
#: ``None`` means no transition happened.
BreakerTransition = Literal["opened", "closed"]


class CircuitBreakerState:
    """Mutable per-``GitHub``-instance breaker state (never a frozen dataclass:
    this is runtime state that mutates on every call, exactly like the
    existing ``_list_cache`` dict on ``GitHub`` -- see that field's
    ``__post_init__`` comment for why mutable per-instance state is the
    established pattern here, distinct from the frozen config/value objects
    CLAUDE.md's invariant covers).

    Three states:

    - ``closed``: every call is allowed through. A transport-class failure
      increments a consecutive-failure counter; ``failure_threshold``
      consecutive transport failures trips to ``open``. Any non-transport
      outcome (success *or* a semantic failure -- both mean a response
      reached the caller, so the transport is provably fine right now)
      resets the counter to zero.
    - ``open``: every call is refused without running ``gh`` (``allow_call()``
      returns ``False``) until ``cooldown_seconds`` have elapsed since the
      trip, at which point the *next* ``allow_call()`` transitions to
      ``half_open`` and lets exactly one probe through.
    - ``half_open``: the one probe call is allowed. If it is not a transport
      failure, the breaker closes (counter reset, back to normal). If it is,
      the breaker reopens and the cooldown restarts.

    Not thread-safe: every ``gh`` invocation in this codebase already runs
    serially within one orchestrator pass against one ``GitHub`` instance
    (the #1832 postmortem's own framing -- "called serially" -- describes
    the problem this breaker fixes), so ``half_open`` never needs to arbitrate
    between concurrent probes.
    """

    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds must be >= 0, got {cooldown_seconds}")
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._phase: _Phase = "closed"
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def phase(self) -> _Phase:
        return self._phase

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    def allow_call(self) -> bool:
        """Return whether the caller may spawn ``gh`` right now.

        ``False`` means: fail the call as a value (per errors-as-values),
        never spawn a subprocess. Checking (and, when the cooldown has
        elapsed, transitioning ``open`` -> ``half_open``) happens here rather
        than in a separate "tick" method, so a caller only has to call this
        once per attempted invocation to get correct behavior.
        """
        if self._phase == "closed":
            return True
        if self._phase == "half_open":
            # Exactly one probe in flight at a time; see class docstring for
            # why no concurrency guard is needed against a second caller.
            return True
        # open
        assert self._opened_at is not None
        if self._clock() - self._opened_at >= self._cooldown_seconds:
            self._phase = "half_open"
            return True
        return False

    def record_success(self) -> BreakerTransition | None:
        """Record a ``gh`` invocation that did NOT fail with a transport-class
        error (a genuine success, or a semantic failure -- both prove the
        transport itself is working). Resets the consecutive-failure streak.

        Returns ``"closed"`` iff this call recovered the breaker from
        ``open``/``half_open`` back to ``closed`` (the half-open probe
        succeeded), else ``None``.
        """
        was_tripped = self._phase != "closed"
        self._phase = "closed"
        self._consecutive_failures = 0
        self._opened_at = None
        return "closed" if was_tripped else None

    def record_transport_failure(self) -> BreakerTransition | None:
        """Record a transport-class failure (connect/handshake/DNS/hang).

        Returns ``"opened"`` iff this call tripped the breaker (closed ->
        open on reaching ``failure_threshold``, or half_open -> open on a
        failed probe), else ``None``.
        """
        if self._phase == "half_open":
            # The probe failed: reopen and restart the cooldown. Does not
            # touch the streak counter -- it already sat at/above
            # failure_threshold from the original trip.
            self._phase = "open"
            self._opened_at = self._clock()
            return "opened"

        self._consecutive_failures += 1
        if self._phase == "closed" and self._consecutive_failures >= self._failure_threshold:
            self._phase = "open"
            self._opened_at = self._clock()
            return "opened"
        return None

    def reset(self) -> None:
        """Re-arm to ``closed`` with a clean slate.

        The per-pass reset hook (``Transport.reset_circuit_breaker``, called
        from the loop-body's start-of-pass cache invalidation): a breaker
        tripped by a degraded network during one pass must not carry into
        the next pass's first call, or a transient overnight blip would
        permanently fail-fast every future pass. Deliberately emits no
        transition event -- this is re-initialization, not a recovery.
        """
        self._phase = "closed"
        self._consecutive_failures = 0
        self._opened_at = None


@dataclass(frozen=True)
class GhCircuitBreakerConfig:
    """Per-pass circuit breaker thresholds for ``gh`` transport failures
    (issue #1833, follow-up to the #1832 overnight outage).

    After ``failure_threshold`` consecutive transport-class ``gh`` failures
    (connect/handshake/DNS/hang -- see ``classify_gh_failure`` above) in one
    orchestrator pass, further ``gh`` calls that pass fail immediately as
    values, without spawning a subprocess, until ``cooldown_seconds`` has
    elapsed, at which point one probe call is allowed through. Semantic
    failures (4xx/422/5xx, rate limits, auth) never count toward the
    threshold. Reset to a clean slate at the start of every pass
    (``GitHub.reset_circuit_breaker()``), so a bad pass cannot permanently
    fail-fast every later one.

    Ships enabled with these defaults (owner directive: every new
    feature/knob ships enabled with sensible defaults; config exists only as
    a kill switch, never a default-off opt-in). Five consecutive failures is
    high enough above single-blip noise to avoid false trips while still
    catching the #1832 pattern (every call in a pass failing the same way)
    within the first handful of calls rather than exhausting the whole pass.
    60s balances riding out a brief blip against blocking the bulk of a
    pass's runtime once the network has recovered.

    Lives here rather than in ``config.py`` for the same reason
    ``RunnerCapacityEscalationConfig`` lives in
    ``capacity_starvation_escalation.py`` (see that re-export's comment in
    ``config.py``): new code should not land in that over-cap monolith
    (file-size ratchet, issue #1442). This is a plain two-field dataclass
    with no import of ``charlie_work.config``/``charlie_work.instrumentation``
    of its own, so hosting it here does not compromise this module's
    documented purity (module docstring above) -- ``config.py`` re-exports
    it in place, unchanged, exactly like ``CircuitBreakerState``'s sibling
    types.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
