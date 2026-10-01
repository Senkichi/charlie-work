"""The single wrapper stack around the GitHub adapters (ADR-0006).

``GuardedTransport.send`` is the one place retry, circuit breaker, dry-run,
pass deadline, token handling, per-call fallback and rate-limit accounting
live. Order, outermost to innermost:

1. fill ``{owner}``/``{repo}`` placeholders
2. dry-run: a mutation (derived by ``Request.is_mutation``) returns a
   synthetic success and touches nothing -- no breaker, no network, no budget
3. breaker gate: an open breaker fails the call fast as a value
4. retry loop: deadline check, choose adapter, token, send, 401 re-resolve,
   per-call gh fallback, budget observation, classify, backoff
5. the breaker hears about the FINAL outcome exactly once

It imports neither ``config`` nor ``github_capabilities``: the runtime,
breaker and owner/repo resolver arrive as small protocols, so this package
stays below both in the import graph.

Fallback is per call and never a mode: only ``TOKEN_UNAVAILABLE``,
``ADAPTER_DEFECT`` and ``CONNECT`` fall back, because those three guarantee
the request was not applied and replaying it through gh is safe even for a
mutation. Any HTTP status, any GraphQL error, ``SENT_NO_RESPONSE`` and
``TIMEOUT`` never fall back. With the kill switch on
(``runtime.gh_transport == "gh"``) gh is the configured transport, not a
fallback, so no fallback events are emitted.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from ..api_budget import GitHubRateBudget, GitHubRateWindow, github_headroom, observe_github_rate
from ..instrumentation import log_event
from ..pass_deadline import raise_if_pass_deadline_spent
from ..transient_errors import is_transient_network_error
from .failure_markers import is_pre_connection_text, is_transport_class_text
from .gh_adapter import render_argv
from .legacy_argv import LegacyCli
from .outcome import FailureKind, Outcome, Response, TransportFailure, render_legacy_error
from .request import CliCommand, CliRequest, Request, RestRequest

# What the guard accepts: the typed requests plus the transitional verbatim
# ``gh`` passthrough (``LegacyCli``), which is deleted with ``GitHub.run``.
AnyRequest = Request | LegacyCli

logger = logging.getLogger(__name__)

# Fractional jitter applied to each retry backoff (0.25 => +/- 25%).
_JITTER_FRACTION = 0.25

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_SECONDS = 1.0
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_LONG_CALL_TIMEOUT_SECONDS = 120.0
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_DETAIL_LIMIT = 300

FALLBACK_KINDS = frozenset(
    {FailureKind.TOKEN_UNAVAILABLE, FailureKind.ADAPTER_DEFECT, FailureKind.CONNECT}
)
# Failures that mean "no response reached us": counted by the breaker.
_BREAKER_KINDS = frozenset(
    {
        FailureKind.CONNECT,
        FailureKind.SENT_NO_RESPONSE,
        FailureKind.TIMEOUT,
        FailureKind.CLI_MISSING,
    }
)


class Adapter(Protocol):
    name: str

    def send(self, request: AnyRequest, *, token: str | None, timeout: float) -> Outcome: ...


class GitHubTransport(Protocol):
    """What the capabilities depend on."""

    def send(self, request: Request) -> Outcome: ...


class RuntimePort(Protocol):
    """The slice of ``RuntimeConfig`` the guard reads, live on every call."""

    @property
    def gh_transport(self) -> str: ...
    @property
    def gh_max_retries(self) -> int: ...
    @property
    def gh_retry_base_seconds(self) -> float: ...
    @property
    def gh_timeout_seconds(self) -> float: ...
    @property
    def gh_long_call_timeout_seconds(self) -> float: ...


class BreakerPort(Protocol):
    """Satisfied by ``github_capabilities.circuit_breaker.CircuitBreakerState``."""

    @property
    def consecutive_failures(self) -> int: ...
    @property
    def failure_threshold(self) -> int: ...
    @property
    def cooldown_seconds(self) -> float: ...
    def allow_call(self) -> bool: ...
    def record_success(self) -> str | None: ...
    def record_transport_failure(self) -> str | None: ...


@dataclass(frozen=True)
class Adapters:
    http: Adapter
    gh: Adapter


class RateBudgetHolder:
    """Runtime holder for the observed rate budget: one lock, one replaced value."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = GitHubRateBudget()

    @property
    def value(self) -> GitHubRateBudget:
        with self._lock:
            return self._value

    def observe(self, response: Response, now: float) -> None:
        with self._lock:
            self._value = observe_github_rate(self._value, response.headers, now)


def circuit_open_message(breaker: BreakerPort, description: str) -> str:
    """Same wording as ``circuit_breaker_transport.circuit_breaker_open_message``."""
    return (
        f"GitHub circuit breaker open: refusing to run {description} "
        f"after {breaker.consecutive_failures} consecutive transport-class "
        f"gh failure(s) (threshold {breaker.failure_threshold}); cooldown "
        f"{breaker.cooldown_seconds:g}s"
    )


def _is_rate_limited(response: Response) -> bool:
    if response.status == 429:
        return True
    if response.status == 403 and response.header("x-ratelimit-remaining") == "0":
        return True
    return any(error.type == "RATE_LIMITED" for error in response.graphql_errors)


def is_retryable(outcome: Outcome, *, is_mutation: bool) -> bool:
    """Whether *outcome* may be retried (derived from typed fields, not text)."""
    if isinstance(outcome, TransportFailure):
        if outcome.kind is FailureKind.CONNECT:
            return True  # provably not sent: safe even for a mutation
        if outcome.kind in (FailureKind.SENT_NO_RESPONSE, FailureKind.TIMEOUT):
            return not is_mutation  # a mutation may already have landed
        return False
    if outcome.status == 0:
        # Text-classified (legacy gh prose): a mutation only retries on
        # failures that provably happened before the request was sent.
        text = render_legacy_error(outcome)
        return response_is_transient(outcome) and (not is_mutation or is_pre_connection_text(text))
    if is_mutation:
        return False
    return response_is_transient(outcome)


def response_is_transient(response: Response) -> bool:
    if response.status == 0:
        # A gh-local failure (LegacyCli / CliRequest): only prose to go on.
        return is_transient_network_error(render_legacy_error(response))
    return response.status in _RETRYABLE_STATUSES or _is_rate_limited(response)


def _response_is_transport_class(response: Response) -> bool:
    """A gh-local failure whose prose says no response ever reached gh."""
    return response.status == 0 and is_transport_class_text(render_legacy_error(response))


class GuardedTransport:
    """``GitHubTransport`` over a pair of adapters; see the module docstring."""

    def __init__(
        self,
        adapters: Adapters,
        *,
        runtime: RuntimePort | None = None,
        dry_run: bool = False,
        breaker: BreakerPort | None = None,
        state_path: Path | None = None,
        resolve_owner_repo: Callable[[], tuple[str, str]] | None = None,
        pass_deadline_exceeded: Callable[[], bool] | None = None,
        budget: RateBudgetHolder | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._adapters = adapters
        self._runtime = runtime
        self.dry_run = dry_run
        self._breaker = breaker
        self._state_path = state_path
        self._resolve_owner_repo = resolve_owner_repo
        self._exceeded = pass_deadline_exceeded
        self.budget = budget if budget is not None else RateBudgetHolder()
        self._sleep = sleep
        self._jitter = jitter
        self._now = now
        self._token: str | None = None
        self._token_lock = threading.Lock()
        self._breaker_lock = threading.Lock()

    # -- config (read live, like the legacy per-call knob reads) -------------

    def _kill_switch(self) -> bool:
        return self._runtime is not None and self._runtime.gh_transport == "gh"

    @property
    def kill_switch(self) -> bool:
        """True while ``runtime.gh_transport`` routes every request to gh."""
        return self._kill_switch()

    def _max_retries(self) -> int:
        return self._runtime.gh_max_retries if self._runtime else _DEFAULT_MAX_RETRIES

    def _retry_base(self) -> float:
        rt = self._runtime
        return rt.gh_retry_base_seconds if rt else _DEFAULT_RETRY_BASE_SECONDS

    def _timeout(self, request: AnyRequest) -> float:
        long_call = getattr(request, "long_call", False)
        if long_call:
            rt = self._runtime
            return rt.gh_long_call_timeout_seconds if rt else _DEFAULT_LONG_CALL_TIMEOUT_SECONDS
        return self._runtime.gh_timeout_seconds if self._runtime else _DEFAULT_TIMEOUT_SECONDS

    def fresh_rate_window(self, resource: str, max_age_seconds: float) -> GitHubRateWindow | None:
        """The observed window for *resource* if seen within *max_age_seconds*
        and not yet reset (B15), else ``None``: the caller must ask GitHub."""
        now = self._now()
        window = github_headroom(self.budget.value, resource, now)
        if window is None or now - window.observed_epoch > max_age_seconds:
            return None
        return window

    def set_pass_deadline_exceeded(self, check: Callable[[], bool] | None) -> None:
        self._exceeded = check

    def reset_circuit_breaker(self) -> None:
        reset = getattr(self._breaker, "reset", None)
        if callable(reset):
            with self._breaker_lock:
                reset()

    # -- the stack -----------------------------------------------------------

    def send(self, request: AnyRequest) -> Outcome:
        resolved = self._fill_placeholders(request)
        if isinstance(resolved, TransportFailure):
            return resolved
        request = resolved
        if self.dry_run and request.is_mutation:
            return Response(200, (), "", "dry_run")
        command = [request.describe()]
        if self._breaker is not None:
            with self._breaker_lock:
                allowed = self._breaker.allow_call()
            if not allowed:
                detail = circuit_open_message(self._breaker, request.describe())
                return TransportFailure(FailureKind.CIRCUIT_OPEN, detail, "guard")
        max_retries = self._max_retries()
        outcome: Outcome | None = None
        for attempt in range(max_retries + 1):
            raise_if_pass_deadline_spent(self._exceeded, command)
            outcome = self._attempt(request)
            if isinstance(outcome, Response):
                self.budget.observe(outcome, self._now())
            if attempt >= max_retries or not is_retryable(
                outcome, is_mutation=request.is_mutation
            ):
                break
            raise_if_pass_deadline_spent(self._exceeded, command)
            delay = self._retry_base() * (2**attempt)
            wait = max(
                0.0, delay + self._jitter(-_JITTER_FRACTION * delay, _JITTER_FRACTION * delay)
            )
            logger.warning(
                "GitHub %s failed (attempt %d/%d); retrying in %.2fs",
                request.describe(),
                attempt + 1,
                max_retries + 1,
                wait,
            )
            self._sleep(wait)
        assert outcome is not None
        self._note_breaker(outcome)
        return outcome

    def _fill_placeholders(self, request: AnyRequest) -> AnyRequest | TransportFailure:
        if not isinstance(request, RestRequest) or not request.needs_repo:
            return request
        if self._resolve_owner_repo is None:
            return TransportFailure(
                FailureKind.ADAPTER_DEFECT, "no owner/repo resolver for a templated route", "guard"
            )
        try:
            owner, repo = self._resolve_owner_repo()
            return request.resolve(owner, repo)
        except Exception as exc:  # noqa: BLE001 - resolver failures are values at this seam
            return TransportFailure(
                FailureKind.ADAPTER_DEFECT, f"{type(exc).__name__}: {exc}", "guard"
            )

    # -- one attempt ---------------------------------------------------------

    def _attempt(self, request: AnyRequest) -> Outcome:
        timeout = self._timeout(request)
        if isinstance(request, (CliRequest, LegacyCli)) or self._kill_switch():
            return self._adapters.gh.send(request, token=None, timeout=timeout)
        outcome = self._send_http(request, timeout)
        if isinstance(outcome, TransportFailure) and outcome.kind in FALLBACK_KINDS:
            self._emit_fallback(request, outcome)
            return self._adapters.gh.send(request, token=None, timeout=timeout)
        return outcome

    def _send_http(self, request: Request, timeout: float) -> Outcome:
        token = self._token_or_resolve()
        if token is None:
            return TransportFailure(FailureKind.TOKEN_UNAVAILABLE, "gh auth token failed", "guard")
        outcome = self._adapters.http.send(request, token=token, timeout=timeout)
        if isinstance(outcome, Response) and outcome.status == 401:
            # A 401 was not applied, so one re-resolve + resend is safe for mutations too.
            token = self._token_or_resolve(refresh=True)
            if token is None:
                detail = "gh auth token re-resolution failed after HTTP 401"
                return TransportFailure(FailureKind.TOKEN_UNAVAILABLE, detail, "guard")
            outcome = self._adapters.http.send(request, token=token, timeout=timeout)
        return outcome

    def _token_or_resolve(self, *, refresh: bool = False) -> str | None:
        with self._token_lock:
            if refresh:
                self._token = None
            if self._token is not None:
                return self._token
        # Recursive guarded call: a CliRequest always routes to gh, so no cycle.
        result = self.send(CliRequest(CliCommand.AUTH_TOKEN))
        token = result.body.strip() if isinstance(result, Response) and result.ok else ""
        with self._token_lock:
            self._token = token or None
            return self._token

    # -- side effects --------------------------------------------------------

    def _emit_fallback(self, request: Request, failure: TransportFailure) -> None:
        if self._state_path is None:
            return
        payload = {
            "command": " ".join(render_argv(request)[0]),
            "request": request.describe(),
            "reason": failure.kind.value,
            "detail": failure.detail[:_DETAIL_LIMIT],
            "mutation": request.is_mutation,
        }
        # write-gate-exempt(issue=1834): GitHub client layer has no WriteGate; transport fallback bookkeeping is lock-free
        log_event(self._state_path, "github_transport_fallback", payload)

    def _note_breaker(self, outcome: Outcome) -> None:
        """Record the FINAL outcome once: a transport-class failure counts
        toward opening; any response (even an error status) proves the
        transport works and resets the streak."""
        breaker = self._breaker
        if breaker is None:
            return
        counts = (
            outcome.kind in _BREAKER_KINDS
            if isinstance(outcome, TransportFailure)
            else _response_is_transport_class(outcome)
        )
        with self._breaker_lock:
            transition = breaker.record_transport_failure() if counts else breaker.record_success()
            payload = {
                "consecutive_failures": breaker.consecutive_failures,
                "failure_threshold": breaker.failure_threshold,
                "cooldown_seconds": breaker.cooldown_seconds,
            }
        if self._state_path is None:
            return
        if transition == "opened":
            # write-gate-exempt(issue=1833): GitHub client layer has no WriteGate; breaker state is lock-free
            log_event(self._state_path, "github_circuit_opened", payload)
        elif transition == "closed":
            # write-gate-exempt(issue=1833): GitHub client layer has no WriteGate; breaker state is lock-free
            log_event(
                self._state_path,
                "github_circuit_closed",
                {"cooldown_seconds": payload["cooldown_seconds"]},
            )


__all__ = [
    "Adapter",
    "Adapters",
    "AnyRequest",
    "BreakerPort",
    "FALLBACK_KINDS",
    "GitHubTransport",
    "GuardedTransport",
    "RateBudgetHolder",
    "RuntimePort",
    "circuit_open_message",
    "is_retryable",
]
