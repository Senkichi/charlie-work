"""Build the per-``GitHub`` ``GuardedTransport`` (ADR-0006).

One place assembles the guard from the ``GitHub`` instance's breaker state,
runtime knobs, pass-deadline hook and owner/repo resolver. Everything that can
change after construction (the pass-deadline predicate, ``time.sleep``,
``random.uniform``, the runtime's retry knobs) is read live through a lambda,
so arming a deadline or patching a clock after ``GitHub(...)`` still takes
effect, exactly as it did when ``run`` read them per call.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import layout
from ..github_transport.gh_adapter import GhAdapter
from ..github_transport.guarded import Adapters, GuardedTransport, RateBudgetHolder
from ..github_transport.http_adapter import HttpAdapter
from ..github_transport.shared_budget import SharedBudgetFile
from . import http_cache
from .circuit_breaker_transport import circuit_breaker_state_path

if TYPE_CHECKING:
    from ..github import GitHub

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_SECONDS = 1.0
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_LONG_CALL_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class _DefaultRuntime:
    """Stand-in for ``runtime=None``: HTTP with the default retry knobs."""

    gh_transport: str = "http"
    gh_max_retries: int = _DEFAULT_MAX_RETRIES
    gh_retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS
    gh_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    gh_long_call_timeout_seconds: float = _DEFAULT_LONG_CALL_TIMEOUT_SECONDS


class _LiveBreaker:
    """``BreakerPort`` that resolves ``gh._circuit_breaker_state`` on every call.

    The state object is replaced wholesale by ``reset``-style code and by
    tests (``object.__setattr__(gh, "_circuit_breaker_state", ...)``), so the
    guard must not hold a stale reference.
    """

    def __init__(self, gh: "GitHub") -> None:
        self._gh = gh

    @property
    def consecutive_failures(self) -> int:
        return self._gh._circuit_breaker_state.consecutive_failures

    @property
    def failure_threshold(self) -> int:
        return self._gh._circuit_breaker_state.failure_threshold

    @property
    def cooldown_seconds(self) -> float:
        return self._gh._circuit_breaker_state.cooldown_seconds

    def allow_call(self) -> bool:
        return self._gh._circuit_breaker_state.allow_call()

    def record_success(self) -> str | None:
        return self._gh._circuit_breaker_state.record_success()

    def record_transport_failure(self) -> str | None:
        return self._gh._circuit_breaker_state.record_transport_failure()

    def reset(self) -> None:
        self._gh._circuit_breaker_state.reset()


def etag_cache_path(runtime: object | None, repo_root: Path) -> Path:
    state_dir = getattr(runtime, "state_dir", None) or layout.DEFAULT_STATE_DIR
    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    return layout.http_etag_cache_path(root.resolve())


def build_guarded_transport(gh: "GitHub") -> GuardedTransport:
    """The guard for *gh*; ``gh.adapters`` overrides the real adapters (tests)."""
    runtime = gh.runtime if gh.runtime is not None else _DefaultRuntime()
    adapters = gh.adapters
    if adapters is None:
        adapters = Adapters(
            http=HttpAdapter(
                cache=http_cache.FileEtagCache(etag_cache_path(gh.runtime, gh.repo_root))
            ),
            gh=GhAdapter(gh.repo_root),
        )

    def deadline_exceeded() -> bool:
        check = gh._pass_deadline_exceeded
        return bool(check is not None and check())

    # Issue #2442: one observed budget per token across every client and
    # process on this host, unless the governor's kill switch is off.
    shared = (
        SharedBudgetFile(layout.github_budget_path())
        if getattr(runtime, "github_budget_governor", True)
        else None
    )
    return GuardedTransport(
        adapters,
        budget=RateBudgetHolder(shared=shared),
        runtime=runtime,  # type: ignore[arg-type]
        dry_run=gh.dry_run,
        breaker=_LiveBreaker(gh),
        state_path=circuit_breaker_state_path(gh.runtime, gh.repo_root),
        resolve_owner_repo=lambda: gh._repo_owner_name(),
        pass_deadline_exceeded=deadline_exceeded,
        sleep=lambda seconds: time.sleep(seconds),
        jitter=lambda low, high: random.uniform(low, high),
        now=lambda: time.time(),
    )
