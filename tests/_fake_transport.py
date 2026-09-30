"""Scripted fakes for the GitHub transport (ADR-0006). No network, no clock.

``FakeAdapter`` stands in for one adapter at the ``Adapter`` seam: it plays
back a queue of ``Outcome`` values (or exceptions) and records every call.
``FakeTransport`` stands in for the whole ``GitHubTransport`` (the seam the
capabilities and ``pagination`` depend on) and records every request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from charlie_work.github_transport import (
    Adapters,
    CliCommand,
    CliRequest,
    FailureKind,
    GuardedTransport,
    Outcome,
    Request,
    Response,
    TransportFailure,
)


def ok(body: object = "", *, headers: dict[str, str] | None = None, status: int = 200) -> Response:
    text = body if isinstance(body, str) else json.dumps(body)
    pairs = tuple((k.lower(), v) for k, v in (headers or {}).items())
    return Response(status, pairs, text, "http")


def failure(kind: FailureKind, detail: str = "boom", adapter: str = "http") -> TransportFailure:
    return TransportFailure(kind, detail, adapter)  # type: ignore[arg-type]


@dataclass
class Call:
    request: Request
    token: str | None
    timeout: float


@dataclass
class FakeAdapter:
    """Plays back *script* in order; the last entry repeats once exhausted.

    ``token`` (gh fakes): ``gh auth token`` requests are answered with it and
    do not consume the script, so a test scripts only the API traffic.
    """

    name: str = "http"
    script: list[Outcome | BaseException] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    token: str | None = None
    _served: int = 0

    def send(self, request: Request, *, token: str | None, timeout: float) -> Outcome:
        self.calls.append(Call(request, token, timeout))
        if (
            self.token is not None
            and isinstance(request, CliRequest)
            and request.command is CliCommand.AUTH_TOKEN
        ):
            return token_ok(self.token)
        if not self.script:
            raise AssertionError(f"{self.name} adapter received an unscripted call: {request}")
        item = self.script[min(self._served, len(self.script) - 1)]
        self._served += 1
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def requests(self) -> list[Request]:
        return [call.request for call in self.calls]

    @property
    def api_requests(self) -> list[Request]:
        return [r for r in self.requests if not isinstance(r, CliRequest)]


@dataclass
class FakeTransport:
    """Whole-transport fake: ``handler(request) -> Outcome`` per request."""

    handler: Callable[[Request], Outcome]
    requests: list[Request] = field(default_factory=list)

    def send(self, request: Request) -> Outcome:
        self.requests.append(request)
        return self.handler(request)


@dataclass
class Sleeps:
    """Injected sleep: records delays instead of waiting."""

    delays: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def token_ok(token: str = "tok-1") -> Response:
    return Response(200, (), token + "\n", "gh", returncode=0)


@dataclass(frozen=True)
class Runtime:
    """Stand-in for ``RuntimeConfig`` (satisfies ``RuntimePort``)."""

    gh_transport: str = "http"
    gh_max_retries: int = 2
    gh_retry_base_seconds: float = 1.0
    gh_timeout_seconds: float = 30.0
    gh_long_call_timeout_seconds: float = 120.0


def build_guard(
    *,
    http: FakeAdapter | None = None,
    gh: FakeAdapter | None = None,
    runtime: Runtime | None = None,
    dry_run: bool = False,
    breaker: object | None = None,
    state_path: Path | None = None,
    sleeps: Sleeps | None = None,
    exceeded: Callable[[], bool] | None = None,
    owner_repo: tuple[str, str] | None = ("octo", "hello"),
) -> tuple[GuardedTransport, FakeAdapter, FakeAdapter, Sleeps]:
    """A ``GuardedTransport`` over fakes with deterministic sleep/jitter/clock.

    Pass a gh fake with ``token=`` so ``http`` calls get a token; API requests
    that reached gh are in ``gh.api_requests``.
    """
    http = http if http is not None else FakeAdapter("http", [ok({})])
    gh = gh if gh is not None else FakeAdapter("gh", token="tok-1")
    sleeps = sleeps if sleeps is not None else Sleeps()
    guard = GuardedTransport(
        Adapters(http=http, gh=gh),
        runtime=runtime if runtime is not None else Runtime(),
        dry_run=dry_run,
        breaker=breaker,  # type: ignore[arg-type]
        state_path=state_path,
        resolve_owner_repo=(lambda: owner_repo) if owner_repo is not None else None,
        pass_deadline_exceeded=exceeded,
        sleep=sleeps,
        jitter=lambda lo, hi: 0.0,
        now=lambda: 1000.0,
    )
    return guard, http, gh, sleeps
