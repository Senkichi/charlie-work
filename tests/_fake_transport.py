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
    GraphQLError,
    GraphQLRequest,
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


def graphql_ok(data: object) -> Response:
    """A GraphQL ``{"data": ...}`` reply."""
    return ok({"data": data})


def graphql_failure(message: str, kind: str = "NOT_FOUND") -> Response:
    """A GraphQL reply whose ``errors`` array is populated (``Response.ok`` is False)."""
    return Response(
        200, (), '{"data": null}', "http", graphql_errors=(GraphQLError(message, kind),)
    )


def check_run(name: str, state: str = "SUCCESS", url: str = "", started: str = "") -> dict:
    """One ``CheckRun`` rollup context in the shape the checks document selects.

    *state* is a conclusion for a finished run, or ``IN_PROGRESS`` / ``QUEUED``.
    """
    done = state not in ("IN_PROGRESS", "QUEUED", "PENDING")
    return {
        "__typename": "CheckRun",
        "name": name,
        "status": "COMPLETED" if done else state,
        "conclusion": state if done else None,
        "detailsUrl": url,
        "startedAt": started or None,
        "completedAt": None,
        "checkSuite": {"workflowRun": None},
    }


def checks_reply(*contexts: dict, next_cursor: str | None = None) -> Response:
    """The reply to the ``pr checks`` document for a PR with *contexts*.

    *next_cursor* marks a page with more contexts after it.
    """
    page_info = {"hasNextPage": next_cursor is not None, "endCursor": next_cursor}
    rollup = {"contexts": {"nodes": list(contexts), "pageInfo": page_info}}
    return graphql_ok({"repository": {"pullRequest": {"statusCheckRollup": rollup}}})


def graphql_variables(request: Request) -> dict:
    """The variables of a ``GraphQLRequest`` as a dict."""
    return json.loads(request.variables)  # type: ignore[attr-defined]


def connection_page(connection: str, nodes: list, *, next_cursor: str | None = None) -> Response:
    """One page of ``repository { <connection> { nodes pageInfo } }``."""
    page_info = {"hasNextPage": next_cursor is not None, "endCursor": next_cursor}
    return graphql_ok({"repository": {connection: {"nodes": nodes, "pageInfo": page_info}}})


def paged_connection(connection: str, total: int, page_size: int = 100):
    """A handler serving *total* ``{"number": i}`` nodes in pages of *page_size*."""

    def handler(request: Request) -> Response:
        after = graphql_variables(request).get("after")
        start = int(after) if after else 0
        end = min(start + page_size, total)
        nodes = [{"number": i} for i in range(start, end)]
        return connection_page(connection, nodes, next_cursor=str(end) if end < total else None)

    return handler


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
    # When set, answers every non-token request instead of ``script`` (for
    # reads whose reply depends on the request, e.g. a per-number issue view).
    handler: Callable[[Request], Outcome] | None = None
    _served: int = 0

    def send(self, request: Request, *, token: str | None, timeout: float) -> Outcome:
        self.calls.append(Call(request, token, timeout))
        if (
            self.token is not None
            and isinstance(request, CliRequest)
            and request.command is CliCommand.AUTH_TOKEN
        ):
            return token_ok(self.token)
        if self.handler is not None:
            return self.handler(request)
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


def rest_sent(adapter: FakeAdapter) -> list[tuple[str, str, object]]:
    """``sent`` restricted to REST requests (drops GraphQL reads)."""
    return [call for call in sent(adapter) if call[0]]


def merge_adapter(put_reply: Outcome, state: str = "CLEAN") -> FakeAdapter:
    """Answers the ``mergeStateStatus`` read with *state*, every REST call with *put_reply*."""

    def handler(request: Request) -> Outcome:
        if isinstance(request, GraphQLRequest):
            return graphql_ok({"repository": {"pullRequest": {"mergeStateStatus": state}}})
        return put_reply

    return FakeAdapter("http", handler=handler)


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


def make_github(
    repo_root: Path,
    *,
    http: FakeAdapter | None = None,
    gh: FakeAdapter | None = None,
    runtime: object | None = None,
    dry_run: bool = False,
):
    """A real ``GitHub`` whose adapters are scripted fakes (no network, no gh).

    The owner/repo slug is pre-seeded, so ``{owner}/{repo}`` placeholders
    resolve without a git remote. Sleep is not injected here: tests that
    retry patch ``charlie_work.github.time.sleep`` as they always have.
    """
    from charlie_work.config import RuntimeConfig
    from charlie_work.github import GitHub

    http = http if http is not None else FakeAdapter("http", [ok({})])
    gh = gh if gh is not None else FakeAdapter("gh", token="tok-1")
    github = GitHub(
        repo_root,
        dry_run=dry_run,
        runtime=runtime if runtime is not None else RuntimeConfig(),  # type: ignore[arg-type]
        adapters=Adapters(http=http, gh=gh),
    )
    github._list_cache[("_repo_owner_name",)] = ("octo", "hello")
    # invalidate_list_cache() (once per pass) clears the cache seed above, so pin the
    # resolver itself: the slug never falls through to a real `git remote get-url`.
    object.__setattr__(github, "_repo_owner_name", lambda: ("octo", "hello"))
    # The Transport collaborator owns the real resolver and calls its own copy, so
    # pin that one too: otherwise a list read after invalidate_list_cache() shells
    # out to git. It still reads the cache first, so a test can seed another slug.
    github._transport._repo_owner_name = lambda: github._list_cache.setdefault(  # type: ignore[method-assign]
        ("_repo_owner_name",), ("octo", "hello")
    )
    return github, http, gh


def gh_kill_switch_runtime(**overrides: object):
    """A ``RuntimeConfig`` with ``gh_transport: gh``: every request is a gh
    subprocess, so tests that patch ``subprocess.run`` keep seeing real argv."""
    from charlie_work.config import RuntimeConfig

    return RuntimeConfig(gh_transport="gh", **overrides)  # type: ignore[arg-type]


def sent(adapter: FakeAdapter) -> list[tuple[str, str, object]]:
    """(method, route, decoded JSON body) of every REST request *adapter* saw.

    The guard fills ``{owner}/{repo}`` before the adapter sees a request; this
    maps ``make_github``'s seeded slug back so assertions read like the
    capability's own route templates.
    """
    out: list[tuple[str, str, object]] = []
    for request in adapter.api_requests:
        route = getattr(request, "route", "").replace("repos/octo/hello/", "repos/{owner}/{repo}/")
        body = getattr(request, "body", None)
        out.append((getattr(request, "method", ""), route, json.loads(body) if body else None))
    return out


@dataclass
class FakeRaw:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    will_close: bool = False

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers.items())

    def read(self) -> bytes:
        return self.body


class FakeSock:
    def settimeout(self, seconds: float) -> None:
        self.timeout = seconds


class FakeConn:
    """Stand-in for HTTPSConnection: queue of responses or exceptions."""

    def __init__(self, script: list, *, connect_error: BaseException | None = None) -> None:
        self.script = list(script)
        self.connect_error = connect_error
        self.sock: FakeSock | None = None
        self.timeout = 0.0
        self.requests: list[tuple[str, str, bytes | None, dict]] = []
        self.closed = False

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.sock = FakeSock()

    def request(self, method: str, path: str, body=None, headers=None) -> None:
        self.requests.append((method, path, body, dict(headers or {})))

    def getresponse(self) -> FakeRaw:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True
        self.sock = None
