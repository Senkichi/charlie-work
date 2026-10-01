"""Pooled stdlib HTTPS adapter (ADR-0006): the default GitHub transport.

Stdlib only (``http.client.HTTPSConnection`` + ``ssl``); no HTTP client
dependency. The adapter speaks the generic ``Request`` values directly and
returns ``Outcome`` values -- it never raises for a network or HTTP
condition. Retry, breaker, dry-run, deadline, token resolution and the 401
re-resolve all live in ``guarded.GuardedTransport``, not here.

Connection pool: a lock-guarded LIFO free list of keep-alive connections to
``api.github.com``. A connection is checked out for exactly one request, so
concurrent callers (the ``issue_view`` fan-out) never share a socket, and is
returned on success or dropped on any failure. At most
``_MAX_IDLE_CONNECTIONS`` idle connections are kept.

Failure mapping distinguishes *where* in the exchange the failure happened,
because that decides whether a mutation may be replayed:

* connect()/handshake failed (DNS, refused, TLS)     -> ``CONNECT`` (not sent)
* timeout after the request was written              -> ``TIMEOUT``
* any other OS/SSL/HTTP error after the request      -> ``SENT_NO_RESPONSE``
* no token / malformed request or response body      -> ``TOKEN_UNAVAILABLE`` /
                                                        ``ADAPTER_DEFECT``
"""

from __future__ import annotations

import json
import select
import socket
import ssl
import threading
from dataclasses import dataclass, field
from http.client import HTTPException, HTTPSConnection
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from .outcome import (
    GITHUB_API_HOST,
    FailureKind,
    Outcome,
    Response,
    TransportFailure,
    graphql_errors_from_body,
    normalize_headers,
)
from .request import GraphQLRequest, Request, RestRequest
from .token_hygiene import is_well_formed, redact

# Matches the `issue_view` fan-out's max_workers ceiling: more idle sockets
# than concurrent callers could ever use would only hold server resources.
_MAX_IDLE_CONNECTIONS = 8

_API_VERSION = "2022-11-28"
_USER_AGENT = "charlie-work-http-transport"
_NOT_MODIFIED = 304
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFECT_ERRORS = (KeyError, TypeError, AttributeError, ValueError)


def _socket_dropped(sock: Any) -> bool:
    """Whether an idle pooled socket was closed by the peer.

    An idle keep-alive socket should never be readable: readable means EOF
    (the server idle-closed it), a TLS close_notify, or stray bytes -- in
    every case it must not carry a new request. Without this check the
    request "succeeds" into the kernel buffer and only ``getresponse()``
    fails, which is indistinguishable from "sent, no response" and so
    cannot be retried for a mutation. Test doubles without a real file
    descriptor are treated as live.
    """
    try:
        fileno = sock.fileno()
    except AttributeError:
        return False
    except OSError:
        return True
    if not isinstance(fileno, int):
        return False
    if fileno < 0:
        return True
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


def _new_connection(host: str, timeout: float) -> HTTPSConnection:
    return HTTPSConnection(host, timeout=timeout)


class CachedEntry(Protocol):
    etag: str
    status: int
    body: str


class EtagCache(Protocol):
    """Conditional-GET store (``github_capabilities.http_cache`` backs it).

    Injected rather than imported so this package stays below the capability
    collaborators in the import graph.
    """

    def get(self, path: str) -> CachedEntry | None: ...

    def record(self, path: str, *, etag: str, status: int, body: str) -> None: ...


@dataclass(frozen=True)
class _Prepared:
    method: str
    path: str
    body: bytes | None
    # Holds ``Authorization: Bearer <token>``: never part of a repr.
    headers: dict[str, str] = field(repr=False)
    idempotent: bool = False


class HttpAdapter:
    """``Adapter`` implementation over pooled ``HTTPSConnection`` objects."""

    name = "http"

    def __init__(
        self,
        *,
        host: str = GITHUB_API_HOST,
        cache: EtagCache | None = None,
        connection_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        self._host = host
        self._cache = cache
        self._factory = connection_factory
        self._lock = threading.Lock()
        self._idle: list[Any] = []

    # -- pool ---------------------------------------------------------------

    def _make(self, host: str, timeout: float) -> Any:
        factory = self._factory if self._factory is not None else _new_connection
        return factory(host, timeout)

    def _checkout(self, timeout: float, *, fresh: bool = False) -> tuple[Any, bool]:
        """Return ``(connection, reused)``; dead idle sockets are discarded."""
        while not fresh:
            with self._lock:
                conn = self._idle.pop() if self._idle else None
            if conn is None:
                break
            sock = getattr(conn, "sock", None)
            if sock is not None and _socket_dropped(sock):
                self._close(conn)
                continue
            conn.timeout = timeout
            if sock is not None:
                sock.settimeout(timeout)
            return conn, True
        return self._make(self._host, timeout), False

    def _checkin(self, conn: Any) -> None:
        with self._lock:
            if len(self._idle) < _MAX_IDLE_CONNECTIONS:
                self._idle.append(conn)
                return
        self._close(conn)

    @staticmethod
    def _close(conn: Any) -> None:
        try:
            conn.close()
        except (OSError, HTTPException):
            pass

    def idle_connections(self) -> int:
        with self._lock:
            return len(self._idle)

    # -- request building ---------------------------------------------------

    def _prepare(self, request: Request, token: str, cached: CachedEntry | None) -> _Prepared:
        headers = {
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": _API_VERSION,
            "User-Agent": _USER_AGENT,
            "Host": self._host,
        }
        if isinstance(request, GraphQLRequest):
            payload = {"query": request.document, "variables": json.loads(request.variables)}
            headers["Accept"] = "application/vnd.github+json"
            headers["Content-Type"] = "application/json"
            return _Prepared(
                "POST",
                "/graphql",
                json.dumps(payload).encode("utf-8"),
                headers,
                idempotent=not request.is_mutation,
            )
        if isinstance(request, RestRequest):
            headers["Accept"] = request.accept
            body = request.body.encode("utf-8") if request.body is not None else None
            if body is not None:
                headers["Content-Type"] = "application/json"
            if cached is not None:
                headers["If-None-Match"] = cached.etag
            return _Prepared(
                request.method,
                "/" + request.target(),
                body,
                headers,
                idempotent=not request.is_mutation,
            )
        raise TypeError(f"HttpAdapter cannot send {type(request).__name__}")

    # -- send ---------------------------------------------------------------

    def send(self, request: Request, *, token: str | None, timeout: float) -> Outcome:
        if not isinstance(request, (RestRequest, GraphQLRequest)):
            return self._defect(f"HttpAdapter cannot send {type(request).__name__}")
        if not token:
            return TransportFailure(FailureKind.TOKEN_UNAVAILABLE, "no bearer token", "http")
        if not is_well_formed(token):
            # Never reaches the wire (http.client would reject the header and echo
            # it); the detail deliberately says nothing about the value.
            detail = "bearer token is not a valid header value"
            return TransportFailure(FailureKind.TOKEN_UNAVAILABLE, detail, "http")
        try:
            outcome = self._send(request, token, timeout)
        except _DEFECT_ERRORS as exc:
            outcome = self._defect(f"{type(exc).__name__}: {exc}")
        return self._scrubbed(outcome, token)

    @staticmethod
    def _scrubbed(outcome: Outcome, token: str) -> Outcome:
        """The adapter boundary: no failure detail leaves carrying the credential.

        Exception text (``http.client`` echoes rejected header values, an OS error
        can name a request line) is the only free-form text the adapter builds, so
        redacting here covers every failure kind at the one place the token is known.
        """
        if not isinstance(outcome, TransportFailure):
            return outcome
        detail = redact(outcome.detail, token)
        if detail == outcome.detail:
            return outcome
        return TransportFailure(outcome.kind, detail, outcome.adapter)

    def _send(self, request: RestRequest | GraphQLRequest, token: str, timeout: float) -> Outcome:
        cached = None
        if (
            isinstance(request, RestRequest)
            and request.method == "GET"
            and self._cache is not None
        ):
            cached = self._cache.get("/" + request.target())
        prepared = self._prepare(request, token, cached)
        outcome = self._exchange(prepared, timeout)
        if not isinstance(outcome, Response):
            return outcome
        if isinstance(request, GraphQLRequest):
            return self._finish_graphql(request, outcome)
        if outcome.status == _NOT_MODIFIED and cached is not None:
            return Response(cached.status, outcome.headers, cached.body, "http")
        if self._cache is not None and request.method == "GET":
            self._remember(prepared.path, outcome)
        if request.follow_redirect and outcome.status in _REDIRECT_STATUSES:
            return self._follow_redirect(outcome, timeout)
        return outcome

    def _remember(self, path: str, response: Response) -> None:
        etag = response.header("etag")
        if self._cache is not None and response.status == 200 and etag:
            self._cache.record(path, etag=etag, status=200, body=response.body)

    def _finish_graphql(self, request: GraphQLRequest, response: Response) -> Outcome:
        if not 200 <= response.status < 300:
            return response  # the status carries the failure; body need not be JSON
        errors = graphql_errors_from_body(response.body)
        if errors is None:
            detail = "GraphQL response was not a JSON object"
            if request.is_mutation:
                # The server answered 2xx, so the mutation may have been
                # applied: ADAPTER_DEFECT would fall back and replay it via
                # gh. SENT_NO_RESPONSE is neither retried nor fallen back.
                return TransportFailure(FailureKind.SENT_NO_RESPONSE, detail, "http")
            return self._defect(detail)
        if not errors:
            return response
        return Response(response.status, response.headers, response.body, "http", errors)

    def _defect(self, detail: str) -> TransportFailure:
        return TransportFailure(FailureKind.ADAPTER_DEFECT, detail, "http")

    # -- wire ----------------------------------------------------------------

    def _exchange(self, prepared: _Prepared, timeout: float) -> Outcome:
        outcome, reused = self._exchange_once(prepared, timeout, fresh=False)
        if (
            reused
            and prepared.idempotent
            and isinstance(outcome, TransportFailure)
            and outcome.kind is FailureKind.SENT_NO_RESPONSE
        ):
            # The peer closed a pooled socket in the instant after the
            # liveness check. Replaying a read on a fresh connection is
            # safe; a mutation is never replayed here (it may have landed).
            outcome, _ = self._exchange_once(prepared, timeout, fresh=True)
        return outcome

    def _exchange_once(
        self, prepared: _Prepared, timeout: float, *, fresh: bool
    ) -> tuple[Outcome, bool]:
        conn, reused = self._checkout(timeout, fresh=fresh)
        try:
            if getattr(conn, "sock", None) is None:
                conn.connect()
        except TimeoutError as exc:
            self._close(conn)
            return TransportFailure(
                FailureKind.CONNECT, f"connect timed out: {exc}", "http"
            ), reused
        except (OSError, ssl.SSLError, HTTPException) as exc:
            self._close(conn)
            return TransportFailure(FailureKind.CONNECT, str(exc), "http"), reused
        try:
            conn.request(
                prepared.method, prepared.path, body=prepared.body, headers=prepared.headers
            )
            raw = conn.getresponse()
            body_bytes = raw.read()
            status = raw.status
            headers = normalize_headers(raw.getheaders())
            will_close = bool(getattr(raw, "will_close", False))
        except (TimeoutError, socket.timeout) as exc:
            self._close(conn)
            return TransportFailure(FailureKind.TIMEOUT, f"read timed out: {exc}", "http"), reused
        except (OSError, ssl.SSLError, HTTPException) as exc:
            self._close(conn)
            return TransportFailure(FailureKind.SENT_NO_RESPONSE, str(exc), "http"), reused
        if will_close:
            self._close(conn)
        else:
            self._checkin(conn)
        text = body_bytes.decode("utf-8", errors="replace")
        return Response(status, headers, text, "http"), reused

    def _follow_redirect(self, redirect: Response, timeout: float) -> Outcome:
        """Fetch a signed-URL redirect target (job logs) without credentials."""
        location = redirect.header("location")
        if not location:
            return redirect
        target = urlsplit(location)
        if target.scheme != "https" or not target.hostname:
            # The query of a signed URL carries a credential: never record it.
            shown = f"{target.scheme}://{target.netloc}{target.path}"
            return self._defect(f"refusing redirect to non-https location: {shown!r}")
        path = target.path or "/"
        if target.query:
            path = f"{path}?{target.query}"
        conn = self._make(target.netloc, timeout)
        try:
            conn.request("GET", path, headers={"User-Agent": _USER_AGENT})
            raw = conn.getresponse()
            body_bytes = raw.read()
            status = raw.status
            headers = normalize_headers(raw.getheaders())
        except (TimeoutError, socket.timeout) as exc:
            return TransportFailure(FailureKind.TIMEOUT, f"redirect read timed out: {exc}", "http")
        except (OSError, ssl.SSLError, HTTPException) as exc:
            return TransportFailure(FailureKind.SENT_NO_RESPONSE, f"redirect fetch: {exc}", "http")
        finally:
            self._close(conn)
        return Response(status, headers, body_bytes.decode("utf-8", errors="replace"), "http")
