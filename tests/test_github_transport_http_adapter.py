"""HttpAdapter against a scripted connection: no network (ADR-0006).

The connection is injected via ``connection_factory``; the autouse conftest
guard makes a forgotten injection fail instead of dialling api.github.com.
"""

from __future__ import annotations

import json
import socket
import ssl
from dataclasses import dataclass

import pytest
from _fake_transport import FakeConn, FakeRaw

from charlie_work.github_transport import (
    CliCommand,
    CliRequest,
    FailureKind,
    GraphQLRequest,
    HttpAdapter,
    Response,
    RestRequest,
    TransportFailure,
)
from charlie_work.github_transport import http_adapter as http_adapter_module

GET = RestRequest.of("GET", "repos/o/r/pulls/1", query={"state": "all"})
POST = RestRequest.of("POST", "repos/o/r/issues", body={"title": "t"})
QUERY = GraphQLRequest.of("query Q($n: Int!) { a }", {"n": 1})


def _adapter(conn: FakeConn, **kwargs) -> HttpAdapter:
    return HttpAdapter(connection_factory=lambda host, timeout: conn, **kwargs)


def _send(adapter: HttpAdapter, request, token: str | None = "tok"):
    return adapter.send(request, token=token, timeout=5.0)


def test_rest_get_builds_an_authenticated_request_and_returns_a_response() -> None:
    conn = FakeConn([FakeRaw(200, {"X-RateLimit-Remaining": "9"}, b'{"n": 1}')])
    out = _send(_adapter(conn), GET)
    assert isinstance(out, Response) and out.ok and out.adapter == "http"
    assert out.json() == {"n": 1}
    assert out.header("x-ratelimit-remaining") == "9"  # names are lower-cased
    method, path, body, headers = conn.requests[0]
    assert (method, path, body) == ("GET", "/repos/o/r/pulls/1?state=all", None)
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Accept"] == "application/vnd.github+json"


def test_a_non_2xx_status_is_a_response_not_a_failure() -> None:
    out = _send(_adapter(FakeConn([FakeRaw(404, body=b'{"message": "Not Found"}')])), GET)
    assert isinstance(out, Response) and out.status == 404 and not out.ok


def test_mutation_body_is_sent_as_json() -> None:
    conn = FakeConn([FakeRaw(201, body=b"{}")])
    _send(_adapter(conn), POST)
    method, _, body, headers = conn.requests[0]
    assert method == "POST" and json.loads(body) == {"title": "t"}
    assert headers["Content-Type"] == "application/json"


def test_graphql_posts_to_the_graphql_endpoint() -> None:
    conn = FakeConn([FakeRaw(200, body=b'{"data": {"a": 1}}')])
    out = _send(_adapter(conn), QUERY)
    assert isinstance(out, Response) and out.ok
    method, path, body, _ = conn.requests[0]
    assert (method, path) == ("POST", "/graphql")
    assert json.loads(body) == {"query": QUERY.document, "variables": {"n": 1}}


def test_graphql_errors_inside_a_200_are_parsed_and_make_the_response_not_ok() -> None:
    payload = {
        "data": None,
        "errors": [
            {"message": "nope", "type": "NOT_FOUND", "path": ["repository", "pullRequest"]},
            {"message": "slow", "extensions": {"code": "RATE_LIMITED"}},
        ],
    }
    out = _send(_adapter(FakeConn([FakeRaw(200, body=json.dumps(payload).encode())])), QUERY)
    assert isinstance(out, Response)
    assert out.status == 200 and not out.ok
    assert [(e.message, e.type) for e in out.graphql_errors] == [
        ("nope", "NOT_FOUND"),
        ("slow", "RATE_LIMITED"),
    ]
    assert out.graphql_errors[0].path == ("repository", "pullRequest")


def test_a_graphql_200_whose_body_is_not_an_object_is_an_adapter_defect() -> None:
    out = _send(_adapter(FakeConn([FakeRaw(200, body=b"[1]")])), QUERY)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


def test_graphql_errors_on_a_non_2xx_are_left_to_the_status() -> None:
    out = _send(_adapter(FakeConn([FakeRaw(502, body=b"<html>bad gateway</html>")])), QUERY)
    assert isinstance(out, Response) and out.status == 502


def test_no_token_is_token_unavailable_without_touching_the_network() -> None:
    conn = FakeConn([])
    out = _send(_adapter(conn), GET, token=None)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.TOKEN_UNAVAILABLE
    assert conn.requests == []


def test_cli_requests_are_an_adapter_defect() -> None:
    out = _send(_adapter(FakeConn([])), CliRequest(CliCommand.AUTH_TOKEN))
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


@pytest.mark.parametrize(
    "error", [socket.gaierror("no dns"), ConnectionRefusedError("refused"), ssl.SSLError("tls")]
)
def test_failure_to_connect_is_connect_and_the_request_is_never_written(error) -> None:
    conn = FakeConn([], connect_error=error)
    out = _send(_adapter(conn), POST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.CONNECT
    assert conn.requests == [] and conn.closed


def test_a_connect_timeout_is_connect() -> None:
    conn = FakeConn([], connect_error=TimeoutError("slow"))
    out = _send(_adapter(conn), GET)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.CONNECT


def test_a_read_timeout_after_the_request_was_sent_is_timeout() -> None:
    conn = FakeConn([TimeoutError("read")])
    out = _send(_adapter(conn), POST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.TIMEOUT
    assert len(conn.requests) == 1 and conn.closed


def test_a_reset_after_send_is_sent_no_response() -> None:
    out = _send(_adapter(FakeConn([ConnectionResetError("reset")])), POST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.SENT_NO_RESPONSE


def test_a_malformed_request_is_an_adapter_defect_not_a_raise() -> None:
    class Broken(FakeConn):
        def request(self, *a, **k):  # noqa: ANN002, ANN003
            raise TypeError("bad header value")

    out = _send(_adapter(Broken([])), GET)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


def test_keep_alive_connections_are_pooled_and_reused() -> None:
    conn = FakeConn([FakeRaw(200, body=b"{}"), FakeRaw(200, body=b"{}")])
    made: list[FakeConn] = []

    def factory(host: str, timeout: float) -> FakeConn:
        made.append(conn)
        return conn

    adapter = HttpAdapter(connection_factory=factory)
    _send(adapter, GET)
    _send(adapter, GET)
    assert len(made) == 1 and adapter.idle_connections() == 1


def test_a_connection_the_server_will_close_is_not_pooled() -> None:
    conn = FakeConn([FakeRaw(200, body=b"{}", will_close=True)])
    adapter = _adapter(conn)
    _send(adapter, GET)
    assert adapter.idle_connections() == 0 and conn.closed


class FakeCache:
    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def get(self, path: str):
        return self.entries.get(path)

    def record(self, path: str, *, etag: str, status: int, body: str) -> None:
        @dataclass
        class Entry:
            etag: str
            status: int
            body: str

        self.entries[path] = Entry(etag, status, body)


def test_etag_conditional_get_serves_a_304_from_the_cache_as_a_200() -> None:
    cache = FakeCache()
    conn = FakeConn(
        [FakeRaw(200, {"ETag": '"abc"'}, b'{"n": 1}'), FakeRaw(304, {"ETag": '"abc"'}, b"")]
    )
    adapter = _adapter(conn, cache=cache)
    first = _send(adapter, GET)
    second = _send(adapter, GET)
    assert isinstance(first, Response) and isinstance(second, Response)
    assert second.status == 200 and second.json() == {"n": 1}
    assert conn.requests[0][3].get("If-None-Match") is None
    assert conn.requests[1][3]["If-None-Match"] == '"abc"'


def test_mutations_never_use_or_fill_the_etag_cache() -> None:
    cache = FakeCache()
    conn = FakeConn([FakeRaw(201, {"ETag": '"x"'}, b"{}")])
    _send(_adapter(conn, cache=cache), POST)
    assert cache.entries == {}


def test_a_signed_redirect_is_followed_without_credentials() -> None:
    logs = RestRequest.of("GET", "repos/o/r/actions/jobs/1/logs", follow_redirect=True)
    first = FakeConn([FakeRaw(302, {"Location": "https://blob.example.net/logs?sig=1"})])
    second = FakeConn([FakeRaw(200, body=b"log text")])
    conns = iter([first, second])
    adapter = HttpAdapter(connection_factory=lambda host, timeout: next(conns))
    out = _send(adapter, logs)
    assert isinstance(out, Response) and out.body == "log text"
    assert "Authorization" not in second.requests[0][3]


def test_a_redirect_to_a_non_https_location_is_refused() -> None:
    logs = RestRequest.of("GET", "repos/o/r/actions/jobs/1/logs", follow_redirect=True)
    conn = FakeConn([FakeRaw(302, {"Location": "http://evil.example.net/x"})])
    out = _send(_adapter(conn), logs)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


def test_conftest_guard_blocks_an_uninjected_real_connection() -> None:
    with pytest.raises(AssertionError, match="real network in tests"):
        http_adapter_module._new_connection("api.github.com", 1.0)
    with pytest.raises(AssertionError, match="real network in tests"):
        HttpAdapter().send(GET, token="tok", timeout=1.0)
