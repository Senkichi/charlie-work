"""Server hardening: exclusive bind, timeouts, headers on every path, watchdog, HEAD, 500."""

from __future__ import annotations

import http.client
import http.server
import logging
import socket
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work.dashboard import now_needs_me, serve, server
from charlie_work.dashboard.config import DashboardConfig
from charlie_work.dashboard.now_types import SourcesRead
from charlie_work.dashboard.read_model import (
    ModelState,
    ReadModel,
    collector_stall,
    refresh_model,
    start_workers,
)
from charlie_work.dashboard.serve import serve_dashboard, watch_head_drift
from charlie_work.dashboard.server import CSP, DashboardSources, ServerError, make_server

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _collect(now: datetime):
    return SourcesRead(repos=()), ()


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def served(clock: _Clock) -> Iterator[Any]:
    cfg = DashboardConfig(port=0, collector_interval_seconds=10)
    srv = make_server(cfg, DashboardSources(_collect), clock=clock)
    assert not isinstance(srv, ServerError)
    refresh_model(srv.holder, _collect, clock)
    t = threading.Thread(target=srv.httpd.serve_forever, kwargs={"poll_interval": 0.05})
    t.daemon = True
    t.start()
    yield srv
    srv.httpd.shutdown()
    srv.httpd.server_close()


def _raw(port: int, payload: bytes, timeout: float = 5.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(payload)
        chunks = []
        try:
            while data := s.recv(65536):
                chunks.append(data)
        except (TimeoutError, ConnectionError):
            pass
        return b"".join(chunks)


def _host(srv: Any) -> bytes:
    return f"Host: 127.0.0.1:{srv.port}\r\n".encode()


# (1) exclusive bind -------------------------------------------------------------------
def test_second_dashboard_on_same_port_is_a_server_error(served) -> None:
    again = make_server(DashboardConfig(port=served.port), DashboardSources(_collect))
    assert isinstance(again, ServerError) and "cannot bind" in again.message


def test_listening_stdlib_server_blocks_bind() -> None:
    other = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    try:
        port = other.server_address[1]
        res = make_server(DashboardConfig(port=port), DashboardSources(_collect))
        assert isinstance(res, ServerError)
    finally:
        other.server_close()


def test_rebind_after_close_still_works(served) -> None:
    port = served.port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/healthz")
    conn.getresponse().read()
    conn.close()
    served.httpd.shutdown()
    served.httpd.server_close()
    again = make_server(DashboardConfig(port=port), DashboardSources(_collect))
    assert not isinstance(again, ServerError)
    again.httpd.server_close()


@pytest.mark.skipif(sys.platform != "win32", reason="SO_EXCLUSIVEADDRUSE is Windows-only")
def test_exclusive_addr_use_set_on_windows(served) -> None:
    value = served.httpd.socket.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE)
    assert value


# (2) timeouts / Host before body ----------------------------------------------------
def test_idle_connection_is_dropped(served, monkeypatch) -> None:
    monkeypatch.setattr(server._Handler, "timeout", 0.3)
    start = time.monotonic()
    out = _raw(served.port, b"GET / HTTP/1.1\r\nHost: x", timeout=5)
    assert out == b"" and time.monotonic() - start < 3


def test_bad_host_post_answers_without_reading_body(served) -> None:
    # Content-Length promises 100 bytes, only 3 arrive: reading first would hang.
    payload = b"POST / HTTP/1.1\r\nHost: evil.com\r\nContent-Length: 100\r\n\r\nabc"
    start = time.monotonic()
    out = _raw(served.port, payload, timeout=3)
    assert b" 421 " in out.split(b"\r\n", 1)[0]
    assert time.monotonic() - start < 2.5


def test_oversized_body_is_not_drained(served) -> None:
    payload = b"POST / HTTP/1.1\r\n" + _host(served) + b"Content-Length: 99999999\r\n\r\nabc"
    out = _raw(served.port, payload, timeout=3)
    assert b" 405 " in out.split(b"\r\n", 1)[0]


# (3) headers on every response ------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"FOO / HTTP/1.1\r\nHost: evil.com\r\n\r\n", b" 501 "),
        (b"TRACE / HTTP/1.1\r\n{host}\r\n", b" 501 "),
        (b"GARBAGE\r\n\r\n", b" 400 "),
        (b"GET / HTTP/9.9\r\n{host}\r\n", b" 505 "),
        (b"POST / HTTP/1.1\r\n{host}Content-Length: 0\r\n\r\n", b" 405 "),
        (b"GET /nope HTTP/1.1\r\n{host}\r\n", b" 404 "),
    ],
)
def test_every_response_carries_security_headers(served, payload: bytes, status: bytes) -> None:
    payload = payload.replace(b"{host}", _host(served))
    head = _raw(served.port, payload).split(b"\r\n\r\n", 1)[0]
    assert status in head.split(b"\r\n", 1)[0]
    lowered = head.lower()
    assert f"content-security-policy: {CSP}".lower().encode() in lowered
    assert b"x-content-type-options: nosniff" in lowered
    assert b"cache-control: no-store" in lowered
    assert lowered.count(b"content-security-policy") == 1


# (4) watchdog -----------------------------------------------------------------------
def test_collector_stall_logic() -> None:
    fresh = ModelState(last_attempt_at=NOW)
    assert collector_stall(fresh, NOW + timedelta(seconds=29), 10) is None
    assert "stalled" in (collector_stall(fresh, NOW + timedelta(seconds=31), 10) or "")
    assert collector_stall(ModelState(), NOW, 10) is None  # not started yet
    dead = ModelState(collector_dead="Boom: x", last_attempt_at=NOW)
    assert "died" in (collector_stall(dead, NOW, 10) or "")


def test_hung_collector_is_visible_on_healthz_and_banner(served, clock: _Clock) -> None:
    clock.now = NOW + timedelta(seconds=45)  # > 3 x 10s since the last tick started
    conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    conn.request("GET", "/healthz")
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 503 and b'"ok": false' in body and b"stalled" in body
    conn.request("GET", "/now/fragment")
    frag = conn.getresponse().read()
    conn.close()
    assert b"collector is stalled" in frag


def test_healthy_collector_healthz_ok(served) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    conn.request("GET", "/healthz")
    resp = conn.getresponse()
    assert resp.status == 200 and b'"ok": true' in resp.read()
    conn.close()


class _Boom(BaseException):
    pass


def test_baseexception_kills_thread_but_is_surfaced(clock: _Clock) -> None:
    holder, stop = ReadModel(), threading.Event()

    def collect(now: datetime):
        raise _Boom("fatal")

    (t, *_) = start_workers(
        holder,
        stop,
        collect=collect,
        rollup=None,
        collector_interval=1,
        rollup_interval=1,
        clock=clock,
    )
    t.join(timeout=5)
    stop.set()
    assert not t.is_alive()
    assert "_Boom: fatal" in (holder.get().collector_dead or "")
    assert "died" in (collector_stall(holder.get(), clock.now, 1) or "")


def test_exception_per_tick_keeps_model_and_thread(clock: _Clock) -> None:
    holder = ReadModel()
    refresh_model(holder, _collect, clock)
    good = holder.get().model

    def bad(now: datetime):
        raise RuntimeError("x")

    refresh_model(holder, bad, clock)
    assert holder.get().model is good and holder.get().collector_error


# (5) render error -> 500 --------------------------------------------------------------
def test_render_error_is_a_500_with_escaped_page(served, monkeypatch) -> None:
    def boom(*a: Any, **k: Any) -> str:
        raise KeyError("<script>alert(1)</script>")

    monkeypatch.setattr(server, "render_fragment", boom)
    out = _raw(served.port, b"GET /now/fragment HTTP/1.1\r\n" + _host(served) + b"\r\n")
    head, _, body = out.partition(b"\r\n\r\n")
    assert b" 500 " in head.split(b"\r\n", 1)[0]
    assert CSP.encode() in head
    assert b"<script>" not in body and b"&lt;script&gt;" in body


# (6) aborted-request noise ------------------------------------------------------------
def test_aborted_connection_logs_one_line_no_traceback(served, caplog, capsys) -> None:
    with caplog.at_level(logging.INFO, logger="charlie_work.dashboard"):
        try:
            raise ConnectionResetError(10054, "reset")
        except ConnectionResetError:
            served.httpd.handle_error(None, ("127.0.0.1", 1234))
    assert capsys.readouterr().err == ""
    lines = [r for r in caplog.records if "aborted" in r.getMessage()]
    assert len(lines) == 1 and lines[0].exc_info is None


def test_other_errors_still_reach_stdlib_handler(served, capsys) -> None:
    try:
        raise ValueError("real bug")
    except ValueError:
        served.httpd.handle_error(None, ("127.0.0.1", 1234))
    assert "real bug" in capsys.readouterr().err


# (7) HEAD ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/", "/now/fragment", "/healthz", "/api/now.json"])
def test_head_matches_get_without_body(served, path: str) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    conn.request("HEAD", path)
    resp = conn.getresponse()
    assert resp.status == 200 and resp.read() == b""
    assert int(resp.getheader("Content-Length")) > 0
    assert resp.getheader("Content-Security-Policy") == CSP
    conn.close()


def test_head_on_bad_host_is_421(served) -> None:
    out = _raw(served.port, b"HEAD / HTTP/1.1\r\nHost: evil.com\r\n\r\n")
    assert b" 421 " in out.split(b"\r\n", 1)[0]


# (8) HEAD-drift baseline --------------------------------------------------------------
def test_drift_watcher_uses_startup_baseline() -> None:
    stop, drifted = threading.Event(), threading.Event()
    # The first read the watcher sees is already the NEW head: without a startup baseline
    # it would adopt it and never restart.
    watch_head_drift(lambda: "new", stop, drifted, 0.01, baseline="old")
    assert drifted.is_set() and stop.is_set()


def test_serve_reads_baseline_before_sources_and_bind(tmp_path, monkeypatch) -> None:
    order: list[str] = []

    def sources(*a: Any, **k: Any) -> DashboardSources:
        order.append("sources")
        return DashboardSources(_collect)

    def bind(*a: Any, **k: Any) -> ServerError:
        order.append("bind")
        return ServerError("stop")

    monkeypatch.setattr(serve, "default_sources", sources)
    monkeypatch.setattr(serve, "make_server", bind)

    def head() -> str:
        order.append("head")
        return "a"

    result = serve_dashboard(DashboardConfig(port=0), str(tmp_path), read_head=head)
    assert not result.ok and order == ["head", "sources", "bind"]


# (9) malformed issue number -----------------------------------------------------------
def _needs(issue: dict[str, Any]):
    labels = SimpleNamespace(operator_queue="op", human_needed="hn")
    repo = SimpleNamespace(
        key="o/r",
        repo_root="/r",
        snapshot=SimpleNamespace(data={"issues": [issue], "prs": []}),
        escalated_since=(),
    )
    src = SimpleNamespace(labels=labels, repos=(repo,), pause=None, supervisor_heartbeat=None)
    return now_needs_me.needs_me_items(src, (), (), NOW, 100.0, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["9; calc", None, 0, -3, True, 2.5])
def test_malformed_issue_number_never_renders_as_hash_zero(bad: Any) -> None:
    items = _needs({"number": bad, "title": "t", "labels": ["op"]})
    assert items, "the unreadable row must stay visible"
    for item in items:
        assert "#0" not in item.reason
        assert item.command is None and item.number is None


def test_valid_issue_number_still_gets_a_command() -> None:
    (item,) = _needs({"number": 7, "title": "t", "labels": ["op"]})
    assert item.number == 7 and "--issue 7" in (item.command or "")
