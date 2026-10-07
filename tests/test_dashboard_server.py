"""Real-HTTP tests for the dashboard server (ephemeral port, fake collector)."""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from charlie_work.dashboard.config import DashboardConfig
from charlie_work.dashboard.now_types import SourcesRead
from charlie_work.dashboard.read_model import ReadModel, refresh_model
from charlie_work.dashboard.server import (
    CSP,
    DashboardSources,
    ServerError,
    make_server,
    run_server,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


class Fake:
    def __init__(self) -> None:
        self.fail = False
        self.calls = 0

    def collect(self, now: datetime):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom <script>")
        return SourcesRead(repos=()), ()


@pytest.fixture
def fake() -> Fake:
    return Fake()


@pytest.fixture
def served(fake: Fake) -> Iterator[Any]:
    config = DashboardConfig(port=0, poll_interval_seconds=7)
    server = make_server(config, DashboardSources(fake.collect), clock=lambda: NOW)
    assert not isinstance(server, ServerError)
    refresh_model(server.holder, fake.collect, lambda: NOW)
    stop = threading.Event()
    # Serve only (collector thread is exercised via refresh_model above and run_server below).
    t = threading.Thread(target=server.httpd.serve_forever, kwargs={"poll_interval": 0.05})
    t.daemon = True
    t.start()
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()
    stop.set()


def _get(server, path: str, host: str | None = None, method: str = "GET"):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.putrequest(method, path, skip_host=host is not None)
    if host is not None:
        conn.putheader("Host", host)
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


@pytest.mark.parametrize(
    ("path", "ctype"),
    [
        ("/", "text/html"),
        ("/now", "text/html"),
        ("/now/fragment", "text/html"),
        ("/api/now.json", "application/json"),
        ("/healthz", "application/json"),
        ("/static/dashboard.css", "text/css"),
        ("/static/htmx.min.js", "text/javascript"),
        ("/static/base.css", "text/css"),
    ],
)
def test_routes_status_and_content_type(served, path: str, ctype: str) -> None:
    resp, body = _get(served, path)
    assert resp.status == 200 and resp.getheader("Content-Type", "").startswith(ctype)
    assert body
    assert resp.getheader("Content-Security-Policy") == CSP
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Referrer-Policy") == "no-referrer"


def test_page_links_assets_and_fragment_has_as_of(served) -> None:
    _, page = _get(served, "/")
    text = page.decode()
    assert "/static/dashboard.css" in text and "/static/htmx.min.js" in text
    assert 'hx-get="/now/fragment"' in text and 'hx-trigger="every 7s"' in text
    _, frag = _get(served, "/now/fragment")
    assert b"as of " in frag and b"(local)" in frag and not frag.startswith(b"<!doctype")


def test_api_and_healthz_payloads(served) -> None:
    _, body = _get(served, "/api/now.json")
    assert json.loads(body)["model"]["generated_at"] == NOW.isoformat()
    _, body = _get(served, "/healthz")
    assert json.loads(body) == {"ok": True, "model_age_seconds": 0.0, "reason": None}


@pytest.mark.parametrize(
    "path",
    [
        "/static/../pyproject.toml",
        "/static/%2e%2e/pyproject.toml",
        "/static/%2e%2e%2fpyproject.toml",
        "/static/D:x",
        "/static/..\\pyproject.toml",
        "/static/fonts/../../theme.py",
        "/static//etc/passwd",
        "/static/nope.css",
        "/static/",
        "/static/fonts",
        "/nope",
    ],
)
def test_static_allowlist_and_unknown_paths_404(served, path: str) -> None:
    resp, _ = _get(served, path)
    assert resp.status == 404


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1:1", "localhost", "127.0.0.1"])
def test_bad_host_header_421(served, host: str) -> None:
    resp, _ = _get(served, "/healthz", host=host)
    assert resp.status == 421


def test_localhost_host_header_ok(served) -> None:
    resp, _ = _get(served, "/healthz", host=f"localhost:{served.port}")
    assert resp.status == 200


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_other_methods_405(served, method: str) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    conn.request(method, "/", body=b"x")
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 405 and resp.getheader("Allow") == "GET, HEAD"
    conn.close()


def test_collector_failure_keeps_previous_model_and_surfaces_error(served, fake: Fake) -> None:
    fake.fail = True
    refresh_model(served.holder, fake.collect, lambda: NOW)
    state = served.holder.get()
    assert state.model is not None and state.collector_failing_since == NOW
    _, frag = _get(served, "/now/fragment")
    assert b"collector failing since" in frag and b"as of " in frag
    assert b"<script>" not in frag and b"&lt;script&gt;" in frag  # escaped
    fake.fail = False
    refresh_model(served.holder, fake.collect, lambda: NOW)
    assert served.holder.get().collector_error is None


def test_failure_before_first_model_is_not_blank() -> None:
    holder = ReadModel()
    fake = Fake()
    fake.fail = True
    refresh_model(holder, fake.collect, lambda: NOW)
    assert holder.get().model is None and holder.get().collector_error


def test_non_loopback_host_is_error_value() -> None:
    result = make_server(DashboardConfig(host="0.0.0.0", port=0), DashboardSources(Fake().collect))
    assert isinstance(result, ServerError) and "loopback" in result.message


def test_run_server_collects_serves_and_stops(fake: Fake) -> None:
    config = DashboardConfig(port=0, collector_interval_seconds=1)
    server = make_server(config, DashboardSources(fake.collect), clock=lambda: NOW)
    assert not isinstance(server, ServerError)
    stop = threading.Event()
    runner = threading.Thread(target=run_server, args=(server, stop), daemon=True)
    runner.start()
    deadline = time.monotonic() + 5
    while server.holder.get().model is None and time.monotonic() < deadline:
        time.sleep(0.02)
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=5) as r:
        assert r.status == 200
    stop.set()
    runner.join(timeout=5)
    assert not runner.is_alive() and fake.calls >= 1
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=1)


@pytest.mark.parametrize("name", ["dashboard.js", "theme-init.js"])
def test_scripts_served_as_javascript_and_page_references_them(served, name: str) -> None:
    resp, body = _get(served, f"/static/{name}")
    assert resp.status == 200
    assert resp.getheader("Content-Type", "").startswith("text/javascript")  # RFC 9239
    assert body
    _, page = _get(served, "/")
    assert f'src="/static/{name}"' in page.decode()


def test_theme_init_is_synchronous_in_head_and_dashboard_js_deferred(served) -> None:
    _, page = _get(served, "/")
    text = page.decode()
    assert '<script src="/static/theme-init.js"></script>' in text
    assert text.index("theme-init.js") < text.index("</head>")
    assert '<script src="/static/dashboard.js" defer></script>' in text
    assert 'id="theme-toggle"' in text and 'id="keyhelp"' in text


def test_route_registry_matches_what_the_server_serves(served) -> None:
    """Registered routes answer; a known view that is not registered really is a 404."""
    from charlie_work.dashboard.pages import routes

    for prefix in sorted(routes.ROUTES):
        resp, body = _get(served, prefix)
        # A view answers 200; a bare drill-down prefix is handled by its page handler
        # (a house-style 404 naming the expected shape), never the plain-text fallthrough.
        assert resp.getheader("Content-Type", "").startswith("text/html"), prefix
        if prefix in ("/now", "/history"):
            assert resp.status == 200, prefix
        else:
            assert resp.status == 404 and b'class="dmissing"' in body, prefix
    unbuilt = [v.href for v in routes.VIEWS if not routes.is_routed(v.href)]
    for href in unbuilt:
        resp, _ = _get(served, href)
        assert resp.status == 404, href
