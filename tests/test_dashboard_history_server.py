"""/history over real HTTP: served from the per-(tab, range) cache, never per request."""

from __future__ import annotations

import http.client
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _dashboard_metrics_fixtures import NOW, build_base, build_quality
from _dashboard_rollup_fixtures import Fleet

from charlie_work.dashboard import rollup
from charlie_work.dashboard.config import DashboardConfig
from charlie_work.dashboard.now_types import SourcesRead
from charlie_work.dashboard.server import CSP, DashboardSources, ServerError, make_server


def _collect(now):
    return SourcesRead(repos=()), ()


def _serve(history_db: Path | None) -> Any:
    server = make_server(
        DashboardConfig(port=0),
        DashboardSources(_collect, history_db=history_db),
        clock=lambda: NOW,
    )
    assert not isinstance(server, ServerError)
    threading.Thread(
        target=server.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    return server


def _get(server, path: str) -> tuple[int, str, http.client.HTTPResponse]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, body, resp


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    build_base(f)
    build_quality(f)
    assert rollup.run_rollup(f.sources(), NOW).errors == ()
    f.close()
    server = _serve(f.sources().db_path)
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


def test_history_is_served_from_the_cache(served) -> None:
    status, body, resp = _get(served, "/history?tab=quality&range=7d")
    assert status == 200 and resp.getheader("Content-Security-Policy") == CSP
    assert '<a id="tab-quality" href="/history?tab=quality&amp;range=7d" aria-current' in body
    assert 'class="takeaway"' in body
    assert served.httpd.app.history.misses == 1
    _get(served, "/history?range=7d&tab=quality")
    _get(served, "/history?tab=QUALITY&range=7d")
    assert served.httpd.app.history.misses == 1  # same (tab, range): no second query
    _get(served, "/history")  # defaults: flow, 7d
    assert served.httpd.app.history.misses == 2


def test_history_unknown_params_fall_back_and_subpaths_404(served) -> None:
    status, body, _ = _get(served, "/history?tab=<script>&range=1y")
    assert status == 200 and 'id="tab-flow" href="/history?tab=flow&amp;range=7d" aria' in body
    assert "<script>&" not in body
    assert _get(served, "/history/flow")[0] == 404


def test_history_without_a_rollup_db_says_so() -> None:
    server = _serve(None)
    try:
        status, body, _ = _get(server, "/history?tab=flow")
        assert status == 200 and "Rollup not available" in body and "<svg" not in body
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()


def test_now_header_links_to_history(served) -> None:
    _, body, _ = _get(served, "/now")
    assert '<a href="/history" data-go="h">History</a>' in body
