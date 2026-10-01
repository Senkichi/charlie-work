# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Drill-down routes over real HTTP on synthetic fleets (rolled-up dashboard.db, events.db)."""

from __future__ import annotations

import http.client
import re
import threading
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from _dashboard_page_fixtures import _parse
from _dashboard_rollup_fixtures import ALPHA, BETA, NOW, fleet  # noqa: F401

from charlie_work import instrumentation
from charlie_work.config import LabelConfig
from charlie_work.dashboard import rollup, sources
from charlie_work.dashboard.config import DashboardConfig
from charlie_work.dashboard.now_types import RepoRead, SourcesRead
from charlie_work.dashboard.pages import routes
from charlie_work.dashboard.read_model import refresh_model
from charlie_work.dashboard.server import CSP, DashboardSources, ServerError, make_server
from charlie_work.dashboard.server_drill import HANDLERS
from charlie_work.dashboard.sources import SnapshotRead

CID = "ed62ee91e2e5"
L = LabelConfig()
HOSTILE = "<script>alert(1)</script>"


def _snapshot(issues: list[dict]) -> SnapshotRead:
    return SnapshotRead(
        NOW - timedelta(seconds=20), 20.0, {"issues": issues, "workers": [{}]}, None
    )


def _collect(now):
    alpha = [
        {
            "number": 2199,
            "title": f"t {HOSTILE}",
            "labels": [L.in_progress],
            "dispatchable": False,
        },
        {
            "number": 2226,
            "title": "queued one",
            "labels": [L.queued, L.ready],
            "dispatchable": True,
        },
    ]
    beta = [{"number": 7, "title": "beta q", "labels": [L.queued]}]
    repos = (
        RepoRead(ALPHA, "C:/a", _snapshot(alpha), reviewers_live=1, worker_cap=2),
        RepoRead(BETA, "C:/b", _snapshot(beta)),
    )
    return SourcesRead(repos=repos, global_worker_cap=4, done_24h=1), ()


@pytest.fixture
def served(fleet) -> Iterator[Any]:
    for i, (kind, payload) in enumerate(
        [
            ("loop_started", {"pass": 1}),
            ("merge_failed", {"error": HOSTILE, f"k{HOSTILE}": "x\x1b[31m"}),
        ]
    ):
        fleet.monkeypatch.setattr(
            instrumentation, "_now_iso", lambda i=i: f"2026-10-01T08:00:0{i}Z"
        )
        instrumentation.log_event(fleet.alpha / "state.json", kind, payload, correlation_id=CID)
    for ts, kind, p in [  # issue 900: dispatched, PR opened, review claimed (still reviewing)
        ("2026-10-01T09:00:00Z", "dispatch", {"issue_numbers": [900]}),
        (
            "2026-10-01T09:40:00Z",
            "worker_handoff_pr_opened",
            {"issue_number": 900, "pr_number": 901},
        ),
        ("2026-10-01T10:00:00Z", "review_dispatch_claim", {"pr_numbers": [901]}),
    ]:
        fleet.emit(fleet.alpha, ts, kind, p)
    assert rollup.run_rollup(fleet.sources(), NOW).errors == ()
    fleet.close()
    fleet_dir = str(fleet.dir)
    server = make_server(
        DashboardConfig(port=0),
        DashboardSources(
            _collect,
            history_db=fleet.sources().db_path,
            repos=lambda: sources.enumerate_repos(fleet_dir),
        ),
        clock=lambda: NOW,
    )
    assert not isinstance(server, ServerError)
    refresh_model(server.holder, _collect, lambda: NOW)
    threading.Thread(
        target=server.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


def _get(server, path: str) -> tuple[int, str, http.client.HTTPResponse]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, body, resp


OK_PATHS = (
    f"/repo/{ALPHA}",
    f"/repo/{BETA}",  # the no-remote lane: /repo/local/<name>
    f"/repo/{ALPHA}?view=runners",
    f"/issue/{ALPHA}/2199",
    f"/pr/{ALPHA}/2208",
    f"/pass/{ALPHA}/{CID}",
    "/flow/queued",
    "/flow/in-progress",
    f"/flow/queued?repo={BETA}",
    "/flow/dispatchable",
    "/flow/needs-rework",
    "/flow/active",
    "/flow/done-24h",
)


@pytest.mark.parametrize("path", OK_PATHS)
def test_each_drill_route_renders_in_house_style(served, path: str) -> None:
    status, body, resp = _get(served, path)
    assert status == 200, body[:400]
    assert resp.getheader("Content-Type") == "text/html; charset=utf-8"
    assert resp.getheader("Content-Security-Policy") == CSP
    assert body.startswith("<!doctype html>") and body.count("<h1") == 1
    assert '<nav class="crumbs"' in body and '<a class="crumb" href="/now">Now</a>' in body
    assert "as of <b><time" in body and "(local)" in body  # same header as Now
    assert 'class="fresh"' in body and 'id="theme-toggle"' in body
    assert '<link rel="stylesheet" href="/static/drill.css">' in body
    parsed = _parse(body)  # the parser, not a regex: a regex misses `</script >`
    assert all(a.get("src", "").startswith("/static/") for t, a in parsed.tags if t == "script")
    assert all(b == "" for b in parsed.script_bodies)  # no inline script (CSP)
    assert " style=" not in body and "<style" not in body
    assert HOSTILE not in body


NOT_FOUND = (
    "/repo/owner/other",
    "/repo/owner",
    "/repo/owner/alpha/extra",
    "/repo/%2E%2E/x",
    "/repo/owner%2Falpha/x",
    "/issue/owner/alpha/abc",
    "/issue/owner/alpha/0",
    "/issue/owner/alpha/99999",  # valid but never seen
    "/pr/owner/alpha/-3",
    "/pass/owner/alpha/bad%20id",
    "/pass/owner/alpha/deadbeef0000",
    "/pass/owner/other/" + CID,
    "/flow/nope",
    "/flow/queued?repo=..%2Fx",
    "/flow/queued?repo=owner/other",
    "/issue",
    "/repo/",
)


@pytest.mark.parametrize("path", NOT_FOUND)
def test_bad_or_unknown_input_is_a_house_style_404(served, path: str) -> None:
    status, body, resp = _get(served, path)
    assert status == 404, path
    assert resp.getheader("Content-Type") == "text/html; charset=utf-8"
    assert 'class="dmissing"' in body and '<a class="crumb" href="/now">' in body


def test_pass_page_is_ordered_dense_and_escapes_hostile_payload(served) -> None:
    _, body, _ = _get(served, f"/pass/{ALPHA}/{CID}")
    assert body.index("loop_started") < body.index("merge_failed")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body and HOSTILE not in body
    assert "\x1b" not in body  # control characters stripped by the model
    assert '<span aria-hidden="true">✕</span> error' in body  # glyph + word, not colour
    local = NOW.replace(hour=8).astimezone().strftime("%H:%M:%S")
    assert f'<time datetime="2026-10-01T08:00:00Z">{local}</time>' in body
    assert f'href="/repo/{ALPHA}"' in body  # breadcrumb back through the repo


def test_repo_page_links_passes_and_marks_the_section_now_linked(served) -> None:
    _, body, _ = _get(served, f"/repo/{ALPHA}?view=runners")
    assert f'href="/pass/{ALPHA}/{CID}"' in body
    assert 'class="dsec focus" id="cap"' in body
    assert f'href="/flow/queued?repo={ALPHA.replace("/", "%2F")}"' in body
    assert 'id="needs"' in body and 'id="stages"' in body


def test_issue_page_draws_stage_lanes_in_local_time(served) -> None:
    _, body, _ = _get(served, f"/pr/{ALPHA}/2208")
    assert 'href="/issue/owner/alpha/2199"' in body  # the PR's issue
    assert "&lt;script&gt;" in body and "Merged" in body  # snapshot title escaped; timeline
    _, issue, _ = _get(served, f"/issue/{ALPHA}/900")
    assert '<svg class="lane-chart"' in issue
    # in progress 09:00 -> 09:40, PR open 09:40 -> 10:00, reviewing 10:00 -> now (12:00, open)
    assert "Longest in Reviewing: 2h00m (67% of staged time), approx." in issue
    assert "40m00s · 1 visit · approx." in issue and "2h00m · 1 visit · open · approx." in issue
    assert 'stroke-dasharray="4 4"' in issue  # approx. bands are dashed, not colour alone
    start = NOW.replace(hour=9).astimezone().strftime("%Y-%m-%d %H:%M")
    assert f'<time datetime="2026-10-01T09:00:00Z">{start}</time>' in issue  # local window
    assert 'href="/pr/owner/alpha/901"' in issue
    assert "Needs rework</text>" in issue and "not visited" in issue


def test_flow_page_lists_the_issues_now_counts(served) -> None:
    _, body, _ = _get(served, "/flow/queued")
    assert "Now shows <b>2</b> across the fleet" in body
    assert 'href="/issue/owner/alpha/2226"' in body and 'href="/issue/local/beta/7"' in body
    _, one, _ = _get(served, f"/flow/queued?repo={BETA}")
    assert 'href="/issue/local/beta/7"' in one and "/issue/owner/alpha/2226" not in one
    _, done, _ = _get(served, "/flow/done-24h")
    assert "Now shows <b>1</b> merged in the last 24h" in done
    assert 'href="/issue/owner/alpha/2199"' in done and 'class="dnote"' not in done


def test_drill_routes_say_unavailable_before_the_first_model(fleet) -> None:
    server = make_server(DashboardConfig(port=0), DashboardSources(_collect), clock=lambda: NOW)
    assert not isinstance(server, ServerError)
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
    try:
        status, body, _ = _get(server, f"/repo/{ALPHA}")
        assert status == 503 and "Not available" in body and "collecting" in body
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()


def test_route_registry_equals_the_server_handlers() -> None:
    assert set(HANDLERS) | {"/now", "/history"} == routes.ROUTES


def test_api_json_does_not_carry_the_raw_sources(served) -> None:
    _, body, _ = _get(served, "/api/now.json")
    assert '"sources"' not in body and HOSTILE not in body


def test_drill_css_uses_only_tokens_and_is_served(served) -> None:
    from charlie_work.dashboard.theme import generate_css, static_asset

    css = static_asset("drill.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[\w-]+)\s*:", generate_css()))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    assert used and used <= defined, sorted(used - defined)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|transparent", css)
    assert not re.search(r"^\s*--[\w-]+\s*:", css, re.M)  # no new custom properties
    status, _, resp = _get(served, "/static/drill.css")
    assert status == 200 and resp.getheader("Content-Type") == "text/css; charset=utf-8"
