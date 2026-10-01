"""Loopback-only, read-only HTTP server for the fleet dashboard (ADR-0008).

``make_server`` builds (never starts) the server and returns a ``ServerError`` value on
a config problem (non-loopback host, bind failure). ``run_server`` starts the collector
thread(s) plus the accept loop and blocks until a stop ``Event`` is set. The server never
writes fleet state and never calls GitHub: all data comes from ``ReadModel``.
"""

from __future__ import annotations

import dataclasses
import html
import ipaddress
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from typing import Any

from .config import DashboardConfig
from .pages.now import render_fragment, render_now
from .read_model import (
    Clock,
    Collect,
    ModelState,
    ReadModel,
    RollupRun,
    collector_stall,
    start_workers,
    to_plain,
)
from .server_http import (
    CSP,
    MAX_DRAIN_BYTES,
    DashboardHTTPServer,
    SecureHandler,
)
from .theme import assets

__all__ = ["CSP"]

log = logging.getLogger("charlie_work.dashboard")

_STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}


@dataclass(frozen=True)
class ServerError:
    """Returned (not raised) by ``make_server`` when the server cannot be built."""

    message: str


@dataclass(frozen=True)
class DashboardSources:
    """Injected data sources: the collector pass and the optional rollup pass."""

    collect: Collect
    rollup: RollupRun | None = None


def default_sources(fleet_dir_override: str | None, collector_interval: float) -> DashboardSources:
    """Real fleet sources (read-only collector + alarm leaves, and the rollup)."""
    from .alarm_feed import loop_pass_findings
    from .now_collect import collect_sources_read
    from .rollup import rollup_sources, run_rollup

    def collect(now: datetime):
        read = collect_sources_read(now, fleet_dir_override)
        read = dataclasses.replace(read, collector_interval_seconds=float(collector_interval))
        return read, loop_pass_findings(now, fleet_dir_override)

    def rollup(now: datetime) -> tuple[str, ...]:
        return run_rollup(rollup_sources(fleet_dir_override), now).errors

    return DashboardSources(collect, rollup)


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.version == 4 and addr.is_loopback


@dataclass(frozen=True)
class DashboardServer:
    httpd: ThreadingHTTPServer
    holder: ReadModel
    config: DashboardConfig
    sources: DashboardSources
    clock: Clock

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _App:
    """Per-server state the handler reads (set on the httpd instance)."""

    def __init__(self, holder: ReadModel, config: DashboardConfig, clock: Clock, port: int):
        self.holder = holder
        self.config = config
        self.clock = clock
        self.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


class _Handler(SecureHandler):
    @property
    def app(self) -> _App:
        return self.server.app  # type: ignore[attr-defined]

    # -- plumbing ---------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":  # HEAD carries the headers (incl. length), never a body
            self.wfile.write(body)

    def _text(self, status: int, text: str, extra: dict[str, str] | None = None) -> None:
        self._send(status, (text + "\n").encode(), "text/plain; charset=utf-8", extra)

    def _html(self, html_text: str, status: int = 200) -> None:
        self._send(status, html_text.encode("utf-8"), "text/html; charset=utf-8")

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    # -- dispatch ---------------------------------------------------------
    def _guard(self) -> bool:
        """DNS-rebinding defence: only our own loopback Host values pass."""
        if self.headers.get("Host", "") not in self.app.allowed_hosts:
            self.close_connection = True  # never read a body from an unvetted peer
            self._text(421, "misdirected request")
            return False
        return True

    def _stalled(self, state: ModelState) -> str | None:
        return collector_stall(
            state, self.app.clock(), float(self.app.config.collector_interval_seconds)
        )

    def _route(self) -> None:
        if not self._guard():
            return
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        state = self.app.holder.get()
        poll = self.app.config.poll_interval_seconds
        if path in ("/", "/now"):
            self._html(render_now(state, poll_seconds=poll, stalled=self._stalled(state)))
        elif path == "/now/fragment":
            self._html(render_fragment(state, poll, self._stalled(state)))
        elif path == "/api/now.json":
            self._json(to_plain(state))
        elif path == "/healthz":
            stall = self._stalled(state)
            age = (
                (self.app.clock() - state.collected_at).total_seconds()
                if state.collected_at
                else None
            )
            self._json(
                {"ok": stall is None, "model_age_seconds": age, "reason": stall},
                200 if stall is None else 503,
            )
        elif path == "/static/dashboard.css":
            self._send(200, assets.stylesheet().encode("utf-8"), "text/css; charset=utf-8")
        elif path.startswith("/static/"):
            self._static(path[len("/static/") :])
        else:
            self._text(404, "not found")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except (ConnectionError, TimeoutError):
            raise  # an aborted peer: logged as one line by the server's handle_error
        except Exception as exc:  # noqa: BLE001 - a render bug is a 500, never a dropped socket
            log.exception("dashboard request failed: %s %s", self.command, self.path)
            self._html(
                "<!doctype html><title>Fleet error</title><h1>Internal error</h1>"
                f"<p>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>",
                500,
            )

    do_HEAD = do_GET  # noqa: N815 - same routes, no body (see ``_send``)

    def _static(self, relpath: str) -> None:
        try:
            data = assets.static_asset(relpath).read_bytes()
        except (ValueError, FileNotFoundError, OSError):
            self._text(404, "not found")
            return
        suffix = "." + relpath.rsplit(".", 1)[-1] if "." in relpath else ""
        self._send(200, data, _STATIC_TYPES.get(suffix, "application/octet-stream"))

    def _method_not_allowed(self) -> None:
        if not self._guard():  # Host check BEFORE touching the body
            return
        # Drain a small body so closing does not RST the socket before the client reads the
        # 405 (observed on Windows); a larger one just gets the connection cut. The socket
        # timeout bounds a body that never arrives.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= MAX_DRAIN_BYTES:
            self.rfile.read(length)
        else:
            self.close_connection = True
        self._text(405, "method not allowed", {"Allow": "GET, HEAD"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed  # noqa: N815


def make_server(
    config: DashboardConfig,
    sources: DashboardSources,
    *,
    clock: Callable[[], datetime] = _utcnow,
) -> DashboardServer | ServerError:
    """Bind (but do not serve) the dashboard; config problems come back as values."""
    if not _is_loopback_host(config.host):
        return ServerError(
            f"dashboard.host {config.host!r} is not loopback; the dashboard has no auth "
            "and only binds 127.0.0.1"
        )
    try:
        httpd = DashboardHTTPServer((config.host, config.port), _Handler)
    except OSError as exc:
        return ServerError(f"cannot bind {config.host}:{config.port}: {exc}")
    holder = ReadModel()
    httpd.app = _App(holder, config, clock, int(httpd.server_address[1]))  # type: ignore[attr-defined]
    return DashboardServer(httpd, holder, config, sources, clock)


def run_server(server: DashboardServer, stop: threading.Event) -> None:
    """Start the collector thread(s) and serve until ``stop`` is set, then shut down."""
    start_workers(
        server.holder,
        stop,
        collect=server.sources.collect,
        rollup=server.sources.rollup,
        collector_interval=server.config.collector_interval_seconds,
        rollup_interval=server.config.rollup_interval_seconds,
        clock=server.clock,
    )
    thread = threading.Thread(
        target=server.httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    thread.start()
    try:
        stop.wait()
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
        thread.join(timeout=5)
