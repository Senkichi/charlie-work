"""Hardened stdlib HTTP plumbing for the dashboard: bind exclusivity, timeouts, headers.

Everything here is route-agnostic. ``DashboardHTTPServer`` refuses to share a port (on
Windows ``SO_REUSEADDR`` lets a second listener bind a port that is already LISTENING and
traffic goes to an arbitrary one) and logs aborted connections as one line.
``SecureHandler`` puts the security headers on EVERY response from the single
``end_headers`` choke point, so stdlib ``send_error`` paths (400/414/431/501/505) and 500s
carry them too, and bounds how long any connection may hold a thread.
"""

from __future__ import annotations

import logging
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger("charlie_work.dashboard")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "img-src 'self' data:; frame-ancestors 'none'"
)
SECURITY_HEADERS = (
    ("Content-Security-Policy", CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cache-Control", "no-store"),
)
# Idle / half-sent connections are dropped after this long instead of pinning a thread.
CONNECTION_TIMEOUT_SECONDS = 10.0
# A request body is never needed (read-only server); drain at most this much so closing
# does not RST the socket before the client reads the response.
MAX_DRAIN_BYTES = 65536


class DashboardHTTPServer(ThreadingHTTPServer):
    """Exclusive-bind threading server that never prints aborted-request tracebacks."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if sys.platform == "win32" and exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            # Browsers abort on every reload/tab close; one line, no traceback.
            log.info("client %s aborted: %s: %s", client_address[0], type(exc).__name__, exc)
            return
        super().handle_error(request, client_address)


class SecureHandler(BaseHTTPRequestHandler):
    """Base handler: connection timeout plus security headers on every response."""

    timeout = CONNECTION_TIMEOUT_SECONDS
    server_version = "charlie-dashboard"
    sys_version = ""

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # An unparseable request line leaves request_version at HTTP/0.9, for which the
        # stdlib writes a bare body with no status line or headers at all.
        if self.request_version == "HTTP/0.9":
            self.request_version = "HTTP/1.0"
        super().send_error(code, message, explain)

    def end_headers(self) -> None:
        for key, value in SECURITY_HEADERS:
            self.send_header(key, value)
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        log.debug("%s - %s", self.address_string(), format % args)

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        log.warning("%s - %s", self.address_string(), format % args)
