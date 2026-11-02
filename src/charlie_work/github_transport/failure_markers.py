"""Failure-text marker sets shared by the gh adapter and the circuit breaker.

The ``gh`` CLI reports network trouble only as stderr prose. The text
classifiers live here, once, so the gh adapter (translating stderr into a
typed ``TransportFailure``) and ``github_capabilities.circuit_breaker`` (the
legacy text classifier, deleted at the end of the migration) cannot drift.
"""

from __future__ import annotations

# Failures that provably occurred before the request reached GitHub. A
# mutation is only retried on these, preserving at-most-once semantics.
PRE_CONNECTION_MARKERS: tuple[str, ...] = (
    "tls handshake timeout",
    "connection refused",
    "could not connect",
    "error connecting to",
)

# Connect/handshake/DNS marker set. Deliberately narrower than the retry
# allowlist in ``transient_errors``: a pattern here must mean "no response
# reached the caller at all". Moved verbatim from ``circuit_breaker.py``.
TRANSPORT_CLASS_MARKERS: tuple[str, ...] = (
    # TLS handshake timeout (Go net/http, the #1832 signature).
    "tls handshake timeout",
    "connection refused",
    "connection reset",
    "connection was reset",
    "could not connect",
    "couldn't connect",
    "error connecting to",
    "i/o timeout",
    "remote end hung up",
    "empty reply from server",
    # DNS resolution failure (Go's net.DNSError / Winsock via "connectex").
    "no such host",
    "could not resolve host",
    "connectex",
)


def is_pre_connection_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in PRE_CONNECTION_MARKERS)


def is_transport_class_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRANSPORT_CLASS_MARKERS)
