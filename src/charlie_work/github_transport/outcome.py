"""Transport outcome values (ADR-0006).

A non-2xx status is a ``Response``, not a failure: GitHub answered. So is a
200 whose GraphQL ``errors`` array is non-empty. A ``TransportFailure`` means
no usable answer came back. Adapters return these as values and never raise
for network or HTTP conditions (CLAUDE.md: errors from external processes
come back as values). The one exception allowed to cross the transport is
``pass_deadline.PassDeadlineExceeded``, a ``BaseException`` by design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal

GITHUB_API_HOST = "api.github.com"


@dataclass(frozen=True)
class GraphQLError:
    message: str
    type: str | None = None  # "NOT_FOUND", "RATE_LIMITED", extensions.code, ...
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class Response:
    status: int  # HTTP status; 200/0 for gh-local CLI results (see returncode)
    headers: tuple[tuple[str, str], ...]  # lower-cased names
    body: str
    adapter: Literal["http", "gh", "dry_run"]
    graphql_errors: tuple[GraphQLError, ...] = ()
    returncode: int | None = None  # only set for CliRequest results

    def header(self, name: str) -> str | None:
        wanted = name.lower()
        for key, value in self.headers:
            if key == wanted:
                return value
        return None

    def json(self) -> Any:
        """Parse the body; raises ``ValueError`` on non-JSON (caller converts)."""
        return json.loads(self.body)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and not self.graphql_errors


class FailureKind(Enum):
    CONNECT = "connect"  # DNS/refused/TLS handshake: provably not sent
    SENT_NO_RESPONSE = "sent_no_response"  # reset/EOF after send: a mutation may have landed
    TIMEOUT = "timeout"
    TOKEN_UNAVAILABLE = "token_unavailable"
    ADAPTER_DEFECT = "adapter_defect"  # KeyError/TypeError/... inside an adapter; malformed body
    CLI_MISSING = "cli_missing"  # gh binary absent
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)
class TransportFailure:
    kind: FailureKind
    detail: str
    adapter: Literal["http", "gh", "guard"]


Outcome = Response | TransportFailure


def normalize_headers(pairs: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(k).lower(), str(v)) for k, v in pairs)


def parse_graphql_errors(errors: Any) -> tuple[GraphQLError, ...]:
    """Convert a raw GraphQL ``errors`` array into typed values.

    Tolerant by design: a non-list or malformed element degrades to a
    message-only error rather than raising inside an adapter.
    """
    if not isinstance(errors, list):
        return ()
    parsed: list[GraphQLError] = []
    for item in errors:
        if not isinstance(item, dict):
            parsed.append(GraphQLError(message=str(item)))
            continue
        extensions = item.get("extensions")
        code = extensions.get("code") if isinstance(extensions, dict) else None
        kind = item.get("type") or code
        path = item.get("path")
        parsed.append(
            GraphQLError(
                message=str(item.get("message", item)),
                type=str(kind) if kind else None,
                path=tuple(path) if isinstance(path, list) else (),
            )
        )
    return tuple(parsed)


def graphql_errors_from_body(body: str) -> tuple[GraphQLError, ...] | None:
    """Errors in a GraphQL response body, or ``None`` when the body is not
    a JSON object (an adapter defect the caller must surface)."""
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parse_graphql_errors(parsed.get("errors"))


def extract_message(status: int, body: str) -> str:
    """GitHub's own REST error message, passed through verbatim so it keeps
    the substrings ("Bad credentials", "API rate limit exceeded", ...) the
    repo's text classifiers match on."""
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str) and parsed["message"]:
        return parsed["message"]
    return body.strip()[:200] or f"HTTP {status}"


def render_graphql_errors(errors: tuple[GraphQLError, ...]) -> str:
    """``GraphQL: {message} ({dotted.path})`` for the first error -- gh's own
    format, which existing text consumers match."""
    if not errors:
        return "GraphQL: unknown error"
    first = errors[0]
    if first.path:
        return f"GraphQL: {first.message} ({'.'.join(str(p) for p in first.path)})"
    return f"GraphQL: {first.message}"


def render_legacy_error(outcome: Outcome) -> str:
    """The single place an outcome becomes gh-shaped error text.

    Returns ``""`` for a successful ``Response``. Text classifiers that still
    read prose (``_is_not_found_gh_error``, the breaker markers) keep working
    on this output until they are deleted with the legacy shim.
    """
    if isinstance(outcome, Response):
        if outcome.graphql_errors:
            return render_graphql_errors(outcome.graphql_errors)
        if outcome.status == 0:  # gh-local command failure: body carries its stderr
            return outcome.body.strip() or f"gh exited {outcome.returncode}"
        if 200 <= outcome.status < 300:
            return ""
        return f"gh: {extract_message(outcome.status, outcome.body)} (HTTP {outcome.status})"
    kind, detail = outcome.kind, outcome.detail
    if kind is FailureKind.CONNECT:
        if "error connecting to" in detail.lower():
            return detail
        return f"error connecting to https://{GITHUB_API_HOST}: {detail}"
    if kind is FailureKind.SENT_NO_RESPONSE:
        return f"connection reset (request sent, no response): {detail}"
    if kind is FailureKind.CLI_MISSING:
        return "GitHub CLI `gh` is not installed or not on PATH."
    if kind is FailureKind.TOKEN_UNAVAILABLE:
        return f"GitHub token unavailable: {detail}"
    if kind is FailureKind.ADAPTER_DEFECT:
        return f"GitHub transport defect ({outcome.adapter}): {detail}"
    return detail  # TIMEOUT, CIRCUIT_OPEN: the detail is already the message
