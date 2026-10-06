"""Typed-request send helpers for the capability collaborators (ADR-0006).

A capability builds a ``Request`` and calls one of these with itself as the
first argument; the transport (reached through the owner's ``_transport_v2``)
returns an ``Outcome`` and the helper renders it into the shape the method
has always exposed: a raised ``GitHubError``, a ``bool``, or a
``GitHubRunResult``. The raise-vs-return contract therefore lives in
``_outcome``, not in each capability.
"""

from __future__ import annotations

from typing import Any

from ..github_transport.capability import capability_name, capability_scope
from ..github_transport.json_read import JsonRead
from ..github_transport.outcome import Outcome, Response
from ..github_transport.request import GraphQLRequest, RestRequest
from ._base import GitHubRunResult
from ._outcome import expect_json, expect_ok, failure_text, is_success, to_run_result

AnyTypedRequest = RestRequest | GraphQLRequest


def send(collab: Any, request: AnyTypedRequest) -> Outcome:
    """Send *request* through the owner's guarded transport."""
    with capability_scope(capability_name(collab)):
        return collab._transport_v2.send(request)


def send_ok(collab: Any, request: AnyTypedRequest) -> bool:
    """True iff the request succeeded (never raises; dry-run is a success)."""
    return is_success(send(collab, request))


def send_text(collab: Any, request: AnyTypedRequest) -> str:
    """Stripped body of a successful request; raises ``GitHubError`` otherwise."""
    return expect_ok(send(collab, request), command=request.describe())


def send_json(collab: Any, request: AnyTypedRequest) -> Any:
    """Parsed JSON body of a successful request; raises ``GitHubError`` otherwise."""
    return expect_json(send(collab, request), command=request.describe())


def send_result(
    collab: Any, request: AnyTypedRequest, *, json_output: bool = False
) -> GitHubRunResult:
    """The ``allow_failure=True`` shape: errors come back as a value."""
    return to_run_result(
        send(collab, request), json_output=json_output, command=request.describe()
    )


def send_graphql(collab: Any, request: GraphQLRequest) -> tuple[Any, str | None]:
    """Send a GraphQL request; return ``(parsed body, error text)``.

    The body is returned whenever the call was answered 2xx and parses, even
    if it carries per-node ``errors`` (the partial ``data`` GitHub still
    sends, #1933); ``error`` is then the rendered errors. A transport failure
    or non-2xx answer returns ``(None, text)``. Never raises.
    """
    outcome = send(collab, request)
    if not isinstance(outcome, Response) or not 200 <= outcome.status < 300:
        return None, failure_text(outcome)
    try:
        body = outcome.json()
    except ValueError:
        return None, "GraphQL response was not JSON"
    if outcome.ok:
        return body, None
    return body, failure_text(outcome)


def send_read(collab: Any, read: JsonRead) -> Outcome:
    """Execute a ``--json`` read (GraphQL under the hood) for the repo's slug.

    ``_repo_owner_name`` may raise ``GitHubError`` (no readable remote); that
    propagates exactly as it does for a ``{owner}/{repo}`` REST route.
    """
    owner, name = collab._repo_owner_name()
    with capability_scope(capability_name(collab)):
        return read.execute(collab._transport_v2, owner, name)


def read_json(collab: Any, read: JsonRead) -> Any:
    """Parsed gh-dialect JSON of a read; raises ``GitHubError`` on failure."""
    return expect_json(send_read(collab, read), command=read.describe())


def read_result(collab: Any, read: JsonRead) -> GitHubRunResult:
    """The ``allow_failure=True`` shape of a read: errors come back as a value."""
    return to_run_result(send_read(collab, read), json_output=True, command=read.describe())


def status_of(outcome: Outcome) -> int | None:
    """HTTP status of an answered request, ``None`` for a transport failure."""
    return outcome.status if isinstance(outcome, Response) else None


__all__ = [
    "send",
    "send_graphql",
    "read_json",
    "read_result",
    "send_json",
    "send_ok",
    "send_read",
    "send_result",
    "send_text",
    "status_of",
]
