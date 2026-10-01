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

from ..github_transport.outcome import Outcome, Response
from ..github_transport.request import GraphQLRequest, RestRequest
from ._base import GitHubRunResult
from ._outcome import expect_json, expect_ok, is_success, to_run_result

AnyTypedRequest = RestRequest | GraphQLRequest


def send(collab: Any, request: AnyTypedRequest) -> Outcome:
    """Send *request* through the owner's guarded transport."""
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


def status_of(outcome: Outcome) -> int | None:
    """HTTP status of an answered request, ``None`` for a transport failure."""
    return outcome.status if isinstance(outcome, Response) else None


__all__ = [
    "send",
    "send_json",
    "send_ok",
    "send_result",
    "send_text",
    "status_of",
]
