"""Which capability a request is spent on (issue #2439).

The guard attributes every request's rate-limit points to a *capability*
name. Callers that know theirs (the capability collaborators, the legacy
``gh`` argv shim) open a ``capability_scope``; anything else is named from
the request itself, so no request is ever unattributed. The scope is a
``ContextVar``: a lane runs on its own pool thread, so scopes never leak
between lanes.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .request import CliRequest, GraphQLRequest, Request, RestRequest

_SCOPE: ContextVar[str | None] = ContextVar("github_capability", default=None)
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def capability_name(owner: object) -> str:
    """``PullRequests`` -> ``pull_requests``: a collaborator's capability."""
    return _CAMEL.sub("_", type(owner).__name__).lower()


@contextmanager
def capability_scope(name: str) -> Iterator[None]:
    token = _SCOPE.set(name)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current_capability() -> str | None:
    return _SCOPE.get()


def derive_capability(request: Request) -> str:
    """A name from the request alone: the REST route family or the GraphQL
    operation type."""
    if isinstance(request, RestRequest):
        parts = [p for p in request.route.split("/") if p]
        if parts and parts[0] == "repos":
            parts = parts[3:]  # repos/{owner}/{repo}/<family>/...
        return f"rest.{parts[0] if parts else 'repo'}"
    if isinstance(request, GraphQLRequest):
        return f"graphql.{request.operation}"
    if isinstance(request, CliRequest):
        return "cli." + "_".join(request.command.value)
    return "unknown"


def resolve_capability(request: Request) -> str:
    return current_capability() or derive_capability(request)
