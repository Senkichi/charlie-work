"""Generic GitHub transport (ADR-0006).

One ``Request`` value in, one ``Outcome`` value out. The HTTP adapter is the
default and the ``gh`` adapter is the per-call fallback (and the kill switch,
``runtime.gh_transport: gh``). ``GuardedTransport`` is the single wrapper
stack: dry-run, breaker, deadline, retry, fallback and rate-limit accounting.

This package sits below ``github_capabilities``; it never imports
``github.py``, ``config.py`` or the capability collaborators.
"""

from __future__ import annotations

from .gh_adapter import GhAdapter
from .guarded import (
    Adapter,
    Adapters,
    BreakerPort,
    GitHubTransport,
    GuardedTransport,
    RateBudgetHolder,
    RuntimePort,
)
from .budget_pass import budget_pass_payload, emit_github_budget_pass
from .capability import capability_name, capability_scope, current_capability
from .http_adapter import HttpAdapter
from .outcome import (
    FailureKind,
    GraphQLError,
    Outcome,
    Response,
    TransportFailure,
    render_legacy_error,
)
from .pagination import paginate_graphql, paginate_rest
from .request import CliCommand, CliRequest, GraphQLRequest, Request, RestRequest

__all__ = [
    "Adapter",
    "Adapters",
    "BreakerPort",
    "CliCommand",
    "CliRequest",
    "FailureKind",
    "GhAdapter",
    "GitHubTransport",
    "GraphQLError",
    "GraphQLRequest",
    "GuardedTransport",
    "HttpAdapter",
    "Outcome",
    "RateBudgetHolder",
    "Request",
    "Response",
    "RestRequest",
    "RuntimePort",
    "TransportFailure",
    "budget_pass_payload",
    "capability_name",
    "capability_scope",
    "current_capability",
    "emit_github_budget_pass",
    "paginate_graphql",
    "paginate_rest",
    "render_legacy_error",
]
