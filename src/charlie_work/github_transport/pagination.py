"""Pagination above the transport (ADR-0006).

Both helpers are plain loops over ``GitHubTransport.send``, so every page is
its own guarded call with its own retry, and the page cap applies to every
adapter (the gh adapter never uses ``--paginate``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from .guarded import GitHubTransport
from .outcome import (
    GITHUB_API_HOST,
    FailureKind,
    GraphQLError,
    Outcome,
    Response,
    TransportFailure,
)
from .request import GraphQLRequest, RestRequest, canonical_json

# Moved verbatim from http_transport._MAX_PAGINATE_PAGES.
MAX_PAGES = 50


def next_link(headers: Mapping[str, str]) -> str | None:
    """Extract the ``rel="next"`` URL from a ``Link`` header, if present."""
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in link.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url = segments[0].strip().lstrip("<").rstrip(">")
        rel_part = ";".join(segments[1:])
        if 'rel="next"' in rel_part:
            return url
    return None


def path_from_url(url: str) -> str:
    """Reduce an absolute ``https://api.github.com/...`` Link URL to a bare
    request path -- pagination Link headers always stay on the same host."""
    marker = f"https://{GITHUB_API_HOST}"
    if url.startswith(marker):
        return url[len(marker) :] or "/"
    return url


def _defect(detail: str) -> TransportFailure:
    return TransportFailure(FailureKind.ADAPTER_DEFECT, detail, "guard")


def _next_request(request: RestRequest, url: str) -> RestRequest | TransportFailure:
    parts = urlsplit(url)
    if parts.netloc and parts.netloc != GITHUB_API_HOST:
        return _defect(f"refusing to follow pagination link off host: {url!r}")
    route = parts.path.lstrip("/")
    if not route:
        return _defect(f"pagination link has no path: {url!r}")
    return replace(request, route=route, query=tuple(parse_qsl(parts.query)))


def _list_items(response: Response, items_key: str | None) -> list[Any] | TransportFailure:
    try:
        parsed = response.json()
    except ValueError:
        return _defect("paginated REST body was not JSON")
    if items_key is not None:
        parsed = parsed.get(items_key) if isinstance(parsed, dict) else None
    if not isinstance(parsed, list):
        return _defect("paginated REST body was not a list")
    return parsed


def paginate_rest(
    transport: GitHubTransport,
    request: RestRequest,
    *,
    max_pages: int = MAX_PAGES,
    items_key: str | None = None,
) -> Outcome:
    """All pages of a list endpoint as one ``Response`` whose body is the
    concatenated JSON list. ``items_key`` names the array inside an object
    body (``workflow_runs``, ``items``, ...). The first non-ok page is the
    outcome; exceeding *max_pages* with a next link still present is a
    defect, never a silent truncation."""
    collected: list[Any] = []
    current = request
    for _ in range(max_pages):
        outcome = transport.send(current)
        if not isinstance(outcome, Response) or not outcome.ok:
            return outcome
        items = _list_items(outcome, items_key)
        if isinstance(items, TransportFailure):
            return items
        collected.extend(items)
        url = next_link(dict(outcome.headers))
        if url is None:
            return replace(outcome, body=canonical_json(collected))
        following = _next_request(current, url)
        if isinstance(following, TransportFailure):
            return following
        current = following
    return _defect(f"pagination exceeded {max_pages} pages with a next link still present")


def _connection_at(data: Any, path: Sequence[str]) -> dict[str, Any] | None:
    node = data
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def paginate_graphql(
    transport: GitHubTransport,
    request: GraphQLRequest,
    *,
    connection_path: Sequence[str],
    limit: int,
    max_pages: int = MAX_PAGES,
) -> Outcome:
    """Walk a GraphQL connection. The document must declare ``$after`` and
    select ``pageInfo { hasNextPage endCursor }`` and ``nodes`` at
    *connection_path* (under ``data``). Returns the last page's body with the
    collected nodes (capped at *limit*) substituted in, so callers parse one
    response shape whether or not there were several pages."""
    nodes: list[Any] = []
    errors: list[GraphQLError] = []
    variables = json.loads(request.variables)
    cursor: str | None = None
    for _ in range(max_pages):
        page_vars = {**variables, "after": cursor}
        page = replace(request, variables=canonical_json(page_vars))
        outcome = transport.send(page)
        if not isinstance(outcome, Response) or not 200 <= outcome.status < 300:
            return outcome
        if outcome.graphql_errors and not request.partial_ok:
            return outcome
        try:
            body = json.loads(outcome.body)
        except ValueError:
            return _defect("paginated GraphQL body was not JSON")
        connection = _connection_at(
            body.get("data") if isinstance(body, dict) else None, connection_path
        )
        page_nodes = connection.get("nodes") if connection is not None else None
        if connection is None or not isinstance(page_nodes, list):
            return _defect(f"no connection with nodes at {'.'.join(connection_path)}")
        errors.extend(outcome.graphql_errors)
        nodes.extend(page_nodes)
        page_info = connection.get("pageInfo")
        page_info = page_info if isinstance(page_info, dict) else {}
        cursor = page_info.get("endCursor")
        if len(nodes) >= limit or not page_info.get("hasNextPage") or not cursor:
            merged = copy.deepcopy(body)
            target = _connection_at(merged["data"], connection_path)
            assert target is not None
            target["nodes"] = nodes[:limit]
            return replace(outcome, body=canonical_json(merged), graphql_errors=tuple(errors))
    return _defect(f"pagination exceeded {max_pages} pages with a next page still present")
