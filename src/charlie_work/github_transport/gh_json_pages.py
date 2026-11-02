"""Follow-up pages for the nested connections gh walks to the end (ADR-0006, F-F).

``gh pr view|list`` and ``gh issue view`` page ``comments``,
``closingIssuesReferences`` and ``statusCheckRollup`` fully (finder.go's
``preloadPr*``). The first GraphQL page is capped at 100, so a node whose
connection reports ``hasNextPage`` is completed here, by node id, before it is
normalized: a PR with more than 100 check contexts must not read as having
only the first 100 (a pending required check past the 100th would be invisible
to the merge gate).

``labels`` and ``assignees`` stay at one page of 100: gh does not page them
either (recorded in ``gt-review-dispositions.md``).

Pure: no I/O. ``json_read`` runs the loop; this module builds the documents and
merges a page into a node without mutating either.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import gh_json_fields as fields_mod
from .gh_json_fields import IncompletePageError


@dataclass(frozen=True)
class _Paged:
    """Where a paged field's connection lives in a node and how to re-select it."""

    path: tuple[str, ...]  # keys from the node to the connection object
    selection: str  # the follow-up selection (takes ``$after``)
    issue_too: bool = False  # the connection also exists on ``Issue`` (comments)


_PAGED: dict[str, _Paged] = {
    "comments": _Paged(("comments",), fields_mod.comments_selection(after=True), issue_too=True),
    "closingIssuesReferences": _Paged(
        ("closingIssuesReferences",), fields_mod.closing_issues_selection(after=True)
    ),
    "statusCheckRollup": _Paged(
        ("statusCheckRollup", "contexts"), fields_mod.rollup_selection(after=True)
    ),
}


def _id_key(item: Any) -> Any:
    """The identity of one list entry: its ``id`` (else ``databaseId``/``number``)."""
    if isinstance(item, dict):
        for key in ("id", "databaseId", "number"):
            if item.get(key) is not None:
                return (key, item[key])
    return None


def dedupe_by_id(items: list[Any]) -> list[Any]:
    """*items* without a repeated id, first occurrence kept, order preserved.

    A list that changes between page reads (offset paging) returns the entry
    at the page boundary twice, and a repeated entry must not be counted or
    acted on twice. An entry with no id is kept as is.
    """
    seen: set[Any] = set()
    unique: list[Any] = []
    for item in items:
        key = _id_key(item)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        unique.append(item)
    return unique


def _connection(node: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    current: Any = node
    for key in path:
        current = current.get(key) if isinstance(current, dict) else None
    return current if isinstance(current, dict) else None


def _next_cursor(connection: dict[str, Any] | None) -> str | None:
    info = connection.get("pageInfo") if connection else None
    if not isinstance(info, dict) or not info.get("hasNextPage"):
        return None
    cursor = info.get("endCursor")
    if not isinstance(cursor, str) or not cursor:
        raise IncompletePageError("a connection has a next page but no end cursor")
    return cursor


def pending_pages(
    resource: str, fields: str | Iterable[str], node: dict[str, Any]
) -> list[tuple[str, str]]:
    """``(field, end cursor)`` for each requested paged connection of *node* with more pages."""
    out: list[tuple[str, str]] = []
    for name in fields_mod.field_names(fields):
        paged = _PAGED.get(name)
        if paged is None or not _is_paged(resource, name):
            continue
        cursor = _next_cursor(_connection(node, paged.path))
        if cursor is not None:
            out.append((name, cursor))
    return out


def _is_paged(resource: str, name: str) -> bool:
    return name in fields_mod.paged_fields(resource)


def page_document(resource: str, field: str) -> str:
    """One follow-up page of *field* for the node with id ``$id``."""
    paged = _PAGED[field]
    fragments = [f"... on PullRequest{{{paged.selection}}}"]
    if resource == "issue" and paged.issue_too:
        fragments.insert(0, f"... on Issue{{{paged.selection}}}")
    return "query($id:ID!,$after:String){node(id:$id){" + " ".join(fragments) + "}}"


def absorb_page(
    node: dict[str, Any], field: str, page: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """*node* with *page*'s items appended to *field*'s connection, plus the next cursor."""
    paged = _PAGED[field]
    fresh = _connection(page, paged.path)
    if fresh is None or not isinstance(fresh.get("nodes"), list):
        raise IncompletePageError(f"follow-up page for {field} had no connection")
    return _extended(node, paged.path, fresh), _next_cursor(fresh)


def _extended(
    node: dict[str, Any], path: tuple[str, ...], fresh: dict[str, Any]
) -> dict[str, Any]:
    key, rest = path[0], path[1:]
    child = node.get(key)
    child = child if isinstance(child, dict) else {}
    if rest:
        return {**node, key: _extended(child, rest, fresh)}
    nodes = dedupe_by_id([*(child.get("nodes") or []), *fresh["nodes"]])
    return {**node, key: {**child, "nodes": nodes, "pageInfo": fresh.get("pageInfo")}}
