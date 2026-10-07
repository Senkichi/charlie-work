"""One paged REST read of a repo's open issues, mapped to the ``gh issue list`` shape (#2443).

``Issues.issue_list`` used to issue one label-filtered GraphQL ``issue list``
per distinct ``(state, labels)`` per pass (3-5 calls, ~1 point each, fleet-wide
a large share of the hourly GraphQL budget). The REST ``issues`` endpoint
costs no GraphQL points, and it travels through the pooled HTTP transport's
ETag cache, so an unchanged repo answers every page with a ``304``.

This module is the single seam: ``fetch_open_issues`` reads *every* open issue
once (pages of 100, newest first, Link-header pagination through
``paginate_rest`` -- each page is its own guarded, ETag-keyed call), drops the
pull requests the endpoint interleaves, and maps each REST object to exactly
the dict ``ISSUE_LIST_FIELDS`` produced under gh's GraphQL dialect, so no
caller changes. Label selection happens locally in ``filter_by_labels``.

Kill switch: ``CHARLIE_WORK_ISSUE_LIST_REST=off`` restores the GraphQL path.
It is also restored per call: a REST failure (``fetch_open_issues`` raises
``GitHubError``) is answered by the caller with the GraphQL read plus a
``github_transport_fallback`` event (``emit_fallback``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from ci_fleet.github import GitHubError

from ..github_transport.pagination import paginate_rest
from ..github_transport.request import RestRequest
from ..instrumentation import log_event
from ._outcome import expect_json
from .circuit_breaker_transport import circuit_breaker_state_path

KILL_SWITCH_ENV = "CHARLIE_WORK_ISSUE_LIST_REST"
_OFF_VALUES = frozenset({"off", "0", "false", "no", "gh", "graphql"})
_PER_PAGE = 100


def rest_issue_list_enabled() -> bool:
    """Ships ON; only an explicit off-value in the environment disables it."""
    return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() not in _OFF_VALUES


def _author(user: Any) -> dict[str, Any] | None:
    if not isinstance(user, dict):
        return None
    is_bot = user.get("type") == "Bot"
    login = user.get("login") or ""
    if is_bot and login:
        # GraphQL's Bot.login has no "[bot]" suffix; gh prints it as ``app/<login>``.
        login = f"app/{login.removesuffix('[bot]')}"
    # REST carries no display name on list rows; gh prints "" when absent.
    return {"id": user.get("node_id"), "is_bot": is_bot, "login": login, "name": ""}


def _label(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):  # defensive: the REST list always sends objects
        return {"id": None, "name": raw, "description": "", "color": None}
    return {
        "id": raw.get("node_id"),
        "name": raw.get("name"),
        "description": raw.get("description") or "",
        "color": raw.get("color"),
    }


def map_rest_issue(raw: dict[str, Any]) -> dict[str, Any]:
    """REST issue object -> the ``ISSUE_LIST_FIELDS`` dict gh's GraphQL read returns.

    GraphQL's non-null ``body`` is ``""`` where REST sends ``null``; ``state``
    is the upper-case enum.
    """
    return {
        "number": raw.get("number"),
        "title": raw.get("title"),
        "url": raw.get("html_url"),
        "body": raw.get("body") or "",
        "labels": [_label(item) for item in raw.get("labels") or [] if item],
        "author": _author(raw.get("user")),
        "createdAt": raw.get("created_at"),
        "updatedAt": raw.get("updated_at"),
        "state": str(raw.get("state") or "").upper(),
    }


def fetch_open_issues(collab: Any) -> list[dict[str, Any]]:
    """Every open issue (PRs dropped), gh-shaped. Raises ``GitHubError`` on any failure."""
    request = RestRequest.of(
        "GET",
        "repos/{owner}/{repo}/issues",
        query={"state": "open", "sort": "created", "direction": "desc", "per_page": _PER_PAGE},
        long_call=True,
    )
    outcome = paginate_rest(collab._transport_v2, request)
    payload = expect_json(outcome, command=request.describe())
    if not isinstance(payload, list):
        raise GitHubError("open issues REST read returned a non-list body")
    return [
        map_rest_issue(item)
        for item in payload
        if isinstance(item, dict) and not item.get("pull_request")
    ]


def filter_by_labels(
    issues: Iterable[dict[str, Any]], labels: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Issues carrying *all* of *labels* (gh ``--label`` AND semantics; names are
    case-insensitive on GitHub). No labels means no filtering."""
    wanted = {name.casefold() for name in labels}
    if not wanted:
        return list(issues)
    return [
        issue
        for issue in issues
        if wanted <= {str(label.get("name", "")).casefold() for label in issue.get("labels") or []}
    ]


def emit_fallback(collab: Any, error: Exception) -> None:
    """Record that the REST read failed and the GraphQL path answered instead.

    Same event kind and payload keys as the transport's own http->gh fallback
    (``GuardedTransport._emit_fallback``), so existing dashboards count it.
    """
    # write-gate-exempt(issue=2443): GitHub client layer has no WriteGate; transport fallback bookkeeping is lock-free
    log_event(
        circuit_breaker_state_path(collab.runtime, collab.repo_root),
        "github_transport_fallback",
        {
            "command": "GET repos/{owner}/{repo}/issues?state=open",
            "request": "issue_list REST->GraphQL",
            "reason": "rest_issue_list_failed",
            "detail": str(error)[:300],
            "mutation": False,
        },
    )


__all__ = [
    "KILL_SWITCH_ENV",
    "emit_fallback",
    "fetch_open_issues",
    "filter_by_labels",
    "map_rest_issue",
    "rest_issue_list_enabled",
]
