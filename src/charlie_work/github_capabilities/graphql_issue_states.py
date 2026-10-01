"""Batched GraphQL issue-state query tolerant of per-node failures (#1933).

The implementation of ``Transport._graphql_issue_states``. Kept as
module-level functions rather than ``Transport`` methods for the same reason
as ``circuit_breaker_transport.py`` (see that module's docstring): nothing
here is a ``self.<name>()`` delegation target reached through
``GitHub``'s owner-to-collaborator routing, ``Transport`` is over its
attachment-point member ceiling, and ``transport.py`` sits at the edge of the
800-line file-size ratchet (issue #1442).

Why this exists: one unresolvable issue number in the batched query
(``s_361: issue(number: 361)`` on a deleted/renumbered/hiccuping issue) makes
``gh api graphql`` exit non-zero for the WHOLE query -- but GitHub still
answers HTTP 200 with partial ``data``: every other alias resolves normally
and the bad alias is ``null``, alongside a per-node ``errors`` entry. Both
transports preserve that body: real ``gh`` copies the response body to
stdout before emitting the error (verified in cli/cli
``pkg/cmd/api/api.go``'s ``processResponse`` -- the ``io.Copy(bodyWriter,
responseBody)`` runs ahead of the ``serverError`` branch), and this repo's
pooled HTTP transport does the same (``github_transport.http_adapter``).
Running the query through ``run(..., allow_failure=True)`` therefore
surfaces the parsed body as ``GitHubRunResult.value`` even when ``ok`` is
False, so a single bad alias no longer demotes an entire batch to the
per-issue ``issue_view`` fallback -- which is what timed out
``fleet status --json`` in the field.
"""

from __future__ import annotations

import logging
from typing import Any

from ci_fleet.github import GitHubError

from ..github_transport.request import GraphQLRequest
from ._send import send_graphql

logger = logging.getLogger(__name__)


def _issue_state_fields(chunk: list[int]) -> str:
    return " ".join(f"s_{n}: issue(number: {n}) {{ number state }}" for n in chunk)


def graphql_issue_states(
    transport: Any, issue_numbers: list[int], batch_size: int
) -> dict[int, bool]:
    """Fetch open/closed state for many issue numbers via batched GraphQL.

    ``transport`` is the ``Transport`` collaborator (duck-typed: needs
    ``._transport_v2`` and ``._repo_owner_name()``; ``Any`` because importing the
    class would cycle -- the module map's ``TYPE_CHECKING`` pattern does not
    apply to a call target).

    Returns a mapping ``issue_number -> is_open`` covering the numbers that
    resolved. A requested number ABSENT from the mapping is one the batched
    query could not resolve (``data.repository.s_<n>`` came back ``null``
    alongside a per-node ``errors`` entry) -- the caller per-issue-fetches
    exactly those numbers instead of the whole batch (issue #1933). A
    whole-query failure -- no usable ``data`` at all, or ``run`` itself
    raising -- still propagates as ``GitHubError`` so the caller's existing
    full-set fallback covers it.
    """
    if not issue_numbers:
        return {}

    owner, name = transport._repo_owner_name()
    states: dict[int, bool] = {}

    for i in range(0, len(issue_numbers), batch_size):
        chunk = issue_numbers[i : i + batch_size]
        query = (
            f"query($owner: String!, $name: String!) {{ "
            f"repository(owner: $owner, name: $name) {{ {_issue_state_fields(chunk)} }} "
            f"}}"
        )

        value, error = send_graphql(
            transport,
            GraphQLRequest.of(query, {"owner": owner, "name": name}, partial_ok=True),
        )
        if error is not None:
            data = value.get("data") if isinstance(value, dict) else None
            if not isinstance(data, dict) or not isinstance(data.get("repository"), dict):
                # No usable partial data -- a genuine whole-query failure.
                # Raise so the caller's full-set per-issue fallback applies
                # (the pre-#1933 path).
                raise GitHubError(f"GraphQL query failed: {error}")
            logger.warning(
                "Batched issue-state query returned per-node errors; "
                "keeping the resolved aliases and leaving the unresolved "
                "numbers for scoped per-issue fallback: %s",
                error,
            )

        if not isinstance(value, dict):
            raise GitHubError("GraphQL query returned non-dict JSON")

        data = value.get("data")
        repo = data.get("repository") if isinstance(data, dict) else None
        if not isinstance(repo, dict):
            raise GitHubError("GraphQL response missing repository")

        for number in chunk:
            issue = repo.get(f"s_{number}")
            if not isinstance(issue, dict):
                # Unresolvable node -- deliberately absent from `states` so
                # the caller per-issue-fetches just this number.
                continue
            returned_number = issue.get("number")
            if returned_number is not None:
                number = int(returned_number)
            states[number] = str(issue.get("state") or "").upper() == "OPEN"

    return states
