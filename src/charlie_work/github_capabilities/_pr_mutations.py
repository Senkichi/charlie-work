"""PR mutations that GitHub only offers over GraphQL (ADR-0006, gt-design G5).

``markPullRequestReadyForReview`` and ``enablePullRequestAutoMerge`` have no
REST route, and both take the PR's node id rather than its number. Each is
therefore two guarded requests: a read for ``pullRequest(number:){id}``, then
the mutation (B10: one extra read per call). Under ``--dry-run`` the read still
goes out, and the guard answers the mutation with a synthetic success.

Failures come back as ``GitHubRunResult`` values (never raised), the shape
``pr_ready`` has always had; ``merge_pr`` turns a failed result into the
``GitHubError`` it has always raised.
"""

from __future__ import annotations

from typing import Any

from ci_fleet.github import GitHubError

from ..github_transport.outcome import Response
from ..github_transport.request import GraphQLRequest
from ._base import GitHubRunResult
from ._outcome import failure_text, to_run_result
from ._send import send

_ID_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) {"
    " repository(owner: $owner, name: $name) { pullRequest(number: $number) { id } } }"
)
_READY_MUTATION = (
    "mutation($id: ID!) {"
    " markPullRequestReadyForReview(input: {pullRequestId: $id})"
    " { pullRequest { number isDraft } } }"
)
_AUTO_MERGE_MUTATION = (
    "mutation($id: ID!, $method: PullRequestMergeMethod!) {"
    " enablePullRequestAutoMerge(input: {pullRequestId: $id, mergeMethod: $method})"
    " { pullRequest { number autoMergeRequest { enabledAt } } } }"
)


def _failed(error: str) -> GitHubRunResult:
    return GitHubRunResult(
        ok=False, returncode=1, stdout="", stderr=error, value=None, error=error
    )


def _pull_request_id(collab: Any, number: int) -> tuple[str | None, str]:
    """``(node id, "")`` for PR *number*, or ``(None, error text)``."""
    try:
        owner, name = collab._repo_owner_name()
    except GitHubError as exc:  # no readable remote: a value, as pr_ready never raises
        return None, str(exc)
    request = GraphQLRequest.of(_ID_QUERY, {"owner": owner, "name": name, "number": number})
    outcome = send(collab, request)
    if not isinstance(outcome, Response) or not outcome.ok:
        return None, failure_text(outcome)
    try:
        body = outcome.json()
        node_id = body["data"]["repository"]["pullRequest"]["id"]
    except (ValueError, KeyError, TypeError):
        return None, f"no pull request #{number} in GraphQL response"
    if not isinstance(node_id, str) or not node_id:
        return None, f"no pull request #{number} in GraphQL response"
    return node_id, ""


def _mutate(collab: Any, number: int, document: str, extra: dict[str, Any]) -> GitHubRunResult:
    node_id, error = _pull_request_id(collab, number)
    if node_id is None:
        return _failed(error)
    request = GraphQLRequest.of(document, {"id": node_id, **extra})
    return to_run_result(send(collab, request), json_output=False, command=request.describe())


def mark_ready(collab: Any, number: int) -> GitHubRunResult:
    """Mark a draft PR ready for review."""
    return _mutate(collab, number, _READY_MUTATION, {})


def enable_auto_merge(collab: Any, number: int, strategy: str) -> GitHubRunResult:
    """Enable auto-merge on PR *number* with merge *strategy* (merge|squash|rebase)."""
    return _mutate(collab, number, _AUTO_MERGE_MUTATION, {"method": strategy.upper()})
