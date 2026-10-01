"""A ``gh ... --json`` read as one value, executed over GraphQL (ADR-0006, G4).

``JsonRead`` names what a ``gh issue|pr list|view|checks --json F`` call used
to ask for. ``execute`` builds the document with ``gh_json_fields``, sends it
through the guarded transport (paginating lists), and returns a ``Response``
whose body is the exact JSON ``gh`` printed, so ``_outcome.expect_json`` and
every consumer of ``issue_list`` / ``pr_view`` / ``pr_checks`` is unchanged.

Failures pass through untouched: a GraphQL ``errors`` array is a
``Response`` with ``graphql_errors`` (so ``Response.ok`` is False and the
caller renders the API's message), a transport failure stays a
``TransportFailure``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from . import gh_json_fields as fields_mod
from .guarded import GitHubTransport
from .outcome import FailureKind, Outcome, Response, TransportFailure
from .pagination import paginate_graphql
from .request import GraphQLRequest, RestRequest, canonical_json

Shape = Literal["list", "view", "search", "checks"]
_PAGE = 100


def _defect(detail: str) -> TransportFailure:
    return TransportFailure(FailureKind.ADAPTER_DEFECT, detail, "guard")


@dataclass(frozen=True)
class JsonRead:
    """One ``--json`` read. Always a query (never a mutation)."""

    resource: Literal["issue", "pr"]
    shape: Shape
    fields: str
    number: int | None = None
    state: str = "open"
    labels: tuple[str, ...] = ()
    head: str | None = None
    search: str | None = None
    limit: int = 30
    long_call: bool = False

    is_mutation = False

    def describe(self) -> str:
        what = f"#{self.number}" if self.number is not None else self.shape
        return f"graphql {self.resource} {what} ({self.fields})"

    # -- execution -----------------------------------------------------------

    def execute(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        try:
            if self.shape == "view":
                return self._view(transport, owner, repo)
            if self.shape == "checks":
                return self._checks(transport, owner, repo)
            if self.shape == "search":
                return self._search(transport, owner, repo)
            return self._list(transport, owner, repo)
        except fields_mod.UnknownFieldError as exc:
            return _defect(str(exc))

    def _send(
        self, transport: GitHubTransport, document: str, variables: dict[str, Any]
    ) -> Outcome:
        return transport.send(GraphQLRequest.of(document, variables, long_call=self.long_call))

    def _view(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        document = fields_mod.document_for(self.resource, self.fields, "view")
        outcome = self._send(
            transport, document, {"owner": owner, "name": repo, "number": self.number}
        )
        field = "issue" if self.resource == "issue" else "pullRequest"
        node = _dig(outcome, ("repository", field))
        if isinstance(node, (Response, TransportFailure)):
            return node
        if not isinstance(node, dict):
            return _defect(f"no {self.resource} #{self.number} in GraphQL response")
        assert isinstance(outcome, Response)
        return _with_body(outcome, fields_mod.normalize_node(self.resource, node, self.fields))

    def _list(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        document = fields_mod.document_for(self.resource, self.fields, "list")
        variables: dict[str, Any] = {
            "owner": owner,
            "name": repo,
            "states": fields_mod.states_for(self.resource, self.state),
            "first": min(self.limit, _PAGE),
        }
        if self.resource == "issue":
            variables["labels"] = list(self.labels) or None
            connection = "issues"
        else:
            variables["head"] = self.head
            connection = "pullRequests"
        request = GraphQLRequest.of(document, variables, long_call=self.long_call)
        outcome = paginate_graphql(
            transport, request, connection_path=("repository", connection), limit=self.limit
        )
        nodes = _dig(outcome, ("repository", connection, "nodes"))
        if isinstance(nodes, (Response, TransportFailure)):
            return nodes
        if not isinstance(nodes, list):
            return _defect(f"no {connection} connection in GraphQL response")
        assert isinstance(outcome, Response)
        return _with_body(outcome, fields_mod.normalize_nodes(self.resource, nodes, self.fields))

    def _search(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        document = fields_mod.document_for("pr", self.fields, "search")
        query = f"repo:{owner}/{repo} is:pr is:{self.state} {self.search or ''}".strip()
        outcome = self._send(transport, document, {"q": query, "first": self.limit})
        nodes = _dig(outcome, ("search", "nodes"))
        if isinstance(nodes, (Response, TransportFailure)):
            return nodes
        if not isinstance(nodes, list):
            return _defect("no search nodes in GraphQL response")
        assert isinstance(outcome, Response)
        # A search hit that is not a PullRequest comes back as an empty object.
        pulls = [n for n in nodes if isinstance(n, dict) and n]
        return _with_body(outcome, fields_mod.normalize_nodes("pr", pulls, self.fields))

    def _checks(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        outcome = self._send(
            transport,
            fields_mod.checks_document(),
            {"owner": owner, "name": repo, "number": self.number},
        )
        node = _dig(outcome, ("repository", "pullRequest"))
        if isinstance(node, (Response, TransportFailure)):
            return node
        if not isinstance(node, dict):
            return _defect(f"no pull request #{self.number} in GraphQL response")
        assert isinstance(outcome, Response)
        contexts = fields_mod.checks_contexts(node)
        return _with_body(outcome, fields_mod.normalize_checks(contexts, self.fields))


# gh ``run list --json`` field -> REST ``workflow_runs`` entry key.
_RUN_FIELDS = {
    "databaseId": "id",
    "status": "status",
    "conclusion": "conclusion",
    "createdAt": "created_at",
    "updatedAt": "updated_at",
    "startedAt": "run_started_at",
    "headBranch": "head_branch",
    "headSha": "head_sha",
    "event": "event",
    "workflowName": "name",
    "displayTitle": "display_title",
    "number": "run_number",
    "url": "html_url",
}


@dataclass(frozen=True)
class RunListRead:
    """``gh run list --json F`` as ``GET actions/runs`` (REST; B1: a read).

    ``workflow`` matches the run's workflow name or its file name, which is
    what ``gh run list --workflow`` accepts. The filter runs client-side over
    the newest page (<= 100 runs); ``limit`` then truncates.
    """

    fields: str
    workflow: str | None = None
    branch: str | None = None
    status: str | None = None
    limit: int = 20
    long_call: bool = False

    is_mutation = False

    def describe(self) -> str:
        return f"rest actions/runs ({self.fields})"

    def execute(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        del owner, repo  # the guard fills the route placeholders
        columns = [f.strip() for f in self.fields.split(",") if f.strip()]
        unknown = [c for c in columns if c not in _RUN_FIELDS]
        if unknown:
            return _defect(f"unknown run field(s): {', '.join(unknown)}")
        query: dict[str, Any] = {"per_page": _PAGE}
        if self.branch:
            query["branch"] = self.branch
        if self.status:
            query["status"] = self.status
        request = RestRequest.of(
            "GET", "repos/{owner}/{repo}/actions/runs", query=query, long_call=self.long_call
        )
        outcome = transport.send(request)
        if not isinstance(outcome, Response) or not outcome.ok:
            return outcome
        try:
            runs = json.loads(outcome.body).get("workflow_runs")
        except (ValueError, AttributeError):
            return _defect("actions/runs body was not a JSON object")
        if not isinstance(runs, list):
            return _defect("actions/runs body had no workflow_runs list")
        wanted = [r for r in runs if isinstance(r, dict) and self._matches(r)]
        rows = [{c: run.get(_RUN_FIELDS[c]) for c in columns} for run in wanted[: self.limit]]
        return _with_body(outcome, rows)

    def _matches(self, run: dict[str, Any]) -> bool:
        if not self.workflow:
            return True
        path = str(run.get("path") or "")
        return self.workflow in (run.get("name"), path.rsplit("/", 1)[-1])


def _with_body(outcome: Response, payload: Any) -> Response:
    return replace(outcome, body=canonical_json(payload))


def _dig(outcome: Outcome, path: tuple[str, ...]) -> Any:
    """``data`` at *path*, or *outcome* itself when it is not a usable page.

    Returns the Response/TransportFailure unchanged for any failure (so the
    caller returns it as is) and ``None`` when the path is absent.
    """
    if not isinstance(outcome, Response) or not outcome.ok:
        return outcome
    try:
        node: Any = json.loads(outcome.body)
    except ValueError:
        return _defect("GraphQL body was not JSON")
    node = node.get("data") if isinstance(node, dict) else None
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node
