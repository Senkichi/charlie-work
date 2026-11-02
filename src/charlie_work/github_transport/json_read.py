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
from urllib.parse import quote

from . import gh_json_fields as fields_mod
from . import gh_json_pages as pages_mod
from .guarded import GitHubTransport
from .outcome import FailureKind, Outcome, Response, TransportFailure
from .pagination import MAX_PAGES, paginate_graphql, paginate_rest
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
        if self.resource == "issue" and self.shape == "list" and len(self.labels) > 1:
            # GraphQL ``issues(labels:)`` is OR; ``gh --label a --label b`` is AND.
            # Refuse rather than return the silently-wrong superset; nothing is sent.
            return _defect(
                f"issue list with {len(self.labels)} labels is unsupported: "
                "GraphQL matches any label (OR), gh matched all (AND)"
            )
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
        field = "issueOrPullRequest" if self.resource == "issue" else "pullRequest"
        node = _dig(outcome, ("repository", field))
        if isinstance(node, (Response, TransportFailure)):
            return node
        if not isinstance(node, dict):
            return _defect(f"no {self.resource} #{self.number} in GraphQL response")
        assert isinstance(outcome, Response)
        completed = self._complete(transport, node)
        if not isinstance(completed, dict):
            return completed
        return _with_body(
            outcome, fields_mod.normalize_node(self.resource, completed, self.fields)
        )

    def _complete(
        self, transport: GitHubTransport, node: dict[str, Any]
    ) -> dict[str, Any] | Outcome:
        """*node* with every nested connection gh pages to the end fetched in full.

        The first GraphQL page of ``comments``, ``closingIssuesReferences`` and
        ``statusCheckRollup`` holds 100 items; a connection that reports a next
        page is followed by node id. Silent truncation is never the result: a
        failed or non-terminating follow-up returns the failure (or a defect
        once ``MAX_PAGES`` pages have been read).
        """
        try:
            pending = pages_mod.pending_pages(self.resource, self.fields, node)
            for name, cursor in pending:
                node_id = node.get("id")
                if not isinstance(node_id, str) or not node_id:
                    return _defect(f"{name} has more pages but the node has no id")
                for _ in range(MAX_PAGES):
                    document = pages_mod.page_document(self.resource, name)
                    outcome = self._send(transport, document, {"id": node_id, "after": cursor})
                    page = _dig(outcome, ("node",))
                    if isinstance(page, (Response, TransportFailure)):
                        return page
                    if not isinstance(page, dict):
                        return _defect(f"no node for the next {name} page")
                    node, next_cursor = pages_mod.absorb_page(node, name, page)
                    if next_cursor is None:
                        break
                    cursor = next_cursor
                else:
                    return _defect(f"{name} exceeded {MAX_PAGES} pages with a next page present")
        except fields_mod.IncompletePageError as exc:
            return _defect(str(exc))
        return node

    def _complete_all(self, transport: GitHubTransport, nodes: list[Any]) -> list[Any] | Outcome:
        completed: list[Any] = []
        for item in nodes:
            if not isinstance(item, dict) or not item:
                completed.append(item)
                continue
            full = self._complete(transport, item)
            if not isinstance(full, dict):
                return full
            completed.append(full)
        return completed

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
        completed = self._complete_all(transport, pages_mod.dedupe_by_id(nodes))
        if not isinstance(completed, list):
            return completed
        return _with_body(
            outcome, fields_mod.normalize_nodes(self.resource, completed, self.fields)
        )

    def _search(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        document = fields_mod.document_for("pr", self.fields, "search")
        # ``--state all`` is no qualifier at all (``is:all`` is not a search term).
        state = "" if self.state.strip().lower() == "all" else f"is:{self.state}"
        query = " ".join(
            part for part in (f"repo:{owner}/{repo}", "is:pr", state, self.search or "") if part
        )
        outcome = self._send(transport, document, {"q": query, "first": self.limit})
        nodes = _dig(outcome, ("search", "nodes"))
        if isinstance(nodes, (Response, TransportFailure)):
            return nodes
        if not isinstance(nodes, list):
            return _defect("no search nodes in GraphQL response")
        assert isinstance(outcome, Response)
        # A search hit that is not a PullRequest comes back as an empty object.
        pulls = [n for n in nodes if isinstance(n, dict) and n]
        completed = self._complete_all(transport, pulls)
        if not isinstance(completed, list):
            return completed
        return _with_body(outcome, fields_mod.normalize_nodes("pr", completed, self.fields))

    def _checks(self, transport: GitHubTransport, owner: str, repo: str) -> Outcome:
        # `gh pr checks` walks every page of contexts; a snapshot cut at 100
        # could omit a required check and read as green.
        document = fields_mod.checks_document()
        contexts: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            outcome = self._send(
                transport,
                document,
                {"owner": owner, "name": repo, "number": self.number, "after": cursor},
            )
            node = _dig(outcome, ("repository", "pullRequest"))
            if isinstance(node, (Response, TransportFailure)):
                return node
            if not isinstance(node, dict):
                return _defect(f"no pull request #{self.number} in GraphQL response")
            assert isinstance(outcome, Response)
            contexts = pages_mod.dedupe_by_id([*contexts, *fields_mod.checks_contexts(node)])
            try:
                cursor = fields_mod.checks_next_cursor(node)
            except fields_mod.IncompletePageError as exc:
                return _defect(str(exc))
            if cursor is None:
                return _with_body(outcome, fields_mod.normalize_checks(contexts, self.fields))
        return _defect(f"check contexts exceeded {MAX_PAGES} pages with a next page present")


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


def _is_file_name(workflow: str) -> bool:
    return workflow.isdigit() or workflow.lower().endswith((".yml", ".yaml"))


@dataclass(frozen=True)
class RunListRead:
    """``gh run list --json F`` as a REST read (B1: a read).

    Without ``workflow`` it reads ``GET actions/runs``. With one it resolves
    the workflow exactly as gh does (a numeric id or ``*.yml|*.yaml`` file name,
    suffix case-insensitive, is used as given; anything else is matched
    case-insensitively against the names of the repo's workflows other than
    those in state ``disabled_manually``, and must match exactly one) and reads
    ``GET actions/workflows/{id_or_file}/runs``. Either way ``branch``,
    ``status`` and ``event`` are server-side filters, ``per_page`` is
    ``min(limit, 100)`` so ``limit <= 100`` is one request, and
    ``exclude_pull_requests=true`` is sent like gh does (no ``--json`` column
    reads ``pull_requests``). A ``limit`` under 1 is refused up front: gh
    rejects ``--limit < 1`` rather than listing nothing. The repo-wide list
    is never filtered client-side: it pages by offset over a list that keeps
    changing, and a shifted page repeats an entry. A ``limit`` over 100 pages
    the filtered endpoint and drops a repeated run id.
    """

    fields: str
    workflow: str | None = None
    branch: str | None = None
    status: str | None = None
    event: str | None = None
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
        if self.limit < 1:
            # gh rejects `run list --limit < 1`; refuse before any request.
            return _defect(f"run list needs a --limit of at least 1, got {self.limit}")
        route = self._route(transport)
        if not isinstance(route, str):
            return route
        per_page = min(self.limit, _PAGE)
        # gh sends exclude_pull_requests to shrink run-list payloads; no
        # _RUN_FIELDS column reads pull_requests.
        filters: dict[str, Any] = {"per_page": per_page, "exclude_pull_requests": "true"}
        for name, value in (
            ("branch", self.branch),
            ("status", self.status),
            ("event", self.event),
        ):
            if value:
                filters[name] = value
        wanted: list[dict[str, Any]] = []
        last: Response | None = None
        for page in range(1, MAX_PAGES + 1):
            query = filters if page == 1 else {**filters, "page": page}
            outcome = transport.send(
                RestRequest.of("GET", route, query=query, long_call=self.long_call)
            )
            if not isinstance(outcome, Response) or not outcome.ok:
                return outcome
            last = outcome
            try:
                runs = json.loads(outcome.body).get("workflow_runs")
            except (ValueError, AttributeError):
                return _defect("actions/runs body was not a JSON object")
            if not isinstance(runs, list):
                return _defect("actions/runs body had no workflow_runs list")
            wanted = pages_mod.dedupe_by_id([*wanted, *(r for r in runs if isinstance(r, dict))])
            if len(wanted) >= self.limit or len(runs) < per_page:
                break
        else:
            return _defect(f"actions/runs exceeded {MAX_PAGES} pages without {self.limit} runs")
        assert last is not None
        rows = [{c: run.get(_RUN_FIELDS[c]) for c in columns} for run in wanted[: self.limit]]
        return _with_body(last, rows)

    def _route(self, transport: GitHubTransport) -> str | TransportFailure | Response:
        """The runs route: repo-wide, or the resolved workflow's own."""
        if not self.workflow:
            return "repos/{owner}/{repo}/actions/runs"
        base = "repos/{owner}/{repo}/actions/workflows"
        if _is_file_name(self.workflow):
            return f"{base}/{quote(self.workflow, safe='')}/runs"
        listed = paginate_rest(
            transport,
            RestRequest.of("GET", base, query={"per_page": _PAGE}, long_call=self.long_call),
            items_key="workflows",
        )
        if not isinstance(listed, Response) or not listed.ok:
            return listed
        try:
            workflows = json.loads(listed.body)
        except ValueError:
            return _defect("actions/workflows body was not JSON")
        ids = [
            w.get("id")
            for w in workflows
            if isinstance(w, dict)
            and isinstance(w.get("name"), str)
            and w["name"].casefold() == self.workflow.casefold()
            and w.get("state") != "disabled_manually"
            and w.get("id") is not None
        ]
        if len(ids) != 1:
            return _defect(f"{len(ids)} workflows named {self.workflow!r} (gh needs exactly one)")
        return f"{base}/{ids[0]}/runs"


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
