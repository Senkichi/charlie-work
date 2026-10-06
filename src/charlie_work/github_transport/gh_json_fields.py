"""The ``gh --json`` dialect, expressed as GraphQL documents (ADR-0006, G4).

Every ``gh issue|pr list|view --json F`` read the orchestrator makes has a
field list (``ISSUE_VIEW_FIELDS``, ``PR_LIST_FIELDS`` ...). This module turns
such a list into one GraphQL document (one fragment per gh field name) and
turns the response nodes back into the exact JSON shape ``gh`` printed, so no
consumer of ``issue_list`` / ``pr_view`` / ``pr_checks`` changes.

The mapping is pinned by recorded fixtures (``tests/fixtures/gh_json/``): each
holds ``gh ... --json`` output and the GraphQL response for the same object,
and ``tests/test_github_transport_gh_json_fields.py`` asserts the normalizer
turns the second into the first.

An unknown field name raises ``UnknownFieldError`` when the document is built,
so a typo in a ``*_FIELDS`` constant fails at import-time tests, not in
production (this replaces probing ``gh`` with an invalid field).

Pure: no I/O, no transport. Imported by ``json_read`` (the executor).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

Resource = Literal["issue", "pr"]

# gh prints a zero time.Time for an absent check timestamp.
_ZERO_TIME = "0001-01-01T00:00:00Z"
_PAGE = 100


class UnknownFieldError(ValueError):
    """A ``--json`` field name the dialect has no GraphQL fragment for."""


class IncompletePageError(ValueError):
    """A connection reports a next page the node cannot be followed to."""


@dataclass(frozen=True)
class FieldSpec:
    """One gh field name: its GraphQL selection and the node -> gh value map.

    ``paged`` marks a nested connection that gh itself walks to the end
    (``comments``, ``closingIssuesReferences``, ``statusCheckRollup``): its
    selection carries ``pageInfo`` and ``gh_json_pages`` fetches the rest.
    """

    selection: str
    extract: Callable[[dict[str, Any]], Any]
    paged: bool = False


_PAGE_INFO = "pageInfo{hasNextPage endCursor}"


def _first(after: bool) -> str:
    """The connection arguments of the first page, or of a follow-up page."""
    return f"first:{_PAGE},after:$after" if after else f"first:{_PAGE}"


def _nodes(node: dict[str, Any], key: str) -> list[Any]:
    connection = node.get(key)
    items = connection.get("nodes") if isinstance(connection, dict) else None
    return [item for item in items if item is not None] if isinstance(items, list) else []


def _scalar(name: str) -> FieldSpec:
    return FieldSpec(name, lambda node: node.get(name))


def _string(name: str) -> FieldSpec:
    """A GraphQL nullable that gh prints as ``""`` when absent."""
    return FieldSpec(name, lambda node: node.get(name) or "")


def _author(node: dict[str, Any]) -> dict[str, Any] | None:
    actor = node.get("author")
    if not isinstance(actor, dict):
        return None
    is_bot = actor.get("__typename") == "Bot"
    login = actor.get("login")
    return {
        "id": actor.get("id"),
        "is_bot": is_bot,
        "login": f"app/{login}" if is_bot and login else login,
        "name": actor.get("name") or "",
    }


_AUTHOR = FieldSpec(
    "author{__typename login ... on User{id name} ... on Bot{id}}",
    _author,
)

_LABELS = FieldSpec(
    f"labels(first:{_PAGE}){{nodes{{id name description color}}}}",
    lambda node: [
        {
            "id": label.get("id"),
            "name": label.get("name"),
            "description": label.get("description") or "",
            "color": label.get("color"),
        }
        for label in _nodes(node, "labels")
    ],
)

_ASSIGNEES = FieldSpec(
    f"assignees(first:{_PAGE}){{nodes{{id login name databaseId}}}}",
    lambda node: [
        {
            "id": user.get("id"),
            "login": user.get("login"),
            "name": user.get("name") or "",
            "databaseId": user.get("databaseId"),
        }
        for user in _nodes(node, "assignees")
    ],
)


def _reaction_groups(comment: dict[str, Any]) -> list[dict[str, Any]]:
    groups = comment.get("reactionGroups")
    out: list[dict[str, Any]] = []
    for group in groups if isinstance(groups, list) else []:
        users = group.get("users") if isinstance(group, dict) else None
        total = users.get("totalCount") if isinstance(users, dict) else 0
        if total:
            out.append({"content": group.get("content"), "users": {"totalCount": total}})
    return out


_COMMENT_NODE = (
    "id author{login} authorAssociation body createdAt "
    "includesCreatedEdit isMinimized minimizedReason url viewerDidAuthor "
    "reactionGroups{content users{totalCount}}"
)


def comments_selection(after: bool = False) -> str:
    return f"comments({_first(after)}){{nodes{{{_COMMENT_NODE}}}{_PAGE_INFO}}}"


_COMMENTS = FieldSpec(
    comments_selection(),
    lambda node: [
        {
            "id": comment.get("id"),
            "author": {"login": (comment.get("author") or {}).get("login")},
            "authorAssociation": comment.get("authorAssociation"),
            "body": comment.get("body"),
            "createdAt": comment.get("createdAt"),
            "includesCreatedEdit": comment.get("includesCreatedEdit"),
            "isMinimized": comment.get("isMinimized"),
            "minimizedReason": comment.get("minimizedReason") or "",
            "reactionGroups": _reaction_groups(comment),
            "url": comment.get("url"),
            "viewerDidAuthor": comment.get("viewerDidAuthor"),
        }
        for comment in _nodes(node, "comments")
    ],
    paged=True,
)

_CLOSING_NODE = "id number url repository{id name owner{id login}}"


def closing_issues_selection(after: bool = False) -> str:
    return f"closingIssuesReferences({_first(after)}){{nodes{{{_CLOSING_NODE}}}{_PAGE_INFO}}}"


_CLOSING_ISSUES = FieldSpec(
    closing_issues_selection(),
    lambda node: [
        {
            "id": ref.get("id"),
            "number": ref.get("number"),
            "repository": {
                "id": (ref.get("repository") or {}).get("id"),
                "name": (ref.get("repository") or {}).get("name"),
                "owner": {
                    "id": ((ref.get("repository") or {}).get("owner") or {}).get("id"),
                    "login": ((ref.get("repository") or {}).get("owner") or {}).get("login"),
                },
            },
            "url": ref.get("url"),
        }
        for ref in _nodes(node, "closingIssuesReferences")
    ],
    paged=True,
)

# CheckRun / StatusContext, the two members of a rollup context (gh's names).
_CONTEXT_SELECTION = (
    "__typename "
    "... on CheckRun{name status conclusion detailsUrl startedAt completedAt "
    "checkSuite{workflowRun{event workflow{name}}}} "
    "... on StatusContext{context state targetUrl createdAt description}"
)


def _rollup_entry(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("__typename") == "StatusContext":
        return {
            "__typename": "StatusContext",
            "context": context.get("context"),
            "startedAt": context.get("createdAt"),
            "state": context.get("state"),
            "targetUrl": context.get("targetUrl"),
        }
    run = ((context.get("checkSuite") or {}).get("workflowRun")) or {}
    return {
        "__typename": "CheckRun",
        "completedAt": context.get("completedAt") or _ZERO_TIME,
        "conclusion": context.get("conclusion") or "",
        "detailsUrl": context.get("detailsUrl"),
        "name": context.get("name"),
        "startedAt": context.get("startedAt") or _ZERO_TIME,
        "status": context.get("status"),
        "workflowName": ((run.get("workflow") or {}).get("name")) or "",
    }


def _rollup_contexts(node: dict[str, Any]) -> list[dict[str, Any]]:
    rollup = node.get("statusCheckRollup")
    contexts = rollup.get("contexts") if isinstance(rollup, dict) else None
    return [c for c in _nodes({"contexts": contexts}, "contexts") if isinstance(c, dict)]


def rollup_selection(after: bool = False) -> str:
    return (
        f"statusCheckRollup{{contexts({_first(after)}){{"
        f"nodes{{{_CONTEXT_SELECTION}}}{_PAGE_INFO}}}}}"
    )


_ROLLUP = FieldSpec(
    rollup_selection(),
    lambda node: [_rollup_entry(c) for c in _rollup_contexts(node)],
    paged=True,
)

_COMMON: dict[str, FieldSpec] = {
    "number": _scalar("number"),
    "title": _scalar("title"),
    "url": _scalar("url"),
    "body": _scalar("body"),
    "state": _scalar("state"),
    "createdAt": _scalar("createdAt"),
    "updatedAt": _scalar("updatedAt"),
    "closedAt": _scalar("closedAt"),
    "author": _AUTHOR,
    "labels": _LABELS,
}

_REGISTRY: dict[str, dict[str, FieldSpec]] = {
    "issue": {
        **_COMMON,
        "assignees": _ASSIGNEES,
        "comments": _COMMENTS,
    },
    "pr": {
        **_COMMON,
        "isDraft": _scalar("isDraft"),
        "headRefName": _scalar("headRefName"),
        "baseRefName": _scalar("baseRefName"),
        "headRefOid": _scalar("headRefOid"),
        "isCrossRepository": _scalar("isCrossRepository"),
        "mergeable": _scalar("mergeable"),
        "mergeStateStatus": _scalar("mergeStateStatus"),
        "mergedAt": _scalar("mergedAt"),
        "additions": _scalar("additions"),
        "deletions": _scalar("deletions"),
        "reviewDecision": _string("reviewDecision"),
        "statusCheckRollup": _ROLLUP,
        "closingIssuesReferences": _CLOSING_ISSUES,
    },
}

_STATE_ENUMS = {
    "issue": ("OPEN", "CLOSED"),
    "pr": ("OPEN", "CLOSED", "MERGED"),
}


def paged_fields(resource: str) -> tuple[str, ...]:
    """The field names of *resource* whose connection gh pages to the end."""
    return tuple(name for name, spec in _REGISTRY.get(resource, {}).items() if spec.paged)


def field_names(fields: str | Iterable[str]) -> tuple[str, ...]:
    """Split a gh ``--json`` field list (``"a,b,c"``) into its names."""
    if isinstance(fields, str):
        parts = fields.split(",")
    else:
        parts = list(fields)
    return tuple(part.strip() for part in parts if part.strip())


def _specs(resource: str, fields: str | Iterable[str]) -> list[tuple[str, FieldSpec]]:
    registry = _REGISTRY.get(resource)
    if registry is None:
        raise UnknownFieldError(f"unknown resource {resource!r}")
    names = field_names(fields)
    if not names:
        raise UnknownFieldError(f"empty field list for {resource}")
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise UnknownFieldError(f"unknown --json field(s) for {resource}: {', '.join(unknown)}")
    return [(name, registry[name]) for name in names]


def selection_for(resource: str, fields: str | Iterable[str]) -> str:
    """The GraphQL selection set (without braces) for *fields*."""
    seen: list[str] = []
    specs = _specs(resource, fields)
    if any(spec.paged for _name, spec in specs):
        seen.append("id")  # the node id a follow-up page is fetched by
    for _name, spec in specs:
        if spec.selection not in seen:
            seen.append(spec.selection)
    return " ".join(seen)


def normalize_node(
    resource: str, node: dict[str, Any], fields: str | Iterable[str]
) -> dict[str, Any]:
    """One GraphQL node -> the dict ``gh <resource> ... --json fields`` printed."""
    return {name: spec.extract(node) for name, spec in _specs(resource, fields)}


def normalize_nodes(
    resource: str, nodes: Iterable[Any], fields: str | Iterable[str]
) -> list[dict[str, Any]]:
    return [normalize_node(resource, n, fields) for n in nodes if isinstance(n, dict)]


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

Shape = Literal["list", "view", "search"]


def document_for(resource: str, fields: str | Iterable[str], shape: Shape) -> str:
    """The GraphQL query for one read shape of *resource*.

    ``list``: ``repository.issues|pullRequests`` (variables ``owner``, ``name``,
    ``states``, ``first``, ``after``; issues add ``labels``, PRs add ``head``);
    ``view``: one object by ``number``; ``search``: PRs matching ``q``.
    Every list document selects ``pageInfo`` so ``paginate_graphql`` can walk it.
    """
    sel = selection_for(resource, fields)
    if shape == "view":
        if resource == "issue":
            # ``gh issue view`` accepts a pull request number too (a PR is an issue).
            body = f"issueOrPullRequest(number:$number){{... on Issue{{{sel}}} ... on PullRequest{{{sel}}}}}"
        else:
            body = f"pullRequest(number:$number){{{sel}}}"
        return (
            "query($owner:String!,$name:String!,$number:Int!){"
            f"repository(owner:$owner,name:$name){{{body}}}}}"
        )
    if shape == "search":
        if resource != "pr":
            raise UnknownFieldError("search documents exist for pull requests only")
        return (
            "query($q:String!,$first:Int!){"
            f"search(query:$q,type:ISSUE,first:$first){{nodes{{... on PullRequest{{{sel}}}}}}}}}"
        )
    state_enum = "IssueState" if resource == "issue" else "PullRequestState"
    if resource == "issue":
        head = "$labels:[String!],"
        args = "states:$states,labels:$labels"
        connection = "issues"
    else:
        head = "$head:String,"
        args = "states:$states,headRefName:$head"
        connection = "pullRequests"
    return (
        f"query($owner:String!,$name:String!,$states:[{state_enum}!],{head}"
        "$first:Int!,$after:String){"
        f"repository(owner:$owner,name:$name){{{connection}({args},first:$first,after:$after,"
        "orderBy:{field:CREATED_AT,direction:DESC}){"
        f"nodes{{{sel}}}pageInfo{{hasNextPage endCursor}}}}}}}}"
    )


def checks_document() -> str:
    """The rollup query behind ``gh pr checks``: one page of a PR's check contexts.

    ``gh pr checks`` walks every page of contexts, so this document takes
    ``$after`` and selects ``pageInfo``; ``JsonRead._checks`` follows it.
    """
    return (
        "query($owner:String!,$name:String!,$number:Int!,$after:String){"
        "repository(owner:$owner,name:$name){pullRequest(number:$number){"
        f"{rollup_selection(after=True)}}}}}}}"
    )


def checks_next_cursor(pull_request: dict[str, Any]) -> str | None:
    """``endCursor`` of the next contexts page, or ``None`` on the last page.

    ``hasNextPage`` without a usable ``endCursor`` is a GitHub contract
    violation: it raises ``IncompletePageError`` rather than ending the walk
    as if the page were the last.
    """
    rollup = pull_request.get("statusCheckRollup")
    contexts = rollup.get("contexts") if isinstance(rollup, dict) else None
    info = contexts.get("pageInfo") if isinstance(contexts, dict) else None
    if not isinstance(info, dict) or not info.get("hasNextPage"):
        return None
    cursor = info.get("endCursor")
    if not isinstance(cursor, str) or not cursor:
        raise IncompletePageError("a connection has a next page but no end cursor")
    return cursor


def states_for(resource: str, state: str) -> list[str] | None:
    """gh's ``--state open|closed|merged|all`` as a GraphQL enum list (``None`` = all)."""
    value = state.strip().lower()
    if value == "all":
        return None
    enums = _STATE_ENUMS[resource]
    wanted = value.upper()
    if wanted not in enums:
        raise UnknownFieldError(f"unsupported --state {state!r} for {resource}")
    return [wanted]


# ---------------------------------------------------------------------------
# `gh pr checks`
# ---------------------------------------------------------------------------

_FAIL_STATES = frozenset({"ERROR", "FAILURE", "TIMED_OUT", "ACTION_REQUIRED"})
_SKIP_STATES = frozenset({"SKIPPED", "NEUTRAL"})


def bucket_for(state: str) -> str:
    """Port of gh's ``pr checks`` aggregation: state -> pass/fail/pending/skipping/cancel."""
    if state == "SUCCESS":
        return "pass"
    if state in _SKIP_STATES:
        return "skipping"
    if state in _FAIL_STATES:
        return "fail"
    if state == "CANCELLED":
        return "cancel"
    return "pending"


_CHECK_FIELDS = (
    "bucket",
    "completedAt",
    "description",
    "event",
    "link",
    "name",
    "startedAt",
    "state",
    "workflow",
)


def normalize_checks(contexts: Iterable[Any], fields: str | Iterable[str]) -> list[dict[str, Any]]:
    """Rollup contexts -> ``gh pr checks --json fields``.

    gh sorts newest-first by start time and keeps one entry per check run
    (keyed by name, workflow and triggering event) or per status context
    (keyed by its context, in a separate namespace), then derives ``state`` (a
    CheckRun's conclusion once it is COMPLETED, else its status) and ``bucket``
    from it.  Keying on the bare name would hide a failing run of one workflow
    behind a newer passing run of another.
    """
    names = field_names(fields)
    unknown = [name for name in names if name not in _CHECK_FIELDS]
    if unknown:
        raise UnknownFieldError(f"unknown --json field(s) for checks: {', '.join(unknown)}")
    entries = [_check_entry(c) for c in contexts if isinstance(c, dict)]
    entries.sort(key=lambda e: e["startedAt"], reverse=True)
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, Any]] = []
    for entry in entries:
        if entry["_key"] in seen:
            continue
        seen.add(entry["_key"])
        unique.append(entry)
    return [{name: entry[name] for name in names} for entry in unique]


def _check_entry(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("__typename") == "StatusContext":
        state = str(context.get("state") or "")
        name = context.get("context") or ""
        return {
            "_key": ("status", name),
            "name": name,
            "state": state,
            "bucket": bucket_for(state),
            "link": context.get("targetUrl") or "",
            "startedAt": context.get("createdAt") or _ZERO_TIME,
            "completedAt": _ZERO_TIME,
            "description": context.get("description") or "",
            "event": "",
            "workflow": "",
        }
    status = str(context.get("status") or "")
    conclusion = str(context.get("conclusion") or "")
    state = conclusion if status == "COMPLETED" else status
    run = ((context.get("checkSuite") or {}).get("workflowRun")) or {}
    name = context.get("name") or ""
    event = run.get("event") or ""
    workflow = ((run.get("workflow") or {}).get("name")) or ""
    return {
        "_key": ("run", name, workflow, event),
        "name": name,
        "state": state,
        "bucket": bucket_for(state),
        "link": context.get("detailsUrl") or "",
        "startedAt": context.get("startedAt") or _ZERO_TIME,
        "completedAt": context.get("completedAt") or _ZERO_TIME,
        "description": "",
        "event": event,
        "workflow": workflow,
    }


def checks_contexts(pull_request: dict[str, Any]) -> list[dict[str, Any]]:
    """The raw rollup contexts of a ``pullRequest`` node."""
    return _rollup_contexts(pull_request)


__all__ = [
    "FieldSpec",
    "IncompletePageError",
    "UnknownFieldError",
    "bucket_for",
    "checks_contexts",
    "checks_next_cursor",
    "checks_document",
    "document_for",
    "field_names",
    "normalize_checks",
    "normalize_node",
    "normalize_nodes",
    "paged_fields",
    "selection_for",
    "states_for",
]
