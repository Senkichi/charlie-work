"""The legacy ``GitHub.run(argv)`` edge (ADR-0006, design section 0 item 3).

``GitHubLike.run(argv)`` stays because ``FakeGitHub`` is unchanged and a
handful of callers outside the capability layer still pass gh argv. This
module is the closed, test-pinned table that maps those argv shapes onto
requests. It is an edge adapter for old callers, not the contract: each
capability migrates to typed requests and the table only shrinks
(``tests/test_github_transport_legacy_argv.py`` pins the row count).

Two outcomes for every argv:

* a ``RestRequest`` / ``GraphQLRequest`` for the ``gh api`` REST-GET and
  GraphQL-query shapes the HTTP path has always served, plus the two
  ``gh run cancel|rerun`` mutations, a ``JsonRead`` / ``RunListRead`` for
  the ``gh issue|pr|run ... --json`` reads (executed over GraphQL / REST,
  ``json_read.py``), or
* a ``LegacyCli``, the verbatim ``gh <args>`` passthrough for everything
  else. ``LegacyCli`` is a transitional request: it always routes to the gh
  adapter and is deleted when the last capability has moved off
  ``run(argv)``.

Mutation-ness of a passthrough argv is the old prefix allowlist, moved here
verbatim (``legacy_is_mutating``): a typed request derives it from its
method or operation instead.

This module imports only ``request``; the transport package sits below the
capability layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

from .request import ACCEPT_JSON, GraphQLRequest, Request, RestRequest

if TYPE_CHECKING:  # json_read imports the guard, which imports this module
    from .json_read import JsonRead, RunListRead

# Endpoint path suffix whose request follows a redirect: the Actions job-log
# endpoint answers 302 to a signed URL with a raw-text body (gt-design G3).
_REDIRECT_PATH_SUFFIXES = ("/logs",)
_GRAPHQL_KNOWN_FIELDS = ("query", "owner", "name")
_READONLY_PREFIXES = (
    "issue list",
    "issue view",
    "pr list",
    "pr view",
    "pr diff",
    "pr checks",
    "label list",
    "auth status",
)


# ---------------------------------------------------------------------------
# mutation classification (moved verbatim from github_capabilities/_base.py)
# ---------------------------------------------------------------------------


def graphql_field_value(args: list[str], field: str) -> str | None:
    """Return the raw value of a `gh api graphql -f/--field name=value` pair.

    Handles detached (`-f query=...`), attached shorthand (`-fquery=...`),
    and `--field=query=...` spellings (#919). Returns `None` if the field is
    absent or its value is missing.
    """
    for i, arg in enumerate(args):
        if arg in ("-f", "--raw-field", "-F", "--field"):
            next_arg = args[i + 1] if i + 1 < len(args) else ""
            if "=" in next_arg and next_arg.split("=", 1)[0] == field:
                return next_arg.split("=", 1)[1]
        elif arg.startswith("-f") and len(arg) > 2:
            rest = arg[2:].lstrip("=")
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
        elif arg.startswith(("--field=", "--raw-field=")):
            rest = arg.split("=", 1)[1]
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
    return None


def is_graphql_query(args: list[str]) -> bool:
    """A `gh api graphql -f query='query { ... }'` is a read-only query.

    Fails closed: only an operation that *starts* with the GraphQL `query`
    keyword is treated as read-only. `mutation` or anything unparseable is
    classified as mutating so a stray write never runs under `--dry-run`.
    """
    if len(args) < 2 or args[0] != "api" or args[1] != "graphql":
        return False
    query = graphql_field_value(args, "query")
    if not query:
        return False
    return query.lstrip()[:5].lower() == "query"


def api_is_mutating(args: list[str]) -> bool:
    """Classify a `gh api` invocation, for the --dry-run gate.

    `gh api` defaults to GET, so a bare `gh api <path>` is read-only. The
    classification keys off whether a method is *named* and fails CLOSED when
    a method flag is present but its value cannot be extracted (#914, #917).
    Request parameters (`-f`, `-F`, `--field`, `--raw-field`, `--input`)
    switch gh to POST, so they count as mutating too (#919). A read-only
    `gh api graphql -f query='query { ... }'` is an exception (#923).
    """
    if is_graphql_query(args):
        return False

    for i, arg in enumerate(args):
        if arg in ("-X", "--method"):
            method = args[i + 1] if i + 1 < len(args) else ""
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif arg.startswith("-X"):
            # pflag shorthand accepts an attached value: `-XDELETE` and `-X=DELETE`.
            method = arg[2:].lstrip("=")
        else:
            continue
        # A named-but-unparseable method is not evidence of a read; fail closed.
        return not method or method.upper() not in ("GET", "HEAD")
    param_prefixes = ("--raw-field", "--field", "--input")
    return any(arg.startswith(param_prefixes) or arg[:2] in ("-f", "-F") for arg in args)


def legacy_is_mutating(args: list[str]) -> bool:
    """Whether the gh argv (without the leading ``gh``) may change state."""
    if not args:
        return False
    text = " ".join(args)
    if text.startswith("api"):
        return api_is_mutating(args)
    return not any(text.startswith(prefix) for prefix in _READONLY_PREFIXES)


# ---------------------------------------------------------------------------
# the request values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyCli:
    """A verbatim ``gh <args>`` invocation (transitional passthrough)."""

    args: tuple[str, ...]
    long_call: bool = False

    @property
    def is_mutation(self) -> bool:
        return legacy_is_mutating(list(self.args))

    def describe(self) -> str:
        return " ".join(("gh", *self.args))


@dataclass(frozen=True)
class Translated:
    """One table lookup: the request to send and whether to follow pages."""

    request: "Request | LegacyCli | JsonRead | RunListRead"
    paginate: bool = False


def _split_route(path: str) -> tuple[str, list[tuple[str, str]]]:
    route, _, query = path.partition("?")
    return route, parse_qsl(query, keep_blank_values=True)


def _rest_row(args: list[str], long_call: bool) -> Translated | None:
    """`gh api [--paginate] [-H Accept: X] PATH`: the REST-GET row.

    Fails closed: any other flag, a second positional, or a header other than
    ``Accept`` is not in the table.
    """
    path: str | None = None
    accept = ACCEPT_JSON
    paginate = False
    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "--paginate":
            paginate = True
            i += 1
            continue
        if arg == "-H":
            if i + 1 >= len(args):
                return None
            header_text = args[i + 1]
            i += 2
        elif arg.startswith("-H") and len(arg) > 2:
            header_text = arg[2:]
            i += 1
        elif arg.startswith("-"):
            return None
        else:
            if path is not None:
                return None
            path = arg
            i += 1
            continue
        name, sep, value = header_text.partition(":")
        if not sep or name.strip().lower() != "accept":
            return None
        accept = value.strip()
    if path is None:
        return None
    route, query = _split_route(path)
    follow = route.endswith(_REDIRECT_PATH_SUFFIXES)
    try:
        request = RestRequest(
            "GET",
            route,
            query=tuple(query),
            accept=accept,
            long_call=long_call,
            follow_redirect=follow,
        )
    except ValueError:
        return None
    return Translated(request, paginate=paginate)


def _graphql_args_are_recognized(args: list[str]) -> bool:
    """Every arg after ``api graphql`` is a ``-f``/``--field`` spelling of one
    of the fields the row understands (query, owner, name)."""
    i = 2
    while i < len(args):
        arg = args[i]
        if arg in ("-f", "--raw-field", "-F", "--field"):
            if i + 1 >= len(args):
                return False
            name = args[i + 1].split("=", 1)[0] if "=" in args[i + 1] else None
            if name not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 2
            continue
        if arg.startswith("-f") and len(arg) > 2:
            rest = arg[2:].lstrip("=")
            name = rest.split("=", 1)[0] if "=" in rest else None
            if name not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 1
            continue
        if arg.startswith(("--field=", "--raw-field=")):
            rest = arg.split("=", 1)[1]
            name = rest.split("=", 1)[0] if "=" in rest else None
            if name not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 1
            continue
        return False
    return True


def _graphql_row(args: list[str], long_call: bool) -> Translated | None:
    if not _graphql_args_are_recognized(args):
        return None
    query = graphql_field_value(args, "query") or ""
    variables = {
        name: value
        for name in ("owner", "name")
        if (value := graphql_field_value(args, name)) is not None
    }
    try:
        return Translated(GraphQLRequest.of(query, variables, long_call=long_call))
    except ValueError:
        return None


def _run_row(args: list[str], long_call: bool) -> Translated | None:
    """`gh run cancel ID` / `gh run rerun ID [--failed]`: the two run mutations
    that callers outside the capability layer still spell as argv (design a.2).
    """
    if len(args) < 3 or args[0] != "run" or not args[2].isdigit():
        return None
    run_id, rest = args[2], args[3:]
    if args[1] == "cancel" and not rest:
        suffix = "cancel"
    elif args[1] == "rerun" and not rest:
        suffix = "rerun"
    elif args[1] == "rerun" and rest == ["--failed"]:
        suffix = "rerun-failed-jobs"
    else:
        return None
    route = f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/{suffix}"
    return Translated(RestRequest("POST", route, long_call=long_call))


_STATES = frozenset({"open", "closed", "merged", "all"})


def _flag_values(
    args: list[str], start: int, allowed: frozenset[str]
) -> dict[str, list[str]] | None:
    """``--flag value`` pairs from *start* on; None for anything not in *allowed*
    (a bare token, a ``--flag=value`` spelling, a missing value)."""
    found: dict[str, list[str]] = {}
    i = start
    while i < len(args):
        flag = args[i]
        if flag not in allowed or i + 1 >= len(args):
            return None
        found.setdefault(flag, []).append(args[i + 1])
        i += 2
    return found


def _single(values: dict[str, list[str]], flag: str) -> str | None:
    seen = values.get(flag)
    return seen[-1] if seen else None


def _limit(values: dict[str, list[str]], default: int) -> int | None:
    raw = _single(values, "--limit")
    if raw is None:
        return default
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def _json_row(args: list[str], long_call: bool) -> Translated | None:
    """`gh issue|pr view|list|checks ... --json F`: the dialect reads.

    ``--json F`` must be the last pair. Lists take ``--state/--limit/--label/
    --head/--search``; anything else is not in the table (fails closed).
    """
    from .json_read import JsonRead

    if len(args) < 4 or args[0] not in ("issue", "pr") or args[-2] != "--json":
        return None
    resource, verb, fields = args[0], args[1], args[-1]
    body = args[:-2]
    if verb in ("view", "checks") and len(body) == 3 and body[2].isdigit():
        if verb == "checks" and resource != "pr":
            return None
        return Translated(
            JsonRead(resource, verb, fields, number=int(body[2]), long_call=long_call)  # type: ignore[arg-type]
        )
    if verb != "list":
        return None
    values = _flag_values(
        body, 2, frozenset({"--state", "--limit", "--label", "--head", "--search"})
    )
    if values is None:
        return None
    limit = _limit(values, 30)
    state = _single(values, "--state") or "open"
    if limit is None or state not in _STATES:
        return None
    head, search = _single(values, "--head"), _single(values, "--search")
    labels = tuple(values.get("--label", ()))
    if search is not None:
        if resource != "pr" or head is not None or labels:
            return None
        return Translated(
            JsonRead(
                "pr",
                "search",
                fields,
                state=state,
                search=search,
                limit=limit,
                long_call=long_call,
            )
        )
    if head is not None and resource != "pr" or labels and resource != "issue":
        return None
    return Translated(
        JsonRead(
            resource,  # type: ignore[arg-type]
            "list",
            fields,
            state=state,
            labels=labels,
            head=head,
            limit=limit,
            long_call=long_call,
        )
    )


def _run_list_row(args: list[str], long_call: bool) -> Translated | None:
    """`gh run list [--workflow W] [--branch B] [--status S] [--limit N] --json F`."""
    from .json_read import RunListRead

    if len(args) < 4 or args[1] != "list" or args[-2] != "--json":
        return None
    values = _flag_values(
        args[:-2], 2, frozenset({"--workflow", "--branch", "--status", "--limit"})
    )
    if values is None:
        return None
    limit = _limit(values, 20)
    if limit is None:
        return None
    return Translated(
        RunListRead(
            fields=args[-1],
            workflow=_single(values, "--workflow"),
            branch=_single(values, "--branch"),
            status=_single(values, "--status"),
            limit=limit,
            long_call=long_call,
        )
    )


def request_for_argv(
    args: list[str], *, long_call: bool = False, use_requests: bool = True
) -> Translated:
    """Map gh argv (without the leading ``gh``) onto a request.

    ``use_requests=False`` (the ``gh_transport: gh`` kill switch) forces the
    verbatim passthrough, so the old gh behaviour is exactly preserved while
    the switch is on.
    """
    passthrough = Translated(LegacyCli(tuple(args), long_call))
    if use_requests and args and args[0] in ("issue", "pr"):
        return _json_row(args, long_call) or passthrough
    if use_requests and args and args[0] == "run":
        return _run_row(args, long_call) or _run_list_row(args, long_call) or passthrough
    if not use_requests or not args or args[0] != "api" or api_is_mutating(args):
        return passthrough
    if is_graphql_query(args):
        return _graphql_row(args, long_call) or passthrough
    return _rest_row(args, long_call) or passthrough


__all__ = [
    "LegacyCli",
    "Translated",
    "api_is_mutating",
    "graphql_field_value",
    "is_graphql_query",
    "legacy_is_mutating",
    "request_for_argv",
]
