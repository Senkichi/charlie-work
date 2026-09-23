"""Translate `gh api` REST-GET / graphql argv shapes into HTTP request plans
(issue #1834).

Scope (disclosed, deliberate narrowing -- see the PR description for the
full rationale): this translator only covers `gh api <path>` GET calls
(optionally `--paginate`, optionally `-H "Header: value"`) and `gh api
graphql -f query=...` *read-only* queries. Every `--json` structured
subcommand (`issue list`, `pr view`, `pr checks`, `label list`, `run
list`, ...) and every mutation is left untranslated on purpose:

* Mutations are excluded because `run()`'s error-translation and circuit
  breaker paths must stay purely input->output, never re-derive gh's own
  request-building for a write it does not attempt to reproduce.
* `--json` structured commands are excluded because `gh` builds a
  *different* GraphQL query per subcommand-plus-field-list combination
  internally (not part of its documented, stable interface), so faithfully
  reproducing them here would mean silently vendoring gh's own query
  construction and accepting drift risk with no compile-time signal when
  gh's internals change. `gh api graphql` itself (the explicit, versioned
  GraphQL surface) has no such problem, which is why it IS translated.

Both exclusions fall back to `gh` by construction: `is_http_candidate`
returning False is the only thing `GitHub.run()` consults before choosing
the subprocess path, so nothing needs to remember these two rules anywhere
else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._base import _api_is_mutating, _graphql_field_value, _is_graphql_query

# Endpoint path suffixes gh handles with response-shape behavior this
# translator does not replicate (redirect-follow raw-text bodies, etc.) --
# excluded from candidacy so they always fall back to gh by construction.
# `/logs` is the Actions job-log endpoint (`workflow._actions_job_log_text`):
# GitHub responds with a 302 to a time-limited storage URL and gh follows it,
# returning the raw log text rather than a JSON body.
_EXCLUDED_PATH_SUFFIXES = ("/logs",)


@dataclass(frozen=True)
class HttpRequestPlan:
    """One translated HTTP request, ready for `http_transport.py` to send."""

    method: str  # "GET" for REST, "POST" for graphql (GitHub's graphql
    # endpoint is POST-only even for a read-only query -- this is a transport
    # detail of GitHub's API, not a mutation classification, and is unrelated
    # to `_is_mutating`'s dry-run semantics).
    path: str  # already {owner}/{repo}-substituted, leading "/"
    headers: tuple[tuple[str, str], ...]
    body: bytes | None
    paginate: bool


@dataclass(frozen=True)
class _RestArgs:
    """The parsed shape of a REST-GET `gh api` call: the endpoint plus the
    two flags this module understands how to translate."""

    path: str
    headers: tuple[tuple[str, str], ...]
    paginate: bool


def _rest_path(args: list[str]) -> str | None:
    """Return the REST endpoint -- the single non-flag positional after
    `"api"` -- or `None` if the shape is not recognized.

    The endpoint is NOT always `args[1]`: the only `--paginate` call site in
    this codebase, `workflow._gh_api_list`, calls
    `["api", "--paginate", path]`, which puts a flag before the path. This
    walks the whole argv instead of assuming a fixed position, and returns
    `None` (not a candidate) when there is no positional, more than one, or
    any flag other than `--paginate`/`-H` -- see `_parse_rest_args`, which
    this delegates to.
    """
    parsed = _parse_rest_args(args)
    return parsed.path if parsed is not None else None


def _parse_rest_args(args: list[str]) -> _RestArgs | None:
    """Parse the REST-GET argv (after `"api"`), returning `None` unless
    every arg is the single endpoint positional, `--paginate`, or a
    `-H <value>` / `-H<value>` header -- the only shapes every `gh api` GET
    call site in this codebase uses.

    Fails CLOSED on anything else: an unrecognized flag such as `--slurp`,
    `-i`/`--include`, `-q`/`--jq`, `--template`/`-t`, `--method GET`,
    `--hostname`, `--cache`, `-p`/`--preview`, or a second positional,
    returns `None` rather than being silently ignored -- those shapes would
    otherwise be served over HTTP with output that differs from `gh`'s own
    (e.g. `--include` prepends response headers to stdout; `gh` never
    reaches this translator at all for them since `is_http_candidate`
    rejects the call up front).
    """
    path: str | None = None
    headers: list[tuple[str, str]] = []
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
        if ":" in header_text:
            name, _, value = header_text.partition(":")
            headers.append((name.strip(), value.strip()))
    if path is None:
        return None
    return _RestArgs(path=path, headers=tuple(headers), paginate=paginate)


_GRAPHQL_KNOWN_FIELDS = ("query", "owner", "name")


def _graphql_field_name(field_arg: str) -> str | None:
    if "=" in field_arg:
        return field_arg.split("=", 1)[0]
    return None


def _graphql_args_are_recognized(args: list[str]) -> bool:
    """True when every arg after `"api" "graphql"` is a `-f`/`--field`
    field-value spelling for one of `query`/`owner`/`name` -- the only
    fields `_build_graphql_plan` reads and the only ones
    `Transport._graphql_query` (the sole `gh api graphql` call site in this
    codebase) ever passes.

    Mirrors `_graphql_field_value`'s own parsing (detached `-f`/
    `--raw-field`/`-F`/`--field`, attached `-fname=value`, and
    `--field=`/`--raw-field=`) but fails CLOSED: a header, `--jq`, an
    unrecognized field name, or a malformed/missing value makes the whole
    call not a candidate, rather than `_build_graphql_plan` silently
    dropping the field it doesn't recognize.
    """
    i = 2
    while i < len(args):
        arg = args[i]
        if arg in ("-f", "--raw-field", "-F", "--field"):
            if i + 1 >= len(args):
                return False
            field = _graphql_field_name(args[i + 1])
            if field not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 2
            continue
        if arg.startswith("-f") and len(arg) > 2:
            field = _graphql_field_name(arg[2:].lstrip("="))
            if field not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 1
            continue
        if arg.startswith(("--field=", "--raw-field=")):
            field = _graphql_field_name(arg.split("=", 1)[1])
            if field not in _GRAPHQL_KNOWN_FIELDS:
                return False
            i += 1
            continue
        return False
    return True


def is_http_candidate(args: list[str]) -> bool:
    """True when `args` (argv after the leading `gh` token) is a REST-GET or
    GraphQL-query `gh api` call this module can build an HTTP request for.

    Callers must not call `build_request_plan` unless this returns True.

    Fails CLOSED, for both shapes: an arg this module does not specifically
    recognize is never silently ignored (dropped) and served over HTTP with
    output that could differ from `gh`'s own -- it makes the whole call not
    a candidate, falling back to the real `gh` subprocess instead. See
    `_parse_rest_args`/`_graphql_args_are_recognized` for the exact per-shape
    rules.
    """
    if not args or args[0] != "api":
        return False
    if _api_is_mutating(args):
        return False
    if _is_graphql_query(args):
        return _graphql_args_are_recognized(args)
    path = _rest_path(args)
    if path is None:
        return False
    if any(path.split("?", 1)[0].endswith(suffix) for suffix in _EXCLUDED_PATH_SUFFIXES):
        return False
    return True


def build_request_plan(args: list[str], owner: str, repo: str) -> HttpRequestPlan:
    """Build the HTTP request plan for an `is_http_candidate(args) is True` argv.

    `owner`/`repo` come from `Transport._repo_owner_name()` -- the same
    resolution `_graphql_query`'s `-f owner=...`/`-f name=...` fields and
    every literal `{owner}`/`{repo}` REST-path placeholder already depend on.
    """
    if _is_graphql_query(args):
        return _build_graphql_plan(args, owner, repo)
    return _build_rest_get_plan(args, owner, repo)


def _build_graphql_plan(args: list[str], owner: str, repo: str) -> HttpRequestPlan:
    query = _graphql_field_value(args, "query") or ""
    variables: dict[str, Any] = {}
    # Only `owner`/`name` are ever passed as `-f` variables to `gh api
    # graphql` in this codebase (`Transport._graphql_query`) -- every other
    # value the query needs is interpolated directly into the query text
    # itself (`_graphql_issue_states`, `_graphql_issue_dependencies`).
    for var_name in ("owner", "name"):
        value = _graphql_field_value(args, var_name)
        if value is not None:
            variables[var_name] = value
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    return HttpRequestPlan(
        method="POST",
        path="/graphql",
        headers=(("Content-Type", "application/json"),),
        body=body,
        paginate=False,
    )


def _build_rest_get_plan(args: list[str], owner: str, repo: str) -> HttpRequestPlan:
    """Build the REST-GET plan. `args` must already satisfy
    `is_http_candidate` (guaranteed by `build_request_plan`'s contract), so
    `_parse_rest_args` re-parsing it here cannot return `None` -- this reuses
    that single parser instead of re-deriving the path/header/paginate scan a
    second time, so a future shape change only needs to be taught to one
    function.
    """
    parsed = _parse_rest_args(args)
    if parsed is None:
        raise ValueError(f"build_request_plan called with a non-candidate REST shape: {args!r}")

    substituted = parsed.path.replace("{owner}", owner).replace("{repo}", repo)
    if not substituted.startswith("/"):
        substituted = "/" + substituted

    return HttpRequestPlan(
        method="GET",
        path=substituted,
        headers=parsed.headers,
        body=None,
        paginate=parsed.paginate,
    )


__all__ = ["HttpRequestPlan", "build_request_plan", "is_http_candidate"]
