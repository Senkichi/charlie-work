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


def _rest_path(args: list[str]) -> str | None:
    """Return the REST endpoint positional (`args[1]`) or None if absent/flag.

    Every REST call site in this codebase places the endpoint immediately
    after `"api"` (`["api", "rate_limit"]`, `["api", f"repos/{{owner}}/..."]`,
    ...) -- there is no call shape here that puts a flag before the path.
    """
    if len(args) < 2:
        return None
    path = args[1]
    if path.startswith("-"):
        return None
    return path


def is_http_candidate(args: list[str]) -> bool:
    """True when `args` (argv after the leading `gh` token) is a REST-GET or
    GraphQL-query `gh api` call this module can build an HTTP request for.

    Callers must not call `build_request_plan` unless this returns True.
    """
    if not args or args[0] != "api":
        return False
    if _api_is_mutating(args):
        return False
    # --jq applies client-side jq filtering gh performs on the response
    # before printing it; replicating that would mean vendoring jq
    # semantics. Excluded -- falls back to gh.
    if any(arg == "--jq" or arg.startswith("--jq=") for arg in args):
        return False
    if _is_graphql_query(args):
        return True
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
    raw_path = _rest_path(args) or ""
    substituted = raw_path.replace("{owner}", owner).replace("{repo}", repo)
    if not substituted.startswith("/"):
        substituted = "/" + substituted

    headers: list[tuple[str, str]] = []
    paginate = False
    i = 2
    while i < len(args):
        arg = args[i]
        if arg == "-H":
            header_text = args[i + 1] if i + 1 < len(args) else ""
            i += 2
        elif arg.startswith("-H") and len(arg) > 2:
            header_text = arg[2:]
            i += 1
        elif arg == "--paginate":
            paginate = True
            i += 1
            continue
        else:
            i += 1
            continue
        if ":" in header_text:
            name, _, value = header_text.partition(":")
            headers.append((name.strip(), value.strip()))

    return HttpRequestPlan(
        method="GET",
        path=substituted,
        headers=tuple(headers),
        body=None,
        paginate=paginate,
    )


__all__ = ["HttpRequestPlan", "build_request_plan", "is_http_candidate"]
