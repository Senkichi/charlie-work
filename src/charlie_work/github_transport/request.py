"""Generic GitHub request values (ADR-0006).

A request is one of three shapes: a REST call (method + route + body), a
GraphQL call (document + variables), or a gh-local CLI command (token / auth
status). All are frozen and hashable so recording fakes can key on them.

Mutation-ness is *derived* here, never listed: REST from the HTTP method,
GraphQL from the operation type the document declares. Dry-run suppression
(``guarded.GuardedTransport``) reads ``Request.is_mutation`` and nothing
else, so a new call site cannot be forgotten by a hand-kept allowlist.

This module imports nothing from ``github.py``, ``config.py`` or the
capability collaborators (it sits below them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Any, Literal, Mapping
from urllib.parse import urlencode

RestMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
_REST_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

OperationType = Literal["query", "mutation", "subscription"]
_OPERATION_KEYWORDS = frozenset({"query", "mutation", "subscription"})

ACCEPT_JSON = "application/vnd.github+json"
ACCEPT_DIFF = "application/vnd.github.v3.diff"


class CliCommand(Enum):
    """Closed set of gh-local commands that never touch the HTTP API."""

    AUTH_TOKEN = ("auth", "token")
    AUTH_STATUS = ("auth", "status")


def canonical_json(value: Any) -> str:
    """Deterministic JSON text, so equal payloads compare and hash equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_text(value: Any, *, default: str | None) -> str | None:
    if value is None:
        return default
    if isinstance(value, str):
        # Already text: re-canonicalise so two spellings of one payload match.
        return canonical_json(json.loads(value))
    return canonical_json(value)


@dataclass(frozen=True)
class RestRequest:
    method: RestMethod
    route: str  # "repos/{owner}/{repo}/pulls/42"; placeholders filled by the guard
    query: tuple[tuple[str, str], ...] = ()
    body: str | None = None  # canonical JSON text
    accept: str = ACCEPT_JSON
    long_call: bool = False
    follow_redirect: bool = False  # only job /logs (302 to a signed URL)

    def __post_init__(self) -> None:
        if self.method not in _REST_METHODS:
            raise ValueError(f"unsupported REST method: {self.method!r}")
        if not self.route or "?" in self.route:
            raise ValueError(f"route must be non-empty and carry no query string: {self.route!r}")
        if self.route.startswith("/"):
            object.__setattr__(self, "route", self.route.lstrip("/"))
        if self.body is not None and self.method == "GET":
            raise ValueError("a GET request cannot carry a body")

    @classmethod
    def of(
        cls,
        method: RestMethod,
        route: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any = None,
        accept: str = ACCEPT_JSON,
        long_call: bool = False,
        follow_redirect: bool = False,
    ) -> "RestRequest":
        """Build from mappings; canonicalises query and body to immutable text."""
        pairs = tuple((str(k), str(v)) for k, v in (query or {}).items())
        return cls(
            method,
            route,
            query=pairs,
            body=_canonical_text(body, default=None),
            accept=accept,
            long_call=long_call,
            follow_redirect=follow_redirect,
        )

    @property
    def is_mutation(self) -> bool:
        return self.method != "GET"

    @property
    def needs_repo(self) -> bool:
        return "{owner}" in self.route or "{repo}" in self.route

    def resolve(self, owner: str, repo: str) -> "RestRequest":
        """Return a copy with ``{owner}``/``{repo}`` filled in."""
        if not owner or not repo or "/" in owner or "/" in repo:
            raise ValueError(f"invalid owner/repo: {owner!r}/{repo!r}")
        route = self.route.replace("{owner}", owner).replace("{repo}", repo)
        return replace(self, route=route)

    def target(self) -> str:
        """Route plus URL-encoded query string (no leading slash)."""
        if not self.query:
            return self.route
        return f"{self.route}?{urlencode(self.query)}"

    def describe(self) -> str:
        return f"{self.method} {self.route}"


@dataclass(frozen=True)
class GraphQLRequest:
    document: str
    variables: str = "{}"  # canonical JSON text
    partial_ok: bool = False  # 200 + errors + data counts as success-with-errors
    long_call: bool = False

    def __post_init__(self) -> None:
        # Fail at construction: a document whose operation type cannot be
        # determined must never reach dry-run suppression undecided.
        _operation_of(self.document)
        variables = json.loads(self.variables)
        if not isinstance(variables, dict):
            raise ValueError("GraphQL variables must be a JSON object")

    @classmethod
    def of(
        cls,
        document: str,
        variables: Mapping[str, Any] | None = None,
        *,
        partial_ok: bool = False,
        long_call: bool = False,
    ) -> "GraphQLRequest":
        text = _canonical_text(dict(variables or {}), default="{}")
        assert text is not None
        return cls(document, text, partial_ok=partial_ok, long_call=long_call)

    @property
    def operation(self) -> OperationType:
        return _operation_of(self.document)

    @property
    def is_mutation(self) -> bool:
        # subscription counts as a mutation: fails closed for dry-run.
        return self.operation != "query"

    def describe(self) -> str:
        return f"graphql {self.operation}"


@dataclass(frozen=True)
class CliRequest:
    command: CliCommand

    @property
    def is_mutation(self) -> bool:
        return False

    def describe(self) -> str:
        return "gh " + " ".join(self.command.value)


Request = RestRequest | GraphQLRequest | CliRequest


# ---------------------------------------------------------------------------
# GraphQL operation-type lexer
# ---------------------------------------------------------------------------


def _skip_string(text: str, i: int) -> int:
    """Return the index just past the string literal starting at ``text[i]``."""
    if text.startswith('"""', i):
        end = text.find('"""', i + 3)
        while end != -1 and text[end - 1] == "\\":
            end = text.find('"""', end + 1)
        return len(text) if end == -1 else end + 3
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i + 1
        i += 1
    return len(text)


@lru_cache(maxsize=512)
def _lex_operations(document: str) -> tuple[OperationType, ...]:
    """Operation types declared at the top level of *document*, in order.

    Skips whitespace, commas, ``#`` comments and string literals. Tracks
    brace/paren depth so selection sets, variable defaults and directive
    arguments never read as top-level tokens. A leading ``{`` (no keyword) is
    the anonymous-query shorthand; ``fragment`` definitions are not operations.
    """
    ops: list[OperationType] = []
    depth = 0  # brace depth
    parens = 0
    expect_body = False  # a keyword header was seen; its `{` is the body
    i, n = 0, len(document)
    while i < n:
        ch = document[i]
        if ch in " \t\r\n,\ufeff":
            i += 1
        elif ch == "#":
            while i < n and document[i] not in "\r\n":
                i += 1
        elif ch == '"':
            i = _skip_string(document, i)
        elif ch == "(":
            parens += 1
            i += 1
        elif ch == ")":
            parens = max(0, parens - 1)
            i += 1
        elif ch == "{":
            if depth == 0 and parens == 0:
                if expect_body:
                    expect_body = False
                else:
                    ops.append("query")  # anonymous shorthand
            depth += 1
            i += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            i += 1
        elif ch.isalpha() or ch == "_":
            j = i
            while j < n and (document[j].isalnum() or document[j] == "_"):
                j += 1
            word = document[i:j]
            if depth == 0 and parens == 0 and not expect_body:
                if word in _OPERATION_KEYWORDS:
                    ops.append(word)  # type: ignore[arg-type]
                    expect_body = True
                elif word == "fragment":
                    expect_body = True
            i = j
        else:
            i += 1
    return tuple(ops)


def _operation_of(document: str) -> OperationType:
    ops = _lex_operations(document)
    if len(ops) != 1:
        raise ValueError(f"GraphQL document must declare exactly one operation, found {len(ops)}")
    return ops[0]
