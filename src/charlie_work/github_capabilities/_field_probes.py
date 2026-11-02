"""Schema probes behind ``Transport.validate_field_lists`` (ADR-0006, B13).

Each registered field list becomes one cheap request (``first:1`` for a list,
``number:0`` for a view) sent through the guarded transport. GitHub validates a
GraphQL selection before it executes anything, so an unknown field comes back
as an ``undefinedField`` error naming it, while a clean schema answers a
``number:0`` view with NOT_FOUND. ``probe_verdict`` reads that outcome as a
value; the caller decides whether to raise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..github_transport.json_read import JsonRead, RunListRead
from ..github_transport.outcome import FailureKind, Outcome, Response, TransportFailure
from ..github_transport.request import RestRequest
from ._base import RUN_LIST_FIELDS
from ._outcome import failure_text
from .checks import PR_CHECKS_FIELDS
from .issues import ISSUE_LIST_FIELDS, ISSUE_VIEW_FIELDS
from .labels import LABEL_LIST_FIELDS
from .pull_requests import MERGED_PR_LIST_FIELDS, PR_LIST_FIELDS, PR_VIEW_FIELDS

Probe = JsonRead | RunListRead | RestRequest

_FIELD_IN_MESSAGE = re.compile(r"Field '([^']+)'")
# Statuses that say GitHub (or our access to it) is unavailable, never that a field was
# rejected: 0 = no HTTP exchange, 401 = bad/expired token, 403 = primary or secondary
# rate limit or a permission gap, 404 = repository not visible, 429 = rate limited.
_UNAVAILABLE_STATUSES = (0, 401, 403, 404, 429)
_UNMAPPED = re.compile(r"^unknown (?:--json|run) field\(s\)(?: for [a-z]+)?: (.+)$")


@dataclass(frozen=True)
class ProbeVerdict:
    """What one probe proved: ``rejected`` fields, a transport ``skip``, or a ``detail``.

    Only ``rejected`` is evidence that a configured field list is wrong. ``skip`` (GitHub
    unavailable: transport failure, auth, quota) and a bare ``detail`` (inconclusive) are
    both non-evidence, so the caller must not turn either into a startup error (#1833).
    """

    rejected: tuple[str, ...] | None = None
    skip: bool = False
    detail: str = ""


def field_list_probes(
    reconcile_pr_fields: str, reconcile_issue_fields: str
) -> list[tuple[str, str, Probe]]:
    """``(constant name, fields, probe)`` for every registered list."""
    return [
        (
            "ISSUE_LIST_FIELDS",
            ISSUE_LIST_FIELDS,
            JsonRead("issue", "list", ISSUE_LIST_FIELDS, state="open", limit=1),
        ),
        (
            "ISSUE_VIEW_FIELDS",
            ISSUE_VIEW_FIELDS,
            JsonRead("issue", "view", ISSUE_VIEW_FIELDS, number=0),
        ),
        (
            "PR_LIST_FIELDS",
            PR_LIST_FIELDS,
            JsonRead("pr", "list", PR_LIST_FIELDS, state="open", limit=1),
        ),
        (
            "MERGED_PR_LIST_FIELDS",
            MERGED_PR_LIST_FIELDS,
            JsonRead("pr", "list", MERGED_PR_LIST_FIELDS, state="merged", limit=1),
        ),
        ("PR_VIEW_FIELDS", PR_VIEW_FIELDS, JsonRead("pr", "view", PR_VIEW_FIELDS, number=0)),
        (
            "PR_CHECKS_FIELDS",
            PR_CHECKS_FIELDS,
            JsonRead("pr", "checks", PR_CHECKS_FIELDS, number=0),
        ),
        (
            "LABEL_LIST_FIELDS",
            LABEL_LIST_FIELDS,
            RestRequest.of("GET", "repos/{owner}/{repo}/labels", query={"per_page": 1}),
        ),
        (
            "RECONCILE_PR_FIELDS",
            reconcile_pr_fields,
            JsonRead("pr", "list", reconcile_pr_fields, state="all", limit=1),
        ),
        (
            "RECONCILE_ISSUE_FIELDS",
            reconcile_issue_fields,
            JsonRead("issue", "list", reconcile_issue_fields, state="open", limit=1),
        ),
        ("RUN_LIST_FIELDS", RUN_LIST_FIELDS, RunListRead(RUN_LIST_FIELDS, limit=1)),
    ]


def _rejected_fields(response: Response) -> tuple[str, ...]:
    names: list[str] = []
    for error in response.graphql_errors:
        if error.type != "undefinedField" and "doesn't exist on type" not in error.message:
            continue
        match = _FIELD_IN_MESSAGE.search(error.message)
        if match:
            names.append(match.group(1))
        elif error.path and isinstance(error.path[-1], str):
            names.append(error.path[-1])
        else:
            names.append(error.message)
    return tuple(names)


def probe_verdict(outcome: Outcome) -> ProbeVerdict:
    """Read one probe's outcome. All-default (``ProbeVerdict()``) means the list is valid."""
    if isinstance(outcome, TransportFailure):
        if outcome.kind is not FailureKind.ADAPTER_DEFECT:
            return ProbeVerdict(skip=True, detail=failure_text(outcome))
        unmapped = _UNMAPPED.match(outcome.detail)
        if unmapped:
            return ProbeVerdict(rejected=tuple(n.strip() for n in unmapped.group(1).split(",")))
        if outcome.detail.startswith("no "):
            return ProbeVerdict()  # a view of #0 answered with no node: the schema accepted it
        return ProbeVerdict(detail=failure_text(outcome))
    if outcome.ok:
        return ProbeVerdict()
    rejected = _rejected_fields(outcome)
    if rejected:
        return ProbeVerdict(rejected=rejected)
    errors = outcome.graphql_errors
    if errors and all(error.type == "NOT_FOUND" for error in errors):
        return ProbeVerdict()
    if (
        outcome.status in _UNAVAILABLE_STATUSES
        or outcome.status >= 500
        or any(error.type == "RATE_LIMITED" for error in errors)
    ):
        return ProbeVerdict(skip=True, detail=failure_text(outcome))
    return ProbeVerdict(detail=failure_text(outcome))
