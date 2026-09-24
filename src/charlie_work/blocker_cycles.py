"""Blocker-cycle detection over the open-issue dependency graph (issue #1848).

When open issues block each other in a loop -- A waits on B while B waits on
A, or an issue lists itself -- none of them ever dispatches: every member
carries ``automated-ready`` and looks armed, but each is gated by a blocker
that is itself gated. The per-issue dependency gate
(``_get_open_blockers_for_issue``) only answers "does THIS issue have an open
blocker", so the loop is invisible in every per-issue view -- the issues sit
armed and stall forever with no signal naming the reason. Bulk-generated
ticket sets (``/to-tickets``, plan-to-issues) make such loops more likely.

This module builds the directed blocker graph over *open* issues only -- an
edge to a closed issue cannot be part of any loop -- and enumerates every
distinct elementary cycle, including one-member self-references.
``OrchestratorApp.intake`` runs the scan once per intake pass, logs one
warning per distinct cycle, and records one ``blocker_cycle`` event per
cycle. The scan is reporting-only: it writes no labels or comments and
changes no dispatch decision.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .github import (
    GitHubError,
    GitHubLike,
    get_github_issue_dependencies,
    parse_blockers,
)

logger = logging.getLogger(__name__)


def declared_blockers_by_issue(
    gh: GitHubLike, issues: list[dict[str, Any]]
) -> dict[int, set[int]]:
    """Resolve each issue's declared blockers (body + GitHub-native deps).

    Mirrors the resolution ``_get_open_blockers_for_issue`` performs -- the
    union of ``parse_blockers`` on the issue body and
    ``get_github_issue_dependencies`` -- but over the whole issue set at once:
    native dependencies are fetched in one batched ``issue_dependencies``
    query when the client supports it (the ``_prefetch_blocker_data``
    contract, issue #870), falling back to per-issue REST calls otherwise.
    Unlike the dispatch-gate check, a self-reference is kept -- ``A -> A`` is
    exactly the one-member cycle this scan exists to report.
    """
    numbers: list[int] = []
    body_by_number: dict[int, str] = {}
    for issue in issues:
        number = issue.get("number")
        if number is None:
            continue
        number = int(number)
        if number not in body_by_number:
            numbers.append(number)
            body_by_number[number] = issue.get("body") or ""
    if not numbers:
        return {}
    if hasattr(gh, "issue_dependencies"):
        try:
            deps_by_number = gh.issue_dependencies(numbers)
        except (GitHubError, OSError, ValueError, TypeError):
            # The batch method falls back internally; if it raises anyway,
            # resolve per issue -- the same fallback _prefetch_blocker_data
            # uses.
            deps_by_number = {
                number: get_github_issue_dependencies(gh, number) for number in numbers
            }
    else:
        # Test doubles and older GitHub-like objects without the batch method.
        deps_by_number = {number: get_github_issue_dependencies(gh, number) for number in numbers}
    return {
        number: set(parse_blockers(body_by_number[number]))
        | {int(dep) for dep in deps_by_number.get(number) or []}
        for number in numbers
    }


def open_blocker_edges(gh: GitHubLike, issues: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Directed edges ``issue -> open blocker`` for each issue in ``issues``.

    ``issues`` is the graph's vertex set (callers pass the open-issue list).
    An edge to a blocker that is not currently open is dropped -- a closed
    issue cannot be part of a loop -- as is an edge to a blocker that does
    not exist. Open-ness is resolved through ``gh.are_issues_open``, the same
    primitive the dependency gate uses, so "closed" cannot drift between this
    scan and dispatch.
    """
    declared = declared_blockers_by_issue(gh, issues)
    all_blockers = sorted({b for targets in declared.values() for b in targets})
    open_targets = set(gh.are_issues_open(all_blockers)) if all_blockers else set()
    return {
        number: sorted(blocker for blocker in targets if blocker in open_targets)
        for number, targets in declared.items()
    }


def find_blocker_cycles(edges: Mapping[int, Iterable[int]]) -> list[list[int]]:
    """Every distinct elementary cycle in the blocker graph.

    ``edges`` maps an issue number to the issues it is blocked by. Each cycle
    is returned once, in cycle order (each member is blocked by the next),
    rotated so its least member comes first; a self-loop ``A -> A`` yields
    ``[A]``. The result list is sorted, so output is deterministic.

    Enumeration is least-member-anchored DFS: a cycle is emitted only by the
    search rooted at its smallest member, and intermediate members are
    restricted to nodes greater than the anchor, so each elementary cycle is
    found exactly once. Elementary-cycle counts can grow exponentially in
    dense graphs; blocker graphs are sparse by construction (a handful of
    declared blockers per issue), so no artificial cap is applied.
    """
    adjacency = {node: sorted(set(targets)) for node, targets in edges.items()}
    cycles: list[list[int]] = []
    for start in sorted(adjacency):
        # Iterative DFS rooted at `start`, restricted to intermediate nodes
        # greater than `start`. Reaching `start` again closes a cycle whose
        # least member is `start`.
        path = [start]
        in_path = {start}
        stack = [iter(adjacency[start])]
        while stack:
            advanced = False
            for nxt in stack[-1]:
                if nxt == start:
                    cycles.append(list(path))
                elif nxt > start and nxt not in in_path:
                    path.append(nxt)
                    in_path.add(nxt)
                    stack.append(iter(adjacency.get(nxt, ())))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                in_path.discard(path.pop())
    cycles.sort()
    return cycles


def detect_open_blocker_cycles(gh: GitHubLike) -> list[list[int]]:
    """Scan the open-issue blocker graph for cycles; warn once per cycle.

    Reporting-only (issue #1848): performs only reads, logs one warning per
    distinct cycle, and returns the cycles for the caller to record as
    events. Fail-open: a transient ``gh`` failure resolves to an empty list
    rather than breaking the intake pass that called it.
    """
    try:
        open_issues = gh.issue_list(state="open")
        cycles = find_blocker_cycles(open_blocker_edges(gh, open_issues))
    except Exception:
        logger.warning("blocker-cycle scan failed; skipping cycle report", exc_info=True)
        return []
    for cycle in cycles:
        logger.warning(
            "blocker cycle detected among open issues: %s",
            " -> ".join(f"#{number}" for number in (*cycle, cycle[0])),
        )
    return cycles
