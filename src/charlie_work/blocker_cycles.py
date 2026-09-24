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
edge to a closed issue cannot be part of any loop -- and enumerates distinct
elementary cycles, including one-member self-references, up to
``MAX_REPORTED_CYCLES`` per pass. ``OrchestratorApp.intake`` runs the scan
once per intake pass, logs one warning per reported cycle (plus one
truncation warning when the cap bites), and records one ``blocker_cycle``
event per reported cycle. The scan is reporting-only: it writes no labels or
comments and changes no dispatch decision.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .github import (
    GitHubError,
    GitHubLike,
    get_github_issue_dependencies,
    parse_blockers,
)

logger = logging.getLogger(__name__)

# Bound on cycles enumerated and reported per intake pass. Elementary-cycle
# counts grow combinatorially inside a dense strongly connected component (k
# issues each listing all the others yields sum_{l=2..k} C(k,l)*(l-1)! cycles
# -- k=9 is already 125,664), so without a cap a malformed bulk-generated
# component would flood the log and hold the state lock recording events
# every pass. Enumeration stops at the cap; the truncation is reported
# explicitly rather than silently dropped.
MAX_REPORTED_CYCLES = 100


@dataclass(frozen=True)
class BlockerCycleScan:
    """Outcome of one open-issue blocker-cycle scan.

    ``cycles`` holds at most ``MAX_REPORTED_CYCLES`` entries, each in cycle
    order rotated so its least member comes first. ``truncated`` is True when
    more distinct cycles exist than were reported -- the unreported remainder
    is deliberately never enumerated, so no total count is available.
    """

    cycles: list[list[int]]
    truncated: bool


def declared_blockers_by_issue(
    gh: GitHubLike, issues: list[dict[str, Any]]
) -> dict[int, set[int]]:
    """Resolve each issue's declared blockers (body + GitHub-native deps).

    Mirrors the resolution ``_get_open_blockers_for_issue`` performs -- the
    union of ``parse_blockers`` on the issue body and
    ``get_github_issue_dependencies`` -- but over the whole issue set at once:
    native dependencies are fetched in one batched ``issue_dependencies``
    query (the ``_prefetch_blocker_data`` contract, issue #870), which is
    part of the ``GitHubLike`` protocol every client satisfies. Unlike the
    dispatch-gate check, a self-reference is kept -- ``A -> A`` is exactly
    the one-member cycle this scan exists to report.
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
    try:
        deps_by_number = gh.issue_dependencies(numbers)
    except (GitHubError, OSError, ValueError, TypeError):
        # The batch method falls back internally; if it raises anyway,
        # resolve per issue -- the same fallback _prefetch_blocker_data
        # uses.
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


def _strongly_connected_components(
    adjacency: Mapping[int, list[int]],
) -> list[list[int]]:
    """Kosaraju's algorithm over ``adjacency``, fully iterative: O(V + E).

    Every vertex -- including target-only nodes with no outgoing edges --
    must appear as a key in ``adjacency``. Returns the components in no
    particular order; a singleton component can still be cyclic (a self-loop
    edge), which callers check separately.
    """
    # Pass 1: iterative DFS on the forward graph, recording finish order.
    visited: set[int] = set()
    finish_order: list[int] = []
    for root in adjacency:
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, successors = stack[-1]
            advanced = False
            for nxt in successors:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, iter(adjacency[nxt])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                finish_order.append(node)

    # Pass 2: DFS on the reversed graph in reverse finish order; each tree
    # is one strongly connected component.
    reversed_adjacency: dict[int, list[int]] = {node: [] for node in adjacency}
    for node, targets in adjacency.items():
        for target in targets:
            reversed_adjacency[target].append(node)

    components: list[list[int]] = []
    seen: set[int] = set()
    for root in reversed(finish_order):
        if root in seen:
            continue
        seen.add(root)
        component = [root]
        stack = [root]
        while stack:
            node = stack.pop()
            for predecessor in reversed_adjacency[node]:
                if predecessor not in seen:
                    seen.add(predecessor)
                    component.append(predecessor)
                    stack.append(predecessor)
        components.append(component)
    return components


def find_blocker_cycles(
    edges: Mapping[int, Iterable[int]], limit: int | None = None
) -> list[list[int]]:
    """Distinct elementary cycles in the blocker graph, up to ``limit``.

    ``edges`` maps an issue number to the issues it is blocked by. Each cycle
    is returned once, in cycle order (each member is blocked by the next),
    rotated so its least member comes first; a self-loop ``A -> A`` yields
    ``[A]``. The result list is sorted, so output is deterministic. When
    ``limit`` is given, enumeration stops as soon as ``limit`` cycles have
    been found -- callers wanting a truncation signal ask for one more than
    they intend to report.

    The cost of the scan is driven by the graph's *cyclic* structure, not by
    its size or path count: strongly connected components are computed first
    (linear), and cycle enumeration runs only inside components of two or
    more members, since every elementary cycle lies wholly inside one SCC.
    An acyclic graph -- however many distinct paths it contains -- therefore
    costs one linear pass and returns ``[]``. Inside a non-trivial SCC the
    enumeration is least-member-anchored DFS: a cycle is emitted only by the
    search rooted at its smallest member, intermediate members are restricted
    to nodes greater than the anchor within the component, so each elementary
    cycle is found exactly once. Singleton components are handled separately
    by the self-loop check.
    """
    adjacency: dict[int, list[int]] = {}
    all_targets: set[int] = set()
    for node, targets in edges.items():
        deduped = sorted(set(targets))
        adjacency[node] = deduped
        all_targets.update(deduped)
    for target in all_targets:
        adjacency.setdefault(target, [])

    cycles: list[list[int]] = []

    def at_limit() -> bool:
        return limit is not None and len(cycles) >= limit

    for component in _strongly_connected_components(adjacency):
        if len(component) == 1:
            # A singleton SCC is cyclic only via a self-reference edge.
            (node,) = component
            if node in adjacency[node]:
                cycles.append([node])
                if at_limit():
                    cycles.sort()
                    return cycles
            continue
        members = set(component)
        for start in sorted(component):
            # Iterative DFS rooted at `start`, restricted to intermediate
            # nodes greater than `start` inside this component. Reaching
            # `start` again closes a cycle whose least member is `start`.
            path = [start]
            in_path = {start}
            stack = [iter(adjacency[start])]
            while stack:
                advanced = False
                for nxt in stack[-1]:
                    if nxt == start:
                        cycles.append(list(path))
                        if at_limit():
                            cycles.sort()
                            return cycles
                    elif nxt > start and nxt in members and nxt not in in_path:
                        path.append(nxt)
                        in_path.add(nxt)
                        stack.append(iter(adjacency[nxt]))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    in_path.discard(path.pop())
    cycles.sort()
    return cycles


def detect_open_blocker_cycles(gh: GitHubLike) -> BlockerCycleScan:
    """Scan the open-issue blocker graph for cycles; warn once per cycle.

    Reporting-only (issue #1848): performs only reads, logs one warning per
    reported cycle, and returns a ``BlockerCycleScan`` for the caller to
    record as events. At most ``MAX_REPORTED_CYCLES`` cycles are reported per
    pass; when more exist, ``truncated`` is set and one truncation warning is
    logged -- the unreported remainder is never enumerated. Fail-open: a
    transient ``gh`` failure resolves to an empty scan rather than breaking
    the intake pass that called it.
    """
    try:
        open_issues = gh.issue_list(state="open")
        # Ask for one more than the report cap: a result longer than the cap
        # is what proves truncation without enumerating the full set.
        cycles = find_blocker_cycles(
            open_blocker_edges(gh, open_issues), limit=MAX_REPORTED_CYCLES + 1
        )
    except Exception:
        logger.warning("blocker-cycle scan failed; skipping cycle report", exc_info=True)
        return BlockerCycleScan([], False)
    truncated = len(cycles) > MAX_REPORTED_CYCLES
    reported = cycles[:MAX_REPORTED_CYCLES]
    for cycle in reported:
        logger.warning(
            "blocker cycle detected among open issues: %s",
            " -> ".join(f"#{number}" for number in (*cycle, cycle[0])),
        )
    if truncated:
        logger.warning(
            "blocker-cycle report truncated: more than %d distinct cycles "
            "detected among open issues; reporting the first %d",
            MAX_REPORTED_CYCLES,
            MAX_REPORTED_CYCLES,
        )
    return BlockerCycleScan(reported, truncated)
