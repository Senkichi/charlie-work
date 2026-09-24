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
``MAX_REPORTED_CYCLES`` per pass, with per-component caps on both reported
cycles and DFS work so one malformed component can neither starve other
components' cycles nor stall the scan. ``OrchestratorApp.intake`` runs the scan
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

# Per-component bounds applied inside ``find_blocker_cycles``. Capping each
# non-trivial SCC's reported cycles strictly below the per-pass cap keeps one
# dense malformed component from consuming the whole report and dropping
# genuine cycles in other components. The step budget bounds the DFS *work*
# per component, which the cycle cap alone cannot do: inside a cyclic SCC the
# search can walk exponentially many dead-end paths between real cycles, so
# an output-only cap still stalls the synchronous intake pass. On either cap
# the component's enumeration stops and the scan reports truncation.
MAX_CYCLES_PER_COMPONENT = 25
MAX_COMPONENT_DFS_STEPS = 50_000


@dataclass(frozen=True)
class CycleEnumeration:
    """Outcome of ``find_blocker_cycles``.

    ``cycles`` holds at most ``limit`` entries (when a limit was given), each
    in cycle order rotated so its least member comes first, sorted. ``truncated``
    is True when enumeration stopped before the whole graph was searched --
    the global ``limit`` was reached with cyclic components still unexamined,
    a component's cycle cap was exceeded, or its DFS step budget ran out --
    so unreported cycles may exist and no total count is available. ``steps``
    is the number of DFS edge-expansions performed, the work unit the
    per-component ``component_step_budget`` bounds.
    """

    cycles: list[list[int]]
    truncated: bool
    steps: int


@dataclass(frozen=True)
class BlockerCycleScan:
    """Outcome of one open-issue blocker-cycle scan.

    ``cycles`` holds at most ``MAX_REPORTED_CYCLES`` entries, each in cycle
    order rotated so its least member comes first. ``truncated`` is True when
    enumeration stopped before the whole graph was searched -- the report
    cap, a per-component cap, or the per-component DFS step budget -- so
    unreported cycles may exist; no total count is available.
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


def _is_cyclic_component(adjacency: Mapping[int, list[int]], component: list[int]) -> bool:
    """True when an SCC can yield a cycle: non-trivial, or a self-loop."""
    return len(component) > 1 or component[0] in adjacency[component[0]]


def _component_cycles(
    adjacency: Mapping[int, list[int]],
    component: list[int],
    cycle_limit: int | None,
    step_budget: int | None,
) -> tuple[list[list[int]], bool, int]:
    """Elementary cycles wholly inside one non-trivial SCC.

    Iterative least-member-anchored DFS: a cycle is emitted only by the
    search rooted at its smallest member, and intermediate members are
    restricted to nodes greater than the anchor within the component, so
    each elementary cycle is found exactly once.

    Returns ``(cycles, truncated, steps)`` where ``cycles`` holds at most
    ``cycle_limit`` entries. Enumeration of the component stops early --
    reported as ``truncated`` -- when a ``cycle_limit + 1``-th cycle is
    found or when ``step_budget`` edge-expansions are exhausted, whichever
    comes first. Both caps are per-component: the cycle cap keeps one dense
    component from consuming the whole report, and the step cap keeps a
    component with exponentially many dead-end paths from stalling the
    scan. ``steps`` counts edge-expansions so callers can observe the work
    the budget bounds.
    """
    members = set(component)
    found: list[list[int]] = []
    steps = 0
    exhausted = False
    for start in sorted(members):
        if exhausted:
            break
        path = [start]
        in_path = {start}
        stack = [iter(adjacency[start])]
        while stack:
            advanced = False
            for nxt in stack[-1]:
                steps += 1
                if (step_budget is not None and steps >= step_budget) or (
                    cycle_limit is not None and len(found) > cycle_limit
                ):
                    exhausted = True
                    break
                if nxt == start:
                    found.append(list(path))
                elif nxt > start and nxt in members and nxt not in in_path:
                    path.append(nxt)
                    in_path.add(nxt)
                    stack.append(iter(adjacency[nxt]))
                    advanced = True
                    break
            if exhausted:
                break
            if not advanced:
                stack.pop()
                in_path.discard(path.pop())
    truncated = exhausted or (cycle_limit is not None and len(found) > cycle_limit)
    if cycle_limit is not None:
        del found[cycle_limit:]
    return found, truncated, steps


def find_blocker_cycles(
    edges: Mapping[int, Iterable[int]],
    limit: int | None = None,
    *,
    component_cycle_limit: int | None = MAX_CYCLES_PER_COMPONENT,
    component_step_budget: int | None = MAX_COMPONENT_DFS_STEPS,
) -> CycleEnumeration:
    """Distinct elementary cycles in the blocker graph, up to ``limit``.

    ``edges`` maps an issue number to the issues it is blocked by. Each cycle
    is returned once, in cycle order (each member is blocked by the next),
    rotated so its least member comes first; a self-loop ``A -> A`` yields
    ``[A]``. The returned ``CycleEnumeration`` carries a sorted cycle list,
    a ``truncated`` marker set whenever enumeration stopped before the
    graph was fully searched (the global ``limit`` was reached with cyclic
    components still unexamined, a component's ``component_cycle_limit``
    was exceeded, or its ``component_step_budget`` ran out), and the DFS
    edge-expansion count.

    Strongly connected components are computed first (linear); every
    elementary cycle lies wholly inside one SCC, so an acyclic graph --
    however many distinct paths it contains -- costs one pass and returns
    empty. Cyclic components are enumerated in canonical order (least
    member first); singleton components contribute only a self-loop.

    Inside a non-trivial SCC the search can still walk exponentially many
    dead-end paths between real cycles, so work -- not just output -- is
    bounded: ``component_step_budget`` caps edge-expansions per component
    and ``component_cycle_limit`` caps cycles reported per component, which
    also keeps one malformed component from consuming the whole report and
    starving genuine cycles elsewhere. Because component order, anchor
    order, and adjacency order are all canonical, the reported set is
    deterministic and independent of the order ``edges`` (or the underlying
    issue list) happened to enumerate issues in.
    """
    adjacency: dict[int, list[int]] = {}
    all_targets: set[int] = set()
    for node, targets in edges.items():
        deduped = sorted(set(targets))
        adjacency[node] = deduped
        all_targets.update(deduped)
    for target in all_targets:
        adjacency.setdefault(target, [])

    # Canonical component order -- least member first -- so which cycles get
    # reported under the caps never depends on input enumeration order.
    components = sorted(_strongly_connected_components(adjacency), key=min)

    cycles: list[list[int]] = []
    steps = 0
    truncated = False
    for index, component in enumerate(components):
        if len(component) == 1:
            # A singleton SCC is cyclic only via a self-reference edge.
            (node,) = component
            if node in adjacency[node]:
                cycles.append([node])
        else:
            found, component_truncated, component_steps = _component_cycles(
                adjacency, component, component_cycle_limit, component_step_budget
            )
            cycles.extend(found)
            steps += component_steps
            truncated = truncated or component_truncated
        if limit is not None and len(cycles) >= limit:
            if len(cycles) > limit:
                # Cycles were found that cannot be reported.
                truncated = True
            else:
                # Exactly at the cap: truncated iff any later component is
                # still cyclic -- a cheap structural check, not enumeration.
                truncated = truncated or any(
                    _is_cyclic_component(adjacency, later) for later in components[index + 1 :]
                )
            break
    if limit is not None:
        del cycles[limit:]
    cycles.sort()
    return CycleEnumeration(cycles, truncated, steps)


def detect_open_blocker_cycles(gh: GitHubLike) -> BlockerCycleScan:
    """Scan the open-issue blocker graph for cycles; warn once per cycle.

    Reporting-only (issue #1848): performs only reads, logs one warning per
    reported cycle, and returns a ``BlockerCycleScan`` for the caller to
    record as events. At most ``MAX_REPORTED_CYCLES`` cycles are reported per
    pass; enumeration also stops early when a component exceeds its cycle cap
    or DFS step budget. Any early stop sets ``truncated`` and logs one
    truncation warning -- the unreported remainder is never enumerated.
    Fail-open: a transient ``gh`` failure resolves to an empty scan rather
    than breaking the intake pass that called it.
    """
    try:
        open_issues = gh.issue_list(state="open")
        enumeration = find_blocker_cycles(
            open_blocker_edges(gh, open_issues), limit=MAX_REPORTED_CYCLES
        )
    except Exception:
        logger.warning("blocker-cycle scan failed; skipping cycle report", exc_info=True)
        return BlockerCycleScan([], False)
    reported = enumeration.cycles
    for cycle in reported:
        logger.warning(
            "blocker cycle detected among open issues: %s",
            " -> ".join(f"#{number}" for number in (*cycle, cycle[0])),
        )
    if enumeration.truncated:
        logger.warning(
            "blocker-cycle report truncated: enumeration stopped early after "
            "%d edge-expansions (per-pass cap %d, per-component cap %d, "
            "per-component DFS budget %d); reporting %d cycle(s), "
            "unreported cycles may remain",
            enumeration.steps,
            MAX_REPORTED_CYCLES,
            MAX_CYCLES_PER_COMPONENT,
            MAX_COMPONENT_DFS_STEPS,
            len(reported),
        )
    return BlockerCycleScan(reported, enumeration.truncated)
