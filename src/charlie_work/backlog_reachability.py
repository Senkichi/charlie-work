"""Backlog-reachability free-function pair (issue #1283 standalone extraction).

Extracted verbatim from ``workflow.py``: the shared open-blocker check
(``_get_open_blockers_for_issue``) and the unfiltered-backlog reachability
classifier (``classify_backlog_reachability``) that calls it. The two are
call-graph connected (``classify_backlog_reachability`` calls
``_get_open_blockers_for_issue`` directly) but disconnected from every
Phase-A family (A1-A5) already extracted from ``workflow.py`` -- confirmed
by the A6 recon and operator-approved in issue #1283's newest comment as a
standalone standard-shape PR rather than folded into another family.

``workflow.py`` re-exports both symbols via a facade import block (mirroring
``config.py``'s ``RunnerAllocationConfig`` re-export pattern and this
repo's own ``dispatch_selection.py``/``escalation.py``/``verdict_parsing.py``/
``rework_prompts.py``/``ci_findings.py`` precedents), so existing import
paths and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import OrchestratorConfig
from .github import (
    GitHubLike,
    get_github_issue_dependencies,
    label_names,
    parse_blockers,
)


def _get_open_blockers_for_issue(
    gh: GitHubLike, issue: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Standalone blocker check — the shared core of ``_get_open_blockers``.

    Both the dispatch candidate filter (``_filter_blocked_issues`` via the
    ``_get_open_blockers`` method) and ``classify_backlog_reachability`` must
    answer the same question: does this issue have any *open* blockers? This
    function is the single implementation of that check so the two paths cannot
    diverge — a dependency-gate change made for dispatch automatically applies
    to reachability, and vice versa.

    Returns ``(declared_blockers, open_blockers)`` — both sorted lists of issue
    numbers. ``declared_blockers`` is every blocker mentioned in the issue body
    or GitHub-native dependencies; ``open_blockers`` is the subset that are
    currently open. Fail-open: a transient API error resolves to ``([], [])``,
    so the issue is treated as unblocked (matching the dispatch path's
    behaviour — a failed lookup does not filter a candidate out).
    """
    logger = logging.getLogger(__name__)
    issue_number = int(issue["number"])
    body = issue.get("body", "")
    body_blockers = parse_blockers(body)
    gh_blockers = get_github_issue_dependencies(gh, issue_number)
    all_blockers = sorted(set(body_blockers + gh_blockers))
    if not all_blockers:
        return [], []
    open_blockers = gh.are_issues_open(all_blockers)
    if issue_number in open_blockers:
        logger.warning(
            f"Issue #{issue_number} has self-referencing blocker declaration - ignoring"
        )
        open_blockers.discard(issue_number)
        all_blockers.remove(issue_number)
    return sorted(all_blockers), sorted(open_blockers)


def classify_backlog_reachability(
    gh: GitHubLike,
    config: OrchestratorConfig,
    operator_claimed: set[int] | None = None,
    *,
    ready_open_count: int | None = None,
) -> dict[str, Any]:
    """Issue #944: observe the *unfiltered* open backlog and record why nothing dispatched.

    The dispatch path (``_dispatch_impl``) sources candidates from
    ``issue_list(labels=[ready], state="all")`` -- filtered *at the source*. An
    issue without ``automated-ready`` is therefore not rejected by
    ``_is_dispatchable``; it is never **fetched**. No counter, event field, or
    status line placed downstream of that query can distinguish "no work
    exists" from "the entire backlog is unreachable", because by then the
    unreachable issues have been filtered out of existence.

    That gap hid a four-day total dispatch stall (2026-07-31 -> 2026-08-05):
    87 open issues, 0 dispatchable, while every pass reported ``ok=1`` and every
    ``dispatch`` event carried ``issue_numbers: []``. A content-empty stream
    defeats content-keyed alerting exactly as silence does.

    This runs one *unfiltered* ``issue_list(state="open")`` and bins every open
    issue by the first ``_is_dispatchable`` arm that rejects it, so a zero
    dispatch count comes with a reason rather than an absence.

    **An empty fetch is reported as ``observed: False``, never as an empty
    backlog.** ``GitHubClient._list_json`` returns ``[]`` both when the repo
    genuinely has no open issues and when the underlying ``gh`` call fails
    (``run()`` returning a non-list is coerced to ``[]``). Reporting that as
    ``open_total: 0, dispatchable: 0`` would read as "nothing to do, all
    healthy" -- reintroducing the exact failure this function exists to
    detect, one layer up. Callers must treat ``observed: False`` as "unknown",
    not as "empty".

    ``ready_open_count`` is an optional cross-check: the unfiltered open list
    must be a superset of the ready-labelled open issues the caller already
    fetched. ``consistent`` is tri-state -- None when the check did not run,
    True when it ran and passed, False when the unfiltered fetch is missing
    issues the caller already saw. It is never True by default, because a
    reassuring value standing in for an unrun check is this bug wearing a
    different field name.
    """
    reachability: dict[str, Any] = {
        "observed": False,
        # Tri-state, and deliberately not defaulted to True: None means the
        # cross-check did not RUN (the fetch failed, or the caller supplied no
        # ``ready_open_count``), True means it ran and passed, False means it
        # ran and failed. Defaulting an unverified claim to "verified" would be
        # the exact defect this function exists to catch, one field over -- a
        # healthy-looking default standing in for an unknown state.
        "consistent": None,
        "open_total": 0,
        "dispatchable": 0,
        "missing_ready": 0,
        "terminal_label": 0,
        "active_label": 0,
        "operator_claimed": 0,
        # Issue #1110: an automated-ready issue with no agent: label that is
        # blocked by an open predecessor passes every label-only check above
        # but is permanently (and correctly) unselectable by dispatch, which
        # applies a further dependency gate (_filter_blocked_issues). Without
        # this bin those issues were counted ``dispatchable`` by reachability
        # while being unselectable, producing false dispatch_stale alarms for
        # a deliberately sequenced cohort tail. ``dispatchable`` now counts
        # only issues that pass BOTH the label gate and the dependency gate;
        # this bin holds the issues the dependency gate rejects. The bins
        # still partition -- every fetched issue lands in exactly one.
        "blocked_by_open_dependency": 0,
        # An issue with no ``number`` cannot be dispatched or named as an
        # example, but it must still be BINNED rather than skipped: the
        # renderer joins the non-zero reasons, so a backlog of these would
        # print "N open, 0 dispatchable ()" -- firing the alarm while naming
        # no cause. Every fetched issue lands in exactly one bin, so the bins
        # always sum to ``open_total``.
        "unidentified": 0,
        "unreachable_examples": {},
    }

    issues = gh.issue_list(state="open")
    if not issues:
        # Ambiguous by construction -- see the docstring. Say nothing rather
        # than say "empty".
        return reachability

    claimed = operator_claimed or set()
    ready_seen = 0
    examples: dict[str, list[int]] = {}
    for issue in issues:
        number = issue.get("number")
        if number is None:
            reachability["unidentified"] += 1
            continue
        number = int(number)
        names = label_names(issue)
        if config.labels.ready not in names:
            reason = "missing_ready"
        else:
            ready_seen += 1
            if names & config.labels.terminal:
                reason = "terminal_label"
            elif names & config.labels.active:
                reason = "active_label"
            elif number in claimed:
                reason = "operator_claimed"
            else:
                # Issue #1110: the label-only checks above mirror
                # _is_dispatchable, but the dispatch path applies a further
                # dependency gate (_filter_blocked_issues) that this function
                # never modeled. Run the same blocker check the dispatch
                # candidate filter runs and bin dependency-blocked issues
                # distinctly, so dispatch_staleness can key off the
                # post-dependency-gate candidate count instead of the
                # label-only count. Fail-open: a transient API error resolves
                # to no open blockers (matching dispatch -- a failed lookup
                # does not filter a candidate out), so the issue bins as
                # ``dispatchable`` rather than ``blocked_by_open_dependency``.
                _declared, open_blockers = _get_open_blockers_for_issue(gh, issue)
                if open_blockers:
                    reason = "blocked_by_open_dependency"
                else:
                    reason = "dispatchable"
        reachability[reason] += 1
        if reason != "dispatchable":
            bucket = examples.setdefault(reason, [])
            if len(bucket) < 5:
                bucket.append(number)

    reachability["observed"] = True
    reachability["open_total"] = len(issues)
    reachability["unreachable_examples"] = {k: sorted(v) for k, v in sorted(examples.items())}
    if ready_open_count is not None:
        # The unfiltered list must be a superset of the ready-labelled OPEN
        # issues the caller already fetched. If it is missing some, it cannot
        # be a superset and the fetch is unreliable. One-directional by
        # design: a failure of the *filtered* query yields ready_open_count=0
        # and passes here, but that case is loud elsewhere (a non-zero
        # dispatchable count alongside a dispatch of nothing).
        reachability["consistent"] = ready_seen >= ready_open_count
    return reachability
