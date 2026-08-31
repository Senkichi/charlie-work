"""Backlog-reachability free-function family (issue #1283 standalone extraction).

Extracted from ``workflow.py``: the shared open-blocker check
(``_get_open_blockers_for_issue``) and the unfiltered-backlog reachability
classifier (``classify_backlog_reachability``) that calls it. The two are
call-graph connected (``classify_backlog_reachability`` calls
``_get_open_blockers_for_issue`` directly) but disconnected from every
Phase-A family (A1-A5) already extracted from ``workflow.py`` -- confirmed
by the A6 recon and operator-approved in issue #1283's newest comment as a
standalone standard-shape PR rather than folded into another family.

Issue #1337 added four more functions extracted here so the monolith does
not grow past its file-size ratchet high-water mark:
``scan_merged_pr_references`` (the merged-PR -> issue reference scan, moved
verbatim from ``_merged_pr_referenced_issue_numbers``), the mention-coverage
map builder ``compute_mention_coverage_map``, the fail-open merged-PR fetch
``fetch_merged_prs_fail_open``, and the dispatch-side fetch/reuse resolver
``resolve_dispatch_mention_coverage``. The classifier's mention-coverage arm
and the dispatch-side exclusion both call ``scan_merged_pr_references`` (via
the app's thin wrapper), so the exclusion semantics cannot drift between the
two paths.

``workflow.py`` re-exports all symbols via a facade import block (mirroring
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
    GitHubError,
    GitHubLike,
    get_github_issue_dependencies,
    issue_numbers_mentioned_by_pr,
    label_names,
    linked_issue_number,
    parse_blockers,
)
from .state import load_state_locked


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
    mention_covered: dict[int, list[int]] | None = None,
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

    ``mention_covered`` (issue #1337) maps issue numbers to the merged PR
    numbers that mention them in free text and whose mention-only dispatch
    exclusion has NOT been lifted by operator re-arm. An issue in this map
    bins as ``mention_covered_awaiting_operator`` -- not ``dispatchable`` --
    mirroring the dispatch-side exclusion computed from the same
    ``_merged_pr_referenced_issue_numbers`` + ``_mention_rearmed_issue_numbers``
    helpers the caller invokes. The map is derived from (not a parallel
    re-implementation of) the dispatch predicate: the caller computes it by
    calling the same methods ``_dispatch_impl`` uses, so a change to the
    exclusion semantics automatically applies here. ``mention_covered_prs``
    carries the issue->PR-numbers detail so the reason names the mentioning
    PR(s), not just the exclusion. Fail-open: a ``None`` or empty map (e.g.
    when the merged-PR list fetch failed) leaves the bin at zero and issues
    classify as ``dispatchable``, matching the blocker check's fail-open
    behaviour -- the classifier is advisory and must not raise.
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
        # only issues that pass the label gate, the merged-PR mention
        # exclusion, AND the dependency gate; this bin holds the issues the
        # dependency gate rejects (after the mention exclusion has already
        # passed). The bins still partition -- every fetched issue lands in
        # exactly one.
        "blocked_by_open_dependency": 0,
        # Issue #1337: an automated-ready issue that a merged PR mentions in
        # free text (and whose mention-only exclusion has not been lifted by
        # operator re-arm) is excluded from dispatch by _dispatch_impl's
        # merged_pr_issue_numbers filter. Without this bin the classifier
        # reported such an issue as ``dispatchable`` on every heartbeat check
        # forever, while dispatch silently dropped it each pass -- the exact
        # contradiction ("dispatchable across N beats but never dispatched")
        # that triggered a manual investigation for #1059. The map is derived
        # from the same _merged_pr_referenced_issue_numbers +
        # _mention_rearmed_issue_numbers helpers dispatch uses, so the
        # exclusion semantics cannot drift between the two paths. Checked
        # BEFORE the dependency gate to mirror dispatch's filter order (label
        # gate -> merged-PR exclusion -> dependency gate): an issue that is
        # both mention-covered and blocked by an open dependency bins here,
        # because dispatch drops it at the merged_pr_issue_numbers exclusion
        # and it never reaches _filter_blocked_issues.
        "mention_covered_awaiting_operator": 0,
        # An issue with no ``number`` cannot be dispatched or named as an
        # example, but it must still be BINNED rather than skipped: the
        # renderer joins the non-zero reasons, so a backlog of these would
        # print "N open, 0 dispatchable ()" -- firing the alarm while naming
        # no cause. Every fetched issue lands in exactly one bin, so the bins
        # always sum to ``open_total``.
        "unidentified": 0,
        "unreachable_examples": {},
        # Issue #1337: issue_number -> sorted PR numbers that mention it, for
        # issues binned as ``mention_covered_awaiting_operator``. Carried
        # separately from ``unreachable_examples`` (which maps reason ->
        # issue numbers) so the reason names the mentioning PR(s), not just
        # the exclusion.
        "mention_covered_prs": {},
    }

    issues = gh.issue_list(state="open")
    if not issues:
        # Ambiguous by construction -- see the docstring. Say nothing rather
        # than say "empty".
        return reachability

    claimed = operator_claimed or set()
    covered = mention_covered or {}
    covered_prs: dict[int, list[int]] = {}
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
                # Issue #1337: model the merged-PR mention-only dispatch
                # exclusion. An issue in ``covered`` is excluded from
                # dispatch by _dispatch_impl's merged_pr_issue_numbers
                # filter (and has NOT been re-armed by the operator).
                # Without this arm it binned as ``dispatchable`` forever
                # while dispatch silently dropped it each pass. The map
                # is derived from the same helpers dispatch uses, so the
                # predicate cannot drift. Placed BEFORE the dependency
                # gate to mirror dispatch's filter order (label gate ->
                # merged-PR exclusion -> dependency gate): both the dry-run
                # and real _dispatch_impl paths apply the
                # merged_pr_issue_numbers exclusion in the candidate list
                # comprehension and only run _filter_blocked_issues on the
                # already-filtered list, so an issue that is BOTH
                # mention-covered and blocked by an open dependency is
                # dropped by the mention exclusion and never reaches the
                # dependency gate. The classifier bins it the same way --
                # ``mention_covered_awaiting_operator`` -- so the reason
                # names the still-active exclusion rather than the
                # dependency gate the issue never reached.
                if number in covered:
                    reason = "mention_covered_awaiting_operator"
                    covered_prs[number] = sorted(covered[number])
                else:
                    # Issue #1110: the label-only checks above mirror
                    # _is_dispatchable, but the dispatch path applies a
                    # further dependency gate (_filter_blocked_issues) that
                    # this function never modeled. Run the same blocker
                    # check the dispatch candidate filter runs and bin
                    # dependency-blocked issues distinctly, so
                    # dispatch_staleness can key off the
                    # post-dependency-gate candidate count instead of the
                    # label-only count. Fail-open: a transient API error
                    # resolves to no open blockers (matching dispatch -- a
                    # failed lookup does not filter a candidate out), so
                    # the issue bins as ``dispatchable`` rather than
                    # ``blocked_by_open_dependency``.
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
    reachability["mention_covered_prs"] = {k: v for k, v in sorted(covered_prs.items())}
    if ready_open_count is not None:
        # The unfiltered list must be a superset of the ready-labelled OPEN
        # issues the caller already fetched. If it is missing some, it cannot
        # be a superset and the fetch is unreliable. One-directional by
        # design: a failure of the *filtered* query yields ready_open_count=0
        # and passes here, but that case is loud elsewhere (a non-zero
        # dispatchable count alongside a dispatch of nothing).
        reachability["consistent"] = ready_seen >= ready_open_count
    return reachability


# ---------------------------------------------------------------------------
# Issue #1337: merged-PR mention-coverage helpers, extracted from workflow.py
# so the monolith does not grow past its file-size ratchet high-water mark.
# The classifier's mention-coverage arm and the dispatch-side exclusion both
# go through ``scan_merged_pr_references`` (via the app's thin wrapper), so
# the exclusion semantics cannot drift between the two paths.
# ---------------------------------------------------------------------------


def scan_merged_pr_references(
    issues: list[dict[str, Any]],
    merged_prs: list[dict[str, Any]],
    branch_prefix: str,
) -> tuple[set[int], set[int], set[int], dict[int, list[int]]]:
    """Return ready issues already covered by a merged PR, split by trust level.

    Extracted verbatim from ``workflow.py``'s ``_merged_pr_referenced_issue_numbers``
    method (issue #1337 moved it here so the mention-PR tracking it added does
    not grow the monolith past its ratchet high-water mark). The thin wrapper
    on ``OrchestratorApp`` passes ``self.config.dispatch.branch_prefix``.

    Returns a ``(bound, mention_only, bound_pr_numbers,
    mention_pr_numbers_by_issue)`` 4-tuple:

    * ``bound`` -- ``linked_issue_number`` binds the PR to the issue by a
      hijack-safe signal (same-repo branch-prefix or closing-action verb).
      This is the same trust level issue #220 uses to close issues at
      merge time, so callers MAY treat a bound issue as safe to close here
      too (belt-and-suspenders in case #220's merge-time close hasn't
      landed, e.g. a crash between merge and label transition).
    * ``mention_only`` -- the PR merely contains an ``issue #N`` /
      ``issues #N`` text reference (same-repo PRs only -- cross-repo
      provenance is never trusted here, and ``isCrossRepository`` only
      describes head-branch provenance, not which repo the *text* refers
      to, so it cannot resolve a cross-repo mention collision either).
      This is advisory only per ``issue_numbers_mentioned_by_pr``'s
      contract: issue #203 never authorized closing an issue on a bare
      mention. Callers MUST exclude these from dispatch and flag them for
      a human -- never close the issue or transition it toward "merged".
    * ``bound_pr_numbers`` -- the PR numbers that bound to a managed issue.
      Used to finalize state.json ``prs`` entries for externally-merged PRs
      (issue #427: Aviator mergequeue handoff).
    * ``mention_pr_numbers_by_issue`` -- issue_number -> sorted PR numbers
      that mention it in free text (issue #1337). Built in the same scan
      loop as ``mention_only`` so there is no second implementation to
      drift. Only carries entries for issues in ``mention_only`` (bound
      issues are excluded from ``mention_only`` and therefore from this
      map too). Consumed by the backlog-reachability classifier to name
      the mentioning PR(s) in its ``mention_covered_awaiting_operator``
      reason.

    The bound/mention sets are intersected with the supplied issue set so a
    stray mention of an issue not in the dispatch queue does not get
    actioned. ``bound`` takes precedence: an issue bound by one merged PR
    but only mentioned by another is reported solely in ``bound``.
    """
    ready_issue_numbers = {int(issue["number"]) for issue in issues}
    bound: set[int] = set()
    bound_pr_numbers: set[int] = set()
    mention_only: set[int] = set()
    # Issue #1337: track which PR mentions each issue in the same scan
    # loop so the classifier can name the mentioning PR(s) without a
    # second implementation of the mention scan. ``mention_prs`` maps
    # issue_number -> list of PR numbers (accumulated across PRs; a
    # single issue can be mentioned by multiple merged PRs).
    mention_prs: dict[int, list[int]] = {}
    for pr in merged_prs:
        if str(pr.get("state") or "").upper() != "MERGED":
            continue
        pr_number = int(pr["number"])
        # Issue #1229 scoping decision: this call site is deliberately NOT
        # threaded through branch_issue_validator. The ``bound`` set only
        # feeds the backlog-reachability classifier's mention-only exclusion
        # (an issue wrongly seen as "bound" is excluded from dispatch as
        # already-covered, a missed dispatch recovered on the next
        # classification pass); no issue-label transition or state escalation
        # keys off it. A stale branch-name binding can at worst misclassify a
        # ready issue as bound, not escalate the wrong issue. (Contrast
        # ``detect_mergequeue_wedged``, whose ``issue_number`` DOES drive
        # ``_escalate_issue`` and is validator-threaded.)
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=branch_prefix,
        )
        if issue_number is not None and issue_number in ready_issue_numbers:
            bound.add(issue_number)
            bound_pr_numbers.add(pr_number)
        # isCrossRepository describes the PR's own head-branch provenance
        # (fork vs. same-repo), not which repo a free-text "#N" refers to.
        # It cannot fully guard a cross-repo mention collision, but it does
        # guard the common case of a fork PR's text being trusted at all.
        if pr.get("isCrossRepository") is False:
            for mentioned in issue_numbers_mentioned_by_pr(pr):
                if mentioned in ready_issue_numbers:
                    mention_only.add(mentioned)
                    mention_prs.setdefault(mentioned, []).append(pr_number)
    mention_only -= bound
    # Drop bound issues from the mention-PR map: ``bound`` takes
    # precedence, so a bound issue is never reported as mention-only.
    for bound_issue in bound:
        mention_prs.pop(bound_issue, None)
    mention_pr_numbers_by_issue = {
        issue_number: sorted(pr_numbers)
        for issue_number, pr_numbers in sorted(mention_prs.items())
    }
    return bound, mention_only, bound_pr_numbers, mention_pr_numbers_by_issue


def compute_mention_coverage_map(
    issues: list[dict[str, Any]],
    resolved_merged_prs: list[dict[str, Any]],
    app: Any,
) -> dict[int, list[int]]:
    """Issue #1337: compute the merged-PR mention coverage map for the
    backlog-reachability classifier.

    Returns ``issue_number -> sorted PR numbers`` for issues whose
    mention-only dispatch exclusion has NOT been lifted by operator
    re-arm. The classifier bins these as
    ``mention_covered_awaiting_operator`` instead of ``dispatchable``,
    mirroring the dispatch-side exclusion computed from the same
    ``scan_merged_pr_references`` + ``_mention_rearmed_issue_numbers``
    helpers ``_dispatch_impl`` uses -- so a change to the exclusion
    semantics automatically applies to the classifier (no second
    implementation to drift).

    ``bound`` exclusions are NOT included: a bound issue is closed by the
    dispatch path and leaves the open backlog, so it never reaches the
    classifier's ``dispatchable`` arm. Only mention-only issues stay open
    while being excluded -- the exact contradiction (#1059 reported
    dispatchable forever while dispatch silently dropped it) this map
    exists to resolve.

    Takes the already-resolved merged-PR list so the caller controls the
    fetch (and its error semantics): ``_dispatch_impl`` resolves once via
    ``_resolve_merged_prs`` and reuses the list for both the coverage map
    and the candidate filter (no second ``merged_pr_list`` call);
    ``status()`` fetches via ``fetch_merged_prs_fail_open`` so a transient
    ``GitHubError`` leaves the map empty (issues classify as
    ``dispatchable``) rather than crashing the status command -- the
    classifier is advisory and must not raise.
    """
    if not issues or not resolved_merged_prs:
        return {}
    _bound, mention_only, _bound_pr_numbers, mention_pr_numbers_by_issue = (
        app._merged_pr_referenced_issue_numbers(issues, resolved_merged_prs)
    )
    if not mention_only:
        return {}
    # Compute the re-arm lift using the same helper dispatch uses, so the
    # classifier mirrors the dispatch-side predicate exactly. The state
    # read holds the advisory lock (issue #310's
    # test_no_unlocked_load_state_in_production_code invariant).
    state = load_state_locked(app.paths.state_file)
    already_flagged = {
        int(num)
        for num, entry in state.get("issues", {}).items()
        if isinstance(entry, dict) and entry.get("merged_pr_mention_flagged_at")
    }
    rearmed, _ = app._mention_rearmed_issue_numbers(mention_only, issues, state, already_flagged)
    excluded = mention_only - rearmed
    return {
        issue_number: mention_pr_numbers_by_issue.get(issue_number, [])
        for issue_number in sorted(excluded)
    }


def fetch_merged_prs_fail_open(gh: GitHubLike) -> list[dict[str, Any]]:
    """Issue #1337: fetch ``merged_pr_list`` with fail-open semantics for
    the advisory reachability classifier.

    A ``GitHubError`` resolves to ``[]`` (issues classify as
    ``dispatchable``) rather than crashing the status command -- the
    classifier is advisory and must not raise. The dispatch path's own
    fetch (``_resolve_merged_prs``) re-fetches and RAISES on the same
    error so dispatch defers rather than re-dispatching covered issues.
    """
    try:
        return gh.merged_pr_list()
    except GitHubError:
        return []


def resolve_dispatch_mention_coverage(
    issues: list[dict[str, Any]],
    merged_prs_outcome: Any,
    gh: GitHubLike,
    app: Any,
) -> tuple[dict[int, list[int]], list[dict[str, Any]] | None]:
    """Issue #1337: resolve the mention-coverage map for ``_dispatch_impl``,
    reusing an already-fetched merged-PR list when available.

    Returns ``(mention_covered, fetched_merged_prs)``:

    * ``mention_covered`` -- the coverage map to pass to
      ``classify_backlog_reachability``.
    * ``fetched_merged_prs`` -- ``None`` when no fresh fetch happened (the
      outcome was already called, or the fetch failed); otherwise the
      freshly-fetched list. The caller rebinds ``merged_prs`` to a
      ``_MergedPRListOutcome(fetched, called=True)`` so the later
      ``_resolve_merged_prs`` calls and the tripwire reuse the same list
      -- no second API call.

    Three branches mirror the original inline wiring:

    1. ``issues`` is empty -> empty map, no fetch (matching
       ``_resolve_merged_prs``'s issue #361 guard).
    2. ``merged_prs_outcome`` was already fetched (``called=True``):
       reuse its items. An error outcome leaves the map empty (fail-open
       for the advisory classifier; ``_resolve_merged_prs`` re-raises so
       dispatch defers).
    3. Not yet fetched (``called=False`` -- the common case when there are
       no closed-ready issues): fetch ``merged_pr_list()`` here with
       fail-open try/except. A fetch failure leaves the map empty;
       ``_resolve_merged_prs`` re-fetches and RAISES so dispatch defers
       rather than re-dispatching covered issues.
    """
    if not issues:
        return {}, None
    if merged_prs_outcome is not None and merged_prs_outcome.called:
        if merged_prs_outcome.error is not None:
            return {}, None
        return compute_mention_coverage_map(issues, merged_prs_outcome.items, app), None
    try:
        fetched = gh.merged_pr_list()
    except GitHubError:
        return {}, None
    return compute_mention_coverage_map(issues, fetched, app), fetched
