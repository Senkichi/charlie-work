"""Dead-worker-reap dispatch delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, leaf L06 (issue #1637; design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp`` (``self`` binds through the descriptor
protocol exactly as a lexical method did).

Names reached through ``_wf.`` (module-object seam, design Section 3.1 rule 2,
#1627): ``charlie_work.workflow`` module-level definitions ``CommandResult``,
``ConcurrencyGovernorResult``, ``_MergedPRListOutcome`` (classes),
``_state_lock_busy_result`` (free function); and Tier-D names patched on
``charlie_work.workflow`` by the suite, so the moved body must keep intercepting
those patches: ``_count_live_sessions``, ``count_fleet_live_sessions``,
``_log_worker_census``. All other free names are imported directly from their
defining module (a three-form, six-alias patch census confirms no test patches
any of them on ``charlie_work.workflow``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.github import GitHubError, GraphQLBudgetError
from charlie_work.instrumentation import log_event
from charlie_work.janitor import JanitorVerdict
from charlie_work.safe_ref import require_valid_sha
from charlie_work.state import StateLockBusy
from charlie_work.dead_worker_reap import _is_pr_updated_at_older_than


def _detect_ci_run_never_created(
    self,
    pr: dict[str, Any],
    verdict: JanitorVerdict,
    *,
    known_head: str | None = None,
) -> str | None:
    """Return the head SHA when Actions never created a run for it, else None.

    A "Required check(s) missing" janitor failure is ambiguous between
    "still pending" and "GitHub never created a workflow run for this
    head at all" -- a sibling repo measured 11 PRs stuck behind this exact
    failure for 4+ days: the webhook delivered (other check-suite apps
    registered) but no github-actions run object ever appeared, and
    nothing distinguished that from ordinary CI latency. This queries
    Actions directly, once the head has had a grace period to actually
    start CI, so the distinction is diagnosable via events.db instead of
    blending into the janitor gate's generic bookkeeping. Detection
    only -- no retry or retrigger here; that is follow-up policy.

    Shared by both the escalated-PR path (``review()``'s early-return
    branch, which recomputes janitor diagnostics for visibility only)
    and the normal janitor-gate path below it -- a PR stuck 4+ days is
    itself a strong escalation candidate, so the check must not be
    reachable only from the non-escalated branch.

    Does NOT take ``state_lock`` and must never be called while holding
    it -- it makes a ``gh api`` call, and this repo's state lock must
    never span external I/O (mopup-rate-limit-lock-contention). Callers
    take the lock only to persist the already-computed result.

    ``known_head`` is the previously-persisted ``ci_run_never_created_head``
    marker, if any. When it matches the PR's current head, the event has
    already fired for this exact head and would be deduped anyway, so the
    ``gh api`` call is skipped -- otherwise a PR that stays escalated for
    days would re-query Actions on every single pass forever (this is the
    same population the grace period targets).
    """
    if not verdict.missing_required_checks:
        return None
    grace_minutes = self.config.auto_merge.ci_run_never_created_grace_minutes
    if grace_minutes <= 0:
        return None
    raw_head_sha = str(pr.get("headRefOid") or "") or None
    if raw_head_sha is None:
        return None
    try:
        head_sha = require_valid_sha(raw_head_sha, context="_detect_ci_run_never_created head_sha")
    except ValueError:
        return None
    if known_head is not None and known_head == head_sha:
        return None
    if not _is_pr_updated_at_older_than(pr, datetime.now(UTC), grace_minutes):
        return None
    head_runs = self.gh.workflow_runs_for_head(head_sha)
    # None means the query itself failed (rate limit, transient error) --
    # fail closed: only a successful, empty response is positive evidence
    # of "never created", never the absence of a successful response.
    if head_runs is not None and len(head_runs) == 0:
        return head_sha
    return None


def _apply_concurrency_governor(
    self,
    dispatch_limit: int,
    *,
    live_count: int | None = None,
    apply_open_pr_backpressure: bool = False,
) -> _wf.ConcurrencyGovernorResult:
    """Apply global concurrency governor cap to a dispatch limit.

    Returns a ConcurrencyGovernorResult with the potentially-clamped limit
    and all related fields. This eliminates Pyright's reportPossiblyUnbound
    warnings by ensuring live_count is always bound together with the
    clamped flag.

    Args:
        dispatch_limit: The requested dispatch limit
        live_count: Optional pre-computed live worker count. If None and
            max_concurrent > 0, this will compute it via _count_live_sessions.
        apply_open_pr_backpressure: When True (fresh-issue dispatch only),
            also clamp to ``max(0, max_open_agent_prs - open_pr_count)``
            where ``open_pr_count`` is the number of open agent PRs whose
            head ref matches ``dispatch.branch_prefix``. Rework, recovery,
            and loop-level callers leave this False -- they reduce
            verification debt rather than adding to it (issue #1129).
    """
    max_concurrent = self.config.dispatch.max_concurrent_sessions
    fleet_max = self.config.fleet.global_max_concurrent_sessions
    open_pr_max = self.config.dispatch.max_open_agent_prs if apply_open_pr_backpressure else 0
    available_slots = dispatch_limit
    clamped = False
    fleet_live_count = 0
    open_pr_count = 0

    if max_concurrent > 0:
        if live_count is None:
            sessions_dir = self._layout.sessions_dir
            live_count = _wf._count_live_sessions(sessions_dir, self.paths.state_file)
        available_slots = max(0, max_concurrent - live_count)
        if available_slots < dispatch_limit:
            dispatch_limit = available_slots
            clamped = True

    if fleet_max > 0:
        fleet_live_count, _skipped_repos = _wf.count_fleet_live_sessions(self.fleet_dir_override)
        fleet_available = max(0, fleet_max - fleet_live_count)
        if fleet_available < dispatch_limit:
            dispatch_limit = fleet_available
            clamped = True

    if open_pr_max > 0:
        # Issue #1129: count open agent PRs from live GitHub state (the
        # same pr_list() + branch_prefix derivation _merge_train_candidates
        # and the reconciler use). No new state; the count is recomputed
        # each pass and self-corrects after merges/closes.
        branch_prefix = self.config.dispatch.branch_prefix
        open_pr_count = sum(
            1
            for pr in self.gh.pr_list()
            if str(pr.get("headRefName") or "").startswith(branch_prefix)
        )
        open_pr_available = max(0, open_pr_max - open_pr_count)
        if open_pr_available < dispatch_limit:
            # Record a dispatch_backpressure event so "0 dispatched with N
            # dispatchable" is diagnosable from events.db rather than
            # reading as idleness (same discipline as #1091's de-escalation
            # skip attribution). The governor runs outside the state lock,
            # so log_event (the low-level write primitive for events
            # outside state-lock contexts) is used directly.
            #
            # Dry-run never writes the event: log_event is a durable
            # events.db mutation, and a dry-run preview must not record
            # instrumentation (the same write-suppression discipline as the
            # worker_token_escalated marker above -- the escalation event
            # and durable marker stay behind ``not self.dry_run``). The
            # clamp itself (dispatch_limit/clamped below) is NOT dry-run
            # gated: a dry-run preview must report the same clamped
            # selected_count a live pass would, matching the
            # worker_token_missing refusal precedent in _dispatch_impl.
            if not self.dry_run:
                log_event(
                    self.paths.state_file,
                    "dispatch_backpressure",
                    {
                        "open_pr_count": open_pr_count,
                        "max_open_agent_prs": open_pr_max,
                        "requested_limit": dispatch_limit,
                        "clamped_limit": open_pr_available,
                    },
                    repo=self.repo_root.name,
                )
            dispatch_limit = open_pr_available
            clamped = True

    return _wf.ConcurrencyGovernorResult(
        clamped=clamped,
        max_concurrent=max_concurrent,
        live_count=live_count or 0,
        available_slots=available_slots,
        dispatch_limit=dispatch_limit,
        fleet_live_count=fleet_live_count,
        fleet_max=fleet_max,
        open_pr_count=open_pr_count,
        open_pr_max=open_pr_max,
    )


def dispatch(
    self,
    limit: int | None = None,
    *,
    only_issues: str | None = None,
    stalled_entries: list[dict[str, int]] | None = None,
) -> _wf.CommandResult:
    """Dispatch fresh workers for ready issues.

    ``stalled_entries``: pass the result of an already-completed
    ``_detect_and_handle_stalled_sessions`` sweep to reuse it instead of
    re-running the sweep inside this call. ``loop()`` does this because it
    runs the sweep itself at the top of each pass; the sweep is the sole
    writer of Signal-1's inconclusive-probe deferral counter, so re-running
    it here would advance that counter more than once per pass and erode
    the ``max_inconclusive_probe_deferrals`` grace period (issue #343
    Finding 2). Standalone callers leave this as None and the sweep runs
    inside this call as before.
    """
    # Issue #646: unconditional census of every alive worker, logged before
    # any guard below can short-circuit (state lock busy, fleet lock held,
    # GraphQL budget deferred) -- this is the one chokepoint every dispatch
    # path funnels through, whether invoked standalone (`work`/`fleet work`)
    # or from inside a supervised pass (`loop()` -> `_loop_body()` ->
    # `dispatch()`), so it answers "how many suites were running at <time>,
    # from which worktrees, at what cap" regardless of which command
    # launched them. Purely read-only, but explicitly guarded: per-file
    # read errors are already swallowed inside read_worker_records/
    # read_session_records, but this diagnostic must never be the reason a
    # whole dispatch pass aborts, so any other unexpected failure here
    # (formatting, directory-listing races, etc.) is logged and swallowed
    # rather than propagated -- a torn sidecar read is *more* likely, not
    # less, during the exact high-concurrency moment this census exists to
    # diagnose.
    try:
        _wf._log_worker_census(self._layout.sessions_dir)
    except Exception:
        import logging

        logging.getLogger(__name__).warning("worker census failed", exc_info=True)
    # Finalize closed ready-labeled issues whose linked PR merged externally.
    # This runs before fleet lock / GraphQL budget / provider throttle guards
    # so a pass that defers new dispatch still drains the Aviator-merge backlog.
    finalized: set[int] = set()
    merged_pr_outcome: _wf._MergedPRListOutcome = _wf._MergedPRListOutcome()
    try:
        finalized, ready_issues, merged_pr_outcome = self._finalize_externally_merged_issues()
    except StateLockBusy:
        return _wf._state_lock_busy_result(
            "dispatch deferred: state lock held",
            selected_count=0,
            deferred_reason="state_lock_busy",
        )

    # Reuse the merged PR list already fetched by _finalize_externally_merged_issues
    # so the post-merge tripwire in loop() can avoid a second GraphQL call.
    merged_prs_for_result: list[dict[str, Any]] | None = (
        merged_pr_outcome.items
        if merged_pr_outcome.called and merged_pr_outcome.error is None
        else None
    )

    fleet_lock = None
    if self.config.fleet.global_max_concurrent_sessions > 0:
        fleet_lock = try_acquire_fleet_lock(self.fleet_dir_override)
        if fleet_lock is None:
            return _wf.CommandResult(
                True,
                "dispatch deferred: fleet lock held",
                {
                    "selected_count": 0,
                    "deferred_reason": "fleet_lock_held",
                    "merged_prs": merged_prs_for_result,
                    "merged_pr_closed_issue_numbers": sorted(finalized),
                    "merged_pr_referenced_issue_numbers": sorted(finalized),
                },
            )
    try:
        result = self._dispatch_impl(
            limit,
            only_issues=only_issues,
            stalled_entries=stalled_entries,
            ready_issues=ready_issues,
            merged_prs=merged_pr_outcome,
        )
        data = dict(result.data)
        if finalized:
            data["merged_pr_closed_issue_numbers"] = sorted(
                set(data.get("merged_pr_closed_issue_numbers", [])) | finalized
            )
            data["merged_pr_referenced_issue_numbers"] = sorted(
                set(data.get("merged_pr_referenced_issue_numbers", [])) | finalized
            )
        return _wf.CommandResult(result.ok, result.message, data)
    except StateLockBusy:
        return _wf._state_lock_busy_result(
            "dispatch deferred: state lock held",
            selected_count=0,
            deferred_reason="state_lock_busy",
            merged_prs=merged_prs_for_result,
            merged_pr_closed_issue_numbers=sorted(finalized),
            merged_pr_referenced_issue_numbers=sorted(finalized),
        )
    except GraphQLBudgetError as exc:
        return _wf.CommandResult(
            True,
            "dispatch deferred: GraphQL rate limit below threshold",
            {
                "selected_count": 0,
                "deferred_reason": "graphql_rate_limit",
                "graphql_remaining": exc.remaining,
                "graphql_reset": exc.reset_at,
                "graphql_threshold": exc.threshold,
                "merged_prs": merged_prs_for_result,
                "merged_pr_closed_issue_numbers": sorted(finalized),
                "merged_pr_referenced_issue_numbers": sorted(finalized),
            },
        )
    except GitHubError as exc:
        # A GitHubError from _dispatch_impl means a GitHub API call
        # needed for reliable dispatch failed. The two known sources are
        # merged_pr_list() (raised on unusable responses — empty stdout,
        # non-zero exit, unparseable JSON — per #633) and pr_list(); both
        # are fetched before any issue is claimed or worker launched, so
        # deferring here cannot leave a partial claim. Earlier
        # _finalize_externally_merged_issues already recorded its own
        # merged_pr_list failure in merged_pr_outcome, and
        # _resolve_merged_prs re-raises that stored error (branch 2) so
        # this handler covers BOTH the direct-fallback fetch (branch 1,
        # the common case when there are open ready issues but no
        # closed-ready issues this pass) and the finalize-errored re-raise
        # (branch 2). Deferring is the correct response: proceeding with
        # an empty merged-PR set would re-dispatch issues a merged PR
        # already covered (the silent-empty path #633 closed), and letting
        # the error propagate crashes the supervised loop daemon on a
        # transient gh failure. Any claim written before a later
        # GitHubError (e.g. issue_view mid-launch) is recovered by the
        # existing stale-claim sweep on the next pass.
        return _wf.CommandResult(
            True,
            f"dispatch deferred: GitHub API error ({exc})",
            {
                "selected_count": 0,
                "deferred_reason": "github_error",
                "github_error": str(exc),
                "merged_prs": merged_prs_for_result,
                "merged_pr_closed_issue_numbers": sorted(finalized),
                "merged_pr_referenced_issue_numbers": sorted(finalized),
            },
        )
    finally:
        if fleet_lock is not None:
            fleet_lock.release()
