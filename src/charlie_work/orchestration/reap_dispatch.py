"""Dead-worker-reap dispatch delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, leaf L06 (issue #1637; design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp`` (``self`` binds through the descriptor
protocol exactly as a lexical method did).

Names reached through ``_wf.`` (module-object seam, design Section 3.1 rule 2,
#1627): ``charlie_work.workflow`` module-level definitions ``CommandResult``,
``_MergedPRListOutcome`` (class),
``_state_lock_busy_result`` (free function); and Tier-D names patched on
``charlie_work.workflow`` by the suite, so the moved body must keep intercepting
those patches: ``_log_worker_census``. The live-session counters
(``_count_live_sessions``, ``count_fleet_live_sessions``) are reached through
``host.current().sessions``, whose Real late-binds ``charlie_work.workflow.*`` so
those patches still intercept. All other free names are imported directly from their
defining module (a three-form, six-alias patch census confirms no test patches
any of them on ``charlie_work.workflow``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.dispatch_deferral import records_deferral
from charlie_work import layout
from charlie_work.ci_absence import CiAbsence, runs_terminally_without_jobs
from charlie_work.ci_headroom import ci_headroom_available
from charlie_work.concurrency_governor_result import ConcurrencyGovernorResult
from charlie_work.fleet_paths import fleet_dir
from charlie_work.fleet_registry import registered_state_dirs, try_acquire_fleet_lock
from charlie_work.github import GitHubError, GraphQLBudgetError
from charlie_work.host_load import measure_host_load
from charlie_work.instrumentation import log_event
from charlie_work.janitor import JanitorVerdict
from charlie_work.safe_ref import require_valid_sha
from charlie_work.state import StateLockBusy
from charlie_work.worker_launch_gate import acquire_fleet_launch_lock
from charlie_work.dead_worker_sweep.effects_pr import _safe_repo_slug
from charlie_work.dead_worker_sweep.effects_rework import _is_pr_updated_at_older_than


def _detect_ci_absence(
    self,
    pr: dict[str, Any],
    verdict: JanitorVerdict,
    *,
    known_head: str | None = None,
    reprobe_known_head: bool = False,
) -> CiAbsence | None:
    """Classify a terminally-absent required check for this head, else None.

    Returns ``CiAbsence(kind="never_created")`` when Actions created no run
    object for the head, or ``CiAbsence(kind="workflow_no_jobs")`` (issue
    #1681) when it created runs but every one is ``completed`` and none can
    ever report the missing required checks. ``_detect_ci_run_never_created``
    is the original, never-created-only view of this.

    Original contract (``never_created``): the head SHA when Actions never
    created a run for it.

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

    The marker only records ``never_created``, so honoring it forever would
    blind this detector to a same-head transition to ``workflow_no_jobs``
    (e.g. the stale-checks retrigger itself creates a run for the head that
    then completes with no jobs). ``reprobe_known_head=True`` lets the caller
    lift the skip for one bounded re-query; the caller owns the bound (see
    ``review()``: once per ``stale_checks_retrigger_attempts`` value).
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
        head_sha = require_valid_sha(raw_head_sha, context="_detect_ci_absence head_sha")
    except ValueError:
        return None
    if known_head is not None and known_head == head_sha and not reprobe_known_head:
        return None
    if not _is_pr_updated_at_older_than(pr, datetime.now(UTC), grace_minutes):
        return None
    head_runs = self.gh.workflow_runs_for_head(head_sha)
    # None means the query itself failed (rate limit, transient error) --
    # fail closed: only a successful, empty response is positive evidence
    # of "never created", never the absence of a successful response.
    if head_runs is None:
        return None
    if len(head_runs) == 0:
        return CiAbsence(kind="never_created", head_sha=head_sha)
    # Issue #1681: a run object exists, so "never created" is structurally
    # blind to this head; every run being completed with no jobs is the
    # terminal "workflow file rejected" signature.
    if runs_terminally_without_jobs(head_runs):
        return CiAbsence(kind="workflow_no_jobs", head_sha=head_sha)
    return None


def _detect_ci_run_never_created(
    self,
    pr: dict[str, Any],
    verdict: JanitorVerdict,
    *,
    known_head: str | None = None,
) -> str | None:
    """Return the head SHA when Actions never created a run for it, else None.

    The never-created-only view of ``_detect_ci_absence`` (see there for the
    full rationale); the escalated-PR path and the stale-checks retrigger
    consume this. Does not take ``state_lock``.
    """
    absence = self._detect_ci_absence(pr, verdict, known_head=known_head)
    if absence is not None and absence.kind == "never_created":
        return absence.head_sha
    return None


def _apply_concurrency_governor(
    self,
    dispatch_limit: int,
    *,
    live_count: int | None = None,
    apply_open_pr_backpressure: bool = False,
) -> ConcurrencyGovernorResult:
    """Apply global concurrency governor cap to a dispatch limit.

    Returns a ConcurrencyGovernorResult with the potentially-clamped limit
    and all related fields. This eliminates Pyright's reportPossiblyUnbound
    warnings by ensuring live_count is always bound together with the
    clamped flag.

    Args:
        dispatch_limit: The requested dispatch limit
        live_count: Optional pre-computed live worker count. If None and
            max_concurrent > 0 or ci_capacity_headroom_ratio > 0, this will
            compute it via _count_live_sessions -- the CI-headroom clamp
            below needs it too, as a floor on demand ci_fleet's last
            allocation pass has not measured yet (issue #1770 review
            finding 3), independent of whether max_concurrent itself is
            enabled.
        apply_open_pr_backpressure: When True (fresh-issue dispatch only),
            also clamp to ``max(0, max_open_agent_prs - open_pr_count)``
            where ``open_pr_count`` is the number of open agent PRs whose
            head ref matches ``dispatch.branch_prefix``, and (issue #1770)
            to the repo's live CI headroom (see ``ci_capacity_headroom_ratio``
            below) -- ``open_pr_count`` is likewise computed whenever either
            of those two terms is enabled, not only ``max_open_agent_prs``,
            for the same floor-on-demand reason. Rework, recovery, and
            loop-level callers leave this False -- they reduce verification
            debt (open-PR count, CI demand) rather than adding to it (issue
            #1129).
    """
    max_concurrent = self.config.dispatch.max_concurrent_sessions
    fleet_max = self.config.fleet.global_max_concurrent_sessions
    open_pr_max = self.config.dispatch.max_open_agent_prs if apply_open_pr_backpressure else 0
    ci_headroom_ratio = (
        self.config.dispatch.ci_capacity_headroom_ratio if apply_open_pr_backpressure else 0.0
    )
    # Issue #1843: unlike the two WIP-shaping terms above, the host-load
    # clamp applies to EVERY caller -- rework and recovery launches spawn
    # real local suites too, and the host does not care which lane
    # oversubscribed it. Read unconditionally; the probe below still only
    # runs when a launch could actually happen (dispatch_limit > 0).
    host_load_max = self.config.dispatch.host_load_max_pytest_processes
    host_load_max_trees = self.config.dispatch.host_load_max_pytest_trees
    # Issue #1770 review finding 10: captured once, before any term below can
    # tighten ``dispatch_limit``, so every ``dispatch_backpressure`` event
    # this call writes reports the same "requested" baseline -- the caller's
    # original ask -- rather than whatever the running value happened to be
    # when that particular term fired. Two clamps firing in one pass (e.g.
    # open_pr_max then ci_headroom) would otherwise write two events with
    # different, unlabelled "requested_limit" values, and reconstructing the
    # pass would require knowing the term order.
    original_dispatch_limit = dispatch_limit
    available_slots = dispatch_limit
    clamped = False
    clamped_by: str | None = None
    fleet_live_count = 0
    open_pr_count = 0
    ci_headroom: int | None = None
    host_load_reading = None

    # Issue #1770 review finding 3: live_count/open_pr_count are also the
    # cheapest available floor on "demand ci_fleet's last allocation pass
    # cannot see yet" (a freshly dispatched worker, or an already-open PR,
    # neither of which necessarily has a queued/in_progress Actions run at
    # the instant of that snapshot). Computed whenever the CI-headroom clamp
    # is live, not only when its own governor term (max_concurrent/
    # open_pr_max) is independently enabled, so the floor is populated even
    # for a repo that opted into *only* the CI-headroom clamp.
    if max_concurrent > 0 or ci_headroom_ratio > 0:
        if live_count is None:
            sessions_dir = self._layout.sessions_dir
            live_count = self.host.sessions.live_workers(sessions_dir, self.paths.state_file)

    if max_concurrent > 0:
        available_slots = max(0, max_concurrent - (live_count or 0))
        if available_slots < dispatch_limit:
            dispatch_limit = available_slots
            clamped = True
            clamped_by = "max_concurrent"

    if fleet_max > 0:
        fleet_live_count, _skipped_repos = self.host.sessions.fleet_live_workers(
            self.fleet_dir_override
        )
        fleet_available = max(0, fleet_max - fleet_live_count)
        if fleet_available < dispatch_limit:
            dispatch_limit = fleet_available
            clamped = True
            clamped_by = "fleet_max"

    if open_pr_max > 0 or ci_headroom_ratio > 0:
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

    if open_pr_max > 0:
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
            # instrumentation. The clamp itself (dispatch_limit/clamped
            # below) is NOT dry-run gated: a dry-run preview must report the
            # same clamped selected_count a live pass would, matching the
            # provider_throttled deferral precedent in _dispatch_impl.
            if not self.dry_run:
                log_event(
                    self.paths.state_file,
                    "dispatch_backpressure",
                    {
                        "open_pr_count": open_pr_count,
                        "max_open_agent_prs": open_pr_max,
                        "requested_limit": original_dispatch_limit,
                        "clamped_limit": open_pr_available,
                    },
                    repo=self.repo_root.name,
                )
            dispatch_limit = open_pr_available
            clamped = True
            clamped_by = "open_pr_max"

    if ci_headroom_ratio > 0:
        # Issue #1770: clamp fresh dispatch to the repo's live CI headroom.
        # ``_safe_repo_slug`` returns "?" (never raises) when the GitHub
        # lookup itself fails -- passing that through is deliberate rather
        # than a special case: "?" never matches a real ``ci_fleet`` target,
        # so ``ci_headroom_available`` takes its own "unconfigured" fail-open
        # path for it, the same one a genuinely hosted-runner repo (no
        # self-hosted registration, so no entry in ci_fleet's plan at all --
        # derived from ci_fleet's own live discovery, never a repo-name
        # list) takes. One fail-open path covers both cases.
        fleet_state_path = layout.state_file_path(fleet_dir(override=self.fleet_dir_override))
        ci_headroom = ci_headroom_available(
            _safe_repo_slug(self.gh),
            headroom_ratio=ci_headroom_ratio,
            fleet_state_path=fleet_state_path,
            # Finding 3: floor demand at the local work already in flight
            # for this repo (live worker sessions + already-open agent PRs)
            # so a burst of fresh dispatch cannot keep re-granting the same
            # full headroom pass after pass while ci_fleet's own snapshot
            # still lags behind it.
            min_in_flight_demand=(live_count or 0) + open_pr_count,
            # Finding 8: route the fail-open diagnostic to THIS repo's own
            # events.db under the same repo spelling dispatch_backpressure
            # below uses, not the fleet-wide store the runner_allocation
            # read above requires -- so both halves of one clamp decision
            # are discoverable together from one repo's event store.
            diagnostic_state_path=self.paths.state_file,
            diagnostic_repo=self.repo_root.name,
        )
        if ci_headroom is not None and ci_headroom < dispatch_limit:
            # Same dispatch_backpressure event kind open_pr_max writes to
            # above (design doc Section 4 step 3): one existing
            # consumer/dashboard sees both reasons, distinguished by
            # ``clamped_by``. Same dry-run write-suppression discipline as
            # the open_pr_max block -- the clamp itself always applies so a
            # dry-run preview matches a live pass's selected_count.
            if not self.dry_run:
                log_event(
                    self.paths.state_file,
                    "dispatch_backpressure",
                    {
                        "clamped_by": "ci_headroom",
                        "ci_headroom": ci_headroom,
                        "ci_headroom_ratio": ci_headroom_ratio,
                        "requested_limit": original_dispatch_limit,
                        "clamped_limit": ci_headroom,
                    },
                    repo=self.repo_root.name,
                )
            dispatch_limit = ci_headroom
            clamped = True
            clamped_by = "ci_headroom"

    if (host_load_max > 0 or host_load_max_trees > 0) and dispatch_limit > 0:
        # Issue #1843: defer worker launches while the host is saturated.
        # Applies to every governor caller (loop wave budget, rework, fresh
        # dispatch) -- a launch spawns a real local suite whichever lane asks
        # for it, unlike the WIP-shaping terms above. Probed only when the
        # running limit is still positive: a pass already clamped to 0
        # cannot launch anyway, so spending a subprocess spawn on the probe
        # would buy no decision. ``measure_host_load`` returns None on probe
        # failure (fail-open -- dispatch proceeds; it logs a rate-limited
        # host_load_unavailable event itself), so only a real over-threshold
        # reading ever reaches the clamp.
        # Issue #1943: the reading is scoped to orchestrator-attributable
        # trees. ``measure_host_load`` builds in the ``.var/charlie-work``
        # state-dir convention marker; the paths below add this repo's
        # resolved roots (``runtime.state_dir`` and
        # ``claude_code.worktrees_dir`` overrides included -- the latter is
        # only reachable via ``self._layout.worktrees``, never
        # ``self.paths.worktrees``) and every fleet-registered ``state_dir``,
        # so a sibling repo's overridden layout stays attributable too. A
        # suite with no managed path in any member/ancestor command line --
        # e.g. a CI runner's tree under ``C:\actions-runners\*`` -- no longer
        # feeds either count.
        host_load_reading = measure_host_load(
            diagnostic_state_path=self.paths.state_file,
            diagnostic_repo=self.repo_root.name,
            scope_paths=(
                self.paths.root,
                self._layout.worktrees,
                *registered_state_dirs(self.fleet_dir_override),
            ),
        )
        host_load_limit: int | None = None
        host_load_term: str | None = None
        if host_load_reading is not None:
            # Issue #1903: two terms over one reading. The process term is
            # the fan-out brake -- raw count scales with ``-n``, not with
            # real load, so it keeps its strict ``>`` trip to 0 and only
            # fires on abnormal suite width (its recalibrated default is
            # ~3x cores). The tree term is the actual governor: suite
            # count is in the same units as ``dispatch_limit`` (one launch
            # ≈ one suite), so it clamps by headroom
            # (``cap - live trees``) and grants partial capacity near the
            # cap instead of dropping straight to 0. The brake is checked
            # first: a single hyper-wide suite can hold process count
            # over its cap while tree count stays low, and that is
            # exactly the case where granting headroom would be wrong.
            if host_load_max > 0 and (host_load_reading.pytest_process_count > host_load_max):
                host_load_limit = 0
                host_load_term = "pytest_processes"
            elif host_load_max_trees > 0:
                tree_headroom = max(0, host_load_max_trees - host_load_reading.pytest_tree_count)
                if tree_headroom < dispatch_limit:
                    host_load_limit = tree_headroom
                    host_load_term = "pytest_trees"
            if host_load_limit is not None:
                # Same dispatch_backpressure kind as the
                # open_pr_max/ci_headroom clamps above: one existing
                # consumer sees every deferral reason, distinguished by
                # ``clamped_by``. Same dry-run write-suppression
                # discipline -- the clamp applies either way so a dry-run
                # preview reports the same deferral a live pass would.
                if not self.dry_run:
                    log_event(
                        self.paths.state_file,
                        "dispatch_backpressure",
                        {
                            "clamped_by": "host_load",
                            "host_load_term": host_load_term,
                            "host_load_pytest_processes": host_load_reading.pytest_process_count,
                            "host_load_pytest_trees": host_load_reading.pytest_tree_count,
                            "host_load_max_pytest_processes": host_load_max,
                            "host_load_max_pytest_trees": host_load_max_trees,
                            "requested_limit": original_dispatch_limit,
                            "clamped_limit": host_load_limit,
                        },
                        repo=self.repo_root.name,
                    )
                dispatch_limit = host_load_limit
                clamped = True
                clamped_by = "host_load"

    return ConcurrencyGovernorResult(
        clamped=clamped,
        max_concurrent=max_concurrent,
        live_count=live_count or 0,
        available_slots=available_slots,
        dispatch_limit=dispatch_limit,
        fleet_live_count=fleet_live_count,
        fleet_max=fleet_max,
        open_pr_count=open_pr_count,
        open_pr_max=open_pr_max,
        ci_headroom=ci_headroom,
        ci_headroom_ratio=ci_headroom_ratio,
        host_load_max_pytest_processes=host_load_max,
        host_load_max_pytest_trees=host_load_max_trees,
        host_load_pytest_processes=(
            host_load_reading.pytest_process_count if host_load_reading is not None else None
        ),
        host_load_pytest_trees=(
            host_load_reading.pytest_tree_count if host_load_reading is not None else None
        ),
        clamped_by=clamped_by,
    )


@records_deferral("dispatch")
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

    # Issue #2055: mint the fleet-launch-lock handle here so the ``finally``
    # below covers every impl exit path, but do NOT take the OS lock yet --
    # the pending handle is realized by issue_worker_launch_permit (bounded
    # wait) immediately before the governor, after _dispatch_impl's
    # issue_list / blocker-prefetch / reachability / stall-sweep scan.
    launch_lock = acquire_fleet_launch_lock(self, acquire=try_acquire_fleet_lock)
    try:
        result = self._dispatch_impl(
            limit,
            only_issues=only_issues,
            stalled_entries=stalled_entries,
            ready_issues=ready_issues,
            merged_prs=merged_pr_outcome,
            launch_lock=launch_lock,
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
        launch_lock.release()


def _probe_ci_absence(
    self,
    pr: dict[str, Any],
    verdict: JanitorVerdict,
    pr_state: dict[str, Any] | None,
) -> tuple[str | None, str | None, int]:
    """Run ``_detect_ci_absence`` for ``review()``'s janitor gate (issue #1681).

    Returns ``(never_created_head, workflow_no_jobs_head, attempts)``. A head
    already recorded as never-created is re-probed once per
    ``stale_checks_retrigger_attempts`` value (bounded by the retrigger cap), so
    a same-head transition to ``workflow_no_jobs`` is not masked by the dedup
    marker. Does NOT take ``state_lock``.
    """
    pr_state = pr_state or {}
    raw_attempts = pr_state.get("stale_checks_retrigger_attempts")
    attempts = raw_attempts if isinstance(raw_attempts, int) else 0
    # ``workflow_no_jobs`` is terminal for a head: the persisted marker is the
    # classification, so a later same-head pass must not fall back to the
    # (stale) ``ci_run_never_created_head`` marker, which would re-blind the
    # detector and drop the PR into the retrigger lane. Still gated on the
    # live "required check missing" signal and the current head matching.
    cached_no_jobs_head = pr_state.get("workflow_no_jobs_head")
    if (
        cached_no_jobs_head
        and verdict.missing_required_checks
        and str(pr.get("headRefOid") or "") == cached_no_jobs_head
    ):
        return None, cached_no_jobs_head, attempts
    absence = self._detect_ci_absence(
        pr,
        verdict,
        known_head=pr_state.get("ci_run_never_created_head"),
        reprobe_known_head=pr_state.get("ci_absence_probed_attempts") != attempts,
    )
    kind = absence.kind if absence is not None else None
    head_sha = absence.head_sha if absence is not None else None
    return (
        head_sha if kind == "never_created" else None,
        head_sha if kind == "workflow_no_jobs" else None,
        attempts,
    )


def _record_ci_absence_state(
    self,
    state: dict[str, Any],
    pr_state_update: dict[str, Any],
    existing_pr_state: dict[str, Any],
    pr: dict[str, Any],
    verdict: JanitorVerdict,
    issue_number: int | None,
    *,
    never_created_head: str | None,
    workflow_no_jobs_head: str | None,
    attempts: int,
) -> dict[str, Any]:
    """Persist the absence markers + once-per-head events; return ``state``.

    Must be called inside ``state_lock``. Marker writes go into
    ``pr_state_update`` (the caller's fresh dict) which is then stored on
    ``state``. The ``workflow_no_jobs`` event dedup is event-only: routing is
    NOT gated on the marker, so a rework that pushes nothing new is re-routed
    (and capped by ``record_review``), never re-parked.
    """
    pr_number = pr_state_update["number"]
    pr_state_update["ci_absence_probed_attempts"] = attempts
    no_jobs_new = (
        workflow_no_jobs_head is not None
        and existing_pr_state.get("workflow_no_jobs_head") != workflow_no_jobs_head
    )
    never_created_new = (
        never_created_head is not None
        and existing_pr_state.get("ci_run_never_created_head") != never_created_head
    )
    if workflow_no_jobs_head is not None:
        pr_state_update["workflow_no_jobs_head"] = workflow_no_jobs_head
    if never_created_head is not None:
        pr_state_update["ci_run_never_created_head"] = never_created_head
    state["prs"][str(pr_number)] = pr_state_update

    def payload(head: str | None) -> dict[str, Any]:
        return {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "head_sha": head,
            "branch": pr.get("headRefName"),
            "missing_checks": list(verdict.missing_required_checks),
        }

    if no_jobs_new:
        state = self._record_event(state, "workflow_no_jobs", payload(workflow_no_jobs_head))
    if never_created_new:
        state = self._record_event(state, "ci_run_never_created", payload(never_created_head))
    return state


def _route_workflow_no_jobs(
    self,
    pr: dict[str, Any],
    pr_number: int,
    issue_number: int,
    verdict: JanitorVerdict,
    head_sha: str,
) -> Any:
    """Route a ``workflow_no_jobs`` head to rework (issue #1681).

    Retrigger cannot fix a rejected workflow file, so ``review()`` returns this
    instead of re-parking the PR as ``janitor_blocked``.
    """
    _wf.transition(self.gh, self.config.labels, issue_number, "review_started")
    missing = ", ".join(verdict.missing_required_checks)
    diagnostic = (
        f"workflow file invalid: run completed with no jobs created for head "
        f"{head_sha}; required check(s) {missing} can never "
        f"report for this head. Fix the workflow file and push."
    )
    return self.record_review(
        pr_number,
        "request_changes",
        summary=diagnostic,
        reviewed_head=pr.get("headRefOid"),
        required_changes=[diagnostic],
        verdict_provenance="ci_gate_auto_reject",
    )
