"""Loop-tick maintenance-singleton delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 3 (issue #1646, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import charlie_work.workflow as _wf


def _maybe_probe_quota_recovery(self, *, now: datetime | None = None) -> None:
    """Flat-interval Haiku probe for early quota/rate-limit throttle recovery.

    Runs every loop pass but is a near-no-op unless a throttle that a
    green ambient-CLI probe could actually clear is currently active
    (``is_quota_probe_actionable``) -- an operator switching to a
    different subscription account can make a provider's quota recover
    well before a blanket cooldown (e.g. the 24h
    ``_DEFAULT_QUOTA_COOLDOWN_HOURS`` window) elapses, and there was
    previously no way to detect that early. Deliberately a *flat*
    ``quota_probe.interval_minutes`` schedule (not exponential backoff
    like ``reviewer_quota.probe_after``): the user asked for a fixed
    15-minute recheck, not a growing wait.

    The gate is narrower than "any throttle indicator" -- a devin/api
    adapter throttle or a provider_auth cooldown would survive
    ``clear_quota_throttles`` untouched even on a green probe (see that
    function), so arming/probing for one would just burn Haiku sessions
    every interval for no possible benefit. Because of this, a green
    probe reaching the success branch below is guaranteed to have
    something to clear -- no post-hoc check needed there.

    Two-phase lock pattern so the (possibly tens-of-seconds) subprocess
    call in ``run_quota_probe`` never holds ``state_lock`` and blocks
    every other state read/writer in the process:
      1. Under the lock: decide whether to probe at all, and if not,
         arm/disarm/return without calling out to the CLI.
      2. Outside the lock: run the actual probe subprocess.
      3. Under the lock again: re-read state (it may have changed while
         unlocked) and apply the outcome.

    ``now`` (issue #828) is the injectable clock for this whole method:
    both internal ``datetime.now(UTC)`` samples below (the first-arm
    branch and the red-probe reschedule branch) resolve from the same
    seeded instant instead of each independently racing the wall clock.
    Defaults to ``datetime.now(UTC)`` when omitted, so production
    behavior is byte-identical; tests can freeze it and assert exact
    equality on the scheduled ``next_probe_at`` instead of a
    wall-clock-tolerance proximity check.
    """
    if not self.config.quota_probe.enabled:
        return

    resolved_now = now if now is not None else datetime.now(UTC)
    state_file = self.paths.state_file
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if not _wf.is_quota_probe_actionable(state):
            if _wf.is_quota_probe_armed(state):
                state = _wf.disarm_quota_probe(state)
                _wf.save_state(state_file, state)
            return
        if not _wf.is_quota_probe_armed(state):
            next_probe_at = (
                (resolved_now + timedelta(minutes=self.config.quota_probe.interval_minutes))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            state = _wf.arm_quota_probe(state, next_probe_at)
            _wf.save_state(state_file, state)
            return
        if not _wf.is_quota_probe_due(state):
            return

    probe_ok = _wf.run_quota_probe(repo_root=self.repo_root, config=self.config)

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if not _wf.is_quota_probe_actionable(state):
            # Cleared or became non-actionable while the probe ran unlocked.
            state = _wf.disarm_quota_probe(state)
            _wf.save_state(state_file, state)
            return
        if probe_ok:
            state = _wf.clear_quota_throttles(state)
            state = _wf.disarm_quota_probe(state)
            state = self._record_event(state, "quota_probe_succeeded", {})
        else:
            next_probe_at = (
                (resolved_now + timedelta(minutes=self.config.quota_probe.interval_minutes))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            state = _wf.arm_quota_probe(state, next_probe_at)
            # Share the failure with dispatch_reviews's probe_mode gate:
            # a red flat probe confirmed the window is still closed, so
            # also bump reviewer_quota.probe_after (when the reviewer
            # quota is exhausted) to stop dispatch_reviews from
            # independently launching a real reviewer session into the
            # same window on this or a nearby pass (issue #663).
            state = _wf.defer_reviewer_probe_after(state, next_probe_at)
            state = self._record_event(state, "quota_probe_failed", {})
        _wf.save_state(state_file, state)


def _maybe_reclaim_superseded_main_ci(self) -> None:
    """Cancel superseded, not-yet-started ``main`` CI runs every pass (#863, #815).

    ``main_ci_reclaim.reclaim_superseded_main_ci_runs`` owns the actual
    detection/cancellation logic and its safety invariant (never a
    started run, never main's current tip, only strict ancestors of
    tip) -- see that module's docstring. This method is wiring only.

    Deliberately has NO cadence gate, unlike ``_maybe_reconcile_drift``
    and ``_maybe_probe_quota_recovery`` above: running on every pass is
    the mechanism that closes #815 (the reaper workflow gets only one
    scheduling chance per main push and can permanently lose the race to
    the stale run starting first). Since this lane needs no runner --
    just a local ``git fetch`` plus a couple of cheap ``gh api`` calls --
    running it unconditionally on every pass gives a superseded run
    repeated chances to be reclaimed across passes, which a runner-bound
    workflow retry could never provide regardless of how often it were
    scheduled.

    ``self.config.runners.default_branch`` is reused rather than adding
    a second ``default_branch`` knob on ``MainCiReclaimConfig`` --
    "what is this repo's default branch" is one fact, not one per
    feature, and a second copy would only risk drifting out of sync with
    the first.

    Event policy (issue tracker: "signal without a consumer" is this
    repo's #1 recurring defect class, so this is deliberate rather than
    an oversight): unlike ``_maybe_reconcile_drift``, which emits one
    summary event on every call because its cadence is bounded to once
    per ``interval_minutes``, this method has no cadence gate and the
    overwhelming majority of passes will find zero candidates -- writing
    a ``main_ci_reclaim_completed`` event every single pass would add
    events.db volume with zero diagnostic value and crowd the capped
    200-entry ``state.json`` ring. So a durable event is written only
    when there is something worth a durable record: an actual
    cancellation (``main_ci_reclaim_cancelled``) or a pass-level failure
    (``main_ci_reclaim_failed``). Both are consumed the same way
    ``reconcile_pass_*`` events already are: ``query_events``/
    ``event_counts_by_kind`` for structured aggregation, and
    ``events_by_correlation_id`` to reconstruct a specific pass. The
    zero-candidate case still gets an immediate, cheap consumer via
    ``logger.debug`` below -- proof-of-life without a durable write.

    Wrapped in exception containment for the same reason as every other
    ``_maybe_*`` lane in this method: ``supervise.py``'s
    ``except Exception`` sits outside its ``while True``, so one
    uncaught exception here would kill the whole daemon rather than one
    pass.
    """
    if not self.config.main_ci_reclaim.enabled:
        return

    import logging

    logger = logging.getLogger(__name__)
    state_file = self.paths.state_file
    try:
        result = _wf.reclaim_superseded_main_ci_runs(
            self.gh,
            self.repo_root,
            default_branch=self.config.runners.default_branch,
            workflow_filename=self.config.main_ci_reclaim.workflow_filename,
        )
    except Exception as exc:  # noqa: BLE001 - containment is deliberate; see docstring
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            state = self._record_event(
                state,
                "main_ci_reclaim_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            self.write_gate.save_state(state)
        logger.warning("main_ci_reclaim pass raised an exception", exc_info=True)
        return

    if not result.ok:
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            state = self._record_event(state, "main_ci_reclaim_failed", {"error": result.error})
            self.write_gate.save_state(state)
        logger.warning("main_ci_reclaim pass failed: %s", result.error)
        return

    if not result.cancelled:
        logger.debug(
            "main_ci_reclaim: no superseded main CI runs to reclaim this pass "
            "(tip=%s, checked=%d)",
            result.tip_sha,
            result.candidates_checked,
        )
        return

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        state = self._record_event(
            state,
            "main_ci_reclaim_cancelled",
            {
                "tip_sha": result.tip_sha,
                "cancelled_run_ids": [run.run_id for run in result.cancelled],
                "candidates_checked": result.candidates_checked,
                "skipped_not_ancestor": result.skipped_not_ancestor,
                "skipped_started_before_cancel": result.skipped_started_before_cancel,
                "cancel_errors": list(result.cancel_errors),
            },
        )
        self.write_gate.save_state(state)
    for run in result.cancelled:
        logger.info(
            "main_ci_reclaim: cancelled superseded main CI run %s (sha %s, was %s, created %s)",
            run.run_id,
            run.head_sha,
            run.status_before_cancel,
            run.created_at,
        )


def _maybe_reclaim_worktrees(self, *, now: datetime | None = None) -> dict[str, Any] | None:
    """Cadence-gated merged-PR worktree reclamation on the fleet pass.

    Runs ``clean_worktrees`` -- the same junction-safe, merge-gated,
    liveness-gated sweep behind ``charlie worktree-clean`` -- on a flat
    ``worktree_reclamation.interval_minutes`` schedule. Before this call
    site the sweep only ran when an operator remembered the standalone
    subcommand, so worktrees for merged PRs accumulated indefinitely
    (issue #636: 77 of 81 dead on the host this was measured on).

    Two-phase lock pattern (same shape as ``_maybe_probe_quota_recovery``):
    the schedule is advanced under the lock and the sweep itself runs
    outside it, because ``clean_worktrees`` makes a live ``gh pr view`` call
    per candidate worktree and must not hold ``state_lock`` while it does --
    holding the lock across a per-candidate GitHub fan-out would block every
    other state reader/writer in the process for the sweep's duration.

    ``dry_run`` is threaded honestly: a ``--dry-run`` fleet pass runs the
    sweep in preview mode, which removes nothing (the preview-vs-act class
    tracked in #614-#619). Under live mode a ``worktrees_reclaimed`` event
    is emitted so a maintenance action that left no trace is
    indistinguishable from one that never ran (lesson from #595/#621);
    under ``dry_run`` the event is suppressed per #1324 (WriteGate).
    The event payload carries a bounded ``skipped_examples`` list (each
    entry's own ``reason`` string) alongside the exact ``skipped_count``,
    plus ``worktrees_registered``/``worktrees_out_of_scope`` -- so a
    worktree stuck for days can be diagnosed from events.db alone,
    without catching the sweep live (issue #1012).

    The cadence schedule itself is advanced regardless of ``dry_run``.
    This is deliberate, not an instance of the #614-#619 class:
    ``clean_worktrees`` makes its live ``gh pr view`` call per candidate
    unconditionally -- ``dry_run`` only gates the final ``git worktree
    remove`` -- so the GitHub-quota cost this interval exists to bound is
    identical in preview and live mode. Not advancing the schedule under
    ``dry_run`` would let a repeated preview pass re-run the full
    per-candidate fan-out every time, defeating the cadence gate.

    Returns a small summary dict when the sweep ran (for the loop result's
    ``data``), or ``None`` when reclamation is disabled or not due this
    pass.

    ``now`` (issue #828) is the injectable clock used to compute
    ``next_run_at`` below, BEFORE the ``clean_worktrees`` call's live
    ``gh pr view`` fan-out. Defaults to ``datetime.now(UTC)`` when
    omitted, so production behavior is byte-identical; tests can freeze
    it and assert exact equality on the scheduled ``next_run_at`` instead
    of a wall-clock-tolerance proximity check that would otherwise race
    the fan-out's own duration.
    """
    if not self.config.worktree_reclamation.enabled:
        return None
    resolved_now = now if now is not None else datetime.now(UTC)
    state_file = self.paths.state_file
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if not _wf.is_worktree_reclamation_due(state):
            return None
        # Advance the schedule BEFORE running the sweep so a concurrent
        # pass (or a sweep that takes longer than one poll interval) cannot
        # double-fire. The next run is interval_minutes away regardless of
        # how long the sweep itself takes.
        next_run_at = (
            (resolved_now + timedelta(minutes=self.config.worktree_reclamation.interval_minutes))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        state = _wf.schedule_worktree_reclamation(state, next_run_at)
        _wf.save_state(state_file, state)

    # Use the same resolved worktrees root dispatch and `charlie
    # worktree-clean` use (self._layout.worktrees, from
    # paths.resolved_layout) rather than re-deriving the
    # claude_code.worktrees_dir/runtime.state_dir sentinel logic inline --
    # that duplication across call sites is the exact shape of bug
    # layout.py's module docstring documents as a past production
    # incident (create and sweep sides silently disagreeing on the root).
    state = _wf.load_state_locked(state_file)
    result = _wf.clean_worktrees(
        self.repo_root,
        self._layout.worktrees,
        state,
        self.config,
        self.gh,
        dry_run=self.dry_run,
    )
    orphans = result.data.get("orphans", {})
    skipped_full = result.data.get("skipped", [])
    summary = {
        "dry_run": self.dry_run,
        "ok": result.ok,
        "removed": len(result.data.get("removed", [])),
        "planned": len(result.data.get("planned", [])),
        "skipped_count": len(skipped_full),
        "failed": len(result.data.get("failed", [])),
        "orphans_removed": len(orphans.get("removed", [])),
        "orphans_planned": len(orphans.get("planned", [])),
        "orphans_failed": len(orphans.get("failed", [])),
        "message": result.message,
        # issue #1012: clean_worktrees computes a distinct `reason` per
        # skipped worktree (at least nine distinct strings across the
        # merged/closed-unmerged/liveness gates), but until this fix only
        # `skipped_count` reached this durable payload -- "11 skipped" with
        # no way to tell which reason, or whether a specific stuck
        # worktree was even a candidate. Truncated to
        # `_MAX_SKIPPED_WORKTREE_EXAMPLES` so a standing backlog can't
        # re-emit the full list into events.db every interval;
        # `skipped_count` above is still the exact, untruncated count.
        "skipped_examples": skipped_full[: _wf._MAX_SKIPPED_WORKTREE_EXAMPLES],
        # Distinguishes "never a candidate" (outside worktrees_dir or off
        # the dispatch branch prefix -- an operator-created worktree, for
        # instance) from "considered and skipped": neither was
        # observable from this event before.
        "worktrees_registered": result.data.get("worktrees_registered", 0),
        "worktrees_out_of_scope": result.data.get("worktrees_out_of_scope", 0),
    }
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        state = self._record_event(state, "worktrees_reclaimed", summary)
        _wf.save_state(state_file, state)
    return summary


def _maybe_reconcile_drift(self, *, now: datetime | None = None) -> None:
    """Periodic in-loop repair of GitHub label / state.json divergence.

    merge-lane-recovery plan §6-B / D-8. ``OrchestratorApp.reconcile()``
    (``detect_drift`` + ``apply_fixes``) was previously reachable only
    via the operator-invoked ``charlie mop-up --fix`` CLI command
    (``cli.py``'s ``mop-up`` handler) -- the fleet has never run its own
    repair. This wires it into the loop on a fixed cadence so a
    divergence like a failed escalation label write (the PRIMARY defect:
    ``status`` flips to ``escalated`` but the paired ``human_needed``
    label transition silently fails, leaving a stale ``needs-rework``
    label on a dead issue forever) is corrected automatically instead of
    only when an operator remembers to intervene.

    This method is wiring only -- it does not reimplement drift
    detection or repair. ``self._reconcile_locked(fix=True)`` owns that (and
    already threads ``state_path`` so ``apply_fixes`` emits one
    ``"reconcile"`` event per repaired drift item for free -- see
    ``reconcile.py``). It also owns the safety invariant this
    workstream exists to preserve: an open escalated issue's
    ``status`` is never rewritten (D-2), and reconcile never rewrites
    ``status`` to an active dispatch/rework value. Any status writes
    it does make are to terminal or passive values (``closed``,
    ``merged``, ``open_passive``) or to clear a stale/missing key;
    a closed-while-escalated issue is finalized to ``closed`` like
    any other, but no escalated issue is ever re-entered into the
    machine. This method does not touch ``status``.

    Calls with ``skip_dead_session_sweep=True``: this pass already ran
    the loop's own stall/dead lanes (``_detect_and_handle_stalled_sessions``
    / ``_classify_dead_sessions_and_update_throttle_state``) earlier in
    ``_loop_body``, with grace-period semantics
    (``max_inconclusive_probe_deferrals``) that reconcile.py's own
    dead-session sweep predates and does not implement. Without this,
    reconcile would re-scan the same sessions a few calls later and
    unconditionally reap any not-alive one, silently defeating that
    grace period every time this pass runs. See ``detect_drift``'s
    docstring in ``reconcile.py`` for the full rationale. Launch-stalled
    detection and live-session tracking (the other two things gated on
    ``repo_root`` in ``detect_drift``) are unaffected -- they have no
    counterpart in the loop's own lanes and keep running.

    Calls ``_reconcile_locked`` directly, NOT ``reconcile()`` -- this is
    the whole point of the D-8a split and reverting it silently disables
    this entire method. ``reconcile()`` acquires ``supervisor.lock``
    before doing anything else, and every production caller of ``loop()``
    already holds that exact lock for the full duration of the call:
    ``cli.py``'s ``bash-rats`` handler, ``fleet_dispatch.py``, and
    ``supervise.py`` (which holds it across *every* pass of its
    ``while True``). Byte-range locks taken via ``msvcrt.locking`` with
    ``LK_NBLCK`` are per-handle and non-reentrant *even within a single
    process* -- ``file_lock.py`` keeps no reentrancy bookkeeping -- so a
    second acquisition from inside the same process fails exactly like a
    foreign process's would. Going through ``reconcile()`` here therefore
    always took the ``supervisor_lock_held`` early return and never ran.

    Two-phase lock pattern, mirroring ``_maybe_probe_quota_recovery``:
    ``_reconcile_locked`` acquires ``state_lock`` itself internally to
    run drift detection/repair, and ``state_lock`` wraps a non-reentrant
    advisory file lock (a plain per-path ``threading.Lock``, not an
    ``RLock``) -- calling it while this method already held the same lock
    would deadlock. So:
      1. Under our own (short) lock: decide whether reconcile is due at
         all. If not, return without calling out.
      2. Outside any lock held by this method: call
         ``self._reconcile_locked(fix=True)``. This may still defer (the
         existing GraphQL rate-limit check) -- that path is preserved.
         It can no longer report ``supervisor_lock_held``, because it
         never attempts that acquisition; the lock is already held by
         ``loop()``'s caller, which is the precondition this method
         relies on rather than something it works around.
      3. Under our own (short) lock again: persist the next-due
         timestamp and emit exactly one summary event for this pass,
         shaped by the outcome (completed / deferred / failed) so a
         deferred or failed pass is distinguishable from a silent
         no-op rather than failing silently (D-8, B-AC3).

    The call is wrapped in exception containment, and that containment is
    load-bearing rather than defensive habit: ``supervise.py``'s
    ``except Exception`` sits *outside* its ``while True``, so a single
    uncaught exception from here terminates the whole daemon rather than
    one pass. ``_fetch_prs``/``_fetch_issues`` reach GitHub via
    ``gh.run(..., json_output=True)`` with ``allow_failure=False``, which
    *raises* ``GitHubError``/``GitHubNotFoundError`` once retries are
    exhausted -- so without this, a GitHub outage is a live daemon-kill
    path. A failed pass re-arms the timer like any other outcome, so a
    persistent failure degrades to one logged error per interval instead
    of a hot loop.

    ``now`` (issue #828) is the injectable clock used to compute
    ``next_reconcile_at`` below, BEFORE ``self._reconcile_locked(...)``
    acquires ``state_lock`` and does its own (potentially slow) drift
    detection/repair. Defaults to ``datetime.now(UTC)`` when omitted, so
    production behavior is byte-identical; tests can freeze it and
    assert exact equality on the scheduled ``next_reconcile_at`` instead
    of a wall-clock-tolerance proximity check that would otherwise race
    the reconcile call's own duration.
    """
    if not self.config.reconcile_pass.enabled:
        return

    resolved_now = now if now is not None else datetime.now(UTC)
    state_file = self.paths.state_file
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if not _wf.is_reconcile_due(state):
            return

    next_reconcile_at = (
        (resolved_now + timedelta(minutes=self.config.reconcile_pass.interval_minutes))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # D-8a: _reconcile_locked, never reconcile(). See this method's
    # docstring -- reconcile() would re-acquire the supervisor lock that
    # loop()'s caller already holds and silently no-op every pass.
    try:
        result = self._reconcile_locked(
            fix=True,
            skip_dead_session_sweep=True,
            dry_run=self.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - containment is deliberate; see docstring
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            state = _wf.arm_reconcile_pass(state, next_reconcile_at)
            state = self._record_event(
                state,
                "reconcile_pass_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            self.write_gate.save_state(state)
        return

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        state = _wf.arm_reconcile_pass(state, next_reconcile_at)
        data = result.data
        if data.get("pass_skipped"):
            state = self._record_event(
                state,
                "reconcile_pass_skipped",
                {"reason": data.get("reason")},
            )
        elif data.get("deferred"):
            if not data.get("reconcile_pass_event_recorded"):
                state = self._record_event(
                    state,
                    "reconcile_pass_deferred",
                    {
                        "deferred_reason": data.get("deferred_reason"),
                        "graphql_remaining": data.get("graphql_remaining"),
                        "graphql_reset": data.get("graphql_reset"),
                        "graphql_threshold": data.get("graphql_threshold"),
                    },
                )
        else:
            drift_before = data.get("drift_before", 0)
            drift_after = data.get("drift_after", 0)
            state = self._record_event(
                state,
                "reconcile_pass_completed",
                {
                    "ok": result.ok,
                    "drift_detected": drift_before,
                    "drift_fixed": drift_before - drift_after,
                    "drift_remaining": drift_after,
                },
            )
        self.write_gate.save_state(state)


def _maybe_emit_operator_queue_depth(self) -> None:
    """Emit the ``operator_queue_depth`` gauge event when depth exceeds threshold.

    Issue #1314 item 3. The operator-queue depth gauge makes a silently
    growing queue of mechanical escalations visible in ``events.db``
    rather than only via GitHub label queries. The gauge is checked
    every loop pass (or on the dedicated
    ``operator_queue_review_interval_minutes`` cadence if configured),
    and the ``operator_queue_depth`` warning event is emitted only when
    the depth exceeds the configured ``operator_queue_depth_threshold``
    -- the same "checked every pass, emitted only when the condition
    holds" pattern ``dispatch_stale`` uses.

    The consumer is ``heartbeat_check.py``'s ``check_warning_events``,
    which buckets the kind (registered in
    ``EXPECTED_OPERATIONAL_KINDS``) into a summarized count so a
    chronically deep queue does not drown out genuinely rare warnings.
    That bucketing wiring lands in the same PR as the signal, per the
    signal-without-a-consumer rule.

    Threshold 0 disables the alert entirely (no event emitted regardless
    of depth), preserving the pre-feature silent-queue behavior for
    fleets that have not yet opted in.

    ``dry_run`` short-circuits the entire gauge before any lock or state
    read: the gauge emits a warning event and persists the
    ``next_operator_queue_review_at`` arm timestamp via a raw
    ``save_state`` (outside WriteGate), so running it under
    ``dry_run=True`` would both leak an ``operator_queue_depth`` row into
    ``events.db`` and mutate ``state.json`` -- violating the C1.2
    "byte-identical to a pass that never ran" dry-run invariant
    (``test_loop_wrapper_telemetry_is_the_only_delta_under_dry_run_true``
    pins that invariant; its fixture has zero escalated issues, so it
    only exercises the depth<=threshold early return and would not catch
    a deep-queue dry-run leak on its own). The guard is at the top rather
    than after the threshold check so a dry-run pass pays neither the
    lock acquisition nor the state load.
    """
    if self.dry_run:
        return
    threshold = self.config.deescalation.operator_queue_depth_threshold
    if threshold <= 0:
        return

    state_file = self.paths.state_file
    review_interval = self.config.deescalation.operator_queue_review_interval_minutes
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if review_interval > 0 and not _wf.is_operator_queue_review_due(state):
            return
        depth_set = _wf.operator_queue_depth(state)
        depth = len(depth_set)
        if depth <= threshold:
            return
        next_review_at = (
            (datetime.now(UTC) + timedelta(minutes=review_interval))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        state = _wf.arm_operator_queue_review(state, next_review_at)
        state = self._record_event(
            state,
            "operator_queue_depth",
            {
                "depth": depth,
                "threshold": threshold,
                "issue_numbers": sorted(depth_set),
            },
        )
        _wf.save_state(state_file, state)


def _is_dispatchable(
    self,
    issue: dict[str, Any],
    operator_claimed: set[int] | None = None,
) -> bool:
    # Closed issues are never dispatchable, regardless of labels.
    if str(issue.get("state") or "OPEN").upper() != "OPEN":
        return False
    names = _wf.label_names(issue)
    if self.config.labels.ready not in names:
        return False
    if names & self.config.labels.terminal:
        return False
    if names & self.config.labels.active:
        return False
    if operator_claimed is None:
        state = _wf.load_state_locked(self.paths.state_file)
        operator_claimed = _wf.operator_claimed_issues(state)
    return int(issue["number"]) not in operator_claimed


def _mark_foreign_issue_ref(self, pr_number: int, issue_number: int, reason: str) -> bool:
    """Record a not-found for this (pr, issue) pair, parking durably once
    confirmations reach the configured threshold.

    A PR opened against the wrong fleet repo (e.g. its branch references
    another repo's issue number) can never be processed here: every pass
    would re-derive the same issue number and re-fail the same GitHub
    lookup forever. Persist a ``foreign_issue_ref`` marker in the PR's
    state entry so subsequent passes skip it with zero GitHub calls.

    Issue #1132: a single not-found is no longer enough to park durably —
    a transient GraphQL repo-resolution failure (network/infra dip) can
    produce the same ``GitHubNotFoundError`` shape. The marker now carries
    a ``confirmations`` counter; the PR is only *confirmed* (skipped on
    subsequent passes) once the counter reaches
    ``review.foreign_issue_ref_confirm_passes`` (default 2). A transient
    window (minutes) clears before two 5-minute passes complete.

    Legacy markers (pre-#1132, no ``confirmations`` field) are treated as
    already-confirmed by the early-skip in ``_loop_body`` (``confirmed =
    confirmations is None or ...``), so they never reach this method via
    the loop. The ``same_issue and prev_conf is None`` branch below is a
    defensive guard kept for direct/non-loop call sites: without it a
    legacy marker would hit ``prev_conf + 1`` (``None + 1`` -> TypeError)
    and could re-emit the one-shot digest. It does NOT silently migrate
    the field on existing production markers — that handling lives in the
    early-skip, which leaves the marker untouched (no ``confirmations``
    key is written). See
    ``test_legacy_marker_without_confirmations_skipped_silently``.

    Returns True only when the confirmation threshold is first reached for
    this (pr, issue) pair, so the caller emits the one-shot attention
    digest exactly once.
    """
    confirm_passes = self.config.review.foreign_issue_ref_confirm_passes
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        marker = pr_state.get("foreign_issue_ref") or {}
        same_issue = marker.get("issue") == issue_number
        prev_conf = marker.get("confirmations")
        if same_issue and prev_conf is None:
            # Defensive guard for a legacy marker (pre-#1132, no
            # ``confirmations`` field) reaching this method via a
            # non-loop call site. Unreachable via ``_loop_body``: the
            # early-skip there treats ``confirmations is None`` as
            # confirmed and skips before this method is called, leaving
            # the marker untouched (no migration). Kept to avoid a
            # ``None + 1`` TypeError on the ``elif same_issue`` branch
            # and to suppress a spurious digest re-emit.
            new_conf = confirm_passes
            emit = False
        elif same_issue:
            new_conf = prev_conf + 1
            emit = new_conf >= confirm_passes and prev_conf < confirm_passes
        else:
            # Fresh marker (new issue ref or first sighting).
            new_conf = 1
            emit = new_conf >= confirm_passes
        detected_at = marker.get("detected_at") or _wf.utc_now()
        state["prs"][str(pr_number)] = {
            **pr_state,
            "number": pr_number,
            "foreign_issue_ref": {
                "issue": issue_number,
                "detected_at": detected_at,
                "reason": reason,
                "confirmations": new_conf,
            },
        }
        _wf.save_state(self.paths.state_file, state)
    return emit
