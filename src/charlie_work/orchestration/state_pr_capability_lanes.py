"""Per-pass ``loop()`` lanes that only make sense on a PR-publishing backend.

Track 2 Phase B delegate leaf: the two method bodies below were moved
verbatim out of ``orchestration/state_maintenance.py`` (which was over its
file-size ratchet mark) and are re-attached onto ``OrchestratorApp`` by
``workflow_delegation._install_delegates`` -- ``self`` binds through the
descriptor protocol exactly as it did there, and every ``_wf.``
reach-through keeps the Tier-D monkeypatch seam (the suite patches
``reclaim_superseded_main_ci_runs`` and the ``state_lock``/``load_state``
primitives on the ``charlie_work.workflow`` module object).

Issue #1810: both lanes assume a GitHub/PR-capable backend. On a repo whose
backend cannot publish pull requests (``local_issues.LocalFileGitHub`` --
no remote at all) each was a guaranteed failure event every pass:
``_maybe_reclaim_superseded_main_ci`` starts with ``git fetch origin``, and
``_maybe_reconcile_drift`` -> ``detect_drift`` -> ``_fetch_prs`` calls
``gh.run(...)`` which raises ``GitHubError``. Both are therefore gated on
``local_work_park.publishes_pull_requests`` -- the same capability
predicate ``dead_worker_reap`` consults -- so a non-publishing backend
skips the lane outright rather than recording ``main_ci_reclaim_failed`` /
``reconcile_pass_failed`` every pass with zero repair value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import charlie_work.workflow as _wf
from charlie_work.local_work_park import publishes_pull_requests


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
    # Issue #1810: the reclaim starts with ``git fetch origin`` -- a repo
    # whose backend cannot publish pull requests has no remote at all, so
    # the call is a guaranteed main_ci_reclaim_failed every pass. Skip it
    # on the same capability predicate dead_worker_reap already gates on,
    # rather than catching the fetch failure per call site.
    if not publishes_pull_requests(self.gh):
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

    if result.fetch_attempts > 1:
        # One event per retried call, never per attempt (git_retry.py's own
        # on_retry contract) -- so this only fires on the rare pass where the
        # leading `git fetch` actually hit a transient network blip.
        # Recorded regardless of the pass's eventual ok/failure outcome
        # below: the retry is a fact about the fetch, independent of what a
        # later step in the same pass does.
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            state = self._record_event(
                state,
                "git_network_retry",
                {
                    "site": "main_ci_reclaim",
                    "command": "git fetch",
                    "attempts": result.fetch_attempts,
                    "ok": result.ok,
                },
            )
            self.write_gate.save_state(state)

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
    # Issue #1810: ``_reconcile_locked`` -> ``detect_drift`` -> ``_fetch_prs``
    # calls ``gh.run(...)``, which raises ``GitHubError`` on a backend that
    # cannot publish pull requests -- a guaranteed reconcile_pass_failed
    # every due pass with zero repair value. Skip the whole call on the
    # same capability predicate dead_worker_reap already gates on.
    if not publishes_pull_requests(self.gh):
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
