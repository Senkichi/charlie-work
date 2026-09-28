"""Cooperative in-pass deadline machinery (issue #1948).

``fleet_loop``'s per-pass wall-clock budget (``max_pass_runtime_seconds``,
issue #1832) used to be consulted only *between* repo lanes, so one lane
could overrun the deadline by tens of minutes of sequential gh timeouts
and retry backoffs. This module holds the whole mechanism that now bounds
a lane mid-flight:

* :class:`PassDeadlineExceeded` -- the refusal ``GitHub.run()`` raises once
  the armed predicate reports the budget spent, deliberately a distinct
  type rather than a ``GitHubRunResult`` failure value or a plain
  ``GitHubError`` (see the class docstring).
* :func:`set_pass_deadline_exceeded` -- per-lane arming hook for a
  ``GitHub`` client; called by ``fleet_lanes._run_fleet_repo_lane``.
* :func:`raise_if_pass_deadline_spent` -- the single refusal path
  ``GitHub.run()`` calls at each yield point.
* :func:`pass_deadline_spent` -- duck-typed probe for sites that must not
  burn attempt budget on a spent pass (merge_ready's alarm counter).
* :class:`PassDeadline` -- the latched check-and-trip tracker
  ``reap_loop._loop_body`` drives its sub-phase yield points with.
* :func:`pass_deadline_suspended` -- disarm a lane's hook around a
  mutating sequence that is already past its irreversible step (a
  refusal there could only strand half-finished cleanup -- or skip the
  pass's merge bookkeeping entirely).
* :func:`run_deadline_guarded_maintenance` -- the per-pass maintenance
  lane, extracted from ``_loop_body`` for the module-size ratchet.

The wiring policy (moved here from ``_loop_body`` so this module owns the
whole contract): every GitHub-touching sub-phase goes through
``deadline.phase`` / ``deadline.call`` -- a boundary check returning a
deferred placeholder/fallback, plus a catch for a refusal raised
mid-phase. The per-PR review/merge scan breaks at the top of each
iteration. Purely local work (orphan-process sweeps, stall scans, digest
emission, marker writes) stays UNguarded -- a deadline must never strand
bookkeeping the pass already committed to. Once observed True the latch
stays set so later checks are free, but the pass-end marker re-queries
the predicate rather than trusting the latch, so a deadline that trips
inside the FINAL guarded step still marks the pass
``data["deadline_deferred"]``. A ``None`` predicate (every non-fleet
caller) disables every check.

The exception to "guard every gh call" is a sequence already past its
irreversible step: once ``merge_pr`` has merged, refusing the trailing
label/close/branch cleanup -- or the open-PR update / superseded-run
cancel tail -- does not un-merge anything -- it only strands
reconcile-visible bookkeeping (and, past the tail, silently drops the
merge's own events and ``merges[]`` entry). ``pass_deadline_suspended``
is the scoped valve for exactly that case; callers keep it tight because
the suspend window is also a refusal-free zone for anything called
inside it.

This module imports nothing from ``charlie_work`` at module level:
``github.py`` imports it, so an import back would be circular.
``run_deadline_guarded_maintenance`` binds ``charlie_work.workflow``
lazily inside the function body -- by call time the import is long
settled, and the ``_wf.<name>`` seam keeps the same test-patch targets
the inlined code had.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class PassDeadlineExceeded(BaseException):
    """The armed in-pass deadline refused a ``gh`` call (issue #1948).

    Deliberately a ``BaseException`` -- the ``asyncio.CancelledError``
    cancellation precedent: a refusal is cooperative cancellation, and
    cooperative cancellation must be invisible to ``except Exception``.
    The lane's call graph contains broad ``except Exception`` sites whose
    *fallbacks* are the dangerous part, not the swallow itself:

    * ``workflow.merge_ready``'s comment-post handler swallows and falls
      through to the ``consecutive_failed_merge_attempts`` increment.
    * ``workflow``'s dead-worker rework lane swallows ``gh.pr_view`` and
      routes on the *stale* ``pr_data`` snapshot -- a post-deadline
      mutation decided on pre-deadline data.
    * ``github.build_branch_issue_validator`` swallows ``issue_list`` and
      returns ``None`` -- fail-OPEN, so a post-deadline pass would trust
      stale branch-name bindings it would otherwise reject.
    * ``state_rework_routing``'s stranded-marker restorer swallows
      ``issue_view`` and defers -- benign alone, but indistinguishable.

    As a ``BaseException`` no ``except Exception`` / ``except GitHubError``
    site can intercept it; it lands only on the explicit
    ``except PassDeadlineExceeded`` boundary catches (the per-PR scan,
    ``_loop_impl``, ``_run_fleet_repo_lane``) or a genuinely broad
    ``except BaseException`` forwarding site (``fleet_status`` delivers it
    through the future -- still correct). ``finally`` cleanup still runs:
    lane locks release normally.

    It is also deliberately NOT a ``GitHubError`` subclass: the codebase's
    ~37 ``except GitHubError`` handlers translate caught errors into
    failure *values* -- ``errors[]`` entries, ``is_open=False`` in the
    ``are_issues_open`` per-issue fallback, ``return False`` in
    ``close_issue``/``pr_update_branch`` (which merge_ready reads as
    ``sync_failed`` -> ``consecutive_failed_merge_attempts`` increment),
    the unauthorized-merge tripwire's "transient gh failure" skip record.
    A refusal subclassing GitHubError would be swallowed by every one of
    those sites and re-enter exactly the failure vocabulary the deadline
    exists to bypass -- the review finding this type answers.

    And it is never a ``GitHubRunResult``: ``run()`` raises it under
    ``allow_failure=True`` too, because every existing allow_failure
    consumer reads a refusal-shaped result as a real gh failure
    (``pr_checks`` -> ``None`` -> unavailable -> merge-attempt counter,
    ``_run_bool`` -> ``False`` -> handoff-failed counter,
    ``issue_dependencies`` -> fail-open ``{}`` corrupting the dependency
    gate). A refusal is control flow -- "stop now, the budget is spent" --
    not a transport failure, so it is never fed to the circuit breaker
    either.
    """


def pass_deadline_refusal(command: list[str]) -> PassDeadlineExceeded:
    """Build the refusal ``GitHub.run()`` raises once the budget is spent.

    Single source for the refusal message text -- the lane-level tests
    match on "in-pass deadline" and downstream diagnostics join the
    refused argv back into it.
    """
    return PassDeadlineExceeded(f"in-pass deadline exceeded; gh call refused: {' '.join(command)}")


def raise_if_pass_deadline_spent(exceeded: Callable[[], bool] | None, command: list[str]) -> None:
    """Raise the deadline refusal when the armed *exceeded* predicate trips.

    The single refusal path for ``GitHub.run()`` -- every yield point
    (pre-attempt, pre-backoff on both retry branches) calls this so the
    armed/spent/raise ordering lives in exactly one place.
    """
    if exceeded is not None and exceeded():
        raise pass_deadline_refusal(command)


def set_pass_deadline_exceeded(gh: Any, check: Callable[[], bool] | None) -> None:
    """Arm (or clear, with None) the cooperative in-pass deadline hook on *gh*.

    ``fleet_loop``'s per-pass budget was previously only consulted between
    repo lanes, so one lane could overrun the deadline by tens of minutes
    of sequential gh timeouts and retry backoffs. Arming this hook on the
    lane's own client lets ``run()`` refuse a call -- or abort a retry
    chain -- the moment the budget is spent.

    A module-level function rather than a ``GitHub`` member: the class's
    lexical surface is frozen at ``__post_init__``/``run`` by the
    githublike-protocol tests -- capability access goes through delegates,
    and this hook is per-lane runtime state, not a capability. Written
    through ``object.__setattr__`` because ``GitHub`` is a frozen
    dataclass; fleet lanes build a fresh client per repo per pass, so an
    armed hook cannot leak into a later pass. Non-fleet callers never arm
    it, keeping the check a no-op for single-repo passes.
    """
    object.__setattr__(gh, "_pass_deadline_exceeded", check)


def pass_deadline_spent(gh: Any) -> bool:
    """True when *gh* carries an armed deadline hook that currently reports spent.

    Duck-typed ``getattr`` rather than ``isinstance``: this module must
    not import ``charlie_work.github`` (it is imported BY ``github.py``),
    and ``GitHubLike`` test doubles simply lack the attribute -> False.
    """
    check = getattr(gh, "_pass_deadline_exceeded", None)
    return bool(check is not None and check())


@contextmanager
def pass_deadline_suspended(gh: Any) -> Iterator[None]:
    """Disarm *gh*'s deadline hook for the scope, then re-arm it.

    For a mutating sequence already past its irreversible step --
    ``merge_ready``'s whole post-``merge_pr`` region: the finalize trio
    (issue-label transition, issue close, head-branch delete) AND the
    deferral tail (``_update_open_agent_prs``,
    ``cancel_superseded_runs``). Refusing mid-sequence cannot undo the
    merge, so the refusal's only effect would be stranding half-finished
    cleanup for reconcile to find later -- or, in the tail's case,
    propagating out of ``merge_ready`` before the ``merge_ready`` /
    ``merge_succeeded`` events and the ``merges[]`` result are produced
    (a permanent loss: the durable merged status makes the next pass
    short-circuit without re-emitting them). Running the calls is both
    correct and faster than the reconcile fallback. The hook is re-armed
    on exit so the code after the block still sees a spent budget, and
    ``pass_deadline_spent(gh)`` keeps reporting True for the
    merge-counter guard.

    A no-op on an unarmed client (``GitHubLike`` doubles, non-fleet
    passes): nothing was armed, nothing to re-arm.
    """
    check = getattr(gh, "_pass_deadline_exceeded", None)
    if check is None:
        yield
        return
    object.__setattr__(gh, "_pass_deadline_exceeded", None)
    try:
        yield
    finally:
        object.__setattr__(gh, "_pass_deadline_exceeded", check)


class PassDeadline:
    """Latched check-and-trip tracker for one reap-loop pass body (issue #1948).

    Bundles what ``_loop_body`` used to spell out inline -- the
    ``deadline_hit`` latch, the ``_deadline_now()`` closure, and the
    ``_phase_skipped()`` placeholder factory -- behind a small interface:

    * :meth:`now` -- fresh check-and-latch: queries the predicate until it
      first returns True, then stays True. Called at every sub-phase
      boundary AND once at end-of-pass, so a deadline tripped inside the
      final guarded operation still marks the pass ``deadline_deferred``
      (the stale-latch bug the earlier shape had).
    * :meth:`trip` -- latch the deadline spent when a refusal escaped a
      callee as ``PassDeadlineExceeded`` instead of being observed via
      the predicate.
    * :meth:`skipped` -- the ``deadline_deferred`` placeholder
      CommandResult a skipped sub-phase contributes to the pass data.
    * :meth:`phase` -- guarded CommandResult-producing sub-phase: skip
      when spent, run otherwise, and convert a mid-phase refusal into the
      same deferred placeholder so already-collected pass data survives.
    * :meth:`call` -- the same guard for non-CommandResult steps (list
      fetches, void maintenance calls), returning a caller-chosen
      fallback.
    * :meth:`preflight` -- the pass-start deferred shortcut: a lane that
      arrives on an already-spent budget returns a complete-but-empty
      result instead of starting the first sub-phase on a dead budget.
    * :meth:`finalize_pass` -- the pass-end deferred check: a fresh
      predicate evaluation (not the earlier latch read) that marks the
      pass's ``data`` and returns the ``loop_pass_deadline_deferred``
      event payload to emit, or ``None``.

    ``result_type`` is injected (``CommandResult``'s ``(ok, message,
    data)`` shape) because this module cannot import ``workflow`` --
    ``workflow`` reaches this module through ``github`` -> the import
    direction must stay one-way.
    """

    def __init__(
        self,
        exceeded: Callable[[], bool] | None,
        result_type: Callable[[bool, str, dict[str, Any]], Any],
    ) -> None:
        self._exceeded = exceeded
        self._result_type = result_type
        self._hit = False
        # Seed the latch once at construction so "already spent at pass
        # start" is indistinguishable downstream from "tripped later".
        self.now()

    @property
    def hit(self) -> bool:
        """Latched state -- does NOT re-query the predicate (use now())."""
        return self._hit

    def now(self) -> bool:
        """Check-and-latch: query the predicate until it first trips."""
        if not self._hit and self._exceeded is not None:
            self._hit = bool(self._exceeded())
        return self._hit

    def trip(self) -> None:
        """Latch the deadline spent (a refusal escaped a callee as an
        exception rather than being observed via the predicate)."""
        self._hit = True

    def skipped(self, name: str) -> Any:
        """The deferred placeholder CommandResult a skipped phase reports."""
        return self._result_type(
            True,
            f"{name} skipped: in-pass deadline reached",
            {"deadline_deferred": True},
        )

    def phase(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run a CommandResult-producing sub-phase, or its placeholder.

        A mid-phase ``PassDeadlineExceeded`` (a nested gh call refused
        after this phase started on a live budget) is caught, latched,
        and recorded with the same deferred placeholder -- the pass keeps
        whatever partial results the earlier phases produced.
        """
        if self.now():
            return self.skipped(name)
        try:
            return fn()
        except PassDeadlineExceeded:
            self.trip()
            return self.skipped(name)

    def call(self, fn: Callable[[], Any], fallback: Any) -> Any:
        """Guarded non-CommandResult step: skip-or-run, refusal -> fallback."""
        if self.now():
            return fallback
        try:
            return fn()
        except PassDeadlineExceeded:
            self.trip()
            return fallback

    def preflight(self, message: str) -> Any | None:
        """Deferred result for a pass arriving on an already-spent budget.

        The lane can be submitted just as the budget runs out (the
        pool-slot wait or label-ensure can consume the remainder after
        fleet_loop's between-lane re-check) -- return a complete-but-empty
        pass instead of starting the first sub-phase on a dead budget.
        ``None`` means the budget is live; proceed normally.
        """
        if self.now():
            return self._result_type(True, message, {"deadline_deferred": True})
        return None

    def finalize_pass(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """End-of-pass deferred check; return the event payload or None.

        A FRESH predicate evaluation via :meth:`now`, not the earlier
        latch read: a deadline tripped inside the final guarded step (the
        last PR's review/merge work, or worktree reclamation) must still
        mark the pass partial -- otherwise fleet_loop counts an
        uninspected repo as observed. When spent, ``deadline_deferred``
        is set on *data* and the ``loop_pass_deadline_deferred`` event
        payload is returned for the caller to emit; the caller owns the
        write so the emit site stays next to its sibling ``log_event``
        calls under the same write-gate exemption.
        """
        if not self.now():
            return None
        data["deadline_deferred"] = True
        return {
            "reviews_completed": len(data.get("reviews", [])),
            "merges_completed": len(data.get("merges", [])),
            "open_tracked_prs": data.get("open_tracked_prs", 0),
        }


def run_deadline_guarded_maintenance(
    deadline: PassDeadline,
    app: Any,
    *,
    now: Any = None,
) -> list[dict[str, Any]]:
    """The per-pass maintenance lane, guarded step by step (issue #1948).

    Extracted from ``reap_loop._loop_body`` to hold that module under its
    file-size ratchet mark -- same extraction precedent as
    ``escalation._stale_template_warning_suppressed`` (#1894). Every
    GitHub-touching step goes through ``deadline.call`` so a spent budget
    skips-or-falls-back instead of starting work; the orphan *process*
    sweep is pure local bookkeeping and stays unconditional. Returns the
    dead-session classification's reaped-entry list -- ``_loop_body``
    feeds it into the notify digest's health transitions.

    ``app`` is the ``OrchestratorApp`` instance (``_loop_body``'s
    ``self``); ``charlie_work.workflow`` is bound lazily inside the body
    because a module-level import would cycle github -> pass_deadline ->
    workflow -> github.
    """
    import charlie_work.workflow as _wf

    sessions_dir = app._layout.sessions_dir
    # Classify dead sessions and update throttle state (production loop path)
    # This detects provider throttling from worker deaths and sets cooldown
    # Also reconciles labels for dead sessions with no open PR (issue #118)
    # Issue #343 Finding 2: the stall lane at the top of _loop_body already
    # ran this pass and is the sole writer of the inconclusive-probe
    # deferral counter for a not-alive worker -- this lane is told not to
    # persist it again on top of that write.
    reaped = deadline.call(
        lambda: _wf._classify_dead_sessions_and_update_throttle_state(
            sessions_dir,
            app.paths.state_file,
            app.gh,
            app.config,
            write_gate=app.write_gate,
            persist_inconclusive_probe_counter=False,
            now=now,
            fleet_dir_override=app.fleet_dir_override,
        ),
        [],
    )

    # Flat-interval Haiku probe for early quota/rate-limit recovery (see
    # docstring): only does real work when a throttle indicator is active.
    # `now` (issue #828) is this pass's single injected clock, forwarded
    # so the probe's own cadence-scheduling samples stay consistent with
    # the rest of this pass instead of independently racing wall clock.
    deadline.call(lambda: app._maybe_probe_quota_recovery(now=now), None)

    # Periodic in-loop reconcile (merge-lane-recovery §6-B): repairs
    # GitHub label / state.json divergence on a fixed cadence instead of
    # only when an operator runs `charlie mop-up --fix`. Placed before
    # the dispatch calls below so labels it repairs (e.g. a stale
    # `needs-rework` on an issue state already marked `escalated`) are
    # visible to this same pass's dispatch decisions, not just the next.
    deadline.call(lambda: app._maybe_reconcile_drift(now=now), None)

    # Issues #863/#815: reclaim superseded, not-yet-started main CI runs
    # every pass -- no runner needed, so unlike the workflow-based
    # reaper this cannot lose the race for the capacity it exists to
    # free. See _maybe_reclaim_superseded_main_ci's docstring.
    deadline.call(lambda: app._maybe_reclaim_superseded_main_ci(), None)

    # Issue #783: periodic re-evaluation of `mechanical` escalations --
    # the only automated re-entry from `agent:human-needed` for pure
    # process failures (a dead rework worker, a redispatch cap, a
    # stalled janitor-gate rework, ...) whose underlying PR artifact has
    # since become mergeable and janitor-clean. `judgment` escalations
    # and any pre-existing escalation with no recorded reason_class are
    # untouched by construction (see _maybe_deescalate_mechanical).
    deadline.call(lambda: app._maybe_deescalate_mechanical(), None)

    # Sweep for orphan processes in dead session worktrees (issue #139)
    # This catches detached/daemonized processes that survived session kills
    _wf._sweep_orphan_processes_for_dead_sessions(
        sessions_dir, app.paths.state_file, app.config, write_gate=app.write_gate
    )

    # Detect and handle orphaned workers using state.json PID records (issue #207)
    # This fallback detects dead workers even when session sidecar files are orphaned.
    # Pass the review callback so a head-advanced request_changes finding can be
    # routed to the review-pending path instead of being re-emitted as drift.
    deadline.call(
        lambda: _wf._detect_and_handle_orphaned_workers(
            sessions_dir,
            app.paths.state_file,
            app.config,
            app.gh,
            write_gate=app.write_gate,
            review_callback=app.review,
            fleet_dir_override=app.fleet_dir_override,
        ),
        None,
    )
    return reaped
