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

This module imports nothing from ``charlie_work``: ``github.py`` imports
it, so an import back would be circular.
"""

from __future__ import annotations

from collections.abc import Callable
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
