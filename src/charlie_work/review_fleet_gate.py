"""Fleet-wide reviewer-concurrency cap -- the review-dispatch twin of the worker cap (#2084).

``fleet.global_max_concurrent_reviews`` caps live REVIEW sessions across every
registered repo. Until it existed, reviewers were bounded only per repo
(``review_dispatch.max_concurrent_reviews``), so the fleet-wide reviewer count
was the SUM of those caps -- which a provider rate limit on the reviewer
harness (#2038) does not respect. This module mirrors, piece for piece, how
``fleet.global_max_concurrent_sessions`` bounds workers:

=====================================  =====================================
worker lane                            review dispatch (this module)
=====================================  =====================================
``count_fleet_live_sessions``          ``fleet_registry.count_fleet_live_reviews``
``_apply_concurrency_governor``        :func:`read_fleet_review_cap` +
(``fleet_max`` / ``fleet_available``)  ``_select_review_dispatch_candidates``
``acquire_fleet_launch_lock`` +        :func:`acquire_fleet_review_launch_lock` +
``issue_worker_launch_permit`` lock    :func:`fleet_review_lock_deferral`
step (``fleet_lock_held``)
``records_deferral`` (``dispatch_      :func:`record_review_lane_result`
deferred`` / ``dispatch_starved``)
``fleet_concurrency_limit`` /          :meth:`FleetReviewCap.report_fields`
``fleet_live_session_count``
=====================================  =====================================

Like the worker cap, the knob is read from the repo's layered config on every
pass (``fleet_dispatch`` loads it per repo per pass), so flipping it needs no
supervisor restart, and ``0`` disables it entirely: no lock is taken and no
fleet count is read.

The fleet lock is the SAME file the worker lanes use. Each lane opts in only
when ITS cap is enabled, and the lock is held across count -> claim -> launch
so two repos cannot both read a stale fleet reviewer count and over-dispatch
the shared cap. Both reviewer launch sites (``dispatch_reviews`` and the
no-remote ``_local_dispatch_reviewers``) go through it; the AST guard in
``tests/test_review_fleet_gate_guard.py`` fails if a new one bypasses it.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from charlie_work.dispatch_deferral import deferral_reason, record_lane_result
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.worker_launch_gate import (
    REASON_FLEET_LOCK_HELD,
    FleetLaunchLock,
    FleetLockAcquirer,
    acquire_fleet_launch_lock,
    fleet_lock_held_extra,
)

if TYPE_CHECKING:
    from charlie_work.workflow import OrchestratorApp

logger = logging.getLogger(__name__)

# Lane name for ``dispatch_deferred`` / ``dispatch_starved`` events and the
# per-lane consecutive-deferral streak; matches the ``dispatch_reviews`` key
# the loop already files this lane's result under.
REVIEW_DISPATCH_LANE = "dispatch_reviews"

# ``clamped_by`` value when the fleet cap, not the per-repo caps, bound the
# launch count -- the same spelling the worker governor uses.
CLAMPED_BY_FLEET_MAX = "fleet_max"


@dataclass(frozen=True)
class FleetReviewCap:
    """One read of the fleet reviewer budget: the cap and the live count under it."""

    fleet_max: int
    fleet_live_count: int

    @property
    def available(self) -> int:
        return max(0, self.fleet_max - self.fleet_live_count)

    def report_fields(self, *, clamped: bool = False) -> dict[str, Any]:
        """Governor fields for the review-dispatch result and event payloads.

        Deliberately NOT the worker lane's ``fleet_concurrency_limit`` /
        ``fleet_live_session_count`` spellings: ``reap_loop`` lifts those keys
        into the loop payload and ``heartbeat_check`` reads them as the WORKER
        fleet's saturation, so reusing them would make a full reviewer lane
        read as a saturated worker fleet.
        """
        fields: dict[str, Any] = {
            "fleet_review_concurrency_limit": self.fleet_max,
            "fleet_live_review_count": self.fleet_live_count,
            "fleet_available_review_slots": self.available,
        }
        if clamped:
            fields["clamped_by"] = CLAMPED_BY_FLEET_MAX
        return fields


def _workflow() -> Any:
    # Reached through ``charlie_work.workflow`` (resolved at call time -- it
    # imports the orchestration delegates that import this module) so a test
    # patching ``charlie_work.workflow.count_fleet_live_reviews`` intercepts,
    # exactly as ``_wf.count_fleet_live_sessions`` does for the worker cap.
    import charlie_work.workflow as wf

    return wf


def acquire_fleet_review_launch_lock(
    app: OrchestratorApp, *, acquire: FleetLockAcquirer | None = None
) -> FleetLaunchLock:
    """Mint the review lane's fleet-launch-lock handle -- the OS lock is NOT taken here.

    A no-op handle when ``fleet.global_max_concurrent_reviews`` is 0 (no
    cross-repo accounting to serialize). Use as a context manager so every
    exit path releases; the OS lock is realized later by
    :func:`fleet_review_lock_deferral`, after the lane's lock-free scans.
    """
    return acquire_fleet_launch_lock(
        app,
        acquire=acquire if acquire is not None else try_acquire_fleet_lock,
        cap=app.config.fleet.global_max_concurrent_reviews,
    )


def fleet_review_lock_deferral(
    app: OrchestratorApp, launch_lock: FleetLaunchLock | None
) -> dict[str, Any] | None:
    """Realize the fleet lock (bounded jittered wait); ``None`` on success.

    On exhaustion returns the ``deferred_reason: fleet_lock_held`` payload
    fields (plus lock-holder diagnostics) for the lane to return as an
    ok-deferral, which ``dispatch_deferral`` then records.
    """
    wait_seconds = app.config.fleet.launch_lock_wait_seconds
    if launch_lock is None or launch_lock.ensure_acquired(wait_seconds) is not None:
        return None
    return {
        "deferred_reason": REASON_FLEET_LOCK_HELD,
        **fleet_lock_held_extra(app, wait_seconds),
    }


def read_fleet_review_cap(app: OrchestratorApp) -> FleetReviewCap | None:
    """Read the fleet reviewer budget now; ``None`` when the cap is disabled.

    Callers that launch must hold the fleet lock (see
    :func:`fleet_review_lock_deferral`) so the returned live count cannot go
    stale before their launches land; a dry-run preview reads it lock-free.
    """
    fleet_max = app.config.fleet.global_max_concurrent_reviews
    if fleet_max <= 0:
        return None
    live_count, _skipped_repos = _workflow().count_fleet_live_reviews(app.fleet_dir_override)
    return FleetReviewCap(fleet_max=fleet_max, fleet_live_count=live_count)


def record_review_lane_result(app: OrchestratorApp, result: Any) -> None:
    """Emit ``dispatch_deferred`` / ``dispatch_starved`` for a review-lane ``fleet_lock_held``.

    Only the fleet-lock deferral is this module's: ``dispatch_reviews`` has
    other ok-deferrals (``reviewer_quota_probe_backoff``) that fire every pass
    for hours and were never starvation signals, so they neither record nor
    touch the streak. A completed pass (no deferral) resets the streak, same
    as the worker lanes -- but only while the cap is enabled, so a repo that
    never opted in pays nothing.
    """
    reason = deferral_reason(result)
    if reason is None:
        if app.config.fleet.global_max_concurrent_reviews > 0:
            record_lane_result(app, REVIEW_DISPATCH_LANE, result)
    elif reason == REASON_FLEET_LOCK_HELD:
        record_lane_result(app, REVIEW_DISPATCH_LANE, result)


def fleet_review_lock(
    to_result: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a review-dispatch entry point (``self`` first) with the fleet lock handle.

    Mints the pass's pending handle (:func:`acquire_fleet_review_launch_lock`),
    passes it as ``launch_lock=``, releases it on every exit path -- including
    a raise -- and records a ``fleet_lock_held`` deferral afterwards. A
    decorator rather than a wrapper method so ``OrchestratorApp``'s member
    surface is unchanged. ``to_result`` adapts an entry point that returns a
    plain dict (the local path) into the ``CommandResult`` shape
    :func:`record_review_lane_result` reads.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with acquire_fleet_review_launch_lock(self) as launch_lock:
                result = fn(self, *args, launch_lock=launch_lock, **kwargs)
            record_review_lane_result(self, to_result(result) if to_result else result)
            return result

        return wrapper

    return decorate
