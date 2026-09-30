"""The single enforcement point for fleet-wide worker-launch limits (issue #2041).

Every worker lane (fresh ``dispatch``, remote ``dispatch_rework`` normal and
rescue tiers, and the no-remote ``_local_dispatch_rework``) launches workers
through :func:`_launch_workers`, and :func:`_launch_workers` launches only
against a :class:`WorkerLaunchPermit` minted by
:func:`issue_worker_launch_permit`. Before this module each lane re-implemented
the gate sequence by convention, and the local rework lane never did (#2039):
``local/mdls`` launched Devin workers uncounted against
``fleet.global_max_concurrent_sessions`` and straight through provider
throttle / operator-hold cooldowns.

Gate sequence -- the ONE implementation, in the order the remote lanes have
always applied it (the provider-throttle deferral payload carries the
governor's report fields, so the governor runs before the throttle read):

1. fleet lock (``try_acquire_fleet_lock``) when
   ``fleet.global_max_concurrent_sessions > 0`` -- held across
   governor -> claim -> launch so independently-running repos cannot both read
   a stale fleet live count and over-dispatch the shared cap;
2. ``_apply_concurrency_governor`` (per-repo cap, fleet cap, host load, and --
   fresh dispatch only -- open-PR / CI-headroom backpressure);
3. provider throttle (``is_throttled``) read under the state lock.

The fresh and remote-rework lanes take the fleet lock at their entry point
(before the stall sweep and candidate scan) via :func:`acquire_fleet_launch_lock`
and hand it to :func:`issue_worker_launch_permit`; the local lane lets the
permit acquire it. Either way only this module constructs a valid lock handle
or permit: validity is a private mint token checked by identity at launch
time, so a hand-built permit is refused rather than trusted.

``tests/test_worker_launch_gate_guard.py`` AST-scans ``src/charlie_work`` and
fails if any function other than :func:`_launch_workers` references
``dispatch_sessions`` -- a new lane that bypasses the gate fails CI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from charlie_work.adapters import AdapterSettings, SessionDispatchResult, SessionRequest
from charlie_work.fleet_registry import try_acquire_fleet_lock

if TYPE_CHECKING:
    from charlie_work.workflow import OrchestratorApp


def _workflow() -> Any:
    # ``charlie_work.workflow`` imports the orchestration delegates, which
    # import this module -- resolve it at call time, not import time. Going
    # through it (rather than ``state`` / ``adapters`` directly) keeps every
    # existing ``charlie_work.workflow.*`` monkeypatch site intercepting.
    import charlie_work.workflow as wf

    return wf


logger = logging.getLogger(__name__)

# Identity-checked proof that a lock handle / permit came from this module.
_MINT = object()

REASON_FLEET_LOCK_HELD = "fleet_lock_held"
REASON_PROVIDER_THROTTLED = "provider_throttled"
REASON_CONCURRENCY_CAP = "concurrency_cap"
REASON_LAUNCH_LOCK_INVALID = "launch_lock_invalid"

FleetLockAcquirer = Callable[[str | None], Any]


class FleetLaunchLock:
    """The fleet-wide launch lock, held for one governor -> claim -> launch window.

    A resource handle, not a value object: it tracks whether it has been
    released. ``lock`` is ``None`` when the fleet cap is disabled
    (``global_max_concurrent_sessions == 0``) -- no cross-repo accounting
    exists to serialize, so no OS lock is taken, but the handle is still
    required so every lane goes through the same path.
    """

    __slots__ = ("_lock", "_released", "_token")

    def __init__(self, lock: Any, token: object) -> None:
        self._lock = lock
        self._released = False
        self._token = token

    @property
    def valid(self) -> bool:
        return self._token is _MINT and not self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._lock is not None:
            self._lock.release()

    def __enter__(self) -> FleetLaunchLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass(frozen=True)
class WorkerLaunchDeferral:
    """The gate refused to issue a permit this pass. Nothing may launch.

    ``governor`` is populated when the governor ran before the deferring gate
    (the provider-throttle case), so the deferral payload reports the same
    governor fields the lanes always reported.
    """

    reason: str
    throttled_until: Any = None
    governor: Any = None

    def report_fields(self) -> dict[str, Any]:
        """Payload fields for this deferral (excludes ``deferred_reason``)."""
        fields: dict[str, Any] = {}
        if self.reason == REASON_PROVIDER_THROTTLED:
            fields["throttled_until"] = self.throttled_until
        if self.governor is not None and self.governor.any_term_enabled:
            fields.update(self.governor.report_fields())
        return fields


@dataclass
class _LaunchLedger:
    """Launches already made against one permit.

    The one mutable part of a permit: the remote rework lane's normal and
    rescue tiers launch in two calls that must share one budget.
    """

    launched: int = 0


@dataclass(frozen=True)
class WorkerLaunchPermit:
    """Authorization to launch at most ``max_launches`` workers this pass.

    ``max_launches`` may be 0 (the governor clamped everything): the lane
    still runs its bookkeeping (deferred-by-concurrency reporting, review
    routing, escalations) but :func:`_launch_workers` accepts only an empty
    batch. Use as a context manager to release a fleet lock the permit
    acquired itself; a lock borrowed from the lane entry point is released by
    that entry point.
    """

    requested: int
    max_launches: int
    governor: Any
    _launch_lock: FleetLaunchLock = field(repr=False, compare=False)
    _owns_lock: bool = field(default=False, repr=False, compare=False)
    _ledger: _LaunchLedger = field(default_factory=_LaunchLedger, repr=False, compare=False)
    _token: object = field(default=None, repr=False, compare=False)

    @property
    def clamped(self) -> bool:
        return self.max_launches < self.requested

    @property
    def remaining(self) -> int:
        return max(0, self.max_launches - self._ledger.launched)

    def release(self) -> None:
        if self._owns_lock:
            self._launch_lock.release()

    def __enter__(self) -> WorkerLaunchPermit:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def acquire_fleet_launch_lock(
    app: OrchestratorApp, *, acquire: FleetLockAcquirer = try_acquire_fleet_lock
) -> FleetLaunchLock | WorkerLaunchDeferral:
    """Gate 1: take the fleet lock when the fleet cap is enabled.

    ``acquire`` is the lock primitive; lanes pass their own module-level
    ``try_acquire_fleet_lock`` name so existing tests that patch it there
    keep intercepting. Whether a lock is required is decided here, never by
    the caller.
    """
    if app.config.fleet.global_max_concurrent_sessions <= 0:
        return FleetLaunchLock(None, _MINT)
    lock = acquire(app.fleet_dir_override)
    if lock is None:
        return WorkerLaunchDeferral(REASON_FLEET_LOCK_HELD)
    return FleetLaunchLock(lock, _MINT)


def issue_worker_launch_permit(
    app: OrchestratorApp,
    requested: int,
    *,
    live_count: int | None = None,
    apply_open_pr_backpressure: bool = False,
    launch_lock: FleetLaunchLock | None = None,
    acquire: FleetLockAcquirer = try_acquire_fleet_lock,
) -> WorkerLaunchPermit | WorkerLaunchDeferral:
    """Run the fleet lock -> governor -> provider throttle gates; mint a permit.

    ``launch_lock``: a lock the lane already took at its entry point via
    :func:`acquire_fleet_launch_lock` (borrowed; the lane releases it). When
    ``None`` the permit acquires the lock itself and releases it on
    ``__exit__`` -- or immediately, if it defers.

    ``live_count`` / ``apply_open_pr_backpressure`` pass straight through to
    ``_apply_concurrency_governor`` (open-PR backpressure is fresh-dispatch
    only, issue #1129). They are forwarded only when non-default so the
    governor is called exactly as each lane always called it.
    """
    owns_lock = launch_lock is None
    if launch_lock is None:
        acquired = acquire_fleet_launch_lock(app, acquire=acquire)
        if isinstance(acquired, WorkerLaunchDeferral):
            return acquired
        launch_lock = acquired
    elif not (isinstance(launch_lock, FleetLaunchLock) and launch_lock.valid):
        return WorkerLaunchDeferral(REASON_LAUNCH_LOCK_INVALID)

    _wf = _workflow()
    try:
        governor_kwargs: dict[str, Any] = {}
        if live_count is not None:
            governor_kwargs["live_count"] = live_count
        if apply_open_pr_backpressure:
            governor_kwargs["apply_open_pr_backpressure"] = True
        gov = app._apply_concurrency_governor(requested, **governor_kwargs)

        with _wf.state_lock(app.paths.state_file):
            state = _wf.load_state(app.paths.state_file)
            throttled = _wf.is_throttled(state)
            throttled_until = state.get("throttled_until")
    except BaseException:
        if owns_lock:
            launch_lock.release()
        raise

    if throttled:
        if owns_lock:
            launch_lock.release()
        return WorkerLaunchDeferral(
            REASON_PROVIDER_THROTTLED, throttled_until=throttled_until, governor=gov
        )
    return WorkerLaunchPermit(
        requested=requested,
        max_launches=max(0, int(gov.dispatch_limit)),
        governor=gov,
        _launch_lock=launch_lock,
        _owns_lock=owns_lock,
        _token=_MINT,
    )


def _refusal_reason(permit: object, count: int) -> str | None:
    if not isinstance(permit, WorkerLaunchPermit) or permit._token is not _MINT:
        return "no valid worker launch permit (not issued by issue_worker_launch_permit)"
    if not permit._launch_lock.valid:
        return "worker launch permit's fleet lock is no longer held"
    if count > permit.remaining:
        return (
            f"{count} launch(es) requested but the permit allows "
            f"{permit.remaining} of {permit.max_launches}"
        )
    return None


def _launch_workers(
    app: OrchestratorApp,
    permit: WorkerLaunchPermit,
    settings: AdapterSettings,
    requests: list[SessionRequest],
) -> list[SessionDispatchResult]:
    """Launch ``requests`` against ``permit`` -- the only ``dispatch_sessions`` caller.

    Refuses (never raises -- errors from launches come back as values) when
    the permit was not minted by :func:`issue_worker_launch_permit`, its
    fleet lock has been released, or the batch exceeds the permit's remaining
    budget: every request comes back as a failed ``SessionDispatchResult``
    and nothing is launched. Successive calls on one permit share its budget.
    """
    reason = _refusal_reason(permit, len(requests))
    if reason is not None:
        logger.error("worker launch refused for %d request(s): %s", len(requests), reason)
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter=settings.adapter,
                ok=False,
                error=f"worker launch refused: {reason}",
            )
            for request in requests
        ]
    permit._ledger.launched += len(requests)
    _wf = _workflow()
    # Reached through the workflow re-export so test fakes that patch
    # ``charlie_work.workflow.dispatch_sessions`` keep intercepting.
    return _wf.dispatch_sessions(
        app.repo_root,
        app._layout.session_manifest,
        app._layout.session_results,
        settings,
        requests,
    )
