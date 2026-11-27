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

Gate sequence -- the ONE implementation, inside
:func:`issue_worker_launch_permit` (issue #2055 ordering):

P. fleet pause (issue #2766) -- a lock-free read of ``fleet-pause.json``, ahead
   of everything below. The supervisor reads the flag only between fleet passes,
   but a fleet pass walks the registry for minutes, so this is what stops the
   repos it has not reached yet from launching after the operator paused. Defers
   ``fleet_paused`` (``ok=True``, so ``dispatch_deferred`` records it) with
   ``paused_at`` / ``pause_reason``; ``--ignore-pause`` (``app.ignore_pause``)
   bypasses it. Running workers, reviews, reconcile and salvage do not pass
   through the permit and are untouched;
0. provider-throttle / operator-hold PRE-CHECK -- a lock-free ``state.json``
   read. A throttled repo defers here without ever touching the fleet lock,
   so a throttled fleet cannot saturate it. The governor has not run at this
   point, so a pre-check deferral carries ``throttled_until`` and NO governor
   report fields -- only the step-4 deferral carries those;
1. fleet lock (``try_acquire_fleet_lock``) realized when
   ``fleet.global_max_concurrent_sessions > 0`` or
   ``fleet.max_concurrent_sessions_by_harness`` is non-empty, retried with jitter for up
   to ``fleet.launch_lock_wait_seconds`` -- held only across
   governor -> claim -> launch so independently-running repos cannot both
   read a stale fleet live count and over-dispatch the shared cap, and a
   briefly-held lock no longer costs a whole pass;
2. ``live_count`` computed under the lock (post-stall-sweep state);
3. ``_apply_concurrency_governor`` (per-repo cap, fleet cap, host load, and --
   fresh dispatch only -- open-PR / CI-headroom backpressure);
4. provider throttle (``is_throttled``) read under the state lock -- the
   authoritative check (it also catches a throttle set after step 0), still
   ordered after the governor so the provider-throttled deferral payload
   carries the governor's report fields as it always has.

Role chain (issue #2086): before gate 0 the worker role chain is resolved
against the fleet quota ledger (:func:`charlie_work.role_selection.select_for_launch`).
Every entry restricted defers ``provider_throttled`` with the chain fields; a
per-repo throttle window the ledger fully explains is *covered* (selection
already routed past the restricted entry) and does not block gates 0 or 4.
The selection rides on the permit, and :func:`_launch_workers` launches the
worker-role settings on the selected entry, stamps each launched session's
sidecar with it, and emits ``role_fallback_selected`` past the primary. A
length-1 chain never reads the ledger and behaves exactly as before.

Per-harness ceilings (issue #2757): ``fleet.global_max_concurrent_sessions``
counts every live session whatever its harness, but a provider quota (e.g. a
shared Devin account) binds only the sessions that draw on it.
``fleet.max_concurrent_sessions_by_harness`` adds a ceiling per harness. The
governor measures the fleet-wide live count per harness under the fleet lock
(step 3); right after it :func:`_apply_harness_caps` bounds the launch by the
role-chain entry selected above. An entry whose harness is full is passed over
for the next entry whose harness has room (the same fall-through a quota
restriction takes); with none left -- a length-1 chain, or every entry full or
restricted -- the launch is clamped to zero with ``clamped_by: harness_max``
(a deferral, never a provider throttle). The total cap still bounds every
harness, so a launch on an uncapped harness is limited only by it. A
``model:<tier>`` group that lands on a different capped harness than the
permit's is bounded in :func:`_tier_groups`. The rescue tier (role ``""``)
launches on its own configured harness exactly as configured and is not
bounded by a per-harness ceiling; it still counts against the total.

Model tier (TIS-CW-6): :func:`_launch_workers` splits a worker-role batch by the
issues' ``model:<tier>`` labels (``model_tier``). Each tier the chain can
serve launches on the first unrestricted entry of that model family and emits
``worker_model_tier_selected``; unlabelled issues, and tiers the chain cannot
serve (``worker_model_tier_fallback``), launch on the permit's selection.

The fresh and remote-rework lanes mint a *pending* lock handle at their entry
point via :func:`acquire_fleet_launch_lock` -- which does NOT touch the OS
lock -- and hand it to :func:`issue_worker_launch_permit`; the local lane lets
the permit mint one itself. Either way only this module constructs a valid
lock handle or permit: validity is a private mint token checked by identity at
launch time, so a hand-built permit is refused rather than trusted.

``tests/test_worker_launch_gate_guard.py`` AST-scans ``src/charlie_work`` and
fails if any function other than :func:`_launch_workers` touches the
``worker_launch`` port, or anything other than the port's Real /
:func:`_launch_workers` itself / the ``workflow`` re-export references
``dispatch_sessions`` -- a new lane that bypasses the gate fails CI.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

# The dotted form, not ``from charlie_work import model_tier``: the dormant-module
# import graph (tests/test_dormant_fleet_marking.py) cannot see the bare-package
# form, and this is model_tier's only importer.
import charlie_work.model_tier as model_tier
from charlie_work import layout, role_selection
from charlie_work.adapters import (
    AdapterSettings,
    SessionDispatchResult,
    SessionRequest,
    manifest_adapter_label,
    write_session_results,
)
from charlie_work.atomic_write import write_json_atomic
from charlie_work.fleet_lock_holder import fleet_lock_held_extra
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.worker_launch_gates import (
    CLAMPED_BY_HARNESS_MAX,  # noqa: F401  (deliberate re-export)
    REASON_CONCURRENCY_CAP,  # noqa: F401  (deliberate re-export)
    REASON_FLEET_LOCK_HELD,
    REASON_FLEET_PAUSED,  # noqa: F401  (deliberate re-export)
    REASON_LAUNCH_LOCK_INVALID,
    REASON_PROVIDER_THROTTLED,
    WorkerLaunchDeferral,
    _apply_harness_caps,
    _harness_free_slots,
    lock_free_launch_gates,
)
from charlie_work.host.clock import Clock
from charlie_work.host.launch import worker_error_results

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

FleetLockAcquirer = Callable[[str | None], Any]

# Bounded-wait retry cadence for ``FleetLaunchLock.ensure_acquired`` (issue
# #2055): the lock primitive itself stays non-blocking; the wait budget and
# the jitter live here so every lane shares one policy.
_LOCK_RETRY_MIN_SECONDS = 0.05
_LOCK_RETRY_MAX_SECONDS = 0.15


class FleetLaunchLock:
    """The fleet-wide launch lock, held for one governor -> claim -> launch window.

    A resource handle, not a value object: it tracks whether it has been
    released. Two-phase (issue #2055): the lane mints the handle at its entry
    point via :func:`acquire_fleet_launch_lock` so the lane's ``finally``
    release covers every impl exit path, but the OS lock itself is realized
    later -- by :meth:`ensure_acquired` inside
    :func:`issue_worker_launch_permit`, immediately before the governor -- so
    it is never held across the pass's read-only scan
    (``issue_list``/``pr_list``/``issue_view``/stall sweep). A handle minted
    with ``acquire=None`` is a no-op (fleet cap disabled): no cross-repo
    accounting exists to serialize, so no OS lock is taken, but the handle is
    still required so every lane goes through the same path.
    """

    __slots__ = (
        "_lock",
        "_released",
        "_token",
        "_acquire",
        "_fleet_dir_override",
        "_holder_repo",
        "_wrote_holder",
        "_clock",
        "_acquired_at",
        "_released_at",
        "_permit_issued_at",
        "_launch_started_at",
        "_launch_done_at",
    )

    def __init__(
        self,
        lock: Any,
        token: object,
        *,
        acquire: FleetLockAcquirer | None = None,
        fleet_dir_override: str | None = None,
        holder_repo: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._lock = lock
        self._released = False
        self._token = token
        self._acquire = acquire
        self._fleet_dir_override = fleet_dir_override
        self._holder_repo = holder_repo
        self._wrote_holder = False
        self._clock = clock
        self._acquired_at: float | None = None
        self._released_at: float | None = None
        self._permit_issued_at: float | None = None
        self._launch_started_at: float | None = None
        self._launch_done_at: float | None = None

    @property
    def valid(self) -> bool:
        return self._token is _MINT and not self._released

    def ensure_acquired(self, wait_seconds: float) -> float | None:
        """Realize the OS lock, retrying with jitter for up to ``wait_seconds``.

        Returns the seconds spent acquiring (``0.0`` when the first try lands
        or the handle is a cap-disabled no-op), or ``None`` when the wait
        budget was exhausted -- the handle is then dead (``valid`` is False
        and ``release()`` has nothing to release), which is the
        ``fleet_lock_held`` outcome.
        """
        if self._released:
            return None
        if self._acquire is None:
            return 0.0
        acquire = self._acquire
        start = time.monotonic()
        deadline = start + max(0.0, wait_seconds)
        lock = acquire(self._fleet_dir_override)
        while lock is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(
                min(remaining, random.uniform(_LOCK_RETRY_MIN_SECONDS, _LOCK_RETRY_MAX_SECONDS))
            )
            lock = acquire(self._fleet_dir_override)
        self._acquire = None
        if lock is None:
            self._released = True
            return None
        self._lock = lock
        if self._clock is not None:
            self._acquired_at = self._clock.monotonic()
        self._write_holder()
        return time.monotonic() - start

    def note_permit_issued(self) -> None:
        """Issue #3139: boundary between the ``governor`` and ``selection`` phases.

        Recorded only when :func:`issue_worker_launch_permit` mints a
        ``WorkerLaunchPermit`` -- a deferral never reaches selection, so its
        ``governor`` phase spans the whole hold.
        """
        if self._clock is not None and self._acquired_at is not None:
            self._permit_issued_at = self._clock.monotonic()

    def note_launch_started(self) -> None:
        """Issue #3139: boundary between the ``selection`` and ``launch`` phases.

        First-write-wins: the rework lane launches a normal and a rescue
        batch through two ``_launch_workers`` calls; ``selection`` ends at
        the FIRST attempt.
        """
        if (
            self._clock is not None
            and self._acquired_at is not None
            and self._launch_started_at is None
        ):
            self._launch_started_at = self._clock.monotonic()

    def note_launch_done(self) -> None:
        """Issue #3139: boundary between the ``launch`` and ``post_launch`` phases.

        Last-write-wins: a second ``_launch_workers`` call (rework rescue
        tier) extends ``launch`` to the end of the LAST attempt.
        """
        if self._clock is not None and self._acquired_at is not None:
            self._launch_done_at = self._clock.monotonic()

    def event_fields(self) -> dict[str, Any]:
        """Issue #3089: ``launch_lock_held_seconds`` (Host clock) for the aggregate
        Event; issue #3139: ``launch_lock_phase_seconds`` beside it -- one entry
        per phase the lane reached while it held the lock (``governor``,
        ``selection``, ``launch``, ``post_launch``), each running to the next
        phase's boundary so the entries sum to the hold time. A cap-disabled
        no-op handle reports nothing; a realizable handle that was never
        realized reports an exact ``0`` hold and an empty phase map.
        Elapsed-so-far if not yet released."""
        if self._acquired_at is None:
            if self._acquire is None:
                return {}
            return {"launch_lock_held_seconds": 0.0, "launch_lock_phase_seconds": {}}
        if self._clock is None:
            return {}
        end = self._released_at if self._released_at is not None else self._clock.monotonic()
        starts: list[tuple[str, float]] = [("governor", self._acquired_at)]
        if self._permit_issued_at is not None:
            starts.append(("selection", self._permit_issued_at))
        if self._launch_started_at is not None:
            starts.append(("launch", self._launch_started_at))
        if self._launch_done_at is not None:
            starts.append(("post_launch", self._launch_done_at))
        phases = {
            name: (starts[i + 1][1] if i + 1 < len(starts) else end) - start
            for i, (name, start) in enumerate(starts)
        }
        return {
            "launch_lock_held_seconds": end - self._acquired_at,
            "launch_lock_phase_seconds": phases,
        }

    def _write_holder(self) -> None:
        """Record who holds the lock so a starved waiter can report it (issue #2055)."""
        if self._holder_repo is None:
            return
        path = layout.fleet_lock_holder_path(override=self._fleet_dir_override)
        payload = {
            "repo": self._holder_repo,
            "pid": os.getpid(),
            "acquired_at": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, payload)
        except OSError:
            return  # best-effort metadata -- never the reason a launch fails
        self._wrote_holder = True

    def _clear_holder(self) -> None:
        if not self._wrote_holder:
            return
        self._wrote_holder = False
        try:
            layout.fleet_lock_holder_path(override=self._fleet_dir_override).unlink()
        except OSError:
            pass

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._lock is None:
            return
        if self._acquired_at is not None and self._clock is not None:
            self._released_at = self._clock.monotonic()
        # Drop our holder metadata BEFORE releasing the OS lock: done the
        # other way around, a successor could write its own sidecar between
        # the two steps and have it unlinked from under it.
        self._clear_holder()
        self._lock.release()

    def __enter__(self) -> FleetLaunchLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass
class _LaunchLedger:
    """Launches already made against one permit.

    The one mutable part of a permit: the remote rework lane's normal and
    rescue tiers launch in two calls that must share one budget.
    """

    launched: int = 0
    # Issue #2757: free slots left under each ``max_concurrent_sessions_by_harness``
    # ceiling when the permit was issued, drawn down by tier groups that launch on
    # a harness other than the permit's own (see ``_tier_groups``).
    harness_free: dict[str, int] = field(default_factory=dict)


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
    # Issue #2086: the worker role-chain entry this permit launches on.
    role_selection: Any = field(default=None, repr=False, compare=False)

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
    app: OrchestratorApp,
    *,
    acquire: FleetLockAcquirer = try_acquire_fleet_lock,
    cap: int | None = None,
) -> FleetLaunchLock:
    """Mint this lane's fleet-launch-lock handle -- the OS lock is NOT taken here.

    Issue #2055: the handle is minted at lane entry so the lane's ``finally``
    release covers every impl exit path, but the real acquisition happens
    inside :func:`issue_worker_launch_permit` (bounded wait,
    ``fleet.launch_lock_wait_seconds``) immediately before the governor --
    the lock is never held across the pass's read-only scan. It can therefore
    never defer: ``fleet_lock_held`` is produced by the permit, not here.

    ``acquire`` is the lock primitive the handle will realize through; lanes
    pass their own module-level ``try_acquire_fleet_lock`` name so existing
    tests that patch it there keep intercepting. Whether a lock is required
    is decided here, never by the caller. ``cap`` is the fleet-wide cap the
    lock serializes accounting for: the worker cap by default, or the
    reviewer cap for the review lane (issue #2084) -- both share the one
    fleet lock file, each lane opting in only when ITS cap is enabled.
    """
    if cap is None:
        cap = app.config.fleet.global_max_concurrent_sessions
        if cap <= 0 and app.config.fleet.max_concurrent_sessions_by_harness:
            # Issue #2757: a per-harness ceiling is cross-repo accounting too -- two
            # repos reading the same stale per-harness live count would both launch
            # past it -- so it needs the lock even with no total cap configured.
            cap = 1
    if cap <= 0:
        return FleetLaunchLock(None, _MINT)
    return FleetLaunchLock(
        None,
        _MINT,
        acquire=acquire,
        fleet_dir_override=app.fleet_dir_override,
        holder_repo=app.repo_root.name,
        clock=app.host.clock,
    )


def issue_worker_launch_permit(
    app: OrchestratorApp,
    requested: int,
    *,
    live_count: int | None = None,
    apply_open_pr_backpressure: bool = False,
    launch_lock: FleetLaunchLock | None = None,
    acquire: FleetLockAcquirer = try_acquire_fleet_lock,
) -> WorkerLaunchPermit | WorkerLaunchDeferral:
    """Run the pause + throttle pre-checks -> fleet lock -> governor -> throttle gates.

    ``launch_lock``: a handle the lane minted at its entry point via
    :func:`acquire_fleet_launch_lock` (borrowed; the lane releases it). When
    ``None`` the permit mints one itself and releases it on ``__exit__`` --
    or immediately, if it defers.

    ``live_count`` is computed under the fleet lock when the caller did not
    supply it (issue #2055: the count is post-stall-sweep state and must be
    read while the lock is held, not during the lock-free scan).
    ``apply_open_pr_backpressure`` passes straight through to
    ``_apply_concurrency_governor`` (open-PR backpressure is fresh-dispatch
    only, issue #1129); it is forwarded only when set so the governor is
    called exactly as each lane always called it.
    """
    _wf = _workflow()

    # Issue #3139: the lock-free gates are shared with the lanes'
    # nothing-to-dispatch early return so an empty pass on a paused or
    # throttled repo still reports the same deferral.
    selection, preflight = lock_free_launch_gates(app)
    if preflight is not None:
        return preflight
    assert selection is not None  # only the pause deferral skips selection

    owns_lock = launch_lock is None
    if launch_lock is None:
        launch_lock = acquire_fleet_launch_lock(app, acquire=acquire)
    elif not (isinstance(launch_lock, FleetLaunchLock) and launch_lock.valid):
        return WorkerLaunchDeferral(REASON_LAUNCH_LOCK_INVALID)

    # Issue #2055 gate 1: realize the OS lock HERE -- after the scans, right
    # before the governor -- with a bounded jittered wait so a briefly-held
    # lock does not cost a whole pass.
    wait_seconds = app.config.fleet.launch_lock_wait_seconds
    waited = launch_lock.ensure_acquired(wait_seconds)
    if waited is None:
        return WorkerLaunchDeferral(
            REASON_FLEET_LOCK_HELD, ok=True, extra=fleet_lock_held_extra(app, wait_seconds)
        )

    try:
        if live_count is None:
            live_count = app.host.sessions.live_workers(
                app._layout.sessions_dir, app.paths.state_file
            )
        governor_kwargs: dict[str, Any] = {"live_count": live_count}
        if apply_open_pr_backpressure:
            governor_kwargs["apply_open_pr_backpressure"] = True
        gov = app._apply_concurrency_governor(requested, **governor_kwargs)
        selection, gov = _apply_harness_caps(app, selection, gov)

        with _wf.state_lock(app.paths.state_file):
            state = _wf.load_state(app.paths.state_file)
            throttled_until = state.get("throttled_until")
            throttled = _wf.is_throttled(state) and not role_selection.window_covered(
                throttled_until,
                selection,
                reason=state.get("throttle_reason"),
                adapter_kind=state.get("throttle_adapter_kind"),
                harness=state.get("throttle_harness"),
                model=state.get("throttle_model"),
            )
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
    # Issue #3139: the governor phase ends here -- a deferral never reaches
    # this mark, so its ``governor`` phase runs to release as the sole entry.
    launch_lock.note_permit_issued()
    return WorkerLaunchPermit(
        requested=requested,
        max_launches=max(0, int(gov.dispatch_limit)),
        governor=gov,
        _launch_lock=launch_lock,
        _owns_lock=owns_lock,
        _ledger=_LaunchLedger(harness_free=_harness_free_slots(app, gov)),
        _token=_MINT,
        role_selection=selection,
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


def _tier_groups(
    app: OrchestratorApp,
    selection: Any,
    settings: AdapterSettings,
    requests: list[SessionRequest],
    harness_free: dict[str, int] | None = None,
) -> list[tuple[Any, str | None, list[SessionRequest]]]:
    """Split a worker batch by ``model:<tier>`` label (TIS-CW-6): ``(selection, tier, requests)``.

    The first group launches on the permit's own selection (the fleet's default
    tier): unlabelled requests, plus every request whose tier the chain cannot
    serve, each recorded as ``worker_model_tier_fallback``. Each tier the chain
    serves gets its own group on that tier's entry. Rescue-tier settings (role
    ``""``), a permit without a selection, or an empty prefix give one group,
    launched exactly as before TIS-CW-6; so does an empty batch, so the manifest is
    still written.

    ``harness_free`` (issue #2757, the permit ledger's free slots under each
    ``fleet.max_concurrent_sessions_by_harness`` ceiling) bounds a tier group that
    would land on a capped harness *other than the permit's own*: the permit's
    ``max_launches`` was clamped by the permit's harness only, so a tier entry on
    a different capped harness needs its own check. A group that does not fit in
    that harness's free slots launches on the permit's selection instead
    (``worker_model_tier_fallback`` with reason ``harness_cap``); one that fits
    draws the slots down so a second tier on the same harness sees what is left.
    A tier group on the permit's own harness needs no check -- the whole batch is
    already within the permit's clamp.
    """
    prefix = app.config.labels.model_tier_prefix
    if selection is None or settings.role != "worker" or not prefix:
        return [(selection, None, list(requests))]
    default: list[SessionRequest] = []
    by_tier: dict[str, list[SessionRequest]] = {}
    for request in requests:
        tiers = model_tier.model_tiers(request.labels, prefix)
        if len(tiers) == 1:
            by_tier.setdefault(tiers[0], []).append(request)
            continue
        if tiers:
            model_tier.emit_tier_fallback(
                app.write_gate,
                tier=",".join(tiers),
                reason=model_tier.CONFLICTING,
                numbers=[request.issue_number],
            )
        default.append(request)
    served: list[tuple[Any, str | None, list[SessionRequest]]] = []
    for tier, members in sorted(by_tier.items()):
        tier_selection, reason = model_tier.select_for_config(selection.chain, tier, app.config)
        if tier_selection is None:
            model_tier.emit_tier_fallback(
                app.write_gate,
                tier=tier,
                reason=reason,
                numbers=[request.issue_number for request in members],
            )
            default.extend(members)
            continue
        harness = tier_selection.entry.harness
        own_harness = selection.entry is not None and selection.entry.harness == harness
        if harness_free is not None and harness in harness_free and not own_harness:
            free = harness_free[harness]
            if len(members) > free:
                model_tier.emit_tier_fallback(
                    app.write_gate,
                    tier=tier,
                    reason=model_tier.HARNESS_CAP,
                    numbers=[request.issue_number for request in members],
                )
                default.extend(members)
                continue
            harness_free[harness] = free - len(members)
        served.append((tier_selection, tier, members))
    if default or not served:
        return [(selection, None, default), *served]
    return served


def _launch_workers(
    app: OrchestratorApp,
    permit: WorkerLaunchPermit,
    settings: AdapterSettings,
    requests: list[SessionRequest],
) -> list[SessionDispatchResult]:
    """Launch ``requests`` against ``permit`` -- the only worker-launch-port caller.

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
    # Issue #3139: the launch phase brackets the attempts themselves -- a
    # refused batch never reaches this mark, so its time stays in selection.
    permit._launch_lock.note_launch_started()
    _wf = _workflow()
    groups = _tier_groups(
        app, permit.role_selection, settings, requests, permit._ledger.harness_free
    )
    results: list[SessionDispatchResult] = []
    adapters: set[str] = set()
    for selection, tier, group in groups:
        if results and group and settings.launch_stagger_seconds > 0 and not settings.dry_run:
            # A batch split by tier is still one burst: pace the boundary launch
            # the way ``dispatch_sessions`` paces launches within a group.
            time.sleep(settings.launch_stagger_seconds)
        group_settings = settings
        if selection is not None and settings.role == "worker" and selection.is_fallback:
            # Issue #2086: worker-role settings follow the chain entry -- the
            # permit's, or the one a ``model:<tier>`` label picked (TIS-CW-6);
            # rescue-tier settings (role "") launch exactly as configured.
            group_settings = role_selection.worker_settings_for(app, selection)
        adapters.add(group_settings.adapter)
        # Reached through the host's worker-launch port (issue #2229); the
        # Real resolves ``adapters.dispatch_sessions`` at call time so test
        # fakes that patch it on ``charlie_work.adapters`` keep intercepting.
        try:
            group_results = app.host.worker_launch.launch(
                app.repo_root,
                app._layout.session_manifest,
                app._layout.session_results,
                group_settings,
                group,
            )
        except Exception as exc:
            # The permit ledger already debited this batch above; a raise at
            # the launch seam still comes back as per-request failure values
            # (never a propagate) so the pass completes and the lane's normal
            # path releases the permit's fleet lock.
            group_results = worker_error_results(
                app.repo_root, group_settings, group, f"launch failed: {exc}"
            )
        results.extend(group_results)
        if selection is not None and group_settings.role == "worker":
            launched = [result.issue_number for result in group_results if result.ok]
            for number in launched:
                role_selection.stamp_launch(
                    group_settings.sessions_dir, number, "worker", selection
                )
            if launched and tier is None:
                role_selection.emit_fallback_selected(
                    app.write_gate, role="worker", selection=selection, numbers=launched
                )
            elif launched:
                model_tier.emit_tier_selected(
                    app.write_gate, tier=tier, selection=selection, numbers=launched
                )
    if len(groups) > 1:
        # Each group's ``dispatch_sessions`` overwrote the manifest and results
        # files with its own subset; rewrite both once with the whole batch, as
        # dispatch_rework does for its rescue-tier split.
        _wf.write_session_manifest(
            app._layout.session_manifest, requests, adapter=manifest_adapter_label(adapters)
        )
        write_session_results(app._layout.session_results, results)
    permit._launch_lock.note_launch_done()
    return results
