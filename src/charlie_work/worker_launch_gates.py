"""The deferral vocabulary and lock-free gates of the worker-launch gate.

Split out of ``worker_launch_gate.py`` (issue #3139, file-size ratchet): the
permit's gate sequence runs in two stages -- everything decidable WITHOUT
the fleet lock (here), then the locked gates in
:func:`charlie_work.worker_launch_gate.issue_worker_launch_permit`. The
lanes' nothing-to-dispatch early return calls
:func:`lock_free_launch_gates` directly so an empty pass still reports the
same deferral a paused or throttled repo would have produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from charlie_work import role_selection
from charlie_work.fleet_pause import FLEET_PAUSED, launch_pause_fields
from charlie_work.state import is_throttled, read_state_snapshot

if TYPE_CHECKING:
    from charlie_work.workflow import OrchestratorApp

REASON_FLEET_LOCK_HELD = "fleet_lock_held"
REASON_FLEET_PAUSED = FLEET_PAUSED
REASON_PROVIDER_THROTTLED = role_selection.REASON_PROVIDER_THROTTLED
REASON_CONCURRENCY_CAP = "concurrency_cap"
REASON_LAUNCH_LOCK_INVALID = "launch_lock_invalid"

# ``ConcurrencyGovernorResult.clamped_by`` when a per-harness ceiling (issue #2757)
# is the term that bound ``dispatch_limit``.
CLAMPED_BY_HARNESS_MAX = "harness_max"


@dataclass(frozen=True)
class WorkerLaunchDeferral:
    """The gate refused to issue a permit this pass. Nothing may launch.

    ``governor`` is populated only when the governor ran before the
    deferring gate -- i.e. the authoritative provider-throttle read (gate
    4), whose payload reports the same governor fields the lanes always
    reported. A pre-check (gate 0) throttle deferral has ``governor=None``:
    it carries ``throttled_until`` and no governor report fields. ``ok`` is the
    ``CommandResult.ok`` the lanes should surface: ``fleet_lock_held`` is an
    ok-deferral (``True``) so ``dispatch_deferral`` records it and counts it
    toward ``dispatch_starved`` -- that is the visibility the starvation
    alarm is built on -- while ``provider_throttled`` and
    ``launch_lock_invalid`` stay ``False`` (a policy backoff / a bug, each
    surfaced as a not-ok result). ``extra`` carries reason-specific payload
    fields (e.g. lock-wait / holder metadata for ``fleet_lock_held``).
    """

    reason: str
    throttled_until: Any = None
    governor: Any = None
    ok: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def report_fields(self) -> dict[str, Any]:
        """Payload fields for this deferral (excludes ``deferred_reason``)."""
        fields: dict[str, Any] = dict(self.extra)
        if self.reason in (REASON_PROVIDER_THROTTLED, role_selection.REASON_HARNESS_AUTH_BLOCKED):
            fields["throttled_until"] = self.throttled_until
        if self.governor is not None and self.governor.any_term_enabled:
            fields.update(self.governor.report_fields())
        return fields


def lock_free_launch_gates(
    app: OrchestratorApp,
) -> tuple[role_selection.RoleSelection | None, WorkerLaunchDeferral | None]:
    """The gates of ``issue_worker_launch_permit`` decidable without the lock.

    The #2766 pause gate, the #2086 role-chain resolution against the fleet
    quota ledger, and the #2055 gate-0 provider-throttle/operator-hold
    pre-check -- every deferral decidable lock-free, in the permit's order.
    Returns ``(selection, deferral)``: ``selection`` is the resolved worker
    role chain entry (``None`` only when the pause gate fired first) and
    ``deferral`` is ``None`` when the pass may proceed to the locked gates.

    Issue #3139: the lanes' nothing-to-dispatch early return runs the same
    gates in place of the permit, so an empty pass on a paused or throttled
    repo still reports the deferral instead of a clean "nothing to do".
    """
    # Issue #2766 pause gate: the flag is read fresh, lock-free, before anything
    # else -- a fleet pass that began unpaused still stops launching once it is set.
    if (paused := launch_pause_fields(app)) is not None:
        return None, WorkerLaunchDeferral(REASON_FLEET_PAUSED, ok=True, extra=paused)

    # Issue #2055 gate 0: lock-free provider-throttle/operator-hold
    # pre-check -- a cheap unlocked ``state.json`` read via
    # ``read_state_snapshot`` (issue #3141: the sanctioned non-mutating
    # reader -- an unlocked ``load_state`` could quarantine a live file on a
    # transient read failure). A throttled repo
    # defers here without ever contending for the fleet lock (a throttled
    # fleet previously still saturated it: the lock was taken before the
    # throttle check and held across the whole scan). The authoritative
    # throttle read still runs under the state lock after the governor
    # below, so a throttle set after this check cannot slip through. The
    # deferred payload from THAT path keeps its governor fields; a pre-check
    # deferral carries none -- the governor has not run.
    # Issue #2086: resolve the worker role chain against the fleet quota
    # ledger first. Every entry restricted is the chained form of "provider
    # throttled"; a per-repo window the ledger explains is covered below.
    selection = role_selection.select_for_config(app.config.worker.chain, app.config)
    if selection.exhausted:
        return selection, WorkerLaunchDeferral(
            selection.exhausted_reason,
            throttled_until=selection.report_fields().get("chain_retry_at"),
            extra=selection.report_fields(),
        )
    # An unreadable state (None) reads as "not throttled" here -- the
    # authoritative locked read below still runs before any launch.
    pre_state = read_state_snapshot(app.paths.state_file) or {}
    if is_throttled(pre_state) and not role_selection.window_covered(
        pre_state.get("throttled_until"),
        selection,
        reason=pre_state.get("throttle_reason"),
        adapter_kind=pre_state.get("throttle_adapter_kind"),
        harness=pre_state.get("throttle_harness"),
        model=pre_state.get("throttle_model"),
    ):
        return selection, WorkerLaunchDeferral(
            REASON_PROVIDER_THROTTLED, throttled_until=pre_state.get("throttled_until")
        )
    return selection, None


def _harness_free_slots(app: OrchestratorApp, gov: Any) -> dict[str, int]:
    """Free slots under each configured per-harness ceiling (empty when none is set)."""
    free: dict[str, int] = {}
    if not app.config.fleet.max_concurrent_sessions_by_harness:
        return free
    for harness in gov.harness_caps:
        slots = gov.harness_available(harness)
        if slots is not None:
            free[harness] = slots
    return free


def _apply_harness_caps(
    app: OrchestratorApp, selection: role_selection.RoleSelection, gov: Any
) -> tuple[role_selection.RoleSelection, Any]:
    """Bound the launch by the selected entry's harness ceiling (issue #2757).

    ``fleet.max_concurrent_sessions_by_harness`` limits only the sessions that
    draw on a given harness's quota, so it can only be enforced once the harness
    is known -- i.e. after role-chain selection -- and, like the fleet cap, on a
    live count taken under the fleet lock (the governor measured it). Unset map:
    ``gov`` and ``selection`` come back untouched.

    * Selected entry's harness has no ceiling, or has room: the launch is bounded
      to the free slots (``clamped_by="harness_max"`` when that binds).
    * Selected entry's harness is full: fall through to the next chain entry that
      is neither quota-restricted nor on a full harness
      (:func:`role_selection.reselect_within_harness_caps`), exactly as a quota
      restriction routes past an entry, then bound by that entry's free slots.
    * No such entry (a length-1 chain, or every entry restricted / full): the
      launch is clamped to zero with ``clamped_by="harness_max"`` -- a deferral,
      not a provider throttle, so no repo-wide throttle window is implied.
    """
    # Gated on config, the source of truth, not on ``gov``: the governor only
    # populates the harness fields when the same map is set.
    if not app.config.fleet.max_concurrent_sessions_by_harness or selection.entry is None:
        return selection, gov
    reselected = role_selection.reselect_within_harness_caps(selection, gov.harness_available)
    if reselected.entry is None:
        if gov.dispatch_limit > 0:
            gov = replace(gov, dispatch_limit=0, clamped=True, clamped_by=CLAMPED_BY_HARNESS_MAX)
        return selection, gov
    free = gov.harness_available(reselected.entry.harness)
    if free is not None and free < gov.dispatch_limit:
        gov = replace(gov, dispatch_limit=free, clamped=True, clamped_by=CLAMPED_BY_HARNESS_MAX)
    return reselected, gov
