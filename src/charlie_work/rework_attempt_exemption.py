"""Provider-throttle rework deaths never count as a rework attempt (issue #2282).

A rework session that dies because the provider throttled it (``rate_limited``,
``quota_exhausted``, ...) says nothing about the work. Before #2282 such a death
still used up one rework attempt. The *death* was exempt: #1684/#1917/#2002
skipped the ``worker_death_at`` credit. The *dispatch* was not. Every rework
dispatch stamps ``redispatch_at`` (``_dispatch_rework_impl``), and every
redispatch cap derives from that list:

* the no-op cap, ``no_op_count = len(redispatch_at) - len(worker_death_at)``
  (``_dispatch_rework_impl`` head check -> ``no_op_rework_cap_exceeded``;
  ``_reap_restore_rework_requested`` -> ``redispatch_cap_exceeded``);
* the death-loop cap, ``_paired_death_count(redispatch_at, worker_death_at)``
  (``worker_death_loop``);
* the raw redispatch cap at the next rework dispatch
  (``len(redispatch_at) > max_auto_redispatch``).

So skipping only the death credit made things worse. An uncredited throttle
death left its dispatch stamp in ``redispatch_at`` with no matching death,
which is exactly what a no-op looks like. Three throttle waves escalated a
healthy PR as ``no_op_rework_cap_exceeded`` (charlie-work #2254 / PR #2264).

:func:`exempt_provider_throttle_rework_death` is the single point of
enforcement. Every lane that resolves a dead rework session's ``failure_kind``
calls it: the orphan sweep's credit seam
(``dead_worker_classification.classify_and_credit_dead_worker``) and the
dead-session restore lane (``dead_worker_sweep.effects_rework``). For a
throttle death it:

1. **refunds** the dead session's own dispatch stamp from ``redispatch_at``.
   That is the latest stamp at or after the epoch's ``dispatched_at``. With
   the stamp gone, every redispatch-derived cap above sees the throttled
   attempt as never having happened, with no change to any reader;
2. marks the PR's ``last_rework_was_startup_death`` exemption flag. This is the
   #1106 flag ``_route_janitor_gate_failure_to_rework`` already honors, so the
   PR-level ``no_op_rework_attempts`` / ``conflict_rework_attempts`` counters
   are not advanced either. ``last_rework_exemption`` records which exemption
   set it;
3. emits ``rework_attempt_exempted_provider_throttle`` so the exemption can be
   observed.

The ``worker_death_at`` non-credit stays at its existing call sites, which
already gate on the same predicate.

The exempt kinds are derived, never listed here. They are the fleet ledger's
``RESTRICTING_FAILURE_KINDS`` (the kinds that restrict a model) united with
``PROVIDER_THROTTLE_FAILURE_KINDS`` (the kinds the ``worker_death_at`` credit
gate already exempts). The union keeps the refund and the non-credit in
lockstep. A kind exempt from one and not the other would turn a throttle
death back into a counted no-op.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .role_quota_ledger import RESTRICTING_FAILURE_KINDS
from .throttle_signatures import PROVIDER_THROTTLE_FAILURE_KINDS
from .write_gate import WriteGate, require_write_gate

PROVIDER_THROTTLE_EXEMPT_KINDS: frozenset[str] = (
    RESTRICTING_FAILURE_KINDS | PROVIDER_THROTTLE_FAILURE_KINDS
)
EXEMPTION_EVENT_KIND = "rework_attempt_exempted_provider_throttle"
PROVIDER_THROTTLE_EXEMPTION = "provider_throttle"
STARTUP_DEATH_EXEMPTION = "startup_death"


def is_provider_throttle_rework_death(failure_kind: str | None) -> bool:
    """True when a dead rework session's ``failure_kind`` is a provider throttle."""
    return failure_kind in PROVIDER_THROTTLE_EXEMPT_KINDS


def _parse(ts: Any) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _refundable_stamp(redispatch_at: Sequence[Any], dispatched_at: Any) -> int | None:
    """Index of the dead session's own dispatch stamp in ``redispatch_at``, or None.

    That is the latest stamp at or after the epoch's ``dispatched_at``. The
    rework dispatch writes ``dispatched_at`` first and the ``redispatch_at``
    stamp right after it. When no stamp qualifies, nothing is refunded:
    ``dispatched_at`` is unknown, or the dispatch never stamped, like a fresh
    implementer dispatch. A guess could refund an earlier, genuinely counted
    attempt.
    """
    start = _parse(dispatched_at)
    if start is None:
        return None
    best: tuple[datetime, int] | None = None
    for index, raw in enumerate(redispatch_at):
        stamp = _parse(raw)
        if stamp is None or stamp < start:
            continue
        if best is None or stamp >= best[0]:
            best = (stamp, index)
    return None if best is None else best[1]


def exempt_provider_throttle_rework_death(
    state: dict[str, Any],
    issue_number: int,
    entry: dict[str, Any],
    failure_kind: str | None,
    *,
    dispatched_at: Any,
    pr_number: int | None,
    source: str,
    write_gate: WriteGate,
) -> bool:
    """Exempt a provider-throttle rework death from every rework-attempt counter.

    Returns False and changes nothing when ``failure_kind`` is not a provider
    throttle. Otherwise it changes ``entry`` (the caller's locked issue entry)
    and ``state`` in place, as the sweep's other in-lock helpers do, and
    returns True. The refund pops one stamp. The PR flag is cleared by the
    next rework dispatch. A repeated call for the same death therefore finds
    nothing left to refund.
    """
    write_gate = require_write_gate(write_gate)
    if not is_provider_throttle_rework_death(failure_kind):
        return False
    raw = entry.get("redispatch_at")
    redispatch_at = list(raw) if isinstance(raw, list) else []
    index = _refundable_stamp(redispatch_at, dispatched_at)
    refunded = redispatch_at.pop(index) if index is not None else None
    if refunded is not None:
        entry["redispatch_at"] = redispatch_at
    if pr_number is not None:
        prs = state.setdefault("prs", {})
        prs[str(pr_number)] = {
            **prs.get(str(pr_number), {}),
            "last_rework_failure_kind": failure_kind,
            "last_rework_was_startup_death": True,
            "last_rework_exemption": PROVIDER_THROTTLE_EXEMPTION,
        }
    state.update(
        write_gate.append_event(  # event-consumer: audit-only -- observability for the #2282 exemption; the refund and PR flag above are the enforcement, pinned by tests/test_rework_throttle_exemption.py.
            state,
            "rework_attempt_exempted_provider_throttle",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "failure_kind": failure_kind,
                "source": source,
                "refunded_redispatch_at": refunded,
                "redispatch_count": len(redispatch_at),
            },
        )
    )
    return True
