"""Classify a no-PR dead worker's log before its orphan fate resolves (issue #2274).

Reconcile used to be the only lane that classified a dead worker from its log,
and it runs about every 31 minutes. A death inside that interval reached the
orphan sweep's no-PR arm first, which only read an *existing*
``dead_worker_failure_kind`` stamp. The redispatch and phantom lanes then
overwrote or reaped the sidecar, so a rate-limit/quota death never reached the
role-quota ledger (#2086) and ``FateResult.throttled`` was always false.

``resolve_fate`` (no-PR stage) is the first place each pass handles a dead
worker, so the classification runs there. It runs in the sweep's lock-free pre
phase, and the lock phase reloads ``state.json`` and discards the pre-phase
copy, so the work is split in two:

* :func:`classify_before_fate` (pre phase): classifies the log through
  ``dead_worker_classification.classify_dead_worker_log``. The adapter helper
  durably stamps the sidecar and records the ledger restriction right away.
  The kind stamp goes onto the pre-phase entry so this pass's fate sees it.
  No cooldown is armed and no event is emitted there, because both would be
  dropped or emitted twice.
* :func:`persist_pre_classified` (lock phase, right after the reload): persists
  each classification once through ``worker_fate.persist_failure``. That stamps
  the entry, arms the per-repo cooldown, and emits ``throttle_window_set``
  through the sweep's write gate.

Under ``dry_run`` nothing is classified, because the adapter helper writes the
sidecar and the ledger outside the write gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import dead_worker_classification, worker_fate
from .apply_context import SweepContext


@dataclass(frozen=True)
class PreClassifiedDeath:
    """One pre-phase classification, carried to the lock phase."""

    worker_pid: Any
    adapter_kind: str
    failure: worker_fate.FailureEvidence


def classify_before_fate(ctx: SweepContext, number: int, entry: dict[str, Any]) -> dict[str, Any]:
    """Classify ``number``'s dead worker and return the entry its fate should read.

    Returns ``entry`` unchanged when it is already stamped, the sweep is a dry
    run, or the log shows no classifiable signature. Otherwise it returns a
    stamped copy, which is also seated in the pre-phase ``ctx.state``, and
    queues the classification for :func:`persist_pre_classified`.
    """
    if ctx.write_gate.dry_run or worker_fate.persisted_failure(entry).kind is not None:
        return entry
    classified = dead_worker_classification.classify_dead_worker_log(
        entry, ctx.sessions_dir, number, ctx.config, now=ctx.now
    )
    if classified is None:
        return entry
    adapter_kind, failure = classified
    ctx.pre_classified[number] = PreClassifiedDeath(
        worker_pid=entry.get("worker_pid"), adapter_kind=adapter_kind, failure=failure
    )
    # Stamp only: a ``throttled_until=None`` evidence leaves the cooldown alone
    # and emits nothing. The lock phase arms it once.
    stamped_state = worker_fate.persist_failure(
        {"issues": {str(number): entry}},
        number,
        worker_fate.FailureEvidence(kind=failure.kind, throttled_until=None, fresh=True),
        adapter_kind=adapter_kind,
        now=ctx.now,
        source="dead_worker_classification",
    )
    stamped = stamped_state["issues"][str(number)]
    issues = ctx.state.get("issues")
    if isinstance(issues, dict) and str(number) in issues:
        issues[str(number)] = stamped
    return stamped


def persist_pre_classified(ctx: SweepContext) -> None:
    """Persist every pre-phase classification onto the freshly locked ``ctx.state``.

    An entry that has been re-dispatched since the pre phase (a different
    ``worker_pid``), already stamped, or removed is skipped. A new epoch's stamp
    must never describe an older death. The ledger restriction and the sidecar
    stamp are already durable, so the skip loses nothing those lanes need.
    """
    for number, death in ctx.pre_classified.items():
        entry = ctx.issue_entry(number)
        if not entry or entry.get("worker_pid") != death.worker_pid:
            continue
        if worker_fate.persisted_failure(entry).kind is not None:
            continue
        ctx.state = worker_fate.persist_failure(
            ctx.state,
            number,
            death.failure,
            adapter_kind=death.adapter_kind,
            now=ctx.now,
            source="dead_worker_classification",
            write_gate=ctx.write_gate,
        )
