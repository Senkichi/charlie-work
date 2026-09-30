"""Rule 1's stale-evidence reporting: build, dedup and emit
``worker_evidence_stale`` for every ``resolve_fate`` consumer.

Split out of ``worker_fate.py`` (file-size ratchet). The fate/evidence types
are only annotations here, so they are imported under ``TYPE_CHECKING`` --
``worker_fate`` re-exports this module's public names and importing it back at
runtime would cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .instrumentation import log_event
from .state import load_state, load_state_locked, save_state, state_lock

if TYPE_CHECKING:
    from .worker_fate import StaleEvidence, WorkerFate

# --------------------------------------------------------------------------
# B6 (wf-review-opus.md) / design doc §5: rule 1's stale-evidence event.
# ``FateBasis.stale`` (rule 1/7's freshness step) was populated from the
# start, but nothing built the event the plan promised ("Stale evidence is
# ignored **and emits a stale-evidence event**") or read ``basis.stale`` at
# all -- a prior-dispatch outcome (including a real pushed branch) could be
# silently dropped with zero signal in events.db. These two functions are
# the module's own answer (design doc §5); callers wire them in per site,
# same as every other Group B consumer.
# --------------------------------------------------------------------------


def stale_evidence_key(stale: StaleEvidence) -> str:
    """Dedup key for one ``StaleEvidence``: once per ``(source, written_at)``.

    Design doc §5's dedup key, keyed by ``entry["stale_evidence_reported"]``
    at the caller -- a dead worker re-swept every pass must not re-emit the
    same stale candidate and flood the 2000-entry events ring.
    """
    written = stale.written_at.isoformat() if stale.written_at is not None else "unknown"
    return f"{stale.source}:{written}"


def stale_evidence_events(
    entry: Mapping[str, Any], fate: WorkerFate
) -> list[tuple[str, dict[str, Any]]]:
    """Build ``(kind, payload)`` pairs for not-yet-reported rule-1 stale evidence.

    Design doc §5: kind ``worker_evidence_stale`` (level warning), one event
    per ``StaleEvidence`` in ``fate.basis.stale`` whose :func:`stale_evidence_key`
    is not already in ``entry.get("stale_evidence_reported")``. Payload:
    ``issue_number, source, reason, written_at, dispatched_at, evidence_head,
    live_head, adapter`` -- ``dispatched_at`` and ``adapter`` come from
    ``entry`` itself (``FateBasis`` does not carry either; every fate-computing
    call site already stamps ``entry["dispatched_at"]``/``entry["adapter"]``
    from the same evidence used to resolve the fate).

    Pure: does not mutate ``entry``. The caller emits the returned events
    through ``_record_event``/``append_event``/``log_event`` (ADR-0005; see
    CLAUDE.md's instrumentation invariant for which one applies where) and
    merges :func:`stale_evidence_key` for each of ``fate.basis.stale`` into
    ``entry["stale_evidence_reported"]`` itself, in the same state-lock
    section, so the dedup marker and the emitted events never drift apart.
    """
    basis = fate.basis
    already_reported = set(entry.get("stale_evidence_reported") or ())
    dispatched_at = entry.get("dispatched_at")
    if dispatched_at is None and basis.dispatched_at is not None:
        dispatched_at = basis.dispatched_at.isoformat()
    adapter = entry.get("adapter")
    events: list[tuple[str, dict[str, Any]]] = []
    for stale in basis.stale:
        key = stale_evidence_key(stale)
        if key in already_reported:
            continue
        # N-d: one legacy terminal record yields two ``StaleEvidence`` (outcome
        # and ``ended_at``/``exit_code``) sharing a key -- emit it once.
        already_reported.add(key)
        events.append(
            (
                "worker_evidence_stale",
                {
                    "issue_number": basis.issue_number,
                    "source": str(stale.source),
                    "reason": str(stale.reason),
                    "written_at": stale.written_at.isoformat() if stale.written_at else None,
                    "dispatched_at": dispatched_at,
                    "evidence_head": stale.evidence_head,
                    "live_head": stale.live_head,
                    "adapter": adapter,
                },
            )
        )
    return events


def stale_terminal_fate(
    issue_number: int, ended_at: datetime | None, dispatched_at: datetime | None
) -> WorkerFate:
    """An evidence-only carrier fate for a stale terminal record that
    ``fresh_terminal_record`` dropped, so it reaches :func:`report_stale_evidence`
    through the ordinary ``on_fate`` collector. It decides nothing: its dedup
    key ``(terminal, ended_at)`` equals the one ``resolve_fate`` builds for the
    same record, so the two paths never double-report.
    """
    # Runtime import: ``worker_fate`` re-exports this module, so a top-level
    # import would cycle; by call time it is fully loaded.
    from .worker_fate import (
        Crashed,
        EvidenceSource,
        FateBasis,
        StaleEvidence,
        StaleReason,
    )

    stale = StaleEvidence(
        source=EvidenceSource.TERMINAL,
        reason=StaleReason.OLDER_THAN_DISPATCH,
        written_at=ended_at,
        evidence_head=None,
        live_head=None,
    )
    basis = FateBasis(
        issue_number=issue_number,
        pid_alive=False,
        outcome=None,
        exit_code=None,
        stale=(stale,),
        rule="R1-stale-terminal",
        dispatched_at=dispatched_at,
    )
    return Crashed(basis=basis, failure=None)


def collect_fate(into: dict[int, list[WorkerFate]], fate: WorkerFate) -> None:
    """Accumulate ``fate`` under its issue -- the ``on_fate`` collector body.

    Keeps EVERY fate a pass resolves for an issue, never just the last: the
    with-PR sweep resolves the same outcome twice (once against the live PR
    head, once without it), and the first carries ``HEAD_MISMATCH`` evidence
    the second cannot (B6 residue). :func:`report_stale_evidence` merges them.
    """
    into.setdefault(fate.basis.issue_number, []).append(fate)


def report_stale_evidence(
    state_file: Path,
    fates: Mapping[int, Sequence[WorkerFate]],
    *,
    dry_run: bool,
) -> None:
    """Emit ``worker_evidence_stale`` for every fate's not-yet-reported stale
    evidence, then persist the dedup marker.

    The one reporting path for every ``resolve_fate`` consumer (B6,
    wf-r2-s6): the orphan sweep's precompute, the live-handoff lane, the
    dispatch-time phantom-worker lane and the rework-outcome readers all hand
    their fates here instead of each dropping ``basis.stale`` on the floor.

    ``dry_run`` is required (keyword-only) and makes this a no-op: the events
    ring and the dedup marker are local writes a dry run must not make -- and a
    persisted marker would suppress the real event on the next live pass.
    Taking it here means no caller can forget the gate.

    Several fates for one issue are merged: their stale evidence is unioned and
    deduped by :func:`stale_evidence_key`, first fate's wording winning.

    Never call this while holding ``state_lock`` -- it takes the lock itself
    (not reentrant) and emits outside it (CLAUDE.md instrumentation
    invariant). Callers already inside a lock collect their fates and report
    after releasing it. Dedup is per ``(source, written_at)`` against the
    entry's ``stale_evidence_reported`` marker (design doc §5), so re-sweeping
    the same dead worker every pass emits once; an issue with no state entry
    still emits (there is nothing to dedup against) but persists no marker.
    """
    if dry_run or not fates:
        return
    snapshot = load_state_locked(state_file)
    pending: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for issue_number, issue_fates in fates.items():
        entry = snapshot.get("issues", {}).get(str(issue_number))
        entry_map: dict[str, Any] = dict(entry) if isinstance(entry, dict) else {}
        reported = set(entry_map.get("stale_evidence_reported") or ())
        events: list[tuple[str, dict[str, Any]]] = []
        for fate in issue_fates:
            entry_map["stale_evidence_reported"] = sorted(reported)
            events.extend(stale_evidence_events(entry_map, fate))
            reported.update(stale_evidence_key(s) for s in fate.basis.stale)
        if events:
            pending[issue_number] = events
    if not pending:
        return
    for events in pending.values():
        for kind, payload in events:
            log_event(
                state_file,
                kind,  # event-consumer: audit-only -- always ``worker_evidence_stale`` (the one kind ``stale_evidence_events`` builds); an operator-visibility warning that rule 1 ignored a leftover outcome, with no state mutation keyed off it
                payload,
                level="warning",
            )
    with state_lock(state_file):
        locked_state = load_state(state_file)
        for issue_number in pending:
            locked_entry = locked_state.get("issues", {}).get(str(issue_number))
            if not isinstance(locked_entry, dict):
                continue
            already = set(locked_entry.get("stale_evidence_reported") or ())
            for fate in fates[issue_number]:
                already.update(stale_evidence_key(s) for s in fate.basis.stale)
            locked_entry["stale_evidence_reported"] = sorted(already)
        save_state(state_file, locked_state)
