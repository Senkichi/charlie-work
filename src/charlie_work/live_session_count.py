"""The single live-session counter shared by the worker and review lanes (issue #2230).

Both lanes count sidecar-alive processes (``worker.iter_workers`` +
``WorkerView.is_alive``, the recycling-safe pid/start-time probe), then --
when a ``state_file`` is given -- corroborate the sidecar count against
state.json's own dispatch records so a session whose sidecar went missing
for a still-live process (a "ghost", issue #343) still counts against the
cap instead of reading as free capacity. The lanes differ only in which
state map, status predicate and pid fields the corroboration consults;
``LiveSessionKind`` carries those differences as data.

Consumers reach this module directly: the ``host.sessions`` Real resolves
``count_live_sessions`` (per lane) at call time, the fleet-wide walkers in
``fleet_registry`` pass the lane constant themselves, and
``orchestration.dispatch_state`` / ``dead_worker_sweep.ports`` call
``_ghost_pid_alive`` for state.json pid probes. The thin lane delegates and
the ``workflow.*`` re-exports that used to front this module were deleted in
issue #2235 once their patch surfaces migrated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import host as _host
from . import state as _state
from .state import load_state_locked

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSessionKind:
    """The lane-specific state.json fields a ghost corroboration consults.

    Worker lane: ``issues`` entries whose ``status`` is ``"dispatched"``
    carry ``worker_pid`` / ``worker_process_start_time``. Review lane:
    ``prs`` entries whose ``review_dispatch_status`` is
    ``"review_dispatch_dispatched"`` carry ``reviewer_pid`` /
    ``reviewer_process_start_time``. The remaining fields only steer the
    ``[reconcile]`` line printed for each counted ghost.
    """

    state_map: str
    status_field: str
    status_value: str
    pid_field: str
    start_time_field: str
    subject_label: str  # "issue" / "PR" in the [reconcile] line
    sidecar_label: str  # "session" / "review" in the [reconcile] line
    cap_label: str  # the cap the ghost is counted against


WORKER_LANE = LiveSessionKind(
    state_map="issues",
    status_field="status",
    status_value="dispatched",
    pid_field="worker_pid",
    start_time_field="worker_process_start_time",
    subject_label="issue",
    sidecar_label="session",
    cap_label="the concurrency governor",
)

REVIEW_LANE = LiveSessionKind(
    state_map="prs",
    status_field="review_dispatch_status",
    status_value="review_dispatch_dispatched",
    pid_field="reviewer_pid",
    start_time_field="reviewer_process_start_time",
    subject_label="PR",
    sidecar_label="review",
    cap_label="the local review cap",
)


def read_fleet_census_state(state_file: Path, name_with_owner: str) -> dict[str, Any] | None:
    """One registered repo's ``state.json`` for the fleet-wide live census.

    Always a lock-free ``state.read_state_snapshot`` (issue #3141): the census
    runs under the fleet launch lock, so it must not queue on another repo's
    state lock -- a slow holder there used to freeze every other lane's
    launch gate for the whole hold. ``read_state_snapshot`` also never
    quarantines/renames the file, so a census cannot damage the state it only
    came to count from (issue #310).

    ``None`` means the file could not be read or parsed. The caller must mark
    the repo skipped and count only its live sidecars (never silently count 0
    ghosts); this helper leaves the loud signal -- a warning log plus a
    ``fleet_census_state_unreadable`` event in that repo's events.db.
    """
    snapshot = _state.read_state_snapshot(state_file)
    if snapshot is None:
        logger.warning(
            "Skipping ghost corroboration for %s in the fleet live-count: "
            "state file %s is unreadable -- counting live sidecars only",
            name_with_owner,
            state_file,
        )
        try:
            from .instrumentation import log_event

            # write-gate-exempt(issue=3141): no write_gate param; sibling raw calls remain
            log_event(
                state_file,
                "fleet_census_state_unreadable",
                {"repo": name_with_owner, "state_file": str(state_file)},
                repo=name_with_owner,
                level="warning",
            )
        except Exception:  # noqa: BLE001 — instrumentation never breaks the census
            logger.debug("Failed to record fleet_census_state_unreadable for %s", name_with_owner)
    return snapshot


def _ghost_pid_alive(entry: dict[str, Any], kind: LiveSessionKind) -> bool:
    """pid + start-time liveness for a state.json dispatch record.

    Shared body of the worker- and reviewer-lane state.json pid probes
    (``dispatch_selection._reviewer_pid_alive`` wraps it for REVIEW_LANE): a
    missing pid is dead; otherwise the host liveness probe decides (live pid
    whose recorded start time matches -- recycling-safe; indeterminate
    probes fail open per ``process_utils.is_pid_alive``).
    """
    pid = entry.get(kind.pid_field)
    if pid is None:
        return False
    return _host.current().probe.is_alive(pid, entry.get(kind.start_time_field))


def count_live_sessions_with_adapter_kinds(
    directory: Path,
    state_file: Path | None,
    kind: LiveSessionKind,
    *,
    state_snapshot: dict[str, Any] | None = None,
) -> tuple[int, dict[str, int]]:
    """One ``iter_workers`` pass returning the corroborated total AND the per-kind counts.

    The single-pass body behind :func:`count_live_sessions` (which returns the
    total) and the fleet governor's also-needed per-``adapter_kind`` breakdown:
    one sidecar walk feeds both results, then
    the ghost corroboration adds to the total only (issue #2764 -- a caller
    that needs both numbers, the fleet governor, walks the sessions dir once).

    Sidecar pass: every ``iter_workers`` record whose pid probe is alive
    counts once, keyed on ``issue_number`` so the corroboration pass never
    double-counts a session that has both a live sidecar and a dispatch
    record.

    Corroboration pass (only when ``state_file`` is given): each
    ``kind.state_map`` entry still marked dispatched whose recorded pid is
    alive but has no live sidecar counts too, and is printed as a loud
    reconcile signal rather than silently treated as free capacity. Without
    it, a reap lane that strands state.json's dispatch record (or any other
    path that loses the sidecar of a still-live process) makes a running
    worker/reviewer invisible to concurrency accounting and the next pass
    can launch past the configured cap (issue #343). A ghost has no sidecar
    and therefore no known adapter, so it never enters the per-kind counts.

    ``state_snapshot``: an already-read state dict to corroborate against,
    used verbatim -- no state read happens at all. The fleet census passes
    the ``read_state_snapshot`` result (lock-free: the census walks other
    repos' state under the fleet launch lock and must not queue on their
    state locks, issue #3141) and passes ``state_file=None`` when the
    snapshot came back ``None`` (unreadable), which skips the corroboration
    pass entirely -- the repo is then marked skipped and counts live
    sidecars only, never 0 ghosts silently. Callers that pass only
    ``state_file`` keep the locked read: their count gates a mutation of
    that same state.
    """
    from .worker import iter_workers

    live_numbers: set[int] = set()
    live_count = 0
    counts: dict[str, int] = {}
    for w in iter_workers(directory):
        if w.is_alive():
            live_count += 1
            live_numbers.add(w.issue_number)
            counts[w.adapter_kind] = counts.get(w.adapter_kind, 0) + 1

    state: dict[str, Any] | None = state_snapshot
    if state is None and state_file is not None:
        state = load_state_locked(state_file)
    if state is not None:
        for number_str, entry in state.get(kind.state_map, {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get(kind.status_field) != kind.status_value:
                continue
            try:
                number = int(number_str)
            except (TypeError, ValueError):
                continue
            if number in live_numbers:
                continue
            if _ghost_pid_alive(entry, kind):
                live_count += 1
                print(
                    f"[reconcile] {kind.subject_label} {number}: {kind.pid_field} "
                    f"{entry.get(kind.pid_field)} is alive with no live "
                    f"{kind.sidecar_label} sidecar (ghost) -- counting it against "
                    f"{kind.cap_label} instead of treating the slot as free",
                    flush=True,
                )

    return live_count, counts


def count_live_sessions(
    directory: Path,
    state_file: Path | None,
    kind: LiveSessionKind,
    *,
    state_snapshot: dict[str, Any] | None = None,
) -> int:
    """Count live sessions under ``directory``, corroborated against ``state_file``.

    Thin wrapper over :func:`count_live_sessions_with_adapter_kinds` (issue
    #2764): the total half of its single sidecar pass, unchanged in behavior.
    ``state_snapshot`` is forwarded verbatim (issue #3141).
    """
    return count_live_sessions_with_adapter_kinds(
        directory, state_file, kind, state_snapshot=state_snapshot
    )[0]
