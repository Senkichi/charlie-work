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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import host as _host
from .state import load_state_locked


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


def count_live_sessions(directory: Path, state_file: Path | None, kind: LiveSessionKind) -> int:
    """Count live sessions under ``directory``, corroborated against ``state_file``.

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
    can launch past the configured cap (issue #343).
    """
    from .worker import iter_workers

    live_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(directory):
        if w.is_alive():
            live_count += 1
            live_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
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

    return live_count
