"""Session-liveness and death bookkeeping helpers for the dead-worker sweep.

Moved verbatim from the retired ``dead_worker_reap`` module: startup-death
classification, the ``session_failed_relabeled`` event payload, live-session counts,
the orphan-process sweep for dead sessions and the worker census.
"""

from __future__ import annotations


import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import (
    OrchestratorConfig,
)
from ..orphan_sweep import sweep_orphan_processes
from .. import host as _host
from ..state import (
    load_state,
    load_state_locked,
    state_lock,
)
from ..worker import WorkerHealth, WorkerView
from ..write_gate import WriteGate, require_write_gate


# Issue #1106: a rework session that dies at CLI startup (before the worker's
# first tool action) is not a no-op/conflict rework attempt — the cap counters
# should only count sessions that *ran* and produced no useful change.  A
# ``launch_failed`` (process never launched) is always a startup death; a
# ``stalled`` session that died within this threshold is also a startup death
# (CLI error, nonzero exit within seconds, empty diff AND empty transcript).
# The threshold is deliberately generous: a genuine stall (worker ran for
# minutes but got stuck) must NOT be misclassified as a startup death, so the
# bound is set above the worst-case CLI startup time but well below the
# shortest genuine-work session.
#
# The ``runtime_seconds`` passed to ``_is_startup_death`` must be bounded by
# the worker's *actual process runtime* (time from start to death), NOT by the
# elapsed time until the orchestrator's classification pass runs.  Using
# ``WorkerView.runtime_seconds()`` (which is ``now - started_at``) would let
# ordinary polling latency — the gap between when the CLI died and when the
# reaper pass classifies it — silently push a 5-second startup death past the
# 60-second threshold and defeat the exemption.  ``_worker_death_bounded_runtime_seconds``
# derives the runtime from the log file's last-modified time (frozen at death
# for a dead process) instead.
STARTUP_DEATH_THRESHOLD_SECONDS: int = 60


def _is_startup_death(failure_kind: str | None, runtime_seconds: float) -> bool:
    """Classify whether a dead rework session died at CLI startup.

    Returns True when the session never reached the worker's first tool
    action — the cap counters in ``_route_janitor_gate_failure_to_rework``
    must not count these as no-op/conflict rework attempts (issue #1106).

    ``runtime_seconds`` must be the worker's *death-bounded* runtime (time
    from ``started_at`` to the last real log activity / death), as computed
    by ``_worker_death_bounded_runtime_seconds`` — NOT
    ``WorkerView.runtime_seconds()``, which measures elapsed time until
    classification and is polluted by polling latency.
    """
    if failure_kind is None:
        return False
    # ``launch_failed``: the process never launched at all.
    if failure_kind == "launch_failed":
        return True
    # ``stalled`` with a very short runtime: the CLI exited before the
    # worker did any real work (e.g. "Refusing to run in an untrusted
    # workspace").  A longer runtime means the worker genuinely ran and
    # got stuck — that IS a no-op rework attempt the cap should count.
    if failure_kind == "stalled" and runtime_seconds < STARTUP_DEATH_THRESHOLD_SECONDS:
        return True
    return False


def _worker_death_bounded_runtime_seconds(worker: WorkerView) -> float:
    """Return the worker's runtime bounded by actual process death.

    This is the signal ``_is_startup_death`` must use instead of
    ``WorkerView.runtime_seconds()`` (which is ``now - started_at`` and
    measures elapsed time until the *classification pass*, not until death).
    A CLI that dies at 5 seconds but is not classified until 300 seconds
    later must still be recognized as a 5-second startup death, not a
    300-second stall.

    The death-bounded runtime is derived from the log file's last-modified
    time: once the CLI process exits, the log stops being written and its
    mtime freezes at the death moment.  A fresh ``stat()`` of the log file
    is the most accurate signal; the sidecar's recorded ``last_activity_at``
    (updated each pass by ``update_worker_log_stat``) is the fallback when
    the log file is gone.  When neither is available — the CLI never wrote
    anything, e.g. a ``launch_failed`` that still got a PID — the runtime is
    0.0, which is a startup death by construction.
    """
    from datetime import UTC, datetime

    try:
        started_at = datetime.fromisoformat(worker.started_at)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return 0.0

    # Prefer a fresh stat of the log file — its mtime is frozen at death for
    # a dead process, so this is the tightest death-bounded signal.
    death_ts: float | None = None
    log_stat = worker.log_stat()
    if log_stat is not None:
        death_ts = log_stat.st_mtime
    if death_ts is None and worker.last_activity_at is not None:
        # Fall back to the sidecar's recorded last-activity timestamp.
        from ..worker import _iso_to_timestamp

        death_ts = _iso_to_timestamp(worker.last_activity_at)
    if death_ts is None:
        # No log activity was ever recorded — the CLI never wrote anything,
        # which is a startup death by construction (runtime 0.0).
        return 0.0
    return max(0.0, death_ts - started_at.timestamp())


def _session_failed_relabeled_payload(
    *,
    issue_number: int,
    reason: str,
    failure_kind: str | None = None,
    removed_labels: list[str] | tuple[str, ...] = (),
    added_ready: bool = False,
    label_write_ok: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build the payload for a ``session_failed_relabeled`` event.

    Mirrors ``_escalate_issue`` (#750): ``reason`` is keyword-only and
    required, so a relabel event can never be emitted without saying *why*
    it fired. ``failure_kind`` is an optional refinement (the classifier's
    verdict, which may be ``None`` when classification could not determine
    a kind); when present it is recorded alongside ``reason``, but
    ``reason`` is the canonical field a reader filters on.

    Every prior call site spelled "why" under a different key --
    ``reason`` (orphan sweep), ``failure_kind`` only (dead-worker
    no-open-PR), both (phantom live worker), or English prose in
    ``detail`` (reconcile) -- so a query on any one key silently missed
    the rows written by the others. Routing all sites through this
    builder makes the omission unrepresentable: there is no way to call
    it without a ``reason``. Issue #978.
    """
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "reason": reason,
        "removed_labels": sorted(removed_labels),
        "added_ready": added_ready,
        "label_write_ok": label_write_ok,
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind
    if extra:
        payload.update(extra)
    return payload


def _emit_session_failed_relabeled(
    state: dict[str, Any],
    *,
    issue_number: int,
    reason: str,
    failure_kind: str | None = None,
    removed_labels: list[str] | tuple[str, ...] = (),
    added_ready: bool = False,
    label_write_ok: bool = True,
    state_path: Path | None = None,
    write_gate: WriteGate,
    **extra: Any,
) -> dict[str, Any]:
    """Emit a ``session_failed_relabeled`` event via the shared payload builder.

    Thin wrapper around :func:`_session_failed_relabeled_payload` plus
    :func:`append_event`; see the payload builder's docstring for the
    required-``reason`` invariant (issue #978).

    Issue #1264 (W6 PR3): ``write_gate`` is declared explicitly (not folded
    into ``**extra``) so a caller that forgets it gets ``require_write_gate``'s
    loud ``TypeError`` instead of the gate silently landing in the event
    payload. ``state_path`` is consequently now vestigial (the gate
    auto-binds its own ``state_path``) but kept as a parameter rather than
    removed, the same call PR2 made for the analogous
    ``_merge_on_write_save`` seam (adversarial-review finding F3), to avoid
    churning callers for no behavioral gain.
    """
    write_gate = require_write_gate(write_gate)
    return write_gate.append_event(
        state,
        "session_failed_relabeled",
        _session_failed_relabeled_payload(
            issue_number=issue_number,
            reason=reason,
            failure_kind=failure_kind,
            removed_labels=removed_labels,
            added_ready=added_ready,
            label_write_ok=label_write_ok,
            **extra,
        ),
    )


def _count_live_sessions(sessions_dir: Path, state_file: Path | None = None) -> int:
    """Count the number of currently alive worker sessions across both adapters.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the total count of sessions with alive PIDs.

    When ``state_file`` is given, this also corroborates the sidecar-based
    count against state.json's own dispatched-issue ``worker_pid``/
    ``worker_process_start_time`` records (issue #343). A sidecar can go
    missing for a still-live process -- a "ghost" -- if a reap lane removes
    it on ambiguous evidence (see ``_classify_dead_sessions_and_update_
    throttle_state``'s corroboration gate) or through any other path that
    strands state.json's dispatch record. Because the governor's dispatch
    cap (``_apply_concurrency_governor``) is built on top of this count, a
    ghost previously made a live process invisible to concurrency accounting
    and let the next pass over-dispatch past the configured cap. state.json's
    worker_pid fields are not touched by sidecar reaping, so any issue still
    recorded as ``dispatched`` whose worker_pid is alive (pid + start-time,
    recycling-safe) but has no corresponding live sidecar is counted here
    too, and printed as a loud reconcile signal rather than silently treated
    as free capacity.
    """
    from ..worker import iter_workers

    live_issue_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(sessions_dir):
        if w.is_alive():
            live_count += 1
            live_issue_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
        for issue_number_str, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict) or entry.get("status") != "dispatched":
                continue
            try:
                issue_number = int(issue_number_str)
            except (TypeError, ValueError):
                continue
            if issue_number in live_issue_numbers:
                continue
            if _worker_pid_alive(entry):
                live_count += 1
                print(
                    f"[reconcile] issue {issue_number}: worker_pid "
                    f"{entry.get('worker_pid')} is alive with no live session "
                    "sidecar (ghost) -- counting it against the concurrency "
                    "governor instead of treating the slot as free",
                    flush=True,
                )

    return live_count


def _detect_stalled_sessions(
    sessions_dir: Path, config: OrchestratorConfig, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Detect stalled sessions (live PID but dead agent) without handling them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. This is a read-only detection function for status/roll-call.

    ``now`` (issue #828) is the injectable clock, following the convention
    established by PR #827 / ``get_rate_limit_defer_until``: defaults to
    ``datetime.now(UTC)`` when omitted, so production behavior is
    byte-identical.

    Returns a list of {issue, pid, health, terminal_tool, terminal_reason}
    dicts for stalled sessions. ``health`` distinguishes STALLED from DEAD
    (issue #261) so digest callers can surface dead-worker terminal cause
    instead of collapsing everything to "STALLED". ``terminal_tool``/
    ``terminal_reason`` are populated only for DEAD entries with a matching
    post-mortem sidecar (best-effort — absent when extraction found nothing).
    """
    from datetime import UTC, datetime
    from ..post_mortem import read_post_mortem
    from ..worker import classify_worker_health, iter_workers, real_activity_probe_for

    if not config.watchdog.enabled:
        return []

    stalled_entries: list[dict[str, Any]] = []
    if now is None:
        now = datetime.now(UTC)

    for w in iter_workers(sessions_dir):
        if w.pid is None or w.error is not None:
            continue

        # Issues #280/#301: corroborate against real-session activity for read-only
        # detection too (status/dry-run/digest).
        probe = real_activity_probe_for(w, config, now)
        health = classify_worker_health(w, config, now, probe)

        # Both STALLED and DEAD are considered "stalled" for reporting purposes
        if health in (WorkerHealth.STALLED, WorkerHealth.DEAD):
            entry: dict[str, Any] = {
                "issue": w.issue_number,
                "pid": w.pid,
                "health": health.name,
            }
            if health is WorkerHealth.DEAD:
                post_mortem = read_post_mortem(sessions_dir, w.issue_number)
                if post_mortem is not None:
                    entry["terminal_tool"] = post_mortem.terminal_tool
                    entry["terminal_reason"] = post_mortem.terminal_reason
            stalled_entries.append(entry)

    return stalled_entries


# NOTE: the former `_kill_orphan_pid` free function that lived here was
# hoisted to `process_utils.kill_orphan_pid` verbatim (issue #1264, W6 PR3,
# R6a) so that `write_gate.py` can wrap it as `WriteGate.kill_process`
# without importing this module. The two call sites below
# (`_detect_and_handle_stalled_sessions`) are unconditional and out of this
# PR's scope -- see issue #1311's sibling filing for that function's own
# dry-run leak -- so they call the hoisted primitive raw, unchanged in
# behavior. The one in-scope call site (`_sweep_orphan_processes_for_dead_sessions`)
# now goes through `write_gate.kill_process` instead.


def _worker_pid_alive(entry: dict[str, Any]) -> bool:
    """Check if a worker PID from state.json is alive, with start-time verification.

    This helper deduplicates the PID liveness check used across dispatch and
    orphaned worker detection. It checks both PID liveness and process identity
    via start time to detect PID recycling.

    Args:
        entry: A state.json issue entry containing worker_pid and optionally
               worker_process_start_time.

    Returns:
        True if the worker PID is alive and the start time matches (if available),
        False otherwise.
    """
    worker_pid = entry.get("worker_pid")
    if worker_pid is None:
        return False

    return _host.current().probe.is_alive(worker_pid, entry.get("worker_process_start_time"))


def _orphan_head_fingerprint(remote_sha: str | None, local_sha: str | None) -> str:
    """Combine remote and local branch head SHAs into a single progress fingerprint.

    A change in either SHA indicates progress -- a remote push or a local
    (possibly stranded) commit. Used by the orphan-sweep redispatch cap
    (issue #1243) to distinguish a no-progress death loop (unchanged fingerprint
    across attempts) from a moving head that is the salvage path's job.
    """
    return f"{remote_sha or 'none'}:{local_sha or 'none'}"


# Issue #1153: minimum number of zero-artifact attempts (all ``ahead_of_main
# == 0``) in a post-mortem sidecar before the orphan sweep escalates instead
# of relabeling to ``automated-ready`` for another redispatch. The first
# zero-artifact attempt is the initial dispatch; the second is the first
# redispatch. Escalating before the *second* redispatch (the third attempt
# overall) means the threshold is 2: two attempts have already produced zero
# artifacts, so a third would almost certainly do the same.
_ZERO_ARTIFACT_ESCALATION_THRESHOLD = 2


def _is_zero_artifact_dispatch_loop(sessions_dir: Path, issue_number: int) -> bool:
    """Return True when prior dispatch attempts all produced zero artifacts.

    Issue #1153: an issue whose post-mortem sidecar records ``>= 2`` attempts
    where *every* attempt's ``ahead_of_main`` is ``0`` is in a zero-artifact
    dispatch loop -- each worker session ran, produced no commits ahead of
    the base ref, and was swept as a dead worker with no open PR. The
    post-mortem file already contains exactly the signal needed
    (``attempts[].ahead_of_main == 0`` repeated); this helper reads it so the
    orphan sweep can escalate to ``agent:human-needed`` instead of relabeling
    to ``automated-ready`` for yet another fruitless redispatch.

    Returns ``False`` when there is no sidecar, fewer than the threshold
    number of attempts, any attempt has a non-zero ``ahead_of_main``, or any
    attempt's ``ahead_of_main`` is ``None`` (unknown -- cannot confirm
    zero-artifact, so do not escalate on ambiguous evidence).
    """
    from ..post_mortem import read_post_mortem

    record = read_post_mortem(sessions_dir, issue_number)
    if record is None:
        return False
    if len(record.attempts) < _ZERO_ARTIFACT_ESCALATION_THRESHOLD:
        return False
    # Every attempt must have a confirmed ahead_of_main == 0. An attempt
    # with ahead_of_main == None is ambiguous (the count could not be
    # computed) -- do not escalate on ambiguous evidence.
    return all(attempt.ahead_of_main == 0 for attempt in record.attempts)


def _sweep_orphan_processes_for_dead_sessions(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
) -> None:
    """Sweep for orphan processes in worktrees of dead sessions.

    This is called from the production loop to detect and clean up orphaned
    processes that survived session kills (e.g., detached/daemonized processes
    like nohup-style background processes). This addresses issue #139.

    On Windows: Uses PowerShell Get-CimInstance Win32_Process to find processes
    whose CommandLine references the worktree path of dead sessions.
    On POSIX: Not implemented (returns empty list).

    Detected orphans are killed automatically and logged to state.json.

    Issue #1264 (W6 PR3): this is one of the three unconditional
    ``_loop_body`` call sites named by issue #1311's dry-run leak. Both the
    orphan kill and the event/state write below now go through
    ``write_gate`` -- the kill via the 6th gate method, ``kill_process``
    (R6a: wraps the ``_kill_orphan_pid`` primitive hoisted to
    ``process_utils.kill_orphan_pid``).
    """
    write_gate = require_write_gate(write_gate)
    from .. import worker_fate
    from ..devin_shell import read_session_records
    from ..claude_code import read_worker_records

    # Only run on Windows where the issue occurs
    if os.name != "nt":
        return

    # Collect worktree paths of dead sessions
    dead_worktree_paths: set[str] = set()

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not worker_fate.is_alive(record.pid, record.process_start_time):
            dead_worktree_paths.add(record.worktree_path)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not worker_fate.is_alive(record.pid, record.process_start_time):
            dead_worktree_paths.add(record.worktree_path)

    # Sweep for orphans in each dead worktree
    for worktree_path in dead_worktree_paths:
        orphan_processes = sweep_orphan_processes(worktree_path)
        if orphan_processes:
            # Kill detected orphans
            killed_orphans: list[int] = []
            for orphan in orphan_processes:
                write_gate.kill_process(orphan["pid"])
                killed_orphans.append(orphan["pid"])

            # Log the event with image/cmdline of each killed process so the
            # respawn source in dead worktrees can be identified and shut off.
            with state_lock(state_file):
                state = load_state(state_file)
                state = write_gate.append_event(
                    state,
                    "orphan_processes_killed",
                    {
                        "worktree_path": worktree_path,
                        "orphan_pids": [o["pid"] for o in orphan_processes],
                        "killed_orphans": killed_orphans,
                        "orphan_processes": [
                            {
                                "pid": o["pid"],
                                "name": o.get("name"),
                                "command_line": o.get("command_line"),
                            }
                            for o in orphan_processes
                        ],
                    },
                )
                write_gate.save_state(state)


def _log_worker_census(sessions_dir: Path) -> None:
    """Log one INFO line per loop pass listing every currently-alive worker.

    Issue #646: the launch-time log in claude_code.py/devin_shell.py answers
    "what cap did this worker launch with", but says nothing about how many
    are running *right now* -- the question a box-saturation incident needs
    answered ("how many suites were running at 11:33, from which worktrees,
    at what cap"). This sweep answers it directly from log content alone,
    with no process forensics required.

    Deliberately read-only (no state mutation): it carries none of the
    fragile sole-writer invariants _detect_and_handle_stalled_sessions/
    _sweep_orphan_processes_for_dead_sessions must protect. It also only
    ever iterates *alive* records, so — unlike a dead/exit-transition sweep —
    it has no "months of accumulated stale sidecar" flooding problem even if
    old sidecars are never pruned from sessions_dir.

    Called from the top of ``dispatch()`` (see its docstring) -- the one
    chokepoint every dispatch path funnels through, standalone (`work`/`fleet
    work`) or supervised (`loop()` -> `_loop_body()` -> `dispatch()`) -- so it
    runs unconditionally regardless of how long the orchestrator process
    itself lives: a one-shot ``charlie fleet work`` CLI invocation logs
    exactly one census line before exiting; a long-lived ``charlie fleet
    supervise`` logs one per pass. Both answer the diagnostic question above
    from log content alone -- no need to correlate against a live process
    list.
    """
    import logging

    logger = logging.getLogger(__name__)

    from .. import worker_fate
    from ..claude_code import read_worker_records
    from ..devin_shell import read_session_records

    now = datetime.now(UTC)

    def _age_seconds(started_at: str) -> int | None:
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return None
        return int((now - started).total_seconds())

    entries: list[str] = []
    # adapter_kind=None also covers the "api" adapter, which delegates to
    # claude_code.launch_claude_worker under the hood and shares its sidecar
    # schema (and therefore xdist_cap/liveness) unchanged.
    for record in read_worker_records(sessions_dir, adapter_kind=None):
        if (
            record.pid is None
            or record.error is not None
            or not worker_fate.is_alive(record.pid, record.process_start_time)
        ):
            continue
        entries.append(
            f"(adapter={record.adapter_kind} issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )
    for record in read_session_records(sessions_dir):
        if (
            record.pid is None
            or record.error is not None
            or not worker_fate.is_alive(record.pid, record.process_start_time)
        ):
            continue
        entries.append(
            f"(adapter=devin-shell issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )

    logger.info("worker census: n_alive=%d %s", len(entries), " ".join(entries) or "[]")


def _issues_with_live_workers(sessions_dir: Path) -> set[int]:
    """Return the set of issue numbers that have currently alive worker sessions.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness through the host session port.
    Returns the set of issue numbers with alive PIDs.
    """
    return _host.current().sessions.live_issue_numbers(sessions_dir)
