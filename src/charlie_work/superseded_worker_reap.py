"""Superseded-worker reap at the rework launch trigger (issue #1494).

A rework dispatch supersedes the issue's previously dispatched worker, but
every producer of ``rework_requested`` — the janitor-gate/worktree-rescue
stall path, the verdict reconcile, the stranded-commit restore — flips the
issue status without checking that worker's liveness, and
``worktree._check_worktree_writer_marker`` deliberately exempts a marker
owned by one of our own live sessions. #1337 ran two workers against one
worktree this way. The launch trigger is the enforcement point: before a
replacement worker dispatches, every recorded worker pid for the issue is
reaped via the fingerprinted ``write_gate.kill_process_tree``.

This module was split out of ``dead_worker_reap.py`` during the PR #1926
rework review: that module is a verbatim-move extraction family (issue
#1317) whose member set and cap band are pinned by
``tests/test_dead_worker_reap_split.py``, so a new member written after the
move lives here rather than growing an already over-cap file.

Members:

- ``_reap_superseded_workers`` -- collect every recorded worker pid for an
  issue (session sidecars plus the state ``worker_pid`` /
  ``last_known_worker_pid`` fallbacks), verify liveness against the
  recorded ``process_start_time`` fingerprint, kill each live worker's
  process tree, and sweep orphaned children in its worktree. Returns the
  pids still alive after the reap attempt.
- ``_reap_superseded_workers_for_launch`` -- the launch-lane wrapper shared
  by ``_dispatch_rework_impl`` (remote rework) and
  ``_local_dispatch_rework`` (local/no-remote rework): runs the reap over a
  batch of ``SessionRequest``\\ s and partitions them into launchable
  requests and synthetic ``PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND`` blocked
  results, so the two lanes share a single enforcement point that cannot
  drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .adapters import SessionDispatchResult, SessionRequest
from .config import PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND
from .orphan_sweep import sweep_orphan_processes
from .process_utils import is_pid_alive
from .worktree import worktree_path_for_branch
from .write_gate import WriteGate, require_write_gate


def _reap_superseded_workers(
    issue_number: int,
    issue_entry: dict[str, Any],
    sessions_dir: Path,
    *,
    worktree_path: Path | None = None,
    write_gate: WriteGate,
) -> list[int]:
    """Kill every still-live worker process recorded for ``issue_number``.

    Issue #1494: every producer of ``rework_requested`` — the
    janitor-gate/worktree-rescue stall path this issue covers, the verdict
    reconcile, the stranded-commit restore — flips the issue status without
    checking the prior worker's liveness, and
    ``worktree._check_worktree_writer_marker`` deliberately exempts a marker
    owned by one of our own live sessions ("dispatch/rework state guards
    prevent double dispatch"). When that state guard is the thing that
    leaked, a still-running prior worker admits its replacement into the
    same worktree — #1337 ran two workers against one worktree this way.
    The rework launch trigger is the enforcement point: before a
    replacement worker dispatches, every recorded worker pid for the issue
    is reaped via the same fingerprinted ``write_gate.kill_process_tree``
    the stall detector uses.

    Candidate pids come from both records of the issue's prior workers:

    * the session sidecars in ``sessions_dir`` — covers an earlier worker
      launched under a different adapter (e.g. a devin-shell worker
      superseded by a rescue-tier claude-code worker are distinct sidecar
      files, so state.json's single ``worker_pid`` sees only the newest);
    * the issue entry's ``worker_pid``/``worker_process_start_time``,
      falling back to ``last_known_worker_*`` — the same precedence
      ``_probe_recovery_liveness`` uses — covering a prior worker whose
      sidecar was already reaped.

    The PID-recycling defense mirrors ``_reap_idle_foreign_writer``: a
    fingerprint mismatch means the recorded worker is already gone (the pid
    was recycled) and is not a survivor; a *live* pid carrying no
    ``process_start_time`` fingerprint at all cannot be reaped safely —
    ``kill_process_tree`` would have nothing to re-verify against — so it
    blocks the launch instead of killing on bare pid liveness.

    Each reaped tree is followed by an orphan sweep of the worker's
    recorded worktree and a ``superseded_worker_reaped`` audit event; a pid
    that survives the kill attempt (or a live pid with no fingerprint) is
    returned and emits ``superseded_worker_reap_failed``.

    Returns the pids still alive after the reap attempt. An empty list
    means no recorded prior worker is running and the replacement may
    launch; a non-empty list means the caller must not launch.
    """
    write_gate = require_write_gate(write_gate)
    from .worker import iter_workers

    candidates: dict[int, dict[str, Any]] = {}
    for w in iter_workers(sessions_dir):
        # ``error is not None`` mirrors the stall lane's skip of
        # launch-failure records: an errored sidecar's pid field can carry
        # a *diagnostic* pid (e.g. the foreign/live worker a failed launch
        # reported), not this issue's own worker.
        if w.issue_number != issue_number or w.error is not None:
            continue
        if not isinstance(w.pid, int) or w.pid <= 0:
            continue
        candidates[w.pid] = {
            "process_start_time": w.process_start_time,
            "session_id": w.session_id,
            "adapter_kind": w.adapter_kind,
            "worktree_path": w.worktree_path,
            "source": "sidecar",
        }

    for pid_key, start_key in (
        ("worker_pid", "worker_process_start_time"),
        ("last_known_worker_pid", "last_known_worker_process_start_time"),
    ):
        raw_pid = issue_entry.get(pid_key)
        try:
            pid = int(raw_pid) if raw_pid is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid is None or pid <= 0:
            continue
        if pid in candidates:
            # The sidecar and state can disagree on the fingerprint's
            # availability: prefer a recorded fingerprint over none so a
            # reaper-eligible worker is not failed closed just because the
            # sidecar predates process_start_time capture.
            if candidates[pid]["process_start_time"] is None:
                candidates[pid]["process_start_time"] = issue_entry.get(start_key)
            continue
        candidates[pid] = {
            "process_start_time": issue_entry.get(start_key),
            "session_id": None,
            "adapter_kind": None,
            "worktree_path": str(worktree_path) if worktree_path is not None else "",
            "source": "state",
        }

    survivors: list[int] = []
    for pid, candidate in candidates.items():
        raw_start = candidate["process_start_time"]
        start_time = raw_start if isinstance(raw_start, (int, float)) else None
        if not is_pid_alive(pid, start_time):
            # Dead already, or the pid was recycled and its fingerprint no
            # longer matches — the recorded worker is gone either way.
            continue
        if start_time is None:
            # Live but unfingerprinted: kill_process_tree would have no
            # identity to re-verify, so this pid cannot be reaped without
            # risking an unrelated process that now holds the pid — the
            # same refusal _reap_idle_foreign_writer applies to
            # fingerprint-less markers. Fail closed: block the launch.
            # ``kind=`` by keyword: the event-kind scanner reads positional
            # arg 1 as the kind (the ``log_event(state_path, kind, ...)``
            # shape), so a positional kind here would leave the payload dict
            # unresolved.
            write_gate.log_event(
                kind="superseded_worker_reap_failed",
                payload={
                    "issue_number": issue_number,
                    "pid": pid,
                    "session_id": candidate["session_id"],
                    "adapter_kind": candidate["adapter_kind"],
                    "source": candidate["source"],
                    "worktree_path": candidate["worktree_path"],
                    "reason": "live prior worker has no process_start_time fingerprint",
                },
            )
            survivors.append(pid)
            continue
        killed_pids = write_gate.kill_process_tree(pid, start_time)
        if pid not in killed_pids and is_pid_alive(pid, start_time):
            # The kill was refused (identity re-verification failed) or the
            # process survived it — gate launch on confirmed death, the same
            # "plan is a snapshot" re-check _reap_idle_foreign_writer does.
            write_gate.log_event(
                kind="superseded_worker_reap_failed",
                payload={
                    "issue_number": issue_number,
                    "pid": pid,
                    "process_start_time": start_time,
                    "session_id": candidate["session_id"],
                    "adapter_kind": candidate["adapter_kind"],
                    "source": candidate["source"],
                    "worktree_path": candidate["worktree_path"],
                    "killed_pids": killed_pids,
                    "reason": "kill_process_tree did not terminate the process",
                },
            )
            survivors.append(pid)
            continue
        # Sweep detached/daemonized children the tree kill can leave
        # behind — the same sweep the stall lane runs after each reap.
        orphan_pids: list[int] = []
        sweep_path = candidate["worktree_path"] or (
            str(worktree_path) if worktree_path is not None else ""
        )
        if sweep_path:
            orphan_processes = sweep_orphan_processes(sweep_path)
            for orphan in orphan_processes:
                write_gate.kill_process(orphan["pid"])
                killed_pids.append(orphan["pid"])
            orphan_pids = [o["pid"] for o in orphan_processes]
        write_gate.log_event(
            kind="superseded_worker_reaped",
            payload={
                "issue_number": issue_number,
                "pid": pid,
                "process_start_time": start_time,
                "session_id": candidate["session_id"],
                "adapter_kind": candidate["adapter_kind"],
                "source": candidate["source"],
                "worktree_path": candidate["worktree_path"],
                "killed_pids": killed_pids,
                "orphan_pids": orphan_pids if orphan_pids else None,
            },
        )
    return survivors


def _reap_superseded_workers_for_launch(
    requests: Sequence[SessionRequest],
    issues: Mapping[str, Any],
    sessions_dir: Path,
    *,
    repo_root: Path,
    worktrees_dir: Path,
    adapter_label: Callable[[SessionRequest], str],
    write_gate: WriteGate,
) -> tuple[list[SessionRequest], list[SessionDispatchResult]]:
    """Reap superseded prior workers for a batch of rework launch requests.

    Issue #1494: the single enforcement point shared by the remote
    (``_dispatch_rework_impl``) and local (``_local_dispatch_rework``)
    rework lanes so the reap-then-block behavior cannot drift between them.
    For each request, every recorded prior-worker pid for the issue is
    reaped; a pid that survives (or is live with no start-time fingerprint
    to kill against) produces a synthetic pre-launch
    ``SessionDispatchResult`` carrying
    ``PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND`` — a member of
    ``PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS``, so the refusal accrues
    to ``blocked_environment_at`` rather than consuming the redispatch cap
    and the claim releases back to ``rework_requested``.

    ``issues`` is the caller's state snapshot's ``issues`` mapping (used to
    cover a prior worker whose sidecar was already reaped);
    ``adapter_label`` resolves the adapter label recorded on a blocked
    result so each lane reports the adapter that *would* have launched.

    Returns ``(launchable_requests, blocked_results)``: requests with no
    surviving prior worker, and one blocked result per refused launch.
    """
    launchable: list[SessionRequest] = []
    blocked: list[SessionDispatchResult] = []
    for request in requests:
        issue_entry = issues.get(str(request.issue_number), {})
        surviving = _reap_superseded_workers(
            request.issue_number,
            issue_entry if isinstance(issue_entry, dict) else {},
            sessions_dir,
            worktree_path=worktree_path_for_branch(repo_root, request.branch_name, worktrees_dir),
            write_gate=write_gate,
        )
        if surviving:
            blocked.append(
                SessionDispatchResult(
                    issue_number=request.issue_number,
                    issue_title=request.issue_title,
                    prompt_path=str(request.prompt_path),
                    branch_name=request.branch_name,
                    adapter=adapter_label(request),
                    ok=False,
                    error=(
                        "prior worker still alive after reap "
                        f"(pid(s): {surviving}); refusing to launch a "
                        "second worker into the same worktree"
                    ),
                    failure_kind=PRIOR_WORKER_STILL_ALIVE_FAILURE_KIND,
                    pid=surviving[0],
                )
            )
            continue
        launchable.append(request)
    return launchable, blocked
