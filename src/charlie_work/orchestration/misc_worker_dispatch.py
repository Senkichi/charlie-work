"""Worker-dispatch and worktree-safety delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 batch 3 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
here unwrapped onto ``OrchestratorApp`` (``self`` binds via the descriptor
protocol exactly as the lexical methods did).

Workflow-defined names are reached through ``_wf.<name>`` so every existing
``charlie_work.workflow`` monkeypatch seam keeps landing: the ``CommandResult``
class and the ``_state_lock_busy_result`` helper. Every other free name is
imported directly from its defining module (none is patched on
``charlie_work.workflow``). ``dispatch_rework`` calls the L01-moved
``self._dispatch_rework_impl`` -- a cross-delegate call that resolves on the
class through the installer.
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from pathlib import Path
from typing import Any

from charlie_work.adapters import SessionRequest
from charlie_work.dead_worker_reap import _emit_session_failed_relabeled
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.github import label_names
from charlie_work.state import StateLockBusy
from charlie_work.worker import iter_workers
from charlie_work.worktree import (
    WorktreeProbeFailedError,
    _worktree_refuse_to_reset_reason,
    inspect_worktree_state,
    read_worker_outcome,
    worktree_path_for_branch,
)


def _route_phantom_live_worker(
    self,
    state: dict[str, Any],
    request: SessionRequest,
    full_issue: dict[str, Any],
    sessions_dir: Path,
) -> tuple[str, str | None, dict[str, Any]]:
    """Route a phantom ``live_worker_redispatch_averted`` result as dead.

    The adapter reported a live worker, but the recorded PID failed the
    OS-level liveness + identity check (issue #523). Remove the stale
    sidecar/marker so the session no longer occupies a concurrency slot,
    then strip active labels and restore ``automated-ready`` so the issue
    is dispatchable again, recording a single
    ``session_failed_relabeled`` attention event.

    This mirrors ``_classify_dead_sessions_and_update_throttle_state``'s
    no-open-PR relabel path. The open-PR/rework-routing case is
    intentionally NOT handled here: a dispatched request's issue can never
    have an open tracked PR, because ``_dispatch_impl`` builds
    ``pr_by_issue`` once and candidate selection excludes every issue in
    it (so ``request.issue_number`` is never in ``pr_by_issue``). A phantom
    worker whose issue later opens a PR is routed to rework by the
    dead-session reaper lane, not by dispatch.

    Issue #1122: before reaping, inspect the worktree using the same
    ``inspect_worktree_state`` single enforcement point the reaper lane
    uses (issue #252). If the worktree is COMPLETED or the worker reported
    a successful push (``.worker-outcome.json`` with
    ``push_succeeded=true``), do NOT reap the sidecar or strip labels --
    leave the sidecar untouched so the reaper lane
    (``_classify_dead_sessions_and_update_throttle_state``) can salvage the
    pushed branch on a subsequent pass. Reaping here would destroy the
    sidecar the reaper lane is keyed off, making salvage impossible and
    escalating review-ready pushed work to a human.

    Returns ``(status, dispatched_at, state)``. The status is
    ``"dispatch_failed"`` so the caller's entry-building frees the slot;
    ``dispatched_at`` is ``None`` because no worker was actually launched.
    """
    issue_number = request.issue_number

    # Issue #1122: inspect the worktree before reaping. A completed or
    # push-succeeded worktree must be left for the reaper lane's salvage
    # path (_attempt_salvage). Reaping the sidecar here would destroy the
    # key the reaper lane iterates over, making salvage impossible.
    # Issue #1130: relax the preserve condition from ``COMPLETED`` to
    # ``ahead_count > 0`` — a worktree with committed-but-unpushed work
    # that also has shim/scaffolding dirt (not in ``injected_paths``)
    # classifies as PARTIAL, not COMPLETED, but the committed work is
    # still salvageable. The reaper lane's salvage condition was relaxed
    # to match (see ``_classify_dead_sessions_and_update_throttle_state``).
    phantom_workers = [w for w in iter_workers(sessions_dir) if w.issue_number == issue_number]
    for w in phantom_workers:
        worktree_path = Path(w.worktree_path)
        inspection = inspect_worktree_state(
            worktree_path,
            self.config.dispatch.base_ref,
            self.config.dispatch.injected_paths,
            self.config.dispatch.materialize_dirs,
        )
        worker_outcome = read_worker_outcome(worktree_path)
        reported_push = (
            isinstance(worker_outcome, dict)
            and worker_outcome.get("push_succeeded") is True
            and worker_outcome.get("pr_created") is False
        )
        if inspection.ahead_count > 0 or reported_push:
            # Preserve the sidecar so the reaper lane can salvage. Do NOT
            # strip labels -- the issue should stay in its active state
            # until salvage moves it to pr_open, preventing re-dispatch
            # into the occupied worktree.
            state = _emit_session_failed_relabeled(
                state,
                issue_number=issue_number,
                reason="phantom_live_worker_completed_work_preserved",
                failure_kind="live_worker_redispatch_averted",
                removed_labels=[],
                added_ready=False,
                label_write_ok=True,
                worktree_state=inspection.state.value,
                reported_push=reported_push,
                state_path=self.paths.state_file,
                write_gate=self.write_gate,
            )
            return "dispatch_failed", None, state

    # Reap the stale sidecar and matching worktree writer marker. Reuse
    # WorkerView.reap_sidecar so the adapter-specific path derivation and
    # session-id-gated marker removal stay in one place.
    for w in phantom_workers:
        w.reap_sidecar(sessions_dir)

    # Strip active labels and ensure the ready label is present so the
    # issue becomes dispatchable. This mirrors
    # _classify_dead_sessions_and_update_throttle_state's no-open-PR
    # relabel path, gated on an active label actually being present so a
    # terminal-only issue is never given ``ready`` back spuriously.
    issue_labels = label_names(full_issue)
    active_labels = issue_labels & self.config.labels.active
    if not active_labels:
        state = _emit_session_failed_relabeled(
            state,
            issue_number=issue_number,
            reason="phantom_live_worker_pid_dead",
            failure_kind="live_worker_redispatch_averted",
            removed_labels=[],
            added_ready=False,
            label_write_ok=True,
            state_path=self.paths.state_file,
            write_gate=self.write_gate,
        )
        return "dispatch_failed", None, state

    needs_ready = self.config.labels.ready not in issue_labels
    label_write_ok = True
    for label in sorted(active_labels):
        if not self.gh.remove_issue_label(issue_number, label):
            label_write_ok = False
    if needs_ready:
        if not self.gh.add_issue_label(issue_number, self.config.labels.ready):
            label_write_ok = False
    state = _emit_session_failed_relabeled(
        state,
        issue_number=issue_number,
        reason="phantom_live_worker_pid_dead",
        failure_kind="live_worker_redispatch_averted",
        removed_labels=sorted(active_labels),
        added_ready=needs_ready,
        label_write_ok=label_write_ok,
        state_path=self.paths.state_file,
        write_gate=self.write_gate,
    )
    return "dispatch_failed", None, state


def _worktree_still_unsafe(self, issue_number: int, state: dict[str, Any]) -> str | None:
    """Re-run the worktree safety check for an issue (issue #849).

    Returns a reason string if the issue's worktree is still unsafe to
    reset, or ``None`` if the worktree is safe (or does not exist). Used
    by both de-escalation paths (``unescalate`` and
    ``_deescalate_mechanical_issue``) to refuse clearing a
    ``worktree_unsafe`` escalation while the underlying cause — a set of
    bytes on disk — still persists. Without this check, clearing the
    label reports success for an operation that changed nothing causal,
    and the next rework dispatch reproduces the escalation deterministically.

    Fails closed: a probe failure (``WorktreeProbeFailedError``) is
    treated as "still unsafe" so a transient index lock cannot clear a
    real blocker.
    """
    issue_entry = state.get("issues", {}).get(str(issue_number), {})
    if not isinstance(issue_entry, dict):
        return None
    branch = issue_entry.get("branch_name")
    if not branch or not isinstance(branch, str):
        return None
    wt_path = worktree_path_for_branch(self.repo_root, branch, self._layout.worktrees)
    if not wt_path.is_dir():
        # No worktree on disk — the blocker is gone (or was never this
        # issue's worktree). Clearing is safe.
        return None
    try:
        return _worktree_refuse_to_reset_reason(
            self.repo_root,
            branch,
            self.config.dispatch.base_ref,
            wt_path,
            self.config.dispatch.injected_paths,
            self.config.dispatch.materialize_dirs,
        )
    except (WorktreeProbeFailedError, RuntimeError):
        # Fail closed: a probe failure means we cannot confirm safety,
        # so treat the worktree as still unsafe.
        return "worktree safety probe failed; cannot confirm clean"


def dispatch_rework(
    self,
    limit: int | None = None,
    *,
    only_issues: str | None = None,
    stalled_entries: list[dict[str, int]] | None = None,
) -> _wf.CommandResult:
    """Dispatch rework workers for issues in needs-rework state with open PRs.

    ``stalled_entries``: pass the result of an already-completed
    ``_detect_and_handle_stalled_sessions`` sweep to skip re-running the
    sweep inside this call — same contract as ``dispatch()`` (issue #343
    Finding 2: the sweep writes Signal-1's inconclusive-probe deferral
    counter, so it must run at most once per pass). Standalone callers
    leave this as None and the sweep runs inside this call as before.
    """
    fleet_lock = None
    if self.config.fleet.global_max_concurrent_sessions > 0:
        fleet_lock = try_acquire_fleet_lock(self.fleet_dir_override)
        if fleet_lock is None:
            return _wf.CommandResult(
                True,
                "rework dispatch deferred: fleet lock held",
                {
                    "adapter": self.config.worker.harness,
                    "selected_count": 0,
                    "deferred_reason": "fleet_lock_held",
                },
            )
    try:
        return self._dispatch_rework_impl(
            limit, only_issues=only_issues, stalled_entries=stalled_entries
        )
    except StateLockBusy:
        return _wf._state_lock_busy_result(
            "rework dispatch deferred: state lock held",
            adapter=self.config.worker.harness,
            selected_count=0,
            deferred_reason="state_lock_busy",
        )
    finally:
        if fleet_lock is not None:
            fleet_lock.release()
