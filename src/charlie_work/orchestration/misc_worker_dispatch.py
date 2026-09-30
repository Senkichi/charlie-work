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
from charlie_work.dispatch_deferral import records_deferral
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from charlie_work import worker_fate
from charlie_work.adapters import SessionRequest
from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.dead_worker_reap import _emit_session_failed_relabeled
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.worker_launch_gate import WorkerLaunchDeferral, acquire_fleet_launch_lock
from charlie_work.github import label_names
from charlie_work.salvage_events import repeats_last_salvage_failure
from charlie_work.state import StateLockBusy
from charlie_work.worker import iter_workers
from charlie_work.foreign_worktree import OPERATOR_MARKER_KIND, read_worktree_marker
from charlie_work.worktree import (
    WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
    WorktreeProbeFailedError,
    _archive_unreachable_tip_if_applicable,
    _has_origin_remote,
    _worktree_refuse_to_reset_reason,
    _worktree_unsafe_kind_from_reason,
    inspect_worktree_state,
    read_worker_outcome,
    salvage_push_stranded_commits,
    worktree_path_for_branch,
)


def _route_phantom_live_worker(
    self,
    state: dict[str, Any],
    request: SessionRequest,
    full_issue: dict[str, Any],
    sessions_dir: Path,
    on_fate: Callable[[worker_fate.WorkerFate], None] | None = None,
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

    The caller holds ``state_lock`` (and ``report_stale_evidence`` takes it
    itself), so each resolved fate is handed to ``on_fate`` instead of
    reported here; the caller reports rule 1's stale evidence (B6) once it
    has released the lock.
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

        # Obtain this worker's fate from the module (design doc §8, step
        # B5, A11): a dispatched request's issue can never have an open
        # tracked PR (see the docstring above), so `pr_known=True,
        # open_pr_number=None` is a real invariant here, not a guess.
        # `local_ahead` carries the local, unverified `ahead_count` (local vs
        # base): no remote read happens at dispatch time, so it cannot be
        # split into pushed vs unpushed. `unpushed=None` marks that; row 6
        # resolves a dead worker with local commits and an unknown remote to
        # `Stranded` (rule "R3+R9-remote-unknown"), which the branch below
        # preserves for the salvage lane.
        try:
            outcome_mtime = datetime.fromtimestamp(
                (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
            )
        except OSError:
            outcome_mtime = None
        fate = worker_fate.resolve_fate(
            worker_fate.FateEvidence(
                issue_number=issue_number,
                adapter=w.adapter_kind,
                dispatched_at=worker_fate.parse_iso_timestamp(w.started_at),
                pid_alive=False,
                health=None,
                terminal=None,
                worktree_outcome=(
                    worker_fate.OutcomeEvidence(
                        source=worker_fate.EvidenceSource.WORKTREE,
                        written_at=outcome_mtime,
                        outcome=worker_outcome.get("outcome"),
                        push_succeeded=worker_outcome.get("push_succeeded"),
                        pr_created=worker_outcome.get("pr_created"),
                        head_sha=worker_outcome.get("head_sha"),
                        raw=worker_outcome,
                    )
                    if isinstance(worker_outcome, dict)
                    else None
                ),
                branch=worker_fate.BranchEvidence(
                    remote_head_sha=None,
                    remote_ahead=None,
                    unpushed=None,
                    local_ahead=inspection.ahead_count,
                    open_pr_number=None,
                    pr_known=True,
                ),
                failure=None,
            ),
            now=datetime.now(UTC),
        )
        if on_fate is not None:
            on_fate(fate)
        # Rule 1/2 (design doc §8, step B5): preserve for salvage when the
        # resolved fate is `Stranded` (local-only commits) -- a fresh
        # `blocked` declaration now resolves to `fate=Blocked` (rule 2:
        # fresh blocked beats push flags) and no longer gets folded into
        # the salvage-preserve branch just because local commits or a push
        # claim are also present.
        #
        # `PushedWithoutPr` is named alongside `Stranded` for the case it
        # can prove, but `_is_pushed` (row 2/3) needs `branch.remote_ahead`
        # or a matching `remote_head_sha` to confirm a push, and neither is
        # ever populated here -- this call site deliberately does no
        # remote read at dispatch time (see the evidence-construction
        # comment above), so `resolve_fate` alone can never *return*
        # `PushedWithoutPr`/`Completed` from this evidence shape; an
        # unconfirmable, fresh, self-reported push falls through to
        # `Crashed` (row 10) instead. `fate.basis.outcome` still carries
        # the rule-1-freshness-gated winning candidate regardless of which
        # row picked it, so `declared_push` reads the same self-reported
        # claim the legacy `reported_push` check trusted directly, gated
        # on rule 1 instead of trusted unconditionally.
        # B9 (wf-review-opus.md): the legacy `reported_push` check above
        # required an EXPLICIT `pr_created is False`. This refactored
        # `declared_push` loosened it to `pr_created is not True`, which
        # also admits `None` -- an outcome that omits `pr_created` now
        # gets the same "confirmed pushed, no PR" preservation treatment
        # as an explicit `false`. That widening was never one of the nine
        # reviewed flips; restore the strict legacy gate.
        fresh_outcome = fate.basis.outcome
        declared_push = (
            fresh_outcome is not None
            and fresh_outcome.push_succeeded is True
            and fresh_outcome.pr_created is False
        )
        # B4 (wf-review-opus.md), rule 2: a fresh `blocked` declaration
        # beats push flags everywhere, but this dispatch-time helper has
        # no remote read (see above) and is not the single point of
        # enforcement for the blocked-escalation invariant -- the
        # dead-session reaper lane already owns that (`workflow.py`'s
        # `fates.get(issue_number) == Blocked` branch, and the
        # `orphaned_worker_sweep.py` sites fixed for B2/B3), keyed off the
        # very sidecar this function would otherwise reap. Before this
        # fix, `Blocked` was neither `Stranded` nor `PushedWithoutPr` and
        # `declared_push` was false for a pure blocked outcome, so control
        # fell through to the reap-and-`ready` path below: the sidecar the
        # reaper lane keys off was destroyed, and the issue was
        # re-queued into the identical wall it had just declared itself
        # blocked on. Folding `Blocked` into this same preserve branch
        # (rather than escalating inline here, which would duplicate that
        # enforcement point and reintroduce the entry-identity clobber the
        # caller's `is_phantom_live_worker` arm does not guard against --
        # it never re-reads `entry` from `state["issues"]` the way the
        # dispatch-failed-cap and blocked-environment arms do after their
        # own `_escalate_issue` calls) leaves the sidecar and labels
        # untouched instead: the reaper lane's next pass resolves the same
        # fresh `Blocked` fate with real branch evidence and escalates it
        # exactly as it already does for a dead worker.
        if (
            isinstance(
                fate, (worker_fate.Stranded, worker_fate.PushedWithoutPr, worker_fate.Blocked)
            )
            or declared_push
        ):
            # Preserve the sidecar so the reaper lane can act on it --
            # either salvaging a pushed/stranded branch or escalating a
            # declared-blocked worker. Do NOT strip labels -- the issue
            # should stay in its active state until that lane resolves it,
            # preventing re-dispatch into the occupied worktree or wall.
            state = _emit_session_failed_relabeled(
                state,
                issue_number=issue_number,
                reason=(
                    "phantom_live_worker_declared_blocked_preserved"
                    if isinstance(fate, worker_fate.Blocked)
                    else "phantom_live_worker_completed_work_preserved"
                ),
                failure_kind="live_worker_redispatch_averted",
                removed_labels=[],
                added_ready=False,
                label_write_ok=True,
                worktree_state=inspection.state.value,
                reported_push=reported_push,
                worker_fate=type(fate).__name__,
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


def _worktree_still_unsafe(
    self, issue_number: int, state: dict[str, Any], *, dry_run: bool = False
) -> str | None:
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

    Rule 3 of the Worker fate resolution (a stranded branch -- committed
    but not on the remote -- is salvaged, never merely "stuck") is applied
    here as salvage-before-clear, the same fact
    ``_route_phantom_live_worker`` resolves to ``worker_fate.Stranded``.
    When the reason is ``WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`` (no
    worker-authored dirt, only commits absent from the remote) the commits
    are made durable first (``_salvage_stranded_before_clear``) and the
    escalation clears only once they are:

    - repo WITH an origin remote: the ff-only
      ``salvage_push_stranded_commits`` push, then
      ``_worktree_refuse_to_reset_reason`` is re-run and must be clean.
      Both callers always have an open PR, so this deliberately does NOT
      go through ``_attempt_salvage`` (its PR-open would duplicate it).
    - repo with NO origin remote: the #1944 archive park. The re-check
      cannot turn clean there (nothing is ever "on the remote"), so a
      verified archive ref is the clearing evidence instead.

    Any salvage skip or failure (diverged remote, live or operator writer
    marker, push or probe error, archive declined) returns the ORIGINAL
    reason so the label stays. ``dry_run`` (the command's own flag or the
    write gate's) never salvages, so it never clears a local-commits
    reason.
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

    def _refuse_reason() -> str | None:
        return _worktree_refuse_to_reset_reason(
            self.repo_root,
            branch,
            self.config.dispatch.base_ref,
            wt_path,
            self.config.dispatch.injected_paths,
            self.config.dispatch.materialize_dirs,
        )

    try:
        reason = _refuse_reason()
        if (
            not reason
            or _worktree_unsafe_kind_from_reason(reason) != WORKTREE_UNSAFE_KIND_LOCAL_COMMITS
            or dry_run
            or self.write_gate.dry_run
        ):
            return reason
        mode = self._salvage_stranded_before_clear(issue_number, branch, wt_path)
        if mode is None:
            return reason
        if mode == "parked":
            return None
        return _refuse_reason()
    except (WorktreeProbeFailedError, RuntimeError):
        # Fail closed: a probe failure means we cannot confirm safety,
        # so treat the worktree as still unsafe.
        return "worktree safety probe failed; cannot confirm clean"


def _salvage_stranded_before_clear(
    self, issue_number: int, branch: str, wt_path: Path
) -> str | None:
    """Publish or park a stranded branch before its escalation may clear.

    Returns ``"pushed"`` (ff-only push to origin landed; the caller
    re-checks), ``"parked"`` (no origin remote; tip archived by the #1944
    park path), or ``None`` when the work is not proven durable (the label
    must stay). Emits ``worktree_unsafe_stranded_salvaged`` or
    ``worktree_unsafe_stranded_salvage_failed`` (with ``skip_reason``).
    """
    mode: str | None = None
    skip_reason: str | None = None
    payload: dict[str, Any] = {"issue_number": issue_number, "branch": branch}
    if _has_origin_remote(self.repo_root):
        result = salvage_push_stranded_commits(
            self.repo_root,
            branch,
            wt_path,
            base_ref=self.config.dispatch.base_ref,
            dry_run=False,
        )
        if result.pushed:
            mode = "pushed"
            payload.update(
                commit_count=result.commit_count,
                old_remote_sha=result.old_remote_sha,
                new_remote_sha=result.new_remote_sha,
            )
        else:
            skip_reason = result.skip_reason or result.error or "push_failed"
    elif not self.config.dispatch.archive_unreachable_local_commits:
        skip_reason = "archive_disabled"
    else:
        marker = read_worktree_marker(wt_path)
        marker_pid = marker.get("pid") if marker is not None else None
        # Same fingerprint the with-origin path passes (worktree.py), so a
        # recycled marker PID does not read as a live writer.
        marker_start_time = marker.get("process_start_time") if marker is not None else None
        if marker is not None and marker.get("kind") == OPERATOR_MARKER_KIND:
            skip_reason = "operator_claimed"
        elif isinstance(marker_pid, int) and worker_fate.is_alive(marker_pid, marker_start_time):
            skip_reason = "live_writer_marker"
        elif _archive_unreachable_tip_if_applicable(
            self.repo_root, branch, wt_path, self.paths.state_file, set(), issue_number
        ):
            mode = "parked"
        else:
            skip_reason = "archive_declined"

    payload["mode"] = mode
    if mode is None:
        payload["skip_reason"] = skip_reason
    kind = (
        "worktree_unsafe_stranded_salvaged"
        if mode is not None
        else "worktree_unsafe_stranded_salvage_failed"
    )
    with _wf.state_lock(self.paths.state_file):
        fresh = _wf.load_state(self.paths.state_file)
        # N3: the de-escalation sweep re-runs this every interval for a
        # persistently unsalvageable branch; record a failure only when it
        # differs from the newest failure already logged for this issue.
        if mode is not None or not repeats_last_salvage_failure(fresh, payload):
            fresh = self._record_event(fresh, kind, payload)
            self.write_gate.save_state(fresh)
    return mode


@records_deferral("dispatch_rework")
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
    # Issue #2041: gate 1 of the shared worker-launch gate, held across
    # governor -> claim -> launch; _dispatch_rework_impl hands it to
    # issue_worker_launch_permit.
    launch_lock = acquire_fleet_launch_lock(self, acquire=try_acquire_fleet_lock)
    if isinstance(launch_lock, WorkerLaunchDeferral):
        return _wf.CommandResult(
            True,
            "rework dispatch deferred: fleet lock held",
            {
                "adapter": self.config.worker.harness,
                "selected_count": 0,
                "deferred_reason": launch_lock.reason,
            },
        )
    try:
        return self._dispatch_rework_impl(
            limit,
            only_issues=only_issues,
            stalled_entries=stalled_entries,
            launch_lock=launch_lock,
        )
    except StateLockBusy:
        return _wf._state_lock_busy_result(
            "rework dispatch deferred: state lock held",
            adapter=self.config.worker.harness,
            selected_count=0,
            deferred_reason="state_lock_busy",
        )
    finally:
        launch_lock.release()
