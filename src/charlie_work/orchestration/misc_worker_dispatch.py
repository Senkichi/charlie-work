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
from charlie_work.command_result import CommandResult
from charlie_work.dispatch_deferral import records_deferral
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from charlie_work import local_work_park, worker_fate
from charlie_work.adapters import SessionRequest
from charlie_work.blocked_worker_escalation import resolve_fate_exempting_blocked
from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.dead_worker_sweep.effects_sessions import _emit_session_failed_relabeled
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.worker_launch_gate import acquire_fleet_launch_lock
from charlie_work.github import label_names
from charlie_work.salvage_events import repeats_last_salvage_failure
from charlie_work.state import PASSIVE_OPEN_STATUS, StateLockBusy
from charlie_work.worker import WorkerView, iter_workers
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


@dataclass(frozen=True)
class _PhantomWorkerAssessment:
    """Pre-lock evidence for one phantom sidecar (issue #2262).

    Worktree inspection, the worker-outcome read, and fate resolution are
    pure reads — ``_precheck_phantom_live_worker`` runs them before the state
    lock so the in-lock route only applies effects.
    """

    worker: WorkerView
    worktree_path: Path
    inspection: Any
    worker_outcome: Any
    fate: worker_fate.WorkerFate
    reported_push: bool
    declared_push: bool
    escalatable_blocked: bool

    @property
    def preserving(self) -> bool:
        """A preserving assessment keeps the sidecar for the reaper lane —
        salvage a pushed/stranded branch, or escalate a declared-blocked
        worker. Same triple as the pre-#2262 preserve branch."""
        return (
            isinstance(self.fate, (worker_fate.Stranded, worker_fate.PushedWithoutPr))
            or self.escalatable_blocked
            or self.declared_push
        )


@dataclass(frozen=True)
class _PhantomPrecheck:
    """Pre-lock verdict for the phantom-live-worker lane (issue #2262)."""

    workers: tuple[WorkerView, ...]
    assessments: tuple[_PhantomWorkerAssessment, ...]
    salvage: local_work_park.LocalParkResult | None


def _assess_phantom_worker(
    self,
    w: WorkerView,
    sessions_dir: Path,
    issue_number: int,
    now: datetime,
) -> _PhantomWorkerAssessment:
    """Gather one phantom worker's evidence and resolve its fate (issue #2262).

    Pure reads — this runs pre-lock inside ``_precheck_phantom_live_worker``
    so the in-lock route applies effects only.
    """
    worktree_path = Path(w.worktree_path)
    # Issue #1122: the same ``inspect_worktree_state`` single enforcement
    # point the reaper lane uses (issue #252); #1130 relaxed the preserve
    # condition from COMPLETED to ``ahead_count > 0``.
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

    # Fate resolution (design doc §8, step B5, A11): a dispatched request's
    # issue can never have an open tracked PR — ``_dispatch_impl`` builds
    # ``pr_by_issue`` once and candidate selection excludes every issue in it
    # — so `pr_known=True, open_pr_number=None` is a real invariant, not a
    # guess. `local_ahead` carries the unverified `ahead_count` (local vs
    # base): no remote read happens at dispatch time, so it cannot be split
    # into pushed vs unpushed; `unpushed=None` marks that. Row 6 resolves a
    # dead worker with local commits and an unknown remote to `Stranded`.
    try:
        outcome_mtime = datetime.fromtimestamp(
            (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
        )
    except OSError:
        outcome_mtime = None
    fate = resolve_fate_exempting_blocked(
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
        sessions_dir=sessions_dir,
        issue_number=issue_number,
        now=now,
    )
    # B9 (wf-review-opus.md): the legacy `reported_push` check required an
    # EXPLICIT `pr_created is False`. The strict gate is restored via
    # `fate.basis.outcome` — the rule-1-freshness-gated winning candidate —
    # rather than trusting the raw outcome dict unconditionally.
    fresh_outcome = fate.basis.outcome
    declared_push = (
        fresh_outcome is not None
        and fresh_outcome.push_succeeded is True
        and fresh_outcome.pr_created is False
    )
    # Issue #2010: a permission-denial ``blocked`` (a worker-config defect the
    # reaper lane will NOT escalate) is exempt — ``resolve_fate_exempting_blocked``
    # already dropped that claim, so a ``Blocked`` fate here is always
    # escalatable.
    escalatable_blocked = isinstance(fate, worker_fate.Blocked)
    return _PhantomWorkerAssessment(
        worker=w,
        worktree_path=worktree_path,
        inspection=inspection,
        worker_outcome=worker_outcome,
        fate=fate,
        reported_push=reported_push,
        declared_push=declared_push,
        escalatable_blocked=escalatable_blocked,
    )


def _precheck_phantom_live_worker(
    self,
    request: SessionRequest,
    full_issue: dict[str, Any],
    sessions_dir: Path,
    prev_entry: Mapping[str, Any],
    on_fate: Callable[[worker_fate.WorkerFate], None] | None = None,
) -> _PhantomPrecheck:
    """Pre-lock half of the phantom-live-worker route (issue #2262).

    Runs in ``_dispatch_impl`` BEFORE the state lock: resolves every phantom
    sidecar's fate (pure reads) and — when the in-lock fall-through would
    otherwise reap + requeue — probes the dead worker's branch for committed
    work through ``local_work_park.salvage_dead_worker_commits``, the shared
    "dead worker with commits" seam all dead-worker loci route through.

    The probe MUST run pre-lock: ``salvage_dead_worker_commits`` acquires
    ``state_lock`` itself (inside ``_attempt_salvage`` /
    ``park_unpublishable_work``) and ``state_lock`` is non-reentrant, so the
    in-lock route can only consume the verdict. This is the same
    pre-lock/two-phase pattern the dead-worker sweep already uses.

    Probe gating mirrors the pre-#2262 preserve semantics: a fresh ``Blocked``
    fate keeps vetoing everything (the reaper lane's escalation owns it). On
    a no-PR backend the probe then runs whenever it can — parking now beats
    deferring to the reaper, since both end the same way and only the park
    records the verdict this pass; this is also what covers the no-sidecar
    case (the incident), where no assessment exists to preserve. On a
    PR-capable backend a preserving fate still defers to the reaper lane —
    the probe only fires on the fall-through, which is what routes a phantom
    with no sidecar but a committed branch through the same
    push+PR ``_attempt_salvage`` path the other dead-worker loci use.
    """
    issue_number = request.issue_number
    now = self.host.clock.now()
    phantom_workers = tuple(
        w for w in iter_workers(sessions_dir) if w.issue_number == issue_number
    )
    assessments = tuple(
        self._assess_phantom_worker(w, sessions_dir, issue_number, now) for w in phantom_workers
    )
    if on_fate is not None:
        for assessment in assessments:
            on_fate(assessment.fate)

    preserving = next((a for a in assessments if a.preserving), None)
    blocked = next((a for a in assessments if a.escalatable_blocked), None)
    salvage: local_work_park.LocalParkResult | None = None
    if blocked is None and (
        not local_work_park.publishes_pull_requests(self.gh) or preserving is None
    ):
        issue_labels = label_names(full_issue)
        # Prefer the sidecar's recorded branch — it is the branch the dead
        # worker actually worked on; the state entry's ``branch_name`` (and
        # the title-slug fallback inside the helper) cover the rest.
        worker_branch = next((a.worker.branch for a in assessments if a.worker.branch), None)
        entry_snapshot = dict(prev_entry)
        if worker_branch:
            entry_snapshot["branch_name"] = worker_branch
        worker_outcome = next(
            (a.worker_outcome for a in assessments if isinstance(a.worker_outcome, dict)),
            None,
        )
        salvage = local_work_park.salvage_dead_worker_commits(
            gh=self.gh,
            config=self.config,
            repo_root=self.repo_root,
            worktrees_dir=self._layout.worktrees,
            state={"issues": {str(issue_number): entry_snapshot}},
            issue_number=issue_number,
            issue=full_issue,
            active_labels=issue_labels & self.config.labels.active,
            issue_labels=issue_labels,
            state_file=self.paths.state_file,
            worker_outcome=worker_outcome,
            write_gate=self.write_gate,
        )
    return _PhantomPrecheck(
        workers=phantom_workers,
        assessments=assessments,
        salvage=salvage,
    )


def _route_phantom_live_worker(
    self,
    state: dict[str, Any],
    request: SessionRequest,
    full_issue: dict[str, Any],
    sessions_dir: Path,
    *,
    precheck: _PhantomPrecheck,
) -> tuple[str, str | None, dict[str, Any]]:
    """Route a phantom ``live_worker_redispatch_averted`` result as dead.

    The adapter reported a live worker, but the recorded PID failed the
    OS-level liveness + identity check (issue #523). The pre-lock verdicts
    are already computed in ``precheck`` (fate per sidecar, plus the
    ``salvage_dead_worker_commits`` probe when the fall-through would
    otherwise run — see ``_precheck_phantom_live_worker``); this function
    only applies effects in order:

    1. ``parked`` — the dead worker's branch carried commits and the probe
       already published them (``agent:review-ready`` + ``open_passive`` on a
       no-PR backend, push + PR on a PR-capable one). Reap the stale sidecar
       and park the entry passive-open. NEVER requeue: requeueing is what
       sent the next dispatch's worktree reset to archive the commits
       (issue #2262).
    2. ``preserving`` fate — keep the sidecar and labels so the reaper lane
       can escalate a declared-blocked worker or salvage a pushed/stranded
       branch (issues #1122/#1130, unchanged).
    3. ``park_failed`` / ``probe_failed`` — defer: keep the sidecar, labels,
       and the ``dispatched`` status so the dead-worker sweep's own park
       probe retries next pass. A transient park/probe failure must never be
       the reason committed work is requeued (issue #1971's rule, applied
       here).
    4. Otherwise — the original reap + relabel: remove the stale
       sidecar/marker, strip active labels, restore ``automated-ready``.

    Returns ``(status, dispatched_at, state)``.
    """
    issue_number = request.issue_number
    phantom_workers = precheck.workers
    salvage = precheck.salvage
    issue_labels = label_names(full_issue)
    active_labels = issue_labels & self.config.labels.active

    if salvage is not None and salvage.status == "parked":
        for w in phantom_workers:
            w.reap_sidecar(sessions_dir)
        state = _emit_session_failed_relabeled(
            state,
            issue_number=issue_number,
            reason="phantom_live_worker_commits_salvaged",
            failure_kind="live_worker_redispatch_averted",
            removed_labels=sorted(active_labels),
            added_ready=False,
            label_write_ok=True,
            state_path=self.paths.state_file,
            write_gate=self.write_gate,
        )
        return PASSIVE_OPEN_STATUS, None, state

    for assessment in precheck.assessments:
        if assessment.preserving:
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
                    if assessment.escalatable_blocked
                    else "phantom_live_worker_completed_work_preserved"
                ),
                failure_kind="live_worker_redispatch_averted",
                removed_labels=[],
                added_ready=False,
                label_write_ok=True,
                worktree_state=assessment.inspection.state.value,
                reported_push=assessment.reported_push,
                worker_fate=type(assessment.fate).__name__,
                state_path=self.paths.state_file,
                write_gate=self.write_gate,
            )
            return "dispatch_failed", None, state

    if salvage is not None and salvage.status in ("park_failed", "probe_failed"):
        # Defer to the dead-worker sweep: the entry stays ``dispatched`` (the
        # sweep's ``dead_orphans`` lane re-probes and parks next pass) and the
        # sidecar is kept (the sidecar lane can retry too). The original
        # dispatch timestamp is preserved so redispatch windowing still
        # counts from the real dispatch. ``ready`` is stripped so the entry
        # is not dispatchable while the park is pending — with a dead/popped
        # PID and a ready-only label set it would otherwise be re-selected
        # next pass and relaunch before the sweep's retry. The reclaim path
        # re-adds ``ready`` when the deferral budget expires.
        removed_ready = False
        if self.config.labels.ready in issue_labels:
            removed_ready = self.write_gate.apply_issue_labels(
                self.gh,
                self.config.labels,
                issue_number,
                remove=(self.config.labels.ready,),
                cause="phantom_live_worker_salvage_deferred",
            ).ok
        state = _emit_session_failed_relabeled(
            state,
            issue_number=issue_number,
            reason="phantom_live_worker_salvage_deferred",
            failure_kind="live_worker_redispatch_averted",
            removed_labels=[self.config.labels.ready] if removed_ready else [],
            added_ready=False,
            label_write_ok=False,
            salvage_status=salvage.status,
            salvage_error=salvage.error,
            state_path=self.paths.state_file,
            write_gate=self.write_gate,
        )
        prev_entry = (state.get("issues") or {}).get(str(issue_number))
        dispatched_at = (
            prev_entry.get("dispatched_at") if isinstance(prev_entry, dict) else None
        ) or _wf.utc_now()
        return "dispatched", dispatched_at, state

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
    # Issue #2226: the strip+ready write goes through the WriteGate's
    # canonical seam so the return to ``ready`` is recorded in events.db
    # bound to this repo; the conditional add preserves the
    # no-redundant-write contract.
    label_result = self.write_gate.apply_issue_labels(
        self.gh,
        self.config.labels,
        issue_number,
        add=(self.config.labels.ready,) if needs_ready else (),
        remove=sorted(active_labels),
        to_state="ready",
        cause="session_failed_relabeled",
    )
    label_write_ok = label_result.ok
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
) -> CommandResult:
    """Dispatch rework workers for issues in needs-rework state with open PRs.

    ``stalled_entries``: pass the result of an already-completed
    ``_detect_and_handle_stalled_sessions`` sweep to skip re-running the
    sweep inside this call — same contract as ``dispatch()`` (issue #343
    Finding 2: the sweep writes Signal-1's inconclusive-probe deferral
    counter, so it must run at most once per pass). Standalone callers
    leave this as None and the sweep runs inside this call as before.
    """
    # Issue #2055: mint the fleet-launch-lock handle here so the ``finally``
    # below covers every impl exit path, but do NOT take the OS lock yet --
    # the pending handle is realized by issue_worker_launch_permit (bounded
    # wait) immediately before the governor, after _dispatch_rework_impl's
    # pr_list / issue_view / stall-sweep scan.
    launch_lock = acquire_fleet_launch_lock(self, acquire=try_acquire_fleet_lock)
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
