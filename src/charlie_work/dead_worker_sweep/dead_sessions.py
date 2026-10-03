"""The dead-session lane: reap dead sessions, classify their failures, update throttle state.

``classify_dead_sessions`` is called from the production loop to detect provider
throttling from worker deaths and set the cooldown window in ``state.json``. Every
decision it makes is a pure function in ``decide_dead_sessions``; this module reads
the world (GitHub, state, the worktree), asks, and applies the answer in the same
order the lane always did. The no-open-PR reclaim and the open-PR rework routing
live in ``dead_sessions_reclaim``.

Also reconciles labels for dead sessions with no open PR (issue #118): a dead worker
with no open PR is recoverable and is relabeled as dispatchable (remove active
labels, ensure the ready label is present).

Issue #252: a dead worker with a clean worktree and unpushed commits
(completed-but-unpublished) gets its branch pushed and a PR created, and the issue
moves to ``pr_open`` instead of being re-dispatched.

Issue #266: launch-failure sidecars (``pid=None``, ``error`` set) are terminal by
construction and are reaped immediately, reported in the returned list.

Issue #295: dead or launch-failed rework sessions with an open PR and a
request_changes verdict (or a rework prompt on disk) are restored to
``rework_requested`` so ``dispatch_rework`` can re-select them.

``persist_inconclusive_probe_counter`` (issue #343 Finding 2) controls whether this
lane persists Signal-1's inconclusive-probe deferral counter for a not-alive,
pid-bearing worker. It defaults to True so the function is self-sufficient when
called on its own. ``loop()`` always runs the sibling stall lane (``run_stalled_sweep``,
the sole writer of this counter for an ALIVE-but-stalled worker and the first lane to
see every worker each pass) immediately before this one and passes False, so the
counter is not incremented twice in one pass (0->1 in the stall lane, ->2 here), which
halved the deferral grace period and opened Finding 1's pass-2 phantom-sidecar window.
``classify_worker_health``'s own cap check reads whatever value is on the sidecar, so
suppressing the write here never changes the DEAD-vs-deferred decision.

``now`` (issue #822) is the injectable clock for the pass: it seeds the worker-health
and probe timing and is forwarded to every throttle classification, so one instant
serves the whole pass. It defaults to ``datetime.now(UTC)``.

Collaborators that the suite patches on their defining module (``worker``,
``worker_fate``, ``post_mortem``, ``state``, ``worker_literal_tmp``) are looked up as
module attributes at call time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import post_mortem, role_quota_ledger, worker, worker_fate, worker_literal_tmp
from .. import state as state_mod
from ..config import OrchestratorConfig
from ..dispatch_selection import _windowed_redispatch_at
from ..escalation import _escalate_issue, _escalation_edge
from ..fleet_registry import managed_repo_names
from ..host import current as _host_current
from ..github import GitHubLike, build_branch_issue_validator, label_names
from ..issue_linking import linked_issue_number
from ..paths import resolved_layout
from ..worktree import (
    WorktreeState,
    inspect_worktree_state,
    read_worker_outcome,
    worktree_path_for_branch,
)
from ..write_gate import WriteGate, require_write_gate
from ..process_utils import find_worker_terminal_status
from .dead_sessions_reclaim import reclaim_or_route_dead_session
from .decide_dead_sessions import (
    BACKGROUND_EXIT_FAILURE_KIND,
    exited_with_background_work,
    wants_unsafe_salvage,
)
from .decide_dead_sessions_plan import (
    EmitBackgroundExit,
    EmitProviderSuspended,
    EscalateLaunchFailure,
    PersistFailure,
    ReapSidecar,
    ReclaimOrRoute,
    RestoreRework,
    Step,
    WarnLiteralTmp,
    plan_dead_classification,
    plan_dead_reap,
    plan_launch_escalation,
    plan_launch_failed,
    wants_salvage_from_unsafe,
)
from .effects_pr import _attempt_salvage, _dispatching_repo_name
from .effects_rework import _reap_restore_rework_requested


@dataclass(frozen=True)
class SessionPass:
    """Everything one sweep pass reads once and every worker shares."""

    sessions_dir: Path
    state_file: Path
    gh: GitHubLike
    config: OrchestratorConfig
    write_gate: WriteGate
    repo_root: Any
    open_prs_by_issue: dict[int, list[dict[str, Any]]]
    now_for_health: datetime
    fleet_repos: Any
    dispatching_repo_name: str


def _open_prs_by_issue(
    gh: GitHubLike, config: OrchestratorConfig
) -> dict[int, list[dict[str, Any]]]:
    """Open PRs keyed by the issue they link, for the "no open PR" guard.

    Issue #1229: branch-name-derived issue numbers are validated against the
    open-issue set so a stale branch name (e.g. ``agent/issue-709-...`` left over
    from a merged PR, reused by an unrelated issue-less PR) cannot bind a phantom
    open PR to a dead worker's issue and skip its escalation/salvage.
    """
    prs = gh.pr_list()
    open_prs_by_issue: dict[int, list[dict[str, Any]]] = {}
    branch_validator = build_branch_issue_validator(gh)
    for pr in prs:
        if str(pr.get("state") or "").upper() != "OPEN":
            continue
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
        )
        if issue_number is not None:
            open_prs_by_issue.setdefault(issue_number, []).append(pr)
    return open_prs_by_issue


def _escalate_launch_failure(ctx: SessionPass, w: worker.WorkerView, failure_kind: str) -> None:
    """A deterministic launch failure with no open PR: salvage first, else escalate."""
    gh, config, write_gate, state_file = ctx.gh, ctx.config, ctx.write_gate, ctx.state_file
    try:
        issue = gh.issue_view(w.issue_number)
    except Exception:
        issue = None
    issue_labels = label_names(issue) if issue else set()
    active_labels = issue_labels & config.labels.active

    # Issue #1130: before escalating a ``worktree_unsafe`` launch failure, attempt
    # salvage -- redispatch found the worktree holding unpushed commits, exactly the
    # work salvage exists to recover. Issue #807: the ``ahead_count > 0`` gate below
    # filters shim dirt (no commits ahead) so salvage only fires for genuine commits.
    salvaged_from_unsafe = False
    if wants_unsafe_salvage(
        failure_kind,
        has_repo_root=ctx.repo_root is not None,
        branch=w.branch,
        active_labels=active_labels,
    ):
        worktrees_dir = resolved_layout(config, ctx.repo_root).worktrees
        wt_path = worktree_path_for_branch(ctx.repo_root, w.branch, worktrees_dir)
        unsafe_inspection = inspect_worktree_state(
            wt_path,
            config.dispatch.base_ref,
            config.dispatch.injected_paths,
            config.dispatch.materialize_dirs,
        )
        if wants_salvage_from_unsafe(ahead_count=unsafe_inspection.ahead_count):
            salvaged_from_unsafe, _ = _attempt_salvage(
                gh=gh,
                config=config,
                repo_root=ctx.repo_root,
                worktree_path=wt_path if wt_path.is_dir() else ctx.repo_root,
                branch=w.branch,
                base_ref=unsafe_inspection.resolved_base_ref or config.dispatch.base_ref,
                issue_number=w.issue_number,
                active_labels=active_labels,
                issue_labels=issue_labels,
                state_file=state_file,
                failure_kind=failure_kind,
                issue_title=issue.get("title") if issue else None,
                # Issue #1241: pass the live issue so the shared supersession
                # check's closed-issue branch fires here too.
                issue=issue,
                # cw#1771: read from the REAL worktree dir, never the ``repo_root``
                # fallback passed as ``worktree_path`` above -- a stray
                # `.worker-outcome.json` at the main checkout root must never be
                # mistaken for this branch's drafted PR content.
                worker_outcome=(read_worker_outcome(wt_path) if wt_path.is_dir() else None),
                write_gate=write_gate,
            )
    if salvaged_from_unsafe:
        return

    with state_mod.state_lock(state_file):
        state = state_mod.load_state(state_file)
        entry = state["issues"].get(str(w.issue_number), {})
        plan = plan_launch_escalation(
            _windowed_redispatch_at(
                entry, window_minutes=config.watchdog.redispatch_window_minutes
            ),
            failure_kind,
            now=_host_current().clock.now(),
            active_labels=active_labels,
        )
        redispatch_at = list(plan.redispatch_at)
        state = _escalate_issue(
            state,
            w.issue_number,
            reason=plan.reason,
            reason_class=plan.reason_class,
            issue_extra={"redispatch_at": redispatch_at},
        )
        state["issues"][str(w.issue_number)].pop("worker_pid", None)
        state["issues"][str(w.issue_number)].pop("worker_process_start_time", None)
        write_gate.save_state(state)
        write_gate.transition(
            gh,
            config.labels,
            w.issue_number,
            _escalation_edge("redispatch_escalated", plan.reason_class),
        )
        state = write_gate.append_event(
            state,
            "session_failed_escalated",
            {
                "issue_number": w.issue_number,
                "failure_kind": failure_kind,
                "removed_labels": list(plan.removed_labels),
                "redispatch_count": len(redispatch_at),
            },
        )
        write_gate.save_state(state)


def _persist_failure(
    ctx: SessionPass,
    w: worker.WorkerView,
    failure_kind: str,
    throttled_until: Any,
    *,
    launch_failure: bool,
) -> None:
    """Persist the classification and throttle window on the issue entry.

    Issue #1917: the classification lands on the issue entry so the state.json-keyed
    orphan sweep -- which runs after the sidecar is reaped -- can exempt
    provider-throttle outcomes from its timed reap and orphan-redispatch cap.

    ``launch_failure`` selects the audit-trail source; each ``persist_failure`` call
    carries its ``source=`` as a string literal (the repo-wide audit guard), kept equal
    to the plan's ``*_PERSIST_SOURCE`` constants by a test.
    """
    # The dead session's stamped role-chain entry rides the evidence so the
    # per-repo window it arms is attributed to the ``(harness, model)`` that
    # died, not the adapter as a whole (issue #2279). The sidecar still
    # exists here: ``plan_dead_reap`` orders ``PersistFailure`` before
    # ``ReapSidecar``.
    role_key = role_quota_ledger.role_key_for_session(
        ctx.sessions_dir, w.adapter_kind, w.issue_number
    )
    harness, model = role_key if role_key is not None else (None, None)
    evidence = worker_fate.FailureEvidence.from_classification(
        failure_kind, throttled_until, fresh=True, harness=harness, model=model
    )
    with state_mod.state_lock(ctx.state_file):
        state = state_mod.load_state(ctx.state_file)
        if launch_failure:
            state = worker_fate.persist_failure(
                state,
                w.issue_number,
                evidence,
                adapter_kind=w.adapter_kind,
                now=ctx.now_for_health,
                source="dead_sessions_launch_failure",
                write_gate=ctx.write_gate,
            )
        else:
            state = worker_fate.persist_failure(
                state,
                w.issue_number,
                evidence,
                adapter_kind=w.adapter_kind,
                now=ctx.now_for_health,
                source="dead_sessions_reap",
                write_gate=ctx.write_gate,
            )
        ctx.write_gate.save_state(state)


def _run_steps(steps: tuple[Step, ...], handlers: dict[type, Callable[[Any], None]]) -> None:
    """Execute a decided plan in order; an unhandled step type is a wiring bug (KeyError)."""
    for step in steps:
        handlers[type(step)](step)


def _reap_entry(w: worker.WorkerView, failure_kind: str | None) -> dict[str, Any]:
    return {
        "issue_number": w.issue_number,
        "adapter_kind": w.adapter_kind,
        "failure_kind": failure_kind,
        "error": w.error,
        "pid": w.pid,
    }


def _reap_launch_failed(ctx: SessionPass, w: worker.WorkerView, profile: Any) -> dict[str, Any]:
    """Issue #266: a launch-failure sidecar is terminal; classify, stamp, escalate, reap."""
    config, write_gate, state_file = ctx.config, ctx.write_gate, ctx.state_file
    failure_kind: str | None = "launch_failed"
    throttled_until = None
    if profile is not None and profile.record_failure is not None:
        failure_kind, throttled_until = profile.record_failure(
            ctx.sessions_dir,
            w.issue_number,
            fallback_kind=failure_kind,
            config=config,
            now=ctx.now_for_health,
        )
    steps = plan_launch_failed(failure_kind, has_open_pr=w.issue_number in ctx.open_prs_by_issue)
    _run_steps(
        steps,
        {
            # A throttle-caused launch failure must persist its window just like the
            # dead-session branch -- otherwise the governor relaunches straight into
            # the same throttled provider.
            PersistFailure: lambda _: _persist_failure(
                ctx,
                w,
                failure_kind,  # type: ignore[arg-type]  # the plan only persists a kind
                throttled_until,
                launch_failure=True,
            ),
            EscalateLaunchFailure: lambda _: _escalate_launch_failure(
                ctx,
                w,
                failure_kind,  # type: ignore[arg-type]  # the plan only escalates a kind
            ),
            ReapSidecar: lambda _: w.reap_sidecar(
                ctx.sessions_dir, api_config=config.api_worker, state_dir=state_file.parent
            ),
            # Issue #295: a launch-failed rework session must still be returned to
            # rework_requested so its owning lane can re-dispatch it.
            RestoreRework: lambda _: _reap_restore_rework_requested(
                state_file,
                ctx.gh,
                config,
                ctx.open_prs_by_issue,
                w,
                failure_kind=failure_kind,
                repo_root=ctx.repo_root,
                write_gate=write_gate,
            ),
        },
    )
    return _reap_entry(w, failure_kind)


def _emit_provider_suspended(
    ctx: SessionPass, w: worker.WorkerView, failure_kind: str | None
) -> None:
    """Issue #1342: a distinct error-level event on the FIRST detection of a suspension.

    ``provider_suspended`` is terminal (no cooldown) and escalates on this same pass;
    the sidecar was just reaped, so the event fires once per episode.
    """
    with state_mod.state_lock(ctx.state_file):
        state = state_mod.load_state(ctx.state_file)
        state = ctx.write_gate.append_event(
            state,
            "api_worker_provider_suspended",
            {
                "issue_number": w.issue_number,
                "pid": w.pid,
                "process_start_time": w.process_start_time,
                "provider": w.provider,
                "failure_kind": failure_kind,
            },
            level="error",
        )
        ctx.write_gate.save_state(state)


def _emit_background_exit(
    ctx: SessionPass,
    w: worker.WorkerView,
    failure_kind: str | None,
    ahead_count: int,
) -> None:
    """Issue #2096: visible and countable, not a generic orphan."""
    with state_mod.state_lock(ctx.state_file):
        state = state_mod.load_state(ctx.state_file)
        state = ctx.write_gate.append_event(
            state,
            # event-consumer: audit-only -- operator-visible forensic record of a
            # worker that exited 0 with uncommitted work (issue #2096); the
            # reap and escalation decisions key off ``failure_kind``, not this event
            BACKGROUND_EXIT_FAILURE_KIND,
            {
                "issue_number": w.issue_number,
                "pid": w.pid,
                "adapter_kind": w.adapter_kind,
                "failure_kind": failure_kind,
                "ahead_count": ahead_count,
            },
            level="warning",
        )
        ctx.write_gate.save_state(state)


def _reap_dead(
    ctx: SessionPass,
    w: worker.WorkerView,
    profile: Any,
    *,
    persist_inconclusive_probe_counter: bool,
) -> dict[str, Any] | None:
    """A not-alive worker: confirm death, classify, stamp, reap, then reclaim or route."""
    config, state_file = ctx.config, ctx.state_file
    # Issue #755: the confirmed-dead decision (including the
    # max_inconclusive_probe_deferrals grace period and counter) is owned by one
    # helper shared with reconcile.detect_drift.
    if not worker.is_worker_confirmed_dead(
        w,
        config,
        ctx.now_for_health,
        ctx.sessions_dir,
        persist_inconclusive_probe_counter=persist_inconclusive_probe_counter,
    ):
        return None

    # Inspect the worktree before classifying and relabeling: the single
    # enforcement point for issue #252.
    worktree_path = Path(w.worktree_path)
    inspection = inspect_worktree_state(
        worktree_path,
        config.dispatch.base_ref,
        config.dispatch.injected_paths,
        config.dispatch.materialize_dirs,
    )
    is_completed = inspection.state == WorktreeState.COMPLETED
    classification = plan_dead_classification(
        is_completed=is_completed,
        worktree_unknown=inspection.state == WorktreeState.UNKNOWN,
        background_exit=not is_completed
        and exited_with_background_work(
            find_worker_terminal_status(ctx.sessions_dir, w.issue_number),
            adapter_kind=w.adapter_kind,
            dirty=inspection.dirty,
            ahead_count=inspection.ahead_count,
            pid=w.pid,
        ),
    )

    def record_post_mortem() -> None:
        # Diagnostic; for a completed worktree its worker_blocked verdict is ignored
        # because the worktree itself proves the work was completed.
        post_mortem.classify_and_record(
            ctx.sessions_dir, config, w, now=_host_current().clock.now()
        )

    if classification.post_mortem_first:
        record_post_mortem()
    failure_kind: str | None = None
    throttled_until = None
    if profile is not None and profile.record_failure is not None:
        # session_completed (issue #656): the inspection is ground truth that the
        # session produced real work, so log-tail marker matching is skipped.
        extra = {"session_completed": True} if classification.session_completed else {}
        failure_kind, throttled_until = profile.record_failure(
            ctx.sessions_dir,
            w.issue_number,
            fallback_kind=classification.fallback_kind,
            config=config,
            now=ctx.now_for_health,
            **extra,
        )
    if not classification.post_mortem_first:
        record_post_mortem()

    entry = _reap_entry(w, failure_kind)
    _run_steps(
        plan_dead_reap(failure_kind),
        {
            PersistFailure: lambda _: _persist_failure(
                ctx,
                w,
                failure_kind,  # type: ignore[arg-type]  # the plan only persists a kind
                throttled_until,
                launch_failure=False,
            ),
            # Reap the sidecar to prevent phantom sessions from PID recycling (#113).
            ReapSidecar: lambda _: w.reap_sidecar(
                ctx.sessions_dir, api_config=config.api_worker, state_dir=state_file.parent
            ),
            # Issue #1780: fires at most once per session: the sidecar was just reaped.
            WarnLiteralTmp: lambda _: worker_literal_tmp.emit_literal_tmp_path_warning(
                state_file, w, ctx.write_gate
            ),
            EmitBackgroundExit: lambda _: _emit_background_exit(
                ctx, w, failure_kind, inspection.ahead_count
            ),
            EmitProviderSuspended: lambda _: _emit_provider_suspended(ctx, w, failure_kind),
            ReclaimOrRoute: lambda _: reclaim_or_route_dead_session(
                ctx,
                w,
                failure_kind=failure_kind,
                inspection=inspection,
                is_completed=is_completed,
                worktree_path=worktree_path,
            ),
        },
    )
    return entry


def classify_dead_sessions(
    sessions_dir: Path,
    state_file: Path,
    gh: GitHubLike,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    persist_inconclusive_probe_counter: bool = True,
    now: datetime | None = None,
    fleet_dir_override: str | None = None,
) -> list[dict[str, Any]]:
    """Check for dead sessions, classify their failures, and update throttle state.

    Returns one row per reaped session. See the module docstring for the lane's
    contract and the issues that shaped it.
    """
    write_gate = require_write_gate(write_gate)
    now_for_health = now if now is not None else _host_current().clock.now()

    repo_root = getattr(gh, "repo_root", None)
    open_prs = _open_prs_by_issue(gh, config)
    # Issue #1244: pre-compute fleet info for the cross-repo scope tripwire. The
    # managed-repo set is derived from the fleet registry, never a hardcoded list,
    # and computed once before the worker loop because the registry is read from disk
    # and ``gh.name_with_owner()`` is a network call.
    ctx = SessionPass(
        sessions_dir=sessions_dir,
        state_file=state_file,
        gh=gh,
        config=config,
        write_gate=write_gate,
        repo_root=repo_root,
        open_prs_by_issue=open_prs,
        now_for_health=now_for_health,
        fleet_repos=managed_repo_names(fleet_dir_override),
        dispatching_repo_name=(
            _dispatching_repo_name(gh, repo_root) if repo_root is not None else ""
        ),
    )

    reaped: list[dict[str, Any]] = []
    for w in worker.iter_workers(sessions_dir):
        # The Adapter seam (design doc section 7): one profile lookup per worker,
        # reused by every classification branch.
        profile = worker_fate.profile_for(w.adapter_kind)
        if w.pid is None and w.error is not None:
            reaped.append(_reap_launch_failed(ctx, w, profile))
            continue
        if not w.is_alive():
            entry = _reap_dead(
                ctx,
                w,
                profile,
                persist_inconclusive_probe_counter=persist_inconclusive_probe_counter,
            )
            if entry is not None:
                reaped.append(entry)
    return reaped
