"""The stalled-session lane's apply shell: ``run_stalled_sweep``.

Same contract as the orphan sweep's shell (see ``apply.py``): per worker, ask
``decide_stalled`` what to do given the results observed so far, apply the new
commits, run the one request the plan still needs, and decide again. The plan
is deterministic, so a commit prefix that changes between rounds, or a request
that comes back unanswered, is a bug in the decision module: the shell logs an
error-level event and leaves that worker untouched for the pass.

Issue #1325: every process kill and every state/event write goes through
``write_gate`` so a ``dry_run=True`` gate suppresses it. The sidecar-writing
helpers take no gate; ``decide_stalled`` withholds their commits under dry-run.
The ``write_gate`` parameter follows the Convention B explicit-threading
pattern (``require_write_gate()``).

Every collaborator is looked up on its defining module at call time (never
bound at import), so suite patches on ``charlie_work.worker.*``,
``charlie_work.post_mortem.*`` and friends stay live.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import devin_shell, no_pr_orphan_fate, post_mortem, role_quota_ledger, worker, worker_fate
from . import effects_sessions
from ..config import OrchestratorConfig
from ..process_utils import find_worker_terminal_status
from ..worktree import read_worker_outcome
from ..state import load_state, load_state_locked, set_throttled_until, state_lock
from ..write_gate import WriteGate, require_write_gate
from .decide_stalled import DEFER_SOURCE, REAP_SOURCE, decide_stalled
from .stalled_model import (
    STALLED_REQUEST_TYPES,
    AdapterFacts,
    HealthProbe,
    KillOrphans,
    KillTree,
    LogTail,
    MarkBudgetExceeded,
    ProbeCompletedHandoff,
    ProbeHealth,
    ProbeRateLimitDefer,
    ReadAdapterProfile,
    ReadLogTail,
    RecordFailure,
    RecordPostMortem,
    RecordRoleQuota,
    StalledFacts,
    StampSidecar,
    StateTxn,
    ThrottleArm,
)

logger = logging.getLogger(__name__)

# A worker needs a handful of requests; far above that means a loop that never converges.
MAX_ROUNDS_PER_WORKER = 64

PLAN_VIOLATION_KIND = "stalled_sweep_plan_violation"


def _completed_handoff(
    w: worker.WorkerView, *, state_file: Path, sessions_dir: Path, now: datetime
) -> bool:
    """True when ``worker_fate`` finds a fresh, non-blocked completed outcome.

    Same freshness rules as the orphan sweep (``resolve_no_pr_orphan_fate``): a
    terminal record from an earlier dispatch, or a stale worktree outcome, does
    not count. Completed means a declared push or a clean (exit 0) terminal.
    """
    entry = (load_state_locked(state_file).get("issues") or {}).get(str(w.issue_number), {})
    worktree = Path(w.worktree_path) if w.worktree_path else None
    fate = no_pr_orphan_fate.resolve_no_pr_orphan_fate(
        issue_number=w.issue_number,
        entry=entry,
        terminal=find_worker_terminal_status(sessions_dir, w.issue_number),
        worktree_path=worktree,
        worktree_outcome_raw=read_worker_outcome(worktree) if worktree is not None else None,
        now=now,
    )
    outcome = fate.basis.outcome
    if outcome is not None:
        return outcome.outcome != "blocked" and (
            outcome.push_succeeded is True or outcome.outcome == "completed"
        )
    return fate.basis.exit_code == 0


def _serve(
    request: Any,
    w: worker.WorkerView,
    *,
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    now: datetime,
    write_gate: WriteGate,
) -> Any:
    """Run one request's effect and return its result."""
    if isinstance(request, ReadAdapterProfile):
        profile = worker_fate.profile_for(w.adapter_kind)
        return AdapterFacts(
            over_budget=bool(
                profile is not None
                and profile.over_budget is not None
                and profile.over_budget(w, config)
            ),
            can_record_failure=profile is not None and profile.record_failure is not None,
        )
    if isinstance(request, ProbeHealth):
        # Issues #280/#301: corroborate sidecar mtime against real-session
        # activity before deciding whether to kill the worker.
        probe = worker.real_activity_probe_for(w, config, now)
        health = worker.classify_worker_health(w, config, now, probe)
        return HealthProbe(
            probe=probe,
            health=health,
            next_inconclusive_count=worker._next_inconclusive_probe_deferred_count(
                w, probe, health
            ),
        )
    if isinstance(request, ProbeRateLimitDefer):
        return devin_shell.get_rate_limit_defer_until(
            Path(w.log_path),
            config.watchdog.rate_limit_defer_slack_minutes,
            now,
            config.runtime.throttle_error_markers,
            config.runtime.throttle_resume_margin_s,
        )
    if isinstance(request, KillTree):
        # Start-time verification (inside the primitive) prevents PID recycling kills.
        return tuple(write_gate.kill_process_tree(w.pid, w.process_start_time))
    if isinstance(request, KillOrphans):
        # Catches detached/daemonized processes that survived the tree kill.
        orphans = effects_sessions.sweep_orphan_processes(w.worktree_path)
        for orphan in orphans or []:
            write_gate.kill_process(orphan["pid"])
        return tuple(o["pid"] for o in orphans or [])
    if isinstance(request, RecordFailure):
        profile = worker_fate.profile_for(w.adapter_kind)
        assert profile is not None and profile.record_failure is not None  # decided upstream
        return profile.record_failure(
            sessions_dir, w.issue_number, fallback_kind="stalled", config=config, now=now
        )
    if isinstance(request, ProbeCompletedHandoff):
        return _completed_handoff(w, state_file=state_file, sessions_dir=sessions_dir, now=now)
    if isinstance(request, ReadLogTail):
        log_path = Path(w.log_path)
        last_line = None
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                last_line = lines[-1].strip()
        except OSError:
            pass
        return LogTail(
            last_line=last_line,
            mtime=str(datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)),
        )
    raise TypeError(f"unknown stalled request {request!r}")  # pragma: no cover


def _mark_budget_exceeded(sessions_dir: Path, w: worker.WorkerView) -> None:
    """Stamp ``budget_exceeded`` directly (not via the classifier) so a coincidental
    throttle/auth log-tail match cannot override the verdict."""
    from ..claude_code import _sidecar_path as _api_sidecar_path
    from ..claude_code import _write_json_atomic as _api_write_json_atomic

    api_sidecar = _api_sidecar_path(sessions_dir, w.issue_number, "api")
    try:
        with api_sidecar.open("r", encoding="utf-8") as handle:
            api_payload = json.load(handle)
        if isinstance(api_payload, dict):
            api_payload["failure_kind"] = "budget_exceeded"
            _api_write_json_atomic(api_sidecar, api_payload)
    except (OSError, json.JSONDecodeError):
        pass


def _arm_throttle(
    state: dict[str, Any], arm: ThrottleArm, write_gate: WriteGate
) -> dict[str, Any]:
    """``set_throttled_until`` with its ``source`` spelled as a literal at each site
    (the audit trail's grep contract): one branch per source the decision may name."""
    if arm.source == REAP_SOURCE:
        return set_throttled_until(
            state,
            arm.until,
            source="stalled_sessions_reap",
            reason=arm.reason,
            adapter_kind=arm.adapter_kind,
            write_gate=write_gate,
        )
    if arm.source == DEFER_SOURCE:
        return set_throttled_until(
            state,
            arm.until,
            source="stalled_sessions_rate_limit_defer",
            reason=arm.reason,
            adapter_kind=arm.adapter_kind,
            write_gate=write_gate,
        )
    raise ValueError(f"unknown stalled throttle source {arm.source!r}")


def _state_txn(
    commit: StateTxn,
    *,
    state_file: Path,
    now: datetime,
    write_gate: WriteGate,
) -> None:
    """One ``state_lock`` window: throttle, failure stamp and event, saved once."""
    with state_lock(state_file):
        state = load_state(state_file)
        if commit.throttle is not None:
            state = _arm_throttle(state, commit.throttle, write_gate)
        if commit.stamp is not None:
            # #1917: persist the classification on the issue entry so the
            # state.json-keyed orphan sweep can exempt provider-throttle deaths.
            # ``throttled_until=None``: the cooldown (if any) was armed by the
            # throttle txn, which deliberately never stamps a kind.
            state = worker_fate.persist_failure(
                state,
                commit.stamp.issue,
                worker_fate.FailureEvidence(
                    kind=commit.stamp.kind, throttled_until=None, fresh=True
                ),
                adapter_kind=commit.stamp.adapter_kind,
                now=now,
                source="stalled_sessions_reap",
                write_gate=write_gate,
            )
        if commit.event_kind is not None:
            state = write_gate.append_event(
                state,
                # event-consumer: audit-only -- forwards ``commit.event_kind`` unchanged; the consumer
                # or audit-only justification lives at the ``event_kind=`` literal in ``decide_stalled``
                commit.event_kind,
                dict(commit.event_payload),
            )
        write_gate.save_state(state)


def _apply(
    commit: Any,
    w: worker.WorkerView,
    *,
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    now: datetime,
    write_gate: WriteGate,
) -> None:
    if isinstance(commit, StampSidecar):
        worker.update_worker_log_stat(sessions_dir, w, **dict(commit.fields))
    elif isinstance(commit, MarkBudgetExceeded):
        _mark_budget_exceeded(sessions_dir, w)
    elif isinstance(commit, RecordRoleQuota):
        role_quota_ledger.record_for_session(
            sessions_dir,
            w.adapter_kind,
            commit.issue,
            commit.until,
            reason=commit.reason,
            source=commit.source,
        )
    elif isinstance(commit, RecordPostMortem):
        post_mortem.classify_and_record(sessions_dir, config, w, now=now)
    elif isinstance(commit, StateTxn):
        _state_txn(commit, state_file=state_file, now=now, write_gate=write_gate)
    else:  # pragma: no cover - STALLED_COMMIT_TYPES is closed
        raise TypeError(f"unknown stalled commit {commit!r}")


def _violation(write_gate: WriteGate, w: worker.WorkerView, detail: str) -> None:
    logger.error("stalled sweep plan violation for issue %s: %s", w.issue_number, detail)
    write_gate.log_event(
        # event-consumer: audit-only -- a plan violation is a bug in ``decide_stalled``; the
        # error-level row is the audit record and the ``logger.error`` above is the alert
        kind=PLAN_VIOLATION_KIND,
        payload={"issue_number": w.issue_number, "detail": detail[:500]},
        level="error",
    )


def _handle_worker(
    w: worker.WorkerView,
    *,
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    now: datetime,
    write_gate: WriteGate,
) -> dict[str, int] | None:
    facts = StalledFacts(worker=w, config=config, now=now, dry_run=write_gate.dry_run)
    observed: dict[Any, Any] = {}
    applied: list[Any] = []
    for _round in range(MAX_ROUNDS_PER_WORKER):
        plan = decide_stalled(facts, observed)
        if tuple(plan.commits[: len(applied)]) != tuple(applied):
            _violation(write_gate, w, "commit prefix changed between rounds")
            return None
        for commit in plan.commits[len(applied) :]:
            _apply(
                commit,
                w,
                sessions_dir=sessions_dir,
                state_file=state_file,
                config=config,
                now=now,
                write_gate=write_gate,
            )
            applied.append(commit)
        if plan.done:
            return plan.entry
        (request,) = plan.requests
        if request in observed or not isinstance(request, STALLED_REQUEST_TYPES):
            _violation(write_gate, w, f"unservable request {request!r}")
            return None
        observed[request] = _serve(
            request,
            w,
            sessions_dir=sessions_dir,
            state_file=state_file,
            config=config,
            now=now,
            write_gate=write_gate,
        )
    _violation(write_gate, w, f"exceeded {MAX_ROUNDS_PER_WORKER} rounds")
    return None


def run_stalled_sweep(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    *,
    write_gate: WriteGate,
    now: datetime | None = None,
) -> list[dict[str, int]]:
    """Detect stalled sessions (live PID but dead agent) and handle them.

    Returns ``[{"issue", "pid"}, ...]`` for the sessions handled, so the caller
    can exclude them from dispatch in the same pass. ``now`` (issue #828) is the
    injectable clock, sampled once so a single pass never straddles two samples.
    Runs before the dead-session lane, so it applies the same classify-then-
    fallback treatment itself (issue #246); see ``decide_stalled`` for the
    decision table and the module docstring for the write-gate discipline.
    """
    write_gate = require_write_gate(write_gate)
    if not config.watchdog.enabled:
        return []
    if now is None:
        now = datetime.now(UTC)

    stalled_entries: list[dict[str, int]] = []
    for w in worker.iter_workers(sessions_dir):
        entry = _handle_worker(
            w,
            sessions_dir=sessions_dir,
            state_file=state_file,
            config=config,
            now=now,
            write_gate=write_gate,
        )
        if entry is not None:
            stalled_entries.append(entry)
    return stalled_entries
