"""The dead-worker sweep's apply shell: ``run_orphan_sweep``.

The shell owns every side effect. For each phase (``pre``, ``lock``, ``post``) it
loops: ask ``decide`` what to do given the results observed so far, apply the new
commits, run the one request the plan still needs, record its result, decide
again. ``decide`` is pure and replays from the start every round, so the shell
only has to keep the observed-results map and the count of commits it has
already applied.

Two gates protect the loop. A *plan violation* (the commit prefix changed
between rounds, or a request illegal in the phase) aborts the phase before any
save; a request that comes back unchanged (``no_progress``) does the same. Both
log an error-level event and end the sweep with nothing half-written.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import stalled_review_reap
from ..config import OrchestratorConfig
from ..paths import resolved_layout
from ..write_gate import WriteGate, require_write_gate
from .apply_commits import LOCK_ILLEGAL_COMMITS, POST_ILLEGAL_COMMITS, apply_commit
from .apply_context import SweepContext
from .apply_requests_lock import serve
from .decide import PhaseOrderError, decide
from .model import (
    LOCK_LEGAL,
    POST_LEGAL,
    REQUEST_TYPES,
    FetchOpenPrs,
    Phase,
    RepoFacts,
    SweepFacts,
)
from .ports import SweepPorts, ports_from_workflow

logger = logging.getLogger(__name__)

# Far above any real sweep (a handful of requests per dead worker); a backstop
# against a decide/apply loop that never converges, not a tuning knob.
MAX_ROUNDS_PER_PHASE = 5000

PLAN_VIOLATION_KIND = "dead_worker_sweep_plan_violation"
NO_PROGRESS_KIND = "dead_worker_sweep_no_progress"

_LOCK_ONLY = tuple(t for t in LOCK_LEGAL if t.__name__ != "ReadReviewDecision")
PRE_LEGAL = tuple(t for t in REQUEST_TYPES if t not in POST_LEGAL and t not in _LOCK_ONLY)
_LEGAL: dict[str, tuple[type, ...]] = {
    "pre": PRE_LEGAL,
    "lock": LOCK_LEGAL,
    "post": POST_LEGAL,
}


class SweepAborted(Exception):
    """A phase gate tripped; the sweep ends without saving anything further."""


@dataclass
class _Run:
    ctx: SweepContext
    base_facts: SweepFacts
    observed: dict[Any, Any]


def _abort(ctx: SweepContext, kind: str, phase: str, detail: str) -> SweepAborted:
    logger.error("dead-worker sweep %s in %s phase: %s", kind, phase, detail)
    ctx.write_gate.log_event(
        # event-consumer: audit-only -- a sweep abort is a decide-module bug; the error-level
        # row is the audit record and the ``logger.error`` above is the alert
        kind=kind,
        payload={"phase": phase, "detail": detail[:500]},
        level="error",
    )
    return SweepAborted(f"{kind}: {detail}")


def _run_phase(run: _Run, facts: SweepFacts, sweep_events: list[Any]) -> None:
    ctx = run.ctx
    phase = facts.phase
    legal = _LEGAL[phase]
    applied: list[Any] = []
    for _round in range(MAX_ROUNDS_PER_PHASE):
        try:
            plan = decide(facts, run.observed)
        except PhaseOrderError as exc:  # a decide bug: abort loudly like any other gate
            raise _abort(ctx, PLAN_VIOLATION_KIND, phase, f"PhaseOrderError: {exc}") from exc
        if tuple(plan.commits[: len(applied)]) != tuple(applied):
            raise _abort(ctx, PLAN_VIOLATION_KIND, phase, "commit prefix changed between rounds")
        for commit in plan.commits[len(applied) :]:
            if phase == "lock" and isinstance(commit, LOCK_ILLEGAL_COMMITS):
                raise _abort(
                    ctx, PLAN_VIOLATION_KIND, phase, f"{type(commit).__name__} illegal in lock"
                )
            if phase == "post" and isinstance(commit, POST_ILLEGAL_COMMITS):
                raise _abort(
                    ctx, PLAN_VIOLATION_KIND, phase, f"{type(commit).__name__} illegal in post"
                )
            apply_commit(ctx, phase, commit, sweep_events)
            applied.append(commit)
        if not plan.requests:
            return
        (request,) = plan.requests
        if request in run.observed:
            raise _abort(ctx, NO_PROGRESS_KIND, phase, f"{request!r} already observed")
        if not isinstance(request, legal):
            raise _abort(
                ctx, PLAN_VIOLATION_KIND, phase, f"{type(request).__name__} illegal in {phase}"
            )
        run.observed[request] = serve(ctx, request)
    raise _abort(ctx, NO_PROGRESS_KIND, phase, f"exceeded {MAX_ROUNDS_PER_PHASE} rounds")


def _facts(run: _Run, phase: Phase, locked: Any) -> SweepFacts:
    base = run.base_facts
    stamp = run.ctx.ports.utc_now()
    run.ctx.stamp = stamp  # handlers that write stamps (credit) share the phase's stamp
    return SweepFacts(
        phase=phase,
        now=base.now,
        stamp=stamp,
        config=base.config,
        snapshot=base.snapshot,
        locked=locked,
        pid_alive=base.pid_alive,
        repo=base.repo,
        review_available=base.review_available,
    )


def _pid_liveness(state: dict[str, Any], ports: SweepPorts) -> dict[int, bool]:
    alive: dict[int, bool] = {}
    for key, entry in (state.get("issues") or {}).items():
        if isinstance(entry, dict) and entry.get("status") == "dispatched":
            alive[int(key)] = bool(ports.worker_pid_alive(entry))
    return alive


def run_orphan_sweep(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh: Any,
    *,
    write_gate: WriteGate,
    review_callback: Callable[[int], Any] | None = None,
    record_review_callback: Callable[..., Any] | None = None,
    enrich_checks_callback: Callable[..., list[dict[str, Any]]] | None = None,
    fleet_dir_override: str | None = None,
    ports: SweepPorts | None = None,
) -> None:
    """Detect and handle orphaned workers (same contract as the original sweep)."""
    write_gate = require_write_gate(write_gate)
    ports = ports or ports_from_workflow()

    with ports.state_lock(state_file):
        state = ports.load_state(state_file)

    now = datetime.now(UTC)
    repo_root = getattr(gh, "repo_root", None)
    worktrees_dir = resolved_layout(config, repo_root).worktrees if repo_root is not None else None
    ctx = SweepContext(
        sessions_dir=sessions_dir,
        state_file=state_file,
        config=config,
        gh=gh,
        write_gate=write_gate,
        ports=ports,
        review_callback=review_callback,
        record_review_callback=record_review_callback,
        enrich_checks_callback=enrich_checks_callback,
        fleet_dir_override=fleet_dir_override,
        repo_root=repo_root,
        worktrees_dir=worktrees_dir,
        state=state,
        now=now,
        pid_alive=_pid_liveness(state, ports),
    )
    base = SweepFacts(
        phase="pre",
        now=now,
        stamp="",
        config=config,
        snapshot=copy.deepcopy(state),
        locked=None,
        pid_alive=dict(ctx.pid_alive),
        repo=RepoFacts(
            has_repo_root=repo_root is not None,
            has_worktrees=worktrees_dir is not None,
            repo_root_is_path=isinstance(repo_root, Path),
        ),
        review_available=review_callback is not None,
    )
    run = _Run(ctx=ctx, base_facts=base, observed={})
    try:
        _run_sweep(run)
    except SweepAborted:
        return


def _run_sweep(run: _Run) -> None:
    ctx = run.ctx
    _run_phase(run, _facts(run, "pre", None), [])
    if FetchOpenPrs() not in run.observed:
        return  # the pre flow early-exited: nothing dead, nothing to hand off

    with ctx.ports.state_lock(ctx.state_file):
        ctx.state = ctx.ports.load_state(ctx.state_file)
        locked = copy.deepcopy(ctx.state)
        sweep_events: list[Any] = []
        _run_phase(run, _facts(run, "lock", locked), sweep_events)
        ctx.state = stalled_review_reap._append_sweep_events(
            ctx.state,
            sweep_events,
            max_size=ctx.config.runtime.event_ring_size,
            state_file=ctx.state_file,
            write_gate=ctx.write_gate,
        )
        ctx.write_gate.save_state(ctx.state)

    _run_phase(run, _facts(run, "post", locked), [])
