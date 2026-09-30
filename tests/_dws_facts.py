"""Builders and a driver for the dead-worker sweep decision-table tests.

``decide`` is pure, so a scenario is: build :class:`SweepFacts`, answer each
request the plan asks for from a mapping, and read the trace. Hoisted here (the
``tests/_*.py`` convention) so test modules never import each other.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.dead_worker_sweep import RepoFacts, SweepFacts, SweepPlan, decide

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
STAMP = "2026-09-30T12:00:00Z"
OLD = "2024-01-01T00:00:00Z"


def sweep_config(**watchdog: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, **watchdog),
    )


def dispatched(**fields: Any) -> dict[str, Any]:
    return {
        "status": "dispatched",
        "worker_pid": 99999,
        "dispatched_at": OLD,
        "branch_name": "agent/7-x",
        **fields,
    }


def state_with(issues: Mapping[int, Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"issues": {str(n): dict(e) for n, e in issues.items()}, "prs": {}, **extra}


def make_facts(
    state: Mapping[str, Any],
    *,
    phase: str = "pre",
    alive: Mapping[int, bool] | None = None,
    config: OrchestratorConfig | None = None,
    review_available: bool = True,
    repo: RepoFacts | None = None,
    stamp: str = STAMP,
    now: datetime = NOW,
) -> SweepFacts:
    """Facts for ``phase``; ``locked`` is a deepcopy of ``state`` for lock/post."""
    pid_alive = (
        dict(alive)
        if alive is not None
        else {
            int(k): False
            for k, v in state.get("issues", {}).items()
            if isinstance(v, dict) and v.get("status") == "dispatched"
        }
    )
    return SweepFacts(
        phase=phase,  # type: ignore[arg-type]
        now=now,
        stamp=stamp,
        config=config or sweep_config(),
        snapshot=copy.deepcopy(dict(state)),
        locked=copy.deepcopy(dict(state)) if phase != "pre" else None,
        pid_alive=pid_alive,
        repo=repo or RepoFacts(False, False, False),
        review_available=review_available,
    )


def for_phase(facts: SweepFacts, phase: str) -> SweepFacts:
    locked = copy.deepcopy(dict(facts.snapshot)) if phase != "pre" else None
    return replace(facts, phase=phase, locked=locked)  # type: ignore[arg-type]


Answer = Any  # a result value, or a callable(request) -> result


@dataclass(frozen=True)
class Trace:
    requests: tuple[Any, ...]
    commits: tuple[Any, ...]
    observed: Mapping[Any, Any]

    def request_types(self) -> list[str]:
        return [type(r).__name__ for r in self.requests]

    def commit_types(self) -> list[str]:
        return [type(c).__name__ for c in self.commits]

    def commits_of(self, kind: type) -> list[Any]:
        return [c for c in self.commits if isinstance(c, kind)]

    def emitted(self) -> list[str]:
        return [c.kind for c in self.commits if type(c).__name__ == "Emit"]


def _answer(answers: Mapping[Any, Answer], request: Any) -> Any:
    for key in (request, type(request)):
        if key in answers:
            value = answers[key]
            return value(request) if callable(value) else value
    raise AssertionError(f"scenario has no answer for {request!r}")


def drive(
    facts: SweepFacts,
    answers: Mapping[Any, Answer],
    *,
    observed: dict[Any, Any] | None = None,
    max_rounds: int = 200,
    on_round: Callable[[SweepPlan], None] | None = None,
) -> Trace:
    """Run ``decide`` to a fixed point, answering each request from ``answers``.

    ``answers`` maps an exact request or a request type to a result value (or a
    callable taking the request). ``observed`` is shared across phases by the
    caller, as the shell does.
    """
    seen: list[Any] = []
    observed = {} if observed is None else observed
    plan = decide(facts, observed)
    for _ in range(max_rounds):
        if on_round is not None:
            on_round(plan)
        if not plan.requests:
            return Trace(tuple(seen), plan.commits, observed)
        (request,) = plan.requests
        assert request not in observed, f"decide re-asked {request!r}"
        seen.append(request)
        observed[request] = _answer(answers, request)
        plan = decide(facts, observed)
    raise AssertionError("decide did not converge")


def drive_all_phases(
    pre_facts: SweepFacts, answers: Mapping[Any, Answer]
) -> tuple[Trace, Trace | None, Trace | None]:
    """Pre, then lock, then post, sharing one ``observed`` map like the shell."""
    from charlie_work.dead_worker_sweep.model import FetchOpenPrs

    observed: dict[Any, Any] = {}
    pre = drive(pre_facts, answers, observed=observed)
    if FetchOpenPrs() not in observed:
        return pre, None, None
    lock = drive(for_phase(pre_facts, "lock"), answers, observed=observed)
    post = drive(for_phase(pre_facts, "post"), answers, observed=observed)
    return pre, lock, post


@dataclass(frozen=True)
class ReapRun:
    """Result of one timed-backstop (#654) decision for a single dead entry."""

    reaped: bool
    events: tuple[tuple[str, Mapping[str, Any]], ...]
    commits: tuple[Any, ...]
    entry: Mapping[str, Any]  # the draft after the flow: re-arm mutations land here


def run_reap(
    entry: Mapping[str, Any],
    *,
    issue: int = 42,
    pr_data: Mapping[str, Any] | None = None,
    reap_minutes: float = 60,
    max_rearms: int = 0,
    throttled_until: str | None = None,
    local_park_deferred: Mapping[int, str] | None = None,
    now: datetime = NOW,
    exit_code: int | None = None,
) -> ReapRun:
    """Drive ``reap_flow`` (the dead-dispatched backstop) for one dispatched entry.

    Replaces direct calls to the old ``maybe_reap_dead_dispatched_worker``: the
    decision is pure, so the row is "given this entry, PR, throttle window and
    clock, the backstop escalates / re-arms / defers / stays quiet".
    """
    from charlie_work.dead_worker_sweep.decide_common import Draft, LockAcc, run_flow
    from charlie_work.dead_worker_sweep.decide_reap import reap_flow
    from charlie_work.dead_worker_sweep.model import (
        Emit,
        PreOutcome,
        ReadTerminal,
        TerminalFacts,
    )

    state = state_with({issue: entry})
    facts = make_facts(
        state,
        phase="lock",
        config=sweep_config(
            dead_dispatched_reap_minutes=reap_minutes, max_auto_redispatch=max_rearms
        ),
        now=now,
    )
    pre = PreOutcome(
        orphans=(issue,),
        no_pr_orphans=(),
        pr_by_issue={} if pr_data is None else {issue: pr_data},
        details={},
        candidates={},
        escalations={},
        deferred=dict(local_park_deferred or {}),
        reclaim_results={},
        unreviewed={},
        live_candidates={},
        pr_already_open={},
        salvage_events=(),
        heads={},
        early_exit=False,
        now=now,
    )
    draft = Draft(issue, entry)
    acc = LockAcc(throttled_until=throttled_until)
    observed = {ReadTerminal(issue): TerminalFacts(exit_code=exit_code, duration_seconds=None)}
    reaped, commits, pending = run_flow(reap_flow(facts, pre, draft, pr_data, acc), observed)
    assert pending is None, f"reap flow asked for an unanswered request: {pending!r}"
    events = tuple((c.kind, c.payload) for c in commits if isinstance(c, Emit))
    return ReapRun(bool(reaped), events, commits, dict(draft.work))
