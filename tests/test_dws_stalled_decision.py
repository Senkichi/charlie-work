"""Decision-table tests for the stalled-session lane, through ``decide_stalled``.

``decide_stalled(facts, observed)`` is pure: each row builds one worker's facts,
answers the requests the plan asks for from a mapping keyed by request type, and
asserts the commits and the returned entry. (Named ``test_dws_*`` so the dormant
module guard does not mistake this for a one-module-one-test file.)
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
from _dws_facts import NOW, sweep_config
from charlie_work.dead_worker_sweep.decide_stalled import decide_stalled
from charlie_work.dead_worker_sweep.stalled_model import (
    AdapterFacts,
    HealthProbe,
    KillOrphans,
    KillTree,
    LogTail,
    MarkBudgetExceeded,
    ProbeHealth,
    ProbeRateLimitDefer,
    ReadAdapterProfile,
    ReadLogTail,
    RecordFailure,
    RecordPostMortem,
    StalledFacts,
    StampSidecar,
    StateTxn,
)
from charlie_work.worker import WorkerHealth, WorkerView

ISSUE = 7


def _worker(**kw: Any) -> WorkerView:
    base: dict[str, Any] = {
        "adapter_kind": "devin",
        "issue_number": ISSUE,
        "repo_key": "",
        "pid": 4242,
        "started_at": "2026-09-30T10:00:00Z",
        "process_start_time": 111.0,
        "log_path": "x.log",
        "worktree_path": "wt",
        "error": None,
        "failure_kind": None,
        "reclaimed": None,
    }
    return WorkerView(**{**base, **kw})


def _facts(*, dry_run: bool = False, config=None, **worker_kw: Any) -> StalledFacts:
    return StalledFacts(
        worker=_worker(**worker_kw),
        config=config or sweep_config(),
        now=NOW,
        dry_run=dry_run,
    )


def _answers(
    *,
    health: WorkerHealth = WorkerHealth.STALLED,
    over_budget: bool = False,
    can_record: bool = True,
    defer: str | None = None,
    failure: tuple[str | None, str | None] = ("stalled", None),
    next_count: int = 0,
) -> dict[type, Any]:
    return {
        ReadAdapterProfile: AdapterFacts(over_budget, can_record),
        ProbeHealth: HealthProbe(None, health, next_count),
        ProbeRateLimitDefer: defer,
        KillTree: (11, 12),
        KillOrphans: (13,),
        RecordFailure: failure,
        ReadLogTail: LogTail("last line", "2026-09-30 11:00:00+00:00"),
    }


def _drive(facts: StalledFacts, answers: dict[type, Any]):
    """Answer requests until done; returns ``(plan, requests_in_order)``."""
    observed: dict[Any, Any] = {}
    asked: list[Any] = []
    for _ in range(32):
        plan = decide_stalled(facts, observed)
        if plan.done:
            return plan, asked
        (request,) = plan.requests
        assert request not in observed, f"request {request!r} asked twice"
        asked.append(request)
        observed[request] = answers[type(request)]
    raise AssertionError("decide_stalled did not converge")


def _kinds(plan) -> list[str]:
    return [c.event_kind for c in plan.commits if isinstance(c, StateTxn) and c.event_kind]


def test_a_worker_without_a_pid_or_with_an_error_is_left_alone() -> None:
    for kw in ({"pid": None}, {"error": "boom"}):
        plan, asked = _drive(_facts(**kw), _answers())
        assert (plan.entry, plan.commits, asked) == (None, (), [])


def test_a_healthy_worker_only_gets_its_sidecar_stamped() -> None:
    plan, asked = _drive(_facts(), _answers(health=WorkerHealth.HEALTHY, next_count=2))
    assert plan.entry is None
    assert plan.commits == (
        StampSidecar(ISSUE),
        StampSidecar(ISSUE, (("inconclusive_probe_deferred_count", 2),)),
    )
    assert [type(r) for r in asked] == [ReadAdapterProfile, ProbeHealth]


def test_a_stalled_worker_is_reaped_with_failure_stamp_and_event() -> None:
    plan, asked = _drive(_facts(), _answers())
    assert plan.entry == {"issue": ISSUE, "pid": 4242}
    assert [type(r) for r in asked] == [
        ReadAdapterProfile,
        ProbeHealth,
        ProbeRateLimitDefer,
        KillTree,
        KillOrphans,
        RecordFailure,
        ReadLogTail,
    ]
    assert RecordPostMortem(ISSUE) in plan.commits
    assert _kinds(plan) == ["session_stalled"]
    txn = plan.commits[-1]
    assert txn.stamp is not None and txn.stamp.kind == "stalled"
    payload = dict(txn.event_payload)
    assert payload["killed_pids"] == [11, 12, 13]
    assert payload["orphan_pids"] == [13]
    assert payload["failure_kind"] == "stalled"


def test_a_dead_worker_logs_session_exited_and_skips_the_rate_limit_probe() -> None:
    plan, asked = _drive(_facts(), _answers(health=WorkerHealth.DEAD))
    assert _kinds(plan) == ["session_exited"]
    assert ProbeRateLimitDefer not in [type(r) for r in asked]


def test_a_throttle_failure_arms_the_cooldown_before_the_reap_event() -> None:
    plan, _ = _drive(_facts(), _answers(failure=("rate_limited", "2026-09-30T13:00:00Z")))
    arm, reap = plan.commits[-2], plan.commits[-1]
    assert arm.throttle is not None and arm.throttle.until == "2026-09-30T13:00:00Z"
    assert arm.throttle.source == "stalled_sessions_reap"
    assert arm.event_kind is None
    assert reap.event_kind == "session_stalled" and reap.stamp.kind == "rate_limited"


def test_no_adapter_failure_recorder_means_no_stamp() -> None:
    plan, asked = _drive(_facts(), _answers(can_record=False))
    assert RecordFailure not in [type(r) for r in asked]
    assert plan.commits[-1].stamp is None


def test_a_rate_limit_signature_defers_the_kill() -> None:
    until = "2026-09-30T12:10:00Z"
    plan, asked = _drive(_facts(), _answers(defer=until))
    assert plan.entry is None
    assert KillTree not in [type(r) for r in asked]
    assert StampSidecar(ISSUE, (("rate_limit_defer_until", until),)) in plan.commits
    txn = plan.commits[-1]
    assert txn.event_kind == "session_rate_limit_deferred"
    assert txn.throttle.source == "stalled_sessions_rate_limit_defer"
    assert txn.throttle.reason == "rate_limited"


def test_an_unexpired_stored_defer_window_leaves_the_worker_alone() -> None:
    plan, asked = _drive(_facts(rate_limit_defer_until="2026-09-30T12:30:00Z"), _answers())
    assert plan.entry is None
    assert KillTree not in [type(r) for r in asked]


def test_an_expired_stored_defer_window_does_not_block_the_reap() -> None:
    plan, _ = _drive(_facts(rate_limit_defer_until="2026-09-30T11:00:00Z"), _answers())
    assert plan.entry == {"issue": ISSUE, "pid": 4242}


def test_disabling_rate_limit_defer_skips_the_probe() -> None:
    cfg = sweep_config(rate_limit_defer_enabled=False)
    _, asked = _drive(_facts(config=cfg), _answers(defer="2026-09-30T12:10:00Z"))
    assert ProbeRateLimitDefer not in [type(r) for r in asked]


def test_an_over_budget_worker_is_killed_and_marked_before_any_health_verdict() -> None:
    plan, asked = _drive(_facts(), _answers(over_budget=True, health=WorkerHealth.HEALTHY))
    assert plan.entry == {"issue": ISSUE, "pid": 4242}
    assert MarkBudgetExceeded(ISSUE) in plan.commits
    assert _kinds(plan) == ["session_budget_exceeded"]
    assert RecordFailure not in [type(r) for r in asked]


def test_dry_run_withholds_sidecar_and_classification_commits() -> None:
    plan, asked = _drive(_facts(dry_run=True), _answers())
    assert plan.entry == {"issue": ISSUE, "pid": 4242}
    assert all(isinstance(c, StateTxn) for c in plan.commits)
    assert RecordFailure not in [type(r) for r in asked]
    assert plan.commits[-1].stamp is None
    # Kills and the reap event still go through the gate (which suppresses them itself).
    assert KillTree in [type(r) for r in asked] and _kinds(plan) == ["session_stalled"]


def test_dry_run_over_budget_skips_only_the_sidecar_mark() -> None:
    plan, _ = _drive(_facts(dry_run=True), _answers(over_budget=True))
    assert MarkBudgetExceeded(ISSUE) not in plan.commits
    assert _kinds(plan) == ["session_budget_exceeded"]


@pytest.mark.parametrize("health", [WorkerHealth.STALLED, WorkerHealth.DEAD, WorkerHealth.HEALTHY])
def test_decide_stalled_is_deterministic_and_does_not_mutate_its_inputs(health) -> None:
    facts = _facts()
    answers = _answers(health=health)
    first, _ = _drive(facts, answers)
    frozen = copy.deepcopy(answers)
    second, _ = _drive(replace(facts), answers)
    assert first == second
    assert answers == frozen


def test_commits_are_a_growing_prefix_as_results_arrive() -> None:
    facts, answers = _facts(), _answers()
    observed: dict[Any, Any] = {}
    previous: tuple = ()
    while True:
        plan = decide_stalled(facts, observed)
        assert plan.commits[: len(previous)] == previous
        previous = plan.commits
        if plan.done:
            break
        (request,) = plan.requests
        observed[request] = answers[type(request)]
