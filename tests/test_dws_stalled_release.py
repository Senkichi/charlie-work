"""The stalled lane's one-shot release of a tool call blocked on a pipe holder (#2794).

``decide_stalled`` is pure, so each row answers the requests the plan asks for and
reads the trace. The question every row settles is *kill the session or leave it
running*: a STALLED opencode worker whose pipe holders could be killed is released
once and left alone; everything else keeps the pre-#2794 reap. (Named ``test_dws_*``
so the dormant-module guard does not read this as a one-module-one-test file.)
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from _dws_facts import NOW, sweep_config
from charlie_work.dead_worker_sweep.decide_stalled import RELEASE_RESUME_GRACE, decide_stalled
from charlie_work.dead_worker_sweep.stalled_model import (
    AdapterFacts,
    HealthProbe,
    KillOrphans,
    KillTree,
    LogTail,
    ProbeCompletedHandoff,
    ProbeHealth,
    ProbeRateLimitDefer,
    ReadAdapterProfile,
    ReadLogTail,
    RecordFailure,
    ReleaseBlockedTool,
    StalledFacts,
    StampSidecar,
    StateTxn,
)
from charlie_work.opencode_config import OpenCodeConfig
from charlie_work.worker import WorkerHealth, WorkerView

ISSUE = 7
RELEASED_PIDS = (501, 502)


def _worker(**kw: Any) -> WorkerView:
    base: dict[str, Any] = {
        "adapter_kind": "opencode",
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


def _facts(
    *, dry_run: bool = False, release_enabled: bool = True, **worker_kw: Any
) -> StalledFacts:
    config = replace(sweep_config(), opencode=OpenCodeConfig(release_blocked_tool=release_enabled))
    return StalledFacts(worker=_worker(**worker_kw), config=config, now=NOW, dry_run=dry_run)


def _answers(
    *,
    health: WorkerHealth = WorkerHealth.STALLED,
    can_release: bool = True,
    released: tuple[int, ...] = RELEASED_PIDS,
    defer: str | None = None,
) -> dict[type, Any]:
    return {
        ReadAdapterProfile: AdapterFacts(False, True, can_release),
        ProbeHealth: HealthProbe(None, health, 0),
        ProbeRateLimitDefer: defer,
        ReleaseBlockedTool: released,
        KillTree: (11,),
        KillOrphans: (13,),
        RecordFailure: ("stalled", None),
        ProbeCompletedHandoff: False,
        ReadLogTail: LogTail("last line", "2026-09-30 11:00:00+00:00"),
    }


def _drive(facts: StalledFacts, answers: dict[type, Any]):
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


def _asked_types(asked: list[Any]) -> list[type]:
    return [type(r) for r in asked]


def _is_reaped(plan, asked: list[Any]) -> bool:
    return (
        plan.entry == {"issue": ISSUE, "pid": 4242}
        and KillTree(ISSUE) in asked
        and "session_stalled" in _kinds(plan)
    )


def test_a_stalled_worker_with_pipe_holders_is_released_not_reaped() -> None:
    plan, asked = _drive(_facts(), _answers())

    assert plan.entry is None
    assert ReleaseBlockedTool(ISSUE) in asked
    assert KillTree(ISSUE) not in asked and KillOrphans(ISSUE) not in asked
    assert _kinds(plan) == ["session_tool_released"]
    assert StampSidecar(ISSUE, (("tool_released_at", NOW.isoformat()),)) in plan.commits
    payload = dict(next(c for c in plan.commits if isinstance(c, StateTxn)).event_payload)
    assert payload["released_pids"] == list(RELEASED_PIDS)
    assert payload["pid"] == 4242 and payload["issue_number"] == ISSUE


def test_no_holder_to_release_falls_through_to_the_ordinary_reap() -> None:
    plan, asked = _drive(_facts(), _answers(released=()))

    assert _is_reaped(plan, asked)
    assert _asked_types(asked).index(ReleaseBlockedTool) < _asked_types(asked).index(KillTree)
    assert "session_tool_released" not in _kinds(plan)
    assert not any(
        isinstance(c, StampSidecar) and ("tool_released_at", NOW.isoformat()) in c.fields
        for c in plan.commits
    )


def test_a_worker_released_moments_ago_is_left_to_resume() -> None:
    released_at = (NOW - RELEASE_RESUME_GRACE + timedelta(seconds=30)).isoformat()
    plan, asked = _drive(_facts(tool_released_at=released_at), _answers())

    assert plan.entry is None
    assert ReleaseBlockedTool not in _asked_types(asked), "release is one-shot per dispatch"
    assert KillTree(ISSUE) not in asked
    assert _kinds(plan) == []


def test_a_worker_still_stalled_after_the_resume_grace_is_reaped() -> None:
    """The release did not get it moving: the pre-#2794 reap applies, without a second release."""
    released_at = (NOW - RELEASE_RESUME_GRACE - timedelta(seconds=30)).isoformat()
    plan, asked = _drive(_facts(tool_released_at=released_at), _answers())

    assert _is_reaped(plan, asked)
    assert ReleaseBlockedTool not in _asked_types(asked)


def test_an_unparseable_release_stamp_does_not_wedge_the_worker_as_released() -> None:
    plan, asked = _drive(_facts(tool_released_at="not-a-timestamp"), _answers())

    assert ReleaseBlockedTool(ISSUE) in asked
    assert plan.entry is None


def test_the_kill_switch_restores_the_outright_reap() -> None:
    plan, asked = _drive(_facts(release_enabled=False), _answers())

    assert _is_reaped(plan, asked)
    assert ReleaseBlockedTool not in _asked_types(asked)


def test_a_harness_without_the_capability_is_reaped_without_asking_to_release() -> None:
    plan, asked = _drive(_facts(adapter_kind="devin"), _answers(can_release=False))

    assert _is_reaped(plan, asked)
    assert ReleaseBlockedTool not in _asked_types(asked)


@pytest.mark.parametrize("health", [WorkerHealth.DEAD, WorkerHealth.HEALTHY])
def test_only_a_stalled_worker_is_ever_released(health: WorkerHealth) -> None:
    plan, asked = _drive(_facts(), _answers(health=health))

    assert ReleaseBlockedTool not in _asked_types(asked)
    assert "session_tool_released" not in _kinds(plan)


def test_a_rate_limit_defer_wins_over_a_release() -> None:
    """A provider throttle explains the silence; nothing is held, nothing is killed."""
    plan, asked = _drive(_facts(), _answers(defer="2026-09-30T13:00:00Z"))

    assert plan.entry is None
    assert ReleaseBlockedTool not in _asked_types(asked)
    assert "session_rate_limit_deferred" in _kinds(plan)


def test_dry_run_records_the_release_event_but_stamps_no_sidecar() -> None:
    plan, asked = _drive(_facts(dry_run=True), _answers())

    assert ReleaseBlockedTool(ISSUE) in asked
    assert plan.entry is None
    assert _kinds(plan) == ["session_tool_released"]
    assert not any(isinstance(c, StampSidecar) for c in plan.commits)
