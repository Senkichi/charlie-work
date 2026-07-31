"""Tests for host-wide elastic runner-slot allocation.

Covers the pure allocator (runner_allocation.py), the host/GitHub layer
(runner_slots.py), and the pass that wires them together
(runner_allocation_pass.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from charlie_work.config import RunnerAllocationConfig
from charlie_work.instrumentation import close_db, query_events
from charlie_work.runner_allocation import (
    AllocationPlan,
    RepoDemand,
    RepoTarget,
    RunnerInstance,
    SlotAction,
    SlotChange,
    allocate_slots,
    annotate_busy,
    derive_budget,
    next_idle_streaks,
    plan_allocation,
    plan_summary,
    repo_slug_from_github_url,
    runner_capacity_starved_events,
)
from charlie_work.runner_allocation_pass import resolve_inputs, run_allocation_pass
from charlie_work.runner_slots import (
    apply_allocation,
    discover_runner_instances,
    load_idle_streaks,
    load_tie_break_offset,
    park_runner_slot,
    save_idle_streaks,
)


CW = "Senkichi/charlie-work"
JC = "Senkichi/job-cannon"
PUB = "Senkichi/jobcannon"


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


def test_value_objects_are_frozen() -> None:
    """Allocation value objects follow the project's frozen-dataclass rule."""
    instance = RunnerInstance(path=Path("x"), name="n", repo=CW, running=True)
    with pytest.raises(Exception):
        instance.running = False  # type: ignore[misc]

    demand = RepoDemand(repo=CW)
    with pytest.raises(Exception):
        demand.queued_jobs = 3  # type: ignore[misc]

    plan = AllocationPlan(budget=1, budget_reason="r", targets=(), changes=())
    with pytest.raises(Exception):
        plan.budget = 2  # type: ignore[misc]


def test_repo_demand_sums_queued_and_in_progress() -> None:
    """Demand counts both jobs waiting for a slot and jobs holding one."""
    assert RepoDemand(repo=CW, queued_jobs=4, in_progress_jobs=2).demand == 6


# --------------------------------------------------------------------------
# repo_slug_from_github_url
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Senkichi/charlie-work", CW),
        ("https://github.com/Senkichi/charlie-work/", CW),
        ("https://github.com/Senkichi/charlie-work.git", CW),
        ("https://ghe.example.com/Senkichi/charlie-work", CW),
        ("https://github.com/Senkichi", None),
        ("", None),
    ],
)
def test_repo_slug_from_github_url(url: str, expected: str | None) -> None:
    """Repo ownership is derived from the runner's own registration URL."""
    assert repo_slug_from_github_url(url) == expected


# --------------------------------------------------------------------------
# derive_budget
# --------------------------------------------------------------------------


def test_derive_budget_prefers_configured_value() -> None:
    budget, reason = derive_budget(6, cpu_count=32)
    assert budget == 6
    assert "configured" in reason


def test_derive_budget_falls_back_to_half_the_cores() -> None:
    budget, reason = derive_budget(0, cpu_count=16)
    assert budget == 8
    assert "16 logical cores" in reason


def test_derive_budget_never_returns_zero() -> None:
    """A single-core host still gets one slot."""
    assert derive_budget(0, cpu_count=1)[0] == 1


# --------------------------------------------------------------------------
# allocate_slots — the pure policy core
# --------------------------------------------------------------------------


def test_idle_repo_capacity_flows_to_the_busy_repo() -> None:
    """The whole point: a saturated repo gets slots an idle repo isn't using.

    Under fixed per-repo allocation, charlie-work would hold 2 slots while
    job-cannon queued behind its own 2. Here job-cannon takes the slack.
    """
    targets = allocate_slots(
        demands={CW: 0, JC: 6},
        capacities={CW: 4, JC: 5},
        budget=6,
        min_per_repo=1,
    )
    assert targets[JC] == 5
    assert targets[CW] == 1


def test_no_repo_exceeds_its_registered_runner_count() -> None:
    """Capacity is a hard ceiling — you cannot run a runner that isn't configured."""
    targets = allocate_slots(
        demands={CW: 0, JC: 40},
        capacities={CW: 2, JC: 3},
        budget=10,
        min_per_repo=1,
    )
    assert targets[JC] == 3
    assert targets[CW] == 1


def test_spare_budget_is_left_unused_when_demand_is_low() -> None:
    """Idle listeners nobody needs are not worth their RAM."""
    targets = allocate_slots(
        demands={CW: 0, JC: 0},
        capacities={CW: 4, JC: 5},
        budget=8,
        min_per_repo=1,
    )
    assert targets == {CW: 1, JC: 1}


def test_every_repo_keeps_a_floor_so_its_queue_is_never_unclaimed() -> None:
    """A repo with zero demand still keeps a listener online.

    With every listener offline, a queued job waits for one to come back
    instead of starting — the floor is what keeps pickup latency at zero.
    """
    targets = allocate_slots(
        demands={CW: 0, JC: 99, PUB: 0},
        capacities={CW: 2, JC: 5, PUB: 1},
        budget=7,
        min_per_repo=1,
    )
    assert targets[CW] >= 1
    assert targets[PUB] >= 1
    assert sum(targets.values()) <= 7


def test_floor_is_capped_by_capacity() -> None:
    """A repo with no registered runners cannot be given a floor."""
    targets = allocate_slots(
        demands={CW: 5},
        capacities={CW: 0},
        budget=4,
        min_per_repo=2,
    )
    assert targets[CW] == 0


def test_two_hungry_repos_share_fairly_rather_than_first_come_first_served() -> None:
    """Water-filling: a burst in one repo cannot starve another that has work."""
    targets = allocate_slots(
        demands={CW: 10, JC: 10},
        capacities={CW: 5, JC: 5},
        budget=6,
        min_per_repo=1,
    )
    assert targets[CW] == 3
    assert targets[JC] == 3


def test_unequal_demand_splits_by_need_not_evenly() -> None:
    targets = allocate_slots(
        demands={CW: 1, JC: 10},
        capacities={CW: 5, JC: 5},
        budget=6,
        min_per_repo=1,
    )
    assert targets[CW] == 1
    assert targets[JC] == 5


def test_budget_smaller_than_the_floors_serves_the_hungriest_first() -> None:
    """When the host cannot host one listener per repo, work wins over fairness."""
    targets = allocate_slots(
        demands={CW: 0, JC: 8, PUB: 0},
        capacities={CW: 2, JC: 5, PUB: 1},
        budget=1,
        min_per_repo=1,
    )
    assert targets[JC] == 1
    assert sum(targets.values()) == 1


def test_allocation_never_overcommits_the_budget() -> None:
    for budget in range(0, 12):
        targets = allocate_slots(
            demands={CW: 3, JC: 7, PUB: 2},
            capacities={CW: 4, JC: 5, PUB: 1},
            budget=budget,
            min_per_repo=1,
        )
        assert sum(targets.values()) <= budget, f"overcommitted at budget={budget}"


def test_pinned_repos_hold_their_slots_and_shrink_the_elastic_budget() -> None:
    """An unmeasurable repo keeps what it has; the rest share what's left."""
    targets = allocate_slots(
        demands={CW: 0, JC: 9},
        capacities={CW: 4, JC: 5},
        budget=6,
        min_per_repo=1,
        pinned={CW: 3},
    )
    assert targets[CW] == 3
    assert targets[JC] == 3


def test_pins_can_exceed_the_budget_but_never_compound_it() -> None:
    """Pins are the only path to an over-budget host — and they stop there.

    Two unmeasurable repos holding 3 slots each add up to 6 against a budget of
    4. The allocator cannot park them (unknown demand could mean unseen work),
    but it must not hand the elastic repo slots on top of the overcommit.
    """
    targets = allocate_slots(
        demands={CW: 0, JC: 0, PUB: 5},
        capacities={CW: 3, JC: 3, PUB: 4},
        budget=4,
        min_per_repo=1,
        pinned={CW: 3, JC: 3},
    )
    assert targets == {CW: 3, JC: 3, PUB: 0}
    assert sum(targets.values()) == 6  # over budget, and entirely by pins


def test_over_budget_pins_are_reported_to_the_operator() -> None:
    """An over-budget host must not read as a healthy one."""
    plan = plan_allocation(
        _instances(
            {
                CW: [("cw-1", True, False), ("cw-2", True, False), ("cw-3", True, False)],
                JC: [("jc-1", True, False), ("jc-2", True, False), ("jc-3", True, False)],
            }
        ),
        {
            CW: RepoDemand(CW, ok=False, error="403"),
            JC: RepoDemand(JC, ok=False, error="403"),
        },
        budget=4,
        budget_reason="configured",
        min_per_repo=1,
        idle_streaks={},
        demand_idle_samples=3,
    )
    assert plan.changes == ()  # nothing moves while demand is unreadable
    assert any("above the 4-slot budget" in note for note in plan.notes)


def test_over_budget_from_busy_listeners_is_reported() -> None:
    """A plan whose targets fit the budget can still push running over it.

    Reproduced from issue #601: A(cap 3, 2 running both busy, demand 0),
    B(cap 5, 1 running, demand 10), budget 3. Targets A=1 B=2 sum to 3, so
    the old sum-of-targets check never fired. But A's two busy listeners
    cannot be parked, while B starts one more — running goes 3 -> 4. The
    post-plan running total must be the warn condition, not the target sum.
    """
    plan = plan_allocation(
        _instances(
            {
                CW: [("cw-1", True, True), ("cw-2", True, True), ("cw-3", False, False)],
                JC: [("jc-1", True, False), ("jc-2", False, False), ("jc-3", False, False)],
            }
        ),
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=10)},
        budget=3,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 99},
        demand_idle_samples=3,
    )
    # B starts one listener; A's busy listeners stay running.
    starts = [c for c in plan.changes if c.action is SlotAction.START]
    assert len(starts) == 1 and starts[0].repo == JC
    parks = [c for c in plan.changes if c.action is SlotAction.PARK]
    assert parks == []  # A's surplus listeners are busy — cannot park
    # Post-plan running = 2 (A, busy) + 2 (B, 1 existing + 1 start) = 4 > 3.
    assert any("above the 3-slot budget" in note for note in plan.notes)


def test_over_budget_from_demotion_hysteresis_is_attributed_correctly() -> None:
    """The over-budget note must name the real cause, not always blame busy
    listeners or unmeasurable repos (issue #601 review finding).

    Reproduced against the allocator: budget 1, one repo holding two idle,
    non-busy listeners, demand 0, slack streak below demand_idle_samples, and
    no other repo waiting. Nothing is pinned and nothing is busy — the overage
    is entirely the demotion-hysteresis grace period holding the surplus slot.
    The note must say so, and must NOT claim busy listeners or unmeasurable
    repos.
    """
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW)},
        budget=1,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 0},
        demand_idle_samples=3,
    )
    assert plan.changes == ()  # grace period holds the surplus; no park
    over = [n for n in plan.notes if "above the 1-slot budget" in n]
    assert len(over) == 1
    note = over[0]
    assert "in demotion hysteresis" in note
    # The misattribution the review flagged must not survive:
    assert "busy listeners" not in note
    assert "unmeasurable" not in note


def test_over_budget_note_lists_each_present_cause() -> None:
    """When multiple holds stack up, the note names all of them.

    Two pinned repos (unmeasurable) plus a busy listener on a third repo push
    running above budget for two distinct reasons at once.
    """
    instances = _instances(
        {
            CW: [("cw-1", True, False), ("cw-2", True, False)],
            JC: [("jc-1", True, False), ("jc-2", True, False)],
            PUB: [("pub-1", True, True), ("pub-2", True, True)],
        }
    )
    plan = plan_allocation(
        instances,
        {
            CW: RepoDemand(CW, ok=False, error="403"),
            JC: RepoDemand(JC, ok=False, error="403"),
            PUB: RepoDemand(PUB),
        },
        budget=4,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={PUB: 99},
        demand_idle_samples=3,
    )
    over = [n for n in plan.notes if "above the 4-slot budget" in n]
    assert len(over) == 1
    note = over[0]
    assert "pinned (unmeasurable)" in note
    assert "busy" in note
    assert "in demotion hysteresis" not in note  # PUB's streak is mature, not in grace


def test_floor_shortfall_tie_break_rotates_with_offset() -> None:
    """The same repo must not lose every name tie on every pass (issue #601).

    demand A=5 B=1 C=1, caps 5, budget 2, min_per_repo 1: only two repos get
    a floor slot. With offset 0, C is starved (B wins the tie). With offset 1,
    B is starved (C wins the tie). The rotation spreads the shortfall.
    """
    args = dict(
        demands={"A": 5, "B": 1, "C": 1},
        capacities={"A": 5, "B": 1, "C": 1},
        budget=2,
        min_per_repo=1,
    )
    targets0 = allocate_slots(**args, tie_break_offset=0)  # type: ignore[arg-type]
    assert targets0 == {"A": 1, "B": 1, "C": 0}

    targets1 = allocate_slots(**args, tie_break_offset=1)  # type: ignore[arg-type]
    assert targets1 == {"A": 1, "B": 0, "C": 1}

    targets2 = allocate_slots(**args, tie_break_offset=2)  # type: ignore[arg-type]
    assert targets2 == {"A": 1, "B": 1, "C": 0}


def test_water_fill_tie_break_rotates_with_offset() -> None:
    """The water-fill step's name tie-break also rotates (issue #601).

    Two repos with equal demand and an odd budget: the extra slot goes to
    CW with offset 0, to JC with offset 1.
    """
    args = dict(
        demands={CW: 3, JC: 3},
        capacities={CW: 3, JC: 3},
        budget=3,
        min_per_repo=1,
    )
    targets0 = allocate_slots(**args, tie_break_offset=0)  # type: ignore[arg-type]
    assert targets0 == {CW: 2, JC: 1}

    targets1 = allocate_slots(**args, tie_break_offset=1)  # type: ignore[arg-type]
    assert targets1 == {CW: 1, JC: 2}


def test_allocation_is_deterministic() -> None:
    """Same inputs, same plan — ties break on repo name."""
    args = dict(
        demands={CW: 5, JC: 5, PUB: 5},
        capacities={CW: 5, JC: 5, PUB: 5},
        budget=7,
        min_per_repo=1,
    )
    first = allocate_slots(**args)  # type: ignore[arg-type]
    for _ in range(5):
        assert allocate_slots(**args) == first  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# plan_allocation
# --------------------------------------------------------------------------


def _instances(spec: dict[str, list[tuple[str, bool, bool]]]) -> list[RunnerInstance]:
    """Build instances from {repo: [(name, running, busy), ...]}."""
    out: list[RunnerInstance] = []
    for repo, entries in spec.items():
        for name, running, busy in entries:
            out.append(
                RunnerInstance(
                    path=Path("/runners") / name,
                    name=name,
                    repo=repo,
                    running=running,
                    busy=busy,
                )
            )
    return out


def test_plan_starts_parked_runners_for_a_starved_repo() -> None:
    instances = _instances(
        {
            JC: [("jc-1", True, True), ("jc-2", True, True), ("jc-3", False, False)],
            CW: [("cw-1", True, False)],
        }
    )
    plan = plan_allocation(
        instances,
        {JC: RepoDemand(JC, queued_jobs=5), CW: RepoDemand(CW)},
        budget=4,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={},
        demand_idle_samples=3,
    )
    starts = [c for c in plan.changes if c.action is SlotAction.START]
    assert [c.runner_name for c in starts] == ["jc-3"]


def test_plan_never_parks_a_runner_that_is_executing_a_job() -> None:
    """Reclaiming a busy slot would abort a CI job mid-flight."""
    instances = _instances(
        {
            CW: [("cw-1", True, True), ("cw-2", True, True)],
            JC: [("jc-1", False, False), ("jc-2", False, False)],
        }
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=4)},
        budget=2,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 99},
        demand_idle_samples=3,
    )
    assert [c for c in plan.changes if c.action is SlotAction.PARK] == []
    assert any("executing jobs" in note for note in plan.notes)


def test_plan_defers_demotion_while_nobody_is_waiting() -> None:
    """Asymmetric hysteresis: parking a slot nobody wants can wait."""
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 1},
        demand_idle_samples=3,
    )
    assert plan.changes == ()
    assert any("slack for 1/3" in note for note in plan.notes)


def test_plan_demotes_once_the_slack_streak_matures() -> None:
    """A mature streak still parks when the budget is already fully used."""
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW)},
        budget=2,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 3},
        demand_idle_samples=3,
    )
    parks = [c for c in plan.changes if c.action is SlotAction.PARK]
    assert [c.runner_name for c in parks] == ["cw-2"]


def test_plan_holds_mature_slack_when_budget_undersubscribed() -> None:
    """Mature streaks must not park idle slots when the budget has spare room.

    Reproduced from issue #628: a host with budget to spare should not bounce
    the same listeners through park/restore cycles, because the restore latency
    costs real CI time while the parked slots were not displacing any work.
    """
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 3},
        demand_idle_samples=3,
    )
    assert [c for c in plan.changes if c.action is SlotAction.PARK] == []
    assert any(
        "budget undersubscribed" in note and "no repo is waiting" in note for note in plan.notes
    )


def test_plan_does_not_displace_idle_slots_when_pending_starts_fit() -> None:
    """A repo that can start its own runners using spare budget should not force
    an idle repo to park first. Issue #628: the budget-undersubscribed guard
    must not turn into a hard floor when another repo is merely ramping up.
    """
    instances = _instances(
        {
            CW: [
                ("cw-1", True, False),
                ("cw-2", True, False),
                ("cw-3", True, False),
                ("cw-4", True, False),
            ],
            JC: [
                ("jc-1", True, False),
                ("jc-2", False, False),
                ("jc-3", False, False),
                ("jc-4", False, False),
            ],
        }
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=4)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 3},
        demand_idle_samples=3,
    )
    assert [c for c in plan.changes if c.action is SlotAction.PARK] == []
    starts = [c for c in plan.changes if c.action is SlotAction.START]
    assert [c.runner_name for c in starts] == ["jc-2", "jc-3", "jc-4"]
    assert not any("above the" in note and "budget" in note for note in plan.notes)
    assert any("budget undersubscribed" in note for note in plan.notes)
    # The note must report the post-plan occupancy (8/8), not the pre-plan count (5/8).
    assert any("8/8 running" in note for note in plan.notes)


def test_plan_demotes_when_pending_starts_would_exceed_budget() -> None:
    """If pending starts do not fit in the spare budget, the surplus must still
    be reclaimed so the host stays within its configured ceiling."""
    instances = _instances(
        {
            CW: [
                ("cw-1", True, False),
                ("cw-2", True, False),
                ("cw-3", True, False),
                ("cw-4", True, False),
                ("cw-5", True, False),
            ],
            JC: [
                ("jc-1", True, False),
                ("jc-2", False, False),
                ("jc-3", False, False),
                ("jc-4", False, False),
            ],
        }
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=4)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 3},
        demand_idle_samples=3,
    )
    parks = [c for c in plan.changes if c.action is SlotAction.PARK]
    assert [c.runner_name for c in parks] == ["cw-5", "cw-4", "cw-3", "cw-2"]
    starts = [c for c in plan.changes if c.action is SlotAction.START]
    assert [c.runner_name for c in starts] == ["jc-2", "jc-3", "jc-4"]


def test_plan_holds_mature_slack_with_busy_runners() -> None:
    """A budget-undersubscribed hold still reports surplus listeners that are
    executing jobs, so the hold does not look like purely idle slots."""
    instances = _instances(
        {CW: [("cw-1", True, True), ("cw-2", True, True), ("cw-3", True, False)]}
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 3},
        demand_idle_samples=3,
    )
    assert [c for c in plan.changes if c.action is SlotAction.PARK] == []
    budget_notes = [n for n in plan.notes if "budget undersubscribed" in n]
    assert len(budget_notes) == 1
    assert "executing jobs" in budget_notes[0]


def test_plan_holds_mixed_hysteresis_and_budget_surplus() -> None:
    """One repo in its grace period and another with a mature streak can both be
    held in the same pass when the budget is undersubscribed."""
    instances = _instances(
        {
            CW: [("cw-1", True, False), ("cw-2", True, False)],
            JC: [("jc-1", True, False), ("jc-2", True, False)],
        }
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 1, JC: 3},
        demand_idle_samples=3,
    )
    assert [c for c in plan.changes if c.action is SlotAction.PARK] == []
    assert any("slack for 1/3" in note for note in plan.notes)
    assert any("budget undersubscribed" in note for note in plan.notes)
    assert not any("above the" in note and "budget" in note for note in plan.notes)


def test_plan_reclaims_immediately_when_another_repo_is_waiting() -> None:
    """Hysteresis must not make a starved repo wait for a slot sitting idle."""
    instances = _instances(
        {
            CW: [("cw-1", True, False), ("cw-2", True, False)],
            JC: [("jc-1", False, False), ("jc-2", False, False), ("jc-3", False, False)],
        }
    )
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=6)},
        budget=3,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 0},
        demand_idle_samples=3,
    )
    parks = [c for c in plan.changes if c.action is SlotAction.PARK]
    assert [c.runner_name for c in parks] == ["cw-2"]
    assert all("reclaimed for a waiting repo" == c.reason for c in parks)


def test_plan_notes_when_demand_exceeds_registered_capacity() -> None:
    """The operator needs to hear that more runner dirs would help."""
    instances = _instances({JC: [("jc-1", True, True)]})
    plan = plan_allocation(
        instances,
        {JC: RepoDemand(JC, queued_jobs=9)},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={},
        demand_idle_samples=3,
    )
    assert any("exceeds its 1 registered runner" in note for note in plan.notes)


def test_plan_pins_a_repo_whose_demand_could_not_be_measured() -> None:
    """A transient API failure must not become a starved queue."""
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW, ok=False, error="boom")},
        budget=8,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={CW: 99},
        demand_idle_samples=3,
    )
    assert plan.changes == ()
    target = next(t for t in plan.targets if t.repo == CW)
    assert target.pinned is True
    assert target.target == 2
    assert any("unmeasurable" in note for note in plan.notes)


def test_plan_reports_targets_for_every_repo() -> None:
    instances = _instances({CW: [("cw-1", True, False)], JC: [("jc-1", False, False)]})
    plan = plan_allocation(
        instances,
        {CW: RepoDemand(CW), JC: RepoDemand(JC, queued_jobs=1)},
        budget=4,
        budget_reason="test",
        min_per_repo=1,
        idle_streaks={},
        demand_idle_samples=3,
    )
    assert {t.repo for t in plan.targets} == {CW, JC}
    summary = plan_summary(plan)
    assert summary["budget"] == 4
    assert {t["repo"] for t in summary["targets"]} == {CW, JC}


def test_plan_target_capped_by_capacity_when_demand_exceeds_registered_runners() -> None:
    """A repo whose demand exceeds its registered runner count is capped at capacity.

    The target equals the running count (both 1 — the single registered runner),
    so no start or park is issued. The ``needed > len(parked)`` branch that used
    to guard this path was unreachable (target <= capacity == running + parked)
    and has been removed (issue #601).
    """
    instances = _instances({JC: [("jc-1", True, True)]})
    plan = plan_allocation(
        instances,
        {JC: RepoDemand(JC, queued_jobs=5)},
        budget=8,
        budget_reason="test",
        min_per_repo=2,
        idle_streaks={},
        demand_idle_samples=3,
    )
    assert plan.changes == ()


# --------------------------------------------------------------------------
# runner_capacity_starved_events — synthesized target set
# --------------------------------------------------------------------------


def test_runner_capacity_starved_detected_from_synthesized_targets() -> None:
    """A repo whose demand exceeds its registered runner count while the host
    still has spare budget is reported as capacity-starved."""
    plan = AllocationPlan(
        budget=8,
        budget_reason="test",
        targets=(
            RepoTarget(repo=CW, target=2, running=2, demand=13, capacity=2),
            RepoTarget(repo=JC, target=1, running=3, demand=0, capacity=5),
            RepoTarget(repo=PUB, target=1, running=1, demand=0, capacity=1),
        ),
        changes=(),
    )
    events = runner_capacity_starved_events(plan)
    assert len(events) == 1
    assert events[0]["repo"] == CW
    assert events[0]["demand"] == 13
    assert events[0]["capacity"] == 2
    assert events[0]["spare_budget"] == 2


def test_runner_capacity_starved_is_quiet_when_demand_fits_capacity() -> None:
    """Positive control: every repo has demand <= capacity, so the detector
    stays silent.  A detector that only fires on the broken state is not yet
    evidence of correct behavior."""
    plan = AllocationPlan(
        budget=8,
        budget_reason="test",
        targets=(
            RepoTarget(repo=CW, target=2, running=2, demand=2, capacity=2),
            RepoTarget(repo=JC, target=3, running=3, demand=0, capacity=5),
            RepoTarget(repo=PUB, target=1, running=1, demand=0, capacity=1),
        ),
        changes=(),
    )
    assert runner_capacity_starved_events(plan) == []


def test_runner_capacity_starved_is_quiet_when_budget_is_full() -> None:
    """A repo with demand above capacity is not reported when the host has no
    spare budget to use."""
    plan = AllocationPlan(
        budget=6,
        budget_reason="test",
        targets=(
            RepoTarget(repo=CW, target=2, running=2, demand=13, capacity=2),
            RepoTarget(repo=JC, target=3, running=3, demand=0, capacity=5),
            RepoTarget(repo=PUB, target=1, running=1, demand=0, capacity=1),
        ),
        changes=(),
    )
    assert runner_capacity_starved_events(plan) == []


# --------------------------------------------------------------------------
# annotate_busy / next_idle_streaks
# --------------------------------------------------------------------------


def test_annotate_busy_marks_only_the_named_runners() -> None:
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})
    annotated = annotate_busy(instances, {CW: {"cw-1"}})
    assert [(i.name, i.busy) for i in annotated] == [("cw-1", True), ("cw-2", False)]


def test_idle_streak_increments_while_slack_and_resets_on_demand() -> None:
    instances = _instances({CW: [("cw-1", True, False), ("cw-2", True, False)]})

    slack = next_idle_streaks(instances, {CW: RepoDemand(CW)}, {CW: 2})
    assert slack[CW] == 3

    busy = next_idle_streaks(instances, {CW: RepoDemand(CW, queued_jobs=5)}, {CW: 2})
    assert busy[CW] == 0


def test_idle_streak_holds_when_demand_is_unmeasurable() -> None:
    """Missing data must not accrue toward parking a runner."""
    instances = _instances({CW: [("cw-1", True, False)]})
    streaks = next_idle_streaks(instances, {CW: RepoDemand(CW, ok=False, error="x")}, {CW: 2})
    assert streaks[CW] == 2


def test_idle_streak_ignores_capacity_it_cannot_use() -> None:
    """Demand above capacity is not slack, even though it is unmet."""
    instances = _instances({JC: [("jc-1", True, True)]})
    streaks = next_idle_streaks(instances, {JC: RepoDemand(JC, queued_jobs=9)}, {JC: 0})
    assert streaks[JC] == 0


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _make_runner_dir(
    root: Path,
    name: str,
    repo_url: str,
    agent_name: str,
    *,
    script: str = "run.cmd",
    bom: bool = True,
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / script).write_text("@echo off\n", encoding="utf-8")
    payload = json.dumps({"agentName": agent_name, "gitHubUrl": repo_url, "workFolder": "_work"})
    # The real runner writes .runner as UTF-8 with a BOM.
    (path / ".runner").write_text(payload, encoding="utf-8-sig" if bom else "utf-8")
    return path


def test_discovery_reads_repo_and_name_from_the_runner_file(tmp_path: Path) -> None:
    """Ownership comes from .runner, not from directory naming conventions."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "weird-name", f"https://github.com/{JC}", "jc-9800x3d-1")

    instances, notes = discover_runner_instances(root, platform="win32")

    assert len(instances) == 1
    assert instances[0].repo == JC
    assert instances[0].name == "jc-9800x3d-1"
    assert notes == []


def test_discovery_handles_the_bom_the_runner_writes(tmp_path: Path) -> None:
    """A plain utf-8 read of .runner raises on the leading BOM."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "jc-1", f"https://github.com/{JC}", "jc-1", bom=True)

    instances, notes = discover_runner_instances(root, platform="win32")
    assert [i.name for i in instances] == ["jc-1"]
    assert notes == []


def test_discovery_cannot_reach_runners_outside_managed_root(tmp_path: Path) -> None:
    """Structural guard: a runner installed elsewhere on the host is unreachable.

    This host has an unrelated runner *service* installed at C:\\actions-runner
    that must never be touched. Safety comes from the traversal never leaving
    the configured managed root — not from filtering names afterwards.
    """
    managed_root = tmp_path / "actions-runners"
    managed_root.mkdir()
    _make_runner_dir(managed_root, "cw-1", f"https://github.com/{CW}", "cw-1")

    # A sibling install, exactly one directory over.
    foreign_root = tmp_path / "actions-runner"
    foreign_root.mkdir()
    (foreign_root / "run.cmd").write_text("@echo off\n", encoding="utf-8")
    (foreign_root / ".runner").write_text(
        json.dumps({"agentName": "foreign", "gitHubUrl": "https://github.com/other/repo"}),
        encoding="utf-8-sig",
    )

    instances, _ = discover_runner_instances(managed_root, platform="win32")

    assert [i.name for i in instances] == ["cw-1"]
    assert all("foreign" not in i.name for i in instances)
    assert all(foreign_root not in i.path.parents for i in instances)


def test_discovery_does_not_recurse_into_nested_directories(tmp_path: Path) -> None:
    """Only direct children are runners; _work trees must not be mistaken for them."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    nested = root / "cw-1" / "_work" / "nested"
    nested.mkdir(parents=True)
    (nested / "run.cmd").write_text("@echo off\n", encoding="utf-8")
    (nested / ".runner").write_text(
        json.dumps({"agentName": "nested", "gitHubUrl": f"https://github.com/{CW}"}),
        encoding="utf-8-sig",
    )
    (root / "cw-1" / "run.cmd").write_text("@echo off\n", encoding="utf-8")
    (root / "cw-1" / ".runner").write_text(
        json.dumps({"agentName": "cw-1", "gitHubUrl": f"https://github.com/{CW}"}),
        encoding="utf-8-sig",
    )

    instances, _ = discover_runner_instances(root, platform="win32")
    assert [i.name for i in instances] == ["cw-1"]


def test_discovery_reports_skipped_directories(tmp_path: Path) -> None:
    """An unreadable .runner is surfaced, not silently dropped."""
    root = tmp_path / "actions-runners"
    root.mkdir()

    no_runner_file = root / "half-configured"
    no_runner_file.mkdir()
    (no_runner_file / "run.cmd").write_text("@echo off\n", encoding="utf-8")

    corrupt = root / "corrupt"
    corrupt.mkdir()
    (corrupt / "run.cmd").write_text("@echo off\n", encoding="utf-8")
    (corrupt / ".runner").write_text("{not json", encoding="utf-8")

    instances, notes = discover_runner_instances(root, platform="win32")

    assert instances == []
    assert any("half-configured" in note for note in notes)
    assert any("corrupt" in note for note in notes)


def test_discovery_ignores_directories_without_a_launch_script(tmp_path: Path) -> None:
    root = tmp_path / "actions-runners"
    root.mkdir()
    unconfigured = root / "extracted-only"
    unconfigured.mkdir()
    (unconfigured / ".runner").write_text(
        json.dumps({"agentName": "x", "gitHubUrl": f"https://github.com/{CW}"}),
        encoding="utf-8-sig",
    )

    instances, notes = discover_runner_instances(root, platform="win32")
    assert instances == []
    assert notes == []


def test_discovery_on_missing_root_reports_rather_than_raising(tmp_path: Path) -> None:
    instances, notes = discover_runner_instances(tmp_path / "nope", platform="win32")
    assert instances == []
    assert any("does not exist" in note for note in notes)


def test_discovery_uses_the_platform_launch_script(tmp_path: Path) -> None:
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", f"https://github.com/{CW}", "cw-1", script="run.sh")

    assert discover_runner_instances(root, platform="linux")[0] != []
    assert discover_runner_instances(root, platform="win32")[0] == []


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_idle_streaks_round_trip(tmp_path: Path) -> None:
    save_idle_streaks(tmp_path, {CW: 2, JC: 0}, source="prologue")
    assert load_idle_streaks(tmp_path) == {CW: 2, JC: 0}


def test_tie_break_offset_round_trip(tmp_path: Path) -> None:
    """The floor-shortfall rotation offset persists across passes (issue #601)."""
    from charlie_work.runner_slots import load_tie_break_offset

    save_idle_streaks(tmp_path, {CW: 1}, source="prologue", tie_break_offset=3)
    assert load_tie_break_offset(tmp_path) == 3


def test_tie_break_offset_defaults_to_zero_on_missing_file(tmp_path: Path) -> None:
    from charlie_work.runner_slots import load_tie_break_offset

    assert load_tie_break_offset(tmp_path) == 0


def test_tie_break_offset_defaults_to_zero_on_corrupt_state(tmp_path: Path) -> None:
    from charlie_work.runner_slots import load_tie_break_offset

    (tmp_path / "runner-allocation.json").write_text("{ not json", encoding="utf-8")
    assert load_tie_break_offset(tmp_path) == 0


def test_idle_streak_write_is_atomic(tmp_path: Path) -> None:
    """Temp-file + replace, per the project's JSON-write invariant."""
    save_idle_streaks(tmp_path, {CW: 1}, source="prologue")
    assert list(tmp_path.glob("*.tmp")) == []
    payload = json.loads((tmp_path / "runner-allocation.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["repos"][CW]["idle_streak"] == 1


def test_load_idle_streaks_degrades_on_corrupt_state(tmp_path: Path) -> None:
    (tmp_path / "runner-allocation.json").write_text("{ not json", encoding="utf-8")
    assert load_idle_streaks(tmp_path) == {}


def test_load_idle_streaks_ignores_non_integer_entries(tmp_path: Path) -> None:
    (tmp_path / "runner-allocation.json").write_text(
        json.dumps({"repos": {CW: {"idle_streak": "three"}, JC: {"idle_streak": 4}}}),
        encoding="utf-8",
    )
    assert load_idle_streaks(tmp_path) == {JC: 4}


def test_load_idle_streaks_on_missing_file(tmp_path: Path) -> None:
    assert load_idle_streaks(tmp_path) == {}


# --------------------------------------------------------------------------
# Actuation guard
# --------------------------------------------------------------------------


def test_park_refuses_a_runner_that_picked_up_a_job_after_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan is a snapshot; a job can arrive before actuation."""
    monkeypatch.setattr("charlie_work.runner_slots.has_active_job", lambda _path: True)
    instance = RunnerInstance(path=Path("/runners/cw-1"), name="cw-1", repo=CW, running=True)

    ok, message = park_runner_slot(instance, dry_run=False)

    assert ok is False
    assert "picked up a job" in message


def test_park_is_a_noop_when_the_listener_is_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("charlie_work.runner_slots.has_active_job", lambda _path: False)
    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_listener_process", lambda _path: None
    )
    instance = RunnerInstance(path=Path("/runners/cw-1"), name="cw-1", repo=CW, running=True)

    ok, message = park_runner_slot(instance)

    assert ok is True
    assert "already parked" in message


def test_park_declines_a_runner_that_is_still_starting_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live wrapper with no listener yet must not be "parked".

    ``cmd /c`` does not take its child down, so killing the wrapper would leave
    the listener to come online orphaned — a park that silently did not park.
    """
    monkeypatch.setattr("charlie_work.runner_slots.has_active_job", lambda _path: False)
    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_listener_process", lambda _path: None
    )
    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_launcher_process", lambda _path: object()
    )
    instance = RunnerInstance(path=Path("/runners/cw-1"), name="cw-1", repo=CW, running=True)

    ok, message = park_runner_slot(instance)

    assert ok is False
    assert "still starting up" in message


def test_start_does_not_relaunch_a_runner_that_is_mid_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two controllers seconds apart must not both launch one runner.

    ``Runner.Listener.exe`` appears several seconds after the launch script, so
    liveness that only looks for the listener would let a fleet pass and an
    operator's ``runners allocate`` each start the same runner — two listeners
    then race for one registration.
    """
    launches: list[Path] = []
    monkeypatch.setattr("charlie_work.runner_slots.is_runner_launched", lambda _path: True)
    monkeypatch.setattr(
        "charlie_work.runner_slots.launch_runner_listener",
        lambda path, dry_run=False: (launches.append(path), (True, "launched"))[1],
    )
    change = SlotChange(
        repo=JC,
        runner_name="jc-3",
        path=Path("/runners/jc-3"),
        action=SlotAction.START,
        reason="demand 9",
    )
    plan = AllocationPlan(budget=8, budget_reason="configured", targets=(), changes=(change,))

    results = apply_allocation(plan)

    assert launches == []
    assert results[0].ok is True
    assert "already running" in results[0].message


# --------------------------------------------------------------------------
# resolve_inputs
# --------------------------------------------------------------------------


def test_resolve_inputs_falls_back_to_the_scaling_managed_root(tmp_path: Path) -> None:
    """The host path is configured once, not duplicated across two sections."""
    inputs, error = resolve_inputs(
        RunnerAllocationConfig(enabled=True), managed_root_fallback=str(tmp_path)
    )
    assert error is None
    assert inputs is not None
    assert inputs.managed_root == tmp_path


def test_resolve_inputs_requires_a_managed_root() -> None:
    inputs, error = resolve_inputs(RunnerAllocationConfig(enabled=True))
    assert inputs is None
    assert error is not None
    assert "managed_root" in error


def test_resolve_inputs_rejects_a_nonexistent_managed_root(tmp_path: Path) -> None:
    """A typo'd path must fail loudly instead of allocating nothing."""
    inputs, error = resolve_inputs(
        RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path / "gone"))
    )
    assert inputs is None
    assert error is not None
    assert "does not exist" in error


def test_resolve_inputs_notes_an_underived_budget(tmp_path: Path) -> None:
    """An unconfigured budget is reported, since the default cannot see worker load."""
    inputs, error = resolve_inputs(
        RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path))
    )
    assert error is None
    assert inputs is not None
    assert any("max_running_runners unset" in note for note in inputs.notes)


def test_resolve_inputs_is_quiet_when_the_budget_is_configured(tmp_path: Path) -> None:
    inputs, error = resolve_inputs(
        RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path), max_running_runners=8)
    )
    assert error is None
    assert inputs is not None
    assert inputs.budget == 8
    assert inputs.notes == ()


# --------------------------------------------------------------------------
# Busy detection (real body)
# --------------------------------------------------------------------------
#
# Every other test in this file monkeypatches ``has_active_job`` wholesale, so
# the function guarding SAFETY A ("never stop a busy listener") had no coverage
# of its own: inverting its result would have kept the suite green. These drive
# the real body by faking the process tree one level down instead.


def _fake_process_tree(monkeypatch: pytest.MonkeyPatch, children: list[object]) -> None:
    """Point ``has_active_job`` at a synthetic listener with ``children``."""
    psutil = pytest.importorskip("psutil")

    class _FakeListener:
        pid = 4242

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def children(self, recursive: bool = False) -> list[object]:
            return list(children)

    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_listener_process",
        lambda _path: _FakeListener(),
    )
    # ``has_active_job`` imports psutil inside the function, binding this same
    # module object, so patching the attribute here reaches it.
    monkeypatch.setattr(psutil, "Process", _FakeProcess)


class _FakeChild:
    def __init__(self, name: str = "", raises: BaseException | None = None) -> None:
        self._name = name
        self._raises = raises

    def name(self) -> str:
        if self._raises is not None:
            raise self._raises
        return self._name


def test_busy_detection_finds_a_worker_child(monkeypatch: pytest.MonkeyPatch) -> None:
    from charlie_work.runner_slots import has_active_job

    _fake_process_tree(monkeypatch, [_FakeChild("conhost.exe"), _FakeChild("Runner.Worker.exe")])

    assert has_active_job(Path("/runners/cw-1")) is True


def test_busy_detection_reports_idle_when_no_child_is_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from charlie_work.runner_slots import has_active_job

    _fake_process_tree(monkeypatch, [_FakeChild("conhost.exe"), _FakeChild("cmd.exe")])

    assert has_active_job(Path("/runners/cw-1")) is False


def test_busy_detection_fails_closed_when_a_child_name_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable child cannot be ruled out as the worker.

    The docstring promises fail-closed and the outer handler honours it; this
    pins the per-child path, where treating AccessDenied as "not a worker"
    would let a park terminate a listener mid-job and abort a CI run.
    """
    psutil = pytest.importorskip("psutil")
    from charlie_work.runner_slots import has_active_job

    _fake_process_tree(monkeypatch, [_FakeChild(raises=psutil.AccessDenied(pid=99))])

    assert has_active_job(Path("/runners/cw-1")) is True


def test_busy_detection_keeps_scanning_past_a_child_that_vanished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that exited cannot hold a job, but its siblings still can."""
    psutil = pytest.importorskip("psutil")
    from charlie_work.runner_slots import has_active_job

    _fake_process_tree(
        monkeypatch,
        [
            _FakeChild(raises=psutil.NoSuchProcess(pid=99)),
            _FakeChild("Runner.Worker.exe"),
        ],
    )

    assert has_active_job(Path("/runners/cw-1")) is True


# --------------------------------------------------------------------------
# Traversal containment
# --------------------------------------------------------------------------


def test_discovery_skips_a_directory_that_resolves_outside_managed_root(
    tmp_path: Path,
) -> None:
    """``is_dir()`` follows junctions, so containment needs the resolved path.

    The real hazard on this host is the unrelated runner *service* at
    C:\actions-runner: a junction under managed_root pointing at it would make
    it start/park-eligible.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    outsider = tmp_path / "elsewhere" / "actions-runner"
    outsider.mkdir(parents=True)
    (outsider / "run.cmd").write_text("", encoding="utf-8")
    (outsider / ".runner").write_text(
        json.dumps({"gitHubUrl": f"https://github.com/{CW}"}), encoding="utf-8"
    )
    link = managed_root / "sneaky"
    # Prefer a junction on Windows: unlike a symlink it needs no elevation or
    # Developer Mode, so this test actually runs on the self-hosted CI host
    # rather than skipping exactly where the hazard lives.
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(outsider), str(link))
        else:
            link.symlink_to(outsider, target_is_directory=True)
    except (OSError, AttributeError, ImportError, NotImplementedError):
        pytest.skip("cannot create a directory link on this host")

    instances, notes = discover_runner_instances(managed_root, platform="win32")

    assert instances == []
    assert any("resolves outside managed_root" in note for note in notes)


def test_saved_state_records_which_path_wrote_it(tmp_path: Path) -> None:
    """Provenance is what lets the doctor probe distinguish daemon from operator."""
    from charlie_work.runner_slots import load_allocation_stamp

    save_idle_streaks(tmp_path, {CW: 1}, source="cli")
    payload = json.loads((tmp_path / "runner-allocation.json").read_text(encoding="utf-8"))
    assert payload["source"] == "cli"

    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.source == "cli"
    assert stamp.updated_at is not None
    assert stamp.updated_at.tzinfo is not None


def test_allocation_stamp_is_none_when_no_pass_has_run(tmp_path: Path) -> None:
    from charlie_work.runner_slots import load_allocation_stamp

    assert load_allocation_stamp(tmp_path) is None


def test_allocation_stamp_distinguishes_unreadable_from_absent(tmp_path: Path) -> None:
    """A corrupt file must not look like never-ran; the two need different reports."""
    from charlie_work.runner_slots import load_allocation_stamp

    (tmp_path / "runner-allocation.json").write_text("{not json", encoding="utf-8")
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.updated_at is None
    assert stamp.source is None


def test_allocation_stamp_treats_a_naive_timestamp_as_utc(tmp_path: Path) -> None:
    """Ages are computed against an aware now; a naive stamp must not raise."""
    from charlie_work.runner_slots import load_allocation_stamp

    (tmp_path / "runner-allocation.json").write_text(
        json.dumps({"version": 1, "updated_at": "2026-07-25T12:00:00", "repos": {}}),
        encoding="utf-8",
    )
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.updated_at is not None
    assert stamp.updated_at.tzinfo is not None


# --------------------------------------------------------------------------
# run_allocation_pass — end-to-end tie_break_offset threading (issue #601)
# --------------------------------------------------------------------------
#
# The unit tests above prove allocate_slots rotates given a hand-supplied
# offset, and that save/load round-trips the offset. The review finding on
# PR #678 was that the *wiring* — run_allocation_pass reading the offset,
# threading it into plan_allocation, and persisting offset+1 — was never
# exercised end-to-end. This test runs the real pass twice against a fake
# GitHub and a real managed_root + state file to close that gap.


def test_run_allocation_pass_threads_and_persists_tie_break_offset_across_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pass reads tie_break_offset from disk, threads it into the plan, and
    persists offset+1 so the next pass rotates the floor-shortfall loser.

    Floor-shortfall scenario: A (cap 5, demand 5), B (cap 1, demand 1),
    C (cap 1, demand 1), budget 2, min_per_repo 1. Only two repos get a floor
    slot; the third loses a name tie. Pass 1 (offset 0) starves C; pass 2
    (persisted offset 1) starves B. The rotation is observable in the plan's
    targets, proving the offset was read, threaded into allocate_slots, and the
    incremented value written back for the next pass.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    for i in range(5):
        _make_runner_dir(managed_root, f"a-{i}", "https://github.com/o/A", f"a-{i}")
    _make_runner_dir(managed_root, "b-0", "https://github.com/o/B", "b-0")
    _make_runner_dir(managed_root, "c-0", "https://github.com/o/C", "c-0")

    fleet_dir = tmp_path / "fleet"

    demand_by_repo = {"o/A": 5, "o/B": 1, "o/C": 1}

    def fake_measure(gh: object, repo: str, max_runs_scanned: int) -> RepoDemand:
        return RepoDemand(repo=repo, queued_jobs=demand_by_repo[repo])

    monkeypatch.setattr("charlie_work.runner_allocation_pass.measure_repo_demand", fake_measure)
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.fetch_busy_runner_names",
        lambda gh, repo: (set(), None),
    )
    # No real actuation: the runner dirs have no listener processes, so
    # starting/parking them would touch the host. Return no results; the plan
    # and the state write are what this test inspects.
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.apply_allocation",
        lambda plan, dry_run=False: [],
    )

    config = RunnerAllocationConfig(
        enabled=True,
        managed_root=str(managed_root),
        max_running_runners=2,
        min_running_per_repo=1,
        demand_idle_samples=3,
    )

    class _FakeGh:
        """Stand-in: every gh.run call is intercepted by the monkeypatches above."""

    # Pass 1: no state file -> offset 0 -> B wins the name tie, C starved.
    r1 = run_allocation_pass(
        _FakeGh(),  # type: ignore[arg-type]
        config,
        fleet_dir_override=str(fleet_dir),
        source="prologue",
    )
    assert r1.ok is True
    assert r1.plan is not None
    t1 = {t.repo: t.target for t in r1.plan.targets}
    assert t1 == {"o/A": 1, "o/B": 1, "o/C": 0}
    assert load_tie_break_offset(fleet_dir) == 1

    # Pass 2: persisted offset 1 -> C wins the name tie, B starved.
    r2 = run_allocation_pass(
        _FakeGh(),  # type: ignore[arg-type]
        config,
        fleet_dir_override=str(fleet_dir),
        source="prologue",
    )
    assert r2.ok is True
    assert r2.plan is not None
    t2 = {t.repo: t.target for t in r2.plan.targets}
    assert t2 == {"o/A": 1, "o/B": 0, "o/C": 1}
    assert load_tie_break_offset(fleet_dir) == 2


def test_run_allocation_pass_logs_runner_capacity_starved_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass that sees demand above capacity with spare budget persists a
    runner_capacity_starved event to events.db so provisioning can consume it."""
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    _make_runner_dir(managed_root, "cw-1", f"https://github.com/{CW}", "cw-1")
    _make_runner_dir(managed_root, "cw-2", f"https://github.com/{CW}", "cw-2")

    def fake_measure(gh: object, repo: str, max_runs_scanned: int) -> RepoDemand:
        if repo == CW:
            return RepoDemand(repo=repo, queued_jobs=13)
        return RepoDemand(repo=repo)

    monkeypatch.setattr("charlie_work.runner_allocation_pass.measure_repo_demand", fake_measure)
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.fetch_busy_runner_names",
        lambda gh, repo: (set(), None),
    )
    # Pretend both listeners are already running so the plan is converged and
    # does not try to start/park real host processes.
    monkeypatch.setattr(
        "charlie_work.runner_slots.is_runner_launched",
        lambda _path: True,
    )

    fleet_dir = tmp_path / "fleet"
    state_path = fleet_dir / "state.json"

    config = RunnerAllocationConfig(
        enabled=True,
        managed_root=str(managed_root),
        max_running_runners=8,
        min_running_per_repo=1,
        demand_idle_samples=3,
    )

    class _FakeGh:
        pass

    result = run_allocation_pass(
        _FakeGh(),  # type: ignore[arg-type]
        config,
        fleet_dir_override=str(fleet_dir),
        state_path=state_path,
        source="prologue",
    )

    assert result.ok is True
    assert result.plan is not None
    assert result.plan.changes == ()

    close_db(state_path)
    events = query_events(state_path, kind="runner_capacity_starved")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["repo"] == CW
    assert payload["demand"] == 13
    assert payload["capacity"] == 2
    assert payload["spare_budget"] == 6
