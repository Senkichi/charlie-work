"""Tests for host-wide elastic runner-slot allocation.

Covers the pure allocator (runner_allocation.py), the host/GitHub layer
(runner_slots.py), and the pass that wires them together
(runner_allocation_pass.py).

**These modules are no longer on the live path (issue #876).** PR #869
repointed charlie-work's fleet consumers at the extracted ``ci_fleet`` package;
the modules under test here are retained, re-activatable by config, as the
rollback path. A green run of this file therefore means *rollback still works*
-- it does **not** mean fleet allocation is healthy, because the allocator
making live decisions is ci_fleet's. See ``tests/test_dormant_fleet_marking.py``,
which derives that claim from the import graph rather than trusting this
paragraph.
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
    starved_repos,
)
from charlie_work.github import GitHubError
from charlie_work.runner_allocation_pass import resolve_inputs, run_allocation_pass
from charlie_work.runner_slots import (
    apply_allocation,
    discover_runner_instances,
    load_idle_streaks,
    load_tie_break_offset,
    park_runner_slot,
    save_idle_streaks,
)

# Issue #876: this whole module covers the dormant rollback path, not the live
# allocator. Applied at module scope rather than per test because the dormancy is
# a property of the module under test, not of any individual case. Membership is
# enforced against the import graph by tests/test_dormant_fleet_marking.py.
pytestmark = pytest.mark.rollback_path


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
    save_idle_streaks(tmp_path, {CW: 2, JC: 0}, source="prologue", full_pass_interval_seconds=300)
    assert load_idle_streaks(tmp_path) == {CW: 2, JC: 0}


def test_tie_break_offset_round_trip(tmp_path: Path) -> None:
    """The floor-shortfall rotation offset persists across passes (issue #601)."""
    from charlie_work.runner_slots import load_tie_break_offset

    save_idle_streaks(
        tmp_path, {CW: 1}, source="prologue", full_pass_interval_seconds=300, tie_break_offset=3
    )
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
    save_idle_streaks(tmp_path, {CW: 1}, source="prologue", full_pass_interval_seconds=300)
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
# run_allocation_pass: skip provenance (issue #606)
#
# A pass that declines to act used to leave the state file absent-or-stale, so
# the doctor probe attributed every skip to "the daemon never reached
# allocation" (#590). The pass now records the actual reason. ``gh`` is never
# reached on either skip path, so a stand-in is safe here.
# --------------------------------------------------------------------------


def test_run_allocation_pass_records_a_skip_when_no_runners_are_found(
    tmp_path: Path,
) -> None:
    """A real pass with no runners under managed_root writes a skip record."""
    from charlie_work.runner_allocation_pass import run_allocation_pass
    from charlie_work.runner_slots import load_allocation_stamp

    result = run_allocation_pass(
        gh=None,  # type: ignore[arg-type]
        allocation=RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path)),
        fleet_dir_override=str(tmp_path),
        dry_run=False,
        source="prologue",
        full_pass_interval_seconds=300,
    )

    assert result.skipped is True
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.skip_reason is not None
    assert "no configured runners" in stamp.skip_reason
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == 300


def test_run_allocation_pass_records_a_skip_when_inputs_cannot_resolve(
    tmp_path: Path,
) -> None:
    """An unresolvable managed_root writes a skip record naming the cause."""
    from charlie_work.runner_allocation_pass import run_allocation_pass
    from charlie_work.runner_slots import load_allocation_stamp

    result = run_allocation_pass(
        gh=None,  # type: ignore[arg-type]
        allocation=RunnerAllocationConfig(enabled=True),  # no managed_root, no fallback
        fleet_dir_override=str(tmp_path),
        dry_run=False,
        source="prologue",
        full_pass_interval_seconds=300,
    )

    assert result.ok is False
    assert result.error is not None
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.skip_reason is not None
    assert "managed_root" in stamp.skip_reason


def test_run_allocation_pass_does_not_write_state_on_a_dry_run_skip(
    tmp_path: Path,
) -> None:
    """A dry-run skip must not bump updated_at — that would look like a pass."""
    from charlie_work.runner_allocation_pass import run_allocation_pass
    from charlie_work.runner_slots import load_allocation_stamp

    run_allocation_pass(
        gh=None,  # type: ignore[arg-type]
        allocation=RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path)),
        fleet_dir_override=str(tmp_path),
        dry_run=True,
        source="prologue",
        full_pass_interval_seconds=300,
    )

    # Dry-run isolation: no state file at all.
    assert load_allocation_stamp(tmp_path) is None


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

    save_idle_streaks(tmp_path, {CW: 1}, source="cli", full_pass_interval_seconds=300)
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
# Process matchers — real body
# --------------------------------------------------------------------------
#
# ``is_runner_launched`` / ``get_runner_launcher_process`` /
# ``get_runner_listener_process`` are the host-facing matchers that decide
# whether a runner directory has a live listener or a launch script mid-startup.
# Every other test in this file (and the discover tests below) either
# monkeypatches them wholesale or calls them without asserting the
# ``RunnerInstance.running`` field they populate — so a matcher that always
# returned False (or always True) would keep the suite green. These drive the
# real bodies by faking ``psutil.process_iter`` one level down, the same shape
# the busy-detection tests above use for ``has_active_job``.


def _platform_launch_script() -> str:
    return "run.cmd" if sys.platform == "win32" else "run.sh"


class _FakeInfo:
    """Mapping that yields canned attrs, or raises a psutil error on access.

    ``psutil.process_iter(attrs)`` populates ``proc.info`` with the requested
    attrs (None when unreadable). The matchers index ``proc.info[...]`` inside
    a ``try/except (NoSuchProcess, AccessDenied, ZombieProcess)`` block, so to
    exercise the skip-and-continue path the info object itself has to raise.
    """

    def __init__(
        self, data: dict[str, object] | None = None, raises: BaseException | None = None
    ) -> None:
        self._data = data or {}
        self._raises = raises

    def __getitem__(self, key: str) -> object:
        if self._raises is not None:
            raise self._raises
        return self._data.get(key)


class _FakeProc:
    """Stand-in for a ``psutil.Process`` yielded by ``process_iter``."""

    def __init__(self, pid: int, info: _FakeInfo) -> None:
        self.pid = pid
        self.info = info


def _fake_process_iter(monkeypatch: pytest.MonkeyPatch, procs: list[_FakeProc]) -> None:
    """Point ``psutil.process_iter`` at a fixed list of fake processes."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter(procs))


def _launcher_proc(runner_dir: Path, pid: int = 100) -> _FakeProc:
    """A process whose cmdline names this runner's launch script.

    The launcher matcher restricts its scan to plausible wrapper image names
    (``cmd.exe``/``conhost.exe`` on Windows, POSIX shells on Unix) before the
    cmdline substring test, so the fake must carry a name in that allow-list
    or it is skipped before the cmdline is ever inspected.
    """
    script = runner_dir / _platform_launch_script()
    name = "cmd.exe" if sys.platform == "win32" else "bash"
    return _FakeProc(pid, _FakeInfo({"pid": pid, "name": name, "cmdline": [str(script)]}))


def _listener_proc(runner_dir: Path, pid: int = 200) -> _FakeProc:
    """A process matching the listener matcher for the current platform."""
    if sys.platform == "win32":
        info = _FakeInfo(
            {"pid": pid, "name": "Runner.Listener.exe", "cwd": str(runner_dir), "exe": None}
        )
    else:
        script = runner_dir / "run.sh"
        info = _FakeInfo({"pid": pid, "name": None, "cwd": None, "exe": str(script)})
    return _FakeProc(pid, info)


def test_get_runner_launcher_process_matches_a_cmdline_naming_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher matcher finds a wrapper whose cmdline contains run.cmd/sh."""
    from charlie_work.runners import get_runner_launcher_process

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    _fake_process_iter(monkeypatch, [_launcher_proc(runner_dir)])

    proc = get_runner_launcher_process(runner_dir)

    assert proc is not None
    assert proc.pid == 100


def test_get_runner_launcher_process_returns_none_when_no_cmdline_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrelated process must not be mistaken for this runner's wrapper."""
    from charlie_work.runners import get_runner_launcher_process

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    # A plausible wrapper image (so it clears the name filter) but an
    # unrelated cmdline — the matcher must reject it on the substring test,
    # not vacuously skip it on the name filter.
    other_name = "cmd.exe" if sys.platform == "win32" else "bash"
    other = _FakeProc(
        300, _FakeInfo({"pid": 300, "name": other_name, "cmdline": ["cmd", "/c", "unrelated.cmd"]})
    )
    _fake_process_iter(monkeypatch, [other])

    assert get_runner_launcher_process(runner_dir) is None


def test_get_runner_launcher_process_returns_none_when_the_script_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory without a launch script has nothing to match against."""
    from charlie_work.runners import get_runner_launcher_process

    runner_dir = tmp_path / "r" / "cw-1"
    runner_dir.mkdir(parents=True)
    _fake_process_iter(monkeypatch, [_launcher_proc(runner_dir)])

    assert get_runner_launcher_process(runner_dir) is None


def test_get_runner_launcher_process_skips_a_process_that_became_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that vanished mid-scan cannot be a launcher; siblings still can."""
    psutil = pytest.importorskip("psutil")
    from charlie_work.runners import get_runner_launcher_process

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    dead = _FakeProc(400, _FakeInfo(raises=psutil.NoSuchProcess(pid=400)))
    live = _launcher_proc(runner_dir, pid=401)
    _fake_process_iter(monkeypatch, [dead, live])

    proc = get_runner_launcher_process(runner_dir)

    assert proc is not None
    assert proc.pid == 401


def test_is_runner_launched_is_true_when_only_the_wrapper_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-startup: the launch script is up but the listener is not yet.

    This is the window ``is_runner_launched`` exists to cover — a start path
    that only looked for the listener would launch a second copy of a runner
    already coming up.
    """
    from charlie_work.runners import is_runner_launched

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    _fake_process_iter(monkeypatch, [_launcher_proc(runner_dir)])

    assert is_runner_launched(runner_dir) is True


def test_is_runner_launched_is_true_when_the_listener_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.runners import is_runner_launched

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    _fake_process_iter(monkeypatch, [_listener_proc(runner_dir)])

    assert is_runner_launched(runner_dir) is True


def test_is_runner_launched_is_false_when_neither_wrapper_nor_listener_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative case: a configured runner with nothing live is not launched."""
    from charlie_work.runners import is_runner_launched

    runner_dir = _make_runner_dir(
        tmp_path / "r",
        "cw-1",
        f"https://github.com/{CW}",
        "cw-1",
        script=_platform_launch_script(),
    )
    _fake_process_iter(monkeypatch, [])

    assert is_runner_launched(runner_dir) is False


def test_discovery_marks_a_runner_running_when_its_wrapper_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap the issue names: discover tests never asserted ``.running``.

    A matcher that always returned False would leave every discovered runner
    parked; one that always returned True would start nothing. Driving the real
    ``is_runner_launched`` body through ``discover_runner_instances`` pins the
    field that the start/park planner reads.
    """
    root = tmp_path / "runners"
    root.mkdir()
    runner_dir = _make_runner_dir(
        root, "cw-1", f"https://github.com/{CW}", "cw-1", script=_platform_launch_script()
    )
    _fake_process_iter(monkeypatch, [_launcher_proc(runner_dir)])

    instances, _ = discover_runner_instances(root)

    assert len(instances) == 1
    assert instances[0].running is True


def test_discovery_marks_a_runner_parked_when_nothing_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runners"
    root.mkdir()
    _make_runner_dir(
        root, "cw-1", f"https://github.com/{CW}", "cw-1", script=_platform_launch_script()
    )
    _fake_process_iter(monkeypatch, [])

    instances, _ = discover_runner_instances(root)

    assert len(instances) == 1
    assert instances[0].running is False


# --------------------------------------------------------------------------
# run_allocation_pass — end-to-end
# --------------------------------------------------------------------------
#
# The pass wires discovery, GitHub observation, planning, actuation, and state
# persistence together. Three of its branches had no direct coverage (issue
# #602): the dry-run state-write guard, the zero-runner early skip, and the
# mapping of a busy-list API failure onto a pinned (unmeasurable) repo. These
# drive the real pass body with a fake GitHub client and actuation stubbed at
# the runner_slots boundary, so the assertions target the pass's own logic
# rather than the matchers covered above.


class _FakeGitHub:
    """Routing fake for the two ``gh.run`` shapes the pass issues.

    ``fetch_busy_runner_names`` calls ``repos/{repo}/actions/runners`` and
    treats a raised ``GitHubError`` as a busy-list failure (the pinned path).
    ``measure_repo_demand`` calls ``repos/{repo}/actions/runs?status=...`` and
    ``repos/{repo}/actions/runs/{id}/jobs``. Routes on the API path so one
    object can serve several repos with different responses.
    """

    def __init__(
        self,
        *,
        runner_errors: set[str] | None = None,
        runs: dict[str, dict] | None = None,
        jobs: dict[str, dict] | None = None,
    ) -> None:
        self._runner_errors = runner_errors or set()
        self._runs = runs or {}
        self._jobs = jobs or {}
        self.calls: list[list[str]] = []

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        self.calls.append(args)
        path = args[1] if len(args) > 1 else ""
        repo = path.split("repos/", 1)[1].split("/actions", 1)[0] if "repos/" in path else ""

        # runners list: repos/{repo}/actions/runners?per_page=100
        if "/actions/runners" in path and "/runs/" not in path:
            if repo in self._runner_errors:
                raise GitHubError(f"simulated runners-list failure for {repo}")
            return {"runners": []}

        # jobs for one run: repos/{repo}/actions/runs/{run_id}/jobs?per_page=100
        if "/runs/" in path and "/jobs" in path:
            run_id = path.split("/runs/", 1)[1].split("/", 1)[0]
            return self._jobs.get(run_id, {"jobs": []})

        # runs by status: repos/{repo}/actions/runs?status=...&per_page=...
        if "/actions/runs?" in path:
            return self._runs.get(repo, {"workflow_runs": []})

        return {}


def _make_managed_root(tmp_path: Path, runners: list[tuple[str, str, str]]) -> Path:
    """Build a managed root with one directory per (name, repo, agent_name)."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    script = _platform_launch_script()
    for name, repo, agent in runners:
        _make_runner_dir(root, name, f"https://github.com/{repo}", agent, script=script)
    return root


def _stub_actuation(monkeypatch: pytest.MonkeyPatch, *, launched: bool) -> None:
    """Stop the pass from touching real processes.

    Discovery and the start re-check both call ``is_runner_launched``; park
    calls ``has_active_job`` and the two process matchers. Pointing all of
    them at fixed returns keeps the pass deterministic without bypassing the
    pass's own logic — the matchers themselves are tested above.
    """
    monkeypatch.setattr("charlie_work.runner_slots.is_runner_launched", lambda _path: launched)
    monkeypatch.setattr("charlie_work.runner_slots.has_active_job", lambda _path: False)
    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_listener_process", lambda _path: None
    )
    monkeypatch.setattr(
        "charlie_work.runner_slots.get_runner_launcher_process", lambda _path: None
    )
    monkeypatch.setattr(
        "charlie_work.runner_slots.launch_runner_listener",
        lambda path, dry_run=False: (True, f"launched {path.name} (dry_run={dry_run})"),
    )


def test_run_allocation_pass_skips_early_when_no_runners_are_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty managed root short-circuits before any GitHub call."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    gh = _FakeGitHub()

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(tmp_path / "fleet"),
        source="prologue",
        full_pass_interval_seconds=300,
    )

    assert result.ok is True
    assert result.skipped is True
    assert any("no configured runners" in note for note in result.notes)
    # The early skip must not have reached GitHub observation.
    assert gh.calls == []


def test_run_allocation_pass_pins_a_repo_whose_busy_list_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A busy-list API failure maps to an unmeasurable, pinned repo.

    The allocator must not reallocate capacity away from a repo it cannot see
    — a transient API blip would otherwise strand a queue. ``fetch_busy_runner_names``
    returns a busy_error, the pass builds a ``RepoDemand(ok=False)``, and the
    planner pins that repo to its current running count.
    """
    root = _make_managed_root(
        tmp_path,
        [("cw-1", CW, "cw-1"), ("jc-1", JC, "jc-1")],
    )
    _stub_actuation(monkeypatch, launched=True)
    gh = _FakeGitHub(runner_errors={JC})

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(tmp_path / "fleet"),
        source="prologue",
        full_pass_interval_seconds=300,
    )

    assert result.ok is True
    assert result.plan is not None
    pinned = {t.repo: t.pinned for t in result.plan.targets}
    assert pinned[JC] is True
    assert pinned[CW] is False
    assert any("demand unmeasurable" in note and JC in note for note in result.plan.notes)


def test_run_allocation_pass_dry_run_does_not_persist_idle_streaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run plans and reports but must not advance the hysteresis counters.

    The ``if not dry_run: save_idle_streaks(...)`` guard is what keeps a preview
    pass from writing the state file the next real pass reads — advancing the
    slack streak while previewing would let a later real pass park a slot based
    on passes that never actually happened.
    """
    root = _make_managed_root(tmp_path, [("cw-1", CW, "cw-1"), ("cw-2", CW, "cw-2")])
    _stub_actuation(monkeypatch, launched=False)
    gh = _FakeGitHub(
        runs={CW: {"workflow_runs": [{"id": 11}]}},
        jobs={"11": {"jobs": [{"status": "queued", "labels": ["self-hosted"]}] * 5}},
    )
    fleet = tmp_path / "fleet"
    state_path = tmp_path / "state.json"

    save_calls: list[bool] = []
    log_payloads: list[dict] = []
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.save_idle_streaks",
        lambda *a, **k: save_calls.append(True),
    )
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.log_event",
        lambda _sp, _kind, payload, **_k: log_payloads.append(payload),
    )

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(
            enabled=True, managed_root=str(root), max_running_runners=4, min_running_per_repo=1
        ),
        fleet_dir_override=str(fleet),
        state_path=state_path,
        dry_run=True,
        source="prologue",
        full_pass_interval_seconds=300,
    )

    assert result.ok is True
    # The plan wants both parked runners started (demand 5, capacity 2). The
    # stubbed launcher reports success for both, but dry_run is the pass's
    # state-write concern, not the result count — the guard is asserted below.
    assert result.started == 2
    assert all("dry_run=True" in r.message for r in result.results)
    assert save_calls == []
    assert not (fleet / "runner-allocation.json").exists()
    assert log_payloads and log_payloads[-1]["dry_run"] is True


def test_run_allocation_pass_real_run_persists_idle_streaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-dry-run path writes the state file and logs dry_run=False."""
    root = _make_managed_root(tmp_path, [("cw-1", CW, "cw-1"), ("cw-2", CW, "cw-2")])
    _stub_actuation(monkeypatch, launched=False)
    gh = _FakeGitHub(
        runs={CW: {"workflow_runs": [{"id": 11}]}},
        jobs={"11": {"jobs": [{"status": "queued", "labels": ["self-hosted"]}] * 5}},
    )
    fleet = tmp_path / "fleet"
    state_path = tmp_path / "state.json"

    save_calls: list[bool] = []
    log_payloads: list[dict] = []
    # Delegate to the real writer so the state file actually lands.
    real_save = save_idle_streaks
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.save_idle_streaks",
        lambda *a, **k: (save_calls.append(True), real_save(*a, **k))[1],
    )
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.log_event",
        lambda _sp, _kind, payload, **_k: log_payloads.append(payload),
    )

    try:
        result = run_allocation_pass(
            gh,
            RunnerAllocationConfig(
                enabled=True, managed_root=str(root), max_running_runners=4, min_running_per_repo=1
            ),
            fleet_dir_override=str(fleet),
            state_path=state_path,
            dry_run=False,
            source="prologue",
            full_pass_interval_seconds=300,
        )

        assert result.ok is True
        assert result.started == 2  # both parked runners were started (stubbed)
        assert save_calls == [True]
        assert (fleet / "runner-allocation.json").exists()
        assert log_payloads and log_payloads[-1]["dry_run"] is False
        assert log_payloads[-1]["source"] == "prologue"
    finally:
        close_db(state_path)


# --------------------------------------------------------------------------
# Provenance: driving interval + skip reason (issue #606)
# --------------------------------------------------------------------------


def test_save_idle_streaks_records_the_driving_interval(tmp_path: Path) -> None:
    """The interval the pass was driven at is persisted, not re-resolved later."""
    from charlie_work.runner_slots import load_allocation_stamp

    save_idle_streaks(tmp_path, {CW: 1}, source="prologue", full_pass_interval_seconds=120)
    payload = json.loads((tmp_path / "runner-allocation.json").read_text(encoding="utf-8"))
    assert payload["full_pass_interval_seconds"] == 120
    assert payload["skip_reason"] is None

    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.full_pass_interval_seconds == 120
    assert stamp.skip_reason is None


def test_allocation_stamp_reads_interval_and_skip_reason(tmp_path: Path) -> None:
    """Both new provenance fields round-trip through the state file."""
    from charlie_work.runner_slots import load_allocation_stamp

    (tmp_path / "runner-allocation.json").write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-07-28T00:00:00+00:00",
                "source": "prologue",
                "full_pass_interval_seconds": 90,
                "skip_reason": "no configured runners found under /x",
                "repos": {CW: {"idle_streak": 2}},
            }
        ),
        encoding="utf-8",
    )
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.full_pass_interval_seconds == 90
    assert stamp.skip_reason == "no configured runners found under /x"


def test_allocation_stamp_treats_a_pre_interval_file_as_unknown_interval(
    tmp_path: Path,
) -> None:
    """A file written before interval recording falls back to None, not a guess."""
    from charlie_work.runner_slots import load_allocation_stamp

    (tmp_path / "runner-allocation.json").write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-07-28T00:00:00+00:00",
                "source": "prologue",
                "repos": {},
            }
        ),
        encoding="utf-8",
    )
    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.full_pass_interval_seconds is None
    assert stamp.skip_reason is None


def test_save_allocation_skip_records_the_reason_without_touching_repos(
    tmp_path: Path,
) -> None:
    """A skip writes provenance + reason and preserves the prior idle streaks."""
    from charlie_work.runner_slots import load_allocation_stamp, save_allocation_skip

    # A real pass accumulated hysteresis history first.
    save_idle_streaks(tmp_path, {CW: 2, JC: 1}, source="prologue", full_pass_interval_seconds=300)

    save_allocation_skip(
        tmp_path,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no configured runners found under /runners",
    )

    payload = json.loads((tmp_path / "runner-allocation.json").read_text(encoding="utf-8"))
    assert payload["skip_reason"] == "no configured runners found under /runners"
    assert payload["source"] == "prologue"
    assert payload["full_pass_interval_seconds"] == 300
    # The streaks are preserved, not reset — a transient skip must not zero
    # demotion hysteresis.
    assert payload["repos"][CW]["idle_streak"] == 2
    assert payload["repos"][JC]["idle_streak"] == 1

    stamp = load_allocation_stamp(tmp_path)
    assert stamp is not None
    assert stamp.skip_reason == "no configured runners found under /runners"
    assert stamp.full_pass_interval_seconds == 300


def test_save_allocation_skip_preserves_repos_when_no_prior_state(tmp_path: Path) -> None:
    """A skip with no prior file writes an empty repos map, not a missing one."""
    from charlie_work.runner_slots import save_allocation_skip

    save_allocation_skip(
        tmp_path,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="managed_root does not exist: /nope",
    )
    payload = json.loads((tmp_path / "runner-allocation.json").read_text(encoding="utf-8"))
    assert payload["repos"] == {}
    assert payload["skip_reason"] == "managed_root does not exist: /nope"


def test_save_allocation_skip_write_is_atomic(tmp_path: Path) -> None:
    """The skip writer shares the temp-file + replace invariant."""
    from charlie_work.runner_slots import save_allocation_skip

    save_allocation_skip(
        tmp_path,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="none",
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_allocation_skip_preserves_the_tie_break_offset(tmp_path: Path) -> None:
    """A skip must not reset the floor-shortfall rotation (issues #601 + #606).

    save_idle_streaks and save_allocation_skip share one writer; without an
    explicit read-back the skip would write the default 0 and re-starve the
    repo the rotation had just moved past.
    """
    from charlie_work.runner_slots import load_tie_break_offset, save_allocation_skip

    save_idle_streaks(
        tmp_path, {CW: 1}, source="prologue", full_pass_interval_seconds=300, tie_break_offset=3
    )
    save_allocation_skip(
        tmp_path,
        source="prologue",
        full_pass_interval_seconds=300,
        skip_reason="no runners",
    )
    assert load_tie_break_offset(tmp_path) == 3


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
        full_pass_interval_seconds=300,
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
        full_pass_interval_seconds=300,
    )
    assert r2.ok is True
    assert r2.plan is not None
    t2 = {t.repo: t.target for t in r2.plan.targets}
    assert t2 == {"o/A": 1, "o/B": 0, "o/C": 1}
    assert load_tie_break_offset(fleet_dir) == 2


# --------------------------------------------------------------------------
# starved_repos — pure detection (issue #799)
# --------------------------------------------------------------------------
#
# Fast, direct checks of the pure function against hand-built plans. These do
# not exercise run_allocation_pass's edge-triggering — see the
# run_allocation_pass section below for that, driven through the real entry
# point rather than a synthesized plan.


def test_starved_repos_detects_demand_exceeding_capacity_with_spare_budget() -> None:
    """demand > capacity, with the host budget undersubscribed, is starvation."""
    plan = AllocationPlan(
        budget=4,
        budget_reason="test",
        targets=(RepoTarget(repo="Fake/repo-xyz", target=2, running=2, demand=9, capacity=2),),
        changes=(),
    )

    starved = starved_repos(plan)

    assert len(starved) == 1
    signal = starved[0]
    assert signal.repo == "Fake/repo-xyz"
    assert signal.demand == 9
    assert signal.capacity == 2
    assert signal.running == 2
    assert signal.spare_budget == 2  # budget 4 - running 2


def test_starved_repos_silent_when_demand_within_capacity() -> None:
    """Positive control: a healthy target set must signal nothing.

    Without this, a detector that always fires (or never fires) would still
    pass a "fires when starved" test alone.
    """
    plan = AllocationPlan(
        budget=4,
        budget_reason="test",
        targets=(
            RepoTarget(repo="Fake/repo-a", target=1, running=1, demand=1, capacity=2),
            RepoTarget(repo="Fake/repo-b", target=1, running=1, demand=1, capacity=1),
        ),
        changes=(),
    )

    assert starved_repos(plan) == ()


def test_starved_repos_silent_when_budget_fully_subscribed() -> None:
    """demand > capacity alone is not enough; a saturated budget has no slack
    a bigger registration could use, so the signal must stay silent."""
    plan = AllocationPlan(
        budget=2,
        budget_reason="test",
        targets=(RepoTarget(repo="Fake/repo-xyz", target=2, running=2, demand=9, capacity=2),),
        changes=(),
    )

    assert starved_repos(plan) == ()


# --------------------------------------------------------------------------
# run_allocation_pass — runner_capacity_starved / _recovered events (#799)
# --------------------------------------------------------------------------
#
# These drive the REAL run_allocation_pass entrypoint — the same function
# both `charlie runners allocate` (cli.py's run_runners_allocate, which calls
# run_allocation_pass with state_path=paths.state_file) and the fleet
# prologue (fleet_dispatch.py's _run_fleet_allocation_prologue, which calls
# it with state_path=anchor_state) invoke — rather than the private
# _emit_capacity_events helper directly. A test that only called the helper
# would keep passing even if it were no longer wired into the real pass.


def _capacity_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repo: str,
    demand: int,
    capacity: int,
    budget: int,
    state_path: Path | None,
    dry_run: bool = False,
):
    """Run the real run_allocation_pass against one synthesized repo.

    ``repo`` is caller-supplied and deliberately not one of this module's
    CW/JC/PUB constants in most callers below, so a repo that only exists
    because this test invented it still drives the signal correctly --
    proving nothing in the detector or the pass is keyed off a known name.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir(exist_ok=True)
    for i in range(capacity):
        dirname = f"{repo.replace('/', '-')}-{i}"
        if not (managed_root / dirname).exists():
            _make_runner_dir(managed_root, dirname, f"https://github.com/{repo}", dirname)

    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.measure_repo_demand",
        lambda gh, r, max_runs_scanned: RepoDemand(repo=r, queued_jobs=demand),
    )
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.fetch_busy_runner_names",
        lambda gh, r: (set(), None),
    )
    # No real actuation: these runner dirs have no listener process, so
    # starting/parking them would touch the host. The plan and the event log
    # are what these tests inspect.
    monkeypatch.setattr(
        "charlie_work.runner_allocation_pass.apply_allocation",
        lambda plan, dry_run=False: [],
    )

    config = RunnerAllocationConfig(
        enabled=True,
        managed_root=str(managed_root),
        max_running_runners=budget,
        min_running_per_repo=0,
        demand_idle_samples=3,
    )

    class _FakeGh:
        """Stand-in: every gh.run call is intercepted by the monkeypatches above."""

    return run_allocation_pass(
        _FakeGh(),  # type: ignore[arg-type]
        config,
        fleet_dir_override=str(tmp_path / "fleet"),
        state_path=state_path,
        dry_run=dry_run,
        source="prologue",
        full_pass_interval_seconds=300,
    )


def test_run_allocation_pass_emits_runner_capacity_starved_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthesized starved target set produces exactly one starved event,
    written through the real run_allocation_pass entrypoint."""
    state_path = tmp_path / "state.json"
    try:
        result = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-xyz",
            demand=9,
            capacity=1,
            budget=4,
            state_path=state_path,
        )
        assert result.ok is True

        events = query_events(state_path, kind="runner_capacity_starved")
        assert len(events) == 1
        assert events[0]["repo"] == "Fake/repo-xyz"
        assert events[0]["payload"]["demand"] == 9
        assert events[0]["payload"]["capacity"] == 1
        assert events[0]["payload"]["spare_budget"] == 4
        assert events[0]["level"] == "warning"
    finally:
        close_db(state_path)


def test_run_allocation_pass_positive_control_silence_on_healthy_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mandatory positive control: a healthy synthesized target set (demand <=
    capacity) must write zero capacity events through the real entrypoint."""
    state_path = tmp_path / "state.json"
    try:
        result = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-healthy",
            demand=1,
            capacity=2,
            budget=4,
            state_path=state_path,
        )
        assert result.ok is True

        assert query_events(state_path, kind="runner_capacity_starved") == []
        assert query_events(state_path, kind="runner_capacity_recovered") == []
    finally:
        close_db(state_path)


def test_run_allocation_pass_dry_run_writes_no_capacity_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run previews a starved plan without writing the event -- an
    event write is a side effect, and dry-run must have none."""
    state_path = tmp_path / "state.json"
    try:
        result = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-xyz",
            demand=9,
            capacity=1,
            budget=4,
            state_path=state_path,
            dry_run=True,
        )
        assert result.ok is True

        assert query_events(state_path, kind="runner_capacity_starved") == []
    finally:
        close_db(state_path)


def test_run_allocation_pass_emits_exactly_one_starved_event_across_n_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion: N consecutive passes with a persistently starved
    repo produce exactly ONE runner_capacity_starved event, not N.

    demand > capacity while the budget has slack is this host's steady state
    for days at a time (registration only moves on a separate, much slower,
    possibly-disabled provisioning cadence) -- a level-triggered emit would
    write one row per repo per pass forever. This is the test that would have
    passed under that broken design; it is the one that actually matters.
    """
    state_path = tmp_path / "state.json"
    try:
        for _ in range(4):
            result = _capacity_pass(
                tmp_path,
                monkeypatch,
                repo="Fake/repo-xyz",
                demand=9,
                capacity=1,
                budget=4,
                state_path=state_path,
            )
            assert result.ok is True

        events = query_events(state_path, kind="runner_capacity_starved")
        assert len(events) == 1
    finally:
        close_db(state_path)


def test_run_allocation_pass_emits_recovery_event_on_transition_out_of_starvation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo transitioning from starved to healthy fires exactly one
    runner_capacity_recovered event -- and staying healthy afterward does not
    re-fire it. Without this, a reader cannot tell "recovered" from "the
    signal stopped working"."""
    state_path = tmp_path / "state.json"
    try:
        r1 = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-xyz",
            demand=9,
            capacity=1,
            budget=4,
            state_path=state_path,
        )
        assert r1.ok is True
        assert len(query_events(state_path, kind="runner_capacity_starved")) == 1
        assert query_events(state_path, kind="runner_capacity_recovered") == []

        # Demand drops to at-or-below capacity: the repo recovers.
        r2 = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-xyz",
            demand=1,
            capacity=1,
            budget=4,
            state_path=state_path,
        )
        assert r2.ok is True
        assert len(query_events(state_path, kind="runner_capacity_starved")) == 1
        recovered = query_events(state_path, kind="runner_capacity_recovered")
        assert len(recovered) == 1
        assert recovered[0]["repo"] == "Fake/repo-xyz"

        # A further healthy pass must not re-fire the recovery event.
        r3 = _capacity_pass(
            tmp_path,
            monkeypatch,
            repo="Fake/repo-xyz",
            demand=1,
            capacity=1,
            budget=4,
            state_path=state_path,
        )
        assert r3.ok is True
        assert len(query_events(state_path, kind="runner_capacity_starved")) == 1
        assert len(query_events(state_path, kind="runner_capacity_recovered")) == 1
    finally:
        close_db(state_path)
