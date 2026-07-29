"""Policy core for host-wide elastic allocation of self-hosted runner slots.

``runners.py`` scales a *single* repo's pool vertically: it mints registration
tokens, extracts runner packages, and deregisters runners as that repo's queue
pressure moves (epic #231). It cannot take capacity away from an idle repo and
give it to a busy one — every repo's parallelism is fixed by however many
runners were registered to it, so a saturated repo queues behind its own cap
while another repo's runners sit idle on the same machine.

This module supplies the missing dimension. It treats the host as owning one
budget of concurrently *running listeners* and distributes that budget across
every repo that has runners registered under the managed root, in proportion to
each repo's live Actions demand.

The key move is that **registration is never touched**. A configured runner
whose listener is stopped simply reports ``offline`` to GitHub and keeps its
registration; starting the listener again brings it back with the credentials
already on disk. So a slot moves between repos in about a second, with no
registration token, no GitHub write, and no package extraction — cheap enough
to re-evaluate every fleet pass. What ``runners.py`` provisions, this module
schedules.

Everything here is pure: observations in, decisions out, no I/O. The host and
GitHub side lives in ``runner_slots.py``; the pass that wires them together
lives in ``runner_allocation_pass.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class SlotAction(str, Enum):
    """What an allocation plan wants done to one runner slot."""

    START = "start"  # Bring a parked (configured, offline) listener online
    PARK = "park"  # Stop an idle listener, leaving its registration intact


@dataclass(frozen=True)
class RunnerInstance:
    """One configured runner directory on this host.

    ``repo`` and ``name`` come from the runner's own ``.runner`` file, so this
    reflects what the runner is actually registered to — not what its
    directory happens to be named.
    """

    path: Path
    name: str  # GitHub runner name (``.runner`` agentName)
    repo: str  # "owner/name", derived from ``.runner`` gitHubUrl
    running: bool  # A listener process for this directory is alive locally
    busy: bool = False  # GitHub reports this runner as executing a job


@dataclass(frozen=True)
class RepoDemand:
    """Live self-hosted Actions demand for one repo.

    ``queued_jobs`` want a slot now; ``in_progress_jobs`` already occupy one.
    Their sum is how many slots the repo could usefully hold this instant.

    ``ok`` False means the measurement failed (API blip, auth, rate limit).
    Such a repo is *pinned* to its current running count rather than treated
    as idle — reallocating capacity away from a repo we cannot see would turn
    a transient API failure into a starved queue.
    """

    repo: str
    queued_jobs: int = 0
    in_progress_jobs: int = 0
    ok: bool = True
    error: str | None = None
    truncated: bool = False  # Run scan hit max_runs_scanned

    @property
    def demand(self) -> int:
        return self.queued_jobs + self.in_progress_jobs


@dataclass(frozen=True)
class RepoTarget:
    """The allocator's verdict for one repo, with the inputs that produced it."""

    repo: str
    target: int  # Listeners that should be running
    running: int  # Listeners running now
    demand: int  # Slots the repo could use
    capacity: int  # Registered runner directories (hard ceiling)
    pinned: bool = False  # Demand unmeasurable; held at current running count


@dataclass(frozen=True)
class SlotChange:
    """One start/park action against a specific runner directory."""

    repo: str
    runner_name: str
    path: Path
    action: SlotAction
    reason: str


@dataclass(frozen=True)
class AllocationPlan:
    """A complete, immutable allocation decision for the host.

    ``notes`` carries every place the plan was bounded — capacity ceilings,
    budget shortfalls, deferred demotions, busy slots that could not be
    reclaimed. A silently truncated plan reads as "nothing to do" when it
    isn't, so bounds are reported rather than swallowed.
    """

    budget: int
    budget_reason: str
    targets: tuple[RepoTarget, ...]
    changes: tuple[SlotChange, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotChangeResult:
    """Outcome of applying one SlotChange. Errors are values, never raises."""

    change: SlotChange
    ok: bool
    message: str


def repo_slug_from_github_url(url: str) -> str | None:
    """Extract "owner/name" from a runner's registration URL.

    Takes the last two path segments so GitHub Enterprise hosts and any
    ``.git``-suffixed form resolve the same way as github.com.
    """
    if not url:
        return None
    trimmed = url.strip().rstrip("/")
    # Drop scheme, then the host, keeping only path segments.
    without_scheme = trimmed.split("://", 1)[-1]
    segments = [seg for seg in without_scheme.split("/")[1:] if seg]
    if len(segments) < 2:
        return None
    owner, name = segments[-2], segments[-1]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    return f"{owner}/{name}"


def annotate_busy(
    instances: list[RunnerInstance],
    busy_by_repo: Mapping[str, set[str]],
) -> list[RunnerInstance]:
    """Return copies of ``instances`` with GitHub's busy flag applied."""
    return [
        RunnerInstance(
            path=inst.path,
            name=inst.name,
            repo=inst.repo,
            running=inst.running,
            busy=inst.name in busy_by_repo.get(inst.repo, set()),
        )
        for inst in instances
    ]


def derive_budget(configured: int, cpu_count: int | None = None) -> tuple[int, str]:
    """Resolve the host's concurrent-job budget.

    A configured value always wins. The derived fallback is deliberately
    conservative — half the logical cores — because CI jobs here are test
    suites that themselves fan out, and they share the host with worker and
    reviewer processes this function cannot observe. It exists so the feature
    degrades to something sane when unconfigured, not as a tuned recommendation.
    """
    if configured > 0:
        return configured, f"configured max_running_runners={configured}"

    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    derived = max(1, cores // 2)
    return derived, f"derived from {cores} logical cores (cores // 2)"


def allocate_slots(
    demands: Mapping[str, int],
    capacities: Mapping[str, int],
    budget: int,
    min_per_repo: int,
    pinned: Mapping[str, int] | None = None,
    *,
    tie_break_offset: int = 0,
) -> dict[str, int]:
    """Distribute ``budget`` running slots across repos by demand.

    This is the pure heart of the module: observations in, targets out.

    Policy, in order:

    1. Repos whose demand could not be measured are *pinned* to the slot count
       they already hold; that capacity is removed from the budget before
       anything else is decided.
    2. Every remaining repo receives a floor of ``min_per_repo`` slots (capped
       by how many runners it actually has registered) so no repo's queue can
       go unclaimed.
    3. Leftover budget is water-filled one slot at a time to whichever repo has
       the largest unmet demand, breaking ties toward the repo with the fewest
       slots held. This is max-min fair *per pass*: a burst in one repo cannot
       starve another that also has work within a single call.
    4. No repo is ever given more slots than it has registered runners, and
       spare budget beyond total demand is left unused rather than spent on
       listeners nobody needs.

    When floors alone exceed the budget, slots go to the hungriest repos first
    and some repos get nothing; the caller surfaces that as a note.

    Ties (equal hunger in step 2, equal shortfall in step 3) break on a
    *rotated* name order driven by ``tie_break_offset``. Without rotation, the
    same repo loses every name tie on every pass — e.g. demand A=5 B=1 C=1,
    budget 2, ``min_per_repo`` 1 always starves C. The caller increments the
    offset each pass so the loser rotates, spreading floor shortfall fairly
    across passes rather than permanently punishing whichever repo sorts last.
    A single call with a fixed offset is still deterministic.
    """
    pinned = pinned or {}
    targets: dict[str, int] = {}

    # Step 1: honor pins first — unmeasurable repos keep what they hold.
    remaining = budget
    for repo in sorted(pinned):
        held = max(0, min(pinned[repo], capacities.get(repo, 0)))
        targets[repo] = held
        remaining -= held

    elastic = sorted(repo for repo in capacities if repo not in pinned)
    for repo in elastic:
        targets[repo] = 0

    remaining = max(0, remaining)

    def want(repo: str) -> int:
        return min(demands.get(repo, 0), capacities.get(repo, 0))

    # Rotated name priority: ``(index + offset) % len(elastic)`` shifts which
    # repo wins a name tie so the same repo is not permanently punished. With
    # elastic [B, C] and offset 0, B outranks C; with offset 1, C outranks B.
    # The modular addition (not a cyclic list rotation) is what actually flips
    # the relative order of tied repos — a plain rotation preserves adjacency.
    n = len(elastic)
    name_priority = {repo: (i + tie_break_offset) % n for i, repo in enumerate(elastic)}

    floors = {repo: min(min_per_repo, capacities.get(repo, 0)) for repo in elastic}

    # Step 2: floors. If they don't all fit, the hungriest repos are served
    # first — but every repo still gets a shot before anyone gets a second slot.
    if sum(floors.values()) > remaining:
        by_hunger = sorted(elastic, key=lambda r: (-want(r), name_priority[r]))
        progressed = True
        while remaining > 0 and progressed:
            progressed = False
            for repo in by_hunger:
                if remaining == 0:
                    break
                if targets[repo] < floors[repo]:
                    targets[repo] += 1
                    remaining -= 1
                    progressed = True
        return targets

    for repo in elastic:
        targets[repo] = floors[repo]
        remaining -= floors[repo]

    # Step 3: water-fill the remainder by largest unmet demand.
    while remaining > 0:
        candidates = [repo for repo in elastic if targets[repo] < want(repo)]
        if not candidates:
            break
        # Largest shortfall wins; then fewest slots held; then rotated name.
        chosen = min(
            candidates,
            key=lambda r: (-(want(r) - targets[r]), targets[r], name_priority[r]),
        )
        targets[chosen] += 1
        remaining -= 1

    return targets


def plan_allocation(
    instances: list[RunnerInstance],
    demands: Mapping[str, RepoDemand],
    *,
    budget: int,
    budget_reason: str,
    min_per_repo: int,
    idle_streaks: Mapping[str, int],
    demand_idle_samples: int,
    tie_break_offset: int = 0,
) -> AllocationPlan:
    """Turn observations into a concrete set of start/park actions.

    Demotion is asymmetric with promotion. A repo gains slots on the first
    pass that shows demand, but only loses an over-allocated slot once it has
    looked slack for ``demand_idle_samples`` consecutive passes — *unless*
    another repo currently has unmet demand, in which case the idle slot is
    reclaimed immediately. Waiting to hand a spare slot to a repo that is
    actively queuing would defeat the point; waiting before parking a slot
    nobody wants avoids pointless churn.

    A mature slack streak is also left in place when the host's budget is
    undersubscribed and all pending starts still fit within it, because
    parking a slot only to restart it a few passes later wastes CI latency
    for host resources that are not the binding constraint (issue #628).

    Slots executing jobs are never selected for parking, so a plan can be
    smaller than the target arithmetic implies; that gap is reported in notes.
    """
    by_repo: dict[str, list[RunnerInstance]] = {}
    for inst in instances:
        by_repo.setdefault(inst.repo, []).append(inst)

    capacities = {repo: len(insts) for repo, insts in by_repo.items()}
    running_counts = {repo: sum(1 for i in insts if i.running) for repo, insts in by_repo.items()}
    demand_values = {repo: demands.get(repo, RepoDemand(repo)).demand for repo in by_repo}
    pinned = {
        repo: running_counts[repo]
        for repo in by_repo
        if not demands.get(repo, RepoDemand(repo)).ok
    }

    targets = allocate_slots(
        demand_values,
        capacities,
        budget,
        min_per_repo,
        pinned=pinned,
        tie_break_offset=tie_break_offset,
    )

    notes: list[str] = []
    for repo in sorted(by_repo):
        measurement = demands.get(repo, RepoDemand(repo))
        if not measurement.ok:
            notes.append(
                f"{repo}: demand unmeasurable ({measurement.error}); "
                f"pinned at {running_counts[repo]} running slot(s)"
            )
        if measurement.truncated:
            notes.append(
                f"{repo}: run scan hit the max_runs_scanned cap; demand may be under-counted"
            )
        if demand_values[repo] > capacities[repo]:
            notes.append(
                f"{repo}: demand {demand_values[repo]} exceeds its {capacities[repo]} "
                f"registered runner(s) — provision more to use spare budget"
            )

    floors_requested = sum(min(min_per_repo, capacities[r]) for r in capacities if r not in pinned)
    if floors_requested > max(0, budget - sum(pinned.values())):
        notes.append(
            f"budget {budget} cannot cover a {min_per_repo}-slot floor for "
            f"{len(capacities) - len(pinned)} repo(s); hungriest served first"
        )

    # Someone is starved: their fair target is below what they could use.
    contended = any(
        targets.get(repo, 0) < min(demand_values[repo], capacities[repo]) for repo in by_repo
    )

    total_running = sum(running_counts.values())
    pending_starts = sum(max(0, targets.get(repo, 0) - running_counts[repo]) for repo in by_repo)
    post_without_parks = total_running + pending_starts
    # A purely-idle surplus should not be parked while the budget is
    # undersubscribed (current running plus all planned starts still fit),
    # because parking then only saves host resources while a restore costs
    # real CI latency. See issue #628.
    budget_undersubscribed = total_running < budget and post_without_parks <= budget

    changes: list[SlotChange] = []
    target_records: list[RepoTarget] = []

    # The plan's targets sum to at most the budget (pins aside), but the
    # post-plan running total can exceed it when surplus listeners cannot be
    # parked. Three causes are possible, and only the ones actually present
    # belong in the over-budget note — otherwise the operator is told "busy
    # listeners" when the real cause is a demotion grace period (issue #601).
    held_by_pin = sum(pinned.values())
    held_by_hysteresis = 0
    held_by_busy = 0

    for repo in sorted(by_repo):
        target = targets.get(repo, 0)
        running = [i for i in by_repo[repo] if i.running]
        parked = [i for i in by_repo[repo] if not i.running]

        target_records.append(
            RepoTarget(
                repo=repo,
                target=target,
                running=len(running),
                demand=demand_values[repo],
                capacity=capacities[repo],
                pinned=repo in pinned,
            )
        )

        if target > len(running):
            needed = target - len(running)
            # ``target <= capacity == running + parked`` always holds, so
            # ``needed <= len(parked)`` — there is always a parked runner to
            # start. (Issue #601: the ``needed > len(parked)`` guard that used
            # to live here was unreachable and has been removed.)
            for inst in sorted(parked, key=lambda i: i.name)[:needed]:
                changes.append(
                    SlotChange(
                        repo=repo,
                        runner_name=inst.name,
                        path=inst.path,
                        action=SlotAction.START,
                        reason=(
                            f"demand {demand_values[repo]}, target {target}, "
                            f"{len(running)} running"
                        ),
                    )
                )
            continue

        if target < len(running):
            surplus = len(running) - target
            streak = idle_streaks.get(repo, 0)
            if not contended and (streak < demand_idle_samples or budget_undersubscribed):
                if streak < demand_idle_samples:
                    notes.append(
                        f"{repo}: holding {surplus} surplus slot(s) — slack for {streak}/"
                        f"{demand_idle_samples} pass(es) and no repo is waiting"
                    )
                    held_by_hysteresis += surplus
                else:
                    notes.append(
                        f"{repo}: holding {surplus} surplus slot(s) — "
                        f"budget undersubscribed ({total_running}/{budget} running) "
                        f"and no repo is waiting"
                    )
                continue

            reclaimable = sorted(
                (i for i in running if not i.busy), key=lambda i: i.name, reverse=True
            )
            for inst in reclaimable[:surplus]:
                changes.append(
                    SlotChange(
                        repo=repo,
                        runner_name=inst.name,
                        path=inst.path,
                        action=SlotAction.PARK,
                        reason=(
                            "reclaimed for a waiting repo"
                            if contended
                            else f"slack for {streak} consecutive pass(es)"
                        ),
                    )
                )
            if len(reclaimable) < surplus:
                held_busy = surplus - len(reclaimable)
                held_by_busy += held_busy
                notes.append(
                    f"{repo}: {held_busy} surplus slot(s) left running "
                    f"because they are executing jobs"
                )

    # The host can sit above its budget for three reasons the allocator
    # refuses to "fix" by parking: a repo whose demand we cannot read is pinned
    # to its running count, a repo whose surplus listeners are executing jobs,
    # and a repo whose surplus listeners are still inside the demotion
    # hysteresis grace period. In each case parking would strand or abort work
    # (or churn a slot nobody is waiting for), so the overage is reported
    # rather than silently absorbed. The sum of *targets* can equal the budget
    # while the post-plan running total exceeds it — listeners that cannot be
    # parked keep running on top of the starts the plan issues (issue #601).
    # The note names only the causes actually present, so a grace-period
    # overage is not mislabeled as busy listeners or unmeasurable repos.
    post_running = sum(running_counts.values())
    for change in changes:
        if change.action is SlotAction.START:
            post_running += 1
        else:
            post_running -= 1
    if post_running > budget:
        causes: list[str] = []
        if held_by_pin:
            causes.append(f"{held_by_pin} pinned (unmeasurable)")
        if held_by_busy:
            causes.append(f"{held_by_busy} busy")
        if held_by_hysteresis:
            causes.append(f"{held_by_hysteresis} in demotion hysteresis")
        cause_list = "; ".join(causes) if causes else "no single cause tracked"
        notes.append(
            f"{post_running} slot(s) running above the {budget}-slot budget "
            f"({cause_list}); self-corrects as pinned repos become measurable, "
            f"busy jobs finish, or slack streaks mature"
        )

    return AllocationPlan(
        budget=budget,
        budget_reason=budget_reason,
        targets=tuple(target_records),
        changes=tuple(changes),
        notes=tuple(notes),
    )


def next_idle_streaks(
    instances: list[RunnerInstance],
    demands: Mapping[str, RepoDemand],
    previous: Mapping[str, int],
) -> dict[str, int]:
    """Advance the per-repo slack streak used for demotion hysteresis.

    A repo is slack on this pass when it holds more running listeners than it
    could use. Streaks reset the moment demand catches up, and unmeasurable
    repos hold their streak rather than accruing slack on missing data.
    """
    by_repo: dict[str, list[RunnerInstance]] = {}
    for inst in instances:
        by_repo.setdefault(inst.repo, []).append(inst)

    streaks: dict[str, int] = {}
    for repo, insts in by_repo.items():
        measurement = demands.get(repo, RepoDemand(repo))
        running = sum(1 for i in insts if i.running)
        if not measurement.ok:
            streaks[repo] = previous.get(repo, 0)
            continue
        usable = min(measurement.demand, len(insts))
        streaks[repo] = previous.get(repo, 0) + 1 if usable < running else 0
    return streaks


def plan_summary(plan: AllocationPlan) -> dict[str, Any]:
    """Serializable summary for CLI output and event payloads."""
    return {
        "budget": plan.budget,
        "budget_reason": plan.budget_reason,
        "targets": [
            {
                "repo": t.repo,
                "target": t.target,
                "running": t.running,
                "demand": t.demand,
                "capacity": t.capacity,
                "pinned": t.pinned,
            }
            for t in plan.targets
        ],
        "changes": [
            {
                "repo": c.repo,
                "runner": c.runner_name,
                "action": c.action.value,
                "reason": c.reason,
            }
            for c in plan.changes
        ],
        "notes": list(plan.notes),
    }
