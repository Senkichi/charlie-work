"""One end-to-end runner-slot allocation pass: observe, decide, actuate, record.

This is the single entry point both the ``charlie runners allocate`` CLI command
and the fleet pass prologue call, so an operator running it by hand and the
scheduled fleet loop take exactly the same code path.

Ordering matters and is deliberate:

1. Discover the host's configured runners and derive the repo set from their
   ``.runner`` files — never from a configured list, so a repo appears the
   moment a runner is registered to it.
2. Measure each repo's live demand and busy set.
3. Plan against the *whole* host, then actuate.
4. Persist slack history last, and only for a real run — a ``--dry-run`` must
   not advance the hysteresis counters it is previewing. The same guard covers
   the ``runner_capacity_starved``/``runner_capacity_recovered`` events (issue
   #799): a dry-run's whole point is to preview without side effects, and an
   event write is a side effect.

The caller is expected to hold the fleet lock (the fleet pass does). Actuation
re-checks liveness per slot, so a concurrent pass degrades to redundant no-ops
rather than double-starting a listener.

Capacity-starvation signaling (issue #799) is edge-triggered, not
level-triggered. ``demand > capacity`` while the budget has slack can be this
host's steady state for days — registration only changes on a separate, much
slower provisioning cadence (``runners.py``) that may be disabled entirely. A
naive "emit while true" write would turn one real signal into an unbounded
stream of identical rows in the append-only ``events`` table, one per repo per
pass, forever. Instead ``_emit_capacity_events`` fires ``runner_capacity_starved``
once on the pass a repo's condition turns true, then stays silent every
subsequent pass the condition holds, and fires ``runner_capacity_recovered``
once when it turns false again — so a reader can always tell "still starved"
from "signal stopped working" instead of the silence being ambiguous.

This module is **not** the live implementation. Nothing in ``src/`` imports it;
``fleet_dispatch`` and ``cli`` both resolve ``run_allocation_pass`` through
``ci_fleet.charlie_work_adapter``, which re-exports
``ci_fleet.runner_allocation_pass``. Only tests import this copy. Treat the
ci_fleet module as authoritative and keep behavioural changes there; the
rationale below is retained because this copy carries the same logic and the
same trap.

That dedup state cannot live in memory, and the reason is *not* that each pass
is a fresh process — the fleet supervisor calls the pass in-process from inside
its own loop and persists across passes. A module-global would therefore
survive for a whole supervisor lifetime and lose its state only on respawn:
a failure that passes every test and then misfires non-deterministically, which
is worse than one that breaks immediately. So it is derived entirely from
``events.db`` itself (no new state file): the prior signaled state for a repo
is "more ``runner_capacity_starved`` rows than ``runner_capacity_recovered``
rows", which is immune to same-second timestamp collisions (``_now_iso()``
truncates to whole seconds) because it counts rows rather than ordering them
by timestamp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import RunnerAllocationConfig
from .fleet_paths import fleet_dir
from .github import GitHubLike
from .instrumentation import log_event, query_events
from .runner_allocation import (
    AllocationPlan,
    RepoDemand,
    SlotChangeResult,
    annotate_busy,
    derive_budget,
    next_idle_streaks,
    plan_allocation,
    plan_summary,
    starved_repos,
)
from .runner_slots import (
    apply_allocation,
    discover_runner_instances,
    fetch_busy_runner_names,
    load_idle_streaks,
    load_tie_break_offset,
    measure_repo_demand,
    AllocationSource,
    save_allocation_skip,
    save_idle_streaks,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationPassResult:
    """Outcome of one allocation pass. Errors are values, never raises."""

    ok: bool
    plan: AllocationPlan | None = None
    results: tuple[SlotChangeResult, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None
    skipped: bool = False  # Feature disabled or nothing to allocate

    @property
    def started(self) -> int:
        return sum(1 for r in self.results if r.ok and r.change.action.value == "start")

    @property
    def parked(self) -> int:
        return sum(1 for r in self.results if r.ok and r.change.action.value == "park")


@dataclass(frozen=True)
class AllocationInputs:
    """Resolved configuration for a pass, so callers can log what was used."""

    managed_root: Path
    budget: int
    budget_reason: str
    min_per_repo: int
    demand_idle_samples: int
    max_runs_scanned: int
    notes: tuple[str, ...] = field(default=())


def resolve_inputs(
    allocation: RunnerAllocationConfig,
    managed_root_fallback: str = "",
) -> tuple[AllocationInputs | None, str | None]:
    """Resolve the managed root and budget, or explain why the pass cannot run.

    ``managed_root`` falls back to ``runner_scaling.managed_root`` so the path
    is configured once per host rather than duplicated across two sections.
    """
    root_value = allocation.managed_root or managed_root_fallback
    if not root_value:
        return None, (
            "runner_allocation.managed_root is not configured "
            "(and runner_scaling.managed_root is empty)"
        )

    if allocation.min_running_per_repo < 0:
        return None, "runner_allocation.min_running_per_repo must be >= 0"

    root = Path(root_value)
    if not root.exists():
        # The operator explicitly enabled the feature and named this path. A
        # typo that silently allocates nothing is the failure mode this check
        # exists to prevent.
        return None, f"runner_allocation.managed_root does not exist: {root}"

    budget, budget_reason = derive_budget(allocation.max_running_runners)
    notes: list[str] = []
    if allocation.max_running_runners <= 0:
        notes.append(
            f"max_running_runners unset; using {budget} ({budget_reason}). "
            "Set it explicitly once the host's real CI ceiling is known — this "
            "default cannot see worker or reviewer load."
        )

    return (
        AllocationInputs(
            managed_root=root,
            budget=budget,
            budget_reason=budget_reason,
            min_per_repo=allocation.min_running_per_repo,
            demand_idle_samples=allocation.demand_idle_samples,
            max_runs_scanned=max(1, allocation.max_runs_scanned),
            notes=tuple(notes),
        ),
        None,
    )


def _emit_capacity_events(state_path: Path, plan: AllocationPlan) -> None:
    """Edge-triggered ``runner_capacity_starved``/``_recovered`` events (#799).

    Fires ``runner_capacity_starved`` the pass a repo's condition (``demand >
    capacity`` while ``starved_repos`` sees host-wide budget slack) turns
    true, and ``runner_capacity_recovered`` the pass it turns back false.
    Silent on every pass in between, however long the condition persists —
    see the module docstring for why level-triggered emission is wrong here.

    Prior state is derived from ``events.db`` itself, per repo: strictly more
    ``runner_capacity_starved`` rows than ``runner_capacity_recovered`` rows
    means "currently signaled starved". The two kinds are only ever written
    in strict alternation (this function is the sole writer of both), so a
    row *count* comparison determines "which fired last" without needing to
    order across kinds by timestamp — ``_now_iso()`` truncates to whole
    seconds, so two kinds written in the same second would tie under a
    timestamp comparison but never under a count comparison.

    Iterates ``plan.targets`` — the same computed, live-discovered set
    ``plan_allocation`` builds — so no repo name is ever hardcoded here.
    """
    starved_by_repo = {s.repo: s for s in starved_repos(plan)}
    targets_by_repo = {t.repo: t for t in plan.targets}
    spare_budget = plan.budget - sum(t.running for t in plan.targets)

    for repo in sorted(targets_by_repo):
        starved_count = len(query_events(state_path, repo=repo, kind="runner_capacity_starved"))
        recovered_count = len(
            query_events(state_path, repo=repo, kind="runner_capacity_recovered")
        )
        was_starved = starved_count > recovered_count
        is_starved = repo in starved_by_repo

        if is_starved and not was_starved:
            s = starved_by_repo[repo]
            log_event(
                state_path,
                "runner_capacity_starved",
                {
                    "repo": s.repo,
                    "demand": s.demand,
                    "capacity": s.capacity,
                    "running": s.running,
                    "spare_budget": s.spare_budget,
                },
                repo=s.repo,
            )
        elif was_starved and not is_starved:
            t = targets_by_repo[repo]
            log_event(
                state_path,
                "runner_capacity_recovered",
                {
                    "repo": t.repo,
                    "demand": t.demand,
                    "capacity": t.capacity,
                    "running": t.running,
                    "spare_budget": spare_budget,
                },
                repo=t.repo,
            )


def run_allocation_pass(
    gh: GitHubLike,
    allocation: RunnerAllocationConfig,
    *,
    managed_root_fallback: str = "",
    fleet_dir_override: str | None = None,
    state_path: Path | None = None,
    dry_run: bool = False,
    source: AllocationSource,
    full_pass_interval_seconds: int,
) -> AllocationPassResult:
    """Rebalance this host's running runner listeners across repos by demand.

    Args:
        gh: Any authenticated GitHub client — repos are addressed by explicit
            slug, so the client's own repo_root is irrelevant here.
        allocation: The host-wide allocation config (global fleet layer).
        managed_root_fallback: ``runner_scaling.managed_root``, used when the
            allocation section leaves the path unset.
        fleet_dir_override: Optional fleet-directory override for state.
        state_path: Optional ``state.json`` path; when given, the plan is
            recorded to the event log.
        dry_run: Plan and report without starting, parking, or persisting.
        source: Which path is running this pass. Persisted with the state file so
            the doctor probe can tell an unattended pass from an operator's
            manual ``charlie runners allocate`` — both write the same file, and
            only the former is evidence the daemon is rebalancing (issue #590).
        full_pass_interval_seconds: The cadence this pass is being driven at
            (``supervisor.full_pass_interval_seconds`` from the *caller's*
            resolved config). Persisted with the state file so the doctor probe
            measures staleness against the interval the daemon actually used
            rather than re-resolving config through its own load call — a
            per-repo layer that sets the interval would otherwise make the probe
            measure against a cadence the daemon is not running at (issue #606).

    Returns:
        AllocationPassResult — never raises.
    """
    if not allocation.enabled:
        return AllocationPassResult(
            ok=True,
            skipped=True,
            notes=("runner_allocation is disabled in config",),
        )

    inputs, error = resolve_inputs(allocation, managed_root_fallback)
    if inputs is None:
        # A misconfigured root leaves no positive evidence the pass ran, so the
        # doctor probe used to attribute it to "the daemon never reached
        # allocation" (#590) — a different problem with a different fix. Record
        # the actual reason instead (issue #606). A dry-run must not write state
        # (the preview would otherwise bump ``updated_at`` and look like a pass).
        if not dry_run:
            save_allocation_skip(
                fleet_dir(override=fleet_dir_override),
                source=source,
                full_pass_interval_seconds=full_pass_interval_seconds,
                skip_reason=error or "runner_allocation inputs could not be resolved",
            )
        return AllocationPassResult(ok=False, error=error)

    instances, discovery_notes = discover_runner_instances(inputs.managed_root)
    notes = list(inputs.notes) + list(discovery_notes)

    if not instances:
        if not dry_run:
            save_allocation_skip(
                fleet_dir(override=fleet_dir_override),
                source=source,
                full_pass_interval_seconds=full_pass_interval_seconds,
                skip_reason=f"no configured runners found under {inputs.managed_root}",
            )
        return AllocationPassResult(
            ok=True,
            skipped=True,
            notes=tuple(notes + [f"no configured runners found under {inputs.managed_root}"]),
        )

    repos = sorted({inst.repo for inst in instances})

    busy_by_repo: dict[str, set[str]] = {}
    demands: dict[str, RepoDemand] = {}
    for repo in repos:
        busy_names, busy_error = fetch_busy_runner_names(gh, repo)
        busy_by_repo[repo] = busy_names
        if busy_error:
            # Treat an unreadable runner list as an unmeasurable repo: the
            # allocator will pin it rather than risk parking a busy runner it
            # cannot see.
            demands[repo] = RepoDemand(repo=repo, ok=False, error=f"runners list: {busy_error}")
            continue
        demands[repo] = measure_repo_demand(gh, repo, inputs.max_runs_scanned)

    observed = annotate_busy(instances, busy_by_repo)
    state_dir = fleet_dir(override=fleet_dir_override)
    previous_streaks = load_idle_streaks(state_dir)
    tie_break_offset = load_tie_break_offset(state_dir)

    plan = plan_allocation(
        observed,
        demands,
        budget=inputs.budget,
        budget_reason=inputs.budget_reason,
        min_per_repo=inputs.min_per_repo,
        idle_streaks=previous_streaks,
        demand_idle_samples=inputs.demand_idle_samples,
        tie_break_offset=tie_break_offset,
    )

    results = apply_allocation(plan, dry_run=dry_run)

    if not dry_run:
        save_idle_streaks(
            state_dir,
            next_idle_streaks(observed, demands, previous_streaks),
            source=source,
            full_pass_interval_seconds=full_pass_interval_seconds,
            # Advance the rotation so a different repo wins the next name tie.
            tie_break_offset=tie_break_offset + 1,
        )

        # Promote the "demand exceeds registered capacity, budget has slack"
        # condition from a stdout-only note to a queryable, edge-triggered
        # event (issue #799). Gated on ``not dry_run`` for the same reason
        # the hysteresis persist above is: a dry-run previews the plan and
        # must not have side effects (including event writes).
        if state_path is not None:
            _emit_capacity_events(state_path, plan)

    if state_path is not None:
        summary = plan_summary(plan)
        summary["dry_run"] = dry_run
        summary["source"] = source
        summary["applied"] = [
            {"runner": r.change.runner_name, "ok": r.ok, "message": r.message} for r in results
        ]
        log_event(state_path, "runner_allocation", summary)

    return AllocationPassResult(
        ok=all(r.ok for r in results),
        plan=plan,
        results=tuple(results),
        notes=tuple(notes + list(plan.notes)),
    )
