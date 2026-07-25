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
   not advance the hysteresis counters it is previewing.

The caller is expected to hold the fleet lock (the fleet pass does). Actuation
re-checks liveness per slot, so a concurrent pass degrades to redundant no-ops
rather than double-starting a listener.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import RunnerAllocationConfig
from .fleet_paths import fleet_dir
from .github import GitHub
from .instrumentation import log_event
from .runner_allocation import (
    AllocationPlan,
    RepoDemand,
    SlotChangeResult,
    annotate_busy,
    derive_budget,
    next_idle_streaks,
    plan_allocation,
    plan_summary,
)
from .runner_slots import (
    apply_allocation,
    discover_runner_instances,
    fetch_busy_runner_names,
    load_idle_streaks,
    measure_repo_demand,
    AllocationSource,
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


def run_allocation_pass(
    gh: GitHub,
    allocation: RunnerAllocationConfig,
    *,
    managed_root_fallback: str = "",
    fleet_dir_override: str | None = None,
    state_path: Path | None = None,
    dry_run: bool = False,
    source: AllocationSource,
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
        return AllocationPassResult(ok=False, error=error)

    instances, discovery_notes = discover_runner_instances(inputs.managed_root)
    notes = list(inputs.notes) + list(discovery_notes)

    if not instances:
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

    plan = plan_allocation(
        observed,
        demands,
        budget=inputs.budget,
        budget_reason=inputs.budget_reason,
        min_per_repo=inputs.min_per_repo,
        idle_streaks=previous_streaks,
        demand_idle_samples=inputs.demand_idle_samples,
    )

    results = apply_allocation(plan, dry_run=dry_run)

    if not dry_run:
        save_idle_streaks(
            state_dir,
            next_idle_streaks(observed, demands, previous_streaks),
            source=source,
        )

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
