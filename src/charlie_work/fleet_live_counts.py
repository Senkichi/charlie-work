"""Worker-lane fleet live counting in ONE registry walk (issue #2764).

The concurrency governor needs two numbers under the fleet launch lock -- the
fleet-wide live total and the per-``adapter_kind`` breakdown behind
``fleet.max_concurrent_sessions_by_harness`` (#2757). Walking the fleet
registry twice would double a lock other repos wait on (observed
``fleet_lock_held`` deferrals of 15-111s when several repos launch together),
so this module's one walker returns both from a single
``_iter_registered_repo_contexts`` pass: each registered repo contributes
``live_session_count.count_live_sessions_with_adapter_kinds`` (one
``iter_workers`` pass feeding the total -- with its state.json ghost
corroboration -- and the per-kind breakdown).

The two single-number walkers it backs live in ``fleet_registry`` as thin
wrappers over this function, so their public signatures and patch targets are
unchanged.
"""

from __future__ import annotations

import logging

from . import layout

logger = logging.getLogger(__name__)


def count_fleet_live_sessions_with_adapter_kinds(
    fleet_dir_override: str | None,
) -> tuple[int, dict[str, int], list[str]]:
    """One registry walk returning the fleet live total AND its per-kind breakdown.

    ``_iter_registered_repo_contexts`` is resolved from ``fleet_registry`` at
    call time (module-attribute lookup), so a patch on the registry's walk --
    such as the governor's one-walk-per-call test -- still intercepts here.

    Same skip semantics as the single-number walkers that wrap it. Returns
    ``(total_live_count, counts_by_adapter_kind, skipped_repos)``.
    """
    from .fleet_registry import _iter_registered_repo_contexts
    from .live_session_count import WORKER_LANE, count_live_sessions_with_adapter_kinds

    total_live_count = 0
    totals_by_kind: dict[str, int] = {}
    skipped_repos: list[str] = []

    for name_with_owner, repo_root, state_dir, repo_config in _iter_registered_repo_contexts(
        fleet_dir_override, skipped_repos
    ):
        sessions_dir = layout.resolve_state_child(
            repo_config.devin.sessions_dir,
            repo_root=repo_root,
            default=layout.sessions_dir_default(state_dir),
        )
        if not sessions_dir.exists():
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: "
                f"sessions_dir {sessions_dir} does not exist"
            )
            skipped_repos.append(name_with_owner)
            continue

        repo_total, repo_kinds = count_live_sessions_with_adapter_kinds(
            sessions_dir, layout.state_file_path(state_dir), WORKER_LANE
        )
        total_live_count += repo_total
        for kind, count in repo_kinds.items():
            totals_by_kind[kind] = totals_by_kind.get(kind, 0) + count

    return total_live_count, totals_by_kind, skipped_repos
