"""Hold slots for ready ``priority:critical`` issues before the rework lane spends them (issue #2917).

``loop()`` shares one wave budget between the two worker lanes: rework runs
first, then fresh dispatch gets ``wave - rework launched``. Fresh dispatch ranks
its own candidates by ``priority:<level>`` (:mod:`charlie_work.issue_priority`),
but nothing compared a fresh issue with rework, so a ``critical`` issue waited
behind the whole rework queue.

:func:`critical_fresh_reserve` is that comparison, kept at the one point where
the lanes share slots: ``loop()`` hands the rework lane
``wave - critical_fresh_reserve(...)``. Rework already running holds live
slots the governor has already subtracted, so it is never touched, and
rework's own order is unchanged (ordering it by priority too would not have
helped: the starved issue had no critical rework).

The reserve is the number of ready critical fresh issues, bounded by:

* the wave itself ("up to the limit");
* the fresh lane's own headroom (``max_open_agent_prs`` and CI headroom bind
  fresh dispatch only). Without that bound a critical issue held back by
  open-PR backpressure would hold a slot from rework every pass, and rework is
  what drains open PRs: the reserve would deadlock itself.

"Ready" is the dispatch candidate filter outside the lane: the label gate
(``_is_dispatchable``: ready, not active or terminal, owned here, not
operator-claimed) and the dependency gate (``_filter_blocked_issues``), the
same two steps ``status()`` uses. The ``state.json`` in-flight exclusions
inside ``_dispatch_impl`` are not repeated; a critical issue fresh dispatch
still declines for a state reason keeps the reserve for as long as it stays
labelled ready, and every such pass emits ``dispatch_priority_reserve`` naming
it, so the hold is visible rather than silent.

An empty ``labels.priority_prefix`` turns the reserve off (zero reads), and a
GitHub failure or a busy state lock while counting reserves nothing: the pass
behaves as it did before this module existed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .github import GitHubError, label_names
from .issue_priority import is_critical
from .state import StateLockBusy, load_state_locked, operator_claimed_issues

if TYPE_CHECKING:
    from .workflow import OrchestratorApp

logger = logging.getLogger(__name__)


def _ready_critical_issue_numbers(app: OrchestratorApp, prefix: str) -> list[int]:
    """Numbers of the open critical issues the fresh lane's label and dependency gates admit."""
    claimed = operator_claimed_issues(load_state_locked(app.paths.state_file))
    critical = [
        issue
        for issue in app.gh.issue_list(labels=[app.config.labels.ready])
        if is_critical(label_names(issue), prefix) and app._is_dispatchable(issue, claimed)
    ]
    if not critical:
        return []
    unblocked, _blocked, _open_blockers = app._filter_blocked_issues(critical)
    return sorted(int(issue["number"]) for issue in unblocked)


def critical_fresh_reserve(app: OrchestratorApp, wave_limit: int) -> int:
    """Slots of ``wave_limit`` to leave unclaimed for ready critical fresh issues.

    Returns ``0`` when priority is off (empty prefix), the wave is empty, no
    critical issue is ready, or GitHub could not be read. Records a
    ``dispatch_priority_reserve`` event when it reserves (not in a dry run).
    """
    prefix = app.config.labels.priority_prefix
    if wave_limit <= 0 or not prefix:
        return 0
    try:
        numbers = _ready_critical_issue_numbers(app, prefix)
        if not numbers:
            return 0
        headroom = app._apply_concurrency_governor(
            wave_limit, apply_open_pr_backpressure=True
        ).dispatch_limit
    except (GitHubError, StateLockBusy) as exc:
        # Fail open to the pre-reserve pass. StateLockBusy matters here: this runs before
        # ``dispatch_rework`` is entered, so nothing downstream would absorb it into a
        # deferral and it would fail the whole ``loop()`` pass.
        logger.warning("priority reserve skipped, could not read ready issues: %s", exc)
        return 0
    reserve = min(len(numbers), wave_limit, max(0, headroom))
    if reserve:
        app.write_gate.log_event(
            # event-consumer: audit-only -- records slots the rework lane left for ready
            # critical issues; the reservation itself is enforced by the limit loop() passes on
            "dispatch_priority_reserve",
            {
                "issue_numbers": numbers,
                "reserved": reserve,
                "wave_limit": wave_limit,
                "fresh_headroom": headroom,
            },
        )
    return reserve
