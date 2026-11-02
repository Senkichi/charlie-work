"""Classify a dead worker's sidecar before a launch can replace it (issue #2274).

A worker launch writes ``issue-<n>`` sidecar records in several places: the
``live_worker_redispatch_averted`` / worktree-failure record, the post-worktree
failure record, and the launched session's own record. Any of them replaces the
previous session's sidecar. When that previous session died of a provider
rate limit or quota and no lane had classified it yet, its evidence went with
it. Its role was never restricted in the ledger (#2086), and the overwriting
record's own ``failure_kind`` stopped any later classification.

:func:`classify_dead_sidecars` runs once per request in
``adapters.dispatch_sessions``, before any adapter touches the sidecar. That one
point covers every adapter and every overwrite site. It classifies only a
sidecar that

* belongs to this issue,
* has no ``failure_kind`` yet (so a re-run is a cheap no-op), and
* is behind a process that is not alive by the OS-level probe
  (``WorkerView.is_alive``). A live worker's log is never classified.

Classification goes through the adapter profile's ``record_failure`` with no
fallback kind, the same helper the reap lanes use. That helper stamps the
sidecar, records the ledger restriction, and keeps the #656 completion guard
(``terminal_record_proves_completion``). The per-repo cooldown in
``state.json`` is not armed here, because adapters do not own state. Arming it
is the dead-worker sweep's job (``dead_worker_sweep.pre_classification``).

Best effort, never raises: a failure here must not block a launch.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from . import worker_fate
from .config import OrchestratorConfig
from .worker import iter_workers

logger = logging.getLogger(__name__)


def classify_dead_sidecars(
    sessions_dir: Path,
    issue_number: int,
    config: OrchestratorConfig | None,
    *,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Classify every unclassified, dead sidecar for ``issue_number``.

    Returns the failure kinds that were newly resolved, which is empty when
    there was nothing to do or the logs show no signature.
    """
    resolved: list[str] = []
    try:
        views = [
            view
            for view in iter_workers(sessions_dir)
            if view.issue_number == issue_number and view.failure_kind is None
        ]
        for view in views:
            if view.is_alive():
                continue
            profile = worker_fate.profile_for(view.adapter_kind)
            if profile is None or profile.record_failure is None:
                continue
            kind, _until = profile.record_failure(
                sessions_dir, issue_number, config=config, now=now
            )
            if kind is not None:
                resolved.append(kind)
    except Exception:  # best effort: a launch must never fail here
        logger.warning("dead-sidecar classification failed issue=%d", issue_number, exc_info=True)
    return tuple(resolved)
