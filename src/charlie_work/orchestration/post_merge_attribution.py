"""Post-merge outcome attribution pass lane for ``OrchestratorApp`` (#1717).

Track 2 Phase B delegate: wired into the per-pass maintenance lane
(``pass_deadline.run_deadline_guarded_maintenance``) so the two outcome
candidates the reviewer-effort experiment report (#1701) previously
declared not-derivable are recorded as events once they exist. The
mechanical scans live in ``post_merge_attribution_scan`` (a plain module
-- every top-level ``def`` this module declares is auto-attached onto
``OrchestratorApp`` by ``workflow_delegation._install_delegates``, so
app surface stays to lane bodies only).

Both scans are skipped in dry-run (the reports the writes feed are not
consumed there) and on a backend that cannot publish pull requests --
a no-remote repo has no PRs to attribute anything to. Each scan is
contained so one failing scan can neither kill the pass (``supervise.py``
treats an uncaught exception as daemon-fatal) nor block the other.
"""

from __future__ import annotations

import logging
from datetime import datetime

from charlie_work.local_work_park import publishes_pull_requests


def _maybe_post_merge_attribution(self, *, now: datetime | None = None) -> None:
    """Record post-merge outcome attribution events every pass (#1717).

    Lane body for ``pass_deadline.run_deadline_guarded_maintenance``.
    Deliberately NO cadence gate, like ``_maybe_reclaim_superseded_main_ci``:
    both scans' steady-state cost is bounded by their own state (a merged
    entry is probed once per pass until its outcome record is terminal,
    and the follow-up scan reads only the cached open-PR list), so a
    cadence would only delay convergence on the outcome record.

    The scan module is imported lazily inside the body: it reaches
    ``charlie_work.workflow`` for the state primitives, and
    ``charlie_work.workflow`` imports the orchestration package (this
    module) for delegate discovery at its bottom -- a module-level import
    here would cycle exactly the way ``run_deadline_guarded_maintenance``
    documents for its own ``_wf`` binding.
    """
    from charlie_work.post_merge_attribution_scan import (
        probe_merged_main_ci,
        scan_followup_pr_attribution,
    )

    if self.dry_run:
        return
    if not publishes_pull_requests(self.gh):
        return
    try:
        probe_merged_main_ci(self, state_file=self.paths.state_file, now=now)
    except Exception:  # noqa: BLE001 - containment is deliberate; see docstring
        logging.getLogger(__name__).warning(
            "post_merge_attribution: main-CI probe pass failed", exc_info=True
        )
    try:
        scan_followup_pr_attribution(self, state_file=self.paths.state_file)
    except Exception:  # noqa: BLE001 - containment is deliberate; see docstring
        logging.getLogger(__name__).warning(
            "post_merge_attribution: follow-up scan pass failed", exc_info=True
        )
