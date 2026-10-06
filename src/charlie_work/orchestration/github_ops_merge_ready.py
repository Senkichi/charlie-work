"""Merge-ready dry-run command delegate moved out of ``OrchestratorApp``.

The body lives in ``charlie_work.merge_path.preview``: the same pure stages
``merge_ready`` runs, fed by preview gathers and never applied (issue #614).
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``; it stays a method because tests call it
directly.
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from charlie_work.merge_path.preview import preview_merge_ready


def _merge_ready_dry_run(
    self,
    pr_number: int,
    *,
    merge: bool | None = None,
    merge_train_head: int | None = None,
) -> _wf.CommandResult:
    """Dry-run readiness evaluation for ``merge_ready`` (issue #614).

    Reports what ``merge_ready`` would do without a state write, label
    transition, branch update, merge or mergequeue label add. The gate in
    :meth:`merge_ready` sits above its ``state_lock`` so a dry-run never
    reaches the failed-attempt accounting that would advance the counters.
    """
    return preview_merge_ready(self, pr_number, merge=merge, merge_train_head=merge_train_head)
