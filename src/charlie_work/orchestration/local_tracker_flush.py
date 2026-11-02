"""Flush the local issue tracker's accumulated writes at the end of one pass.

Track 2 Phase B shape: a leaf module whose top-level ``def``s delegate-relocate
onto ``OrchestratorApp`` via ``workflow_delegation._install_delegates``.

Issue #2434. Every ``LocalFileGitHub`` write site (frontmatter mutations,
appended comments) records its touched file on
``local_issue_commits.queue_issue_write``; this delegate is the single
per-pass drain point: it calls ``local_issue_commits.flush_tracker_writes``
once, at the end of the loop pass -- the commit batch, not a commit per
label edge. A no-op on the remote ``gh`` backend (no ``issues_dir``
attribute) and under ``dry_run`` (the write gate's zero-writes invariant:
no commit may land from a dry-run pass).

When the flush leaves wanted writes on the floor (HEAD detached, a
merge/rebase in progress, a failed commit group), the pass records the
``local_tracker_writes_deferred`` event here -- the client itself has no
event channel, matching how ``_drain_local_blocker_patch_equiv`` (issue
#1967) drains its own batch. ``charlie doctor``'s tracker-writes check warns
when these events repeat across passes (see ``doctor_local_backend``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from charlie_work.local_issue_commits import flush_tracker_writes

logger = logging.getLogger(__name__)

# Event-consumer note: audit-only -- the pass-level deferral record the
# doctor's "local tracker writes" check consumes (doctor_local_backend.py)
# via query_events; no separate downstream consumer writes it back.
_DEFERRED_KIND = "local_tracker_writes_deferred"


def _flush_pass_tracker_writes(self) -> None:
    """Commit the tracker writes this pass accumulated (issue #2434).

    Called at the very end of every ``_loop_body``, after the per-PR scan
    whose merge finalizations may touch the tracker, and before the pass
    result aggregates. The commit arm itself is deliberate bookkeeping, not
    a WriteGate primitive: the gate's dry-run contract ("zero writes") is
    honored by the dry-run early return above, and the flush never raises
    when git disagrees. ``commit_writes`` reads the config's kill switch
    directly: under ``false`` the flush is a benign skip, not an anomaly, so
    no deferred event fires.
    """
    if self.dry_run:
        return
    issues_dir = getattr(self.gh, "issues_dir", None)
    if not isinstance(issues_dir, Path):
        return
    result = flush_tracker_writes(
        self.repo_root,
        issues_dir,
        commit_writes=getattr(self.config.local_issues, "commit_writes", True),
    )
    if result.committed:
        logger.info(
            "pass flush committed %d tracker write batch(es): %s",
            len(result.committed),
            ", ".join(result.committed),
        )
    if not result.needs_attention:
        return
    try:
        display = issues_dir.relative_to(self.repo_root).as_posix()
    except ValueError:
        display = issues_dir.as_posix()
    reasons = "; ".join(r for r in (result.defer_reason, result.reason) if r)
    self.write_gate.log_event(
        kind=_DEFERRED_KIND,
        payload={
            "issues_dir": display,
            "left_dirty": list(result.left_dirty),
            "reason": reasons or "unknown",
            "committed": list(result.committed),
            "skipped": result.skipped,
        },
    )
