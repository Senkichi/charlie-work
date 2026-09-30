"""Worker fate for the orphan sweep's no-open-PR lane.

Extracted from ``workflow._detect_and_handle_orphaned_workers`` (file-size
ratchet). Two pure-ish builders that assemble ``FateEvidence`` from data the
sweep already read and hand it to ``worker_fate.resolve_fate``:

- :func:`resolve_no_pr_orphan_fate` -- the precompute fate (rules 1/7 freshness
  arbitration between the terminal record's outcome and the worktree's file;
  rule 2's ``Blocked`` and rule 6's persisted throttle ride on it).
- :func:`resolve_pushed_orphan_fate` -- the branch-aware fate for the
  pushed-branch-candidate loop (rules 8/9: only the remote ahead count decides).

Both take the sweep's reads as parameters (the sweep keeps calling
``find_worker_terminal_status`` / ``read_worker_outcome`` through its own
namespace so suite patches on ``charlie_work.workflow.<name>`` stay live).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import worker_fate
from .config import WORKER_OUTCOME_FILENAME
from .state import parse_iso_timestamp


def outcome_evidence(
    raw: dict[str, Any] | None,
    *,
    source: worker_fate.EvidenceSource,
    written_at: datetime | None,
) -> worker_fate.OutcomeEvidence | None:
    """One ``OutcomeEvidence`` from a raw ``.worker-outcome.json`` dict (B3,
    rule 1/7). ``None`` in, ``None`` out (no claim). A present-but-empty dict
    still builds a real all-``None`` evidence: ``worker_fate._carries_no_claim``
    (N1) is what stops it out-ranking a sibling that carries a real claim.
    """
    if not isinstance(raw, dict):
        return None
    return worker_fate.OutcomeEvidence(
        source=source,
        written_at=written_at,
        outcome=raw.get("outcome"),
        push_succeeded=raw.get("push_succeeded"),
        pr_created=raw.get("pr_created"),
        head_sha=raw.get("head_sha"),
        raw=raw,
    )


def _adapter(entry: object) -> str:
    return str(entry.get("adapter") or "unknown") if isinstance(entry, dict) else "unknown"


def resolve_no_pr_orphan_fate(
    *,
    issue_number: int,
    entry: object,
    terminal: dict[str, Any] | None,
    worktree_path: Path | None,
    worktree_outcome_raw: dict[str, Any] | None,
    now: datetime,
) -> worker_fate.WorkerFate:
    """Resolve a no-PR orphan's fate from the terminal record and the worktree.

    ``dispatched_at`` gates freshness (rule 1). The terminal candidate's
    ``written_at`` is the outcome file's own mtime as the watcher recorded it
    (``worker_outcome_written_at``, B5), falling back to ``ended_at`` only for
    records written before that field existed; the worktree candidate's is the
    file's mtime. ``resolve_fate``'s freshness step (rule 7) arbitrates them.
    Branch evidence is unknown at this point (the remote read happens later),
    so the fate is reliable for the blocked check (row 1) and the throttle
    guard (``throttle_failure``), which need none of it.
    """
    terminal_evidence: worker_fate.TerminalEvidence | None = None
    if isinstance(terminal, dict):
        ended_at = parse_iso_timestamp(terminal.get("ended_at")) or now
        outcome_written_at = parse_iso_timestamp(terminal.get("worker_outcome_written_at"))
        terminal_evidence = worker_fate.TerminalEvidence(
            ended_at=ended_at,
            exit_code=terminal.get("exit_code"),
            outcome=outcome_evidence(
                terminal.get("worker_outcome"),
                source=worker_fate.EvidenceSource.TERMINAL,
                written_at=outcome_written_at or ended_at,
            ),
        )
    worktree_evidence: worker_fate.OutcomeEvidence | None = None
    if worktree_path is not None:
        try:
            worktree_mtime = datetime.fromtimestamp(
                (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
            )
        except OSError:
            worktree_mtime = None
        worktree_evidence = outcome_evidence(
            worktree_outcome_raw,
            source=worker_fate.EvidenceSource.WORKTREE,
            written_at=worktree_mtime,
        )
    persisted = worker_fate.persisted_failure(entry if isinstance(entry, dict) else {})
    return worker_fate.resolve_fate(
        worker_fate.FateEvidence(
            issue_number=issue_number,
            adapter=_adapter(entry),
            dispatched_at=parse_iso_timestamp(
                entry.get("dispatched_at") if isinstance(entry, dict) else None
            ),
            pid_alive=False,
            health=None,
            terminal=terminal_evidence,
            worktree_outcome=worktree_evidence,
            branch=worker_fate.BranchEvidence(
                remote_head_sha=None,
                remote_ahead=None,
                unpushed=None,
                open_pr_number=None,
                pr_known=True,
            ),
            # Rule 6, read-through-fate: the persisted classification is fed
            # back in so a throttle death resolves to ``Throttled``.
            failure=persisted.as_evidence(),
        ),
        now=now,
    )


def resolve_pushed_orphan_fate(
    *,
    issue_number: int,
    entry: object,
    precompute: worker_fate.WorkerFate,
    remote_head_sha: str | None,
    ahead_count: int | None,
    now: datetime,
) -> worker_fate.WorkerFate:
    """The branch-aware fate for one no-PR orphan: ``PushedWithoutPr`` exactly
    when the remote shows pushed commits (rules 8/9). ``open_pr_number`` is
    ``None`` -- a no-PR orphan by construction -- and ``unpushed`` is unset (the
    park/reclaim lane owns local-only commits), so this lane never resolves to
    ``Stranded``. Reuses the precompute fate's already-fresh outcome.
    """
    return worker_fate.resolve_fate(
        worker_fate.FateEvidence(
            issue_number=issue_number,
            adapter=_adapter(entry),
            dispatched_at=parse_iso_timestamp(
                entry.get("dispatched_at") if isinstance(entry, dict) else None
            ),
            pid_alive=False,
            health=None,
            terminal=None,
            worktree_outcome=precompute.basis.outcome,
            branch=worker_fate.BranchEvidence(
                remote_head_sha=remote_head_sha,
                remote_ahead=ahead_count,
                unpushed=None,
                open_pr_number=None,
                pr_known=True,
            ),
            failure=None,
        ),
        now=now,
    )
