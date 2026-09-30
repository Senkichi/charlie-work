"""Rule 2 (Worker fate) for a dead worker that still has an open PR: read its
fresh ``blocked`` declaration and escalate it, queueing the label edge.

Split out of ``orphaned_worker_sweep`` (two call sites there, one per review
decision branch) so the sweep stays under its file-size ratchet mark.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import worker_fate
from .claude_code import worker_permission_denied
from .rework_outcome import blocked_worker_outcome
from .worktree import worktree_path_for_branch


def escalatable_blocked_outcome(
    outcome: dict[str, Any] | None, *, sessions_dir: Path, issue_number: int
) -> dict[str, Any] | None:
    """The single gate every blocked-escalation path goes through: ``outcome``
    when it may reach the operator queue, else None.

    Issue #2010: a ``blocked`` declaration (or log tail) that is just the
    headless permission-denial signature is a worker-config defect, not a
    structurally impossible task, so it is never escalated -- it falls through
    to the ordinary redispatch/reset path. Shared by the no-PR lane
    (``workflow``) and the with-PR lane (:func:`dead_worker_blocked_outcome`)
    so neither can skip the exemption.
    """
    if outcome is None:
        return None
    detail = str(outcome.get("detail") or "")
    if worker_permission_denied(sessions_dir, issue_number, detail):
        return None
    return outcome


def dead_worker_blocked_outcome(
    *,
    pr_data: dict[str, Any],
    entry: dict[str, Any],
    issue_number: int,
    sessions_dir: Path,
    repo_root: Any,
    worktrees_dir: Path | None,
    on_fate: Callable[[worker_fate.WorkerFate], None] | None,
) -> dict[str, Any] | None:
    """Rule 2: the dead worker's fresh, escalatable ``blocked`` declaration, else None.

    A #2010 permission-denial declaration is not escalatable (see
    :func:`escalatable_blocked_outcome`).

    N4: the worktree may already be reaped; the terminal record still carries
    the watcher's copy of the outcome, so ``sessions_dir`` is always passed.
    """
    import charlie_work.workflow as _wf

    branch = pr_data.get("headRefName") or entry.get("branch_name")
    worktree_path = (
        worktree_path_for_branch(repo_root, branch, worktrees_dir)
        if branch and isinstance(repo_root, Path) and worktrees_dir is not None
        else None
    )
    return escalatable_blocked_outcome(
        blocked_worker_outcome(
            worktree_path,
            issue_number=issue_number,
            dispatched_at=_wf._parse_iso_timestamp(entry.get("dispatched_at")),
            pr_number=pr_data.get("number") if isinstance(pr_data.get("number"), int) else None,
            on_fate=on_fate,
            sessions_dir=sessions_dir,
        ),
        sessions_dir=sessions_dir,
        issue_number=issue_number,
    )


def escalate_declared_blocked(
    *,
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    entry: dict[str, Any],
    issue_number: int,
    pr_number: int,
    blocked_outcome: dict[str, Any],
    reap_escalations: list[int],
    event_extra: dict[str, Any],
    terminal_pid: Any,
    terminal_exit_code: Any,
    terminal_duration_seconds: Any,
) -> None:
    """Escalate a dead with-PR worker that declared itself blocked (rule 2).

    Queues the issue on ``reap_escalations`` so the pass applies the
    ``escalated`` label edge post-lock, keeping labels and state.json in step.

    ``_escalate_issue`` rebuilds ``state["issues"][key]`` as a brand-new dict
    rather than mutating the one passed in (see escalation.py / CLAUDE.md),
    and the caller's loop tail writes back the *same* ``entry`` object it
    fetched before calling us. Rebinding a local ``entry`` would leave the
    caller's stale, pre-escalation reference to clobber the escalation, so the
    caller-owned ``entry`` is mutated in place: its identity (and thus the
    loop tail's write-back) stays correct.
    """
    import charlie_work.workflow as _wf

    state = _wf._escalate_issue(
        state,
        issue_number,
        reason="worker_declared_blocked",
        reason_class="mechanical",
        pr_number=pr_number,
        issue_extra={"dispatched_at": None},
    )
    entry.clear()
    entry.update(state["issues"][str(issue_number)])
    state["issues"][str(issue_number)] = entry
    reap_escalations.append(issue_number)
    sweep_events.append(
        (
            "worker_declared_blocked",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "previous_status": "dispatched",
                "reason": "worker_declared_blocked",
                **event_extra,
                "reason_kind": str(blocked_outcome.get("reason_kind") or "unknown"),
                "detail": str(blocked_outcome.get("detail") or ""),
                "pid": terminal_pid,
                "exit_code": terminal_exit_code,
                "duration_seconds": terminal_duration_seconds,
            },
        )
    )
