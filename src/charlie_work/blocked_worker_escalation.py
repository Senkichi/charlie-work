"""Rule 2 (Worker fate) for a dead worker that still has an open PR: read its
fresh ``blocked`` declaration and escalate it, queueing the label edge.

Split out of the retired ``orphaned_worker_sweep`` (now
``dead_worker_sweep.decide_with_pr``), which reads this through
``dead_worker_blocked_outcome``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime
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


def is_exempt_blocked(
    fate: worker_fate.WorkerFate, *, sessions_dir: Path, issue_number: int
) -> bool:
    """Whether ``fate`` is a ``Blocked`` the #2010 gate refuses to escalate.

    The fate-level form of :func:`escalatable_blocked_outcome`, for consumers
    that must route such a worker as if it had not declared ``blocked`` at all
    (the pushed-branch PR-open lane, the live-handoff finalize), so the
    exemption decides the routing once instead of only suppressing the
    escalation.
    """
    if not isinstance(fate, worker_fate.Blocked) or fate.basis.outcome is None:
        return False
    return (
        escalatable_blocked_outcome(
            dict(fate.basis.outcome.raw), sessions_dir=sessions_dir, issue_number=issue_number
        )
        is None
    )


def resolve_fate_exempting_blocked(
    evidence: worker_fate.FateEvidence,
    *,
    sessions_dir: Path,
    issue_number: int,
    now: datetime,
) -> worker_fate.WorkerFate:
    """``resolve_fate(evidence)``, deciding the #2010 exemption at the fate.

    Row 1 (a fresh ``blocked`` claim) shadows every branch-evidence row, so a
    permission-denial ``blocked`` that the gate refuses to escalate would still
    mask the fate the branch supports (``Stranded``/``PushedWithoutPr``) and a
    consumer that merely skips the escalation would fall through to a
    reap-and-ready path that destroys unsalvaged work. When the resolved fate is
    such an exempt ``Blocked``, the ``blocked`` claim is dropped from the
    evidence and the fate re-resolved, so the branch evidence decides exactly as
    if the worker had not declared ``blocked``. Push flags and ``head_sha`` are
    kept: only the ``outcome`` verdict is cleared.

    The single implementation for every consumer that routes on fate (the
    dispatch-time phantom-worker path and the pushed-orphan lane), so the
    exemption is evaluated once per resolution and the returned fate is
    ``Blocked`` only when it is genuinely escalatable.
    """
    fate = worker_fate.resolve_fate(evidence, now=now)
    if not is_exempt_blocked(fate, sessions_dir=sessions_dir, issue_number=issue_number):
        return fate
    worktree_outcome = evidence.worktree_outcome
    terminal = evidence.terminal
    return worker_fate.resolve_fate(
        dataclasses.replace(
            evidence,
            worktree_outcome=(
                dataclasses.replace(worktree_outcome, outcome=None)
                if worktree_outcome is not None
                else None
            ),
            terminal=(
                dataclasses.replace(
                    terminal,
                    outcome=(
                        dataclasses.replace(terminal.outcome, outcome=None)
                        if terminal.outcome is not None
                        else None
                    ),
                )
                if terminal is not None
                else None
            ),
        ),
        now=now,
    )


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
