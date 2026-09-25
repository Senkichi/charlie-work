"""Issue #1853: apply a rework worker's ``.worker-outcome.json`` to its PR.

Workers carry no ``gh`` credential by design (``sanitize_env`` strips the
orchestrator's token and points ``GH_CONFIG_DIR`` at an empty directory), so
a rework session cannot run ``gh pr view``/``gh pr edit``/``gh pr comment``
itself. The rework brief instead directs the worker to push its branch,
verify the push with ``git ls-remote``, and report in the outcome file:

- ``head_sha``: the SHA ``git rev-parse HEAD`` produced after the verified
  push — the worker's claim about what the remote head now is.
- ``pr_body`` (optional): a replacement PR body.
- ``pr_comment`` (optional): a comment to post on the PR.

The authenticated orchestrator applies those requests here — but only after
verifying the *live* remote branch head (``git ls-remote``) equals the
reported ``head_sha``. A mismatch means the outcome describes a different
remote state than the one that exists now (a stale outcome, or someone else
pushed after the worker), so nothing is applied.

Applied heads are recorded per issue under ``rework_outcome_applied_heads``
in ``state.json`` so a retried routing pass — e.g. ``review()`` blocked by
the janitor gate, retried next pass — never double-edits the body or
double-posts the comment. The marker is keyed on the reported head, not the
issue, so a later rework round with a new head applies again.

This module never raises: every failure mode resolves to a recorded event
and no PR mutation, so a malformed outcome cannot wedge the dispatch pass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .process_utils import find_worker_terminal_status
from .rework_prompts import _write_text_atomic
from .state import load_state, state_lock
from .worktree import (
    read_worker_outcome,
    remote_branch_head_sha,
    worktree_path_for_branch,
)
from .write_gate import WriteGate

if TYPE_CHECKING:
    from .github import GitHubLike

logger = logging.getLogger(__name__)

# State key holding {issue_number: applied_head_sha} — the dedup record that
# makes outcome application exactly-once per reported head.
APPLIED_HEADS_KEY = "rework_outcome_applied_heads"


def _emit_event(
    state_file: Path,
    write_gate: WriteGate,
    emit: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Run ``emit(state)`` under the state lock and persist the result.

    ``emit`` is a ``state -> state`` callable (a ``write_gate.append_event``
    call with a literal kind — the event-kind registry scanner resolves
    kinds statically, so a parameterized ``kind`` argument here is not
    allowed). ``write_gate.append_event`` is a no-op under dry-run (returns
    ``state`` unchanged) and ``save_state`` likewise, so callers need no
    dry-run branch.
    """
    with state_lock(state_file):
        state = load_state(state_file)
        write_gate.save_state(emit(state))


def _load_issue_branch(state_file: Path, issue_number: int) -> str | None:
    """Return the issue's recorded ``branch_name`` (worktree/outcome locator)."""
    with state_lock(state_file):
        state = load_state(state_file)
    entry = state.get("issues", {}).get(str(issue_number), {})
    if not isinstance(entry, dict):
        return None
    branch = entry.get("branch_name")
    return branch if isinstance(branch, str) and branch else None


def _read_rework_outcome(
    sessions_dir: Path,
    repo_root: Path,
    worktrees_dir: Path,
    issue_number: int,
    branch: str | None,
) -> dict[str, Any] | None:
    """Read the worker's outcome: durable terminal status first, worktree fallback.

    The terminal-status watcher copies the outcome out of the worktree at
    process exit, so the durable copy survives worktree cleanup; the worktree
    file itself is the fallback for a worker whose watcher has not run yet.
    """
    record = find_worker_terminal_status(sessions_dir, issue_number)
    if isinstance(record, dict):
        outcome = record.get("worker_outcome")
        if isinstance(outcome, dict):
            return outcome
    if branch:
        worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
        return read_worker_outcome(worktree_path)
    return None


def apply_rework_worker_outcome(
    gh: GitHubLike,
    *,
    repo_root: Path,
    worktrees_dir: Path,
    sessions_dir: Path,
    state_file: Path,
    write_gate: WriteGate,
    issue_number: int,
    pr_number: int,
) -> None:
    """Apply a rework worker's outcome-file PR edits after head verification.

    Called from ``_route_rework_candidate_to_review`` — the seam where a PR
    head that advanced past the last verdict is routed back to review — so
    the edits land before the review packet is regenerated, and still land
    when the packet itself is blocked (a rerouted-but-blocked review must
    not drop the worker's body/comment updates).

    Never raises; never mutates the PR unless the live remote head equals
    the worker-reported ``head_sha``.
    """
    base_payload = {"issue_number": issue_number, "pr_number": pr_number}

    def _skip(reason: str, **extra: Any) -> None:
        _emit_event(
            state_file,
            write_gate,
            lambda s: write_gate.append_event(
                s,
                "rework_outcome_skipped",
                {**base_payload, "reason": reason, **extra},
            ),
        )

    branch = _load_issue_branch(state_file, issue_number)
    outcome = _read_rework_outcome(sessions_dir, repo_root, worktrees_dir, issue_number, branch)
    if not isinstance(outcome, dict):
        _skip("no_outcome")
        return
    # ``{"outcome": "blocked", ...}`` is the scope-escalation channel the
    # rework brief documents — a different message shape, never a PR edit.
    if outcome.get("outcome") == "blocked":
        _skip("blocked_outcome")
        return

    head_sha = outcome.get("head_sha")
    if not isinstance(head_sha, str) or not head_sha:
        _skip("missing_head_sha")
        return

    pr_body = outcome.get("pr_body")
    pr_comment = outcome.get("pr_comment")
    if (pr_body is not None and not isinstance(pr_body, str)) or (
        pr_comment is not None and not isinstance(pr_comment, str)
    ):
        _skip("invalid_outcome")
        return

    if not branch:
        _skip("missing_branch")
        return

    # Exactly-once: a head already applied for this issue is not re-applied.
    with state_lock(state_file):
        state = load_state(state_file)
        applied_heads = state.get(APPLIED_HEADS_KEY, {})
    if isinstance(applied_heads, dict) and applied_heads.get(str(issue_number)) == head_sha:
        return

    remote_head = remote_branch_head_sha(repo_root, branch)
    if remote_head is None:
        _skip("remote_head_unavailable", reported_head_sha=head_sha)
        return
    if remote_head != head_sha:
        _skip(
            "head_mismatch",
            reported_head_sha=head_sha,
            remote_head_sha=remote_head,
        )
        return

    # Apply outside the state lock — gh calls are network-bound and must
    # never hold it. Both mutations run through the gh capability's own
    # outbound-body guard; a refusal raises GitHubError like any gh failure.
    pr_dir = state_file.parent / "prs" / f"pr-{pr_number}"
    body_updated = False
    comment_posted = False
    try:
        if pr_body and pr_body.strip():
            body_path = pr_dir / "rework-outcome-pr-body.md"
            pr_dir.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(body_path, pr_body)
            gh.pr_edit(pr_number, body_path)
            body_updated = True
        if pr_comment and pr_comment.strip():
            comment_path = pr_dir / "rework-outcome-comment.md"
            pr_dir.mkdir(parents=True, exist_ok=True)
            # Deferred import: ORCHESTRATOR_COMMENT_MARKER lives in
            # workflow.py, which imports the orchestration delegate that
            # calls this module — a top-level import would cycle.
            from .workflow import ORCHESTRATOR_COMMENT_MARKER

            _write_text_atomic(comment_path, f"{ORCHESTRATOR_COMMENT_MARKER}\n{pr_comment}")
            gh.pr_comment(pr_number, comment_path)
            comment_posted = True
    except Exception as exc:
        # Mutations are idempotent (same body text; the marker is only
        # written after both succeed), so a partial failure retries cleanly
        # on the next routing pass.
        error = f"{type(exc).__name__}: {exc}"
        _emit_event(
            state_file,
            write_gate,
            lambda s: write_gate.append_event(
                s,
                "rework_outcome_apply_failed",
                {**base_payload, "error": error},
                level="warning",
            ),
        )
        return

    with state_lock(state_file):
        state = load_state(state_file)
        applied_heads = state.setdefault(APPLIED_HEADS_KEY, {})
        if isinstance(applied_heads, dict):
            applied_heads[str(issue_number)] = head_sha
        state = write_gate.append_event(
            state,
            "rework_outcome_applied",
            {
                **base_payload,
                "head_sha": head_sha,
                "body_updated": body_updated,
                "comment_posted": comment_posted,
            },
        )
        write_gate.save_state(state)


__all__ = [
    "APPLIED_HEADS_KEY",
    "apply_rework_worker_outcome",
]
