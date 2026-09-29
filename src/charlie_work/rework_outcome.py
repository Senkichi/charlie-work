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
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import worker_fate
from .config import WORKER_OUTCOME_FILENAME
from .instrumentation import log_event
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


def _outcome_evidence(
    outcome: dict[str, Any] | None,
    *,
    source: worker_fate.EvidenceSource,
    written_at: datetime | None = None,
) -> worker_fate.OutcomeEvidence | None:
    """Build an ``OutcomeEvidence`` from a raw ``.worker-outcome.json`` dict.

    ``None`` in, ``None`` out (no claim); a present-but-empty dict still
    builds a real (all-``None``-fields) evidence object, matching what
    "an empty terminal outcome dict" means in the design doc (§3 step 0:
    "An empty terminal dict ``{}`` is 'no claim': it is neither stale nor
    decisive" -- distinguished from *absent*, i.e. ``None``, which is
    skipped entirely by the freshness step).
    """
    if not isinstance(outcome, dict):
        return None
    return worker_fate.OutcomeEvidence(
        source=source,
        written_at=written_at,
        outcome=outcome.get("outcome"),
        push_succeeded=outcome.get("push_succeeded"),
        pr_created=outcome.get("pr_created"),
        head_sha=outcome.get("head_sha"),
        raw=outcome,
    )


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

    Obtains this pick from the module (design doc §8, step B1, A13):
    ``worker_fate.resolve_fate``'s freshness step (rules 1/7) already
    implements "terminal, if fresh, else worktree, if fresh, else nothing".
    Neither a ``dispatched_at`` nor a live head is known at this call site
    (this function takes neither), so every candidate is fresh by
    definition (rule 1's legacy mode: an unset ``dispatched_at`` accepts
    unconditionally, and an unknown head never proves a mismatch) -- which
    degrades exactly to "terminal if present (even ``{}``), else worktree",
    FLIP 7's ``isinstance(outcome, dict): return outcome`` pass-through
    with no ``or``-style fallthrough on an empty-but-present terminal dict.
    Contrast ``workflow.py``'s no-PR lane (FLIP 7's other, disagreeing
    site), which still uses the raw ``terminal_outcome or worktree_outcome``
    expression.
    """
    terminal_evidence: worker_fate.TerminalEvidence | None = None
    record = find_worker_terminal_status(sessions_dir, issue_number)
    if isinstance(record, dict):
        terminal_evidence = worker_fate.TerminalEvidence(
            ended_at=worker_fate.parse_iso_timestamp(record.get("ended_at")) or datetime.now(UTC),
            exit_code=record.get("exit_code"),
            outcome=_outcome_evidence(
                record.get("worker_outcome"), source=worker_fate.EvidenceSource.TERMINAL
            ),
        )

    worktree_outcome_evidence: worker_fate.OutcomeEvidence | None = None
    if branch:
        worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
        worktree_outcome_evidence = _outcome_evidence(
            read_worker_outcome(worktree_path), source=worker_fate.EvidenceSource.WORKTREE
        )

    fate = worker_fate.resolve_fate(
        worker_fate.FateEvidence(
            issue_number=issue_number,
            adapter="unknown",
            dispatched_at=None,
            pid_alive=False,
            health=None,
            terminal=terminal_evidence,
            worktree_outcome=worktree_outcome_evidence,
            branch=worker_fate.BranchEvidence(
                has_remote=True,
                remote_head_sha=None,
                remote_ahead=None,
                unpushed=None,
                open_pr_number=None,
                pr_known=False,
            ),
            failure=None,
        ),
        now=datetime.now(UTC),
    )
    return dict(fate.basis.outcome.raw) if fate.basis.outcome is not None else None


def fresh_completed_worker_outcome(
    worktree_path: Path | None,
    *,
    live_head_sha: str | None,
    dispatched_at: datetime | None,
) -> dict[str, Any] | None:
    """Return a worktree ``.worker-outcome.json`` that proves a dead worker
    completed its handoff despite leaving no terminal-status record
    (issue #1911).

    The orphan sweep uses this before crediting a worker death: adapters that
    never write a terminal record (``devin-shell`` — only
    ``claude_code.launch_claude_worker`` runs ``start_terminal_status_watcher``)
    and sessions whose watcher died with the orchestrator both surface as
    ``terminal_exit_code is None``, indistinguishable from a crash by PID
    alone. The on-disk outcome is the durable signal that separates them, but
    only when it is *fresh* — written after this dispatch began
    (``dispatched_at``, the issue entry's dispatch timestamp), so a leftover
    from a previous session does not count — reports ``push_succeeded``, is
    not a ``blocked`` declaration, and pins ``head_sha`` to the live remote
    head.

    Every check fails safe: ``None`` sends the caller back to the existing
    worker-death path. Never raises.

    Obtains this pick from the module (design doc §8, step B1, A12): rule
    1's freshness gate (``written_at > dispatched_at``) and rule 7's
    head-match replace the two literal checks this function used to run by
    hand. No ``pr_by_issue`` is threaded through here (the design note
    limits this wiring to the caller's ``live_head_sha``/``dispatched_at``),
    so ``resolve_fate`` can never prove ``Completed``/``PushedWithoutPr`` at
    this call site (both require ``branch.pr_known``) and always falls
    through to ``Crashed`` instead -- the module's own comment at that row
    already says "pr_created claim ignored" (FLIP 8: rule 8 never reads
    ``pr_created`` to decide, matching this function's own legacy contract,
    which never checked it either -- ``pr_created: true`` is still accepted
    below, unchanged). What still gates the return is ``fate.basis.outcome``
    surviving the freshness step with a confirmed push and a head match --
    the exact two checks this function ran directly before.
    """
    if worktree_path is None or not live_head_sha or dispatched_at is None:
        return None
    try:
        outcome_mtime = datetime.fromtimestamp(
            (worktree_path / WORKER_OUTCOME_FILENAME).stat().st_mtime, tz=UTC
        )
    except OSError:
        return None
    outcome = read_worker_outcome(worktree_path)
    outcome_evidence = _outcome_evidence(
        outcome, source=worker_fate.EvidenceSource.WORKTREE, written_at=outcome_mtime
    )
    if outcome_evidence is None:
        return None

    fate = worker_fate.resolve_fate(
        worker_fate.FateEvidence(
            issue_number=0,  # not carried by this function's signature; unused by resolve_fate
            adapter="unknown",
            dispatched_at=dispatched_at,
            pid_alive=False,
            health=None,
            terminal=None,
            worktree_outcome=outcome_evidence,
            branch=worker_fate.BranchEvidence(
                has_remote=True,
                remote_head_sha=live_head_sha,
                remote_ahead=None,
                unpushed=None,
                open_pr_number=None,
                pr_known=False,
            ),
            failure=None,
        ),
        now=datetime.now(UTC),
    )

    resolved = fate.basis.outcome
    if (
        not isinstance(fate, worker_fate.Blocked)
        and resolved is not None
        and resolved.push_succeeded is True
        and resolved.head_sha == live_head_sha
    ):
        return dict(resolved.raw)
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


def apply_collected_rework_outcomes(
    gh: GitHubLike,
    *,
    outcome_apply_routes: Iterable[tuple[int, int]],
    repo_root: Any,
    worktrees_dir: Path | None,
    sessions_dir: Path,
    state_file: Path,
    write_gate: WriteGate,
) -> None:
    """Apply each completed outcome the orphan sweep collected in-lock.

    ``_detect_and_handle_orphaned_workers`` queues ``(issue_number,
    pr_number)`` routes under its state lock when a dead worker left a
    fresh, on-target outcome; this runs the post-lock half. Each route goes
    through :func:`apply_rework_worker_outcome`, which re-reads the outcome
    file, re-verifies the live remote head against the reported
    ``head_sha``, and dedups per applied head -- so an already-applied entry
    is skipped cheaply and a transient failure retries on the next pass.

    The per-route guard is defense-in-depth: ``apply_rework_worker_outcome``
    is contractually never-raises, but an escaped exception here would
    propagate out of the sweep and starve every later route *and* the
    ``review_routes`` drain that follows it in the caller. A breach is
    recorded (``rework_outcome_apply_failed``, warning) and the loop
    continues; ``log_event`` is used rather than the state-ring emitter
    because it never raises and never contends for the state lock.
    """
    if not isinstance(repo_root, Path) or worktrees_dir is None:
        return
    for issue_number, pr_number in outcome_apply_routes:
        try:
            apply_rework_worker_outcome(
                gh,
                repo_root=repo_root,
                worktrees_dir=worktrees_dir,
                sessions_dir=sessions_dir,
                state_file=state_file,
                write_gate=write_gate,
                issue_number=issue_number,
                pr_number=pr_number,
            )
        except Exception as exc:
            logger.exception(
                "apply_rework_worker_outcome escaped for issue %s / pr %s",
                issue_number,
                pr_number,
            )
            # Breach telemetry must fire inside an except handler where the
            # gate may be in an unknown state; log_event writes events.db
            # (SQLite), not gated state.json, and never raises.
            # write-gate-exempt(issue=1911): deliberate raw breach event; see above
            log_event(
                state_file,
                "rework_outcome_apply_failed",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                level="warning",
            )


__all__ = [
    "APPLIED_HEADS_KEY",
    "apply_collected_rework_outcomes",
    "apply_rework_worker_outcome",
    "fresh_completed_worker_outcome",
]
