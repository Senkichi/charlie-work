"""Post-lock disposition for a rework worker that exited without changing the PR (#2034).

A dead rework worker can leave the PR head unchanged (``dead_worker_clean_exit_no_op``),
or move it without producing a reviewable delta (a merge from main; the janitor then
refuses the head-change review). Before #2034 both paths fingerprinted one
``orphaned_worker_drift`` event and left the issue ``dispatched`` on
``agent:in-progress`` with no live worker, so it dropped out of every lane.

The orphan sweep now collects a :class:`NoOpReworkRoute` for each such finding and
:func:`drain_no_op_rework_routes` gives it a consumer, outside the state lock (it does
network I/O). In priority order:

1. **Live CI red** on the head -> the CI-failure rework lane: a ``request_changes``
   verdict (``ci_gate_auto_reject``, the same provenance ``review()``'s CI gate uses)
   whose ``required_changes`` name the failing step and carry a log excerpt, because
   workers have no ``gh``.
2. **CI green and the worker rebutted the verdict** (``pr_comment`` in its outcome file)
   -> ``review()``, once per head. A packet flips the issue to ``reviewing``.
3. **Otherwise, or a second no-op on a head already given a disposition** -> escalate
   ``agent:human-needed`` with reason ``rework_no_op`` and the worker's comment.

``entry["no_op_handled_head"]`` records the head a tier-1/2 disposition was issued for;
it is what makes "second no-op on the same head" checkable. Nothing here leaves the
issue ``dispatched`` except the deliberate deferrals for CI that has not settled:
``pr_checks`` returned ``None`` (check state unknown), or the required-check summary
is neither ``failed`` nor ``ready`` -- a required check still ``pending``,
``missing``, ``infra_failed``, ``infra_blocked``, or ``unavailable``. Both defer to
the next pass, stamp ``no_op_deferred_head`` once per head for the audit trail, and
are bounded by the #654 ``dead_dispatched_reap_minutes`` backstop the sweep arms.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .checks import _is_failing_run, summarize_checks
from .ci_findings import _required_changes_from_checks
from .markdown_fence import fenced_block
from .rework_outcome import _read_rework_outcome

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .github import GitHubLike
    from .write_gate import WriteGate

logger = logging.getLogger(__name__)

NO_OP_HANDLED_HEAD_KEY = "no_op_handled_head"
NO_OP_DEFERRED_HEAD_KEY = "no_op_deferred_head"
NO_OP_ESCALATION_REASON = "rework_no_op"
_LOG_EXCERPT_LINES = 40
_LOG_EXCERPT_CHARS = 3000
_COMMENT_CHARS = 2000


class NoOpReworkRoute(NamedTuple):
    """One dead-worker no-op finding awaiting a post-lock disposition."""

    issue_number: int
    pr_number: int
    live_head_sha: str
    reason: str
    branch: str | None


def _failing_step_entries(
    gh: GitHubLike, checks: Sequence[dict[str, Any]], failed_names: frozenset[str]
) -> list[str]:
    """``required_changes`` lines naming each failing step and a tail of its log."""
    import charlie_work.workflow as _wf

    entries: list[str] = []
    for check in checks:
        name = str(check.get("name") or "")
        job_id = check.get("databaseId")
        if name not in failed_names or not _is_failing_run(check) or not isinstance(job_id, int):
            continue
        job = gh.actions_job(job_id)
        steps = job.get("steps") if isinstance(job, dict) else None
        failed_steps = [
            str(step.get("name") or "").strip()
            for step in (steps if isinstance(steps, list) else [])
            if isinstance(step, dict) and str(step.get("conclusion") or "").lower() == "failure"
        ]
        if failed_steps:
            entries.append(f"{name}: failing step(s): {', '.join(s for s in failed_steps if s)}")
        log_text = _wf._actions_job_log_text(gh, job_id)
        if log_text and log_text.strip():
            tail = "\n".join(log_text.strip().splitlines()[-_LOG_EXCERPT_LINES:])
            entries.append(
                f"{name}: log excerpt (last {_LOG_EXCERPT_LINES} lines):\n"
                + fenced_block(tail[-_LOG_EXCERPT_CHARS:])
            )
    return entries


def _worker_comment(
    route: NoOpReworkRoute, entry: dict[str, Any], sessions_dir: Path, repo_root: Any, wt: Any
) -> str | None:
    """The rework worker's ``pr_comment`` (its rebuttal), or ``None``."""
    if not isinstance(repo_root, Path) or wt is None:
        return None
    outcome = _read_rework_outcome(
        sessions_dir, repo_root, wt, route.issue_number, route.branch or entry.get("branch_name")
    )
    comment = outcome.get("pr_comment") if isinstance(outcome, dict) else None
    return (
        comment.strip()[:_COMMENT_CHARS] if isinstance(comment, str) and comment.strip() else None
    )


def drain_no_op_rework_routes(
    routes: Sequence[NoOpReworkRoute],
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    state_file: Path,
    write_gate: WriteGate,
    sessions_dir: Path,
    repo_root: Any,
    worktrees_dir: Path | None,
    review_callback: Callable[[int], Any] | None,
    record_review_callback: Callable[..., Any] | None,
    enrich_checks_callback: Callable[..., list[dict[str, Any]]] | None,
) -> None:
    """Give every collected no-op finding a disposition (see module docstring)."""
    import charlie_work.workflow as _wf

    escalations: list[int] = []
    for route in routes:
        try:
            _drain_one(
                route,
                gh=gh,
                config=config,
                state_file=state_file,
                write_gate=write_gate,
                sessions_dir=sessions_dir,
                repo_root=repo_root,
                worktrees_dir=worktrees_dir,
                review_callback=review_callback,
                record_review_callback=record_review_callback,
                enrich_checks_callback=enrich_checks_callback,
                escalations=escalations,
            )
        except Exception:
            # One throwing route must not starve the rest; the issue stays
            # dispatched, and the #654 backstop the sweep armed still converges it.
            logger.exception("no-op rework route escaped for issue %s", route.issue_number)
    edge = _wf._escalation_edge("escalated", "judgment")
    for issue_number in escalations:
        write_gate.transition(gh, config.labels, issue_number, edge)


def _drain_one(
    route: NoOpReworkRoute,
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    state_file: Path,
    write_gate: WriteGate,
    sessions_dir: Path,
    repo_root: Any,
    worktrees_dir: Path | None,
    review_callback: Callable[[int], Any] | None,
    record_review_callback: Callable[..., Any] | None,
    enrich_checks_callback: Callable[..., list[dict[str, Any]]] | None,
    escalations: list[int],
) -> None:
    import charlie_work.workflow as _wf

    issue_number, pr_number, head = route.issue_number, route.pr_number, route.live_head_sha
    with _wf.state_lock(state_file):
        entry = _wf.load_state(state_file)["issues"].get(str(issue_number), {})
    if not isinstance(entry, dict) or entry.get("status") != "dispatched":
        return  # a concurrent transition already resolved it
    comment = _worker_comment(route, entry, sessions_dir, repo_root, worktrees_dir)
    detail = f"{route.reason}: {comment}" if comment else route.reason

    if entry.get(NO_OP_HANDLED_HEAD_KEY) != head:
        checks = gh.pr_checks(pr_number)
        if checks is None:
            # CI state unknown: retry next pass (bounded by the #654 backstop).
            _defer_no_op(state_file, write_gate, route, ci_state="unavailable")
            return
        required = config.auto_merge.required_checks
        if enrich_checks_callback is not None:
            checks = enrich_checks_callback(checks, required)
        summary = summarize_checks(checks, required)
        if not summary.failed and not summary.ready:
            # Required CI has not settled -- a check is pending, missing,
            # infra-failed, infra-blocked, or unavailable -- so red vs green is
            # undecided and any disposition would be a guess. Defer to the next
            # pass, bounded by the same #654 backstop as the checks-is-None
            # case, rather than falling through to rework_no_op escalation.
            _defer_no_op(
                state_file,
                write_gate,
                route,
                ci_state="unsettled",
                unsettled={
                    bucket: sorted(getattr(summary, bucket))
                    for bucket in (
                        "pending",
                        "missing",
                        "infra_failed",
                        "infra_blocked",
                        "unavailable",
                    )
                    if getattr(summary, bucket)
                },
            )
            return
        if summary.failed and record_review_callback is not None:
            failed = tuple(summary.failed)
            required_changes = _failing_step_entries(
                gh, checks, frozenset(failed)
            ) + _required_changes_from_checks(checks, failed, gh.check_run_annotations)
            result = record_review_callback(
                pr_number,
                "request_changes",
                summary=(
                    f"CI failed on {', '.join(failed)}; the previous rework worker exited "
                    "without fixing it. Push a fix."
                ),
                reviewed_head=head,
                required_changes=required_changes,
                verdict_provenance="ci_gate_auto_reject",
            )
            if result.ok:
                _mark_handled(
                    state_file, write_gate, route, "rework_no_op_ci_rework_requested", detail
                )
                return
        elif summary.ready and comment and review_callback is not None:
            # Mark first: the rebuttal review runs once per head even if it fails.
            _mark_handled(state_file, write_gate, route, "rework_no_op_rebuttal_review", detail)
            review = review_callback(pr_number)
            if review.ok and not (
                review.data.get("routed_to_rework") or review.data.get("closed_unmerged_converged")
            ):
                with _wf.state_lock(state_file):
                    state = _wf.load_state(state_file)
                    cur = state["issues"].get(str(issue_number), {})
                    if isinstance(cur, dict) and cur.get("status") == "dispatched":
                        state["issues"][str(issue_number)] = {**cur, "status": "reviewing"}
                        write_gate.save_state(state)
                return

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        cur = state["issues"].get(str(issue_number), {})
        if not isinstance(cur, dict) or cur.get("status") != "dispatched":
            return
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=NO_OP_ESCALATION_REASON,
            reason_class="judgment",
            pr_number=pr_number,
            issue_extra={
                "dispatched_at": None,
                "orphan_drift_fingerprint": None,
                "orphan_drift_at": None,
                NO_OP_DEFERRED_HEAD_KEY: None,
                "no_op_worker_comment": comment,
            },
        )
        state = write_gate.append_event(
            state,
            # event-consumer: audit-only -- the actionable signal is the agent:human-needed
            # escalation applied for this issue; this records why it was escalated
            "rework_no_op_escalated",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "head_sha": head,
                "reason": route.reason,
                "worker_comment": comment,
            },
        )
        write_gate.save_state(state)
    escalations.append(issue_number)


def _defer_no_op(
    state_file: Path,
    write_gate: WriteGate,
    route: NoOpReworkRoute,
    *,
    ci_state: str,
    unsettled: dict[str, list[str]] | None = None,
) -> None:
    """Leave the issue ``dispatched``; record once per head why a pass declined to act.

    A no-op finding whose required CI has not settled gets no disposition this
    pass -- it is retried on a later pass and bounded by the #654
    ``dead_dispatched_reap_minutes`` backstop the sweep armed. The
    ``no_op_deferred_head`` marker both makes the wait inspectable (the issue is
    ``dispatched`` *because* CI was unsettled, not silently) and dedupes the
    audit event: a clean-exit finding re-collects every pass while CI stays
    unsettled, and re-emitting each time is the identical-event spam the
    cost-spirals dedup convention exists to prevent.
    """
    import charlie_work.workflow as _wf

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        cur = state["issues"].get(str(route.issue_number), {})
        if not isinstance(cur, dict) or cur.get("status") != "dispatched":
            return
        if cur.get(NO_OP_DEFERRED_HEAD_KEY) == route.live_head_sha:
            return  # already recorded for this head
        state["issues"][str(route.issue_number)] = {
            **cur,
            NO_OP_DEFERRED_HEAD_KEY: route.live_head_sha,
        }
        state = write_gate.append_event(
            state,
            # event-consumer: audit-only -- the actionable signal is the #654
            # dead-dispatched backstop this deferral waits on; this records why
            # the pass issued no disposition (vs. silently doing nothing).
            "rework_no_op_deferred",
            {
                "issue_number": route.issue_number,
                "pr_number": route.pr_number,
                "head_sha": route.live_head_sha,
                "ci_state": ci_state,
                "unsettled": unsettled or {},
            },
        )
        write_gate.save_state(state)


def _mark_handled(
    state_file: Path,
    write_gate: WriteGate,
    route: NoOpReworkRoute,
    kind: str,
    detail: str,
) -> None:
    """Record the head a tier-1/2 disposition was issued for, plus its audit event."""
    import charlie_work.workflow as _wf

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        cur = state["issues"].get(str(route.issue_number), {})
        if isinstance(cur, dict):
            state["issues"][str(route.issue_number)] = {
                **cur,
                NO_OP_HANDLED_HEAD_KEY: route.live_head_sha,
                NO_OP_DEFERRED_HEAD_KEY: None,
                "orphan_drift_at": None,
            }
        payload = {
            "issue_number": route.issue_number,
            "pr_number": route.pr_number,
            "head_sha": route.live_head_sha,
            "detail": detail,
        }
        if kind == "rework_no_op_ci_rework_requested":
            state = write_gate.append_event(
                state,
                # event-consumer: audit-only -- the actionable signal is the rework_requested
                # status record_review wrote; this names why the no-op was routed there
                "rework_no_op_ci_rework_requested",
                payload,
            )
        else:
            state = write_gate.append_event(
                state,
                # event-consumer: audit-only -- the review() outcome (reviewing / escalation)
                # is the actionable signal; this records that a rebuttal review was attempted
                "rework_no_op_rebuttal_review",
                payload,
            )
        write_gate.save_state(state)
