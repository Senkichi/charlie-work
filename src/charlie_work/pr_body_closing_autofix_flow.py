"""CI-gate PR-body closing-keyword autofix flow for ``review()`` (issue #2108).

``review()``'s CI-red short-circuit used to route every required-check failure to
worker rework. A Lint failure whose only failing step is the closing-keyword gate
(issue #790) with the finding ``via pr body`` is not code a worker can change --
workers have no ``gh`` token, so they cannot edit the body -- and each no-op
rework burned the no-op cap until the PR escalated (#2005 / PR #2021). This
delegate classifies that case as orchestrator-fixable: it rewrites the stray
reference, re-runs the failed Lint run, and emits
``pr_body_closing_keyword_autofixed``. Any failure to repair escalates; it never
falls back to rework. The mechanics live in
:mod:`charlie_work.pr_body_closing_autofix`; this module owns state, events, and
escalation.

These are free functions taking the app, not ``orchestration/`` delegates: the
installer would graft each ``def`` onto ``OrchestratorApp`` and break the
conserved member surface (``test_workflow_delegation_l00``).

Routing contract (every exit of ``autofix_body_closing_kw``):

- not a sole required-check failure on a linked issue, gate-only failure not provable, or a ``via commit message`` finding -> ``None``
  (caller keeps the worker-rework routing);
- per-head attempt cap reached -> escalate;
- body repaired and/or rerun requested -> ``CommandResult(False, ...)``;
- rerun refused because the run is still in progress, or the scan's pr_view /
  pr_commits / compare fetch failed -> held, retried next pass (a fetch failure
  says nothing about the body, and an escalation is terminal while Lint is red).
  A scan-unavailable hold spends the per-head attempt budget, so a persistent
  fetch failure escalates via ``attempt_cap_exceeded`` instead of holding forever;
- body edit or rerun failure, or a body finding with no declared target -> escalate.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work.labels import TransitionOutcome
from charlie_work.pr_body_closing_autofix import (
    MAX_AUTOFIX_ATTEMPTS_PER_HEAD,
    AutofixOutcome,
    SCAN_UNAVAILABLE_REASON,
    AutofixResult,
    autofix_closing_keyword_pr_body,
    failing_runs_are_closing_keyword_gate_only,
)

_ATTEMPTS_KEY = "closing_keyword_autofix_attempts"
_ESCALATION_REASON = "pr_body_closing_keyword_autofix_failed"


def autofix_body_closing_kw(
    app: Any,
    pr: dict[str, Any],
    issue_number: int | None,
    verdict: Any,
    checks: list[dict[str, Any]] | None,
) -> _wf.CommandResult | None:
    """Repair a PR-body-only closing-keyword gate failure; ``None`` when not applicable."""
    if issue_number is None or not verdict.is_check_failure_block:
        return None
    pr_number = int(pr["number"])
    run_ids = failing_runs_are_closing_keyword_gate_only(
        checks, verdict.failed_required_checks, app.gh.actions_job
    )
    if run_ids is None:
        return None

    head_sha = str(pr.get("headRefOid") or "")
    state = _wf.load_state(app.paths.state_file)
    # The mechanical de-escalation sweep zeroes counters with ``0`` (see
    # REWORK_BUDGET_RESET_BY_ESCALATION_REASON); a non-dict means "no attempts".
    raw_attempts = (state["prs"].get(str(pr_number)) or {}).get(_ATTEMPTS_KEY)
    attempts_by_head = dict(raw_attempts) if isinstance(raw_attempts, dict) else {}
    attempts = int(attempts_by_head.get(head_sha, 0))
    if attempts >= MAX_AUTOFIX_ATTEMPTS_PER_HEAD:
        return _escalate_autofix_failure(
            app, pr_number, issue_number, head_sha, "attempt_cap_exceeded"
        )

    result = autofix_closing_keyword_pr_body(
        app.gh,
        pr_number=pr_number,
        run_ids=run_ids,
        branch_prefix=app.config.dispatch.branch_prefix,
        body_path=app.paths.prs / f"pr-{pr_number}" / "closing-keyword-autofix-body.md",
        rerun_already_running=_wf._is_rerun_already_running_error,
    )
    if result.outcome is AutofixOutcome.NOT_APPLICABLE:
        return None
    if result.outcome is AutofixOutcome.FAILED:
        return _escalate_autofix_failure(app, pr_number, issue_number, head_sha, result.reason)
    if result.outcome is AutofixOutcome.HELD:
        if result.reason == SCAN_UNAVAILABLE_REASON:
            # A persistent fetch failure must not hold the Lint-red lane forever:
            # each held pass spends the per-head budget, so the cap escalates it.
            _persist_attempts(app, pr_number, issue_number, head_sha, attempts_by_head)
            return _wf.CommandResult(
                False,
                f"PR #{pr_number} closing-keyword autofix held: PR scan unavailable, retrying",
                {"pr": pr_number, "issue": issue_number, "scan_unavailable": True},
            )
        return _wf.CommandResult(
            False,
            f"PR #{pr_number} closing-keyword autofix held: Lint run still in progress",
            {"pr": pr_number, "issue": issue_number, "already_running": True},
        )
    return _record_autofix(app, pr_number, issue_number, head_sha, attempts_by_head, result)


def _with_attempt(
    state: dict[str, Any],
    pr_number: int,
    issue_number: int,
    head_sha: str,
    attempts_by_head: dict[str, int],
) -> dict[str, Any]:
    """Spend one per-head autofix attempt in ``state['prs']``."""
    state["prs"][str(pr_number)] = {
        **state["prs"].get(str(pr_number), {}),
        "number": pr_number,
        "issue_number": issue_number,
        _ATTEMPTS_KEY: {
            **attempts_by_head,
            head_sha: int(attempts_by_head.get(head_sha, 0)) + 1,
        },
    }
    return state


def _persist_attempts(
    app: Any,
    pr_number: int,
    issue_number: int,
    head_sha: str,
    attempts_by_head: dict[str, int],
) -> None:
    """Spend one per-head attempt for a held pass (no event)."""
    with _wf.state_lock(app.paths.state_file):
        state = _wf.load_state(app.paths.state_file)
        state = _with_attempt(state, pr_number, issue_number, head_sha, attempts_by_head)
        _wf.save_state(app.paths.state_file, state)


def _record_autofix(
    app: Any,
    pr_number: int,
    issue_number: int,
    head_sha: str,
    attempts_by_head: dict[str, int],
    result: AutofixResult,
) -> _wf.CommandResult:
    """Persist the attempt count and emit ``pr_body_closing_keyword_autofixed``."""
    with _wf.state_lock(app.paths.state_file):
        state = _wf.load_state(app.paths.state_file)
        state = _with_attempt(state, pr_number, issue_number, head_sha, attempts_by_head)
        state = app._record_event(
            state,
            # event-consumer: audit-only -- the repair already happened (body edited,
            # Lint re-run requested); the failure sibling escalates and is the
            # actionable signal
            "pr_body_closing_keyword_autofixed",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "head_sha": head_sha,
                "body_edited": result.body_edited,
                "rewritten": list(result.rewritten),
                "run_ids": list(result.run_ids),
            },
        )
        _wf.save_state(app.paths.state_file, state)
    return _wf.CommandResult(
        False,
        f"PR #{pr_number} closing-keyword gate failure repaired by the orchestrator; "
        f"Lint re-run requested (run(s) {', '.join(str(r) for r in result.run_ids)})",
        {
            "pr": pr_number,
            "issue": issue_number,
            "closing_keyword_autofixed": True,
            "body_edited": result.body_edited,
            "rerun_run_ids": list(result.run_ids),
        },
    )


def _escalate_autofix_failure(
    app: Any, pr_number: int, issue_number: int, head_sha: str, reason: str
) -> _wf.CommandResult:
    """Escalate to the operator queue -- never to worker rework, which cannot edit the body."""
    with _wf.state_lock(app.paths.state_file):
        state = _wf.load_state(app.paths.state_file)
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=_ESCALATION_REASON,
            reason_class="mechanical",
            pr_number=pr_number,
            pr_extra={"closing_keyword_autofix_failure": reason},
        )
        state = app._record_event(
            state,
            # event-consumer: audit-only -- terminal record of the escalation that
            # _escalate_issue + the escalated label edge below already acted on
            "pr_body_closing_keyword_autofix_failed",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "head_sha": head_sha,
                "reason": reason,
            },
        )
        _wf.save_state(app.paths.state_file, state)

    edge = _wf._escalation_edge("escalated", "mechanical")
    transition_result = _wf.transition(app.gh, app.config.labels, issue_number, edge)
    label_error = None
    if transition_result.outcome != TransitionOutcome.APPLIED:
        label_error = {"edge": edge, "outcome": transition_result.outcome.value}
    return _wf.CommandResult(
        False,
        f"PR #{pr_number} closing-keyword gate failure could not be repaired "
        f"({reason}); escalated to human",
        {
            "pr": pr_number,
            "issue": issue_number,
            "closing_keyword_autofix_failed": True,
            "label_error": label_error,
        },
    )
