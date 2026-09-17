"""Rescue-candidate partition and rescue-review delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any

from charlie_work import rescue as rescue_helpers
from charlie_work.github import GitHubError
from charlie_work.labels import TransitionOutcome
from charlie_work.rescue_review import extract_report_body
import charlie_work.workflow as _wf


def _partition_rescue_candidates(
    self, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``review_queue()`` candidates into normal vs rescue-marked
    (issue #555), and process the rescue ones via ``_process_rescue_review``.

    Routing keys on the durable ``rescue_attempted`` marker ALONE, never
    on ``self.config.rescue.enabled``: an operator flipping ``enabled``
    off while a rescue is in flight must not cause an already-marked PR
    to fall through to a normal same-family reviewer, or (on a later
    request_changes) to the legacy escalation path without the two
    rescue artifacts. ``enabled`` only gates NEW rescue entry at the
    three cap sites (record_review / _route_janitor_gate_failure_to_
    rework) — routing of PRs that already carry the marker is
    unconditional.

    A rescue-marked PR whose ``pr_state["status"] == "escalated"`` is
    dropped entirely — not processed again, not returned as a normal
    candidate either. ``_process_rescue_review``'s escalation write does
    not advance the review packet's recorded decision, so
    ``review_queue()`` would otherwise keep requeuing it every pass,
    re-running the (blocking, up to ``reviewer_timeout_seconds``)
    cross-family review, reposting the escalation PR comment, and
    re-firing ``rescue_review_escalated`` forever.
    """
    rescue_snapshot = _wf.load_state_locked(self.paths.state_file)
    normal_candidates: list[dict[str, Any]] = []
    rescue_review_results: list[dict[str, Any]] = []
    for c in candidates:
        pr_state = rescue_snapshot.get("prs", {}).get(str(c["pr"]), {})
        if not pr_state.get("rescue_attempted"):
            normal_candidates.append(c)
            continue
        if pr_state.get("status") == "escalated":
            continue
        result = self._process_rescue_review(c)
        rescue_review_results.append({"pr": c["pr"], **result.data})
    return normal_candidates, rescue_review_results


def _process_rescue_review(self, candidate: dict[str, Any]) -> _wf.CommandResult:
    """Run the cross-family rescue review for one rescue-marked PR
    (issue #555) and apply the rescue tier's exit semantics.

    Called from ``dispatch_reviews()`` INSTEAD of the normal Claude
    reviewer dispatch for any candidate whose PR record already carries
    ``rescue_attempted`` — i.e. its rescue rework was already dispatched
    (via the normal rework-dispatch path, just adapter/model-overridden;
    see ``_rescue_adapter_settings``) and has since pushed a new head.

    Synchronous and one-shot (mirrors ``run_cross_family_review``'s own
    contract): no new polling/reaping machinery is introduced. Exit
    semantics per the issue spec:

    - ``approved`` -> recorded through the SAME entry point normal
      reviews use (``record_review``), so the existing ship-it/merge
      path takes over exactly as it would for a normal approval.
    - ``request_changes``/``blocked``/unparseable report -> escalates to
      a human immediately (never loops back into another rework cycle):
      both artifacts (the rescue PR/diff — this PR's current branch and
      head SHA, since the rescue worker pushed directly to it — and the
      cross-family report path) are attached to the escalation event
      payload and posted as a PR comment.
    """
    pr_number = int(candidate["pr"])
    issue_number = candidate.get("issue")

    # Issue #618: short-circuit in dry-run BEFORE any writes or mutations.
    # Threading dry_run to run_cross_family_review alone is actively
    # harmful here: the dry-run branch returns a synthetic failure
    # (ok=False), which drives this function into its escalation arm and
    # would mark a PR escalated during a preview. The short-circuit
    # avoids the mkdir, the cross-family subprocess, the state/label
    # escalation write, and the PR comment entirely.
    if self.dry_run:
        return _wf.CommandResult(
            True,
            f"dry-run: would run rescue review for PR #{pr_number}",
            {
                "pr": pr_number,
                "issue": issue_number,
                "rescue_review_decision": "dry-run",
                "dry_run": True,
            },
        )

    pr = self.gh.pr_view(pr_number)
    head_sha = str(pr.get("headRefOid") or "")
    branch = str(pr.get("headRefName") or "")
    pr_state = _wf.load_state_locked(self.paths.state_file).get("prs", {}).get(str(pr_number), {})
    cause = str(pr_state.get("rescue_cause") or "unknown")

    pr_dir = self.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    diff_text = self.gh.pr_diff(pr_number) or ""
    prompt_text = rescue_helpers.build_rescue_review_prompt(
        pr_number=pr_number,
        issue_number=issue_number,
        branch=branch,
        diff_text=diff_text,
        cause=cause,
    )
    prompt_path = pr_dir / "rescue-review-prompt.md"
    report_path = pr_dir / "rescue-review-report.md"
    cfg = self.config.rescue
    cf_result = _wf.run_cross_family_review(
        model=cfg.reviewer_model,
        command=cfg.reviewer_command,
        repo_root=self.repo_root,
        prompt_text=prompt_text,
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=cfg.reviewer_timeout_seconds,
        head_ref_oid=head_sha,
    )

    verdict: dict[str, Any] | None = None
    if cf_result.ok and report_path.exists():
        report_text = extract_report_body(report_path.read_text(encoding="utf-8"))
        verdict = _wf._extract_verdict_from_text(report_text)

    if verdict is not None and verdict["decision"] == "approved":
        result = self.record_review(
            pr_number,
            "approved",
            summary=verdict["summary"],
            reviewed_head=head_sha,
            required_changes=verdict["required_changes"],
            verdict_provenance="rescue_review",
        )
        return _wf.CommandResult(
            True,
            f"PR #{pr_number} rescue review approved — {result.message}",
            {
                "rescue_review_decision": "approved",
                "cross_family_report": str(report_path),
                "record_review": result.data,
            },
        )

    # request_changes / blocked / unparseable report -> escalate to a
    # human immediately. Never re-enter the rework loop: this PR already
    # spent its one rescue attempt (rescue_attempted stays durably set).
    verdict_summary = verdict["summary"] if verdict else (cf_result.error or "")
    comment_body = rescue_helpers.build_rescue_escalation_comment(
        cause=cause,
        rescue_branch=branch,
        rescue_head_sha=head_sha,
        cross_family_report_path=str(report_path),
        verdict_summary=verdict_summary,
    )
    label_error: dict[str, Any] | None = None
    # Issue #783: an explicit "blocked" rescue verdict is the same kind of
    # human product/security decision as record_review's "blocked" path
    # -- judgment, never auto-cleared. "request_changes" and an
    # unparseable/failed cross-family report both escalate only because
    # the rescue tier structurally cannot loop again (one-shot, no
    # further rework cycle) -- that is a process limit, not a judgment
    # call, so it is mechanical and eligible for re-evaluation.
    rescue_reason_class = (
        "judgment" if verdict is not None and verdict["decision"] == "blocked" else "mechanical"
    )
    rescue_escalation_reason = f"rescue_review_{cause}"
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=rescue_escalation_reason,
            reason_class=rescue_reason_class,
            pr_number=pr_number,
        )
        state = self._record_event(
            state,
            "rescue_review_escalated",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "cause": cause,
                "rescue_branch": branch,
                "rescue_head_sha": head_sha,
                "cross_family_report": str(report_path),
                "cross_family_ok": cf_result.ok,
                "verdict_decision": verdict["decision"] if verdict else None,
            },
        )
        _wf.save_state(self.paths.state_file, state)
    if issue_number is not None:
        rescue_edge = _wf._escalation_edge("escalated", rescue_reason_class)
        result = _wf.transition(self.gh, self.config.labels, int(issue_number), rescue_edge)
        if result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": rescue_edge,
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
    try:
        self._comment_pr(pr_number, comment_body)
    except GitHubError as exc:
        label_error = {**(label_error or {}), "comment_error": str(exc)}
    return _wf.CommandResult(
        True,
        f"PR #{pr_number} rescue review did not approve — escalated to human",
        {
            "rescue_review_decision": verdict["decision"] if verdict else "unparseable",
            "cross_family_report": str(report_path),
            "cross_family_ok": cf_result.ok,
            "escalated": True,
            "label_error": label_error,
        },
    )
