"""Rework-to-review routing and stranded-commit salvage delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf


def _route_rework_candidate_to_review(
    self,
    issue_number: int,
    pr_number: int,
    reviewed_head_sha_before: str | None,
) -> tuple[bool, _wf.CommandResult]:
    """Route a rework_requested issue back to the review lane instead of
    relaunching a worker onto a PR whose rework was already pushed
    (issue #339): the PR head moved past the last request_changes verdict,
    so the previous worker's output is already live and a relaunch would
    find nothing to do, idle, and get watchdog-reaped.

    Reuses ``review()`` — the review lane's own packet-regeneration entry
    point — instead of duplicating its janitor/test-adequacy gating and
    label-transition logic here.

    Returns a ``(routed, review_result)`` pair. ``routed`` is True only
    when ``review()`` actually produced a fresh, undecided packet against
    the new head. ``review()`` has its own early-returns that leave
    GitHub/labels untouched — most notably the deterministic janitor gate
    (conflicting/draft/red-CI), which returns ``ok=False`` *before*
    writing any packet or firing the ``review_started`` transition, and
    without touching ``reviewed_head_sha``. Flipping the issue's status
    to "reviewing" in that case would desync state.json from GitHub
    reality (labels still say needs-rework, no packet exists) with no
    automated recovery path, since the issue would silently drop out of
    dispatch_rework's own candidate pool forever (issue #339 finding 1).
    So the status flip additionally requires ``review_result.ok`` on top
    of the pre-existing "no fresh decision recorded" check: ``review()``
    can itself invoke ``record_review`` (the test-adequacy hard gate
    re-failing on the new head), which already reconciles the issue's
    status and ``reviewed_head_sha`` — that path returns ``ok=True`` but
    must not be re-flipped here either, so both checks are required.

    When ``routed`` is False, the issue's status is left untouched
    (``rework_requested``), so the next dispatch_rework pass naturally
    retries — the block is often transient (e.g. a merge-train branch
    sync resolving a conflict).
    """
    review_result = self.review(pr_number)
    routed = False
    # review() can now return ok=True for a reason OTHER than "a fresh
    # review packet was produced": the janitor-gate conflict/no-op-rework
    # routing (_route_janitor_gate_failure_to_rework) also returns ok=True
    # when it re-requests rework, with no packet and no review_started
    # transition. That outcome must be treated the same as "review()
    # blocked" here -- the issue stays rework_requested for the next
    # dispatch_rework pass, not flipped to "reviewing" -- otherwise this
    # would desync state.json from GitHub reality exactly the way the
    # ok=False janitor-block case already guards against (issue #339
    # finding 1, see this method's docstring).
    routed_to_rework = bool(review_result.data.get("routed_to_rework"))
    # Issue #558: review() also returns ok=True when it converges a
    # CLOSED-unmerged PR's state entry to "closed" at the janitor gate.
    # The PR is dead, not a fresh-packet candidate, so flipping the issue
    # to "reviewing" here would strand it in an ACTIVE_STATE_STATUS no
    # reconcile rule clears while the GitHub issue stays open (the closed
    # PR still links to the issue, so issue_active_label_no_open_pr does
    # not fire; "reviewing" is a VALID_ISSUE_STATUSES member, so the
    # unknown-status recompute sweep skips it). The issue stays
    # rework_requested and the existing closed-unmerged issue-side
    # handling (closed_unmerged_pr_active_labels) finalizes it.
    closed_unmerged_converged = bool(review_result.data.get("closed_unmerged_converged"))
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        entry = state["issues"].get(str(issue_number), {})
        decision_unchanged = pr_state.get("reviewed_head_sha") == reviewed_head_sha_before
        if (
            review_result.ok
            and not routed_to_rework
            and not closed_unmerged_converged
            and decision_unchanged
            and isinstance(entry, dict)
            and entry.get("status") == "rework_requested"
        ):
            state["issues"][str(issue_number)] = {**entry, "status": "reviewing"}
            routed = True
        state = _wf.append_event(
            state,
            "rework_already_pushed",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "review_ok": review_result.ok,
                "routed": routed,
            },
            state_path=self.paths.state_file,
        )
        _wf.save_state(self.paths.state_file, state)
    return routed, review_result


def _salvage_rework_stranded_commits(
    self,
    issue_number: int,
    pr_data: dict[str, Any],
    issue_entry: dict[str, Any],
) -> bool:
    """Salvage-push stranded commits from a dead rework worker's worktree
    before the death-loop escalation gate fires (issue #1239).

    Returns ``True`` when the push succeeded (stranded commits were
    published); ``False`` for any non-push outcome (no stranded commits,
    divergence, remote-head-not-local, missing worktree/branch, or git
    error).  The caller treats a ``True`` result as "the death produced
    completed work — do NOT count it toward ``worker_death_loop``" and
    routes the issue to review instead of escalating.

    Reuses ``salvage_push_stranded_commits`` — the same sanctioned-git
    primitive the fresh-dispatch salvage lane uses (#1248): ls-remote
    before pushing (never trust the sidecar's ``push_succeeded``), never
    force-push, fast-forward only.  Records a state event so the salvage
    is observable even when the death-loop gate would have escalated.
    """
    live_head = pr_data.get("headRefOid")
    if not live_head:
        return False
    branch = issue_entry.get("branch_name") if isinstance(issue_entry, dict) else None
    if not branch:
        return False
    # Issue #1239 round-3: ``issue_entry`` comes from the
    # ``head_check_state`` snapshot loaded at the top of
    # ``dispatch_rework``'s candidate loop.  Between that snapshot and
    # this call the issue's status may have already moved off
    # ``rework_requested`` (e.g. a concurrent loop pass dispatched it,
    # escalated it, or the issue was closed).  Re-check under the state
    # lock BEFORE any network push — a salvage push to the shared origin
    # remote for an issue that is no longer rework_requested would be an
    # unaudited side effect with no event trail if it succeeded.  Mirrors
    # the precondition added to ``_reap_restore_rework_requested`` (which
    # checks ``status == "dispatched"`` because that lane handles
    # already-dispatched workers); this lane's eligible state is
    # ``rework_requested`` because ``dispatch_rework`` only selects
    # candidates with that status.  The network git push below stays
    # outside any lock; the event-recording state_lock scope further down
    # is still load-bearing (status can move again between this check and
    # that scope).
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        fresh_entry = state["issues"].get(str(issue_number), {})
        if not isinstance(fresh_entry, dict) or fresh_entry.get("status") != "rework_requested":
            return False
    wt_path = _wf.worktree_path_for_branch(self.repo_root, branch, self._layout.worktrees)
    result = _wf.salvage_push_stranded_commits(
        self.repo_root,
        branch,
        wt_path,
        base_ref=self.config.dispatch.base_ref,
        dry_run=self.write_gate.dry_run,
    )
    if not result.pushed:
        return False
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state = self.write_gate.append_event(
            state,
            "rework_stranded_commits_salvaged",
            {
                "issue_number": issue_number,
                "previous_status": "rework_requested",
                "new_status": "rework_requested",
                "reason": "death_loop_salvaged",
                "commit_count": result.commit_count,
                "old_remote_sha": result.old_remote_sha,
                "new_remote_sha": result.new_remote_sha,
            },
        )
        self.write_gate.save_state(state)
    return True
