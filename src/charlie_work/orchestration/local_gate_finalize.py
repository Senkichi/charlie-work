"""Base advance for the asynchronous local merge gate (issues #1974, #2189).

Extracted from ``local_merge_gate.py`` to keep that module under its
file-size-ratchet mark. This is the gate's terminal half: once a suite
result proves a head/base pairing green, ``_local_gate_finalize_merge``
advances the base and runs the merged bookkeeping, or routes the
conflict/error outcomes. A *deferred* advance (a dirty base checkout refused
the fast-forward) keeps the green pairing in ``LOCAL_SUITE_PASSED_FIELDS`` so
later passes retry the merge alone while ``_local_gate_passed_pairing_current``
holds -- re-running a suite that already passed on that exact pairing proves
nothing new (issue #2189).

Every top-level ``def`` is installed on ``OrchestratorApp`` by
``workflow_delegation._install_delegates``; ``local_merge_gate.py`` reaches
these through ``self.`` and imports ``LOCAL_SUITE_PASSED_FIELDS`` directly.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work.labels import TransitionOutcome
from charlie_work.local_lane import (
    branch_head_sha,
    local_base_branch,
    merge_branch_into_base,
    resolve_ref_sha,
    worktree_for_branch,
)
from charlie_work.safe_path import contains
from charlie_work.state import without_review_dispatch_claim
from charlie_work.worktree import remove_review_checkout, remove_worktree

# Issue #2189: a green result whose base advance was *deferred* (a dirty base
# checkout refused the fast-forward) keeps the tested pairing here. While the
# live head and base still equal it, later passes retry the merge alone --
# re-running a suite that already passed on that exact pairing proves nothing
# new and burns a full suite run per deferral. Any drift drops the marker and
# the ordinary launch path re-syncs. Cleared by every terminal outcome.
LOCAL_SUITE_PASSED_FIELDS = (
    "local_suite_passed_head",
    "local_suite_passed_base",
)


def _local_gate_passed_pairing_current(self, record: dict[str, Any]) -> bool:
    """Whether a deferred green pairing (issue #2189) still describes the tree.

    True only when the record carries ``LOCAL_SUITE_PASSED_FIELDS`` and both
    the branch head and the base still resolve to the tested SHAs -- the one
    condition under which merging without a re-run is sound. A stale marker
    must not hold the gate: the walk drops it and relaunches that same pass.
    """
    passed_head = record.get("local_suite_passed_head")
    passed_base = record.get("local_suite_passed_base")
    if not passed_head or not passed_base:
        return False
    branch = str(record.get("branch") or record.get("headRefName") or "")
    base_ref = str(record.get("baseRefName") or local_base_branch(self.repo_root) or "HEAD")
    return bool(branch) and (
        branch_head_sha(self.repo_root, branch) == passed_head
        and resolve_ref_sha(self.repo_root, base_ref) == passed_base
    )


def _local_gate_finalize_merge(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    live_head: str,
    decision: dict[str, Any],
) -> bool:
    """Green suite + stable pairing: advance the base and run terminal bookkeeping.

    Same merge outcomes the synchronous gate produced -- deferred / conflict /
    error / merged -- only reached now from a later pass, after the result
    file proved the tested pairing. Returns True only for ``deferred``: the
    green pairing (``LOCAL_SUITE_PASSED_FIELDS``) is kept so the next pass
    retries the merge without re-running the suite (issue #2189). Every other
    outcome is terminal for the pairing and clears it.
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    worktrees_dir = self._layout.worktrees
    outcome = merge_branch_into_base(self.repo_root, base_ref, branch, worktrees_dir=worktrees_dir)
    if outcome.status == "deferred":
        entry["outcome"] = "deferred"
        entry["detail"] = outcome.detail
        self._local_gate_event(
            "local_merge_deferred",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "detail": outcome.detail,
                "retry": "merge_only",
            },
        )
        return True
    if outcome.status in ("conflict", "error"):
        self._local_gate_update(pr_key, {field: None for field in LOCAL_SUITE_PASSED_FIELDS})
    if outcome.status == "conflict":
        entry["outcome"] = "conflict"
        entry["conflicted_paths"] = list(outcome.conflicted_paths)
        entry["routed_to"] = self._local_route_merge_rework(
            pr_number,
            issue_number,
            record,
            decision,
            reason="merge_conflict",
            note=(
                f"Merging {branch!r} into the base branch {base_ref!r} "
                f"conflicted ({len(outcome.conflicted_paths)} path(s)). "
                "Merge the base into your branch and resolve the "
                "conflicts. The code changes are already approved; do not "
                "re-litigate the review."
            ),
        )
        return False
    if outcome.status == "error":
        entry["outcome"] = "error"
        entry["detail"] = outcome.detail
        self._local_merge_error_escalate(pr_number, issue_number, branch, outcome.detail)
        return False

    # Merged (or already contained): terminal bookkeeping.
    gate_head = record.get("local_suite_head")
    merged_sha = outcome.merged_sha or gate_head or live_head
    entry["outcome"] = "merged" if outcome.status == "merged" else "already_merged"
    entry["fast_forward"] = outcome.fast_forward
    entry["merged_sha"] = merged_sha
    # force=True like every other worker-worktree teardown: the branch
    # content is committed to base by construction, so remaining
    # uncommitted material is worker scratch (.venv, build output,
    # worker-tmp) -- a plain remove would refuse on a real .venv dir and
    # leak a worktree per merged issue. ``branch`` also deletes the
    # merged branch ref when auto_merge.delete_branch is configured,
    # matching the remote lane's post-merge cleanup.
    # Issue #1476: a worktree outside the managed dir is a foreign
    # checkout (``ensure_branch_worktree`` returns the registered path
    # wherever the branch happens to live) -- it belongs to whoever
    # created it and is never removed; the branch ref can't be deleted
    # while checked out anyway.
    worktree_path = worktree_for_branch(self.repo_root, branch)
    if worktree_path is not None and contains(worktrees_dir, worktree_path):
        remove_worktree(
            self.repo_root,
            worktree_path,
            force=True,
            branch=(branch if self.config.auto_merge.delete_branch else None),
        )
    remove_review_checkout(self.repo_root, pr_number, reviews_dir=self._layout.reviews_dir)
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(pr_key, {})
        state["prs"][pr_key] = {
            **without_review_dispatch_claim(pr_state),
            "number": pr_number,
            "local": True,
            "status": "merged",
            "merged_at": _wf.utc_now(),
            "merged_sha": merged_sha,
            "merge_method": "ff" if outcome.fast_forward else "no-ff",
            # Issue #1972: the merge-gate rework episode is resolved --
            # both per-kind attempt counters restart from zero. They
            # deliberately survive packet re-mints and rework routing;
            # this write plus the unescalate/de-escalation reset maps
            # are the only resets.
            "local_merge_conflict_rework_attempts": 0,
            "local_suite_failed_rework_attempts": 0,
            "local_suite_infra_relaunch_count": 0,
            "local_merge_rework_reason": None,
            **{field: None for field in LOCAL_SUITE_PASSED_FIELDS},
        }
        issue_entry = state["issues"].get(str(issue_number), {})
        state["issues"][str(issue_number)] = _wf._merged_issue_fields(issue_entry, issue_number)
        state = self._record_event(
            state,
            "merge_succeeded",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "local": True,
                "fast_forward": outcome.fast_forward,
                "merged_sha": merged_sha,
                "detail": outcome.detail,
            },
        )
        self.write_gate.save_state(state)
    transition_result = self.write_gate.transition(
        self.gh, self.config.labels, issue_number, "merged"
    )
    if transition_result.outcome != TransitionOutcome.APPLIED:
        entry["label_error"] = {
            "edge": "merged",
            "outcome": transition_result.outcome.value,
        }
    # The issue itself closes on merge, same as a merged remote PR.
    try:
        self.gh.close_issue(issue_number)
    except Exception:
        entry["close_error"] = True
    return False
