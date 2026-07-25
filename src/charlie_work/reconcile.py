"""Detect and (optionally) fix drift between GitHub reality and local state.

Production incident class this exists for: a human merges a PR by hand (or
closes an issue, or edits labels) outside the orchestrator's own
``merge_ready``/``transition`` codepaths. ``state.json`` and GitHub labels
then permanently disagree with reality — e.g. ``agent:in-progress`` /
``agent:reviewing`` never clears because the label edge that would clear it
(``labels.transition(..., "merged")``) never ran.

``detect_drift`` is read-only: it issues exactly one PR list query and one
issue list query via ``gh.run`` and never calls a mutating GitHub method.
``apply_fixes`` is the only function in this module that mutates GitHub, and
it is never invoked implicitly — callers gate it behind an explicit
``--fix`` flag. It reuses ``labels.transition`` for the one drift
kind that maps onto a standard lifecycle edge (a PR merged outside the
orchestrator behaves exactly like a normal merge once discovered) and issues
direct ``remove_issue_label`` calls only for label combinations that
``labels.transition`` has no edge for (contradictory terminal+active labels).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DETERMINISTIC_ESCALATION_FAILURE_KINDS, OrchestratorConfig
from .github import (
    GitHub,
    GraphQLBudgetError,
    _LIST_LIMIT,
    RECONCILE_ISSUE_FIELDS,
    RECONCILE_PR_FIELDS,
    label_names,
    linked_issue_number,
)
from .labels import TransitionOutcome, transition
from .process_utils import kill_process_tree
from .state import (
    PASSIVE_OPEN_STATUS,
    VALID_ISSUE_STATUSES,
    append_event,
    is_claim_stale,
    set_throttled_until,
    without_review_dispatch_claim,
)
from .worktree import (
    WorktreeState,
    inspect_worktree_state,
    push_branch,
    remove_review_checkout,
    resolve_base_branch_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftItem:
    kind: str
    issue_number: int | None
    pr_number: int | None
    detail: str
    fix_actions: tuple[str, ...]
    remove_labels: tuple[str, ...] = ()
    add_labels: tuple[str, ...] = ()
    branch: str | None = None
    base_branch: str | None = None
    # Target value for a state["issues"/"prs"][n]["status"] rewrite. Only
    # populated by drift kinds whose fix is a pure status recomputation
    # (issue_active_label_with_open_pr, issue_status_normalized,
    # pr_status_normalized) -- None means "clear the status key" for those
    # kinds, and is simply unused (not "clear") for every other kind.
    new_status: str | None = None


# State-machine statuses that mean "this issue is in the orchestrator's pipeline".
# Any of these on a closed GitHub issue is a drift condition (issue #259).
# "escalated" is included so a closed-while-escalated issue gets its state entry
# finalized; the terminal human_needed label is intentionally preserved.
# INVARIANT (guarded by test_fix_reconcile's coverage test): every
# VALID_ISSUE_STATUSES member except the deliberate exclusions below must be
# in this set. The status-normalization sweep skips anything in
# VALID_ISSUE_STATUSES, so a valid status missing here is invisible to BOTH
# sweeps -- a closed issue stuck in it would never be finalized (the exact
# dead zone adding manifest_written/dispatch_failed to the valid set briefly
# created). Deliberate exclusions: "closed" (already terminal), "approved"/
# "blocked" (finalization of closed approved/blocked issues is owned by the
# merged-PR finalization flow, pre-existing behavior).
ACTIVE_STATE_STATUSES: frozenset[str] = frozenset(
    {
        "dispatched",
        "dispatch_pending",
        "manifest_written",
        "dispatch_failed",
        "rework_requested",
        "reviewing",
        "escalated",
    }
)


def _issue_state(issue: dict[str, Any] | None) -> str:
    """Return the upper-cased GitHub state for an issue dict.

    Missing state defaults to OPEN to keep tests/fixtures that pre-date the
    `--state all` issue list (issue #259) behaving as open issues.
    """
    if issue is None:
        return ""
    return str(issue.get("state") or "OPEN").upper()


def _fetch_prs(gh: GitHub) -> list[dict[str, Any]]:
    result = gh.run(
        [
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            str(_LIST_LIMIT),
            "--json",
            RECONCILE_PR_FIELDS,
        ],
        json_output=True,
    )
    return result if isinstance(result, list) else []


def _fetch_issues(gh: GitHub) -> list[dict[str, Any]]:
    result = gh.run(
        [
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            str(_LIST_LIMIT),
            "--json",
            RECONCILE_ISSUE_FIELDS,
        ],
        json_output=True,
    )
    return result if isinstance(result, list) else []


def detect_drift(
    gh: GitHub, state: dict[str, Any], config: OrchestratorConfig, *, repo_root: Path | None = None
) -> list[DriftItem]:
    """Read-only comparison of GitHub reality against ``state``.

    Issues exactly two ``gh.run`` list queries (all PRs, all issues) and
    performs every drift check against those two in-memory snapshots — no
    per-item ``gh`` calls.

    If ``repo_root`` is provided, also checks for dead sessions and classifies
    their failures to update the provider throttle state.
    """
    threshold = config.runtime.graphql_rate_limit_threshold
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold)
    if not sufficient:
        raise GraphQLBudgetError(remaining, reset_at, threshold)

    labels_cfg = config.labels
    prs = _fetch_prs(gh)
    issues = _fetch_issues(gh)
    issues_by_number = {int(issue["number"]): issue for issue in issues if issue.get("number")}
    # Issue #45: the `--state all` query is capped at _LIST_LIMIT. If it returns
    # that many, the snapshot is provably incomplete; refuse to run sweeps that
    # depend on seeing every issue, but still run per-issue checks for the
    # issues that ARE in the snapshot.
    issue_snapshot_truncated = len(issues) >= _LIST_LIMIT
    state_prs: dict[str, Any] = state.get("prs", {})

    drift: list[DriftItem] = []
    prs_linking_issue: dict[int, list[dict[str, Any]]] = {}
    open_prs_by_issue: dict[int, list[dict[str, Any]]] = {}
    # Track issues already handled by session relabel to avoid double-emission
    # with issue_active_label_no_open_pr (both fire for dead-session-with-no-PR-ever)
    issues_handled_by_session_relabel: set[int] = set()
    # Track issues with live sessions to avoid false-positive drift detection
    # (issue #214: don't strip labels from workers that are still running)
    live_session_issue_numbers: set[int] = set()
    # Track issues whose status was already recomputed by
    # issue_active_label_with_open_pr below, so the status-normalization sweep
    # doesn't also emit a second, duplicate drift item for the same issue in
    # the same pass.
    issues_status_repaired: set[int] = set()

    for pr in prs:
        pr_number = pr.get("number")
        if pr_number is None:
            continue
        pr_number = int(pr_number)
        gh_state = str(pr.get("state") or "").upper()
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        if issue_number is not None:
            prs_linking_issue.setdefault(issue_number, []).append(pr)
            if gh_state == "OPEN":
                open_prs_by_issue.setdefault(issue_number, []).append(pr)

        state_entry = state_prs.get(str(pr_number))

        if gh_state == "MERGED":
            state_status = (state_entry or {}).get("status")
            issue = issues_by_number.get(issue_number) if issue_number is not None else None
            issue_still_active = bool(issue is not None and label_names(issue) & labels_cfg.active)
            if state_status != "merged" or issue_still_active:
                fix_actions = [f"mark state prs[{pr_number}].status = 'merged'"]
                if issue_number is not None and issue_still_active:
                    fix_actions.append(
                        f"transition issue #{issue_number} labels via 'merged' event"
                    )
                drift.append(
                    DriftItem(
                        kind="merged_outside_orchestrator",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"PR #{pr_number} is MERGED on GitHub but state status is "
                            f"{state_status!r} and linked issue #{issue_number} still carries "
                            f"active labels"
                            if issue_still_active
                            else (
                                f"PR #{pr_number} is MERGED on GitHub but state status is "
                                f"{state_status!r}"
                            )
                        ),
                        fix_actions=tuple(fix_actions),
                    )
                )
        elif gh_state == "CLOSED":
            issue = issues_by_number.get(issue_number) if issue_number is not None else None
            issue_active_labels = label_names(issue) & labels_cfg.active if issue else set()
            if issue is not None and issue_active_labels:
                drift.append(
                    DriftItem(
                        kind="closed_unmerged_pr_active_labels",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"PR #{pr_number} was closed without merging but issue "
                            f"#{issue_number} still carries active labels "
                            f"{sorted(issue_active_labels)}"
                        ),
                        fix_actions=tuple(
                            f"remove label '{label}' from issue #{issue_number}"
                            for label in sorted(issue_active_labels)
                        ),
                        remove_labels=tuple(sorted(issue_active_labels)),
                    )
                )
            # Issue #558: converge the state PR entry's own status to
            # "closed" when GitHub reports the PR closed-unmerged. Without
            # this, entries stuck in active statuses (janitor_blocked,
            # rework_requested, reviewing, escalated, ...) are invisible to
            # every terminal-status sweep and get re-fetched / re-evaluated
            # every pass forever. This is the PR-side counterpart to the
            # issue-side closed_unmerged_pr_active_labels above; the two are
            # independent and may both fire for the same PR. The linked
            # issue's disposition (label strip / carry-forward redispatch)
            # is left entirely to the existing issue-side handling. MERGED
            # PRs are excluded -- that is merged_outside_orchestrator's job.
            state_status = (state_entry or {}).get("status")
            if state_status is not None and state_status not in ("closed", "merged"):
                drift.append(
                    DriftItem(
                        kind="closed_unmerged_pr_state_converged",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"PR #{pr_number} is CLOSED (unmerged) on GitHub but "
                            f"state status is {state_status!r}; converging to 'closed'"
                        ),
                        fix_actions=(f"set state prs[{pr_number}].status = 'closed'",),
                        new_status="closed",
                    )
                )
        elif gh_state == "OPEN":
            # A PR record that the orchestrator is already tracking (has an
            # entry in state["prs"]) but that never got a status written --
            # e.g. a review packet generation crashed between creating the
            # entry and recording its first status -- is invisible to every
            # status-driven selector. Normalize it to the same passive
            # "reviewing" placeholder issues get in the sibling sweep below;
            # never invent a status for a PR the orchestrator never tracked
            # (state_entry is None) since that may not be one of ours.
            if state_entry is not None and state_entry.get("status") is None:
                drift.append(
                    DriftItem(
                        kind="pr_status_normalized",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"PR #{pr_number} is tracked in state but has no status field; "
                            f"normalizing to {PASSIVE_OPEN_STATUS!r}"
                        ),
                        fix_actions=(
                            f"set state prs[{pr_number}].status = {PASSIVE_OPEN_STATUS!r}",
                        ),
                        new_status=PASSIVE_OPEN_STATUS,
                    )
                )

    pr_numbers_on_github = {int(pr["number"]) for pr in prs if pr.get("number") is not None}
    for pr_number_str in state_prs:
        try:
            pr_number = int(pr_number_str)
        except ValueError:
            continue
        if pr_number not in pr_numbers_on_github:
            drift.append(
                DriftItem(
                    kind="state_pr_missing_on_github",
                    issue_number=state_prs[pr_number_str].get("issue_number"),
                    pr_number=pr_number,
                    detail=f"state has prs[{pr_number}] but gh reports no such PR",
                    fix_actions=(f"drop prs[{pr_number}] from state",),
                )
            )

    # Detect dead sessions and classify failures for provider throttle state
    # This must happen AFTER the PR loop (to populate open_prs_by_issue) but BEFORE
    # the issue loop (to populate issues_handled_by_session_relabel for mutual exclusion)
    if repo_root is not None:
        from .claude_code import update_worker_record_with_failure_classification
        from .devin_shell import update_session_record_with_failure_classification
        from .post_mortem import classify_and_record
        from .worker import (
            _log_is_stalled_at_shim,
            iter_workers,
            real_activity_probe_for,
            update_worker_log_stat,
        )

        sessions_dir = repo_root / config.devin.sessions_dir
        # state_dir root (sibling of state.json) for api-budget ledger settlement
        # on reap (issue #480). Resolved through runtime_paths so an absolute
        # state_dir config is honored identically to state.json itself.
        from .paths import runtime_paths

        state_dir_root = runtime_paths(repo_root, config.runtime.state_dir).root
        if sessions_dir.is_dir():
            for w in iter_workers(sessions_dir):
                # Track live sessions to avoid false-positive drift detection (issue #214)
                if w.is_alive():
                    live_session_issue_numbers.add(w.issue_number)

                # Issue #221: detect launch_stalled sessions (alive but hung at shim marker)
                # Issue #280: corroborate against real-session activity before killing.
                if w.error is None and w.is_alive():
                    now = datetime.now(UTC)
                    log_path = Path(w.log_path)
                    probe = real_activity_probe_for(w, config, now)
                    if _log_is_stalled_at_shim(
                        log_path,
                        config.watchdog.launch_stall_grace_minutes,
                        now,
                        real_activity_probe=probe,
                    ):
                        # Session is alive but stalled at shim marker - classify as launch_stalled
                        if w.adapter_kind == "devin":
                            update_session_record_with_failure_classification(
                                sessions_dir,
                                w.issue_number,
                                fallback_kind="launch_stalled",
                                config=config,
                            )
                        elif w.adapter_kind == "claude-code":
                            update_worker_record_with_failure_classification(
                                sessions_dir,
                                w.issue_number,
                                fallback_kind="launch_stalled",
                                config=config,
                            )
                        elif w.adapter_kind == "api":
                            update_worker_record_with_failure_classification(
                                sessions_dir,
                                w.issue_number,
                                fallback_kind="launch_stalled",
                                config=config,
                                adapter_kind="api",
                            )

                        # Kill the process tree to free the slot
                        if w.pid is not None:
                            kill_process_tree(w.pid, w.process_start_time)

                        # Reap the sidecar to prevent phantom sessions
                        w.reap_sidecar(
                            sessions_dir,
                            api_config=config.api_worker,
                            state_dir=state_dir_root,
                        )

                        # Reconcile labels for launch_stalled sessions with no open PR
                        if w.issue_number not in open_prs_by_issue:
                            issue = issues_by_number.get(w.issue_number)
                            if issue:
                                issue_labels = label_names(issue)
                                active_labels = issue_labels & labels_cfg.active
                                # Only relabel to ready when the GitHub issue is still open;
                                # closed issues are finalized by state_active_status_issue_closed.
                                if active_labels and _issue_state(issue) == "OPEN":
                                    # Remove all active labels and ensure ready label is present
                                    fix_actions = [
                                        f"remove label '{label}' from issue #{w.issue_number}"
                                        for label in sorted(active_labels)
                                    ]
                                    add_labels: tuple[str, ...] = ()
                                    if labels_cfg.ready not in issue_labels:
                                        fix_actions.append(
                                            f"add label '{labels_cfg.ready}' to issue #{w.issue_number}"
                                        )
                                        add_labels = (labels_cfg.ready,)

                                    activity_payload = probe.to_payload()
                                    drift.append(
                                        DriftItem(
                                            kind="session_failed_relabeled",
                                            issue_number=w.issue_number,
                                            pr_number=None,
                                            detail=(
                                                f"issue #{w.issue_number} session launch_stalled "
                                                f"(hung at shim marker), activity_sources={json.dumps(activity_payload)}, "
                                                f"no open PR, reconciling labels from {sorted(active_labels)} to dispatchable"
                                            ),
                                            fix_actions=tuple(fix_actions),
                                            remove_labels=tuple(sorted(active_labels)),
                                            add_labels=add_labels,
                                        )
                                    )
                                    # Mark this issue as handled to avoid double-emission
                                    issues_handled_by_session_relabel.add(w.issue_number)

                if w.error is None and not w.is_alive():
                    # Update log stat fields for progress tracking (final update before classification)
                    update_worker_log_stat(sessions_dir, w)

                    # Issue #252: inspect the worktree before deciding how to classify.
                    # This is the single enforcement point shared with workflow.py.
                    worktree_path = Path(w.worktree_path)
                    inspection = inspect_worktree_state(
                        worktree_path,
                        config.dispatch.base_ref,
                        config.dispatch.injected_paths,
                        config.dispatch.materialize_dirs,
                    )
                    is_completed = inspection.state == WorktreeState.COMPLETED

                    # Issue #261: post-mortem extraction is intertwined with log-tail
                    # classification. For a completed-but-unpublished worktree, we want
                    # failure_kind to be "unpublished_work" even if the terminal log
                    # tail would otherwise look like a tool-rejection (worker_blocked).
                    # Call update_* first when completed, then run classify_and_record
                    # for diagnostics. For non-completed sessions, preserve the original
                    # ordering so worker_blocked still escalates.
                    if is_completed:
                        if w.adapter_kind == "devin":
                            failure_kind, throttled_until = (
                                update_session_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind="unpublished_work",
                                    config=config,
                                )
                            )
                        elif w.adapter_kind == "claude-code":
                            failure_kind, throttled_until = (
                                update_worker_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind="unpublished_work",
                                    config=config,
                                )
                            )
                        elif w.adapter_kind == "api":
                            failure_kind, throttled_until = (
                                update_worker_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind="unpublished_work",
                                    config=config,
                                    adapter_kind="api",
                                )
                            )
                        else:
                            failure_kind, throttled_until = None, None
                        # Diagnostic post-mortem; its worker_blocked verdict is ignored
                        # because the worktree itself proves the work was completed.
                        classify_and_record(sessions_dir, config, w, now=datetime.now(UTC))
                    else:
                        classify_and_record(sessions_dir, config, w, now=datetime.now(UTC))
                        fallback_kind = (
                            "stalled" if inspection.state != WorktreeState.UNKNOWN else None
                        )
                        if w.adapter_kind == "devin":
                            failure_kind, throttled_until = (
                                update_session_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind=fallback_kind,
                                    config=config,
                                )
                            )
                        elif w.adapter_kind == "claude-code":
                            failure_kind, throttled_until = (
                                update_worker_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind=fallback_kind,
                                    config=config,
                                )
                            )
                        elif w.adapter_kind == "api":
                            failure_kind, throttled_until = (
                                update_worker_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind=fallback_kind,
                                    config=config,
                                    adapter_kind="api",
                                )
                            )
                        else:
                            failure_kind, throttled_until = None, None

                    if failure_kind and throttled_until:
                        # Update state with throttle window
                        # This is a no-op drift item that just signals state update
                        drift.append(
                            DriftItem(
                                kind="provider_throttle_detected",
                                issue_number=w.issue_number,
                                pr_number=None,
                                detail=(
                                    f"issue #{w.issue_number} session died with "
                                    f"{failure_kind}, throttling until {throttled_until}"
                                ),
                                fix_actions=(f"set throttled_until={throttled_until}",),
                            )
                        )

                    # Reap the sidecar to prevent phantom sessions from PID recycling (issue #113)
                    # Delete the sidecar file after the session is detected as dead and classified
                    w.reap_sidecar(
                        sessions_dir,
                        api_config=config.api_worker,
                        state_dir=state_dir_root,
                    )

                    # Issue #118: reconcile labels for dead sessions with no open PR
                    # A dead worker with no open PR is recoverable and should be relabeled
                    # as dispatchable (remove active labels, ensure ready label present)
                    # Only count OPEN PRs (not CLOSED/MERGED) for the guard
                    if w.issue_number not in open_prs_by_issue:
                        issue = issues_by_number.get(w.issue_number)
                        # Only relabel salvage/escalate open issues. Closed issues with
                        # active state-machine status are finalized by state_active_status_issue_closed.
                        if issue and _issue_state(issue) == "OPEN":
                            issue_labels = label_names(issue)
                            active_labels = issue_labels & labels_cfg.active
                            if (
                                active_labels
                                and failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                            ):
                                drift.append(
                                    DriftItem(
                                        kind="session_failed_escalated",
                                        issue_number=w.issue_number,
                                        pr_number=None,
                                        detail=(
                                            f"issue #{w.issue_number} session died with "
                                            f"deterministic failure ({failure_kind}), no open PR; "
                                            f"suppressing relabel-to-ready, escalating instead"
                                        ),
                                        fix_actions=(
                                            f"transition issue #{w.issue_number} labels via "
                                            "'redispatch_escalated' event",
                                        ),
                                    )
                                )
                                # Mark this issue as handled to avoid double-emission with
                                # issue_active_label_no_open_pr (both fire for dead-session-with-no-PR-ever)
                                issues_handled_by_session_relabel.add(w.issue_number)
                            elif active_labels and is_completed:
                                # Issue #252: completed-but-unpublished work takes the salvage
                                # path (push + PR) instead of re-dispatching.
                                base_branch = None
                                if inspection.resolved_base_ref:
                                    base_branch = resolve_base_branch_name(
                                        repo_root, inspection.resolved_base_ref
                                    )
                                fix_actions = [
                                    f"push branch '{w.branch}' to origin",
                                    f"create PR for issue #{w.issue_number} from branch '{w.branch}'",
                                ]
                                add_labels = (labels_cfg.pr_open,)
                                for label in sorted(active_labels):
                                    fix_actions.append(
                                        f"remove label '{label}' from issue #{w.issue_number}"
                                    )
                                fix_actions.append(
                                    f"add label '{labels_cfg.pr_open}' to issue #{w.issue_number}"
                                )
                                drift.append(
                                    DriftItem(
                                        kind="session_unpublished_work_salvaged",
                                        issue_number=w.issue_number,
                                        pr_number=None,
                                        detail=(
                                            f"issue #{w.issue_number} session has a clean worktree "
                                            f"with {inspection.ahead_count} unpushed commit(s); "
                                            f"salvaging by pushing branch '{w.branch}' and opening a PR"
                                        ),
                                        fix_actions=tuple(fix_actions),
                                        remove_labels=tuple(sorted(active_labels)),
                                        add_labels=add_labels,
                                        branch=w.branch,
                                        base_branch=base_branch,
                                    )
                                )
                                # Mark this issue as handled to avoid double-emission with
                                # issue_active_label_no_open_pr (both fire for dead-session-with-no-PR-ever)
                                issues_handled_by_session_relabel.add(w.issue_number)
                            elif active_labels:
                                # Remove all active labels and ensure ready label is present
                                fix_actions = [
                                    f"remove label '{label}' from issue #{w.issue_number}"
                                    for label in sorted(active_labels)
                                ]
                                add_labels: tuple[str, ...] = ()
                                if labels_cfg.ready not in issue_labels:
                                    fix_actions.append(
                                        f"add label '{labels_cfg.ready}' to issue #{w.issue_number}"
                                    )
                                    add_labels = (labels_cfg.ready,)

                                drift.append(
                                    DriftItem(
                                        kind="session_failed_relabeled",
                                        issue_number=w.issue_number,
                                        pr_number=None,
                                        detail=(
                                            f"issue #{w.issue_number} session died with "
                                            f"{failure_kind or 'unknown failure'}, no open PR, "
                                            f"reconciling labels from {sorted(active_labels)} to dispatchable"
                                        ),
                                        fix_actions=tuple(fix_actions),
                                        remove_labels=tuple(sorted(active_labels)),
                                        add_labels=add_labels,
                                    )
                                )
                                # Mark this issue as handled to avoid double-emission with
                                # issue_active_label_no_open_pr (both fire for dead-session-with-no-PR-ever)
                                issues_handled_by_session_relabel.add(w.issue_number)

    for issue in issues:
        issue_number = int(issue["number"])
        issue_labels = label_names(issue)
        active_present = issue_labels & labels_cfg.active
        terminal_present = issue_labels & labels_cfg.terminal

        # Escalation is terminal-until-human, and state.json is its ground
        # truth: every escalation call site writes status="escalated" first,
        # then applies the human_needed label as a separate step, so a crash
        # or label-API failure between the two leaves an escalated issue
        # with no workflow labels at all -- exactly the zero-label shape the
        # open-PR self-heal below would otherwise silently re-arm (and the
        # no-open-PR repair would relabel dispatchable). Converge the labels
        # from state instead, and never self-heal an escalated issue; only
        # `charlie unescalate` re-enters the machine.
        tracked_entry = state.get("issues", {}).get(str(issue_number))
        tracked_status = tracked_entry.get("status") if isinstance(tracked_entry, dict) else None
        if tracked_status == "escalated" and _issue_state(issue) == "OPEN":
            needs_human_needed = labels_cfg.human_needed not in issue_labels
            if needs_human_needed or active_present:
                fix_actions = []
                add_labels: tuple[str, ...] = ()
                if needs_human_needed:
                    fix_actions.append(
                        f"add label '{labels_cfg.human_needed}' to issue #{issue_number}"
                    )
                    add_labels = (labels_cfg.human_needed,)
                for label in sorted(active_present):
                    fix_actions.append(f"remove label '{label}' from issue #{issue_number}")
                drift.append(
                    DriftItem(
                        kind="escalated_labels_converged",
                        issue_number=issue_number,
                        pr_number=None,
                        detail=(
                            f"issue #{issue_number} has status 'escalated' in state but "
                            f"its labels {sorted(issue_labels)} do not reflect it; "
                            "converging labels from state ground truth"
                        ),
                        fix_actions=tuple(fix_actions),
                        remove_labels=tuple(sorted(active_present)),
                        add_labels=add_labels,
                    )
                )
            continue

        # Skip issue_active_label_no_open_pr if already handled by session relabel
        # (both fire for dead-session-with-no-PR-ever scenario) or if the session
        # is still alive (issue #214: don't strip labels from running workers).
        # Terminal-labeled issues are excluded: a repair pass must never make a
        # human-needed/done issue dispatchable again by adding `ready` back.
        if (
            active_present
            and not terminal_present
            and not prs_linking_issue.get(issue_number)
            and issue_number not in issues_handled_by_session_relabel
            and issue_number not in live_session_issue_numbers
            and _issue_state(issue) == "OPEN"
        ):
            # Issue #417: this is the only re-entrant, sidecar-independent
            # path capable of ever revisiting a session that was already
            # reaped before this drift check ran. Removing the stale active
            # label without also ensuring `ready` is present left the issue
            # with no dispatch-eligible label at all -- fixed here to mirror
            # the sibling `session_failed_relabeled` kind below.
            needs_ready = labels_cfg.ready not in issue_labels
            fix_actions = [
                f"remove label '{label}' from issue #{issue_number}"
                for label in sorted(active_present)
            ]
            add_labels: tuple[str, ...] = ()
            if needs_ready:
                fix_actions.append(f"add label '{labels_cfg.ready}' to issue #{issue_number}")
                add_labels = (labels_cfg.ready,)
            drift.append(
                DriftItem(
                    kind="issue_active_label_no_open_pr",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} carries active labels "
                        f"{sorted(active_present)} but no PR links to it"
                    ),
                    fix_actions=tuple(fix_actions),
                    remove_labels=tuple(sorted(active_present)),
                    add_labels=add_labels,
                )
            )

        # Issue #515 (generalized): an issue with an OPEN PR already linked to
        # it is self-healable whenever its active labels don't already read
        # exactly "pr_open" -- either because it carries a stale active label
        # (needs_rework, in_progress, queued) left over from a failed rework
        # loop, OR because it carries NO active label at all (e.g. a bare
        # "automated-ready" label survives untouched while the PR that was
        # actually opened for it goes unnoticed). The original gate only
        # checked the first case (`stale_active` truthy); an issue with zero
        # active labels made `stale_active` an empty (falsy) set and was
        # invisible to this self-heal entirely, even though `needs_pr_open`
        # was independently true. Both sub-cases move the labels to pr_open so
        # the normal review/merge lifecycle takes over instead of looping
        # through failed rework dispatches or sitting dispatch-invisible.
        open_prs = open_prs_by_issue.get(issue_number, [])
        if open_prs:
            stale_active = active_present - {labels_cfg.pr_open, labels_cfg.reviewing}
            needs_pr_open = labels_cfg.pr_open not in issue_labels
            if (
                (stale_active or needs_pr_open)
                and not terminal_present
                and issue_number not in issues_handled_by_session_relabel
                and issue_number not in live_session_issue_numbers
                and _issue_state(issue) == "OPEN"
            ):
                pr_number = min(int(pr["number"]) for pr in open_prs)
                fix_actions = [
                    f"remove label '{label}' from issue #{issue_number}"
                    for label in sorted(stale_active)
                ]
                add_labels: tuple[str, ...] = ()
                if needs_pr_open:
                    fix_actions.append(
                        f"add label '{labels_cfg.pr_open}' to issue #{issue_number}"
                    )
                    add_labels = (labels_cfg.pr_open,)
                fix_actions.append(
                    f"set state issues[{issue_number}].status = {PASSIVE_OPEN_STATUS!r}"
                )
                drift.append(
                    DriftItem(
                        kind="issue_active_label_with_open_pr",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"issue #{issue_number} carries stale active labels "
                            f"{sorted(stale_active)} while open PR #{pr_number} links to it"
                            if stale_active
                            else (
                                f"issue #{issue_number} has no active agent label while "
                                f"open PR #{pr_number} links to it"
                            )
                        ),
                        fix_actions=tuple(fix_actions),
                        remove_labels=tuple(sorted(stale_active)),
                        add_labels=add_labels,
                        new_status=PASSIVE_OPEN_STATUS,
                    )
                )
                issues_status_repaired.add(issue_number)

        if terminal_present and active_present:
            drift.append(
                DriftItem(
                    kind="done_label_with_active_labels",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} has terminal labels {sorted(terminal_present)} "
                        f"and active labels {sorted(active_present)} simultaneously"
                    ),
                    fix_actions=tuple(
                        f"remove label '{label}' from issue #{issue_number}"
                        for label in sorted(active_present)
                    ),
                    remove_labels=tuple(sorted(active_present)),
                )
            )

    # Detect stale dispatch_pending claims (crashed phase-2)
    state_issues: dict[str, Any] = state.get("issues", {})
    for issue_number_str, entry in state_issues.items():
        if not isinstance(entry, dict):
            continue
        try:
            issue_number = int(issue_number_str)
        except ValueError:
            continue
        status = entry.get("status")
        if status == "dispatch_pending" and is_claim_stale(entry.get("dispatch_pending_at")):
            drift.append(
                DriftItem(
                    kind="stale_dispatch_pending_claim",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} has a stale dispatch_pending claim "
                        f"(crashed phase-2) and should be re-dispatchable"
                    ),
                    fix_actions=(f"clear dispatch_pending claim for issue #{issue_number}",),
                )
            )

    # Issue #259: a state entry with an active status but a closed GitHub issue
    # means the orchestrator lost the lifecycle edge (e.g. a crash before the
    # worker saw the issue close). Finalize the state and strip any labels that
    # still look active. This sweep depends on a complete issue snapshot, so it is
    # skipped when the issue list is provably truncated (issue #259 review).
    if not issue_snapshot_truncated:
        for issue_number_str, entry in state_issues.items():
            if not isinstance(entry, dict):
                continue
            try:
                issue_number = int(issue_number_str)
            except ValueError:
                continue
            status = entry.get("status")
            if status not in ACTIVE_STATE_STATUSES:
                continue
            issue = issues_by_number.get(issue_number)
            if issue is None or _issue_state(issue) != "CLOSED":
                continue
            issue_labels = label_names(issue)
            active_labels = issue_labels & labels_cfg.active
            terminal_present = issue_labels & labels_cfg.terminal
            # If a terminal+active contradiction is already being repaired, let that
            # drift item remove the active labels; this item finalizes state status.
            remove_labels = () if terminal_present else tuple(sorted(active_labels))
            fix_actions = [f"set state issues[{issue_number}].status = 'closed'"]
            for label in remove_labels:
                fix_actions.append(f"remove label '{label}' from issue #{issue_number}")
            drift.append(
                DriftItem(
                    kind="state_active_status_issue_closed",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} is CLOSED on GitHub but state status is {status!r}"
                    ),
                    fix_actions=tuple(fix_actions),
                    remove_labels=remove_labels,
                )
            )

        # A status value that no code path in workflow.py ever assigns (e.g.
        # "ready", which is only ever a label default, never a status) is
        # invisible to every status-driven selector: it doesn't read as
        # dispatched/rework_requested (so it isn't mistaken for in-flight
        # work) but it also doesn't read as escalated/closed (so
        # state_active_status_issue_closed above never finalizes it either).
        # A record with no "status" key at all falls in the same blind spot.
        # Recompute from ground truth: closed on GitHub wins first, then an
        # open tracked PR (the same passive placeholder issue_active_label_
        # with_open_pr uses above), else the queued-equivalent baseline a
        # never-dispatched issue naturally has -- which means simply having
        # no status key, not a synthesized status string. Never writes
        # "rework_requested": that would trigger a fresh worker dispatch
        # purely from a repair pass. Only emits when the recomputed target
        # actually differs from the current value, so a second pass over an
        # already-normalized record (including the "no status key" baseline)
        # is a no-op.
        for issue_number_str, entry in state_issues.items():
            if not isinstance(entry, dict):
                continue
            try:
                issue_number = int(issue_number_str)
            except ValueError:
                continue
            if issue_number in issues_status_repaired:
                continue  # already normalized by issue_active_label_with_open_pr above
            current_status = entry.get("status")
            if current_status in VALID_ISSUE_STATUSES:
                continue
            issue = issues_by_number.get(issue_number)
            if issue is not None and _issue_state(issue) == "CLOSED":
                target_status: str | None = "closed"
            elif open_prs_by_issue.get(issue_number):
                target_status = PASSIVE_OPEN_STATUS
            else:
                target_status = None
            if target_status == current_status:
                continue
            drift.append(
                DriftItem(
                    kind="issue_status_normalized",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} has status {current_status!r}, which no "
                        f"code path in the orchestrator ever assigns; normalizing to "
                        f"{target_status!r}"
                    ),
                    fix_actions=(
                        f"set state issues[{issue_number}].status = {target_status!r} "
                        f"(was {current_status!r})",
                    ),
                    new_status=target_status,
                )
            )

    # Issue #15 / issue #259: if the issue snapshot hit the page limit, it is
    # provably incomplete. Emit a loud warning and refuse to act on completeness-
    # dependent sweeps for this pass. Full pagination is issue #45's scope.
    if issue_snapshot_truncated:
        logger.warning(
            "Issue snapshot is truncated at the page limit (%d); skipping "
            "state_active_status_issue_closed finalization sweep for this pass",
            _LIST_LIMIT,
        )
        drift.append(
            DriftItem(
                kind="snapshot_truncated",
                issue_number=None,
                pr_number=None,
                detail=(
                    f"issue snapshot returned {_LIST_LIMIT} issues, matching the page limit; "
                    "snapshot may be incomplete"
                ),
                fix_actions=(
                    "skip completeness-dependent sweeps for this pass",
                    "full pagination is tracked in issue #45",
                ),
            )
        )

    return drift


def apply_fixes(
    gh: GitHub,
    state: dict[str, Any],
    drift: list[DriftItem],
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Apply the structured fixes for each drift item and return a NEW state dict.

    ``state`` (and its nested ``issues``/``prs`` dicts) are never mutated in
    place — every touched entry is replaced via ``{**existing, ...}``.

    If ``repo_root`` is provided, merged/closed PR drift items also tear down
    the isolated ``reviews_dir`` checkout and clear any ``review_dispatch_*``
    state so the closed lifecycle cannot be mistaken for a live claim. A PR
    whose reviewer process is still alive is deferred to a later pass (issue
    #504) so the live session is not interrupted.
    """
    new_issues: dict[str, Any] = dict(state.get("issues", {}))
    new_prs: dict[str, Any] = dict(state.get("prs", {}))
    new_state: dict[str, Any] = {**state, "issues": new_issues, "prs": new_prs}

    alive_pr_numbers: set[int] = set()
    if repo_root is not None:
        reviews_dir = repo_root / config.review_dispatch.reviews_dir
        from .worker import _alive_review_worker_issue_numbers

        alive_pr_numbers = _alive_review_worker_issue_numbers(reviews_dir)

    for item in drift:
        if item.kind == "merged_outside_orchestrator":
            # Issue #504: defer if the reviewer process is still running.
            if item.pr_number is not None and item.pr_number in alive_pr_numbers:
                continue
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                existing_pr = new_prs.get(pr_key, {})
                new_pr_state = {
                    **without_review_dispatch_claim(existing_pr),
                    "number": item.pr_number,
                    "status": "merged",
                }
                if item.issue_number is not None:
                    new_pr_state["issue_number"] = item.issue_number
                new_prs[pr_key] = new_pr_state

                checkout_removed = False
                if repo_root is not None:
                    reviews_dir = repo_root / config.review_dispatch.reviews_dir
                    checkout_removed = remove_review_checkout(
                        repo_root, item.pr_number, reviews_dir=reviews_dir
                    )
                checkout_action = (
                    (
                        f"remove review checkout for PR #{item.pr_number}: "
                        f"{'ok' if checkout_removed else 'failed'}"
                    )
                    if repo_root is not None
                    else (
                        f"remove review checkout for PR #{item.pr_number}: skipped (no repo_root)"
                    )
                )
                fix_actions = list(item.fix_actions) + [
                    checkout_action,
                    f"clear review-dispatch fields for prs[{item.pr_number}]",
                ]
                item = DriftItem(
                    kind=item.kind,
                    issue_number=item.issue_number,
                    pr_number=item.pr_number,
                    detail=item.detail,
                    fix_actions=tuple(fix_actions),
                    remove_labels=item.remove_labels,
                    add_labels=item.add_labels,
                )
            if item.issue_number is not None:
                result = transition(gh, config.labels, item.issue_number, "merged")
                # Record transition outcome in the event
                fix_actions = list(item.fix_actions)
                if result.outcome != TransitionOutcome.APPLIED:
                    fix_actions.append(
                        f"transition outcome: {result.outcome.value}, "
                        f"add_failures: {result.add_failures}, "
                        f"remove_failures: {result.remove_failures}"
                    )
                    # Replace item with updated fix_actions for event emission
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "closed_unmerged_pr_active_labels":
            # Issue #504: defer if the reviewer process is still running.
            if item.pr_number is not None and item.pr_number in alive_pr_numbers:
                continue
            checkout_removed = False
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                if pr_key in new_prs:
                    new_prs[pr_key] = without_review_dispatch_claim(new_prs[pr_key])
                if repo_root is not None:
                    reviews_dir = repo_root / config.review_dispatch.reviews_dir
                    checkout_removed = remove_review_checkout(
                        repo_root, item.pr_number, reviews_dir=reviews_dir
                    )
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                # Record label-write failures in the event
                fix_actions = list(item.fix_actions)
                if item.pr_number is not None:
                    checkout_action = (
                        (
                            f"remove review checkout for PR #{item.pr_number}: "
                            f"{'ok' if checkout_removed else 'failed'}"
                        )
                        if repo_root is not None
                        else (
                            f"remove review checkout for PR #{item.pr_number}: skipped (no repo_root)"
                        )
                    )
                    fix_actions.extend(
                        [
                            checkout_action,
                            f"clear review-dispatch fields for prs[{item.pr_number}]",
                        ]
                    )
                if not label_ok:
                    fix_actions.append("label_write_failed: true")
                if fix_actions != list(item.fix_actions):
                    # Replace item with updated fix_actions for event emission
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "state_pr_missing_on_github":
            if item.pr_number is not None:
                new_prs.pop(str(item.pr_number), None)

        elif item.kind == "closed_unmerged_pr_state_converged":
            # Issue #558: converge the state PR entry's own status to
            # "closed" when GitHub reports the PR closed-unmerged. Only the
            # PR entry's status is touched; the linked issue's disposition
            # is left to the existing closed-unmerged issue-side handling
            # (closed_unmerged_pr_active_labels / state_active_status_issue_
            # closed). Any stale review-dispatch claim is also cleared so a
            # dead PR never retains a live-claim-shaped record.
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                existing_pr = new_prs.get(pr_key, {})
                new_prs[pr_key] = {
                    **without_review_dispatch_claim(existing_pr),
                    "number": item.pr_number,
                    "status": "closed",
                }
                if item.issue_number is not None:
                    new_prs[pr_key]["issue_number"] = item.issue_number

        elif item.kind in (
            "issue_active_label_no_open_pr",
            "done_label_with_active_labels",
            "escalated_labels_converged",
        ):
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                # Issue #417: issue_active_label_no_open_pr now carries
                # add_labels=(ready,) when the ready label is missing;
                # escalated_labels_converged carries add_labels=(human_needed,)
                # when the escalation label never landed --
                # done_label_with_active_labels never sets add_labels, so this
                # loop is a no-op for that sibling kind.
                for label in item.add_labels:
                    if not gh.add_issue_label(item.issue_number, label):
                        label_ok = False
                # Record label-write failures in the event
                fix_actions = list(item.fix_actions)
                if not label_ok:
                    fix_actions.append("label_write_failed: true")
                    # Replace item with updated fix_actions for event emission
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "issue_active_label_with_open_pr":
            # Issue #515 (generalized): repair the stale/missing active label
            # and mirror the fix into state as the passive "reviewing"
            # placeholder -- the same status the normal dispatch->pr-open flow
            # writes once a PR is open and no verdict has landed -- so
            # state-driven dispatch_rework stops selecting the issue without
            # falsely implying a review verdict was actually recorded (the
            # previous "approved" write here was itself wrong: no reviewer
            # ever ran).
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                for label in item.add_labels:
                    if not gh.add_issue_label(item.issue_number, label):
                        label_ok = False

                issue_key = str(item.issue_number)
                existing_issue = new_issues.get(issue_key, {})
                new_issue = {
                    **existing_issue,
                    "number": item.issue_number,
                    "status": item.new_status or PASSIVE_OPEN_STATUS,
                    "merge_alert": "OK",
                }
                new_issue.pop("worker_pid", None)
                new_issue.pop("worker_process_start_time", None)
                new_issues[issue_key] = new_issue

                fix_actions = list(item.fix_actions)
                if not label_ok:
                    fix_actions.append("label_write_failed: true")
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "stale_dispatch_pending_claim":
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                # Clear the stale claim by removing the entry entirely
                new_issues.pop(issue_key, None)

        elif item.kind == "issue_status_normalized":
            # A status outside VALID_ISSUE_STATUSES (or missing entirely) is
            # recomputed from ground truth in detect_drift and carried here
            # via item.new_status. None means "no status" (the baseline a
            # never-dispatched issue naturally has) -- drop the key rather
            # than write a synthesized placeholder string.
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                existing_issue = new_issues.get(issue_key, {})
                if item.new_status is None:
                    new_issues[issue_key] = {
                        k: v for k, v in existing_issue.items() if k != "status"
                    }
                else:
                    new_issues[issue_key] = {**existing_issue, "status": item.new_status}

        elif item.kind == "pr_status_normalized":
            # A tracked PR record with no status field is normalized to the
            # passive "reviewing" placeholder (item.new_status).
            if item.pr_number is not None and item.new_status is not None:
                pr_key = str(item.pr_number)
                existing_pr = new_prs.get(pr_key, {})
                new_prs[pr_key] = {**existing_pr, "status": item.new_status}

        elif item.kind == "state_active_status_issue_closed":
            # Issue #259: finalize the state entry and strip any active labels that
            # still remain on the closed issue.
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                existing_issue = new_issues.get(issue_key, {})
                new_issues[issue_key] = {**existing_issue, "status": "closed"}
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                fix_actions = list(item.fix_actions)
                if not label_ok:
                    fix_actions.append("label_write_failed: true")
                    # Replace item with updated fix_actions for event emission
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "provider_throttle_detected":
            # Extract throttled_until from fix_actions
            for action in item.fix_actions:
                if action.startswith("set throttled_until="):
                    throttled_until = action.split("=", 1)[1]
                    new_state = set_throttled_until(new_state, throttled_until)
                    break

        elif item.kind == "session_failed_escalated":
            # Issue #261: worker was killed by a push-gate hook — escalate
            # via the same "redispatch_escalated" label edge workflow.py's
            # reaper uses, instead of relabeling to ready.
            if item.issue_number is not None:
                result = transition(gh, config.labels, item.issue_number, "redispatch_escalated")
                fix_actions = list(item.fix_actions)
                if result.outcome != TransitionOutcome.APPLIED:
                    fix_actions.append(
                        f"transition outcome: {result.outcome.value}, "
                        f"add_failures: {result.add_failures}, "
                        f"remove_failures: {result.remove_failures}"
                    )
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        elif item.kind == "session_unpublished_work_salvaged":
            # Issue #252: push the completed branch, create a PR, and move labels to pr_open.
            # If any step fails, fall back to the normal relabel-to-ready path.
            if item.issue_number is not None and item.branch and item.base_branch:
                repo_root = getattr(gh, "repo_root", None)
                salvage_ok = False
                salvage_error = "repo_root not available"
                pr_number = None
                if repo_root is not None:
                    push_ok, push_error = push_branch(repo_root, item.branch)
                    if push_ok:
                        pr_create = getattr(gh, "pr_create", None)
                        if pr_create is not None:
                            pr_number = pr_create(
                                head=item.branch,
                                base=item.base_branch,
                                title=f"Salvaged work for issue #{item.issue_number}",
                                body=(
                                    f"Closes #{item.issue_number}\n\n"
                                    "Salvaged by the orchestrator from a completed-but-unpublished "
                                    "worker worktree."
                                ),
                            )
                        if pr_number is not None:
                            salvage_ok = True
                        else:
                            salvage_error = "gh pr create failed or returned no PR number"
                    else:
                        salvage_error = push_error or "git push failed"

                if salvage_ok:
                    label_ok = True
                    for label in item.remove_labels:
                        if not gh.remove_issue_label(item.issue_number, label):
                            label_ok = False
                    for label in item.add_labels:
                        if not gh.add_issue_label(item.issue_number, label):
                            label_ok = False
                    fix_actions = list(item.fix_actions)
                    if not label_ok:
                        fix_actions.append("label_write_failed: true")
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                        branch=item.branch,
                        base_branch=item.base_branch,
                    )
                else:
                    # Fallback: treat as session_failed_relabeled and add ready label
                    label_ok = True
                    for label in item.remove_labels:
                        if not gh.remove_issue_label(item.issue_number, label):
                            label_ok = False
                    if config.labels.ready not in item.add_labels:
                        if not gh.add_issue_label(item.issue_number, config.labels.ready):
                            label_ok = False
                    fix_actions = list(item.fix_actions)
                    fix_actions.append(f"salvage_failed: {salvage_error}")
                    if not label_ok:
                        fix_actions.append("label_write_failed: true")
                    item = DriftItem(
                        kind="session_failed_relabeled",
                        issue_number=item.issue_number,
                        pr_number=None,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=(config.labels.ready,),
                        branch=item.branch,
                        base_branch=item.base_branch,
                    )

        elif item.kind == "session_failed_relabeled":
            # Issue #118: reconcile labels for dead sessions with no open PR
            if item.issue_number is not None:
                label_ok = True
                # Remove active labels
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                # Add ready label if needed (structured field)
                for label in item.add_labels:
                    if not gh.add_issue_label(item.issue_number, label):
                        label_ok = False
                # Record label-write failures in the event
                fix_actions = list(item.fix_actions)
                if not label_ok:
                    fix_actions.append("label_write_failed: true")
                    # Replace item with updated fix_actions for event emission
                    item = DriftItem(
                        kind=item.kind,
                        issue_number=item.issue_number,
                        pr_number=item.pr_number,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        new_state = append_event(
            new_state,
            "reconcile",
            {
                "kind": item.kind,
                "issue_number": item.issue_number,
                "pr_number": item.pr_number,
                "fix_actions": list(item.fix_actions),
                "detail": item.detail,
            },
        )

    return new_state
