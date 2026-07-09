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

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import (
    GitHub,
    _LIST_LIMIT,
    RECONCILE_ISSUE_FIELDS,
    RECONCILE_PR_FIELDS,
    label_names,
    linked_issue_number,
)
from .labels import TransitionOutcome, transition
from .process_utils import kill_process_tree
from .state import append_event, is_claim_stale, set_throttled_until


@dataclass(frozen=True)
class DriftItem:
    kind: str
    issue_number: int | None
    pr_number: int | None
    detail: str
    fix_actions: tuple[str, ...]
    remove_labels: tuple[str, ...] = ()
    add_labels: tuple[str, ...] = ()


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
            "open",
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

    Issues exactly two ``gh.run`` list queries (all PRs, open issues) and
    performs every drift check against those two in-memory snapshots — no
    per-item ``gh`` calls.

    If ``repo_root`` is provided, also checks for dead sessions and classifies
    their failures to update the provider throttle state.
    """
    labels_cfg = config.labels
    prs = _fetch_prs(gh)
    issues = _fetch_issues(gh)
    issues_by_number = {int(issue["number"]): issue for issue in issues if issue.get("number")}
    state_prs: dict[str, Any] = state.get("prs", {})

    drift: list[DriftItem] = []
    prs_linking_issue: dict[int, list[dict[str, Any]]] = {}
    open_prs_by_issue: dict[int, list[dict[str, Any]]] = {}
    # Track issues already handled by session relabel to avoid double-emission
    # with issue_active_label_no_open_pr (both fire for dead-session-with-no-PR-ever)
    issues_handled_by_session_relabel: set[int] = set()

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
        from .worker import _log_is_stalled_at_shim, iter_workers, update_worker_log_stat

        sessions_dir = repo_root / config.devin.sessions_dir
        if sessions_dir.is_dir():
            for w in iter_workers(sessions_dir):
                # Issue #221: detect launch_stalled sessions (alive but hung at shim marker)
                # This check runs before the dead session check to catch zombies
                if w.error is None and w.is_alive():
                    now = datetime.now(UTC)
                    log_path = Path(w.log_path)
                    if _log_is_stalled_at_shim(
                        log_path, config.watchdog.launch_stall_grace_minutes, now
                    ):
                        # Session is alive but stalled at shim marker - classify as launch_stalled
                        if w.adapter_kind == "devin":
                            update_session_record_with_failure_classification(
                                sessions_dir, w.issue_number, failure_kind="launch_stalled"
                            )
                        elif w.adapter_kind == "claude-code":
                            update_worker_record_with_failure_classification(
                                sessions_dir, w.issue_number, failure_kind="launch_stalled"
                            )

                        # Kill the process tree to free the slot
                        if w.pid is not None:
                            kill_process_tree(w.pid, w.process_start_time)

                        # Reap the sidecar to prevent phantom sessions
                        w.reap_sidecar(sessions_dir)

                        # Reconcile labels for launch_stalled sessions with no open PR
                        if w.issue_number not in open_prs_by_issue:
                            issue = issues_by_number.get(w.issue_number)
                            if issue:
                                issue_labels = label_names(issue)
                                active_labels = issue_labels & labels_cfg.active
                                if active_labels:
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
                                                f"issue #{w.issue_number} session launch_stalled "
                                                f"(hung at shim marker), no open PR, "
                                                f"reconciling labels from {sorted(active_labels)} to dispatchable"
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

                    # Session exited without error - classify the failure (adapter-specific dispatch)
                    if w.adapter_kind == "devin":
                        failure_kind, throttled_until = (
                            update_session_record_with_failure_classification(
                                sessions_dir, w.issue_number
                            )
                        )
                    elif w.adapter_kind == "claude-code":
                        failure_kind, throttled_until = (
                            update_worker_record_with_failure_classification(
                                sessions_dir, w.issue_number
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
                    w.reap_sidecar(sessions_dir)

                    # Issue #118: reconcile labels for dead sessions with no open PR
                    # A dead worker with no open PR is recoverable and should be relabeled
                    # as dispatchable (remove active labels, ensure ready label present)
                    # Only count OPEN PRs (not CLOSED/MERGED) for the guard
                    if w.issue_number not in open_prs_by_issue:
                        issue = issues_by_number.get(w.issue_number)
                        if issue:
                            issue_labels = label_names(issue)
                            active_labels = issue_labels & labels_cfg.active
                            if active_labels:
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

        # Skip issue_active_label_no_open_pr if already handled by session relabel
        # (both fire for dead-session-with-no-PR-ever scenario)
        if (
            active_present
            and not prs_linking_issue.get(issue_number)
            and issue_number not in issues_handled_by_session_relabel
        ):
            drift.append(
                DriftItem(
                    kind="issue_active_label_no_open_pr",
                    issue_number=issue_number,
                    pr_number=None,
                    detail=(
                        f"issue #{issue_number} carries active labels "
                        f"{sorted(active_present)} but no PR links to it"
                    ),
                    fix_actions=tuple(
                        f"remove label '{label}' from issue #{issue_number}"
                        for label in sorted(active_present)
                    ),
                    remove_labels=tuple(sorted(active_present)),
                )
            )

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

    return drift


def apply_fixes(
    gh: GitHub, state: dict[str, Any], drift: list[DriftItem], config: OrchestratorConfig
) -> dict[str, Any]:
    """Apply the structured fixes for each drift item and return a NEW state dict.

    ``state`` (and its nested ``issues``/``prs`` dicts) are never mutated in
    place — every touched entry is replaced via ``{**existing, ...}``.
    """
    new_issues: dict[str, Any] = dict(state.get("issues", {}))
    new_prs: dict[str, Any] = dict(state.get("prs", {}))
    new_state: dict[str, Any] = {**state, "issues": new_issues, "prs": new_prs}

    for item in drift:
        if item.kind == "merged_outside_orchestrator":
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                new_prs[pr_key] = {**new_prs.get(pr_key, {}), "status": "merged"}
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
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
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

        elif item.kind == "state_pr_missing_on_github":
            if item.pr_number is not None:
                new_prs.pop(str(item.pr_number), None)

        elif item.kind in ("issue_active_label_no_open_pr", "done_label_with_active_labels"):
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
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

        elif item.kind == "stale_dispatch_pending_claim":
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                # Clear the stale claim by removing the entry entirely
                new_issues.pop(issue_key, None)

        elif item.kind == "provider_throttle_detected":
            # Extract throttled_until from fix_actions
            for action in item.fix_actions:
                if action.startswith("set throttled_until="):
                    throttled_until = action.split("=", 1)[1]
                    new_state = set_throttled_until(new_state, throttled_until)
                    break

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
            },
        )

    return new_state
