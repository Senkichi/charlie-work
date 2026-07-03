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
from typing import Any

from .config import OrchestratorConfig
from .github import GitHub, _LIST_LIMIT, label_names, linked_issue_number
from .labels import transition
from .state import append_event, is_claim_stale

_PR_FIELDS = "number,title,url,headRefName,baseRefName,body,state,labels,isCrossRepository"
_ISSUE_FIELDS = "number,title,url,body,labels"


@dataclass(frozen=True)
class DriftItem:
    kind: str
    issue_number: int | None
    pr_number: int | None
    detail: str
    fix_actions: tuple[str, ...]
    remove_labels: tuple[str, ...] = ()


def _fetch_prs(gh: GitHub) -> list[dict[str, Any]]:
    result = gh.run(
        ["pr", "list", "--state", "all", "--limit", str(_LIST_LIMIT), "--json", _PR_FIELDS],
        json_output=True,
    )
    return result if isinstance(result, list) else []


def _fetch_issues(gh: GitHub) -> list[dict[str, Any]]:
    result = gh.run(
        ["issue", "list", "--state", "open", "--limit", str(_LIST_LIMIT), "--json", _ISSUE_FIELDS],
        json_output=True,
    )
    return result if isinstance(result, list) else []


def detect_drift(gh: GitHub, state: dict[str, Any], config: OrchestratorConfig) -> list[DriftItem]:
    """Read-only comparison of GitHub reality against ``state``.

    Issues exactly two ``gh.run`` list queries (all PRs, open issues) and
    performs every drift check against those two in-memory snapshots — no
    per-item ``gh`` calls.
    """
    labels_cfg = config.labels
    prs = _fetch_prs(gh)
    issues = _fetch_issues(gh)
    issues_by_number = {int(issue["number"]): issue for issue in issues if issue.get("number")}
    state_prs: dict[str, Any] = state.get("prs", {})

    drift: list[DriftItem] = []
    prs_linking_issue: dict[int, list[dict[str, Any]]] = {}
    open_prs_by_number: dict[int, dict[str, Any]] = {}
    for pr in prs:
        pr_number = pr.get("number")
        if pr_number is None:
            continue
        pr_number = int(pr_number)
        gh_state = str(pr.get("state") or "").upper()
        if gh_state == "OPEN":
            open_prs_by_number[pr_number] = pr
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        if issue_number is not None:
            prs_linking_issue.setdefault(issue_number, []).append(pr)

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

    for issue in issues:
        issue_number = int(issue["number"])
        issue_labels = label_names(issue)
        active_present = issue_labels & labels_cfg.active
        terminal_present = issue_labels & labels_cfg.terminal

        if active_present and not prs_linking_issue.get(issue_number):
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
                transition(gh, config.labels, item.issue_number, "merged")

        elif item.kind == "closed_unmerged_pr_active_labels":
            if item.issue_number is not None:
                for label in item.remove_labels:
                    gh.remove_issue_label(item.issue_number, label)

        elif item.kind == "state_pr_missing_on_github":
            if item.pr_number is not None:
                new_prs.pop(str(item.pr_number), None)

        elif item.kind in ("issue_active_label_no_open_pr", "done_label_with_active_labels"):
            if item.issue_number is not None:
                for label in item.remove_labels:
                    gh.remove_issue_label(item.issue_number, label)

        elif item.kind == "stale_dispatch_pending_claim":
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                # Clear the stale claim by removing the entry entirely
                new_issues.pop(issue_key, None)

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
