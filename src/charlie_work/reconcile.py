"""Detect and (optionally) fix drift between GitHub reality and local state.

Production incident class this exists for: a human merges a PR by hand (or
closes an issue, or edits labels) outside the orchestrator's own
``merge_ready``/``transition`` codepaths. ``state.json`` and GitHub labels
then permanently disagree with reality — e.g. ``agent:in-progress`` /
``agent:reviewing`` never clears because the label edge that would clear it
(``labels.transition(..., "merged")``) never ran.

``detect_drift`` is read-only: it fetches every PR and every issue via
paginated REST snapshots through ``gh.run``, and never calls a mutating GitHub
method.
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

from .closing_reference import closing_issues_referenced_numbers, validate_closing_reference
from .config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
)
from .escalation import _escalation_edge, _escalation_label, _repair_reason_class
from .github import (
    GitHubError,
    GitHubLike,
    GraphQLBudgetError,
    PR_CLOSING_ISSUES_FIELDS,
    _LIST_LIMIT,
    label_names,
    linked_issue_number,
)
from .instrumentation import log_event, query_events
from .labels import TransitionOutcome, transition
from .paths import resolved_layout, runtime_paths
from .pr_create_retry import create_pr_with_retry
from .process_utils import kill_process_tree
from .review_decision import review_decision as _resolve_review_decision
from .state import (
    DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS,
    ESCALATION_REASON_CLASS_BY_EVENT_KIND,
    ORCHESTRATOR_OWNED_ISSUE_STATUSES,
    PASSIVE_OPEN_STATUS,
    append_event,
    is_claim_stale,
    set_throttled_until,
    utc_now,
    without_review_dispatch_claim,
)
from .worktree import (
    WorktreeState,
    inspect_worktree_state,
    push_branch,
    remove_review_checkout,
    resolve_base_branch_name,
    summarize_branch_work,
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
    # set_throttled_until's reason/adapter_kind for a provider_throttle_detected
    # item -- carried as fields rather than parsed back out of fix_actions so
    # apply_fixes can thread them through exactly as detect_drift computed them.
    # Unused by every other kind.
    throttle_reason: str | None = None
    throttle_adapter_kind: str | None = None
    # Issue #978: structured "why" for session_failed_relabeled drift items,
    # so the machine-readable reason is not buried only in the free-text
    # ``detail`` string. ``reason`` is the canonical path identifier (e.g.
    # "dead_session_no_open_pr"); ``failure_kind`` is the classifier's
    # optional refinement (may be None when classification was inconclusive).
    # Both are threaded into the ``reconcile`` event payload by apply_fixes.
    # Unused by every other kind.
    reason: str | None = None
    failure_kind: str | None = None


# State-machine statuses that mean "this issue is in the orchestrator's pipeline".
# Any of these on a closed GitHub issue is a drift condition (issue #259).
# "escalated" is included so a closed-while-escalated issue gets its state entry
# finalized; the terminal human_needed label is intentionally preserved.
# INVARIANT (guarded by test_fix_reconcile's coverage test): every
# VALID_ISSUE_STATUSES member except the deliberate exclusions below must be
# in this set. The status-normalization sweep below skips anything in
# ORCHESTRATOR_OWNED_ISSUE_STATUSES (VALID_ISSUE_STATUSES minus the
# externally-derived "closed" -- issue #789), so a valid orchestrator-owned
# status missing here is invisible to BOTH sweeps -- a closed issue stuck in
# it would never be finalized (the exact dead zone adding
# manifest_written/dispatch_failed to the valid set briefly created).
# Deliberate exclusions: "closed" (already terminal), "approved"/"blocked"
# (finalization of closed approved/blocked issues is owned by the merged-PR
# finalization flow, pre-existing behavior).
# Issue #1066: membership here does NOT mean uniform treatment across every
# sweep that consumes this set. closed_unmerged_pr_issue_state_converged
# excludes "escalated" via DORMANT_CONVERGENCE_EXCLUDED_STATUSES (defined
# below) because that sweep drops the status key entirely rather than
# converging to a terminal value; "escalated" stays in this set because
# state_active_status_issue_closed and the coverage invariant above still
# need it here.
ACTIVE_STATE_STATUSES: frozenset[str] = frozenset(
    {
        "dispatched",
        "dispatch_pending",
        "manifest_written",
        "dispatch_failed",
        "rework_requested",
        "reviewing",
        # Issue #955: PASSIVE_OPEN_STATUS's own value (distinct from the
        # active "reviewing" above). Both are "still in the open pipeline"
        # -- this set answers "does this need repair convergence?", not "is
        # a reviewer expected?", so the passive placeholder must stay a
        # member or a passively-open entry stops converging when its PR
        # closes unmerged (reconcile.py's closed_unmerged_pr_issue_state_
        # converged) or its issue closes on GitHub (state_active_status_
        # issue_closed).
        "open_passive",
        "escalated",
    }
)

# Issue #1066: the subset of ACTIVE_STATE_STATUSES that is a human-owned
# terminal disposition rather than orchestrator work-in-flight -- currently
# just "escalated". A repair sweep that converges a stale ACTIVE_STATE_STATUSES
# entry to the dormant baseline (dropping the status key entirely, as opposed
# to state_active_status_issue_closed's convergence to the terminal "closed"
# value) must exclude this set: silently dropping "escalated" detaches the
# issue's state from its still-live agent:human-needed label with no repair
# path back into the human queue, since nothing else is watching a
# status-less issue for that label. reconcile.py's own
# ORCHESTRATOR_OWNED_ISSUE_STATUSES import is NOT the right predicate for
# this -- ACTIVE_STATE_STATUSES is entirely a subset of it (both share every
# member except "approved"/"blocked"/"closed", none of which are ever in
# ACTIVE_STATE_STATUSES), so gating on it here would make the affected drift
# kind permanently unreachable rather than narrowly excluding "escalated".
DORMANT_CONVERGENCE_EXCLUDED_STATUSES: frozenset[str] = frozenset({"escalated"})

# Issue #1092: the subset of ACTIVE_STATE_STATUSES for which an active agent
# label on an issue with an open PR is genuinely STALE -- i.e. the orchestrator
# is not currently holding that issue in any lane, so nothing will re-set the
# label and issue_active_label_with_open_pr is free to clear it.
#
# Every other ACTIVE_STATE_STATUSES member CORROBORATES the label: some router
# put the issue in that lane deliberately and will set the label again on its
# next pass. Reverting it there is not a repair, it is one half of a loop --
# reconcile clears the label and resets status to PASSIVE_OPEN_STATUS, the
# router re-sets both, forever, at whatever the reconcile cadence happens to be.
# Measured in job-cannon at a ~29-minute period across 26 issues with zero
# forward progress, because _dispatch_rework_impl selects on
# status == "rework_requested" (workflow.py) and the revert removes the issue
# from that scan entirely.
#
# Derived by SUBTRACTION rather than enumerated, so a status added to
# ACTIVE_STATE_STATUSES later defaults to protected. The asymmetry justifies the
# direction: a wrong "stale" verdict is an infinite loop that starves dispatch,
# while a wrong "corroborated" verdict is one missed self-heal that some other
# sweep or the next status change will pick up.
#
# The two members are here for DIFFERENT reasons; neither generalizes to the
# other:
#
#   "dispatch_failed" -- a rework loop that failed to dispatch is precisely the
#   "left over from a failed rework loop" case the rule was written for (#515).
#   Exempting it would disable the rule's original purpose.
#
#   "open_passive" -- this is the status the rule itself WRITES on repair, so
#   its presence here means "re-fire on an issue this rule already converged".
#   That is reachable and intended: labels can drift after convergence (an
#   external relabel, a partially-applied transition), and repairing labels
#   against an already-correct status is idempotent, not a loop. No router
#   holds an issue in this lane, so nothing re-sets the label afterwards.
#
# "escalated" is deliberately NOT a member, and its absence is load-bearing
# even though it is currently unobservable. Escalated issues never reach this
# predicate at all -- the `tracked_status == "escalated"` branch above
# converges labels from state and `continue`s, and its comment names this rule
# as the thing it is protecting against ("the zero-label shape the open-PR
# self-heal below would otherwise silently re-arm"). Listing it as STALE would
# therefore be inert today but fail OPEN: if that branch ever stopped
# short-circuiting, an escalated issue would get its status silently reset to
# PASSIVE_OPEN_STATUS while `agent:human-needed` stayed live -- the #894
# status/label split-brain that DORMANT_CONVERGENCE_EXCLUDED_STATUSES (above)
# exists to prevent, re-created one rule over. Protected here means this rule
# fails closed if that guard ever moves.
_LABEL_STALE_STATUSES: frozenset[str] = frozenset(
    {
        "dispatch_failed",
        "open_passive",
    }
)
_LABEL_CORROBORATING_STATUSES: frozenset[str] = ACTIVE_STATE_STATUSES - _LABEL_STALE_STATUSES


def _issue_state(issue: dict[str, Any] | None) -> str:
    """Return the upper-cased GitHub state for an issue dict.

    Missing state defaults to OPEN to keep tests/fixtures that pre-date the
    `--state all` issue list (issue #259) behaving as open issues.
    """
    if issue is None:
        return ""
    return str(issue.get("state") or "OPEN").upper()


def _fetch_snapshot(gh: GitHubLike, args: list[str], *, what: str) -> list[dict[str, Any]]:
    """Run a ``gh ... list --json`` query, refusing to degrade a failed read to ``[]``.

    As of issue #756, ``GitHub.run`` itself raises ``GitHubError`` on the
    *success-with-empty-stdout* path (``returncode == 0``, ``allow_failure=False``,
    ``json_output=True``, empty output) rather than returning ``None`` -- so this
    method's own ``isinstance`` check is now defense-in-depth against test
    doubles (``GitHubLike`` fakes) that still return a non-list, plus any future
    ``gh.run`` implementation that reintroduces the ambiguity. Coercing a
    non-list result to ``[]`` here, as both fetchers used to before #742, makes
    "I could not read GitHub" indistinguishable from "GitHub has zero ``what``".

    That distinction is load-bearing. ``detect_drift`` answers "GitHub has zero
    PRs" by flagging every tracked PR ``state_pr_missing_on_github``, and the
    fix handler drops each one out of ``state["prs"]`` -- erasing ``decision``
    and ``reviewed_head_sha`` fleet-wide, so approved PRs read as un-approved.

    It cannot be recovered downstream: ``detect_drift`` receives bit-identical
    inputs (``prs=[]`` plus a non-empty ``state["prs"]``) from a genuinely
    empty GitHub and from a failed fetch, which is exactly why a "suspiciously
    empty" heuristic there is unsound in both directions. The non-list value is
    only visible *here*, so this is the one layer that can preserve it -- after
    which ``[]`` unambiguously means "GitHub has zero ``what``" and the
    downstream sweep is correct by construction rather than by heuristic.

    Raises ``GitHubError`` so existing caller handling applies: the periodic
    in-loop pass records ``reconcile_pass_failed`` and leaves state untouched,
    and the ``reconcile``/``mop-up`` CLI paths surface it through their
    ``except GitHubError`` handlers instead of mutating state on a bad read.
    """
    result = gh.run(args, json_output=True)
    if not isinstance(result, list):
        raise GitHubError(
            f"reconcile: `gh {args[0]} list` returned {type(result).__name__}, not a list; "
            f"refusing to treat an unreadable {what} snapshot as an empty one "
            "(that reading would drop every tracked item from state.json)"
        )
    return result


def _normalize_reconcile_pr(pr: dict[str, Any]) -> dict[str, Any]:
    """Normalize a REST ``pulls`` response to the ``gh pr list --json`` shape.

    Existing ``FakeGitHub`` test doubles already return the normalized
    ``RECONCILE_PR_FIELDS`` shape, so this function is idempotent: if the input
    already carries ``headRefName`` it is returned unchanged.
    """
    if isinstance(pr.get("head"), dict) and pr.get("headRefName") is None:
        head = pr.get("head") or {}
        base = pr.get("base") or {}
        head_repo = (head.get("repo") or {}).get("full_name")
        base_repo = (base.get("repo") or {}).get("full_name")
        is_cross_repository = (
            None if head_repo is None or base_repo is None else head_repo != base_repo
        )

        raw_state = str(pr.get("state") or "open").lower()
        if pr.get("merged") or pr.get("merged_at"):
            state = "MERGED"
        elif raw_state == "closed":
            state = "CLOSED"
        else:
            state = "OPEN"

        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "url": pr.get("html_url") or pr.get("url"),
            "headRefName": head.get("ref"),
            "baseRefName": base.get("ref"),
            "body": pr.get("body"),
            "state": state,
            "labels": pr.get("labels", []),
            "isCrossRepository": is_cross_repository,
            "headRefOid": head.get("sha"),
        }

    return pr


def _fetch_prs(gh: GitHubLike) -> list[dict[str, Any]]:
    """Fetch every PR via the REST ``pulls`` endpoint, paging to exhaustion.

    ``gh pr list --state all --limit 500`` was permanently capped at
    ``_LIST_LIMIT`` (500). Once a repo crossed that limit the PR snapshot was
    always provably incomplete, which made ``detect_drift`` skip the
    ``state_pr_missing_on_github`` sweep on every subsequent pass (issue #762).
    The REST ``pulls`` endpoint is paged and has no hard cap, so we walk it
    until a page comes back with fewer items than requested.
    """
    per_page = 100
    all_prs: list[dict[str, Any]] = []
    page = 1
    while True:
        result = gh.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/pulls?state=all&per_page={per_page}&page={page}",
            ],
            json_output=True,
        )
        if not isinstance(result, list):
            raise GitHubError(
                f"reconcile: `gh api pulls?state=all` returned {type(result).__name__}, not a list; "
                f"refusing to treat an unreadable PR snapshot as an empty one "
                "(that reading would drop every tracked PR from state.json)"
            )
        if not result:
            break
        all_prs.extend(_normalize_reconcile_pr(pr) for pr in result)
        if len(result) < per_page:
            break
        page += 1
    return all_prs


def _repo_slug(gh: GitHubLike) -> str:
    """Return the ``owner/repo`` slug, or ``?`` if the lookup fails.

    This is used only in log records; a lookup failure must not stop the
    reconcile pass, so it falls back to a placeholder.
    """
    try:
        return gh.name_with_owner()
    except Exception:
        return "?"


def _normalize_reconcile_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Normalize a REST ``issues`` response to the ``gh issue list --json`` shape.

    ``gh issue list --json`` names the web URL ``url``; the REST endpoint names
    the same URL ``html_url``. This function is idempotent: if the input
    already carries ``url`` and no ``html_url`` it is returned unchanged.
    """
    if isinstance(issue, dict) and "html_url" in issue:
        normalized = dict(issue)
        normalized["url"] = normalized.pop("html_url")
        return normalized
    return issue


def _fetch_issues(gh: GitHubLike) -> list[dict[str, Any]]:
    """Fetch every issue via the REST ``issues`` endpoint, paging to exhaustion
    and filtering out pull requests.

    ``gh issue list --state all --limit 500`` was permanently capped at
    ``_LIST_LIMIT`` (500). Once a repo crossed that limit the issue snapshot was
    always provably incomplete, so any state-tracked issue outside the most
    recent 500 became permanently unanswerable (issue #762). The REST
    ``issues`` endpoint is paged and has no hard cap, so we walk it until a page
    comes back with fewer items than requested. The endpoint returns both
    issues and PRs; we drop PR-shaped entries so the issue snapshot contains
    only real issues.
    """
    per_page = 100
    all_issues: list[dict[str, Any]] = []
    page = 1
    while True:
        result = gh.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/issues?state=all&per_page={per_page}&page={page}",
            ],
            json_output=True,
        )
        if not isinstance(result, list):
            raise GitHubError(
                f"reconcile: `gh api issues?state=all` returned {type(result).__name__}, not a list; "
                f"refusing to treat an unreadable issue snapshot as an empty one "
                "(that reading would drop every tracked issue from state.json)"
            )
        if not result:
            break
        for issue in result:
            if not isinstance(issue, dict):
                continue
            # The REST ``issues`` endpoint returns both issues and PRs; ``gh
            # issue list`` filters PRs client-side. Skip PR-shaped rows so they
            # do not pollute the issue snapshot and duplicate the PR snapshot.
            if issue.get("pull_request"):
                continue
            all_issues.append(_normalize_reconcile_issue(issue))
        if len(result) < per_page:
            break
        page += 1
    return all_issues


# Aviator (job-cannon/charlie-work's merge-queue bot) owns these strings; they
# are not orchestrator LabelConfig values. Verified live against real Aviator
# check-run output (job-cannon PR #1400, 2026-07-27): conclusion == "failure",
# output.summary starts with "This PR is not ready to merge (currently in
# state blocked): PR has a blocked label, remove to re-queue."
AVIATOR_CHECK_NAME = "aviator/checks"
AVIATOR_BLOCKED_MESSAGE = "PR has a blocked label, remove to re-queue"


def _pr_review_approved_at_head(
    config: OrchestratorConfig, repo_root: Path | None, pr_number: int, head_sha: str
) -> bool:
    """Mirror the single review-decision reader's approval gate (issue #1362).

    Re-adding the Aviator ``mergequeue`` label must never be safer than the
    normal ship_it path, which only queues a PR when its review decision
    resolves to ``decision == "approved"`` at the PR's *current* head.
    Without this check, ``detect_aviator_stale_blocked`` re-queued job-cannon
    PR #1408 (issue #1404) and PR #1392 (issue #1268) for Aviator merge while
    their recorded decisions were ``request_changes``/never-reviewed --
    Aviator then merged both unreviewed once CI was green, since Aviator's
    own admission check knows nothing about ``review-decision.json``.
    Returns ``False`` (fail closed) when *repo_root* is unavailable, no
    decision can be resolved at all, or the resolved decision is stale
    (``reviewed_head_sha`` does not match *head_sha* -- ``review_decision``'s
    own staleness check, not a re-derived comparison here).

    ``detect_mergequeue_not_approved`` (issue #819) reuses this exact
    predicate as its revocation gate: the label is removed whenever this
    returns ``False``, for any of the reasons documented above.
    """
    if repo_root is None:
        return False
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    resolved = _resolve_review_decision(paths.prs / f"pr-{pr_number}", None, head_sha)
    return resolved.decision == "approved" and not resolved.stale


def detect_aviator_stale_blocked(
    gh: GitHubLike, config: OrchestratorConfig, *, repo_root: Path | None = None
) -> list[DriftItem]:
    """Detect PRs stuck behind a stale Aviator ``blocked`` label.

    Aviator sometimes blocks a PR (setting ``blocked`` and stripping
    ``mergequeue``) on a real CI failure, then never re-evaluates once the
    underlying cause clears (a stale branch update, a flaky test passing on
    rerun, ...) -- confirmed live on job-cannon #1387/#1400/#1398/#1392
    (2026-07-27), each stuck for hours with every real CI check green.

    Deliberately NOT folded into ``detect_drift``: that function's contract
    (enforced by ``test_detect_drift_makes_zero_mutating_calls``) is exactly
    two ``gh.run`` list calls and nothing else, specifically to avoid
    repeating issue #361 (an unconditional per-PR GraphQL check-run walk via
    ``statusCheckRollup`` caused 502s). This function instead issues one
    ``commit_check_runs`` REST call per PR ALREADY LABELED ``blocked`` --
    gated on the cheap, already-fetched ``labels`` field first, so the cost
    scales with how many PRs are actually stuck, not with the full open-PR
    count. ``blocked`` is not a common label in steady state.

    Aviator's message lives in a Check Run's ``output.summary`` -- ``gh pr
    checks --json``'s ``description`` field is always empty for App-created
    Check Runs (only legacy Commit Statuses populate it), so
    ``GitHub.pr_checks``/``PR_CHECKS_FIELDS`` cannot see it at all; this is
    the only path that can.

    The stale ``blocked`` label is always cleared once CI is confirmed green
    (Aviator will re-evaluate honestly on its own). The ``mergequeue`` label
    is re-added only when ``_pr_review_approved_at_head`` confirms the PR is
    currently approved at its live head -- otherwise this function would
    reintroduce the exact worker-self-merge-without-review gap issue #502's
    unauthorized-merge tripwire exists to catch.
    """
    drift: list[DriftItem] = []
    for pr in _fetch_prs(gh):
        if str(pr.get("state") or "").upper() != "OPEN":
            continue
        if "blocked" not in label_names(pr):
            continue
        pr_number = pr.get("number")
        head_sha = pr.get("headRefOid")
        if pr_number is None or not head_sha:
            continue
        pr_number = int(pr_number)

        check_runs = gh.commit_check_runs(str(head_sha))
        if not check_runs:
            continue

        # Multiple check-run entries can exist per name after a rerun; the
        # highest numeric "id" is the most recent (GitHub assigns check-run
        # ids monotonically per repo) -- never trust list order.
        latest_by_name: dict[str, dict[str, Any]] = {}
        for run in check_runs:
            name = run.get("name")
            if not name:
                continue
            current = latest_by_name.get(name)
            if current is None or int(run.get("id") or 0) > int(current.get("id") or 0):
                latest_by_name[name] = run

        aviator_run = latest_by_name.get(AVIATOR_CHECK_NAME)
        if aviator_run is None or aviator_run.get("conclusion") != "failure":
            continue
        summary = str((aviator_run.get("output") or {}).get("summary") or "")
        if AVIATOR_BLOCKED_MESSAGE not in summary:
            continue

        other_checks_green = all(
            run.get("conclusion") == "success"
            for name, run in latest_by_name.items()
            if name != AVIATOR_CHECK_NAME
        )
        if not other_checks_green:
            continue

        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        mergequeue_label = config.auto_merge.mergequeue_label
        add_labels: tuple[str, ...] = ()
        if (
            mergequeue_label
            and mergequeue_label not in label_names(pr)
            and _pr_review_approved_at_head(config, repo_root, pr_number, str(head_sha))
        ):
            add_labels = (mergequeue_label,)

        fix_actions = [f"remove label 'blocked' from PR #{pr_number}"]
        if add_labels:
            fix_actions.append(
                f"add label {mergequeue_label!r} to PR #{pr_number} (Aviator re-queue)"
            )
        drift.append(
            DriftItem(
                kind="aviator_stale_blocked",
                issue_number=issue_number,
                pr_number=pr_number,
                detail=(
                    f"PR #{pr_number} has a stale Aviator 'blocked' label -- all real CI "
                    "checks are green but aviator/checks still reports the blocked-label "
                    "failure"
                ),
                fix_actions=tuple(fix_actions),
                remove_labels=("blocked",),
                add_labels=add_labels,
            )
        )
    return drift


def _mergequeue_revocation_detail(
    config: OrchestratorConfig,
    repo_root: Path,
    pr_number: int,
    head_sha: str,
    mergequeue_label: str,
) -> str:
    """Human-readable explanation for why ``mergequeue`` is being revoked.

    Distinguishes "genuinely not approved" (missing/unreadable decision
    file, or a recorded ``request_changes``/never-reviewed verdict -- the
    PR #695 / issue #819 case) from "approved, but at an older head that a
    rebase has since moved past" -- a case ``merge_ready``'s own
    carry-forward machinery (``_check_carry_forward``) may still legitimately
    re-validate and re-approve on a later pass. Both are revoked (see
    ``detect_mergequeue_not_approved``'s docstring for why leaving the
    stale-head case alone is not safe either), but the emitted detail/event
    payload makes the distinction explicit rather than collapsing both into
    one indistinguishable string.
    """
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    resolved = _resolve_review_decision(paths.prs / f"pr-{pr_number}", None, head_sha)
    if resolved.missing:
        return f"no readable review-decision.json for PR #{pr_number}"
    verdict = resolved.decision
    if verdict != "approved":
        return f"recorded decision is {verdict!r}, not 'approved'"
    reviewed_head = resolved.reviewed_head_sha
    reviewed_head_display = str(reviewed_head)[:12] if reviewed_head else repr(reviewed_head)
    return (
        f"approved at stale head {reviewed_head_display} but current head is "
        f"{head_sha[:12]} -- deferring re-validation to merge_ready's carry-forward "
        f"check rather than {mergequeue_label!r} staying authorized on an unvalidated head"
    )


def detect_mergequeue_not_approved(
    gh: GitHubLike, config: OrchestratorConfig, *, repo_root: Path | None = None
) -> list[DriftItem]:
    """Revoke the Aviator ``mergequeue`` label from any open PR not approved at its head.

    Production incident this exists for (issue #819): PR #695 carried
    ``mergequeue`` after its recorded review verdict flipped to
    ``request_changes`` at the live head, and nothing in the orchestrator
    ever calls ``remove_pr_label`` for ``mergequeue`` -- ``remove_pr_label``
    has exactly one call site anywhere in the repo before this change (the
    Aviator ``blocked`` label in ``detect_aviator_stale_blocked``, above).
    ``merge_ready``'s own carry-forward-failure path (workflow.py's
    ``head_moved and not carried_forward`` branch) transitions issue labels
    and returns ``can_merge: False`` but never strips the PR-level
    ``mergequeue`` label either. With ``.aviator/config.yml``'s
    ``number_of_approvals: 0``, the label plus green Lint/Tests checks IS
    Aviator's merge decision -- Aviator never reads ``review-decision.json``.
    Once applied, ``mergequeue`` was otherwise irrevocable, so Aviator
    merged #695 over a standing ``request_changes`` verdict.

    Deliberately NOT folded into ``detect_drift``, for the same reason
    ``detect_aviator_stale_blocked`` is separate: that function's contract
    (``test_detect_drift_makes_zero_mutating_calls``) pins ``detect_drift``
    itself to exactly two ``gh.run`` list calls, to avoid repeating issue
    #361 (an unconditional per-PR GraphQL walk caused 502s). ``_fetch_prs``
    calls ``gh.run`` directly (via ``_fetch_snapshot``), bypassing
    ``GitHub``'s ``_list_cache`` entirely -- that cache only covers
    ``gh.pr_list()``/``gh.issue_list()``, not the raw ``_fetch_snapshot``
    path, and ``_fetch_snapshot`` has no memoization of its own. So this is
    honestly a *third* full ``gh pr list --state all`` round trip per
    reconcile pass: ``detect_drift`` fetches once, ``detect_aviator_stale_blocked``
    already duplicates that fetch (a pre-existing, accepted wart -- same
    file, same list command, same cost class), and this function is a third
    instance of the identical pattern, not a new one. What it does NOT add
    is any *per-PR* call: ``RECONCILE_PR_FIELDS`` already includes ``labels``
    and ``headRefOid`` for every PR that one list call returns, so
    label-membership and head-sha checks are free once the list is in hand.
    The one filesystem read (``_pr_review_approved_at_head`` /
    ``_read_review_decision``) is gated behind PR state == OPEN *and* the PR
    already carrying ``mergequeue`` -- matching ``detect_aviator_stale_blocked``'s
    "gate on the cheap already-fetched labels field first" discipline -- so
    that cost scales with how many PRs are actually in the merge queue, not
    with the total open-PR count. (Collapsing all three ``_fetch_prs`` calls
    into one shared fetch per pass is a legitimate follow-up; out of scope
    for issue #819, which is about closing the revocation gap, not this
    pre-existing duplication.)

    Gate resolution (issue #819's item 4 -- carry-forward interaction):
    reuses ``_pr_review_approved_at_head`` verbatim as a single,
    undifferentiated revocation gate. The label is removed whenever that
    predicate returns ``False``, for *any* of its reasons: recorded decision
    isn't ``"approved"``, the decision file is missing/unreadable, or the
    decision is ``"approved"`` but at a stale head a rebase has since moved
    past. That last sub-case is deliberately revoked rather than deferred,
    for two independent reasons:

    1. Safety gap: leaving it alone is not safe. ``merge_ready``'s
       carry-forward-failure path never strips ``mergequeue`` either (see
       above), so an approved-at-stale-head PR whose rebase turns out to
       have changed real content would keep an unvalidated label and sail
       through Aviator once CI goes green -- the same failure class as #695,
       through a second door.
    2. Cost: reconcile.py cannot cheaply arbitrate this case itself.
       Answering "is this stale-head approval still valid" requires
       ``_check_carry_forward``'s patch-id / line-content diff comparison
       (workflow.py), which needs a per-PR ``gh.pr_diff()`` call -- exactly
       the issue-#361 cost class ``detect_drift`` and this module exist to
       stay out of.

    Revoking does not "fight" carry-forward -- it cannot, because
    carry-forward is unreachable in both directions here: ``merge_ready``
    only attempts ``_check_carry_forward`` ``if approved:`` (the recorded
    decision is already ``"approved"``), so a ``request_changes``/absent
    decision never reaches carry-forward logic at all. For the genuine
    stale-head-approved case, revoke-then-reapply is cooperative rather than
    adversarial: on the PR's next ``merge_ready`` evaluation, a *clean*
    rebase re-validates via carry-forward, ``_update_approval_head`` records
    approval at the new head, and ``add_pr_label`` is idempotent (per its
    own inline comment at the call site) -- so ``mergequeue`` comes right
    back. ``_maybe_reconcile_drift`` running before the per-PR
    ``merge_ready`` loop within the same ``_loop_body`` pass shortens that
    window to sub-pass latency, but that ordering is a latency detail, not
    the safety argument -- revoke-first-then-cooperative-readd is correct
    regardless of pass ordering, because the alternative (never revoking a
    stale-head approval) is the one with a real unreviewed-merge hole.

    ``repo_root is None`` means this function is globally blind, not that
    any individual PR is unapproved: it returns ``[]`` rather than revoking
    ``mergequeue`` fleet-wide from every currently-labeled PR. A blanket
    revocation storm triggered by the detector's own blindness would be a
    false-positive catastrophe, not fail-closed behavior -- exactly what the
    negative (approved-at-head, left alone) test guards against.
    """
    mergequeue_label = config.auto_merge.mergequeue_label
    if not mergequeue_label:
        return []
    if repo_root is None:
        return []
    drift: list[DriftItem] = []
    for pr in _fetch_prs(gh):
        if str(pr.get("state") or "").upper() != "OPEN":
            continue
        if mergequeue_label not in label_names(pr):
            continue
        pr_number = pr.get("number")
        head_sha = pr.get("headRefOid")
        if pr_number is None or not head_sha:
            continue
        pr_number = int(pr_number)
        head_sha = str(head_sha)

        if _pr_review_approved_at_head(config, repo_root, pr_number, head_sha):
            continue

        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        reason = _mergequeue_revocation_detail(
            config, repo_root, pr_number, head_sha, mergequeue_label
        )
        drift.append(
            DriftItem(
                kind="mergequeue_revoked",
                issue_number=issue_number,
                pr_number=pr_number,
                detail=(
                    f"PR #{pr_number} carries {mergequeue_label!r} but is not approved "
                    f"at its current head {head_sha[:12]} ({reason}); revoking to close "
                    "the irrevocable-mergequeue gap (issue #819)"
                ),
                fix_actions=(f"remove label {mergequeue_label!r} from PR #{pr_number}",),
                remove_labels=(mergequeue_label,),
            )
        )
    return drift


def detect_drift(
    gh: GitHubLike,
    state: dict[str, Any],
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
    skip_dead_session_sweep: bool = False,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> list[DriftItem]:
    """Read-only comparison of GitHub reality against ``state``.

    Issues exactly two ``gh.run`` list queries (all PRs, all issues) and
    performs every drift check against those two in-memory snapshots — no
    per-item ``gh`` calls. ``state_path``, if given, additionally allows the
    stale-``human_needed`` check (issue #947, below) to fall back to
    ``events.db`` for issues escalated before that check shipped; passing
    ``None`` (the default, and what every pre-#947 caller/test still passes)
    simply narrows that one check to its state.json-only tiers, not a fourth
    ``gh.run`` call.

    If ``repo_root`` is provided, also checks for dead sessions and classifies
    their failures to update the provider throttle state.

    ``skip_dead_session_sweep`` (merge-lane-recovery §6-B): when True, the
    confirmed-dead-session classify+reap block below is skipped entirely,
    while live-session tracking and launch-stalled detection (both gated on
    the same ``repo_root is not None`` check, immediately above it) still
    run. This lane predates issue #343 and never adopted its single
    enforcement point (``classify_worker_health`` + the
    ``max_inconclusive_probe_deferrals`` grace cap) — it goes straight from
    "pid not alive" to classify-and-reap on the very first sighting. That
    was safe as long as this function was reachable only from manual
    ``mop-up --fix``, run by an operator who has already independently
    satisfied themselves the session is really gone. It stops being safe once
    ``detect_drift`` runs automatically inside the main loop
    (``_maybe_reconcile_drift``): the loop's own stall/dead lanes
    (``_detect_and_handle_stalled_sessions`` /
    ``_classify_dead_sessions_and_update_throttle_state``) already ran this
    exact same pass, immediately before reconcile, and may have deliberately
    *deferred* a not-yet-confirmed-dead session to preserve its grace budget.
    Re-scanning the same ``sessions_dir`` a few calls later with no memory of
    that decision reaps the sidecar out from under the grace period,
    silently halving it every 30 minutes. The periodic in-loop caller passes
    True for this reason; ``mop-up --fix`` (and every existing caller/test)
    defaults to False and keeps today's full behavior.
    """
    threshold = config.runtime.graphql_rate_limit_threshold
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold)
    if not sufficient:
        raise GraphQLBudgetError(remaining, reset_at, threshold)
    now = now if now is not None else datetime.now(UTC)

    labels_cfg = config.labels
    # Issue #1266: a mechanical escalation parks on operator_queue instead of
    # human_needed, so "terminal escalation label present" must watch both --
    # derived from _escalation_edge's own mechanical-edge table (via
    # _escalation_label) rather than a second hardcoded label pair, so the
    # label-repair consumers below and the staleness alert further down can
    # never drift out of sync about what "escalated and parked" looks like.
    escalation_parked_labels = {
        _escalation_label(labels_cfg, "escalated"),
        _escalation_label(labels_cfg, _escalation_edge("escalated", "mechanical")),
    }
    prs = _fetch_prs(gh)
    issues = _fetch_issues(gh)
    issues_by_number = {int(issue["number"]): issue for issue in issues if issue.get("number")}
    # Issue #762: issues are now fetched via a paginated REST ``issues``
    # snapshot. The list is either complete (we stopped because GitHub returned
    # an under-full page) or the fetch raised. It is no longer a fixed
    # ``_LIST_LIMIT`` page, so the snapshot is not permanently incomplete.
    # The per-item ``issue is None`` checks below already fail safe on any
    # genuinely missing issue, and the warning remains defensive dead code.
    issue_snapshot_truncated = False
    state_prs: dict[str, Any] = state.get("prs", {})
    # Issue #762: PRs are now fetched via a paginated REST ``pulls`` snapshot.
    # The list is either complete (we stopped because GitHub returned an
    # under-full page) or the fetch raised. It is no longer a fixed
    # ``_LIST_LIMIT`` page, so the snapshot is not permanently incomplete.
    # state_pr_missing_on_github therefore runs unconditionally below.
    pr_snapshot_incomplete = False

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
            #
            # A tracked PR (state_entry is not None) whose status key is
            # missing is the same blind spot the sibling OPEN-PR repair
            # branch (pr_status_normalized below) covers for OPEN PRs: it
            # normalizes a None status to the passive placeholder. The
            # symmetric convergence here covers a tracked CLOSED-unmerged
            # PR with no status key, so the entry does not linger as a
            # status-less record the closed sweep skips (Minor symmetry gap
            # vs the OPEN branch). Untracked PRs (state_entry is None) are
            # still never invented an entry for.
            state_status = (state_entry or {}).get("status")
            if state_entry is not None and state_status not in ("closed", "merged"):
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

            # Issue #558 (issue-side counterpart): converge the linked
            # issue's state status away from any ACTIVE_STATE_STATUS when
            # its PR is closed-unmerged and the GitHub issue itself is
            # still OPEN. Without this, an issue stuck in
            # "rework_requested" (or "reviewing", "dispatched", ...) is
            # selected by dispatch_rework's state-driven candidate scan
            # every loop pass, which calls gh.issue_view() on it before
            # any open-PR filtering -- a permanent per-pass GitHub fetch
            # with no terminal exit, the exact slow-cost-spiral shape
            # #556/#558 exist to eliminate. The existing
            # closed_unmerged_pr_active_labels rule only strips GitHub
            # labels and never touches state["issues"][n]["status"], and
            # state_active_status_issue_closed only fires when the GitHub
            # issue itself is CLOSED -- so an OPEN issue with a
            # closed-unmerged PR is invisible to both. Converge to the
            # dormant baseline (drop the status key, the same target
            # issue_status_normalized uses for a never-dispatched issue)
            # so the issue drops out of every status-driven selector until
            # a human re-arms it. The linked issue's label disposition
            # remains owned by closed_unmerged_pr_active_labels; the two
            # are independent and may both fire for the same PR.
            #
            # Issue #1066: "escalated" is excluded via
            # DORMANT_CONVERGENCE_EXCLUDED_STATUSES. Unlike the other
            # ACTIVE_STATE_STATUSES members, "escalated" is not orchestrator
            # work-in-flight that should be reset to dormant when its PR dies
            # -- it is a human-owned terminal disposition, and the issue's
            # agent:human-needed label stays live regardless of what happens
            # to this PR. Dropping the status key here would silently detach
            # the state entry from that still-live label with no automated
            # path back into the human queue (fired in production: issue
            # #894 via PR #948). Leaving "escalated" status in place does not
            # make this issue a rework candidate: dispatch_rework's candidate
            # filter is status == "rework_requested" specifically (not
            # general ACTIVE_STATE_STATUSES membership), so this exclusion
            # does not reintroduce that path's per-pass gh.issue_view() cost.
            if issue is not None and _issue_state(issue) == "OPEN" and issue_number is not None:
                issue_entry = state.get("issues", {}).get(str(issue_number))
                issue_status = issue_entry.get("status") if isinstance(issue_entry, dict) else None
                if issue_status in ACTIVE_STATE_STATUSES - DORMANT_CONVERGENCE_EXCLUDED_STATUSES:
                    drift.append(
                        DriftItem(
                            kind="closed_unmerged_pr_issue_state_converged",
                            issue_number=issue_number,
                            pr_number=pr_number,
                            detail=(
                                f"PR #{pr_number} is CLOSED (unmerged) on GitHub but "
                                f"linked issue #{issue_number} state status is "
                                f"{issue_status!r}; converging to dormant (no status)"
                            ),
                            fix_actions=(f"drop status key for issues[{issue_number}]",),
                            new_status=None,
                        )
                    )
        elif gh_state == "OPEN":
            # A PR record that the orchestrator is already tracking (has an
            # entry in state["prs"]) but that never got a status written --
            # e.g. a review packet generation crashed between creating the
            # entry and recording its first status -- is invisible to every
            # status-driven selector. Normalize it to the same passive
            # PASSIVE_OPEN_STATUS placeholder issues get in the sibling sweep
            # below; never invent a status for a PR the orchestrator never
            # tracked (state_entry is None) since that may not be one of
            # ours. Issue #955: this used to write the literal "reviewing"
            # -- the same value ``review()`` writes when a reviewer really
            # is coming -- which made the #487 stalled-review sweep
            # (workflow.py) misidentify this placeholder as an undispatched
            # review packet. PASSIVE_OPEN_STATUS is now a distinct value so
            # this write can never collide with that sweep's target.
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

            # Issue #1153: a same-repo PR whose closing-keyword reference
            # resolves to an issue number that does not exist in this repo
            # (e.g. ``Closes #1497`` where 1497 is a sibling repo's issue)
            # is invisible to every issue-driven sweep: ``linked_issue_number``
            # returned a number, so the PR was added to
            # ``open_prs_by_issue[issue_number]``, but ``issues_by_number``
            # has no entry for it -- no issue-side normalization, no review
            # routing, no merge lane ever sees it. The PR silently becomes
            # ``open_passive`` (if tracked) or is simply never tracked at
            # all. A resolution failure is a signal, not an absence: surface
            # it as a distinct drift item so the ``reconcile`` event makes
            # the cross-repo linkage failure visible to operators and the
            # janitor, instead of silently demoting the PR to passive.
            #
            # Self-healing (mirrors every sibling drift kind in this loop):
            # once ``apply_fixes`` has tracked this PR in state with the
            # cross-repo ``issue_number`` (from a prior reconcile pass), the
            # linkage failure is already surfaced -- the ``reconcile`` event
            # was emitted and the PR is visible to future sweeps. Skip
            # re-emitting so this drift kind does not re-fire on every pass
            # for as long as the PR stays open, which would unboundedly
            # duplicate ``reconcile`` events in the capped events ring and
            # ``events.db``. ``pr_status_normalized`` below self-heals the
            # same way (skips once ``state_entry["status"]`` is no longer
            # None); ``closed_unmerged_pr_state_converged`` above self-heals
            # by skipping once status is ``"closed"``.
            #
            # The self-heal checks ``issue_number`` only, not ``status``:
            # ``apply_fixes`` deliberately does NOT escalate (see the
            # ``apply_fixes`` branch for this kind), because escalating
            # would pre-empt the existing ``foreign_issue_ref`` parking
            # mechanism in the per-PR review loop (``review()`` treats
            # ``status="escalated"`` as terminal and returns early, so
            # ``issue_view`` is never called and the one-shot foreign-PR
            # digest is never emitted). Tracking the PR with the cross-repo
            # ``issue_number`` alone is sufficient for self-healing and
            # preserves the existing parking and alerting path.
            if (
                issue_number is not None
                and issues_by_number.get(issue_number) is None
                and not (
                    isinstance(state_entry, dict)
                    and state_entry.get("issue_number") == issue_number
                )
            ):
                drift.append(
                    DriftItem(
                        kind="pr_linked_issue_not_in_repo",
                        issue_number=issue_number,
                        pr_number=pr_number,
                        detail=(
                            f"PR #{pr_number} has closing reference to issue "
                            f"#{issue_number}, but no such issue exists in this "
                            f"repo; the reference likely targets a sibling repo "
                            f"(cross-repo linkage failure, issue #1153)"
                        ),
                        fix_actions=(
                            f"surface cross-repo linkage failure for PR #{pr_number} "
                            f"(closing ref issue #{issue_number} not in this repo)",
                        ),
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
            is_worker_confirmed_dead,
            iter_workers,
            real_activity_probe_for,
        )

        sessions_dir = resolved_layout(config, repo_root).sessions_dir
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
                                            reason="launch_stalled_no_open_pr",
                                            failure_kind="launch_stalled",
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

                if (
                    not skip_dead_session_sweep
                    and w.error is None
                    and is_worker_confirmed_dead(
                        w,
                        config,
                        now,
                        sessions_dir,
                        persist_inconclusive_probe_counter=True,
                    )
                ):
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
                        # session_completed=True (issue #656): the worktree
                        # inspection just above is ground truth this session
                        # produced real, committable work -- skip log-tail
                        # marker matching entirely rather than let it treat
                        # the session's own completion-summary prose as a
                        # provider-throttle signature. See claude_code.
                        # update_worker_record_with_failure_classification.
                        if w.adapter_kind == "devin":
                            failure_kind, throttled_until = (
                                update_session_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind="unpublished_work",
                                    config=config,
                                    session_completed=True,
                                )
                            )
                        elif w.adapter_kind == "claude-code":
                            failure_kind, throttled_until = (
                                update_worker_record_with_failure_classification(
                                    sessions_dir,
                                    w.issue_number,
                                    fallback_kind="unpublished_work",
                                    config=config,
                                    session_completed=True,
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
                                    session_completed=True,
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
                                throttle_reason=failure_kind,
                                throttle_adapter_kind=w.adapter_kind,
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
                            if active_labels and (
                                failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                                or failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
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
                            elif active_labels and inspection.ahead_count > 0:
                                # Issue #252: completed-but-unpublished work takes the salvage
                                # path (push + PR) instead of re-dispatching.
                                # Issue #1130: relax from ``is_completed`` to
                                # ``ahead_count > 0`` — a worktree with
                                # committed-but-unpushed work plus shim/scaffolding
                                # dirt (not in ``injected_paths``) classifies as
                                # PARTIAL, not COMPLETED, but the committed work
                                # is still salvageable. Salvage pushes the branch
                                # ref (the committed work), not the working tree.
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
                                            f"issue #{w.issue_number} session has a worktree "
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
                                        reason="dead_session_no_open_pr",
                                        failure_kind=failure_kind,
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

        # Issue #947: ``agent:human-needed`` is a forced terminal state with
        # no other alerting -- an issue parked there (e.g. #894) is silently
        # invisible until an operator happens to look. Gated on the GitHub
        # label itself (not ``tracked_status``) so it fires uniformly for
        # every path that can apply the label (``escalated``, ``blocked``,
        # ``redispatch_escalated``, ``merged_pr_mention_flagged``, or a
        # manual add with no state.json entry at all), and placed before the
        # ``tracked_status == "escalated"`` branch below because that branch
        # unconditionally ``continue``s -- inserting after it would silently
        # skip the common case.
        #
        # Issue #1266: ``escalation_parked_labels`` extends this same alert
        # to ``operator_queue`` -- a mechanical escalation that never clears
        # (e.g. the de-escalation sweep itself stops running) must not sit
        # silently forever just because it parked on the *other* terminal
        # label instead of ``human_needed``.
        #
        # Age is resolved with a 3-tier fallback so a legacy escalation
        # (predating this check) still gets a real age instead of
        # masquerading as fresh:
        #   1. ``terminal_since`` -- stamped by ``_escalate_issue`` on every
        #      escalated/blocked transition going forward.
        #   2. ``merged_pr_mention_flagged_at`` -- the one other durable
        #      local timestamp a human_needed transition can carry (issue
        #      #203); that path does not go through ``_escalate_issue``.
        #   3. The most recent escalation-transition event in ``events.db``
        #      for this issue, using the same CI-verified exhaustive kind
        #      registry ``_backfill_missing_reason_classes`` already relies
        #      on (``test_escalation_event_kind_mapping_is_complete`` in
        #      test_deescalation.py) -- covers issues escalated before this
        #      check shipped, e.g. #894.
        # No timestamp found in any tier reports immediately as "never
        # observed" rather than defaulting to fresh: silently treating
        # unknown age as healthy is exactly the failure mode
        # ``classify_backlog_reachability``'s ``observed: False`` return
        # exists to avoid.
        stale_terminal_labels = issue_labels & escalation_parked_labels
        if stale_terminal_labels and _issue_state(issue) == "OPEN":
            since_raw: str | None = None
            if isinstance(tracked_entry, dict):
                since_raw = tracked_entry.get("terminal_since") or tracked_entry.get(
                    "merged_pr_mention_flagged_at"
                )
            if not since_raw and state_path is not None:
                events = query_events(state_path, issue_number=issue_number)
                escalation_kinds = (
                    frozenset(ESCALATION_REASON_CLASS_BY_EVENT_KIND)
                    | DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS
                )
                escalation_events = [e for e in events if e.get("kind") in escalation_kinds]
                if escalation_events:
                    since_raw = escalation_events[-1]["ts"]

            age_days: float | None = None
            if since_raw:
                try:
                    since_dt = datetime.fromisoformat(str(since_raw).replace("Z", "+00:00"))
                    age_days = (now - since_dt).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    # A malformed/naive timestamp must never crash a
                    # read-only drift pass -- fall back to "never observed"
                    # (age_days stays None) the same as no timestamp at all.
                    age_days = None

            threshold_days = config.reconcile_pass.terminal_state_alert_days
            if age_days is None or age_days >= threshold_days:
                # Issue #1266: name whichever parked label is actually
                # present -- human_needed for a judgment escalation,
                # operator_queue for a mechanical one -- instead of
                # hardcoding human_needed into the message regardless of
                # which label is really there. Both are never present
                # together (labels.py's edges remove each other), but pick
                # deterministically (prefer human_needed) for the
                # pathological case where a manual label add put both on
                # the issue at once.
                stale_label = (
                    labels_cfg.human_needed
                    if labels_cfg.human_needed in stale_terminal_labels
                    else labels_cfg.operator_queue
                )
                detail = (
                    f"issue #{issue_number} has been parked in "
                    f"'{stale_label}' for {age_days:.1f} day(s)"
                    if age_days is not None
                    else (
                        f"issue #{issue_number} carries '{stale_label}' with no "
                        "recorded escalation timestamp (age never observed)"
                    )
                )
                drift.append(
                    DriftItem(
                        kind="terminal_state_stale",
                        issue_number=issue_number,
                        pr_number=None,
                        detail=detail,
                        fix_actions=(),
                    )
                )

        if tracked_status == "escalated" and _issue_state(issue) == "OPEN":
            # Issue #1266: a mechanical escalation's correct label is
            # operator_queue, not human_needed -- deriving the target from
            # the issue's reason_class (via the same helpers the label-repair
            # self-heal sweep uses) is what stops this convergence check from
            # clobbering a correctly operator-queued issue back to
            # human_needed on every reconcile pass.
            repair_reason_class = _repair_reason_class(
                tracked_entry if isinstance(tracked_entry, dict) else None
            )
            expected_label = _escalation_label(
                labels_cfg, _escalation_edge("escalated", repair_reason_class)
            )
            needs_expected_label = (
                expected_label is not None and expected_label not in issue_labels
            )
            if needs_expected_label or active_present:
                fix_actions = []
                add_labels: tuple[str, ...] = ()
                if needs_expected_label:
                    fix_actions.append(f"add label '{expected_label}' to issue #{issue_number}")
                    add_labels = (expected_label,)
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
                # Issue #1092: state corroborates the label -- some router is
                # deliberately holding this issue in a lane and will re-set both
                # the label and the status on its next pass, so "repairing" it
                # here just starts a loop. Worker liveness (the check above) is
                # NOT sufficient on its own: "active label + open PR + no live
                # worker" is also the exact, correct state of an issue queued
                # behind the dispatch concurrency cap, which is every issue past
                # the Nth in any rework batch.
                #
                # Deny-list form is deliberate. tracked_status is None when the
                # issue has no state entry at all, and None must FALL THROUGH to
                # let the rule fire -- a brand-new issue whose PR went unnoticed
                # is exactly what the needs_pr_open half exists to catch. The
                # allow-list spelling (`tracked_status in _LABEL_STALE_STATUSES`)
                # reads equivalently and silently excludes that issue forever.
                and tracked_status not in _LABEL_CORROBORATING_STATUSES
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
    # still look active.
    #
    # Issue #857: this loop used to be gated behind `if not
    # issue_snapshot_truncated`, skipping the whole sweep whenever the issue
    # list hit the page limit. That outer gate was unjustified -- the per-item
    # lookup two lines below already fails safe on a missing issue (`.get()` ->
    # None -> continue), which is exactly the same "missing is unanswerable,
    # skip per-item" reasoning the issue_status_normalized loop below makes in
    # its own comment. Dropping the outer gate costs zero extra API calls and
    # restores finalization for every issue inside the snapshot window; issues
    # that fell off the (created-desc) page are still silently skipped, per
    # item, by the `issue is None` check below.
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
    #
    # Issue #789: "closed" is deliberately excluded from the skip-set
    # (ORCHESTRATOR_OWNED_ISSUE_STATUSES, not the full VALID_ISSUE_STATUSES)
    # because GitHub -- not the orchestrator -- owns that value and can
    # invalidate it at any time via a reopen. Re-examining it costs no
    # extra GitHub call: `issues_by_number` below is the same in-memory
    # snapshot the rest of this function already uses, so the common
    # both-closed case (the overwhelming majority of "closed" entries)
    # just confirms target_status == current_status and continues without
    # emitting drift.
    #
    # Issue #859 (PER-ITEM, not the global pr_snapshot_incomplete flag -- see
    # the elif branch below for why): a per-issue index of PRs state.json
    # itself still tracks as open (status not yet "closed"/"merged") whose PR
    # number does not appear anywhere in this pass's `prs` snapshot at all
    # (`pr_numbers_on_github`, built above alongside state_pr_missing_on_github).
    # That absence is the same kind of "unanswered query" #789 already
    # protects on the issue side, scoped to exactly the issues state.json's
    # own bookkeeping says are affected -- not every issue in the loop.
    untrusted_open_pr_by_issue: dict[int, list[int]] = {}
    for _pr_number_str, _pr_entry in state_prs.items():
        if not isinstance(_pr_entry, dict):
            continue
        try:
            _tracked_pr_number = int(_pr_number_str)
        except ValueError:
            continue
        if _tracked_pr_number in pr_numbers_on_github:
            continue  # positively observed (open or closed) in this pass
        _tracked_issue_number = _pr_entry.get("issue_number")
        if _tracked_issue_number is None:
            continue
        if _pr_entry.get("status") in ("closed", "merged"):
            continue
        untrusted_open_pr_by_issue.setdefault(int(_tracked_issue_number), []).append(
            _tracked_pr_number
        )
    # Issues whose issue_status_normalized None-outcome was deferred this
    # pass because of the above -- reported once, after the loop, so the
    # drift log names what was skipped instead of going silent (issue #859
    # review comment 4).
    issue_status_normalization_deferred: dict[int, list[int]] = {}

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
        if current_status in ORCHESTRATOR_OWNED_ISSUE_STATUSES:
            continue
        issue = issues_by_number.get(issue_number)
        if issue is None:
            # The snapshot cannot support any conclusion about an issue it
            # doesn't contain -- absence here is an unanswered query, not
            # evidence the issue is gone (issue #789 review). This matters
            # once the repo passes _LIST_LIMIT: an older closed issue can
            # fall off the `--state all` page while still being genuinely
            # closed, and falling through to `target_status = None` would
            # strip its "closed" status -- a mass wipe with no signal in
            # the drift log to explain it. Skip unconditionally rather
            # than gating on issue_snapshot_truncated: a missing issue is
            # equally unanswerable regardless of *why* it's missing.
            continue
        if _issue_state(issue) == "CLOSED":
            target_status: str | None = "closed"
        elif open_prs_by_issue.get(issue_number):
            target_status = PASSIVE_OPEN_STATUS
        elif untrusted_open_pr_by_issue.get(issue_number):
            # Issue #859 review: the first version of this fix gated on the
            # GLOBAL `pr_snapshot_incomplete` flag, which is monotonic under
            # `--state all` -- once total PR count crosses _LIST_LIMIT it is
            # True on every future pass, forever, turning a `continue` here
            # into a permanent repo-wide kill switch on issue_status_normalized
            # (exactly what issue #857/#860 already fixed once, for the issue
            # side of this same loop, in commit 09721d5: "the per-item lookup
            # already fails safe ... the outer gate added nothing beyond
            # that"). Rejected in review.
            #
            # This is the per-item replacement: `untrusted_open_pr_by_issue`
            # (built above) is keyed on state.json's OWN PR record for this
            # specific issue being both still-open (by state's bookkeeping)
            # and absent from `prs` entirely -- not on the global truncation
            # flag. A negative answer here is unreliable only for the issues
            # this positively implicates; every other issue in the loop falls
            # through to `target_status = None` exactly as before, regardless
            # of whether the overall snapshot happens to be truncated.
            #
            # `target_status = None` is also the correct, legitimate outcome
            # for an issue that is genuinely open with no PRs at all --
            # state.json's own record is what tells the two cases apart here.
            # Recovery for a deferred issue requires its tracked PR number to
            # reappear in a future `prs` snapshot (or for state's PR record to
            # itself converge to "closed"/"merged" via the sweeps above): that
            # is guaranteed only while total PR count stays under
            # _LIST_LIMIT. Above the cap, `--state all` is monotonic (closed
            # PRs never leave it), so an old PR number that has permanently
            # fallen out of the window makes this deferral permanent for that
            # one issue too -- a narrower, self-contained blast radius than
            # the rejected global gate, not a guarantee of eventual
            # convergence.
            issue_status_normalization_deferred[issue_number] = sorted(
                untrusted_open_pr_by_issue[issue_number]
            )
            continue
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

    # Issue #859 review comment 4: name what was actually deferred rather than
    # going silent. This fires only on passes where the per-item check above
    # actually deferred at least one issue -- unlike the two blanket
    # truncation warnings below, it is not tied to a global len(...) >=
    # _LIST_LIMIT flag, so it stays proportionate to what really happened.
    if issue_status_normalization_deferred:
        logger.warning(
            "[repo=%s] issue_status_normalized deferred for %d issue(s) whose "
            "state-tracked open PR(s) are absent from this pass's PR "
            "snapshot: %s",
            _repo_slug(gh),
            len(issue_status_normalization_deferred),
            issue_status_normalization_deferred,
        )
        drift.append(
            DriftItem(
                kind="snapshot_truncated",
                issue_number=None,
                pr_number=None,
                detail=(
                    "issue_status_normalized deferred its None-outcome for "
                    f"{len(issue_status_normalization_deferred)} issue(s) "
                    f"{sorted(issue_status_normalization_deferred)}: state.json "
                    "tracks an open PR for each that is absent from this "
                    "pass's PR snapshot"
                ),
                fix_actions=(
                    "issue_status_normalized skipped these issues' None-outcome "
                    "rather than normalizing away a possibly-stale status; "
                    "re-run once the tracked PR reappears in the snapshot or "
                    "its state record converges to closed/merged",
                    "see issue #859",
                ),
            )
        )

    # Issue #15 / issue #259 / issue #857: if the issue snapshot hit the page
    # limit, it is provably incomplete. state_active_status_issue_closed and
    # issue_status_normalized above still ran against every issue inside the
    # snapshot window -- only issue numbers that fell outside the (created-desc)
    # page were silently skipped per-item (see the `issue is None` checks
    # above). Emit a warning that reports that partial coverage honestly rather
    # than claiming the sweeps were skipped outright.
    if issue_snapshot_truncated:
        logger.warning(
            "[repo=%s] Issue snapshot is truncated at the page limit (%d); "
            "state_active_status_issue_closed and issue_status_normalized ran "
            "against the %d issues in the snapshot, but issues outside that "
            "page window were not evaluated this pass",
            _repo_slug(gh),
            _LIST_LIMIT,
            len(issues),
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
                    "state_active_status_issue_closed and issue_status_normalized "
                    "ran against the in-window snapshot; issues outside the page "
                    "window were not evaluated this pass",
                    "full pagination is tracked in issue #857",
                ),
            )
        )

    # Defensive: this is dead code while _fetch_prs paginates to completion,
    # but if the fetch is ever re-capped the warning (and DriftItem) must still
    # carry the repo slug.
    if pr_snapshot_incomplete:
        logger.warning(
            "[repo=%s] PR snapshot is incomplete (%d returned, %d tracked in state); "
            "skipping state_pr_missing_on_github sweep for this pass",
            _repo_slug(gh),
            len(prs),
            len(state_prs),
        )
        drift.append(
            DriftItem(
                kind="snapshot_truncated",
                issue_number=None,
                pr_number=None,
                detail=(
                    f"PR snapshot returned {len(prs)} PR(s) while state tracks "
                    f"{len(state_prs)}; snapshot may be incomplete or the fetch may "
                    "have failed"
                ),
                fix_actions=(
                    "skip state_pr_missing_on_github sweep for this pass",
                    "full pagination is tracked in issue #857",
                ),
            )
        )

    return drift


def apply_fixes(
    gh: GitHubLike,
    state: dict[str, Any],
    drift: list[DriftItem],
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the structured fixes for each drift item and return a NEW state dict.

    ``state`` (and its nested ``issues``/``prs`` dicts) are never mutated in
    place — every touched entry is replaced via ``{**existing, ...}``.

    If ``repo_root`` is provided, merged/closed PR drift items also tear down
    the isolated ``reviews_dir`` checkout and clear any ``review_dispatch_*``
    state so the closed lifecycle cannot be mistaken for a live claim. A PR
    whose reviewer process is still alive is deferred to a later pass (issue
    #504) so the live session is not interrupted.

    ``state_path`` is threaded through to ``append_event`` so each
    ``"reconcile"`` event is also dual-written to the unlimited ``events.db``
    log, not just the capped 200-entry ring in ``state.json`` — without it,
    fixes like ``merged_outside_orchestrator``, ``aviator_stale_blocked``,
    and ``mergequeue_revoked`` are invisible to
    ``query_events``/``event_counts_by_kind`` entirely. The event's
    top-level ``kind`` is always the literal ``"reconcile"``; the specific
    drift kind (e.g. ``"mergequeue_revoked"``) lives in
    ``payload["kind"]`` — this module's established convention (see
    ``aviator_stale_blocked``) for a "distinct, greppable" event without a
    special case in this emit path for every new drift kind.
    """
    new_issues: dict[str, Any] = dict(state.get("issues", {}))
    new_prs: dict[str, Any] = dict(state.get("prs", {}))
    new_state: dict[str, Any] = {**state, "issues": new_issues, "prs": new_prs}

    alive_pr_numbers: set[int] = set()
    if repo_root is not None:
        reviews_dir = resolved_layout(config, repo_root).reviews_dir
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
                # Issue #747: ``merged_outside_orchestrator`` drift only fires
                # when ``state_status != "merged"`` (detect_drift, above), so
                # this is always a genuine non-merged -> merged transition.
                # Stamp ``merged_at`` so externally-merged entries get the
                # same timestamp field as fleet-merged ones; existing merged
                # entries are never back-dated because they never reach here.
                if existing_pr.get("status") != "merged":
                    new_pr_state["merged_at"] = utc_now()
                new_prs[pr_key] = new_pr_state

                if repo_root is not None:
                    reviews_dir = resolved_layout(config, repo_root).reviews_dir
                    checkout_removed = remove_review_checkout(
                        repo_root, item.pr_number, reviews_dir=reviews_dir
                    )
                    checkout_action = (
                        f"remove review checkout for PR #{item.pr_number}: "
                        f"{'ok' if checkout_removed else 'failed'}"
                    )
                else:
                    checkout_action = (
                        f"remove review checkout for PR #{item.pr_number}: skipped (no repo_root)"
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
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                if pr_key in new_prs:
                    new_prs[pr_key] = without_review_dispatch_claim(new_prs[pr_key])
                if repo_root is not None:
                    reviews_dir = resolved_layout(config, repo_root).reviews_dir
                    checkout_removed = remove_review_checkout(
                        repo_root, item.pr_number, reviews_dir=reviews_dir
                    )
                    checkout_action = (
                        f"remove review checkout for PR #{item.pr_number}: "
                        f"{'ok' if checkout_removed else 'failed'}"
                    )
                else:
                    checkout_action = (
                        f"remove review checkout for PR #{item.pr_number}: skipped (no repo_root)"
                    )
            if item.issue_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_issue_label(item.issue_number, label):
                        label_ok = False
                # Record label-write failures in the event
                fix_actions = list(item.fix_actions)
                if item.pr_number is not None:
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

        elif item.kind == "closed_unmerged_pr_issue_state_converged":
            # Issue #558 (issue-side): drop the linked issue's active
            # status key so dispatch_rework's state-driven candidate scan
            # stops selecting it (and calling gh.issue_view every loop
            # pass). The issue's label disposition is owned by
            # closed_unmerged_pr_active_labels; this only touches the
            # state status, converging to the dormant baseline (no status
            # key) that issue_status_normalized also uses for a
            # never-dispatched issue. Other fields are preserved.
            # Issue #1066: detect_drift never emits this kind for an
            # "escalated" issue status (DORMANT_CONVERGENCE_EXCLUDED_STATUSES),
            # so this branch cannot reach an escalated entry -- no
            # apply-time guard is needed here, only at emission.
            if item.issue_number is not None:
                issue_key = str(item.issue_number)
                existing_issue = new_issues.get(issue_key, {})
                new_issues[issue_key] = {k: v for k, v in existing_issue.items() if k != "status"}

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
                # escalated_labels_converged carries add_labels=(<expected
                # terminal label>,) when it never landed -- human_needed for
                # a judgment escalation, operator_queue for a mechanical one
                # (issue #1266) -- done_label_with_active_labels never sets
                # add_labels, so this loop is a no-op for that sibling kind.
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
            # and mirror the fix into state as the PASSIVE_OPEN_STATUS
            # placeholder -- so state-driven dispatch_rework stops selecting
            # the issue without falsely implying a review verdict was
            # actually recorded (the previous "approved" write here was
            # itself wrong: no reviewer ever ran). Issue #955: this is a
            # distinct value from the active "reviewing" ``review()`` writes
            # once a PR is open and a review packet has actually been
            # generated -- this self-heal never generates one.
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

        elif item.kind in ("aviator_stale_blocked", "mergequeue_revoked"):
            if item.pr_number is not None:
                label_ok = True
                for label in item.remove_labels:
                    if not gh.remove_pr_label(item.pr_number, label):
                        label_ok = False
                for label in item.add_labels:
                    if not gh.add_pr_label(item.pr_number, label):
                        label_ok = False
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
            # A status outside ORCHESTRATOR_OWNED_ISSUE_STATUSES (missing
            # entirely, never assigned by any code path, or "closed" but no
            # longer accurate because the issue was reopened on GitHub --
            # issue #789) is recomputed from ground truth in detect_drift and
            # carried here via item.new_status. None means "no status" (the
            # baseline a never-dispatched issue naturally has) -- drop the key
            # rather than write a synthesized placeholder string.
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
            # PASSIVE_OPEN_STATUS placeholder (item.new_status). Issue #955:
            # distinct from the active "reviewing" review() writes -- see
            # detect_drift's matching comment above.
            if item.pr_number is not None and item.new_status is not None:
                pr_key = str(item.pr_number)
                existing_pr = new_prs.get(pr_key, {})
                new_prs[pr_key] = {**existing_pr, "status": item.new_status}

        elif item.kind == "pr_linked_issue_not_in_repo":
            # Issue #1153: a PR whose closing reference targets an issue that
            # does not exist in this repo (cross-repo linkage failure). Track
            # the PR in state with the cross-repo ``issue_number`` so it is
            # visible to future reconcile sweeps and the drift self-heals
            # (does not re-fire once the PR is tracked with the matching
            # ``issue_number``). The ``reconcile`` event emitted below (every
            # drift item gets one) is the visible signal -- it surfaces the
            # cross-repo linkage failure to operators and the janitor.
            #
            # This does NOT set ``status="escalated"`` because the existing
            # ``foreign_issue_ref`` parking mechanism in the per-PR review
            # loop (``_mark_foreign_issue_ref``) already handles parking and
            # one-shot digest alerting when ``issue_view`` raises
            # ``GitHubNotFoundError``. Escalating here would pre-empt that
            # mechanism: ``review()`` treats ``status="escalated"`` as
            # terminal and returns early, so ``issue_view`` is never called,
            # the ``GitHubNotFoundError`` handler never fires, and the
            # foreign-PR digest is never emitted. It would also break the
            # ``dispatch_reviews`` lane, which skips escalated PRs. Tracking
            # the PR with ``issue_number`` alone preserves both the existing
            # parking/alerting path and the review-dispatch path while adding
            # reconcile visibility -- the blind spot issue #1153 describes.
            if item.pr_number is not None:
                pr_key = str(item.pr_number)
                existing_pr = new_prs.get(pr_key, {})
                pr_entry = {**existing_pr, "number": item.pr_number}
                if item.issue_number is not None:
                    pr_entry["issue_number"] = item.issue_number
                new_prs[pr_key] = pr_entry

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
                    new_state = set_throttled_until(
                        new_state,
                        throttled_until,
                        reason=item.throttle_reason,
                        adapter_kind=item.throttle_adapter_kind,
                    )
                    break

        elif item.kind == "session_failed_escalated":
            # Issue #261: worker was killed by a push-gate hook — escalate
            # via the same "redispatch_escalated" label edge workflow.py's
            # reaper uses, instead of relabeling to ready.
            # Issue #1266: this DriftItem is only ever raised when
            # failure_kind is in DETERMINISTIC_ESCALATION_FAILURE_KINDS (see
            # detect_drift) -- the exact gate workflow.py's own dead-session
            # reapers use to set reason_class="mechanical" unconditionally.
            # This is a first-escalation path (reconcile.py detected a dead
            # worker before any workflow.py sweep did), not a label-repair
            # path, so it must independently resolve the mechanical edge
            # rather than hardcoding "redispatch_escalated" -- otherwise the
            # same dead-worker condition lands on a different label purely
            # because reconcile.py's drift pass caught it first.
            if item.issue_number is not None:
                edge = _escalation_edge("redispatch_escalated", "mechanical")
                result = transition(gh, config.labels, item.issue_number, edge)
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
                        has_pr_create = getattr(gh, "pr_create", None) is not None
                        if has_pr_create:
                            # Same janitor body gate as a worker-authored PR --
                            # boilerplate alone can never satisfy it. Derive the
                            # rationale from the worker's own commit log rather
                            # than injecting the gate's keywords.
                            salvage_body = (
                                f"Closes #{item.issue_number}\n\n"
                                "Salvaged by the orchestrator from a completed-but-unpublished "
                                "worker worktree."
                            )
                            branch_summary = summarize_branch_work(
                                repo_root,
                                item.branch,
                                item.base_branch,
                                test_path_globs=config.test_adequacy.test_path_globs,
                            )
                            if branch_summary:
                                salvage_body = f"{salvage_body}\n\n{branch_summary}"
                            # cw#1263: canonicalize/validate the closing-reference
                            # line the same way workflow.py's `_open_salvage_pr`
                            # does, via the shared `closing_reference` module.
                            # `workflow.py` imports `reconcile.py` (for
                            # `apply_fixes`/`detect_drift`), so importing
                            # `workflow._open_salvage_pr` back into this module
                            # would cycle -- the standalone third module is what
                            # lets both salvage-body builders share one
                            # implementation without either importing the other.
                            closing_ref = validate_closing_reference(
                                salvage_body, item.issue_number, repo=_repo_slug(gh), gh=gh
                            )
                            salvage_body = closing_ref.body
                            if closing_ref.changed and state_path is not None:
                                log_event(
                                    state_path,
                                    "pr_closing_ref_rewritten",
                                    {
                                        "issue_number": item.issue_number,
                                        "findings": list(closing_ref.findings),
                                        "source": "session_unpublished_work_salvaged",
                                    },
                                )
                            # cw#1273: route through the bounded outer retry
                            # + duplicate-PR guard instead of calling
                            # gh.pr_create directly, matching workflow.py's
                            # _open_salvage_pr (the other pr_create call site).
                            retry_result = create_pr_with_retry(
                                gh,
                                head=item.branch,
                                base=item.base_branch,
                                title=f"Salvaged work for issue #{item.issue_number}",
                                body=salvage_body,
                                max_retries=config.runtime.pr_create_retry_max_attempts,
                                base_seconds=config.runtime.pr_create_retry_base_seconds,
                            )
                            pr_number = retry_result.pr_number
                        if pr_number is not None:
                            salvage_ok = True
                            # `pr_number` is falsy (0) under `dry_run`, where no
                            # real PR was opened -- only probe a real, truthy PR
                            # number (mirrors workflow.py::_open_salvage_pr).
                            if pr_number and state_path is not None:
                                query_ok = True
                                try:
                                    pr_view = gh.pr_view(
                                        pr_number, fields=PR_CLOSING_ISSUES_FIELDS
                                    )
                                except Exception:
                                    pr_view = {}
                                    query_ok = False
                                linked_numbers = closing_issues_referenced_numbers(pr_view)
                                # Only log when the query itself succeeded --
                                # a transient `gh` failure must not be conflated
                                # with a genuine unlinked-PR miss (see
                                # workflow.py::_open_salvage_pr for rationale).
                                if query_ok and item.issue_number not in linked_numbers:
                                    log_event(
                                        state_path,
                                        "pr_closing_ref_unlinked",
                                        {
                                            "issue_number": item.issue_number,
                                            "pr_number": pr_number,
                                            "linked_issue_numbers": sorted(linked_numbers),
                                        },
                                    )
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
                        reason="salvage_failed_fallback",
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
                        reason=item.reason,
                        failure_kind=item.failure_kind,
                        detail=item.detail,
                        fix_actions=tuple(fix_actions),
                        remove_labels=item.remove_labels,
                        add_labels=item.add_labels,
                    )

        # Issue #978: thread the structured ``reason``/``failure_kind`` onto
        # the reconcile event payload so the machine-readable "why" is not
        # buried only in the free-text ``detail`` string. Only included when
        # the drift item carries them (session_failed_relabeled kinds); other
        # kinds leave the payload shape unchanged.
        reconcile_payload: dict[str, Any] = {
            "kind": item.kind,
            "issue_number": item.issue_number,
            "pr_number": item.pr_number,
            "fix_actions": list(item.fix_actions),
            "detail": item.detail,
        }
        if item.reason is not None:
            reconcile_payload["reason"] = item.reason
        if item.failure_kind is not None:
            reconcile_payload["failure_kind"] = item.failure_kind
        new_state = append_event(
            new_state,
            "reconcile",
            reconcile_payload,
            state_path=state_path,
        )

    return new_state
