from __future__ import annotations

import functools
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace as dataclasses_replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adapters import (
    AdapterSettings,
    SessionDispatchResult,
    SessionRequest,
    dispatch_sessions,
    write_session_manifest,
    write_session_results,
)
from .api_budget import budget_status as _api_budget_status
from .api_budget import ledger_path as _api_ledger_path
from .api_budget import load_ledger as _api_load_ledger
from .claude_code import (
    _events_path,
    extract_event_text,
    iter_stream_json_events,
    launch_claude_worker,
    parse_claude_events,
    resolve_review_effort,
    run_quota_probe,
)
from .checks import CheckSummary, summarize_checks
from .config import (
    AutoMergeConfig,
    CrossFamilyConfig,
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
)
from .file_lock import try_acquire_byte_range_lock
from .fleet_registry import count_fleet_live_sessions, try_acquire_fleet_lock
from . import layout
from .notify import AttentionDigest, AttentionEntry, emit_digest
from . import rescue as rescue_helpers
from .subprocess_runner import no_console_window_kwargs
from .cross_family import (
    CrossFamilyResult,
    extract_head_ref_oid,
    extract_report_body,
    parse_cross_family_verdict,
    report_body_is_valid,
    run_cross_family_review,
)
from .github import (
    GitHub,
    GitHubError,
    GitHubNotFoundError,
    GitHubRunResult,
    GraphQLBudgetError,
    cancel_superseded_runs,
    detect_prose_only_dependencies,
    get_github_issue_dependencies,
    is_infrastructure_failure,
    issue_numbers_mentioned_by_pr,
    label_names,
    linked_issue_number,
    parse_blockers,
)
from .janitor import (
    _calculate_patch_id,
    _diff_content_signature,
    check_operator_containment,
    check_test_adequacy,
    detect_cross_pr_revert,
    run_janitor,
    DiffContentSignature,
    TestAdequacyFacts,
    TestAdequacyVerdict,
)
from .labels import TransitionOutcome, transition
from .paths import ResolvedLayout, RuntimePaths, resolved_layout
from .prompts import render_prompt
from .reconcile import (
    DriftItem,
    apply_fixes as apply_drift_fixes,
    detect_aviator_stale_blocked,
    detect_drift,
)
from .worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    clean_worktrees,
    inspect_worktree_state,
    push_branch,
    remove_review_checkout,
    remove_worktree_marker,
    resolve_base_branch_name,
    worktree_path_for_branch,
    write_worktree_marker,
)
from . import state as _state
from .state import (
    PASSIVE_OPEN_STATUS,
    StateLockBusy,
    _REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES,
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
    append_event,
    arm_quota_probe,
    arm_reconcile_pass,
    clear_quota_throttles,
    clear_reviewer_quota,
    defer_reviewer_probe_after,
    disarm_quota_probe,
    is_claim_stale,
    is_quota_probe_actionable,
    is_quota_probe_armed,
    is_quota_probe_due,
    is_reconcile_due,
    is_reviewer_probe_ready,
    is_reviewer_quota_exhausted,
    is_throttled,
    is_worktree_reclamation_due,
    load_state,
    load_state_locked,
    mark_reviewer_quota_alerted,
    operator_claimed_issues,
    release_operator_claimed,
    save_state,
    schedule_worktree_reclamation,
    set_operator_claimed,
    set_reviewer_quota_exhausted,
    set_throttled_until,
    stale_operator_claims,
    state_lock,
    utc_now,
    without_review_dispatch_claim,
)
from .instrumentation import correlation_context, log_event, record_loop_pass
from .throttle_signatures import match_throttle_tail, parse_reset_clock_time
from .process_utils import is_pid_alive, kill_process_tree, sweep_orphan_processes
from .worker import WorkerHealth, WorkerView, _alive_review_worker_issue_numbers, iter_workers
from .routing import AdapterChoice, record_adapter_choice, select_adapter


def _diff_file_summary(diff: str) -> tuple[int, list[tuple[str, int, int]]]:
    """Return (total_lines, per_file_stats) from a unified diff.

    ``per_file_stats`` is a list of ``(filename, added, deleted)`` tuples.
    ``total_lines`` counts content lines (not diff headers/meta lines).
    """
    files: list[tuple[str, int, int]] = []
    current_file = ""
    added = 0
    deleted = 0
    total = 0
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current_file:
                files.append((current_file, added, deleted))
            current_file = ""
            added = 0
            deleted = 0
        elif line.startswith("+++ "):
            current_file = line[4:].strip()
            if current_file == "/dev/null":
                current_file = ""
        elif line.startswith("--- "):
            # Use the source file if the dest is /dev/null (deletion)
            if not current_file:
                current_file = line[4:].strip()
                if current_file == "/dev/null":
                    current_file = ""
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
            total += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            total += 1
    if current_file:
        files.append((current_file, added, deleted))
    return total, files


def _diff_size_section(diff: str, threshold: int, diff_path: Path) -> str:
    """Return a prompt section warning about large diffs, or empty string.

    When the diff exceeds ``threshold`` content lines, this returns a Markdown
    section with a per-file summary and instructions to read the diff
    file-by-file rather than in one shot. When the diff is small or the
    threshold is 0, returns an empty string.
    """
    if threshold <= 0:
        return ""
    total, files = _diff_file_summary(diff)
    if total <= threshold:
        return ""
    lines = [
        "",
        "## Large diff guidance",
        "",
        f"This diff has {total} changed lines across {len(files)} file(s). "
        f"Do **not** read the entire diff in one pass — it will waste your "
        f"token budget. Instead, read `$diff_path` file-by-file, starting "
        f"with the files that have the most changes.",
        "",
        "| File | + | - |",
        "|------|---|---|",
    ]
    for name, add, dele in sorted(files, key=lambda x: x[1] + x[2], reverse=True):
        lines.append(f"| `{name}` | +{add} | -{dele} |")
    lines.append("")
    return "\n".join(lines)


# Fields the reviewer actually needs from pr.json. Excludes large fields that
# bloat the reviewer's context without aiding the review: ``comments`` (can be
# huge on active PRs), ``statusCheckRollup`` (separate checks.json is written),
# and ``createdAt``/``updatedAt`` (not used by the review rubric).
_PR_SLIM_FIELDS: frozenset[str] = frozenset(
    {
        "number",
        "title",
        "url",
        "body",
        "headRefOid",
        "baseRefName",
        "headRefName",
        "isDraft",
        "state",
        "labels",
        "author",
        "additions",
        "deletions",
        "mergeable",
        "mergeStateStatus",
        "isCrossRepository",
        "reviewDecision",
    }
)


def _slim_pr_json(pr: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``pr`` with only the fields the reviewer needs.

    Strips ``comments``, ``statusCheckRollup``, ``createdAt``, ``updatedAt``
    and other large fields that inflate the reviewer's token budget without
    aiding the review. The full PR data is not needed by the reviewer prompt
    (which references the file path, not inline content), and the reviewer
    reads the diff from ``diff.patch`` and checks from ``checks.json``.
    """
    return {k: v for k, v in pr.items() if k in _PR_SLIM_FIELDS}


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any]


def _state_lock_busy_result(message: str, **extra: Any) -> CommandResult:
    data: dict[str, Any] = {
        "skipped": True,
        "reason": "state_lock_busy",
        "state_lock_busy": True,
    }
    data.update(extra)
    return CommandResult(True, message, data)


def _truncate_reason(reason: str, max_len: int = 200) -> str:
    if len(reason) <= max_len:
        return reason
    return reason[: max_len - 3] + "..."


def _dispatch_failure_reason(result: SessionDispatchResult) -> str:
    if result.error:
        return result.error
    if result.failure_kind:
        return f"dispatch failed: {result.failure_kind}"
    return "dispatch failed"


def _label_error_reason(label_error: dict[str, Any]) -> str:
    """Format a persisted label_error dict into a short, human-readable reason."""
    edge = label_error.get("edge", "unknown")
    outcome = label_error.get("outcome", "unknown")
    add_failures = label_error.get("add_failures") or []
    remove_failures = label_error.get("remove_failures") or []
    add_labels = [label for _issue, label in add_failures]
    remove_labels = [label for _issue, label in remove_failures]
    parts = [f"label transition '{edge}' {outcome}"]
    if add_labels:
        parts.append(f"add failures: {add_labels}")
    if remove_labels:
        parts.append(f"remove failures: {remove_labels}")
    return "; ".join(parts)


def _parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp from state.json into a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if not isinstance(value, str):
        return None
    ts = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _recent_dispatch_failed_attempts(
    entry: dict[str, Any], now: datetime, window_minutes: int
) -> list[str]:
    """Return ``dispatch_failed_at`` entries inside the redispatch window."""
    attempts = entry.get("dispatch_failed_at") or []
    if not isinstance(attempts, (list, tuple)):
        return []
    window_start = now - timedelta(minutes=window_minutes)
    recent: list[str] = []
    for value in attempts:
        ts = _parse_iso_timestamp(value)
        if ts is not None and ts >= window_start:
            recent.append(value)
    return recent


def _build_failure_map(
    dispatch_results: Sequence[SessionDispatchResult],
    failed_issue_numbers: Iterable[int],
    deferred_by_concurrency: Iterable[int],
    limit: int,
    extra_failures: Mapping[int, str] | None = None,
) -> dict[int, str]:
    failures: dict[int, str] = {}
    for issue_number in deferred_by_concurrency:
        failures[issue_number] = _truncate_reason(f"deferred by concurrency cap (limit: {limit})")
    for result in dispatch_results:
        if result.issue_number in failed_issue_numbers:
            failures[result.issue_number] = _truncate_reason(_dispatch_failure_reason(result))
    for issue_number, reason in (extra_failures or {}).items():
        failures[issue_number] = _truncate_reason(reason)
    return failures


def _guard_state_lock(func: Any) -> Any:
    """Decorator that turns StateLockBusy into a skipped CommandResult."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> CommandResult:
        try:
            return func(self, *args, **kwargs)
        except StateLockBusy:
            return _state_lock_busy_result("state lock held, skipped")

    return wrapper


@dataclass(frozen=True)
class ConcurrencyGovernorResult:
    """Result of applying concurrency governor to a dispatch limit.

    This encapsulates the concurrency limiting logic and ensures all related
    fields are bound together, eliminating Pyright's reportPossiblyUnbound
    warnings for live_count.
    """

    clamped: bool
    max_concurrent: int
    live_count: int
    available_slots: int
    dispatch_limit: int
    fleet_live_count: int = 0
    fleet_max: int = 0

    @property
    def enabled(self) -> bool:
        """Return True if the governor is enabled (max_concurrent > 0)."""
        return self.max_concurrent > 0

    @property
    def fleet_enabled(self) -> bool:
        """Return True if the fleet governor is enabled (fleet_max > 0)."""
        return self.fleet_max > 0

    def report_fields(self) -> dict[str, int]:
        """Return the fields to include in CommandResult.data when clamped."""
        fields = {
            "concurrency_limit": self.max_concurrent,
            "live_session_count": self.live_count,
            "available_slots": self.available_slots,
        }
        if self.fleet_enabled:
            fields["fleet_concurrency_limit"] = self.fleet_max
            fields["fleet_live_session_count"] = self.fleet_live_count
        return fields


@dataclass(frozen=True)
class LocalReviewCapResult:
    """Result of applying the local-only review process cap (issue #370).

    Unlike ``ConcurrencyGovernorResult``, this is not a provider-rate-limit
    clamp: it only bounds the number of concurrent local Claude Code reviewer
    processes, mirroring ``max_local_review_processes`` in
    ``ReviewDispatchConfig``. A value of 0 means unlimited.
    """

    clamped: bool
    max_local: int
    live_count: int
    available_slots: int
    dispatch_limit: int

    def report_fields(self) -> dict[str, int]:
        """Return the fields to include in CommandResult.data."""
        return {
            "max_local_review_processes": self.max_local,
            "live_review_count": self.live_count,
            "available_review_slots": self.available_slots,
        }


@dataclass(frozen=True)
class CarryForwardCheck:
    """Result of comparing a recorded review verdict's content against the
    live PR diff (issues #411/#412 tier 1, #414 tier 2).

    ``tier`` is ``"patch-id"`` when the live diff's stable patch-id matches
    the recorded ``reviewed_patch_id`` outright (issue #412's fast path),
    ``"line-content"`` when the patch-ids differed — which happens on every
    ordinary main advance, since the merge-base moves — but the ordered
    ``+``/``-`` line stream and changed-file set are identical to what was
    recorded at review time (issue #414), or ``None`` when neither tier
    establishes content identity: the caller must treat the verdict as
    stale. ``live_patch_id``/``live_signature`` are always populated (when
    the live diff was fetched) so a carrying-forward caller can persist the
    freshly computed baseline against the new head without recomputing it.
    """

    tier: str | None
    live_patch_id: str
    live_signature: DiffContentSignature

    @property
    def carry_forward(self) -> bool:
        return self.tier is not None


def _janitor_section(warnings: tuple[str, ...]) -> str:
    if not warnings:
        return ""
    lines = "\n".join(f"- {warning}" for warning in warnings)
    return (
        "\n## Janitor warnings (non-blocking)\n\n"
        f"{lines}\n\n"
        "These deterministic pre-checks passed the gate but deserve reviewer attention.\n"
    )


def _ci_status_section(
    checks: list[dict[str, Any]] | None,
    required: tuple[str, ...],
    checks_json_path: Path,
) -> str:
    """Render the $ci_status_section packet block from already-fetched CI data.

    ``run_janitor`` deterministically verifies required checks BEFORE a review
    packet is ever generated: a definitive required-check failure short-
    circuits ``review()`` long before this function is reached (see the
    ``janitor_blocked`` branch). So a reviewer re-reading ``checks.json`` to
    re-confirm what the gate already verified is pure token waste. This
    section states that verification result inline instead, while still
    surfacing everything the gate does NOT resolve: unfetchable CI data, an
    unconfigured required-check list (the gate is a no-op in that case),
    still-pending required checks, and failing non-required/informational
    checks (the gate never blocks on those).

    Pure and I/O-free like ``_janitor_section`` — safe to call every pass.
    """
    if checks is None:
        return (
            "CI status could not be fetched by the orchestrator (`gh` failure). "
            f"Do not assume checks are green — inspect `{checks_json_path}` "
            "directly if CI status matters to this review.\n"
        )

    if not required:
        return (
            "No required checks are configured for this repo, so CI status was "
            "not deterministically verified before dispatch. Inspect "
            f"`{checks_json_path}` if CI status is relevant to your review.\n"
        )

    summary = summarize_checks(checks, required)
    lines: list[str] = []
    if summary.passed:
        lines.append(
            f"Required check(s) passing — verified deterministically by the "
            f"orchestrator before dispatch: {', '.join(summary.passed)}. Do "
            "not spend turns re-inspecting these."
        )
    if summary.pending:
        lines.append(
            f"Required check(s) still pending, not yet confirmed: {', '.join(summary.pending)}."
        )
    lines.append(f"`checks.json` is available at `{checks_json_path}` if a specific doubt arises.")

    non_required_failing, non_required_cancelled = _non_required_check_findings(checks, required)
    if non_required_failing:
        lines.append(
            "Non-required/informational check(s) currently failing (the "
            "janitor gate does not block on these — weigh them yourself): "
            + ", ".join(non_required_failing)
        )
    if non_required_cancelled:
        lines.append(
            "Non-required/informational check(s) cancelled (often infra-transient, "
            "not necessarily a code failure — weigh them yourself): "
            + ", ".join(non_required_cancelled)
        )

    return "\n".join(lines) + "\n"


def _non_required_check_findings(
    checks: list[dict[str, Any]], required: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify non-required checks into (failing, cancelled) name lists.

    Deliberately does NOT reuse ``summarize_checks`` here: its ``else:
    name_failed = True`` catch-all (checks.py) is correct for REQUIRED checks
    (any non-passing/non-pending/non-infra state should gate the janitor) but
    wrong for informational awareness of non-required checks, where it
    silently swept SKIPPED and NEUTRAL conclusions (a path-filtered or
    matrix-conditional job that correctly did not run) into "failed", making
    every such PR's packet falsely claim an unrelated check was "currently
    failing". ``summarize_checks`` itself must not change — it is the
    janitor's required-check semantics — so this classifies non-required
    checks directly from their raw per-run state/bucket instead.

    A genuine failure is FAILURE, INFRA_FAILURE, or any other unrecognized
    terminal state (e.g. TIMED_OUT, ACTION_REQUIRED) — the same "anything
    else is a real failure" posture ``summarize_checks`` takes, minus the
    SKIPPED/NEUTRAL carve-out. CANCELLED is reported separately (worded as
    "cancelled," never "failing") since it is frequently an infra hiccup
    rather than a code problem. Multiple runs sharing a name use worst-of
    semantics, mirroring ``summarize_checks``.
    """
    required_set = set(required)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        name = str(check.get("name") or "")
        if not name or name in required_set:
            continue
        by_name.setdefault(name, []).append(check)

    failing: list[str] = []
    cancelled: list[str] = []
    for name, runs in by_name.items():
        name_failed = False
        name_cancelled = False
        for check in runs:
            state = str(check.get("state") or "").upper()
            bucket = str(check.get("bucket") or "").lower()
            if state == "SUCCESS" or bucket == "pass":
                continue
            if state in {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED"} or bucket == "pending":
                continue
            if not state and not bucket:
                continue
            if state in {"SKIPPED", "NEUTRAL"}:
                # Legitimate non-outcomes (path-filtered/matrix-conditional
                # jobs) — never a failure.
                continue
            if state == "CANCELLED":
                name_cancelled = True
                continue
            # FAILURE, INFRA_FAILURE, or any other unrecognized state
            # (TIMED_OUT, ACTION_REQUIRED, ...): a genuine failure.
            name_failed = True
        if name_failed:
            failing.append(name)
        elif name_cancelled:
            cancelled.append(name)

    return tuple(sorted(failing)), tuple(sorted(cancelled))


def render_test_adequacy_section(
    facts: TestAdequacyFacts | None, warnings: tuple[str, ...]
) -> str:
    """Render the $test_adequacy_section packet block from Tier-1 facts.

    Returns "" when facts is None (gate disabled — caller in review() passes
    None in that case) or when there is nothing to report. Mirrors the
    _janitor_section pattern: plain function, no I/O, safe to call every pass.
    """
    if facts is None:
        return ""
    lines = [
        "## Test-adequacy facts (Tier 1, deterministic)",
        "",
        f"- Added product LOC: {facts.added_product_loc}",
        f"- Added test LOC: {facts.added_test_loc}",
        f"- Assertion-bearing added test lines: {facts.assertion_count}",
        f"- Test files changed: {facts.test_files_changed}",
    ]
    if facts.untested_product_files:
        lines.append("- Untested product files: " + ", ".join(facts.untested_product_files))
    if facts.exempt:
        lines.append(f'- Test-exempt claim: "{facts.exempt_reason}" (verify against the diff)')
    if warnings:
        lines.append("")
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def render_test_adequacy_summary(verdict: TestAdequacyVerdict, exempt_marker: str) -> str:
    """Render a test-adequacy hard-fail verdict as actionable reviewer feedback.

    This summary is passed to record_review as the review summary, which is
    rendered into rework.md. It must read as actionable feedback, not a raw
    dataclass dump.

    Args:
        verdict: The hard-fail TestAdequacyVerdict (ok=False).
        exempt_marker: The exempt marker string from config (e.g., "Test-exempt:").

    Returns:
        A non-empty templated string with untested product files and exemption
        instruction.
    """
    facts = verdict.facts
    untested_files = facts.untested_product_files
    added_loc = facts.added_product_loc

    # Build file list with LOC counts
    file_list = "\n".join(f"  - {file}" for file in untested_files)

    return (
        f"Test adequacy check failed: {added_loc} lines of product code added "
        f"but no test files changed.\n\n"
        f"Untested product files:\n{file_list}\n\n"
        f"To exempt this PR from the test-adequacy gate, add "
        f"'{exempt_marker} <reason>' to the PR body with a clear justification."
    )


def slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "work"


def parse_issue_numbers(only_issues: str) -> list[int]:
    return [int(part) for part in only_issues.replace(" ", "").split(",") if part]


# Maximum recovery-retry candidates allowed per dispatch pass. Recovery retries
# must not consume the same budget as fresh candidates; capping them at one per
# pass prevents one stuck recovery candidate from starving the queue (issue #506).
_MAX_RECOVERY_RETRY_PER_PASS = 1


def _is_recovery_candidate(
    issue: dict[str, Any],
    state: dict[str, Any],
    branch_name_for: Callable[[dict[str, Any]], str],
) -> bool:
    """Return True when ``issue`` is a recovery retry of a previous dispatch.

    A recovery candidate has a state.json entry whose status is ``"dispatched"``
    and whose stored branch_name matches the branch that would be generated for
    the issue today. These candidates are separated from fresh dispatch
    candidates so they cannot monopolize the dispatch budget.
    """
    issue_number = int(issue["number"])
    prev_entry = state.get("issues", {}).get(str(issue_number), {})
    if prev_entry.get("status") != "dispatched":
        return False
    prev_branch = prev_entry.get("branch_name")
    return prev_branch == branch_name_for(issue)


def _select_dispatch_candidates(
    candidates: list[dict[str, Any]],
    dispatch_limit: int,
    state: dict[str, Any],
    branch_name_for: Callable[[dict[str, Any]], str],
    only_issues: str | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Select dispatch candidates, filling fresh slots before recovery retries.

    Fresh candidates are dispatched first; recovery-retry candidates are only
    attempted with remaining slots, and at most one recovery retry is attempted
    per pass. This prevents a stuck recovery candidate from head-of-line
    blocking fresh work under a tight dispatch limit (issue #506).

    Args:
        candidates: Unblocked, sorted candidate issues from GitHub.
        dispatch_limit: Maximum number of issues to select this pass.
        state: Current state.json snapshot for recovery classification.
        branch_name_for: Callable that returns the branch name for an issue.
        only_issues: Optional explicit comma-separated issue numbers to select.

    Returns:
        Tuple of (selected, skipped_issue_numbers, deferred_by_concurrency).
    """
    if only_issues:
        wanted = parse_issue_numbers(only_issues)
        by_number = {int(issue["number"]): issue for issue in candidates}
        ordered = [by_number[number] for number in wanted if number in by_number]
        skipped_issue_numbers = sorted(set(wanted) - set(by_number))
    else:
        ordered = candidates
        skipped_issue_numbers = []

    recovery_flags = [_is_recovery_candidate(issue, state, branch_name_for) for issue in ordered]
    fresh_count = sum(1 for r in recovery_flags if not r)
    recovery_cap = min(
        _MAX_RECOVERY_RETRY_PER_PASS,
        max(0, dispatch_limit - fresh_count),
    )

    selected: list[dict[str, Any]] = []
    recovery_picked = 0
    if only_issues:
        # Preserve the operator's explicit issue order while still capping
        # recovery retries so a stuck recovery issue cannot starve fresh work.
        for issue, is_recovery in zip(ordered, recovery_flags):
            if len(selected) >= dispatch_limit:
                break
            if is_recovery:
                if recovery_picked < recovery_cap:
                    selected.append(issue)
                    recovery_picked += 1
            else:
                selected.append(issue)
    else:
        fresh = [issue for issue, r in zip(ordered, recovery_flags) if not r]
        recovery = [issue for issue, r in zip(ordered, recovery_flags) if r]
        # Fill fresh first, then allow at most one recovery-retry slot.
        selected = fresh[:dispatch_limit] + recovery[:recovery_cap]

    if only_issues:
        selected_numbers = {int(issue["number"]) for issue in selected}
        deferred_by_concurrency = [
            int(issue["number"])
            for issue in ordered
            if int(issue["number"]) not in selected_numbers
        ]
    else:
        deferred_by_concurrency = []

    return selected, skipped_issue_numbers, deferred_by_concurrency


def _count_live_sessions(sessions_dir: Path, state_file: Path | None = None) -> int:
    """Count the number of currently alive worker sessions across both adapters.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the total count of sessions with alive PIDs.

    When ``state_file`` is given, this also corroborates the sidecar-based
    count against state.json's own dispatched-issue ``worker_pid``/
    ``worker_process_start_time`` records (issue #343). A sidecar can go
    missing for a still-live process -- a "ghost" -- if a reap lane removes
    it on ambiguous evidence (see ``_classify_dead_sessions_and_update_
    throttle_state``'s corroboration gate) or through any other path that
    strands state.json's dispatch record. Because the governor's dispatch
    cap (``_apply_concurrency_governor``) is built on top of this count, a
    ghost previously made a live process invisible to concurrency accounting
    and let the next pass over-dispatch past the configured cap. state.json's
    worker_pid fields are not touched by sidecar reaping, so any issue still
    recorded as ``dispatched`` whose worker_pid is alive (pid + start-time,
    recycling-safe) but has no corresponding live sidecar is counted here
    too, and printed as a loud reconcile signal rather than silently treated
    as free capacity.
    """
    from .worker import iter_workers

    live_issue_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(sessions_dir):
        if w.is_alive():
            live_count += 1
            live_issue_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
        for issue_number_str, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict) or entry.get("status") != "dispatched":
                continue
            try:
                issue_number = int(issue_number_str)
            except (TypeError, ValueError):
                continue
            if issue_number in live_issue_numbers:
                continue
            if _worker_pid_alive(entry):
                live_count += 1
                print(
                    f"[reconcile] issue {issue_number}: worker_pid "
                    f"{entry.get('worker_pid')} is alive with no live session "
                    "sidecar (ghost) -- counting it against the concurrency "
                    "governor instead of treating the slot as free",
                    flush=True,
                )

    return live_count


def _detect_stalled_sessions(
    sessions_dir: Path, config: OrchestratorConfig
) -> list[dict[str, Any]]:
    """Detect stalled sessions (live PID but dead agent) without handling them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. This is a read-only detection function for status/roll-call.

    Returns a list of {issue, pid, health, terminal_tool, terminal_reason}
    dicts for stalled sessions. ``health`` distinguishes STALLED from DEAD
    (issue #261) so digest callers can surface dead-worker terminal cause
    instead of collapsing everything to "STALLED". ``terminal_tool``/
    ``terminal_reason`` are populated only for DEAD entries with a matching
    post-mortem sidecar (best-effort — absent when extraction found nothing).
    """
    from datetime import UTC, datetime
    from .post_mortem import read_post_mortem
    from .worker import classify_worker_health, iter_workers, real_activity_probe_for

    if not config.watchdog.enabled:
        return []

    stalled_entries: list[dict[str, Any]] = []
    now = datetime.now(UTC)

    for w in iter_workers(sessions_dir):
        if w.pid is None or w.error is not None:
            continue

        # Issues #280/#301: corroborate against real-session activity for read-only
        # detection too (status/dry-run/digest).
        probe = real_activity_probe_for(w, config, now)
        health = classify_worker_health(w, config, now, probe)

        # Both STALLED and DEAD are considered "stalled" for reporting purposes
        if health in (WorkerHealth.STALLED, WorkerHealth.DEAD):
            entry: dict[str, Any] = {
                "issue": w.issue_number,
                "pid": w.pid,
                "health": health.name,
            }
            if health is WorkerHealth.DEAD:
                post_mortem = read_post_mortem(sessions_dir, w.issue_number)
                if post_mortem is not None:
                    entry["terminal_tool"] = post_mortem.terminal_tool
                    entry["terminal_reason"] = post_mortem.terminal_reason
            stalled_entries.append(entry)

    return stalled_entries


def _kill_orphan_pid(pid: int) -> None:
    """Best-effort kill of a single orphan PID, cross-platform.

    Mirrors the OS branch used by kill_process_tree: taskkill on Windows,
    os.kill(SIGKILL) on POSIX. Never raises - callers treat this as best-effort
    and always record the PID as killed regardless of outcome.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                **no_console_window_kwargs(),
            )
        else:
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        # Best-effort kill - don't fail if the kill attempt fails
        pass


def _detect_and_handle_stalled_sessions(
    sessions_dir: Path, state_file: Path, config: OrchestratorConfig
) -> list[dict[str, int]]:
    """Detect stalled sessions (live PID but dead agent) and handle them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. On detection, the process tree is killed and the sidecar is
    classified via the log-tail-first helper (``update_session_record_with_
    failure_classification`` / ``update_worker_record_with_failure_classification``),
    falling back to failure_kind "stalled" only when the log shows no provider
    throttle signature. This function runs before the dead-session lane in the
    loop() pass order, so it must apply the same classify-then-fallback
    treatment itself — otherwise a worker that actually died on a provider
    rate limit gets permanently mislabeled "stalled" before the dead-session
    lane ever gets a chance to classify it, and ``throttled_until`` never gets
    set (issue #246).

    When classification detects a throttle signature, ``throttled_until`` is
    persisted to state.json (same as the dead-session lane) so the next
    dispatch pass defers instead of relaunching into the same window. A
    session_stalled event is logged with the resolved failure_kind.

    Returns a list of {issue, pid} dicts for stalled sessions (for exclusion from
    dispatch in the same pass).
    """
    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import (
        get_rate_limit_defer_until,
        update_session_record_with_failure_classification,
    )
    from .post_mortem import classify_and_record
    from .worker import (
        _next_inconclusive_probe_deferred_count,
        _api_session_over_budget,
        classify_worker_health,
        iter_workers,
        real_activity_probe_for,
        update_worker_log_stat,
    )

    if not config.watchdog.enabled:
        return []

    stalled_entries: list[dict[str, int]] = []
    now = datetime.now(UTC)

    def _parse_defer_until(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except (ValueError, TypeError):
            return None

    def _is_deferred(w: Any) -> bool:
        """Return True if the worker has a stored defer deadline that has not passed."""
        defer_until = _parse_defer_until(w.rate_limit_defer_until)
        if defer_until is None:
            return False
        return now < defer_until

    for w in iter_workers(sessions_dir):
        if w.pid is None or w.error is not None:
            continue

        # Update log stat fields for progress tracking. This also clears any
        # stale rate-limit defer deadline when the log has resumed growing.
        update_worker_log_stat(sessions_dir, w)

        # Issues #280/#301: corroborate sidecar mtime against real-session activity
        # (sessions.db message_nodes, per-PID Devin log mtime, and Claude Code
        # events.jsonl) before deciding whether to kill the worker.
        probe = real_activity_probe_for(w, config, now)
        health = classify_worker_health(w, config, now, probe)

        # Issue #338: persist Signal-1's inconclusive-probe deferral counter so the
        # escalation cap is tracked across passes. This lane is the sole writer of
        # this counter for a not-alive worker (issue #343 Finding 2): it is the
        # only lane guaranteed to run at least once whenever the watchdog is
        # enabled -- dispatch()/dispatch_rework() each call this lane standalone
        # (no dead-session lane in the same call), and loop() always runs this
        # lane immediately before the dead-session lane
        # (_classify_dead_sessions_and_update_throttle_state). The dead lane
        # deliberately does NOT also persist this counter for a not-alive worker
        # when the watchdog is enabled -- see the comment there -- so it is
        # written at most once per worker per pass instead of twice (0->1 here,
        # then re-read and ->2 there), which halved the effective deferral grace
        # period and was the very mechanism that opened Finding 1's pass-2
        # phantom-sidecar window.
        new_count = _next_inconclusive_probe_deferred_count(w, probe, health)
        update_worker_log_stat(sessions_dir, w, inconclusive_probe_deferred_count=new_count)

        # Issue #484: in-flight api per-session budget kill. Independent of the
        # STALLED/DEAD classification below — an api worker over its
        # ``max_usd_per_session`` cap is killed immediately and sidecar-marked
        # ``budget_exceeded``. The killed session then flows through the
        # EXISTING dead-worker reconciliation on the next pass (with-PR ->
        # review/rework; without-PR -> re-dispatch via select_adapter, whose
        # preflight naturally decides api-again vs fallback). When the cap is
        # 0/unset the check is entirely dormant. Non-api workers are never
        # budget-evaluated. The kill uses the shared ``kill_process_tree``
        # helper (no-console-window discipline on Windows, full process tree
        # reaped) — not reimplemented here.
        if w.adapter_kind == "api" and _api_session_over_budget(w, config):
            killed_pids = kill_process_tree(w.pid, w.process_start_time)
            orphan_pids_budget: list[int] = []
            orphan_processes = sweep_orphan_processes(w.worktree_path)
            if orphan_processes:
                for orphan in orphan_processes:
                    _kill_orphan_pid(orphan["pid"])
                    killed_pids.append(orphan["pid"])
                orphan_pids_budget = [o["pid"] for o in orphan_processes]

            # Set failure_kind="budget_exceeded" on the sidecar via the shared
            # atomic-write helper. Written directly (not through
            # update_worker_record_with_failure_classification) so the
            # budget-exceeded verdict is not overridden by a coincidental
            # throttle/auth log-tail match. The dead-session lane's
            # classification call then short-circuits on the already-set
            # failure_kind.
            from .claude_code import _sidecar_path as _api_sidecar_path
            from .claude_code import _write_json_atomic as _api_write_json_atomic

            api_sidecar = _api_sidecar_path(sessions_dir, w.issue_number, "api")
            try:
                with api_sidecar.open("r", encoding="utf-8") as handle:
                    api_payload = json.load(handle)
                if isinstance(api_payload, dict):
                    api_payload["failure_kind"] = "budget_exceeded"
                    _api_write_json_atomic(api_sidecar, api_payload)
            except (OSError, json.JSONDecodeError):
                pass

            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "session_budget_exceeded",
                    {
                        "issue_number": w.issue_number,
                        "pid": w.pid,
                        "process_start_time": w.process_start_time,
                        "killed_pids": killed_pids,
                        "orphan_pids": orphan_pids_budget if orphan_pids_budget else None,
                        "provider": w.provider,
                    },
                    state_path=state_file,
                )
                save_state(state_file, state)

            stalled_entries.append({"issue": w.issue_number, "pid": w.pid})
            continue

        # If a stalled-looking worker is still within a previously stored rate-limit
        # defer window, skip it. The deadline is re-derived from the sidecar each pass.
        if health == WorkerHealth.STALLED and _is_deferred(w):
            continue

        # Both STALLED and DEAD are considered "stalled" for handling purposes
        if health in (WorkerHealth.STALLED, WorkerHealth.DEAD):
            # Issue #247: before killing a stalled-looking worker, check the log tail
            # for a rate-limit signature. If found and we are not already in a defer
            # window, record a defer deadline and skip the kill this pass.
            if health == WorkerHealth.STALLED and config.watchdog.rate_limit_defer_enabled:
                if not _is_deferred(w) and w.rate_limit_defer_until is None:
                    defer_until = get_rate_limit_defer_until(
                        Path(w.log_path),
                        config.watchdog.rate_limit_defer_slack_minutes,
                        now,
                        config.runtime.throttle_error_markers,
                        config.runtime.throttle_resume_margin_s,
                    )
                    if defer_until is not None:
                        update_worker_log_stat(sessions_dir, w, rate_limit_defer_until=defer_until)
                        with state_lock(state_file):
                            state = load_state(state_file)
                            state = set_throttled_until(
                                state,
                                defer_until,
                                reason="rate_limited",
                                adapter_kind=w.adapter_kind,
                            )
                            state = append_event(
                                state,
                                "session_rate_limit_deferred",
                                {
                                    "issue_number": w.issue_number,
                                    "pid": w.pid,
                                    "defer_until": defer_until,
                                },
                                state_path=state_file,
                            )
                            save_state(state_file, state)
                        continue

            # Kill the process tree (with start-time verification to prevent PID recycling)
            killed_pids = kill_process_tree(w.pid, w.process_start_time)

            # Sweep for orphan processes that survived the tree kill (Windows-only)
            # This catches detached/daemonized processes (e.g., nohup-style background processes)
            orphan_pids: list[int] = []
            orphan_processes = sweep_orphan_processes(w.worktree_path)
            if orphan_processes:
                # Kill detected orphans to prevent them from running rejected code
                for orphan in orphan_processes:
                    _kill_orphan_pid(orphan["pid"])
                    killed_pids.append(orphan["pid"])
                orphan_pids = [o["pid"] for o in orphan_processes]

            # Post-mortem extraction (issue #261): reads the Devin CLI's own
            # session store for a terminal-tool diagnosis (esp. decision:block
            # push-gate hooks) independent of the log tail. Runs BEFORE the
            # log-tail classification below — when it detects a block, it
            # writes failure_kind="worker_blocked" directly into the sidecar,
            # which makes the classification call below a no-op via its
            # existing "skip if already classified" short-circuit. Best-effort
            # and read-only: any DB problem degrades to extraction_error and
            # this never changes what happens next.
            classify_and_record(sessions_dir, config, w, now=now)

            # Classify the sidecar (adapter-specific dispatch): log-tail
            # classification runs first, falling back to failure_kind "stalled"
            # only when the log shows no provider throttle signature.
            resolved_failure_kind: str | None = None
            throttled_until: str | None = None
            if w.adapter_kind == "devin":
                resolved_failure_kind, throttled_until = (
                    update_session_record_with_failure_classification(
                        sessions_dir,
                        w.issue_number,
                        fallback_kind="stalled",
                        config=config,
                    )
                )
            elif w.adapter_kind == "claude-code":
                resolved_failure_kind, throttled_until = (
                    update_worker_record_with_failure_classification(
                        sessions_dir,
                        w.issue_number,
                        fallback_kind="stalled",
                        config=config,
                    )
                )
            elif w.adapter_kind == "api":
                # api sidecars share the claude-code classification helper but
                # land as issue-<n>.api.json and get provider-auth classification
                # (issue #484). adapter_kind="api" selects both the sidecar
                # suffix and the auth-pattern check inside _classify_session_failure.
                resolved_failure_kind, throttled_until = (
                    update_worker_record_with_failure_classification(
                        sessions_dir,
                        w.issue_number,
                        fallback_kind="stalled",
                        config=config,
                        adapter_kind="api",
                    )
                )

            if resolved_failure_kind and throttled_until:
                # A throttle signature was found in the log tail even though
                # the watchdog reaped this worker for stalling — persist the
                # cooldown so the next dispatch pass defers instead of
                # relaunching into the same provider rate limit/quota window.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=resolved_failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    save_state(state_file, state)

            # Log the event
            log_path = Path(w.log_path)
            last_log_line = None
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = log_text.splitlines()
                if lines:
                    last_log_line = lines[-1].strip()
            except OSError:
                pass

            probe_payload = (
                probe.to_payload()
                if probe is not None
                else {
                    "sources": [],
                    "latest_timestamp": None,
                    "latest_source": "probe unavailable",
                }
            )

            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "session_stalled",
                    {
                        "issue_number": w.issue_number,
                        "pid": w.pid,
                        "process_start_time": w.process_start_time,
                        "log_mtime": str(datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)),
                        "last_log_line": last_log_line,
                        "killed_pids": killed_pids,
                        "orphan_pids": orphan_pids if orphan_pids else None,
                        "failure_kind": resolved_failure_kind,
                        "activity_sources": probe_payload.get("sources", []),
                        "latest_real_activity_at": probe_payload.get("latest_timestamp"),
                        "latest_real_activity_source": probe_payload.get("latest_source"),
                    },
                    state_path=state_file,
                )
                save_state(state_file, state)

            stalled_entries.append({"issue": w.issue_number, "pid": w.pid})

    return stalled_entries


def _worker_pid_alive(entry: dict[str, Any]) -> bool:
    """Check if a worker PID from state.json is alive, with start-time verification.

    This helper deduplicates the PID liveness check used across dispatch and
    orphaned worker detection. It checks both PID liveness and process identity
    via start time to detect PID recycling.

    Args:
        entry: A state.json issue entry containing worker_pid and optionally
               worker_process_start_time.

    Returns:
        True if the worker PID is alive and the start time matches (if available),
        False otherwise.
    """
    worker_pid = entry.get("worker_pid")
    if worker_pid is None:
        return False

    return is_pid_alive(worker_pid, entry.get("worker_process_start_time"))


def _reviewer_pid_alive(entry: dict[str, Any]) -> bool:
    """Check if a reviewer PID from state.json is alive, with start-time verification.

    Mirror of ``_worker_pid_alive`` for reviewer processes launched by
    ``dispatch_reviews``. A reviewer is a Claude Code worker whose PID and
    process start time are stored under ``reviewer_pid`` and
    ``reviewer_process_start_time`` in the per-PR state.
    """
    reviewer_pid = entry.get("reviewer_pid")
    if reviewer_pid is None:
        return False

    return is_pid_alive(reviewer_pid, entry.get("reviewer_process_start_time"))


def _count_live_reviews(reviews_dir: Path, state_file: Path | None = None) -> int:
    """Count the number of currently alive reviewer sessions.

    Reads review sidecar files (claude-code only today) from ``reviews_dir``,
    then checks each record's PID liveness using the adapter-agnostic
    ``WorkerView.is_alive`` probe.

    When ``state_file`` is given, the count is corroborated against
    state.json's own ``review_dispatch_dispatched`` records so a missing
    sidecar doesn't let a live reviewer slip through the local cap.
    """
    live_pr_numbers: set[int] = set()
    live_count = 0
    for w in iter_workers(reviews_dir):
        if w.is_alive():
            live_count += 1
            live_pr_numbers.add(w.issue_number)

    if state_file is not None:
        state = load_state_locked(state_file)
        for pr_number_str, entry in state.get("prs", {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("review_dispatch_status") != "review_dispatch_dispatched":
                continue
            try:
                pr_number = int(pr_number_str)
            except (TypeError, ValueError):
                continue
            if pr_number in live_pr_numbers:
                continue
            if _reviewer_pid_alive(entry):
                live_count += 1
                print(
                    f"[reconcile] PR {pr_number}: reviewer_pid {entry.get('reviewer_pid')} "
                    "is alive with no live review sidecar (ghost) -- counting it against "
                    "the local review cap instead of treating the slot as free",
                    flush=True,
                )

    return live_count


def _apply_local_review_cap(
    dispatch_limit: int,
    max_local: int,
    live_count: int,
) -> LocalReviewCapResult:
    """Apply the local-only review process cap to a dispatch limit.

    ``max_local_review_processes`` is a per-host safety valve, not a provider
    governor. A value of 0 means unlimited. Unlike ``_apply_concurrency_governor``,
    this never consults fleet-wide concurrency or rate-limit state.
    """
    if max_local <= 0:
        return LocalReviewCapResult(
            clamped=False,
            max_local=0,
            live_count=live_count,
            available_slots=dispatch_limit,
            dispatch_limit=dispatch_limit,
        )

    available = max(0, max_local - live_count)
    clamped = available < dispatch_limit
    return LocalReviewCapResult(
        clamped=clamped,
        max_local=max_local,
        live_count=live_count,
        available_slots=available,
        dispatch_limit=min(dispatch_limit, available),
    )


def _windowed_redispatch_at(
    entry: dict[str, Any],
    *,
    window_minutes: int,
) -> list[str]:
    """Return redispatch timestamps within the configured window, type-safely.

    Normalizes ``entry["redispatch_at"]`` to a list of strings, filtering out
    non-string entries and timestamps older than ``window_minutes`` from now.
    This prevents crashes when the persisted value is corrupted (e.g., a string
    instead of a list — ``list("abc")`` would yield individual characters that
    crash ``datetime.fromisoformat``).
    """
    raw = entry.get("redispatch_at")
    if not isinstance(raw, list):
        return []
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        try:
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start:
                result.append(t)
        except (ValueError, AttributeError):
            continue
    return result


def _is_review_dispatchable(
    state: dict[str, Any],
    pr_number: int,
    candidate: dict[str, Any],
    *,
    max_attempts: int = 3,
) -> bool:
    """Return True if ``pr_number`` is free to receive a new reviewer dispatch.

    A PR is dispatchable when:
    - No prior review dispatch claim exists.
    - A prior claim is terminal (completed or stale-failed) and the stale timeout
      has elapsed, allowing retry.
    - A dispatched reviewer is no longer alive and its claim has gone stale.
    - The per-PR dispatch attempt count has not reached ``max_attempts``.

    This reuses ``is_claim_stale`` for the timeout and ``_reviewer_pid_alive``
    for liveness, avoiding a parallel mechanism.
    """
    pr_state = state["prs"].get(str(pr_number), {})
    status = pr_state.get("review_dispatch_status")

    # Per-PR dispatch attempt cap: a PR that has been dispatched max_attempts
    # times without producing a verdict is stuck (e.g. every reviewer hits the
    # session limit). Escalation is handled by the caller; here we just block
    # further dispatch.
    attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
    if attempt_count >= max_attempts:
        return False

    if status is None or status == "review_dispatch_completed":
        return True

    if status == "review_dispatch_pending":
        pending_at = pr_state.get("review_dispatch_pending_at")
        return pending_at is None or is_claim_stale(
            pending_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES
        )

    if status == "review_dispatch_dispatched":
        if _reviewer_pid_alive(pr_state):
            return False
        # Dead reviewer: the stalled-review sweep must disposition this claim
        # first — it classifies throttle tails, counts probe failures, reaps
        # the sidecar, and emits events; freeing here does none of that.
        # Racing the sweep on the same 5-minute timeout measured later in the
        # pass livelocked probes during closed quota windows (issue #571):
        # each silent relaunch reset the clock just after the sweep looked,
        # so backoff never engaged and a 429'd probe launched every pass.
        # The longer backstop frees only true orphans the sweep cannot see
        # (e.g. died before its sidecar was written).
        dispatched_at = pr_state.get("review_dispatched_at")
        return dispatched_at is None or is_claim_stale(
            dispatched_at, timeout_minutes=_REVIEW_DEAD_CLAIM_BACKSTOP_TIMEOUT_MINUTES
        )

    if status == "review_dispatch_failed":
        failed_at = pr_state.get("review_dispatch_failed_at")
        return failed_at is None or is_claim_stale(
            failed_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES
        )

    # Unknown status: treat as free so we don't silently orphan PRs.
    return True


_VERDICT_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)

# Absolute path ending in .md, as reviewers reference their summary files in
# final output (e.g. "Full review written to `C:\...\review.md`"). Colons,
# quotes, and whitespace terminate the match so "path:line" refs don't bleed.
_REVIEW_MD_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|~[\\/]|/)[^\s`\"'<>|*?:]+\.md")

_REVIEW_FALLBACK_FILE_MAX_BYTES = 1024 * 1024
_REVIEW_FALLBACK_MTIME_SLACK_S = 120
_REVIEW_FALLBACK_MAX_CANDIDATES = 8


def _validate_review_verdict(data: Any) -> dict[str, Any] | None:
    """Validate one decoded JSON candidate as a review verdict.

    A valid verdict must contain:

    - ``decision`` in ``{"approved", "request_changes", "blocked"}``
    - ``summary`` as a non-empty string, for EVERY decision including
      ``approved`` (issue #597), and not an unfilled ``<...>`` template
      placeholder
    - ``required_changes`` is optional; if present it must be a list of strings

    ``approved`` used to be exempt from the non-empty-summary rule, on the
    reasoning that ``record_review`` only rejects empty summaries where a
    reason is actionable. That exemption is what let a contentless approval
    through: an approval with no stated reason is indistinguishable from a
    reviewer that never formed an opinion, and approvals are the one decision
    that leads directly to a merge. Requiring a reason costs a reviewer one
    sentence; not requiring it cost ten unreviewed merges. A rejected verdict
    is fail-safe here -- the caller records no verdict and the review is
    retried, rather than merging on a verdict nobody stands behind.

    Returns the normalized verdict dict, or ``None`` if invalid.
    """
    if not isinstance(data, dict):
        return None

    decision = data.get("decision")
    if decision not in {"approved", "request_changes", "blocked"}:
        return None

    summary = data.get("summary")
    if not isinstance(summary, str):
        return None
    stripped_summary = summary.strip()
    if not stripped_summary:
        return None
    # An unfilled template placeholder ("<concise summary of the review>") is
    # prompt boilerplate that leaked into the verdict, never a real summary.
    if stripped_summary.startswith("<") and stripped_summary.endswith(">"):
        return None

    required_changes = data.get("required_changes")
    if required_changes is not None and not isinstance(required_changes, list):
        return None
    if required_changes is not None and not all(
        isinstance(item, str) for item in required_changes
    ):
        return None

    return {
        "decision": decision,
        "summary": summary,
        "required_changes": required_changes if required_changes is not None else [],
    }


def _extract_verdict_from_text(text: str) -> dict[str, Any] | None:
    """Extract the last valid fenced JSON verdict block from plain text.

    Accepts fences with or without a ``json`` language tag, scanning from the
    last fence (the final output) backwards.
    """
    for match in reversed(list(_VERDICT_FENCE_RE.finditer(text))):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        verdict = _validate_review_verdict(data)
        if verdict is not None:
            return verdict
    return None


def _extract_verdict_from_stream_json(raw_text: str) -> dict[str, Any] | None:
    """Extract a verdict from tee'd stream-json JSONL text.

    With ``tee_stream_json`` enabled the sidecar log is JSONL where every
    fence lives *inside* a JSON string (``\\n`` as escape sequences), so a
    regex over the raw text can never match. Decode the events and accept a
    fence ONLY from the final output: the ``result`` event's text, or —
    absent a usable one (session killed before the result line) — the single
    last assistant text. Never scan further back: a mid-session draft or an
    echo of the review prompt's own few-shot example (which contains a
    literal ``"decision": "approved"`` fence) must not be recorded as the
    session's verdict when the reviewer produced no final one. A fence-less
    final output returns ``None`` so the caller's no-verdict path
    (turn-limit summary + retry) handles it as designed.
    """
    result_text: str | None = None
    last_assistant_text: str | None = None
    for event in iter_stream_json_events(raw_text):
        text = extract_event_text(event)
        if not text:
            continue
        if event.get("type") == "result":
            result_text = text
        else:
            last_assistant_text = text

    for text in (result_text, last_assistant_text):
        if text:
            verdict = _extract_verdict_from_text(text)
            if verdict is not None:
                return verdict
    return None


def _parse_review_verdict_from_log(log_path: Path) -> dict[str, Any] | None:
    """Extract a fenced JSON verdict block from a reviewer's sidecar log.

    Handles both log formats: plaintext logs (fences matched directly) and
    stream-json JSONL logs produced by ``tee_stream_json`` (fences are
    JSON-escaped inside event strings, so events are decoded first). Returns
    the parsed dict on success, or ``None`` if no valid block is found.
    Malformed/truncated logs and 0-byte logs both return ``None``.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    verdict = _extract_verdict_from_text(log_text)
    if verdict is not None:
        return verdict
    return _extract_verdict_from_stream_json(log_text)


def _parse_review_verdict_from_events(events_path: Path) -> dict[str, Any] | None:
    """Extract a fenced JSON verdict block from a reviewer's events.jsonl.

    Fallback for when ``_parse_review_verdict_from_log`` fails: the log may be
    truncated or the verdict block split across tee buffer boundaries, but the
    structured events.jsonl carries the assistant's text in discrete JSONL
    lines. Decodes real stream-json events (``assistant``/``result``) as well
    as the legacy ``assistant_message`` shape.

    Returns the parsed dict on success, or ``None`` if no valid block is found.
    """
    try:
        raw_text = events_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _extract_verdict_from_stream_json(raw_text)


def _parse_review_verdict_from_files(
    log_path: Path,
    packet_dir: Path,
    started_at: str | None,
) -> tuple[dict[str, Any], str] | None:
    """Last-resort verdict recovery from files the reviewer wrote (issue #566).

    Reviewers sometimes write their review summary (verdict block included) to
    a Markdown file and merely *reference* it in final output instead of
    re-emitting the fenced JSON. Before counting a completed review as a
    failed attempt, scan ``.md`` paths mentioned in the reviewer's decoded
    output text, newest-mention-first.

    **Nothing inside ``packet_dir`` is ever a candidate (issue #597).** Review
    sessions launch with a hard-pinned ``--permission-mode plan`` (see
    ``claude_code._force_review_permission_mode``), so a reviewer cannot write
    any file, anywhere — every file in the packet directory is authored by the
    orchestrator itself. ``review-prompt.md`` is one of them, and it embeds an
    example verdict block. Globbing ``packet_dir`` for ``*.md`` therefore did
    not recover reviewer verdicts; it parsed the orchestrator's own
    instructions and recorded whatever the example said. Because the example
    read ``"decision": "approved"``, a reviewer that emitted no verdict had an
    approval manufactured for it, which then took the merge label. Ten PRs
    across two repos merged unreviewed that way. ``_extract_verdict_from_stream_json``
    already guarded against this exact echo; this path was added later and
    bypassed that guard.

    ``packet_dir`` is still taken as a parameter because it defines the
    exclusion zone: a reviewer that *mentions* a packet path in its prose must
    not pull the prompt back in through the mention branch either.

    Every candidate is mtime-gated to the reviewer session's ``started_at``
    (minus slack): a stale review file from a previous round must never
    resurrect an old verdict for a new head. Without a parseable
    ``started_at`` there is no safe gate, so no fallback is attempted.

    Returns ``(verdict, source_path)`` or ``None``.
    """
    if not started_at:
        return None
    try:
        cutoff = datetime.fromisoformat(started_at.replace("Z", "+00:00")) - timedelta(
            seconds=_REVIEW_FALLBACK_MTIME_SLACK_S
        )
    except ValueError:
        return None
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""

    candidates: list[Path] = []
    seen: set[str] = set()

    texts = [
        text
        for text in (extract_event_text(event) for event in iter_stream_json_events(log_text))
        if text
    ]
    if not texts and log_text:
        texts = [log_text]
    for text in reversed(texts):
        for match in _REVIEW_MD_PATH_RE.finditer(text):
            raw = match.group(0)
            if raw not in seen:
                seen.add(raw)
                candidates.append(Path(raw).expanduser())

    # Issue #597: the packet directory is orchestrator-authored territory (see
    # this function's docstring). Never read anything inside it -- not via a
    # glob, and not via a path the reviewer happened to mention in its prose.
    try:
        excluded_root = packet_dir.resolve()
    except OSError:
        excluded_root = packet_dir

    def _inside_packet_dir(candidate: Path) -> bool:
        try:
            resolved = candidate.resolve()
        except OSError:
            return False
        return resolved == excluded_root or excluded_root in resolved.parents

    # Stat-filter BEFORE capping: the cap bounds expensive file reads, and
    # spurious path-looking mentions in the reviewer's text (nonexistent,
    # stale, oversized) must not starve genuine candidates out of the read
    # budget.
    readable: list[Path] = []
    for candidate in candidates:
        if _inside_packet_dir(candidate):
            continue
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if not candidate.is_file():
            continue
        if stat.st_size > _REVIEW_FALLBACK_FILE_MAX_BYTES:
            continue
        if datetime.fromtimestamp(stat.st_mtime, tz=UTC) < cutoff:
            continue
        readable.append(candidate)
        if len(readable) >= _REVIEW_FALLBACK_MAX_CANDIDATES:
            break

    for candidate in readable:
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        verdict = _extract_verdict_from_text(content)
        if verdict is not None:
            return verdict, str(candidate)

    return None


def _reviewer_session_metrics(events_path: Path, verdict_source: str) -> dict[str, Any] | None:
    """Parse reviewer session telemetry for a recorded verdict (perf/spend visibility).

    Returns a dict of ``tokens``/``cost_usd``/``turn_count``/``tool_call_count``/
    ``verdict_source`` for ``record_review`` to fold into the ``record_review``
    event and the PR's state entry, or ``None`` when the events.jsonl sidecar
    is missing or unparseable (devin workers, or a claude-code session launched
    without tee_stream_json). Never raises: ``parse_claude_events`` already
    tolerates a missing/malformed file, and a missing telemetry file must never
    block recording the verdict itself.
    """
    progress = parse_claude_events(events_path)
    if progress is None:
        return None
    return {
        "tokens": progress.tokens,
        "cost_usd": progress.cost_usd,
        "turn_count": progress.turn_count,
        "tool_call_count": progress.tool_call_count,
        "verdict_source": verdict_source,
    }


# Why a reviewer session ended without a structured verdict. These are the
# ``reason`` values on ``review_verdict_missed`` events; they have disjoint
# remediations, so collapsing them into one label points every diagnostic at
# the wrong fix (issue #588).
REVIEW_MISS_TURN_LIMIT = "turn_limit_summary_posted"
REVIEW_MISS_LAUNCH_FAILED = "launch_failed"
REVIEW_MISS_DIED_MID_SESSION = "died_mid_session"

# Default markers for _extract_review_session_summary's session-limit
# reclassification (issue #651/#652). These are the NARROW
# RuntimeConfig.session_limit_markers, NOT the broad throttle_error_markers:
# reviewer launches force tee_stream_json=True (claude_code.py), making
# log_path and events_path byte-identical, so any marker matched against the
# log tail is also present in the parsed assistant text. The generic markers
# in throttle_error_markers ("rate limit", "usage limit") legitimately appear
# in this codebase's rate-limit/quota domain review commentary and would
# false-positive on real review work. session_limit_markers contains only the
# CLI's own specific session-limit death message phrasing, which is safe to
# match against reviewer text. Callers pass their config's list explicitly so
# a new session-limit phrasing only needs a config change.
_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS = OrchestratorConfig().runtime.session_limit_markers

# Tail length for the raw-log session-limit match in _extract_review_session_summary.
# Mirrors the 2048-char tail used by the stalled-session sweep (the
# ``log_text[-2048:]`` slice at the ``Path(w.log_path).read_text(...)`` call in
# _handle_stalled_review_sessions): the CLI prints its session-limit notice at
# the very end of the log, so the tail isolates the death message from the
# multi-turn analysis prose earlier in the log.
_REVIEW_THROTTLE_TAIL_CHARS = 2048


def _log_tail_throttled(log_path: Path, markers: Sequence[str]) -> bool:
    """Return True when the raw process log's tail contains a session-limit marker.

    Reads the last ``_REVIEW_THROTTLE_TAIL_CHARS`` chars of ``log_path`` and
    matches against ``markers`` via ``match_throttle_tail``. This is the same
    raw-log-tail boundary the stalled-session sweep uses (the
    ``log_text[-2048:]`` slice at the ``Path(w.log_path).read_text(...)`` call
    in ``_handle_stalled_review_sessions``). Missing or unreadable logs do not
    match.

    Note (issue #652 review): because reviewer launches force
    ``tee_stream_json=True``, ``log_path`` and ``events_path`` are
    byte-identical — the raw log tail IS the events content, so matching
    against the raw tail does NOT avoid matching the reviewer's own parsed
    assistant text. The false-positive protection comes from the NARROW
    ``session_limit_markers`` list (specific CLI death-message phrasing, not
    generic domain terms), not from the raw-vs-parsed distinction. The
    ``tool_call_count == 0`` guard in the caller is defense-in-depth.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not log_text:
        return False
    if len(log_text) > _REVIEW_THROTTLE_TAIL_CHARS:
        tail = log_text[-_REVIEW_THROTTLE_TAIL_CHARS:]
    else:
        tail = log_text
    return match_throttle_tail(tail, markers)[0]


# state.json key holding the one-time record of worker-PR merges that predate the
# #502 post-merge tripwire. Named here rather than inlined so the arming logic and
# the tests that pre-arm it cannot drift apart on a string literal.
UNAUTHORIZED_MERGE_BASELINE_KEY = "unauthorized_merge_baseline"


@dataclass(frozen=True)
class ReviewSessionOutcome:
    """A reviewer session that ended without producing a structured verdict.

    ``did_substantial_work`` is the distinction that matters downstream: a
    session that completed turns and then died is a PR-level outcome (the
    review didn't fit its budget), while one that never reached its first turn
    is an environmental failure that says nothing about the PR.
    """

    text: str
    reason: str
    turn_count: int
    tool_call_count: int

    @property
    def did_substantial_work(self) -> bool:
        return self.reason != REVIEW_MISS_LAUNCH_FAILED


def _extract_review_session_summary(
    events_path: Path,
    log_path: Path,
    max_turns: int,
    *,
    session_limit_markers: Sequence[str] | None = None,
) -> ReviewSessionOutcome | None:
    """Summarize and classify a reviewer session that produced no verdict.

    When a reviewer hits the ``--max-turns`` limit (or dies for any other
    reason after doing substantial work), the structured verdict block is
    missing but the events.jsonl contains the assistant's analysis text and
    tool-call metrics. This function reconstructs a human-readable summary
    from those events so the work is not silently lost.

    It also classifies *why* the verdict is missing. A session that never
    reached its first turn did not hit a turn limit -- it never ran -- and the
    text recovered from its log is the process's own error output, not
    analysis. Reporting the two identically hid a 25-hour outage in which 19
    reviewers died on a rejected argv while every signal said "turn limit"
    (issue #588).

    A session whose only output is a provider session-limit notice (e.g.
    Claude Code's "hit your session limit") is also environmental, not a
    PR-level outcome: ``parse_claude_events`` counts that notice as one turn,
    so it fails the ``turn_count == 0`` check and would otherwise fall through
    to ``REVIEW_MISS_DIED_MID_SESSION`` -- the same bucket as a session that
    genuinely did substantial review work. That misclassification silently
    defeats the #583 throttle-rollback guard (``did_substantial_work`` reads
    ``reason != REVIEW_MISS_LAUNCH_FAILED`` and is persisted as
    ``review_turn_limit_summary_posted``), letting a global session-limit
    outage burn a PR's ``review_dispatch_attempt_count`` budget with zero
    actual review work performed (issue #651). When the raw process log's tail
    matches a session-limit marker AND the session made no tool calls, classify
    as ``REVIEW_MISS_LAUNCH_FAILED`` so the rollback guard fires.

    The match is against the raw process log tail (last 2048 chars), mirroring
    the stalled-session sweep pattern at the ``log_text =
    Path(w.log_path).read_text(...)`` call below. Because reviewer launches
    force ``tee_stream_json=True`` (claude_code.py), ``log_path`` and
    ``events_path`` are byte-identical -- the raw log tail IS the events
    content, so matching against the raw tail does NOT avoid matching the
    reviewer's own parsed assistant text (issue #652 review). The
    false-positive protection comes from the NARROW
    ``session_limit_markers`` list (specific CLI death-message phrasing like
    "hit your session limit", NOT generic domain terms like "rate limit" /
    "usage limit" that legitimately appear in this codebase's rate-limit/quota
    review commentary), and the ``tool_call_count == 0`` guard ensures a
    session that made any tool calls (real review actions) is never
    reclassified regardless of what the tail contains.

    ``session_limit_markers`` defaults to
    ``RuntimeConfig.session_limit_markers`` so the marker list stays
    config-driven (single point of enforcement in
    ``throttle_signatures.match_throttle_tail``); callers pass their config's
    list explicitly.

    Returns ``None`` if the events file is missing and the log contains no
    recoverable text (nothing to summarize).
    """
    progress = parse_claude_events(events_path)
    # Also try the plaintext log as a fallback for assistant text.
    assistant_texts: list[str] = []

    if events_path.exists():
        try:
            raw_events = events_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_events = ""
        for event in iter_stream_json_events(raw_events):
            text = extract_event_text(event)
            if text.strip():
                assistant_texts.append(text.strip())

    # Fallback: if no events.jsonl, try the log. When it is a stream-json
    # tee, decode the events; otherwise keep the remaining prose lines with
    # fenced code blocks stripped (those are verdict attempts, not analysis).
    if not assistant_texts:
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        for event in iter_stream_json_events(log_text):
            text = extract_event_text(event)
            if text.strip():
                assistant_texts.append(text.strip())
        if not assistant_texts:
            stripped = re.sub(r"```(?:json)?\s*\n.*?```", "", log_text, flags=re.DOTALL)
            for line in stripped.splitlines():
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith((">", "#", "-")):
                    assistant_texts.append(stripped_line)

    if not assistant_texts:
        return None

    turn_count = progress.turn_count if progress else 0
    tool_call_count = progress.tool_call_count if progress else 0
    tokens = progress.tokens if progress else None
    cost_usd = progress.cost_usd if progress else None

    markers = (
        session_limit_markers
        if session_limit_markers is not None
        else _DEFAULT_REVIEW_SESSION_LIMIT_MARKERS
    )

    # A session with no turns and no tool calls never reached its first turn:
    # the process died at launch and whatever text we recovered is its error
    # output, not reviewer analysis.
    if turn_count == 0 and tool_call_count == 0:
        reason = REVIEW_MISS_LAUNCH_FAILED
    elif max_turns > 0 and turn_count >= max_turns:
        reason = REVIEW_MISS_TURN_LIMIT
    elif tool_call_count == 0 and _log_tail_throttled(log_path, markers):
        # A session that made no tool calls but whose raw process log tail
        # contains a provider session-limit notice (e.g. "hit your session
        # limit") died on the notice, not after review work. The notice
        # gets counted as one turn by parse_claude_events, so it fails the
        # turn_count == 0 check above and would fall through to
        # REVIEW_MISS_DIED_MID_SESSION -- the same bucket as a session that
        # genuinely did substantial review work. That misclassification silently
        # defeats the #583 throttle-rollback guard (did_substantial_work reads
        # reason != REVIEW_MISS_LAUNCH_FAILED and is persisted as
        # review_turn_limit_summary_posted), letting a global session-limit
        # outage burn a PR's attempt budget with zero review work performed
        # (issue #651).
        #
        # Two boundaries make this safe (issue #652 review):
        # (1) Match only the NARROW session_limit_markers (specific CLI
        #     death-message phrasing like "hit your session limit"), NOT the
        #     broad throttle_error_markers. Reviewer launches force
        #     tee_stream_json=True, so log_path and events_path are
        #     byte-identical -- the raw log tail IS the events content, so
        #     matching the raw tail does not avoid the reviewer's own parsed
        #     assistant text. Generic markers ("rate limit", "usage limit")
        #     legitimately appear in this codebase's rate-limit/quota review
        #     commentary and would false-positive on real review work. The
        #     specific session-limit phrasing is the CLI's own death message,
        #     not a domain term, so it is safe to match against reviewer text.
        # (2) Guard on ``tool_call_count == 0``: a session that made any tool
        #     calls did real review actions and is a PR-level outcome
        #     (died_mid_session) regardless of what its log tail says -- a
        #     throttle on the final API call after real work is not a launch
        #     failure. The turn-limit branch above already owns sessions that
        #     exhausted their turn budget, so this only intercepts deaths that
        #     occurred before any tool use.
        reason = REVIEW_MISS_LAUNCH_FAILED
    else:
        reason = REVIEW_MISS_DIED_MID_SESSION

    if reason == REVIEW_MISS_LAUNCH_FAILED:
        parts = ["## Reviewer session failed to start\n"]
        parts.append(
            "The automated reviewer exited before running a single turn, so no "
            "review was performed. This is an environmental or launch failure, "
            "not a judgement about this PR.\n"
        )
        parts.append("\n### Error output from the reviewer process:\n")
    else:
        parts = ["## Reviewer session summary (no verdict produced)\n"]
        if reason == REVIEW_MISS_TURN_LIMIT:
            parts.append(
                f"The automated reviewer hit the {max_turns}-turn limit before "
                f"producing a structured verdict.\n"
            )
        else:
            parts.append(
                f"The automated reviewer ran for {turn_count} turns "
                f"({tool_call_count} tool calls) but did not produce a structured verdict.\n"
            )
        # Include the last few assistant messages — earlier turns are usually
        # tool-use planning; the final messages contain the analysis.
        parts.append("\n### Recent analysis from the reviewer:\n")

    recent = assistant_texts[-3:]
    for text in recent:
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated)"
        parts.append(text)
        parts.append("\n---\n")

    meta_parts: list[str] = []
    if turn_count:
        meta_parts.append(f"turns: {turn_count}")
    if tool_call_count:
        meta_parts.append(f"tool calls: {tool_call_count}")
    if tokens is not None:
        meta_parts.append(f"tokens: {tokens:,}")
    if cost_usd is not None:
        meta_parts.append(f"cost: ${cost_usd:.4f}")
    if meta_parts:
        parts.append(f"\n*{' · '.join(meta_parts)}*")

    return ReviewSessionOutcome(
        text="\n".join(parts),
        reason=reason,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
    )


def _remove_review_checkout_with_warning(
    state: dict[str, Any],
    repo_root: Path,
    reviews_dir: Path,
    pr_number: int,
    *,
    state_file: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Remove an isolated review checkout and emit a one-shot warning on failure.

    Returns ``(new_state, removed)``. On failure, sets a per-PR warning marker
    in ``state["prs"][pr_number]`` and appends a ``review_checkout_removal_failed``
    event, but only if that PR does not already have an active warning marker.
    The marker is cleared when removal succeeds so a future failure can alert
    again. Retry happens on the next sweep pass; the warning is never re-emitted
    per pass.
    """
    removed = remove_review_checkout(repo_root, pr_number, reviews_dir=reviews_dir)
    pr_key = str(pr_number)
    prs = state.get("prs")
    if not isinstance(prs, dict):
        prs = {}
    pr_state = prs.get(pr_key)
    if not isinstance(pr_state, dict):
        pr_state = {}

    if removed:
        if pr_state.get("review_checkout_removal_warned"):
            state = {
                **state,
                "prs": {**prs, pr_key: {**pr_state, "review_checkout_removal_warned": None}},
            }
        return state, True

    if not pr_state.get("review_checkout_removal_warned"):
        state = {
            **state,
            "prs": {**prs, pr_key: {**pr_state, "review_checkout_removal_warned": True}},
        }
        state = append_event(
            state,
            "review_checkout_removal_failed",
            {"pr_number": pr_number, "reviews_dir": str(reviews_dir)},
            state_path=state_file,
        )
    return state, False


def _set_reviewer_quota_exhausted_with_backoff(
    state: dict[str, Any],
    config: OrchestratorConfig,
    now_dt: datetime,
    *,
    reset_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record a quota-exhaustion episode with exponential probe backoff.

    Every consecutive throttle hit without an intervening successful probe
    (a "successful probe" is a recorded verdict from a dead reviewer -- see
    dispatch_reviews's verdict-reap clear, the only proof the quota window is
    actually open) doubles the probe interval, capped at
    ``quota_probe_max_interval_minutes``, so a live provider outage does not
    relaunch a real reviewer session into the wall every
    ``quota_probe_interval_minutes`` forever (cost-spirals.md Finding 2: the
    config comment used to say "No escalation backoff" and meant it literally
    -- provider-throttle stalls are also exempt from the per-PR dispatch
    attempt cap, so this was the one failure mode that could not terminate).
    ``consecutive_probe_failures`` lives inside the existing ``reviewer_quota``
    dict rather than as a new state.py-owned field/helper, matching this
    fix's file scope.

    Issue #612: when the provider's session-limit notice names a specific
    reset time (``reset_at``, the parsed "resets H:MMam/pm (zone)" clock
    time), ``throttled_until`` is that reset plus the configured
    ``throttle_resume_margin_s`` instead of ``now + quota_reset_hours``.
    The provider's own stated reset is a far better backoff target than a
    fixed guess: it avoids both re-spending into a still-closed window
    (fixed window shorter than the real reset) and stalling far longer than
    necessary (fixed window longer than the real reset). The resume margin
    is added because provider reset estimates are floors, not guarantees
    (issue #499). When ``reset_at`` is None (no clock-time notice parsed,
    or the named zone was unavailable), the fixed ``quota_reset_hours``
    window is used as before.

    Returns ``(new_state, quota_record)`` where ``quota_record`` is the
    written ``reviewer_quota`` dict, so callers can emit a
    ``review_quota_exhausted`` event carrying ``throttled_until``,
    ``probe_after``, ``reset_at`` (ISO or None), and
    ``consecutive_probe_failures`` without re-reading state.
    """
    rd = config.review_dispatch
    quota = state.get("reviewer_quota") or {}
    consecutive_failures = int(quota.get("consecutive_probe_failures", 0)) + 1
    interval_minutes = rd.quota_probe_interval_minutes * (2 ** (consecutive_failures - 1))
    if rd.quota_probe_max_interval_minutes > 0:
        interval_minutes = min(interval_minutes, rd.quota_probe_max_interval_minutes)
    if reset_at is not None:
        # Back off until the provider's own stated reset, plus the resume
        # margin (provider resets are floors, not guarantees — issue #499).
        throttled_dt = reset_at + timedelta(seconds=config.runtime.throttle_resume_margin_s)
    else:
        throttled_dt = now_dt + timedelta(hours=rd.quota_reset_hours)
    throttled_until = throttled_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    probe_after = (
        (now_dt + timedelta(minutes=interval_minutes))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_reviewer_quota_exhausted(
        state, throttled_until=throttled_until, probe_after=probe_after
    )
    quota_record = {
        **state["reviewer_quota"],
        "consecutive_probe_failures": consecutive_failures,
        "reset_at": (
            reset_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if reset_at is not None
            else None
        ),
    }
    state = {**state, "reviewer_quota": quota_record}
    return state, quota_record


def _merge_on_write_save(
    state_file: Path,
    state: dict[str, Any],
    *,
    snapshot_prs: dict[str, Any],
    snapshot_reviewer_quota: Any,
    snapshot_events: list[dict[str, Any]],
    event_ring_cap: int,
) -> None:
    """Merge a sweep's computed changes onto fresh on-disk state and save
    (issue #594).

    Any ``dispatch_reviews`` sweep that loads a state snapshot via
    ``load_state_locked``, does slow work (filesystem or network I/O) with no
    lock held, then wants to persist what it computed must call this instead
    of a bare ``save_state(state_file, state)``. Writing the sweep's `state`
    wholesale would restore every field a concurrent writer (e.g. ``charlie
    unescalate``) changed in the gap between the snapshot load and this save
    to its pre-commit value, with no event explaining the reversal -- a
    classic read-modify-write lost update. ``unescalate`` itself already
    defends against this hazard in the other direction with the same
    re-load-and-re-apply-inside-the-lock pattern; this extends the same
    courtesy to the loop's sweeps.

    Re-loads fresh state under the lock and, per PR entry, applies only the
    fields that differ between ``state`` (the sweep's post-work view) and
    ``snapshot_prs`` (the entry-pass view) -- i.e. only the fields the sweep
    itself actually changed. Entries the sweep never touched are left exactly
    as fresh on-disk state has them, so a concurrent writer's commit to an
    untouched entry survives. For a PR the sweep DID touch, its overrides
    (e.g. a terminal ``status`` written because GitHub said the PR merged)
    still win over whatever a concurrent writer set on that same entry in the
    gap -- the sweep is the authority on the fact it just observed.

    The durable event audit is unaffected by any of this: every event was
    already dual-written to events.db via ``append_event(state_path=...)``
    before this call; this only keeps the bounded ``state.json`` event ring
    consistent by appending the sweep's new events onto the fresh ring.
    """
    with state_lock(state_file):
        fresh = load_state(state_file)
        fresh_prs = dict(fresh.get("prs") or {})
        for pr_key, new_entry in (state.get("prs") or {}).items():
            if not isinstance(new_entry, dict):
                fresh_prs[pr_key] = new_entry
                continue
            if new_entry == snapshot_prs.get(pr_key):
                # Untouched by this sweep: preserve the fresh on-disk value so
                # a concurrent writer's commit survives.
                continue
            # Apply only the fields the sweep computed onto the fresh entry:
            # overrides = new_entry fields that differ from the entry-pass
            # snapshot. Fields the sweep did not touch stay as the fresh
            # on-disk value (preserving concurrent writes to them), and the
            # sweep's overrides win for the fields it legitimately owns.
            snapshot_entry = snapshot_prs.get(pr_key) or {}
            overrides = {
                field: value
                for field, value in new_entry.items()
                if value != snapshot_entry.get(field)
            }
            fresh_entry = fresh_prs.get(pr_key)
            fresh_prs[pr_key] = {
                **(fresh_entry if isinstance(fresh_entry, dict) else {}),
                **overrides,
            }
        merged: dict[str, Any] = dict(fresh)
        merged["prs"] = fresh_prs
        if state.get("reviewer_quota") != snapshot_reviewer_quota:
            merged["reviewer_quota"] = state["reviewer_quota"]
        # Identity diff, not a length-based slice: ``append_event`` always
        # rebuilds the events list via ``list(old) + [new]`` (never mutates
        # entries in place), so every snapshot event dict retains its
        # original ``id()`` for the life of this pass. When the ring is at
        # its cap (the normal steady-state -- 2000 entries by default), each
        # append truncates from the front, so the post-sweep list is the SAME
        # length as the snapshot and a length-based slice
        # ``current[len(snapshot):]`` wrongly evaluates to empty, silently
        # dropping this sweep's own events from the merged ring. Filtering by
        # identity against the snapshot's ids is correct regardless of
        # whether eviction happened.
        snapshot_event_ids = {id(e) for e in snapshot_events}
        sweep_appended_events = [
            e for e in (state.get("events") or []) if id(e) not in snapshot_event_ids
        ]
        if sweep_appended_events:
            ring = list(fresh.get("events") or []) + sweep_appended_events
            if len(ring) > event_ring_cap:
                ring = ring[-event_ring_cap:]
            merged["events"] = ring
        save_state(state_file, merged)


def _detect_and_handle_stalled_reviews(
    reviews_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Detect reviewer processes that died without a verdict and free their claims.

    A reviewer is considered stalled/orphaned when its sidecar process is no
    longer alive and the claim timestamp is past the stale-claim timeout
    (``_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES``, currently 5 minutes -- see
    ``state.is_claim_stale``). When that happens, the per-PR
    ``review_dispatch_status`` is moved to ``review_dispatch_failed`` with the
    stale timestamp as ``review_dispatch_failed_at``. The next
    ``dispatch_reviews`` pass can then re-dispatch the PR after the same stale
    timeout elapses. Every reap path also tears down that PR's isolated review
    checkout (``worktree.remove_review_checkout``) so a dead reviewer never
    leaks its detached-HEAD checkout directory.

    A PR that reached ``status == "reviewing"`` but has no
    ``review_dispatch_status`` claim at all (a packet that was generated but
    never dispatched) is also reaped once its review packet is older than the
    stale-claim timeout. The claim is moved to ``review_dispatch_failed`` using
    the packet's own mtime as ``review_dispatch_failed_at`` so the next
    ``dispatch_reviews`` pass can retry it immediately.

    Callers should run ``_reap_review_verdicts`` first: it extracts and records
    any verdict a dead reviewer emitted, so by the time this sweep runs only
    PRs whose log has no parseable verdict fall through to the failed-claim
    retry/backoff path. This function is intentionally simpler than
    ``_detect_and_handle_stalled_sessions``: it performs claim/slot cleanup.
    """
    stalled: list[dict[str, Any]] = []
    sweep_events: list[tuple[str, dict[str, Any]]] = []
    state = load_state_locked(state_file)
    # Capture the entry-pass snapshot so the save block below can diff against
    # it and apply ONLY the entries/fields this sweep computed. Writing the
    # stale snapshot wholesale at save time would clobber any concurrent writer
    # (e.g. ``charlie unescalate``) that committed between this load and the
    # save -- a lost update across the sweep's read-modify-write window (issue
    # #594). The prs values are rebound (never mutated in place) by every
    # branch below, so a shallow copy of the mapping preserves the originals.
    snapshot_prs = dict(state.get("prs") or {})
    snapshot_reviewer_quota = state.get("reviewer_quota")
    snapshot_events = list(state.get("events") or [])
    changed = False
    seen_pr_keys: set[str] = set()
    # One provider-throttle condition per sweep, no matter how many dead
    # reviewers show the same limit signature: the exponential probe backoff
    # counts consecutive failed PROBES, and a wave of N simultaneously
    # throttled reviewers is one observation of the closed provider window,
    # not N. Without this, a 2-reviewer wave incremented the counter twice
    # per sweep and (combined with the un-reaped sidecars below) drove the
    # backoff from its 15-minute base to the 4-hour cap within 40 minutes
    # (observed live 2026-07-24: consecutive_probe_failures=14 from a single
    # quota outage).
    throttle_backoff_applied = False

    for w in iter_workers(reviews_dir):
        pr_key = str(w.issue_number)
        # A live reviewer needs no cleanup.
        if w.is_alive():
            continue
        # Respect the stale-claim timeout so a very recently dead reviewer is
        # not immediately re-dispatched (which can thrash if the underlying
        # launch path is flaky). Old dead reviewers become re-dispatchable.
        if not is_claim_stale(w.started_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES):
            continue

        seen_pr_keys.add(pr_key)
        pr_state = state["prs"].get(pr_key, {})
        status = pr_state.get("review_dispatch_status")
        if status in ("review_dispatch_completed", "review_dispatch_failed"):
            # Already terminal; don't overwrite a completed/failed record, but
            # reap the dead session's sidecar so it stops resurfacing here.
            w.reap_sidecar(reviews_dir)
            continue
        if status is None and pr_state.get("status") in ("merged", "closed"):
            # Lifecycle-reaped terminal PR: the claim was already cleared by
            # the orphan sweep. Without reaping the sidecar here, the dead
            # session resurrects as a phantom failed claim every pass and the
            # orphan sweep re-reaps it — an infinite stalled/reaped ping-pong
            # that floods the event ring (observed on 5 merged PRs, 07-22).
            w.reap_sidecar(reviews_dir)
            continue

        # A dead reviewer's own log may show why it died. If it hit a
        # provider throttle signature (e.g. "You've hit your session
        # limit"), the launch-time quota_hit check in dispatch_reviews never
        # saw it -- that check only fires when launch_claude_worker() itself
        # errors synchronously, but a throttled Claude Code CLI process
        # starts fine and only dies after printing the limit message to its
        # own log. Without this check, every pass here would mark the claim
        # review_dispatch_failed and the next dispatch_reviews pass would
        # relaunch straight into the same limit -- a redispatch loop that
        # runs every stale-claim interval for as long as the provider window
        # is closed, instead of backing off via the same reviewer-quota gate
        # the launch-time path uses (job-cannon PRs #1342/#1343/#1344/#1346,
        # 2026-07-21: 20+ hours of hot redispatch into a session-limit wall).
        throttled = False
        reset_at: datetime | None = None
        try:
            log_text = Path(w.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        if log_text:
            tail = log_text[-2048:] if len(log_text) > 2048 else log_text
            throttled = match_throttle_tail(tail, config.runtime.throttle_error_markers)[0]
            # Issue #612: the session-limit notice names a specific reset
            # clock time in an IANA zone (e.g. "resets 1:20am
            # (America/Los_Angeles)"). Parse it once per dead session so the
            # fleet-wide backoff targets the provider's own stated reset
            # instead of a fixed quota_reset_hours guess. Only parsed on the
            # session that triggers the backoff (the first throttled one);
            # subsequent throttled sessions in the same wave reuse the
            # already-applied backoff, matching the one-increment-per-wave
            # guard below.
            if throttled and not throttle_backoff_applied:
                reset_at = parse_reset_clock_time(tail, datetime.now(UTC))

        if throttled:
            if not throttle_backoff_applied:
                now_dt = datetime.now(UTC)
                state, quota_record = _set_reviewer_quota_exhausted_with_backoff(
                    state, config, now_dt, reset_at=reset_at
                )
                throttle_backoff_applied = True
                # Distinct, queryable event for a quota-dead reviewer session
                # (issue #612): carries the parsed reset time (or None when
                # the notice carried no clock-time form / the zone was
                # unavailable) so the condition is diagnosable as a quota
                # exhaustion rather than collapsing into a generic
                # "no verdict" / "provider_throttled" stall. Emitted once per
                # sweep alongside the single backoff increment.
                state = append_event(
                    state,
                    "review_quota_exhausted",
                    {
                        "throttled_until": quota_record.get("throttled_until"),
                        "probe_after": quota_record.get("probe_after"),
                        "reset_at": quota_record.get("reset_at"),
                        "consecutive_probe_failures": quota_record.get(
                            "consecutive_probe_failures"
                        ),
                        "source": "stalled_review_sweep",
                    },
                    state_path=state_file,
                )
            throttled_until = state.get("reviewer_quota", {}).get("throttled_until")
            # A session that exhausted its full turn budget did real
            # PR-specific work -- its death is a PR-level outcome (the
            # review didn't fit the budget) regardless of what killed the
            # final API call. Only sessions that died WITHOUT such work
            # qualify for the provider-throttle rollback. _reap_review_verdicts
            # (which runs before this sweep) sets
            # review_turn_limit_summary_posted on the pr-state when it
            # extracted a turn-limit miss from this same dead session, and
            # that flag is reset to False on every new claim -- so it is the
            # single source of truth for "this dispatch lifecycle's session
            # did substantial work then died". When it is set, still apply
            # the global quota backoff (the throttle signal itself is real)
            # but treat the death as a normal counted failure so the per-PR
            # attempt cap can converge (issue #583: without this guard a PR
            # whose reviews deterministically hit the turn cap got unlimited
            # free retries whenever the account was near its limit, and the
            # 3-attempt cap never fired).
            if pr_state.get("review_turn_limit_summary_posted"):
                state["prs"][pr_key] = {
                    **pr_state,
                    "number": w.issue_number,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": w.started_at,
                    "review_dispatch_pending_at": None,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                }
                state = append_event(
                    state,
                    "review_dispatch_stalled",
                    {
                        "pr_number": w.issue_number,
                        "pid": w.pid,
                        "started_at": w.started_at,
                        "reason": "provider_throttled_turn_limit_counted",
                        "throttled_until": throttled_until,
                    },
                    state_path=state_file,
                )
                changed = True
                stalled.append(
                    {
                        "pr": w.issue_number,
                        "pid": w.pid,
                        "started_at": w.started_at,
                        "reason": "provider_throttled_turn_limit_counted",
                    }
                )
                remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
                w.reap_sidecar(reviews_dir)
                continue
            # Roll back (not fail) the claim: this is a global condition, not
            # a defect in this PR's review, so it should be immediately
            # re-dispatchable once the quota gate clears -- mirroring the
            # launch-time quota_hit rollback in dispatch_reviews. Also
            # decrement the attempt counter: the reviewer hit a provider
            # limit, not a PR-specific failure, so this must not consume the
            # per-PR dispatch attempt budget.
            rolled_back = without_review_dispatch_claim(pr_state)
            attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
            if attempt_count > 0:
                rolled_back["review_dispatch_attempt_count"] = attempt_count - 1
            state["prs"][pr_key] = rolled_back
            state = append_event(
                state,
                "review_dispatch_stalled",
                {
                    "pr_number": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": "provider_throttled",
                    "throttled_until": throttled_until,
                },
                state_path=state_file,
            )
            changed = True
            stalled.append(
                {
                    "pr": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                    "reason": "provider_throttled",
                }
            )
            remove_review_checkout(repo_root, w.issue_number, reviews_dir=reviews_dir)
            # Reap the sidecar like every other handled path in this sweep.
            # The rolled-back claim is deliberately non-terminal (so the PR
            # re-dispatches once the quota gate clears), which means neither
            # terminal guard above will ever reap this sidecar -- without
            # this line the same dead reviewer resurfaces every sweep, its
            # log tail still matches the throttle signature, and each pass
            # re-applies the exponential backoff for a session that died
            # exactly once (observed live 2026-07-24: two dead reviewers
            # re-counted across ~6 passes pushed probe_after 4 hours out
            # while the provider window was already open again).
            w.reap_sidecar(reviews_dir)
            continue

        state["prs"][pr_key] = {
            **pr_state,
            "number": w.issue_number,
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_failed_at": w.started_at,
            "review_dispatch_pending_at": None,
            "review_dispatched_at": None,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
        }
        sweep_events.append(
            (
                "review_dispatch_stalled",
                {
                    "pr_number": w.issue_number,
                    "pid": w.pid,
                    "started_at": w.started_at,
                },
            )
        )
        changed = True
        stalled.append({"pr": w.issue_number, "pid": w.pid, "started_at": w.started_at})
        state, _ = _remove_review_checkout_with_warning(
            state, repo_root, reviews_dir, w.issue_number, state_file=state_file
        )
        # The failed record above is now the source of truth for redispatch;
        # the dead session's sidecar must go with the checkout or it re-enters
        # this sweep as a phantom on every subsequent pass.
        w.reap_sidecar(reviews_dir)

    # Catch state entries that have no sidecar (launch crashed before sidecar
    # write, or sidecar was deleted) and are past the stale timeout.
    for pr_key, pr_state in list(state.get("prs", {}).items()):
        if pr_key in seen_pr_keys or not isinstance(pr_state, dict):
            continue

        status = pr_state.get("review_dispatch_status")
        if status == "review_dispatch_pending":
            pending_at = pr_state.get("review_dispatch_pending_at")
            if pending_at and is_claim_stale(
                pending_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES
            ):
                state["prs"][pr_key] = {
                    **pr_state,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": pending_at,
                    "review_dispatch_pending_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                }
                sweep_events.append(
                    (
                        "review_dispatch_stalled",
                        {
                            "pr_number": int(pr_key) if pr_key.isdigit() else None,
                            "status": "pending",
                            "pending_at": pending_at,
                        },
                    )
                )
                changed = True
                stalled.append(
                    {
                        "pr": int(pr_key) if pr_key.isdigit() else None,
                        "pending_at": pending_at,
                    }
                )
                if pr_key.isdigit():
                    state, _ = _remove_review_checkout_with_warning(
                        state, repo_root, reviews_dir, int(pr_key), state_file=state_file
                    )
        elif status == "review_dispatch_dispatched":
            reviewer_pid = pr_state.get("reviewer_pid")
            process_start_time = pr_state.get("reviewer_process_start_time")
            pid_alive = reviewer_pid is not None and is_pid_alive(reviewer_pid, process_start_time)
            if pid_alive:
                continue
            dispatched_at = pr_state.get("review_dispatched_at")
            if dispatched_at and is_claim_stale(
                dispatched_at, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES
            ):
                state["prs"][pr_key] = {
                    **pr_state,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": dispatched_at,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                }
                sweep_events.append(
                    (
                        "review_dispatch_stalled",
                        {
                            "pr_number": int(pr_key) if pr_key.isdigit() else None,
                            "status": "dispatched",
                            "dispatched_at": dispatched_at,
                        },
                    )
                )
                changed = True
                stalled.append(
                    {
                        "pr": int(pr_key) if pr_key.isdigit() else None,
                        "dispatched_at": dispatched_at,
                    }
                )
                if pr_key.isdigit():
                    state, _ = _remove_review_checkout_with_warning(
                        state, repo_root, reviews_dir, int(pr_key), state_file=state_file
                    )
        elif status is None and pr_state.get("status") == "reviewing":
            # Issue #487: a review packet was generated but was never claimed or
            # dispatched at all. If the packet is past the stale-claim timeout,
            # move the (missing) claim to failed using the packet's own mtime as
            # the failure timestamp so the next dispatch_reviews pass can retry.
            prompt_path_str = pr_state.get("prompt_path")
            if not prompt_path_str:
                continue
            prompt_path = Path(prompt_path_str)
            if not prompt_path.exists():
                continue

            decision_value: str | None = "missing"
            decision_path_str = pr_state.get("decision_path")
            if decision_path_str:
                decision_path = Path(decision_path_str)
                if decision_path.exists():
                    try:
                        with decision_path.open("r", encoding="utf-8") as handle:
                            decision_data = json.load(handle)
                        if isinstance(decision_data, dict):
                            decision_value = decision_data.get("decision")
                    except (OSError, json.JSONDecodeError):
                        decision_value = "invalid"
            if decision_value not in ("pending", "missing", "invalid", None):
                continue

            prompt_mtime = prompt_path.stat().st_mtime
            packet_age = (
                datetime.fromtimestamp(prompt_mtime, tz=UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            if not is_claim_stale(packet_age, timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES):
                continue

            state["prs"][pr_key] = {
                **pr_state,
                "number": int(pr_key) if pr_key.isdigit() else pr_state.get("number"),
                "review_dispatch_status": "review_dispatch_failed",
                "review_dispatch_failed_at": packet_age,
                "review_dispatch_pending_at": None,
                "review_dispatched_at": None,
                "reviewer_pid": None,
                "reviewer_process_start_time": None,
            }
            sweep_events.append(
                (
                    "review_dispatch_stalled",
                    {
                        "pr_number": int(pr_key) if pr_key.isdigit() else None,
                        "status": "unclaimed",
                        "prompt_mtime": packet_age,
                    },
                )
            )
            changed = True
            stalled.append(
                {
                    "pr": int(pr_key) if pr_key.isdigit() else None,
                    "prompt_mtime": packet_age,
                }
            )
            if pr_key.isdigit():
                state, _ = _remove_review_checkout_with_warning(
                    state, repo_root, reviews_dir, int(pr_key), state_file=state_file
                )

    if changed:
        state = _append_sweep_events(
            state, sweep_events, max_size=config.runtime.event_ring_size, state_file=state_file
        )
        # Merge-on-write (issue #594) -- see ``_merge_on_write_save`` for why a
        # bare ``save_state(state_file, state)`` here would clobber a
        # concurrent writer (e.g. ``charlie unescalate``).
        _merge_on_write_save(
            state_file,
            state,
            snapshot_prs=snapshot_prs,
            snapshot_reviewer_quota=snapshot_reviewer_quota,
            snapshot_events=snapshot_events,
            event_ring_cap=config.runtime.event_ring_size,
        )

    return stalled


def _reap_review_sidecar(reviews_dir: Path, pr_number: int) -> None:
    """Delete a dead reviewer's sidecar so it cannot resurrect as a phantom.

    Reviewer sessions are keyed by PR number in ``reviews_dir``. Never touches
    a live session's sidecar (the governor counts live sidecars — deleting one
    would silently free a slot for over-cap dispatch). Best-effort:
    ``WorkerView.reap_sidecar`` swallows OSError, and a sidecar that survives
    one pass is reaped on the next.
    """
    for w in iter_workers(reviews_dir):
        if w.issue_number == pr_number and not w.is_alive():
            w.reap_sidecar(reviews_dir)


def _reap_completed_review_checkouts(
    repo_root: Path,
    reviews_dir: Path,
    state_file: Path,
) -> list[int]:
    """Remove isolated review checkouts for PRs whose reviewer already
    recorded a verdict, once the reviewer process itself has exited.

    ``record_review`` (called in-process by ``_reap_review_verdicts`` or invoked
    via the ``verdict`` CLI) sets ``review_dispatch_status = "review_dispatch_completed"`` and
    clears ``reviewer_pid``/``reviewer_process_start_time`` on state.json as
    part of recording the verdict — so by the time this sweep can see
    "completed", state.json itself no longer carries a PID to check liveness
    against. The claude-code sidecar in ``reviews_dir`` still does (it isn't
    touched by ``record_review``), so liveness is checked via ``iter_workers``
    instead. This avoids removing the checkout while the reviewer session
    that just wrote the verdict is still in the process of exiting.
    """
    state = load_state_locked(state_file)
    completed_prs = {
        int(pr_key)
        for pr_key, entry in state.get("prs", {}).items()
        if isinstance(entry, dict)
        and entry.get("review_dispatch_status") == "review_dispatch_completed"
        and pr_key.isdigit()
    }
    if not completed_prs:
        return []

    alive_pr_numbers = _alive_review_worker_issue_numbers(reviews_dir)
    reaped: list[int] = []
    for pr_number in sorted(completed_prs):
        if pr_number in alive_pr_numbers:
            continue
        if remove_review_checkout(repo_root, pr_number, reviews_dir=reviews_dir):
            reaped.append(pr_number)
        # The verdict is recorded and the reviewer has exited: the sidecar has
        # served its purpose and must not linger as a phantom session.
        _reap_review_sidecar(reviews_dir, pr_number)
    return reaped


def _reap_orphaned_review_checkouts(
    gh: GitHub,
    repo_root: Path,
    reviews_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
) -> list[int]:
    """Remove isolated review checkouts for PRs whose GitHub lifecycle has
    already reached ``MERGED`` or ``CLOSED`` and whose reviewer process has
    exited.

    This is the review-dispatch-pass counterpart to reconcile.py's
    ``merged_outside_orchestrator`` and ``closed_unmerged_pr_active_labels``
    drift handlers. It runs unconditionally at the top of ``dispatch_reviews``
    so an externally-merged/closed PR never leaves its ``reviews_dir``
    checkout or ``review_dispatch_*`` claim alive indefinitely. A PR whose
    reviewer sidecar is still alive is deferred to a later pass so the live
    session is not interrupted.
    """
    state = load_state_locked(state_file)
    # Capture the entry-pass snapshot so the save block below can merge-on-write
    # instead of restoring this stale snapshot wholesale (issue #594). The
    # candidate loop below does a ``gh.pr_view`` network call per candidate PR,
    # potentially many, before the save -- the same shape of unlocked
    # read-modify-write window that let a concurrent ``charlie unescalate``
    # get silently reverted in ``_detect_and_handle_stalled_reviews``. ``prs``
    # entries are rebound (never mutated in place) below, so a shallow copy of
    # the mapping preserves the originals.
    snapshot_prs = dict(state.get("prs") or {})
    snapshot_reviewer_quota = state.get("reviewer_quota")
    snapshot_events = list(state.get("events") or [])
    candidate_pr_numbers: set[int] = set()

    review_dispatch_keys = (
        "review_dispatch_status",
        "review_dispatch_pending_at",
        "review_dispatched_at",
        "review_dispatch_failed_at",
        "reviewer_pid",
        "reviewer_process_start_time",
    )
    for pr_key, entry in state.get("prs", {}).items():
        if not isinstance(entry, dict):
            continue
        if any(entry.get(field) is not None for field in review_dispatch_keys):
            try:
                candidate_pr_numbers.add(int(pr_key))
            except ValueError:
                continue

    if reviews_dir.is_dir():
        for entry in reviews_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("pr-"):
                suffix = entry.name.split("-", 1)[1]
                try:
                    candidate_pr_numbers.add(int(suffix))
                except ValueError:
                    continue

    if not candidate_pr_numbers:
        return []

    alive_pr_numbers = _alive_review_worker_issue_numbers(reviews_dir)
    reaped: list[int] = []
    sweep_events: list[tuple[str, dict[str, Any]]] = []
    changed = False
    for pr_number in sorted(candidate_pr_numbers):
        try:
            pr = gh.pr_view(pr_number)
        except Exception:
            # PR not found or transient lookup failure; do not act on uncertainty.
            continue
        if not isinstance(pr, dict):
            continue

        gh_state = str(pr.get("state") or "").upper()
        if gh_state not in ("MERGED", "CLOSED"):
            continue

        # Issue #504: a live reviewer process keeps its checkout alive until it exits.
        if pr_number in alive_pr_numbers:
            continue

        pr_key = str(pr_number)
        pr_state = state["prs"].get(pr_key, {})
        new_pr_state = without_review_dispatch_claim(pr_state)
        new_pr_state["number"] = pr_number
        if gh_state == "MERGED":
            new_pr_state["status"] = "merged"
        else:
            # Record the terminal closed state so a future pass does not
            # re-query.  Always overwrite — a stale "reviewing" status left
            # by the review pipeline causes the unclaimed-stalled sweep to
            # re-trigger every pass (infinite ping-pong with this reaper).
            new_pr_state["status"] = "closed"
        state["prs"][pr_key] = new_pr_state
        changed = True

        state, removed = _remove_review_checkout_with_warning(
            state, repo_root, reviews_dir, pr_number, state_file=state_file
        )
        # Reap the sidecar with the checkout: leaving it resurrects the dead
        # session as a phantom failed claim in the stalled sweep next pass,
        # which re-triggers this reap — an infinite ping-pong per merged PR.
        _reap_review_sidecar(reviews_dir, pr_number)
        if removed:
            sweep_events.append(
                (
                    "review_dispatch_lifecycle_reaped",
                    {"pr_number": pr_number, "github_state": gh_state.lower()},
                )
            )
            reaped.append(pr_number)

    if changed:
        state = _append_sweep_events(
            state, sweep_events, max_size=config.runtime.event_ring_size, state_file=state_file
        )
        # Merge-on-write (issue #594) -- see ``_merge_on_write_save`` for why a
        # bare ``save_state(state_file, state)`` here would clobber a
        # concurrent writer (e.g. ``charlie unescalate``). A PR this reap DID
        # touch keeps its terminal ``status`` ("merged"/"closed") regardless
        # of what a concurrent writer set on that same entry in the gap --
        # GitHub's lifecycle state is authoritative once observed.
        _merge_on_write_save(
            state_file,
            state,
            snapshot_prs=snapshot_prs,
            snapshot_reviewer_quota=snapshot_reviewer_quota,
            snapshot_events=snapshot_events,
            event_ring_cap=config.runtime.event_ring_size,
        )

    return reaped


def _append_sweep_events(
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    max_size: int | None = None,
    *,
    state_file: Path | None = None,
) -> dict[str, Any]:
    """Append events collected during a sweep, aggregating same-kind runs.

    A single occurrence of a kind is emitted with the original kind and payload.
    Multiple occurrences of the same kind are emitted as one ``{kind}_sweep`` event
    with a count and a numbers list. This prevents a single bulk sweep from
    flooding the bounded event buffer and evicting unrelated diagnostic history.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind, payload in sweep_events:
        grouped.setdefault(kind, []).append(payload)

    for kind, payloads in grouped.items():
        if len(payloads) == 1:
            state = append_event(
                state, kind, payloads[0], max_size=max_size, state_path=state_file
            )
        else:
            numbers: list[int] = []
            numbers_key = "numbers"
            for payload in payloads:
                if payload.get("issue_number") is not None:
                    numbers.append(payload["issue_number"])
                    numbers_key = "issue_numbers"
                elif payload.get("pr_number") is not None:
                    numbers.append(payload["pr_number"])
                    if numbers_key == "numbers":
                        numbers_key = "pr_numbers"
                elif payload.get("number") is not None:
                    numbers.append(payload["number"])
            state = append_event(
                state,
                f"{kind}_sweep",
                {
                    "count": len(payloads),
                    numbers_key: numbers,
                },
                max_size=max_size,
                state_path=state_file,
            )
    return state


def _detect_and_handle_orphaned_workers(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh: GitHub,
    *,
    review_callback: Callable[[int], Any] | None = None,
) -> None:
    """Detect and handle orphaned workers using state.json PID records.

    This is a fallback for issue #207: when session sidecar files are orphaned
    (e.g., by session-limit reset), the session-file-based stall-reaper cannot
    detect dead workers. This function reads worker PIDs from state.json and
    checks liveness directly, allowing recovery even without session files.

    For issues with status "dispatched" and a recorded worker_pid:
    - If the PID is dead, check the linked PR's last review decision
    - If last decision was "request_changes" and head unchanged, reset to "rework_requested"
    - If last decision was "request_changes" and head advanced, route to the
      review-pending path by calling ``review_callback`` and then flipping the
      issue status to "reviewing"
    - Otherwise, surface as drift for human triage (once per unchanged finding)
    - Do NOT clear worker_pid from state.json after handling (issue #282: the
      recovery path needs the fingerprint to verify the worktree is safe to reset).

    Issue #417: for the no-open-PR case, this is the ONE lane that revisits a
    dead session's GitHub labels independent of its sidecar file -- it is
    keyed entirely off state.json (``status == "dispatched"`` + a dead
    ``worker_pid``), so it runs every pass whether or not a sidecar exists.
    ``_classify_dead_sessions_and_update_throttle_state``'s own no-open-PR
    reclaim (the "issue #118" lane) is a single best-effort attempt per dead
    session: if it is interrupted (process crash/reboot) between writing
    ``redispatch_at`` and swapping the GitHub labels, or if the label API
    calls themselves fail, the sidecar is already reaped and that lane has no
    way to revisit the issue. This sweep closes that gap by re-deriving
    "does this issue still need its labels fixed" from GitHub's *live* label
    state every pass (not from any one-shot flag), so a half-finished reclaim
    -- or one stranded before this fix ever existed -- gets completed here
    without a human needing to notice.
    """
    if not config.watchdog.enabled:
        return

    def _drift_fingerprint(**parts: Any) -> str:
        """Stable fingerprint for an orphaned-worker drift finding."""
        return json.dumps(parts, sort_keys=True, default=str)

    with state_lock(state_file):
        state = load_state(state_file)

    orphaned_issues: list[int] = []
    for issue_number_str, entry in state.get("issues", {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "dispatched":
            continue

        if not _worker_pid_alive(entry):
            orphaned_issues.append(int(issue_number_str))

    if not orphaned_issues:
        return

    # Fetch PRs once before acquiring the lock (avoid network I/O under lock)
    prs = gh.pr_list()
    pr_by_issue: dict[int, dict[str, Any]] = {}
    for pr in prs:
        linked = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        if linked is not None:
            pr_by_issue[linked] = pr

    # Issue #417: ground-truth label reclaim for the no-open-PR orphans, done
    # OUTSIDE the state lock (network I/O) and with a single bulk issue-list
    # call rather than one gh.issue_view per orphan -- this sweep's GitHub
    # cost must stay bounded regardless of how many stale "dispatched" entries
    # have accumulated in state.json over time.
    no_pr_orphans = [n for n in orphaned_issues if n not in pr_by_issue]
    reclaim_results: dict[int, dict[str, Any]] = {}
    if no_pr_orphans:
        issues_by_number: dict[int, dict[str, Any]] = {}
        for issue in gh.issue_list(state="open"):
            number = issue.get("number")
            if number is not None:
                issues_by_number[int(number)] = issue
        for issue_number in no_pr_orphans:
            issue = issues_by_number.get(issue_number)
            if issue is None:
                # Issue not found in the open snapshot (closed, deleted, or
                # inaccessible) -- nothing safe to reclaim; leave it to the
                # existing diagnostic drift path below.
                continue
            issue_labels = label_names(issue)
            active_labels = issue_labels & config.labels.active
            # Gate the WHOLE reclaim on an active label actually being
            # present, matching reconcile.py's issue_active_label_no_open_pr
            # pattern (~536-580) so all three sites agree. An issue with no
            # active label -- e.g. one carrying only a terminal label like
            # agent:human-needed/agent:done/agent:blocked -- has nothing here
            # to reclaim. A prior `if not active_labels and not needs_ready`
            # gate proceeded whenever EITHER half was false, which wrongly
            # added `ready` back onto a terminal-only issue that also had a
            # stale dispatched/dead-worker/no-PR state.json entry (already
            # fully reconciled issues, or terminal-only ones, both correctly
            # fall through here without any GitHub call).
            if not active_labels:
                continue
            needs_ready = config.labels.ready not in issue_labels
            label_write_ok = True
            for label in sorted(active_labels):
                if not gh.remove_issue_label(issue_number, label):
                    label_write_ok = False
            if needs_ready:
                if not gh.add_issue_label(issue_number, config.labels.ready):
                    label_write_ok = False
            reclaim_results[issue_number] = {
                "removed_labels": sorted(active_labels),
                "added_ready": needs_ready,
                "label_write_ok": label_write_ok,
            }

    # Issue #439: route dead workers with stuck pre-review PRs to rework before
    # the state-update sweep. PR views are fetched outside the state lock; the
    # route helper updates state/labels in its own critical section. The second
    # lock below will then skip issues that have already moved to
    # rework_requested/escalated.
    pre_review_routed: set[int] = set()
    state_snapshot = state
    now = datetime.now(UTC)
    for issue_number in orphaned_issues:
        pr_data = pr_by_issue.get(issue_number)
        if not pr_data:
            continue
        pr_number = int(pr_data["number"])
        pr_state = state_snapshot.get("prs", {}).get(str(pr_number), {})
        last_decision = pr_state.get("decision")
        reviewed_head_sha = pr_state.get("reviewed_head_sha")
        live_head_sha = pr_data.get("headRefOid")
        if (
            last_decision == "request_changes"
            and reviewed_head_sha
            and live_head_sha
            and reviewed_head_sha == live_head_sha
        ):
            # Let the second-lock request_changes restoration path handle this;
            # do not overwrite an existing review feedback prompt.
            continue
        try:
            pr_view = gh.pr_view(pr_number)
        except Exception:
            pr_view = None
        enriched = pr_view if pr_view else pr_data
        is_candidate, reason = _is_pre_review_rework_candidate(enriched, config, now)
        if is_candidate:
            route_result = _route_dead_worker_to_pre_review_rework(
                state_file,
                gh,
                config,
                enriched,
                issue_number,
                reason,
                failure_kind=None,
            )
            if route_result is not None:
                pre_review_routed.add(issue_number)

    # Handle orphaned workers. Head-advanced request_changes findings are
    # collected and routed to the review lane outside the state lock (review()
    # itself acquires the lock and may call transition()).
    review_routes: list[tuple[int, int, str, str, str]] = []
    with state_lock(state_file):
        state = load_state(state_file)
        sweep_events: list[tuple[str, dict[str, Any]]] = []
        for issue_number in orphaned_issues:
            entry = state["issues"].get(str(issue_number), {})
            if not isinstance(entry, dict):
                continue

            # Re-verify status (state may have changed between lock windows)
            if entry.get("status") != "dispatched":
                continue

            # Issue #282: do not clear the liveness fingerprint here. The worker
            # is dead (``_worker_pid_alive`` returned False), but the PID record
            # may still be needed by the recovery path to decide whether the
            # worktree is safe to reset.

            pr_data = pr_by_issue.get(issue_number)

            if pr_data:
                pr_number = int(pr_data["number"])
                # Check the last review decision from state
                pr_state = state.get("prs", {}).get(str(pr_number), {})
                last_decision = pr_state.get("decision")
                reviewed_head_sha = pr_state.get("reviewed_head_sha")
                live_head_sha = pr_data.get("headRefOid")

                if last_decision == "request_changes" and reviewed_head_sha and live_head_sha:
                    if reviewed_head_sha == live_head_sha:
                        # Safe to reset to rework_requested - PR head unchanged since request_changes
                        entry["status"] = "rework_requested"
                        entry["dispatched_at"] = None
                        sweep_events.append(
                            (
                                "orphaned_worker_recovered",
                                {
                                    "issue_number": issue_number,
                                    "pr_number": pr_number,
                                    "previous_status": "dispatched",
                                    "new_status": "rework_requested",
                                    "reason": "dead_worker_with_request_changes",
                                },
                            )
                        )
                    else:
                        # PR head has changed - route to review if possible,
                        # otherwise surface as a drift finding (once per fingerprint).
                        fingerprint = _drift_fingerprint(
                            reason="dead_worker_with_head_change",
                            reviewed_head_sha=reviewed_head_sha,
                            live_head_sha=live_head_sha,
                        )
                        if entry.get("orphan_drift_fingerprint") == fingerprint:
                            # Already handled/failed for this exact head advance;
                            # don't re-emit or retry.
                            state["issues"][str(issue_number)] = entry
                            continue
                        if review_callback is not None:
                            review_routes.append(
                                (
                                    issue_number,
                                    pr_number,
                                    reviewed_head_sha,
                                    live_head_sha,
                                    fingerprint,
                                )
                            )
                        else:
                            entry["orphan_drift_fingerprint"] = fingerprint
                            entry["orphan_drift_at"] = utc_now()
                            sweep_events.append(
                                (
                                    "orphaned_worker_drift",
                                    {
                                        "issue_number": issue_number,
                                        "pr_number": pr_number,
                                        "previous_status": "dispatched",
                                        "last_decision": last_decision,
                                        "reviewed_head_sha": reviewed_head_sha,
                                        "live_head_sha": live_head_sha,
                                        "reason": "dead_worker_with_head_change",
                                    },
                                )
                            )
                else:
                    # Not a simple request_changes case - surface as drift once.
                    fingerprint = _drift_fingerprint(
                        reason="dead_worker_unsafe_to_auto_reset",
                        last_decision=last_decision or "",
                        pr_number=pr_number,
                    )
                    if entry.get("orphan_drift_fingerprint") != fingerprint:
                        entry["orphan_drift_fingerprint"] = fingerprint
                        entry["orphan_drift_at"] = utc_now()
                        sweep_events.append(
                            (
                                "orphaned_worker_drift",
                                {
                                    "issue_number": issue_number,
                                    "pr_number": pr_number,
                                    "previous_status": "dispatched",
                                    "last_decision": last_decision,
                                    "reason": "dead_worker_unsafe_to_auto_reset",
                                },
                            )
                        )
            else:
                # Issue #417: report (and, on success, resolve) the ground-truth
                # label reclaim computed above before falling back to the
                # unresolved-drift diagnostic. This is what makes the reap
                # convergent -- an interrupted or partially-failed attempt by
                # the sidecar-based lane is finished here, and a genuinely
                # failed label write is retried again next pass (this reclaim
                # never gates on orphan_flagged_at, only the diagnostic below
                # does).
                reclaim = reclaim_results.get(issue_number)
                if reclaim is not None:
                    sweep_events.append(
                        (
                            "session_failed_relabeled",
                            {
                                "issue_number": issue_number,
                                "reason": "dead_worker_no_open_pr_orphan_sweep",
                                **reclaim,
                            },
                        )
                    )
                    if reclaim["label_write_ok"]:
                        # Fully reclaimed: labels are correct, nothing left to
                        # flag as unresolved drift. `status` deliberately stays
                        # "dispatched" here (matching the sidecar-based lane's
                        # own issue #282 fingerprint-preservation choice), so
                        # this same entry would otherwise be re-discovered by
                        # this sweep's very next pass and, having no more
                        # labels left to touch, fall through to the
                        # orphan_flagged_at diagnostic below and emit a
                        # spurious orphaned_worker_drift for an issue that is
                        # already fixed. Mark it flagged now so that never
                        # happens -- this lane's reclaim retry above never
                        # gates on this flag, only the diagnostic does.
                        entry["orphan_flagged_at"] = utc_now()
                        state["issues"][str(issue_number)] = entry
                        continue

                # No open PR - emit drift event, leave (further) recovery to
                # this same sweep's next pass, which re-attempts the ground-
                # truth label reclaim above unconditionally regardless of the
                # flag set here (issue #118 mop-up remains a manual fallback).
                # Issue #259: mark the entry so it is not re-flagged every pass.
                # Suppress ONLY the duplicate no-open-PR event; with-PR recovery
                # paths must run regardless of the flag.
                if entry.get("orphan_flagged_at"):
                    state["issues"][str(issue_number)] = entry
                    continue
                entry["orphan_flagged_at"] = utc_now()
                sweep_events.append(
                    (
                        "orphaned_worker_drift",
                        {
                            "issue_number": issue_number,
                            "previous_status": "dispatched",
                            "reason": "dead_worker_no_open_pr",
                        },
                    )
                )

            state["issues"][str(issue_number)] = entry

        state = _append_sweep_events(
            state, sweep_events, max_size=config.runtime.event_ring_size, state_file=state_file
        )
        save_state(state_file, state)

    # Route head-advanced request_changes findings to the review lane outside
    # the state lock. review() generates the packet, fires the review_started
    # label transition, and returns ok when a fresh packet is produced. We then
    # flip the issue status to "reviewing" so it is not re-detected as an orphan
    # on every subsequent pass. If review() fails, we record a drift fingerprint
    # so the identical finding is not re-emitted every pass.
    for (
        issue_number,
        pr_number,
        reviewed_head_sha_before,
        live_head_sha,
        fingerprint,
    ) in review_routes:
        if review_callback is None:
            continue
        review_result = review_callback(pr_number)
        routed = False
        # See _route_rework_candidate_to_review's matching comment: review()
        # can return ok=True for the janitor-gate conflict/no-op-rework route
        # (no packet, no review_started transition) as well as for a real
        # packet. Only a real packet should flip this orphaned-but-dispatched
        # issue to "reviewing".
        routed_to_rework = bool(review_result.data.get("routed_to_rework"))
        # Issue #558: review() also returns ok=True when it converges a
        # CLOSED-unmerged PR's state entry to "closed" at the janitor gate.
        # That is not a fresh packet -- the PR is dead, not transiently
        # blocked -- so it must NOT flip this issue to "reviewing" (an
        # ACTIVE_STATE_STATUS no reconcile rule clears while the GitHub
        # issue itself stays open: issue_active_label_no_open_pr sees the
        # closed PR still links to the issue, issue_active_label_with_open_pr
        # sees no OPEN PR, and the unknown-status recompute sweep skips
        # "reviewing" because it is a VALID_ISSUE_STATUSES member). The
        # issue's disposition is left to the existing closed-unmerged
        # issue-side handling (closed_unmerged_pr_active_labels). Neither
        # the "reviewing" flip nor the transient-block drift fingerprint
        # below applies to a permanently-dead PR.
        closed_unmerged_converged = bool(review_result.data.get("closed_unmerged_converged"))
        with state_lock(state_file):
            state = load_state(state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            entry = state["issues"].get(str(issue_number), {})
            decision_unchanged = pr_state.get("reviewed_head_sha") == reviewed_head_sha_before
            if (
                review_result.ok
                and not routed_to_rework
                and not closed_unmerged_converged
                and decision_unchanged
                and isinstance(entry, dict)
                and entry.get("status") == "dispatched"
            ):
                state["issues"][str(issue_number)] = {**entry, "status": "reviewing"}
                routed = True
            elif (
                not review_result.ok
                and not routed_to_rework
                and isinstance(entry, dict)
                and entry.get("status") == "dispatched"
            ):
                # Review failed: mark the drift fingerprint so the next pass
                # does not retry/re-emit for this unchanged head.
                state["issues"][str(issue_number)] = {
                    **entry,
                    "orphan_drift_fingerprint": fingerprint,
                    "orphan_drift_at": utc_now(),
                }
            state = append_event(
                state,
                "orphaned_worker_routed_to_review"
                if review_result.ok
                else "orphaned_worker_drift",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "review_ok": review_result.ok,
                    "routed": routed,
                    "live_head_sha": live_head_sha,
                    "reviewed_head_sha": reviewed_head_sha_before,
                    "reason": "dead_worker_with_head_change",
                },
                state_path=state_file,
            )
            save_state(state_file, state)


def _sweep_orphan_processes_for_dead_sessions(
    sessions_dir: Path, state_file: Path, config: OrchestratorConfig
) -> None:
    """Sweep for orphan processes in worktrees of dead sessions.

    This is called from the production loop to detect and clean up orphaned
    processes that survived session kills (e.g., detached/daemonized processes
    like nohup-style background processes). This addresses issue #139.

    On Windows: Uses PowerShell Get-CimInstance Win32_Process to find processes
    whose CommandLine references the worktree path of dead sessions.
    On POSIX: Not implemented (returns empty list).

    Detected orphans are killed automatically and logged to state.json.
    """
    from .devin_shell import is_session_alive, read_session_records
    from .claude_code import is_worker_alive, read_worker_records

    # Only run on Windows where the issue occurs
    if os.name != "nt":
        return

    # Collect worktree paths of dead sessions
    dead_worktree_paths: set[str] = set()

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not is_session_alive(record):
            dead_worktree_paths.add(record.worktree_path)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue
        if not is_worker_alive(record):
            dead_worktree_paths.add(record.worktree_path)

    # Sweep for orphans in each dead worktree
    for worktree_path in dead_worktree_paths:
        orphan_processes = sweep_orphan_processes(worktree_path)
        if orphan_processes:
            # Kill detected orphans
            killed_orphans: list[int] = []
            for orphan in orphan_processes:
                _kill_orphan_pid(orphan["pid"])
                killed_orphans.append(orphan["pid"])

            # Log the event with image/cmdline of each killed process so the
            # respawn source in dead worktrees can be identified and shut off.
            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "orphan_processes_killed",
                    {
                        "worktree_path": worktree_path,
                        "orphan_pids": [o["pid"] for o in orphan_processes],
                        "killed_orphans": killed_orphans,
                        "orphan_processes": [
                            {
                                "pid": o["pid"],
                                "name": o.get("name"),
                                "command_line": o.get("command_line"),
                            }
                            for o in orphan_processes
                        ],
                    },
                    state_path=state_file,
                )
                save_state(state_file, state)


def _log_worker_census(sessions_dir: Path) -> None:
    """Log one INFO line per loop pass listing every currently-alive worker.

    Issue #646: the launch-time log in claude_code.py/devin_shell.py answers
    "what cap did this worker launch with", but says nothing about how many
    are running *right now* -- the question a box-saturation incident needs
    answered ("how many suites were running at 11:33, from which worktrees,
    at what cap"). This sweep answers it directly from log content alone,
    with no process forensics required.

    Deliberately read-only (no state mutation): it carries none of the
    fragile sole-writer invariants _detect_and_handle_stalled_sessions/
    _sweep_orphan_processes_for_dead_sessions must protect. It also only
    ever iterates *alive* records, so — unlike a dead/exit-transition sweep —
    it has no "months of accumulated stale sidecar" flooding problem even if
    old sidecars are never pruned from sessions_dir.

    Called from the top of ``dispatch()`` (see its docstring) -- the one
    chokepoint every dispatch path funnels through, standalone (`work`/`fleet
    work`) or supervised (`loop()` -> `_loop_body()` -> `dispatch()`) -- so it
    runs unconditionally regardless of how long the orchestrator process
    itself lives: a one-shot ``charlie fleet work`` CLI invocation logs
    exactly one census line before exiting; a long-lived ``charlie fleet
    supervise`` logs one per pass. Both answer the diagnostic question above
    from log content alone -- no need to correlate against a live process
    list.
    """
    import logging

    logger = logging.getLogger(__name__)

    from .claude_code import is_worker_alive, read_worker_records
    from .devin_shell import is_session_alive, read_session_records

    now = datetime.now(UTC)

    def _age_seconds(started_at: str) -> int | None:
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return None
        return int((now - started).total_seconds())

    entries: list[str] = []
    # adapter_kind=None also covers the "api" adapter, which delegates to
    # claude_code.launch_claude_worker under the hood and shares its sidecar
    # schema (and therefore xdist_cap/is_worker_alive) unchanged.
    for record in read_worker_records(sessions_dir, adapter_kind=None):
        if record.pid is None or record.error is not None or not is_worker_alive(record):
            continue
        entries.append(
            f"(adapter={record.adapter_kind} issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None or not is_session_alive(record):
            continue
        entries.append(
            f"(adapter=devin-shell issue={record.issue_number} "
            f"worktree={record.worktree_path} pid={record.pid} "
            f"cap={record.xdist_cap} age_s={_age_seconds(record.started_at)})"
        )

    logger.info("worker census: n_alive=%d %s", len(entries), " ".join(entries) or "[]")


def _rework_pr_for_worker(
    open_prs_by_issue: dict[int, list[dict[str, Any]]],
    worker: WorkerView,
) -> dict[str, Any] | None:
    """Return the most likely open PR for a dead/launch-failed rework worker.

    Prefer the PR whose ``headRefName`` matches the worker's branch, falling back
    to the lowest PR number for the issue.
    """
    prs = open_prs_by_issue.get(worker.issue_number, [])
    if not prs:
        return None
    for pr in prs:
        if pr.get("headRefName") == worker.branch:
            return pr
    return min(prs, key=lambda pr: int(pr["number"]))


def _reap_restore_rework_requested(
    state_file: Path,
    gh: GitHub,
    config: OrchestratorConfig,
    open_prs_by_issue: dict[int, list[dict[str, Any]]],
    worker: WorkerView,
    failure_kind: str | None = None,
) -> None:
    """Restore a dead/launch-failed rework worker to ``rework_requested``, or
    escalate it to a human when the redispatch cap is exhausted or the
    failure is deterministic.

    Called when a reaped session has an open linked PR. The PR must have a
    ``request_changes`` verdict that is still LIVE for the current head
    (``reviewed_head_sha == live_head_sha``) — ``has_request_changes`` is the
    single source of truth here. A ``rework-prompt.md`` on disk is only ever
    a supplement to that signal, never an independent trigger (issue #315
    finding 1): the prompt file is written once per PR by
    ``_write_rework_prompt`` and is never deleted, so by itself it cannot
    distinguish "still awaiting this exact rework cycle" from "stale leftover
    from an earlier cycle whose head has since been approved or superseded".
    ``has_request_changes`` already re-derives from the PR's current review
    record on every call, so gating on it makes the whole check
    self-invalidating the moment the head advances or gets approved — the
    same property the prompt-existence check was missing.

    Issue #295: this is the rework counterpart to the no-open-PR relabel path.
    It preserves the liveness fingerprint (``worker_pid`` /
    ``worker_process_start_time``) for the recovery probe, matching the
    no-open-PR dead-session escalation branch (issue #282).

    Issue #315 finding 2: this lane must consult the same redispatch-cap and
    deterministic-failure-kind escalation rules the no-open-PR lanes already
    enforce (~line 936 and ~line 1138) — otherwise a rework worker that dies
    at launch every time (or whose failure_kind is confirmed-deterministic,
    e.g. ``worktree_unsafe``) loops rework_requested forever instead of
    escalating to a human. Rework workers always have an open PR, so they
    never reach those lanes' checks; the equivalent must live here.
    """
    pr_data = _rework_pr_for_worker(open_prs_by_issue, worker)
    if pr_data is None:
        return

    pr_number = int(pr_data["number"])
    live_head_sha = pr_data.get("headRefOid")
    prs_dir = state_file.parent / "prs"
    rework_prompt_path = prs_dir / f"pr-{pr_number}" / "rework-prompt.md"

    with state_lock(state_file):
        state = load_state(state_file)
        entry = state["issues"].get(str(worker.issue_number), {})
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            return

        pr_state = state.get("prs", {}).get(str(pr_number), {})
        last_decision = pr_state.get("decision")
        reviewed_head_sha = pr_state.get("reviewed_head_sha")

        has_request_changes = (
            last_decision == "request_changes"
            and reviewed_head_sha is not None
            and live_head_sha is not None
            and reviewed_head_sha == live_head_sha
        )
        # Diagnostic only (issue #315 finding 1) — never gates the restore by
        # itself; see the docstring above.
        has_rework_prompt = rework_prompt_path.exists()

        if not has_request_changes:
            return

        # Issue #315 finding 2: same window-filtered redispatch_at bookkeeping
        # the sibling lanes use (~line 950-961, ~4186-4194), so the cap below
        # is actually consulted instead of silently never growing.
        redispatch_at = _windowed_redispatch_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]

        terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
        should_escalate = (
            terminal_failure or len(redispatch_at) > config.watchdog.max_auto_redispatch
        )

        if should_escalate:
            entry["status"] = "escalated"
            entry["redispatch_at"] = redispatch_at
            entry["escalation_reason"] = (
                failure_kind if terminal_failure else "redispatch_cap_exceeded"
            )
            # Preserve worker_pid/worker_process_start_time (issue #282): the
            # recovery probe still needs the fingerprint even after escalation.
            state["issues"][str(worker.issue_number)] = entry
            state = append_event(
                state,
                "session_failed_escalated",
                {
                    "issue_number": worker.issue_number,
                    "pr_number": pr_number,
                    "failure_kind": failure_kind,
                    "previous_status": "dispatched",
                    "reason": "dead_rework_session_escalated",
                    "redispatch_count": len(redispatch_at),
                },
                state_path=state_file,
            )
            save_state(state_file, state)
        else:
            entry["status"] = "rework_requested"
            entry["dispatched_at"] = None
            entry["redispatch_at"] = redispatch_at
            # Preserve worker_pid (issues #165, #282, #295)
            state["issues"][str(worker.issue_number)] = entry
            state = append_event(
                state,
                "rework_requeued",
                {
                    "issue_number": worker.issue_number,
                    "pr_number": pr_number,
                    "failure_kind": failure_kind,
                    "previous_status": "dispatched",
                    "reason": "dead_rework_session_recovered",
                    "has_request_changes": has_request_changes,
                    "has_rework_prompt": has_rework_prompt,
                },
                state_path=state_file,
            )
            save_state(state_file, state)

    # Transition labels: escalate to human_needed, or rework_requested
    # (needs_rework), removing the stale in_progress label from the failed launch.
    edge = "redispatch_escalated" if should_escalate else "rework_requested"
    result = transition(gh, config.labels, worker.issue_number, edge)
    if result.outcome != TransitionOutcome.APPLIED:
        with state_lock(state_file):
            state = load_state(state_file)
            entry = state["issues"].get(str(worker.issue_number), {})
            entry["label_error"] = {
                "edge": edge,
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
            state["issues"][str(worker.issue_number)] = entry
            save_state(state_file, state)


def _rework_prompt_search_dirs(
    config: OrchestratorConfig, repo_root: Path | None = None
) -> tuple[Path, ...]:
    """Resolve the optional repo-local prompt override directory."""
    prompts_dir = config.runtime.prompts_dir
    if not prompts_dir:
        return ()
    path = Path(prompts_dir)
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    return (path,)


def _render_required_changes_section(decision: dict[str, Any] | None) -> str:
    """Render the ``$required_changes_section`` for a rework brief.

    The findings are read from ``review-decision.json``. ``required_changes``
    -- the most actionable output the review pipeline produces -- is the
    primary source. Giving it its own rendered section (instead of
    multiplexing one prose slot) means an operational dispatch note can no
    longer displace it.

    Measured across the on-disk corpus, ``request_changes`` verdicts with a
    populated ``required_changes`` are the exception (0 of 20 observed):
    ``prompts/review.md`` historically documented the field as optional, so
    reviewers reliably fill in ``summary`` and skip ``required_changes``.
    That ``summary`` is real, substantive review content -- discarding it
    because the structured list is empty silently sends a worker a brief
    with nothing to act on. So for a ``request_changes`` verdict this
    function degrades through three tiers: (1) the enumerated
    ``required_changes`` list when non-empty, (2) the verdict's ``summary``
    rendered verbatim and clearly labelled as prose when the list is empty,
    (3) an explicit, loud "findings unavailable, check the PR on GitHub"
    instruction when BOTH are empty. Tier 3 is a hard invariant: this
    function must never render (or omit into silence) something a worker
    could read as "there is nothing to change" when a decision actively
    requires rework.

    Rendered for ``request_changes`` and, defensively, ``blocked`` verdicts
    (routing to rework via the decision-agnostic janitor gates -- merge
    conflict / no-op-rework repair -- can carry forward whatever verdict was
    last on disk, including ``blocked``). ``approved`` never renders
    anything here; its findings (if any) reach the reviewer via
    ``$prior_review_section`` instead, not the worker's rework brief.

    For ``blocked`` specifically, the enumerated list and the summary
    fallback (tiers 1-2) are intentionally suppressed even when populated.
    ``prompts/review.md`` requires ``required_changes`` for ``blocked`` just
    as it does for ``request_changes``, but "what must change before this PR
    can be approved" language is the wrong framing for the routes that reach
    this decision-agnostic branch (merge-conflict / no-CI / cross-pr-revert
    routes, which explicitly tell the worker "do not re-litigate the
    review") -- an ``approved`` verdict can also legitimately carry a
    non-empty ``required_changes`` left over from an earlier round, which is
    the same contradiction. Tier 3 (the both-empty escape hatch) still
    applies to ``blocked`` -- suppressing content is fine, but suppressing
    it AND leaving the worker with no signal that something was withheld is
    not.

    Returns an empty string only when there is no decision, or the decision
    is not ``request_changes``/``blocked``, or it is ``blocked`` with some
    (but not zero) findings content -- every other case renders something.
    """
    if not isinstance(decision, dict):
        return ""
    verdict = decision.get("decision")
    if verdict not in ("request_changes", "blocked"):
        return ""

    raw_required_changes = decision.get("required_changes")
    changes = (
        [str(item).strip() for item in raw_required_changes if str(item).strip()]
        if isinstance(raw_required_changes, list)
        else []
    )
    raw_summary = decision.get("summary")
    summary_text = raw_summary.strip() if isinstance(raw_summary, str) else ""

    if verdict == "request_changes" and changes:
        lines = [
            "## Required changes",
            "",
            "Address every item below. These are the reviewer's structured "
            "findings — the authoritative list of what must change before this "
            "PR can be approved.",
            "",
        ]
        lines.extend(f"- {change}" for change in changes)
        lines.append("")
        return "\n".join(lines)

    if verdict == "request_changes" and summary_text:
        lines = [
            "## Required changes",
            "",
            "The reviewer did not record a structured findings list for this "
            "verdict. This is their summary, rendered verbatim so it is not "
            "lost — treat it as the findings to address before this PR can "
            "be approved:",
            "",
            summary_text,
            "",
        ]
        return "\n".join(lines)

    if not changes and not summary_text:
        lines = [
            "## Required changes",
            "",
            "**REVIEWER FINDINGS UNAVAILABLE.** No structured findings list "
            "and no summary were recorded for this verdict. This is NOT a "
            "signal that there is nothing to change — it means the findings "
            "did not make it into `review-decision.json`. Inspect the PR's "
            "review comments and review threads on GitHub directly before "
            "doing anything else.",
            "",
        ]
        return "\n".join(lines)

    # blocked verdict carrying required_changes and/or a summary (but not
    # both empty): suppressed by design (see docstring) so nothing renders.
    return ""


def _read_review_decision(decision_path: Path) -> dict[str, Any] | None:
    """Read a PR's ``review-decision.json`` as a dict, or ``None`` if absent.

    Mirrors ``OrchestratorApp._review_decision``'s read shape but returns
    ``None`` (rather than a sentinel) for missing/invalid files so the caller
    can distinguish "no verdict on disk" from "a verdict with an empty
    findings list" — only the latter should render an (empty) section.
    """
    if not decision_path.exists():
        return None
    try:
        with decision_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_verdict_newer_than_brief(decision_path: Path, brief_path: Path) -> bool:
    """Return True when the verdict file is strictly newer than the brief.

    Used by ``dispatch_rework`` to detect a stale brief that has drifted from
    a corrected ``review-decision.json`` (issue #632: the brief on disk is
    authoritative and ``dispatch_rework`` reads it verbatim, so without this
    check a hand-corrected verdict — the #510 case — never reaches the
    worker). Comparison uses nanosecond mtimes; an equal timestamp (the
    normal verdict path writes the decision immediately before the brief) is
    treated as not-stale so the fresh brief is not pointlessly rewritten.
    """
    if not decision_path.exists() or not brief_path.exists():
        return False
    return decision_path.stat().st_mtime_ns > brief_path.stat().st_mtime_ns


def _write_rework_prompt(
    state_file: Path,
    pr: dict[str, Any],
    issue_number: int | None,
    dispatch_note: str,
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Write a rework brief for a PR under the shared ``prs/pr-{N}`` directory.

    This module-level helper lets both the OrchestratorApp review path and the
    dead-session recovery path produce the same ``rework-prompt.md`` artifact.

    Single point of enforcement for issue #632: the reviewer's structured
    ``required_changes`` are read from ``review-decision.json`` here — not
    threaded through by callers — so the three call sites cannot omit them.
    The ``dispatch_note`` (formerly the ``$review_summary`` slot) carries the
    operational/review-prose note and is kept separate from the findings, so
    a churn/rescue message accompanies the findings instead of replacing
    them. The raw note is also written to a sidecar
    (``rework-dispatch-note.txt``) so a stale brief can be regenerated at
    dispatch time without losing its note.
    """
    pr_number = int(pr["number"])
    pr_dir = state_file.parent / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "rework-prompt.md"
    decision = _read_review_decision(pr_dir / "review-decision.json")
    required_changes_section = _render_required_changes_section(decision)
    prompt = render_prompt(
        config.dispatch.rework_template,
        {
            "pr_number": pr_number,
            "pr_title": pr.get("title", ""),
            "pr_url": pr.get("url", ""),
            "issue_number": issue_number or "UNKNOWN",
            "dispatch_note": dispatch_note,
            "required_changes_section": required_changes_section,
            "branch_name": pr.get("headRefName", ""),
        },
        search_dirs=_rework_prompt_search_dirs(config, repo_root=repo_root),
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    # Sidecar: the raw dispatch note, so a dispatch-time regeneration (when
    # review-decision.json is newer than the brief) can reproduce the note
    # without parsing the rendered markdown.
    (pr_dir / "rework-dispatch-note.txt").write_text(dispatch_note, encoding="utf-8")
    return prompt_path


def _is_pr_updated_at_older_than(
    pr: dict[str, Any],
    now: datetime,
    minutes: int,
) -> bool:
    """Return True when ``pr["updatedAt"]`` is more than ``minutes`` old.

    Parses ISO-8601 timestamps with an optional ``Z`` suffix, normalizes
    naive datetimes to UTC, and tolerates missing or malformed values.
    """
    updated_at = pr.get("updatedAt")
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - updated).total_seconds() > minutes * 60


def _is_pre_review_rework_candidate(
    pr: dict[str, Any],
    config: OrchestratorConfig,
    now: datetime,
) -> tuple[bool, str]:
    """Detect PRs that are stuck before review and need a rework cycle.

    Returns ``(True, reason)`` when either:

    * ``mergeable`` is ``CONFLICTING`` — the branch cannot be merged and CI
      will not run because GitHub cannot build a merge ref; or
    * ``mergeStateStatus`` is ``DIRTY`` — the rework branch conflicts with the
      base, so the merge ref cannot be built and no ``pull_request`` CI can run; or
    * ``statusCheckRollup`` is empty and the PR's ``updatedAt`` is older than
      ``watchdog.pre_review_rework_stale_minutes`` — the worker opened a PR
      and then died before any checks were created.
    """
    mergeable = str(pr.get("mergeable") or "").upper()
    if mergeable == "CONFLICTING":
        return True, "merge_conflict"

    merge_state = str(pr.get("mergeStateStatus") or "").upper()
    if merge_state == "DIRTY":
        return True, "rework_branch_conflict"

    stale_minutes = config.watchdog.pre_review_rework_stale_minutes
    if stale_minutes <= 0:
        return False, ""

    status_rollup = pr.get("statusCheckRollup")
    if status_rollup:
        return False, ""

    if _is_pr_updated_at_older_than(pr, now, stale_minutes):
        return True, "stale_empty_checks"

    return False, ""


def _is_readiness_no_ci_stall(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    config: AutoMergeConfig,
    now: datetime,
) -> bool:
    """Detect an approved PR whose required checks have never started.

    Returns True when:
      * ``pr_checks`` returned a parseable (non-None) list;
      * none of the configured ``required_checks`` appear in that list;
      * the PR's ``updatedAt`` is older than ``readiness_no_ci_minutes``.

    The required check names come from ``config.required_checks``; no names are
    hard-coded. ``updatedAt`` is the best available proxy for "head SHA pushed"
    in the ``gh pr view`` JSON field list.
    """
    no_ci_minutes = config.readiness_no_ci_minutes
    if no_ci_minutes <= 0:
        return False
    required = config.required_checks
    if not required:
        return False
    seen = {str(check.get("name") or "") for check in checks}
    if any(name in seen for name in required):
        return False
    return _is_pr_updated_at_older_than(pr, now, no_ci_minutes)


def _route_dead_worker_to_pre_review_rework(
    state_file: Path,
    gh: GitHub,
    config: OrchestratorConfig,
    pr: dict[str, Any],
    issue_number: int,
    reason: str,
    *,
    failure_kind: str | None = None,
) -> dict[str, Any] | None:
    """Route a dead worker's stuck pre-review PR to the rework pipeline.

    Writes a rebase-onto-main brief, transitions the issue to ``needs_rework``,
    and updates state.json to ``rework_requested``. Idempotent: if the issue is
    already ``rework_requested`` or ``escalated``, this is a no-op.

    Enforces ``watchdog.max_auto_redispatch`` and escalates deterministic
    failures immediately, mirroring the existing redispatch-escalation logic.
    """
    pr_number = int(pr["number"])
    if reason == "merge_conflict":
        summary = (
            "The PR branch has a merge conflict with the base branch. "
            "Rebase the branch onto the current base branch, resolve the conflicts, "
            "and push. The code changes are already approved; do not re-litigate the review."
        )
    elif reason == "rework_branch_conflict":
        summary = (
            "The rework branch conflicts with the current base branch; GitHub cannot "
            "build the merge ref, so no pull_request CI will run. Resolve the conflicts "
            "manually and push."
        )
        if failure_kind is None:
            failure_kind = "rework_branch_conflict"
    else:
        summary = (
            "The PR was opened but no CI checks have been created after the stale threshold. "
            "Rebase the branch onto the current base branch and push to trigger a fresh CI run. "
            "The existing changes are pre-approved; do not re-litigate the review."
        )

    with state_lock(state_file):
        state = load_state(state_file)
        state.setdefault("issues", {})
        state.setdefault("prs", {})
        entry = state["issues"].get(str(issue_number), {})
        if not isinstance(entry, dict):
            entry = {}
        current_status = entry.get("status")
        if current_status in ("rework_requested", "escalated"):
            return None

        redispatch_at = _windowed_redispatch_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]

        terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
        if terminal_failure or len(redispatch_at) > config.watchdog.max_auto_redispatch:
            entry = {
                **entry,
                "number": issue_number,
                "status": "escalated",
                "redispatch_at": redispatch_at,
                "escalation_reason": failure_kind
                if terminal_failure
                else "redispatch_cap_exceeded",
                "pre_review_rework_reason": reason,
            }
            state["issues"][str(issue_number)] = entry
            save_state(state_file, state)
            result = transition(gh, config.labels, issue_number, "redispatch_escalated")
            if result.outcome != TransitionOutcome.APPLIED:
                entry = state["issues"].get(str(issue_number), {})
                if isinstance(entry, dict):
                    entry = {
                        **entry,
                        "label_error": {
                            "edge": "redispatch_escalated",
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        },
                    }
                    state["issues"][str(issue_number)] = entry
                    save_state(state_file, state)
            return {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "reason": reason,
                "escalated": True,
                "escalation_reason": entry["escalation_reason"],
            }

        repo_root = getattr(gh, "repo_root", None)
        _write_rework_prompt(state_file, pr, issue_number, summary, config, repo_root=repo_root)
        entry = {
            **entry,
            "number": issue_number,
            "status": "rework_requested",
            "dispatched_at": None,
            "pre_review_rework_reason": reason,
        }
        state["issues"][str(issue_number)] = entry
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "number": pr_number,
            "issue_number": issue_number,
            "status": "rework_requested",
        }
        state = append_event(
            state,
            "pre_review_rework_routed",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "reason": reason,
                "failure_kind": failure_kind,
            },
            state_path=state_file,
        )
        save_state(state_file, state)

    result = transition(gh, config.labels, issue_number, "rework_requested")
    label_error = None
    if result.outcome != TransitionOutcome.APPLIED:
        label_error = {
            "edge": "rework_requested",
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        }
        with state_lock(state_file):
            state = load_state(state_file)
            entry = state["issues"].get(str(issue_number), {})
            if isinstance(entry, dict):
                entry = {**entry, "label_error": label_error}
                state["issues"][str(issue_number)] = entry
                save_state(state_file, state)

    return {
        "issue_number": issue_number,
        "pr_number": pr_number,
        "reason": reason,
        "label_error": label_error,
    }


def _classify_dead_sessions_and_update_throttle_state(
    sessions_dir: Path,
    state_file: Path,
    gh: GitHub,
    config: OrchestratorConfig,
    *,
    persist_inconclusive_probe_counter: bool = True,
) -> list[dict[str, Any]]:
    """Check for dead sessions, classify their failures, and update throttle state.

    This is called from the production loop to detect provider throttling
    from worker deaths and set the cooldown window in state.json.

    Also reconciles labels for dead sessions with no open PR (issue #118):
    a dead worker with no open PR is recoverable and should be relabeled
    as dispatchable (remove active labels, ensure ready label present).

    Issue #252: if a dead worker has a clean worktree with unpushed commits
    (completed-but-unpublished), push the branch, create a PR, and move the
    issue to ``pr_open`` instead of re-dispatching.

    Issue #266: launch-failure sidecars (pid=None, error set) are terminal by
    construction and are reaped immediately, reported in the returned list.

    Issue #295: dead/launch-failed rework sessions with an open PR and a
    request_changes verdict (or a rework prompt on disk) are restored to
    ``rework_requested`` so ``dispatch_rework`` can re-select them.

    ``persist_inconclusive_probe_counter`` (issue #343 Finding 2): controls
    whether this lane persists Signal-1's inconclusive-probe deferral counter
    for a not-alive, pid-bearing worker. Defaults to True so this function
    remains fully self-sufficient when called on its own (as every existing
    unit test does, and as any future standalone caller would expect).
    ``loop()`` is the one caller that always runs the sibling stall lane
    (``_detect_and_handle_stalled_sessions``, the sole writer of this counter
    for an ALIVE-but-stalled worker, and -- unconditionally, regardless of
    liveness -- the first lane to see every worker each pass) immediately
    before this one, in the same pass; it passes False there so this lane
    does not ALSO increment the same counter on top of what the stall lane
    just wrote a moment earlier -- that double counting (0->1 in the stall
    lane, then re-read and ->2 here) halved the effective deferral grace
    period and was the very mechanism that opened Finding 1's pass-2
    phantom-sidecar window. classify_worker_health's own cap-check always
    reads whatever value is currently on the sidecar, regardless of which
    lane -- or how many passes ago -- last wrote it, so suppressing the
    write here never affects the DEAD-vs-deferred decision made below,
    only which lane's write ends up on disk for a given pass.
    """
    from .claude_code import update_worker_record_with_failure_classification
    from .devin_shell import update_session_record_with_failure_classification
    from .post_mortem import classify_and_record
    from .state import append_event, load_state, save_state, set_throttled_until, state_lock
    from .worker import (
        _next_inconclusive_probe_deferred_count,
        classify_worker_health,
        iter_workers,
        real_activity_probe_for,
        update_worker_log_stat,
    )
    from .worktree import WorktreeState

    now_for_health = datetime.now(UTC)

    # Fetch open PRs for the "no open PR" guard
    prs = gh.pr_list()
    open_prs_by_issue: dict[int, list[dict[str, Any]]] = {}
    for pr in prs:
        pr_state = str(pr.get("state") or "").upper()
        if pr_state != "OPEN":
            continue
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        if issue_number is not None:
            open_prs_by_issue.setdefault(issue_number, []).append(pr)

    repo_root = getattr(gh, "repo_root", None)

    reaped: list[dict[str, Any]] = []

    for w in iter_workers(sessions_dir):
        if w.pid is None and w.error is not None:
            # Launch-failure sidecar: terminal by construction (issue #266).
            # The process never launched, so it can never transition to live.
            failure_kind = "launch_failed"
            throttled_until = None
            if w.adapter_kind == "devin":
                failure_kind, throttled_until = update_session_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                )
            elif w.adapter_kind == "claude-code":
                failure_kind, throttled_until = update_worker_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                )
            elif w.adapter_kind == "api":
                failure_kind, throttled_until = update_worker_record_with_failure_classification(
                    sessions_dir,
                    w.issue_number,
                    fallback_kind=failure_kind,
                    config=config,
                    adapter_kind="api",
                )
            if failure_kind and throttled_until:
                # A throttle-caused launch failure must persist its window just
                # like the dead-session branch below — otherwise the governor
                # relaunches straight into the same throttled provider.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    save_state(state_file, state)

            if (
                failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                and w.issue_number not in open_prs_by_issue
            ):
                try:
                    issue = gh.issue_view(w.issue_number)
                except Exception:
                    issue = None
                issue_labels = label_names(issue) if issue else set()
                active_labels = issue_labels & config.labels.active
                with state_lock(state_file):
                    state = load_state(state_file)
                    entry = state["issues"].get(str(w.issue_number), {})
                    now = datetime.now(UTC)
                    redispatch_at = _windowed_redispatch_at(
                        entry, window_minutes=config.watchdog.redispatch_window_minutes
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    entry["status"] = "escalated"
                    entry["escalation_reason"] = failure_kind
                    entry["redispatch_at"] = redispatch_at
                    entry.pop("worker_pid", None)
                    entry.pop("worker_process_start_time", None)
                    state["issues"][str(w.issue_number)] = entry
                    save_state(state_file, state)
                    transition(gh, config.labels, w.issue_number, "redispatch_escalated")
                    state = append_event(
                        state,
                        "session_failed_escalated",
                        {
                            "issue_number": w.issue_number,
                            "failure_kind": failure_kind,
                            "removed_labels": sorted(active_labels),
                            "redispatch_count": len(redispatch_at),
                        },
                        state_path=state_file,
                    )
                    save_state(state_file, state)

            w.reap_sidecar(
                sessions_dir,
                api_config=config.api_worker,
                state_dir=state_file.parent,
            )
            reaped.append(
                {
                    "issue_number": w.issue_number,
                    "adapter_kind": w.adapter_kind,
                    "failure_kind": failure_kind,
                    "error": w.error,
                    "pid": w.pid,
                }
            )
            # Issue #295: a launch-failed rework session must still be returned to
            # rework_requested so its owning lane can re-dispatch it.
            _reap_restore_rework_requested(
                state_file, gh, config, open_prs_by_issue, w, failure_kind=failure_kind
            )
            continue
        if not w.is_alive():
            # Update log stat fields for progress tracking (final update before classification)
            update_worker_log_stat(sessions_dir, w)

            # Issue #343: this lane used to treat "not w.is_alive()" as sufficient
            # grounds to relabel the issue and reap the sidecar, bypassing the
            # real-activity corroboration + inconclusive-probe deferral cap that
            # classify_worker_health already enforces for the sibling stall/kill
            # lane (_detect_and_handle_stalled_sessions, issues #280/#301/#307/#338).
            # That let a fail-open reap remove the sidecar of a worker whose
            # liveness signal was merely ambiguous (or transiently wrong) while the
            # governor's dispatch cap (_count_live_sessions) counts live sidecars,
            # not live processes -- a wrongly-reaped sidecar silently frees a slot
            # for over-cap dispatch even though the underlying process may still be
            # running. Route through the same single enforcement point here so a
            # worker is only ever treated as DEAD -- and only ever loses its
            # sidecar -- once classify_worker_health agrees, with the same
            # escalation cap (max_inconclusive_probe_deferrals) guaranteeing a
            # genuinely-dead worker behind a permanently-broken probe still gets
            # reaped after N deferred passes (never an unconditional "never-reap").
            #
            # Issue #426: the launch-failure lane above handles ``pid is None``
            # sidecars. Sidecars that carry a real (dead) pid *and* a stale
            # ``error`` string (e.g. ``live_worker_redispatch_averted``) must not
            # be invisible to the confirmed-dead lane. Removing the ``w.error is
            # None`` gate lets classify_worker_health decide, with the same
            # max_inconclusive_probe_deferrals cap, instead of leaving them stuck
            # forever. The stall lane skips ``w.error is not None`` workers, so
            # the dead lane must persist the Signal-1 counter for those workers
            # even when loop() asks it not to double-write for ``w.error is None``
            # workers.
            if w.pid is not None:
                probe = real_activity_probe_for(w, config, now_for_health)
                health = classify_worker_health(w, config, now_for_health, probe)
                # Issue #343 Finding 2: persist_inconclusive_probe_counter (see
                # the docstring) lets loop() suppress this write when it just
                # ran the sibling stall lane a moment earlier in the same pass
                # -- that lane already persisted this exact counter for a
                # not-alive worker, and writing it again here double-increments
                # it. Every other caller (including every existing unit test)
                # leaves this at its default True, so this lane remains fully
                # self-sufficient when called on its own.
                #
                # Issue #426: the stall lane unconditionally skips
                # ``w.error is not None`` workers, so for those sidecars this
                # dead lane is the only writer of the counter. Persist it even
                # when loop() passes False.
                if persist_inconclusive_probe_counter or w.error is not None:
                    new_deferred_count = _next_inconclusive_probe_deferred_count(w, probe, health)
                    update_worker_log_stat(
                        sessions_dir, w, inconclusive_probe_deferred_count=new_deferred_count
                    )
                if health is not WorkerHealth.DEAD:
                    # Corroboration vetoed the DEAD verdict (fresh real-session
                    # activity) or the probe was inconclusive and the deferral
                    # cap has not yet been reached -- defer to next pass instead
                    # of reaping a sidecar we cannot yet prove is safe to remove.
                    continue

            # Inspect the worktree before deciding how to classify and relabel.
            # This is the single enforcement point for issue #252.
            worktree_path = Path(w.worktree_path)
            inspection = inspect_worktree_state(
                worktree_path,
                config.dispatch.base_ref,
                config.dispatch.injected_paths,
                config.dispatch.materialize_dirs,
            )
            is_completed = inspection.state == WorktreeState.COMPLETED

            # Post-mortem extraction (issue #261) is intertwined with log-tail
            # classification. For a completed-but-unpublished worktree, we want
            # failure_kind to be "unpublished_work" even if the terminal log
            # tail would otherwise look like a tool-rejection (worker_blocked).
            # Call update_* first when completed, then run classify_and_record
            # for diagnostics (it will no-op on the sidecar because failure_kind
            # is already set). For non-completed sessions, preserve the original
            # ordering so worker_blocked still escalates.
            if is_completed:
                # session_completed=True (issue #656): the worktree inspection
                # just above is ground truth that this session produced real,
                # committable work -- it cannot also have died to a provider
                # quota/rate-limit/auth failure, so log-tail marker matching
                # (which would otherwise treat the session's own completion
                # summary prose as fair game) is skipped entirely.
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
                fallback_kind = "stalled" if inspection.state != WorktreeState.UNKNOWN else None
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
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(
                        state,
                        throttled_until,
                        reason=failure_kind,
                        adapter_kind=w.adapter_kind,
                    )
                    save_state(state_file, state)

            # Reap the sidecar to prevent phantom sessions from PID recycling (issue #113)
            # Delete the sidecar file after the session is detected as dead and classified
            w.reap_sidecar(
                sessions_dir,
                api_config=config.api_worker,
                state_dir=state_file.parent,
            )
            reaped.append(
                {
                    "issue_number": w.issue_number,
                    "adapter_kind": w.adapter_kind,
                    "failure_kind": failure_kind,
                    "error": w.error,
                    "pid": w.pid,
                }
            )

            # Issue #118: reconcile labels for dead sessions with no open PR
            if w.issue_number not in open_prs_by_issue:
                try:
                    issue = gh.issue_view(w.issue_number)
                except Exception:
                    # Issue may have been deleted or we lack access; skip relabel
                    continue
                issue_labels = label_names(issue)
                active_labels = issue_labels & config.labels.active
                # Gate the WHOLE reclaim on an active label actually being
                # present, matching reconcile.py's issue_active_label_no_open_pr
                # pattern (~536-580) so all three sites agree. An issue with
                # no active label -- e.g. one carrying only a terminal label
                # like agent:human-needed/agent:done/agent:blocked -- has
                # nothing here to reclaim; it must never get `ready` added
                # back just because it also has a stale
                # dispatched/dead-worker/no-PR state.json entry. (A prior
                # revision gated on "not active_labels and not needs_ready" to
                # also repair a remove-succeeded-but-add-failed partial
                # failure once the active label was already gone -- but that
                # made this lane indistinguishable from "issue is legitimately
                # terminal-only", which is the regression this gate now
                # avoids. `needs_ready` is still honored below whenever an
                # active label IS present, so the common
                # remove-and-add-together case is unaffected.)
                if not active_labels:
                    continue
                needs_ready = config.labels.ready not in issue_labels

                # Issue #252: completed-but-unpublished work takes the salvage
                # path (push + PR) instead of re-dispatching.
                salvage_error: str | None = None
                if is_completed and repo_root is not None:
                    salvaged, salvage_error = _attempt_salvage(
                        gh=gh,
                        config=config,
                        repo_root=repo_root,
                        worktree_path=worktree_path,
                        branch=w.branch,
                        base_ref=inspection.resolved_base_ref or "",
                        issue_number=w.issue_number,
                        active_labels=active_labels,
                        issue_labels=issue_labels,
                        state_file=state_file,
                        failure_kind=failure_kind,
                    )
                    if salvaged:
                        continue
                    # Salvage failed: fall through to the normal relabel path below.

                # Track redispatch count for escalation cap (issue #165)
                # This relabel-to-ready path is a redispatch event
                with state_lock(state_file):
                    state = load_state(state_file)
                    entry = state["issues"].get(str(w.issue_number), {})
                    now = datetime.now(UTC)
                    redispatch_at = _windowed_redispatch_at(
                        entry, window_minutes=config.watchdog.redispatch_window_minutes
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    # issue #261: a worker_blocked verdict (extracted from the
                    # Devin CLI's session store — see post_mortem.classify_and_record)
                    # means the worker was killed by a push-gate hook, not a
                    # generic stall/crash. Hot-redispatching it just repeats the
                    # same block, so it bypasses the redispatch-count cap entirely
                    # and escalates on the very first occurrence.
                    terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                    if (
                        terminal_failure
                        or len(redispatch_at) > config.watchdog.max_auto_redispatch
                    ):
                        # Escalate to human review instead of relabeling to ready
                        entry["status"] = "escalated"
                        entry["redispatch_at"] = redispatch_at
                        entry["escalation_reason"] = (
                            failure_kind if terminal_failure else "redispatch_cap_exceeded"
                        )
                        # Issue #282: preserve the liveness fingerprint for the
                        # recovery path. The PID is already verified dead by the
                        # time we reach this branch, but clearing it removes the
                        # only signal the recovery probe can cross-check.
                        state["issues"][str(w.issue_number)] = entry
                        save_state(state_file, state)
                        transition(gh, config.labels, w.issue_number, "redispatch_escalated")
                        state = append_event(
                            state,
                            "session_failed_escalated",
                            {
                                "issue_number": w.issue_number,
                                "failure_kind": failure_kind,
                                "removed_labels": sorted(active_labels),
                                "redispatch_count": len(redispatch_at),
                            },
                            state_path=state_file,
                        )
                        save_state(state_file, state)
                        continue
                    else:
                        entry["redispatch_at"] = redispatch_at
                        state["issues"][str(w.issue_number)] = entry
                        save_state(state_file, state)
                # Remove all active labels and ensure ready label is present.
                # Issue #417: check (and record) the bool return values instead
                # of silently discarding them. A False here means this pass's
                # label swap did not fully land -- the issue remains eligible
                # for _detect_and_handle_orphaned_workers' no-open-PR sweep to
                # finish the reclaim on a later pass, since that lane
                # re-derives "does this still need fixing" from GitHub's live
                # label state every pass rather than from any flag written
                # here (and never touches redispatch_at, so a retry there
                # cannot double-count this as a second redispatch event).
                label_write_ok = True
                for label in sorted(active_labels):
                    if not gh.remove_issue_label(w.issue_number, label):
                        label_write_ok = False
                if needs_ready:
                    if not gh.add_issue_label(w.issue_number, config.labels.ready):
                        label_write_ok = False
                # Record the relabel event
                with state_lock(state_file):
                    state = load_state(state_file)
                    # Issue #282: preserve the liveness fingerprint so the
                    # recovery path can verify the worker is dead before removing
                    # the worktree, even after the session is classified as dead.
                    state = append_event(
                        state,
                        "session_failed_relabeled",
                        {
                            "issue_number": w.issue_number,
                            "failure_kind": failure_kind,
                            "removed_labels": sorted(active_labels),
                            "added_ready": needs_ready,
                            "label_write_ok": label_write_ok,
                            "salvage_failed": is_completed,
                            "salvage_error": salvage_error,
                        },
                        state_path=state_file,
                    )
                    save_state(state_file, state)
            else:
                # Issue #295: open PR with request_changes or rework prompt =>
                # restore to rework_requested for its owning lane.
                #
                # Issue #315 finding 1: a completed worktree (ahead of base and
                # clean) proves the worker finished its work — even if this
                # reap pass's PR-list snapshot (fetched once, at line ~889)
                # hasn't caught up to a fresh push yet. Never roll a worker
                # that finished (and, per the open-PR guard above, already has
                # a PR reflecting or about to reflect that work) back to
                # rework_requested just because a launch-failure classifier
                # ran on its now-dead sidecar.
                if not is_completed:
                    pr_data = _rework_pr_for_worker(open_prs_by_issue, w)
                    if pr_data is not None:
                        pr_number = int(pr_data["number"])
                        try:
                            pr_view = gh.pr_view(pr_number)
                        except Exception:
                            pr_view = None
                        enriched = pr_view if pr_view else pr_data
                        is_candidate, reason = _is_pre_review_rework_candidate(
                            enriched, config, now_for_health
                        )
                        if is_candidate:
                            _route_dead_worker_to_pre_review_rework(
                                state_file,
                                gh,
                                config,
                                enriched,
                                w.issue_number,
                                reason,
                                failure_kind=failure_kind,
                            )
                        else:
                            _reap_restore_rework_requested(
                                state_file,
                                gh,
                                config,
                                open_prs_by_issue,
                                w,
                                failure_kind=failure_kind,
                            )
                    else:
                        _reap_restore_rework_requested(
                            state_file,
                            gh,
                            config,
                            open_prs_by_issue,
                            w,
                            failure_kind=failure_kind,
                        )

    return reaped


def _attempt_salvage(
    *,
    gh: GitHub,
    config: OrchestratorConfig,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    failure_kind: str | None,
) -> tuple[bool, str | None]:
    """Push a completed branch and open a PR, then move labels to ``pr_open``.

    Returns ``(ok, error)``. Errors are recorded as values and never raised.
    """
    push_ok, push_error = push_branch(repo_root, branch, worktree_path=worktree_path)
    if not push_ok:
        return False, push_error

    base_branch = resolve_base_branch_name(repo_root, base_ref)
    title = f"Salvaged work for issue #{issue_number}"
    body = f"Closes #{issue_number}\n\nSalvaged by the orchestrator from a completed-but-unpublished worker worktree."
    pr_number = gh.pr_create(head=branch, base=base_branch, title=title, body=body)
    if pr_number is None:
        pr_error = "gh pr create failed or returned no PR number"
        return False, pr_error

    for label in sorted(active_labels):
        gh.remove_issue_label(issue_number, label)
    if config.labels.pr_open not in issue_labels:
        gh.add_issue_label(issue_number, config.labels.pr_open)

    with state_lock(state_file):
        state = load_state(state_file)
        # Issue #282: preserve the liveness fingerprint so the recovery path
        # can verify the worker is dead before the worktree is reclaimed.
        state = append_event(
            state,
            "session_salvaged",
            {
                "issue_number": issue_number,
                "failure_kind": failure_kind,
                "removed_labels": sorted(active_labels),
                "pr_number": pr_number,
            },
            state_path=state_file,
        )
        save_state(state_file, state)
    return True, None


def _issues_with_live_workers(sessions_dir: Path) -> set[int]:
    """Return the set of issue numbers that have currently alive worker sessions.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the set of issue numbers with alive PIDs.
    """
    from .worker import iter_workers

    return {w.issue_number for w in iter_workers(sessions_dir) if w.is_alive()}


def _build_attention_digest(
    state_file: Path,
    health_transitions: dict[int, dict[str, Any]],
    repo: str,
    state_field: str = "health",
) -> AttentionDigest | None:
    """Build an AttentionDigest from health transitions observed in a pass.

    Args:
        state_file: Path to state.json for reading/writing per-issue health baseline
        health_transitions: Dict mapping issue_number to transition data:
            {
                issue_number: {
                    "adapter_kind": str,
                    "health": str,  # current health (e.g., "STALLED", "RUNAWAY", "DEAD")
                    "last_log_line": str | None,
                    "pid": int | None,
                    "terminal_tool": str | None,  # issue #261: post-mortem terminal tool (DEAD only)
                    "terminal_reason": str | None,  # issue #261: one-line terminal cause
                }
            }
        repo: Repository name for the digest
        state_field: The state["issues"][n] field to read/write for transition
            comparison. Defaults to "health"; callers tracking a separate alert
            dimension (e.g. merge_alert) can pass their own field name.

    Returns:
        AttentionDigest if there are transitions, None otherwise. Updates per-issue
        health field in state.json to the current health for transition comparison
        on the next pass.
    """
    if not health_transitions:
        return None

    from .state import load_state, save_state, state_lock
    from .state import utc_now

    entries: list[AttentionEntry] = []

    with state_lock(state_file):
        state = load_state(state_file)

        for issue_number, transition in health_transitions.items():
            current_health = transition["health"]

            # Read the last persisted health for this issue
            issue_key = str(issue_number)
            issue_entry = state.get("issues", {}).get(issue_key, {})
            last_health = issue_entry.get(state_field)

            # Only include if health changed (or no previous health persisted)
            if last_health != current_health:
                entries.append(
                    AttentionEntry(
                        issue_number=issue_number,
                        adapter_kind=transition["adapter_kind"],
                        health=current_health,
                        previous_health=last_health,
                        last_log_line=transition.get("last_log_line"),
                        pid=transition.get("pid"),
                        terminal_tool=transition.get("terminal_tool"),
                        terminal_reason=transition.get("terminal_reason"),
                    )
                )

                # Update the persisted health for this issue
                state["issues"][issue_key] = {
                    **issue_entry,
                    state_field: current_health,
                }

        # Save the updated health baselines
        if entries:
            save_state(state_file, state)

    if not entries:
        return None

    return AttentionDigest(
        generated_at=utc_now(),
        repo=repo,
        transitions=tuple(entries),
    )


def _is_pending_only(summary: CheckSummary) -> bool:
    """Return True if the only reason the PR cannot merge is in-flight checks.

    A summary whose only defect is pending checks is not a structural merge
    failure; it should not arm the failed-attempt alarm.
    """
    return (
        bool(summary.pending)
        and not summary.failed
        and not summary.missing
        and not summary.infra_failed
        and not summary.unavailable
    )


def _format_merge_attempt_alarm_message(
    pr_number: int,
    attempts: int,
    summary: CheckSummary,
    mergeable: str | None = None,
    merge_state_status: str | None = None,
) -> str:
    """Human-readable alarm message for an approved PR that cannot merge.

    The message is surfaced in pass warnings, the merge_failed_attempt_alarm
    state event, and the notify digest terminal_reason.
    """
    buckets: list[str] = []
    if summary.missing:
        # Issue #253 signature: required checks missing while the PR is still open
        buckets.append("required checks missing while GitHub shows the PR open")
    if summary.pending:
        buckets.append(f"pending: {', '.join(summary.pending)}")
    if summary.failed:
        buckets.append(f"failed: {', '.join(summary.failed)}")
    if summary.infra_failed:
        buckets.append(f"infra_failed: {', '.join(summary.infra_failed)}")
    if summary.unavailable:
        # gh reported no parseable check list at all (see summarize_checks'
        # `checks is None` branch) — distinct from "all required checks
        # passed", so it must not fall into the passed-but-unmergeable
        # bucket below.
        buckets.append(f"unavailable: {', '.join(summary.unavailable)}")
    if not buckets:
        # Issue #751: every check-summary bucket above is empty, which means
        # the checks the bot tracks are not why this PR is stuck — GitHub's
        # own mergeability signal is (a lagging/absent CONFLICTING reading, a
        # BLOCKED merge state, branch protection, or a merge-base freshness
        # result that isn't one of the explicitly modelled branches above).
        # Surface what `pr_view` already fetched instead of discarding it;
        # only fall back to the generic "unknown" text when GitHub hasn't
        # reported anything usable either, so that case stays distinguishable.
        norm_mergeable = str(mergeable or "").upper()
        norm_merge_state = str(merge_state_status or "").upper()
        known_mergeable = bool(norm_mergeable) and norm_mergeable != "UNKNOWN"
        known_merge_state = bool(norm_merge_state) and norm_merge_state != "UNKNOWN"
        if known_mergeable or known_merge_state:
            buckets.append(
                f"mergeable={norm_mergeable or 'UNKNOWN'}, "
                f"mergeStateStatus={norm_merge_state or 'UNKNOWN'}, "
                "all required checks passed"
            )
        else:
            buckets.append("check summary unknown")
    checks_str = "; ".join(buckets)
    pass_str = "pass" if attempts == 1 else "passes"
    return f"PR #{pr_number} approved but unmergeable for {attempts} {pass_str}: {checks_str}"


def _format_stale_base_alarm_message(pr_number: int, attempts: int, reason: str) -> str:
    """Human-readable alarm message for an approved PR whose base is not current."""
    pass_str = "pass" if attempts == 1 else "passes"
    if reason == "base_stale":
        detail = "base is stale"
    elif reason == "compare_unavailable":
        detail = "base freshness comparison unavailable"
    else:
        detail = f"base is not current (reason: {reason})"
    return f"PR #{pr_number} approved but {detail} for {attempts} consecutive {pass_str}"


# Sentinel used to distinguish "no base-current signal was supplied" from
# an explicit ``None`` (compare API unavailable) in _should_update_pr_branch.
class _BaseCurrentUnset:
    __slots__ = ()


_BASE_CURRENT_UNSET = _BaseCurrentUnset()


@dataclass(frozen=True)
class _MergedPRListOutcome:
    items: list[dict[str, Any]] = field(default_factory=list)
    error: GitHubError | None = None
    called: bool = False


class OrchestratorApp:
    def __init__(
        self,
        repo_root: Path,
        paths: RuntimePaths,
        config: OrchestratorConfig,
        gh: GitHub,
        *,
        dry_run: bool = False,
        fleet_dir_override: str | None = None,
    ):
        self.repo_root = repo_root
        self.paths = paths
        self.config = config
        # Single resolved view of every sentinel-style state-child config
        # value (devin.sessions_dir/session_manifest/session_results,
        # review_dispatch.reviews_dir, notify.file_path, claude_code.worktrees_dir)
        # -- see paths.resolved_layout. Safe to cache once: self.config is
        # assigned only here, never reassigned on a live instance.
        self._layout = resolved_layout(config, repo_root)
        self.gh = gh
        self.dry_run = dry_run
        self.fleet_dir_override = fleet_dir_override
        # Make the event ring cap config-driven (issue #525).
        _state.EVENT_RING_SIZE = config.runtime.event_ring_size
        prompts_dir = config.runtime.prompts_dir
        if prompts_dir:
            override = Path(prompts_dir)
            if not override.is_absolute():
                override = repo_root / override
            self.prompt_dirs: tuple[Path, ...] = (override,)
        else:
            self.prompt_dirs = ()
        self.paths.ensure()

        # Startup self-check: validate gh --json field lists against the
        # installed CLI. Fail fast before any dispatch/review/merge work.
        if isinstance(self.gh, GitHub):
            self.gh.validate_field_lists()

    @property
    def layout(self) -> ResolvedLayout:
        """Public, read-only view of the resolved state-child layout.

        This is the contract module-level helpers (e.g. ``supervise.run_supervised``)
        that take an ``app`` argument should use, mirroring the existing public
        ``paths`` attribute -- callers outside this class should never reach into
        the private ``self._layout`` cache directly.
        """
        return self._layout

    def _render(self, template_name: str, values: dict[str, Any]) -> str:
        return render_prompt(template_name, values, search_dirs=self.prompt_dirs)

    def _record_event(
        self, state: dict[str, Any], kind: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Append an event to state.json and the unlimited events.jsonl log.

        This is the single instrumentation entry point for OrchestratorApp
        methods. It wraps ``append_event`` with ``self.paths.state_file`` and
        the repo name so every event is dual-written: once to the 200-entry
        convenience cache in ``state.json`` and once to the append-only
        ``events.jsonl`` audit log.
        """
        return append_event(
            state,
            kind,
            payload,
            state_path=self.paths.state_file,
            repo=self.repo_root.name,
        )

    def _resolve(self, value: str) -> Path:
        # pathlib keeps an absolute right-hand side as-is, so this handles
        # both repo-relative and absolute config paths.
        return self.repo_root / value

    def _adapter_settings(self, *, adapter: str | None = None) -> AdapterSettings:
        claude = self.config.claude_code
        devin = self.config.devin
        api_worker = self.config.api_worker
        resolved_adapter = adapter if adapter is not None else devin.adapter
        # Use adapter-specific venv_source and worker_env
        if resolved_adapter == "devin-shell":
            venv_source = self._resolve(devin.venv_source) if devin.venv_source else None
            worker_env = devin.worker_env
        elif resolved_adapter == "claude-code":
            venv_source = self._resolve(claude.venv_source) if claude.venv_source else None
            worker_env = claude.worker_env
        elif resolved_adapter == "api":
            # api workers are Claude Code CLI processes with provider env
            # injected, so they reuse the claude-code venv/env resolution
            # (shared venv junction, worker_env overrides). The provider
            # routing vars (ANTHROPIC_BASE_URL/AUTH_TOKEN/MODEL) are merged
            # inside launch_api_worker, over any worker_env values, so an
            # operator's worker_env cannot accidentally override the provider.
            venv_source = self._resolve(claude.venv_source) if claude.venv_source else None
            worker_env = claude.worker_env
        else:
            venv_source = None
            worker_env = {}
        return AdapterSettings(
            adapter=resolved_adapter,
            dispatch_command=devin.dispatch_command,
            command_timeout_seconds=devin.command_timeout_seconds,
            sessions_dir=self._layout.sessions_dir,
            shell_command=devin.shell_command,
            claude_command=claude.command,
            worktrees_dir=self._layout.worktrees,
            venv_source=venv_source,
            worker_env=worker_env,
            worker_model=devin.worker_model,
            materialize_dirs=self.config.dispatch.materialize_dirs,
            dry_run=self.dry_run,
            base_ref=self.config.dispatch.base_ref,
            tee_stream_json=claude.tee_stream_json,
            launch_stagger_seconds=self.config.dispatch.launch_stagger_seconds,
            api_worker_config=api_worker if resolved_adapter == "api" else None,
            config=self.config,
        )

    def _routing_inputs(self) -> tuple[Any, bool, bool, list[int]]:
        """Compute pass-level routing inputs shared across all issues (issue #482).

        Returns ``(budget, api_key_present, provider_in_cooldown,
        live_api_sessions)`` — the inputs to ``routing.select_adapter`` for one
        dispatch pass. Per-issue inputs (``rework``, ``issue_labels``) are
        supplied by the caller.

        * ``budget``: ``routing.BudgetStatus`` from ``api_budget.budget_status``
          over the loaded ledger.
        * ``api_key_present``: whether the active provider's ``api_key_env``
          names a non-empty environment variable.
        * ``provider_in_cooldown``: from the existing ``throttled_until`` state
          mechanism (``state.is_throttled``). The dispatch-level throttle gate
          already defers when this is True, so this is defense-in-depth for the
          future api-specific cooldown (issue ⑦ in the design decomposition).
        * ``live_api_sessions``: a **mutable** one-element list ``[count]``
          holding the number of in-flight api workers (alive workers with
          ``adapter_kind == "api"`` counted via ``worker.iter_workers``). The
          list is mutated in place by ``_select_adapter_for_issue``: each issue
          routed to the api adapter increments ``live_api_sessions[0]`` so that
          subsequent issues in the same pass see the updated in-flight count and
          fall back when ``max_concurrent_sessions`` is reached. Without this
          running increment, every api-eligible issue in a batch would see the
          same stale count and all bypass the concurrency cap simultaneously.
        """
        api_config = self.config.api_worker
        # Budget status from the on-disk spend ledger (atomic load with
        # corrupt-file recovery; missing file = empty ledger).
        ledger = _api_load_ledger(_api_ledger_path(self.paths.state_file.parent))
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        budget = _api_budget_status(ledger, api_config.budget, today)
        # API key present from the environment (name from the active provider).
        provider = api_config.providers.get(api_config.provider)
        api_key_present = bool(provider and os.environ.get(provider.api_key_env))
        # Provider in cooldown from the existing throttle-cooldown state.
        state = load_state_locked(self.paths.state_file)
        provider_in_cooldown = is_throttled(state)
        # Live api sessions counted via iter_workers filtered to adapter_kind == "api".
        # Wrapped in a mutable list so _select_adapter_for_issue can increment it
        # as issues are routed to api within the same pass (concurrency cap fix).
        sessions_dir = self._layout.sessions_dir
        live_api_sessions = sum(
            1 for w in iter_workers(sessions_dir) if w.adapter_kind == "api" and w.is_alive()
        )
        return budget, api_key_present, provider_in_cooldown, [live_api_sessions]

    def _select_adapter_for_issue(
        self,
        *,
        rework: bool,
        issue_labels: set[str],
        routing_inputs: tuple[Any, bool, bool, list[int]],
    ) -> AdapterChoice:
        """Call ``routing.select_adapter`` for one issue (issue #482).

        The single point of adapter routing enforcement: no inline
        adapter-choice conditionals live in workflow.py. Per-issue inputs
        (``rework``, ``issue_labels``) are passed in; pass-level inputs come
        from ``_routing_inputs``.

        When the choice resolves to the api adapter, the mutable
        ``live_api_sessions`` counter inside ``routing_inputs`` is incremented
        in place so the next issue in the same pass sees the updated in-flight
        count. This prevents a whole batch of api-eligible issues from all
        passing the concurrency preflight against a stale count and exceeding
        ``max_concurrent_sessions``.
        """
        budget, api_key_present, provider_in_cooldown, live_api_sessions = routing_inputs
        choice = select_adapter(
            rework=rework,
            issue_labels=issue_labels,
            complexity_high_label=self.config.labels.complexity_high,
            api_config=self.config.api_worker,
            budget=budget,
            api_key_present=api_key_present,
            provider_in_cooldown=provider_in_cooldown,
            live_api_sessions=live_api_sessions[0],
            default_adapter=self.config.devin.adapter,
        )
        if choice.kind == "api":
            live_api_sessions[0] += 1
        return choice

    def _dispatch_partitioned(
        self,
        session_requests: list[SessionRequest],
        adapter_choices: dict[int, AdapterChoice],
    ) -> list[SessionDispatchResult]:
        """Partition requests by ``AdapterChoice.kind`` and dispatch per group (issue #482).

        When ``adapter_choices`` is empty (api disabled), dispatches all
        requests as a single group with the default ``AdapterSettings`` —
        byte-identical to the pre-#482 behavior. When non-empty, partitions
        by kind and invokes ``dispatch_sessions`` once per non-empty group
        with that group's ``AdapterSettings`` (built via ``_adapter_settings``
        parameterized by kind). ``launch_stagger_seconds`` applies within each
        group as today.

        After all groups are dispatched, a combined session manifest and
        results file are written to the standard paths so the on-disk
        artifacts reflect every session in the pass (per-group calls each
        overwrite these files; the combined write at the end is authoritative).
        """
        manifest_path = self._layout.session_manifest
        results_path = self._layout.session_results

        if not adapter_choices:
            # No routing (api disabled) — single group, today's behavior.
            return dispatch_sessions(
                self.repo_root,
                manifest_path,
                results_path,
                self._adapter_settings(),
                session_requests,
            )

        # Partition by adapter kind, preserving the original request order
        # within each group (stable partition).
        groups: dict[str, list[SessionRequest]] = {}
        for req in session_requests:
            kind = adapter_choices[req.issue_number].kind
            groups.setdefault(kind, []).append(req)

        all_results: list[SessionDispatchResult] = []
        for kind, group_requests in groups.items():
            group_results = dispatch_sessions(
                self.repo_root,
                manifest_path,
                results_path,
                self._adapter_settings(adapter=kind),
                group_requests,
            )
            all_results.extend(group_results)

        # Write combined manifest and results reflecting all sessions in the
        # pass. Per-group dispatch_sessions calls each overwrite these files;
        # this final write is the authoritative on-disk artifact.
        write_session_manifest(manifest_path, session_requests, adapter="mixed")
        write_session_results(results_path, all_results)
        return all_results

    def _rescue_adapter_settings(self) -> AdapterSettings:
        """AdapterSettings for a rescue-tier rework dispatch (issue #555).

        Mirrors the "claude-code" branch of ``_adapter_settings()`` exactly
        (same venv/worker_env/command resolution), but forces
        ``adapter="claude-code"`` regardless of the primary configured
        ``devin.adapter`` — the rescue tier always uses the claude-code
        adapter — and overrides ``claude_code.model`` to
        ``rescue.worker_model`` via a one-off config copy. This is the
        "adapter/model overridden from RescueConfig" the rescue rework
        reuses the existing rework-dispatch path with, never a parallel
        launch path.
        """
        claude = self.config.claude_code
        rescue_config = dataclasses_replace(
            self.config,
            claude_code=dataclasses_replace(claude, model=self.config.rescue.worker_model),
        )
        return AdapterSettings(
            adapter="claude-code",
            sessions_dir=self._layout.sessions_dir,
            claude_command=claude.command,
            worktrees_dir=self._layout.worktrees,
            venv_source=self._resolve(claude.venv_source) if claude.venv_source else None,
            worker_env=claude.worker_env,
            materialize_dirs=self.config.dispatch.materialize_dirs,
            dry_run=self.dry_run,
            base_ref=self.config.dispatch.base_ref,
            tee_stream_json=claude.tee_stream_json,
            launch_stagger_seconds=self.config.dispatch.launch_stagger_seconds,
            config=rescue_config,
        )

    def _apply_concurrency_governor(
        self, dispatch_limit: int, *, live_count: int | None = None
    ) -> ConcurrencyGovernorResult:
        """Apply global concurrency governor cap to a dispatch limit.

        Returns a ConcurrencyGovernorResult with the potentially-clamped limit
        and all related fields. This eliminates Pyright's reportPossiblyUnbound
        warnings by ensuring live_count is always bound together with the
        clamped flag.

        Args:
            dispatch_limit: The requested dispatch limit
            live_count: Optional pre-computed live worker count. If None and
                max_concurrent > 0, this will compute it via _count_live_sessions.
        """
        max_concurrent = self.config.dispatch.max_concurrent_sessions
        fleet_max = self.config.fleet.global_max_concurrent_sessions
        available_slots = dispatch_limit
        clamped = False
        fleet_live_count = 0

        if max_concurrent > 0:
            if live_count is None:
                sessions_dir = self._layout.sessions_dir
                live_count = _count_live_sessions(sessions_dir, self.paths.state_file)
            available_slots = max(0, max_concurrent - live_count)
            if available_slots < dispatch_limit:
                dispatch_limit = available_slots
                clamped = True

        if fleet_max > 0:
            fleet_live_count, _skipped_repos = count_fleet_live_sessions(self.fleet_dir_override)
            fleet_available = max(0, fleet_max - fleet_live_count)
            if fleet_available < dispatch_limit:
                dispatch_limit = fleet_available
                clamped = True

        return ConcurrencyGovernorResult(
            clamped=clamped,
            max_concurrent=max_concurrent,
            live_count=live_count or 0,
            available_slots=available_slots,
            dispatch_limit=dispatch_limit,
            fleet_live_count=fleet_live_count,
            fleet_max=fleet_max,
        )

    @_guard_state_lock
    def status(self) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        prs = self.gh.pr_list()
        state = load_state_locked(self.paths.state_file)
        operator_claimed = operator_claimed_issues(state)
        stale_claims = stale_operator_claims(state)
        active_issues = [
            issue for issue in issues if label_names(issue) & self.config.labels.active
        ]
        available_issues = [
            issue for issue in issues if self._is_dispatchable(issue, operator_claimed)
        ]

        # Check for blocked issues (dependency gate)
        truly_available, blocked_issues, _open_blockers_by_issue = self._filter_blocked_issues(
            available_issues
        )

        # Check for stalled sessions (read-only for status/roll-call)
        sessions_dir = self._layout.sessions_dir
        stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)

        # Build workers list with health classification
        from .worker import classify_worker_health, iter_workers, real_activity_probe_for

        worker_views = list(iter_workers(sessions_dir))
        now = datetime.now(UTC)
        workers = []
        for view in worker_views:
            if not view.is_alive():
                continue
            probe = real_activity_probe_for(view, self.config, now)
            workers.append(
                self._summarize_worker(view, classify_worker_health(view, self.config, now, probe))
            )

        # Observe runner pool if feature is enabled
        runners_data = None
        if self.config.runner_scaling.enabled:
            from .runners import format_runner_pool_state, observe_runner_pool

            try:
                # No state_dir here, so nothing is written either way today --
                # threaded anyway so the call site stays honest if one is ever
                # added, and so the dry-run call-site guard stays a bright line.
                pool_state = observe_runner_pool(
                    self.gh, self.config.runner_scaling, dry_run=self.gh.dry_run
                )
                runners_data = format_runner_pool_state(pool_state)
            except Exception:
                # Don't fail status() if runner observation fails
                runners_data = None

        linked_prs = [
            self._summarize_pr(pr)
            for pr in prs
            if linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            is not None
        ]
        data = {
            "ready_issue_count": len(issues),
            "available_issue_count": len(truly_available),
            "active_issue_count": len(active_issues),
            "open_linked_pr_count": len(linked_prs),
            "state_file": str(self.paths.state_file),
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "issues": [self._summarize_issue(issue) for issue in issues],
            "prs": linked_prs,
            "last_generated_at": state.get("generated_at"),
            "blocked": [
                {"issue": issue_number, "blockers": blockers}
                for issue_number, blockers in sorted(blocked_issues.items())
            ],
            "stalled": stalled_entries,
            "workers": workers,
            "operator_claimed": sorted(operator_claimed),
            "stale_claims": sorted(stale_claims),
        }

        # Add runners section if feature is enabled and observation succeeded
        if runners_data is not None:
            data["runners"] = runners_data

        return CommandResult(True, "status complete", data)

    def claim(self, issue_number: int, release: bool = False) -> CommandResult:
        """Record or release an operator claim on an issue.

        A claimed issue is excluded from fresh dispatch and rework dispatch
        regardless of its labels. When a worktree for the issue already
        exists, a ``.charlie-writer.json`` marker is written or removed so the
        protection is mutual: the orchestrator refuses to dispatch into a
        worktree with a live foreign writer marker.
        """
        try:
            issue = self.gh.issue_view(issue_number)
        except GitHubError as exc:
            return CommandResult(
                False, f"issue #{issue_number} not found: {exc}", {"issue_number": issue_number}
            )

        branch_name = self._branch_name(issue)
        worktree_path = worktree_path_for_branch(
            self.repo_root, branch_name, self._layout.worktrees
        )

        marker_written = False
        if not release and worktree_path.is_dir():
            # Operator markers intentionally do not encode the CLI's transient
            # PID; liveness is keyed off operator_claimed_at in state.json.
            try:
                write_worktree_marker(
                    worktree_path,
                    0,
                    OPERATOR_MARKER_SESSION_ID,
                    kind=OPERATOR_MARKER_KIND,
                )
                marker_written = True
            except OSError:
                # Marker write failure is not fatal, but record it in the result.
                pass
        elif release:
            # Best-effort marker removal; state release is what matters. Only
            # remove a marker that belongs to this operator claim, never a
            # worker marker from an active session.
            try:
                remove_worktree_marker(worktree_path, session_id=OPERATOR_MARKER_SESSION_ID)
            except OSError:
                pass

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            if release:
                state = release_operator_claimed(state, issue_number)
            else:
                state = set_operator_claimed(state, issue_number)
            state = self._record_event(
                state,
                "operator_claim_released" if release else "operator_claim",
                {
                    "issue_number": issue_number,
                    "branch_name": branch_name,
                    "worktree_path": str(worktree_path),
                    "marker_written": marker_written,
                },
            )
            save_state(self.paths.state_file, state)

        message = (
            f"operator claim released for issue #{issue_number}"
            if release
            else f"operator claim recorded for issue #{issue_number}"
        )
        return CommandResult(
            True,
            message,
            {
                "issue_number": issue_number,
                "branch_name": branch_name,
                "worktree_path": str(worktree_path),
                "released": release,
                "marker_written": marker_written,
            },
        )

    def bootstrap_labels(self) -> CommandResult:
        descriptions = {
            self.config.labels.ready: "Issue is ready for deterministic agentic automation.",
            self.config.labels.queued: "Issue is queued by the orchestrator.",
            self.config.labels.in_progress: "A worker is implementing this issue.",
            self.config.labels.pr_open: "A worker PR exists for this issue.",
            self.config.labels.reviewing: "The orchestrator is adversarially reviewing the worker PR.",
            self.config.labels.needs_rework: "The worker PR needs another implementation cycle.",
            self.config.labels.blocked: "Automation is blocked and needs intervention.",
            self.config.labels.done: "Automation completed and the issue was merged or resolved.",
            self.config.labels.human_needed: "A human product or security decision is needed.",
            self.config.labels.prose_only_deps: "Issue has prose-only dependencies that need structured blocker declarations.",
            self.config.labels.merge_hold: "Approved PR is held out of the merge queue by operator request.",
            self.config.labels.complexity_high: (
                "Routing hint: route to the api worker (multi-module, "
                "cross-cutting invariant, or prior escalation)."
            ),
        }
        for label in self.config.labels.all:
            # The ready marker is green; the complexity routing hint gets a
            # distinct amber so it is visually separable from workflow state.
            if label == self.config.labels.ready:
                color = "0E8A16"
            elif label == self.config.labels.complexity_high:
                color = "BFD4F2"
            else:
                color = "5319E7"
            self.gh.label_create(label, color, descriptions[label])
        # Verify: check which labels actually exist after creation attempts.
        # label_create uses allow_failure=True, so silent failures are possible
        # (e.g. no auth, wrong repo). Don't report success we can't vouch for.
        try:
            live = {
                str(item.get("name") or "")
                for item in self.gh.label_list()
                if isinstance(item, dict)
            }
            missing = [name for name in self.config.labels.all if name not in live]
        except GitHubError as exc:
            return CommandResult(
                False,
                f"labels created but verification failed: {exc}",
                {"labels": self.config.labels.all, "missing": None},
            )
        if missing:
            return CommandResult(
                False,
                f"bootstrap incomplete — {len(missing)} label(s) still missing: {missing}",
                {"labels": self.config.labels.all, "missing": missing},
            )
        return CommandResult(
            True, "labels ensured", {"labels": self.config.labels.all, "missing": []}
        )

    @_guard_state_lock
    def intake(self) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        written: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        prose_only_deps_issues: list[int] = []
        # Gather all network results and write files outside the lock
        for issue in issues:
            issue_number = int(issue["number"])
            try:
                full_issue = self.gh.issue_view(issue_number)
            except GitHubError as exc:
                failed.append({"issue": issue_number, "error": str(exc)})
                continue
            issue_dir = self.paths.issues / f"issue-{issue_number}"
            issue_dir.mkdir(parents=True, exist_ok=True)
            issue_json = issue_dir / "issue.json"
            self._write_json(issue_json, full_issue)
            prompt_path = self._write_worker_prompt(full_issue)

            # Check for prose-only dependencies (issue #225)
            body_text = full_issue.get("body", "")
            has_prose_deps = detect_prose_only_dependencies(body_text)
            has_structured_blockers = bool(parse_blockers(body_text))

            # If prose-only dependencies exist without structured blockers, label for human attention
            if has_prose_deps and not has_structured_blockers:
                prose_only_deps_issues.append(issue_number)
                try:
                    self.gh.add_issue_label(issue_number, self.config.labels.prose_only_deps)
                except Exception:
                    # Label add failure is non-blocking for intake
                    pass

            written.append(
                {
                    "issue": issue_number,
                    "prompt_path": str(prompt_path),
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "labels": sorted(label_names(full_issue)),
                    "updated_at": full_issue.get("updatedAt"),
                }
            )
        # Single lock for all state updates
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            for entry in written:
                issue_number = entry["issue"]
                # Merge-update, never replace: intake used to clobber dispatch
                # status recorded by earlier passes (production-confirmed).
                state["issues"][str(issue_number)] = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "title": entry["title"],
                    "url": entry["url"],
                    "labels": entry["labels"],
                    "prompt_path": entry["prompt_path"],
                    "updated_at": entry["updated_at"],
                }
            for failure in failed:
                state = self._record_event(
                    state,
                    "intake_failed",
                    {"issue_number": failure["issue"], "error": failure["error"]},
                )
            if prose_only_deps_issues:
                state = self._record_event(
                    state,
                    "intake_prose_only_deps",
                    {"issue_numbers": sorted(prose_only_deps_issues)},
                )
            state = self._record_event(
                state, "intake", {"issue_count": len(issues), "failed_count": len(failed)}
            )
            save_state(self.paths.state_file, state)
        message = "intake complete"
        if failed:
            message = f"intake completed with {len(failed)} failure(s)"
        if prose_only_deps_issues:
            message += (
                f", {len(prose_only_deps_issues)} issue(s) labeled with prose-only dependencies"
            )
        return CommandResult(
            not failed,
            message,
            {
                "issues": written,
                "failed": failed,
                "prose_only_deps_issues": prose_only_deps_issues,
            },
        )

    def _finalize_externally_merged_issues(
        self,
        ready_issues: list[dict[str, Any]] | None = None,
    ) -> tuple[set[int], list[dict[str, Any]], _MergedPRListOutcome]:
        """Finalize closed ready-labeled issues whose linked PR merged externally,
        and strip the ready/active labels from closed ready issues that have no
        merged PR binding them (issue #429/#433).

        Runs before dispatch capacity guards (fleet lock, GraphQL budget, provider
        throttle) so a pass that defers new work still drains the backlog of
        externally-merged issues (e.g. Aviator MergeQueue handoffs).  It first
        binds candidates against the cheap most-recent-500 ``merged_pr_list()``;
        only issues whose merged PR falls outside that window incur a per-issue
        ``gh pr list --search`` lookup.

        Per-issue lookups are capped at ``dispatch.finalize_limit`` and processed
        oldest-first (by ``createdAt``, then issue number). A consecutive-failure
        circuit breaker stops the pass after 3 failed lookups so a transient
        Search API rate limit does not monopolize the shared token.
        """
        if ready_issues is None:
            ready_issues = self.gh.issue_list(
                labels=[self.config.labels.ready],
                state="all",
            )
        closed_ready = [
            issue
            for issue in ready_issues
            if str(issue.get("state") or "OPEN").upper() == "CLOSED"
        ]
        if not closed_ready:
            return set(), ready_issues, _MergedPRListOutcome()

        finalize_limit = self.config.dispatch.finalize_limit
        if finalize_limit <= 0:
            return set(), ready_issues, _MergedPRListOutcome()

        # Try the cheap 500-window binding first; if the GraphQL-budget guard
        # refuses the call, fall back to per-issue search for all candidates.
        bound_issue_numbers: set[int] = set()
        mention_only_issue_numbers: set[int] = set()
        merged_pr_outcome = _MergedPRListOutcome()
        try:
            merged_prs = self.gh.merged_pr_list()
        except GitHubError as exc:
            merged_pr_outcome = _MergedPRListOutcome([], exc, called=True)
            merged_prs = []
        else:
            merged_pr_outcome = _MergedPRListOutcome(merged_prs, called=True)
        for pr in merged_prs:
            if str(pr.get("state") or "").upper() != "MERGED":
                continue
            bound = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if bound is not None:
                bound_issue_numbers.add(bound)
            # isCrossRepository describes the PR's own head-branch provenance
            # (fork vs. same-repo). It cannot fully guard a cross-repo mention
            # collision, but it does guard the common case of a fork PR's text
            # being trusted at all.
            if pr.get("isCrossRepository") is False:
                for mentioned in issue_numbers_mentioned_by_pr(pr):
                    mention_only_issue_numbers.add(mentioned)

        # Mention-only references are advisory; they are not a binding, but
        # they also must not be stripped as "unmerged" — dispatch() will flag
        # them for a human decision.
        mention_only_issue_numbers -= bound_issue_numbers

        # Only unbound closed issues are candidates for per-issue search or strip.
        unbound_issues = [
            issue for issue in closed_ready if int(issue["number"]) not in bound_issue_numbers
        ]

        def _finalization_order(issue: dict[str, Any]) -> tuple[str, int]:
            return (str(issue.get("createdAt") or ""), int(issue["number"]))

        # Slice BEFORE any per-issue lookup so a large backlog cannot exhaust
        # the GitHub Search API bucket in a single pass.
        candidates = sorted(unbound_issues, key=_finalization_order)[:finalize_limit]

        issue_pr_map: dict[int, list[dict[str, Any]]] = {}
        closed_unmerged_ready_issues: set[int] = set()
        consecutive_failures = 0
        for issue in candidates:
            if consecutive_failures >= 3:
                break
            issue_number = int(issue["number"])
            merged_prs = self.gh.merged_prs_for_issue(
                issue_number,
                self.config.dispatch.branch_prefix,
            )
            if not getattr(merged_prs, "ok", True):
                consecutive_failures += 1
                continue
            consecutive_failures = 0
            if merged_prs:
                issue_pr_map[issue_number] = list(merged_prs)
            elif issue_number not in mention_only_issue_numbers:
                # Confirmed closed ready issue with no merged PR binding it.
                closed_unmerged_ready_issues.add(issue_number)

        # Persist state first, then apply labels outside the lock.
        if issue_pr_map or closed_unmerged_ready_issues:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                for issue_number, prs in issue_pr_map.items():
                    issue_key = str(issue_number)
                    issue_entry = state["issues"].get(issue_key, {})
                    state["issues"][issue_key] = {
                        **issue_entry,
                        "number": issue_number,
                        "status": "closed",
                    }
                    for pr in prs:
                        pr_number = int(pr["number"])
                        pr_key = str(pr_number)
                        pr_entry = state["prs"].get(pr_key, {})
                        state["prs"][pr_key] = {
                            **pr_entry,
                            "number": pr_number,
                            "status": "merged",
                            "merged": True,
                            "issue_number": issue_number,
                        }
                if issue_pr_map:
                    state = self._record_event(
                        state,
                        "finalize_externally_merged",
                        {
                            "issue_numbers": sorted(issue_pr_map.keys()),
                            "pr_numbers": sorted(
                                {int(pr["number"]) for prs in issue_pr_map.values() for pr in prs}
                            ),
                        },
                    )
                for issue_number in closed_unmerged_ready_issues:
                    issue_key = str(issue_number)
                    issue_entry = state["issues"].get(issue_key, {})
                    state["issues"][issue_key] = {
                        **issue_entry,
                        "number": issue_number,
                        "status": "closed",
                    }
                if closed_unmerged_ready_issues:
                    state = self._record_event(
                        state,
                        "dispatch_closed_unmerged_ready_stripped",
                        {"issue_numbers": sorted(closed_unmerged_ready_issues)},
                    )
                save_state(self.paths.state_file, state)

        for issue_number in issue_pr_map:
            transition(self.gh, self.config.labels, issue_number, "merged")
            self.gh.close_issue(issue_number)

        for issue_number in closed_unmerged_ready_issues:
            transition(self.gh, self.config.labels, issue_number, "closed_unmerged")

        finalized: set[int] = set(issue_pr_map.keys())
        removed = finalized | closed_unmerged_ready_issues
        remaining = [issue for issue in ready_issues if int(issue["number"]) not in removed]
        return finalized, remaining, merged_pr_outcome

    def dispatch(
        self,
        limit: int | None = None,
        *,
        only_issues: str | None = None,
        stalled_entries: list[dict[str, int]] | None = None,
    ) -> CommandResult:
        """Dispatch fresh workers for ready issues.

        ``stalled_entries``: pass the result of an already-completed
        ``_detect_and_handle_stalled_sessions`` sweep to reuse it instead of
        re-running the sweep inside this call. ``loop()`` does this because it
        runs the sweep itself at the top of each pass; the sweep is the sole
        writer of Signal-1's inconclusive-probe deferral counter, so re-running
        it here would advance that counter more than once per pass and erode
        the ``max_inconclusive_probe_deferrals`` grace period (issue #343
        Finding 2). Standalone callers leave this as None and the sweep runs
        inside this call as before.
        """
        # Issue #646: unconditional census of every alive worker, logged before
        # any guard below can short-circuit (state lock busy, fleet lock held,
        # GraphQL budget deferred) -- this is the one chokepoint every dispatch
        # path funnels through, whether invoked standalone (`work`/`fleet work`)
        # or from inside a supervised pass (`loop()` -> `_loop_body()` ->
        # `dispatch()`), so it answers "how many suites were running at <time>,
        # from which worktrees, at what cap" regardless of which command
        # launched them. Purely read-only, but explicitly guarded: per-file
        # read errors are already swallowed inside read_worker_records/
        # read_session_records, but this diagnostic must never be the reason a
        # whole dispatch pass aborts, so any other unexpected failure here
        # (formatting, directory-listing races, etc.) is logged and swallowed
        # rather than propagated -- a torn sidecar read is *more* likely, not
        # less, during the exact high-concurrency moment this census exists to
        # diagnose.
        try:
            _log_worker_census(self._layout.sessions_dir)
        except Exception:
            import logging

            logging.getLogger(__name__).warning("worker census failed", exc_info=True)
        # Finalize closed ready-labeled issues whose linked PR merged externally.
        # This runs before fleet lock / GraphQL budget / provider throttle guards
        # so a pass that defers new dispatch still drains the Aviator-merge backlog.
        finalized: set[int] = set()
        merged_pr_outcome: _MergedPRListOutcome = _MergedPRListOutcome()
        try:
            finalized, ready_issues, merged_pr_outcome = self._finalize_externally_merged_issues()
        except StateLockBusy:
            return _state_lock_busy_result(
                "dispatch deferred: state lock held",
                selected_count=0,
                deferred_reason="state_lock_busy",
            )

        # Reuse the merged PR list already fetched by _finalize_externally_merged_issues
        # so the post-merge tripwire in loop() can avoid a second GraphQL call.
        merged_prs_for_result: list[dict[str, Any]] | None = (
            merged_pr_outcome.items
            if merged_pr_outcome.called and merged_pr_outcome.error is None
            else None
        )

        fleet_lock = None
        if self.config.fleet.global_max_concurrent_sessions > 0:
            fleet_lock = try_acquire_fleet_lock(self.fleet_dir_override)
            if fleet_lock is None:
                return CommandResult(
                    True,
                    "dispatch deferred: fleet lock held",
                    {
                        "selected_count": 0,
                        "deferred_reason": "fleet_lock_held",
                        "merged_prs": merged_prs_for_result,
                        "merged_pr_closed_issue_numbers": sorted(finalized),
                        "merged_pr_referenced_issue_numbers": sorted(finalized),
                    },
                )
        try:
            result = self._dispatch_impl(
                limit,
                only_issues=only_issues,
                stalled_entries=stalled_entries,
                ready_issues=ready_issues,
                merged_prs=merged_pr_outcome,
            )
            data = dict(result.data)
            if finalized:
                data["merged_pr_closed_issue_numbers"] = sorted(
                    set(data.get("merged_pr_closed_issue_numbers", [])) | finalized
                )
                data["merged_pr_referenced_issue_numbers"] = sorted(
                    set(data.get("merged_pr_referenced_issue_numbers", [])) | finalized
                )
            return CommandResult(result.ok, result.message, data)
        except StateLockBusy:
            return _state_lock_busy_result(
                "dispatch deferred: state lock held",
                selected_count=0,
                deferred_reason="state_lock_busy",
                merged_prs=merged_prs_for_result,
                merged_pr_closed_issue_numbers=sorted(finalized),
                merged_pr_referenced_issue_numbers=sorted(finalized),
            )
        except GraphQLBudgetError as exc:
            return CommandResult(
                True,
                "dispatch deferred: GraphQL rate limit below threshold",
                {
                    "selected_count": 0,
                    "deferred_reason": "graphql_rate_limit",
                    "graphql_remaining": exc.remaining,
                    "graphql_reset": exc.reset_at,
                    "graphql_threshold": exc.threshold,
                    "merged_prs": merged_prs_for_result,
                    "merged_pr_closed_issue_numbers": sorted(finalized),
                    "merged_pr_referenced_issue_numbers": sorted(finalized),
                },
            )
        except GitHubError as exc:
            # A GitHubError from _dispatch_impl means a GitHub API call
            # needed for reliable dispatch failed. The two known sources are
            # merged_pr_list() (raised on unusable responses — empty stdout,
            # non-zero exit, unparseable JSON — per #633) and pr_list(); both
            # are fetched before any issue is claimed or worker launched, so
            # deferring here cannot leave a partial claim. Earlier
            # _finalize_externally_merged_issues already recorded its own
            # merged_pr_list failure in merged_pr_outcome, and
            # _resolve_merged_prs re-raises that stored error (branch 2) so
            # this handler covers BOTH the direct-fallback fetch (branch 1,
            # the common case when there are open ready issues but no
            # closed-ready issues this pass) and the finalize-errored re-raise
            # (branch 2). Deferring is the correct response: proceeding with
            # an empty merged-PR set would re-dispatch issues a merged PR
            # already covered (the silent-empty path #633 closed), and letting
            # the error propagate crashes the supervised loop daemon on a
            # transient gh failure. Any claim written before a later
            # GitHubError (e.g. issue_view mid-launch) is recovered by the
            # existing stale-claim sweep on the next pass.
            return CommandResult(
                True,
                f"dispatch deferred: GitHub API error ({exc})",
                {
                    "selected_count": 0,
                    "deferred_reason": "github_error",
                    "github_error": str(exc),
                    "merged_prs": merged_prs_for_result,
                    "merged_pr_closed_issue_numbers": sorted(finalized),
                    "merged_pr_referenced_issue_numbers": sorted(finalized),
                },
            )
        finally:
            if fleet_lock is not None:
                fleet_lock.release()

    def _dispatch_impl(
        self,
        limit: int | None = None,
        *,
        only_issues: str | None = None,
        stalled_entries: list[dict[str, int]] | None = None,
        ready_issues: list[dict[str, Any]] | None = None,
        merged_prs: _MergedPRListOutcome | None = None,
    ) -> CommandResult:
        # Issue #427: include closed ready-labeled issues so externally-merged PRs
        # (e.g. Aviator MergeQueue) can be finalized even after GitHub closes the issue.
        if ready_issues is None:
            issues = self.gh.issue_list(
                labels=[self.config.labels.ready],
                state="all",
            )
        else:
            issues = ready_issues
        dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit
        operator_claimed_ready: list[int] = []

        # Gather sessions_dir for stall detection and live worker counting
        sessions_dir = self._layout.sessions_dir

        # Detect and handle stalled sessions before applying concurrency governor.
        # This must run exactly once per pass, not twice (was duplicated in the
        # governor). When the caller already ran the sweep this pass (loop()'s
        # unconditional reaper at the top of its pass), it hands the result down
        # via ``stalled_entries`` and the sweep is NOT re-run here — see
        # dispatch()'s docstring for why re-running it corrupts the Signal-1
        # deferral counter.
        if stalled_entries is None:
            stalled_entries = _detect_and_handle_stalled_sessions(
                sessions_dir, self.paths.state_file, self.config
            )

        # Count live workers after stall handling (stalled workers are killed).
        # Corroborated against state.json (issue #343) so a ghost -- a live
        # worker_pid whose sidecar was removed -- cannot silently free a slot.
        live_count = _count_live_sessions(sessions_dir, self.paths.state_file)

        # Apply global concurrency governor cap with pre-computed live_count
        gov = self._apply_concurrency_governor(dispatch_limit, live_count=live_count)
        dispatch_limit = gov.dispatch_limit

        # Compute the merged PR list (if already fetched) for the tripwire so
        # loop() can reuse it and avoid a second GraphQL call per pass.
        merged_prs_for_tripwire: list[dict[str, Any]] | None = (
            merged_prs.items
            if merged_prs is not None and merged_prs.called and merged_prs.error is None
            else None
        )

        # Apply provider throttle cooldown check
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            if is_throttled(state):
                throttled_until = state.get("throttled_until")
                # Return immediately with deferral reason
                data = {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "skipped_issue_numbers": [],
                    "label_errors": [],
                    "sessions": [],
                    "dispatch_results": [],
                    "merged_prs": merged_prs_for_tripwire,
                    "deferred_reason": "provider_throttled",
                    "throttled_until": throttled_until,
                }
                if gov.enabled or gov.fleet_enabled:
                    data.update(gov.report_fields())
                return CommandResult(
                    False,
                    f"dispatch deferred: provider throttled until {throttled_until}",
                    data,
                )

        def _resolve_merged_prs(
            outcome: _MergedPRListOutcome | None,
        ) -> list[dict[str, Any]]:
            # Both raising branches (the direct fallback below and the
            # outcome.error re-raise) propagate GitHubError to dispatch()'s
            # ``except GitHubError`` handler, which defers the pass. This is
            # deliberate: proceeding with [] would re-dispatch issues a merged
            # PR already covered (the silent-empty path #633 closed). The
            # direct fallback is the COMMON case — it runs whenever there are
            # open ready issues but no closed-ready issues this pass, because
            # _finalize_externally_merged_issues skips the merged_pr_list()
            # fetch entirely when closed_ready is empty (returning an outcome
            # with called=False).
            if outcome is None or not outcome.called:
                return self.gh.merged_pr_list() if issues else []
            if outcome.error is not None and issues:
                raise outcome.error
            return outcome.items if issues else []

        # Dry-run: read-only planning — compute selection and would-be SessionRequests,
        # but skip all state writes, label transitions, and file mutations.
        if self.dry_run:
            selected_issue_numbers: list[int] = []
            skipped_issue_numbers: list[int] = []
            # Detect stalled sessions (read-only for dry-run)
            stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)
            stalled_issues = {entry["issue"] for entry in stalled_entries}
            live_worker_issues = _issues_with_live_workers(sessions_dir)
            prs = self.gh.pr_list()
            # No ready issues means _merged_pr_referenced_issue_numbers() would
            # return empty sets regardless of what merged_pr_list() returns
            # (it intersects against the ready-issue-number set) — skip the
            # expensive listing query entirely rather than fetch-and-discard
            # (issue #361).
            merged_prs = _resolve_merged_prs(merged_prs)
            (
                merged_pr_bound_issue_numbers,
                merged_pr_mention_only_issue_numbers,
                _,
            ) = self._merged_pr_referenced_issue_numbers(issues, merged_prs)
            merged_pr_issue_numbers = (
                merged_pr_bound_issue_numbers | merged_pr_mention_only_issue_numbers
            )
            pr_by_issue = {}
            for pr in prs:
                issue_number = linked_issue_number(
                    pr,
                    is_cross_repository=pr.get("isCrossRepository"),
                    branch_prefix=self.config.dispatch.branch_prefix,
                )
                if issue_number is not None:
                    # If multiple PRs link to the same issue, keep the lowest PR number
                    if issue_number not in pr_by_issue or int(pr["number"]) < int(
                        pr_by_issue[issue_number]["number"]
                    ):
                        pr_by_issue[issue_number] = pr

            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                # Same dispatchability logic as the real dispatch, but read-only
                # Issue #5: also check worker liveness for "dispatched" status
                live_dispatched = set()
                for number, entry in state.get("issues", {}).items():
                    if not isinstance(entry, dict):
                        continue
                    status = entry.get("status")
                    if status == "dispatch_pending" and not is_claim_stale(
                        entry.get("dispatch_pending_at")
                    ):
                        live_dispatched.add(int(number))
                    elif status == "dispatched":
                        # Issue #5: only exclude if the worker is alive OR there's an open PR.
                        # A dead worker with no open PR is recoverable (crashed before PR opened).
                        # A dead worker with an open PR is mid-review and must not be re-dispatched.
                        # Issue #207: also check state.json worker_pid for liveness when session files are orphaned
                        issue_number = int(number)
                        worker_alive = _worker_pid_alive(entry)
                        if (
                            issue_number in live_worker_issues
                            or worker_alive
                            or issue_number in pr_by_issue
                        ):
                            live_dispatched.add(issue_number)
                issues_with_open_tracked_prs = set(pr_by_issue.keys())
            candidates = [
                issue
                for issue in issues
                if self._is_dispatchable(issue)
                and int(issue["number"]) not in live_dispatched
                and int(issue["number"]) not in stalled_issues
                and int(issue["number"]) not in issues_with_open_tracked_prs
                and int(issue["number"]) not in merged_pr_issue_numbers
            ]

            # Apply dependency gate: skip issues with open blockers (dry-run)
            # Done outside the lock to avoid holding it during GitHub API calls
            candidates, blocked_issues, _open_blockers_by_issue = self._filter_blocked_issues(
                candidates
            )

            # Sort candidates by dispatch order
            # Default (oldest) uses dependency-aware ordering; explicit newest uses creation date
            if self.config.dispatch.order == "newest":
                candidates = self._sort_by_dispatch_order(candidates)
            else:
                # Default: use dependency-aware ordering (out-degree) with oldest-first tiebreaker
                candidates = self._sort_by_dependency_depth(candidates)

            # Fill fresh candidates first; recovery retries only get leftover slots
            # and are capped at one per pass (issue #506).
            selected, skipped_issue_numbers, deferred_by_concurrency = _select_dispatch_candidates(
                candidates,
                dispatch_limit,
                state,
                self._branch_name,
                only_issues=only_issues,
            )
            selected_issue_numbers = [int(issue["number"]) for issue in selected]

            # Compute would-be SessionRequests without state mutation
            session_requests: list[SessionRequest] = []
            full_issues: dict[int, dict[str, Any]] = {}
            # Issue #482: compute would-be adapter choices (read-only).
            adapter_choices: dict[int, AdapterChoice] = {}
            api_enabled = self.config.api_worker.enabled
            routing_inputs = self._routing_inputs() if api_enabled else None
            for issue_number in selected_issue_numbers:
                full_issue = self.gh.issue_view(issue_number)
                full_issues[issue_number] = full_issue
                branch_name = self._branch_name(full_issue)

                template: str | None = None
                if api_enabled and routing_inputs is not None:
                    issue_labels = {label["name"] for label in full_issue.get("labels", [])}
                    choice = self._select_adapter_for_issue(
                        rework=False,
                        issue_labels=issue_labels,
                        routing_inputs=routing_inputs,
                    )
                    adapter_choices[issue_number] = choice
                    if choice.kind == "api":
                        template = self.config.api_worker.worker_template

                prompt_path = self._write_worker_prompt(full_issue, template=template)

                # Check if this is a dead-worker recovery (same logic as real dispatch)
                recovery_record: dict[str, Any] | None = None
                prev_entry = state.get("issues", {}).get(str(issue_number), {})
                prev_branch = prev_entry.get("branch_name")
                if prev_branch == branch_name and prev_entry.get("status") == "dispatched":
                    recovery_record = prev_entry

                session_requests.append(
                    SessionRequest(
                        issue_number=issue_number,
                        issue_title=str(full_issue.get("title") or ""),
                        prompt_path=prompt_path,
                        branch_name=branch_name,
                        recovery=recovery_record,
                    )
                )

            # Return planning data without touching state, labels, or manifest/results files
            data = {
                "selected_count": len(session_requests),
                "attempted_count": len(session_requests),
                "failed_count": 0,
                "skipped_issue_numbers": skipped_issue_numbers,
                "deferred_by_concurrency": deferred_by_concurrency,
                "merged_prs": merged_prs,
                "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                "merged_pr_mention_only_issue_numbers": sorted(
                    merged_pr_mention_only_issue_numbers
                ),
                "label_errors": [],
                "sessions": [asdict(request) for request in session_requests],
                "adapter_choices": {
                    str(n): {"kind": c.kind, "provider": c.provider, "reason": c.reason}
                    for n, c in sorted(adapter_choices.items())
                },
                "dispatch_results": [],
                "blocked": [
                    {"issue": issue_number, "blockers": blockers}
                    for issue_number, blockers in sorted(blocked_issues.items())
                ],
                "stalled": stalled_entries,
            }
            if gov.enabled or gov.fleet_enabled:
                data.update(gov.report_fields())
            return CommandResult(
                True,
                f"dry-run: would dispatch {len(session_requests)} issue(s)",
                data,
            )

        # Real dispatch: claim issues, launch workers, update state and labels
        # First lock: claim issues by marking them as dispatch_pending
        selected_issue_numbers: list[int] = []
        skipped_issue_numbers: list[int] = []
        # Use pre-computed stalled_entries from the stall detection above
        stalled_issues = {entry["issue"] for entry in stalled_entries}
        live_worker_issues = _issues_with_live_workers(sessions_dir)
        prs = self.gh.pr_list()
        # No ready issues means _merged_pr_referenced_issue_numbers() would
        # return empty sets regardless of what merged_pr_list() returns (it
        # intersects against the ready-issue-number set) — skip the expensive
        # listing query entirely rather than fetch-and-discard (issue #361).
        merged_prs = _resolve_merged_prs(merged_prs)
        (
            merged_pr_bound_issue_numbers,
            merged_pr_mention_only_issue_numbers,
            merged_pr_bound_pr_numbers,
        ) = self._merged_pr_referenced_issue_numbers(issues, merged_prs)
        merged_pr_issue_numbers = (
            merged_pr_bound_issue_numbers | merged_pr_mention_only_issue_numbers
        )

        # Issue #432: cap merge-finalization per pass so a large backlog cannot
        # monopolize the pass budget. Oldest first (by creation date, then issue
        # number) drains the backlog deterministically.
        issue_by_number = {int(issue["number"]): issue for issue in issues}
        finalize_limit = self.config.dispatch.finalize_limit

        def _finalization_order(issue_numbers: set[int]) -> list[int]:
            return sorted(
                issue_numbers,
                key=lambda n: (issue_by_number.get(n, {}).get("createdAt", ""), n),
            )

        finalizable_bound_issue_numbers = _finalization_order(merged_pr_bound_issue_numbers)[
            :finalize_limit
        ]
        finalizable_mention_issue_numbers = _finalization_order(
            merged_pr_mention_only_issue_numbers
        )[:finalize_limit]

        pr_by_issue = {}
        for pr in prs:
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is not None:
                # If multiple PRs link to the same issue, keep the lowest PR number
                if issue_number not in pr_by_issue or int(pr["number"]) < int(
                    pr_by_issue[issue_number]["number"]
                ):
                    pr_by_issue[issue_number] = pr

        # Close ready issues whose merged PR safely binds to them (hijack-safe:
        # same-repo branch-prefix or closing-action verb — the same trust
        # level issue #220 uses to close at merge time). This is
        # belt-and-suspenders in case #220's merge-time close hasn't landed
        # yet. These are network calls, so they run outside the state lock;
        # the successful closures are persisted to state.json inside the lock
        # below, and the issue numbers are excluded from dispatch candidates
        # regardless of closure success. Issue #432: only the oldest
        # finalize_limit issues are processed per pass, so a one-time backlog
        # cannot monopolize the pass budget.
        closed_merged_pr_issues: set[int] = set()
        for issue_number in finalizable_bound_issue_numbers:
            # Best-effort label transition and issue close. A failure here is
            # non-fatal; the issue is still excluded from dispatch because the
            # merged PR reference exists, and the next pass will retry.
            transition(self.gh, self.config.labels, issue_number, "merged")
            if self.gh.close_issue(issue_number):
                closed_merged_pr_issues.add(issue_number)

        # Issue #203 (redesigned per review): a merged PR that only
        # *mentions* the issue in free text has no hijack-safe binding and
        # must never authorize a close. Flag it for a human instead — the
        # issue is excluded from this pass's candidates (via
        # merged_pr_issue_numbers below) and left OPEN for the operator to
        # decide whether to close it, wire up a proper closing reference, or
        # redispatch it. Issue #432: capped to finalize_limit per pass.
        #
        # Issue #564: one-shot flagging. The flag must fire once per issue,
        # not every pass — otherwise the operator's removal of
        # agent:human-needed is overridden on the next pass and the event
        # stream is spammed with one dispatch_merged_pr_mention_flagged event
        # per pass while the mention persists. Skip issues whose state entry
        # already records merged_pr_mention_flagged_at (set the first time
        # this path flagged them). This follows the emit-on-change dedup
        # pattern established in #556 for dispatch_skip_blocked/janitor_gate.
        #
        # Re-flag semantics: keyed on the timestamp's absence — once flagged,
        # an issue is never re-flagged, even if a NEW merged PR mentions it.
        # The simplest acceptable semantics per issue #564; pinned by
        # test_dispatch_merged_pr_mention_flag_is_one_shot.
        #
        # Known limitation (issue #564 point 2, documented as out of scope):
        # the mention-only *dispatch exclusion* still keys off the raw mention
        # scan (merged_pr_issue_numbers below), not the label. So an operator
        # who removes agent:human-needed to re-arm automation does NOT re-enter
        # dispatch — the scan-based exclusion keeps blocking the issue until it
        # closes or the mentioning PRs are no longer merged/referenced. Keying
        # the exclusion off the label instead would let a deliberate operator
        # requeue take effect, but it widens the blast radius (label-read
        # dependency in the candidate filter) and is left for a follow-up.
        # load_state_locked (not raw load_state) so the read holds the
        # advisory state lock — required by the invariant enforced in
        # test_no_unlocked_load_state_in_production_code. The authoritative
        # timestamp write below is a separate locked critical section; this
        # read is best-effort relative to it but must still hold the lock to
        # avoid racing a concurrent tmp+replace writer (issue #310).
        mention_state = load_state_locked(self.paths.state_file)
        already_flagged_mention_issues = {
            int(num)
            for num, entry in mention_state.get("issues", {}).items()
            if isinstance(entry, dict) and entry.get("merged_pr_mention_flagged_at")
        }
        newly_flagged_mention_issues = [
            n for n in finalizable_mention_issue_numbers if n not in already_flagged_mention_issues
        ]
        # Capture the transition outcome per issue so the dedup marker below
        # is only stamped for issues whose label edge actually took effect.
        # Stamping unconditionally (the pre-fix behavior) meant a
        # PARTIAL_FAILURE label write still recorded
        # merged_pr_mention_flagged_at, permanently suppressing retry (the
        # one-shot guard above keys off the timestamp's presence) with no
        # diagnostic beyond transition()'s own log line. NOTHING_CHANGED is
        # treated the same as APPLIED: per labels.py's _edges(),
        # "merged_pr_mention_flagged" always has a non-empty add tuple, so
        # NOTHING_CHANGED is unreachable for this event today, but it is
        # handled here defensively since a retry would recompute the exact
        # same static edge and produce the same NOTHING_CHANGED outcome again.
        mention_flag_outcomes: list[tuple[int, TransitionOutcome]] = [
            (
                issue_number,
                transition(
                    self.gh, self.config.labels, issue_number, "merged_pr_mention_flagged"
                ).outcome,
            )
            for issue_number in newly_flagged_mention_issues
        ]
        stamped_mention_issues = [
            issue_number
            for issue_number, outcome in mention_flag_outcomes
            if outcome != TransitionOutcome.PARTIAL_FAILURE
        ]

        # Issue #429/#433: closed-unmerged stripping is handled by
        # _finalize_externally_merged_issues, which already performs the
        # capped per-issue merged-PR lookup and removes stale ready/active labels.

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Persist the fact that merged PRs already covered these ready issues.
            # This keeps state.json consistent with the closed GitHub issue and lets
            # reconcile skip the active-status-on-closed-issue drift sweep.
            for issue_number in closed_merged_pr_issues:
                _issue_key = str(issue_number)
                _issue_entry = state["issues"].get(_issue_key, {})
                state["issues"][_issue_key] = {
                    **_issue_entry,
                    "number": issue_number,
                    "status": "closed",
                }
            if closed_merged_pr_issues:
                state = append_event(
                    state,
                    "dispatch_merged_pr_references_closed",
                    {"issue_numbers": sorted(closed_merged_pr_issues)},
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
            # Issue #427: finalize state.json entries for the merged PRs so
            # externally-merged PRs (Aviator mergequeue handoff) do not leave
            # stale prs[...].status == "mergequeue" behind.
            for pr_number in merged_pr_bound_pr_numbers:
                _pr_key = str(pr_number)
                _pr_entry = state["prs"].get(_pr_key, {})
                state["prs"][_pr_key] = {
                    **_pr_entry,
                    "status": "merged",
                    "merged": True,
                }
            if merged_pr_bound_pr_numbers:
                save_state(self.paths.state_file, state)
            # Record a flag timestamp so operators/tooling (e.g. a doctor
            # check) can surface mention-only coverage without re-deriving
            # the mention scan. "status" is deliberately untouched — the
            # issue stays open and its normal state machine intact.
            # Issue #564: only record/emit for issues flagged *this* pass
            # (newly_flagged_mention_issues); already-flagged issues are
            # skipped so the event fires once and the operator's label
            # removal is not overridden on the next pass.
            # Only issues whose transition() outcome was not PARTIAL_FAILURE
            # (stamped_mention_issues, computed above) get the dedup marker —
            # a failed label write must leave it unset so the next pass
            # retries instead of silently leaving the wrong labels forever.
            for issue_number in stamped_mention_issues:
                _issue_key = str(issue_number)
                _issue_entry = state["issues"].get(_issue_key, {})
                state["issues"][_issue_key] = {
                    **_issue_entry,
                    "number": issue_number,
                    "merged_pr_mention_flagged_at": utc_now(),
                }
            if stamped_mention_issues:
                state = append_event(
                    state,
                    "dispatch_merged_pr_mention_flagged",
                    {"issue_numbers": stamped_mention_issues},
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
            # Defence-in-depth against double-dispatch: an issue whose state records
            # a live launched worker (status "dispatched") or a fresh pending claim
            # (status "dispatch_pending" not yet stale) is not re-dispatchable even
            # if its GitHub label write failed after the worker launched.
            # _is_dispatchable is label-only; this closes the launched-but-unlabeled
            # window that would otherwise spawn a second worker on the same issue.
            # Stale claims (crashed phase-2) are excluded to allow re-dispatch.
            # Issue #5: also check worker liveness for "dispatched" status to recover
            # from crashed workers before PR opens.
            live_dispatched = set()
            dispatch_blocked = set()
            now = datetime.now(UTC)
            for number, entry in state.get("issues", {}).items():
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status == "dispatch_pending" and not is_claim_stale(
                    entry.get("dispatch_pending_at")
                ):
                    live_dispatched.add(int(number))
                elif status == "dispatched":
                    # Issue #5: only exclude if the worker is alive OR there's an open PR.
                    # A dead worker with no open PR is recoverable (crashed before PR opened).
                    # A dead worker with an open PR is mid-review and must not be re-dispatched.
                    # Issue #207: also check state.json worker_pid for liveness when session files are orphaned
                    issue_number = int(number)
                    worker_alive = _worker_pid_alive(entry)
                    if (
                        issue_number in live_worker_issues
                        or worker_alive
                        or issue_number in pr_by_issue
                    ):
                        live_dispatched.add(issue_number)
                elif status in ("dispatch_failed", "escalated"):
                    # Issue #461: bound dispatch_failed retries using the same
                    # redispatch-window cap that rework uses. A status already
                    # marked ``escalated`` should also drop out of dispatch.
                    issue_number = int(number)
                    if status == "escalated":
                        dispatch_blocked.add(issue_number)
                    else:
                        recent = _recent_dispatch_failed_attempts(
                            entry,
                            now,
                            self.config.watchdog.redispatch_window_minutes,
                        )
                        if len(recent) > self.config.watchdog.max_auto_redispatch:
                            dispatch_blocked.add(issue_number)
            operator_claimed = operator_claimed_issues(state)
            ready_issue_numbers = {int(issue["number"]) for issue in issues}
            operator_claimed_ready = sorted(operator_claimed & ready_issue_numbers)
            issues_with_open_tracked_prs = set(pr_by_issue.keys())
            candidates = [
                issue
                for issue in issues
                if self._is_dispatchable(issue, operator_claimed)
                and int(issue["number"]) not in live_dispatched
                and int(issue["number"]) not in stalled_issues
                and int(issue["number"]) not in issues_with_open_tracked_prs
                and int(issue["number"]) not in merged_pr_issue_numbers
                and int(issue["number"]) not in dispatch_blocked
            ]
            if operator_claimed_ready:
                state = append_event(
                    state,
                    "dispatch_skip_operator_claimed",
                    {"issue_numbers": operator_claimed_ready},
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)

        # Apply dependency gate: skip issues with open blockers
        # Done outside the lock to avoid holding it during GitHub API calls
        candidates, blocked_issues, open_blockers_by_issue = self._filter_blocked_issues(
            candidates
        )

        # Sort candidates by dispatch order
        # Default (oldest) uses dependency-aware ordering; explicit newest uses creation date
        if self.config.dispatch.order == "newest":
            candidates = self._sort_by_dispatch_order(candidates)
        else:
            # Default: use dependency-aware ordering (out-degree) with oldest-first tiebreaker
            candidates = self._sort_by_dependency_depth(candidates)

        # Re-enter lock to log events and claim issues
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)

            # Log dispatch_skip_blocked events for blocked issues. Dedup
            # (cost-spirals.md Finding 3): a still-blocked issue re-selects
            # every pass with the identical blocker list -- 784 byte-identical
            # events over 18h in the investigated window -- so only emit when
            # the (issue, blockers) content actually changed since the last
            # emission, tracked via a compact snapshot on the issue record.
            if blocked_issues:
                for issue_number, blockers in blocked_issues.items():
                    issue_key = str(issue_number)
                    issue_entry = state["issues"].get(issue_key, {})
                    if not isinstance(issue_entry, dict):
                        issue_entry = {}
                    if issue_entry.get("last_skip_blocked_blockers") != blockers:
                        issue_entry = {
                            **issue_entry,
                            "number": issue_number,
                            "last_skip_blocked_blockers": blockers,
                        }
                        state["issues"][issue_key] = issue_entry
                        state = self._record_event(
                            state,
                            "dispatch_skip_blocked",
                            {"issue": issue_number, "blockers": blockers},
                        )

                    # Blocked-chain attention (pr-lifecycle.md/cost-spirals.md
                    # Finding 3/4): an issue whose every currently-open
                    # blocker is itself dead (escalated, or its tracked PR is
                    # escalated/janitor_blocked) can never unblock through any
                    # automated path. Alert once on transition into that
                    # state -- no label changes, diagnostic only -- instead
                    # of silently re-skipping forever (observed: 4+ days
                    # stuck with zero signal).
                    open_blockers = open_blockers_by_issue.get(issue_number, [])
                    dead_blockers = sorted(
                        b for b in open_blockers if self._is_dead_blocker(b, state, pr_by_issue)
                    )
                    chain_dead = bool(open_blockers) and dead_blockers == sorted(open_blockers)
                    previously_alerted = issue_entry.get("chain_dead_alerted_blockers")
                    if chain_dead and previously_alerted != dead_blockers:
                        state["issues"][issue_key] = {
                            **issue_entry,
                            "number": issue_number,
                            "chain_dead_alerted_blockers": dead_blockers,
                        }
                        state = self._record_event(
                            state,
                            "dispatch_blocked_chain_dead",
                            {"issue": issue_number, "chain_root": dead_blockers},
                        )
                    elif not chain_dead and previously_alerted is not None:
                        # Recovered (or the dead set changed) -- clear the
                        # marker so a future transition back into all-dead
                        # alerts again instead of staying silent forever.
                        state["issues"][issue_key] = {
                            **issue_entry,
                            "number": issue_number,
                            "chain_dead_alerted_blockers": None,
                        }
                save_state(self.paths.state_file, state)

            # Fill fresh candidates first; recovery retries only get leftover slots
            # and are capped at one per pass (issue #506).
            selected, skipped_issue_numbers, deferred_by_concurrency = _select_dispatch_candidates(
                candidates,
                dispatch_limit,
                state,
                self._branch_name,
                only_issues=only_issues,
            )
            selected_issue_numbers = [int(issue["number"]) for issue in selected]
            # Capture previous entries for recovery detection BEFORE overwriting status
            # Issue #81: we need to know if an issue was previously "dispatched" on the same branch
            # to recover from a crashed worker. This snapshot must be taken before we overwrite
            # the status to "dispatch_pending".
            previous_entries: dict[int, dict[str, Any]] = {}
            for issue_number in selected_issue_numbers:
                previous_entries[issue_number] = state["issues"].get(str(issue_number), {})
            # Mark selected issues as "dispatch_pending" to claim them before launching
            for issue_number in selected_issue_numbers:
                entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "dispatch_pending",
                    "dispatch_pending_at": utc_now(),
                }
                # A fresh dispatch supersedes any previous orphan flag.
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                state["issues"][str(issue_number)] = entry
            save_state(self.paths.state_file, state)
        # Do all network calls, file writes, and worker launches outside the lock
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        # Issue #482: per-issue adapter routing. When api_worker is enabled,
        # select_adapter is called once per issue to route it to the api or
        # fallback adapter. When disabled, adapter_choices stays empty and
        # _dispatch_partitioned falls back to a single-group dispatch
        # byte-identical to the pre-#482 behavior.
        adapter_choices: dict[int, AdapterChoice] = {}
        api_enabled = self.config.api_worker.enabled
        routing_inputs = self._routing_inputs() if api_enabled else None
        for issue_number in selected_issue_numbers:
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            branch_name = self._branch_name(full_issue)

            # Determine the adapter for this issue (single point of enforcement:
            # routing.select_adapter). The prompt template follows the choice
            # so the api worker gets the claude-code-flavored prompt.
            template: str | None = None
            if api_enabled and routing_inputs is not None:
                issue_labels = {label["name"] for label in full_issue.get("labels", [])}
                choice = self._select_adapter_for_issue(
                    rework=False,
                    issue_labels=issue_labels,
                    routing_inputs=routing_inputs,
                )
                adapter_choices[issue_number] = choice
                if choice.kind == "api":
                    template = self.config.api_worker.worker_template

            prompt_path = self._write_worker_prompt(full_issue, template=template)

            # Check if this is a dead-worker recovery: the issue has a previous
            # dispatch record with the same branch name (i.e., our own crashed attempt)
            # Use the snapshot captured before status overwrite (Issue #81 fix)
            recovery_record: dict[str, Any] | None = None
            prev_entry = previous_entries.get(issue_number, {})
            prev_branch = prev_entry.get("branch_name")
            if prev_branch == branch_name and prev_entry.get("status") == "dispatched":
                # This is our own crashed attempt - pass the record for recovery
                recovery_record = prev_entry

            session_requests.append(
                SessionRequest(
                    issue_number=issue_number,
                    issue_title=str(full_issue.get("title") or ""),
                    prompt_path=prompt_path,
                    branch_name=branch_name,
                    recovery=recovery_record,
                )
            )
        dispatch_results = self._dispatch_partitioned(session_requests, adapter_choices)
        manifest_path = self._layout.session_manifest
        results_path = self._layout.session_results
        successful_issue_numbers = {
            result.issue_number for result in dispatch_results if result.ok
        }
        # Issue #523: a live_worker_redispatch_averted result claims the prior
        # worker is still alive, but the adapter's probe (_probe_recovery_liveness)
        # can fail closed on an inconclusive real-activity signal (probe_error) or
        # report fresh sessions.db activity even when the recorded wrapper PID is
        # dead/recycled. Verify the PID against the OS (with start-time identity)
        # at the single point where live worker slots are counted — the same
        # is_pid_alive + process_start_time check the review lane uses via
        # _reviewer_pid_alive. A session whose recorded PID is dead is a phantom
        # slot and is routed through the dead-session path below (sidecar reap,
        # label repair) instead of starving fresh dispatch.
        live_worker_issue_numbers: set[int] = set()
        phantom_live_worker_issue_numbers: set[int] = set()
        for result in dispatch_results:
            if result.ok or result.failure_kind != "live_worker_redispatch_averted":
                continue
            if (
                result.pid is not None
                and result.pid > 0
                and is_pid_alive(result.pid, result.process_start_time)
            ):
                live_worker_issue_numbers.add(result.issue_number)
            else:
                phantom_live_worker_issue_numbers.add(result.issue_number)
        failed_issue_numbers = {
            result.issue_number
            for result in dispatch_results
            if not result.ok
            and result.issue_number not in live_worker_issue_numbers
            and result.issue_number not in phantom_live_worker_issue_numbers
        }
        foreign_writer_issue_numbers = {
            result.issue_number
            for result in dispatch_results
            if not result.ok and result.failure_kind == "worktree_foreign_writer"
        }
        # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
        manual = self.config.devin.adapter == "manual"
        label_errors: list[int] = []
        label_error_failures: dict[int, str] = {}
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            for request in session_requests:
                full_issue = full_issues[request.issue_number]
                ok = request.issue_number in successful_issue_numbers
                is_live_worker = request.issue_number in live_worker_issue_numbers
                is_phantom_live_worker = request.issue_number in phantom_live_worker_issue_numbers
                prev_entry = state["issues"].get(str(request.issue_number), {})
                if ok:
                    status = "manifest_written" if manual else "dispatched"
                    dispatched_at = utc_now()
                elif is_live_worker:
                    status = "dispatched"
                    dispatched_at = prev_entry.get("dispatched_at") or utc_now()
                elif is_phantom_live_worker:
                    # Issue #523: the adapter reported a live worker, but the
                    # recorded PID failed the OS-level liveness + identity
                    # check. Route through the dead-session path (sidecar reap,
                    # label repair) instead of keeping the phantom slot
                    # occupied. The slot is freed and the issue becomes
                    # dispatchable again without burning a redispatch attempt.
                    # A dispatched request's issue can never have an open
                    # tracked PR -- candidate selection excludes every issue in
                    # pr_by_issue -- so the rework-routing case is handled by
                    # the dead-session reaper lane, not here.
                    status, dispatched_at, state = self._route_phantom_live_worker(
                        state,
                        request,
                        full_issue,
                        sessions_dir,
                    )
                else:
                    # Issue #461: bound dispatch_failed retries with the same
                    # redispatch-window cap used for rework.
                    now = datetime.now(UTC)
                    all_attempts = list(prev_entry.get("dispatch_failed_at") or [])
                    if not isinstance(all_attempts, list):
                        all_attempts = []
                    all_attempts.append(now.isoformat())
                    recent = _recent_dispatch_failed_attempts(
                        {"dispatch_failed_at": all_attempts},
                        now,
                        self.config.watchdog.redispatch_window_minutes,
                    )
                    # Deterministic launch failures escalate immediately,
                    # mirroring dispatch_rework's post-#550 behavior — fresh
                    # dispatch previously only consulted the redispatch-window
                    # cap, so e.g. a worktree_unsafe failure burned every
                    # capped retry before a human ever heard about it.
                    failed_result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    terminal_failure = (
                        failed_result is not None
                        and failed_result.failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                    )
                    if terminal_failure or len(recent) > self.config.watchdog.max_auto_redispatch:
                        status = "escalated"
                        dispatched_at = None
                    else:
                        status = "dispatch_failed"
                        dispatched_at = None
                entry = {
                    **prev_entry,
                    "number": request.issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "branch_name": request.branch_name,
                    "prompt_path": str(request.prompt_path),
                    "status": status,
                    "dispatched_at": dispatched_at,
                }
                # Clear the claim timestamp on successful upgrade
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                # A successful or live-worker recovery supersedes any previous orphan flag.
                if ok or is_live_worker:
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                    entry.pop("dispatch_failed_at", None)
                    entry.pop("escalation_reason", None)
                elif is_phantom_live_worker:
                    # Issue #523: a phantom live worker is being routed as dead;
                    # do not preserve a stale worker_pid that would keep the slot
                    # occupied, and do not burn a redispatch attempt (the launch
                    # was averted, not failed).
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                    entry.pop("dispatch_failed_at", None)
                    entry.pop("escalation_reason", None)
                    entry.pop("worker_pid", None)
                    entry.pop("worker_process_start_time", None)
                elif status == "escalated":
                    entry["dispatch_failed_at"] = all_attempts
                    entry["escalation_reason"] = (
                        failed_result.failure_kind
                        if terminal_failure and failed_result is not None
                        else "dispatch_failed_cap_exceeded"
                    )
                else:
                    entry["dispatch_failed_at"] = all_attempts
                    entry.pop("escalation_reason", None)
                # Store worker PID and process start time for state-based liveness detection
                # This allows recovery even when session sidecar files are orphaned (issue #207)
                if ok:
                    result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    if result and result.pid is not None:
                        entry["worker_pid"] = result.pid
                        entry["worker_process_start_time"] = result.process_start_time
                state["issues"][str(request.issue_number)] = entry
                # Issue #482: record the adapter choice (including fallback
                # choices) into adapter_history so every routing decision is
                # auditable. record_adapter_choice returns a new state dict
                # (immutable helper); chain it before the atomic save.
                choice = adapter_choices.get(request.issue_number)
                if choice is not None:
                    state = record_adapter_choice(state, request.issue_number, choice, utc_now())
                # Persist the launched worker BEFORE touching GitHub labels: a
                # transient label-write failure (or crash) must never leave a live
                # worker unrecorded and therefore re-dispatchable next wave. The
                # transition is isolated per-issue so one failure never aborts the
                # rest of the batch (orphaning already-launched workers).
                save_state(self.paths.state_file, state)
                if ok or is_live_worker:
                    target = "queued" if manual else "dispatched"
                    result = transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        target,
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": target,
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        entry["label_error"] = label_error
                        label_errors.append(request.issue_number)
                        label_error_failures[request.issue_number] = _label_error_reason(
                            label_error
                        )
                        save_state(self.paths.state_file, state)
                    if is_live_worker:
                        result = next(
                            (
                                r
                                for r in dispatch_results
                                if r.issue_number == request.issue_number
                            ),
                            None,
                        )
                        state = append_event(
                            state,
                            "live_worker_redispatch_averted",
                            {
                                "issue_number": request.issue_number,
                                "branch_name": request.branch_name,
                                "pid": result.pid if result else None,
                                "process_start_time": result.process_start_time
                                if result
                                else None,
                                "probe_result": result.error if result else None,
                            },
                            state_path=self.paths.state_file,
                        )
                        save_state(self.paths.state_file, state)
                elif status == "escalated":
                    # Issue #461: dispatch-failed retry cap exceeded — or a
                    # deterministic launch failure that retrying cannot fix
                    # (escalation_reason carries the failure_kind) — escalate to
                    # human-needed and remove the issue from the dispatch pool.
                    result = transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        "redispatch_escalated",
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": "redispatch_escalated",
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        entry["label_error"] = label_error
                        label_errors.append(request.issue_number)
                        label_error_failures[request.issue_number] = _label_error_reason(
                            label_error
                        )
                        save_state(self.paths.state_file, state)

            # Build dispatch-alert transitions for the notify digest. Averted
            # redispatches surface as DISPATCH_AVERTED; a later successful or
            # non-averted dispatch clears the alert back to OK.
            dispatch_alert_transitions: dict[int, dict[str, Any]] = {}
            live_worker_redispatch_averted: list[dict[str, Any]] = []
            for request in session_requests:
                prev_alert = previous_entries.get(request.issue_number, {}).get("dispatch_alert")
                result = next(
                    (r for r in dispatch_results if r.issue_number == request.issue_number),
                    None,
                )
                is_live_worker = request.issue_number in live_worker_issue_numbers
                if is_live_worker:
                    dispatch_alert_transitions[request.issue_number] = {
                        "adapter_kind": result.adapter if result else "unknown",
                        "health": "DISPATCH_AVERTED",
                        "last_log_line": None,
                        "pid": result.pid if result else None,
                        "terminal_tool": None,
                        "terminal_reason": result.error if result else None,
                    }
                    live_worker_redispatch_averted.append(
                        {
                            "issue_number": request.issue_number,
                            "branch_name": request.branch_name,
                            "pid": result.pid if result else None,
                            "process_start_time": result.process_start_time if result else None,
                            "probe_result": result.error if result else None,
                            "adapter_kind": result.adapter if result else "unknown",
                        }
                    )
                elif prev_alert == "DISPATCH_AVERTED":
                    dispatch_alert_transitions[request.issue_number] = {
                        "adapter_kind": result.adapter if result else "unknown",
                        "health": "OK",
                        "last_log_line": None,
                        "pid": result.pid if result else None,
                        "terminal_tool": None,
                        "terminal_reason": None,
                    }

            for issue_number in foreign_writer_issue_numbers:
                result = next(
                    (r for r in dispatch_results if r.issue_number == issue_number), None
                )
                branch_name = next(
                    (r.branch_name for r in session_requests if r.issue_number == issue_number),
                    None,
                )
                state = append_event(
                    state,
                    "worktree_foreign_writer",
                    {
                        "issue_number": issue_number,
                        "branch_name": branch_name,
                        "pid": result.pid if result else None,
                        "probe_result": result.error if result else None,
                    },
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
            dispatch_failure_map = _build_failure_map(
                dispatch_results,
                failed_issue_numbers,
                deferred_by_concurrency,
                dispatch_limit,
                extra_failures=label_error_failures,
            )
            state = append_event(
                state,
                "dispatch",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "live_worker_issue_numbers": sorted(live_worker_issue_numbers),
                    "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "foreign_writer_issue_numbers": sorted(foreign_writer_issue_numbers),
                    "label_errors": sorted(label_errors),
                    "skipped_issue_numbers": skipped_issue_numbers,
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                    "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
                    "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
                    "failures": dispatch_failure_map,
                },
                state_path=self.paths.state_file,
            )
            save_state(self.paths.state_file, state)
        result_dicts = [result.to_dict() for result in dispatch_results]
        message = "dispatch complete"
        if failed_issue_numbers:
            entries = ", ".join(
                f"#{issue} ({dispatch_failure_map[issue]})"
                for issue in sorted(failed_issue_numbers)
            )
            message = f"dispatch failures: {entries}"
        elif live_worker_issue_numbers:
            message = "dispatch completed with live worker redispatch averted"
        if skipped_issue_numbers:
            message += f" (skipped non-dispatchable: {skipped_issue_numbers})"
        if label_errors:
            message += f" (launched but label write failed: {sorted(label_errors)})"
        if phantom_live_worker_issue_numbers:
            message += (
                f" (reaped phantom live worker slots: {sorted(phantom_live_worker_issue_numbers)})"
            )
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "failures": dispatch_failure_map,
            "live_worker_count": len(live_worker_issue_numbers),
            "phantom_live_worker_count": len(phantom_live_worker_issue_numbers),
            "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
            "foreign_writer_count": len(foreign_writer_issue_numbers),
            "skipped_issue_numbers": skipped_issue_numbers,
            "deferred_by_concurrency": deferred_by_concurrency,
            "merged_prs": merged_prs,
            "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
            "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
            "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
            "label_errors": sorted(label_errors),
            "session_manifest": str(manifest_path),
            "session_results": str(results_path),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": result_dicts,
            "live_worker_redispatch_averted": live_worker_redispatch_averted,
            "stalled": stalled_entries,
            "blocked": [
                {"issue": issue_number, "blockers": blockers}
                for issue_number, blockers in sorted(blocked_issues.items())
            ],
            "operator_claimed_ready": sorted(operator_claimed_ready),
        }
        if gov.enabled or gov.fleet_enabled:
            data.update(gov.report_fields())

        # Emit notification digest if there are health transitions (stalled sessions)
        # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
        if stalled_entries and self.config.notify.enabled:
            health_transitions: dict[int, dict[str, Any]] = {}
            for entry in stalled_entries:
                health_transitions[entry["issue"]] = {
                    "adapter_kind": "unknown",  # Will be filled by #165's full supervisor
                    "health": entry.get("health", "STALLED"),
                    "last_log_line": None,
                    "pid": entry.get("pid"),
                    "terminal_tool": entry.get("terminal_tool"),
                    "terminal_reason": entry.get("terminal_reason"),
                }
            digest = _build_attention_digest(
                self.paths.state_file,
                health_transitions,
                repo=self.repo_root.name,
            )
            if digest:
                emit_digest(self._layout.notify, digest)

        # Emit dispatch-alert digest for live-worker redispatch averted outcomes.
        # This surfaces the silent-stall class of dispatch failures in the same
        # attention pipeline used for stalled workers (issue #506 / #497).
        if dispatch_alert_transitions and self.config.notify.enabled:
            dispatch_digest = _build_attention_digest(
                self.paths.state_file,
                dispatch_alert_transitions,
                repo=self.repo_root.name,
                state_field="dispatch_alert",
            )
            if dispatch_digest:
                emit_digest(self._layout.notify, dispatch_digest)

        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

    @_guard_state_lock
    def review(self, pr_number: int, *, cross_family: bool | None = None) -> CommandResult:
        """Generate a review packet for a PR.

        When config.test_adequacy.enabled, this method may itself issue a
        request_changes verdict and advance/terminate the rework loop (previously
        only record_review/verdict did this). When disabled (default), the method
        never mutates decision state and always returns a packet.

        Args:
            pr_number: The PR number to review.
            cross_family: Whether to enable cross-family review (optional).

        Returns:
            CommandResult with ok=True if a packet was generated, or ok=False if
            the review was blocked (janitor gate, test-adequacy gate) or the PR
            was not found. Two ok=True returns carry NO packet and callers that
            gate a status->"reviewing" flip on ``ok`` must additionally exclude
            them via the data flags: ``routed_to_rework`` (the janitor-gate
            conflict/no-op-rework route re-requests rework with no packet) and
            ``closed_unmerged_converged`` (issue #558: a CLOSED-unmerged PR is
            converged to state status "closed" at the janitor gate -- the PR is
            dead, not a fresh-packet candidate). See
            ``_route_rework_candidate_to_review`` and the dead-worker orphan
            sweep for the canonical gating pattern.
        """
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return CommandResult(False, f"PR #{pr_number} was not found", {})
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
        )

        # Escalation is terminal: once a PR or its linked issue is marked
        # escalated, no further review packet generation or label transitions should
        # occur until a human explicitly de-escalates. This prevents a later loop()
        # pass from clobbering the agent:human-needed label with a review_started
        # transition (issue #384), and it protects PRs that have no resolvable linked
        # issue (cross-repo/fork PRs or branches outside the configured prefix) from
        # falling through to the janitor gate and losing their escalated marker.
        state = load_state_locked(self.paths.state_file)
        pr_state = state.get("prs", {}).get(str(pr_number), {})
        pr_escalated = pr_state.get("status") == "escalated"
        issue_escalated = False
        if issue_number is not None:
            issue_state = state.get("issues", {}).get(str(issue_number), {})
            issue_escalated = issue_state.get("status") == "escalated"
        if pr_escalated or issue_escalated:
            reason = (
                f"issue #{issue_number} is escalated; review skipped"
                if issue_number is not None and issue_escalated
                else f"PR #{pr_number} is escalated; review skipped"
            )
            # Refresh (but never act on) janitor diagnostics while escalated.
            # PRs #1397/#1443 (job-cannon, 2026-07-27) sat with janitor_ok/
            # janitor_failures frozen at hours-stale values because this early
            # return prevented run_janitor from ever re-running once the
            # LINKED ISSUE escalated -- caused there by an unrelated dead
            # rework-worker session, not by the janitor's own verdict. A
            # since-cleared merge conflict or a since-green CI run stayed
            # reported as failing until an operator ran unescalate(), because
            # nothing else re-observes reality for an escalated PR.
            # Escalation must stay terminal for every side effect --status,
            # labels, routing, attempt counters-- only unescalate() may re-arm
            # those (see its docstring) -- so this recomputes janitor_ok/
            # janitor_failures for visibility ONLY, reusing the same
            # dedup'd write/event pattern the janitor gate itself uses
            # (cost-spirals.md Finding 2) so an unrepaired PR doesn't
            # re-log identical failures every pass while escalated either.
            escalated_checks = self.gh.pr_checks(pr_number)
            escalated_diff = self.gh.pr_diff(pr_number)
            with state_lock(self.paths.state_file):
                fresh_state = load_state(self.paths.state_file)
                existing_pr_state = fresh_state["prs"].get(str(pr_number))
                if existing_pr_state is not None:
                    escalated_verdict = run_janitor(
                        pr,
                        escalated_checks,
                        self.config,
                        pr_state=existing_pr_state,
                        repo_root=self.repo_root,
                        pr_diff=escalated_diff,
                    )
                    failures_changed = existing_pr_state.get("janitor_failures") != list(
                        escalated_verdict.failures
                    )
                    fresh_state["prs"][str(pr_number)] = {
                        **existing_pr_state,
                        "janitor_ok": escalated_verdict.ok,
                        "janitor_failures": list(escalated_verdict.failures),
                    }
                    if failures_changed:
                        fresh_state = self._record_event(
                            fresh_state,
                            "janitor_gate",
                            {
                                "pr_number": pr_number,
                                "failures": list(escalated_verdict.failures),
                                "escalated": True,
                            },
                        )
                    save_state(self.paths.state_file, fresh_state)
            return CommandResult(
                True,
                reason,
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "skipped": True,
                    "checks_unavailable": escalated_checks is None,
                },
            )

        issue = self.gh.issue_view(issue_number) if issue_number is not None else {}
        checks = self.gh.pr_checks(pr_number)

        # Load PR state for no-op rework detection (only if PR has verdict history)
        pr_state = None
        if str(pr_number) in state.get("prs", {}):
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})

        # Fetch diff for patch-id based no-op rework detection (issue #222)
        # This is needed before the janitor gate to detect actual content changes
        diff = self.gh.pr_diff(pr_number)

        # Deterministic janitor gate BEFORE any packet/cross-family spend: an
        # obviously-not-ready PR (draft, conflicting, red CI, no issue link)
        # must cost zero review tokens. Most failures don't move labels — they
        # are the worker's/CI's to fix. A definitive required-check failure on
        # a linked-issue PR is routed to rework so the worker can push a fix.
        verdict = run_janitor(
            pr, checks, self.config, pr_state=pr_state, repo_root=self.repo_root, pr_diff=diff
        )
        if not verdict.ok:
            # Issue #558: a PR that is CLOSED (unmerged) on GitHub is dead --
            # every other janitor failure is moot. Converge the state PR
            # entry to "closed" at this boundary (single-point-of-enforcement)
            # so the janitor stops re-fetching and re-evaluating it every
            # pass. The reconcile rule (closed_unmerged_pr_state_converged)
            # is the idempotent backstop for entries the janitor gate never
            # observes (e.g. a PR that closed between passes, or one never
            # routed through review()). MERGED PRs are left to
            # merged_outside_orchestrator's reconcile rule. The linked
            # issue's disposition is left to the existing closed-unmerged
            # issue-side handling.
            if str(pr.get("state") or "").upper() == "CLOSED":
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    existing_pr_state = state["prs"].get(str(pr_number), {})
                    if existing_pr_state.get("status") != "closed":
                        state["prs"][str(pr_number)] = {
                            **without_review_dispatch_claim(existing_pr_state),
                            "number": pr_number,
                            "issue_number": issue_number,
                            "status": "closed",
                        }
                        state = self._record_event(
                            state,
                            "closed_unmerged_pr_state_converged",
                            {"pr_number": pr_number, "issue_number": issue_number},
                        )
                        save_state(self.paths.state_file, state)
                return CommandResult(
                    True,
                    f"PR #{pr_number} is CLOSED (unmerged) on GitHub; "
                    f"converged state status to 'closed'",
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "closed_unmerged_converged": True,
                    },
                )
            # Flake-aware debounce (issue #391): if the only blocker is a failed
            # required check and we have not yet retried the Actions run for this
            # head, trigger one automatic `gh run rerun --failed` and defer rework
            # routing until the next poll. Any rerun-API error is surfaced as an
            # event and we fall through to the existing rework/janitor-block path.
            if verdict.rerun_run_ids:
                rerun_errors: list[str] = []
                triggered_run_ids: list[int] = []
                for run_id in verdict.rerun_run_ids:
                    result = self.gh.run(
                        ["run", "rerun", str(run_id), "--failed"], allow_failure=True
                    )
                    if isinstance(result, GitHubRunResult):
                        if result.ok:
                            triggered_run_ids.append(run_id)
                        else:
                            rerun_errors.append(
                                result.error or f"gh run rerun {run_id} exited {result.returncode}"
                            )
                    elif isinstance(result, str):
                        # Dry-run returns a descriptive string; treat as success.
                        triggered_run_ids.append(run_id)
                    else:
                        rerun_errors.append(
                            f"unexpected result from gh run rerun {run_id}: {result!r}"
                        )

                if triggered_run_ids and not rerun_errors:
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)] = {
                            **state["prs"].get(str(pr_number), {}),
                            "number": pr_number,
                            "issue_number": issue_number,
                            "check_rerun_attempts": verdict.check_rerun_attempts,
                        }
                        state = append_event(
                            state,
                            "flake_rerun_triggered",
                            {
                                "pr_number": pr_number,
                                "run_ids": triggered_run_ids,
                                "head_sha": pr.get("headRefOid"),
                            },
                            state_path=self.paths.state_file,
                        )
                        save_state(self.paths.state_file, state)
                    return CommandResult(
                        False,
                        f"flake rerun triggered for PR #{pr_number}: run(s) "
                        + ", ".join(str(rid) for rid in triggered_run_ids),
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "rerun_run_ids": triggered_run_ids,
                            "checks_unavailable": checks is None,
                        },
                    )

                # Rerun API error: record it, but do not consume the attempt.
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state = append_event(
                        state,
                        "flake_rerun_failed",
                        {
                            "pr_number": pr_number,
                            "run_ids": list(verdict.rerun_run_ids),
                            "errors": rerun_errors,
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)

            if issue_number is not None and verdict.is_check_failure_block:
                transition(self.gh, self.config.labels, issue_number, "review_started")
                summary = f"CI failed on {', '.join(verdict.failed_required_checks)}; push a fix"
                return self.record_review(
                    pr_number,
                    "request_changes",
                    summary=summary,
                    reviewed_head=pr.get("headRefOid"),
                )

            # Merge-conflict and no-op-rework janitor failures used to have no
            # remediation path at all: only is_check_failure_block (above)
            # routed to rework, so a conflicting or diff-unchanged PR fell
            # straight through to the janitor_blocked branch below, which has
            # zero readers anywhere in the codebase (pr-lifecycle.md
            # "janitor_blocked zero readers" finding) -- it just re-logged the
            # identical failure every pass, forever (cost-spirals.md Finding
            # 1: ~700 identical events across 5 PRs in a 19h window). Route
            # both into the same rework machinery the check-failure path
            # uses, decision-agnostic -- the existing merge_ready conflict
            # rework route requires an approved decision (a conflicting
            # branch needs a rebase regardless of its review verdict) -- each
            # bounded by its own small attempt cap so a PR whose rework keeps
            # failing to make progress escalates to a human instead of
            # looping forever.
            is_merge_conflict_block = str(pr.get("mergeable") or "").upper() == "CONFLICTING" or (
                str(pr.get("mergeStateStatus") or "").upper() == "DIRTY"
            )
            # Excludes the case where a required check is ALSO still failing:
            # that combination already has an established, deliberate
            # non-routing behavior (test_janitor_required_check_failure_noop_
            # does_not_reroute, issue #376) -- re-requesting a rework whose
            # only signal is "same diff as last time" while CI is still red
            # is not obviously more productive than waiting, and changing
            # that existing invariant is out of this fix's scope. This only
            # newly routes the PURE no-op-rework case (no co-occurring check
            # failure), which previously had no consumer at all.
            is_no_op_rework_block = verdict.is_no_op_rework and not verdict.failed_required_checks
            if issue_number is not None and (is_merge_conflict_block or is_no_op_rework_block):
                if is_merge_conflict_block:
                    routed = self._route_janitor_gate_failure_to_rework(
                        pr,
                        issue_number,
                        attempts_key="conflict_rework_attempts",
                        max_attempts=self.config.review.max_conflict_rework_attempts,
                        reason="merge_conflict",
                        router=self._request_merge_conflict_rework,
                    )
                else:
                    routed = self._route_janitor_gate_failure_to_rework(
                        pr,
                        issue_number,
                        attempts_key="no_op_rework_attempts",
                        max_attempts=self.config.review.max_no_op_rework_attempts,
                        reason="no_op_rework",
                        router=self._request_no_op_rework_repair,
                    )
                if routed is not None:
                    return routed
                # None: a rework for this issue is already pending, so there is
                # nothing to route -- fall through to the janitor_blocked
                # bookkeeping below and wait for the pending cycle.

            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                existing_pr_state = state["prs"].get(str(pr_number), {})
                # Dedup (cost-spirals.md Finding 2): the janitor gate re-runs
                # every pass and an unrepaired PR produces byte-identical
                # failures for many hours (699 identical events for 5 PRs in
                # a 19h window) -- only log a fresh event when the failure
                # set actually changes from what's already on record.
                failures_changed = existing_pr_state.get("janitor_failures") != list(
                    verdict.failures
                )
                state["prs"][str(pr_number)] = {
                    **existing_pr_state,
                    "number": pr_number,
                    "issue_number": issue_number,
                    "status": "janitor_blocked",
                    "janitor_ok": False,
                    "janitor_failures": list(verdict.failures),
                    "check_rerun_attempts": verdict.check_rerun_attempts,
                }
                if failures_changed:
                    state = self._record_event(
                        state,
                        "janitor_gate",
                        {"pr_number": pr_number, "failures": list(verdict.failures)},
                    )
                save_state(self.paths.state_file, state)
            return CommandResult(
                False,
                f"janitor gate blocked PR #{pr_number}: " + "; ".join(verdict.failures),
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "janitor_ok": False,
                    "janitor_failures": list(verdict.failures),
                    "janitor_warnings": list(verdict.warnings),
                    "checks_unavailable": checks is None,
                },
            )
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(pr_dir / "pr.json", _slim_pr_json(pr))
        self._write_json(pr_dir / "checks.json", checks)
        diff_path = pr_dir / "diff.patch"
        diff_path.write_text(diff, encoding="utf-8")

        # Tier 1 test-adequacy hard gate (issue #179)
        test_adequacy_section = ""
        test_adequacy_verdict = None
        if self.config.test_adequacy.enabled:
            test_adequacy_verdict = check_test_adequacy(diff, pr, self.config.test_adequacy)
            if not test_adequacy_verdict.ok:
                # Same terminal label set as an LLM request_changes:
                # {in_progress} -> review_started -> {in_progress,pr_open,reviewing}
                #               -> rework_requested (inside record_review) -> {in_progress,pr_open,needs_rework}
                if issue_number is not None:
                    transition(self.gh, self.config.labels, issue_number, "review_started")
                summary = render_test_adequacy_summary(
                    test_adequacy_verdict, self.config.test_adequacy.exempt_marker
                )
                return self.record_review(
                    pr_number,
                    "request_changes",
                    summary=summary,
                    reviewed_head=pr.get("headRefOid"),
                )
            # Gate passed while enabled: add Tier-2 packet section (issue #180)
            test_adequacy_section = render_test_adequacy_section(
                test_adequacy_verdict.facts, test_adequacy_verdict.warnings
            )

        # Run containment check for worker edits leaked into operator checkout
        containment_warnings = check_operator_containment(self.repo_root, diff, pr_number)
        # Merge containment warnings with janitor warnings
        merged_warnings = tuple(list(verdict.warnings) + list(containment_warnings))
        # If test-adequacy gate is enabled and passed, merge its warnings too
        if test_adequacy_verdict is not None:
            merged_warnings = tuple(list(merged_warnings) + list(test_adequacy_verdict.warnings))
        cross_family_section, cf_result = self._cross_family_for_pr(
            pr=pr,
            issue=issue,
            pr_dir=pr_dir,
            pr_number=pr_number,
            issue_number=issue_number,
            diff_path=diff_path,
            enabled=cross_family,
        )
        prompt_path = pr_dir / "review-prompt.md"
        decision_path = pr_dir / "review-decision.json"
        diff_size_section = _diff_size_section(
            diff, self.config.review_dispatch.diff_line_threshold, diff_path
        )
        ci_status_section = _ci_status_section(
            checks, self.config.auto_merge.required_checks, pr_dir / "checks.json"
        )

        # Single read of review-decision.json, BEFORE rendering: reused both to
        # build the round-2 $prior_review_section below and, after rendering,
        # by the stale-verdict reset a few lines down. Previously that reset
        # was the only reader, and it ran after the prompt was already
        # rendered — so a prior round's verdict/summary/required_changes were
        # on disk at render time but never surfaced to the reviewer.
        existing_decision = self._review_decision(pr_number)
        prior_reviewed_head_sha = existing_decision.get("reviewed_head_sha")
        # Issue #632 defect 3: a terminal verdict on disk must reach the
        # reviewer even when the head has NOT moved (a PR parked on
        # agent:human-needed, or an operator-corrected verdict). The old gate
        # required prior_reviewed_head_sha != headRefOid, so a same-head
        # re-review rendered an empty $prior_review_section and the corrected
        # findings were invisible. _build_prior_review_section adapts its
        # wording to the same-head vs moved-head case.
        is_round2_review = existing_decision.get("decision") not in (
            "pending",
            None,
            "missing",
            "invalid",
        ) and bool(prior_reviewed_head_sha)
        prior_review_section = (
            self._build_prior_review_section(pr_dir, existing_decision, pr.get("headRefOid"))
            if is_round2_review
            else ""
        )

        prompt = self._render(
            "review.md",
            {
                "pr_number": pr_number,
                "pr_title": pr.get("title", ""),
                "pr_url": pr.get("url", ""),
                "issue_number": issue_number or "UNKNOWN",
                "issue_title": issue.get("title", "UNKNOWN"),
                "issue_url": issue.get("url", ""),
                "pr_json_path": pr_dir / "pr.json",
                "diff_path": pr_dir / "diff.patch",
                "cross_family_section": cross_family_section,
                "janitor_section": _janitor_section(merged_warnings),
                "test_adequacy_section": test_adequacy_section,
                "diff_size_section": diff_size_section,
                "ci_status_section": ci_status_section,
                "prior_review_section": prior_review_section,
            },
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        decision_template = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": "pending",
            "summary": "",
            "required_changes": [],
            "reviewed_at": None,
        }
        if not decision_path.exists():
            self._write_json(decision_path, decision_template)
        else:
            # A verdict is pinned to a specific head. If the PR has moved on,
            # the old verdict is void and must not survive into the new packet.
            # This applies to all terminal decisions (approved, request_changes,
            # blocked), not just approvals: a request_changes on an old head is
            # equally stale when the head has advanced, and carrying forward its
            # summary/required_changes misleads the reviewer into re-issuing the
            # same verdict without examining the new diff.
            if existing_decision.get("decision") not in ("pending", None) and (
                prior_reviewed_head_sha is None or prior_reviewed_head_sha != pr.get("headRefOid")
            ):
                self._write_json(decision_path, decision_template)
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Merge-update, never replace: wholesale assignment here used to erase
            # recorded review decisions on repeated review()/loop() passes
            # (production-confirmed, pr-497).
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "url": pr.get("url"),
                "issue_number": issue_number,
                "prompt_path": str(prompt_path),
                "decision_path": str(decision_path),
                "status": "reviewing",
                "janitor_ok": True,
                "janitor_failures": [],
                "janitor_warnings": list(merged_warnings),
                "cross_family_report": cf_result.report_path if cf_result else None,
                "cross_family_ok": cf_result.ok if cf_result else None,
                "consecutive_failed_merge_attempts": 0,
                "check_rerun_attempts": verdict.check_rerun_attempts,
                # New packet for a (possibly) new head: reset the dispatch
                # attempt counter so the fresh review cycle starts clean.
                "review_dispatch_attempt_count": 0,
                # A clean janitor pass ends the no-op-rework epoch (the
                # janitor's no-op check passing means content actually
                # moved): without this reset, attempts consumed by a long-
                # resolved stall would count against a genuinely new,
                # unrelated one weeks later and escalate it prematurely (the
                # counters are merge-carried forward by every other write to
                # this record).
                "no_op_rework_attempts": 0,
                "no_op_rework_attempts_last_head": None,
                # The conflict epoch resets only on an AFFIRMATIVE mergeable
                # signal: GitHub reports mergeable UNKNOWN/null for a window
                # after every push while it recomputes, and the janitor's
                # conflict check only fails on CONFLICTING/DIRTY -- so a
                # clean pass during that window is not evidence the conflict
                # was resolved, and resetting on it would let a flapping PR
                # relitigate its attempt cap forever.
                **(
                    {
                        "conflict_rework_attempts": 0,
                        "conflict_rework_attempts_last_head": None,
                    }
                    if str(pr.get("mergeable") or "").upper() == "MERGEABLE"
                    else {}
                ),
            }
            if issue_number is not None:
                _issue_key = str(issue_number)
                _issue_entry = state["issues"].get(_issue_key, {})
                state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
            state = append_event(
                state,
                "review_packet",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "cross_family_ok": cf_result.ok if cf_result else None,
                    "cross_family_reused": cf_result.reused if cf_result else None,
                },
                state_path=self.paths.state_file,
            )
            save_state(self.paths.state_file, state)
        # GitHub label side effects are best-effort and isolated: the durable
        # packet above is the authority; a label failure is reported, not fatal.
        label_error: dict[str, Any] | None = None
        if issue_number is not None:
            # Optimization: skip review_started transition if the PR has an unaddressed
            # request_changes decision and the head SHA hasn't changed (nothing new to review).
            # This avoids pointless packet churn and prevents the transition from stripping
            # the needs_rework label from budget-deferred rework candidates.
            should_skip_transition = False
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})
                existing_decision = pr_state.get("decision")
                reviewed_head_sha = pr_state.get("reviewed_head_sha")
                live_head_sha = pr.get("headRefOid")
                if (
                    existing_decision == "request_changes"
                    and reviewed_head_sha is not None
                    and live_head_sha is not None
                    and live_head_sha == reviewed_head_sha
                ):
                    should_skip_transition = True

            if not should_skip_transition:
                result = transition(self.gh, self.config.labels, issue_number, "review_started")
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": "review_started",
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)]["label_error"] = label_error
                        save_state(self.paths.state_file, state)
        message = "review packet generated"
        if label_error:
            message += f" (label update failed: {label_error.get('outcome', label_error)})"
        return CommandResult(
            True,
            message,
            {
                "pr": pr_number,
                "issue": issue_number,
                "prompt_path": str(prompt_path),
                "decision_path": str(decision_path),
                "cross_family_report": cf_result.report_path if cf_result else None,
                "cross_family_ok": cf_result.ok if cf_result else None,
                "cross_family_reused": cf_result.reused if cf_result else None,
                "label_error": label_error,
                "checks_unavailable": False,
            },
        )

    def _sort_review_queue_by_dependency_depth(
        self, queue: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sort review queue so PRs blocking the most downstream work come first.

        Builds the same blocker->dependents graph used by worker dispatch
        against the currently-blocked ready-labeled issues. PRs whose linked
        issue is a blocker for more downstream issues are dispatched first,
        with PR number as a stable tiebreaker.
        """
        import logging

        logger = logging.getLogger(__name__)
        if not queue:
            return queue

        try:
            ready_issues = self.gh.issue_list(
                labels=[self.config.labels.ready],
                state="OPEN",
            )
            blocker_to_dependents: dict[int, list[int]] = {}
            for issue in ready_issues:
                issue_number = int(issue["number"])
                declared_blockers, open_blockers = self._get_open_blockers(issue)
                if not open_blockers:
                    continue
                for blocker in declared_blockers:
                    blocker_to_dependents.setdefault(blocker, []).append(issue_number)

            def sort_key(entry: dict[str, Any]) -> tuple[int, int]:
                return (
                    -len(blocker_to_dependents.get(entry["issue"], [])),
                    entry["pr"],
                )

            return sorted(queue, key=sort_key)
        except Exception:
            logger.warning(
                "Dependency depth sort failed; returning unsorted review queue",
                exc_info=True,
            )
            return queue

    def review_queue(self) -> CommandResult:
        """Enumerate open agent PRs whose review packet is current and awaiting a verdict.

        When a recorded verdict (``approved``, ``request_changes``, or
        ``blocked``) was made against an older head, this method computes the
        live head's stable patch-id. If it matches the recorded
        ``reviewed_patch_id``, the verdict is carried forward to the new head
        (atomic decision-file + state update) and the PR is not queued as
        stale. In dry-run mode the carry-forward write is skipped; the patch-id
        check still runs so the queue reflects real content changes.

        A PR is queued when:

        - It has a linked issue (same as ``review()``).
        - ``prs/pr-N/review-prompt.md`` exists.
        - The recorded decision is ``missing``/``pending`` and the stored packet
          head OID matches the PR's live ``headRefOid``.
        - The recorded decision is a stale ``request_changes``/``blocked``/
          ``approved`` verdict whose patch-id genuinely differs from the live
          head and the packet head is still current.

        Returns:
            CommandResult with a sorted ``queue`` list keyed by repo.
        """
        prs = self.gh.pr_list()
        queue: list[dict[str, Any]] = []

        for pr in prs:
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is None:
                continue

            pr_number = int(pr["number"])
            pr_dir = self.paths.prs / f"pr-{pr_number}"
            prompt_path = pr_dir / "review-prompt.md"
            if not prompt_path.exists():
                continue

            packet_head_sha = self._read_packet_head_oid(pr_number)
            live_head_sha = pr.get("headRefOid")
            if live_head_sha is None:
                continue

            decision = self._review_decision(pr_number)
            decision_value = decision.get("decision")
            reviewed_head_sha = decision.get("reviewed_head_sha")

            if decision_value in ("approved", "request_changes", "blocked"):
                if reviewed_head_sha == live_head_sha:
                    continue

                check = self._check_carry_forward(pr_number, decision)
                if check.carry_forward:
                    # In dry-run mode we still run the content check so the queue
                    # reflects real changes, but we skip the durable head update.
                    if not self.dry_run:
                        try:
                            self._update_approval_head(
                                pr_number,
                                decision,
                                live_head_sha,
                                old_head=reviewed_head_sha,
                                issue_number=issue_number,
                                tier=check.tier or "patch-id",
                                new_patch_id=check.live_patch_id,
                                new_signature=check.live_signature,
                            )
                        except StateLockBusy:
                            # Could not mirror the carry-forward into state.json,
                            # but the decision-file update is the durable source
                            # of truth; proceed as carried-forward.
                            pass
                    continue

                # If the packet is stale, we cannot dispatch a new reviewer from
                # it; the merge gate will route the issue to re-review. Only
                # surface as stale when the packet head is still current.
                if packet_head_sha is None or packet_head_sha != live_head_sha:
                    continue

                queue.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "packet_head_sha": packet_head_sha,
                        "decision": "stale",
                        "reviewed_head_sha": reviewed_head_sha,
                    }
                )
            elif decision_value in ("pending", "missing", "invalid"):
                if packet_head_sha is None or packet_head_sha != live_head_sha:
                    continue

                queue.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "packet_head_sha": packet_head_sha,
                        "decision": decision_value
                        if decision_value in ("pending", "missing")
                        else "missing",
                        "reviewed_head_sha": None,
                    }
                )

        queue = self._sort_review_queue_by_dependency_depth(queue)
        return CommandResult(
            True,
            f"review queue: {len(queue)} PR(s) awaiting verdict",
            {"queue": queue},
        )

    def _reap_review_verdicts(self, reviews_dir: Path) -> dict[str, Any]:
        """Record verdicts for dead reviewers whose sidecar log contains a valid
        fenced JSON verdict block.

        Iterates claude-code review sidecars. For each reviewer that is no
        longer alive and still has a ``review_dispatch_dispatched`` claim, parse
        the log. If a valid verdict block is found, call ``record_review``
        in-process with the packet head pinned as ``reviewed_head`` so the
        verdict is attributed to the diff the reviewer actually read. On
        success ``record_review`` moves the PR to
        ``review_dispatch_completed``. If parsing fails or the verdict is
        malformed, the PR is left for ``_detect_and_handle_stalled_reviews`` to
        retry/backoff using the existing stale-claim path.

        Returns a dict with ``recorded`` and ``missed`` verdict info lists for
        the dispatch result and the fleet attention digest.
        """
        recorded: list[dict[str, Any]] = []
        missed: list[dict[str, Any]] = []

        for w in iter_workers(reviews_dir):
            if w.adapter_kind != "claude-code":
                continue
            if w.is_alive():
                continue

            pr_number = w.issue_number
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})
                if pr_state.get("review_dispatch_status") != "review_dispatch_dispatched":
                    continue
                issue_number = pr_state.get("issue_number")

            verdict_source = "log"
            verdict = _parse_review_verdict_from_log(Path(w.log_path))
            if verdict is None:
                # Fallback: parse the structured events.jsonl. The plaintext log
                # may be truncated or the verdict block split across tee buffer
                # boundaries, but the stream-json events contain the assistant's
                # message text in discrete JSONL lines.
                events_path = _events_path(reviews_dir, pr_number, review=True)
                verdict_source = "events"
                verdict = _parse_review_verdict_from_events(events_path)
            if verdict is None:
                # Last resort (issue #566): the reviewer may have written its
                # verdict to a Markdown file it referenced in final output
                # instead of re-emitting the fenced block. mtime-gated to this
                # session's started_at so stale files never resurrect old
                # verdicts.
                file_hit = _parse_review_verdict_from_files(
                    Path(w.log_path),
                    self.paths.prs / f"pr-{pr_number}",
                    w.started_at,
                )
                if file_hit is not None:
                    verdict, file_source = file_hit
                    verdict_source = f"file:{file_source}"
            if verdict is None:
                # No structured verdict found. Before discarding this reviewer's
                # work, check if it did substantial analysis (e.g. hit the
                # --max-turns limit) and post a summary PR comment so the work
                # is not silently lost. Only post once per dispatch lifecycle.
                pr_state_dict = pr_state
                # One-shot guard. ``review_miss_summary_posted`` covers every
                # miss reason; the legacy turn-limit key is still honoured so
                # PRs mid-lifecycle when this shipped don't get a second
                # comment.
                already_posted = pr_state_dict.get(
                    "review_miss_summary_posted"
                ) or pr_state_dict.get("review_turn_limit_summary_posted")
                if not already_posted:
                    events_path = _events_path(reviews_dir, pr_number, review=True)
                    max_turns = self.config.review_dispatch.review_max_turns
                    outcome = _extract_review_session_summary(
                        events_path,
                        Path(w.log_path),
                        max_turns,
                        session_limit_markers=self.config.runtime.session_limit_markers,
                    )
                    if outcome is not None:
                        try:
                            self._comment_pr(pr_number, outcome.text)
                        except Exception:
                            pass
                        with state_lock(self.paths.state_file):
                            state = load_state(self.paths.state_file)
                            ps = state["prs"].get(str(pr_number), {})
                            # The turn-limit key means "this dispatch
                            # lifecycle's session did substantial work then
                            # died". The provider-throttle sweep reads it to
                            # deny a rollback (#583) and #584 counts it as a
                            # turn-limit death, so a session that never
                            # reached turn 1 must not set it -- that death is
                            # environmental and says nothing about this PR.
                            state["prs"][str(pr_number)] = {
                                **ps,
                                "review_miss_summary_posted": True,
                                "review_turn_limit_summary_posted": (outcome.did_substantial_work),
                            }
                            state = append_event(
                                state,
                                "review_verdict_missed",
                                {
                                    "pr_number": pr_number,
                                    "issue_number": issue_number,
                                    "reason": outcome.reason,
                                    "turn_count": outcome.turn_count,
                                    "tool_call_count": outcome.tool_call_count,
                                },
                                state_path=self.paths.state_file,
                            )
                            save_state(self.paths.state_file, state)
                        missed.append(
                            {
                                "pr": pr_number,
                                "issue": issue_number,
                                "reason": outcome.reason,
                            }
                        )
                continue

            packet_head_sha = self._read_packet_head_oid(pr_number)
            session_metrics = _reviewer_session_metrics(
                _events_path(reviews_dir, pr_number, review=True), verdict_source
            )
            # Fold the review_effort experiment's arm/effort assignment (set
            # at claim time in dispatch_reviews) into session_metrics so the
            # record_review event alone is enough to split spend/quality by
            # arm, without a join back to the dispatch event.
            review_effort_arm = pr_state.get("review_effort_arm")
            review_effort_used = pr_state.get("review_effort_used")
            if review_effort_arm is not None or review_effort_used is not None:
                session_metrics = {
                    **(session_metrics or {}),
                    "review_effort_arm": review_effort_arm,
                    "review_effort_used": review_effort_used,
                }
            result = self.record_review(
                pr_number,
                verdict["decision"],
                summary=verdict["summary"],
                reviewed_head=packet_head_sha,
                required_changes=verdict["required_changes"],
                session_metrics=session_metrics,
            )
            if result.ok:
                recorded.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "decision": verdict["decision"],
                        "verdict_source": verdict_source,
                    }
                )
            else:
                reason = result.message or "record_review failed"
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state = append_event(
                        state,
                        "review_verdict_missed",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "reason": reason,
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)
                missed.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "reason": reason,
                    }
                )

        return {"recorded": recorded, "missed": missed}

    def _record_cross_family_verdicts(self) -> list[dict[str, Any]]:
        """Parse cross-family reports and record verdicts for pending PRs.

        When ``cross_family.auto_verdict`` is enabled and
        ``review_dispatch`` is disabled (the cross-family pass is the sole
        automated review), this scans the review queue for PRs whose
        ``review-decision.json`` is still ``pending`` but whose
        ``cross-family-review.md`` report is valid and non-stale. It
        parses the report into an approved/request_changes verdict and
        records it via ``record_review()``, unblocking the merge lane.

        Returns a list of per-PR result dicts for logging/diagnostics.
        """
        if not self.config.cross_family.auto_verdict:
            return []
        results: list[dict[str, Any]] = []
        queue_result = self.review_queue()
        candidates = queue_result.data.get("queue", [])
        for candidate in candidates:
            pr_number = candidate["pr"]
            decision = candidate.get("decision")
            if decision != "pending":
                continue
            pr_dir = self.paths.prs / f"pr-{pr_number}"
            report_path = pr_dir / "cross-family-review.md"
            if not report_path.exists():
                continue
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except OSError:
                continue
            # Skip stale reports: the head SHA in the report must match the
            # packet head SHA so we don't record a verdict for an old diff.
            report_head = extract_head_ref_oid(report_text)
            packet_head = candidate.get("packet_head_sha")
            if report_head is not None and packet_head is not None and report_head != packet_head:
                continue
            parsed = parse_cross_family_verdict(report_text)
            if parsed is None:
                continue
            verdict_decision = parsed.decision
            record_result = self.record_review(
                pr_number,
                verdict_decision,
                summary=parsed.summary,
                reviewed_head=packet_head,
                required_changes=parsed.required_changes,
            )
            results.append(
                {
                    "pr_number": pr_number,
                    "decision": verdict_decision,
                    "ok": record_result.ok,
                    "message": record_result.message,
                }
            )
        return results

    @_guard_state_lock
    def dispatch_reviews(self, limit: int | None = None) -> CommandResult:
        """Launch Claude Code reviewer sessions concurrently for queued PRs.

        Issue #370: a deterministic loop stage that turns ``review_queue()```
        into launched, sidecar-tracked reviewer processes. Reviewers are
        ``claude_code.launch_claude_worker`` sessions; there is no provider-
        rate-limit governor here — only an optional local-only process cap
        (``max_local_review_processes``) to protect the host from too many
        concurrent reviewer worktrees.

        The double-dispatch protection is a two-phase claim on
        ``state["prs"][pr]``: this method writes ``review_dispatch_pending``,
        launches outside the lock, then upgrades to ``review_dispatch_dispatched``
        or ``review_dispatch_failed``. Overlapping passes see the pending claim
        and skip until it goes stale. A reviewer that dies without a verdict is
        freed by ``_detect_and_handle_stalled_reviews`` after the stale-claim
        timeout, making the PR re-dispatchable.
        """
        if not self.config.review_dispatch.enabled:
            return CommandResult(
                True,
                "review dispatch disabled",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "launched_count": 0,
                },
            )

        reviews_dir = self._layout.reviews_dir

        # Run the verdict-reaper and orphan/stalled sweeps BEFORE the quota
        # gate so dead reviewers are reaped and stale claims are freed even
        # during throttle periods. Without this ordering, a quota deferral
        # returns early and leaves dead reviewer claims stuck — blocking
        # re-dispatch after the quota resets. In dry-run mode we skip
        # these sweeps to stay read-only.
        verdict_result = {"recorded": [], "missed": []}
        if not self.dry_run:
            verdict_result = self._reap_review_verdicts(reviews_dir)
            _detect_and_handle_stalled_reviews(
                reviews_dir, self.paths.state_file, self.config, self.repo_root
            )
            _reap_completed_review_checkouts(self.repo_root, reviews_dir, self.paths.state_file)
            _reap_orphaned_review_checkouts(
                self.gh, self.repo_root, reviews_dir, self.paths.state_file, self.config
            )

        # Clear the reviewer quota if any verdicts were recorded from dead
        # reviewers. This is the only proof the quota window is actually open:
        # a process that merely *started* can still die seconds later from an
        # asynchronous session-limit kill. Run before the quota gate so a
        # successful reap clears the throttle and lets the pass proceed.
        recorded_verdicts = verdict_result.get("recorded", [])
        missed_verdicts = verdict_result.get("missed", [])
        if not self.dry_run and recorded_verdicts:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                if is_reviewer_quota_exhausted(state):
                    state = clear_reviewer_quota(state)
                    # A recorded verdict is proof the provider let a real
                    # review through -- reset the probe backoff so the next
                    # outage starts from the configured base interval again
                    # instead of carrying forward an exponentially-grown one.
                    state = {
                        **state,
                        "reviewer_quota": {
                            **(state.get("reviewer_quota") or {}),
                            "consecutive_probe_failures": 0,
                        },
                    }
                    save_state(self.paths.state_file, state)

        # System-wide reviewer quota gate. If the quota is exhausted and we are
        # not yet due to probe again, defer without touching any PR state.
        # When the probe window opens, only one reviewer is launched until the
        # probe succeeds, at which point the global quota is cleared.
        quota_alert: dict[str, Any] | None = None
        deferred = False
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            if is_reviewer_quota_exhausted(state):
                if not is_reviewer_probe_ready(state):
                    deferred = True
                    # Quota deferral is by design, but it must never be silent:
                    # an exhaustion episode that outlives its probe window
                    # stalls the review lane fleet-wide. Alert once per
                    # episode (marker cleared with the quota record on probe
                    # success). The digest is emitted after the lock releases —
                    # sinks may do network I/O.
                    quota = state.get("reviewer_quota") or {}
                    if not quota.get("alerted_at") and self.config.notify.enabled:
                        save_state(self.paths.state_file, mark_reviewer_quota_alerted(state))
                        quota_alert = dict(quota)
                    probe_mode = False
                else:
                    probe_mode = True
            else:
                probe_mode = False
        if deferred:
            if quota_alert is not None:
                emit_digest(
                    self._layout.notify,
                    AttentionDigest(
                        generated_at=utc_now(),
                        repo=self.repo_root.name,
                        transitions=(
                            AttentionEntry(
                                issue_number=0,
                                adapter_kind="reviewer",
                                health="REVIEWER_QUOTA_EXHAUSTED",
                                previous_health=None,
                                last_log_line=(
                                    f"throttled_until={quota_alert.get('throttled_until')} "
                                    f"probe_after={quota_alert.get('probe_after')}"
                                ),
                                pid=None,
                                terminal_tool=None,
                                terminal_reason=(
                                    "all reviewer launches deferred until the quota probe succeeds"
                                ),
                            ),
                        ),
                    ),
                )
            # Rescue tier (issue #555): the quota gate above governs Claude-
            # family reviewer launches only. Rescue reviews run on the
            # cross-family (Devin) adapter via _process_rescue_review and do
            # not consume Claude quota, so a Claude quota deferral must not
            # freeze them. Still compute the queue and process the rescue
            # partition here; the deferred result below covers normal
            # candidates only, unchanged from before this fix.
            deferred_queue_result = self.review_queue()
            deferred_candidates = deferred_queue_result.data.get("queue", [])
            _deferred_normal, deferred_rescue_results = self._partition_rescue_candidates(
                deferred_candidates
            )
            return CommandResult(
                True,
                "review dispatch deferred: reviewer quota exhausted, probe not ready",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "launched_count": 0,
                    "deferred_reason": "reviewer_quota_probe_backoff",
                    "recorded_verdicts": recorded_verdicts,
                    "missed_verdicts": missed_verdicts,
                    "rescue_review_results": deferred_rescue_results,
                },
            )

        queue_result = self.review_queue()
        candidates = queue_result.data.get("queue", [])
        if not candidates:
            return CommandResult(
                True,
                "review dispatch: no candidates",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "launched_count": 0,
                    "recorded_verdicts": recorded_verdicts,
                    "missed_verdicts": missed_verdicts,
                },
            )

        # Rescue tier (issue #555): a rescue-marked PR's review must run
        # through the cross-family rescue reviewer (_process_rescue_review),
        # never a normal same-family Claude reviewer -- the exit-to-human-on-
        # request_changes rule only holds if this replaces, rather than
        # precedes, the normal dispatch below. Partitioned out here, before
        # any of the normal claim/quota machinery runs on these PRs. Routing
        # keys on the durable rescue_attempted marker alone (never on
        # self.config.rescue.enabled -- see _partition_rescue_candidates).
        candidates, rescue_review_results = self._partition_rescue_candidates(candidates)
        if not candidates:
            return CommandResult(
                True,
                f"review dispatch: {len(rescue_review_results)} rescue review(s) "
                "processed, no normal candidates",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "launched_count": 0,
                    "recorded_verdicts": recorded_verdicts,
                    "missed_verdicts": missed_verdicts,
                    "rescue_review_results": rescue_review_results,
                },
            )

        # Filter out PRs that are already claimed or still have a live reviewer.
        # Also escalate PRs that have exhausted their dispatch attempt budget.
        max_attempts = self.config.review_dispatch.max_review_dispatch_attempts
        escalated_for_labels: list[tuple[int, int | None]] = []
        escalated_skipped: list[int] = []
        # Issue #586: already-escalated PRs skipped by the gate below are
        # collected here so their human-needed label edge can be re-applied
        # out-of-lock. The edge is applied once at escalation time; if that
        # transition() failed (or the PR was escalated by a path that
        # predated the label edge), the label never lands and the PR sits
        # escalated in state but invisible on GitHub -- permanently excluded
        # from dispatch with no human-visible signal.
        escalated_label_repair: list[tuple[int, int | None]] = []
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Escalate PRs whose dispatch attempt count has reached the cap.
            # These PRs are stuck (every reviewer died without a verdict) and
            # must not be re-dispatched indefinitely. Mark them escalated so a
            # human can intervene, mirroring the rework-cycle escalation pattern.
            changed = False
            for c in candidates:
                pr_key = str(c["pr"])
                pr_state = state["prs"].get(pr_key, {})
                # Escalation gate (issue #575): a PR whose pr-state status is
                # "escalated", or whose linked issue's state status is
                # "escalated", is awaiting a human and must never be
                # re-dispatched or re-escalated -- record_review's escalation
                # guard (~line 6823) refuses the verdict anyway, so dispatching
                # here only burns provider quota on a session whose result is
                # thrown away and then silently lost (the live incident behind
                # this fix: issue #480/PR #540). This must run BEFORE the
                # attempt-cap escalation below too: an issue that gets
                # escalated by an independent path (not this attempt-cap
                # branch) while its PR is already sitting at the cap must not
                # trigger a second, bogus "max_review_dispatch_attempts_exceeded"
                # escalation on top of it. No per-pass event is emitted for the
                # skip itself (see review_dispatch_escalated below for the
                # human-facing signal) because this condition holds every pass
                # while a human has not yet resolved the escalation; emitting a
                # duplicate event each pass would spam the event log.
                issue_num_gate = pr_state.get("issue_number") or c.get("issue")
                issue_state_gate = (
                    state["issues"].get(str(issue_num_gate), {})
                    if issue_num_gate is not None
                    else {}
                )
                if (
                    pr_state.get("status") == "escalated"
                    or issue_state_gate.get("status") == "escalated"
                ):
                    escalated_skipped.append(c["pr"])
                    if issue_num_gate is not None:
                        escalated_label_repair.append((int(c["pr"]), issue_num_gate))
                    continue
                attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
                if pr_state.get(
                    "review_dispatch_status"
                ) == "review_dispatch_dispatched" and _reviewer_pid_alive(pr_state):
                    # Never escalate over a LIVE in-flight reviewer (issue
                    # #573): the cap can be reached by attempts that predate
                    # the current launch (e.g. quota deaths whose rollback
                    # decrement was bypassed), and killing the claim
                    # mid-review orphans the imminent verdict — the reaper
                    # only records verdicts for dispatched claims. Let the
                    # review finish: a recorded verdict resets the counter;
                    # a death is dispositioned by the stalled sweep, after
                    # which the cap escalates honestly on a dead claim.
                    continue
                if attempt_count >= max_attempts and pr_state.get("status") != "escalated":
                    issue_num = pr_state.get("issue_number") or c.get("issue")
                    state["prs"][pr_key] = {
                        **pr_state,
                        "status": "escalated",
                        "review_dispatch_status": "review_dispatch_failed",
                        "review_dispatch_failed_at": utc_now(),
                        "review_dispatch_pending_at": None,
                        "review_dispatched_at": None,
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                    if issue_num is not None:
                        state["issues"][str(issue_num)] = {
                            **state["issues"].get(str(issue_num), {}),
                            "number": issue_num,
                            "status": "escalated",
                            "merge_alert": "OK",
                        }
                    state = append_event(
                        state,
                        "review_dispatch_escalated",
                        {
                            "pr_number": c["pr"],
                            "issue_number": issue_num,
                            "attempt_count": attempt_count,
                            "reason": "max_review_dispatch_attempts_exceeded",
                        },
                        state_path=self.paths.state_file,
                    )
                    changed = True
                    escalated_for_labels.append((int(c["pr"]), issue_num))
            if changed:
                save_state(self.paths.state_file, state)
            # The escalation gate above already filtered out escalated
            # candidates (and recorded them in escalated_skipped) before any
            # attempt-count or claim mutation ran for them; here we only need
            # to exclude those same PR numbers so this second pass over
            # candidates doesn't re-select them via _is_review_dispatchable.
            escalated_skipped_set = set(escalated_skipped)
            dispatchable = [
                c
                for c in candidates
                if c["pr"] not in escalated_skipped_set
                and _is_review_dispatchable(state, c["pr"], c, max_attempts=max_attempts)
            ]

        # Apply the human-needed label edge for each fresh escalation, outside
        # the state lock (transition() makes GitHub API calls). This was the
        # one escalation call site that skipped the label edge entirely,
        # leaving PRs escalated in state.json but invisible on GitHub
        # (pr-lifecycle.md: PRs 548/540/531 live escalated-without-label).
        # Mirrors the dead-rework-session sibling: label_error is recorded on
        # the issue entry when the transition does not fully apply. On success
        # label_error is explicitly set to None so the self-heal loop below
        # can distinguish "edge applied and verified" from "edge never
        # attempted" (absent key, e.g. a pre-#556 escalation).
        # Gated by dry_run: --dry-run must not perform live GitHub label
        # mutations or state.json writes (review finding on PR #670).
        if not self.dry_run:
            escalated_label_outcomes: list[tuple[int, dict[str, Any] | None]] = []
            for _pr_num, issue_num in escalated_for_labels:
                if issue_num is None:
                    continue
                result = transition(self.gh, self.config.labels, int(issue_num), "escalated")
                if result.outcome != TransitionOutcome.APPLIED:
                    escalated_label_outcomes.append(
                        (
                            int(issue_num),
                            {
                                "edge": "escalated",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            },
                        )
                    )
                else:
                    escalated_label_outcomes.append((int(issue_num), None))
            if escalated_label_outcomes:
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    for issue_num, label_error in escalated_label_outcomes:
                        entry = state["issues"].get(str(issue_num), {})
                        state["issues"][str(issue_num)] = {
                            **(entry if isinstance(entry, dict) else {}),
                            "number": issue_num,
                            "label_error": label_error,
                        }
                    save_state(self.paths.state_file, state)

        # Issue #586: self-heal the escalated label edge for PRs that were
        # already escalated (skipped by the gate above) but whose
        # human-needed label never landed. The edge is applied once at
        # escalation time; if transition() failed then (label_error is a
        # dict) or the PR was escalated by a path that predated the label
        # edge (label_error key absent), the PR sits escalated in state but
        # invisible on GitHub -- permanently excluded from review dispatch
        # with no human-visible signal that operator action is required.
        # Re-apply the edge here every pass until it sticks, then mark it
        # verified (label_error = None) so steady-state passes skip the
        # GitHub label fetch. This is the single-point enforcement the issue
        # asks for: state status == "escalated" and the agent:human-needed
        # label cannot stay in disagreement past one dispatch_reviews pass.
        # Gated by dry_run: --dry-run must not perform live GitHub label
        # mutations or state.json writes (review finding on PR #670).
        if not self.dry_run:
            escalated_repair_outcomes: list[tuple[int, dict[str, Any] | None]] = []
            import logging

            for pr_num, issue_num in escalated_label_repair:
                if issue_num is None:
                    continue
                with state_lock(self.paths.state_file):
                    repair_state = load_state(self.paths.state_file)
                    repair_entry = repair_state.get("issues", {}).get(str(issue_num), {})
                    repair_pr = repair_state.get("prs", {}).get(str(pr_num), {})
                # Race guard (review finding on PR #670): a concurrent
                # unescalate() may have freed this issue between the
                # escalation gate above and this per-item read. Re-check
                # both the PR's and the issue's current status -- if neither
                # is still "escalated", the issue is no longer awaiting a
                # human and re-applying the agent:human-needed label would
                # silently undo the unescalate (label_error is also cleared
                # by unescalate, so the absent-key branch below would
                # otherwise re-escalate without a status check).
                if (
                    repair_pr.get("status") != "escalated"
                    and repair_entry.get("status") != "escalated"
                ):
                    continue
                # label_error is None  -> verified OK on a prior pass, skip the
                #   GitHub fetch entirely (steady-state cost: zero).
                # label_error is a dict -> prior transition() failed, retry.
                # label_error key absent -> edge never attempted (pre-#556
                #   escalation or a call site that doesn't record label_error),
                #   fetch live labels to decide whether repair is needed.
                if "label_error" in repair_entry and repair_entry["label_error"] is None:
                    continue
                # issue_view/transition make GitHub API calls that can raise on
                # transient errors; a failure to verify one escalated issue must
                # not abort the entire dispatch_reviews pass. Skip the issue this
                # pass and retry on the next -- the escalated status is already
                # durable in state, so no ground truth is lost by deferring.
                try:
                    issue_view = self.gh.issue_view(int(issue_num))
                    if self.config.labels.human_needed in label_names(issue_view):
                        escalated_repair_outcomes.append((int(issue_num), None))
                        continue
                    result = transition(self.gh, self.config.labels, int(issue_num), "escalated")
                except Exception:
                    logging.getLogger(__name__).warning(
                        "escalated label repair for issue %s deferred (GitHub "
                        "fetch failed); will retry next pass",
                        issue_num,
                        exc_info=True,
                    )
                    continue
                if result.outcome == TransitionOutcome.APPLIED:
                    escalated_repair_outcomes.append((int(issue_num), None))
                else:
                    escalated_repair_outcomes.append(
                        (
                            int(issue_num),
                            {
                                "edge": "escalated",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            },
                        )
                    )
            if escalated_repair_outcomes:
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    for issue_num, label_error in escalated_repair_outcomes:
                        entry = state["issues"].get(str(issue_num), {})
                        state["issues"][str(issue_num)] = {
                            **(entry if isinstance(entry, dict) else {}),
                            "number": issue_num,
                            "label_error": label_error,
                        }
                    state = append_event(
                        state,
                        "escalated_label_repaired",
                        {
                            "issue_numbers": [i for i, _ in escalated_repair_outcomes],
                            "failures": [i for i, e in escalated_repair_outcomes if e is not None],
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)

        # Apply the local and provider-token caps. 0 means unlimited for both.
        max_local = self.config.review_dispatch.max_local_review_processes
        max_concurrent = self.config.review_dispatch.max_concurrent_reviews
        live_count = _count_live_reviews(reviews_dir, self.paths.state_file)
        requested_limit = limit if limit is not None else len(dispatchable)
        local_cap = _apply_local_review_cap(requested_limit, max_local, live_count)
        if max_concurrent > 0:
            concurrent_available = max(0, max_concurrent - live_count)
            concurrent_cap = min(local_cap.dispatch_limit, concurrent_available)
        else:
            concurrent_cap = local_cap.dispatch_limit
        # In probe mode, only launch one reviewer at a time to test quota.
        dispatch_limit = 1 if probe_mode else concurrent_cap
        selected = dispatchable[:dispatch_limit]

        if self.dry_run:
            return CommandResult(
                True,
                f"dry-run: would dispatch {len(selected)} reviewer(s)",
                {
                    "selected_count": len(selected),
                    "attempted_count": len(selected),
                    "failed_count": 0,
                    "launched_count": 0,
                    "deferred_count": len(candidates) - len(selected),
                    "escalated_skipped": escalated_skipped,
                    **local_cap.report_fields(),
                },
            )

        # Claim the selected PRs as pending before launching. This is the only
        # place that writes review_dispatch_pending; the upgrade happens after
        # each launch so a crash between claim and upgrade is recoverable via
        # the stale-claim timeout.
        now = utc_now()
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            review_effort_assignments: list[dict[str, Any]] = []
            # Single source of truth for the review_effort experiment arm:
            # resolved ONCE here at claim time and threaded through to the
            # launch call below via resolved_review_effort, so the launch
            # uses exactly this value instead of re-deriving it (agreement
            # by construction, not by convention).
            resolved_review_efforts: dict[int, str] = {}
            for candidate in selected:
                pr_number = candidate["pr"]
                pr_state = state["prs"].get(str(pr_number), {})
                attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
                review_effort_used, review_effort_arm = resolve_review_effort(
                    pr_number, self.config.review_dispatch, self.config.claude_code
                )
                resolved_review_efforts[pr_number] = review_effort_used
                state["prs"][str(pr_number)] = {
                    **pr_state,
                    "number": pr_number,
                    "issue_number": candidate["issue"],
                    "review_dispatch_status": "review_dispatch_pending",
                    "review_dispatch_pending_at": now,
                    "review_dispatched_at": None,
                    "review_dispatch_failed_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                    "review_dispatch_attempt_count": attempt_count + 1,
                    "review_turn_limit_summary_posted": False,
                    "review_miss_summary_posted": False,
                    "review_effort_arm": review_effort_arm,
                    "review_effort_used": review_effort_used,
                }
                review_effort_assignments.append(
                    {
                        "pr_number": pr_number,
                        "review_effort_arm": review_effort_arm,
                        "review_effort_used": review_effort_used,
                    }
                )
            if selected:
                state = append_event(
                    state,
                    "review_dispatch_claim",
                    {
                        "pr_numbers": [c["pr"] for c in selected],
                        "count": len(selected),
                        "review_effort_assignments": review_effort_assignments,
                    },
                    state_path=self.paths.state_file,
                )
            save_state(self.paths.state_file, state)

        # Launch reviewers concurrently: each launch is non-blocking (subprocess.Popen),
        # so this loop quickly spawns all selected processes. Worktree creation
        # is synchronous per-PR but independent across PRs.
        launched: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        quota_hit = False
        # Captured at the moment a launch-time quota hit is detected so the
        # provider's named reset time (issue #612) can be parsed from it
        # after the loop and used for the fleet-wide backoff target.
        quota_hit_error: str | None = None
        for candidate in selected:
            pr_number = candidate["pr"]
            issue_number = candidate["issue"]
            pr_dir = self.paths.prs / f"pr-{pr_number}"
            prompt_path = pr_dir / "review-prompt.md"
            try:
                if not prompt_path.exists():
                    failed.append(
                        {
                            "pr": pr_number,
                            "error": "review-prompt.md not found",
                        }
                    )
                    continue

                pr = self.gh.pr_view(pr_number)
                if not pr:
                    failed.append({"pr": pr_number, "error": f"PR #{pr_number} not found"})
                    continue

                branch = str(pr.get("headRefName", ""))
                if not branch:
                    failed.append({"pr": pr_number, "error": "PR headRefName missing"})
                    continue

                head_sha = pr.get("headRefOid")
                if not head_sha:
                    failed.append({"pr": pr_number, "error": "PR headRefOid missing"})
                    continue

                # Cross-repo PRs are never linked for lifecycle purposes by
                # linked_issue_number, so the only way we see one here is an
                # unexpected state; strip an owner prefix defensively.
                if ":" in branch:
                    branch = branch.split(":", 1)[1]

                prompt_text = prompt_path.read_text(encoding="utf-8")
                claude_cfg = self.config.claude_code
                # Reviewers never use worktrees_dir/venv_source — those key
                # the worker's branch-slug worktree, which create_review_checkout
                # (routed to via review=True + head_sha) never touches. Only
                # repo_root/sessions_dir/env/materialize_dirs/review/head_sha
                # are meaningful for a reviewer launch. `claude_cfg.command`
                # is deliberately NOT forwarded here: it is a worker-tuning
                # field, and launch_claude_worker(review=True, ...) hard-pins
                # the read-only command template regardless of what is passed
                # for command_template, so passing it through would be
                # misleading dead code (PR #397 round-2 review).
                #
                # `config` IS forwarded (unlike the above): launch_claude_worker
                # falls back to a bare default OrchestratorConfig() whenever
                # config is omitted, which was silently discarding every
                # review-only pin (review_effort, review_max_turns, and the
                # review_effort experiment) at the one real dispatch_reviews()
                # launch site — the effort/max-turns pins only ever worked in
                # direct launch_claude_worker(config=...) unit tests, never in
                # an actual dispatch pass. adapters.py's worker-dispatch path
                # (_run_claude_code_adapter) already forwards config the same
                # way.
                launch_kwargs: dict[str, Any] = {
                    "repo_root": self.repo_root,
                    "sessions_dir": reviews_dir,
                    "config": self.config,
                    "env": claude_cfg.worker_env,
                    "materialize_dirs": self.config.dispatch.materialize_dirs,
                    "review": True,
                    "head_sha": head_sha,
                    # Force-enabled for reviewers: the structured events.jsonl
                    # is needed for verdict fallback parsing (issue #540) and
                    # token/turn monitoring. Unlike workers, reviewers always
                    # benefit from the structured output because their verdict
                    # extraction depends on it.
                    "tee_stream_json": True,
                    # The review_effort experiment arm was already resolved
                    # once, above, at claim time (and persisted to state/
                    # telemetry) -- pass it through so launch_claude_worker
                    # uses this value directly instead of re-resolving it.
                    "resolved_review_effort": resolved_review_efforts.get(pr_number),
                }

                record = launch_claude_worker(
                    issue_number=pr_number,
                    branch=branch,
                    prompt_text=prompt_text,
                    **launch_kwargs,
                )
                if record.error or record.pid is None:
                    error_text = record.error or "launch returned no pid"
                    # A quota failure is a global condition, not a per-PR
                    # failure. Stop the pass immediately so the next probe can
                    # retry once the usage window resets.
                    if (
                        record.error
                        and match_throttle_tail(
                            record.error,
                            self.config.runtime.throttle_error_markers,
                        )[0]
                    ):
                        quota_hit = True
                        quota_hit_error = record.error
                        break
                    failed.append({"pr": pr_number, "error": error_text})
                else:
                    launched.append(
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "pid": record.pid,
                            "process_start_time": record.process_start_time,
                        }
                    )
            except (OSError, GitHubError, ValueError) as exc:
                failed.append({"pr": pr_number, "error": f"{type(exc).__name__}: {exc}"})

        # Upgrade claims outside the launch loop. Successful launches become
        # review_dispatch_dispatched. A quota failure rolls back the claim so
        # the PR is not wedged by a global condition; it remains dispatchable
        # once quota is available. Non-quota failures become
        # review_dispatch_failed.
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            launched_prs = {x["pr"] for x in launched}
            failed_prs = {x["pr"] for x in failed}
            for candidate in selected:
                pr_number = candidate["pr"]
                issue_number = candidate["issue"]
                pr_state = state["prs"].get(str(pr_number), {})
                if pr_number in launched_prs:
                    launch_info = next(x for x in launched if x["pr"] == pr_number)
                    state["prs"][str(pr_number)] = {
                        **pr_state,
                        "number": pr_number,
                        "issue_number": issue_number,
                        "review_dispatch_status": "review_dispatch_dispatched",
                        "review_dispatched_at": utc_now(),
                        "review_dispatch_pending_at": None,
                        "review_dispatch_failed_at": None,
                        # A successful launch supersedes any earlier failure:
                        # without this reset the last error string is carried
                        # forward verbatim by the **pr_state spread forever.
                        "review_dispatch_error": None,
                        "reviewer_pid": launch_info["pid"],
                        "reviewer_process_start_time": launch_info["process_start_time"],
                    }
                elif pr_number in failed_prs:
                    fail_info = next(x for x in failed if x["pr"] == pr_number)
                    failed_state = {
                        **pr_state,
                        "number": pr_number,
                        "issue_number": issue_number,
                        "review_dispatch_status": "review_dispatch_failed",
                        "review_dispatch_failed_at": utc_now(),
                        "review_dispatch_pending_at": None,
                        "review_dispatched_at": None,
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                    failed_state["review_dispatch_error"] = fail_info["error"]
                    state["prs"][str(pr_number)] = failed_state
                else:
                    # Quota failure (or not reached due to break) — roll back
                    # the pending claim so the PR stays dispatchable. Also
                    # decrement the attempt counter: no reviewer actually ran,
                    # so this global condition must not consume the per-PR
                    # dispatch attempt budget (3 quota hits would otherwise
                    # escalate a PR that was never reviewed).
                    rolled_back = without_review_dispatch_claim(pr_state)
                    attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
                    if attempt_count > 0:
                        rolled_back["review_dispatch_attempt_count"] = attempt_count - 1
                    state["prs"][str(pr_number)] = rolled_back

            if quota_hit:
                now_dt = datetime.now(UTC)
                # Issue #612: parse the provider's named reset clock time
                # from the launch error (e.g. "resets 1:20am
                # (America/Los_Angeles)") so the backoff targets the stated
                # reset instead of a fixed quota_reset_hours guess.
                reset_at = (
                    parse_reset_clock_time(quota_hit_error, now_dt) if quota_hit_error else None
                )
                state, quota_record = _set_reviewer_quota_exhausted_with_backoff(
                    state, self.config, now_dt, reset_at=reset_at
                )
                # Distinct, queryable event for a launch-time quota hit
                # (issue #612): mirrors the stalled-sweep event so a quota
                # exhaustion is diagnosable from either detection path.
                state = append_event(
                    state,
                    "review_quota_exhausted",
                    {
                        "throttled_until": quota_record.get("throttled_until"),
                        "probe_after": quota_record.get("probe_after"),
                        "reset_at": quota_record.get("reset_at"),
                        "consecutive_probe_failures": quota_record.get(
                            "consecutive_probe_failures"
                        ),
                        "source": "launch_quota_hit",
                    },
                    state_path=self.paths.state_file,
                )

            state = append_event(
                state,
                "review_dispatch",
                {
                    "launched": [x["pr"] for x in launched],
                    "failed": [x["pr"] for x in failed],
                    "quota_hit": quota_hit,
                },
                state_path=self.paths.state_file,
            )
            save_state(self.paths.state_file, state)

        ok = not failed and not quota_hit

        message = f"review dispatch: {len(launched)} launched, {len(failed)} failed"
        if recorded_verdicts or missed_verdicts:
            message += (
                f"; {len(recorded_verdicts)} verdict(s) recorded, {len(missed_verdicts)} missed"
            )
        if quota_hit:
            message = (
                f"review dispatch: reviewer quota hit after {len(launched)} "
                f"launched; will probe again later"
            )
        elif failed:
            message = (
                f"review dispatch completed with {len(failed)} failure(s): "
                f"{len(launched)} launched"
            )

        data: dict[str, Any] = {
            "selected_count": len(selected),
            "attempted_count": len(selected),
            "launched_count": len(launched),
            "failed_count": len(failed),
            "failed": failed,
            "quota_hit": quota_hit,
            "probe_mode": probe_mode,
            "skipped_count": len(dispatchable) - len(selected),
            "deferred_count": len(candidates) - len(dispatchable),
            "escalated_skipped": escalated_skipped,
            "recorded_verdicts": recorded_verdicts,
            "missed_verdicts": missed_verdicts,
            "rescue_review_results": rescue_review_results,
        }
        data.update(local_cap.report_fields())
        return CommandResult(ok, message, data)

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
        rescue_snapshot = load_state_locked(self.paths.state_file)
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

    def _process_rescue_review(self, candidate: dict[str, Any]) -> CommandResult:
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
        pr = self.gh.pr_view(pr_number)
        head_sha = str(pr.get("headRefOid") or "")
        branch = str(pr.get("headRefName") or "")
        pr_state = load_state_locked(self.paths.state_file).get("prs", {}).get(str(pr_number), {})
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
        cf_result = run_cross_family_review(
            model=cfg.reviewer_model,
            command=cfg.reviewer_command or self.config.cross_family.command,
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
            verdict = _extract_verdict_from_text(report_text)

        if verdict is not None and verdict["decision"] == "approved":
            result = self.record_review(
                pr_number,
                "approved",
                summary=verdict["summary"],
                reviewed_head=head_sha,
                required_changes=verdict["required_changes"],
            )
            return CommandResult(
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
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                "status": "escalated",
            }
            if issue_number is not None:
                state["issues"][str(issue_number)] = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "escalated",
                    "merge_alert": "OK",
                }
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
            save_state(self.paths.state_file, state)
        if issue_number is not None:
            result = transition(self.gh, self.config.labels, int(issue_number), "escalated")
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": "escalated",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
        try:
            self._comment_pr(pr_number, comment_body)
        except GitHubError as exc:
            label_error = {**(label_error or {}), "comment_error": str(exc)}
        return CommandResult(
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

    def record_review(
        self,
        pr_number: int,
        decision: str,
        summary: str = "",
        summary_file: Path | None = None,
        comment: bool = False,
        reviewed_head: str | None = None,
        required_changes: Sequence[str] | None = None,
        session_metrics: dict[str, Any] | None = None,
    ) -> CommandResult:
        if decision not in {"approved", "request_changes", "blocked"}:
            return CommandResult(
                False, "decision must be approved, request_changes, or blocked", {}
            )
        summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
        # Issue #11: reject empty summary for request_changes/blocked decisions
        # before any state/label mutation
        if decision in {"request_changes", "blocked"} and not summary_text.strip():
            return CommandResult(
                False,
                f"--summary or --summary-file is required for decision '{decision}'",
                {},
            )
        pr = self.gh.pr_view(pr_number)
        issue_number = (
            linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if pr
            else None
        )

        # Escalation is terminal for verdict recording too, mirroring review()'s
        # guard: without this, a late-arriving verdict (a reviewer that finished
        # after the attempt-cap escalation fired, or a stale reap) silently
        # overwrites status="escalated" and re-enters the PR into the pipeline,
        # which is exactly how escalated PRs were observed re-escalating 2-3x
        # (pr-lifecycle.md: non-durable escalation). A human re-arms the PR with
        # `charlie unescalate`, after which verdicts record normally again.
        guard_state = load_state_locked(self.paths.state_file)
        guard_pr_state = guard_state.get("prs", {}).get(str(pr_number), {})
        guard_issue_state = (
            guard_state.get("issues", {}).get(str(issue_number), {})
            if issue_number is not None
            else {}
        )
        if (
            guard_pr_state.get("status") == "escalated"
            or guard_issue_state.get("status") == "escalated"
        ):
            return CommandResult(
                False,
                f"PR #{pr_number} is escalated; verdict not recorded "
                f"(run `charlie unescalate --pr {pr_number}` to re-arm it first)",
                {"pr": pr_number, "issue": issue_number, "escalated": True},
            )

        pr_dir = self.paths.prs / f"pr-{pr_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        # reviewed_head_sha/reviewed_patch_id must reflect the packet the reviewer
        # actually read (review()'s pr.json/diff.patch), not a fresh fetch made
        # here at verdict time: a commit landing between packet generation and
        # verdict recording would otherwise silently reattribute the decision to
        # a head/diff that was never reviewed. Fall back to a live fetch only
        # when no packet exists (e.g. a decision recorded without a prior
        # review() call).
        packet_head_sha = self._read_packet_head_oid(pr_number)
        live_head_sha = pr.get("headRefOid") if pr else None

        # Issue #467: do not silently pin a verdict to a stale packet when the PR
        # head has advanced since the packet was generated. If the packet head
        # disagrees with the live PR head, require an explicit --reviewed-head
        # choice. When they agree (or no packet exists), preserve the existing
        # packet-first / live-fallback semantics and record where the SHA came from.
        if reviewed_head is not None:
            if packet_head_sha is not None and reviewed_head == packet_head_sha:
                reviewed_head_sha = reviewed_head
                reviewed_head_source = "packet"
            elif live_head_sha is not None and reviewed_head == live_head_sha:
                reviewed_head_sha = reviewed_head
                reviewed_head_source = "live"
            else:
                options: list[str] = []
                if packet_head_sha is not None:
                    options.append(f"packet head {packet_head_sha}")
                if live_head_sha is not None:
                    options.append(f"live head {live_head_sha}")
                options_str = " or ".join(options) if options else "any available head"
                return CommandResult(
                    False,
                    f"--reviewed-head {reviewed_head} does not match {options_str}",
                    {},
                )
        elif (
            packet_head_sha is not None
            and live_head_sha is not None
            and packet_head_sha != live_head_sha
        ):
            return CommandResult(
                False,
                f"review packet head ({packet_head_sha}) differs from live PR head ({live_head_sha}); "
                "use --reviewed-head to choose the head the verdict applies to",
                {},
            )
        elif packet_head_sha is not None:
            reviewed_head_sha = packet_head_sha
            reviewed_head_source = "packet"
        elif live_head_sha is not None:
            reviewed_head_sha = live_head_sha
            reviewed_head_source = "live"
        else:
            return CommandResult(False, "no packet or live PR head available", {})

        # Calculate patch-id for the PR diff to detect actual content changes
        # (issue #222: base-update merges can advance head SHA without changing diff content).
        # All terminal decisions (approved/request_changes/blocked) persist reviewed_patch_id
        # so the review-queue enumerator can carry them forward on content-identical heads.
        # Also record the tier-2 content signature (issue #414): patch-id is
        # unstable across every main advance (the merge-base moves), so the
        # ordered +/- line stream and changed-file set are persisted here,
        # at the moment the diff is freshly known, so a later carry-forward
        # check never needs to reconstruct this historical diff.
        reviewed_patch_id = ""
        reviewed_signature = DiffContentSignature((), frozenset())
        diff: str | None = None
        if pr:
            if reviewed_head_source == "packet":
                diff = self._read_packet_diff(pr_number)
                if diff is None:
                    diff = self.gh.pr_diff(pr_number)
            else:
                diff = self.gh.pr_diff(pr_number)
            reviewed_patch_id = _calculate_patch_id(diff)
            if diff:
                reviewed_signature = _diff_content_signature(diff)
        decision_payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "summary": summary_text,
            "required_changes": list(required_changes) if required_changes is not None else [],
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_head_source": reviewed_head_source,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
            "carried_forward_from": [],
            "reviewed_at": utc_now(),
        }
        decision_path = pr_dir / "review-decision.json"
        # Merge-update (never in-place assignment) and persist BEFORE any GitHub
        # label mutation: a label-write failure or crash must not desync the
        # durable decision/counter from what actually happened.
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            rework_path: str | None = None
            escalated = False
            rescue_dispatched = False
            # Durable per-PR rework counter — NOT derived from the global events
            # log, which append_event truncates to the last 200 entries: on a busy
            # repo that eviction silently reset the count and defeated the cap
            # (a PR could rework forever instead of escalating to a human).
            request_changes_count = int(pr_state.get("request_changes_count", 0))
            if decision == "request_changes":
                # Rework cap: past max_rework_cycles the evidence says iteration
                # thrashes (wrong brief or unimplementable criteria) — escalate to
                # a human instead of dispatching another cycle.
                escalated = request_changes_count >= self.config.review.max_rework_cycles
                # Rescue tier (issue #555): a cap exceedance here is one of the
                # three verdict-driven ("cheap model wasn't good enough")
                # causes the rescue tier is gated on. If enabled and this PR
                # has not already spent its one rescue attempt, route to a
                # bounded Opus rework instead of escalating — never a second
                # rescue for the same PR (rescue_attempted is durable, cleared
                # only by `charlie unescalate`).
                if (
                    escalated
                    and self.config.rescue.enabled
                    and not pr_state.get("rescue_attempted")
                ):
                    escalated = False
                    rescue_dispatched = True
                # Only count a rework cycle when the PR head has actually advanced.
                # If the head is unchanged, the prior cycle's attempt was never
                # delivered (e.g., worker died orphaned), so re-issuing request_changes
                # should not consume the escalation budget. See issue #208.
                head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
                if not escalated and head_advanced:
                    request_changes_count += 1
                if not escalated:
                    rework_summary = (
                        rescue_helpers.build_rescue_rework_summary(
                            "rework_cycle_cap", summary_text
                        )
                        if rescue_dispatched
                        else summary_text
                    )
            decision_payload["escalated"] = escalated
            # Persist the verdict BEFORE rendering the rework brief: the brief
            # reads review-decision.json itself (issue #632, single point of
            # enforcement) to surface required_changes, so the decision file
            # must be on disk first. A label-write failure or crash after this
            # point leaves a durable verdict and a brief consistent with it.
            self._write_json(decision_path, decision_payload)
            if decision == "request_changes" and not escalated:
                rework_path = str(self._write_rework_prompt(pr, issue_number, rework_summary))
            rescue_fields = (
                rescue_helpers.build_rescue_dataclass_kwargs("rework_cycle_cap")
                if rescue_dispatched
                else {}
            )
            if rescue_dispatched:
                rescue_fields["rescue_dispatched_at"] = utc_now()
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "issue_number": issue_number,
                "decision": decision,
                "decision_path": str(decision_path),
                "reviewed_head_sha": reviewed_head_sha,
                "reviewed_patch_id": reviewed_patch_id,
                "carried_forward_from": [],
                "request_changes_count": request_changes_count,
                "status": "escalated" if escalated else decision,
                "consecutive_failed_merge_attempts": 0,
                **rescue_fields,
                # The reviewer agent has recorded its verdict. The hub no longer
                # needs to treat this PR as having an in-flight reviewer; the next
                # stale-head review-queue entry will re-dispatch cleanly.
                "review_dispatch_status": "review_dispatch_completed",
                "reviewer_pid": None,
                "reviewer_process_start_time": None,
                # Reset the dispatch attempt counter: a verdict was produced, so
                # the PR is not stuck. If the head later advances and triggers a
                # new review cycle, the counter starts fresh.
                "review_dispatch_attempt_count": 0,
                # Reviewer session token/cost telemetry (best-effort): merge-update
                # so a call without metrics (e.g. a manual `charlie verdict`)
                # never clobbers metrics recorded by an earlier automated reap.
                "review_session_metrics": (
                    session_metrics
                    if session_metrics is not None
                    else pr_state.get("review_session_metrics")
                ),
            }
            # Update the linked issue's status to reconcile out of rework_requested:
            # the previous worker session is definitionally finished, so the issue
            # status must reflect the actual decision. This prevents state-driven
            # dispatch_rework from selecting approved/blocked PRs for duplicate work.
            if issue_number is not None:
                if decision == "request_changes":
                    if not escalated:
                        state["issues"][str(issue_number)] = {
                            **state["issues"].get(str(issue_number), {}),
                            "number": issue_number,
                            "status": "rework_requested",
                            "merge_alert": "OK",
                        }
                    else:
                        # Clear rework_requested status when escalated to prevent selection
                        state["issues"][str(issue_number)] = {
                            **state["issues"].get(str(issue_number), {}),
                            "number": issue_number,
                            "status": "escalated",
                            "merge_alert": "OK",
                        }
                elif decision == "approved":
                    state["issues"][str(issue_number)] = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "status": "approved",
                        "merge_alert": "OK",
                    }
                    # Clear worker PID when issue is approved (worker is done)
                    state["issues"][str(issue_number)].pop("worker_pid", None)
                    state["issues"][str(issue_number)].pop("worker_process_start_time", None)
                elif decision == "blocked":
                    state["issues"][str(issue_number)] = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "status": "blocked",
                        "merge_alert": "OK",
                    }
                    # Clear worker PID when issue is blocked (worker is done)
                    state["issues"][str(issue_number)].pop("worker_pid", None)
                    state["issues"][str(issue_number)].pop("worker_process_start_time", None)
            event_payload: dict[str, Any] = {
                "pr_number": pr_number,
                "decision": decision,
                "escalated": escalated,
            }
            if session_metrics is not None:
                event_payload["session_metrics"] = session_metrics
            state = self._record_event(state, "record_review", event_payload)
            if rescue_dispatched:
                state = self._record_event(
                    state,
                    "rescue_dispatched",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "cause": "rework_cycle_cap",
                    },
                )
            save_state(self.paths.state_file, state)
        # GitHub label side effects are best-effort and isolated: the durable
        # decision above is the authority; a label failure is reported, not fatal.
        label_error: dict[str, Any] | None = None
        if issue_number is not None:
            if decision == "request_changes":
                target = "escalated" if escalated else "rework_requested"
                result = transition(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    target,
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": target,
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
            elif decision == "blocked":
                result = transition(self.gh, self.config.labels, issue_number, "blocked")
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": "blocked",
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
            elif decision == "approved":
                result = transition(self.gh, self.config.labels, issue_number, "review_approved")
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": "review_approved",
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
        if decision == "request_changes" and comment and summary_text:
            try:
                self._comment_pr(pr_number, summary_text)
            except GitHubError as exc:
                # Comment failure is separate from label transition failure
                if label_error is None:
                    label_error = {"comment_error": str(exc)}
                else:
                    label_error["comment_error"] = str(exc)
        source_note = f"head from {reviewed_head_source}"
        if rescue_dispatched:
            message = (
                f"review recorded — rework cap ({self.config.review.max_rework_cycles}) "
                f"reached, rescue tier dispatched instead of escalating ({source_note})"
            )
        elif escalated:
            message = (
                f"review recorded — rework cap ({self.config.review.max_rework_cycles}) reached, "
                f"escalated to human ({source_note})"
            )
        else:
            message = f"review recorded ({source_note})"
        if label_error:
            message += f" (label update failed: {label_error.get('outcome', label_error)})"
        return CommandResult(
            True,
            message,
            {
                "pr": pr_number,
                "decision": decision,
                "decision_path": str(decision_path),
                "reviewed_head_sha": reviewed_head_sha,
                "reviewed_head_source": reviewed_head_source,
                "live_head_sha": live_head_sha,
                "packet_head_sha": packet_head_sha,
                "rework_path": rework_path,
                "escalated": escalated,
                "rescue_dispatched": rescue_dispatched,
                "request_changes_count": request_changes_count,
                "label_error": label_error,
            },
        )

    # PR-record bookkeeping that must not survive an operator re-arm: attempt
    # counters and caches that would otherwise instantly re-escalate the PR
    # (counters at cap) or feed the pipeline frozen pre-escalation data
    # (janitor/CI caches — pr-lifecycle.md: escalated PRs freeze their cached
    # janitor state forever, e.g. #548 showing "Tests pending" 12h after the
    # checks passed).
    _UNESCALATE_PR_RESET_FIELDS = (
        "review_dispatch_attempt_count",
        "request_changes_count",
        "conflict_rework_attempts",
        "conflict_rework_attempts_last_head",
        "no_op_rework_attempts",
        "no_op_rework_attempts_last_head",
        "review_dispatch_status",
        "review_dispatch_failed_at",
        "review_dispatch_pending_at",
        "review_dispatched_at",
        "reviewer_pid",
        "reviewer_process_start_time",
        "review_turn_limit_summary_posted",
        "review_miss_summary_posted",
        "janitor_ok",
        "janitor_failures",
        "janitor_warnings",
        "escalation_reason",
        "label_error",
        # Rescue tier (issue #555): rescue_attempted is the durable "used my
        # one shot" marker. Only charlie unescalate clears it (this tuple) —
        # every other code path treats a present marker as permanent.
        "rescue_attempted",
        "rescue_cause",
        "rescue_dispatched_at",
    )
    # Issue-record equivalents (dispatch-side caps and stale worker bookkeeping).
    _UNESCALATE_ISSUE_RESET_FIELDS = (
        "dispatch_failed_at",
        "redispatch_at",
        "escalation_reason",
        "label_error",
        "worker_pid",
        "worker_process_start_time",
        "dispatched_at",
    )

    def unescalate(
        self,
        pr_number: int | None = None,
        issue_number: int | None = None,
        *,
        dry_run: bool = False,
    ) -> CommandResult:
        """Operator re-arm for an escalated (or janitor-blocked) PR/issue.

        Escalation is deliberately terminal for every automated path (review()
        and record_review() both hard-stop on it); until this command existed
        the only recovery was hand-editing state.json and labels, which is
        exactly how the status/label desyncs this repair sweep keeps finding
        were produced. This is the sanctioned door back into the pipeline:

        - PR merged/closed on GitHub: normalize the record to that terminal
          state (finalization/reconcile handle the rest); no label changes.
        - PR open: reset status to the passive pr-open state, zero every
          attempt counter and frozen janitor/review cache, and apply the
          ``unescalated_pr_open`` label edge so the next pass re-reviews it
          from scratch.
        - Issue with no live PR: drop the issue back to the never-dispatched
          baseline and strip workflow labels (``unescalated_requeued``) so
          dispatch treats it as fresh.

        Idempotent: a record that is not escalated/janitor_blocked is a no-op
        (ok=True). ``dry_run`` computes and reports the full transition map
        without touching state, labels, or events.
        """
        if pr_number is None and issue_number is None:
            return CommandResult(False, "unescalate requires --pr and/or --issue", {})

        state = load_state_locked(self.paths.state_file)

        # Resolve the PR/issue pair from whichever side was given.
        if pr_number is None and issue_number is not None:
            open_pr_numbers = sorted(
                int(k)
                for k, v in state.get("prs", {}).items()
                if isinstance(v, dict)
                and v.get("issue_number") == issue_number
                and v.get("status") not in ("merged", "closed")
                and k.isdigit()
            )
            if open_pr_numbers:
                pr_number = open_pr_numbers[0]
        pr_state = state.get("prs", {}).get(str(pr_number), {}) if pr_number is not None else {}
        if issue_number is None and pr_number is not None:
            issue_number = pr_state.get("issue_number")
        issue_state = (
            state.get("issues", {}).get(str(issue_number), {}) if issue_number is not None else {}
        )

        pr_stuck = pr_state.get("status") in ("escalated", "janitor_blocked")
        issue_stuck = issue_state.get("status") == "escalated"
        if not pr_stuck and not issue_stuck:
            return CommandResult(
                True,
                f"nothing to unescalate (pr={pr_number} status="
                f"{pr_state.get('status')!r}, issue={issue_number} status="
                f"{issue_state.get('status')!r})",
                {"pr": pr_number, "issue": issue_number, "changed": False},
            )

        # Ground truth decides the re-entry point.
        live_pr = self.gh.pr_view(pr_number) if pr_number is not None else {}
        live_pr_state = str((live_pr or {}).get("state") or "").upper()

        # Issue #214 precedent (reconcile's live_session_issue_numbers guard):
        # a verifiably live worker session means nothing here is stuck, it is
        # IN USE -- refuse entirely rather than re-arm around a running
        # process. Popping issue-side worker_pid/dispatched_at would blind
        # orphan-worker detection; resetting the PR side would zero the
        # conflict/no-op attempt counters for a rework cycle still in flight
        # (defeating the caps) and flip the PR to the passive reviewing
        # status, inviting a concurrent review() against the worker's
        # in-progress push. PR "janitor_blocked" + issue "dispatched" is the
        # NORMAL mid-rework steady state, not a wedge.
        #
        # Issue #625: "live" is no longer just "is the PID alive?". Both the
        # sidecar-based and state.json-based checks route through one
        # predicate (``issue_worker_liveness``) that bounds the state-side
        # check with the watchdog's stall standard -- an alive-but-silent
        # session (no real activity for > stall_minutes, or past the
        # wall-clock deadline with an inconclusive probe) is wedged, not
        # live, and unescalate may proceed. The refusal carries session age
        # and last-activity diagnostics so an operator can tell a wedged
        # worker from a working one.
        if issue_number is not None:
            from datetime import UTC

            from .worker import issue_worker_liveness

            sessions_dir = self._layout.sessions_dir
            verdict = issue_worker_liveness(
                issue_number, issue_state, sessions_dir, self.config, datetime.now(UTC)
            )
        else:
            verdict = None
        if verdict is not None and verdict.live:
            return CommandResult(
                True,
                f"issue #{issue_number} has a live worker session; nothing to "
                f"unescalate (pr={pr_number} left untouched) -- {verdict.reason}",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "issue_worker_alive": True,
                    "issue_worker_last_activity_at": verdict.last_activity_at,
                    "issue_worker_last_activity_source": verdict.last_activity_source,
                    "issue_worker_session_started_at": verdict.session_started_at,
                    "issue_worker_pid": verdict.pid,
                    "issue_worker_source": verdict.source,
                    "changed": False,
                },
            )

        # Compute the TRANSFORMATION from the pre-fetch snapshot (for dry_run
        # reporting), then re-apply it to freshly-loaded entries inside the
        # write lock below -- gh.pr_view has real latency, and writing this
        # snapshot's dicts wholesale would clobber any field a concurrent
        # writer (e.g. a reconcile pass) touched in the meantime.
        transitions: dict[str, list[Any]] = {}
        label_edge: str | None = None

        pr_status_target: str | None = None
        if pr_number is not None and pr_stuck:
            if live_pr_state == "MERGED":
                pr_status_target = "merged"
            elif live_pr_state == "CLOSED":
                pr_status_target = "closed"
            else:
                pr_status_target = PASSIVE_OPEN_STATUS

        def _apply_pr_reset(entry: dict[str, Any]) -> dict[str, Any]:
            updated = dict(entry)
            updated["status"] = pr_status_target
            if pr_status_target == PASSIVE_OPEN_STATUS:
                updated["review_dispatch_attempt_count"] = 0
                updated["request_changes_count"] = 0
                for field_name in self._UNESCALATE_PR_RESET_FIELDS:
                    if field_name in ("review_dispatch_attempt_count", "request_changes_count"):
                        continue
                    updated.pop(field_name, None)
            return updated

        issue_status_action: str = "leave"
        if issue_number is not None:
            if live_pr_state == "OPEN" and pr_number is not None:
                issue_status_action = "passive"
                label_edge = "unescalated_pr_open"
            elif live_pr_state in ("MERGED", "CLOSED"):
                # Terminal PR: leave issue status to finalization/reconcile,
                # which own the closed-issue bookkeeping.
                label_edge = None
            elif issue_stuck:
                # No live PR at all — back to the never-dispatched baseline
                # (a status literal no dispatch selector reads would just
                # recreate the orphan gap reconcile now repairs).
                issue_status_action = "drop"
                label_edge = "unescalated_requeued"

        def _apply_issue_reset(entry: dict[str, Any]) -> dict[str, Any]:
            updated = dict(entry)
            if issue_status_action == "passive":
                updated["status"] = PASSIVE_OPEN_STATUS
            elif issue_status_action == "drop":
                updated.pop("status", None)
            for field_name in self._UNESCALATE_ISSUE_RESET_FIELDS:
                updated.pop(field_name, None)
            return updated

        if pr_number is not None and pr_stuck:
            snapshot_new_pr = _apply_pr_reset(pr_state)
            if snapshot_new_pr.get("status") != pr_state.get("status"):
                transitions["pr.status"] = [pr_state.get("status"), snapshot_new_pr["status"]]
        if issue_number is not None:
            snapshot_new_issue = _apply_issue_reset(issue_state)
            if snapshot_new_issue.get("status") != issue_state.get("status"):
                transitions["issue.status"] = [
                    issue_state.get("status"),
                    snapshot_new_issue.get("status"),
                ]

        if dry_run:
            return CommandResult(
                True,
                f"dry-run: would unescalate pr={pr_number} issue={issue_number} "
                f"(label edge: {label_edge})",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "transitions": transitions,
                    "label_edge": label_edge,
                    "changed": False,
                },
            )

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            if pr_number is not None and pr_stuck:
                fresh_pr = state["prs"].get(str(pr_number), {})
                state["prs"][str(pr_number)] = {
                    **_apply_pr_reset(fresh_pr if isinstance(fresh_pr, dict) else {}),
                    "number": pr_number,
                }
            if issue_number is not None:
                fresh_issue = state["issues"].get(str(issue_number), {})
                state["issues"][str(issue_number)] = {
                    **_apply_issue_reset(fresh_issue if isinstance(fresh_issue, dict) else {}),
                    "number": issue_number,
                }
            state = self._record_event(
                state,
                "unescalate",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "transitions": transitions,
                    "label_edge": label_edge,
                },
            )
            save_state(self.paths.state_file, state)

        label_error = None
        if label_edge is not None and issue_number is not None:
            result = transition(self.gh, self.config.labels, int(issue_number), label_edge)
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": label_edge,
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    entry = state["issues"].get(str(issue_number), {})
                    state["issues"][str(issue_number)] = {
                        **(entry if isinstance(entry, dict) else {}),
                        "number": issue_number,
                        "label_error": label_error,
                    }
                    save_state(self.paths.state_file, state)

        summary = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in transitions.items())
        message = f"unescalated pr={pr_number} issue={issue_number}"
        if summary:
            message += f" ({summary})"
        if label_error:
            message += f" (label update failed: {label_error['outcome']})"
        return CommandResult(
            True,
            message,
            {
                "pr": pr_number,
                "issue": issue_number,
                "transitions": transitions,
                "label_edge": label_edge,
                "label_error": label_error,
                "changed": True,
            },
        )

    @_guard_state_lock
    def merge_ready(
        self,
        pr_number: int,
        *,
        merge: bool | None = None,
        merge_train_head: int | None = None,
    ) -> CommandResult:
        # Idempotence: if state already records this PR as merged, short-circuit
        # to a success no-op. Re-running `ship-it` on a completed PR must not
        # re-attempt `gh pr merge` (which fails on an already-merged PR and
        # propagates GitHubError → exit 2).
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing_pr_state = state["prs"].get(str(pr_number), {})
            if existing_pr_state.get("status") == "merged":
                # Clear any stale merge alert so a reopened issue can re-alert.
                _issue_number = existing_pr_state.get("issue_number")
                if _issue_number is not None:
                    _issue_key = str(_issue_number)
                    _issue_entry = state["issues"].get(_issue_key, {})
                    if _issue_entry.get("merge_alert") != "OK":
                        state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
                        save_state(self.paths.state_file, state)
                return CommandResult(
                    True,
                    f"PR #{pr_number} already merged",
                    {
                        "pr": pr_number,
                        "issue": existing_pr_state.get("issue_number"),
                        "already_merged": True,
                        "merged": True,
                    },
                )
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return CommandResult(False, f"PR #{pr_number} was not found", {})
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
        )
        decision = self._review_decision(pr_number)
        approved = decision.get("decision") == "approved"
        sync_failed = False
        merge_conflict = False
        merge_conflict_routed = False
        check_failure_routed = False
        cross_pr_revert_detected = False
        cross_pr_revert_routed = False
        cross_pr_revert_reason: str | None = None
        issue_status: str | None = None
        label_error: dict[str, Any] | None = None
        rework_label_error: dict[str, Any] | None = None
        if approved:
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            head_moved = reviewed_head_sha is None or live_head_sha != reviewed_head_sha
            carried_forward = False
            if head_moved and live_head_sha:
                old_reviewed_head_sha = reviewed_head_sha
                check = self._check_carry_forward(pr_number, decision)
                if check.carry_forward:
                    self._update_approval_head(
                        pr_number,
                        decision,
                        live_head_sha,
                        old_head=old_reviewed_head_sha,
                        issue_number=issue_number,
                        tier=check.tier or "patch-id",
                        new_patch_id=check.live_patch_id,
                        new_signature=check.live_signature,
                    )
                    pr = self.gh.pr_view(pr_number) or pr
                    decision = self._review_decision(pr_number)
                    reviewed_head_sha = decision.get("reviewed_head_sha")
                    live_head_sha = pr.get("headRefOid")
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)] = {
                            **state["prs"].get(str(pr_number), {}),
                            "number": pr_number,
                            "issue_number": issue_number,
                            "status": "approved",
                            "head_moved": False,
                            "reviewed_head_sha": reviewed_head_sha,
                            "reviewed_patch_id": check.live_patch_id,
                            "carry_forward_tier": check.tier,
                            "carried_forward_from": decision.get("carried_forward_from", []),
                            "live_head_sha": live_head_sha,
                            "consecutive_failed_merge_attempts": 0,
                            "consecutive_stale_base_deferrals": 0,
                        }
                        # The carry-forward event itself is recorded inside
                        # _update_approval_head (issue #638) so every call
                        # site is instrumented uniformly; do NOT re-record
                        # here or the transition is double-counted.
                        save_state(self.paths.state_file, state)
                    carried_forward = True
            if head_moved and not carried_forward:
                message = "PR head moved since approval — re-review required"
                label_error: dict[str, Any] | None = None
                if issue_number is not None:
                    result = transition(
                        self.gh, self.config.labels, issue_number, "review_started"
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": "review_started",
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state["prs"][str(pr_number)] = {
                        **state["prs"].get(str(pr_number), {}),
                        "number": pr_number,
                        "issue_number": issue_number,
                        "status": "reviewing",
                        "head_moved": True,
                        "reviewed_head_sha": reviewed_head_sha,
                        "live_head_sha": live_head_sha,
                        "consecutive_failed_merge_attempts": 0,
                        "consecutive_stale_base_deferrals": 0,
                    }
                    if issue_number is not None:
                        _issue_key = str(issue_number)
                        _issue_entry = state["issues"].get(_issue_key, {})
                        state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
                    state = self._record_event(
                        state,
                        "head_moved",
                        {
                            "pr_number": pr_number,
                            "reviewed_head_sha": reviewed_head_sha,
                            "live_head_sha": live_head_sha,
                        },
                    )
                    save_state(self.paths.state_file, state)
                return CommandResult(
                    False,
                    message,
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "can_merge": False,
                        "merged": False,
                        "head_moved": True,
                        "reviewed_head_sha": reviewed_head_sha,
                        "live_head_sha": live_head_sha,
                        "review_decision": decision,
                        "label_error": label_error,
                    },
                )
            # Genuine merge conflict: gh pr update-branch cannot resolve this.
            # Record the conflict and let the failed-attempt alarm decide when
            # to route the linked issue to rework_requested for a worker to resolve.
            if self._is_merge_conflict(pr):
                state = load_state_locked(self.paths.state_file)
                issue_state = (
                    state["issues"].get(str(issue_number), {}) if issue_number is not None else {}
                )
                issue_status = issue_state.get("status")
                if issue_status in (
                    "dispatched",
                    "dispatch_pending",
                    "manifest_written",
                ):
                    return CommandResult(
                        True,
                        f"PR #{pr_number} merge conflict is being resolved by a rework worker",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "can_merge": False,
                            "merged": False,
                            "review_decision": decision,
                            "merge_conflict": True,
                            "consecutive_failed_merge_attempts": existing_pr_state.get(
                                "consecutive_failed_merge_attempts", 0
                            ),
                            "merge_attempt_alarm": False,
                            "merge_attempt_warning": None,
                        },
                    )
                # The linked issue is in a human-terminal state (escalated to a
                # human decision, or blocked pending one). Never reroute those
                # to rework_requested: transition() has no source-state
                # validation, so doing so would silently strip human_needed
                # and hand the issue back into the automated pipeline behind
                # the human's back. Leave the issue and PR alone; a human
                # must move it out of this state.
                if issue_status in ("escalated", "blocked"):
                    return CommandResult(
                        True,
                        f"PR #{pr_number} merge conflict on issue #{issue_number} "
                        f"awaiting human decision ({issue_status}); not rerouted",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "can_merge": False,
                            "merged": False,
                            "review_decision": decision,
                            "merge_conflict": True,
                            "consecutive_failed_merge_attempts": existing_pr_state.get(
                                "consecutive_failed_merge_attempts", 0
                            ),
                            "merge_attempt_alarm": False,
                            "merge_attempt_warning": None,
                        },
                    )
                merge_conflict = True
                sync_failed = True
                # Conflict-rework dispatch is deferred until the consecutive
                # failed-merge-attempt alarm threshold is reached (below). This
                # debounces transient/stale CONFLICTING readings and keeps the
                # approved verdict intact for the rework push's carry-forward.
            # Head matches the approved SHA. In front-of-train mode, only the
            # head of the approved queue is allowed to proceed, and it must be
            # up-to-date with main before checks are evaluated.
            update_branch_strategy = self.config.auto_merge.update_branch_strategy
            if update_branch_strategy == "front_of_train":
                if merge_train_head is not None and merge_train_head != pr_number:
                    return self._merge_not_ready_result(
                        pr_number,
                        issue_number,
                        decision,
                        existing_pr_state,
                    )
                if merge_train_head is None:
                    try:
                        prs = self.gh.pr_list()
                    except GitHubError:
                        sync_failed = True
                    else:
                        head = self._merge_train_head(prs)
                        if head is not None and head != pr_number:
                            return self._merge_not_ready_result(
                                pr_number,
                                issue_number,
                                decision,
                                existing_pr_state,
                            )
            # Single point of enforcement: derive base freshness from the GitHub
            # compare API once and use it for both the pre-merge sync decision
            # and the merge-base gate. mergeStateStatus can lag and report CLEAN
            # while the branch is actually stale, so it is no longer authoritative.
            base_current: bool | None = None
            if not sync_failed and update_branch_strategy in {"front_of_train", "broadcast"}:
                base_current = self._is_base_current(pr)
                # Aviator MergeQueue handoff (task #10): once this PR has
                # already been parked in Aviator's queue (state status
                # "mergequeue" from a prior merge_ready pass), Aviator owns
                # rebasing it. Calling pr_update_branch here on every
                # subsequent poll would race Aviator's own rebase as a second
                # writer on the same ref — this repo's live config is
                # broadcast, so every poll would otherwise attempt it. The
                # base-freshness *read* above still runs (it only feeds the
                # require_current_base gate below); only the write is
                # skipped. See the merge-train exclusion in
                # _merge_train_candidates for the sibling half of this fix.
                already_in_mergequeue = existing_pr_state.get("status") == "mergequeue"
                if not already_in_mergequeue and self._should_update_pr_branch(pr, base_current):
                    if self.gh.pr_update_branch(pr_number):
                        new_head = self._verify_synced_head(pr_number, live_head_sha)
                        if new_head and new_head != live_head_sha:
                            self._update_approval_head(
                                pr_number,
                                decision,
                                new_head,
                                old_head=live_head_sha,
                                issue_number=issue_number,
                            )
                            pr = self.gh.pr_view(pr_number) or pr
                            decision = self._review_decision(pr_number)
                        elif new_head == live_head_sha:
                            # Already up-to-date; nothing to do.
                            pass
                        else:
                            sync_failed = True
                    else:
                        sync_failed = True
            # merge-base freshness gate: mergeStateStatus can lag, so verify
            # ancestry with the GitHub compare API before merging.
            if not sync_failed and self.config.auto_merge.require_current_base:
                if base_current is None and update_branch_strategy not in {
                    "front_of_train",
                    "broadcast",
                }:
                    base_current = self._is_base_current(pr)
                elif pr.get("headRefOid") != live_head_sha:
                    base_current = self._is_base_current(pr)
                if base_current is not True:
                    base_ref = pr.get("baseRefName")
                    head_sha = pr.get("headRefOid")
                    reason = "compare_unavailable" if base_current is None else "base_stale"
                    return self._merge_deferred_stale_base_result(
                        pr_number,
                        issue_number,
                        decision,
                        base_ref,
                        head_sha,
                        reason,
                    )

            # Cross-PR revert gate: a branch that merges a base commit and then
            # reverts it has a clean PR diff but would silently undo the base
            # change when squash-merged. Detect by enumerating branch commits not
            # on base and matching `Revert "..."` subjects against base commits.
            if not sync_failed:
                cross_pr_revert_reason = detect_cross_pr_revert(pr, self.repo_root)
                if cross_pr_revert_reason:
                    cross_pr_revert_detected = True
                    sync_failed = True
                    if issue_number is not None:
                        state = load_state_locked(self.paths.state_file)
                        issue_state = state["issues"].get(str(issue_number), {})
                        issue_status = issue_state.get("status")
                        if issue_status not in (
                            "escalated",
                            "blocked",
                            "dispatched",
                            "dispatch_pending",
                            "manifest_written",
                        ):
                            if issue_status != "rework_requested":
                                cross_pr_revert_routed = True
                                rework_label_error = self._request_cross_pr_revert_rework(
                                    pr, issue_number, decision, cross_pr_revert_reason
                                )
        checks = self.gh.pr_checks(pr_number)
        checks_unavailable = checks is None

        if checks_unavailable:
            # gh pr checks command itself failed. Treat every required check as
            # unavailable (not missing) and do not merge; callers/loop will count
            # this as an infrastructure error.
            summary = summarize_checks(None, self.config.auto_merge.required_checks)
            enriched_checks: list[dict[str, Any]] = []
        else:
            # Enrich check data with infrastructure failure detection for FAILED checks
            # This implements detection signals 1 (zero-step jobs) and 2 (billing annotations)
            # from issue #210, keeping summarize_checks pure by enriching at the data boundary
            enriched_checks = []
            for check in checks:
                state = str(check.get("state") or "").upper()
                if state == "FAILURE":
                    # Check if this failure is due to infrastructure issues
                    check_run_id = check.get("databaseId")
                    if check_run_id and isinstance(check_run_id, int):
                        # The databaseId from gh pr checks IS the GitHub Actions job id
                        job = self.gh.actions_job(check_run_id)
                        annotations = self.gh.check_run_annotations(check_run_id)
                        if job and is_infrastructure_failure(job, annotations):
                            # Reclassify as infrastructure failure by setting state to a marker
                            # that summarize_checks will route to infra_failed
                            check = {**check, "state": "INFRA_FAILURE"}
                enriched_checks.append(check)
            summary = summarize_checks(enriched_checks, self.config.auto_merge.required_checks)
        # Run containment check for worker edits leaked into operator checkout
        diff = self.gh.pr_diff(pr_number)
        containment_warnings = check_operator_containment(self.repo_root, diff, pr_number)
        if containment_warnings:
            # Log containment warnings as a pre-merge gate warning
            # This is report-only, not blocking (per issue directive)
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state = self._record_event(
                    state,
                    "containment_check",
                    {
                        "pr_number": pr_number,
                        "warnings": list(containment_warnings),
                    },
                )
                save_state(self.paths.state_file, state)
        # Readiness no-CI stall gate (issue #474): an approved PR whose required
        # checks have not appeared within ``readiness_no_ci_minutes`` is routed to
        # rework instead of waiting forever for CI that will never start.
        if (
            not checks_unavailable
            and approved
            and not sync_failed
            and not summary.ready
            and not _is_pending_only(summary)
            and issue_number is not None
        ):
            now = datetime.now(UTC)
            if _is_readiness_no_ci_stall(pr, enriched_checks, self.config.auto_merge, now):
                state = load_state_locked(self.paths.state_file)
                issue_state = state["issues"].get(str(issue_number), {})
                issue_status = issue_state.get("status")
                if issue_status not in (
                    "dispatched",
                    "dispatch_pending",
                    "manifest_written",
                    "escalated",
                    "blocked",
                    "rework_requested",
                ):
                    label_error = self._request_readiness_no_ci_rework(
                        pr, issue_number, decision, summary.missing
                    )
                    return CommandResult(
                        True,
                        f"PR #{pr_number} has not started required CI checks; rework requested",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "can_merge": False,
                            "merged": False,
                            "review_decision": decision,
                            "checks": asdict(summary),
                            "checks_unavailable": False,
                            "label_error": label_error,
                            "readiness_no_ci_stall": True,
                            "merge_conflict": merge_conflict,
                            "merge_attempt_alarm": False,
                            "merge_attempt_warning": None,
                        },
                    )

        can_merge = (
            summary.ready
            and (approved or not self.config.auto_merge.require_approved_review)
            and not sync_failed
        )
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        branch_deleted: bool | None = None
        update_results: list[dict[str, Any]] | None = None
        cancel_results: dict[str, Any] | None = None
        mergequeue_label_applied: bool | None = None
        merge_hold: bool = False
        merge_hold_check_unavailable: bool = False
        if can_merge and should_merge:
            mergequeue_label = self.config.auto_merge.mergequeue_label
            if mergequeue_label:
                # Aviator MergeQueue handoff (task #10): apply the trigger
                # label instead of self-merging. add_pr_label is PR-scoped
                # (issue_number may be None for cross-repo PRs) and idempotent.
                # State records status="mergequeue" (never "merged"), so the
                # idempotency short-circuit at the top of this method does not
                # fire — a re-run of `ship-it` while the queue merge is still
                # pending safely re-applies the (no-op) label. Aviator does
                # its own queue rebase (replacing _update_open_agent_prs) and
                # GitHub auto-closes the linked issue via the PR's "Closes #N"
                # body; branch deletion is configured on the Aviator side.
                # cancel_superseded_runs is the one accepted residual gap (see
                # PR description). Once GitHub reports the PR merged,
                # reconcile.py's merged_outside_orchestrator drift path
                # reconciles status to "merged" and runs the "merged" label
                # transition — no new post-merge bookkeeping is added here.
                #
                # Issue #496: an operator can park an approved PR by adding the
                # configured merge-hold label to the PR or its linked issue.
                # When the hold is present, skip the mergequeue re-add entirely.
                # Scope note: this hold check runs only in mergequeue mode
                # (when ``mergequeue_label`` is set). In direct-merge mode the
                # hold label has no effect — the issue title scopes this to
                # "the mergequeue re-add," so the self-merge branch is unchanged.
                merge_hold = self.config.labels.merge_hold in label_names(pr)
                if not merge_hold and issue_number is not None:
                    try:
                        issue = self.gh.issue_view(issue_number)
                    except (GitHubError, ValueError):
                        merge_hold_check_unavailable = True
                        issue = None
                    if not merge_hold_check_unavailable and (
                        not isinstance(issue, dict) or "labels" not in issue
                    ):
                        merge_hold_check_unavailable = True
                        issue = None
                    if not merge_hold_check_unavailable:
                        issue_labels = label_names(issue) if issue else set()
                        merge_hold = self.config.labels.merge_hold in issue_labels
                if not merge_hold and not merge_hold_check_unavailable:
                    mergequeue_label_applied = self.gh.add_pr_label(pr_number, mergequeue_label)
                if mergequeue_label_applied:
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)] = {
                            **state["prs"].get(str(pr_number), {}),
                            "number": pr_number,
                            "issue_number": issue_number,
                            "status": "mergequeue",
                            "consecutive_failed_merge_attempts": 0,
                        }
                        if issue_number is not None:
                            _issue_key = str(issue_number)
                            _issue_entry = state["issues"].get(_issue_key, {})
                            state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
                        save_state(self.paths.state_file, state)
                # else: add_pr_label failed. The label IS the handoff, so a
                # failure here must not advance status to "mergequeue" —
                # treating a failed label add as success would silently
                # orphan the PR (never self-merged, never picked up by
                # Aviator, and nothing would ever look wrong to state). Leave
                # status/counters untouched; the shared failed-attempt-alarm
                # block below (mergequeue_handoff_failed) increments
                # consecutive_failed_merge_attempts on every retry and can
                # escalate exactly like an unmergeable approved PR.
            else:
                # Merge, then labels, then best-effort branch deletion — in that
                # order. merge_pr is the irreversible step: persist status="merged"
                # to state IMMEDIATELY after it succeeds and BEFORE the label
                # transition, so a transition failure or Ctrl+C can't leave GitHub
                # merged while state.json still shows "reviewing" — which made
                # reconcile false-positive on every clean auto-merge and lost the
                # merged fact entirely on a crash between merge and save.
                merge_output = self.gh.merge_pr(
                    pr_number,
                    self.config.auto_merge.strategy,
                    admin=self.config.auto_merge.admin,
                    merge_flags=self.config.auto_merge.merge_flags,
                )
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state["prs"][str(pr_number)] = {
                        **state["prs"].get(str(pr_number), {}),
                        "number": pr_number,
                        "issue_number": issue_number,
                        "status": "merged",
                        "merged": True,
                        "consecutive_failed_merge_attempts": 0,
                    }
                    if issue_number is not None:
                        _issue_key = str(issue_number)
                        _issue_entry = state["issues"].get(_issue_key, {})
                        state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
                    save_state(self.paths.state_file, state)
                # Label + branch cleanup are best-effort; the merged fact is already
                # durable. A branch-deletion failure (head branch checked out in a
                # worktree) or label failure must never un-record the merge.
                if issue_number is not None:
                    result = transition(self.gh, self.config.labels, issue_number, "merged")
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": "merged",
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                    # Close the linked issue explicitly — idempotent if already closed
                    # via GitHub's keyword automation. This ensures the dependency gate
                    # sees the closure immediately, avoiding the agent:done+OPEN state.
                    self.gh.close_issue(issue_number)
                if self.config.auto_merge.delete_branch:
                    head_ref = str(pr.get("headRefName") or "")
                    branch_deleted = self.gh.delete_branch(head_ref) if head_ref else False
                # Update remaining open agent PRs after successful merge (if configured)
                if self.config.auto_merge.update_branch_strategy in {
                    "broadcast",
                    "front_of_train",
                }:
                    update_results = self._update_open_agent_prs(pr_number)
                # Cancel superseded queued runs on default branch after successful merge (if configured)
                if self.config.runners.enabled and self.config.runners.cancel_superseded_main_runs:
                    cancel_results = cancel_superseded_runs(
                        self.gh,
                        self.config.runners.default_branch,
                        self.config.runners.workflow_name,
                    )
        # Aviator MergeQueue handoff (task #10): the label add IS the handoff
        # — a failed add_pr_label must be treated as a genuine unmergeable
        # pass (like an unmet check or a merge conflict), not silently
        # swallowed as best-effort cleanup. can_merge is True here (checks
        # green, approved), so the "approved and not can_merge" alarm gate
        # below would otherwise never fire for this failure mode.
        mergequeue_handoff_failed = bool(
            self.config.auto_merge.mergequeue_label and mergequeue_label_applied is False
        )
        # Conflict-rework dispatch is debounced to the failed-attempt alarm
        # threshold so a single transient/stale CONFLICTING reading does not
        # clobber an approved verdict. Re-read the issue status and the PR
        # attempt counter immediately before dispatch: the preceding checks,
        # diff, and containment work are network-I/O windows long enough for a
        # concurrent pass to have moved the issue into an in-flight or
        # human-terminal state, and the stale `existing_pr_state` snapshot can
        # diverge from the counter the final persistence block will reload
        # (e.g. a carry-forward reset in this same pass). Dispatch outside the
        # final state-lock because _request_merge_conflict_rework acquires its
        # own lock.
        if (
            merge_conflict
            and approved
            and not can_merge
            and not _is_pending_only(summary)
            and issue_number is not None
        ):
            state = load_state_locked(self.paths.state_file)
            issue_state = state["issues"].get(str(issue_number), {})
            issue_status = issue_state.get("status")
            existing_for_route = state["prs"].get(str(pr_number), {})
            if issue_status not in (
                "dispatched",
                "dispatch_pending",
                "manifest_written",
                "escalated",
                "blocked",
                "rework_requested",
            ):
                new_attempts_for_route = (
                    int(existing_for_route.get("consecutive_failed_merge_attempts", 0)) + 1
                )
                threshold = self.config.auto_merge.failed_attempt_alarm
                if threshold > 0 and new_attempts_for_route >= threshold:
                    merge_conflict_routed = True
                    rework_label_error = self._request_merge_conflict_rework(
                        pr, issue_number, decision
                    )
        # Check-failure rework dispatch (issue #674): an approved PR whose
        # required checks have genuinely failed (a completed FAILURE
        # conclusion, not merely pending/missing/infra_failed/unavailable)
        # never re-enters review()'s janitor gate once approved -- the
        # already_approved fast path in loop() routes straight to
        # merge_ready as long as the head hasn't moved, so nothing else in
        # the orchestrator ever pushes it back to rework. Debounced to the
        # same failed-attempt-alarm threshold as merge-conflict rework
        # (above) so a transient failure or the mergequeue's own
        # speculative-merge retry gets a chance to self-heal first.
        # Deliberately scoped to `summary.failed` only: `missing`/
        # `infra_failed`/`unavailable` checks are not something a code push
        # can reliably fix and are left to the existing warning-only alarm.
        if (
            not merge_conflict
            and not cross_pr_revert_detected
            and not mergequeue_handoff_failed
            and approved
            and not can_merge
            and bool(summary.failed)
            and issue_number is not None
        ):
            state = load_state_locked(self.paths.state_file)
            issue_state = state["issues"].get(str(issue_number), {})
            issue_status = issue_state.get("status")
            existing_for_route = state["prs"].get(str(pr_number), {})
            if issue_status not in (
                "dispatched",
                "dispatch_pending",
                "manifest_written",
                "escalated",
                "blocked",
                "rework_requested",
            ):
                new_attempts_for_route = (
                    int(existing_for_route.get("consecutive_failed_merge_attempts", 0)) + 1
                )
                threshold = self.config.auto_merge.failed_attempt_alarm
                if threshold > 0 and new_attempts_for_route >= threshold:
                    check_failure_routed = True
                    rework_label_error = self._request_check_failure_rework(
                        pr, issue_number, decision, summary
                    )
        if rework_label_error is not None:
            label_error = rework_label_error
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing = state["prs"].get(str(pr_number), {})
            new_attempts = 0
            new_stale_base_deferrals = 0
            merge_attempt_alarm = False
            merge_attempt_warning: str | None = None
            if (
                approved and not can_merge and not _is_pending_only(summary)
            ) or mergequeue_handoff_failed:
                new_attempts = int(existing.get("consecutive_failed_merge_attempts", 0)) + 1
                threshold = self.config.auto_merge.failed_attempt_alarm
                merge_attempt_alarm = threshold > 0 and new_attempts == threshold
                if merge_attempt_alarm:
                    if merge_conflict:
                        pass_str = "pass" if new_attempts == 1 else "passes"
                        if issue_number is None:
                            conflict_detail = "no linked issue, cannot route to rework"
                        elif merge_conflict_routed:
                            if rework_label_error:
                                outcome = rework_label_error.get("outcome", rework_label_error)
                                conflict_detail = (
                                    f"rework dispatch attempted (label update failed: {outcome})"
                                )
                            else:
                                conflict_detail = "rework dispatched"
                        elif issue_status == "rework_requested":
                            conflict_detail = "rework already requested"
                        else:
                            conflict_detail = "rework not routed"
                        merge_attempt_warning = (
                            f"PR #{pr_number} approved but unmergeable for {new_attempts} {pass_str}: "
                            f"merge conflict — {conflict_detail}"
                        )
                    elif cross_pr_revert_detected:
                        pass_str = "pass" if new_attempts == 1 else "passes"
                        if issue_number is None:
                            revert_detail = "no linked issue, cannot route to rework"
                        elif cross_pr_revert_routed:
                            if rework_label_error:
                                outcome = rework_label_error.get("outcome", rework_label_error)
                                revert_detail = (
                                    f"rework dispatch attempted (label update failed: {outcome})"
                                )
                            else:
                                revert_detail = "rework dispatched"
                        elif issue_status == "rework_requested":
                            revert_detail = "rework already requested"
                        else:
                            revert_detail = "rework not routed"
                        merge_attempt_warning = (
                            f"PR #{pr_number} approved but unmergeable for {new_attempts} {pass_str}: "
                            f"cross-PR revert — {revert_detail}"
                        )
                    elif mergequeue_handoff_failed:
                        pass_str = "pass" if new_attempts == 1 else "passes"
                        merge_attempt_warning = (
                            f"PR #{pr_number} approved and checks green but the mergequeue "
                            f"label {self.config.auto_merge.mergequeue_label!r} failed to "
                            f"apply for {new_attempts} {pass_str} — never handed off to "
                            "Aviator"
                        )
                    elif bool(summary.failed):
                        pass_str = "pass" if new_attempts == 1 else "passes"
                        if issue_number is None:
                            check_detail = "no linked issue, cannot route to rework"
                        elif check_failure_routed:
                            if rework_label_error:
                                outcome = rework_label_error.get("outcome", rework_label_error)
                                check_detail = (
                                    f"rework dispatch attempted (label update failed: {outcome})"
                                )
                            else:
                                check_detail = "rework dispatched"
                        elif issue_status == "rework_requested":
                            check_detail = "rework already requested"
                        else:
                            check_detail = "rework not routed"
                        failed_str = ", ".join(summary.failed)
                        merge_attempt_warning = (
                            f"PR #{pr_number} approved but unmergeable for {new_attempts} {pass_str}: "
                            f"required check(s) failed ({failed_str}) — {check_detail}"
                        )
                    else:
                        merge_attempt_warning = _format_merge_attempt_alarm_message(
                            pr_number,
                            new_attempts,
                            summary,
                            mergeable=pr.get("mergeable"),
                            merge_state_status=pr.get("mergeStateStatus"),
                        )
                    state = self._record_event(
                        state,
                        "merge_failed_attempt_alarm",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "attempts": new_attempts,
                            "threshold": threshold,
                            "checks_summary": asdict(summary),
                            # Issue #751: carry GitHub's own mergeability signal
                            # alongside checks_summary so the event is
                            # diagnosable without re-deriving state that has
                            # since changed (mergeable/mergeStateStatus are
                            # already fetched by pr_view — see field list at
                            # _PR_SLIM_FIELDS).
                            "mergeable": pr.get("mergeable"),
                            "merge_state_status": pr.get("mergeStateStatus"),
                            "message": merge_attempt_warning,
                        },
                    )
            if (
                approved
                and can_merge
                and merge_output is None
                and not mergequeue_handoff_failed
                and not merge_hold_check_unavailable
            ):
                # merge=False / auto_merge.enabled=False: can_merge recovered but no
                # merge was attempted. Clear the merge alert so a subsequent
                # degradation can re-fire the digest (last_health == current_health
                # dedup would otherwise drop it). Excluded when the mergequeue
                # handoff itself failed or the merge-hold issue check was
                # unavailable — both are genuine problems, not benign
                # evaluation-only passes, and must not be masked as OK.
                if issue_number is not None:
                    _issue_key = str(issue_number)
                    _issue_entry = state["issues"].get(_issue_key, {})
                    if _issue_entry.get("merge_alert") != "OK":
                        state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
            prs_entry: dict[str, Any] = {
                **existing,
                "number": pr_number,
                "issue_number": issue_number,
                "consecutive_failed_merge_attempts": new_attempts,
                "consecutive_stale_base_deferrals": new_stale_base_deferrals,
            }
            if merge_output:
                prs_entry["status"] = "merged"
                prs_entry["merged"] = True
            state["prs"][str(pr_number)] = prs_entry
            state = self._record_event(
                state,
                "merge_ready",
                {
                    "pr_number": pr_number,
                    "can_merge": can_merge,
                    "merged": bool(merge_output),
                    "merge_hold": merge_hold,
                    "merge_hold_check_unavailable": merge_hold_check_unavailable,
                    "cancel_superseded_runs_results": cancel_results,
                },
            )
            save_state(self.paths.state_file, state)
        data = {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": can_merge,
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "merged": bool(merge_output),
            "merge_output": merge_output,
            "branch_deleted": branch_deleted,
            "review_decision": decision,
            "checks": asdict(summary),
            "checks_unavailable": checks_unavailable,
            "label_error": label_error,
            "update_open_prs_results": update_results,
            "cancel_superseded_runs_results": cancel_results,
            "containment_warnings": list(containment_warnings),
            "consecutive_failed_merge_attempts": new_attempts,
            "consecutive_stale_base_deferrals": new_stale_base_deferrals,
            "merge_attempt_alarm": merge_attempt_alarm,
            "merge_attempt_warning": merge_attempt_warning,
            "merge_conflict": merge_conflict,
            "cross_pr_revert_detected": cross_pr_revert_detected,
            "cross_pr_revert_reason": cross_pr_revert_reason,
            "cross_pr_revert_routed": cross_pr_revert_routed,
            "mergequeue_label_applied": mergequeue_label_applied,
            "merge_hold": merge_hold,
            "merge_hold_check_unavailable": merge_hold_check_unavailable,
        }
        message = "merge readiness evaluated"
        if cross_pr_revert_detected:
            message = f"cross-PR revert detected: {cross_pr_revert_reason}"
        elif checks_unavailable:
            message = "checks unavailable (gh failure)"
        elif merge_hold_check_unavailable:
            message += f" (merge-hold check unavailable for issue #{issue_number} — not handed off to Aviator)"
        elif merge_hold:
            message += (
                f" (merge-hold label {self.config.labels.merge_hold!r} present — left alone)"
            )
        elif mergequeue_label_applied is False:
            message += (
                f" (mergequeue label {self.config.auto_merge.mergequeue_label!r} FAILED to "
                f"apply — not handed off to Aviator; will retry, attempt {new_attempts})"
            )
        elif mergequeue_label_applied is True:
            message += (
                f" (handed off to mergequeue label {self.config.auto_merge.mergequeue_label!r})"
            )
        elif merge_output and label_error:
            message += f" (merged; post-merge label/branch cleanup failed: {label_error})"
        elif label_error:
            message += f" (label update failed: {label_error.get('outcome', label_error)})"
        return CommandResult(
            not (checks_unavailable or merge_hold_check_unavailable), message, data
        )

    @_guard_state_lock
    def spec_review(self, artifact_path: Path) -> CommandResult:
        """Run an explicit cross-family adversarial pass over a spec/plan file.

        Independent of ``cross_family.enabled`` (that flag governs the PR-auto path);
        this command is the pre-execution spec slot and always runs when invoked.
        """
        path = Path(artifact_path)
        artifact_text = path.read_text(encoding="utf-8")
        cfg = self.config.cross_family
        reviews_dir = self.paths.cross_family
        reviews_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(path.stem)
        prompt_text = self._render(
            "cross_family_spec_review.md",
            {"artifact_label": f"`{path}`", "artifact_text": artifact_text},
        )
        result = run_cross_family_review(
            model=cfg.model,
            command=cfg.command,
            repo_root=self.repo_root,
            prompt_text=prompt_text,
            prompt_path=reviews_dir / f"spec-{slug}-prompt.md",
            report_path=reviews_dir / f"spec-{slug}-review.md",
            timeout_seconds=cfg.timeout_seconds,
        )
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state = self._record_event(
                state, "spec_review", {"artifact": str(path), "ok": result.ok, "model": cfg.model}
            )
            save_state(self.paths.state_file, state)
        return CommandResult(
            result.ok,
            "spec cross-family review complete"
            if result.ok
            else f"spec cross-family review failed: {result.error}",
            {
                "artifact": str(path),
                "report_path": result.report_path,
                "model": cfg.model,
                "ok": result.ok,
            },
        )

    def _cross_family_for_pr(
        self,
        *,
        pr: dict[str, Any],
        issue: dict[str, Any],
        pr_dir: Path,
        pr_number: int,
        issue_number: int | None,
        diff_path: Path,
        enabled: bool | None,
    ) -> tuple[str, CrossFamilyResult | None]:
        cfg: CrossFamilyConfig = self.config.cross_family
        use = cfg.enabled if enabled is None else enabled
        if not use or pr.get("isDraft"):
            return "", None
        report_path = pr_dir / "cross-family-review.md"
        # Idempotent: a non-empty, semantically valid SUCCESS report is reused,
        # so repeated review()/loop() passes don't re-burn the cross-family model
        # on the same PR. Failure stubs (headed "(UNAVAILABLE)") and exit-zero
        # but empty/blocked reports must NOT satisfy this check — reusing them
        # turned one codex timeout and one blocked refusal into a permanent
        # silent skip on every subsequent pass.
        # Additionally, reports are invalidated when the PR head SHA changes
        # to prevent reviewing stale code (issue #156).
        if report_path.exists() and report_path.stat().st_size > 0:
            text = report_path.read_text(encoding="utf-8")
            first_line = text.splitlines()[0]
            # The file is a wrapped report (header + caveat + body).  Validate the
            # model body only, not the wrapper text that itself contains bold
            # markdown ("**leads, not verdicts**").
            body = extract_report_body(text)
            stored_head_sha = extract_head_ref_oid(text)
            current_head_sha = pr.get("headRefOid")
            if (
                "(UNAVAILABLE)" not in first_line
                and report_body_is_valid(body)
                and stored_head_sha == current_head_sha
            ):
                return self._cross_family_section(report_path), CrossFamilyResult(
                    ok=True, report_path=str(report_path), model=cfg.model, reused=True
                )
        prompt_text = self._render(
            "cross_family_review.md",
            {
                "pr_number": pr_number,
                "pr_title": pr.get("title", ""),
                "pr_url": pr.get("url", ""),
                "issue_number": issue_number or "UNKNOWN",
                "issue_title": issue.get("title", "UNKNOWN"),
                "pr_json_path": pr_dir / "pr.json",
                "diff_path": diff_path,
            },
        )
        result = run_cross_family_review(
            model=cfg.model,
            command=cfg.command,
            repo_root=self.repo_root,
            prompt_text=prompt_text,
            prompt_path=pr_dir / "cross-family-prompt.md",
            report_path=report_path,
            timeout_seconds=cfg.timeout_seconds,
            dry_run=self.dry_run,
            head_ref_oid=pr.get("headRefOid"),
        )
        return self._cross_family_section(result.report_path), result

    def _build_prior_review_section(
        self,
        pr_dir: Path,
        prior_decision: dict[str, Any],
        new_head_sha: str | None,
    ) -> str:
        """Render ``$prior_review_section`` for a round-2+ review packet.

        Called when the prior decision is a terminal, non-pending verdict
        with a recorded ``reviewed_head_sha``. Two cases:

        - **Moved head** (prior head != live head): a genuine rework round.
          Surfaces round-1 findings plus an interdiff (prior reviewed head
          -> new head) so the reviewer has somewhere to start, without
          losing sight of the full diff: the interdiff is "start here,"
          never "only look here" -- the full diff stays attached and
          remains authoritative for findings outside it.
        - **Same head** (prior head == live head, issue #632 defect 3): a
          PR parked on ``agent:human-needed`` whose head has not advanced,
          or an operator-corrected verdict. The diff is identical, so no
          interdiff is generated; the findings are surfaced so a re-review
          can verify whether they still apply or whether the verdict was
          corrected. Without this branch the corrected verdict was
          invisible to the reviewer (the #510 case).

        Fail-safe posture mirrors janitor.py's patch-id carry-forward
        (``_calculate_patch_id``/``_check_no_op_rework``): every I/O call
        here (``compare_diff``) already returns errors as values (``None``),
        never raises, so a failed/unavailable comparison (404, GC'd SHA,
        rebase/divergence, gh failure) just omits the interdiff and says so
        -- it never blocks packet generation.
        """
        prior_head_sha = prior_decision.get("reviewed_head_sha")
        decision = prior_decision.get("decision") or "unknown"
        summary = str(prior_decision.get("summary") or "").strip()
        required_changes = prior_decision.get("required_changes")
        if not isinstance(required_changes, list):
            required_changes = []

        same_head = bool(prior_head_sha) and prior_head_sha == new_head_sha

        if same_head:
            lines = [
                "",
                "## Prior review (same head)",
                "",
                f"A prior review of this head (`{prior_head_sha}`) recorded "
                f"decision **{decision}**. The diff has not changed since that "
                "review -- the findings below are from the same code you are "
                "reviewing now. Verify whether they still apply or whether the "
                "verdict was corrected (e.g. by an operator hand-edit).",
                "",
            ]
            if summary:
                lines.append(f"Prior summary: {summary}")
                lines.append("")
            if required_changes:
                lines.append("Prior required changes:")
                lines.extend(f"- {change}" for change in required_changes)
                lines.append("")
            lines.append(
                "No interdiff is needed -- the head is unchanged. Re-examine "
                "the full diff and confirm or overturn the prior verdict."
            )
            lines.append("")
            return "\n".join(lines)

        lines = [
            "",
            "## Prior review (round 1, earlier head)",
            "",
            f"A previous review round on an earlier head (`{prior_head_sha}`) "
            f"recorded decision **{decision}**. These are round-1 findings on "
            "a DIFFERENT diff than the one you are reviewing now -- verify "
            "each one against the current code, don't assume it still applies.",
            "",
        ]
        if summary:
            lines.append(f"Round-1 summary: {summary}")
            lines.append("")
        if required_changes:
            lines.append("Round-1 required changes:")
            lines.extend(f"- {change}" for change in required_changes)
            lines.append("")

        interdiff_text = None
        if prior_head_sha and new_head_sha:
            interdiff_text = self.gh.compare_diff(str(prior_head_sha), str(new_head_sha))
        if interdiff_text and interdiff_text.strip():
            interdiff_path = pr_dir / "interdiff.patch"
            interdiff_path.write_text(interdiff_text, encoding="utf-8")
            lines.append(
                f"Interdiff (round-1 head to this head): `{interdiff_path}`. Verify "
                "each required change above is addressed there first -- but the "
                "full diff remains authoritative; findings outside the interdiff "
                "are still in scope."
            )
        else:
            lines.append(
                "Prior-head comparison was unavailable (rebase, divergence, or an "
                "API error) -- no interdiff could be generated. Review the full "
                "diff as usual, with the round-1 findings above in mind."
            )
        lines.append("")
        return "\n".join(lines)

    def reconcile(
        self, *, fix: bool = False, skip_dead_session_sweep: bool = False
    ) -> CommandResult:
        """Detect (and optionally repair) drift between GitHub reality and the
        orchestrator's labels/state — e.g. a PR merged by hand outside
        merge-ready leaving `agent:in-progress` stale forever. Read-only unless
        ``fix`` is passed.

        ``mop-up --fix`` is a state writer and must be mutually exclusive with
        a supervised ``bash-rats``/fleet pass on the same repo. It acquires the
        same ``supervisor.lock`` used by fleet/supervise. If either the
        supervisor lock or the state lock is held, the call returns a skipped
        value result rather than blocking or writing unlocked.

        This is the public entry point: acquire ``supervisor.lock`` (only
        when ``fix``, matching today's behavior), then delegate everything
        else to ``_reconcile_locked``, which requires the lock already be
        held. Do not inline the body back here — the periodic in-loop caller
        (``_maybe_reconcile_drift``) already holds ``supervisor.lock`` via
        its own caller (``loop()``) and must call ``_reconcile_locked``
        directly; re-entering this method would always fail to reacquire the
        same non-reentrant lock and silently no-op (merge-lane-recovery D-8a).
        """
        supervisor_lock = None
        if fix:
            supervisor_lock = try_acquire_byte_range_lock(
                layout.supervisor_lock_path(self.paths.root)
            )
            if supervisor_lock is None:
                return CommandResult(
                    True,
                    "reconcile deferred: supervisor lock held",
                    {"skipped": True, "reason": "supervisor_lock_held"},
                )
        try:
            return self._reconcile_locked(fix=fix, skip_dead_session_sweep=skip_dead_session_sweep)
        finally:
            if supervisor_lock is not None:
                supervisor_lock.release()

    def _reconcile_locked(
        self, *, fix: bool = False, skip_dead_session_sweep: bool = False
    ) -> CommandResult:
        """Run drift detection (and optional repair) against GitHub/state.

        Precondition: the caller MUST already hold ``supervisor.lock`` — this
        method never acquires it itself. ``reconcile()`` is the only method
        that acquires the lock (and only when ``fix``); it delegates here
        immediately afterward. The periodic in-loop caller
        (``_maybe_reconcile_drift``) calls this directly, bypassing
        ``reconcile()``'s lock-acquisition entirely, because ``loop()``'s own
        caller already holds ``supervisor.lock`` for the whole pass —
        acquiring it a second time on the same non-reentrant byte-range lock
        would always fail and silently no-op (merge-lane-recovery D-8a).

        Extracted verbatim from ``reconcile()``'s former body — logic here is
        unchanged from before the split, including the GraphQL rate-limit
        deferral, so ``charlie mop-up --fix`` (which still goes through
        ``reconcile()``) is byte-for-byte unchanged in behaviour.
        """
        try:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                threshold = self.config.runtime.graphql_rate_limit_threshold
                sufficient, remaining, reset_at = self.gh.check_graphql_rate_limit(threshold)
                if not sufficient:
                    state = append_event(
                        state,
                        "graphql_rate_limit_deferred",
                        {
                            "remaining": remaining,
                            "reset": reset_at,
                            "threshold": threshold,
                            "phase": "reconcile",
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)
                    return CommandResult(
                        True,
                        "reconcile deferred: GraphQL rate limit below threshold",
                        {
                            "deferred": True,
                            "deferred_reason": "graphql_rate_limit",
                            "graphql_remaining": remaining,
                            "graphql_reset": reset_at,
                            "graphql_threshold": threshold,
                        },
                    )
                drift = detect_drift(
                    self.gh,
                    state,
                    self.config,
                    repo_root=self.repo_root,
                    skip_dead_session_sweep=skip_dead_session_sweep,
                ) + detect_aviator_stale_blocked(self.gh, self.config, repo_root=self.repo_root)
                fixed = False
                post_fix_drift: list[DriftItem] = []
                if fix and drift:
                    new_state = apply_drift_fixes(
                        self.gh,
                        state,
                        drift,
                        self.config,
                        repo_root=self.repo_root,
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, new_state)
                    # Post-#134: transition() returns TransitionResult with PARTIAL_FAILURE
                    # for failed adds/removes, and apply_fixes records the outcome in the
                    # reconcile event. Re-detect against the new state to verify the repairs
                    # actually landed before reporting success.
                    post_fix_drift = detect_drift(
                        self.gh,
                        new_state,
                        self.config,
                        repo_root=self.repo_root,
                        skip_dead_session_sweep=skip_dead_session_sweep,
                    ) + detect_aviator_stale_blocked(
                        self.gh, self.config, repo_root=self.repo_root
                    )
                    fixed = len(post_fix_drift) == 0
            message = f"found {len(drift)} drift item(s)"
            if fixed:
                message += " — fixed"
            elif drift:
                if fix and post_fix_drift:
                    message += f" — partially fixed — {len(post_fix_drift)} item(s) remain"
                else:
                    message += " (read-only; pass --fix to repair)"
            # ok=False when drift is present and not fixed: scripts and CI can gate
            # on exit code to detect unresolved drift, matching how `doctor` gates.
            ok = not drift or fixed
            return CommandResult(
                ok,
                message,
                {
                    "drift": [asdict(item) for item in drift],
                    "fixed": fixed,
                    "drift_before": len(drift),
                    "drift_after": len(post_fix_drift),
                    "remaining_drift": [asdict(item) for item in post_fix_drift],
                },
            )
        except GraphQLBudgetError as exc:
            # Defensive: detect_drift re-checks the budget and may raise.
            return CommandResult(
                True,
                "reconcile deferred: GraphQL rate limit below threshold",
                {
                    "deferred": True,
                    "deferred_reason": "graphql_rate_limit",
                    "graphql_remaining": exc.remaining,
                    "graphql_reset": exc.reset_at,
                    "graphql_threshold": exc.threshold,
                },
            )
        except StateLockBusy:
            return _state_lock_busy_result("reconcile deferred: state lock held")

    @staticmethod
    def _cross_family_section(report_path: str | Path) -> str:
        return (
            "\n## Cross-family adversarial pass\n\n"
            f"An automated non-Claude adversarial review is at `{report_path}`. Read it, but "
            "treat its findings as **leads, not verdicts** — that model over-escalates severity. "
            "Verify each against live code before folding it in, reject over-escalations with a "
            "reason, and never let it gate the merge on its own.\n"
        )

    def _update_open_agent_prs(self, merged_pr_number: int) -> list[dict[str, Any]]:
        """Update remaining open agent PRs after a successful merge.

        Behavior is controlled by ``auto_merge.update_branch_strategy``:

        - "front_of_train" (default): update only the head of the approved
          queue, so a single merge step causes at most one CI reset on a
          single-runner merge train.
        - "broadcast": update every eligible open tracked PR that is not
          approved-pending-ship and has no required checks in-flight. Intended
          for multi-runner setups. PRs whose current review decision is
          ``request_changes``, escalated, or blocked are skipped.
        - "off": do nothing.

        Per-PR failures (conflicts, network errors) are reported as values and
        never abort the batch operation. A GitHubError from pr_list is also
        reported as a value and never propagates.
        """
        results: list[dict[str, Any]] = []
        mode = self.config.auto_merge.update_branch_strategy
        if mode == "off":
            return results

        if mode == "front_of_train":
            try:
                candidates = self._merge_train_candidates(exclude_pr_number=merged_pr_number)
            except GitHubError as exc:
                return [{"error": f"pr_list failed: {exc}"}]

            if not candidates:
                return results

            # Only the head of the merge-train queue is synced.
            _, pr_number, pr, decision, head = candidates[0]
            base_current = self._is_base_current(pr)
            if base_current is None:
                # Compare API unavailable: report distinctly from "up-to-date" so
                # a GitHub compare-API degradation is visible in telemetry instead
                # of masquerading as every open PR being current.
                return [
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "compare_unavailable",
                    }
                ]
            if not self._should_update_pr_branch(pr, base_current):
                return [
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "up-to-date",
                    }
                ]

            old_head = pr.get("headRefOid")
            if not self.gh.pr_update_branch(pr_number):
                return [
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "error": "pr_update_branch failed",
                    }
                ]

            new_head = self._verify_synced_head(pr_number, old_head)
            if new_head is None:
                return [
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "error": "post-sync head verification failed",
                    }
                ]
            if new_head == old_head:
                return [
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "up-to-date",
                    }
                ]

            self._update_approval_head(
                pr_number,
                decision,
                new_head,
                old_head=old_head,
                issue_number=linked_issue_number(
                    pr,
                    is_cross_repository=pr.get("isCrossRepository"),
                    branch_prefix=self.config.dispatch.branch_prefix,
                ),
            )
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": True,
                    "new_head": new_head,
                }
            ]

        # mode == "broadcast": update every eligible PR.
        try:
            prs = self.gh.pr_list()
        except GitHubError as exc:
            # Report the pr_list failure as a value instead of raising
            return [{"error": f"pr_list failed: {exc}"}]
        branch_prefix = self.config.dispatch.branch_prefix
        required_checks = self.config.auto_merge.required_checks

        for pr in prs:
            pr_number = int(pr.get("number", 0))
            if pr_number == merged_pr_number:
                continue

            # Skip fork PRs
            if pr.get("isCrossRepository"):
                continue

            # Only update PRs with the configured branch prefix
            head = str(pr.get("headRefName") or "")
            if not head.startswith(branch_prefix):
                continue

            # Derive eligibility from the recorded review decision. Never
            # update-branch a PR whose current decision is request_changes,
            # escalated, or blocked — rework or human intervention will replace
            # the head, so the CI run would be guaranteed-wasted time.
            decision = self._review_decision(pr_number)
            decision_value = decision.get("decision")
            if decision_value in {"request_changes", "blocked"} or decision.get("escalated"):
                results.append(
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "not_approved",
                    }
                )
                continue

            # Skip approved-pending-ship PRs to avoid invalidating their approvals.
            # These will get base-updated when they themselves are merged (GitHub
            # merges handle base freshness) or by a later pass after they merge.
            if decision_value == "approved":
                reviewed_head_sha = decision.get("reviewed_head_sha")
                live_head_sha = pr.get("headRefOid")
                if reviewed_head_sha is not None and live_head_sha == reviewed_head_sha:
                    # PR is approved and head hasn't moved since approval — skip update
                    results.append(
                        {
                            "pr_number": pr_number,
                            "head_ref": head,
                            "updated": False,
                            "skipped_reason": "approved-pending-ship",
                        }
                    )
                    continue

            # Skip PRs with required checks in PENDING/IN_PROGRESS to avoid cancelling in-flight CI
            # This prevents the wedge described in issue #209 where update-branch cancels
            # matrix jobs and aggregate-gate checks permanently fail against the frozen CANCELLED state.
            status_rollup = pr.get("statusCheckRollup")
            if status_rollup and required_checks:
                # statusCheckRollup is a flat array of check objects (CheckRun or StatusContext)
                # CheckRun uses 'status' field, StatusContext uses 'state' field
                has_pending_required = False
                for check in status_rollup:
                    check_name = check.get("name") or check.get("context")
                    if check_name in required_checks:
                        # Check if this required check is in a pending/in-progress state
                        # For CheckRuns: status != COMPLETED means in-flight
                        # For StatusContext: state != SUCCESS/FAILURE/ERROR means in-flight
                        status = check.get("status") or check.get("state", "")
                        # Treat any non-terminal status as in-flight (safer than enumerating)
                        # Terminal states: COMPLETED (CheckRun), SUCCESS/FAILURE/ERROR (StatusContext)
                        if status.upper() != "COMPLETED" and status.upper() not in {
                            "SUCCESS",
                            "FAILURE",
                            "ERROR",
                        }:
                            has_pending_required = True
                            break

                if has_pending_required:
                    results.append(
                        {
                            "pr_number": pr_number,
                            "head_ref": head,
                            "updated": False,
                            "skipped_reason": "pending-required-checks",
                        }
                    )
                    continue

            # Use the same compare-derived base-current signal as the front-of-train
            # path and merge_ready so broadcast mode also skips up-to-date PRs and
            # syncs stale ones even when mergeStateStatus is CLEAN.
            base_current = self._is_base_current(pr)
            if base_current is None:
                # Compare API unavailable: report distinctly from "up-to-date" so
                # a GitHub compare-API degradation is visible in telemetry instead
                # of masquerading as every open PR being current.
                results.append(
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "compare_unavailable",
                    }
                )
                continue
            if not self._should_update_pr_branch(pr, base_current):
                results.append(
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "up-to-date",
                    }
                )
                continue

            # Attempt to update the branch
            success = self.gh.pr_update_branch(pr_number)
            results.append(
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": success,
                }
            )

        return results

    def _merge_not_ready_result(
        self,
        pr_number: int,
        issue_number: int | None,
        decision: dict[str, Any],
        existing_pr_state: dict[str, Any],
    ) -> CommandResult:
        """Return a non-mergeable result for an approved PR that is not the train head."""
        return CommandResult(
            True,
            f"PR #{pr_number} is not the head of the merge-train queue",
            {
                "pr": pr_number,
                "issue": issue_number,
                "can_merge": False,
                "auto_merge_enabled": self.config.auto_merge.enabled,
                "merged": False,
                "merge_output": None,
                "branch_deleted": None,
                "review_decision": decision,
                "checks": asdict(summarize_checks([], self.config.auto_merge.required_checks)),
                "checks_unavailable": False,
                "label_error": None,
                "update_open_prs_results": None,
                "cancel_superseded_runs_results": None,
                "containment_warnings": [],
                "consecutive_failed_merge_attempts": existing_pr_state.get(
                    "consecutive_failed_merge_attempts", 0
                ),
                "merge_attempt_alarm": False,
                "merge_attempt_warning": None,
                "merge_conflict": False,
            },
        )

    def _merge_deferred_stale_base_result(
        self,
        pr_number: int,
        issue_number: int | None,
        decision: dict[str, Any],
        base_ref: str | None,
        head_sha: str | None,
        reason: str = "base_stale",
    ) -> CommandResult:
        """Return a non-mergeable result for an approved PR whose base is stale.

        Records a ``merge_deferred_stale_base`` event so operators can see that
        the merge was deferred because the PR's merge-base is not the current
        base branch tip.
        """
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing = state["prs"].get(str(pr_number), {})
            new_stale_base_deferrals = int(existing.get("consecutive_stale_base_deferrals", 0)) + 1
            threshold = self.config.auto_merge.failed_attempt_alarm
            stale_base_alarm = threshold > 0 and new_stale_base_deferrals == threshold
            stale_base_warning: str | None = None
            if stale_base_alarm:
                stale_base_warning = _format_stale_base_alarm_message(
                    pr_number, new_stale_base_deferrals, reason
                )
                state = append_event(
                    state,
                    "merge_deferred_stale_base_alarm",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "base_ref": base_ref,
                        "head_sha": head_sha,
                        "reason": reason,
                        "attempts": new_stale_base_deferrals,
                        "threshold": threshold,
                        "message": stale_base_warning,
                    },
                    state_path=self.paths.state_file,
                )
            state = append_event(
                state,
                "merge_deferred_stale_base",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "base_ref": base_ref,
                    "head_sha": head_sha,
                    "reason": reason,
                },
                state_path=self.paths.state_file,
            )
            state["prs"][str(pr_number)] = {
                **existing,
                "number": pr_number,
                "issue_number": issue_number,
                "consecutive_stale_base_deferrals": new_stale_base_deferrals,
            }
            save_state(self.paths.state_file, state)
        return CommandResult(
            True,
            f"PR #{pr_number} base is stale; merge deferred until base is current",
            {
                "pr": pr_number,
                "issue": issue_number,
                "can_merge": False,
                "auto_merge_enabled": self.config.auto_merge.enabled,
                "merged": False,
                "merge_output": None,
                "branch_deleted": None,
                "review_decision": decision,
                "checks": asdict(summarize_checks([], self.config.auto_merge.required_checks)),
                "checks_unavailable": False,
                "label_error": None,
                "update_open_prs_results": None,
                "cancel_superseded_runs_results": None,
                "containment_warnings": [],
                "stale_base": True,
                "consecutive_failed_merge_attempts": existing.get(
                    "consecutive_failed_merge_attempts", 0
                ),
                "consecutive_stale_base_deferrals": new_stale_base_deferrals,
                "merge_attempt_alarm": stale_base_alarm,
                "merge_attempt_warning": stale_base_warning,
                "merge_conflict": False,
            },
        )

    def _is_merge_conflict(self, pr: dict[str, Any]) -> bool:
        """Detect a genuine content conflict that gh pr update-branch cannot resolve.

        GitHub exposes this through ``mergeable`` (CONFLICTING) and through
        ``mergeStateStatus`` (DIRTY). Both are already fetched by ``pr_view``.
        """
        return (
            str(pr.get("mergeable") or "").upper() == "CONFLICTING"
            or str(pr.get("mergeStateStatus") or "").upper() == "DIRTY"
        )

    def _route_to_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        summary: str,
        event_kind: str,
        extra_payload: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Route an approved PR to rework with a custom summary and event kind.

        Writes a rework prompt, transitions the linked issue to
        ``rework_requested`` (same label set as a non-escalated request_changes),
        and appends the requested state event. The PR's ``review-decision.json``
        is intentionally left untouched so the approved verdict is re-confirmed
        after the worker push moves the head. Any durable review fields already
        present in the PR state are preserved, and callers may pass additional
        keys to record without overwriting those verdict fields.
        """
        pr_number = int(pr["number"])
        self._write_rework_prompt(pr, issue_number, summary)

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            issue_key = str(issue_number)
            issue_entry = state["issues"].get(issue_key, {})
            state["issues"][issue_key] = {
                **issue_entry,
                "number": issue_number,
                "status": "rework_requested",
                "merge_alert": "OK",
            }
            pr_entry = state["prs"].get(str(pr_number), {})
            state["prs"][str(pr_number)] = {
                **pr_entry,
                "number": pr_number,
                "issue_number": issue_number,
                "status": "rework_requested",
                "decision": pr_entry.get("decision"),
                "reviewed_head_sha": pr_entry.get("reviewed_head_sha"),
                "reviewed_patch_id": pr_entry.get("reviewed_patch_id"),
                **(extra_state or {}),
            }
            payload = {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "base_ref": pr.get("baseRefName"),
                "head_sha": pr.get("headRefOid"),
                "reviewed_head_sha": decision.get("reviewed_head_sha"),
            }
            if extra_payload:
                payload.update(extra_payload)
            state = self._record_event(state, event_kind, payload)
            save_state(self.paths.state_file, state)

        result = transition(self.gh, self.config.labels, issue_number, "rework_requested")
        if result.outcome == TransitionOutcome.APPLIED:
            return None
        return {
            "edge": "rework_requested",
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        }

    def _request_merge_conflict_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Route a PR with a genuine merge conflict to rework.

        Called from two sites: merge_ready's approved+alarm-threshold path
        (the original use -- ``decision == "approved"`` is enforced by that
        caller, not here) and review()'s janitor gate
        (``_route_janitor_gate_failure_to_rework``), which is decision-
        agnostic -- a conflicting branch needs a rebase regardless of its
        review verdict -- and passes ``extra_state`` to thread its own
        attempt counter through the same state write.
        """
        summary = (
            "The PR branch has a merge conflict with the base branch after a base update. "
            "Merge the base branch into the PR branch, resolve the conflicts, and push."
        )
        if decision.get("decision") == "approved":
            summary += " The code changes are already approved; do not re-litigate the review."
        requested_at = utc_now()
        merged_extra_state = {"conflict_rework_requested_at": requested_at}
        if extra_state:
            merged_extra_state.update(extra_state)
        return self._route_to_rework(
            pr,
            issue_number,
            decision,
            summary,
            "merge_conflict_rework_requested",
            extra_payload={"conflict_rework_requested_at": requested_at},
            extra_state=merged_extra_state,
        )

    def _request_check_failure_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        summary: CheckSummary,
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Route an approved PR whose required checks have genuinely failed to rework.

        Called only from merge_ready's approved+alarm-threshold path (issue
        #674), mirroring ``_request_merge_conflict_rework``. Once a PR is
        approved and its head stops moving, loop()'s already_approved fast
        path never calls review() again -- it goes straight to merge_ready --
        so a required check that starts genuinely failing after approval
        (not merely pending/missing/infra_failed/unavailable) previously had
        no path back to rework at all. Debounced to the same
        failed-attempt-alarm threshold as merge-conflict rework so a
        transient failure or the mergequeue's own speculative-merge retry
        gets a chance to self-heal first.
        """
        failed_str = ", ".join(summary.failed)
        text = (
            f"CI failed on required check(s) after approval: {failed_str}. The code "
            "changes are already approved; do not re-litigate the review -- push a fix "
            "for the failing check(s)."
        )
        requested_at = utc_now()
        merged_extra_state = {"check_failure_rework_requested_at": requested_at}
        if extra_state:
            merged_extra_state.update(extra_state)
        return self._route_to_rework(
            pr,
            issue_number,
            decision,
            text,
            "check_failure_rework_requested",
            extra_payload={
                "check_failure_rework_requested_at": requested_at,
                "failed_checks": list(summary.failed),
            },
            extra_state=merged_extra_state,
        )

    def _request_no_op_rework_repair(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Route a PR whose last rework cycle pushed no actual change to rework.

        The janitor's no-op-rework check (``janitor._check_no_op_rework``)
        only detects the condition (unchanged patch-id/head since the last
        request_changes verdict); nothing previously consumed it
        (pr-lifecycle.md Finding 1's "no-op-rework-never-escalated"
        sub-case). This is that consumer, called from
        ``_route_janitor_gate_failure_to_rework`` and shaped like
        ``_request_merge_conflict_rework``.
        """
        summary = (
            "The previous rework cycle produced no actual content change (the diff or head "
            "matches the last request_changes verdict). Check the branch worktree for "
            "unpushed commits and push the real fix, or explain in the PR body why no "
            "further change was needed."
        )
        return self._route_to_rework(
            pr,
            issue_number,
            decision,
            summary,
            "no_op_rework_repair_requested",
            extra_state=extra_state,
        )

    def _route_janitor_gate_failure_to_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        *,
        attempts_key: str,
        max_attempts: int,
        reason: str,
        router: Callable[..., dict[str, Any] | None],
    ) -> CommandResult | None:
        """Shared cap/escalation wrapper for janitor-gate rework routing.

        ``router`` is ``_request_merge_conflict_rework`` or
        ``_request_no_op_rework_repair`` -- both ultimately call
        ``_route_to_rework`` and return ``None`` on a clean label transition
        or a ``label_error`` dict on partial failure.

        Returns ``None`` (caller falls through to the passive janitor_blocked
        wait) when a rework for the issue is already pending
        (``rework_requested``, ``dispatched``, or the two-phase-claim crash
        window ``dispatch_pending``): the janitor re-detects the same
        conflict/no-op every pass, and review() is re-invoked every pass, so
        routing -- and burning an attempt -- on each detection would escalate
        a PR whose rework worker simply hasn't run yet within two loop
        passes. Attempts must count completed-but-still-failing rework
        CYCLES, not loop passes and not individual pushes. For
        ``merge_conflict`` the cycle signal is a SETTLED head change: the
        issue must be back in ``rework_requested`` (a live ``dispatched``
        session may push any number of intermediate WIP commits -- burning
        per push would escalate a PR whose worker is actively fixing it) and
        the head must differ from the recorded
        ``<attempts_key>_last_head`` baseline. A truthy baseline is
        required: with no baseline on record (the rework predates this
        bookkeeping, or the head was transiently unavailable at routing
        time) the head is recorded as the new baseline instead of guessing
        that a cycle completed. A qualifying settled change burns one
        attempt (bounding the push-conflicted-heads-forever loop) but does
        not re-route -- the issue is already queued and the worktree layer
        injects the conflict notice at (re)launch. For ``no_op_rework``,
        head unchanged is the detection signal itself, so pending cycles
        are instead bounded by dispatch_rework's dead-session redispatch
        cap.

        Once ``max_attempts`` is exceeded, escalate using the same
        ``transition()`` helper the other escalation call sites use so
        ``agent:human-needed`` actually lands (pr-lifecycle.md Finding 3: the
        review-dispatch attempt-cap escalation was the one call site that
        skipped this).
        """
        pr_number = int(pr["number"])
        head_sha = str(pr.get("headRefOid") or "")
        last_head_key = f"{attempts_key}_last_head"
        stall_since_key = f"{attempts_key}_stall_since"
        stall_head_key = f"{attempts_key}_stall_head"
        # The attempt count is read here but persisted under a later, separate
        # lock (this write for the burn path, _route_to_rework's for the
        # routing path) -- a deliberate deviation from the single-lock RMW
        # pattern used elsewhere: two overlapping passes on the same PR can
        # at worst under-count by one, which only DELAYS escalation by one
        # cycle and self-corrects on the next pass. Making the routing path
        # atomic would mean threading a counter callback through
        # _route_to_rework's shared state write; not worth it for that
        # failure direction.
        snapshot = load_state_locked(self.paths.state_file)
        existing_pr_state = snapshot.get("prs", {}).get(str(pr_number), {})
        issue_status = snapshot.get("issues", {}).get(str(issue_number), {}).get("status")
        rework_pending = issue_status in ("rework_requested", "dispatched", "dispatch_pending")
        counted_head = existing_pr_state.get(last_head_key)
        if rework_pending:
            settled_new_conflicted_head = (
                reason == "merge_conflict"
                and issue_status == "rework_requested"
                and bool(head_sha)
                and head_sha != counted_head
            )
            if not settled_new_conflicted_head:
                # Issue #765: "no progress at all" and "progress still in
                # flight" both land here (a live dispatched session's WIP
                # pushes, and a stalled rework_requested queue item with an
                # unmoved head, are indistinguishable from settled_new_
                # conflicted_head's point of view). The cap/rescue check
                # below is unreachable from here -- it only runs once a
                # cycle SETTLES -- so a PR that never settles can wait here
                # forever. _check_janitor_rework_stall separates the two
                # cases, but NOT by issue status alone -- reconcile's
                # issue_active_label_with_open_pr self-heal periodically
                # flips issue_status away from rework_requested without the
                # head moving, so status is only used to decide whether to
                # ADVANCE the clock; only a head change (or escalation)
                # clears it.
                stall_result = self._check_janitor_rework_stall(
                    pr_number=pr_number,
                    issue_number=issue_number,
                    issue_status=issue_status,
                    head_sha=head_sha,
                    reason=reason,
                    attempts_key=attempts_key,
                    stall_since_key=stall_since_key,
                    stall_head_key=stall_head_key,
                    stall_since=existing_pr_state.get(stall_since_key),
                    stall_head=existing_pr_state.get(stall_head_key),
                )
                if stall_result is not None:
                    return stall_result
                return None
            if not counted_head:
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state["prs"][str(pr_number)] = {
                        **state["prs"].get(str(pr_number), {}),
                        "number": pr_number,
                        "issue_number": issue_number,
                        last_head_key: head_sha,
                        # Defensive: the stall clock is only ever set from
                        # the not-settled branch above, so this should
                        # already be unset -- but a head that was
                        # transiently unavailable (see settled_new_
                        # conflicted_head's bool(head_sha) guard) could have
                        # started the clock before this baseline existed.
                        stall_since_key: None,
                        stall_head_key: None,
                    }
                    save_state(self.paths.state_file, state)
                return None
        attempts = int(existing_pr_state.get(attempts_key, 0)) + 1

        if (
            max_attempts > 0
            and attempts > max_attempts
            and self.config.rescue.enabled
            and not existing_pr_state.get("rescue_attempted")
        ):
            # Rescue tier (issue #555): conflict-rework and no-op-rework caps
            # are both eligible ("cheap model wasn't good enough") causes.
            # Route to a bounded Opus rework via the SAME router this
            # function already uses for the non-cap-exceeded routing path
            # below, instead of escalating — never a second rescue for the
            # same PR (rescue_attempted is durable, cleared only by
            # `charlie unescalate`).
            #
            # Ordering note (reviewed, deliberate): the marker is committed
            # under this lock BEFORE `router(...)` runs its own separate
            # state_lock write below. A crash in that narrow window leaves
            # `rescue_attempted=True` with no rework actually dispatched —
            # the PR's one rescue slot is spent for nothing and the NEXT cap
            # exceedance escalates straight to a human instead of retrying
            # the rescue. This is intentional, not an oversight: the
            # alternative (write the marker AFTER `router()` succeeds) fails
            # the other way — a crash between a successful `router()` write
            # and the marker write leaves the PR eligible for a SECOND
            # rescue attempt on the next cap exceedance, violating the "one
            # rescue per PR" invariant the whole feature is built around
            # (issue #555: "no upward-migrating cost spiral"). Under-rescue
            # (safe, wasteful) is a strictly better failure mode than
            # over-rescue (invariant violation) here, so marker-first is
            # kept. This mirrors this same function's pre-existing,
            # explicitly-accepted two-lock deviation for `attempts_key`
            # itself (see the docstring above and the comment at the
            # `snapshot = load_state_locked(...)` read a few lines up) —
            # not a new risk introduced by the rescue tier, the same
            # "self-corrects, at worst delays by one cycle" tradeoff this
            # function already makes elsewhere. Making this fully atomic
            # would require plumbing an extra_payload-style rescue-marker
            # parameter through `_request_merge_conflict_rework`/
            # `_request_no_op_rework_repair` (both also called from
            # merge_ready's approved+alarm-threshold path with no rescue
            # concept), which is a larger blast-radius change than this
            # fix-forward warrants; revisit only if the wasted-slot case is
            # ever observed live.
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    attempts_key: attempts,
                    **rescue_helpers.build_rescue_dataclass_kwargs(reason),
                    "rescue_dispatched_at": utc_now(),
                }
                state = self._record_event(
                    state,
                    "rescue_dispatched",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "cause": reason,
                        "attempts": attempts,
                    },
                )
                save_state(self.paths.state_file, state)
            decision = self._review_decision(pr_number)
            route_extra_state: dict[str, Any] = {attempts_key: attempts}
            if head_sha:
                route_extra_state[last_head_key] = head_sha
            label_error = router(pr, issue_number, decision, extra_state=route_extra_state)
            return CommandResult(
                True,
                f"PR #{pr_number} janitor {reason} rework cap exceeded "
                f"({attempts}/{max_attempts}); rescue tier dispatched instead of escalating",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "janitor_ok": False,
                    "routed_to_rework": True,
                    "rescue_dispatched": True,
                    "rework_reason": reason,
                    "label_error": label_error,
                },
            )

        if max_attempts > 0 and attempts > max_attempts:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    "status": "escalated",
                    attempts_key: attempts,
                }
                state["issues"][str(issue_number)] = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "escalated",
                    "merge_alert": "OK",
                }
                state = self._record_event(
                    state,
                    "janitor_rework_escalated",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "reason": reason,
                        "attempts": attempts,
                    },
                )
                save_state(self.paths.state_file, state)
            result = transition(self.gh, self.config.labels, issue_number, "escalated")
            label_error = None
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": "escalated",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
            return CommandResult(
                False,
                f"PR #{pr_number} janitor {reason} rework cap exceeded "
                f"({attempts}/{max_attempts}); escalated",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "janitor_ok": False,
                    "escalated": True,
                    "label_error": label_error,
                },
            )

        if rework_pending:
            # merge_conflict with a settled head that moved off the recorded
            # baseline: a rework cycle completed without clearing the
            # conflict. Burn the attempt so cycles stay bounded, but the
            # issue is already queued for rework -- record and fall through
            # to the passive janitor_blocked wait.
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    attempts_key: attempts,
                    last_head_key: head_sha,
                    # Issue #765: the head moved -- real progress -- so any
                    # stall clock started while it was stuck no longer
                    # applies. A later re-stall must count from zero, not
                    # from a timestamp accumulated during unrelated history.
                    stall_since_key: None,
                    stall_head_key: None,
                }
                state = self._record_event(
                    state,
                    "janitor_rework_cycle_failed",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "reason": reason,
                        "attempts": attempts,
                        "head_sha": head_sha,
                    },
                )
                save_state(self.paths.state_file, state)
            return None

        decision = self._review_decision(pr_number)
        # Issue #765: no unconditional stall_since_key/stall_head_key clear
        # here. A genuinely fresh rework request has no clock running yet, so
        # omitting the keys is a no-op; a reconcile-induced re-route (the
        # head is unchanged, only issue_status flipped and flipped back) must
        # NOT lose already-accumulated stall time, since that would let
        # reconcile's unrelated label self-heal silently defeat this bound.
        route_extra_state: dict[str, Any] = {attempts_key: attempts}
        if head_sha:
            route_extra_state[last_head_key] = head_sha
        label_error = router(pr, issue_number, decision, extra_state=route_extra_state)
        return CommandResult(
            True,
            f"PR #{pr_number} routed to rework ({reason}, attempt {attempts}/{max_attempts})",
            {
                "pr": pr_number,
                "issue": issue_number,
                "routed_to_rework": True,
                "rework_reason": reason,
                "label_error": label_error,
            },
        )

    def _check_janitor_rework_stall(
        self,
        *,
        pr_number: int,
        issue_number: int,
        issue_status: str | None,
        head_sha: str,
        reason: str,
        attempts_key: str,
        stall_since_key: str,
        stall_head_key: str,
        stall_since: Any,
        stall_head: Any,
    ) -> CommandResult | None:
        """Stall bound orthogonal to the settled-head signal (issue #765).

        Called only from ``_route_janitor_gate_failure_to_rework``'s
        not-settled branch, where a rework is pending but has not produced
        that function's own progress signal. That combination is ambiguous
        on its own -- it covers both a live ``dispatched``/``dispatch_
        pending`` session still doing real work (must NOT escalate here; a
        genuinely dead session is WatchdogConfig.stall_minutes's job, not
        this function's) and a PR that is queued with nobody working it at
        all (``rework_requested``, no progress). The latter can never reach
        the cap/rescue check in the caller, because that check only runs
        once a cycle SETTLES, and a stalled PR's cycle never settles -- so
        without this, it waits here forever (issue #765: PR #696, 55
        ``rework_already_pushed`` events, attempts already at cap, rescue
        never attempted).

        HOLD vs. RESET is deliberately split on two different signals:

        - Only a HEAD CHANGE since the clock started clears it. This is
          checked regardless of ``issue_status`` -- a live ``dispatched``
          session pushing WIP commits is real progress and must reset the
          clock even though it isn't the "stalled" status itself, and (the
          motivating case, found while building this fix) reconcile's
          ``issue_active_label_with_open_pr`` self-heal flips ``issue_status``
          away from ``rework_requested`` on its own periodic cadence
          *without touching the branch* -- an earlier draft keyed the clear
          on status alone, which reconcile would have reset roughly every
          reconcile pass, making the bound practically unreachable whenever
          reconcile is enabled (the common case). Head-only clearing makes
          the clock immune to that oscillation.
        - Only ``issue_status == "rework_requested"`` ADVANCES the clock
          (starts it, or lets elapsed time accrue toward the threshold). Any
          other pending status (``dispatched``, ``dispatch_pending``, or a
          transient reconcile-driven flip with the head unchanged) HOLDS --
          returns ``None`` without escalating and without touching the
          clock, so already-accumulated stall time survives status churn
          that isn't accompanied by real progress.

        Returns ``None`` while waiting/holding, or after clearing a clock
        whose head moved. Returns a terminal ``CommandResult`` --
        ``agent:human-needed`` applied via the same ``transition()`` helper
        the cap-exceeded path uses -- once the threshold is exceeded.
        """
        progressed = (
            stall_since is not None
            and stall_head is not None
            and bool(head_sha)
            and head_sha != stall_head
        )
        if progressed:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    stall_since_key: None,
                    stall_head_key: None,
                }
                save_state(self.paths.state_file, state)
            stall_since = None
            stall_head = None

        stalled_candidate = issue_status == "rework_requested"
        if not stalled_candidate:
            # HOLD: do not clear. Only a head change (above) clears the
            # clock; a live worker or a transient status flip with the head
            # unchanged must not lose accumulated stall time.
            return None

        started = _parse_iso_timestamp(stall_since) if stall_since is not None else None
        if started is None or stall_head is None:
            # (Re)start the clock. Covers: no clock running yet; an
            # unparseable/corrupt timestamp; and a clock inherited from
            # before this bound had a head anchor at all (stall_since set,
            # stall_head missing -- pre-redesign state, or any other write
            # path that recorded one key without the other). In that last
            # case we cannot tell whether the head moved during the
            # already-elapsed time, so we discard it and re-anchor rather
            # than trust elapsed time of unknown provenance -- the safe
            # direction on rollout is one extra window before the bound can
            # fire, never a spurious escalation from state that never
            # actually observed a stalled head.
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    stall_since_key: utc_now(),
                    stall_head_key: head_sha or None,
                }
                save_state(self.paths.state_file, state)
            return None

        threshold_minutes = self.config.review.rework_stall_minutes
        if threshold_minutes <= 0:
            return None
        elapsed_minutes = (datetime.now(UTC) - started).total_seconds() / 60
        if elapsed_minutes < threshold_minutes:
            return None

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            attempts_so_far = int(state["prs"].get(str(pr_number), {}).get(attempts_key, 0))
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                "status": "escalated",
                stall_since_key: None,
                stall_head_key: None,
            }
            state["issues"][str(issue_number)] = {
                **state["issues"].get(str(issue_number), {}),
                "number": issue_number,
                "status": "escalated",
                "merge_alert": "OK",
            }
            state = self._record_event(
                state,
                "janitor_rework_stalled",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "reason": reason,
                    "attempts": attempts_so_far,
                    "stalled_minutes": round(elapsed_minutes, 1),
                    "stall_since": stall_since,
                    "head_sha": head_sha,
                },
            )
            save_state(self.paths.state_file, state)
        result = transition(self.gh, self.config.labels, issue_number, "escalated")
        label_error = None
        if result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": "escalated",
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
        return CommandResult(
            False,
            f"PR #{pr_number} janitor {reason} rework stalled "
            f"({round(elapsed_minutes)}m with no progress while {issue_status}); escalated",
            {
                "pr": pr_number,
                "issue": issue_number,
                "janitor_ok": False,
                "escalated": True,
                "escalation_reason": "stalled",
                "label_error": label_error,
            },
        )

    def _request_readiness_no_ci_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        missing_checks: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """Route an approved PR whose required CI checks have not started to rework."""
        summary = (
            "The PR was approved but the required CI checks have not started after the "
            f"configured readiness timeout ({self.config.auto_merge.readiness_no_ci_minutes} minutes). "
            f"Required checks: {', '.join(missing_checks)}. Push an empty commit or amend the head "
            "to re-trigger CI. The code changes are already approved; do not re-litigate the review."
        )
        requested_at = utc_now()
        return self._route_to_rework(
            pr,
            issue_number,
            decision,
            summary,
            "readiness_no_ci_rework_requested",
            extra_payload={
                "missing_checks": missing_checks,
                "readiness_no_ci_rework_requested_at": requested_at,
            },
            extra_state={
                "readiness_no_ci_rework_requested_at": requested_at,
            },
        )

    def _request_cross_pr_revert_rework(
        self,
        pr: dict[str, Any],
        issue_number: int,
        decision: dict[str, Any],
        reason: str,
    ) -> dict[str, Any] | None:
        """Route an approved PR whose branch silently reverts a base commit to rework."""
        summary = (
            f"{reason}. Remove the revert commit (or the merge+revert pair) from the PR "
            "history, or add an explicit 'allow-revert: <reason>' line to the PR body if the "
            "revert is intentional. Then push the corrected branch and re-request review."
        )
        return self._route_to_rework(
            pr, issue_number, decision, summary, "cross_pr_revert_rework_requested"
        )

    def _merge_train_head(self, prs: list[dict[str, Any]] | None = None) -> int | None:
        """Return the PR number of the head of the merge-train queue, or None.

        The head is the earliest approved-pending-ship PR (same-repo, matching
        the configured branch prefix) ordered by reviewed_at (falling back to
        updatedAt), then by PR number for determinism.
        """
        candidates = self._merge_train_candidates(prs=prs)
        return candidates[0][1] if candidates else None

    def _merge_train_candidates(
        self,
        prs: list[dict[str, Any]] | None = None,
        exclude_pr_number: int | None = None,
    ) -> list[tuple[str, int, dict[str, Any], dict[str, Any], str]]:
        """Return approved-pending-ship candidates sorted by approval time.

        Each tuple contains (sort_key, pr_number, pr, decision, head_ref).
        """
        if prs is None:
            prs = self.gh.pr_list()

        branch_prefix = self.config.dispatch.branch_prefix
        # Aviator MergeQueue handoff (task #10): a PR already parked in
        # Aviator's queue (state status "mergequeue") must never occupy
        # charlie's merge-train head — Aviator now owns serialization for it.
        # Without this exclusion the parked PR keeps winning "earliest
        # reviewed" on every poll until GitHub reports it merged, so under
        # front_of_train no other approved PR is ever attempted while Aviator
        # is still processing it. Reading state is only necessary when the
        # mergequeue handoff feature is actually configured.
        state_prs = (
            load_state_locked(self.paths.state_file).get("prs", {})
            if self.config.auto_merge.mergequeue_label
            else {}
        )
        candidates: list[tuple[str, int, dict[str, Any], dict[str, Any], str]] = []
        for pr in prs:
            pr_number = int(pr.get("number", 0))
            if pr_number == exclude_pr_number:
                continue
            if pr.get("isCrossRepository"):
                continue
            head = str(pr.get("headRefName") or "")
            if not head.startswith(branch_prefix):
                continue
            if (state_prs.get(str(pr_number)) or {}).get("status") == "mergequeue":
                continue
            decision = self._review_decision(pr_number)
            if decision.get("decision") != "approved":
                continue
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            if reviewed_head_sha is None or live_head_sha != reviewed_head_sha:
                continue
            reviewed_at = decision.get("reviewed_at") or pr.get("updatedAt") or ""
            candidates.append((str(reviewed_at), pr_number, pr, decision, head))

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def _check_carry_forward(self, pr_number: int, decision: dict[str, Any]) -> CarryForwardCheck:
        """Determine whether ``decision``'s verdict can carry forward to the
        PR's live head, and via which tier (issues #411/#412, #414).

        Tier 1 (fast path): the live diff's stable patch-id equals the
        recorded ``reviewed_patch_id`` — the effective content is provably
        identical modulo hunk-context.

        Tier 2 (line-content, issue #414): patch-ids differ — which happens
        on every ordinary main advance, since ``git patch-id --stable``
        hashes hunk-boundary context and the merge-base moves whenever main
        does — but the ordered ``+``/``-`` line stream and changed-file set
        recorded at review time are identical to the live diff's. Reordered,
        added, removed, or altered lines, or a changed file set, are real
        content changes and do NOT carry forward. Tier 2 is INELIGIBLE
        whenever either side's diff touched a binary file: a binary payload
        emits no ``+``/``-`` lines, so the signature is blind to it — two
        diffs with genuinely different binary content at the same path
        would otherwise compare equal (review follow-up on issue #414).

        Fails closed (``tier=None``) on any missing data, diff-fetch
        failure, or binary content: a decision recorded before tier-2
        existed (no signature stored), a PR whose diff cannot be fetched,
        a binary file on either side, or a genuine content difference all
        report "cannot carry forward" — never carry forward on
        uncertainty. Tier 2 is pure string parsing of the diff text already
        fetched for tier 1 — it needs no additional git/gh calls and so has
        no failure mode of its own beyond that shared fetch.

        Eligibility for BOTH tiers is gated on ``reviewed_patch_id`` being
        recorded at all (matching #412's original behavior exactly): a
        "blocked" verdict, or any other decision that never computed one,
        has no baseline to compare against, full stop. A pure-rename or
        mode-only diff also has an empty ``reviewed_patch_id`` (no ``@@``
        hunk) despite having a valid tier-2 signature on file — that
        specific case is intentionally left conservative (stays stale)
        rather than gating on the signature fields' presence instead, which
        was tried and reverted: ``record_review`` unconditionally records a
        (possibly trivially-empty) signature for every approved/
        request_changes decision, so gating on "signature present" instead
        of "patch-id present" made an unrelated placeholder/no-op diff look
        like a valid tier-2 baseline and wrongly carried forward verdicts
        whose head had genuinely moved to unrelated content. Tracked as a
        narrow follow-up, not fixed here.
        """
        live_diff = self.gh.pr_diff(pr_number) or ""
        if not live_diff:
            return CarryForwardCheck(None, "", DiffContentSignature((), frozenset()))

        live_patch_id = _calculate_patch_id(live_diff)
        live_signature = _diff_content_signature(live_diff)

        reviewed_patch_id = decision.get("reviewed_patch_id") or ""
        if not reviewed_patch_id:
            # No baseline recorded at all (e.g. a "blocked" verdict never
            # computes a patch-id) — nothing to compare against.
            return CarryForwardCheck(None, live_patch_id, live_signature)

        if live_patch_id and live_patch_id == reviewed_patch_id:
            return CarryForwardCheck("patch-id", live_patch_id, live_signature)

        reviewed_changed_lines = decision.get("reviewed_changed_lines")
        reviewed_changed_files = decision.get("reviewed_changed_files")
        if reviewed_changed_lines is None or reviewed_changed_files is None:
            # Decision predates tier-2 (no signature recorded) — cannot
            # establish content identity; fail closed to stale.
            return CarryForwardCheck(None, live_patch_id, live_signature)

        if decision.get("reviewed_has_binary") or live_signature.has_binary:
            # A binary payload emits no +/- content lines, so the signature
            # cannot see it — never rely on its silence for content it
            # never observed (issue #414 review follow-up).
            return CarryForwardCheck(None, live_patch_id, live_signature)

        lines_match = tuple(reviewed_changed_lines) == live_signature.changed_lines
        files_match = frozenset(reviewed_changed_files) == live_signature.changed_files
        if lines_match and files_match:
            return CarryForwardCheck("line-content", live_patch_id, live_signature)

        return CarryForwardCheck(None, live_patch_id, live_signature)

    def _update_approval_head(
        self,
        pr_number: int,
        decision: dict[str, Any],
        new_head: str,
        old_head: str | None = None,
        *,
        issue_number: int | None = None,
        tier: str = "verified-sync",
        new_patch_id: str | None = None,
        new_signature: DiffContentSignature | None = None,
    ) -> None:
        """Persist an updated review head for a PR whose branch was synced.

        Keeps the verdict valid when the branch was base-updated or rebased
        without content changes. Updates both review-decision.json and
        state.json, appending the old head to ``carried_forward_from`` for audit.

        ``tier`` records which mechanism justified the update, for audit:
        ``"patch-id"``/``"line-content"`` from :meth:`_check_carry_forward`
        (issues #412/#414), or the default ``"verified-sync"`` for the
        structurally-verified (``_verify_synced_head``) front-of-train/
        broadcast sync callers, which never compared patch-ids at all.

        When ``new_patch_id``/``new_signature`` are supplied — they are
        already computed as part of the tier check that led here — they
        replace the stored baseline so ``reviewed_head_sha`` and the
        patch-id/content-signature fields always describe the SAME head
        consistently. Leaving a patch-id recorded against a stale head would
        pointlessly defeat that head's own future tier-1 fast path: patch-id
        is unstable across every main advance, not just the one just
        carried past.

        A carry-forward event is recorded here, inside the locked section
        that already persists the transition, so every call site is
        instrumented by construction — a new caller cannot forget it
        (issue #638). The event kind is tier-dependent so the three
        mechanisms (``verdict_carried_forward_clean_rebase`` for patch-id,
        ``verdict_carried_forward_line_content`` for line-content, and
        ``verdict_carried_forward_verified_sync`` for the structurally-
        verified sync that never compared patch-ids) stay separately
        auditable in the events log. Callers that previously recorded the
        event themselves must NOT do so anymore, or the transition would be
        double-counted.
        """
        decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
        updated_decision = dict(decision)
        updated_decision["reviewed_head_sha"] = new_head
        updated_decision["carry_forward_tier"] = tier
        if new_patch_id is not None:
            updated_decision["reviewed_patch_id"] = new_patch_id
        if new_signature is not None:
            updated_decision["reviewed_changed_lines"] = list(new_signature.changed_lines)
            updated_decision["reviewed_changed_files"] = sorted(new_signature.changed_files)
            updated_decision["reviewed_has_binary"] = new_signature.has_binary
        carried_forward: list[str] = list(updated_decision.get("carried_forward_from", []))
        if old_head is not None and old_head != new_head and old_head not in carried_forward:
            carried_forward.append(old_head)
        updated_decision["carried_forward_from"] = carried_forward
        self._write_json(decision_path, updated_decision)

        if tier == "patch-id":
            event_kind = "verdict_carried_forward_clean_rebase"
        elif tier == "line-content":
            event_kind = "verdict_carried_forward_line_content"
        else:
            event_kind = "verdict_carried_forward_verified_sync"

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "decision": pr_state.get("decision") or decision.get("decision") or "approved",
                "status": pr_state.get("status") or decision.get("decision") or "approved",
                "reviewed_head_sha": new_head,
                "reviewed_patch_id": new_patch_id
                if new_patch_id is not None
                else (
                    pr_state.get("reviewed_patch_id") or decision.get("reviewed_patch_id") or ""
                ),
                "carry_forward_tier": tier,
                "carried_forward_from": carried_forward,
            }
            state = self._record_event(
                state,
                event_kind,
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "old_reviewed_head_sha": old_head,
                    "new_head_sha": new_head,
                    "patch_id": new_patch_id,
                    "carry_forward_tier": tier,
                    "carried_forward_from": carried_forward,
                },
            )
            save_state(self.paths.state_file, state)

    def _verify_synced_head(self, pr_number: int, old_head_sha: str) -> str | None:
        """Verify that the new head of a PR is a valid base-sync merge commit.

        After ``gh pr update-branch`` advances the PR head, we must not bless the
        new SHA until we confirm it is a GitHub-generated merge commit (web-flow)
        whose parents include the previously approved head. This closes the
        approval-integrity TOCTOU: a racing push to the PR branch could otherwise
        be mistaken for a base update and auto-merged without review.
        """
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return None
        new_head_sha = pr.get("headRefOid")
        if not new_head_sha:
            return None
        if new_head_sha == old_head_sha:
            return new_head_sha

        commit = self.gh.commit(new_head_sha)
        if not commit:
            return None

        parents = commit.get("parents") or []
        if len(parents) != 2:
            return None
        parent_shas = [str(p.get("sha")) for p in parents if isinstance(p, dict)]
        if old_head_sha not in parent_shas:
            return None

        committer = commit.get("committer") or {}
        if not isinstance(committer, dict):
            committer = {}
        commit_committer = commit.get("commit", {}).get("committer") or {}
        if not isinstance(commit_committer, dict):
            commit_committer = {}
        committer_login = committer.get("login")
        committer_name = commit_committer.get("name")
        # Both identity signals must match a GitHub-generated merge. The git
        # metadata name is pusher-settable, and the account login can be
        # spoofed via the committer email, so accepting either alone would let
        # a crafted racing push get blessed as a base sync. Fail closed.
        if committer_login != "web-flow" or committer_name != "GitHub":
            return None

        return new_head_sha

    def _should_update_pr_branch(
        self,
        pr: dict[str, Any],
        base_current: bool | None | _BaseCurrentUnset = _BASE_CURRENT_UNSET,
    ) -> bool:
        """Return True if the PR branch should be synced against its base.

        When a ``base_current`` signal is supplied, it is authoritative:
        ``True`` means the branch is already up-to-date and no sync is needed;
        ``False`` means the branch is stale and should be synced; ``None`` means
        the compare API is unavailable, so fail-closed and do not sync.

        If no ``base_current`` signal is supplied, fall back to the legacy
        mergeStateStatus heuristic for backward compatibility.
        """
        if isinstance(base_current, _BaseCurrentUnset):
            status = str(pr.get("mergeStateStatus") or "").upper()
            return status not in {"CLEAN", "UNSTABLE", "HAS_HOOKS"}
        return base_current is False

    def _is_base_current(self, pr: dict[str, Any]) -> bool | None:
        """Return True if the PR's merge-base is the current tip of its base.

        Uses the GitHub compare API to derive ancestry rather than timestamps
        or mergeStateStatus, which can lag. Returns ``None`` when the comparison
        cannot be completed so callers can decide how to fail-safe.
        """
        base_ref = pr.get("baseRefName") or self.config.runners.default_branch
        head_sha = pr.get("headRefOid")
        if not base_ref or not head_sha:
            return None
        comparison = self.gh.compare(base_ref, head_sha)
        if not comparison:
            return None
        base_commit = comparison.get("base_commit")
        merge_base_commit = comparison.get("merge_base_commit")
        if not isinstance(base_commit, dict) or not isinstance(merge_base_commit, dict):
            return None
        base_sha = base_commit.get("sha")
        merge_base_sha = merge_base_commit.get("sha")
        if not base_sha or not merge_base_sha:
            return None
        return bool(base_sha == merge_base_sha)

    def _record_review_or_error(
        self,
        review_result: CommandResult,
        errors: list[dict[str, Any]],
        reviews: list[dict[str, Any]],
    ) -> bool:
        """Append a review result to reviews or errors if checks are unavailable.

        Returns True if the caller should continue to the next PR (the unavailable
        case has already been recorded as an error).
        """
        if review_result.data.get("checks_unavailable"):
            errors.append({"pr": review_result.data.get("pr"), "error": review_result.message})
            return True
        reviews.append(review_result.data)
        return False

    def _record_merge_or_error(
        self,
        merge_result: CommandResult,
        errors: list[dict[str, Any]],
        merges: list[dict[str, Any]],
    ) -> None:
        """Append a merge_ready result to merges or errors if a gh check failed."""
        if merge_result.data.get("checks_unavailable") or merge_result.data.get(
            "merge_hold_check_unavailable"
        ):
            errors.append({"pr": merge_result.data.get("pr"), "error": merge_result.message})
        else:
            merges.append(merge_result.data)

    def _mark_foreign_issue_ref(self, pr_number: int, issue_number: int, reason: str) -> bool:
        """Durably park a PR whose linked issue does not exist in this repo.

        A PR opened against the wrong fleet repo (e.g. its branch references
        another repo's issue number) can never be processed here: every pass
        would re-derive the same issue number and re-fail the same GitHub
        lookup forever. Persist a ``foreign_issue_ref`` marker in the PR's
        state entry so subsequent passes skip it with zero GitHub calls.

        Returns True only on the first marking for this (pr, issue) pair so
        the caller can emit a one-shot attention event.
        """
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            marker = pr_state.get("foreign_issue_ref") or {}
            if marker.get("issue") == issue_number:
                return False
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "foreign_issue_ref": {
                    "issue": issue_number,
                    "detected_at": utc_now(),
                    "reason": reason,
                },
            }
            save_state(self.paths.state_file, state)
        return True

    @_guard_state_lock
    def loop(self, limit: int | None = None, *, merge: bool | None = None) -> CommandResult:
        return self._loop_impl(limit, merge=merge)

    def _loop_impl(self, limit: int | None, *, merge: bool | None) -> CommandResult:
        # merge=False runs the full pass (intake, dispatch, reviews, readiness
        # evaluation + labels) but skips the actual `gh pr merge` — for
        # operators sequencing same-surface PR cascades by hand, where the
        # pr_list (newest-first) merge order would land PRs in the wrong order.
        loop_start = time.monotonic()
        with correlation_context() as cid:
            start_ts = utc_now()
            log_event(
                self.paths.state_file,
                "loop_started",
                {"limit": limit, "merge": merge},
                repo=self.repo_root.name,
                correlation_id=cid,
            )
            record_loop_pass(self.paths.state_file, cid, start_ts)
            result = self._loop_body(limit, merge=merge)
            elapsed = time.monotonic() - loop_start
            log_event(
                self.paths.state_file,
                "loop_completed",
                {
                    "ok": result.ok,
                    "message": result.message,
                    "elapsed_seconds": round(elapsed, 2),
                    "error_count": len(result.data.get("errors", [])),
                },
                repo=self.repo_root.name,
                correlation_id=cid,
            )
            record_loop_pass(
                self.paths.state_file,
                cid,
                start_ts,
                completed_at=utc_now(),
                ok=result.ok,
                elapsed_seconds=round(elapsed, 2),
                error_count=len(result.data.get("errors", [])),
                merge_count=len(result.data.get("merges", [])),
                review_count=len(result.data.get("reviews", [])),
            )
            return result

    def _maybe_probe_quota_recovery(self) -> None:
        """Flat-interval Haiku probe for early quota/rate-limit throttle recovery.

        Runs every loop pass but is a near-no-op unless a throttle that a
        green ambient-CLI probe could actually clear is currently active
        (``is_quota_probe_actionable``) -- an operator switching to a
        different subscription account can make a provider's quota recover
        well before a blanket cooldown (e.g. the 24h
        ``_DEFAULT_QUOTA_COOLDOWN_HOURS`` window) elapses, and there was
        previously no way to detect that early. Deliberately a *flat*
        ``quota_probe.interval_minutes`` schedule (not exponential backoff
        like ``reviewer_quota.probe_after``): the user asked for a fixed
        15-minute recheck, not a growing wait.

        The gate is narrower than "any throttle indicator" -- a devin/api
        adapter throttle or a provider_auth cooldown would survive
        ``clear_quota_throttles`` untouched even on a green probe (see that
        function), so arming/probing for one would just burn Haiku sessions
        every interval for no possible benefit. Because of this, a green
        probe reaching the success branch below is guaranteed to have
        something to clear -- no post-hoc check needed there.

        Two-phase lock pattern so the (possibly tens-of-seconds) subprocess
        call in ``run_quota_probe`` never holds ``state_lock`` and blocks
        every other state read/writer in the process:
          1. Under the lock: decide whether to probe at all, and if not,
             arm/disarm/return without calling out to the CLI.
          2. Outside the lock: run the actual probe subprocess.
          3. Under the lock again: re-read state (it may have changed while
             unlocked) and apply the outcome.
        """
        if not self.config.quota_probe.enabled:
            return

        state_file = self.paths.state_file
        with state_lock(state_file):
            state = load_state(state_file)
            if not is_quota_probe_actionable(state):
                if is_quota_probe_armed(state):
                    state = disarm_quota_probe(state)
                    save_state(state_file, state)
                return
            if not is_quota_probe_armed(state):
                next_probe_at = (
                    (
                        datetime.now(UTC)
                        + timedelta(minutes=self.config.quota_probe.interval_minutes)
                    )
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                state = arm_quota_probe(state, next_probe_at)
                save_state(state_file, state)
                return
            if not is_quota_probe_due(state):
                return

        probe_ok = run_quota_probe(repo_root=self.repo_root, config=self.config)

        with state_lock(state_file):
            state = load_state(state_file)
            if not is_quota_probe_actionable(state):
                # Cleared or became non-actionable while the probe ran unlocked.
                state = disarm_quota_probe(state)
                save_state(state_file, state)
                return
            if probe_ok:
                state = clear_quota_throttles(state)
                state = disarm_quota_probe(state)
                state = self._record_event(state, "quota_probe_succeeded", {})
            else:
                next_probe_at = (
                    (
                        datetime.now(UTC)
                        + timedelta(minutes=self.config.quota_probe.interval_minutes)
                    )
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                state = arm_quota_probe(state, next_probe_at)
                # Share the failure with dispatch_reviews's probe_mode gate:
                # a red flat probe confirmed the window is still closed, so
                # also bump reviewer_quota.probe_after (when the reviewer
                # quota is exhausted) to stop dispatch_reviews from
                # independently launching a real reviewer session into the
                # same window on this or a nearby pass (issue #663).
                state = defer_reviewer_probe_after(state, next_probe_at)
                state = self._record_event(state, "quota_probe_failed", {})
            save_state(state_file, state)

    def _maybe_reclaim_worktrees(self) -> dict[str, Any] | None:
        """Cadence-gated merged-PR worktree reclamation on the fleet pass.

        Runs ``clean_worktrees`` -- the same junction-safe, merge-gated,
        liveness-gated sweep behind ``charlie worktree-clean`` -- on a flat
        ``worktree_reclamation.interval_minutes`` schedule. Before this call
        site the sweep only ran when an operator remembered the standalone
        subcommand, so worktrees for merged PRs accumulated indefinitely
        (issue #636: 77 of 81 dead on the host this was measured on).

        Two-phase lock pattern (same shape as ``_maybe_probe_quota_recovery``):
        the schedule is advanced under the lock and the sweep itself runs
        outside it, because ``clean_worktrees`` makes a live ``gh pr view`` call
        per candidate worktree and must not hold ``state_lock`` while it does --
        holding the lock across a per-candidate GitHub fan-out would block every
        other state reader/writer in the process for the sweep's duration.

        ``dry_run`` is threaded honestly: a ``--dry-run`` fleet pass runs the
        sweep in preview mode, which removes nothing (the preview-vs-act class
        tracked in #614-#619). A ``worktrees_reclaimed`` event is always emitted
        when the sweep runs, so a maintenance action that left no trace is
        indistinguishable from one that never ran (lesson from #595/#621).

        The cadence schedule itself is advanced regardless of ``dry_run``.
        This is deliberate, not an instance of the #614-#619 class:
        ``clean_worktrees`` makes its live ``gh pr view`` call per candidate
        unconditionally -- ``dry_run`` only gates the final ``git worktree
        remove`` -- so the GitHub-quota cost this interval exists to bound is
        identical in preview and live mode. Not advancing the schedule under
        ``dry_run`` would let a repeated preview pass re-run the full
        per-candidate fan-out every time, defeating the cadence gate.

        Returns a small summary dict when the sweep ran (for the loop result's
        ``data``), or ``None`` when reclamation is disabled or not due this
        pass.
        """
        if not self.config.worktree_reclamation.enabled:
            return None
        state_file = self.paths.state_file
        with state_lock(state_file):
            state = load_state(state_file)
            if not is_worktree_reclamation_due(state):
                return None
            # Advance the schedule BEFORE running the sweep so a concurrent
            # pass (or a sweep that takes longer than one poll interval) cannot
            # double-fire. The next run is interval_minutes away regardless of
            # how long the sweep itself takes.
            next_run_at = (
                (
                    datetime.now(UTC)
                    + timedelta(minutes=self.config.worktree_reclamation.interval_minutes)
                )
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            state = schedule_worktree_reclamation(state, next_run_at)
            save_state(state_file, state)

        # Use the same resolved worktrees root dispatch and `charlie
        # worktree-clean` use (self._layout.worktrees, from
        # paths.resolved_layout) rather than re-deriving the
        # claude_code.worktrees_dir/runtime.state_dir sentinel logic inline --
        # that duplication across call sites is the exact shape of bug
        # layout.py's module docstring documents as a past production
        # incident (create and sweep sides silently disagreeing on the root).
        state = load_state_locked(state_file)
        result = clean_worktrees(
            self.repo_root,
            self._layout.worktrees,
            state,
            self.config,
            self.gh,
            dry_run=self.dry_run,
        )
        orphans = result.data.get("orphans", {})
        summary = {
            "dry_run": self.dry_run,
            "ok": result.ok,
            "removed": len(result.data.get("removed", [])),
            "planned": len(result.data.get("planned", [])),
            "skipped": len(result.data.get("skipped", [])),
            "failed": len(result.data.get("failed", [])),
            "orphans_removed": len(orphans.get("removed", [])),
            "orphans_planned": len(orphans.get("planned", [])),
            "orphans_failed": len(orphans.get("failed", [])),
            "message": result.message,
        }
        with state_lock(state_file):
            state = load_state(state_file)
            state = self._record_event(state, "worktrees_reclaimed", summary)
            save_state(state_file, state)
        return summary

    def _maybe_reconcile_drift(self) -> None:
        """Periodic in-loop repair of GitHub label / state.json divergence.

        merge-lane-recovery plan §6-B / D-8. ``OrchestratorApp.reconcile()``
        (``detect_drift`` + ``apply_fixes``) was previously reachable only
        via the operator-invoked ``charlie mop-up --fix`` CLI command
        (``cli.py``'s ``mop-up`` handler) -- the fleet has never run its own
        repair. This wires it into the loop on a fixed cadence so a
        divergence like a failed escalation label write (the PRIMARY defect:
        ``status`` flips to ``escalated`` but the paired ``human_needed``
        label transition silently fails, leaving a stale ``needs-rework``
        label on a dead issue forever) is corrected automatically instead of
        only when an operator remembers to intervene.

        This method is wiring only -- it does not reimplement drift
        detection or repair. ``self._reconcile_locked(fix=True)`` owns that (and
        already threads ``state_path`` so ``apply_fixes`` emits one
        ``"reconcile"`` event per repaired drift item for free -- see
        ``reconcile.py``). It also owns the safety invariant this
        workstream exists to preserve: reconcile only ever converges labels
        *from* state, never the reverse -- an escalated issue's ``status``
        is never rewritten (D-2). This method does not touch ``status``.

        Calls with ``skip_dead_session_sweep=True``: this pass already ran
        the loop's own stall/dead lanes (``_detect_and_handle_stalled_sessions``
        / ``_classify_dead_sessions_and_update_throttle_state``) earlier in
        ``_loop_body``, with grace-period semantics
        (``max_inconclusive_probe_deferrals``) that reconcile.py's own
        dead-session sweep predates and does not implement. Without this,
        reconcile would re-scan the same sessions a few calls later and
        unconditionally reap any not-alive one, silently defeating that
        grace period every time this pass runs. See ``detect_drift``'s
        docstring in ``reconcile.py`` for the full rationale. Launch-stalled
        detection and live-session tracking (the other two things gated on
        ``repo_root`` in ``detect_drift``) are unaffected -- they have no
        counterpart in the loop's own lanes and keep running.

        Calls ``_reconcile_locked`` directly, NOT ``reconcile()`` -- this is
        the whole point of the D-8a split and reverting it silently disables
        this entire method. ``reconcile()`` acquires ``supervisor.lock``
        before doing anything else, and every production caller of ``loop()``
        already holds that exact lock for the full duration of the call:
        ``cli.py``'s ``bash-rats`` handler, ``fleet_dispatch.py``, and
        ``supervise.py`` (which holds it across *every* pass of its
        ``while True``). Byte-range locks taken via ``msvcrt.locking`` with
        ``LK_NBLCK`` are per-handle and non-reentrant *even within a single
        process* -- ``file_lock.py`` keeps no reentrancy bookkeeping -- so a
        second acquisition from inside the same process fails exactly like a
        foreign process's would. Going through ``reconcile()`` here therefore
        always took the ``supervisor_lock_held`` early return and never ran.

        Two-phase lock pattern, mirroring ``_maybe_probe_quota_recovery``:
        ``_reconcile_locked`` acquires ``state_lock`` itself internally to
        run drift detection/repair, and ``state_lock`` wraps a non-reentrant
        advisory file lock (a plain per-path ``threading.Lock``, not an
        ``RLock``) -- calling it while this method already held the same lock
        would deadlock. So:
          1. Under our own (short) lock: decide whether reconcile is due at
             all. If not, return without calling out.
          2. Outside any lock held by this method: call
             ``self._reconcile_locked(fix=True)``. This may still defer (the
             existing GraphQL rate-limit check) -- that path is preserved.
             It can no longer report ``supervisor_lock_held``, because it
             never attempts that acquisition; the lock is already held by
             ``loop()``'s caller, which is the precondition this method
             relies on rather than something it works around.
          3. Under our own (short) lock again: persist the next-due
             timestamp and emit exactly one summary event for this pass,
             shaped by the outcome (completed / deferred / failed) so a
             deferred or failed pass is distinguishable from a silent
             no-op rather than failing silently (D-8, B-AC3).

        The call is wrapped in exception containment, and that containment is
        load-bearing rather than defensive habit: ``supervise.py``'s
        ``except Exception`` sits *outside* its ``while True``, so a single
        uncaught exception from here terminates the whole daemon rather than
        one pass. ``_fetch_prs``/``_fetch_issues`` reach GitHub via
        ``gh.run(..., json_output=True)`` with ``allow_failure=False``, which
        *raises* ``GitHubError``/``GitHubNotFoundError`` once retries are
        exhausted -- so without this, a GitHub outage is a live daemon-kill
        path. A failed pass re-arms the timer like any other outcome, so a
        persistent failure degrades to one logged error per interval instead
        of a hot loop.
        """
        if not self.config.reconcile_pass.enabled:
            return

        state_file = self.paths.state_file
        with state_lock(state_file):
            state = load_state(state_file)
            if not is_reconcile_due(state):
                return

        next_reconcile_at = (
            (datetime.now(UTC) + timedelta(minutes=self.config.reconcile_pass.interval_minutes))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        # D-8a: _reconcile_locked, never reconcile(). See this method's
        # docstring -- reconcile() would re-acquire the supervisor lock that
        # loop()'s caller already holds and silently no-op every pass.
        try:
            result = self._reconcile_locked(fix=True, skip_dead_session_sweep=True)
        except Exception as exc:  # noqa: BLE001 - containment is deliberate; see docstring
            with state_lock(state_file):
                state = load_state(state_file)
                state = arm_reconcile_pass(state, next_reconcile_at)
                state = self._record_event(
                    state,
                    "reconcile_pass_failed",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                save_state(state_file, state)
            return

        with state_lock(state_file):
            state = load_state(state_file)
            state = arm_reconcile_pass(state, next_reconcile_at)
            data = result.data
            if data.get("skipped"):
                state = self._record_event(
                    state,
                    "reconcile_pass_skipped",
                    {"reason": data.get("reason")},
                )
            elif data.get("deferred"):
                state = self._record_event(
                    state,
                    "reconcile_pass_deferred",
                    {
                        "deferred_reason": data.get("deferred_reason"),
                        "graphql_remaining": data.get("graphql_remaining"),
                        "graphql_reset": data.get("graphql_reset"),
                        "graphql_threshold": data.get("graphql_threshold"),
                    },
                )
            else:
                drift_before = data.get("drift_before", 0)
                drift_after = data.get("drift_after", 0)
                state = self._record_event(
                    state,
                    "reconcile_pass_completed",
                    {
                        "ok": result.ok,
                        "drift_detected": drift_before,
                        "drift_fixed": drift_before - drift_after,
                        "drift_remaining": drift_after,
                    },
                )
            save_state(state_file, state)

    def _loop_body(self, limit: int | None, *, merge: bool | None) -> CommandResult:
        # Every pass must observe a fresh GitHub snapshot. The list cache
        # dedupes calls within one pass, but a long-running supervisor
        # (charlie fleet supervise) reuses this app -- and therefore one
        # GitHub instance -- across many passes; without this, issues filed
        # or PRs opened after the first pass stay invisible until the daemon
        # restarts (observed live: intake frozen at a stale issue set for the
        # daemon's entire lifetime).
        self.gh.invalidate_list_cache()
        sessions_dir = self._layout.sessions_dir
        # Issue #646: the worker census now logs from inside dispatch() itself
        # (the one chokepoint every dispatch path funnels through, including
        # standalone `work`/`fleet work` which never reach this method) --
        # see dispatch()'s docstring. Not re-logged here to avoid a duplicate
        # census line within the same pass.
        # Unconditional sweep: reap stalled/orphaned sessions even when this pass
        # has zero ready/rework candidates and never reaches dispatch()'s reaper call.
        # The result is handed down to dispatch_rework()/dispatch() below so the
        # sweep runs exactly once per pass — it is the sole writer of Signal-1's
        # inconclusive-probe deferral counter, and re-running it inside each
        # dispatch lane advanced the counter up to 3x per pass, collapsing the
        # max_inconclusive_probe_deferrals "N passes of grace" into a single pass
        # (issue #343 Finding 2).
        loop_stalled_entries = _detect_and_handle_stalled_sessions(
            sessions_dir, self.paths.state_file, self.config
        )
        intake = self.intake()
        # Share a single wave budget between fresh and rework dispatch
        # Rework-first, then fresh fills the remainder
        # Resolve the effective budget once
        effective_limit = limit if limit is not None else self.config.dispatch.default_limit

        # Apply global concurrency governor cap to the total wave budget
        gov = self._apply_concurrency_governor(effective_limit)
        effective_limit = gov.dispatch_limit

        # Classify dead sessions and update throttle state (production loop path)
        # This detects provider throttling from worker deaths and sets cooldown
        # Also reconciles labels for dead sessions with no open PR (issue #118)
        sessions_dir = self._layout.sessions_dir
        # Issue #343 Finding 2: the stall lane at the top of this method
        # (line ~4100) already ran this pass and is the sole writer of the
        # inconclusive-probe deferral counter for a not-alive worker -- tell
        # this lane not to persist it again on top of that write.
        reaped = _classify_dead_sessions_and_update_throttle_state(
            sessions_dir,
            self.paths.state_file,
            self.gh,
            self.config,
            persist_inconclusive_probe_counter=False,
        )

        # Flat-interval Haiku probe for early quota/rate-limit recovery (see
        # docstring): only does real work when a throttle indicator is active.
        self._maybe_probe_quota_recovery()

        # Periodic in-loop reconcile (merge-lane-recovery §6-B): repairs
        # GitHub label / state.json divergence on a fixed cadence instead of
        # only when an operator runs `charlie mop-up --fix`. Placed before
        # the dispatch calls below so labels it repairs (e.g. a stale
        # `needs-rework` on an issue state already marked `escalated`) are
        # visible to this same pass's dispatch decisions, not just the next.
        self._maybe_reconcile_drift()

        # Sweep for orphan processes in dead session worktrees (issue #139)
        # This catches detached/daemonized processes that survived session kills
        _sweep_orphan_processes_for_dead_sessions(sessions_dir, self.paths.state_file, self.config)

        # Detect and handle orphaned workers using state.json PID records (issue #207)
        # This fallback detects dead workers even when session sidecar files are orphaned.
        # Pass the review callback so a head-advanced request_changes finding can be
        # routed to the review-pending path instead of being re-emitted as drift.
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            self.paths.state_file,
            self.config,
            self.gh,
            review_callback=self.review,
        )

        # Detect stalled sessions for notification (read-only, stateful via _build_attention_digest)
        stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)
        health_transitions: dict[int, dict[str, Any]] = {}
        for entry in stalled_entries:
            health_transitions[entry["issue"]] = {
                "adapter_kind": "unknown",  # Will be filled by #165's full supervisor
                "health": entry.get("health", "STALLED"),
                "last_log_line": None,
                "pid": entry.get("pid"),
                "terminal_tool": entry.get("terminal_tool"),
                "terminal_reason": entry.get("terminal_reason"),
            }

        # Emit notification digest if there are health transitions
        if health_transitions and self.config.notify.enabled:
            digest = _build_attention_digest(
                self.paths.state_file,
                health_transitions,
                repo=self.repo_root.name,
            )
            if digest:
                emit_digest(self._layout.notify, digest)

        dispatch_rework = self.dispatch_rework(
            effective_limit, stalled_entries=loop_stalled_entries
        )
        rework_count = dispatch_rework.data.get("selected_count", 0)
        fresh_limit = max(0, effective_limit - rework_count)
        dispatch = self.dispatch(fresh_limit, stalled_entries=loop_stalled_entries)

        # Issue #370: launch reviewers for queued PRs. This runs after worker
        # dispatch so a completed worker's review packet can be picked up by the
        # same loop pass only if the reviewer finishes immediately (tests); in
        # production the per-PR merge lane below fires on the next poll.
        dispatch_reviews = self.dispatch_reviews()

        # Auto-record cross-family verdicts for pending PRs when the
        # cross-family pass is the sole automated review (review_dispatch
        # disabled). Without this, PRs with valid cross-family reports but
        # no recorded verdict pile up in "reviewing" status forever.
        if not self.dry_run:
            self._record_cross_family_verdicts()

        reviews: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        # Issue #502: post-merge tripwire. Detect any merged worker PR that was
        # not approved by the orchestrator's adversarial review gate.
        #
        # Reuse the merged PR list already fetched by dispatch() to avoid a
        # redundant fetch per loop pass. dispatch() returns an empty list (not
        # the fetched merged PRs) when there are no ready issues, so coerce an
        # empty reuse list back to None — the tripwire must then fetch its own
        # list to stay armed even when the queue is idle (a worker self-merge can
        # land regardless of whether issues are ready).
        #
        # Deliberate trade-off: because empty-means-unknown is indistinguishable
        # from empty-means-no-merged-PRs, an idle queue costs one extra
        # merged_pr_list() call per pass. That is a paginated REST fetch, not a
        # GraphQL check-run walk (merged_pr_list is REST-only by construction —
        # issue #361), so the cost is bounded and does not risk the gateway 502s
        # that motivated #361. Staying armed while idle is worth it: the whole
        # point of the tripwire is to catch merges the orchestrator did not
        # perform, which are exactly the ones that can happen on a quiet pass.
        # Replacing this with an explicit "not fetched" sentinel threaded out of
        # dispatch() would remove the extra call; that is tracked in #446 with
        # the rest of the per-pass fetch consolidation, and is deliberately not
        # done here to keep this security fix narrow.
        merged_prs_for_tripwire: list[dict[str, Any]] | None = dispatch.data.get("merged_prs")
        if not merged_prs_for_tripwire:
            merged_prs_for_tripwire = None
        for unauthorized in self._detect_unauthorized_merges(merged_prs_for_tripwire):
            reviewed_sha = unauthorized.get("reviewed_head_sha")
            live_sha = unauthorized.get("live_head_sha")
            if (
                unauthorized["decision"] == "approved"
                and reviewed_sha is not None
                and live_sha is not None
                and reviewed_sha != live_sha
            ):
                reason = f"approved for head {reviewed_sha!r} but merged head is {live_sha!r}"
            else:
                reason = (
                    f"without an approved review decision (decision={unauthorized['decision']!r})"
                )
            errors.append(
                {
                    "pr": unauthorized["pr"],
                    "issue": unauthorized["issue"],
                    "error": (
                        f"PR #{unauthorized['pr']} ({unauthorized['head']}) is MERGED "
                        f"{reason}; possible worker self-merge"
                    ),
                }
            )

        foreign_transitions: dict[int, dict[str, Any]] = {}
        open_tracked_prs = 0
        skipped_reviews = 0
        prs = self.gh.pr_list()
        # Snapshot for foreign-PR markers only: markers change at most once
        # per PR, so a single point-in-time read at loop start is sufficient.
        state_snapshot = load_state_locked(self.paths.state_file)
        merge_train_head = (
            self._merge_train_head(prs)
            if self.config.auto_merge.update_branch_strategy == "front_of_train"
            else None
        )
        for pr in prs:
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is None:
                continue
            pr_number = int(pr["number"])
            parked = (state_snapshot["prs"].get(str(pr_number)) or {}).get(
                "foreign_issue_ref"
            ) or {}
            if parked.get("issue") == issue_number:
                # Foreign/unlinked PR: its claimed issue does not exist in this
                # repo (e.g. opened against the wrong fleet repo). Skip all
                # per-PR work with zero GitHub calls until the marker is
                # cleared or the PR's linked-issue ref changes.
                continue
            # Count every PR with a resolvable linked issue (includes skipped ones)
            open_tracked_prs += 1
            is_merge_head = merge_train_head is None or pr_number == merge_train_head
            # Per-PR isolation: one PR's merge conflict or gh failure must not
            # abort review/merge of every remaining PR in the batch.
            try:
                # Idempotence: if the PR already has an approved decision in
                # state and isn't in a rework/blocked state, skip the expensive
                # review() pass (packet regeneration + label transitions) and
                # go straight to merge_ready. This prevents a second loop() pass
                # from rewriting the review packet or re-firing labels for a PR
                # that's simply waiting on pending checks.
                state = load_state_locked(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})
                already_approved = pr_state.get("decision") == "approved" and pr_state.get(
                    "status"
                ) not in ("request_changes", "escalated", "blocked")
                if already_approved:
                    reviewed_head_sha = pr_state.get("reviewed_head_sha")
                    live_head_sha = pr.get("headRefOid")
                    head_matches = (
                        reviewed_head_sha is not None
                        and live_head_sha is not None
                        and live_head_sha == reviewed_head_sha
                    )
                    if head_matches and is_merge_head:
                        merge_result = self.merge_ready(
                            pr_number, merge=merge, merge_train_head=merge_train_head
                        )
                        self._record_merge_or_error(merge_result, errors, merges)
                    elif not head_matches:
                        review = self.review(pr_number)
                        if self._record_review_or_error(review, errors, reviews):
                            continue
                        decision = self._review_decision(pr_number)
                        if decision.get("decision") == "approved" and is_merge_head:
                            merge_result = self.merge_ready(
                                pr_number, merge=merge, merge_train_head=merge_train_head
                            )
                            self._record_merge_or_error(merge_result, errors, merges)
                else:
                    # Same-head packet skip: if we already have a review packet
                    # for this exact head SHA and no decision has been recorded
                    # yet, skip regenerating the packet. This prevents repeated
                    # supervised passes from re-firing review_started transitions
                    # and regenerating packets every poll cycle while the operator
                    # is still reading. The packet remains current; verdict file
                    # appearance triggers a delta → the merge lane fires normally.
                    live_head_sha = pr.get("headRefOid")
                    packet_head_sha = self._read_packet_head_oid(pr_number)
                    if (
                        live_head_sha is not None
                        and packet_head_sha is not None
                        and live_head_sha == packet_head_sha
                    ):
                        # Packet is current — skip regenerating it. But an
                        # operator may have written review-decision.json
                        # directly without state.json reflecting it yet (the
                        # already_approved branch above only fires once
                        # state.json has the decision), so the verdict would
                        # otherwise stay invisible until the head moves. Check
                        # the decision file directly and proceed to merge on
                        # approval, same as the decided path.
                        skipped_reviews += 1
                        decision = self._review_decision(pr_number)
                        if decision.get("decision") == "approved" and is_merge_head:
                            merge_result = self.merge_ready(
                                pr_number, merge=merge, merge_train_head=merge_train_head
                            )
                            self._record_merge_or_error(merge_result, errors, merges)
                    else:
                        review = self.review(pr_number)
                        if self._record_review_or_error(review, errors, reviews):
                            continue
                        decision = self._review_decision(pr_number)
                        if decision.get("decision") == "approved" and is_merge_head:
                            merge_result = self.merge_ready(
                                pr_number, merge=merge, merge_train_head=merge_train_head
                            )
                            self._record_merge_or_error(merge_result, errors, merges)
            except GitHubNotFoundError as exc:
                # Permanent: the PR's claimed issue (or another object it
                # references) does not exist in this repo. Park it durably and
                # alert once instead of failing the pass every 5 minutes
                # forever — retrying can never succeed.
                log_event(
                    self.paths.state_file,
                    "github_not_found_error",
                    {"pr_number": pr_number, "issue_number": issue_number, "error": str(exc)},
                    repo=self.repo_root.name,
                )
                if self._mark_foreign_issue_ref(pr_number, issue_number, str(exc)):
                    foreign_transitions[pr_number] = {
                        "adapter_kind": "unknown",
                        "health": "FOREIGN_ISSUE_REF",
                        "last_log_line": str(exc),
                        "terminal_reason": (
                            f"linked issue #{issue_number} not found in this repo; "
                            f"PR #{pr_number} parked until the marker is cleared"
                        ),
                    }
            except GitHubError as exc:
                log_event(
                    self.paths.state_file,
                    "github_error",
                    {"pr_number": pr_number, "issue_number": issue_number, "error": str(exc)},
                    repo=self.repo_root.name,
                )
                errors.append({"pr": pr_number, "error": str(exc)})
        warnings: list[str] = []
        merge_alert_transitions: dict[int, dict[str, Any]] = {}
        for merge_entry in merges:
            warning = merge_entry.get("merge_attempt_warning")
            if warning:
                warnings.append(warning)
            if merge_entry.get("merge_attempt_alarm") and merge_entry.get("issue") is not None:
                issue = merge_entry["issue"]
                merge_alert_transitions[issue] = {
                    "adapter_kind": "unknown",
                    "health": "MERGE_BLOCKED",
                    "last_log_line": None,
                    "pid": None,
                    "terminal_tool": None,
                    "terminal_reason": warning,
                }

        # Emit a merge-lane alert digest when a PR crosses the threshold.
        if merge_alert_transitions and self.config.notify.enabled:
            digest = _build_attention_digest(
                self.paths.state_file,
                merge_alert_transitions,
                repo=self.repo_root.name,
                state_field="merge_alert",
            )
            if digest:
                emit_digest(self._layout.notify, digest)

        # One-shot alert for newly parked foreign PRs. Dedupe comes from the
        # durable state marker (_mark_foreign_issue_ref returns True exactly
        # once per (pr, issue) pair), so this digest is built directly rather
        # than through the per-issue health-baseline machinery.
        if foreign_transitions and self.config.notify.enabled:
            emit_digest(
                self._layout.notify,
                AttentionDigest(
                    generated_at=utc_now(),
                    repo=self.repo_root.name,
                    transitions=tuple(
                        AttentionEntry(
                            issue_number=pr_num,
                            adapter_kind=t["adapter_kind"],
                            health=t["health"],
                            previous_health=None,
                            last_log_line=t["last_log_line"],
                            pid=None,
                            terminal_tool=None,
                            terminal_reason=t["terminal_reason"],
                        )
                        for pr_num, t in foreign_transitions.items()
                    ),
                ),
            )

        ok = (
            intake.ok and dispatch.ok and dispatch_rework.ok and dispatch_reviews.ok and not errors
        )
        message = "loop complete"
        if errors:
            message = f"loop completed with {len(errors)} PR error(s)"
        elif not intake.ok:
            message = "loop completed with intake failures"
        elif not dispatch.ok:
            message = dispatch.message
        elif not dispatch_rework.ok:
            message = dispatch_rework.message
        elif not dispatch_reviews.ok:
            message = "loop completed with review dispatch failures"
        data = {
            "intake": intake.data,
            "dispatch": dispatch.data,
            "dispatch_rework": dispatch_rework.data,
            "dispatch_reviews": dispatch_reviews.data,
            "reviews": reviews,
            "merges": merges,
            "errors": errors,
            "warnings": warnings,
            "open_tracked_prs": open_tracked_prs,
            "skipped_reviews": skipped_reviews,
            "reaped": reaped,
        }
        # Propagate concurrency info from dispatch results
        if gov.enabled or gov.fleet_enabled:
            data.update(gov.report_fields())
        # Prefer the dispatch-scoped governor values (they reflect sidecars
        # written by this pass and the most accurate fleet-wide live count).
        for lane in ("dispatch", "dispatch_rework"):
            for key in (
                "concurrency_limit",
                "live_session_count",
                "available_slots",
                "fleet_concurrency_limit",
                "fleet_live_session_count",
            ):
                if key in data[lane]:
                    data[key] = data[lane][key]
        # Cadence-gated merged-PR worktree reclamation (issue #636). Runs at
        # the END of the pass so the per-candidate `gh pr view` fan-out never
        # contends with the dispatch/review/merge lanes for state_lock or
        # GitHub quota during the critical window. Gated by
        # worktree_reclamation.interval_minutes, so it fires at most once per
        # interval regardless of poll frequency or backlog size.
        reclamation = self._maybe_reclaim_worktrees()
        if reclamation is not None:
            data["worktrees_reclaimed"] = reclamation
        return CommandResult(
            ok,
            message,
            data,
        )

    def dispatch_rework(
        self,
        limit: int | None = None,
        *,
        only_issues: str | None = None,
        stalled_entries: list[dict[str, int]] | None = None,
    ) -> CommandResult:
        """Dispatch rework workers for issues in needs-rework state with open PRs.

        ``stalled_entries``: pass the result of an already-completed
        ``_detect_and_handle_stalled_sessions`` sweep to skip re-running the
        sweep inside this call — same contract as ``dispatch()`` (issue #343
        Finding 2: the sweep writes Signal-1's inconclusive-probe deferral
        counter, so it must run at most once per pass). Standalone callers
        leave this as None and the sweep runs inside this call as before.
        """
        fleet_lock = None
        if self.config.fleet.global_max_concurrent_sessions > 0:
            fleet_lock = try_acquire_fleet_lock(self.fleet_dir_override)
            if fleet_lock is None:
                return CommandResult(
                    True,
                    "rework dispatch deferred: fleet lock held",
                    {
                        "adapter": self.config.devin.adapter,
                        "selected_count": 0,
                        "deferred_reason": "fleet_lock_held",
                    },
                )
        try:
            return self._dispatch_rework_impl(
                limit, only_issues=only_issues, stalled_entries=stalled_entries
            )
        except StateLockBusy:
            return _state_lock_busy_result(
                "rework dispatch deferred: state lock held",
                adapter=self.config.devin.adapter,
                selected_count=0,
                deferred_reason="state_lock_busy",
            )
        finally:
            if fleet_lock is not None:
                fleet_lock.release()

    def _dispatch_rework_impl(
        self,
        limit: int | None = None,
        *,
        only_issues: str | None = None,
        stalled_entries: list[dict[str, int]] | None = None,
    ) -> CommandResult:
        """Dispatch rework workers for issues in needs-rework state with open PRs.

        This is only for non-manual adapters. The manual adapter's human-paste
        path remains intact.

        Candidate selection is STATE-DRIVEN: an issue is a rework candidate iff
        state["issues"][n]["status"] == "rework_requested" and it has an open PR.
        The label is used for display only and never for selection.
        """
        if self.config.devin.adapter == "manual":
            return CommandResult(
                True,
                "rework dispatch skipped for manual adapter",
                {"adapter": "manual", "selected_count": 0},
            )

        sessions_dir = self._layout.sessions_dir
        # Unconditional stall reaper call, matching dispatch()'s — previously this
        # only ran when max_concurrent_sessions > 0 via the governor. Skipped
        # only when the caller (loop()) already ran the sweep this pass and
        # handed its result down — see dispatch_rework()'s docstring.
        if stalled_entries is None:
            _detect_and_handle_stalled_sessions(sessions_dir, self.paths.state_file, self.config)

        # Note: orphaned-worker detection is intentionally NOT re-run here.
        # loop() already runs _detect_and_handle_orphaned_workers once per pass
        # (with the review callback needed to route head-advanced findings to
        # review). Re-running it here produced duplicate drift events (#457).

        # Load state to find rework_requested issues (state-driven selection)
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)

        operator_claimed = operator_claimed_issues(state)
        operator_claimed_skipped: list[int] = []

        # Build the open-PR index BEFORE any per-issue gh.issue_view() fetch.
        # pr_list() returns only OPEN PRs by contract (--state open), so an
        # issue whose PR is closed-unmerged (or that never had one) is absent
        # from pr_by_issue. Issue #558: without this ordering, the candidate
        # scan below called gh.issue_view() for every rework_requested issue
        # every pass -- including issues whose PR closed-unmerged between
        # reconcile sweeps -- a permanent per-pass GitHub fetch with no
        # terminal exit (the exact slow-cost-spiral shape #556/#558 exist to
        # eliminate). Filtering by open PR first cuts the fetch to genuine
        # candidates only. pr_list() is cached within a pass, so calling it
        # here vs. later is the same GitHub call.
        prs = self.gh.pr_list()
        pr_by_issue: dict[int, dict[str, Any]] = {}
        for pr in prs:
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is not None:
                # If multiple PRs link to the same issue, keep the lowest PR number
                if issue_number not in pr_by_issue or int(pr["number"]) < int(
                    pr_by_issue[issue_number]["number"]
                ):
                    pr_by_issue[issue_number] = pr

        # Find issues with rework_requested status. Only fetch the full issue
        # from GitHub for issues that actually have an open PR -- a
        # rework_requested issue with no open PR is not a launch candidate
        # (the PR was closed-unmerged or never existed), so the per-issue
        # gh.issue_view fetch is skipped entirely.
        rework_issues = []
        for number, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "rework_requested":
                issue_number = int(number)
                # Issue #400: operator-claimed issues are not rework-dispatchable.
                if issue_number in operator_claimed:
                    operator_claimed_skipped.append(issue_number)
                    continue
                # Issue #558: skip the gh.issue_view fetch for issues with no
                # open PR -- not a rework candidate, and the fetch is the
                # permanent per-pass cost this gate exists to eliminate.
                if issue_number not in pr_by_issue:
                    continue
                # Fetch the full issue from GitHub to get labels and other metadata
                try:
                    full_issue = self.gh.issue_view(issue_number)
                    rework_issues.append(full_issue)
                except GitHubError:
                    # Skip issues that can't be fetched (deleted, etc.)
                    continue

        rework_limit = limit if limit is not None else self.config.dispatch.default_limit

        # Apply global concurrency governor cap
        gov = self._apply_concurrency_governor(rework_limit)
        rework_limit = gov.dispatch_limit

        # Apply provider throttle cooldown check
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            if is_throttled(state):
                throttled_until = state.get("throttled_until")
                # Return immediately with deferral reason
                data = {
                    "adapter": self.config.devin.adapter,
                    "selected_count": 0,
                    "deferred_reason": "provider_throttled",
                    "throttled_until": throttled_until,
                }
                if gov.enabled or gov.fleet_enabled:
                    data.update(gov.report_fields())
                return CommandResult(
                    False,
                    f"rework dispatch deferred: provider throttled until {throttled_until}",
                    data,
                )

        # pr_list() returns only open PRs by contract (--state open); its field
        # list does not include "state", so no per-PR state check here.
        # rework_issues already contains only issues with an open PR (the
        # fetch loop above skipped issues absent from pr_by_issue), so this
        # filter is now a no-op kept for clarity/defense-in-depth.
        candidates = [issue for issue in rework_issues if int(issue["number"]) in pr_by_issue]

        # Issue #339: a rework worker relaunched onto a PR whose rework was
        # already pushed (PR head moved past the last request_changes verdict)
        # finds nothing to do, idles, and is watchdog-reaped — burning a
        # session and a concurrency slot. Filter those candidates out here,
        # before any dispatch_pending claim, and route them to the review
        # lane instead. A sync-merge-only head advance (patch-id unchanged)
        # is NOT treated as "already reworked" — the same patch still needs a
        # genuine rework cycle, so it remains a legitimate launch candidate.
        head_check_state = load_state_locked(self.paths.state_file)
        routed_to_review: list[int] = []
        head_indeterminate: list[int] = []
        no_op_rework_escalated: list[int] = []
        filtered_candidates = []
        for issue in candidates:
            issue_number = int(issue["number"])
            pr_data = pr_by_issue[issue_number]
            pr_number = int(pr_data["number"])
            live_head_sha = pr_data.get("headRefOid")
            pr_state = head_check_state.get("prs", {}).get(str(pr_number), {})
            reviewed_head_sha = pr_state.get("reviewed_head_sha")

            if not reviewed_head_sha:
                # No recorded request_changes head to compare against —
                # nothing to disambiguate; proceed as a legitimate candidate.
                filtered_candidates.append(issue)
                continue
            if not live_head_sha:
                # Live head cannot be determined — fail closed against a
                # wasted launch, but don't strand the issue: status is left
                # untouched so the next pass retries with fresh PR data.
                head_indeterminate.append(issue_number)
                continue
            if live_head_sha == reviewed_head_sha:
                # Head hasn't moved since request_changes. Check if previous
                # rework attempts for this head already exhausted the
                # redispatch cap — if so, escalate immediately instead of
                # dispatching another worker that will also produce no
                # changes. This is a safety net for cases where the restore
                # path's escalation didn't stick (race/crash between the
                # restore and the state write).
                issue_entry = head_check_state.get("issues", {}).get(str(issue_number), {})
                if isinstance(issue_entry, dict):
                    prior_redispatch = _windowed_redispatch_at(
                        issue_entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    )
                    if len(prior_redispatch) >= self.config.watchdog.max_auto_redispatch:
                        no_op_rework_escalated.append(issue_number)
                        continue
                filtered_candidates.append(issue)
                continue

            # Head moved since the request_changes verdict. Disambiguate a
            # real content push from a sync-merge-only advance using the same
            # patch-id helper the janitor's no-op-rework gate relies on
            # (issue #222).
            reviewed_patch_id = pr_state.get("reviewed_patch_id")
            diff = self.gh.pr_diff(pr_number)
            live_patch_id = _calculate_patch_id(diff) if diff else ""
            if not reviewed_patch_id or not live_patch_id:
                # Can't establish content identity (no recorded baseline, or
                # the diff fetch itself failed) — fail closed rather than
                # guess; retry next pass instead of stranding the issue.
                head_indeterminate.append(issue_number)
                continue
            if live_patch_id == reviewed_patch_id:
                # Sync-merge only: the patch itself is unchanged, so the
                # rework is still genuinely outstanding.
                filtered_candidates.append(issue)
                continue

            routed_to_review.append(issue_number)

        candidates = filtered_candidates
        # A content change was detected above, but review() may itself fail to
        # produce a packet (deterministic janitor gate: conflicting/draft/red
        # CI — see review()'s early-return before any packet/label write).
        # Only issues review() actually routed get reported as routed_to_review;
        # the rest keep their rework_requested status and are retried next pass
        # (issue #339 finding 1: never report a routing that didn't happen —
        # doing so desyncs state.json from GitHub labels/PR state with no
        # automated recovery path).
        confirmed_routed_to_review: list[int] = []
        review_blocked_retry: list[int] = []
        for routed_issue_number in routed_to_review:
            routed_pr_number = int(pr_by_issue[routed_issue_number]["number"])
            reviewed_head_sha_before = (
                head_check_state.get("prs", {})
                .get(str(routed_pr_number), {})
                .get("reviewed_head_sha")
            )
            routed, _review_result = self._route_rework_candidate_to_review(
                routed_issue_number, routed_pr_number, reviewed_head_sha_before
            )
            if routed:
                confirmed_routed_to_review.append(routed_issue_number)
            else:
                review_blocked_retry.append(routed_issue_number)
        routed_to_review = confirmed_routed_to_review

        # Escalate no-op rework issues that have exhausted the redispatch cap
        # without the PR head ever advancing. Each of these would have burned
        # another worker session on an unchanged diff.
        if no_op_rework_escalated:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                for issue_number in no_op_rework_escalated:
                    entry = state.get("issues", {}).get(str(issue_number), {})
                    if not isinstance(entry, dict):
                        entry = {}
                    current_status = entry.get("status")
                    if current_status == "escalated":
                        continue
                    redispatch_at = _windowed_redispatch_at(
                        entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]
                    entry = {
                        **entry,
                        "number": issue_number,
                        "status": "escalated",
                        "redispatch_at": redispatch_at,
                        "escalation_reason": "redispatch_cap_exceeded",
                        "dispatched_at": None,
                    }
                    state["issues"][str(issue_number)] = entry
                    state = append_event(
                        state,
                        "session_failed_escalated",
                        {
                            "issue_number": issue_number,
                            "previous_status": "rework_requested",
                            "reason": "no_op_rework_cap_exceeded",
                            "redispatch_count": len(redispatch_at),
                        },
                        state_path=self.paths.state_file,
                    )
                save_state(self.paths.state_file, state)
            for issue_number in no_op_rework_escalated:
                transition(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    "redispatch_escalated",
                )

        if only_issues:
            wanted = parse_issue_numbers(only_issues)
            by_number = {int(issue["number"]): issue for issue in candidates}
            selected = [by_number[number] for number in wanted if number in by_number]
            # Apply concurrency governor cap to explicit issue selection
            if len(selected) > rework_limit:
                deferred_by_concurrency = [
                    int(issue["number"]) for issue in selected[rework_limit:]
                ]
                selected = selected[:rework_limit]
            else:
                deferred_by_concurrency = []
        else:
            selected = candidates[:rework_limit]
            deferred_by_concurrency = []

        if not selected:
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "failures": _build_failure_map([], set(), deferred_by_concurrency, rework_limit),
                "deferred_by_concurrency": deferred_by_concurrency,
                "routed_to_review": sorted(routed_to_review),
                "skipped_head_indeterminate": sorted(head_indeterminate),
                "review_blocked_retry": sorted(review_blocked_retry),
                "operator_claimed_skipped": sorted(operator_claimed_skipped),
                "no_op_rework_escalated": sorted(no_op_rework_escalated),
            }
            if gov.enabled or gov.fleet_enabled:
                data.update(gov.report_fields())
            return CommandResult(
                True,
                "no rework candidates found",
                data,
            )

        # First lock: claim issues by marking them as dispatch_pending
        selected_issue_numbers: list[int] = []
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Filter out issues whose PR is in escalated state (rework cap exhausted)
            selected = [
                issue
                for issue in selected
                if state["prs"]
                .get(str(pr_by_issue[int(issue["number"])]["number"]), {})
                .get("status")
                != "escalated"
            ]
            live_dispatched = set()
            for number, entry in state.get("issues", {}).items():
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status == "dispatched":
                    live_dispatched.add(int(number))
                elif status == "dispatch_pending" and not is_claim_stale(
                    entry.get("dispatch_pending_at")
                ):
                    live_dispatched.add(int(number))
            # Filter out already-dispatched issues
            selected = [issue for issue in selected if int(issue["number"]) not in live_dispatched]
            selected_issue_numbers = [int(issue["number"]) for issue in selected]
            # Mark selected issues as "dispatch_pending"
            for issue_number in selected_issue_numbers:
                entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "dispatch_pending",
                    "dispatch_pending_at": utc_now(),
                }
                # A fresh dispatch supersedes any previous orphan flag.
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                state["issues"][str(issue_number)] = entry
            save_state(self.paths.state_file, state)

        if not selected_issue_numbers:
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "failures": _build_failure_map([], set(), deferred_by_concurrency, rework_limit),
                "deferred_by_concurrency": deferred_by_concurrency,
                "routed_to_review": sorted(routed_to_review),
                "skipped_head_indeterminate": sorted(head_indeterminate),
                "review_blocked_retry": sorted(review_blocked_retry),
                "no_op_rework_escalated": sorted(no_op_rework_escalated),
            }
            if gov.enabled or gov.fleet_enabled:
                data.update(gov.report_fields())
            return CommandResult(
                True,
                "all rework candidates already dispatched",
                data,
            )

        # Do all network calls, file writes, and worker launches outside the lock
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        skipped_issue_numbers: list[int] = []
        missing_prompt_failures: dict[int, str] = {}
        # Issue #482: per-issue adapter routing for rework. Rework issues
        # route to the api adapter when preflight passes (policy:rework),
        # falling back to the default adapter on any preflight failure.
        # The rework prompt is already on disk (written during the review
        # flow with config.dispatch.rework_template); api_worker.rework_template
        # defaults to the same "rework.md", so no re-render is needed here.
        adapter_choices: dict[int, AdapterChoice] = {}
        api_enabled = self.config.api_worker.enabled
        routing_inputs = self._routing_inputs() if api_enabled else None
        # Rescue tier (issue #555): an issue is rescue-marked when the rescue
        # interception sites (record_review / _route_janitor_gate_failure_to_
        # rework) already stamped `rescue_attempted` on its PR record instead
        # of escalating. Those issues must launch via the claude-code adapter
        # pinned to `rescue.worker_model`, regardless of the primary
        # configured `devin.adapter` — tracked separately here so the SAME
        # candidate-selection/session-request/state-bookkeeping code below
        # handles them, only the final dispatch_sessions() call differs.
        #
        # Always loaded (never gated on self.config.rescue.enabled): routing
        # of a PR that already carries the durable marker must not depend on
        # the current config value. If an operator flips rescue.enabled off
        # while a rescue rework is queued (rework_requested + marker set,
        # worker not yet launched), it must still launch via the rescue
        # adapter/model -- enabled only gates NEW rescue entry at the three
        # cap sites, never routing of an already-marked PR.
        rescue_issue_numbers: set[int] = set()
        rescue_state_snapshot = load_state_locked(self.paths.state_file)
        for issue_number in selected_issue_numbers:
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            pr = pr_by_issue[issue_number]
            pr_number = int(pr["number"])
            # Use the existing PR branch instead of creating a new one
            branch_name = pr.get("headRefName", "")
            # Use the rework prompt from the PR directory
            rework_prompt_path = self.paths.prs / f"pr-{pr_number}" / "rework-prompt.md"
            if not rework_prompt_path.exists():
                # Skip if rework prompt doesn't exist — record as rework_requested
                # to release the claim and allow retry (issue #116)
                skipped_issue_numbers.append(issue_number)
                missing_prompt_failures[issue_number] = (
                    f"missing rework prompt: {rework_prompt_path}"
                )
                continue
            # Issue #632: the brief on disk is authoritative and can drift
            # arbitrarily far from a corrected verdict (an operator
            # hand-editing review-decision.json is the #510 case). Regenerate
            # it from the verdict + the preserved dispatch-note sidecar when
            # the verdict is newer than the brief, so a stale brief cannot
            # outlive a corrected verdict. The note sidecar is written by
            # _write_rework_prompt; if it is absent (a brief predating the
            # fix) the brief is regenerated with an empty note — the findings
            # are the critical part and must not stay hidden.
            decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
            if _is_verdict_newer_than_brief(decision_path, rework_prompt_path):
                note_path = self.paths.prs / f"pr-{pr_number}" / "rework-dispatch-note.txt"
                dispatch_note = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
                self._write_rework_prompt(pr, issue_number, dispatch_note)
            # Rescue tier (issue #555): rescue-marked PRs bypass per-issue
            # adapter routing — they always launch via the claude-code adapter
            # pinned to rescue.worker_model, regardless of the primary
            # configured devin.adapter or api_worker routing preflight.
            pr_state_for_rescue = rescue_state_snapshot.get("prs", {}).get(str(pr_number), {})
            if pr_state_for_rescue.get("rescue_attempted"):
                rescue_issue_numbers.add(issue_number)
            else:
                # Route this non-rescue rework issue through the single
                # enforcement point (issue #482).
                if api_enabled and routing_inputs is not None:
                    issue_labels = {label["name"] for label in full_issue.get("labels", [])}
                    choice = self._select_adapter_for_issue(
                        rework=True,
                        issue_labels=issue_labels,
                        routing_inputs=routing_inputs,
                    )
                    adapter_choices[issue_number] = choice
            session_requests.append(
                SessionRequest(
                    issue_number=issue_number,
                    issue_title=str(full_issue.get("title") or ""),
                    prompt_path=rework_prompt_path,
                    branch_name=branch_name,
                    rework=True,
                )
            )

        if not session_requests:
            # Release the dispatch_pending claims for all skipped issues
            no_session_failure_map = _build_failure_map(
                [],
                set(),
                deferred_by_concurrency,
                rework_limit,
                extra_failures=missing_prompt_failures,
            )
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                for issue_number in skipped_issue_numbers:
                    full_issue = full_issues[issue_number]
                    entry = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "title": full_issue.get("title"),
                        "url": full_issue.get("url"),
                        # Issue #116: restore to rework_requested for retry (missing prompt may be transient)
                        "status": "rework_requested",
                        "dispatched_at": None,
                    }
                    entry.pop("dispatch_pending_at", None)
                    entry.pop("label_error", None)
                    state["issues"][str(issue_number)] = entry
                state = append_event(
                    state,
                    "dispatch_rework",
                    {
                        "issue_numbers": [],
                        "failed_issue_numbers": [],
                        "skipped_issue_numbers": sorted(skipped_issue_numbers),
                        "deferred_by_concurrency": deferred_by_concurrency,
                        "label_errors": [],
                        "operator_claimed_skipped": sorted(operator_claimed_skipped),
                        "failures": no_session_failure_map,
                    },
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "failures": no_session_failure_map,
                "deferred_by_concurrency": deferred_by_concurrency,
                "routed_to_review": sorted(routed_to_review),
                "skipped_head_indeterminate": sorted(head_indeterminate),
                "review_blocked_retry": sorted(review_blocked_retry),
                "operator_claimed_skipped": sorted(operator_claimed_skipped),
                "no_op_rework_escalated": sorted(no_op_rework_escalated),
            }
            if gov.enabled or gov.fleet_enabled:
                data.update(gov.report_fields())
            return CommandResult(
                True,
                "no valid rework prompts found",
                data,
            )

        manifest_path = self._layout.session_manifest
        results_path = self._layout.session_results
        # Rescue tier (issue #555) + per-issue routing (issue #482): split
        # the batch so rescue-marked issues launch via the claude-code adapter
        # pinned to rescue.worker_model (see _rescue_adapter_settings),
        # while every other candidate is dispatched via the per-issue routing
        # partition (_dispatch_partitioned). Rescue issues bypass routing
        # entirely — they always use the rescue adapter/model. Reuses the
        # same dispatch_sessions()/launch_claude_worker() path for both; the
        # only difference is which AdapterSettings/config is passed in.
        normal_requests = [
            r for r in session_requests if r.issue_number not in rescue_issue_numbers
        ]
        rescue_requests = [r for r in session_requests if r.issue_number in rescue_issue_numbers]
        dispatch_results: list[SessionDispatchResult] = []
        if normal_requests:
            # adapter_choices only contains non-rescue issues (rescue issues
            # were never routed in the loop above), so the partition is
            # correct.
            dispatch_results.extend(self._dispatch_partitioned(normal_requests, adapter_choices))
        if rescue_requests:
            dispatch_results.extend(
                dispatch_sessions(
                    self.repo_root,
                    manifest_path,
                    results_path,
                    self._rescue_adapter_settings(),
                    rescue_requests,
                )
            )
        # _dispatch_partitioned and dispatch_sessions each overwrite
        # manifest/results on each call; rewrite once more with the combined
        # batch so the on-disk observability files
        # (session-manifest.json/session-results.json) reflect the full pass,
        # not just the last sub-call.
        write_session_manifest(manifest_path, session_requests, adapter=self.config.devin.adapter)
        write_session_results(results_path, dispatch_results)

        successful_issue_numbers = {
            result.issue_number for result in dispatch_results if result.ok
        }
        failed_issue_numbers = {
            result.issue_number for result in dispatch_results if not result.ok
        }

        # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
        label_errors: list[int] = []
        label_error_failures: dict[int, str] = {}
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Record skipped issues (missing rework prompt) as rework_requested
            # This handles the mixed case where some issues have prompts and some don't.
            # Missing rework-prompt.md may be transient (review agent hasn't written it yet),
            # so restore to rework_requested for retry (issue #116).
            for issue_number in skipped_issue_numbers:
                full_issue = full_issues[issue_number]
                entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "status": "rework_requested",
                    "dispatched_at": None,
                }
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                state["issues"][str(issue_number)] = entry
            for request in session_requests:
                full_issue = full_issues[request.issue_number]
                ok = request.issue_number in successful_issue_numbers
                # Issue #482: record the adapter choice into adapter_history
                # before constructing the entry, so the {**entry} spread below
                # preserves the adapter_history field.
                choice = adapter_choices.get(request.issue_number)
                if choice is not None:
                    state = record_adapter_choice(state, request.issue_number, choice, utc_now())
                entry = {
                    **state["issues"].get(str(request.issue_number), {}),
                    "number": request.issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "branch_name": request.branch_name,
                    "prompt_path": str(request.prompt_path),
                    # On failure, restore to rework_requested so the issue can be retried
                    # in the next pass (issue #116). On success, mark as dispatched.
                    "status": "dispatched" if ok else "rework_requested",
                    "dispatched_at": utc_now() if ok else None,
                }
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                # A successful dispatch supersedes any previous orphan flag.
                if ok:
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                # Store worker PID and process start time for state-based liveness detection
                # This allows recovery even when session sidecar files are orphaned (issue #207)
                if ok:
                    result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    if result and result.pid is not None:
                        entry["worker_pid"] = result.pid
                        entry["worker_process_start_time"] = result.process_start_time
                if ok:
                    # Track redispatch count for escalation cap (issue #165)
                    now = datetime.now(UTC)
                    redispatch_at = _windowed_redispatch_at(
                        entry, window_minutes=self.config.watchdog.redispatch_window_minutes
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    if len(redispatch_at) > self.config.watchdog.max_auto_redispatch:
                        # Escalate to human review
                        entry["status"] = "escalated"
                        entry["redispatch_at"] = redispatch_at
                        entry["escalation_reason"] = "redispatch_cap_exceeded"
                        state["issues"][str(request.issue_number)] = entry
                        save_state(self.paths.state_file, state)
                        result = transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            "redispatch_escalated",
                        )
                        if result.outcome != TransitionOutcome.APPLIED:
                            label_error = {
                                "edge": "redispatch_escalated",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            entry["label_error"] = label_error
                            label_errors.append(request.issue_number)
                            label_error_failures[request.issue_number] = _label_error_reason(
                                label_error
                            )
                            save_state(self.paths.state_file, state)
                        continue
                    else:
                        entry["redispatch_at"] = redispatch_at
                        state["issues"][str(request.issue_number)] = entry
                        save_state(self.paths.state_file, state)
                        result = transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            "rework_dispatched",
                        )
                        if result.outcome != TransitionOutcome.APPLIED:
                            label_error = {
                                "edge": "rework_dispatched",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            entry["label_error"] = label_error
                            label_errors.append(request.issue_number)
                            label_error_failures[request.issue_number] = _label_error_reason(
                                label_error
                            )
                            save_state(self.paths.state_file, state)
                else:
                    # Track every rework-dispatch attempt, successful or not,
                    # against the same redispatch window used on the success path.
                    # Failed attempts that repeat without ever succeeding
                    # eventually trip max_auto_redispatch and escalate instead of
                    # looping forever (issue #515).
                    failed_result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    failure_kind = failed_result.failure_kind if failed_result else None
                    now = datetime.now(UTC)
                    redispatch_at = _windowed_redispatch_at(
                        entry, window_minutes=self.config.watchdog.redispatch_window_minutes
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                    if (
                        terminal_failure
                        or len(redispatch_at) > self.config.watchdog.max_auto_redispatch
                    ):
                        # Escalate to human review
                        entry["status"] = "escalated"
                        entry["redispatch_at"] = redispatch_at
                        entry["escalation_reason"] = (
                            failure_kind if terminal_failure else "redispatch_cap_exceeded"
                        )
                        entry["dispatched_at"] = None
                        state["issues"][str(request.issue_number)] = entry
                        save_state(self.paths.state_file, state)
                        result = transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            "redispatch_escalated",
                        )
                        if result.outcome != TransitionOutcome.APPLIED:
                            label_error = {
                                "edge": "redispatch_escalated",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            entry["label_error"] = label_error
                            label_errors.append(request.issue_number)
                            label_error_failures[request.issue_number] = _label_error_reason(
                                label_error
                            )
                            save_state(self.paths.state_file, state)
                        continue
                    entry["status"] = "rework_requested"
                    entry["dispatched_at"] = None
                    entry["redispatch_at"] = redispatch_at
                    state["issues"][str(request.issue_number)] = entry
                    save_state(self.paths.state_file, state)
            rework_failure_map = _build_failure_map(
                dispatch_results,
                failed_issue_numbers,
                deferred_by_concurrency,
                rework_limit,
                extra_failures={**missing_prompt_failures, **label_error_failures},
            )
            state = append_event(
                state,
                "dispatch_rework",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "skipped_issue_numbers": sorted(skipped_issue_numbers),
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "label_errors": sorted(label_errors),
                    "operator_claimed_skipped": sorted(operator_claimed_skipped),
                    "failures": rework_failure_map,
                },
                state_path=self.paths.state_file,
            )
            save_state(self.paths.state_file, state)

        result_dicts = [result.to_dict() for result in dispatch_results]
        message = "rework dispatch complete"
        if failed_issue_numbers:
            entries = ", ".join(
                f"#{issue} ({rework_failure_map[issue]})" for issue in sorted(failed_issue_numbers)
            )
            message = f"rework dispatch failures: {entries}"
        if label_errors:
            message += f" (launched but label write failed: {sorted(label_errors)})"
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "failures": rework_failure_map,
            "deferred_by_concurrency": deferred_by_concurrency,
            "label_errors": sorted(label_errors),
            "session_manifest": str(manifest_path),
            "session_results": str(results_path),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": result_dicts,
            "routed_to_review": sorted(routed_to_review),
            "skipped_head_indeterminate": sorted(head_indeterminate),
            "review_blocked_retry": sorted(review_blocked_retry),
            "operator_claimed_skipped": sorted(operator_claimed_skipped),
            "no_op_rework_escalated": sorted(no_op_rework_escalated),
        }
        if gov.enabled or gov.fleet_enabled:
            data.update(gov.report_fields())

        # Emit notification digest if there are health transitions (stalled sessions)
        # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
        sessions_dir = self._layout.sessions_dir
        stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)
        if stalled_entries and self.config.notify.enabled:
            health_transitions: dict[int, dict[str, Any]] = {}
            for entry in stalled_entries:
                health_transitions[entry["issue"]] = {
                    "adapter_kind": "unknown",  # Will be filled by #165's full supervisor
                    "health": entry.get("health", "STALLED"),
                    "last_log_line": None,
                    "pid": entry.get("pid"),
                    "terminal_tool": entry.get("terminal_tool"),
                    "terminal_reason": entry.get("terminal_reason"),
                }
            digest = _build_attention_digest(
                self.paths.state_file,
                health_transitions,
                repo=self.repo_root.name,
            )
            if digest:
                emit_digest(self._layout.notify, digest)

        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

    def _route_rework_candidate_to_review(
        self,
        issue_number: int,
        pr_number: int,
        reviewed_head_sha_before: str | None,
    ) -> tuple[bool, CommandResult]:
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
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
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
            state = append_event(
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
            save_state(self.paths.state_file, state)
        return routed, review_result

    def _merged_pr_referenced_issue_numbers(
        self,
        issues: list[dict[str, Any]],
        merged_prs: list[dict[str, Any]],
    ) -> tuple[set[int], set[int], set[int]]:
        """Return ready issues already covered by a merged PR, split by trust level.

        Returns a ``(bound, mention_only, bound_pr_numbers)`` triple:

        * ``bound`` — ``linked_issue_number`` binds the PR to the issue by a
          hijack-safe signal (same-repo branch-prefix or closing-action verb).
          This is the same trust level issue #220 uses to close issues at
          merge time, so callers MAY treat a bound issue as safe to close here
          too (belt-and-suspenders in case #220's merge-time close hasn't
          landed, e.g. a crash between merge and label transition).
        * ``mention_only`` — the PR merely contains an ``issue #N`` /
          ``issues #N`` text reference (same-repo PRs only — cross-repo
          provenance is never trusted here, and ``isCrossRepository`` only
          describes head-branch provenance, not which repo the *text* refers
          to, so it cannot resolve a cross-repo mention collision either).
          This is advisory only per ``issue_numbers_mentioned_by_pr``'s
          contract: issue #203 never authorized closing an issue on a bare
          mention. Callers MUST exclude these from dispatch and flag them for
          a human — never close the issue or transition it toward "merged".
        * ``bound_pr_numbers`` — the PR numbers that bound to a managed issue.
          Used to finalize state.json ``prs`` entries for externally-merged PRs
          (issue #427: Aviator mergequeue handoff).

        The bound/mention sets are intersected with the supplied issue set so a
        stray mention of an issue not in the dispatch queue does not get
        actioned. ``bound`` takes precedence: an issue bound by one merged PR
        but only mentioned by another is reported solely in ``bound``.
        """
        ready_issue_numbers = {int(issue["number"]) for issue in issues}
        bound: set[int] = set()
        bound_pr_numbers: set[int] = set()
        mention_only: set[int] = set()
        for pr in merged_prs:
            if str(pr.get("state") or "").upper() != "MERGED":
                continue
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is not None and issue_number in ready_issue_numbers:
                bound.add(issue_number)
                bound_pr_numbers.add(int(pr["number"]))
            # isCrossRepository describes the PR's own head-branch provenance
            # (fork vs. same-repo), not which repo a free-text "#N" refers to.
            # It cannot fully guard a cross-repo mention collision, but it does
            # guard the common case of a fork PR's text being trusted at all.
            if pr.get("isCrossRepository") is False:
                for mentioned in issue_numbers_mentioned_by_pr(pr):
                    if mentioned in ready_issue_numbers:
                        mention_only.add(mentioned)
        mention_only -= bound
        return bound, mention_only, bound_pr_numbers

    def _is_dispatchable(
        self,
        issue: dict[str, Any],
        operator_claimed: set[int] | None = None,
    ) -> bool:
        # Closed issues are never dispatchable, regardless of labels.
        if str(issue.get("state") or "OPEN").upper() != "OPEN":
            return False
        names = label_names(issue)
        if self.config.labels.ready not in names:
            return False
        if names & self.config.labels.terminal:
            return False
        if names & self.config.labels.active:
            return False
        if operator_claimed is None:
            state = load_state_locked(self.paths.state_file)
            operator_claimed = operator_claimed_issues(state)
        return int(issue["number"]) not in operator_claimed

    def _route_phantom_live_worker(
        self,
        state: dict[str, Any],
        request: SessionRequest,
        full_issue: dict[str, Any],
        sessions_dir: Path,
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Route a phantom ``live_worker_redispatch_averted`` result as dead.

        The adapter reported a live worker, but the recorded PID failed the
        OS-level liveness + identity check (issue #523). Remove the stale
        sidecar/marker so the session no longer occupies a concurrency slot,
        then strip active labels and restore ``automated-ready`` so the issue
        is dispatchable again, recording a single
        ``session_failed_relabeled`` attention event.

        This mirrors ``_classify_dead_sessions_and_update_throttle_state``'s
        no-open-PR relabel path. The open-PR/rework-routing case is
        intentionally NOT handled here: a dispatched request's issue can never
        have an open tracked PR, because ``_dispatch_impl`` builds
        ``pr_by_issue`` once and candidate selection excludes every issue in
        it (so ``request.issue_number`` is never in ``pr_by_issue``). A phantom
        worker whose issue later opens a PR is routed to rework by the
        dead-session reaper lane, not by dispatch.

        Returns ``(status, dispatched_at, state)``. The status is
        ``"dispatch_failed"`` so the caller's entry-building frees the slot;
        ``dispatched_at`` is ``None`` because no worker was actually launched.
        """
        issue_number = request.issue_number

        # Reap the stale sidecar and matching worktree writer marker. Reuse
        # WorkerView.reap_sidecar so the adapter-specific path derivation and
        # session-id-gated marker removal stay in one place.
        for w in iter_workers(sessions_dir):
            if w.issue_number == issue_number:
                w.reap_sidecar(sessions_dir)

        # Strip active labels and ensure the ready label is present so the
        # issue becomes dispatchable. This mirrors
        # _classify_dead_sessions_and_update_throttle_state's no-open-PR
        # relabel path, gated on an active label actually being present so a
        # terminal-only issue is never given ``ready`` back spuriously.
        issue_labels = label_names(full_issue)
        active_labels = issue_labels & self.config.labels.active
        if not active_labels:
            state = append_event(
                state,
                "session_failed_relabeled",
                {
                    "issue_number": issue_number,
                    "failure_kind": "live_worker_redispatch_averted",
                    "reason": "phantom_live_worker_pid_dead",
                    "removed_labels": [],
                    "added_ready": False,
                    "label_write_ok": True,
                },
                state_path=self.paths.state_file,
            )
            return "dispatch_failed", None, state

        needs_ready = self.config.labels.ready not in issue_labels
        label_write_ok = True
        for label in sorted(active_labels):
            if not self.gh.remove_issue_label(issue_number, label):
                label_write_ok = False
        if needs_ready:
            if not self.gh.add_issue_label(issue_number, self.config.labels.ready):
                label_write_ok = False
        state = append_event(
            state,
            "session_failed_relabeled",
            {
                "issue_number": issue_number,
                "failure_kind": "live_worker_redispatch_averted",
                "reason": "phantom_live_worker_pid_dead",
                "removed_labels": sorted(active_labels),
                "added_ready": needs_ready,
                "label_write_ok": label_write_ok,
            },
            state_path=self.paths.state_file,
        )
        return "dispatch_failed", None, state

    def _get_open_blockers(self, issue: dict[str, Any]) -> tuple[list[int], list[int]]:
        """Check if an issue has any open blocker issues.

        Parses the issue body for blocker declarations and checks GitHub's
        native issue dependencies. Returns both declared blockers and open blockers.

        Args:
            issue: The issue dict from GitHub API

        Returns:
            Tuple of (declared_blockers, open_blockers). Both are lists of issue numbers.
            declared_blockers includes all blockers mentioned in the issue body or
            GitHub dependencies. open_blockers is the subset that are currently open.
        """
        import logging

        logger = logging.getLogger(__name__)
        issue_number = int(issue["number"])
        body = issue.get("body", "")

        # Parse blockers from issue body
        body_blockers = parse_blockers(body)

        # Get GitHub native dependencies
        gh_blockers = get_github_issue_dependencies(self.gh, issue_number)

        # Combine and deduplicate
        all_blockers = sorted(set(body_blockers + gh_blockers))

        if not all_blockers:
            return [], []

        # Check which blockers are still open
        open_blockers = self.gh.are_issues_open(all_blockers)

        # Filter out self-references (malformed markers like "Blocked by #123" on issue #123)
        if issue_number in open_blockers:
            logger.warning(
                f"Issue #{issue_number} has self-referencing blocker declaration - ignoring"
            )
            open_blockers.discard(issue_number)
            all_blockers.remove(issue_number)

        return sorted(all_blockers), sorted(open_blockers)

    @staticmethod
    def _is_dead_blocker(
        blocker_number: int,
        state: dict[str, Any],
        pr_by_issue: dict[int, dict[str, Any]],
    ) -> bool:
        """True when a blocker issue can never resolve through any automated path.

        Used by dispatch()'s blocked-chain attention check: "dead" means the
        blocker issue itself is escalated, or its tracked open PR's status is
        escalated/janitor_blocked. Pure local-state lookup, no GitHub calls --
        this only names an already-known dead end, it never widens one.
        """
        issue_entry = state.get("issues", {}).get(str(blocker_number), {})
        if isinstance(issue_entry, dict) and issue_entry.get("status") == "escalated":
            return True
        pr = pr_by_issue.get(blocker_number)
        if pr is not None:
            pr_number = pr.get("number")
            if pr_number is not None:
                pr_status = state.get("prs", {}).get(str(pr_number), {}).get("status")
                if pr_status in ("escalated", "janitor_blocked"):
                    return True
        return False

    def _filter_blocked_issues(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[int]]]:
        """Filter out issues with open blockers from the candidate list.

        This is a shared helper used by both dry-run and real dispatch paths
        to ensure single-point-of-enforcement for the dependency gate logic.

        Args:
            candidates: List of candidate issue dicts from GitHub API

        Returns:
            Tuple of (filtered_candidates, blocked_issues, open_blockers_by_issue).
            filtered_candidates is the input list with blocked issues removed.
            blocked_issues maps blocked issue numbers to their full declared
            blocker list (open + closed) -- unchanged, this is the exact
            shape the dispatch_skip_blocked event payload has always used.
            open_blockers_by_issue maps the same issue numbers to only the
            currently-open subset: a closed blocker isn't actually blocking
            anymore, so it must not count when deciding whether every
            blocker of an issue is "dead" (see dispatch()'s blocked-chain
            attention check).
        """
        blocked_issues: dict[int, list[int]] = {}
        open_blockers_by_issue: dict[int, list[int]] = {}
        for issue in candidates:
            issue_number = int(issue["number"])
            declared_blockers, open_blockers = self._get_open_blockers(issue)
            if open_blockers:
                blocked_issues[issue_number] = declared_blockers
                open_blockers_by_issue[issue_number] = open_blockers

        filtered_candidates = [
            issue for issue in candidates if int(issue["number"]) not in blocked_issues
        ]
        return filtered_candidates, blocked_issues, open_blockers_by_issue

    def _sort_by_dependency_depth(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort unblocked candidates by out-degree (number of blocked dependents).

        Prioritizes issues that block the most downstream issues, so a wave
        drains the critical path and maximizes unblocking. Issues are sorted
        descending by their count of currently-blocked dependents, with
        creation date (oldest first) as a tiebreaker.

        This metric is computed against the full ready-labeled issue set
        before filtering, not just the unblocked candidates, to capture
        the true unblocking impact of each issue.

        Args:
            candidates: List of unblocked candidate issue dicts from GitHub API

        Returns:
            List of candidates sorted by out-degree (descending), then by creation date.
        """
        if not candidates:
            return []

        # Fetch the full set of ready-labeled issues to compute out-degree
        # We need issues that are blocked (not just candidates) to count dependents
        ready_issues = self.gh.issue_list(
            labels=[self.config.labels.ready],
            state="OPEN",
        )

        # Build reverse-adjacency map: blocker_number -> [dependents that are still blocked]
        blocker_to_dependents: dict[int, list[int]] = {}

        for issue in ready_issues:
            issue_number = int(issue["number"])
            declared_blockers, open_blockers = self._get_open_blockers(issue)

            # Only count dependents that are currently blocked (have open blockers)
            if not open_blockers:
                continue

            for blocker in declared_blockers:
                if blocker not in blocker_to_dependents:
                    blocker_to_dependents[blocker] = []
                blocker_to_dependents[blocker].append(issue_number)

        # Compute out-degree for each candidate (number of blocked dependents)
        out_degree: dict[int, int] = {}
        for issue in candidates:
            issue_number = int(issue["number"])
            out_degree[issue_number] = len(blocker_to_dependents.get(issue_number, []))

        # Sort by out-degree (descending), then by creation date (ascending for oldest-first)
        def sort_key(issue: dict[str, Any]) -> tuple[int, str]:
            issue_number = int(issue["number"])
            # Use negative out_degree for descending sort
            degree = -out_degree.get(issue_number, 0)
            # Parse creation date; if missing, use high sentinel to sort last
            created_at = issue.get("createdAt", "9999-12-31T23:59:59Z")
            return (degree, created_at)

        return sorted(candidates, key=sort_key)

    def _sort_by_dispatch_order(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort candidates by dispatch order (oldest-first or newest-first).

        Uses the createdAt field from GitHub API to sort by creation date.
        Default is oldest-first (ascending), but can be configured to newest-first
        (descending) via dispatch.order config.

        Args:
            candidates: List of candidate issue dicts from GitHub API

        Returns:
            Sorted list of candidates by creation date according to dispatch.order
        """
        if self.config.dispatch.order == "newest":
            # Sort by createdAt descending (newest first)
            return sorted(
                candidates,
                key=lambda issue: issue.get("createdAt", ""),
                reverse=True,
            )
        else:
            # Sort by createdAt ascending (oldest first, default)
            return sorted(
                candidates,
                key=lambda issue: issue.get("createdAt", ""),
                reverse=False,
            )

    def _branch_name(self, issue: dict[str, Any]) -> str:
        return f"{self.config.dispatch.branch_prefix}-{int(issue['number'])}-{slugify(str(issue.get('title') or 'work'))}"

    def _write_worker_prompt(self, issue: dict[str, Any], *, template: str | None = None) -> Path:
        issue_number = int(issue["number"])
        issue_dir = self.paths.issues / f"issue-{issue_number}"
        issue_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = issue_dir / "worker-prompt.md"
        prompt = self._render(
            template or self.config.dispatch.worker_template,
            {
                "issue_number": issue_number,
                "issue_title": issue.get("title", ""),
                "issue_url": issue.get("url", ""),
                "issue_body": issue.get("body", ""),
                "branch_name": self._branch_name(issue),
                "worker_model_tier": self.config.dispatch.worker_model_tier,
            },
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt_path

    def _write_rework_prompt(
        self, pr: dict[str, Any], issue_number: int | None, dispatch_note: str
    ) -> Path:
        return _write_rework_prompt(
            self.paths.state_file,
            pr,
            issue_number,
            dispatch_note,
            self.config,
            repo_root=self.repo_root,
        )

    def _detect_unauthorized_merges(
        self, merged_prs: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Scan recently-merged PRs for worker branches whose merged head does not match an approved review decision.

        Workers are forbidden from merging their own PRs (issue #502). This
        post-merge tripwire catches a bypass after it happens: any merged PR
        whose head branch matches the configured worker branch prefix and whose
        head SHA is not covered by an approved ``.var/charlie-work/prs/pr-N/review-decision.json``
        was merged without the orchestrator's adversarial review gate. GitHub
        errors are swallowed so a transient ``gh`` failure cannot crash the
        fleet pass.

        Findings are bounded to merges this control could actually have governed
        — see ``_apply_unauthorized_merge_baseline``, which every return path
        goes through so the bound cannot be bypassed by a future caller.
        """
        candidates: list[dict[str, Any]] = []
        prefix = self.config.dispatch.branch_prefix
        if merged_prs is None:
            try:
                merged_prs = self.gh.merged_pr_list()
            except GitHubError:
                # No list means nothing observed, not "nothing wrong" — return
                # empty without arming, so a transient gh failure on the very
                # first pass cannot bake an empty baseline and permanently
                # exempt the real history it never saw.
                #
                # This covers every raising failure in merged_pr_list(): gh
                # missing, unparseable JSON, non-zero exit, AND gh exiting 0
                # with empty stdout (the silent-empty case #633 closed —
                # merged_pr_list() now raises GitHubError on a non-list result
                # instead of coercing None to []). The empty-stdout path no
                # longer arms an empty baseline silently; it is reported here
                # as a raising failure and the next pass re-arms from real
                # data.
                return []

        for pr in merged_prs:
            if str(pr.get("state") or "").upper() != "MERGED":
                continue
            if pr.get("isCrossRepository") is True:
                continue
            head = str(pr.get("headRefName") or "")
            if not head.startswith(prefix):
                continue
            raw_number = pr.get("number")
            try:
                pr_number = int(raw_number) if raw_number is not None else None
            except (TypeError, ValueError):
                continue
            if pr_number is None:
                continue
            decision = self._review_decision(pr_number)
            decision_value = decision.get("decision")
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            approved = decision_value == "approved"
            head_matches = (
                reviewed_head_sha is not None
                and live_head_sha is not None
                and reviewed_head_sha == live_head_sha
            )
            if not approved or not head_matches:
                issue_number = linked_issue_number(
                    pr,
                    is_cross_repository=pr.get("isCrossRepository"),
                    branch_prefix=prefix,
                )
                candidates.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "head": head,
                        "decision": decision_value,
                        "reviewed_head_sha": reviewed_head_sha,
                        "live_head_sha": live_head_sha,
                    }
                )
        return self._apply_unauthorized_merge_baseline(candidates)

    def _apply_unauthorized_merge_baseline(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Bound the tripwire to merges it could actually have governed.

        The tripwire asserts a policy — every merged worker PR is covered by an
        ``approved`` decision recorded at its merged head — that this
        repository's history predates. ``merged_pr_list()`` returns the last 500
        closed PRs, and many of those merged before the review gate existed,
        under the #597 hollow-verdict bug, or with no decision file at all.
        Measured against the live repo before this landed, an unbounded first
        pass yields 48 findings.

        Those 48 would append to ``loop()``'s ``errors`` bucket on *every* pass
        — there is no dedupe and the 500-PR window keeps them in scope for a
        very long time — pinning ``ok=False`` permanently and burying any real
        self-merge in constant background noise. That is the same "noise, not
        signal" failure this tripwire exists to prevent, arriving from the other
        direction: a control that can never go quiet is not a control.

        So the first pass *arms* rather than alarms. It records exactly which PRs
        were already merged-and-uncovered, emits them once as an
        ``unauthorized_merge_baseline_armed`` event so the backlog stays
        auditable, and reports nothing. Every later pass reports only merges
        absent from that baseline.

        Why an explicit set of PR numbers and not a high-water PR number: a
        number watermark also exempts any PR that was already open when the
        control armed but merges afterwards. That is not hypothetical here — at
        arming there were three open worker PRs (#510, #585, #630), every one
        numbered below the highest merged PR (#631), so a number watermark would
        have permanently exempted all three, including the PR that adds this
        tripwire. The set is derived at runtime from live data, never hard-coded,
        and suppresses precisely the merges that already happened.

        Cost, stated plainly: a genuine bypass that lands in the same window as
        the arming pass is baselined rather than reported. It still appears in
        the armed event's PR list, which is why that event carries the full set
        rather than just a count.
        """
        import logging

        logger = logging.getLogger(__name__)
        key = UNAUTHORIZED_MERGE_BASELINE_KEY

        def _suppressed(baseline: Any) -> set[int]:
            if not isinstance(baseline, dict):
                return set()
            raw = baseline.get("pre_existing_prs") or []
            return {int(n) for n in raw if isinstance(n, int) and not isinstance(n, bool)}

        state = load_state_locked(self.paths.state_file)
        if isinstance(state.get(key), dict):
            pre_existing = _suppressed(state.get(key))
            return [c for c in candidates if c["pr"] not in pre_existing]

        # ---- arming pass ----
        pre_existing_now = sorted({int(c["pr"]) for c in candidates})

        if self.dry_run:
            # A preview must not write state (issues #609/#613/#621). Report what
            # an armed pass would report — nothing — without persisting, so the
            # preview is both write-free and truthful about post-arm behaviour.
            logger.info(
                "DRY-RUN: unauthorized-merge tripwire would arm with %d pre-existing "
                "uncovered merge(s); not persisting",
                len(pre_existing_now),
            )
            return []

        with state_lock(self.paths.state_file):
            locked = load_state(self.paths.state_file)
            if isinstance(locked.get(key), dict):
                # Another pass armed between the read above and this lock. Its
                # baseline wins, so arming is idempotent and never re-widens.
                pre_existing = _suppressed(locked.get(key))
                return [c for c in candidates if c["pr"] not in pre_existing]
            locked[key] = {
                "armed_at": utc_now(),
                "pre_existing_prs": pre_existing_now,
            }
            locked = self._record_event(
                locked,
                "unauthorized_merge_baseline_armed",
                {
                    "pre_existing_count": len(pre_existing_now),
                    "pre_existing_prs": pre_existing_now,
                },
            )
            save_state(self.paths.state_file, locked)

        logger.warning(
            "unauthorized-merge tripwire armed: %d pre-existing uncovered merge(s) "
            "recorded as baseline and will not be reported again (%s)",
            len(pre_existing_now),
            ", ".join(f"#{n}" for n in pre_existing_now) or "none",
        )
        return []

    def _review_decision(self, pr_number: int) -> dict[str, Any]:
        decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
        if not decision_path.exists():
            return {"decision": "missing"}
        try:
            with decision_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"decision": "invalid"}
        return value if isinstance(value, dict) else {"decision": "invalid"}

    def _read_packet_head_oid(self, pr_number: int) -> str | None:
        """Return the ``headRefOid`` stored in the existing review packet for
        ``pr_number``, or ``None`` if no packet exists or it cannot be read.

        Used by ``loop()`` to detect same-head PRs whose review packet is already
        current, so repeated supervised passes don't regenerate the packet or
        re-fire ``review_started`` label transitions while the operator is still
        reading the packet.
        """
        pr_json_path = self.paths.prs / f"pr-{pr_number}" / "pr.json"
        if not pr_json_path.exists():
            return None
        try:
            with pr_json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        value = data.get("headRefOid")
        return str(value) if value is not None else None

    def _read_packet_diff(self, pr_number: int) -> str | None:
        """Return the diff text stored in the existing review packet for
        ``pr_number``, or ``None`` if no packet exists or it cannot be read.

        Mirrors ``_read_packet_head_oid``: keeps ``reviewed_patch_id`` derived
        from the diff the reviewer actually saw rather than a live re-fetch.
        """
        diff_path = self.paths.prs / f"pr-{pr_number}" / "diff.patch"
        if not diff_path.exists():
            return None
        try:
            return diff_path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _comment_pr(self, pr_number: int, summary: str) -> None:
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        body_path = pr_dir / "review-comment.md"
        body_path.write_text(summary, encoding="utf-8")
        self.gh.pr_comment(pr_number, body_path)

    def _summarize_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        declared_blockers, open_blockers = self._get_open_blockers(issue)
        return {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "url": issue.get("url"),
            "labels": sorted(label_names(issue)),
            "dispatchable": self._is_dispatchable(issue),
            "dependencies": {
                "declared": declared_blockers,
                "open": open_blockers,
            },
        }

    def _summarize_pr(self, pr: dict[str, Any]) -> dict[str, Any]:
        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "url": pr.get("url"),
            "issue_number": linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            ),
            "head": pr.get("headRefName"),
            "is_draft": pr.get("isDraft"),
            "reviewDecision": pr.get("reviewDecision"),
        }

    def _summarize_worker(self, view, health) -> dict[str, Any]:
        """Summarize a worker's state for the status() workers list.

        Args:
            view: WorkerView with worker state
            health: WorkerHealth enum value from classify_worker_health

        Returns:
            Dict with worker summary fields for status() JSON output
        """
        from .claude_code import parse_claude_events
        from .post_mortem import _events_path_from_log

        # Resolve repo_key: use view.repo_key if present, otherwise fall back to gh.name_with_owner()
        # This handles both fleet mode (repo_key populated by iter_workers) and single-repo mode
        repo = view.repo_key
        if not repo:
            try:
                repo = self.gh.name_with_owner()
            except GitHubError:
                # If gh fails, use a fallback to avoid breaking status()
                repo = "unknown"

        # Parse tool calls and usage for Claude Code sessions
        tool_calls = None
        tokens = None
        cost_usd = None

        if view.adapter_kind == "claude-code":
            # Canonical derivation (issue #329): supports both plain
            # issue-<n>.claude.log and rework-layout issue-<n>-rework.claude.log,
            # matching the events.jsonl sibling that claude_code actually writes.
            events_path = _events_path_from_log(Path(view.log_path))
            progress = parse_claude_events(events_path)
            if progress is not None:
                tool_calls = progress.tool_call_count
                tokens = progress.tokens
                cost_usd = progress.cost_usd
        # For devin sessions, these fields remain None (no structured stream)

        # Calculate budget remaining (if configured)
        budget_remaining = None
        if tokens is not None and self.config.watchdog.token_budget is not None:
            budget_remaining = max(0, self.config.watchdog.token_budget - tokens)
        elif cost_usd is not None and self.config.watchdog.cost_budget_usd is not None:
            budget_remaining = max(0, self.config.watchdog.cost_budget_usd - cost_usd)

        return {
            "repo": repo,
            "issue": view.issue_number,
            "adapter": view.adapter_kind,
            "health": health.value,
            "runtime_seconds": view.runtime_seconds(),
            "last_activity_at": view.last_activity_at,
            "tool_calls": tool_calls,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "budget_remaining": budget_remaining,
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
