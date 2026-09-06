from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adapters import (
    SessionDispatchResult,
    SessionRequest,
    cleanup_stale_session_tmp_files,
    dispatch_sessions,
    write_session_manifest,  # noqa: F401  (deliberate re-export; patched on the workflow module in tests)
)
from .claude_code import (
    launch_claude_worker,
    resolve_review_effort,
    run_quota_probe,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
)
from .api_worker import launch_api_worker
from .devin_shell import launch_devin_session
from .checks import (
    CheckSummary,
    summarize_checks,
)
from .citation_check import (
    CitationVerdict,
)
from .config import (
    ApiWorkerConfig,
    AutoMergeConfig,
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
    PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS,
    ReviewDispatchConfig,
)
from .env_sanitize import worker_github_token_findings
from .harnesses import REVIEWER_HARNESSES
from .fleet_registry import count_fleet_live_sessions, managed_repo_names  # noqa: F401  (deliberate re-export; count_fleet_live_sessions used by moved L06 delegates via _wf.)
from . import layout, status_snapshot  # noqa: F401  (deliberate re-export; layout reached via _wf.layout by orchestration/misc_reconcile.py)
from .main_ci_reclaim import reclaim_superseded_main_ci_runs  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .rescue_review import (
    LEGACY_VACUOUS_SUMMARY,
    run_cross_family_review,  # noqa: F401  (deliberate re-export; patched on the workflow module in tests)
)
from .cross_repo_gate import CrossRepoGateResult, cross_repo_gate, cross_repo_scope_gate
from .github import (
    GitHub,
    GitHubError,
    GitHubLike,
    GitHubRunResult,
    build_branch_issue_validator,
    cancel_superseded_runs,
    detect_prose_only_dependencies,
    issue_numbers_mentioned_by_pr,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    label_names,
    parse_blockers,
)
from .issue_linking import linked_issue_number

# LOAD-BEARING RE-EXPORT -- NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# Issue #1460: the attachment-budget dispatch clause and packet-section
# renderer live in `charlie_work.attachment_budget_prompt`, following the
# #1442 ratchet's prescribed remedy for over-cap files (extract + re-export
# through this facade block, matching the `.dispatch_selection` /
# `.escalation` / `.verdict_parsing` / `.rework_prompts` / `.ci_findings` /
# `.backlog_reachability` / `.stalled_review_reap` lineage above).
from .attachment_budget_prompt import (  # noqa: F401  (deliberate re-export)
    ATTACHMENT_BUDGET_CLAUSE as _ATTACHMENT_BUDGET_CLAUSE,
    render_attachment_budget_section,
)
from .cross_pr_revert import (  # noqa: F401  (deliberate re-export)
    CrossPrRevertResult,
    CrossPrRevertStatus,
    detect_cross_pr_revert,
)
from .janitor import (
    _calculate_patch_id,  # noqa: F401  (deliberate re-export; patched on the workflow module in tests)
    _diff_content_signature,  # noqa: F401  (deliberate re-export; reached via _wf. by orchestration/state_record_review.py)
    check_operator_containment,
    check_test_adequacy,
    is_stale_ci_verdict,
    run_janitor,
    DiffContentSignature,
    TestAdequacyFacts,
    TestAdequacyVerdict,
)
from .diff_coverage_probe import StaticProbeVerdict, run_static_probe
from .labels import TransitionOutcome, transition
from .paths import ResolvedLayout, RuntimePaths, resolved_layout
from .prompts import (
    PromptTemplateError,
    render_prompt,  # noqa: F401  (deliberate re-export; patched on the workflow module in tests, reached via _wf.render_prompt by orchestration/prompt_ops.py -- Tier D, #1627)
    resolve_template,
    unsupplied_placeholders,
)
from .reconcile import (  # noqa: F401  (deliberate re-export; reconcile symbols reached via charlie_work.reconcile by orchestration/misc_reconcile.py; keeps reconcile.py live in the import-reachability graph, see tests/test_dormant_fleet_marking.py)
    DriftItem,
    apply_fixes as apply_drift_fixes,
    detect_aviator_stale_blocked,
    detect_drift,
    detect_mergequeue_not_approved,
    detect_mergequeue_wedged,
)
from .review_decision import (
    record_decision,
    review_decision,
)
from .worktree import (
    _reap_idle_foreign_writer,
    clean_worktrees,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    read_worktree_marker,
    read_worker_outcome,
    remote_branch_ahead_count,
    remote_branch_head_sha,
    remove_review_checkout,  # noqa: F401  (deliberate re-export; patched on the workflow module and reached via _wf. by orchestration/misc_review_verdicts.py)
    remove_worktree_marker,  # noqa: F401  (deliberate re-export; used by moved L01 b4 delegates via _wf.)
    salvage_push_stranded_commits,
    worktree_head_sha,
    worktree_path_for_branch,
    write_worktree_marker,  # noqa: F401  (deliberate re-export; used by moved L01 b4 delegates via _wf.)
)
from . import state as _state
from .unescalate_reset_fields import (
    REWORK_BUDGET_RESET_BY_ESCALATION_REASON,
    UNESCALATE_ISSUE_RESET_FIELDS,
    UNESCALATE_PR_RESET_FIELDS,
)
from .state import (
    PASSIVE_OPEN_STATUS,
    StateLockBusy,
    append_event,
    arm_operator_queue_review,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    arm_quota_probe,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    arm_reconcile_pass,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    clear_quota_throttles,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    clear_reviewer_quota,
    defer_reviewer_probe_after,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    clear_escalation,
    clear_escalation_on_issue_prs,
    disarm_quota_probe,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    escalation_reason_class,
    is_claim_stale,
    is_operator_queue_review_due,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    is_quota_probe_actionable,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    is_quota_probe_armed,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    is_quota_probe_due,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    is_reconcile_due,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    is_reviewer_probe_ready,
    is_reviewer_quota_exhausted,
    is_throttled,
    is_worktree_reclamation_due,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    load_state,
    load_state_locked,
    mark_reviewer_quota_alerted,
    operator_claimed_issues,
    release_operator_claimed,  # noqa: F401  (deliberate re-export; used by moved L01 b4 delegates via _wf.)
    save_state,
    schedule_worktree_reclamation,  # noqa: F401  (deliberate re-export; used by moved L01 b3 delegates via _wf.)
    set_operator_claimed,  # noqa: F401  (deliberate re-export; used by moved L01 b4 delegates via _wf.)
    stale_operator_claims,
    state_lock,
    utc_now,
    without_review_dispatch_claim,
)
from .instrumentation import (
    current_correlation_id,
    log_event,
    query_events,  # noqa: F401  (deliberate re-export; used by moved L02 state_stale_checks delegate via _wf.)
)
from .preflight import (
    run_preflight,  # noqa: F401  (deliberate re-export; Tier D patch target + used by moved L05 _loop_impl delegate via _wf.)
)
from .throttle_signatures import match_throttle_tail, parse_reset_clock_time
from .process_utils import (
    find_worker_terminal_status,
    is_pid_alive,
)
from .write_gate import WriteGate, require_write_gate

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A: the dispatch-selection free-function family (candidate
# selection for fresh/rework dispatch and for review dispatch, plus the two
# frozen dataclasses those selectors return) now lives in
# `charlie_work.dispatch_selection`. Re-exported here, not re-declared, so
# every existing `charlie_work.workflow.<name>` import and monkeypatch target
# keeps resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig`.
from .dispatch_selection import (  # noqa: F401  (deliberate re-export)
    LocalReviewCapResult,
    ReviewDispatchSelection,
    parse_issue_numbers,
    _MAX_RECOVERY_RETRY_PER_PASS,
    _MAX_DEFERRED_CONCURRENCY_EXAMPLES,
    _is_recovery_candidate,
    _select_dispatch_candidates,
    _select_rework_candidates,
    _reviewer_pid_alive,
    _count_live_reviews,
    _apply_local_review_cap,
    _windowed_redispatch_at,
    _windowed_worker_death_at,
    _windowed_orphan_redispatch_at,
    _windowed_blocked_environment_at,
    _windowed_foreign_writer_reaps,
    _is_review_dispatchable,
    _select_review_dispatch_candidates,
)
from . import orchestration as _orchestration
from .workflow_delegation import _install_delegates, discover_delegate_modules

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A: the escalation free-function family (the shared
# escalation-flag check, the de-escalation skip-outcome builder, the
# escalate-issue mutator, and the escalated-label repair predicates) now
# lives in `charlie_work.escalation`. Re-exported here, not re-declared, so
# every existing `charlie_work.workflow.<name>` import and monkeypatch target
# keeps resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` block
# above.
from .escalation import (  # noqa: F401  (deliberate re-export)
    _escalation_flags,
    _deescalation_skip,
    _escalate_issue,
    _escalated_label_needs_repair,
    _collect_escalated_label_subjects,
    _escalation_edge,
    _escalation_label,
    _repair_reason_class,
    _worker_launched_before_cap_escalation,
    _cap_escalation_pr_extra,
    _reset_linked_pr_status_to_passive_open,
    _MECHANICAL_ESCALATION_EDGES,
)

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A: the reviewer-verdict-parsing free-function family
# (fenced-JSON extraction, stream-json event decoding, mtime-gated file
# fallback recovery, and reviewer-session-summary reconstruction), plus the
# frozen dataclass the summary function returns, now lives in
# `charlie_work.verdict_parsing`. Re-exported here, not re-declared, so
# every existing `charlie_work.workflow.<name>` import and monkeypatch target
# keeps resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` /
# `.escalation` blocks above.
from .verdict_parsing import (  # noqa: F401  (deliberate re-export)
    ReviewSessionOutcome,
    REVIEW_MISS_TURN_LIMIT,
    REVIEW_MISS_LAUNCH_FAILED,
    REVIEW_MISS_DIED_MID_SESSION,
    CAUSE_UNKNOWN,
    _VERDICT_FENCE_RE,
    _REVIEW_MD_PATH_RE,
    _REVIEW_FALLBACK_FILE_MAX_BYTES,
    _REVIEW_FALLBACK_MTIME_SLACK_S,
    _REVIEW_FALLBACK_MAX_CANDIDATES,
    _DEFAULT_REVIEW_SESSION_LIMIT_MARKERS,
    _EXTRACTED_VERDICT_SOURCES,
    _REVIEW_THROTTLE_TAIL_CHARS,
    _RESULT_EVENT_CAUSE_FIELDS,
    _validate_review_verdict,
    _extract_verdict_from_text,
    _extract_verdict_from_stream_json,
    _parse_review_verdict_from_log,
    _parse_review_verdict_from_events,
    _parse_review_verdict_from_files,
    _reviewer_session_metrics,
    _log_tail_throttled,
    _extract_terminating_cause,
    _extract_review_session_summary,
    REVIEW_SESSION_FAILED_HEADING,
    REVIEW_SESSION_SUMMARY_HEADING,
    body_has_crash_signature,
    is_extracted_verdict_source,
    provenance_caveat_for,
)

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A: the rework-prompt-rendering free-function family (the
# required-changes-section tiered renderer, the review-decision JSON reader
# and freshness comparison, the per-round retry/distinct-verdict archive
# numbering, and the pure-render / atomic-write split behind
# ``_write_rework_prompt``) now lives in `charlie_work.rework_prompts`.
# Re-exported here, not re-declared, so every existing
# `charlie_work.workflow.<name>` import and monkeypatch target keeps
# resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` /
# `.escalation` / `.verdict_parsing` blocks above.
from .rework_prompts import (  # noqa: F401  (deliberate re-export)
    _EXTERNAL_FINDINGS_POINTER,
    _EXTERNAL_FINDINGS_SECTION_INTRO,
    _REQUIRED_CHANGES_TIER1_INTRO,
    _ROUND_COMPARE_KEYS,
    _existing_round_numbers,
    _finish_required_changes_section,
    _is_verdict_newer_than_brief,
    _next_round_number,
    _provenance_caveat_from_decision,
    _read_review_decision,
    _render_external_findings_section,
    _render_required_changes_section,
    _render_round_findings,
    _render_rework_prompt,
    _round_history_entries,
    _rework_prompt_search_dirs,
    _write_rework_prompt,
    _write_text_atomic,
)

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A: the CI-checks-findings free-function family (the
# CI-status packet section renderer, the dispatch-staleness detector, and
# the required-check-annotation-to-required-changes converter -- three
# call-graph-disconnected sub-clusters combined under one destination
# theme, see the module docstring) now lives in `charlie_work.ci_findings`.
# Re-exported here, not re-declared, so every existing
# `charlie_work.workflow.<name>` import and monkeypatch target keeps
# resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` /
# `.escalation` / `.verdict_parsing` / `.rework_prompts` blocks above.
from .ci_findings import (  # noqa: F401  (deliberate re-export)
    _ci_status_section,
    _non_required_check_findings,
    _backlog_is_non_empty,
    _latest_non_empty_dispatch,
    _parse_iso_ts,
    check_dispatch_staleness,
    _annotation_to_required_change,
    _required_changes_from_checks,
)

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283: the backlog-reachability free-function pair (the shared
# open-blocker check and the unfiltered-backlog reachability classifier that
# calls it) -- call-graph disconnected from every Phase-A family (A1-A5)
# above, extracted as its own standalone standard-shape PR per operator
# approval -- now lives in `charlie_work.backlog_reachability`. Re-exported
# here, not re-declared, so every existing `charlie_work.workflow.<name>`
# import and monkeypatch target keeps resolving unchanged — the same
# pattern `config.py` uses for `RunnerAllocationConfig` and this file's own
# `.dispatch_selection` / `.escalation` / `.verdict_parsing` /
# `.rework_prompts` / `.ci_findings` blocks above.
from .backlog_reachability import (  # noqa: F401  (deliberate re-export)
    _get_open_blockers_for_issue,
    classify_backlog_reachability,
    compute_mention_coverage_map,
    fetch_merged_prs_fail_open,
    resolve_dispatch_mention_coverage,
    scan_merged_pr_references,
)

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1283 Phase A, PR 6 of 6 (staged split): the stalled-review-reap
# free-function family (10 units -- see the module docstring for the full
# list and the corrected call-graph cohesion rationale; the 25-member
# dead-worker/session half is spun off to issue #1317) now lives in
# `charlie_work.stalled_review_reap`. Re-exported here, not re-declared, so
# every existing `charlie_work.workflow.<name>` import and monkeypatch
# target keeps resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` /
# `.escalation` / `.verdict_parsing` / `.rework_prompts` / `.ci_findings` /
# `.backlog_reachability` blocks above.
from .stalled_review_reap import (  # noqa: F401  (deliberate re-export)
    _remove_review_checkout_with_warning,
    _set_reviewer_quota_exhausted_with_backoff,
    _merge_on_write_save,
    _ThrottleClassification,
    _detect_and_handle_stalled_reviews,
    _reap_review_sidecar,
    _reap_completed_review_checkouts,
    _reap_orphaned_review_checkouts,
    _classify_review_dispatch_stalled_level,
    _append_sweep_events,
)
from .queue_sync_coverage import (  # noqa: F401  (deliberate re-export)
    _QUEUE_SYNC_RETRY_ATTEMPTS,
    _QUEUE_SYNC_RETRY_BACKOFF_SECONDS,
    _QUEUE_SYNC_RETRY_SLEEP,
    _QueueSyncCoverageResult,
    _fetch_commit_retrying,
    _fetch_compare_retrying,
    _queue_sync_merge_covered,
)


# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# issue #1317 (spun off from #1283's A6 recon): the dead-worker/session-reap
# free-function family (25 functions plus the two threshold constants below
# -- see the module docstring for the full list, the deliberate
# `_detect_and_handle_orphaned_workers` exclusion rationale, and the
# disclosed call-graph judgment calls) now lives in
# `charlie_work.dead_worker_reap`. Re-exported here, not re-declared, so
# every existing `charlie_work.workflow.<name>` import and monkeypatch
# target keeps resolving unchanged — the same pattern `config.py` uses for
# `RunnerAllocationConfig` and this file's own `.dispatch_selection` /
# `.escalation` / `.verdict_parsing` / `.rework_prompts` / `.ci_findings` /
# `.backlog_reachability` / `.stalled_review_reap` blocks above.
#
# `_detect_and_handle_orphaned_workers` itself deliberately stays defined
# below in this file -- see its docstring and issue #1317 for why.
from .dead_worker_reap import (  # noqa: F401  (deliberate re-export)
    STARTUP_DEATH_THRESHOLD_SECONDS,
    _is_startup_death,
    _worker_death_bounded_runtime_seconds,
    _session_failed_relabeled_payload,
    _emit_session_failed_relabeled,
    _count_live_sessions,
    _detect_stalled_sessions,
    _detect_and_handle_stalled_sessions,
    _worker_pid_alive,
    _orphan_head_fingerprint,
    _ZERO_ARTIFACT_ESCALATION_THRESHOLD,
    _is_zero_artifact_dispatch_loop,
    _sweep_orphan_processes_for_dead_sessions,
    _log_worker_census,
    _rework_pr_for_worker,
    _reap_restore_rework_requested,
    _is_pr_updated_at_older_than,
    _is_pre_review_rework_candidate,
    _route_dead_worker_to_pre_review_rework,
    _classify_dead_sessions_and_update_throttle_state,
    _safe_repo_slug,
    _dispatching_repo_name,
    _open_salvage_pr,
    _salvage_already_landed,
    _attempt_salvage,
    _open_pr_for_orphaned_branch,
    _issues_with_live_workers,
)


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


def _max_touched_file_line_count(diff: str, repo_root: Path) -> int:
    """Return the largest line count among files touched by ``diff`` (issue #1439).

    The review turn cap scales with the size of the files a diff touches, not
    the diff itself: a PR threading a 25k-line monolith costs the reviewer far
    more grep -> Read-window navigation than the diff's line count suggests.
    File sizes are read from ``repo_root`` (the orchestrator's checkout). For
    files not present there (e.g. newly added in the PR), the per-file added
    line count from the diff is used as the size -- a new file's size IS its
    added lines. Returns 0 for an empty diff.
    """
    _total, files = _diff_file_summary(diff)
    max_lines = 0
    for name, added, _deleted in files:
        if not name:
            continue
        candidate = name
        # Unified diff ``+++`` paths are prefixed with ``b/``; strip it so the
        # path resolves under ``repo_root``. A literal ``b/...`` file would be
        # vanishingly rare and only makes the lookup miss (falling back to the
        # added-line count), never reads the wrong file.
        if candidate.startswith("b/"):
            candidate = candidate[2:]
        path = repo_root / candidate
        line_count = 0
        if path.exists() and path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    line_count = sum(1 for _ in handle)
            except OSError:
                line_count = 0
        else:
            # New file (not present at repo_root): its size is the added lines.
            line_count = added
        if line_count > max_lines:
            max_lines = line_count
    return max_lines


@dataclass(frozen=True)
class OverCapFileFinding:
    """One file whose post-diff line count exceeds the repo size cap and whose
    diff adds code to it (issue #1445).

    ``line_count`` is the post-diff line count (base file at ``repo_root`` plus
    the diff's net added lines, or the added-line count for a new file).
    ``added_lines`` is the diff's gross added content lines for this file --
    the quantity the rubric flags as "new code added to an over-cap file".
    """

    filename: str
    line_count: int
    cap: int
    added_lines: int


def _over_cap_file_findings(
    diff: str, repo_root: Path, cap: int
) -> tuple[OverCapFileFinding, ...]:
    """Detect files this diff adds code to that are over the repo size cap.

    Returns findings only for files that (a) have at least one added content
    line in the diff and (b) whose post-diff line count exceeds ``cap``. A
    ``cap`` of 0 disables the check (returns ``()``) -- mirrors
    ``turn_cap_large_file_threshold``'s 0-disables convention.

    File sizes are read from ``repo_root`` (the orchestrator's checkout) plus
    the diff's net added lines, mirroring ``_max_touched_file_line_count``; a
    new file's size is its added lines. Best-effort: a file that cannot be read
    is skipped, never raises -- the same posture as the static probe and
    ``check_operator_containment``.
    """
    if cap <= 0:
        return ()
    _total, files = _diff_file_summary(diff)
    findings: list[OverCapFileFinding] = []
    for name, added, deleted in files:
        if not name or added <= 0:
            continue
        # Unified diff ``+++`` paths are prefixed with ``b/``; strip it so the
        # path resolves under ``repo_root`` and the reported filename is the
        # repo-relative path (not the diff's ``b/``-prefixed form). A literal
        # ``b/...`` file would be vanishingly rare and only makes the lookup
        # miss (falling back to the added-line count), never reads the wrong
        # file -- mirrors ``_max_touched_file_line_count``'s stance.
        filename = name[2:] if name.startswith("b/") else name
        path = repo_root / filename
        if path.exists() and path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    base = sum(1 for _ in handle)
            except OSError:
                continue
            line_count = base + added - deleted
        else:
            # New file (not present at repo_root): its size is the added lines.
            line_count = added
        if line_count > cap:
            findings.append(OverCapFileFinding(filename, line_count, cap, added))
    return tuple(findings)


def render_over_cap_section(findings: tuple[OverCapFileFinding, ...] | None) -> str:
    """Render the ``$over_cap_section`` packet block (issue #1445).

    Returns ``""`` when ``findings`` is ``None`` (cap disabled -- the caller in
    ``review()`` passes ``None`` when ``file_size_cap_lines`` is 0), mirroring
    ``render_static_probe_section``'s disabled contract. When enabled, this
    ALWAYS renders visible text -- even with zero findings -- rather than ``"``
    for a clean pass, mirroring ``render_static_probe_section``'s never-silent
    contract: an advisory probe that goes silent on a clean run is
    indistinguishable, from the rendered packet alone, from one that never ran.
    """
    if findings is None:
        return ""
    if not findings:
        return "File-size cap: no over-cap additions in this diff.\n"
    lines = ["**Over-cap file additions (issue #1445):**"]
    for finding in findings:
        lines.append(
            f"- `{finding.filename}`: {finding.line_count} lines "
            f"(cap {finding.cap}), +{finding.added_lines} added -- "
            "REPORTABLE FINDING. Suggested remedy: extract the new code to a "
            "domain module (facade re-export block in the monolith, "
            "implementation in the module), matching the #1283-era extractions. "
            "Then run `python scripts/refresh_file_size_ratchet.py` and commit "
            "the resulting `file_size_ratchet_baseline.json` tightening in this "
            "PR -- the script's default mode is lower-only (never raises a "
            "mark), so it is safe to run mid-PR (#1495)."
        )
    return "\n".join(lines) + "\n"


def structure_turn_cap_multiplier(
    max_touched_file_lines: int, config: ReviewDispatchConfig
) -> int:
    """Structure-aware multiplier for the review turn cap (issue #1439).

    Returns ``turn_cap_large_file_multiplier`` (clamped to
    ``turn_cap_max_multiplier``) when ``max_touched_file_lines`` exceeds
    ``turn_cap_large_file_threshold``, else 1. A threshold of 0 disables the
    structure bonus (every diff uses the base cap).
    """
    threshold = config.turn_cap_large_file_threshold
    if threshold <= 0:
        return 1
    if max_touched_file_lines > threshold:
        return min(config.turn_cap_large_file_multiplier, config.turn_cap_max_multiplier)
    return 1


def resolve_review_turn_cap(
    base_max_turns: int,
    structure_multiplier: int,
    miss_streak: int,
    config: ReviewDispatchConfig,
) -> int:
    """Final reviewer turn cap after structure + miss-escalation (issue #1439).

    ``effective_multiplier = min(structure_multiplier + miss_streak,
    turn_cap_max_multiplier)``, clamped to a floor of 1.
    ``final_cap = base_max_turns * effective_multiplier``. A ``base_max_turns``
    of 0 (unlimited) stays 0.
    """
    if base_max_turns <= 0:
        return 0
    effective = min(
        max(structure_multiplier + miss_streak, 1),
        config.turn_cap_max_multiplier,
    )
    return base_max_turns * effective


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
        "pass_skipped": True,
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


# No leading underscore: this is a deliberate cross-module wire contract
# with fleet_dispatch._add_launch_failures, which matches this prefix to
# exclude concurrency-deferred issues from launch-failure attention events.
# Matching the reason string (rather than cross-referencing the
# `deferred_by_concurrency` list) is deliberate: that list is truncated to
# _MAX_DEFERRED_CONCURRENCY_EXAMPLES entries in the persisted payload
# (issue #1005), but this `failures` map is not -- a set-membership check
# against the truncated list would silently re-report the 6th+ deferred
# issue as a genuine launch failure.
DEFERRED_BY_CONCURRENCY_REASON_PREFIX = "deferred by concurrency cap"


def _build_failure_map(
    dispatch_results: Sequence[SessionDispatchResult],
    failed_issue_numbers: Iterable[int],
    deferred_by_concurrency: Iterable[int],
    limit: int,
    extra_failures: Mapping[int, str] | None = None,
) -> dict[int, str]:
    failures: dict[int, str] = {}
    for issue_number in deferred_by_concurrency:
        failures[issue_number] = _truncate_reason(
            f"{DEFERRED_BY_CONCURRENCY_REASON_PREFIX} (limit: {limit})"
        )
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
    # Issue #1129: open-PR backpressure fields. Populated only when the
    # governor was called with ``apply_open_pr_backpressure=True`` (fresh-issue
    # dispatch) and ``dispatch.max_open_agent_prs`` is > 0. Left at 0 for
    # rework/recovery/loop paths, which are exempt from this clamp.
    open_pr_count: int = 0
    open_pr_max: int = 0

    @property
    def enabled(self) -> bool:
        """Return True if the governor is enabled (max_concurrent > 0)."""
        return self.max_concurrent > 0

    @property
    def fleet_enabled(self) -> bool:
        """Return True if the fleet governor is enabled (fleet_max > 0)."""
        return self.fleet_max > 0

    @property
    def open_pr_enabled(self) -> bool:
        """Return True if the open-PR backpressure clamp is enabled (open_pr_max > 0)."""
        return self.open_pr_max > 0

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
        if self.open_pr_enabled:
            fields["open_pr_count"] = self.open_pr_count
            fields["open_pr_max"] = self.open_pr_max
        return fields


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


def _summary_is_vacuous(summary: str) -> bool:
    """True when ``summary`` carries no reviewer content.

    Covers a blank/whitespace-only string, and the historical
    ``LEGACY_VACUOUS_SUMMARY`` placeholder (issue #784) that pre-#784
    cross-family code emitted for a verdict that was, in fact, content-free.
    This is the single definition of "there is nothing here to act on",
    used by ``record_review`` (write-time, issue #792): is there anything to
    derive from ``summary`` into ``required_changes`` when the caller
    didn't supply a structured list?

    Deliberately does *not* do anything smarter than an equality check
    against one known, exported constant -- a referent/keyword classifier
    over free-form reviewer prose was evaluated and rejected (issue #792)
    because it misclassified genuine, specific findings (e.g. architectural
    gaps with no file/line reference, or "CI failed on Lint; push a fix")
    as content-free. The only text this function is entitled to call
    vacuous is text our *own* code emitted as a known placeholder, not text
    a reviewer wrote that merely looks terse or unstructured.
    """
    stripped = summary.strip()
    return not stripped or stripped == LEGACY_VACUOUS_SUMMARY


# Statuses that mean an issue already has a rework routed (or an equivalent
# gate) in flight, or is otherwise spoken for -- shared by every "should I
# route this PR to rework_requested" check so they cannot drift apart.
# Originally the cross-PR-revert gate in ``merge_ready`` (below); issue #784
# AC-8 Case 2 reuses it verbatim for ``review_queue``'s stranded-verdict
# check rather than inventing a second list.
_REWORK_ALREADY_ROUTED_STATUSES = (
    "escalated",
    "blocked",
    "dispatched",
    "dispatch_pending",
    "manifest_written",
    "rework_requested",
)


def _janitor_section(warnings: tuple[str, ...]) -> str:
    if not warnings:
        return ""
    lines = "\n".join(f"- {warning}" for warning in warnings)
    return (
        "\n## Janitor warnings (non-blocking)\n\n"
        f"{lines}\n\n"
        "These deterministic pre-checks passed the gate but deserve reviewer attention.\n"
    )


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


def render_static_probe_section(verdict: StaticProbeVerdict | None) -> str:
    """Render the $static_probe_section packet block from a StaticProbeVerdict.

    Returns "" when ``verdict`` is None (probe disabled -- the caller in
    review() passes None in that case), mirroring
    ``render_test_adequacy_section``'s disabled contract.

    When enabled, this ALWAYS renders visible text -- even with zero findings
    and zero warnings -- rather than "" for a clean pass. This mirrors
    render_test_adequacy_section's own behavior of never going silent once
    its gate is enabled (that function always shows its facts block
    regardless of whether there is a problem to report): an advisory probe
    that goes silent on a clean run is indistinguishable, from the rendered
    packet alone, from one that never ran at all (issue #1260/#1261 design
    item 7's "zero-reads-as-green" concern applies to the clean case too,
    not only to internal errors).
    """
    if verdict is None:
        return ""

    lines: list[str] = []
    if verdict.warnings:
        lines.extend(f"- {warning}" for warning in verdict.warnings)

    if verdict.branch_findings:
        if lines:
            lines.append("")
        lines.append("**Branch-coverage heuristic (W3):**")
        for finding in verdict.branch_findings:
            reason = (
                "no added test/assertion lines in this diff"
                if finding.reason == "no_test_adds"
                else f"branch:test-add ratio high ({finding.branch_adds}:{finding.test_adds})"
            )
            lines.append(f"- `{finding.filename}`: {finding.branch_adds} branch-adds, {reason}")

    if verdict.unwired_findings:
        if lines:
            lines.append("")
        lines.append("**Unwired-symbol probe (W20):**")
        for finding in verdict.unwired_findings:
            lines.append(
                f"- `{finding.symbol}` ({finding.kind}) in `{finding.filename}`: referenced "
                "only from tests/, no src/ caller found"
            )

    if not lines:
        return "Static probe: no findings.\n"

    return "\n".join(lines) + "\n"


def slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "work"


# Maximum per-worktree skip entries carried in the persisted
# `worktrees_reclaimed` event payload. Same idiom as
# `_MAX_DEFERRED_CONCURRENCY_EXAMPLES`: a full count (`skipped_count`) alongside a
# truncated example list (`skipped_examples`), so a standing backlog of
# stuck worktrees doesn't re-emit the full list into events.db every
# interval, while a normal-sized backlog still carries every reason string
# through (issue #1012 -- clean_worktrees computes a distinct `reason` per
# skip but the event used to keep only `skipped_count`).
_MAX_SKIPPED_WORKTREE_EXAMPLES = 20

# Bound on concurrent `gh` subprocesses spawned by _prefetch_blocker_data() to
# warm the dependency cache across every ready issue. I/O-bound fan-out width,
# not a CPU budget; kept modest to stay clear of GitHub's secondary rate
# limits. See issue #870.
_MAX_PREFETCH_WORKERS = 8


# state.json key holding the one-time record of worker-PR merges that predate the
# #502 post-merge tripwire. Named here rather than inlined so the arming logic and
# the tests that pre-arm it cannot drift apart on a string literal.
UNAUTHORIZED_MERGE_BASELINE_KEY = "unauthorized_merge_baseline"

# state.json key holding the ongoing acknowledgment set for post-arming
# unauthorized-merge findings (issue #673). The baseline (#510) suppresses
# *pre-arming* history once; this set suppresses *post-arming* findings that a
# human or automated remediation has explicitly triaged, so a confirmed,
# already-actioned finding stops pinning ok=False on every subsequent pass.
# Like the baseline it is an explicit set of PR numbers (never a high-water
# mark) and is only ever populated by an explicit `charlie tripwire ack` action
# — the tripwire never auto-acknowledges, or it would defeat itself.
UNAUTHORIZED_MERGE_ACK_KEY = "unauthorized_merge_acknowledged"

# state.json key holding the durable first-detection record for post-arming
# unauthorized-merge findings (issue #933). A `{pr_number: {detected_at, head,
# issue, decision, ...}}` map, written the first time a finding is reported and
# not rewritten while the finding stays open, so the
# `unauthorized_merge_detected` event fires exactly once per PR instead of once
# per pass. `ack_unauthorized_merge` deletes the entry, so the map holds
# *currently-open announced findings* — bounded by the open set, not growing
# with every PR ever found, and a finding re-detected after its ack is withdrawn
# announces again instead of returning silently.
#
# Once-per-PR is the whole point. The tripwire re-detects the same unacked
# finding on every pass — that is the 21-pass ok=False streak #933 was filed on
# — so a per-pass event would be a second unbounded stream restating one fact
# indefinitely, which is the "a control that can never go quiet is not a
# control" failure the baseline and the ack set both exist to prevent.
#
# This is a *record*, not a suppressor: unlike the baseline and the ack set it
# is never consulted by `_apply_unauthorized_merge_baseline`, so a finding
# stays reported (and keeps pinning ok=False) until it is explicitly acked.
# Presence here silences the event, never the finding.
UNAUTHORIZED_MERGE_DETECTED_KEY = "unauthorized_merge_detected"


# Issue #934: an operator who legitimately adjudicates and merges a worker PR
# whose recorded review decision is stale, absent, or pending has no way to
# record that adjudication, so every legitimate operator merge becomes a
# tripwire finding. ``merge_authorize`` writes an ``authorized_override`` into
# the PR's ``review-decision.json``; the tripwire and ``merge-check`` treat an
# override whose ``authorized_sha`` matches the live head as explicit
# authorization. This helper extracts and validates that override from a
# decision dict so both consumers share one enforcement point.
#
# An override that does not name a non-empty SHA or a non-empty reason is not a
# valid override — it is treated as absent so the control never reads a
# malformed record as authorization. A control that can be silenced with an
# empty reason is no control, and an authorization that does not name the SHA
# it authorizes reintroduces the exact hole (#802/#804: rebase moved the head
# after the decision was recorded) the SHA-binding exists to close.


def _authorized_override_matches(decision: dict[str, Any], live_head_sha: str | None) -> bool:
    """Return ``True`` if ``decision`` carries a valid override for ``live_head_sha``.

    Shared by ``_detect_unauthorized_merges`` (the post-merge tripwire) and
    ``merge_check`` (the pre-merge preflight) so the two paths cannot drift on
    what counts as authorization. Both must answer the same question — "is
    there an explicit, SHA-bound, reason-bearing authorization for this head?"
    — or a merge that passes the preflight still trips the post-merge control.
    """
    override = decision.get("authorized_override")
    if not isinstance(override, dict):
        return False
    authorized_sha = override.get("authorized_sha")
    reason = override.get("reason")
    return (
        isinstance(authorized_sha, str)
        and bool(authorized_sha)
        and authorized_sha == live_head_sha
        and isinstance(reason, str)
        and bool(reason.strip())
    )


# Caps for the error detail carried in the `loop_completed` payload. The
# loop-body `errors` list is unbounded by construction — one entry per PR that
# raised — and this payload lands in both the capped 200-entry `events` array in
# state.json and the unlimited events.db, so a single bad pass could otherwise
# push dozens of full exception strings through both.
_LOOP_ERROR_MAX_PRS = 20
_LOOP_ERROR_MAX_DETAILS = 5
_LOOP_ERROR_DETAIL_CHARS = 300

# Cap for the issue-number list in the ``rework_issue_fetch_skipped`` event.
# A repo-wide gh outage can make every rework candidate's ``issue_view``
# fail in one pass; the list is capped so one such pass cannot push an
# unbounded payload through the 200-entry event ring.
_REWORK_ISSUE_FETCH_SKIPPED_MAX_ISSUES = 20
_REWORK_ISSUE_FETCH_SKIPPED_REASON_CHARS = 300
# Cap for the distinct-reason set. A single pass can mix failure modes that
# want opposite operator responses -- one issue 404'd because it was deleted
# (stop retrying), another timed out transiently (retry). Keeping only the
# first exception as ``reason`` (issue #970) collapses those into one
# representative and loses the distinction. This mirrors ``summarize_loop_errors``'
# ``max_details`` idiom: a small distinct set, capped with an explicit
# ``reasons_truncated`` count rather than silently dropped.
_REWORK_ISSUE_FETCH_SKIPPED_MAX_REASONS = 5


def _build_rework_issue_fetch_skip_payload(
    failures: list[tuple[int, Exception]],
    *,
    max_issue_numbers: int = _REWORK_ISSUE_FETCH_SKIPPED_MAX_ISSUES,
    reason_chars: int = _REWORK_ISSUE_FETCH_SKIPPED_REASON_CHARS,
    max_reasons: int = _REWORK_ISSUE_FETCH_SKIPPED_MAX_REASONS,
) -> dict[str, Any]:
    """Summarize per-issue ``gh.issue_view`` failures into a bounded payload.

    ``rework_issue_fetch_skipped`` is intentionally one event per pass, not
    one per issue, so a repo-wide ``gh`` outage does not emit N events for
    the same root cause. The issue list is capped with an explicit
    ``issue_numbers_truncated`` count so the record does not understate the
    number of affected issues. The reason uses the first exception as a
    representative root cause.

    ``reasons`` keeps up to ``max_reasons`` *distinct* ``(reason, error_type)``
    pairs (issue #970). A single pass can mix failure modes that want opposite
    operator responses -- one issue 404'd because it was deleted (stop
    retrying), another timed out transiently (retry) -- and a single
    representative ``reason`` collapses that distinction. Dedup is by
    ``(reason, error_type)`` so a repeated identical outage does not crowd out
    a rarer distinct one, matching ``summarize_loop_errors``' ``error_details``
    shape. ``reason``/``error_type`` are retained as the first representative
    for backward compatibility with the #940-style consumer that reads a
    single ``reason``.
    """
    if not failures:
        return {
            "issue_numbers": [],
            "issue_numbers_truncated": 0,
            "reason": "",
            "error_type": "",
            "reasons": [],
            "reasons_truncated": 0,
        }

    issue_numbers = sorted({issue for issue, _ in failures})
    representative = failures[0][1]
    reason = str(representative) or representative.__class__.__name__
    if len(reason) > reason_chars:
        reason = reason[: reason_chars - 3] + "..."

    # Distinct (reason, error_type) pairs, preserving first-seen order so the
    # representative above is always ``reasons[0]``. Dedup is on the truncated
    # reason text + class name, not the raw exception, so two exceptions with
    # the same message and class count as one root cause.
    seen: set[tuple[str, str]] = set()
    distinct: list[dict[str, Any]] = []
    for _, exc in failures:
        text = str(exc) or exc.__class__.__name__
        if len(text) > reason_chars:
            text = text[: reason_chars - 3] + "..."
        cls = exc.__class__.__name__
        key = (text, cls)
        if key in seen:
            continue
        seen.add(key)
        distinct.append({"reason": text, "error_type": cls})

    return {
        "issue_numbers": issue_numbers[:max_issue_numbers],
        "issue_numbers_truncated": max(0, len(issue_numbers) - max_issue_numbers),
        "reason": reason,
        "error_type": representative.__class__.__name__,
        "reasons": distinct[:max_reasons],
        "reasons_truncated": max(0, len(distinct) - max_reasons),
    }


def summarize_loop_errors(
    errors: list[dict[str, Any]],
    *,
    max_prs: int = _LOOP_ERROR_MAX_PRS,
    max_details: int = _LOOP_ERROR_MAX_DETAILS,
    detail_chars: int = _LOOP_ERROR_DETAIL_CHARS,
) -> dict[str, Any]:
    """Summarize `_loop_body`'s `errors` list into a bounded event payload.

    `loop_completed` previously kept only `error_count`, so a pass that reported
    ``"loop completed with 1 PR error(s)"`` left no stored artifact naming *which*
    PR — diagnosing a live finding meant reading the detector source and
    re-deriving the candidate set by hand (issue #933).

    PR numbers are the cheap, high-value half and are emitted first and most
    generously; free-text detail is the expensive half and is truncated hard.
    Both are capped with an explicit ``*_truncated`` count rather than silently
    dropped, so a reader can always tell a complete list from an elided one — a
    silent cap would reproduce the same "the record understates the problem"
    failure this function exists to fix.

    The two ``*_truncated`` counts are over **different populations**, which the
    similar names hide: ``error_prs_truncated`` counts elided *distinct PRs*
    (deduped), while ``error_details_truncated`` counts elided *error entries*
    (not deduped — several entries can share one PR). So ``error_prs_truncated:
    6`` alongside ``error_details_truncated: 23`` is consistent, not a
    contradiction; they answer "how many more PRs" and "how many more failures".
    """
    pr_numbers: list[int] = []
    for entry in errors:
        raw = entry.get("pr")
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        if raw not in pr_numbers:
            pr_numbers.append(raw)

    details: list[str] = []
    for entry in errors[:max_details]:
        text = str(entry.get("error") or "")
        if len(text) > detail_chars:
            text = text[: detail_chars - 3] + "..."
        details.append(text)

    return {
        "error_prs": pr_numbers[:max_prs],
        "error_prs_truncated": max(0, len(pr_numbers) - max_prs),
        "error_details": details,
        "error_details_truncated": max(0, len(errors) - max_details),
    }


# ---------------------------------------------------------------------------
# Issue #1132: foreign_issue_ref self-heal helpers.
#
# ``_mark_foreign_issue_ref`` is an existing OrchestratorApp method (modified
# in-place for confirmation counting); these are module-level free functions
# because OrchestratorApp is at its attachment-point ceiling and cannot take
# new members. They operate on state.json under ``state_lock`` so they are
# safe to call from the per-PR loop body.
# ---------------------------------------------------------------------------


def _should_reprobe_foreign_marker(
    marker: dict[str, Any], now: datetime, reprobe_hours: int
) -> bool:
    """True if a parked ``foreign_issue_ref`` marker is old enough to re-probe.

    The cadence is measured from ``last_reprobe_at`` if present (the most
    recent re-probe), otherwise from ``detected_at`` (the original park
    time). A marker with no parseable timestamp is re-probed immediately —
    it predates the #1132 schema and its age is unknown.
    """
    if reprobe_hours <= 0:
        return False
    threshold = timedelta(hours=reprobe_hours)
    anchor = _parse_iso_timestamp(marker.get("last_reprobe_at"))
    if anchor is None:
        anchor = _parse_iso_timestamp(marker.get("detected_at"))
    if anchor is None:
        return True
    return (now - anchor) >= threshold


def _clear_foreign_issue_ref_marker(state_file: Path, pr_number: int) -> bool:
    """Remove the ``foreign_issue_ref`` marker from a PR's state entry.

    Returns True if a marker was present and removed, False otherwise.
    Used by the self-heal re-probe path (issue #1132): when a parked marker's
    issue now resolves via ``issue_view``, the marker is cleared so the next
    pass resumes per-PR processing instead of skipping.
    """
    with state_lock(state_file):
        state = load_state(state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        if "foreign_issue_ref" not in pr_state:
            return False
        del pr_state["foreign_issue_ref"]
        state["prs"][str(pr_number)] = pr_state
        save_state(state_file, state)
    return True


def _touch_foreign_issue_ref_marker(state_file: Path, pr_number: int, issue_number: int) -> None:
    """Reset the re-probe clock on an existing ``foreign_issue_ref`` marker.

    Called after a re-probe still raised ``GitHubNotFoundError`` (the issue is
    genuinely absent): ``last_reprobe_at`` is stamped so the next re-probe is
    gated from this point, not from the original park time. Preserves
    ``detected_at`` and ``confirmations``.
    """
    with state_lock(state_file):
        state = load_state(state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        marker = pr_state.get("foreign_issue_ref") or {}
        if marker.get("issue") != issue_number:
            return
        ts = utc_now()
        pr_state["foreign_issue_ref"] = {
            **marker,
            "last_reprobe_at": ts,
        }
        state["prs"][str(pr_number)] = pr_state
        save_state(state_file, state)


# The two statuses that mark an issue as parked in the ``agent:human-needed``
# sink. Every escalation transition (``_escalate_issue``) sets one of these;
# the de-escalation sweep selects on the same pair (``_maybe_deescalate_mechanical``).
# Centralized so the sink census and the sweep cannot drift apart on what
# "in the sink" means. Issue #1083.
_SINK_STATUSES: frozenset[str] = frozenset({"escalated", "blocked"})

# Issue #1383: cross-pass infra_blocked escalation tracking. The
# OrchestratorApp instance is rebuilt per repo per pass (fleet_loop), so
# instance-level state cannot track "persistence across N passes." This
# module-level dict, keyed by repo_root string, survives across passes
# within the same supervisor process -- exactly the scope AC3 needs:
# "one operator escalation per window, not one per PR per pass." Each
# entry records the consecutive-pass count of infra_blocked observations,
# the datetime of the last ``infra_blocked_escalated`` emission (or None
# if none has been emitted yet), and the correlation_id of the pass that
# last incremented the counter (so multiple infra-blocked PRs encountered
# within the same ``review()`` pass increment the counter at most once).
# Reset to zero/None/None when a pass observes no infra_blocked PRs (the
# infra condition cleared).
_infra_blocked_window: dict[str, dict[str, Any]] = {}


def sink_census(state: dict[str, Any]) -> set[int]:
    """Return the set of issue numbers currently parked in the sink.

    The sink is the set of issues whose state entry carries a terminal
    ``status`` of ``"escalated"`` or ``"blocked"`` -- the in-state mirror of
    the escalation GitHub labels (``agent:human-needed`` for a judgment
    escalation, ``agent:operator-queue`` for a mechanical one since issue
    #1266; both share the same ``status`` value, so this census counts both
    without needing to know which). This is a point-in-time census read
    directly from ``state.json``'s ``issues`` map, deliberately not a
    GitHub-label query: it is cheap, deterministic, and matches the same
    source of truth the orchestrator's own de-escalation sweep selects
    candidates from, so the metric and the sweep agree on the population.

    Used by ``_loop_impl`` to compute the issue #1083 sink metric: a
    before/after diff around ``_loop_body`` yields arrivals
    (``after - before``), and the after-set size is the population.
    """
    issues = state.get("issues", {})
    if not isinstance(issues, dict):
        return set()
    parked: set[int] = set()
    for num, entry in issues.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in _SINK_STATUSES and str(num).isdigit():
            parked.add(int(num))
    return parked


def operator_queue_depth(state: dict[str, Any]) -> set[int]:
    """Return the set of issue numbers currently parked on the operator queue.

    Issue #1314 item 3. The operator queue is the subset of the sink
    (``sink_census``) whose entries carry ``reason_class == "mechanical"`` --
    the in-state mirror of the ``agent:operator-queue`` GitHub label (issue
    #1266). ``status == "blocked"`` is never mechanical (``blocked`` is a
    reviewer verdict, always ``reason_class == "judgment"``), so only
    ``status == "escalated"`` entries with the mechanical class are counted.

    This is a point-in-time census read directly from ``state.json``'s
    ``issues`` map, deliberately not a GitHub-label query: it is cheap,
    deterministic, and matches the same source of truth the de-escalation
    sweep selects candidates from, so the gauge and the sweep agree on the
    population.
    """
    issues = state.get("issues", {})
    if not isinstance(issues, dict):
        return set()
    queued: set[int] = set()
    for num, entry in issues.items():
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("status") == "escalated"
            and entry.get("reason_class") == "mechanical"
            and str(num).isdigit()
        ):
            queued.add(int(num))
    return queued


def _detect_and_handle_orphaned_workers(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh: GitHubLike,
    *,
    write_gate: WriteGate,
    review_callback: Callable[[int], Any] | None = None,
    fleet_dir_override: str | None = None,
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
    - If last decision was "approved" and the PR state carries
      ``status="rework_requested"`` (evidence the post-approval rework lane
      dispatched this worker) and head is unchanged since review, reset to
      "rework_requested" -- same as the request_changes branch (issue #1109)
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

    Issue #1122: this sweep is NOT gated on ``watchdog.enabled``. The watchdog
    flag controls log-mtime stall detection (``_detect_stalled_sessions`` /
    ``_detect_and_handle_stalled_sessions``), which is unrelated to this
    function's dead-pid state-keyed recovery. A deployment that disables
    watchdog (e.g. to work around a shim log-mtime blindness) must not lose
    the #935 pushed-branch salvage backstop, the #417 ground-truth label
    reclaim, or the orphan drift diagnostics -- all of which are keyed off
    state.json PID records, not log mtimes.
    """
    write_gate = require_write_gate(write_gate)

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
    # Issue #1229: validate branch-name-derived issue numbers against the
    # open-issue set so a stale branch name (e.g. agent/issue-709-… left over
    # from a merged PR #709, reused by an unrelated issue-less PR) cannot bind
    # the PR to a non-existent or closed issue here. Without this, a stale
    # binding would populate pr_by_issue[<wrong n>] and either mask a real
    # orphan's no-open-PR reclaim (the orphan is wrongly seen as "has a PR")
    # or route the #417 ground-truth label reclaim at the wrong issue subject.
    branch_validator = build_branch_issue_validator(gh)
    for pr in prs:
        linked = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
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
    # Issue #1153: issues escalated to ``agent:human-needed`` by the
    # zero-artifact dispatch loop guard, instead of being relabeled to
    # ``automated-ready`` for another fruitless redispatch. Keyed by issue
    # number; each value carries the label-write outcome and the attempt
    # count that triggered the escalation.
    zero_artifact_escalations: dict[int, dict[str, Any]] = {}
    # Issue #1244: issues escalated to ``agent:human-needed`` by the
    # cross-repo scope tripwire, instead of being relabeled to
    # ``automated-ready`` for another fruitless redispatch.  A dead worker
    # whose issue title names another managed repo hopped to that repo's
    # worktree; redispatching repeats the hop forever.  Keyed by issue
    # number; each value carries the label-write outcome and the scope
    # gate's reason.
    cross_repo_scope_escalations: dict[int, dict[str, Any]] = {}
    # Issue #1453: issues escalated to the operator queue because the worker
    # itself declared the task structurally impossible (a ``blocked`` outcome
    # in ``.worker-outcome.json``).  The worker deliberately concluded it
    # cannot do the task (e.g. cross-repo scope, missing dependency); without
    # this channel the orphan sweep would redispatch to another worker that
    # hits the identical wall, burning the full redispatch cap.  Keyed by
    # issue number; each value carries the label-write outcome and the
    # worker's ``reason_kind`` / ``detail`` so the operator queue entry is
    # actionable without reading the worktree.
    worker_declared_blocked_escalations: dict[int, dict[str, Any]] = {}
    # Issue #1453: pre-computed worker outcomes for all no-PR orphans, read
    # once before the first pre-lock loop so the blocked-outcome check can
    # fire before reclaim adds ``automated-ready``.  Reused by the second
    # loop (pushed-branch candidates) without re-reading.
    worker_outcomes: dict[int, dict[str, Any] | None] = {}
    issues_by_number: dict[int, dict[str, Any]] = {}
    # Issue #935 / #1453: compute repo_root and worktrees_dir once, before any
    # of the pre-lock loops, so the first loop (reclaim/escalation) can read
    # worker outcomes for the blocked-outcome check and the second loop
    # (pushed-branch candidates) can reuse the same pre-computed outcomes.
    repo_root = getattr(gh, "repo_root", None)
    worktrees_dir = None
    if repo_root is not None:
        worktrees_dir = resolved_layout(config, repo_root).worktrees

    if no_pr_orphans:
        for issue in gh.issue_list(state="open"):
            number = issue.get("number")
            if number is not None:
                issues_by_number[int(number)] = issue
        # Issue #1244: pre-compute the fleet's managed repo names and the
        # dispatching repo name once for the whole loop — the fleet registry
        # is read from disk and gh.name_with_owner() is a network call.
        sweep_repo_root = repo_root
        sweep_fleet_repos = managed_repo_names(fleet_dir_override)
        sweep_dispatching_repo_name = (
            _dispatching_repo_name(gh, sweep_repo_root) if sweep_repo_root is not None else ""
        )

        # Issue #1453: pre-read worker outcomes for all no-PR orphans so the
        # first loop (reclaim/escalation) can detect a ``blocked`` outcome
        # BEFORE reclaiming (which would add ``automated-ready`` and trigger a
        # fruitless redispatch). The durable terminal status is authoritative
        # because it is written after the worker exits and survives worktree
        # removal; the worktree file is the live fallback. Stored for the
        # second loop to reuse without re-reading.
        for issue_number in no_pr_orphans:
            issue = issues_by_number.get(issue_number)
            if issue is None:
                continue
            entry = state.get("issues", {}).get(str(issue_number), {})
            branch = entry.get("branch_name") if isinstance(entry, dict) else None
            if not branch:
                branch = (
                    f"{config.dispatch.branch_prefix}-{issue_number}-"
                    f"{slugify(str(issue.get('title') or 'work'))}"
                )
            worktree_path = None
            if repo_root is not None and worktrees_dir is not None:
                worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
            terminal = find_worker_terminal_status(sessions_dir, issue_number)
            terminal_outcome = terminal.get("worker_outcome") if terminal else None
            worktree_outcome = read_worker_outcome(worktree_path) if worktree_path else None
            worker_outcomes[issue_number] = terminal_outcome or worktree_outcome

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

            # Issue #1453: a worker that deliberately concluded it CANNOT do
            # the task writes a ``blocked`` outcome in
            # ``.worker-outcome.json`` instead of exiting PR-less with no
            # signal.  Without this channel the orphan sweep classifies the
            # dead worker as ``dead_worker_no_open_pr`` and redispatches --
            # to another worker that hits the identical wall, burning the
            # full redispatch cap.  The worker's own declaration is the most
            # authoritative signal: check it FIRST, before the zero-artifact
            # and cross-repo heuristic escalations, and route directly to the
            # operator queue with zero redispatches.  The pre-lock label
            # relabeling (remove active, add human-needed as fallback) mirrors
            # the zero-artifact / cross-repo scope pattern; the post-lock
            # ``reap_escalations`` transition applies the operator-queue
            # label edge.
            worker_outcome = worker_outcomes.get(issue_number)
            if isinstance(worker_outcome, dict) and worker_outcome.get("outcome") == "blocked":
                label_write_ok = True
                for label in sorted(active_labels):
                    if not gh.remove_issue_label(issue_number, label):
                        label_write_ok = False
                if config.labels.human_needed not in issue_labels:
                    if not gh.add_issue_label(issue_number, config.labels.human_needed):
                        label_write_ok = False
                worker_declared_blocked_escalations[issue_number] = {
                    "removed_labels": sorted(active_labels),
                    "label_write_ok": label_write_ok,
                    "reason_kind": str(worker_outcome.get("reason_kind") or "unknown"),
                    "detail": str(worker_outcome.get("detail") or ""),
                }
                continue

            # Issue #1153: before relabeling to ``automated-ready`` for
            # another redispatch, check whether prior attempts all produced
            # zero artifacts (``ahead_of_main == 0``). A repeated
            # zero-artifact dispatch loop means each worker session ran,
            # determined the fix belongs in a sibling repo, hopped to its
            # worktree, did the work there, and exited with ``ahead_of_main:
            # 0`` in *this* repo's tree -- swept as a dead worker with no
            # open PR, relabeled, redispatched, forever. Escalate to
            # ``agent:human-needed`` instead of burning another dispatch.
            if _is_zero_artifact_dispatch_loop(sessions_dir, issue_number):
                label_write_ok = True
                for label in sorted(active_labels):
                    if not gh.remove_issue_label(issue_number, label):
                        label_write_ok = False
                if config.labels.human_needed not in issue_labels:
                    if not gh.add_issue_label(issue_number, config.labels.human_needed):
                        label_write_ok = False
                zero_artifact_escalations[issue_number] = {
                    "removed_labels": sorted(active_labels),
                    "label_write_ok": label_write_ok,
                }
                continue

            # Issue #1244: cross-repo scope tripwire. Before relabeling to
            # ``automated-ready`` for another redispatch, check whether the
            # issue's title names another managed repo in the fleet. A dead
            # worker whose issue scope targets a sibling repo hopped to that
            # repo's worktree; redispatching repeats the hop forever. Escalate
            # to ``agent:human-needed`` instead of burning another dispatch.
            # This is the transition tripwire (Option 2): catches issues
            # dispatched before the intake-time scope gate (Option 1) existed.
            scope_result = cross_repo_scope_gate(
                str(issue.get("title") or ""),
                str(issue.get("body") or ""),
                sweep_dispatching_repo_name,
                sweep_fleet_repos,
            )
            if not scope_result.passed:
                label_write_ok = True
                for label in sorted(active_labels):
                    if not gh.remove_issue_label(issue_number, label):
                        label_write_ok = False
                if config.labels.human_needed not in issue_labels:
                    if not gh.add_issue_label(issue_number, config.labels.human_needed):
                        label_write_ok = False
                cross_repo_scope_escalations[issue_number] = {
                    "removed_labels": sorted(active_labels),
                    "label_write_ok": label_write_ok,
                    "reason": scope_result.reason,
                }
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

    # Issue #935: for the no-open-PR orphans, determine whether the worker
    # pushed a branch and reported push-succeeded-but-PR-failed. This is done
    # outside the state lock because it can touch origin. The second lock below
    # will use these pre-computed candidates to decide whether to open the PR
    # itself instead of re-dispatching.
    # ``repo_root`` / ``worktrees_dir`` were computed above, before the first
    # pre-lock loop, so the blocked-outcome check could read worker outcomes.

    # Issue #1248: salvage-push committed-but-unpushed work from dead workers'
    # worktrees BEFORE anything below classifies them. A worker that finished
    # its work locally but died at the final push produces zero remote delta,
    # so every downstream branch -- request_changes auto-reset, the no-op
    # rework detector, the #935 pushed-branch lane -- reads it as "did
    # nothing" and redispatches (or caps out) on work that is already done.
    # ``salvage_push_stranded_commits`` publishes the work only when it is a
    # pure fast-forward of the remote tip (never force; diverged worktrees
    # fall through to the existing ``dead_worker_unsafe_to_auto_reset``
    # handling untouched). After a successful push the PR snapshot's
    # ``headRefOid`` is refreshed so the in-lock classification sees the head
    # advance and routes to review instead of counting a death, and the #935
    # candidate detection below sees the branch on origin and opens a PR for
    # it. Runs pre-lock: it is network I/O.
    salvage_pushes: dict[int, dict[str, Any]] = {}
    if repo_root is not None and worktrees_dir is not None:
        for issue_number in orphaned_issues:
            pr_data = pr_by_issue.get(issue_number)
            if pr_data is not None and pr_data.get("isCrossRepository"):
                # The head branch lives in a fork; this checkout cannot (and
                # must not) push there.
                continue
            entry = state.get("issues", {}).get(str(issue_number), {})
            branch = pr_data.get("headRefName") if pr_data is not None else None
            if not branch and isinstance(entry, dict):
                branch = entry.get("branch_name")
            if not branch:
                # No recorded branch: never guess a ref name to push to.
                continue
            worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
            result = salvage_push_stranded_commits(
                repo_root,
                branch,
                worktree_path,
                base_ref=config.dispatch.base_ref,
                dry_run=write_gate.dry_run,
            )
            if result.pushed:
                salvage_pushes[issue_number] = {
                    "issue_number": issue_number,
                    "pr_number": int(pr_data["number"]) if pr_data is not None else None,
                    "branch": branch,
                    "old_remote_sha": result.old_remote_sha,
                    "new_remote_sha": result.new_remote_sha,
                    "commit_count": result.commit_count,
                }
                if pr_data is not None and result.new_remote_sha:
                    # Refresh the snapshot so classification below compares
                    # against the salvaged head, not the pre-push one.
                    pr_data["headRefOid"] = result.new_remote_sha
            elif result.error:
                salvage_pushes[issue_number] = {
                    "issue_number": issue_number,
                    "pr_number": int(pr_data["number"]) if pr_data is not None else None,
                    "branch": branch,
                    "old_remote_sha": result.old_remote_sha,
                    "commit_count": result.commit_count,
                    "error": result.error,
                }
            # skip_reason outcomes are silent by design: "up_to_date" /
            # "no_worktree" / "no_stranded_commits" describe the overwhelming
            # majority of dead workers and would flood events.db every pass.

    no_pr_issue_details: dict[int, dict[str, Any]] = {}
    pushed_branch_candidates: dict[int, dict[str, Any]] = {}
    state_snapshot = state
    for issue_number in no_pr_orphans:
        issue = issues_by_number.get(issue_number)
        if issue is None:
            # Issue not open/visible: we cannot safely mutate labels, and a
            # closed issue should not get a new PR.
            continue
        issue_labels = label_names(issue)
        active_labels = issue_labels & config.labels.active
        no_pr_issue_details[issue_number] = {
            "issue": issue,
            "issue_labels": issue_labels,
            "active_labels": active_labels,
        }

        entry = state_snapshot.get("issues", {}).get(str(issue_number), {})
        branch = entry.get("branch_name")
        if not branch:
            branch = (
                f"{config.dispatch.branch_prefix}-{issue_number}-"
                f"{slugify(str(issue.get('title') or 'work'))}"
            )

        worktree_path = None
        if repo_root is not None and worktrees_dir is not None:
            worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)

        # Issue #1453: reuse the outcome pre-computed above (before the first
        # pre-lock loop) instead of re-reading the terminal status and
        # worktree file.  The blocked-outcome check already ran in the first
        # loop; here we only need the outcome for the push/PR-failure signal.
        worker_outcome = worker_outcomes.get(issue_number)

        reported_push = (
            isinstance(worker_outcome, dict)
            and worker_outcome.get("push_succeeded") is True
            and worker_outcome.get("pr_created") is False
        )

        ahead_count = None
        ahead_error = None
        if repo_root is not None:
            ahead_count, ahead_error = remote_branch_ahead_count(
                repo_root, branch, config.dispatch.base_ref
            )

        # Issue #1243: compute the branch head SHA (remote + local worktree)
        # for the no-open-PR redispatch cap. An unchanged head across attempts
        # is the "no progress" signal that increments toward the cap; a moving
        # head (remote push or local stranded commits) resets it. Done here,
        # outside the state lock, because both probes touch git/network.
        remote_head_sha = None
        local_head_sha = None
        if repo_root is not None:
            remote_head_sha = remote_branch_head_sha(repo_root, branch)
        if worktree_path is not None:
            local_head_sha = worktree_head_sha(worktree_path)
        no_pr_issue_details.setdefault(issue_number, {}).update(
            {
                "branch": branch,
                "remote_head_sha": remote_head_sha,
                "local_head_sha": local_head_sha,
            }
        )

        # Treat a branch as a PR-open candidate when:
        # - the worker itself reported a successful push with a failed PR, OR
        # - the branch exists on origin and is ahead of the base (has commits).
        if reported_push or (ahead_count is not None and ahead_count > 0):
            pushed_branch_candidates[issue_number] = {
                "branch": branch,
                "worktree_path": worktree_path,
                "worker_outcome": worker_outcome,
                "reported_push": reported_push,
                "ahead_count": ahead_count,
                "ahead_error": ahead_error,
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
        # Issue #1362 Stage 1: read through the single review-decision
        # reader (flat file, falling back to the highest archived round)
        # rather than state.json's ``decision``/``reviewed_head_sha``
        # fields, which can lag a concurrent record_review/void.
        resolved_decision = review_decision(
            state_file.parent / "prs" / f"pr-{pr_number}", None, pr_data.get("headRefOid")
        )
        if resolved_decision.decision == "request_changes" and not resolved_decision.stale:
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
                write_gate=write_gate,
            )
            if route_result is not None:
                pre_review_routed.add(issue_number)

    # Issue #1128: for dead workers that already have an OPEN PR with no
    # review verdict yet (``last_decision`` is null/absent), pre-compute the
    # issue's live GitHub labels outside the state lock. The second-lock
    # sweep uses these to transition the issue from ``agent:in-progress`` to
    # ``agent:pr-open`` -- the same LabelConfig-driven swap the
    # ``orphaned_worker_opened_pr`` lane uses -- so review dispatch can claim
    # the salvage PR. Without this, the ``dead_worker_unsafe_to_auto_reset``
    # branch advanced no label and the issue sat on ``agent:in-progress``
    # indefinitely, asserting a live worker the reconciler had just confirmed
    # dead. ``pr_by_issue`` only contains OPEN PRs (``gh pr list --state
    # open``), so presence there is the "PR is OPEN" precondition.
    pr_orphan_unreviewed_details: dict[int, dict[str, Any]] = {}

    def _has_no_review_verdict_yet(issue_number: int) -> bool:
        """True when the PR carries no terminal review verdict yet.

        Issue #1362 Stage 1 fix: the old state-based predicate
        (``not pr_state.get("decision")``) was true for both "no decision
        file at all" and "a pending placeholder decision" -- both count as
        "no verdict yet" per this lane's #1128 intent (see the comment
        above). Checking only ``.missing`` narrowed that: a dead-worker
        orphan PR that reached a pending packet would be silently excluded
        from the ``agent:in-progress`` -> ``agent:pr-open`` advance,
        re-stranding the issue exactly like the original #1128 bug.
        """
        resolved = review_decision(
            state_file.parent / "prs" / f"pr-{int(pr_by_issue[issue_number]['number'])}",
            None,
            pr_by_issue[issue_number].get("headRefOid"),
        )
        return resolved.missing or resolved.decision == "pending"

    pr_orphans_unreviewed = [
        n for n in orphaned_issues if n in pr_by_issue and _has_no_review_verdict_yet(n)
    ]
    if pr_orphans_unreviewed:
        # ``issues_by_number`` is only populated above when there were
        # no-open-PR orphans; build it here when this lane is the sole
        # consumer so the label read stays a single bulk ``issue_list`` call.
        if not issues_by_number:
            for issue in gh.issue_list(state="open"):
                number = issue.get("number")
                if number is not None:
                    issues_by_number[int(number)] = issue
        for issue_number in pr_orphans_unreviewed:
            issue = issues_by_number.get(issue_number)
            if issue is None:
                # Issue not open/visible -- cannot safely mutate labels;
                # the conservative drift path below handles it.
                continue
            issue_labels = label_names(issue)
            active_labels = issue_labels & config.labels.active
            pr_orphan_unreviewed_details[issue_number] = {
                "issue_labels": issue_labels,
                "active_labels": active_labels,
            }

    # Handle orphaned workers. Head-advanced request_changes findings are
    # collected and routed to the review lane outside the state lock (review()
    # itself acquires the lock and may call transition()).
    review_routes: list[tuple[int, int, str, str, str]] = []
    # Issue #654: dead dispatched workers time-escalated inside the lock
    # collected here for the post-lock transition() call (network I/O).
    reap_escalations: list[int] = []
    with state_lock(state_file):
        state = load_state(state_file)
        sweep_events: list[tuple[str, dict[str, Any]]] = []
        # Issue #1248: record the pre-lock salvage pushes (and push failures)
        # in the same event stream as the classification they feed, so a
        # ``dead_worker_with_head_change`` routed below is attributable to its
        # salvage rather than looking like a spontaneous worker push.
        for salvage_payload in salvage_pushes.values():
            if salvage_payload.get("error"):
                sweep_events.append(("salvage_push_failed", salvage_payload))
            else:
                sweep_events.append(("salvage_pushed_stranded_commits", salvage_payload))
        for issue_number in orphaned_issues:
            entry = state["issues"].get(str(issue_number), {})
            if not isinstance(entry, dict):
                continue

            # Re-verify status (state may have changed between lock windows)
            if entry.get("status") != "dispatched":
                continue

            # Issue #654: time-based escape for a dead dispatched worker whose
            # drift was already surfaced on a prior pass (``orphan_drift_at`` is
            # set) but whose PR state did not qualify for auto-reset -- a clean
            # exit with no push (issue #773's ``dead_worker_clean_exit_no_op``
            # branch), a non-request_changes decision, a head change without a
            # review callback, or a PR-create failure on a pushed branch. In all
            # of these the specific sub-branch below emits drift once, sets
            # ``orphan_drift_at``, then on every subsequent pass the fingerprint
            # match short-circuits to ``continue`` -- so the dispatch label
            # (``agent:in-progress``) holds indefinitely. The label is the
            # one-writer-per-branch mutex, so no re-dispatch can proceed on that
            # branch until a worker that no longer exists reports back. After
            # ``dead_dispatched_reap_minutes`` since the drift was first
            # surfaced, escalate to ``agent:human-needed`` so a human can inspect
            # the worktree for unpushed commits and decide whether to salvage or
            # re-dispatch. This runs BEFORE the specific sub-branches so it is a
            # pure backstop: on the first pass ``orphan_drift_at`` is not yet set
            # and the specific sub-branch runs normally (either resetting
            # immediately or emitting the first drift). Only issues that already
            # have drift recorded and have exceeded the grace period are
            # escalated here. 0 disables the escape (pre-#654 hold-forever).
            orphan_drift_at = entry.get("orphan_drift_at")
            if orphan_drift_at is not None and config.watchdog.dead_dispatched_reap_minutes > 0:
                drift_dt = _parse_iso_timestamp(orphan_drift_at)
                if (
                    drift_dt is not None
                    and (now - drift_dt).total_seconds() / 60
                    >= config.watchdog.dead_dispatched_reap_minutes
                ):
                    pr_data_for_reap = pr_by_issue.get(issue_number)
                    pr_number_for_reap = (
                        int(pr_data_for_reap["number"]) if pr_data_for_reap else None
                    )
                    terminal = find_worker_terminal_status(sessions_dir, issue_number)
                    terminal_exit_code = terminal.get("exit_code") if terminal else None
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="dead_dispatched_worker_reap",
                        reason_class="mechanical",
                        pr_number=pr_number_for_reap,
                        issue_extra={
                            "dispatched_at": None,
                            "orphan_drift_fingerprint": None,
                            "orphan_drift_at": None,
                        },
                    )
                    sweep_events.append(
                        (
                            "dead_dispatched_worker_reaped",
                            {
                                "issue_number": issue_number,
                                "pr_number": pr_number_for_reap,
                                "previous_status": "dispatched",
                                "reason": "dead_dispatched_worker_reap",
                                "orphan_drift_at": orphan_drift_at,
                                "reap_minutes": (config.watchdog.dead_dispatched_reap_minutes),
                                "exit_code": terminal_exit_code,
                            },
                        )
                    )
                    reap_escalations.append(issue_number)
                    continue

            # Issue #282: do not clear the liveness fingerprint here. The worker
            # is dead (``_worker_pid_alive`` returned False), but the PID record
            # may still be needed by the recovery path to decide whether the
            # worktree is safe to reset.

            pr_data = pr_by_issue.get(issue_number)

            if pr_data:
                pr_number = int(pr_data["number"])
                pr_state = state.get("prs", {}).get(str(pr_number), {})
                live_head_sha = pr_data.get("headRefOid")
                # Issue #1362 Stage 1: read the last review decision through
                # the single file-first reader (flat file, falling back to
                # the highest archived round) instead of state.json's
                # decision/reviewed_head_sha fields, which can lag a
                # concurrent record_review/void -- the #1340 divergence
                # class AC1 exists to eliminate.
                resolved_decision = review_decision(
                    state_file.parent / "prs" / f"pr-{pr_number}", None, live_head_sha
                )
                last_decision = resolved_decision.decision
                reviewed_head_sha = resolved_decision.reviewed_head_sha

                # Issue #773: measurement-first payload enrichment. A dead PID
                # alone cannot distinguish a worker that crashed from one that
                # exited 0 having pushed nothing -- both present identically
                # to `_worker_pid_alive`. `find_worker_terminal_status` reads
                # the durable record `start_terminal_status_watcher`
                # (process_utils.py) writes at the moment a claude-code worker
                # actually exits; it returns None for legacy sessions, sessions
                # from adapters that don't write one (e.g. devin-shell), or
                # any session whose watcher never got to run (e.g. orchestrator
                # restart mid-session). `terminal_exit_code` is deliberately
                # left as None in all of those cases rather than guessed at --
                # every event below records it as-is so the two populations
                # (confirmed clean exit vs. everything else) are queryable
                # retrospectively even before they're fully separable.
                terminal = find_worker_terminal_status(sessions_dir, issue_number)
                terminal_pid = entry.get("worker_pid")
                terminal_exit_code = terminal.get("exit_code") if terminal else None
                terminal_duration_seconds = terminal.get("duration_seconds") if terminal else None

                if last_decision == "request_changes" and reviewed_head_sha and live_head_sha:
                    if reviewed_head_sha == live_head_sha:
                        if terminal_exit_code == 0:
                            # The worker exited cleanly (exit code 0) rather
                            # than crashing -- e.g. it was handed an empty
                            # rework brief with nothing left to act on. Do NOT
                            # auto-reset to rework_requested: that would spend
                            # one of max_auto_redispatch's attempts on a
                            # worker that never had anything to change,
                            # eventually escalating a benign no-op to
                            # agent:human-needed (issue #773). Surface it once
                            # instead, via the same fingerprinted
                            # surface-once convergence the other drift
                            # branches below already use, so a human/janitor
                            # can decide whether the review itself needs
                            # revisiting -- retrying a dispatch that already
                            # proved it produces no change on this exact head
                            # would just repeat the no-op.
                            fingerprint = _drift_fingerprint(
                                reason="dead_worker_clean_exit_no_op",
                                reviewed_head_sha=reviewed_head_sha,
                            )
                            if entry.get("orphan_drift_fingerprint") == fingerprint:
                                state["issues"][str(issue_number)] = entry
                                continue
                            entry["orphan_drift_fingerprint"] = fingerprint
                            entry["orphan_drift_at"] = utc_now()
                            sweep_events.append(
                                (
                                    "orphaned_worker_drift",
                                    {
                                        "issue_number": issue_number,
                                        "pr_number": pr_number,
                                        "previous_status": "dispatched",
                                        "reason": "dead_worker_clean_exit_no_op",
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                    },
                                )
                            )
                        else:
                            # No terminal record, or a non-zero/None exit code:
                            # unchanged from pre-#773 behavior -- safe to reset
                            # to rework_requested (PR head unchanged since
                            # request_changes).
                            entry["status"] = "rework_requested"
                            entry["dispatched_at"] = None
                            # Issue #1134: record this as a worker death, not
                            # a no-op.  A death redispatch must not count
                            # against the no-op rework cap — the worker may
                            # have completed its work but died before pushing
                            # (salvageable stranded commits).  A separate
                            # death counter with its own escalation reason
                            # (worker_death_loop) lets the operator triage
                            # "check the worktree" vs. "worker is spinning."
                            death_ts = utc_now()
                            prior_deaths = entry.get("worker_death_at")
                            if not isinstance(prior_deaths, list):
                                prior_deaths = []
                            entry["worker_death_at"] = prior_deaths + [death_ts]
                            sweep_events.append(
                                (
                                    "orphaned_worker_recovered",
                                    {
                                        "issue_number": issue_number,
                                        "pr_number": pr_number,
                                        "previous_status": "dispatched",
                                        "new_status": "rework_requested",
                                        "reason": "dead_worker_with_request_changes",
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                        "worker_death_at": death_ts,
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
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                    },
                                )
                            )
                else:
                    # Not a simple request_changes case.
                    # Issue #1109: a dead worker on an approved PR is not
                    # unclassifiable when the post-approval rework lane
                    # (#674 -> PR #685, plus the merge-conflict and no-op
                    # rework lanes that share ``_route_to_rework``) dispatched
                    # it. Those lanes set the PR state status to
                    # ``rework_requested`` while preserving
                    # ``decision="approved"``, and the worker is dispatched to
                    # fix CI/a conflict without re-litigating the review. If
                    # that worker dies before pushing (head unchanged since
                    # review), the issue previously wedged in ``dispatched``
                    # forever because this sweep refused to auto-reset on a
                    # non-``request_changes`` decision -- no redispatch, no
                    # cap consumption, invisible to every downstream lane.
                    # Treat ``decision="approved"`` + PR-state
                    # ``rework_requested`` + head unchanged as safe to
                    # auto-reset, mirroring the request_changes branch above
                    # (including the #773 clean-exit-no-op sub-case so a
                    # benign exit-0 worker does not burn redispatch attempts).
                    # ``dead_worker_unsafe_to_auto_reset`` is kept only for
                    # genuinely unclassifiable decisions -- an approved PR
                    # whose PR state does not carry ``rework_requested`` has
                    # no evidence a post-approval rework lane dispatched this
                    # worker, so auto-resetting would be a guess.
                    #
                    # Issue #1128: when the dead worker has an OPEN PR with no
                    # review verdict yet (``last_decision`` is null/absent),
                    # the "unsafe to auto-reset" judgment stays -- the PR
                    # carries the work, so re-dispatching would duplicate it --
                    # but leaving the issue on ``agent:in-progress`` makes the
                    # state machine assert a live worker the reconciler just
                    # confirmed dead, and review dispatch (which keys off
                    # ``agent:pr-open``) never sees the salvage PR. Transition
                    # to ``pr-open`` via the same LabelConfig-driven swap the
                    # ``orphaned_worker_opened_pr`` lane uses. On label write
                    # failure, fall through to the conservative drift path so
                    # the next pass re-attempts rather than resetting the
                    # worker.
                    pr_state_status = pr_state.get("status")
                    if (
                        last_decision == "approved"
                        and pr_state_status == "rework_requested"
                        and reviewed_head_sha
                        and live_head_sha
                        and reviewed_head_sha == live_head_sha
                    ):
                        if terminal_exit_code == 0:
                            # Clean exit with no push -- same #773 rationale
                            # as the request_changes branch: do not spend a
                            # redispatch attempt on a worker that produced no
                            # change on this exact head.
                            fingerprint = _drift_fingerprint(
                                reason="dead_worker_clean_exit_no_op",
                                reviewed_head_sha=reviewed_head_sha,
                            )
                            if entry.get("orphan_drift_fingerprint") == fingerprint:
                                state["issues"][str(issue_number)] = entry
                                continue
                            entry["orphan_drift_fingerprint"] = fingerprint
                            entry["orphan_drift_at"] = utc_now()
                            sweep_events.append(
                                (
                                    "orphaned_worker_drift",
                                    {
                                        "issue_number": issue_number,
                                        "pr_number": pr_number,
                                        "previous_status": "dispatched",
                                        "reason": "dead_worker_clean_exit_no_op",
                                        "decision": "approved",
                                        "pr_state_status": pr_state_status,
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                    },
                                )
                            )
                        else:
                            # No terminal record, or a non-zero/None exit
                            # code: safe to reset to rework_requested (PR head
                            # unchanged since the approved review, and the
                            # post-approval rework lane dispatched this
                            # worker). Records this as a worker death with a
                            # distinct reason so the death counter (issue
                            # #1134) and the redispatch cap (issue #165) apply
                            # exactly as they do for request_changes.
                            entry["status"] = "rework_requested"
                            entry["dispatched_at"] = None
                            death_ts = utc_now()
                            prior_deaths = entry.get("worker_death_at")
                            if not isinstance(prior_deaths, list):
                                prior_deaths = []
                            entry["worker_death_at"] = prior_deaths + [death_ts]
                            sweep_events.append(
                                (
                                    "orphaned_worker_recovered",
                                    {
                                        "issue_number": issue_number,
                                        "pr_number": pr_number,
                                        "previous_status": "dispatched",
                                        "new_status": "rework_requested",
                                        "reason": "dead_worker_with_approved_rework",
                                        "decision": "approved",
                                        "pr_state_status": pr_state_status,
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                        "worker_death_at": death_ts,
                                    },
                                )
                            )
                    else:
                        # Not the #1109 approved+rework_requested classified
                        # case. This branch covers two populations that share
                        # one fingerprinted drift fallback below:
                        #   (a) #1128: ``last_decision`` is None or "pending"
                        #       (open PR, no terminal review verdict yet) --
                        #       try advancing to ``pr-open``; on label-write
                        #       failure or missing details, fall through to
                        #       the shared drift. Issue #1362 Stage 1: this
                        #       must match ``_has_no_review_verdict_yet``'s
                        #       predicate above (``.missing or .decision ==
                        #       "pending"``) or a pending-packet PR would be
                        #       precomputed into ``pr_orphan_unreviewed_details``
                        #       but never consulted here, re-stranding the
                        #       issue on ``agent:in-progress``.
                        #   (b) genuinely unclassifiable decisions -- fall
                        #       through to the shared drift directly.
                        if last_decision is None or last_decision == "pending":
                            details = pr_orphan_unreviewed_details.get(issue_number)
                            if details is not None:
                                active_labels = details["active_labels"]
                                issue_labels = details["issue_labels"]
                                label_write_ok = True
                                for label in sorted(active_labels):
                                    if not gh.remove_issue_label(issue_number, label):
                                        label_write_ok = False
                                if config.labels.pr_open not in issue_labels:
                                    if not gh.add_issue_label(issue_number, config.labels.pr_open):
                                        label_write_ok = False
                                if label_write_ok:
                                    entry["status"] = PASSIVE_OPEN_STATUS
                                    entry["dispatched_at"] = None
                                    # Clear any prior drift fingerprint so a
                                    # later regression on this issue re-surfaces.
                                    entry["orphan_drift_fingerprint"] = None
                                    entry["orphan_drift_at"] = None
                                    sweep_events.append(
                                        (
                                            "orphaned_worker_advanced_to_pr_open",
                                            {
                                                "issue_number": issue_number,
                                                "pr_number": pr_number,
                                                "previous_status": "dispatched",
                                                "new_status": PASSIVE_OPEN_STATUS,
                                                "reason": "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr",
                                                "removed_labels": sorted(active_labels),
                                                "pid": terminal_pid,
                                                "exit_code": terminal_exit_code,
                                                "duration_seconds": terminal_duration_seconds,
                                                "label_write_ok": True,
                                            },
                                        )
                                    )
                                    state["issues"][str(issue_number)] = entry
                                    continue
                                # Label write failed -- fall through to the
                                # fingerprinted drift path so the next pass
                                # re-attempts the transition (the drift
                                # fingerprint gates only re-emission of the
                                # diagnostic, not the transition retry above,
                                # which runs first on every pass).
                        # Genuinely unclassifiable decision, or #1128 label-
                        # write failure -- surface as drift once. One shared
                        # fingerprinted fallback for both lanes (#1109 keeps
                        # its own classified branch above; this covers
                        # everything else).
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
                                        "pid": terminal_pid,
                                        "exit_code": terminal_exit_code,
                                        "duration_seconds": terminal_duration_seconds,
                                    },
                                )
                            )
            else:
                # Issue #935: before reclaim/drift, try to open a PR for a branch
                # that the worker pushed but could not create a PR for.
                candidate = pushed_branch_candidates.get(issue_number)
                if candidate is not None:
                    details = no_pr_issue_details.get(issue_number, {})
                    # ``getattr(gh, "repo_root", None)`` is not statically typed,
                    # so it could in principle be any non-``Path`` value. Narrow
                    # it to ``Path | None`` before passing it to the salvage
                    # helper; ``None`` is a value the helper already handles by
                    # returning an error, which preserves the existing drift/hold
                    # behavior for no-repo-root orphans.
                    salvage_repo_root = repo_root if isinstance(repo_root, Path) else None
                    pr_number, pr_error, _closing_ref = _open_pr_for_orphaned_branch(
                        gh=gh,
                        config=config,
                        repo_root=salvage_repo_root,
                        branch=candidate["branch"],
                        base_ref=config.dispatch.base_ref,
                        issue_number=issue_number,
                        active_labels=details.get("active_labels", set()),
                        issue_labels=details.get("issue_labels", set()),
                        issue_title=(details.get("issue") or {}).get("title"),
                        state_file=state_file,
                    )
                    if pr_number is not None:
                        entry["status"] = PASSIVE_OPEN_STATUS
                        entry["pr_number"] = pr_number
                        sweep_events.append(
                            (
                                "orphaned_worker_opened_pr",
                                {
                                    "issue_number": issue_number,
                                    "pr_number": pr_number,
                                    "branch_name": candidate["branch"],
                                    "worker_reported": candidate["reported_push"],
                                    "ahead_count": candidate["ahead_count"],
                                    "previous_status": "dispatched",
                                    "reason": "dead_worker_branch_pushed_no_pr",
                                    "label_write_ok": pr_error is None,
                                    "pr_error": pr_error,
                                },
                            )
                        )
                        state["issues"][str(issue_number)] = entry
                        continue

                    # PR creation failed after the bounded outer retry
                    # (pr_create_retry.py) and the duplicate-PR guard both
                    # exhausted -- the branch is pushed and stranded (cw#1273).
                    # Reuses this sweep's existing _drift_fingerprint dedup
                    # path rather than inventing a parallel one; the "reason"
                    # value is preserved unchanged from before cw#1273 so the
                    # fingerprint (and any existing dedup state already on
                    # disk from before this change) stays stable.
                    fingerprint = _drift_fingerprint(
                        reason="dead_worker_branch_pushed_pr_create_failed",
                        branch_name=candidate["branch"],
                        error=pr_error or "unknown",
                    )
                    if entry.get("orphan_drift_fingerprint") == fingerprint:
                        state["issues"][str(issue_number)] = entry
                        continue
                    entry["orphan_drift_fingerprint"] = fingerprint
                    entry["orphan_drift_at"] = utc_now()
                    sweep_events.append(
                        (
                            "pr_create_failed_branch_stranded",
                            {
                                "issue_number": issue_number,
                                "branch_name": candidate["branch"],
                                "previous_status": "dispatched",
                                "reason": "dead_worker_branch_pushed_pr_create_failed",
                                "pr_create_error": pr_error,
                                "worker_reported": candidate["reported_push"],
                                "ahead_count": candidate["ahead_count"],
                            },
                        )
                    )
                    state["issues"][str(issue_number)] = entry
                    continue

                # Issue #1453: check whether the worker itself declared the
                # task structurally impossible (a ``blocked`` outcome in
                # ``.worker-outcome.json``, computed above, outside the state
                # lock).  The worker's own declaration is the most
                # authoritative signal -- check it FIRST, before the
                # zero-artifact and cross-repo heuristic escalations, so a
                # deliberate blocked analysis routes directly to the operator
                # queue with zero redispatches instead of burning the full
                # redispatch cap re-dispatching structurally impossible tasks.
                # The event payload carries ``reason_kind`` and ``detail`` so
                # the operator queue entry is actionable without reading the
                # worktree.
                blocked_escalation = worker_declared_blocked_escalations.get(issue_number)
                if blocked_escalation is not None:
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="worker_declared_blocked",
                        reason_class="mechanical",
                    )
                    escalated_entry = state["issues"][str(issue_number)]
                    escalated_entry["orphan_flagged_at"] = utc_now()
                    state["issues"][str(issue_number)] = escalated_entry
                    sweep_events.append(
                        (
                            "worker_declared_blocked",
                            {
                                "issue_number": issue_number,
                                "previous_status": "dispatched",
                                "reason": "worker_declared_blocked",
                                "reason_kind": blocked_escalation["reason_kind"],
                                "detail": blocked_escalation["detail"],
                                "removed_labels": blocked_escalation["removed_labels"],
                                "label_write_ok": blocked_escalation["label_write_ok"],
                            },
                        )
                    )
                    reap_escalations.append(issue_number)
                    continue

                # Issue #1153: check whether this issue was escalated to
                # ``agent:human-needed`` by the zero-artifact dispatch loop
                # guard (computed above, outside the state lock). If so,
                # record the escalation in state and emit a visible event --
                # do NOT fall through to the reclaim path (which would
                # re-add ``automated-ready`` and trigger another fruitless
                # redispatch).
                escalation = zero_artifact_escalations.get(issue_number)
                if escalation is not None:
                    # Route through ``_escalate_issue`` so the paired
                    # ``escalation_reason`` / ``reason_class`` /
                    # ``terminal_since`` fields are written atomically and
                    # the #750 structural guard (status="escalated" only
                    # inside the helper) continues to hold.
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="zero_artifact_dispatch_loop",
                        reason_class="mechanical",
                    )
                    escalated_entry = state["issues"][str(issue_number)]
                    escalated_entry["orphan_flagged_at"] = utc_now()
                    state["issues"][str(issue_number)] = escalated_entry
                    sweep_events.append(
                        (
                            "session_failed_escalated",
                            _session_failed_relabeled_payload(
                                issue_number=issue_number,
                                reason="zero_artifact_dispatch_loop",
                                removed_labels=escalation["removed_labels"],
                                added_ready=False,
                                label_write_ok=escalation["label_write_ok"],
                            ),
                        )
                    )
                    continue

                # Issue #1244: check whether this issue was escalated to
                # ``agent:human-needed`` by the cross-repo scope tripwire
                # (computed above, outside the state lock). If so, record the
                # escalation in state and emit a visible event — do NOT fall
                # through to the reclaim path (which would re-add
                # ``automated-ready`` and trigger another fruitless
                # redispatch that hops to the sibling repo again).
                scope_escalation = cross_repo_scope_escalations.get(issue_number)
                if scope_escalation is not None:
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="cross_repo_hop",
                        reason_class="mechanical",
                    )
                    escalated_entry = state["issues"][str(issue_number)]
                    escalated_entry["orphan_flagged_at"] = utc_now()
                    state["issues"][str(issue_number)] = escalated_entry
                    sweep_events.append(
                        (
                            "session_failed_escalated",
                            _session_failed_relabeled_payload(
                                issue_number=issue_number,
                                reason="cross_repo_hop",
                                removed_labels=scope_escalation["removed_labels"],
                                added_ready=False,
                                label_write_ok=scope_escalation["label_write_ok"],
                            ),
                        )
                    )
                    continue

                # Issue #1243: per-issue redispatch cap with stall detection.
                # The no-open-PR orphan-sweep redispatch path is the only
                # redispatch loop without a bound -- without this cap, a
                # persistent post-exit condition that leaves no open PR
                # reproduces the #709 infinite loop (worker exits -> sweep
                # strips agent:in-progress -> issue returns to the dispatchable
                # pool -> next pass redispatches -> repeat). The cap counts
                # dead *dispatches* via a timestamp list
                # (``orphan_redispatch_at``) deduplicated on dispatch identity,
                # mirroring the ``redispatch_at``/``worker_death_at`` pattern
                # used by the parallel worker_death_loop cap elsewhere in this
                # module.
                # The previous implementation derived the count from
                # ``len(adapter_history)``, but that list only grew when
                # ``api_worker.enabled`` is ``True`` (the per-issue adapter
                # selector that wrote it was deleted in Phase 2 Track B,
                # PR #1517) -- so in the default (non-API-routed)
                # configuration the counter never incremented and the cap
                # never fired, leaving the #709 infinite loop unbounded in
                # production. The timestamp list is appended on every sweep
                # pass through this code, regardless of the configured
                # adapter.
                # "No progress" is measured, not assumed: the branch head SHA
                # (remote ls-remote + local worktree) is compared across
                # attempts. A moving head is the salvage path's job, not
                # escalation. Parallel to the rework lane's worker_death_loop
                # (fires at death_count > max_auto_redispatch).
                head_details = no_pr_issue_details.get(issue_number, {})
                current_head = _orphan_head_fingerprint(
                    head_details.get("remote_head_sha"),
                    head_details.get("local_head_sha"),
                )
                prior_head = entry.get("orphan_redispatch_head_sha")
                now_ts = utc_now()

                head_changed = prior_head is not None and current_head != prior_head
                first_observation = prior_head is None
                # Identity of the dispatch whose dead worker this pass is
                # observing. The #417 reclaim deliberately leaves the entry's
                # status/worker_pid untouched (issue #282 fingerprint
                # preservation), so the same dead entry is re-discovered by
                # every subsequent sweep pass until a real redispatch replaces
                # dispatched_at/worker_pid. Counting passes would therefore
                # escalate after a few sweeps with zero actual redispatch
                # attempts (e.g. during fleet-capacity dispatch delays);
                # instead, each dead dispatch is counted exactly once, keyed
                # by this identity.
                dispatch_identity = (
                    f"{entry.get('dispatched_at') or 'none'}:{entry.get('worker_pid') or 'none'}"
                )
                prior_dispatch = entry.get("orphan_redispatch_counted_dispatch")
                # Read the windowed timestamp list. On progress or first
                # observation, reset it to [now] so only attempts after this
                # point count toward the cap. Otherwise, append this pass's
                # timestamp only when it observes a dispatch not yet counted
                # -- re-observing the same dead dispatch on a later sweep
                # pass is not a redispatch attempt. (The timestamp list, not
                # adapter_history, is still what the count derives from:
                # adapter_history only grew when api_worker.enabled was True,
                # and the writer was deleted in Phase 2 Track B.)
                orphan_redispatch_at = _windowed_orphan_redispatch_at(
                    entry, window_minutes=config.watchdog.redispatch_window_minutes
                )
                if head_changed or first_observation:
                    orphan_redispatch_at = [now_ts]
                elif dispatch_identity != prior_dispatch:
                    orphan_redispatch_at = orphan_redispatch_at + [now_ts]

                redispatch_count = len(orphan_redispatch_at)

                if redispatch_count > config.watchdog.max_auto_redispatch and not head_changed:
                    # Cap exceeded with no progress -- escalate instead of
                    # recording the relabel event that would return the issue
                    # to the dispatchable pool. The labels were already
                    # stripped in the pre-lock reclaim, but the post-lock
                    # transition() call (via reap_escalations) will change
                    # them to agent:human-needed.
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="orphan_sweep_redispatch_cap_exceeded",
                        reason_class="mechanical",
                        issue_extra={
                            "dispatched_at": None,
                            "orphan_redispatch_head_sha": current_head,
                            "orphan_redispatch_at": orphan_redispatch_at,
                            "orphan_redispatch_counted_dispatch": None,
                            "orphan_flagged_at": None,
                            "orphan_drift_fingerprint": None,
                            "orphan_drift_at": None,
                        },
                    )
                    sweep_events.append(
                        (
                            "orphan_sweep_redispatch_escalated",
                            {
                                "issue_number": issue_number,
                                "previous_status": "dispatched",
                                "reason": "orphan_sweep_redispatch_cap_exceeded",
                                "redispatch_count": redispatch_count,
                                "branch_head_sha": head_details.get("remote_head_sha"),
                                "worktree_head_sha": head_details.get("local_head_sha"),
                                "orphan_redispatch_at_len": len(orphan_redispatch_at),
                            },
                        )
                    )
                    reap_escalations.append(issue_number)
                    continue

                # Cap not exceeded (or progress detected): persist the head
                # fingerprint, timestamp list, and counted dispatch identity
                # so the next orphan-sweep pass can compare against them. On
                # progress or first observation, the list was just (re)seeded
                # above; this persists it.
                entry["orphan_redispatch_head_sha"] = current_head
                entry["orphan_redispatch_at"] = orphan_redispatch_at
                entry["orphan_redispatch_counted_dispatch"] = dispatch_identity

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
                            _session_failed_relabeled_payload(
                                issue_number=issue_number,
                                reason="dead_worker_no_open_pr_orphan_sweep",
                                **reclaim,
                            ),
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
                # Issue #1230: ``orphan_drift_at`` arms the
                # ``dead_dispatched_reap_minutes`` time-based backstop checked
                # at the top of this loop.  It must be backfilled whenever it
                # is missing, INDEPENDENTLY of the ``orphan_flagged_at``
                # duplicate-event guard below.  An entry that was flagged
                # before this stamp existed (pre-#1230 builds set only
                # ``orphan_flagged_at``) or via the reclaim-success branch
                # above (which deliberately omits ``orphan_drift_at``) has
                # ``orphan_flagged_at`` set but ``orphan_drift_at`` absent, so
                # the guard's early-return permanently blocks the backstop
                # from ever arming.  That is the wedge: the issue is not
                # re-dispatchable (a terminal label like ``agent:human-needed``
                # excludes it from the dispatchable pool), not sweepable
                # (status is not ``escalated``), and the backstop never fires
                # because ``orphan_drift_at`` was never set.  Backfill from
                # ``orphan_flagged_at`` so the grace period is measured from
                # when the drift was first observed -- an already-wedged entry
                # (e.g. a sibling repo's #1421, flagged 4+ days ago) converges on the very
                # next sweep pass instead of waiting another full grace window.
                # The reclaim-success branch above deliberately does NOT set
                # ``orphan_drift_at`` -- that path leaves the issue ``ready``
                # and re-dispatchable, so the backstop should not fire on the
                # pass that reclaimed it.  This backfill only arms the backstop
                # for entries that reach THIS drift branch (nothing to reclaim
                # or reclaim failed), which means the issue is not on the
                # normal re-dispatch path and the backstop is the correct
                # convergence mechanism.
                if (
                    entry.get("orphan_drift_at") is None
                    and entry.get("orphan_flagged_at") is not None
                ):
                    entry["orphan_drift_at"] = entry["orphan_flagged_at"]
                if entry.get("orphan_flagged_at"):
                    state["issues"][str(issue_number)] = entry
                    continue
                drift_ts = utc_now()
                entry["orphan_flagged_at"] = drift_ts
                entry["orphan_drift_at"] = drift_ts
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
            state,
            sweep_events,
            max_size=config.runtime.event_ring_size,
            state_file=state_file,
            write_gate=write_gate,
        )
        write_gate.save_state(state)

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
            state = write_gate.append_event(
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
            )
            write_gate.save_state(state)

    # Issue #654: apply the ``escalated`` label edge for dead dispatched
    # workers that exceeded the reap grace period. The state.json update
    # (``_escalate_issue``) was done inside the lock above; ``transition``
    # does network I/O (GitHub label API) so it runs here, outside the lock,
    # matching the pattern ``_check_janitor_rework_stall`` uses. A failed
    # transition leaves the issue escalated in state.json with a stale label
    # -- the next pass's ``_detect_and_handle_orphaned_workers`` will not
    # re-escalate (status is no longer ``dispatched``), but reconcile's
    # ground-truth label sweep will eventually converge the label.
    for issue_number in reap_escalations:
        write_gate.transition(
            gh, config.labels, issue_number, _escalation_edge("escalated", "mechanical")
        )


# Issue #713: canonical key sets each prompt writer supplies to
# ``render_prompt`` -- the explicit ``values`` dict, excluding the dynamic
# ``$section_*`` partials ``section_variables`` discovers on disk. Used by
# ``check_prompt_template_drift`` to fail fast -- at supervisor startup and
# in CI -- when a repo-local flat whole-file override references a
# placeholder the writer no longer provides. This is the durable,
# structural fix for the bug class where a sibling repo's flat ``rework.md``
# override kept ``$review_summary`` after commit 5844c34 (PR #661) renamed
# the writer's slot to ``$dispatch_note`` / ``$required_changes_section``:
# ``render_prompt``'s strict mode catches the crash at dispatch time, but
# nothing caught it *before* dispatch, so it stayed live-armed on a running
# process until the next rework dispatch actually fired.
#
# The subset direction is deliberate: an override legitimately uses fewer
# placeholders than the writer supplies (a sibling repo's ``worker.md`` uses 6
# of these 8 worker keys), so the check fails only when the template reaches
# for a placeholder the writer never provides -- never when it merely
# ignores one the writer does. The reverse direction (every supplied key
# used) is not an error; at most a lint.
#
# Kept honest against the real writers by the registry-drift guard in
# ``tests/test_prompt_template_drift_check.py``, which monkeypatches
# ``render_prompt`` to capture the keys the writer actually passes and
# asserts these frozensets match exactly.
WORKER_PROMPT_KEYS: frozenset[str] = frozenset(
    # OrchestratorApp._write_worker_prompt -- the literal ``values`` dict
    # passed to ``self._render`` (see the writer below in this file).
    {
        "issue_number",
        "issue_title",
        "issue_url",
        "issue_body",
        "issue_body_block",
        "issue_comments",
        "branch_name",
        # Issue #1444: the module-map section, derived from the live tree at
        # packet build time by ``build_module_map``. Empty string when the map
        # could not be derived (fail-soft: omitted section + a
        # ``worker_module_map_failed`` warning event), never a dispatch
        # failure.
        "module_map",
        # Issue #1460: the attachment-point placement clause, gated on
        # `.attachment-budgets.json` presence. Empty string when the marker
        # is absent or fails to load (fail-soft: omitted clause + a
        # ``worker_attachment_budget_failed`` warning event), never a
        # dispatch failure. Deliberately NOT added to REWORK_PROMPT_KEYS --
        # the rework lane receives budget findings via the review packet's
        # `$attachment_budget_section` instead.
        "attachment_budget",
    }
)
REWORK_PROMPT_KEYS: frozenset[str] = frozenset(
    # _render_rework_prompt -- the literal ``values`` dict passed to
    # ``render_prompt`` (see the writer below in this file).
    {
        "pr_number",
        "pr_title",
        "pr_url",
        "issue_number",
        "dispatch_note",
        "dispatch_note_block",
        "required_changes_section",
        "branch_name",
    }
)

# Issue #1265: every writer of a review-decision verdict must say WHERE the
# decision came from. This is a distinct field from the pre-existing
# ``session_metrics.verdict_source`` (set in ``_reap_review_verdicts``,
# meaning "which parser found the reviewer's fenced verdict block" --
# log/events/file:{source}). ``verdict_provenance`` instead means "which
# mechanism produced this verdict at all" and is required (no default,
# anywhere) on every ``record_review`` call and on the two writers that
# bypass ``record_review`` entirely (``_update_approval_head``'s carry-
# forward patch, and the pending-reset template). Follows the same
# plain-string-checked-against-a-set convention ``record_review`` already
# uses for ``decision`` -- deliberately not an Enum type.
#
# Mapping (one call site can share a value; a value need not have exactly
# one call site -- ``carried_forward`` has none in ``record_review`` at all,
# it is stamped directly by ``_update_approval_head``):
#   ci_gate_auto_reject       -- review()'s CI-gate reject (sole failing
#                                check, and the co-occurring-janitor-failure
#                                variant added by #1258/#1286)
#   test_adequacy_auto_reject -- review()'s test-adequacy hard-gate reject
#                                (issue #179; a distinct mechanism from the
#                                CI gate, not folded into it)
#   fresh_llm_review          -- _reap_review_verdicts (a live reviewer's
#                                parsed verdict)
#   stranded_reconciliation   -- _reconcile_stranded_verdicts (issue #736: a
#                                completed on-disk verdict whose state write
#                                was lost, re-ingested from review-
#                                decision.json -- not a fresh decision)
#   rescue_review             -- rescue-tier review's approve
#   operator_manual           -- cli.py's ``charlie verdict`` command
#   carried_forward           -- _update_approval_head (never via
#                                record_review; stamped directly on the
#                                decision-file patch and all three
#                                verdict_carried_forward_* event payloads)
VERDICT_PROVENANCE_VALUES: frozenset[str] = frozenset(
    {
        "ci_gate_auto_reject",
        "test_adequacy_auto_reject",
        "fresh_llm_review",
        "stranded_reconciliation",
        "rescue_review",
        "operator_manual",
        "carried_forward",
    }
)


class PromptOverrideDriftError(RuntimeError):
    """One or more configured prompt templates reference placeholders their
    writer does not supply (issue #713).

    Raised at supervisor startup (and asserted in CI) by
    :func:`check_prompt_template_drift` so a repo-local flat whole-file
    override that drifted out of sync with the orchestrator's writer -- e.g.
    a flat ``rework.md`` still referencing ``$review_summary`` after the
    writer renamed it to ``$dispatch_note`` -- fails fast before any
    dispatch, instead of staying live-armed until the next dispatch crashes
    with an uncaught :class:`PromptTemplateError`.
    """

    def __init__(self, errors: Sequence["PromptTemplateError"]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(str(error) for error in self.errors)
        super().__init__(
            f"prompt template drift detected (issue #713); refusing to start: {details}"
        )


def check_prompt_template_drift(
    config: OrchestratorConfig, *, search_dirs: Sequence[Path] = ()
) -> list[PromptTemplateError]:
    """Static placeholder-subset check for every configured prompt template.

    For each template dispatch will render -- the configured worker/rework
    templates from ``DispatchConfig`` and ``ApiWorkerConfig`` -- resolve it the
    way dispatch would (repo-local override first, then the package default)
    and assert every ``$placeholder`` it references (after expanding the
    ``$section_*`` partials it pulls in) is a subset of the key set the
    corresponding writer supplies together with the section variables
    discovered on disk. Returns a list of :class:`PromptTemplateError` values,
    one per drifting template; an empty list means every configured template
    is safe to render.

    Pure static check: no dispatch, no worker, no network. It reads template
    and section files off disk only, so it runs at supervisor startup
    (``OrchestratorApp.__init__``) and in CI -- catching the #713 bug class
    (a flat override armed with a stale placeholder) before it can crash a
    live dispatch.

    A configured template name that resolves to no file (a typo, or a custom
    name neither the override nor the package ships) is skipped here: that is
    a separate config error dispatch surfaces as a ``FileNotFoundError``, not
    placeholder drift, and conflating the two would muddy the drift report.
    """
    pairs: list[tuple[str, frozenset[str]]] = [
        (config.dispatch.worker_template, WORKER_PROMPT_KEYS),
        (config.dispatch.rework_template, REWORK_PROMPT_KEYS),
        (config.api_worker.worker_template, WORKER_PROMPT_KEYS),
        (config.api_worker.rework_template, REWORK_PROMPT_KEYS),
    ]
    errors: list[PromptTemplateError] = []
    seen: set[str] = set()
    for template_name, keys in pairs:
        if template_name in seen:
            continue
        seen.add(template_name)
        resolved = resolve_template(template_name, search_dirs)
        if not resolved.is_file():
            continue
        missing = unsupplied_placeholders(template_name, keys, search_dirs=search_dirs)
        if missing:
            errors.append(PromptTemplateError(resolved, missing))
    return errors


ORCHESTRATOR_COMMENT_MARKER = "<!-- charlie-work:orchestrator-generated -->"
"""Provenance marker stamped into every PR comment the orchestrator writes.

Issue #950's external-findings ingestion cannot use author identity to tell its
own output apart from a human's: the orchestrator authenticates with a *user*
token, so ``gh api user`` reports ``type=User`` and its comments are
indistinguishable from genuine review comments by the same person. Filtering on
the login would suppress exactly the human findings the feature exists to
capture.

So provenance travels in the body instead. ``_comment_pr`` stamps this marker;
``_is_orchestrator_comment`` skips anything carrying it. An HTML comment is
invisible in rendered markdown, so the posted comment is unchanged for readers.

Note this covers only comments written by *this* process. A worker's own rework
reply is machine-generated too and is not stamped here -- it is excluded from
ingestion by a *temporal* cutoff instead (issue #998): ``_collect_external_findings``
drops any comment posted after the reviewed head commit's committer date, and a
worker's reply is by construction posted after the rework commit it describes.
"""


def _is_orchestrator_comment(item: dict[str, Any]) -> bool:
    """Return True if a comment body *begins* with the orchestrator's provenance marker.

    Deliberately a prefix test, not a substring test. ``_comment_pr`` writes the
    marker as the literal first line, so a prefix check catches every comment this
    process posts -- and a substring check would catch strictly more than that, in
    the one direction that costs real findings.

    The extra case a substring check would swallow: GitHub's "Quote reply" inserts
    the *raw markdown* of the quoted comment, HTML comments included, as a
    blockquote above the reply. Quote-replying to one of our ``request_changes``
    comments is a natural way for a human to respond point by point, and the
    resulting body contains the marker without starting with it. Under a substring
    test that whole comment -- quote plus whatever new finding the human wrote
    below it -- is dropped from ``required_changes`` silently.

    Dropping a genuine human finding is worse than ingesting one of our own
    comments, so this predicate is written to fail toward ingestion.
    """
    body = item.get("body")
    return isinstance(body, str) and body.lstrip().startswith(ORCHESTRATOR_COMMENT_MARKER)


def _is_bot_comment(item: dict[str, Any]) -> bool:
    """Return True if a comment/review author is a GitHub App/bot account.

    Uses the API-supplied ``user.type``/``author.type`` or ``author.is_bot``
    discriminator. Does not rely on a hardcoded login list.

    This catches genuine GitHub Apps (Aviator, dependabot, ...) but *not* the
    orchestrator itself, which posts under a user token -- see
    ``ORCHESTRATOR_COMMENT_MARKER``.
    """
    user = item.get("user")
    if isinstance(user, dict) and user.get("type") == "Bot":
        return True
    author = item.get("author")
    if isinstance(author, dict):
        if author.get("type") == "Bot" or author.get("is_bot") is True:
            return True
    return False


def _gh_api_list(gh: GitHubLike, path: str) -> list[dict[str, Any]]:
    """Call ``gh api --paginate <path>`` and return a list, swallowing errors."""
    result = gh.run(
        ["api", "--paginate", path],
        json_output=True,
        allow_failure=True,
    )
    if isinstance(result, GitHubRunResult):
        if not result.ok or not isinstance(result.value, list):
            return []
        return result.value
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _commit_timestamp(gh: GitHubLike, sha: str | None) -> str | None:
    """Return the committer-date ISO 8601 timestamp of commit ``sha``, or None.

    Wraps ``gh.commit(sha)`` (``gh api repos/{owner}/{repo}/commits/{sha}``) and
    reads ``commit.committer.date`` -- the moment the commit landed on the
    branch, which is what the external-findings upper bound (issue #998) needs
    to compare against comment timestamps. The author date is user-settable and
    can be backdated, so it is *not* used as the cutoff; the committer date is
    the GitHub-recorded landing time.

    Errors are returned as ``None``, never raised -- consistent with this
    repo's errors-as-values invariant. A ``None`` return tells
    ``_collect_external_findings`` to skip the upper bound entirely (fail
    toward ingestion): losing a genuine human finding is the expensive
    direction of this filter, and a missing commit timestamp must not silently
    drop comments.
    """
    if not sha:
        return None
    result = gh.commit(sha)
    value = result.value if isinstance(result, GitHubRunResult) and result.ok else None
    if not isinstance(value, dict):
        return None
    commit = value.get("commit")
    if not isinstance(commit, dict):
        return None
    committer = commit.get("committer")
    if isinstance(committer, dict):
        date = committer.get("date")
        if isinstance(date, str) and date:
            return date
    # Fall back to the author date only if the committer date is entirely
    # absent -- rare, but a squash with a crafted author/committer split could
    # leave committer empty. Author date is less authoritative (settable) but
    # still bounds the comment window better than no cutoff at all.
    author = commit.get("author")
    if isinstance(author, dict):
        date = author.get("date")
        if isinstance(date, str) and date:
            return date
    return None


def _collect_external_findings(
    gh: GitHubLike,
    pr_number: int,
    *,
    since: str | None = None,
    before: str | None = None,
) -> list[str]:
    """Collect non-bot human feedback from the three external PR surfaces.

    Fetches issue-level comments, review bodies, and inline review comments.
    Returns a flat list of markdown bodies ready to be folded into
    ``review-decision.json`` at record time.

    Three provenance/content filters apply, and they are deliberately
    different in kind:

    * **Bots** are excluded by author (``user.type == "Bot"``) -- GitHub Apps
      such as Aviator have their own identity, so identity is a sound
      discriminator for them.
    * **The orchestrator's own comments** are excluded by *body marker*, not by
      author. The orchestrator posts through a user token, so its comments are
      indistinguishable by identity from a genuine human review by the same
      person, and filtering on that login would drop exactly the findings #950
      exists to ingest. See ``ORCHESTRATOR_COMMENT_MARKER``.
    * **Reviewer-session crash summaries** are excluded by *body content*
      (issue #1269, W12): ``_extract_review_session_summary`` posts these
      when a reviewer dies without a verdict, and every one predating the
      ``ORCHESTRATOR_COMMENT_MARKER`` stamp (added #1242/55cecd9) carries no
      marker for the filter above to catch -- they were ingested as if they
      were genuine human findings. See ``body_has_crash_signature``. This
      filter only stops *new* ingestion; already-persisted poisoned records
      are cleaned separately, at render time, by
      ``rework_prompts._render_required_changes_section``.

    ``since``, when given, is the previous round's *upper* bound -- the
    ``before`` timestamp that round persisted in ``review-decision.json``
    (falling back to that round's ``reviewed_at`` only when no ``before`` was
    recorded; see ``record_review``'s contiguity note). An item timestamped
    at or before ``since`` was already visible -- either surfaced in a prior
    round's ``required_changes`` or predating review entirely -- so it is
    skipped. This is what stops a rework loop from re-ingesting the same stale
    findings on every round: without it, ``_write_rework_prompt`` buries the
    one new finding under the full comment history every time.

    ``before``, when given, is the *current* round's ``reviewed_head_sha``
    committer-date timestamp (see ``_commit_timestamp``). An item timestamped
    strictly after ``before`` is skipped. This is the upper bound that closes
    issue #998: a worker's own rework reply ("Reworked in <sha>, here is what
    I changed") is machine-generated, posted through the worker's path (no
    ``ORCHESTRATOR_COMMENT_MARKER``), and -- critically -- posted *after* the
    rework commit it describes, which is exactly the head the reviewer is now
    reading. Cutting ingestion off at that head's commit time drops the reply
    without any cooperation from the worker's posting path, and it also stops
    findings the worker has already addressed from being re-presented every
    round. The same account/``type=User`` constraint that defeated an
    identity-based fix in #997 applies here too, so the cutoff is temporal,
    not by author: a genuine human comment posted *before* the reviewed head
    is still ingested (asserted positively by
    ``test_worker_rework_reply_is_not_ingested_as_external_finding``).

    Contiguity contract: ``since`` and ``before`` together partition the
    comment timeline into per-round half-open windows ``(since, before]``, and
    the next round's ``since`` is this round's persisted ``before``. A comment
    in the gap ``(before, reviewed_at]`` -- posted after the reviewed head
    landed but before the verdict was written -- is excluded by ``before``
    this round and recovered by ``since`` next round. Deriving ``since`` from
    ``reviewed_at`` instead would drop such a comment forever (it satisfies
    ``item_dt <= reviewed_at``), violating the fail-toward-ingestion
    invariant. This is pinned by
    ``test_human_comment_in_before_to_reviewed_at_gap_surfaces_next_round``.

    GitHub's list endpoints disagree on which field carries the timestamp --
    issue comments and inline review comments use ``created_at``, top-level
    review bodies use ``submitted_at`` -- so both are checked per item. An
    item with neither field (or an unparsable one) is never skipped by either
    bound: losing a genuine finding is the expensive direction of this filter
    (see ``test_human_quote_reply_to_orchestrator_comment_is_still_ingested``).
    Likewise, if ``before`` cannot be parsed (or the commit timestamp could
    not be resolved at the call site) the upper bound is not applied -- the
    filter fails toward ingestion, never toward silent suppression.

    Issue #1269 (W12): the three surfaces are walked and combined into one
    list first, and *that* combined list is deduplicated by exact-string
    equality (``dict.fromkeys``, first-occurrence order preserved) at the
    end -- the end-position dedup is required precisely because duplicates
    are cross-surface, not just within one: a reviewer-session crash summary
    is frequently reposted byte-for-byte across issue comments and review
    bodies within the same round (the crash-recovery path posts to more
    than one surface), so deduplicating each surface independently before
    combining would miss exactly the duplicates this exists to catch. Exact
    equality only, no whitespace normalization beyond the existing
    ``.strip()`` above: two genuinely different findings that happen to
    share a substring must stay distinct.
    """
    bodies: list[str] = []
    owner_repo = "{owner}/{repo}"

    surfaces = (
        # A PR is also an issue, so issue-level comments live here.
        f"repos/{owner_repo}/issues/{pr_number}/comments",
        # Top-level PR review bodies.
        f"repos/{owner_repo}/pulls/{pr_number}/reviews",
        # Inline review comments on specific diff lines.
        f"repos/{owner_repo}/pulls/{pr_number}/comments",
    )

    since_dt = _parse_iso_timestamp(since) if since else None
    before_dt = _parse_iso_timestamp(before) if before else None

    for path in surfaces:
        for item in _gh_api_list(gh, path):
            if _is_bot_comment(item) or _is_orchestrator_comment(item):
                continue
            item_dt = _parse_iso_timestamp(item.get("created_at") or item.get("submitted_at"))
            if item_dt is not None:
                if since_dt is not None and item_dt <= since_dt:
                    continue
                if before_dt is not None and item_dt > before_dt:
                    continue
            body_value = item.get("body")
            if not isinstance(body_value, str):
                continue
            body = body_value.strip()
            if body and not body_has_crash_signature(body):
                bodies.append(body)

    # Issue #1269 (W12): collapse exact-duplicate bodies (see docstring
    # above). dict.fromkeys preserves first-occurrence order in O(n).
    return list(dict.fromkeys(bodies))


# Issue #1268 (W11), item 2: bound how much verdict prose a single
# ``record_review`` events.db row can carry. events.db is append-only and
# unbounded (unlike the capped in-memory events ring), so unbounded reviewer
# prose would grow the DB file without limit; the round-K archive (above) is
# the full-text source of truth, this is an observability-only copy. A
# distinct helper from ``_truncate_reason`` deliberately: that one has
# existing callers depending on its 200-char/``"..."`` contract, and
# retuning it for a 16KB/``"...truncated"`` use would risk their behavior.
_EVENT_TEXT_MAX_BYTES = 16 * 1024
_EVENT_TRUNCATE_MARKER = "...truncated"


def _truncate_for_event(text: str, max_bytes: int = _EVENT_TEXT_MAX_BYTES) -> str:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes for an event payload.

    Byte-based, not character-based -- the budget is a DB-row-size concern,
    and reviewer prose is not guaranteed ASCII. ``errors="ignore"`` on the
    final decode can drop a partial multi-byte character sitting exactly at
    the cut boundary, so the truncated result's encoded length is only
    guaranteed to be ``<= max_bytes``, not exactly equal to it.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker_bytes = _EVENT_TRUNCATE_MARKER.encode("utf-8")
    keep = max(0, max_bytes - len(marker_bytes))
    return encoded[:keep].decode("utf-8", errors="ignore") + _EVENT_TRUNCATE_MARKER


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
        and not summary.infra_blocked
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
    if summary.infra_blocked:
        buckets.append(f"infra_blocked: {', '.join(summary.infra_blocked)}")
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


def _is_rerun_already_running_error(error: str) -> bool:
    """Return True if a ``gh run rerun`` error means the run is still in progress.

    GitHub refuses to rerun a workflow while any job in the run is still
    running. The CLI surfaces this with the phrase ``already running`` in the
    error, e.g. ``run <id> cannot be rerun; This workflow is already running``
    or ``The workflow run containing this job is already running``. Other
    refusal reasons (expired run, broken workflow file, missing permissions,
    ...) do not contain this phrase and should not be treated as retryable.
    """
    return "already running" in error.lower()


def _try_reap_blocked_foreign_writer(
    failed_result: Any,
    config: OrchestratorConfig,
    state_file: Path,
    issue_number: int,
    sessions_dir: Path,
) -> bool:
    """Attempt to reap an idle foreign writer at the blocked-environment cap.

    Issue #1423: when the ``blocked_environment_at`` counter exhausts the
    ``max_auto_redispatch`` cap for a ``worktree_foreign_writer`` failure,
    the default action is to escalate the issue to a human. But a writer that
    was active on earlier passes and has since gone idle is a zombie the fleet
    itself launched and forgot — escalating it wastes a human's attention on
    a process the stall detector would have reaped if it could see it.

    This helper reads the marker from the failed dispatch's worktree path and
    delegates to ``_reap_idle_foreign_writer`` (the single enforcement point
    for every reap-candidate guard + the activity probe + kill path). When the
    writer is idle past the threshold, it is reaped and the caller resets the
    counter instead of escalating. When the writer is still active, the caller
    escalates as before — escalation is reserved for a writer that is alive
    *and* active.

    Issue #1443 finding 2: the pid is no longer read from ``failed_result``.
    ``_reap_idle_foreign_writer`` derives it from the marker (single source of
    truth) so the pid can never disagree with the ``process_start_time``
    fingerprint passed to ``kill_process_tree``. ``failed_result`` is used
    only to locate the worktree path and confirm the failure kind.

    Issue #1443 review: ``sessions_dir`` is a required parameter (no default).
    Every production caller passes it explicitly; removing the default converts
    a silently-disabled own-live-session guard into an immediate ``TypeError``
    if a future call site drops or reorders the argument — the exact bug shape
    #1443 was filed to fix.

    Returns ``True`` when the writer was reaped, ``False`` otherwise (including
    when the failed result is not a foreign-writer block or the marker is gone).
    """
    if failed_result is None:
        return False
    failure_kind = getattr(failed_result, "failure_kind", None)
    if failure_kind != "worktree_foreign_writer":
        return False
    worktree_path_str = getattr(failed_result, "worktree_path", None)
    if not worktree_path_str:
        return False
    worktree_path = Path(worktree_path_str)
    marker = read_worktree_marker(worktree_path)
    if marker is None:
        return False
    return _reap_idle_foreign_writer(
        worktree_path,
        marker,
        config,
        sessions_dir,
        state_file=state_file,
        issue_number=issue_number,
    )


def _has_other_open_pr(
    state: dict[str, Any], issue_number: int | None, exclude_pr_number: int | None
) -> bool:
    """Return True if any PR for ``issue_number`` is open besides ``exclude_pr_number``.

    Used by ``unescalate`` to decide whether a terminal PR on GitHub is the
    *only* PR for the issue (issue #1391): when it is, the issue can be
    dropped to baseline in the same call instead of requiring a second
    ``unescalate --issue N`` to take the no-live-PR path. "Open" here means
    the state.json status is not ``merged`` or ``closed`` — the same
    predicate the PR resolution at the top of ``unescalate`` uses.
    """
    if issue_number is None:
        return False
    for k, v in state.get("prs", {}).items():
        if not isinstance(v, dict) or not k.isdigit():
            continue
        if v.get("issue_number") != issue_number:
            continue
        if exclude_pr_number is not None and int(k) == exclude_pr_number:
            continue
        if v.get("status") not in ("merged", "closed"):
            return True
    return False


def _launch_review_claude_code(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    return launch_claude_worker(
        issue_number=pr_number,
        branch=branch,
        prompt_text=prompt_text,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        config=config,
        env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        # Force-enabled for reviewers: the structured events.jsonl is needed
        # for verdict fallback parsing (issue #540) and token/turn monitoring.
        tee_stream_json=True,
        resolved_review_effort=resolved_review_effort,
        max_turns_override=max_turns_override,
        model_override=model_override,
    )


def _launch_review_devin_shell(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    # devin_shell has no notion of review-effort/turn-cap resolution (those
    # are claude-code CLI concepts -- --effort and --max-turns flags); a
    # devin-routed reviewer runs with the CLI's own defaults for both.
    return launch_devin_session(
        pr_number,
        branch,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        config=config,
        worker_env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        worker_model=model_override or "",
    )


def _launch_review_api(
    *,
    pr_number: int,
    branch: str,
    prompt_path: Path,
    prompt_text: str,
    head_sha: str,
    repo_root: Path,
    reviews_dir: Path,
    config: OrchestratorConfig,
    worker_env: dict[str, str],
    materialize_dirs: tuple[str, ...],
    resolved_review_effort: str | None,
    max_turns_override: int | None,
    model_override: str | None,
    api_worker_config: ApiWorkerConfig | None,
) -> Any:
    # model_override is deliberately unused here: an api-routed reviewer
    # always runs the configured provider's pinned model (see
    # api_worker.launch_api_worker's own model_override=provider.model),
    # the same as an api-routed worker -- reviewer.model has no effect on
    # this harness.
    assert api_worker_config is not None  # only None for a non-api harness
    return launch_api_worker(
        pr_number,
        branch,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        api_worker_config=api_worker_config,
        worker_env=worker_env,
        materialize_dirs=materialize_dirs,
        review=True,
        head_sha=head_sha,
        resolved_review_effort=resolved_review_effort,
        max_turns_override=max_turns_override,
        config=config,
    )


# Single dispatch table keyed by ``reviewer.harness`` name -- this, not a
# per-harness if/elif chain, is what ``OrchestratorApp.dispatch_reviews``
# consumes (issue #1513). Every launcher above shares one keyword-only
# signature so the call site does not need to know which positional/keyword
# convention the underlying launch function uses (``launch_claude_worker``/
# ``launch_api_worker`` take ``prompt_text``; ``launch_devin_session`` takes
# ``prompt_path``); each returns a record with ``.error``/``.pid``/
# ``.process_start_time``, which is all the post-launch handling below reads.
# The assertion is the drift guard: it fails at import time if a harness is
# ever added to (or removed from) ``harnesses.REVIEWER_HARNESSES`` without a
# matching entry here, the same pattern ``adapters._ADAPTER_DISPATCHERS``
# uses for the worker side.
_REVIEW_LAUNCHERS: dict[str, Callable[..., Any]] = {
    "claude-code": _launch_review_claude_code,
    "devin-shell": _launch_review_devin_shell,
    "api": _launch_review_api,
}

assert set(_REVIEW_LAUNCHERS) == REVIEWER_HARNESSES, (
    "workflow._REVIEW_LAUNCHERS must launch exactly the harnesses "
    "harnesses.REVIEWER_HARNESSES declares review-capable -- keep both in sync"
)


class OrchestratorApp:
    def __init__(
        self,
        repo_root: Path,
        paths: RuntimePaths,
        config: OrchestratorConfig,
        gh: GitHubLike,
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
        self.write_gate = WriteGate(
            dry_run=self.dry_run, state_path=self.paths.state_file, repo=self.repo_root.name
        )
        self.fleet_dir_override = fleet_dir_override
        # Issue #1363: config_freshness's "exactly once per change" semantics
        # need a mtime cache that outlives a single pass but not the process
        # -- an in-memory dict on the (per-process, per-repo) app instance is
        # exactly that. Fresh at supervisor startup by design: an edit that
        # landed before this instance's first pass is not "stale since load"
        # for THIS instance, only an edit after.
        self._preflight_config_mtimes: dict[str, float] = {}
        # Issue #1001: same-instance once-only escalation flag for the
        # worker-github-token gate. A missing token is a standing condition;
        # the gate must not emit an event every loop pass. The cross-instance
        # source of truth is the durable ``worker_token_escalated`` marker in
        # state.json (fleet_loop rebuilds this app per repo per pass, so an
        # instance flag alone resets every pass). This in-memory flag is a
        # same-instance optimization that also suppresses re-entry under
        # dry-run, where the durable marker is never written. It is cleared
        # when the condition resolves (all findings ok) alongside the marker.
        self._worker_token_escalated = False
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

        # Issue #713: fail fast at startup if any configured prompt template
        # (repo-local override or package default) references a placeholder its
        # writer does not supply. A flat whole-file override that drifted out of
        # sync -- e.g. a stale ``$review_summary`` after the writer renamed it to
        # ``$dispatch_note`` -- would otherwise stay live-armed until the next
        # dispatch crashes with an uncaught PromptTemplateError (which is not
        # caught anywhere in src/). This mirrors the gh field-list self-check
        # below: a pure static gate that runs before any dispatch/review work.
        drift_errors = check_prompt_template_drift(self.config, search_dirs=self.prompt_dirs)
        if drift_errors:
            raise PromptOverrideDriftError(drift_errors)

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

    @_guard_state_lock
    def status(self, *, use_cache: bool = True) -> CommandResult:
        if use_cache and (cached := status_snapshot.read_status_snapshot(self)) is not None:
            return cached
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

        # Warm the blocker/dependency cache for every ready issue concurrently
        # before the serial per-issue lookups below run. See
        # _prefetch_blocker_data's docstring and issue #870: this is what
        # turns O(N) serial `gh` calls into one parallel batch.
        self._prefetch_blocker_data(issues)

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
                self._summarize_worker(
                    view,
                    classify_worker_health(view, self.config, now, probe),
                    probe=probe,
                )
            )

        # Observe runner pool if feature is enabled
        runners_data = None
        if self.config.runner_scaling.enabled:
            from ci_fleet.charlie_work_adapter import format_runner_pool_state, observe_runner_pool

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
            # Issue #944: ready_issue_count above is the ready-FILTERED count,
            # so "0 ready issues" and "0 issues at all" print identically. This
            # is the unfiltered view: how big the backlog really is and which
            # gate is rejecting it.
            # len(issues) with no OPEN filter is correct HERE and would be a bug
            # in _dispatch_impl: this method's query (above) is
            # issue_list(labels.ready) with no state, and GitHubClient defaults
            # to state="open" (github.py: `effective_state = state or "open"`),
            # so `issues` is already open-only. _dispatch_impl passes
            # state="all" and must therefore count OPEN itself, or closed
            # ready-labelled issues would inflate the count and report
            # consistent=False on every healthy pass.
            "backlog_reachability": classify_backlog_reachability(
                self.gh,
                self.config,
                operator_claimed,
                ready_open_count=len(issues),
                # Issue #1337: model the merged-PR mention-only dispatch
                # exclusion so a mention-covered issue bins as
                # ``mention_covered_awaiting_operator`` instead of
                # ``dispatchable``. Fail-open fetch via
                # fetch_merged_prs_fail_open (a merged_pr_list failure
                # leaves the map empty -- the classifier is advisory).
                mention_covered=compute_mention_coverage_map(
                    issues,
                    fetch_merged_prs_fail_open(self.gh),
                    self,
                ),
            ),
            "snapshot_written_at": None,
            "cache_age_seconds": None,
        }

        # Add runners section if feature is enabled and observation succeeded
        if runners_data is not None:
            data["runners"] = runners_data

        return CommandResult(True, "status complete", data)

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
            issue_json = issue_dir / "issue.json"
            # Issue #618: in dry-run, skip all file mutations (issue dir,
            # issue.json, worker-prompt.md) — the preview must not touch disk.
            if not self.dry_run:
                issue_dir.mkdir(parents=True, exist_ok=True)
                self._write_json(issue_json, full_issue)
            prompt_path = self._write_worker_prompt(full_issue, dry_run=self.dry_run)

            # Check for prose-only dependencies (issue #225)
            body_text = full_issue.get("body", "")
            has_prose_deps = detect_prose_only_dependencies(body_text)
            has_structured_blockers = bool(parse_blockers(body_text))

            # If prose-only dependencies exist without structured blockers, label for human attention
            if has_prose_deps and not has_structured_blockers:
                prose_only_deps_issues.append(issue_number)
                if not self.dry_run:
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
        # Single lock for all state updates — skipped in dry-run (issue #618)
        if not self.dry_run:
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
        if self.dry_run:
            message = f"dry-run: would intake {len(written)} issue(s)"
        elif failed:
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

    def _dispatch_impl(
        self,
        limit: int | None = None,
        *,
        only_issues: str | None = None,
        stalled_entries: list[dict[str, int]] | None = None,
        ready_issues: list[dict[str, Any]] | None = None,
        merged_prs: _MergedPRListOutcome | None = None,
    ) -> CommandResult:
        # Issue #1001: worker GitHub token gate. Before dispatching to an
        # adapter family that routes through sanitize_env's merge, consult the
        # same predicate doctor._check_worker_github_token uses. If no
        # worker_env token is configured, escalate once (not per pass) and
        # either refuse (when dispatch.require_worker_github_token is True) or
        # warn and proceed (the default, so the gate does not take the fleet
        # down on a config that has not yet been provisioned — see the issue
        # #1001 sequencing hazard comment in config.py).
        #
        # The once-only guarantee must hold across OrchestratorApp
        # reconstruction: fleet_dispatch.fleet_loop builds a fresh app per
        # repo per pass, so an instance-level flag alone resets every pass
        # and re-escalates indefinitely. The durable marker
        # ``worker_token_escalated`` in state.json is the cross-instance
        # source of truth; the instance-level ``_worker_token_escalated``
        # flag (initialized in __init__) is a same-instance optimization
        # that also covers dry-run, where the durable marker is never
        # written. The marker is cleared when the condition resolves (all
        # findings ok), so a future regression re-escalates.
        token_findings = worker_github_token_findings(self.config)
        missing_findings = [f for f in token_findings if not f.ok]
        if missing_findings:
            if not self._worker_token_escalated:
                self._worker_token_escalated = True
                # Record the escalation event once. Payload carries only
                # config_key names and adapter contexts — never a token
                # value or prefix (issue #1001 acceptance criterion).
                #
                # Dry-run never writes: the escalation event and the
                # durable marker are state mutations (state_lock +
                # save_state), so they are gated on ``not self.dry_run``
                # — the same read-only contract documented at the
                # merge_ready dry-run gate (~line 15452, "Dry-run never
                # writes") and modelled on this function's own
                # top-of-body dry-run short-circuit. The in-memory
                # once-only flag is still set under dry-run so a dry-run
                # pass does not re-enter this block on the next pass;
                # the event and marker are emitted on the first real
                # (non-dry-run) dispatch. ``self.dry_run`` is fixed at
                # construction, so a dry-run instance cannot later
                # "forget" the flag and skip a real write.
                #
                # The durable marker is read inside the lock (not before it)
                # so the cross-instance once-only guarantee holds without an
                # unlocked load_state — issue #310's
                # test_no_unlocked_load_state_in_production_code lint forbids
                # any load_state outside a state_lock block. The in-memory
                # ``_worker_token_escalated`` flag is the first gate
                # (same-instance), so the lock is entered at most once per
                # instance lifetime (the flag's False→True transition); the
                # inner ``if not state.get(...)`` re-check is the
                # authoritative cross-instance gate and no-ops when a prior
                # instance already set the marker.
                if not self.dry_run:
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        if not state.get("worker_token_escalated", False):
                            state = self._record_event(
                                state,
                                "worker_token_missing",
                                {
                                    "findings": [
                                        {
                                            "config_key": f.config_key,
                                            "context": f.context,
                                        }
                                        for f in missing_findings
                                    ],
                                },
                                level="warning",
                            )
                            state["worker_token_escalated"] = True
                            save_state(self.paths.state_file, state)
            # The refusal itself is NOT dry-run-gated: a dry-run preview must
            # report the same deferral a live pass would take (matching the
            # fleet_lock_held / graphql_rate_limit deferral precedent in this
            # function). Only the escalation event / durable marker writes
            # above stay behind ``not self.dry_run``.
            if self.config.dispatch.require_worker_github_token:
                return CommandResult(
                    True,
                    "dispatch deferred: no sanctioned worker GitHub token "
                    "(set devin.worker_env / claude_code.worker_env "
                    "{'GH_TOKEN': '<scoped-PAT>'}; see issue #1001)",
                    {
                        "selected_count": 0,
                        "attempted_count": 0,
                        "failed_count": 0,
                        "skipped_issue_numbers": [],
                        "label_errors": [],
                        "sessions": [],
                        "dispatch_results": [],
                        "deferred_reason": "worker_token_missing",
                        "missing_config_keys": [f.config_key for f in missing_findings],
                    },
                )
        else:
            self._worker_token_escalated = False
            # Condition resolved: clear the durable marker so a future
            # regression re-escalates. Dry-run never writes — a dry-run pass
            # that observes a now-healthy config must not mutate the marker
            # set by a prior real pass (and cannot have set it itself).
            if not self.dry_run:
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    if state.get("worker_token_escalated", False):
                        state["worker_token_escalated"] = False
                        save_state(self.paths.state_file, state)

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

        # Issue #1110 rework: classify_backlog_reachability now runs the same
        # per-issue blocker check the dispatch candidate filter runs (issue
        # #1110 wired _get_open_blockers_for_issue into its else-branch). That
        # check calls get_github_issue_dependencies + are_issues_open per issue
        # -- exactly the N+1 serial `gh` pattern issue #870 built
        # _prefetch_blocker_data to eliminate. Warm the pass-scoped cache for
        # every ready issue once, *before* reachability's serial per-issue
        # lookups run, mirroring how status() warms the cache before its own
        # classify_backlog_reachability call. ``issues`` here is the
        # ready-labelled state="all" set; reachability fetches its own open
        # list, but its blocker check only runs on ready-labelled OPEN issues,
        # which are a subset of this set, so this warm-up covers every
        # dependency lookup reachability will make. The later
        # _filter_blocked_issues(candidates) call below benefits too:
        # candidates are a further subset, so the cache is already warm for
        # them as well. Harmless to call twice (second call is a cache hit).
        self._prefetch_blocker_data(issues)

        # Issue #944: observe the UNFILTERED backlog alongside the filtered
        # candidate query above. This does not participate in selection and
        # must not change dispatch behaviour -- it exists so that a zero
        # dispatch count carries a reason. Done here, outside the state lock,
        # because it is network I/O.
        # Issue #1337: resolve the mention-coverage map, reusing an
        # already-fetched merged-PR outcome when available and fetching
        # (fail-open) otherwise. See resolve_dispatch_mention_coverage's
        # docstring for the three-branch logic. A fresh fetch rebinds
        # ``merged_prs`` so the later _resolve_merged_prs calls and the
        # tripwire reuse the same list -- no second API call.
        mention_covered, _fetched_merged_prs = resolve_dispatch_mention_coverage(
            issues, merged_prs, self.gh, self
        )
        if _fetched_merged_prs is not None:
            merged_prs = _MergedPRListOutcome(_fetched_merged_prs, called=True)
        backlog_reachability = classify_backlog_reachability(
            self.gh,
            self.config,
            operator_claimed_issues(load_state_locked(self.paths.state_file)),
            ready_open_count=sum(
                1 for issue in issues if str(issue.get("state") or "OPEN").upper() == "OPEN"
            ),
            mention_covered=mention_covered,
        )

        # Gather sessions_dir for stall detection and live worker counting
        sessions_dir = self._layout.sessions_dir
        # Issue #1393: clean up stranded .json.tmp session sidecar files from
        # interrupted atomic writes before this pass writes new ones.
        cleanup_stale_session_tmp_files(sessions_dir)

        # Detect and handle stalled sessions before applying concurrency governor.
        # This must run exactly once per pass, not twice (was duplicated in the
        # governor). When the caller already ran the sweep this pass (loop()'s
        # unconditional reaper at the top of its pass), it hands the result down
        # via ``stalled_entries`` and the sweep is NOT re-run here — see
        # dispatch()'s docstring for why re-running it corrupts the Signal-1
        # deferral counter.
        if stalled_entries is None:
            stalled_entries = _detect_and_handle_stalled_sessions(
                sessions_dir,
                self.paths.state_file,
                self.config,
                write_gate=self.write_gate,
            )

        # Count live workers after stall handling (stalled workers are killed).
        # Corroborated against state.json (issue #343) so a ghost -- a live
        # worker_pid whose sidecar was removed -- cannot silently free a slot.
        live_count = _count_live_sessions(sessions_dir, self.paths.state_file)

        # Apply global concurrency governor cap with pre-computed live_count.
        # Issue #1129: fresh-issue dispatch also applies open-PR backpressure
        # (max_open_agent_prs), pacing new PR creation to the review/merge lane.
        gov = self._apply_concurrency_governor(
            dispatch_limit,
            live_count=live_count,
            apply_open_pr_backpressure=True,
        )
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
                if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
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
            resolved_merged_prs = _resolve_merged_prs(merged_prs)
            (
                merged_pr_bound_issue_numbers,
                merged_pr_mention_only_issue_numbers,
                _,
                _,
            ) = self._merged_pr_referenced_issue_numbers(issues, resolved_merged_prs)
            merged_pr_issue_numbers = (
                merged_pr_bound_issue_numbers | merged_pr_mention_only_issue_numbers
            )
            pr_by_issue = {}
            # Issue #1229: validate branch-name-derived issue numbers against the
            # real open-issue set so a stale branch name (e.g. agent/issue-709-…
            # left over from a merged issue/PR #709, reused by an unrelated
            # issue-less PR) cannot populate pr_by_issue[<wrong n>] and make the
            # dry-run report wrongly claim a real, dispatchable issue already has
            # an open PR. Same validator as the real dispatch-claim path below so
            # the two cannot diverge.
            branch_validator = self._make_branch_issue_validator()
            for pr in prs:
                issue_number = linked_issue_number(
                    pr,
                    is_cross_repository=pr.get("isCrossRepository"),
                    branch_prefix=self.config.dispatch.branch_prefix,
                    branch_issue_validator=branch_validator,
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
                # Issue #1336: lift the mention-only exclusion for re-armed
                # issues (read-only detection for dry-run parity with the
                # real dispatch path -- never stamps state).
                _already_flagged_dry = {
                    int(num)
                    for num, entry in state.get("issues", {}).items()
                    if isinstance(entry, dict) and entry.get("merged_pr_mention_flagged_at")
                }
                rearmed_mention_issues, _ = self._mention_rearmed_issue_numbers(
                    merged_pr_mention_only_issue_numbers,
                    issues,
                    state,
                    _already_flagged_dry,
                )
                merged_pr_issue_numbers = merged_pr_bound_issue_numbers | (
                    merged_pr_mention_only_issue_numbers - rearmed_mention_issues
                )
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
            (
                selected,
                skipped_issue_numbers,
                deferred_by_concurrency_full,
                deferred_by_concurrency_count,
            ) = _select_dispatch_candidates(
                candidates,
                dispatch_limit,
                state,
                self._branch_name,
                only_issues=only_issues,
            )
            # Issue #1005 review: this dry-run branch never persists an event
            # or calls _build_failure_map, but truncate for display parity
            # with the real-dispatch payload below -- keep the untruncated
            # list around under its own name so nothing downstream mistakes
            # it for complete.
            deferred_by_concurrency = deferred_by_concurrency_full[
                :_MAX_DEFERRED_CONCURRENCY_EXAMPLES
            ]
            selected_issue_numbers = [int(issue["number"]) for issue in selected]

            # Compute would-be SessionRequests without state mutation
            session_requests: list[SessionRequest] = []
            full_issues: dict[int, dict[str, Any]] = {}
            # Issue #1010: dry-run cross-repo gate — report which issues would
            # be escalated without mutating state or labels.
            # Issue #1244: dry-run cross-repo *scope* gate — report issues whose
            # title names another managed repo.
            dry_run_cross_repo_escalated: dict[int, str] = {}
            fleet_repos = managed_repo_names(self.fleet_dir_override)
            dispatching_repo_name = _dispatching_repo_name(self.gh, self.repo_root)
            for issue_number in selected_issue_numbers:
                full_issue = self.gh.issue_view(issue_number)
                full_issues[issue_number] = full_issue
                branch_name = self._branch_name(full_issue)

                # Pre-flight gate: report cross-repo targets without escalating.
                gate_result = cross_repo_gate(str(full_issue.get("body") or ""), self.repo_root)
                if not gate_result.passed:
                    dry_run_cross_repo_escalated[issue_number] = gate_result.reason
                    continue

                # Pre-flight scope gate: report cross-repo scope targets.
                scope_result = cross_repo_scope_gate(
                    str(full_issue.get("title") or ""),
                    str(full_issue.get("body") or ""),
                    dispatching_repo_name,
                    fleet_repos,
                )
                if not scope_result.passed:
                    dry_run_cross_repo_escalated[issue_number] = scope_result.reason
                    continue

                prompt_path = self._write_worker_prompt(full_issue, dry_run=True)

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
                "deferred_by_concurrency_count": deferred_by_concurrency_count,
                "merged_prs": resolved_merged_prs,
                "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                "merged_pr_mention_only_issue_numbers": sorted(
                    merged_pr_mention_only_issue_numbers
                ),
                "merged_pr_mention_rearmed_issue_numbers": sorted(rearmed_mention_issues),
                "label_errors": [],
                "cross_repo_escalated_issue_numbers": sorted(dry_run_cross_repo_escalated),
                "sessions": [asdict(request) for request in session_requests],
                "dispatch_results": [],
                "blocked": [
                    {"issue": issue_number, "blockers": blockers}
                    for issue_number, blockers in sorted(blocked_issues.items())
                ],
                "stalled": stalled_entries,
            }
            if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
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
        resolved_merged_prs = _resolve_merged_prs(merged_prs)
        (
            merged_pr_bound_issue_numbers,
            merged_pr_mention_only_issue_numbers,
            merged_pr_bound_pr_numbers,
            _,
        ) = self._merged_pr_referenced_issue_numbers(issues, resolved_merged_prs)
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
        # Issue #1229: validate branch-name-derived issue numbers against the
        # real open-issue set so a stale branch name (e.g. agent/issue-709-…
        # left over from a merged issue/PR #709, reused by an unrelated
        # issue-less PR) cannot populate pr_by_issue[<wrong n>] and make the
        # dispatcher believe a real, dispatchable issue already has an open PR,
        # silently skipping dispatch for it. This is the same phantom-binding
        # failure class already fixed at the dead-session escalation guard
        # (_classify_dead_sessions_and_update_throttle_state) and the
        # orphaned-worker sweep (_detect_and_handle_orphaned_workers); all
        # route through _make_branch_issue_validator so the open-issue fetch
        # cannot diverge between call surfaces.
        branch_validator = self._make_branch_issue_validator()
        for pr in prs:
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
                branch_issue_validator=branch_validator,
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
        # Issue #1336 follow-up to #564 point 2: the mention-only *dispatch
        # exclusion* previously keyed off the raw mention scan
        # (merged_pr_issue_numbers below) alone, so an operator who removed
        # agent:human-needed to re-arm automation could NOT re-enter dispatch
        # -- the scan-based exclusion kept blocking the issue until it closed
        # or the mentioning PRs were no longer merged/referenced. The
        # exclusion now lifts for an issue once it was flagged in a prior pass
        # and the operator has removed agent:human-needed: the re-arm is
        # detected from the already-loaded issue labels (no new per-pass API
        # call) and recorded durably in state.json as ``mention_rearmed_at``,
        # so the candidate filter keys the lift off the state signal rather
        # than a per-pass label read -- the blast-radius concern the original
        # comment raised. ``bound`` exclusions stay scan-based; the safe
        # default (never-flagged or still-carries-human-needed stays
        # excluded) is preserved. See ``_mention_rearmed_issue_numbers`` and
        # the re-arm block inside the state lock below.
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
                    # event-consumer: audit-only -- records the issue-status "closed" mutation
                    # already applied inline above; no downstream consumer needed
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
                _bound_pr_state = {
                    **_pr_entry,
                    "status": "merged",
                    "merged": True,
                }
                # Issue #747: stamp ``merged_at`` only on a genuine non-merged
                # -> merged transition; preserve the original observation time
                # on entries already recorded as merged.
                if _pr_entry.get("status") != "merged":
                    _bound_pr_state["merged_at"] = utc_now()
                state["prs"][_pr_key] = _bound_pr_state
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
                    # Issue #783: a merged PR that only mentions the issue in
                    # free text is a hijack-safety judgment call, not a
                    # process failure -- never auto-de-escalated. "status" is
                    # deliberately untouched (see above), so this reason_class
                    # is carried for completeness/consistency even though the
                    # de-escalation sweep never visits this issue via status.
                    "reason_class": escalation_reason_class("judgment"),
                }
            if stamped_mention_issues:
                state = append_event(
                    state,
                    "dispatch_merged_pr_mention_flagged",
                    {"issue_numbers": stamped_mention_issues},
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
            # Issue #1336: lift the mention-only dispatch exclusion once the
            # operator has re-armed a previously-flagged issue (removed
            # agent:human-needed). The re-arm is recorded durably as
            # ``mention_rearmed_at`` so the exclusion keys off a state.json
            # signal rather than a per-pass GitHub label read in the
            # candidate filter -- the blast-radius concern the #564 point-2
            # comment raised when it documented this as out of scope.
            #
            # Safe default preserved: an issue never flagged, flagged this
            # pass, or flagged and still carrying agent:human-needed stays
            # excluded. ``bound`` exclusions are never lifted -- those PRs
            # genuinely bound to the issue by a hijack-safe signal.
            #
            # The durable stamp + event emission is routed through
            # ``_stamp_mention_rearm`` (Convention A: ``self.write_gate.*``)
            # rather than raw ``append_event``/``save_state`` so the R9
            # shrink-only ratchet on workflow.py's raw-primitive count
            # (issue #1264 W6 PR4) is not increased -- the re-arm writes are
            # new territory this wave does not convert, and a raw
            # ``append_event``+``save_state`` pair would trip the ratchet
            # (baseline 266). The existing flag block above stays raw (it is
            # part of the ratchet's baseline); only the NEW re-arm writes go
            # through the gate.
            rearmed_mention_issues, newly_rearmed_mention_issues = (
                self._mention_rearmed_issue_numbers(
                    merged_pr_mention_only_issue_numbers,
                    issues,
                    state,
                    already_flagged_mention_issues,
                )
            )
            state = self._stamp_mention_rearm(state, newly_rearmed_mention_issues)
            # Recompute the exclusion set so the candidate filter below and
            # the result payload reflect the lifted mention-only exclusions.
            # ``bound`` exclusions are never lifted.
            merged_pr_issue_numbers = merged_pr_bound_issue_numbers | (
                merged_pr_mention_only_issue_numbers - rearmed_mention_issues
            )
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
                    # event-consumer: audit-only -- records a skip already enforced by the
                    # `candidates` filter above; the skip itself already happened
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
            (
                selected,
                skipped_issue_numbers,
                deferred_by_concurrency_full,
                deferred_by_concurrency_count,
            ) = _select_dispatch_candidates(
                candidates,
                dispatch_limit,
                state,
                self._branch_name,
                only_issues=only_issues,
            )
            # Issue #1005 review: _build_failure_map must see every deferred
            # issue (deferred_by_concurrency_full) so each one keeps its
            # per-issue "failures" entry -- only the persisted event and
            # CommandResult.data payloads truncate, via
            # deferred_by_concurrency below. Truncating before
            # _build_failure_map silently dropped failures entries for the
            # 6th+ deferred issue; caught in review before merge.
            deferred_by_concurrency = deferred_by_concurrency_full[
                :_MAX_DEFERRED_CONCURRENCY_EXAMPLES
            ]
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
        # Issue #1000: per-issue citation-drift verdicts and the fingerprint that
        # dedups the flag-comment across passes. Populated in the loop below;
        # consumed in the second state-lock section to stamp the issue record and
        # emit one ``dispatch_citation_drift_flagged`` event per drift change.
        citation_drift_stamps: dict[int, tuple[str, list[CitationVerdict]]] = {}
        # Issue #1010: pre-flight cross-repo gate. Issues whose referenced
        # file paths are all absent from the target repo are escalated to
        # human-needed instead of dispatching a worker that will wander to a
        # sibling repo's shared checkout.
        # Issue #1244: pre-flight cross-repo *scope* gate. Issues whose title
        # names another managed repo in the fleet (e.g. "other-repo: ...") are
        # escalated too — their deliverables live in that repo, not this one,
        # so the dispatching lane can never finalize them.  The managed-repo
        # set is derived from the fleet registry, never a hardcoded list.
        cross_repo_escalated: dict[int, CrossRepoGateResult] = {}
        fleet_repos = managed_repo_names(self.fleet_dir_override)
        dispatching_repo_name = _dispatching_repo_name(self.gh, self.repo_root)
        for issue_number in selected_issue_numbers:
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            branch_name = self._branch_name(full_issue)

            # Pre-flight gate: refuse to dispatch when the issue's referenced
            # code does not exist in this repo (issue #1010).
            gate_result = cross_repo_gate(str(full_issue.get("body") or ""), self.repo_root)
            if not gate_result.passed:
                cross_repo_escalated[issue_number] = gate_result
                continue

            # Pre-flight scope gate: refuse to dispatch when the issue's title
            # names another managed repo (issue #1244).
            scope_result = cross_repo_scope_gate(
                str(full_issue.get("title") or ""),
                str(full_issue.get("body") or ""),
                dispatching_repo_name,
                fleet_repos,
            )
            if not scope_result.passed:
                cross_repo_escalated[issue_number] = scope_result
                continue

            prompt_path = self._write_worker_prompt(full_issue)

            # Issue #1000: verify path:line citations in the issue body against
            # the working tree before a worker is sent to them. Drift is flagged
            # (a comment on the issue, which the worker sees via $issue_comments)
            # rather than auto-edited -- the correction needs judgment. The flag
            # is deduped by fingerprint so a still-stale issue is not re-commented
            # every pass, and a newly-stale citation re-alerts. Non-blocking: a
            # verification failure never aborts dispatch, and dispatch itself is
            # not gated on drift -- the comment is the signal, not a hold.
            citation_drift_stamps.update(
                self._check_issue_citations(issue_number, full_issue, previous_entries)
            )

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
        manifest_path = self._layout.session_manifest
        results_path = self._layout.session_results
        dispatch_results = dispatch_sessions(
            self.repo_root,
            manifest_path,
            results_path,
            self._adapter_settings(),
            session_requests,
        )
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
        manual = self.config.worker.harness == "manual"
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
                # Issues #837 / #779: the dispatch outcome (status/dispatched_at)
                # and its escalation bookkeeping (dispatch_failed_at /
                # escalation_reason / reason_class) used to be decided in two
                # textually separate if/elif chains keyed on the same four
                # predicates (ok / is_live_worker / is_phantom_live_worker /
                # else) -- one chain bound all_attempts/failed_result/
                # terminal_failure only in its `else`, a second chain ~70 lines
                # down read them only in its matching `elif`/`else`. That was
                # safe only because the two enumerations happened to agree;
                # nothing enforced it, and pyright could not prove it (reported
                # possibly-unbound). Collapsed into one chain below so each
                # branch binds and consumes its own locals -- there is no
                # second enumeration left to drift out of sync with the first.
                # status/dispatched_at are the only values that still cross out
                # of this chain (applied once, after it); every arm binds them
                # unconditionally, so a future arm that forgets to set one is a
                # possibly-unbound error at type-check time, not a silently
                # stale value copied from prev_entry.
                entry = {
                    **prev_entry,
                    "number": request.issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "branch_name": request.branch_name,
                    "prompt_path": str(request.prompt_path),
                }
                # Clear the claim timestamp on successful upgrade
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                if ok:
                    status = "manifest_written" if manual else "dispatched"
                    dispatched_at = utc_now()
                    # A successful recovery supersedes any previous orphan flag.
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                    entry.pop("dispatch_failed_at", None)
                    clear_escalation(entry)
                    clear_escalation_on_issue_prs(state, request.issue_number)
                elif is_live_worker:
                    status = "dispatched"
                    dispatched_at = prev_entry.get("dispatched_at") or utc_now()
                    # A live-worker recovery supersedes any previous orphan flag.
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                    entry.pop("dispatch_failed_at", None)
                    clear_escalation(entry)
                    clear_escalation_on_issue_prs(state, request.issue_number)
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
                    # A phantom live worker is being routed as dead; do not
                    # preserve a stale worker_pid that would keep the slot
                    # occupied, and do not burn a redispatch attempt (the launch
                    # was averted, not failed).
                    entry.pop("orphan_flagged_at", None)
                    entry.pop("orphan_drift_fingerprint", None)
                    entry.pop("orphan_drift_at", None)
                    entry.pop("dispatch_failed_at", None)
                    clear_escalation(entry)
                    clear_escalation_on_issue_prs(state, request.issue_number)
                    entry.pop("worker_pid", None)
                    entry.pop("worker_process_start_time", None)
                else:
                    # Issue #461: bound dispatch_failed retries with the same
                    # redispatch-window cap used for rework.
                    now = datetime.now(UTC)
                    failed_result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    failure_kind = (
                        failed_result.failure_kind if failed_result is not None else None
                    )
                    # Issue #1393: a pre-launch environment block (e.g.
                    # worktree_foreign_writer) never started a worker session,
                    # so it must NOT count against the dispatch_failed cap
                    # (which measures worker output, not environment hygiene).
                    # Use a separate blocked_environment_at counter and
                    # escalate with the correct reason + blocking path after
                    # the same cap.
                    blocked_environment = (
                        failure_kind in PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS
                    )
                    if blocked_environment:
                        blocked_environment_at = _windowed_blocked_environment_at(
                            entry,
                            window_minutes=self.config.watchdog.redispatch_window_minutes,
                        ) + [now.isoformat().replace("+00:00", "Z")]
                        blocking_error = failed_result.error if failed_result else None
                        entry["blocked_environment_at"] = blocked_environment_at
                        if len(blocked_environment_at) > self.config.watchdog.max_auto_redispatch:
                            # Issue #1423: before escalating a blocked-environment
                            # cap exhaustion for a foreign writer, attempt to reap
                            # it one more time. A writer that was active on earlier
                            # passes but has since gone idle is reaped here instead
                            # of escalating a zombie to a human. Escalation is
                            # reserved for a writer that is alive *and* active.
                            #
                            # Review finding: bound the number of auto-reaps per
                            # issue before falling back to escalation. Each
                            # successful reap resets ``blocked_environment_at`` to
                            # ``[]``, so without a separate cross-pass cap a
                            # persistently-blocked worktree loops forever between
                            # reap and redispatch. ``foreign_writer_reaps`` is a
                            # windowed counter that survives the reset; once it
                            # reaches ``max_foreign_writer_reaps`` the issue
                            # escalates instead of reaping again.
                            max_reaps = self.config.watchdog.max_foreign_writer_reaps
                            prior_reaps = _windowed_foreign_writer_reaps(
                                entry,
                                window_minutes=self.config.watchdog.redispatch_window_minutes,
                            )
                            if len(prior_reaps) < max_reaps and _try_reap_blocked_foreign_writer(
                                failed_result,
                                self.config,
                                self.paths.state_file,
                                request.issue_number,
                                sessions_dir,
                            ):
                                entry["blocked_environment_at"] = []
                                entry["foreign_writer_reaps"] = prior_reaps + [
                                    now.isoformat().replace("+00:00", "Z")
                                ]
                                status = "dispatch_failed"
                                dispatched_at = None
                                clear_escalation(entry)
                                clear_escalation_on_issue_prs(state, request.issue_number)
                                state = self._record_event(  # event-consumer: audit-only -- records a foreign-writer reap (issue #1423) already enforced by the blocked_environment_at reset and foreign_writer_reaps counter; consumed by tests/test_charlie_work.py regression tests.
                                    state,
                                    "dispatch_blocked_environment_reaped",
                                    {
                                        "issue_number": request.issue_number,
                                        "failure_kind": failure_kind,
                                        "pid": failed_result.pid if failed_result else None,
                                        "blocked_environment_count": len(blocked_environment_at),
                                        "foreign_writer_reap_count": len(prior_reaps) + 1,
                                    },
                                )
                            else:
                                status = "escalated"
                                dispatched_at = None
                                reason_class = "mechanical"
                                state = _escalate_issue(
                                    state,
                                    request.issue_number,
                                    reason="dispatch_blocked_environment",
                                    reason_class=reason_class,
                                    issue_extra=entry,
                                )
                                entry = dict(state["issues"][str(request.issue_number)])
                                state = self._record_event(
                                    state,
                                    "session_failed_escalated",
                                    {
                                        "issue_number": request.issue_number,
                                        "previous_status": "dispatch_pending",
                                        "reason": "dispatch_blocked_environment",
                                        "failure_kind": failure_kind,
                                        "blocking_error": blocking_error,
                                        "blocked_environment_count": len(blocked_environment_at),
                                    },
                                )
                        else:
                            status = "dispatch_failed"
                            dispatched_at = None
                            clear_escalation(entry)
                            clear_escalation_on_issue_prs(state, request.issue_number)
                            state = self._record_event(  # event-consumer: audit-only -- records a pre-launch environment block (issue #1393) already enforced by the blocked_environment_at counter and the dispatch_blocked_environment escalation; consumed by tests/test_charlie_work.py regression tests.
                                state,
                                "dispatch_blocked_environment",
                                {
                                    "issue_number": request.issue_number,
                                    "failure_kind": failure_kind,
                                    "blocking_error": blocking_error,
                                    "blocked_environment_count": len(blocked_environment_at),
                                },
                            )
                    else:
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
                        terminal_failure = (
                            failed_result is not None
                            and failed_result.failure_kind
                            in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                        )
                        # Issue #807: a deterministic judgment failure escalates
                        # immediately but as ``reason_class="judgment"``.
                        deterministic_judgment = (
                            failed_result is not None
                            and failed_result.failure_kind
                            in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
                        )
                        immediate_escalation = terminal_failure or deterministic_judgment
                        entry["dispatch_failed_at"] = all_attempts
                        if (
                            immediate_escalation
                            or len(recent) > self.config.watchdog.max_auto_redispatch
                        ):
                            status = "escalated"
                            dispatched_at = None
                            reason_class = "judgment" if deterministic_judgment else "mechanical"
                            state = _escalate_issue(
                                state,
                                request.issue_number,
                                reason=(
                                    failed_result.failure_kind
                                    if (
                                        immediate_escalation
                                        and failed_result is not None
                                        and failed_result.failure_kind is not None
                                    )
                                    else "dispatch_failed_cap_exceeded"
                                ),
                                reason_class=reason_class,
                                issue_extra=entry,
                            )
                            # Re-read the escalation fields _escalate_issue merged in,
                            # but keep ``entry`` a decoupled copy: the mutations below
                            # must not reach state until the single atomic write at the
                            # end of this block.
                            entry = dict(state["issues"][str(request.issue_number)])
                        else:
                            status = "dispatch_failed"
                            dispatched_at = None
                            clear_escalation(entry)
                            clear_escalation_on_issue_prs(state, request.issue_number)
                entry["status"] = status
                entry["dispatched_at"] = dispatched_at
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
                # Issue #1000: stamp the citation-drift fingerprint computed in
                # the outside-lock loop. The comment was already posted there
                # (best-effort); this persists the dedup marker so a still-stale
                # issue is not re-commented next pass, and emits one event per
                # drift change. ``citation_drift_flagged_at`` records the last
                # time drift was observed, not the first -- it moves with every
                # newly-detected drift state, matching the fingerprint's
                # re-alert semantics.
                drift_stamp = citation_drift_stamps.get(request.issue_number)
                if drift_stamp is not None:
                    fp, drift_verdicts = drift_stamp
                    entry["citation_drift_fingerprint"] = fp
                    if fp:
                        entry["citation_drift_flagged_at"] = utc_now()
                        state = self._record_event(
                            state,
                            "dispatch_citation_drift_flagged",
                            {
                                "issue": request.issue_number,
                                "drifted_citations": [
                                    {
                                        "citation": v.citation.raw,
                                        "path": v.citation.path,
                                        "line": v.citation.line,
                                        "end_line": v.citation.end_line,
                                        "status": v.status.value,
                                        "resolved_path": v.resolved_path.replace("\\", "/")
                                        if v.resolved_path
                                        else None,
                                        "candidates": list(v.candidates) if v.candidates else None,
                                    }
                                    for v in drift_verdicts
                                ],
                            },
                        )
                    else:
                        # Drift resolved since the last pass: clear the marker
                        # so a future regression re-alerts. No event -- the
                        # interesting transition is into drift, not out of it.
                        entry.pop("citation_drift_flagged_at", None)
                state["issues"][str(request.issue_number)] = entry
                # Persist the launched worker BEFORE touching GitHub labels: a
                # transient label-write failure (or crash) must never leave a live
                # worker unrecorded and therefore re-dispatchable next wave. The
                # transition is isolated per-issue so one failure never aborts the
                # rest of the batch (orphaning already-launched workers).
                save_state(self.paths.state_file, state)
                # This re-tests the same four predicates as the outcome chain
                # above for a fourth time (label transitions), rather than
                # branching on the outcome directly. It cannot be folded into
                # that chain: GitHub labels must only be touched after
                # save_state has persisted the launched worker (comment above),
                # so this enumeration is structurally separated from the one
                # that decides status/escalation. It does not read
                # all_attempts/failed_result/terminal_failure (issues #837,
                # #779 do not apply here), only the already-bound ok /
                # is_live_worker / status locals, so there is no possibly-
                # unbound hazard -- just a fourth place that must be kept in
                # sync with the outcome predicates if a new outcome is added.
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
                    # Issue #807: a deterministic judgment failure uses
                    # ``reason_class="judgment"`` so the label lands on
                    # human-needed, not operator_queued.
                    edge = _escalation_edge("redispatch_escalated", reason_class)
                    result = transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        edge,
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": edge,
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

            # Issue #1010: escalate issues blocked by the cross-repo pre-flight
            # gate. Their referenced file paths are all absent from the target
            # repo, so dispatching a worker would send it to a sibling repo's
            # shared checkout. Escalate to human-needed with a cross_repo_target
            # reason and record the event — the issue stays in the dispatch
            # pool's state as escalated, not dispatch_pending.
            for issue_number, gate_result in sorted(cross_repo_escalated.items()):
                reason = gate_result.reason
                prev_entry = state["issues"].get(str(issue_number), {})
                entry = {
                    **prev_entry,
                    "number": issue_number,
                    "title": full_issues.get(issue_number, {}).get("title"),
                    "url": full_issues.get(issue_number, {}).get("url"),
                }
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                state = _escalate_issue(
                    state,
                    issue_number,
                    reason=reason,
                    reason_class="mechanical",
                    issue_extra=entry,
                )
                # Issue #1583: report ``neutral_paths`` and ``missing_paths``
                # in the event payload so the operator can see from the
                # event alone which citation tripped the gate (today the
                # payload carried only the count, embedded in ``reason``).
                # For the cross-repo *scope* gate (issue #1244) both tuples
                # are empty by construction -- the scope gate does not deal
                # in file paths -- so the fields are present but empty,
                # which is accurate.
                state = append_event(
                    state,
                    "dispatch_cross_repo_escalated",
                    {
                        "issue_number": issue_number,
                        "reason": reason,
                        "neutral_paths": list(gate_result.neutral_paths),
                        "missing_paths": list(gate_result.missing_paths),
                    },
                    state_path=self.paths.state_file,
                )
                save_state(self.paths.state_file, state)
                # Transition labels (operator_queue for mechanical, following
                # the same pattern as the redispatch_escalated path above).
                edge = _escalation_edge("redispatch_escalated", "mechanical")
                result = transition(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    edge,
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": edge,
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
                    escalated_entry = state["issues"].get(str(issue_number), {})
                    escalated_entry["label_error"] = label_error
                    state["issues"][str(issue_number)] = escalated_entry
                    label_errors.append(issue_number)
                    label_error_failures[issue_number] = _label_error_reason(label_error)
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
                deferred_by_concurrency_full,
                dispatch_limit,
                extra_failures=label_error_failures,
            )
            # Issue #946: warn when a non-empty backlog has not produced a
            # non-empty dispatch event for longer than the configured threshold.
            dispatch_staleness = check_dispatch_staleness(
                self.paths.state_file,
                self.config.dispatch,
                backlog_reachability,
                recent_issue_numbers=sorted(successful_issue_numbers),
                now=datetime.now(UTC),
            )
            if dispatch_staleness["stale"]:
                state = self._record_event(state, "dispatch_stale", dispatch_staleness)
            state = append_event(
                state,
                "dispatch",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "live_worker_issue_numbers": sorted(live_worker_issue_numbers),
                    "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "foreign_writer_issue_numbers": sorted(foreign_writer_issue_numbers),
                    "cross_repo_escalated_issue_numbers": sorted(cross_repo_escalated),
                    "label_errors": sorted(label_errors),
                    "skipped_issue_numbers": skipped_issue_numbers,
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "deferred_by_concurrency_count": deferred_by_concurrency_count,
                    "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                    "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
                    "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
                    "merged_pr_mention_rearmed_issue_numbers": sorted(
                        newly_rearmed_mention_issues
                    ),
                    "failures": dispatch_failure_map,
                    # Issue #944: why zero, when it is zero. Every other field
                    # here describes issues the ready-filtered query returned;
                    # this one describes the backlog that query cannot see.
                    "backlog_reachability": backlog_reachability,
                    # Issue #946: cadence-staleness diagnostic, always present so
                    # the capped state.json ring carries the signal.
                    "dispatch_staleness": dispatch_staleness,
                    # Issue #1005: the capacity axis. backlog_reachability answers
                    # "why zero" for supply (which issues exist/are reachable);
                    # this answers it for capacity (whether there was a slot to put
                    # one in). Always present -- gov.report_fields() is safe to call
                    # unclamped -- and explicit about `clamped` so a reader does not
                    # have to redo the arithmetic (available_slots == 0 alone does
                    # not say whether the repo cap or the fleet cap was binding).
                    # `dispatch_limit` is included explicitly (report_fields() does
                    # not carry it) because it is the only field that reflects a
                    # fleet-cap clamp: `available_slots` is only recomputed when the
                    # repo governor itself is enabled, so a fleet-only clamp leaves
                    # `available_slots` at its pre-fleet value while `dispatch_limit`
                    # still shows the true (possibly zero) effective limit.
                    "concurrency_governor": {
                        "clamped": gov.clamped,
                        "dispatch_limit": gov.dispatch_limit,
                        **gov.report_fields(),
                    },
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
        if cross_repo_escalated:
            message += f" (cross-repo escalated: {sorted(cross_repo_escalated)})"
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "failures": dispatch_failure_map,
            "live_worker_count": len(live_worker_issue_numbers),
            "phantom_live_worker_count": len(phantom_live_worker_issue_numbers),
            "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
            "foreign_writer_count": len(foreign_writer_issue_numbers),
            "cross_repo_escalated_issue_numbers": sorted(cross_repo_escalated),
            "skipped_issue_numbers": skipped_issue_numbers,
            "deferred_by_concurrency": deferred_by_concurrency,
            "deferred_by_concurrency_count": deferred_by_concurrency_count,
            "merged_prs": resolved_merged_prs,
            "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
            "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
            "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
            "merged_pr_mention_rearmed_issue_numbers": sorted(newly_rearmed_mention_issues),
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
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
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
    def review(
        self,
        pr_number: int,
    ) -> CommandResult:
        """Generate a review packet for a PR.

        When config.test_adequacy.enabled, this method may itself issue a
        request_changes verdict and advance/terminate the rework loop (previously
        only record_review/verdict did this). When disabled (default), the method
        never mutates decision state and always returns a packet.

        Args:
            pr_number: The PR number to review.

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
            branch_issue_validator=self._make_branch_issue_validator(),
        )

        # Issue #617: dry-run must short-circuit before the escalation check,
        # the packet writes (pr.json, checks.json, diff.patch, review-prompt.md,
        # review-decision.json), and the rework-budget counter resets
        # (review_dispatch_attempt_count=0, no_op_rework_attempts=0,
        # status="reviewing") -- all of which mutate state.json or packet files
        # and were reachable because review() had no top-level dry_run gate (the
        # flag only reached the nested _cross_family_for_pr call). Mirror
        # _dispatch_impl's top-of-function dry-run short-circuit: return the
        # computed plan, touch nothing. A preview that overwrites a terminal
        # verdict with "pending" or silently extends a PR's retry budget is
        # worse than the bug it replaces, so this gate is a single early return
        # before any branch that has an escalation or state-write arm.
        if self.dry_run:
            return CommandResult(
                True,
                f"dry-run: would generate review packet for PR #{pr_number}",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "dry_run": True,
                    "checks_unavailable": False,
                },
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
        issue_state = (
            state.get("issues", {}).get(str(issue_number), {})
            if issue_number is not None
            else None
        )
        pr_escalated, issue_escalated = _escalation_flags(pr_state, issue_state)
        if pr_escalated or issue_escalated:
            reason = (
                f"issue #{issue_number} is escalated; review skipped"
                if issue_number is not None and issue_escalated
                else f"PR #{pr_number} is escalated; review skipped"
            )
            # Refresh (but never act on) janitor diagnostics while escalated.
            # PRs #1397/#1443 (a sibling repo, 2026-07-27) sat with janitor_ok/
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
            escalated_verdict = None
            known_ci_run_never_created_head = None
            with state_lock(self.paths.state_file):
                fresh_state = load_state(self.paths.state_file)
                existing_pr_state = fresh_state["prs"].get(str(pr_number))
                if existing_pr_state is not None:
                    known_ci_run_never_created_head = existing_pr_state.get(
                        "ci_run_never_created_head"
                    )
                    escalated_verdict = run_janitor(
                        pr,
                        escalated_checks,
                        self.config,
                        pr_state=existing_pr_state,
                        repo_root=self.repo_root,
                        pr_diff=escalated_diff,
                        review_decision=self._review_decision(pr_number),
                    )
                    failures_changed = existing_pr_state.get("janitor_failures") != list(
                        escalated_verdict.failures
                    )
                    fresh_state["prs"][str(pr_number)] = {
                        **existing_pr_state,
                        "janitor_ok": escalated_verdict.ok,
                        "janitor_failures": list(escalated_verdict.failures),
                        "is_missing_checks_only_block": escalated_verdict.is_missing_checks_only_block,
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

            # A sibling repo, 2026-08-06/07: GitHub Actions silently created no
            # workflow run for pushed heads; detection added so the janitor
            # gate distinguishes "CI never started" from "CI failed". An
            # escalated PR returns from this early branch before the
            # non-escalated janitor-gate block below ever runs -- and a PR
            # stuck 4+ days behind a missing-checks failure (the exact
            # condition this detects) is itself a strong escalation
            # candidate, so the detector must be reachable from here too, not
            # only the non-escalated path. Deliberately called OUTSIDE the
            # state_lock above: it makes a `gh api` call, and the state lock
            # must never span external I/O (mopup-rate-limit-lock-contention)
            # -- a stalled/rate-limited call would otherwise hold the lock
            # against every other writer. `known_ci_run_never_created_head`
            # short-circuits the call once this exact head has already been
            # recorded, so a PR that stays escalated for days doesn't re-query
            # Actions every single pass forever.
            ci_run_never_created_head_sha = None
            if escalated_verdict is not None:
                ci_run_never_created_head_sha = self._detect_ci_run_never_created(
                    pr,
                    escalated_verdict,
                    known_head=known_ci_run_never_created_head,
                )
            # `escalated_verdict is not None` is runtime-redundant here (the
            # only way ci_run_never_created_head_sha is non-None is if it was
            # set above and that branch requires escalated_verdict to be set
            # too) but Pyright's narrowing doesn't carry across the two `if`
            # blocks, so escalated_verdict.missing_required_checks below reads
            # as reportOptionalMemberAccess without it. Spelling out the
            # correlation documents why the two are never non-None one
            # without the other.
            if ci_run_never_created_head_sha is not None and escalated_verdict is not None:
                with state_lock(self.paths.state_file):
                    fresh_state = load_state(self.paths.state_file)
                    existing_pr_state = fresh_state["prs"].get(str(pr_number))
                    if existing_pr_state is not None and (
                        existing_pr_state.get("ci_run_never_created_head")
                        != ci_run_never_created_head_sha
                    ):
                        fresh_state["prs"][str(pr_number)] = {
                            **existing_pr_state,
                            "ci_run_never_created_head": ci_run_never_created_head_sha,
                        }
                        fresh_state = self._record_event(
                            fresh_state,
                            "ci_run_never_created",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "head_sha": ci_run_never_created_head_sha,
                                "branch": pr.get("headRefName"),
                                "missing_checks": list(escalated_verdict.missing_required_checks),
                                "escalated": True,
                            },
                        )
                        save_state(self.paths.state_file, fresh_state)

            # Issue #776: escalation must stay terminal for the reason that
            # caused it, but must not become a one-way door that ALSO blocks
            # an unrelated, independently-capped remediation -- e.g. a dead
            # request-changes-fix worker exhausting the unrelated watchdog
            # redispatch cap used to permanently wall off a PR that
            # separately developed a merge conflict, with no path back short
            # of a human running `charlie unescalate` (real corpus: issues
            # #592/#648/#606). A merge-conflict or no-op-rework janitor
            # failure is routed through the SAME attempts_key-capped wrapper
            # the non-escalated janitor gate below uses
            # (_route_janitor_gate_failure_to_rework); that wrapper's own
            # rework_pending check does not treat "escalated" as in-flight,
            # so it re-derives its own dispatch/escalate decision from
            # conflict_rework_attempts/no_op_rework_attempts -- a cap that is
            # untouched by (and therefore still fresh after) an escalation
            # from a different lane, but already exhausted -- so still
            # correctly refused -- if THIS lane is what escalated the issue
            # previously. This never re-litigates the original verdict or
            # reason for escalation: only these two deterministic,
            # zero-token checks are attempted here; packet generation and the
            # LLM reviewer remain skipped below.
            if issue_number is not None and escalated_verdict is not None:
                is_merge_conflict_block = str(
                    pr.get("mergeable") or ""
                ).upper() == "CONFLICTING" or (
                    str(pr.get("mergeStateStatus") or "").upper() == "DIRTY"
                )
                # Issue #1111: mirror the non-escalated path's stale-CI-verdict
                # suppression — a request_changes verdict citing only required
                # checks that are green now must not burn no_op_rework_attempts
                # (the predicate itself refuses escalated decisions, so this is
                # a no-op for verdicts recorded as escalated).
                escalated_stale_ci = False
                if (
                    escalated_verdict.is_no_op_rework
                    and not escalated_verdict.failed_required_checks
                ):
                    required = self.config.auto_merge.required_checks
                    escalated_stale_ci = (
                        bool(required)
                        and escalated_checks is not None
                        and is_stale_ci_verdict(
                            self._review_decision(pr_number),
                            summarize_checks(escalated_checks, required),
                        )
                    )
                is_no_op_rework_block = (
                    escalated_verdict.is_no_op_rework
                    and not escalated_verdict.failed_required_checks
                    and not escalated_stale_ci
                )
                if is_merge_conflict_block or is_no_op_rework_block:
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

            return CommandResult(
                True,
                reason,
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "pass_skipped": True,
                    "checks_unavailable": escalated_checks is None,
                },
            )

        issue = self.gh.issue_view(issue_number) if issue_number is not None else {}
        checks = self.gh.pr_checks(pr_number)
        # Issue #1383: reclassify fleet-wide infra failures (Actions budget /
        # runner outage) as INFRA_BLOCKED at the data boundary, BEFORE the
        # janitor gate / rework-routing decision. A budget-failed check has a
        # FAILURE conclusion (not CANCELLED/INFRA_FAILURE), so without this
        # enrichment it lands in summary.failed and is routed to rework --
        # burning no-op rework caps on a healthy PR. Enriching here is the
        # single point of enforcement; the janitor and merge_ready both see
        # the reclassified checks.
        checks_unavailable = checks is None
        if not checks_unavailable:
            checks = self._enrich_checks_infra_blocked(
                checks, self.config.auto_merge.required_checks
            )

        # Load PR state for no-op rework detection (only if PR has verdict history)
        pr_state = None
        if str(pr_number) in state.get("prs", {}):
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})

        # Fetch diff for patch-id based no-op rework detection (issue #222)
        # This is needed before the janitor gate to detect actual content changes
        diff = self.gh.pr_diff(pr_number)

        # Deterministic janitor gate BEFORE any packet spend: an
        # obviously-not-ready PR (draft, conflicting, red CI, no issue link)
        # must cost zero review tokens. Most failures don't move labels — they
        # are the worker's/CI's to fix. A definitive required-check failure on
        # a linked-issue PR is routed to rework so the worker can push a fix.
        # Issue #1598: pass the bound issue's live labels so the janitor can
        # surface a human-merge-label warning in the verdict. Only fetched
        # when human_merge_labels is configured (default empty → skipped).
        _hm_issue_labels: set[str] | None = None
        if self.config.dispatch.human_merge_labels:
            _hm_issue_num = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if _hm_issue_num is not None:
                try:
                    _hm_issue_labels = label_names(self.gh.issue_view(_hm_issue_num))
                except (GitHubError, ValueError):
                    _hm_issue_labels = None
        verdict = run_janitor(
            pr,
            checks,
            self.config,
            pr_state=pr_state,
            repo_root=self.repo_root,
            pr_diff=diff,
            review_decision=self._review_decision(pr_number),
            issue_labels=_hm_issue_labels,
        )

        # Issue #1116: the stale-CI skip let a reworked-but-unchanged PR
        # through the gate it was permanently wedged behind. Record it so the
        # packet-rebuild -> fresh-review sequence that follows is attributable
        # to the skip rather than looking like a spontaneous unblock. Dedup on
        # the head sha (mirroring the failures_changed / draft_hold_reason
        # pattern elsewhere in this function): the gate re-passes on every
        # poll while the PR waits on review-dispatch capacity, and only the
        # first pass per head is signal (cost-spirals.md Finding 2).
        if verdict.ok and verdict.no_op_check_skipped_stale_ci and not self.dry_run:
            gate_pass_head = str(pr.get("headRefOid") or "") or None
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                existing_pr_state = state["prs"].get(str(pr_number), {})
                if existing_pr_state.get("stale_ci_gate_pass_head") != gate_pass_head:
                    state["prs"][str(pr_number)] = {
                        **existing_pr_state,
                        "stale_ci_gate_pass_head": gate_pass_head,
                    }
                    state = self._record_event(
                        state,
                        "stale_ci_verdict_gate_pass",
                        {"pr_number": pr_number, "head_sha": gate_pass_head},
                    )
                    save_state(self.paths.state_file, state)

        # Issue #820: reconcile the operator merge-hold marker for the #818
        # draft-auto-ready actuator unconditionally, on every review() pass
        # for this PR -- not only when we're about to act on it below --
        # so the stored reason can never go stale from a *different*
        # branch (packet success further down, a co-occurring second
        # failure routed to the general janitor_blocked path, a merge-
        # conflict/no-op-rework/flake-rerun branch, ...) that doesn't
        # revisit the held branch and therefore wouldn't otherwise touch
        # this field. A marker only cleared by its own branch needs a
        # clear-site at every other exit from review() -- miss one and the
        # dedup below silently stops re-firing after a lift/re-park cycle
        # that passed through one of those other branches in between.
        # Reconciling here once, unconditionally, is the single point of
        # truth instead.
        #
        # Semantics mirror merge_ready's mergequeue-handoff hold check:
        # PR-label-or-issue-label, with an unavailable/degraded check
        # failing safe (treated the same as "held", never as "no hold").
        # `issue` was already fetched unconditionally above for packet
        # building, so this adds no extra `gh` calls.
        merge_hold = self.config.labels.merge_hold in label_names(pr)
        merge_hold_check_unavailable = False
        if not merge_hold and issue_number is not None:
            if not isinstance(issue, dict) or "labels" not in issue:
                merge_hold_check_unavailable = True
            else:
                merge_hold = self.config.labels.merge_hold in label_names(issue)
        draft_hold_reason = (
            ("operator_merge_hold" if merge_hold else "merge_hold_check_unavailable")
            if verdict.is_draft_only_block and (merge_hold or merge_hold_check_unavailable)
            else None
        )
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing_pr_state = state["prs"].get(str(pr_number), {})
            draft_hold_reason_changed = (
                existing_pr_state.get("draft_ready_held_reason") != draft_hold_reason
            )
            if draft_hold_reason_changed:
                state["prs"][str(pr_number)] = {
                    **existing_pr_state,
                    "draft_ready_held_reason": draft_hold_reason,
                }
                save_state(self.paths.state_file, state)

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

            # Auto-ready otherwise-mergeable draft PRs (issue #818): a draft
            # is an externally-imposed state like any other the fleet
            # reconciles (base movement, flaky checks, ...). When draft is
            # the ONLY janitor failure -- every other gate already passed --
            # call `gh pr ready` and defer the actual review to the next
            # poll, mirroring the flake-rerun debounce below: trigger a
            # cheap, idempotent, self-correcting actuator rather than
            # mutating this frozen verdict and continuing same-pass. Errors
            # come back as values (GitHubRunResult.ok/.error), never
            # exceptions, so a failed `gh pr ready` must NOT be treated as
            # success -- the PR stays janitor_blocked and is not routed
            # toward merge.
            if verdict.is_draft_only_block:
                # Issue #820: an operator can park a PR by drafting it, which
                # is indistinguishable from the externally-imposed draft state
                # #818 was written to auto-clear. `draft_hold_reason` /
                # `draft_hold_reason_changed` were already computed and
                # reconciled into state unconditionally above (single point
                # of truth -- see the comment there for why a branch-local
                # clear was insufficient); this only decides whether to act
                # on the already-current value.
                if draft_hold_reason is not None:
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        existing_pr_state = state["prs"].get(str(pr_number), {})
                        state["prs"][str(pr_number)] = {
                            **existing_pr_state,
                            "number": pr_number,
                            "issue_number": issue_number,
                            "status": "janitor_blocked",
                            "janitor_ok": False,
                            "janitor_failures": list(verdict.failures),
                            "is_missing_checks_only_block": verdict.is_missing_checks_only_block,
                        }
                        if draft_hold_reason_changed:
                            state = self._record_event(
                                state,
                                "draft_pr_ready_held",
                                {
                                    "pr_number": pr_number,
                                    "issue_number": issue_number,
                                    "reason": draft_hold_reason,
                                },
                            )
                        save_state(self.paths.state_file, state)
                    return CommandResult(
                        False,
                        f"PR #{pr_number} is a draft; auto-ready held ({draft_hold_reason})",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "draft_ready_held": True,
                            "draft_ready_held_reason": draft_hold_reason,
                            "checks_unavailable": checks is None,
                        },
                    )

                ready_result = self.gh.pr_ready(pr_number)
                if ready_result.ok:
                    log_event(
                        self.paths.state_file,
                        "draft_pr_ready_triggered",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "head_sha": pr.get("headRefOid"),
                        },
                        repo=self.repo_root.name,
                    )
                    return CommandResult(
                        False,
                        f"PR #{pr_number} was a draft; marked ready for review "
                        "(deferring to next pass)",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "draft_readied": True,
                            "checks_unavailable": checks is None,
                        },
                    )
                # `gh pr ready` failed: park as janitor_blocked (same as the
                # pre-fix behavior) plus a dedicated error field so repeated
                # identical failures dedupe instead of re-firing the event
                # every pass -- the underlying `verdict.failures` is constant
                # ("PR is a draft") regardless of the actuator's outcome, so
                # the existing failures_changed dedup below would never fire
                # on its own for this class of park.
                draft_ready_error = (
                    ready_result.error or f"gh pr ready exited {ready_result.returncode}"
                )
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    existing_pr_state = state["prs"].get(str(pr_number), {})
                    error_changed = existing_pr_state.get("draft_ready_error") != draft_ready_error
                    state["prs"][str(pr_number)] = {
                        **existing_pr_state,
                        "number": pr_number,
                        "issue_number": issue_number,
                        "status": "janitor_blocked",
                        "janitor_ok": False,
                        "janitor_failures": list(verdict.failures),
                        "draft_ready_error": draft_ready_error,
                        "is_missing_checks_only_block": verdict.is_missing_checks_only_block,
                    }
                    if error_changed:
                        state = append_event(
                            state,
                            "draft_pr_ready_failed",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "error": draft_ready_error,
                            },
                            state_path=self.paths.state_file,
                        )
                    save_state(self.paths.state_file, state)
                return CommandResult(
                    False,
                    f"PR #{pr_number} is a draft and `gh pr ready` failed: {draft_ready_error}",
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "draft_ready_failed": True,
                        "checks_unavailable": checks is None,
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
                            # event-consumer: audit-only -- records a successfully-triggered
                            # rerun (the action already happened); flake_rerun_failed/escalated
                            # are the actionable siblings and are separately consumed
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

                # Issue #992: a refusal because the containing run is still in
                # progress is a retry-later signal, not a reason to route the
                # PR to rework. The attempt stays unconsumed (we did not write
                # check_rerun_attempts above), so a later pass can retry once
                # the run completes.
                if (
                    not triggered_run_ids
                    and rerun_errors
                    and all(_is_rerun_already_running_error(e) for e in rerun_errors)
                ):
                    return CommandResult(
                        False,
                        f"flake rerun for PR #{pr_number} refused: "
                        + "workflow run(s) still in progress",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "rerun_run_ids": list(verdict.rerun_run_ids),
                            "rerun_errors": rerun_errors,
                            "already_running": True,
                            "checks_unavailable": checks is None,
                        },
                    )

            # Infra-failure auto-rerun + escalation (issue #841): a job-level
            # `timeout-minutes` kill is an infra failure (CANCELLED on the
            # self-hosted-era runners, possibly TIMED_OUT on hosted runners),
            # which summarize_checks correctly buckets into infra_failed
            # (blocks merge via CheckSummary.ready) via _classify_check_run --
            # both conclusions route to the infra bucket, so the rerun path
            # matches regardless of which runner reports. But nothing before
            # this retried or escalated it -- classify_check_failures only
            # iterates summary.failed (a code push can't fix an infra kill), so
            # an infra-failed PR sat blocked forever behind only a diagnostic
            # merge_failed_attempt_alarm event. `gh run rerun RUN_ID` is
            # dispatched WITHOUT --failed: the job never completed
            # (cancelled/timed out, not failed), so
            # --failed's "rerun the failed jobs in this run" semantics do not
            # apply -- omitting it reruns the whole run, which is the correct
            # behavior for a run that never produced a completed job to target.
            if verdict.infra_rerun_run_ids:
                infra_rerun_errors: list[str] = []
                infra_triggered_run_ids: list[int] = []
                for run_id in verdict.infra_rerun_run_ids:
                    result = self.gh.run(["run", "rerun", str(run_id)], allow_failure=True)
                    if isinstance(result, GitHubRunResult):
                        if result.ok:
                            infra_triggered_run_ids.append(run_id)
                        else:
                            infra_rerun_errors.append(
                                result.error or f"gh run rerun {run_id} exited {result.returncode}"
                            )
                    elif isinstance(result, str):
                        # Dry-run returns a descriptive string; treat as success.
                        infra_triggered_run_ids.append(run_id)
                    else:
                        infra_rerun_errors.append(
                            f"unexpected result from gh run rerun {run_id}: {result!r}"
                        )

                if infra_triggered_run_ids and not infra_rerun_errors:
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)] = {
                            **state["prs"].get(str(pr_number), {}),
                            "number": pr_number,
                            "issue_number": issue_number,
                            "infra_rerun_attempts": verdict.infra_rerun_attempts,
                        }
                        state = append_event(
                            state,
                            "infra_rerun_triggered",
                            {
                                "pr_number": pr_number,
                                "run_ids": infra_triggered_run_ids,
                                "head_sha": pr.get("headRefOid"),
                            },
                            state_path=self.paths.state_file,
                        )
                        save_state(self.paths.state_file, state)
                    return CommandResult(
                        False,
                        f"infra rerun triggered for PR #{pr_number}: run(s) "
                        + ", ".join(str(rid) for rid in infra_triggered_run_ids),
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "infra_rerun_run_ids": infra_triggered_run_ids,
                            "checks_unavailable": checks is None,
                        },
                    )

                # Rerun API error: record it, but do not consume the attempt.
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state = append_event(
                        state,
                        "infra_rerun_failed",
                        {
                            "pr_number": pr_number,
                            "run_ids": list(verdict.infra_rerun_run_ids),
                            "errors": infra_rerun_errors,
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)

            if (
                issue_number is not None
                and verdict.is_infra_failure_block
                and verdict.infra_definitive_failed
            ):
                # Attempt cap exhausted (or no parseable run id at all): there
                # is no code-fix rework path for an infra failure, so escalate
                # straight to a human instead of looping forever on a PR that
                # can never clear the gate on its own -- this is the bug
                # issue #841 fixes (previously: a diagnostic
                # merge_failed_attempt_alarm event and nothing else).
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state = _escalate_issue(
                        state,
                        issue_number,
                        reason="infra_rerun_cap_exceeded",
                        reason_class="mechanical",
                        pr_number=pr_number,
                        pr_extra={"infra_rerun_attempts": verdict.infra_rerun_attempts},
                    )
                    state = append_event(
                        state,
                        "infra_rerun_escalated",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "checks": list(verdict.infra_definitive_failed),
                        },
                        state_path=self.paths.state_file,
                    )
                    save_state(self.paths.state_file, state)
                edge = _escalation_edge("escalated", "mechanical")
                result = transition(self.gh, self.config.labels, issue_number, edge)
                label_error = None
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": edge,
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
                return CommandResult(
                    False,
                    f"PR #{pr_number} infra-failed check(s) exhausted rerun cap: "
                    + ", ".join(verdict.infra_definitive_failed),
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "infra_escalated": True,
                        "label_error": label_error,
                    },
                )

            # Issue #1383: infra_blocked required checks (fleet-wide Actions
            # budget/runner outage) are held without dispatching rework and
            # without burning rework/no-op attempt counters. The PR is not
            # transitioned to review_started (there is nothing for the worker
            # to fix), no request_changes verdict is recorded, and the
            # check-failure rework route below is skipped entirely. A distinct
            # ``check_infra_blocked`` warning event is emitted per affected PR
            # so the heartbeat consumer (AC4) and the cross-pass escalation
            # tracker can see it. The operator-facing
            # ``infra_blocked_escalated`` error event is emitted at most once
            # per ``escalation_window_minutes`` window across ALL affected PRs
            # (AC3), not once per PR per pass -- tracked in the module-level
            # ``_infra_blocked_window`` dict because the app instance is
            # rebuilt per pass.
            if verdict.is_infra_blocked_block:
                log_event(
                    self.paths.state_file,
                    "check_infra_blocked",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "checks": list(verdict.infra_blocked_checks),
                    },
                    repo=self.repo_root.name,
                    level="warning",
                )
                # Cross-pass persistence tracking + windowed escalation.
                cfg = self.config.auto_merge.infra_blocked
                repo_key = str(self.repo_root)
                window = _infra_blocked_window.setdefault(
                    repo_key,
                    {
                        "consecutive_passes": 0,
                        "last_escalation": None,
                        "last_pass_cid": None,
                    },
                )
                # Issue #1383: increment at most once per loop pass (per
                # correlation_id), not once per infra-blocked PR encountered
                # within review(). Without this gate, N concurrent
                # infra-blocked PRs in one pass would reach
                # persistence_passes=N in a single pass and fire
                # ``infra_blocked_escalated`` immediately, contradicting AC3
                # ("persistence across N passes, not per PR per pass"). The
                # same correlation_id-dedup pattern is used by the
                # reset-on-clear logic in ``_loop_impl``. When no correlation
                # context is active (direct ``review()`` calls outside
                # ``loop()``, e.g. in tests), ``cid`` is None and each call
                # increments -- matching the pre-fix direct-call behavior.
                cid = current_correlation_id()
                if cid is None or window.get("last_pass_cid") != cid:
                    window["consecutive_passes"] = window.get("consecutive_passes", 0) + 1
                    window["last_pass_cid"] = cid
                last_esc = window.get("last_escalation")
                now_dt = datetime.now(UTC)
                should_escalate = window["consecutive_passes"] >= cfg.persistence_passes
                if should_escalate and (
                    last_esc is None
                    or (now_dt - last_esc).total_seconds() >= cfg.escalation_window_minutes * 60
                ):
                    log_event(
                        self.paths.state_file,
                        "infra_blocked_escalated",
                        {
                            "consecutive_passes": window["consecutive_passes"],
                            "persistence_passes": cfg.persistence_passes,
                            "window_minutes": cfg.escalation_window_minutes,
                        },
                        repo=self.repo_root.name,
                        level="error",
                    )
                    window["last_escalation"] = now_dt
                return CommandResult(
                    False,
                    f"PR #{pr_number} infra-blocked (billing/runner outage): "
                    + ", ".join(verdict.infra_blocked_checks),
                    {
                        "infra_blocked": True,
                        "checks": list(verdict.infra_blocked_checks),
                        "checks_unavailable": False,
                    },
                )

            if issue_number is not None and verdict.is_check_failure_block:
                transition(self.gh, self.config.labels, issue_number, "review_started")
                summary = f"CI failed on {', '.join(verdict.failed_required_checks)}; push a fix"
                # Issue #771: name the failure, not just the check. Populated
                # from the failing check run(s)' GitHub annotations when
                # available, falling back to the check's own run link when
                # there are no annotations; degrades all the way to [] only
                # when GitHub gave us neither, in which case record_review's
                # summary-only fallback still renders the message above via
                # _render_required_changes_section.
                required_changes = _required_changes_from_checks(
                    checks, verdict.failed_required_checks, self.gh.check_run_annotations
                )
                # Issue #1258: dedicated provenance for the CI-red
                # short-circuit -- record_review() itself only logs a
                # decision-agnostic "record_review" event, with nothing
                # naming this as the deterministic janitor gate rather than
                # a human/LLM verdict. Additive: the short-circuit's own
                # return/routing above is unchanged.
                log_event(
                    self.paths.state_file,
                    "review_dispatch_skipped_ci_red",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "failed_required_checks": list(verdict.failed_required_checks),
                        "co_occurring_failures": [],
                        "co_occurring": False,
                    },
                    repo=self.repo_root.name,
                )
                return self.record_review(
                    pr_number,
                    "request_changes",
                    summary=summary,
                    reviewed_head=pr.get("headRefOid"),
                    required_changes=required_changes,
                    verdict_provenance="ci_gate_auto_reject",
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
            #
            # Issue #1111: ALSO excluded is the stale-CI-verdict case — the
            # request_changes verdict's only findings cite required checks
            # that are all green right now (a transient failure the reviewer
            # saw has recovered on the same content). Routing rework there is
            # a guaranteed no-op that burns no_op_rework_attempts toward a
            # manufactured human escalation; review_queue() re-queues the PR
            # for a fresh review instead, so this pass just waits.
            stale_ci_verdict = False
            if verdict.is_no_op_rework and not verdict.failed_required_checks:
                required = self.config.auto_merge.required_checks
                stale_ci_verdict = (
                    bool(required)
                    and checks is not None
                    and is_stale_ci_verdict(
                        self._review_decision(pr_number),
                        summarize_checks(checks, required),
                    )
                )
            is_no_op_rework_block = (
                verdict.is_no_op_rework
                and not verdict.failed_required_checks
                and not stale_ci_verdict
            )
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

            # Issue #1258: a required check can fail alongside a janitor
            # failure that is neither the sole failure (is_check_failure_block
            # above already routes that) nor a bare merge conflict (already
            # routed above via _route_janitor_gate_failure_to_rework). That
            # combination used to fall straight through to the passive
            # janitor_blocked bookkeeping below -- neither reviewed nor
            # routed to rework, silently re-logging the same failure set
            # forever (the same "zero readers" dead end issue #765/#776 fixed
            # for the merge-conflict/no-op-rework cases). Route it through
            # the same record_review(request_changes) machinery the
            # sole-failure short-circuit above uses, decision-agnostic, with
            # required_changes/summary naming BOTH the failing check(s) and
            # the co-occurring janitor failure(s) so a rework worker sees the
            # whole picture in one pass.
            #
            # is_merge_conflict_block is excluded: without it, a
            # conflict+red-check PR whose merge-conflict rework is already
            # pending (routed returned None just above) would ALSO get
            # routed through here -- a double-dispatch regression, not a
            # fix.
            #
            # verdict.is_no_op_rework (the raw flag, NOT is_no_op_rework_block)
            # is also excluded, and this exclusion is load-bearing, not
            # redundant with is_no_op_rework_block above: is_no_op_rework_block
            # itself structurally requires `not verdict.failed_required_checks`
            # and so can never be True here -- but the underlying
            # `verdict.is_no_op_rework` flag CAN still be True with a
            # co-occurring required-check failure (same diff pushed again
            # while CI is still red), and that exact combination is
            # test_janitor_required_check_failure_noop_does_not_reroute's
            # issue #376 invariant: re-requesting a rework whose only signal
            # is "same diff as last time" is not productive while CI is
            # still red, and changing that is out of this fix's scope. A
            # PR in that state still stalls in janitor_blocked below after
            # this diff -- an accepted carve-out, not an oversight.
            #
            # verdict.infra_definitive_failed is also excluded: an
            # infra-failed required check (CANCELLED/TIMED_OUT job-level
            # kill, issue #841) co-occurring with a genuine required-check
            # failure has its OWN dedicated remediation path (infra-rerun +
            # escalation, above) that this branch must not shadow or
            # double-dispatch against -- there is no code-fix rework for an
            # infra kill, so routing it through record_review(request_
            # changes) as if a code push could fix it would be a
            # mis-classification, not a fix.
            # test_janitor_mixed_genuine_failure_and_infra_failure_routes_to_rework_not_infra_rerun
            # (issue #847, pre-existing) pins this exact combination falling
            # through to janitor_blocked untouched, unchanged by this diff.
            # infra_definitive_failed is populated here whenever an
            # infra_failed required check co-occurs with a genuine
            # failed_required_checks entry, regardless of whether the infra
            # check's run id is parseable: is_infra_failure_block requires
            # `not failed_required_checks`, so it is always False in that
            # combination, which forces classify_infra_failures'
            # record_attempts=False path -- every infra_failed name lands in
            # definitive_failed (never rerun_run_ids) on this pass. Checking
            # infra_definitive_failed alone is therefore sufficient; it is
            # never empty here while infra_rerun_run_ids is non-empty.
            is_co_occurring_check_failure_block = (
                bool(verdict.failed_required_checks)
                and not verdict.is_check_failure_block
                and not is_merge_conflict_block
                and not verdict.is_no_op_rework
                and not verdict.infra_definitive_failed
            )
            if issue_number is not None and is_co_occurring_check_failure_block:
                transition(self.gh, self.config.labels, issue_number, "review_started")
                # verdict.failures always ends with the "Required check(s)
                # failed: ..." message when failed_required_checks is
                # truthy (janitor.run_janitor appends it last, nothing after
                # it) -- so failures[:-1] is exactly the co-occurring
                # janitor failure text(s), independent of which check(s)
                # _check_body/_check_title_conventional/etc. produced.
                co_occurring_failures = list(verdict.failures[:-1])
                summary = (
                    f"CI failed on {', '.join(verdict.failed_required_checks)}; push a fix. "
                    f"Also blocked by: {'; '.join(co_occurring_failures)}"
                )
                required_changes = (
                    _required_changes_from_checks(
                        checks, verdict.failed_required_checks, self.gh.check_run_annotations
                    )
                    + co_occurring_failures
                )
                log_event(
                    self.paths.state_file,
                    "review_dispatch_skipped_ci_red",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "failed_required_checks": list(verdict.failed_required_checks),
                        "co_occurring_failures": co_occurring_failures,
                        "co_occurring": True,
                    },
                    repo=self.repo_root.name,
                )
                return self.record_review(
                    pr_number,
                    "request_changes",
                    summary=summary,
                    reviewed_head=pr.get("headRefOid"),
                    required_changes=required_changes,
                    verdict_provenance="ci_gate_auto_reject",
                )

            # See _detect_ci_run_never_created for why this check exists
            # (a sibling repo measured 11 PRs stuck 4+ days behind an ambiguous
            # "Required check(s) missing" janitor failure). Detection lives
            # entirely in that method; the retrigger policy below (issue
            # #1274, W17) is this method's follow-up.
            ci_run_never_created_head_sha = self._detect_ci_run_never_created(
                pr,
                verdict,
                known_head=(pr_state or {}).get("ci_run_never_created_head"),
            )
            ci_run_never_created = ci_run_never_created_head_sha is not None

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
                pr_state_update = {
                    **existing_pr_state,
                    "number": pr_number,
                    "issue_number": issue_number,
                    "status": "janitor_blocked",
                    "janitor_ok": False,
                    "janitor_failures": list(verdict.failures),
                    "check_rerun_attempts": verdict.check_rerun_attempts,
                    # Issue #1133: structured flag so _is_dead_blocker can
                    # distinguish the transient "checks not reported yet"
                    # population from durably-stuck janitor_blocked PRs without
                    # parsing failure-message text.
                    "is_missing_checks_only_block": verdict.is_missing_checks_only_block,
                }
                # At most once per (pr, head_sha): only emit when this head
                # hasn't already been flagged as never-created.
                ci_run_never_created_new_head = (
                    ci_run_never_created
                    and existing_pr_state.get("ci_run_never_created_head")
                    != ci_run_never_created_head_sha
                )
                if ci_run_never_created:
                    pr_state_update["ci_run_never_created_head"] = ci_run_never_created_head_sha
                state["prs"][str(pr_number)] = pr_state_update
                if ci_run_never_created_new_head:
                    state = self._record_event(
                        state,
                        "ci_run_never_created",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "head_sha": ci_run_never_created_head_sha,
                            "branch": pr.get("headRefName"),
                            "missing_checks": list(verdict.missing_required_checks),
                        },
                    )
                if failures_changed:
                    # Issue #818: a draft co-occurring with another real
                    # failure (e.g. draft + empty body) is not auto-readied
                    # -- `is_draft_only_block` is False -- but the park is
                    # still worth distinguishing from routine janitor_gate
                    # bookkeeping so it's surfaced by heartbeat_check.py's
                    # check_draft_pr_blocked_events (issue #1366) rather than
                    # only a manual `gh pr list` sweep.
                    event_kind = "draft_pr_blocked" if verdict.is_draft else "janitor_gate"
                    state = self._record_event(
                        state,
                        event_kind,
                        {"pr_number": pr_number, "failures": list(verdict.failures)},
                    )
                save_state(self.paths.state_file, state)

            # Issue #1274 (W17): follow-up retrigger policy for the detection
            # above. `_detect_ci_run_never_created` dedupes per (pr, head_sha)
            # -- once a head is recorded (`pr_state_update["ci_run_never_created_head"]`
            # just above, which carries forward across passes via its
            # `**existing_pr_state` spread regardless of whether THIS pass's
            # own detector call fired), it returns None on every later pass
            # for that SAME head -- the ordinary steady state (no new
            # commits, checks still absent). Gating retrigger eligibility on
            # this pass's live detector return alone would make the feature
            # inert after its first pass, so read the marker that was just
            # persisted above (which already reflects either this pass's own
            # fresh positive or an earlier pass's) instead. Combine it with
            # the LIVE "still missing" signal (`verdict.missing_required_checks`,
            # recomputed fresh every pass by `run_janitor` regardless of the
            # detector's own dedup) to decide whether a retrigger is in scope
            # at all this pass. Deliberately NOT gated on
            # `is_missing_checks_only_block` (binding comment item 5 on issue
            # #1274): a retriggered run is harmless alongside an unrelated
            # co-occurring janitor failure (merge conflict, empty body, ...)
            # and gives the rework loop real signal -- the three PRs this
            # item exists to fix (#1186/#1192/#1214) all carry a co-occurring
            # failure.
            stale_checks_head_sha = str(pr.get("headRefOid") or "") or None
            stale_checks_retrigger_in_scope = (
                issue_number is not None
                and stale_checks_head_sha is not None
                and pr_state_update.get("ci_run_never_created_head") == stale_checks_head_sha
                and bool(verdict.missing_required_checks)
            )
            if stale_checks_retrigger_in_scope:
                stale_checks_retrigger_result = self._attempt_stale_checks_retrigger(
                    pr,
                    pr_number=pr_number,
                    issue_number=issue_number,
                    head_sha=stale_checks_head_sha,
                    existing_pr_state=pr_state_update,
                )
                if stale_checks_retrigger_result is not None:
                    return stale_checks_retrigger_result

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
        # Issue #592: stamp the digest of the resolved review template +
        # referenced section partials into pr.json so a future template edit
        # is detectable as packet staleness alongside a head-SHA change.
        # Without this, a static-head PR keeps being handed a packet rendered
        # from the old template indefinitely -- the orchestrator never
        # regenerates because headRefOid is treated as the packet's only input.
        pr_json = _slim_pr_json(pr)
        pr_json["prompt_template_sha"] = self._review_template_sha()
        # Issue #1439: stamp the structure-aware turn-cap multiplier into the
        # packet so the reviewer launch reads a structure-aware budget instead
        # of the flat ``review_max_turns``. The multiplier (not the absolute
        # cap) is stored so the dispatch path can re-derive the final cap from
        # the live ``review_max_turns`` config + the per-PR miss streak at
        # launch time -- a packet may outlive a config edit, and the base cap
        # is config's to own.
        _max_touched_lines = _max_touched_file_line_count(diff, self.repo_root)
        pr_json["review_turn_cap_structure_multiplier"] = structure_turn_cap_multiplier(
            _max_touched_lines, self.config.review_dispatch
        )
        pr_json["review_turn_cap_max_touched_lines"] = _max_touched_lines
        self._write_json(pr_dir / "pr.json", pr_json)
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
                    verdict_provenance="test_adequacy_auto_reject",
                )
            # Gate passed while enabled: add Tier-2 packet section (issue #180)
            test_adequacy_section = render_test_adequacy_section(
                test_adequacy_verdict.facts, test_adequacy_verdict.warnings
            )

        # Static diff-coverage / unwired-symbol probe (issues #1260/#1261).
        # Advisory-only, never blocking -- see CoverageProbeConfig's docstring.
        # Computed in the same packet-build phase as the Tier-1/2 test-adequacy
        # gate above, on the same diff text already in hand; no PR-branch
        # checkout is created for this (see diff_coverage_probe module docstring).
        static_probe_section = ""
        if self.config.coverage_probe.enabled:
            static_probe_verdict = run_static_probe(
                diff, self.repo_root, self.config.coverage_probe
            )
            static_probe_section = render_static_probe_section(static_probe_verdict)
            if static_probe_verdict.branch_findings or static_probe_verdict.unwired_findings:
                # These two kinds are the substrate for the 2-week
                # false-positive measurement window -- a flag that isn't
                # logged can't be measured.
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    if static_probe_verdict.branch_findings:
                        state = self._record_event(
                            state,
                            # event-consumer: audit-only -- measurement-window telemetry for the
                            # false-positive rate (see comment above), not consumed downstream by design
                            "coverage_probe_flagged",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "flagged_files": [
                                    f.filename for f in static_probe_verdict.branch_findings
                                ],
                            },
                            level="warning",
                        )
                    if static_probe_verdict.unwired_findings:
                        state = self._record_event(
                            state,
                            # event-consumer: audit-only -- measurement-window telemetry for the
                            # false-positive rate (see comment above), not consumed downstream by design
                            "unwired_symbol",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "symbols": [
                                    f.symbol for f in static_probe_verdict.unwired_findings
                                ],
                            },
                            level="warning",
                        )
                    save_state(self.paths.state_file, state)

        # Run containment check for worker edits leaked into operator checkout
        containment_warnings = check_operator_containment(self.repo_root, diff, pr_number)
        # Merge containment warnings with janitor warnings
        merged_warnings = tuple(list(verdict.warnings) + list(containment_warnings))
        # If test-adequacy gate is enabled and passed, merge its warnings too
        if test_adequacy_verdict is not None:
            merged_warnings = tuple(list(merged_warnings) + list(test_adequacy_verdict.warnings))
        prompt_path = pr_dir / "review-prompt.md"
        decision_path = pr_dir / "review-decision.json"
        diff_size_section = _diff_size_section(
            diff, self.config.review_dispatch.diff_line_threshold, diff_path
        )
        ci_status_section = _ci_status_section(
            checks, self.config.auto_merge.required_checks, pr_dir / "checks.json"
        )
        # Issue #1445: over-cap file-addition finding. Advisory-only, never
        # blocking -- the rubric line in review.md flags an over-cap addition
        # as a REPORTABLE FINDING and this probe surfaces the concrete files.
        # ``file_size_cap_lines`` of 0 (default) disables the probe: the section
        # renders "" (rubric prose stays present), mirroring the
        # ``coverage_probe.enabled=False`` disabled contract.
        _file_size_cap = self.config.review_dispatch.file_size_cap_lines
        over_cap_section = render_over_cap_section(
            _over_cap_file_findings(diff, self.repo_root, _file_size_cap)
            if _file_size_cap > 0
            else None
        )

        # Issue #1460: attachment-budget review-packet section. Cheap gate
        # first -- most PRs touch neither `.attachment-budgets.json` nor a
        # baselined host file, so the reconstruct/build path below is
        # skipped for them entirely (section renders "").
        attachment_budget_section = self._build_attachment_budget_section(diff, pr_number)

        # Issue #1036: compare-and-swap the head immediately before committing
        # this packet's outputs (prompt + decision). ``pr`` was snapshotted
        # once, at the top of this method, and everything since -- diff
        # fetch, janitor, containment check -- can take minutes (385s
        # observed in production). The stale-verdict reset a
        # few lines below used to compare a recorded verdict's pinned head
        # against that same build-start snapshot with `!=`, which is
        # symmetric and has the wrong sign whenever the SNAPSHOT is the
        # stale side: a verdict recorded mid-build at the genuinely current
        # head was voided because it disagreed with an old snapshot, not
        # because it was actually behind.
        #
        # A directional (ancestry) fix on the reset alone is not enough --
        # verified against this same method's control flow: ``pr`` also
        # feeds ``_build_prior_review_section`` and the ``review.md`` render
        # below, so sparing only the reset would commit a current-head
        # verdict next to a prompt describing an older diff, with nothing in
        # state recording that mismatch. So this re-reads the live head and,
        # if it no longer matches the snapshot (or can't be read at all --
        # fail closed), discards the WHOLE packet: neither the prompt nor
        # the decision reset is written, no label/status transition fires,
        # and the existing verdict (if any) is left exactly as it was. The
        # next pass rebuilds the packet against the then-current head. A
        # verdict pinned to the (unchanged) live head is therefore never
        # touched by this method in the same pass that would otherwise have
        # voided it -- which is what "preserve a verdict pinned to a
        # descendant of the snapshot" reduces to once the snapshot itself is
        # kept fresh at commit time.
        #
        # Issue #1072: this guard covers ONLY the packet-commit tail exit
        # (the prompt + decision write below). The two earlier verdict-
        # writing exits -- the CI-required-check-failure block and the
        # Tier-1 test-adequacy hard gate -- return through record_review()
        # before reaching this point. The same head-moved invariant is now
        # enforced for those exits (and every other record_review() caller)
        # by the compare-and-swap guard pushed into record_review() itself,
        # so the invariant holds by construction for all verdict writes from
        # this method, not just the one this guard happens to sit on.
        live_pr_for_commit = self.gh.pr_view(pr_number)
        live_head_for_commit = live_pr_for_commit.get("headRefOid")
        snapshot_head_for_commit = pr.get("headRefOid")
        if live_head_for_commit is None or live_head_for_commit != snapshot_head_for_commit:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state = self._record_event(
                    state,
                    "review_packet_discarded_head_moved",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "snapshot_head_sha": snapshot_head_for_commit,
                        "live_head_sha": live_head_for_commit,
                    },
                    level="warning",
                )
                save_state(self.paths.state_file, state)
            return CommandResult(
                False,
                f"PR #{pr_number}: head moved during packet build "
                f"({snapshot_head_for_commit} -> {live_head_for_commit!r}); "
                "discarding packet, will rebuild next pass",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "reason": "head_moved_during_build",
                    "snapshot_head_sha": snapshot_head_for_commit,
                    "live_head_sha": live_head_for_commit,
                },
            )

        # Single read of review-decision.json, BEFORE rendering: used to build
        # the round-2 $prior_review_section below. The stale-verdict reset a
        # few lines down re-reads the file inside the state lock (issue #1340:
        # the outside-lock read here may be stale w.r.t. a concurrent
        # record_review, and the reset write must be atomic with the state
        # write). Previously that reset was the only reader, and it ran after
        # the prompt was already rendered — so a prior round's
        # verdict/summary/required_changes were on disk at render time but
        # never surfaced to the reviewer.
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
                "janitor_section": _janitor_section(merged_warnings),
                "test_adequacy_section": test_adequacy_section,
                "static_probe_section": static_probe_section,
                "diff_size_section": diff_size_section,
                "ci_status_section": ci_status_section,
                "over_cap_section": over_cap_section,
                "attachment_budget_section": attachment_budget_section,
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
            # Issue #1265: an explicit None sentinel, not an omitted key --
            # "pending" is not a verdict yet, so it has no provenance to
            # report, but the key's presence keeps the no-default contract
            # structurally checkable (a bypass writer that forgets the key
            # entirely looks identical to one that never considered it).
            "verdict_provenance": None,
        }
        # Issue #868: review_dispatch.enabled gates whether landing this PR in
        # "reviewing" is safe. Disabled means dispatch_reviews()'s launch+reap
        # machinery never services this state (its own internal gate skips
        # it too), so stamping "reviewing"/agent:reviewing here would strand
        # the PR under a label nothing runs to clear. The packet itself is
        # still written either way — only the state/label transition is
        # gated, so a human can still read the prompt manually.
        dispatch_disabled = not self.config.review_dispatch.enabled
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            # Issue #1351: the dispatch attempt counter is reset only for a
            # genuinely new head, not on every packet write. ``review()`` is
            # called every loop pass for a PR stuck on a stale
            # ``decision``/``reviewed_head_sha`` pair (the ``already_approved``
            # branch with ``head_matches=False``), so an unconditional reset
            # raced with ``dispatch_reviews()``'s claim-time increment and the
            # counter oscillated 1->0->1->0... never reaching
            # ``max_review_dispatch_attempts`` -- the cap was silently inert and
            # reviewers re-dispatched indefinitely with no escalation.
            # ``review_dispatch_attempt_last_head`` baselines the head the
            # current attempt count is counting against, mirroring the
            # ``no_op_rework_attempts_last_head`` /
            # ``conflict_rework_attempts_last_head`` pattern: the counter resets
            # only when the packetized head advances past it.
            _existing_pr_state = state["prs"].get(str(pr_number), {})
            _packet_head = pr.get("headRefOid")
            _prior_attempt_last_head = _existing_pr_state.get("review_dispatch_attempt_last_head")
            _fresh_dispatch_cycle = (
                _prior_attempt_last_head is None or _prior_attempt_last_head != _packet_head
            )
            _preserved_attempt_count = int(
                _existing_pr_state.get("review_dispatch_attempt_count", 0)
            )
            # Issue #1340: the pending-reset decision-file write is inside the
            # state lock, and when a stale-head terminal verdict is voided the
            # state.json ``decision`` field is cleared in the same locked
            # section, so the file and state can never diverge (file "pending"
            # while state carries the stale decision).
            # The ``existing_decision`` read at line ~9429 was outside any lock
            # and may be stale w.r.t. a concurrent record_review; re-read the
            # on-disk decision here, inside the lock. The #1036 compare-and-swap
            # above already proved the PR head is stable, so the only concurrent
            # writer of this file is record_review, which holds this same lock.
            live_decision = self._review_decision(pr_number)
            live_decision_value = live_decision.get("decision")
            live_reviewed_head_sha = live_decision.get("reviewed_head_sha")
            # A verdict is pinned to a specific head. If the PR has moved on,
            # the old verdict is void and must not survive into the new packet.
            # This applies to all terminal decisions (approved, request_changes,
            # blocked), not just approvals: a request_changes on an old head is
            # equally stale when the head has advanced, and carrying forward its
            # summary/required_changes misleads the reviewer into re-issuing the
            # same verdict without examining the new diff.
            #
            # Only a real terminal decision (approved/request_changes/blocked)
            # on disk can be voided. ``_review_decision`` returns
            # ``{"decision": "missing"}`` for both a missing file and a
            # corrupt/unreadable one (issue #1362's ``review_decision``
            # collapses both into one sentinel) -- that is not a terminal
            # verdict, so it never triggers the void path -- the fresh-
            # template write below handles the missing-file case, and a
            # corrupt file is left for a human rather than silently
            # overwritten (mirroring the original code's ``else`` branch,
            # which only reset on a real terminal decision).
            voided_stale_verdict = live_decision_value in (
                "approved",
                "request_changes",
                "blocked",
            ) and (
                live_reviewed_head_sha is None or live_reviewed_head_sha != pr.get("headRefOid")
            )
            if not decision_path.exists() or voided_stale_verdict:
                # Issue #1362 Stage 2: routed through the single writer so the
                # placeholder is head-stamped like every other verdict --
                # ``reviewed_head_sha`` lets a "pending" that is actually
                # pinned to a dead head be told apart from a genuinely
                # unreviewed current head (the mechanism the issue exists
                # for). ``pr`` is the head snapshot already validated
                # unchanged by the compare-and-swap guard above this block.
                # ``archive_round=False``: a pending placeholder is not a
                # reviewer round -- it carries no decision/summary content,
                # so archiving it would mint a content-free round ahead of
                # the first real verdict (and, on the rework path, before
                # every subsequent verdict whenever the head moves and the
                # packet is rebuilt), shifting round numbers and polluting
                # ``_build_prior_review_section``'s rendered history with
                # phantom "Round N (decision: pending)" entries that never
                # came from a reviewer. This is the flat-file head-stamp
                # only; the reader still resolves it correctly since it
                # reads the flat file first.
                record_decision(
                    pr_dir, decision_template, pr.get("headRefOid"), archive_round=False
                )
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
                # Issue #1340: when the decision file was just voided back to
                # "pending" (stale head), mirror that into state.json so the two
                # stores agree on the decision. Without this, state.json
                # retained the stale ``decision`` while the file said
                # "pending" -- a file-trusting consumer (the packet-current skip
                # in loop()) saw "pending" while state-trusting paths acted on
                # the stale verdict. ``reviewed_head_sha`` is intentionally
                # NOT cleared: ``_route_rework_candidate_to_review`` uses it
                # to detect whether ``review()`` recorded a new decision vs.
                # just wrote a fresh pending packet, and clearing it would
                # break that detection (issue #339).
                **({"decision": "pending"} if voided_stale_verdict else {}),
                **({} if dispatch_disabled else {"status": "reviewing"}),
                "janitor_ok": True,
                "janitor_failures": [],
                "is_missing_checks_only_block": False,
                "janitor_warnings": list(merged_warnings),
                "consecutive_failed_merge_attempts": 0,
                "check_rerun_attempts": verdict.check_rerun_attempts,
                # Issue #1351: reset the dispatch attempt counter only when
                # the packet is for a genuinely new head. Rebuilding the
                # packet for the same head (e.g. a stale already_approved /
                # head_matches=False loop that calls review() every pass) must
                # NOT zero the counter -- otherwise review()'s reset races
                # with dispatch_reviews()'s claim-time increment and
                # max_review_dispatch_attempts never fires. The companion
                # ``review_dispatch_attempt_last_head`` baselines the head the
                # count is against (mirroring no_op/conflict_rework_attempts_
                # last_head); the counter resets only when the head advances.
                "review_dispatch_attempt_count": (
                    0 if _fresh_dispatch_cycle else _preserved_attempt_count
                ),
                "review_dispatch_attempt_last_head": _packet_head,
                # Reset the unreadable-log streak: a new packet is a new
                # review cycle, so a prior persistent-unreadable condition
                # must not carry forward and immediately count against the
                # fresh attempt budget (issue #1069).
                "review_log_unreadable_streak": 0,
                # Issue #1439: reset the turn-limit miss streak on a fresh
                # dispatch cycle (new head). A same-head rebuild must NOT zero
                # it -- mirroring review_dispatch_attempt_count's same-head
                # preservation (issue #1351) -- or the cap-aware backstop
                # never converges and a PR that keeps hitting the turn limit
                # on the same head redispatches forever at the base cap.
                "review_turn_limit_miss_streak": (
                    0
                    if _fresh_dispatch_cycle
                    else int(_existing_pr_state.get("review_turn_limit_miss_streak", 0))
                ),
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
                    # Issue #1439: structure-aware turn cap stamped into the
                    # packet, mirrored into the event for auditability.
                    "review_turn_cap_structure_multiplier": pr_json.get(
                        "review_turn_cap_structure_multiplier", 1
                    ),
                    "review_turn_cap_max_touched_lines": pr_json.get(
                        "review_turn_cap_max_touched_lines", 0
                    ),
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
                # Issue #1362 Stage 1: single review-decision reader instead
                # of state.json's decision/reviewed_head_sha fields.
                resolved_decision = review_decision(pr_dir, None, pr.get("headRefOid"))
                if resolved_decision.decision == "request_changes" and not resolved_decision.stale:
                    should_skip_transition = True
                # Issue #384, re-checked here and not only at the top of
                # review(): the `review_started` edge removes every workflow
                # label it is not adding, `human_needed` included. The guard at
                # the head of this method reads state before any of review()'s
                # own work, so it cannot see an escalation applied *during* this
                # pass -- issue #1099's now-deleted cross-family regen-budget
                # escalation (role-config phase 2, track A) used to do exactly
                # that, from a mid-review call a few hundred lines above.
                # Without this re-read an escalating pass would apply
                # `human_needed` and then strip it moments later, leaving the
                # issue escalated in state with no label on GitHub: the
                # #586/#594 escalated-without-label shape, reintroduced from
                # inside the pass rather than from the next one.
                #
                # The sibling mid-review escalation (the infra-rerun cap) avoids
                # this by returning immediately. This path cannot -- the model
                # already ran and its packet is worth writing -- so it suppresses
                # the label edge instead. Enforced at the boundary for *any*
                # mid-review escalation, so the next one added here inherits
                # the fix.
                issue_state_now = (
                    state.get("issues", {}).get(str(issue_number), {})
                    if issue_number is not None
                    else None
                )
                if any(_escalation_flags(pr_state, issue_state_now)):
                    should_skip_transition = True

            if not should_skip_transition and not dispatch_disabled:
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
                "label_error": label_error,
                "checks_unavailable": False,
            },
        )

    # Deliberately NOT @_guard_state_lock: that decorator returns a CommandResult
    # on StateLockBusy, and this method returns a plain dict that gets embedded in
    # dispatch_reviews' payload. Its only caller IS @_guard_state_lock-decorated,
    # so a busy lock here propagates up and skips the whole pass -- which is the
    # right granularity anyway.

    @_guard_state_lock
    def dispatch_reviews(
        self, limit: int | None = None, *, now: datetime | None = None
    ) -> CommandResult:
        """Launch reviewer sessions concurrently for queued PRs.

        Issue #370: a deterministic loop stage that turns ``review_queue()```
        into launched, sidecar-tracked reviewer processes. Reviewers are
        launched through whichever harness ``self.config.reviewer.harness``
        names (issue #1513: claude-code, devin-shell, or api -- see
        ``_REVIEW_LAUNCHERS`` below); there is no provider-rate-limit
        governor here — only an optional local-only process cap
        (``max_local_review_processes``) to protect the host from too many
        concurrent reviewer worktrees.

        The double-dispatch protection is a two-phase claim on
        ``state["prs"][pr]``: this method writes ``review_dispatch_pending``,
        launches outside the lock, then upgrades to ``review_dispatch_dispatched``
        or ``review_dispatch_failed``. Overlapping passes see the pending claim
        and skip until it goes stale. A reviewer that dies without a verdict is
        freed by ``_detect_and_handle_stalled_reviews`` after the stale-claim
        timeout, making the PR re-dispatchable.

        ``now`` (issue #828) is the injectable clock for this entire pass: it
        is resolved once below and forwarded to every ``is_claim_stale`` check
        this method drives, directly (via ``_is_review_dispatchable``) and
        indirectly (via ``_detect_and_handle_stalled_reviews``), instead of
        each one independently racing the wall clock. Defaults to
        ``datetime.now(UTC)`` when omitted, so production behavior is
        byte-identical; tests can freeze it and assert exact equality instead
        of a wall-clock-tolerance proximity check.
        """
        resolved_now = now if now is not None else datetime.now(UTC)
        reviews_dir = self._layout.reviews_dir

        # Run the verdict-reaper and orphan/stalled sweeps BEFORE the quota
        # gate so dead reviewers are reaped and stale claims are freed even
        # during throttle periods. Without this ordering, a quota deferral
        # returns early and leaves dead reviewer claims stuck — blocking
        # re-dispatch after the quota resets. In dry-run mode we skip
        # these sweeps to stay read-only.
        #
        # Issue #868: these sweeps must ALSO run ahead of the
        # ``review_dispatch.enabled`` gate below, for the same reason —
        # disabling dispatch must not disable the cleanup a *previously*
        # dispatched (or previously stamped-reviewing) claim still needs. A
        # dead reviewer's claim that goes stale while the flag happens to be
        # off must still be freed, or it stays stuck even after the flag is
        # re-enabled (the reap is what makes it re-dispatchable). Before this
        # fix these sweeps sat below the ``enabled`` early return and were
        # unreachable whenever dispatch was disabled.
        verdict_result = {"recorded": [], "missed": []}
        reconciled_verdicts: list[dict[str, Any]] = []
        if not self.dry_run:
            verdict_result = self._reap_review_verdicts(reviews_dir)
            # Issue #736: ingest completed on-disk verdicts that were never
            # recorded into state. This runs BEFORE the stale-claim sweep so
            # the sweep's ``decision_already_recorded`` skip (issue #734)
            # never fires for a PR this reconciliation just ingested -- after
            # ``record_review`` the PR's ``review_dispatch_status`` is
            # ``review_dispatch_completed`` and the stale-claim branch (which
            # only matches ``review_dispatch_status is None``) no longer
            # applies. Like the other sweeps here, this runs ahead of the
            # ``review_dispatch.enabled`` gate (issue #868) so a stranded
            # verdict is ingested even when dispatch is disabled fleet-wide.
            reconciled_verdicts = self._reconcile_stranded_verdicts()
            _detect_and_handle_stalled_reviews(
                reviews_dir,
                self.paths.state_file,
                self.config,
                self.repo_root,
                write_gate=self.write_gate,
                now=resolved_now,
            )
            _reap_completed_review_checkouts(self.repo_root, reviews_dir, self.paths.state_file)
            _reap_orphaned_review_checkouts(
                self.gh,
                self.repo_root,
                reviews_dir,
                self.paths.state_file,
                self.config,
                write_gate=self.write_gate,
            )
        recorded_verdicts = verdict_result.get("recorded", [])
        missed_verdicts = verdict_result.get("missed", [])

        # Issue #1088: this MUST stay above the ``review_dispatch.enabled`` early
        # return below. The sweep it drives is #586's, and #586's whole purpose is
        # that an escalated PR cannot sit invisible on GitHub -- a guarantee that
        # has nothing to do with whether new reviewers are being launched. Placing
        # it below the gate (where its input set used to be built) is what made it
        # dead code in every deployed fleet. The reaper sweeps immediately above
        # are the existing precedent for real work happening on this side of the
        # gate.
        repaired_labels = self._repair_escalated_labels()

        if not self.config.review_dispatch.enabled:
            # Issue #868 part 3: this is a real no-op for LAUNCHING new
            # reviewers, but the reaper sweeps above may still have done real
            # work (freeing a stale claim). Carry that through explicitly
            # instead of returning a bare ``ok=True`` that reads identically
            # to "nothing happened at all" — and mark ``disabled`` so a
            # caller doesn't have to string-match the message to tell this
            # apart from a real dispatch pass.
            return CommandResult(
                True,
                "review dispatch disabled",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "launched_count": 0,
                    "disabled": True,
                    "recorded_verdicts": recorded_verdicts,
                    "missed_verdicts": missed_verdicts,
                    # Issue #736: stranded-verdict reconciliation runs above
                    # the gate, so its results are reported on the disabled
                    # path for the same reason #868 reports the reaper results
                    # here -- this is the ONLY path the reconciliation runs on
                    # in a deployed fleet with dispatch disabled.
                    "reconciled_verdicts": reconciled_verdicts,
                    # Issue #1088: reported on the disabled path for the same
                    # reason #868 reports the reaper results here -- this is the
                    # ONLY path the sweep runs on in a deployed fleet, so a
                    # payload that omitted it would make the fix unobservable
                    # from the outside and indistinguishable from the dead code
                    # it replaces.
                    "escalated_labels_repaired": repaired_labels,
                },
            )

        # Issue #617: dry-run must short-circuit BEFORE the quota-alert marker
        # write, the rescue partition, and the attempt-cap escalation block --
        # all of which mutate state.json (or fire a real emit_digest) and sat
        # strictly before the old dry-run gate at the bottom of this method.
        # Mirror _dispatch_impl's top-of-function dry-run short-circuit: compute
        # the would-be selection read-only and return the plan, touching
        # nothing. A preview that escalates or writes a quota-alert marker is
        # worse than the bug it replaces, so this gate is a single early return,
        # not a branch inside the escalation arm.
        if self.dry_run:
            quota_state = load_state_locked(self.paths.state_file)
            if is_reviewer_quota_exhausted(quota_state) and not is_reviewer_probe_ready(
                quota_state
            ):
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
                        "reconciled_verdicts": reconciled_verdicts,
                    },
                )
            probe_mode_dry = bool(
                is_reviewer_quota_exhausted(quota_state) and is_reviewer_probe_ready(quota_state)
            )
            queue_result = self.review_queue()
            all_candidates = queue_result.data.get("queue", [])
            # Issue #617 rework: exclude rescue-marked PRs from the dry-run
            # candidate computation via a read-only state check, NOT a call
            # to _partition_rescue_candidates (which has real side effects:
            # it runs cross-family reviews, records verdicts, and escalates).
            # A rescue-marked PR must never appear in the preview as a normal
            # dispatch/escalation candidate — the real branch routes it
            # through _process_rescue_review instead.
            rescue_marked_dry: list[dict[str, Any]] = []
            candidates_dry: list[dict[str, Any]] = []
            for c in all_candidates:
                if quota_state.get("prs", {}).get(str(c["pr"]), {}).get("rescue_attempted"):
                    rescue_marked_dry.append(c)
                else:
                    candidates_dry.append(c)
            selection = _select_review_dispatch_candidates(
                candidates_dry,
                quota_state,
                self.config.review_dispatch,
                reviews_dir,
                self.paths.state_file,
                resolved_now,
                limit,
                probe_mode_dry,
            )
            # Issue #1251: mirror the real path's empty-diff pre-flight gate
            # in the dry-run preview so the two cannot diverge. Read-only:
            # no events emitted, no state mutation.
            dry_selected: list[dict[str, Any]] = []
            dry_skipped_empty_diff: list[int] = []
            for c in selection.selected:
                diff_text = self._read_packet_diff(c["pr"])
                if diff_text is not None and not diff_text.strip():
                    dry_skipped_empty_diff.append(c["pr"])
                else:
                    dry_selected.append(c)
            # Issue #1131: mirror the real path's already-approved pre-claim
            # gate in the dry-run preview so the two cannot diverge. Read-only:
            # no events emitted, no state mutation.
            dry_skipped_already_approved: list[int] = []
            dry_selected_after_approval: list[dict[str, Any]] = []
            for c in dry_selected:
                pr_dir = self.paths.prs / f"pr-{c['pr']}"
                resolved = review_decision(pr_dir, None, c.get("packet_head_sha"))
                if resolved.decision == "approved" and not resolved.stale and not resolved.missing:
                    dry_skipped_already_approved.append(c["pr"])
                else:
                    dry_selected_after_approval.append(c)
            dry_selected = dry_selected_after_approval
            return CommandResult(
                True,
                f"dry-run: would dispatch {len(dry_selected)} reviewer(s)",
                {
                    "selected_count": len(dry_selected),
                    "attempted_count": len(dry_selected),
                    "failed_count": 0,
                    "launched_count": 0,
                    "deferred_count": len(all_candidates) - len(dry_selected),
                    "escalated_skipped": selection.escalated_skipped,
                    "skipped_empty_diff": dry_skipped_empty_diff,
                    "skipped_already_approved": dry_skipped_already_approved,
                    "merge_conflict_routed": [c["pr"] for c in selection.merge_conflict_routed],
                    "rescue_marked_excluded": [c["pr"] for c in rescue_marked_dry],
                    "recorded_verdicts": recorded_verdicts,
                    "missed_verdicts": missed_verdicts,
                    "reconciled_verdicts": reconciled_verdicts,
                    **selection.local_cap.report_fields(),
                },
            )

        # Clear the reviewer quota if any verdicts were recorded from dead
        # reviewers. This is the only proof the quota window is actually open:
        # a process that merely *started* can still die seconds later from an
        # asynchronous session-limit kill. Run before the quota gate so a
        # successful reap clears the throttle and lets the pass proceed.
        # (recorded_verdicts/missed_verdicts are computed above, ahead of the
        # disabled-gate return, so issue #868's disabled payload can report
        # them too.)
        if not self.dry_run and recorded_verdicts:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                if is_reviewer_quota_exhausted(state):
                    state = clear_reviewer_quota(state)
                    # A recorded verdict is proof the provider let a real
                    # review through -- reset the probe backoff so the next
                    # outage starts from the configured base interval again
                    # instead of carrying forward an exponentially-grown one.
                    # Stamp the same recovery marker a green flat probe would,
                    # so a later dead-reviewer reap sweep can suppress backoff
                    # from a throttle signature that predates this recovery
                    # (issue #662). Use microsecond precision for sub-second
                    # comparisons against a dead session's log mtime.
                    state = {
                        **state,
                        "reviewer_quota": {
                            **(state.get("reviewer_quota") or {}),
                            "consecutive_probe_failures": 0,
                            "last_probe_cleared_at": datetime.now(UTC)
                            .isoformat()
                            .replace("+00:00", "Z"),
                        },
                    }
                    self.write_gate.save_state(state)

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
                        self.write_gate.save_state(mark_reviewer_quota_alerted(state))
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
                    "reconciled_verdicts": reconciled_verdicts,
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
                    "reconciled_verdicts": reconciled_verdicts,
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
                    "reconciled_verdicts": reconciled_verdicts,
                    "rescue_review_results": rescue_review_results,
                },
            )

        # Issue #617 rework: the PR-selection logic (escalated-skip,
        # merge-conflict routing, dispatchable-list construction,
        # local/concurrent cap computation) is shared with the dry-run branch
        # via _select_review_dispatch_candidates, so the two copies cannot
        # diverge. The read-only selection runs first; the attempt-cap
        # escalation (which mutates state) runs separately below. At-cap PRs
        # are filtered by _is_review_dispatchable regardless of whether they
        # have already been escalated, so the dispatchable list is stable
        # across the escalation mutation.
        #
        # Route merge-conflicting PRs to rework instead of dispatching a
        # reviewer (issue #1497): a CONFLICTING PR sitting in
        # ``agent:reviewing`` with a current packet would otherwise be
        # dispatched every pass, but the reviewer's verdict cannot merge —
        # and if the PR never reaches the merge lane (e.g. no verdict is
        # ever recorded), ``review()``'s own janitor-gate conflict route
        # never runs either, so the PR sits in ``agent:reviewing`` forever
        # with zero dispatch events. Routing here mirrors ``review()``'s
        # janitor-gate merge-conflict path
        # (``_route_janitor_gate_failure_to_rework`` with
        # ``reason="merge_conflict"``), including the same attempt cap and
        # escalation to ``agent:human-needed``.
        max_attempts = self.config.review_dispatch.max_review_dispatch_attempts
        # Issue #1439: backstop cap on consecutive turn-limit misses.
        max_turn_limit_misses = self.config.review_dispatch.max_consecutive_turn_limit_misses
        # Issue #586's repair set used to be collected here, from the PRs this
        # candidate loop skipped. Issue #1088: that made the sweep unreachable,
        # because this loop runs below the ``review_dispatch.enabled`` early
        # return and both deployed fleets run that flag false -- the set was
        # always empty. The repair now derives its own subjects from state in
        # ``_repair_escalated_labels()``, called above that early return.
        selection_state = load_state_locked(self.paths.state_file)
        selection = _select_review_dispatch_candidates(
            candidates,
            selection_state,
            self.config.review_dispatch,
            reviews_dir,
            self.paths.state_file,
            resolved_now,
            limit,
            probe_mode,
        )
        escalated_skipped = selection.escalated_skipped
        merge_conflict_routed = selection.merge_conflict_routed
        dispatchable = selection.dispatchable
        local_cap = selection.local_cap
        selected = selection.selected

        # Issue #1251: pre-flight empty-diff gate. A PR whose diff.patch is
        # empty (zero-file diff vs base) must not burn a paid reviewer
        # session reviewing nothing. The packet build (review()) already
        # fetches and writes diff.patch; an empty patch is the signal. Skip
        # dispatch for these PRs and emit review_dispatch_skipped_empty_diff
        # instead of claiming. review_dispatch_attempt_count is NOT incremented --
        # an empty diff is not a review attempt and must not walk toward
        # escalation. Do not auto-close the PR from this path; closing is a
        # separate judgment (the #1221 fix owns the duplicate-PR lifecycle).
        # A missing diff.patch is NOT treated as empty -- that is a missing
        # packet (a different problem), not a zero-delta PR.
        skipped_empty_diff: list[int] = []
        skipped_empty_diff_issues: dict[int, int | None] = {}
        selected_with_diff: list[dict[str, Any]] = []
        for c in selected:
            diff_text = self._read_packet_diff(c["pr"])
            if diff_text is not None and not diff_text.strip():
                skipped_empty_diff.append(c["pr"])
                skipped_empty_diff_issues[c["pr"]] = c.get("issue")
            else:
                selected_with_diff.append(c)
        selected = selected_with_diff
        if skipped_empty_diff:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                for pr_number in skipped_empty_diff:
                    state = self.write_gate.record_event(
                        state,
                        "review_dispatch_skipped_empty_diff",
                        {
                            "pr_number": pr_number,
                            "issue_number": skipped_empty_diff_issues.get(pr_number),
                        },
                        level="warning",
                    )
                self.write_gate.save_state(state)

        # Issue #1131: re-check the decision file at claim time. review_queue's
        # snapshot can be stale when a dispatch pass interleaves with an
        # operator unescalate->approve->merge: the queue was built before the
        # approved verdict was recorded, so an already-approved PR reaches
        # claim time looking reviewable (unescalation cleared
        # ``review_dispatch_status`` and removed ``agent:human-needed``, and
        # ``_is_review_dispatchable`` keys off those, not the decision file).
        # Skip any PR whose decision file now records an approval for the head
        # this pass would review -- launching a fresh paid reviewer is
        # redundant, and a late request_changes from that reviewer clobbers
        # the merge-authorizing decision file (issue #1131). Mirrors the
        # empty-diff gate above: a read-only pre-claim filter that emits an
        # event and never increments ``review_dispatch_attempt_count`` (an
        # already-approved PR is not a review attempt). ``packet_head_sha`` is
        # the head the reviewer would read; for every queued candidate
        # review_queue already verified it equals the live head, so an
        # approval pinned to it is an approval for the live head.
        skipped_already_approved: list[int] = []
        skipped_already_approved_issues: dict[int, int | None] = {}
        skipped_already_approved_heads: dict[int, str | None] = {}
        selected_not_approved: list[dict[str, Any]] = []
        for c in selected:
            pr_number = c["pr"]
            packet_head = c.get("packet_head_sha")
            pr_dir = self.paths.prs / f"pr-{pr_number}"
            resolved = review_decision(pr_dir, None, packet_head)
            if resolved.decision == "approved" and not resolved.stale and not resolved.missing:
                skipped_already_approved.append(pr_number)
                skipped_already_approved_issues[pr_number] = c.get("issue")
                skipped_already_approved_heads[pr_number] = packet_head
            else:
                selected_not_approved.append(c)
        selected = selected_not_approved
        if skipped_already_approved:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                for pr_number in skipped_already_approved:
                    state = self.write_gate.record_event(
                        state,
                        "review_dispatch_skipped_already_approved",
                        {
                            "pr_number": pr_number,
                            "issue_number": skipped_already_approved_issues.get(pr_number),
                            "reviewed_head_sha": skipped_already_approved_heads.get(pr_number),
                        },
                        level="warning",
                    )
                self.write_gate.save_state(state)

        # Escalate PRs whose dispatch attempt count has reached the cap.
        # These PRs are stuck (every reviewer died without a verdict) and
        # must not be re-dispatched indefinitely. Mark them escalated so a
        # human can intervene, mirroring the rework-cycle escalation pattern.
        # The shared helper already classified escalated-skip and
        # merge-conflict-routed PRs read-only; this loop only considers the
        # remaining candidates, skipping live in-flight reviewers (issue
        # #573) so a mid-session verdict is not orphaned. Escalation gate
        # (issue #575): an already-escalated PR is in ``escalated_skipped``
        # and skipped here too, so an issue escalated by an independent path
        # does not get a second bogus "max_review_dispatch_attempts_exceeded"
        # escalation on top of it.
        escalated_for_labels: list[tuple[int, int | None]] = []
        escalated_skipped_set = set(escalated_skipped)
        merge_conflict_pr_set = {c["pr"] for c in merge_conflict_routed}
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            changed = False
            for c in candidates:
                if c["pr"] in escalated_skipped_set or c["pr"] in merge_conflict_pr_set:
                    continue
                pr_key = str(c["pr"])
                pr_state = state["prs"].get(pr_key, {})
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
                attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
                # Issue #1439: a PR whose consecutive turn-limit miss streak
                # has reached the backstop cap escalates with a distinct
                # reason so the diagnosis points at "reviewer ran out of
                # navigation budget on a monolith" rather than the generic
                # "every reviewer died without a verdict". Both reasons route
                # to ``agent:human-needed`` via the same mechanical edge below.
                _turn_limit_streak = int(pr_state.get("review_turn_limit_miss_streak", 0))
                _escalation_reason: str | None = None
                if (
                    max_turn_limit_misses > 0
                    and _turn_limit_streak >= max_turn_limit_misses
                    and pr_state.get("status") != "escalated"
                ):
                    _escalation_reason = "max_consecutive_turn_limit_misses_exceeded"
                elif attempt_count >= max_attempts and pr_state.get("status") != "escalated":
                    _escalation_reason = "max_review_dispatch_attempts_exceeded"
                if _escalation_reason is not None:
                    issue_num = pr_state.get("issue_number") or c.get("issue")
                    state = _escalate_issue(
                        state,
                        issue_num,
                        reason=_escalation_reason,
                        reason_class="mechanical",
                        pr_number=int(c["pr"]),
                        pr_extra={
                            "review_dispatch_status": "review_dispatch_failed",
                            "review_dispatch_failed_at": utc_now(),
                            "review_dispatch_pending_at": None,
                            "review_dispatched_at": None,
                            "reviewer_pid": None,
                            "reviewer_process_start_time": None,
                        },
                    )
                    state = self.write_gate.append_event(
                        state,
                        "review_dispatch_escalated",
                        {
                            "pr_number": c["pr"],
                            "issue_number": issue_num,
                            "attempt_count": attempt_count,
                            "reason": _escalation_reason,
                            # Issue #1439: surface the streak that drove the
                            # turn-limit backstop escalation for diagnosis.
                            "turn_limit_miss_streak": _turn_limit_streak,
                        },
                    )
                    changed = True
                    escalated_for_labels.append((int(c["pr"]), issue_num))
            if changed:
                self.write_gate.save_state(state)

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
            # Issue #1266: every producer feeding escalated_for_labels above
            # escalates with reason_class="mechanical" (max_review_dispatch_
            # attempts_exceeded is a process-attempt-cap limit, never a
            # judgment call), so the edge is resolved once for the batch.
            escalated_edge = _escalation_edge("escalated", "mechanical")
            escalated_label_outcomes: list[tuple[int, dict[str, Any] | None]] = []
            for _pr_num, issue_num in escalated_for_labels:
                if issue_num is None:
                    continue
                result = self.write_gate.transition(
                    self.gh, self.config.labels, int(issue_num), escalated_edge
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    escalated_label_outcomes.append(
                        (
                            int(issue_num),
                            {
                                "edge": escalated_edge,
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
                    self.write_gate.save_state(state)

        # Issue #1497: route merge-conflicting candidates to rework outside the
        # state lock. ``_route_janitor_gate_failure_to_rework`` acquires its own
        # ``state_lock`` and calls ``transition()`` (GitHub API), so it cannot
        # run inside the critical section above. This mirrors ``review()``'s
        # janitor-gate merge-conflict path exactly — same ``reason``,
        # same ``attempts_key``, same ``max_attempts`` cap, same escalation to
        # ``agent:human-needed`` when the cap is exceeded — so a CONFLICTING PR
        # discovered by either path converges to the same state. The method is
        # idempotent: if a rework is already pending it returns ``None`` (no-op),
        # and if the lane's own cap already escalated it returns ``None`` too.
        # Gated by dry_run: --dry-run must not perform live GitHub label
        # mutations or state.json writes (same gate as the escalation label
        # edge above).
        merge_conflict_results: list[dict[str, Any]] = []
        if merge_conflict_routed and not self.dry_run:
            for c in merge_conflict_routed:
                pr_number = int(c["pr"])
                issue_num = c.get("issue")
                if issue_num is None:
                    continue
                pr = self.gh.pr_view(pr_number)
                if pr is None:
                    continue
                # Re-check mergeable from the fresh pr_view: pr_list's
                # ``mergeable`` can be UNKNOWN (GitHub computes it
                # asynchronously), and the conflict may have resolved between
                # the queue build and now. ``mergeStateStatus == "DIRTY"`` from
                # pr_list is already a reliable signal, but pr_view's
                # ``mergeable`` is the authoritative reading.
                if not self._is_merge_conflict(pr):
                    continue
                routed = self._route_janitor_gate_failure_to_rework(
                    pr,
                    issue_num,
                    attempts_key="conflict_rework_attempts",
                    max_attempts=self.config.review.max_conflict_rework_attempts,
                    reason="merge_conflict",
                    router=self._request_merge_conflict_rework,
                )
                if routed is not None:
                    merge_conflict_results.append(
                        {
                            "pr": pr_number,
                            "issue": issue_num,
                            "routed_to_rework": routed.data.get("routed_to_rework", False),
                            "escalated": routed.data.get("escalated", False),
                            "rescue_dispatched": routed.data.get("rescue_dispatched", False),
                        }
                    )

        # Claim the selected PRs as pending before launching. This is the only
        # place that writes review_dispatch_pending; the upgrade happens after
        # each launch so a crash between claim and upgrade is recoverable via
        # the stale-claim timeout. Named distinctly from `now`/`resolved_now`
        # above (issue #828's injectable clock) -- this is a write-time stamp,
        # not a staleness-comparison read, and utc_now() returns a formatted
        # string rather than a datetime.
        claim_stamp = utc_now()
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            review_effort_assignments: list[dict[str, Any]] = []
            # Single source of truth for the review_effort experiment arm:
            # resolved ONCE here at claim time and threaded through to the
            # launch call below via resolved_review_effort, so the launch
            # uses exactly this value instead of re-deriving it (agreement
            # by construction, not by convention).
            resolved_review_efforts: dict[int, str] = {}
            # Issue #1439: resolve the structure-aware turn cap ONCE here at
            # claim time (mirroring resolved_review_efforts) and thread it
            # through to the launch call below, so the cap the reviewer runs
            # with is the cap recorded on the claim -- not a re-derivation that
            # could diverge if the miss streak changes between claim and launch.
            resolved_review_turn_caps: dict[int, int] = {}
            for candidate in selected:
                pr_number = candidate["pr"]
                pr_state = state["prs"].get(str(pr_number), {})
                attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
                review_effort_used, review_effort_arm = resolve_review_effort(
                    pr_number, self.config.reviewer, self.config.claude_code
                )
                resolved_review_efforts[pr_number] = review_effort_used
                # Issue #1439: read the structure multiplier stamped into the
                # packet at build time (review()) and the per-PR turn-limit
                # miss streak, then resolve the final cap. A missing packet
                # (e.g. a pre-#1439 packet) yields multiplier 1 -- the base
                # cap, preserving the pre-fix flat budget.
                _structure_mult = self._read_packet_turn_cap_multiplier(pr_number)
                _miss_streak = int(pr_state.get("review_turn_limit_miss_streak", 0))
                _resolved_cap = resolve_review_turn_cap(
                    self.config.review_dispatch.review_max_turns,
                    _structure_mult,
                    _miss_streak,
                    self.config.review_dispatch,
                )
                resolved_review_turn_caps[pr_number] = _resolved_cap
                state["prs"][str(pr_number)] = {
                    **pr_state,
                    "number": pr_number,
                    "issue_number": candidate["issue"],
                    "review_dispatch_status": "review_dispatch_pending",
                    "review_dispatch_pending_at": claim_stamp,
                    "review_dispatched_at": None,
                    "review_dispatch_failed_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                    "review_dispatch_attempt_count": attempt_count + 1,
                    "review_turn_limit_summary_posted": False,
                    "review_miss_summary_posted": False,
                    "review_effort_arm": review_effort_arm,
                    "review_effort_used": review_effort_used,
                    # Issue #1439: record the resolved cap on the dispatch
                    # record so the launch and any post-mortem read the same
                    # value without re-deriving it.
                    "review_turn_cap": _resolved_cap,
                }
                review_effort_assignments.append(
                    {
                        "pr_number": pr_number,
                        "review_effort_arm": review_effort_arm,
                        "review_effort_used": review_effort_used,
                        "review_turn_cap": _resolved_cap,
                    }
                )
            if selected:
                state = self.write_gate.append_event(
                    state,
                    "review_dispatch_claim",
                    {
                        "pr_numbers": [c["pr"] for c in selected],
                        "count": len(selected),
                        "review_effort_assignments": review_effort_assignments,
                    },
                )
            self.write_gate.save_state(state)

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
        # Issue #1513: which harness launches reviewers is now a config knob,
        # not always claude-code. ``_adapter_settings`` is the existing
        # single point that resolves an adapter-specific worker_env (and, for
        # "api", the ApiWorkerConfig) -- reused here rather than
        # re-deriving that resolution for reviewer dispatch. Its
        # worker_model/venv_source fields are NOT reused: both are tuned for
        # worker dispatch's routed-fallback case and venv_source is moot for
        # a review checkout (create_review_checkout never materializes a
        # venv), so the reviewer's model is resolved from
        # ``self.config.reviewer.model`` below instead, exactly as before.
        reviewer_harness = self.config.reviewer.harness
        reviewer_adapter_settings = self._adapter_settings(adapter=reviewer_harness)
        reviewer_launcher = _REVIEW_LAUNCHERS.get(reviewer_harness)
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
                # Reviewers never use worktrees_dir/venv_source — those key
                # the worker's branch-slug worktree, which create_review_checkout
                # (routed to via review=True + head_sha) never touches. Only
                # repo_root/sessions_dir/env/materialize_dirs/review/head_sha
                # are meaningful for a reviewer launch. The harness's own
                # worker-tuning command-template override (e.g.
                # ``claude_code.command``/``devin.shell_command``) is
                # deliberately NOT forwarded here: each launch function
                # hard-pins its own read-only command template for
                # ``review=True`` regardless of what is passed for
                # ``command_template``, so passing one through would be
                # misleading dead code (PR #397 round-2 review, extended to
                # devin-shell by issue #1513).
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
                if reviewer_launcher is None:
                    # Unreachable in production: ReviewerRoleConfig.__post_init__
                    # already restricts reviewer.harness to
                    # harnesses.REVIEWER_HARNESSES, which _REVIEW_LAUNCHERS'
                    # keys are asserted equal to at import time. Guarded
                    # anyway so a config object built directly (bypassing
                    # validation, e.g. in a test) fails as a per-PR error
                    # value rather than a raise.
                    failed.append(
                        {
                            "pr": pr_number,
                            "error": f"unsupported reviewer harness: {reviewer_harness!r}",
                        }
                    )
                    continue

                record = reviewer_launcher(
                    pr_number=pr_number,
                    branch=branch,
                    prompt_path=prompt_path,
                    prompt_text=prompt_text,
                    head_sha=head_sha,
                    repo_root=self.repo_root,
                    reviews_dir=reviews_dir,
                    config=self.config,
                    worker_env=reviewer_adapter_settings.worker_env,
                    materialize_dirs=self.config.dispatch.materialize_dirs,
                    # The review_effort experiment arm was already resolved
                    # once, above, at claim time (and persisted to state/
                    # telemetry) -- pass it through so the launcher uses this
                    # value directly instead of re-resolving it. Ignored by
                    # the devin-shell launcher (no such CLI concept there).
                    resolved_review_effort=resolved_review_efforts.get(pr_number),
                    # Issue #1439: structure-aware turn cap resolved at claim
                    # time from the packet's structure multiplier + the per-PR
                    # miss streak. Overrides the flat ``review_max_turns`` so a
                    # PR threading a monolith gets a raised budget and a
                    # turn-limit miss escalates the next dispatch's cap.
                    # Ignored by the devin-shell launcher (no such CLI concept
                    # there).
                    max_turns_override=resolved_review_turn_caps.get(pr_number),
                    # Issue TBD (role-config Phase 1): without this, every
                    # claude-code reviewer launch would fall back to
                    # launch_claude_worker's own default model resolution,
                    # which is resolved_config.worker.model (see
                    # _apply_model_pin's caller in claude_code.py) -- so a
                    # worker/reviewer split configuration (different
                    # worker.model vs reviewer.model) would resolve correctly
                    # on the config object but never actually change which
                    # model the reviewer launches with. Passing
                    # resolved_config.reviewer.model explicitly here as
                    # model_override is the single enforcement point (used
                    # verbatim by the devin-shell launcher too; ignored by
                    # the api launcher, which always pins the provider's own
                    # model).
                    model_override=self.config.reviewer.model or None,
                    api_worker_config=reviewer_adapter_settings.api_worker_config,
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
                #
                # Routed through self.write_gate (issue #1264, W6 PR2) rather
                # than a bare append_event: this whole block sits under
                # ``with state_lock(...)`` with no local dry_run check of its
                # own -- the only thing keeping it from running under
                # dry_run=True today is the exhaustive early return several
                # hundred lines above, in a different branch of this same
                # method. Gate-internal checking makes that protection
                # structural instead of relying on the distant guard staying
                # correct (R7).
                state = self.write_gate.append_event(
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
                )

            state = self.write_gate.append_event(
                state,
                "review_dispatch",
                {
                    "launched": [x["pr"] for x in launched],
                    "failed": [x["pr"] for x in failed],
                    "quota_hit": quota_hit,
                },
            )
            self.write_gate.save_state(state)

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
            "skipped_empty_diff": skipped_empty_diff,
            "skipped_already_approved": skipped_already_approved,
            "merge_conflict_results": merge_conflict_results,
            "recorded_verdicts": recorded_verdicts,
            "missed_verdicts": missed_verdicts,
            "reconciled_verdicts": reconciled_verdicts,
            "rescue_review_results": rescue_review_results,
            "escalated_labels_repaired": repaired_labels,
        }
        data.update(local_cap.report_fields())
        return CommandResult(ok, message, data)

    # Re-arm field sets live in unescalate_reset_fields.py (extracted under the
    # #1442 ratchet); the aliases keep ``self._...`` call sites and tests intact.
    _UNESCALATE_PR_RESET_FIELDS = UNESCALATE_PR_RESET_FIELDS
    _UNESCALATE_ISSUE_RESET_FIELDS = UNESCALATE_ISSUE_RESET_FIELDS
    _REWORK_BUDGET_RESET_BY_ESCALATION_REASON = REWORK_BUDGET_RESET_BY_ESCALATION_REASON

    # Deliberately NOT @_guard_state_lock: merge_check takes no state lock, and
    # the guard's contract is to return a *successful* skip (ok=True) when the
    # lock is held. On an authorization preflight that would be fail-open —
    # "cannot tell" rendered as "yes". Keep this method lock-free and pure.
    @_guard_state_lock
    def merge_ready(
        self,
        pr_number: int,
        *,
        merge: bool | None = None,
        merge_train_head: int | None = None,
    ) -> CommandResult:
        # Issue #614: dry-run short-circuit. Under --dry-run the GitHub
        # client's synthetic stubs make ``merge_pr`` / ``add_pr_label`` appear
        # to succeed without raising, which the real path below interprets as
        # proof of a merge and durably records to state.json — stranding the PR
        # in a terminal "merged" or "mergequeue" status that never actually
        # happened. The gate must sit above the ``state_lock`` (the idempotency
        # short-circuit below conditionally writes) and return the computed
        # readiness verdict without persisting, mirroring ``_dispatch_impl``'s
        # dry-run gate. Gating lower — after the idempotency block but before
        # the merge — would drop into the shared failed-attempt-alarm block
        # and increment ``consecutive_failed_merge_attempts``, advancing the
        # state machine instead of merely previewing it.
        if self.dry_run:
            return self._merge_ready_dry_run(
                pr_number, merge=merge, merge_train_head=merge_train_head
            )
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
        # Aviator MergeQueue silent-revert detector (issue #823): add_pr_label's
        # boolean return only proves the POST succeeded, not that the label
        # survived -- Aviator can accept the mergequeue label and then
        # asynchronously strip it 2-3 seconds later (draft PR, a failing
        # required check, base mismatch, paused queue, ...), and every such
        # rejection was previously indistinguishable from success. Detected
        # cross-pass, never same-pass: existing_pr_state is the PRIOR pass's
        # persisted status (loaded above, before this pass's live fetch) and
        # label_names(pr) is THIS pass's live labels. A same-pass re-read
        # would race Aviator's 2-3s revert window and could still observe the
        # label present.
        mergequeue_label_reverted = bool(
            self.config.auto_merge.mergequeue_label
            and existing_pr_state.get("status") == "mergequeue"
            and self.config.auto_merge.mergequeue_label not in label_names(pr)
        )
        # Issue #1402: distinguish reconcile's own cooperative self-revocation
        # (stale-head approval, pending carry-forward re-validation -- the #819
        # gap fix) from Aviator's #823 silent rejection. reconcile records
        # ``mergequeue_revoked_reason`` in ``state["prs"][n]`` when IT removes
        # the label; its absence means the label disappeared for some other
        # reason (Aviator stripped it, or a pre-#1402 reconcile pass that did
        # not record the reason). Only ``"stale_head_pending_carry_forward"``
        # is a self-revocation -- ``"not_approved"`` is a genuine revocation
        # whose counter-increment path is already suppressed by ``can_merge``
        # being False (the decision is not ``"approved"``), so it does not
        # need the same exclusion.
        mergequeue_self_revoked_stale_head = bool(
            mergequeue_label_reverted
            and existing_pr_state.get("mergequeue_revoked_reason")
            == "stale_head_pending_carry_forward"
        )
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
            branch_issue_validator=self._make_branch_issue_validator(),
        )
        decision = self._review_decision(pr_number)
        approved = decision.get("decision") == "approved"
        sync_failed = False
        merge_conflict = False
        merge_conflict_routed = False
        merge_conflict_escalated = False
        check_failure_routed = False
        cross_pr_revert_detected = False
        cross_pr_revert_routed = False
        cross_pr_revert_undetermined = False
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
                # Escalation gate (issue #833). merge_ready()'s escalation
                # handling is per-route by design, not a single top-of-function
                # gate: the readiness-no-CI-stall (below), check-failure, and
                # cross-PR-revert lanes already exclude "escalated"; the
                # merge-conflict-rework lane deliberately does NOT (issue #776
                # -- an unrelated-cause escalation must not permanently wall
                # off a PR from its own capped remediation; real corpus:
                # issues #592/#648/#606). This head_moved branch had NO
                # escalation awareness at all: it unconditionally re-applied
                # the "review_started" label transition and rewrote status to
                # "reviewing" on every ~30 min loop pass, even for a PR
                # already flagged agent:human-needed -- a livelock, not a
                # remediation, because nothing was waiting on the re-review it
                # kept re-requesting (real corpus: issue #602 / PR #679).
                # Mirrors review()'s entry gate (which uses the same
                # ``_escalation_flags`` predicate) but scoped to just these two
                # mutations, not the whole function, so the merge-conflict-
                # rework lane a few lines below keeps working.
                #
                # State is re-read here rather than reusing a flag computed at
                # function entry: ``_check_carry_forward``/
                # ``_update_approval_head`` above, and the merge-conflict-
                # rework dispatch further below, can each write a new status
                # for this PR/issue mid-call, so only a read taken at this
                # exact point is trustworthy.
                escalation_state = load_state_locked(self.paths.state_file)
                pr_escalated, issue_escalated = _escalation_flags(
                    escalation_state.get("prs", {}).get(str(pr_number), {}),
                    escalation_state.get("issues", {}).get(str(issue_number), {})
                    if issue_number is not None
                    else None,
                )
                escalated = pr_escalated or issue_escalated
                # Issue #868: a second, independent call site that stamps
                # "reviewing"/agent:reviewing — separate from review()'s own
                # packet-generation path, which gates the same way. Disabled
                # dispatch means nothing services "reviewing" here either, so
                # the state/label stamp is gated on the same flag. The
                # head_moved bookkeeping below (state field + event) stays
                # unconditional: it's read only by this function's own next
                # pass (via _check_carry_forward/reviewed_head_sha
                # comparisons), never jointly with status=="reviewing", so
                # gating just the reviewing stamp doesn't orphan it.
                dispatch_disabled = not self.config.review_dispatch.enabled
                if not escalated:
                    if issue_number is not None and not dispatch_disabled:
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
                            **({} if dispatch_disabled else {"status": "reviewing"}),
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
                        "escalated": escalated,
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
                # "blocked" is a human reviewer verdict (record_review's
                # decision == "blocked"), not a cap escalation: transition()
                # has no source-state validation, so rerouting it to
                # rework_requested would silently strip that verdict and hand
                # the issue back into the automated pipeline behind the
                # human's back. Leave it alone; a human must move it out.
                #
                # "escalated" is deliberately NOT excluded here (issue #776):
                # it used to be, which let an escalation for ANY reason --
                # e.g. a dead request-changes-fix worker exhausting the
                # unrelated watchdog redispatch cap -- permanently wall off a
                # PR that separately develops a merge conflict, with no path
                # back short of a human running `charlie unescalate` (real
                # corpus: issues #592/#648/#606). The unified conflict-rework
                # dispatch below (_route_janitor_gate_failure_to_rework)
                # re-derives its own decision from conflict_rework_attempts,
                # a cap left untouched by an escalation from a different lane
                # -- but already exhausted, so still correctly refused, if
                # THIS lane is what escalated the issue previously. Falling
                # through is safe: it is the same merge_conflict=True /
                # sync_failed=True path already taken today by a fresh,
                # not-yet-escalated conflict before the failed-attempt-alarm
                # threshold below is reached.
                if issue_status == "blocked":
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
            #
            # Whether base currency gates THIS merge is `require_current_base OR
            # protection.strict` -- protection may raise the requirement, never
            # lower it below what the operator configured (issue #875). Issue
            # #812 made this purely protection-derived, which meant a repo
            # running `strict: false` (as this one does) disabled its own
            # merge gate and merged stale-base PRs whose merged tree was never
            # tested. The broadcast sweep still uses the protection-only
            # `_is_base_freshness_required` -- that write costs N CI cycles
            # across every open PR, where this one costs a single cycle for the
            # PR actually merging. See _is_base_currency_gated.
            #
            # This single flag intentionally drives BOTH the pr_update_branch
            # write below and the deferral gate further down. Decoupling them --
            # gating merges while leaving the write suppressed -- would deadlock:
            # a stale PR would defer forever with nothing to make it current.
            # `update_branch_strategy == "off"` forces the whole thing off
            # regardless: with no sync mechanism at all, requiring currency would
            # be that same inescapable deadlock -- the hazard __post_init__
            # already blocks for the config-only case (require_current_base=True
            # + strategy=off).
            base_current: bool | None = None
            base_currency_gated = False
            if not sync_failed and update_branch_strategy != "off":
                base_currency_gated = self._is_base_currency_gated(
                    pr.get("baseRefName") or self.config.runners.default_branch
                )
            if (
                not sync_failed
                and base_currency_gated
                and update_branch_strategy in {"front_of_train", "broadcast"}
            ):
                base_current = self._is_base_current(pr)
                # Aviator MergeQueue handoff (task #10): once this PR has
                # already been parked in Aviator's queue (state status
                # "mergequeue" from a prior merge_ready pass), Aviator owns
                # rebasing it. Calling pr_update_branch here on every
                # subsequent poll would race Aviator's own rebase as a second
                # writer on the same ref — this repo's live config is
                # broadcast, so every poll would otherwise attempt it. The
                # base-freshness *read* above still runs (it only feeds the
                # deferral gate below); only the write is skipped. See the
                # merge-train exclusion in _merge_train_candidates for the
                # sibling half of this fix.
                already_in_mergequeue = existing_pr_state.get("status") == "mergequeue"
                if not already_in_mergequeue and self._should_update_pr_branch(pr, base_current):
                    if self.gh.pr_update_branch(pr_number):
                        new_head = self._verify_synced_head(pr_number, live_head_sha)
                        if new_head is None:
                            sync_failed = True
                        elif new_head != live_head_sha:
                            self._update_approval_head(
                                pr_number,
                                decision,
                                new_head,
                                old_head=live_head_sha,
                                issue_number=issue_number,
                            )
                            pr = self.gh.pr_view(pr_number) or pr
                            decision = self._review_decision(pr_number)
                        else:
                            # Already up-to-date; nothing to do.
                            pass
                    else:
                        sync_failed = True
            # merge-base freshness gate: mergeStateStatus can lag, so verify
            # ancestry with the GitHub compare API before merging. Skipped
            # entirely when base freshness is not required (issue #812).
            if not sync_failed and base_currency_gated:
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
            #
            # The gate returns an explicit verdict (issue #1068): REVERT_DETECTED
            # blocks and routes to rework; UNDETERMINED (the local git history
            # needed to decide was unavailable) blocks the merge but does NOT
            # route to rework — an unverified gate fails closed rather than
            # folding into the same None as "verified clean", which silently
            # disabled the gate and could let a real cross-PR revert merge
            # unflagged. Only CLEAN lets the merge advance.
            #
            # ``blocks_merge`` is the single enforcement point for sync_failed
            # (both REVERT_DETECTED and UNDETERMINED block); the status chain
            # below is routing/dispatch only — which flag to set and whether to
            # route to rework — not a second enforcement of the block.
            if not sync_failed:
                cross_pr_revert_verdict = detect_cross_pr_revert(pr, self.repo_root)
                cross_pr_revert_reason = cross_pr_revert_verdict.reason
                if cross_pr_revert_verdict.blocks_merge:
                    sync_failed = True
                    if cross_pr_revert_verdict.status is CrossPrRevertStatus.REVERT_DETECTED:
                        cross_pr_revert_detected = True
                        if issue_number is not None:
                            state = load_state_locked(self.paths.state_file)
                            issue_state = state["issues"].get(str(issue_number), {})
                            issue_status = issue_state.get("status")
                            if issue_status not in _REWORK_ALREADY_ROUTED_STATUSES:
                                cross_pr_revert_routed = True
                                rework_label_error = self._request_cross_pr_revert_rework(
                                    pr, issue_number, decision, cross_pr_revert_reason
                                )
                    elif cross_pr_revert_verdict.status is CrossPrRevertStatus.UNDETERMINED:
                        # Fail closed: refuse to merge on an unverified gate. This
                        # is not a detected revert, so no rework routing — the PR
                        # is simply held until the gate can verify (issue #1068).
                        cross_pr_revert_undetermined = True
        checks = self.gh.pr_checks(pr_number)
        checks_unavailable = checks is None

        if checks_unavailable:
            # gh pr checks command itself failed. Treat every required check as
            # unavailable (not missing) and do not merge; callers/loop will count
            # this as an infrastructure error.
            summary = summarize_checks(None, self.config.auto_merge.required_checks)
            enriched_checks: list[dict[str, Any]] = []
        else:
            # Issue #1383: use the shared data-boundary enrichment so the
            # merge path classifies infra_blocked checks identically to the
            # review path. The old inline enrichment reclassified a FAILURE
            # check with an infra signature to ``INFRA_FAILURE`` (routing it
            # to ``CheckSummary.infra_failed``); the shared helper rewrites
            # the same population to ``INFRA_BLOCKED`` instead, so those
            # checks now land in ``CheckSummary.infra_blocked`` rather than
            # ``infra_failed``. Both buckets block merge (``CheckSummary.ready``
            # is False for either), so the merge gate is unchanged -- only the
            # bucket/failure-message differs. ``infra_failed`` is NOT emptied
            # for its original population: ``_enrich_checks_infra_blocked``
            # only touches ``FAILURE``-conclusion checks, so genuine
            # ``CANCELLED``/``INFRA_FAILURE``/``TIMED_OUT`` checks still route
            # to ``infra_failed`` exactly as before (the #841 auto-rerun path
            # in ``review()`` is fed by ``run_janitor``, which sees raw checks
            # -- it never enriched FAILURE to INFRA_FAILURE pre-#1383, so the
            # #1383 change reroutes zero-step FAILURE from
            # ``is_check_failure_block`` (rework) to ``is_infra_blocked_block``
            # (hold), not from ``is_infra_failure_block``).
            enriched_checks = self._enrich_checks_infra_blocked(
                checks, self.config.auto_merge.required_checks
            )
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
                    "containment_check",  # event-consumer: audit-only -- report-only per issue directive, not a blocking gate
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

        # Issue #1060: derive ``can_merge`` from a single dict of gate inputs
        # and persist that same dict in the ``merge_ready`` event below. The
        # gate and the record now read from one source, so a future added
        # condition cannot silently fall out of the persisted payload the way
        # ``mergequeue_label_applied`` did (it existed only in the in-memory
        # return dict, never in events.db, so a query for it was vacuously 0
        # for every PR that has ever existed). A hand-maintained list of keys
        # drifts from the expression it describes; this dict *is* the
        # expression.
        merge_gate_inputs = {
            "summary_ready": summary.ready,
            "approved": approved,
            "require_approved_review": self.config.auto_merge.require_approved_review,
            "sync_failed": sync_failed,
        }
        can_merge = (
            merge_gate_inputs["summary_ready"]
            and (merge_gate_inputs["approved"] or not merge_gate_inputs["require_approved_review"])
            and not merge_gate_inputs["sync_failed"]
        )
        # Issue #840: escalation gate on the merge-execution block. An
        # approved, green, conflict-free PR whose linked issue (or the PR
        # itself) is escalated (status == "escalated", i.e. a human has been
        # asked to intervene via agent:human-needed) must NOT be actually
        # merged while that flag is up -- even when the specific reason for
        # escalation is unrelated to mergeability (e.g. a dead request-
        # changes-fix worker exhausting the watchdog's redispatch cap; real
        # corpus: issues #592/#648/#606). Silently completing an irreversible
        # merge while escalated defeats the purpose of raising the flag.
        #
        # This is a plain gate on the merge-execution block ONLY -- it does
        # NOT modify ``can_merge``, so the failed-attempt-alarm counter block
        # below still sees ``can_merge`` True and zeroes the streak via the
        # ``elif can_merge:`` branch (a green pass held by escalation is not
        # a failed pass). Modifying ``can_merge`` itself would make
        # ``approved and not can_merge`` true, incrementing the counter every
        # pass and eventually firing a spurious merge_attempt_alarm for a PR
        # that isn't failing to merge for a mergeability reason -- a
        # diagnostic regression in the same spirit as the unbounded-counter
        # bug issue #777(b) fixed for PR #679.
        #
        # State is re-read here rather than reusing a flag computed at
        # function entry: the carry-forward / head_moved / merge-conflict
        # branches above can each write a new status for this PR/issue
        # mid-call, so only a read taken at this exact point is trustworthy
        # (same pattern as the head_moved branch's escalation read above).
        # Issue #1598: human-merge-labels gate. When the bound issue carries
        # any label in ``dispatch.human_merge_labels``, the PR is never
        # queued or merged by the fleet. Detection (issue fetch,
        # malformed-dict handling, label intersection) is shared with
        # ``_merge_ready_dry_run`` via ``_human_merge_hold_check`` so the two
        # paths cannot drift. The check reads live issue labels at decision
        # time (same source the merge-hold check below uses), so an operator
        # adding or removing the label mid-flight takes effect on the next
        # pass. If the issue was previously escalated with
        # ``reason_class="policy"`` but the label has been removed, the
        # escalation is cleared here so the PR becomes fleet-mergeable again
        # — ``charlie unescalate`` is not required.
        human_merge_hold, human_merge_check_unavailable = self._human_merge_hold_check(
            issue_number
        )
        # De-escalation: the operator removed the human-merge label after a
        # previous hand-off. Clear the ``"policy"`` escalation so the merge
        # block below is not gated by ``escalated_merge_hold`` and the PR can
        # be fleet-merged. Skipped entirely when human_merge_labels is
        # unconfigured or the PR has no bound issue (zero overhead, matching
        # the pre-refactor guard), and when the check was unavailable (no
        # reliable signal to de-escalate on).
        if (
            self.config.dispatch.human_merge_labels
            and issue_number is not None
            and not human_merge_hold
            and not human_merge_check_unavailable
        ):
            _hm_snap = load_state_locked(self.paths.state_file)
            _hm_issue_entry = _hm_snap.get("issues", {}).get(str(issue_number), {})
            if (
                isinstance(_hm_issue_entry, dict)
                and _hm_issue_entry.get("status") == "escalated"
                and _hm_issue_entry.get("reason_class") == "policy"
            ):
                with state_lock(self.paths.state_file):
                    _hm_state = load_state(self.paths.state_file)
                    _hm_entry = _hm_state["issues"].get(str(issue_number), {})
                    if (
                        isinstance(_hm_entry, dict)
                        and _hm_entry.get("status") == "escalated"
                        and _hm_entry.get("reason_class") == "policy"
                    ):
                        _hm_entry["status"] = PASSIVE_OPEN_STATUS
                        clear_escalation(_hm_entry)
                        _hm_entry.pop("label_error", None)
                        _hm_state["issues"][str(issue_number)] = _hm_entry
                        clear_escalation_on_issue_prs(_hm_state, issue_number)
                        _reset_linked_pr_status_to_passive_open(_hm_state, pr_number)
                        _hm_state = self._record_event(
                            _hm_state,
                            "human_merge_label_removed",  # event-consumer: audit-only -- records the policy de-escalation when an operator removes a human-merge label (issue #1598); consumed by tests/test_human_merge_labels_1598.py.
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                            },
                        )
                        save_state(self.paths.state_file, _hm_state)
        _escalation_snap = load_state_locked(self.paths.state_file)
        _pr_escalated, _issue_escalated = _escalation_flags(
            _escalation_snap.get("prs", {}).get(str(pr_number), {}),
            _escalation_snap.get("issues", {}).get(str(issue_number), {})
            if issue_number is not None
            else None,
        )
        escalated_merge_hold = can_merge and (_pr_escalated or _issue_escalated)
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        # Issue #747: timestamp of the fleet's own merge_pr success (None until
        # the direct-merge branch runs and succeeds). Used for the
        # ``merge_succeeded`` event payload and the ``merged_at`` state field.
        _merged_at: str | None = None
        branch_deleted: bool | None = None
        update_results: list[dict[str, Any]] | None = None
        cancel_results: dict[str, Any] | None = None
        mergequeue_label_applied: bool | None = None
        merge_hold: bool = False
        merge_hold_check_unavailable: bool = False
        if (
            can_merge
            and should_merge
            and not escalated_merge_hold
            and not human_merge_hold
            and not human_merge_check_unavailable
        ):
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
                        _existing_pr_entry = state["prs"].get(str(pr_number), {})
                        _pr_update: dict[str, Any] = {
                            **_existing_pr_entry,
                            "number": pr_number,
                            "issue_number": issue_number,
                            "status": "mergequeue",
                            # Issue #1402: clear the self-revocation reason now
                            # that the label has been re-applied (carry-forward
                            # re-validated the approval), so a subsequent
                            # Aviator #823 silent rejection is not misread as a
                            # stale self-revocation. Setting to None (rather
                            # than deleting the key) keeps the entry shape
                            # stable for callers that .get() it defensively.
                            "mergequeue_revoked_reason": None,
                        }
                        # Issue #823 ordering hazard: do NOT zero the counter
                        # when this re-add is recovering from a cross-pass
                        # revert (mergequeue_label_reverted). This write runs
                        # BEFORE mergequeue_handoff_failed is computed and its
                        # increment is persisted further down in this same
                        # pass -- unconditionally zeroing here would hand that
                        # later increment a false 0 baseline every single
                        # reverted pass, so the counter would land on 1
                        # forever instead of accumulating, making the
                        # escalation this issue adds permanently unreachable
                        # for any threshold > 1. Verified empirically: without
                        # this guard a 3-pass simulation (revert, revert,
                        # revert) produces 1, 1, 1; with it, 1, 2, 3.
                        if not mergequeue_label_reverted:
                            _pr_update["consecutive_failed_merge_attempts"] = 0
                        state["prs"][str(pr_number)] = _pr_update
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
                # Issue #747: stamp ``merged_at`` at the irreversible step so
                # merge throughput/latency are computable from state.json.
                # Existing merged entries are not back-dated — this only fires
                # on a genuine non-merged -> merged transition (merge_pr just
                # succeeded), so the timestamp is the real merge time.
                _merged_at = utc_now()
                with state_lock(self.paths.state_file):
                    state = load_state(self.paths.state_file)
                    state["prs"][str(pr_number)] = {
                        **state["prs"].get(str(pr_number), {}),
                        "number": pr_number,
                        "issue_number": issue_number,
                        "status": "merged",
                        "merged": True,
                        "merged_at": _merged_at,
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
        # Issue #1598: human-merge hand-off. When the bound issue carries a
        # configured human_merge_labels label and the PR is merge-ready
        # (approved, checks green, no conflicts), the fleet does NOT merge or
        # queue it. Instead it transitions the issue to agent:operator-queue
        # via the ``human_merge_required`` edge, escalates with
        # ``reason_class="policy"``, and posts one orchestrator-generated PR
        # comment saying the PR is approved and awaits a human merge. The
        # comment is posted once (tracked via the PR state field
        # ``human_merge_comment_posted``), not every pass. On subsequent
        # passes ``escalated_merge_hold`` is True (the issue is now
        # escalated), so this block does not re-run — the merge block above
        # is also gated by ``escalated_merge_hold`` and stays skipped.
        # ``charlie unescalate`` is not required after the human merges: the
        # existing merged-PR reconcile path closes out the issue as it does
        # today, and the de-escalation block above clears a ``"policy"``
        # escalation if the operator removes the label without merging.
        human_merge_label_error: dict[str, Any] | None = None
        human_merge_comment_posted = False
        if human_merge_hold and can_merge and should_merge and not escalated_merge_hold:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                _existing_pr = state["prs"].get(str(pr_number), {})
                human_merge_comment_posted = bool(_existing_pr.get("human_merge_comment_posted"))
                state = _escalate_issue(
                    state,
                    issue_number,
                    reason="human_merge_required",
                    reason_class="policy",
                    pr_number=pr_number,
                    pr_extra={"human_merge_comment_posted": True},
                )
                state = self._record_event(
                    state,
                    "human_merge_required",  # event-consumer: audit-only -- records the human-merge hand-off (issue #1598); consumed by tests/test_human_merge_labels_1598.py.
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "comment_posted": not human_merge_comment_posted,
                    },
                )
                save_state(self.paths.state_file, state)
            # Transition labels (operator_queue via the human_merge_required
            # edge). Done outside the state lock because transition() makes
            # its own GitHub calls and does not touch state.json.
            if issue_number is not None:
                result = transition(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    "human_merge_required",
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    human_merge_label_error = {
                        "edge": "human_merge_required",
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
            # Post the PR comment once (not every pass).
            if not human_merge_comment_posted and issue_number is not None:
                try:
                    self._comment_pr(
                        pr_number,
                        "This PR is approved and all checks are green, but the linked "
                        "issue carries a human-merge label. The fleet will not auto-merge "
                        "it — a human merge is required.",
                    )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "human_merge comment post failed pr=%d", pr_number, exc_info=True
                    )
        # Aviator MergeQueue handoff (task #10): the label add IS the handoff
        # — a failed add_pr_label must be treated as a genuine unmergeable
        # pass (like an unmet check or a merge conflict), not silently
        # swallowed as best-effort cleanup. can_merge is True here (checks
        # green, approved), so the "approved and not can_merge" alarm gate
        # below would otherwise never fire for this failure mode.
        #
        # Issue #823: folded into the SAME signal (rather than a parallel
        # mechanism) is mergequeue_label_reverted -- an outright POST failure
        # and a POST that succeeded but was silently reverted by Aviator by
        # the next pass are the same underlying failure (the PR never
        # actually made it into Aviator's queue) and must share the same
        # counter-increment/escalation path.
        #
        # The revert term is additionally gated on can_merge: a revert can
        # also be Aviator declining the PR for a genuine reason (e.g. a
        # required check went red), in which case can_merge is False on this
        # pass and the failure is a check failure, not a handoff failure. If
        # mergequeue_handoff_failed were True here it would shadow the
        # check-failure-rework dispatch gate below (`not
        # mergequeue_handoff_failed`), permanently blocking rework for a
        # fixable check failure. The counter still increments in that case
        # via the pre-existing "approved and not can_merge" disjunct in the
        # final counter-persistence block, so escalation is unaffected --
        # only the handoff-specific attribution and the handoff-specific
        # (non-)dispatch of check-failure-rework are.
        mergequeue_handoff_failed = bool(
            self.config.auto_merge.mergequeue_label and mergequeue_label_applied is False
        ) or (mergequeue_label_reverted and can_merge and not mergequeue_self_revoked_stale_head)
        # Conflict-rework dispatch is debounced to the failed-attempt alarm
        # threshold so a single transient/stale CONFLICTING reading does not
        # clobber an approved verdict. Re-read the issue status and the PR
        # attempt counter immediately before dispatch: the preceding checks,
        # diff, and containment work are network-I/O windows long enough for a
        # concurrent pass to have moved the issue into an in-flight or
        # human-terminal state, and the stale `existing_pr_state` snapshot can
        # diverge from the counter the final persistence block will reload
        # (e.g. a carry-forward reset in this same pass). Dispatch outside the
        # final state-lock because _route_janitor_gate_failure_to_rework
        # acquires its own lock.
        #
        # Issues #776/#777: this used to call _request_merge_conflict_rework
        # directly, bypassing conflict_rework_attempts entirely -- the only
        # gate was this threshold check, so a PR could be re-dispatched to
        # rework on every pass forever with no cap, and consecutive_failed_
        # merge_attempts (a diagnostic-only alarm counter, see
        # AutoMergeConfig.failed_attempt_alarm) was the only "cap" a human
        # could point to. It is now routed through the SAME
        # attempts_key-capped wrapper review()'s janitor gate uses, so there
        # is exactly one place in the codebase that increments
        # conflict_rework_attempts and decides dispatch vs. rescue vs.
        # escalation for this lane -- and the attempt is counted (line
        # `attempts = ... + 1` inside that wrapper) BEFORE any subsequent
        # worker-launch failure (e.g. a deterministic worktree_unsafe at
        # actual dispatch time, real corpus: PR #679/issue #602) can
        # escalate, so a launch failure no longer bypasses the cap.
        # "manifest_written"/"blocked" remain pre-checked here because they
        # are merge_ready()-specific concerns the shared wrapper does not
        # know about: manifest_written means an unrelated worker is being
        # launched for this issue right now (preempting it would race that
        # launch), and blocked is a human reviewer verdict that must never be
        # silently routed around. "dispatched"/"dispatch_pending"/
        # "rework_requested" and "escalated" are deliberately NOT
        # pre-excluded any more: the wrapper's own rework_pending check
        # already treats the first three as in-flight (returning None, a
        # no-op here), and its own attempts_key cap -- untouched by an
        # escalation from an unrelated lane, already exhausted if this lane
        # is what escalated the issue -- is what correctly decides
        # "escalated" (issue #776's fix; real corpus: issues #592/#648/#606).
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
            if issue_status not in ("manifest_written", "blocked"):
                new_attempts_for_route = (
                    int(existing_for_route.get("consecutive_failed_merge_attempts", 0)) + 1
                )
                threshold = self.config.auto_merge.failed_attempt_alarm
                if threshold > 0 and new_attempts_for_route >= threshold:
                    routed = self._route_janitor_gate_failure_to_rework(
                        pr,
                        issue_number,
                        attempts_key="conflict_rework_attempts",
                        max_attempts=self.config.review.max_conflict_rework_attempts,
                        reason="merge_conflict",
                        router=self._request_merge_conflict_rework,
                    )
                    if routed is not None:
                        merge_conflict_routed = bool(routed.data.get("routed_to_rework"))
                        merge_conflict_escalated = bool(routed.data.get("escalated"))
                        rework_label_error = routed.data.get("label_error")
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
            # Issue #861: default to PRESERVING the prior count, not resetting
            # to 0. A pending-only pass (checks still in flight) falls through
            # both branches below and previously clobbered an accumulated
            # count with 0 every time, defeating the threshold debounce.
            # Indeterminate passes must leave the counter untouched; only a
            # genuine-success pass (the `elif can_merge:` below) or an
            # explicit merge (handled earlier, before this block re-reads
            # `existing`) may zero it.
            new_attempts = int(existing.get("consecutive_failed_merge_attempts", 0))
            new_stale_base_deferrals = 0
            merge_attempt_alarm = False
            merge_attempt_warning: str | None = None
            if (
                approved and not can_merge and not _is_pending_only(summary)
            ) or mergequeue_handoff_failed:
                new_attempts = int(existing.get("consecutive_failed_merge_attempts", 0)) + 1
                threshold = self.config.auto_merge.failed_attempt_alarm
                # Issue #777(b): this counter previously had no ceiling and
                # grew unbounded across passes (real corpus: PR #679 reached
                # 12 and climbing) because nothing ever reset it once a
                # rework was dispatched and pending. It is not purely
                # diagnostic -- the merge-conflict rework-dispatch block
                # above and the check-failure rework-dispatch block above
                # both read this same counter as a `>= threshold` debounce
                # gating *first*-dispatch timing -- but it does not bound
                # *repeat* dispatch; that functional bound is
                # conflict_rework_attempts/no_op_rework_attempts, enforced by
                # _route_janitor_gate_failure_to_rework above. Clamp at
                # threshold + 1 (not at threshold): clamping AT threshold
                # would make `merge_attempt_alarm` below re-fire every pass
                # once clamped, turning a one-time alarm into a cost spiral;
                # threshold + 1 keeps the alarm's "== threshold" one-shot
                # semantics intact while still giving every trigger sharing
                # this counter (check-failure, cross-PR-revert, mergequeue
                # handoff) a stable ">= threshold" debounce reading forever
                # after the first alarm fires, instead of an ever-growing one.
                if threshold > 0:
                    new_attempts = min(new_attempts, threshold + 1)
                merge_attempt_alarm = threshold > 0 and new_attempts == threshold
                if merge_attempt_alarm:
                    if merge_conflict:
                        pass_str = "pass" if new_attempts == 1 else "passes"
                        if issue_number is None:
                            conflict_detail = "no linked issue, cannot route to rework"
                        elif merge_conflict_escalated:
                            conflict_detail = "conflict-rework cap exhausted; escalated to a human"
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
            elif can_merge:
                # Genuine success: checks are ready (or an explicit merge/
                # mergequeue-handoff already reset the persisted value above,
                # which this re-read of `existing` picked up) and this is not
                # a mergequeue-handoff failure (guaranteed by the `if` above:
                # mergequeue_handoff_failed can only be True when can_merge is
                # True, so if it were True this branch would not be reached).
                # A confirmed-mergeable pass is real positive information, so
                # it is safe -- and correct -- to zero the streak here, unlike
                # the pending-only case above which conveys no information.
                new_attempts = 0
            if (
                approved
                and can_merge
                and merge_output is None
                and not mergequeue_handoff_failed
                and not merge_hold_check_unavailable
                and not human_merge_hold
                and not human_merge_check_unavailable
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
            # Issue #1401: track time-in-mergequeue for the wedge watchdog.
            # ``mergequeue_since`` is the moment the PR entered Aviator's queue
            # at its current head; ``mergequeue_head_sha`` is that head. A head
            # advance (Aviator rebase) resets both, so the watchdog's
            # time-in-queue trigger measures true no-progress dwell, not wall
            # time since the first handoff. Cleared whenever the PR is not
            # currently in mergequeue so a later re-entry starts a fresh window.
            _current_head = pr.get("headRefOid")
            if prs_entry.get("status") == "mergequeue" and _current_head:
                if existing.get("mergequeue_head_sha") == _current_head and existing.get(
                    "mergequeue_since"
                ):
                    prs_entry["mergequeue_since"] = existing["mergequeue_since"]
                    prs_entry["mergequeue_head_sha"] = existing["mergequeue_head_sha"]
                else:
                    prs_entry["mergequeue_since"] = utc_now()
                    prs_entry["mergequeue_head_sha"] = _current_head
            else:
                prs_entry.pop("mergequeue_since", None)
                prs_entry.pop("mergequeue_head_sha", None)
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
                    "human_merge_hold": human_merge_hold,
                    "human_merge_check_unavailable": human_merge_check_unavailable,
                    "cancel_superseded_runs_results": cancel_results,
                    # Issue #1060: persist the Aviator handoff outcome so a
                    # query for it is no longer vacuously 0 for every PR. This
                    # key previously existed only in the in-memory return dict
                    # below, never in events.db.
                    "mergequeue_label_applied": mergequeue_label_applied,
                    # Issue #1060: persist the three gate inputs alongside the
                    # conclusion so a ``can_merge=False`` can be diagnosed from
                    # events.db alone. Spread from the same dict the gate reads
                    # (``merge_gate_inputs`` above) so a future added condition
                    # cannot silently fall out of the record.
                    **merge_gate_inputs,
                },
            )
            # Issue #747: emit a dedicated terminal success event on the
            # fleet's own direct-merge path. ``merge_output`` is truthy only
            # when ``merge_pr`` actually ran and succeeded (it stays None for
            # mergequeue handoffs, deferrals, conflicts, and skipped/deferred
            # passes), so this fires exactly once per real fleet merge and is
            # silent on every other outcome — the negative control the issue
            # requires. ``actor="fleet"`` distinguishes these from
            # externally-merged PRs, which are recorded by
            # ``finalize_externally_merged`` / ``merged_outside_orchestrator``
            # and never emit this kind.
            if merge_output:
                state = self._record_event(
                    state,
                    "merge_succeeded",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "actor": "fleet",
                        "merge_method": self.config.auto_merge.strategy,
                        "merged_at": _merged_at,
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
            "cross_pr_revert_undetermined": cross_pr_revert_undetermined,
            "mergequeue_label_applied": mergequeue_label_applied,
            "merge_hold": merge_hold,
            "merge_hold_check_unavailable": merge_hold_check_unavailable,
            "human_merge_hold": human_merge_hold,
            "human_merge_check_unavailable": human_merge_check_unavailable,
            "human_merge_label_error": human_merge_label_error,
            "escalated_merge_hold": escalated_merge_hold,
            # Issue #1060: surface the gate inputs in the in-memory verdict
            # too, for diagnostic parity with the persisted event.
            **merge_gate_inputs,
        }
        message = "merge readiness evaluated"
        if cross_pr_revert_detected:
            message = f"cross-PR revert detected: {cross_pr_revert_reason}"
        elif cross_pr_revert_undetermined:
            message = (
                f"cross-PR revert gate undetermined (merge blocked, fail-closed): "
                f"{cross_pr_revert_reason}"
            )
        elif checks_unavailable:
            message = "checks unavailable (gh failure)"
        elif escalated_merge_hold:
            message += " (escalated — merge held while agent:human-needed is up)"
        elif human_merge_hold:
            message += " (human-merge label on issue — fleet will not auto-merge)"
        elif human_merge_check_unavailable:
            message += (
                f" (human-merge label check unavailable for issue #{issue_number} — not merged)"
            )
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
    def loop(
        self,
        limit: int | None = None,
        *,
        merge: bool | None = None,
        now: datetime | None = None,
    ) -> CommandResult:
        # ``now`` (issue #822, extended #828) is this pass's injectable clock.
        # ``_loop_body`` forwards it, unresolved, to every cadence-gated lane
        # that samples wall-clock time: dead-session throttle classification
        # (``_classify_dead_sessions_and_update_throttle_state``), the
        # stalled-session reaper (``_detect_and_handle_stalled_sessions``),
        # and the quota-probe/worktree-reclaim/drift-reconcile schedulers
        # (``_maybe_probe_quota_recovery``, ``_maybe_reclaim_worktrees``,
        # ``_maybe_reconcile_drift``). Each callee independently defaults to
        # ``datetime.now(UTC)`` when it receives None, so production behavior
        # is byte-identical when this argument is omitted (as all current
        # production callers do); tests can freeze one ``now`` and assert
        # exact equality instead of a wall-clock-tolerance proximity check.
        return self._loop_impl(limit, merge=merge, now=now)

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

        Issue #1133: ``janitor_blocked`` conflates a durably-stuck population
        (failed checks, merge conflict, body gate, CI-never-created) with a
        transient one -- a brand-new PR whose required checks simply haven't
        reported yet, which self-heals within one CI cycle. The transient
        case is identified structurally by ``is_missing_checks_only_block``
        (the SOLE janitor failure is "Required check(s) missing") combined
        with the absence of a ``ci_run_never_created_head`` marker (which
        would mean CI was confirmed to have never started for this head -- a
        durable condition). Such a PR is NOT dead: it is actively progressing
        and will unblock on the next janitor pass once CI reports. The
        ``escalated`` status stays dead unconditionally.
        """
        issue_entry = state.get("issues", {}).get(str(blocker_number), {})
        if isinstance(issue_entry, dict) and issue_entry.get("status") == "escalated":
            return True
        pr = pr_by_issue.get(blocker_number)
        if pr is not None:
            pr_number = pr.get("number")
            if pr_number is not None:
                pr_state = state.get("prs", {}).get(str(pr_number), {})
                pr_status = pr_state.get("status")
                if pr_status == "escalated":
                    return True
                if pr_status == "janitor_blocked":
                    # Issue #1133: a brand-new PR whose only janitor failure is
                    # "Required check(s) missing" (checks not reported yet) is
                    # transient, not dead -- unless ``ci_run_never_created_head``
                    # is set, which means CI was confirmed to have never started
                    # for this head (the durable population the alert exists for).
                    # Branch on the structured flag, never on failure-message
                    # text (same rule as is_draft_only_block consumers).
                    if pr_state.get("is_missing_checks_only_block") and not pr_state.get(
                        "ci_run_never_created_head"
                    ):
                        return False
                    return True
        return False

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)


# Track 2 Phase B delegation install (#1631, umbrella #1582). Every submodule of
# `charlie_work.orchestration` contributes its top-level `def`s as delegates on
# `OrchestratorApp`; later leaves add submodules there without editing this line.
# At L00 the package is empty, so `_DELEGATE_MODULES` is empty and the install is
# a no-op -- `OrchestratorApp`'s lexical member surface is unchanged.
_DELEGATE_MODULES = discover_delegate_modules(_orchestration)
_install_delegates(OrchestratorApp, _DELEGATE_MODULES)
