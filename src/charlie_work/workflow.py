from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import CLI_NAME
from .adapters import AdapterSettings, SessionRequest, dispatch_sessions
from .checks import CheckSummary, summarize_checks
from .config import CrossFamilyConfig, DETERMINISTIC_ESCALATION_FAILURE_KINDS, OrchestratorConfig
from .fleet_registry import count_fleet_live_sessions, try_acquire_fleet_lock
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .cross_family import (
    CrossFamilyResult,
    extract_head_ref_oid,
    extract_report_body,
    report_body_is_valid,
    run_cross_family_review,
)
from .github import (
    GitHub,
    GitHubError,
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
    check_operator_containment,
    check_test_adequacy,
    run_janitor,
    TestAdequacyFacts,
    TestAdequacyVerdict,
)
from .labels import TransitionOutcome, transition
from .paths import RuntimePaths
from .prompts import render_prompt
from .reconcile import DriftItem, apply_fixes as apply_drift_fixes, detect_drift
from .worktree import inspect_worktree_state, push_branch, resolve_base_branch_name
from .state import (
    append_event,
    is_claim_stale,
    is_throttled,
    load_state,
    load_state_locked,
    save_state,
    set_throttled_until,
    state_lock,
    utc_now,
)
from .process_utils import is_pid_alive, kill_process_tree, sweep_orphan_processes
from .worker import WorkerHealth, WorkerView


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any]


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


def slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "work"


def parse_issue_numbers(only_issues: str) -> list[int]:
    return [int(part) for part in only_issues.replace(" ", "").split(",") if part]


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
        # escalation cap is tracked across passes.
        new_count = _next_inconclusive_probe_deferred_count(w, probe, health)
        update_worker_log_stat(sessions_dir, w, inconclusive_probe_deferred_count=new_count)

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
                    )
                    if defer_until is not None:
                        update_worker_log_stat(sessions_dir, w, rate_limit_defer_until=defer_until)
                        with state_lock(state_file):
                            state = load_state(state_file)
                            state = set_throttled_until(state, defer_until)
                            state = append_event(
                                state,
                                "session_rate_limit_deferred",
                                {
                                    "issue_number": w.issue_number,
                                    "pid": w.pid,
                                    "defer_until": defer_until,
                                },
                            )
                            save_state(state_file, state)
                        continue

            # Kill the process tree (with start-time verification to prevent PID recycling)
            killed_pids = kill_process_tree(w.pid, w.process_start_time)

            # Sweep for orphan processes that survived the tree kill (Windows-only)
            # This catches detached/daemonized processes (e.g., nohup-style background processes)
            orphan_pids = sweep_orphan_processes(w.worktree_path)
            if orphan_pids:
                # Kill detected orphans to prevent them from running rejected code
                for orphan_pid in orphan_pids:
                    _kill_orphan_pid(orphan_pid)
                    killed_pids.append(orphan_pid)

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

            if resolved_failure_kind and throttled_until:
                # A throttle signature was found in the log tail even though
                # the watchdog reaped this worker for stalling — persist the
                # cooldown so the next dispatch pass defers instead of
                # relaunching into the same provider rate limit/quota window.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(state, throttled_until)
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


def _append_sweep_events(
    state: dict[str, Any], sweep_events: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Append events collected during a sweep, aggregating same-kind runs.

    A single occurrence of a kind is emitted with the original kind and payload.
    Multiple occurrences of the same kind are emitted as one ``{kind}_sweep`` event
    with a count and issue-numbers list. This prevents a single bulk sweep from
    flooding the bounded event buffer and evicting unrelated diagnostic history.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind, payload in sweep_events:
        grouped.setdefault(kind, []).append(payload)

    for kind, payloads in grouped.items():
        if len(payloads) == 1:
            state = append_event(state, kind, payloads[0])
        else:
            issue_numbers = [
                payload["issue_number"]
                for payload in payloads
                if payload.get("issue_number") is not None
            ]
            state = append_event(
                state,
                f"{kind}_sweep",
                {
                    "count": len(payloads),
                    "issue_numbers": issue_numbers,
                },
            )
    return state


def _detect_and_handle_orphaned_workers(
    sessions_dir: Path, state_file: Path, config: OrchestratorConfig, gh: GitHub
) -> None:
    """Detect and handle orphaned workers using state.json PID records.

    This is a fallback for issue #207: when session sidecar files are orphaned
    (e.g., by session-limit reset), the session-file-based stall-reaper cannot
    detect dead workers. This function reads worker PIDs from state.json and
    checks liveness directly, allowing recovery even without session files.

    For issues with status "dispatched" and a recorded worker_pid:
    - If the PID is dead, check the linked PR's last review decision
    - If last decision was "request_changes" and head unchanged, reset to "rework_requested"
    - Otherwise, surface as drift for human triage
    - Do NOT clear worker_pid from state.json after handling (issue #282: the
      recovery path needs the fingerprint to verify the worktree is safe to reset).
    """
    if not config.watchdog.enabled:
        return

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

    # Handle orphaned workers
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
                        # PR head has changed - surface as drift for human triage
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
                    # Not a simple request_changes case - surface as drift
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
                # No open PR - emit drift event, leave recovery to mop-up
                # Mop-up will handle label transition back to ready (issue #118)
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

        state = _append_sweep_events(state, sweep_events)
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
        orphan_pids = sweep_orphan_processes(worktree_path)
        if orphan_pids:
            # Kill detected orphans
            killed_orphans = []
            for orphan_pid in orphan_pids:
                _kill_orphan_pid(orphan_pid)
                killed_orphans.append(orphan_pid)

            # Log the event
            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "orphan_processes_killed",
                    {
                        "worktree_path": worktree_path,
                        "orphan_pids": orphan_pids,
                        "killed_orphans": killed_orphans,
                    },
                )
                save_state(state_file, state)


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
        now = datetime.now(UTC)
        window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
        prior = [
            t
            for t in entry.get("redispatch_at", [])
            if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
        ]
        redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

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


def _classify_dead_sessions_and_update_throttle_state(
    sessions_dir: Path, state_file: Path, gh: GitHub, config: OrchestratorConfig
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
            if failure_kind and throttled_until:
                # A throttle-caused launch failure must persist its window just
                # like the dead-session branch below — otherwise the governor
                # relaunches straight into the same throttled provider.
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(state, throttled_until)
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
                    # Append to prior history within the redispatch window rather
                    # than overwriting it, matching the dead-session lane below
                    # (~line 1001-1006).
                    window_start = now - timedelta(
                        minutes=config.watchdog.redispatch_window_minutes
                    )
                    prior = [
                        t
                        for t in entry.get("redispatch_at", [])
                        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
                    ]
                    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]
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
                    )
                    save_state(state_file, state)

            w.reap_sidecar(sessions_dir)
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
        if w.error is None and not w.is_alive():
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
            # Only workers with a real pid are corroborated. A pid=None worker
            # (launch never spawned a process, or the pid was already cleared)
            # has no liveness signal to second-guess -- is_alive() is trivially
            # and unambiguously False -- so it keeps the prior immediate-reap
            # behavior, matching _detect_and_handle_stalled_sessions's existing
            # "if w.pid is None ...: continue" guard before it ever probes.
            if w.pid is not None:
                probe = real_activity_probe_for(w, config, now_for_health)
                health = classify_worker_health(w, config, now_for_health, probe)
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
            inspection = inspect_worktree_state(worktree_path, config.dispatch.base_ref)
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
                else:
                    failure_kind, throttled_until = None, None

            if failure_kind and throttled_until:
                # Update state with throttle window
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(state, throttled_until)
                    save_state(state_file, state)

            # Reap the sidecar to prevent phantom sessions from PID recycling (issue #113)
            # Delete the sidecar file after the session is detected as dead and classified
            w.reap_sidecar(sessions_dir)
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
                if not active_labels:
                    continue

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
                    window_start = now - timedelta(
                        minutes=config.watchdog.redispatch_window_minutes
                    )
                    prior = [
                        t
                        for t in entry.get("redispatch_at", [])
                        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
                    ]
                    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]
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
                        )
                        save_state(state_file, state)
                        continue
                    else:
                        entry["redispatch_at"] = redispatch_at
                        state["issues"][str(w.issue_number)] = entry
                        save_state(state_file, state)
                # Remove all active labels (error-as-value)
                for label in sorted(active_labels):
                    gh.remove_issue_label(w.issue_number, label)
                # Ensure ready label is present (error-as-value)
                if config.labels.ready not in issue_labels:
                    gh.add_issue_label(w.issue_number, config.labels.ready)
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
                            "added_ready": config.labels.ready not in issue_labels,
                            "salvage_failed": is_completed,
                            "salvage_error": salvage_error,
                        },
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
                    _reap_restore_rework_requested(
                        state_file, gh, config, open_prs_by_issue, w, failure_kind=failure_kind
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
    pr_number: int, attempts: int, summary: CheckSummary
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
    if not buckets:
        buckets.append("check summary unknown")
    checks_str = "; ".join(buckets)
    pass_str = "pass" if attempts == 1 else "passes"
    return f"PR #{pr_number} approved but unmergeable for {attempts} {pass_str}: {checks_str}"


# Sentinel used to distinguish "no base-current signal was supplied" from
# an explicit ``None`` (compare API unavailable) in _should_update_pr_branch.
class _BaseCurrentUnset:
    __slots__ = ()


_BASE_CURRENT_UNSET = _BaseCurrentUnset()


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
        self.gh = gh
        self.dry_run = dry_run
        self.fleet_dir_override = fleet_dir_override
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

    def _render(self, template_name: str, values: dict[str, Any]) -> str:
        return render_prompt(template_name, values, search_dirs=self.prompt_dirs)

    def _resolve(self, value: str) -> Path:
        # pathlib keeps an absolute right-hand side as-is, so this handles
        # both repo-relative and absolute config paths.
        return self.repo_root / value

    def _adapter_settings(self) -> AdapterSettings:
        claude = self.config.claude_code
        devin = self.config.devin
        adapter = devin.adapter
        # Use adapter-specific venv_source and worker_env
        if adapter == "devin-shell":
            venv_source = self._resolve(devin.venv_source) if devin.venv_source else None
            worker_env = devin.worker_env
        elif adapter == "claude-code":
            venv_source = self._resolve(claude.venv_source) if claude.venv_source else None
            worker_env = claude.worker_env
        else:
            venv_source = None
            worker_env = {}
        return AdapterSettings(
            adapter=adapter,
            dispatch_command=devin.dispatch_command,
            command_timeout_seconds=devin.command_timeout_seconds,
            sessions_dir=self._resolve(devin.sessions_dir),
            shell_command=devin.shell_command,
            claude_command=claude.command,
            worktrees_dir=self._resolve(claude.worktrees_dir) if claude.worktrees_dir else None,
            venv_source=venv_source,
            worker_env=worker_env,
            worker_model=devin.worker_model,
            materialize_dirs=self.config.dispatch.materialize_dirs,
            dry_run=self.dry_run,
            base_ref=self.config.dispatch.base_ref,
            tee_stream_json=claude.tee_stream_json,
            launch_stagger_seconds=self.config.dispatch.launch_stagger_seconds,
            config=self.config,
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
                sessions_dir = self._resolve(self.config.devin.sessions_dir)
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

    def status(self) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        prs = self.gh.pr_list()
        state = load_state_locked(self.paths.state_file)
        active_issues = [
            issue for issue in issues if label_names(issue) & self.config.labels.active
        ]
        available_issues = [issue for issue in issues if self._is_dispatchable(issue)]

        # Check for blocked issues (dependency gate)
        truly_available, blocked_issues = self._filter_blocked_issues(available_issues)

        # Check for stalled sessions (read-only for status/roll-call)
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
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
                pool_state = observe_runner_pool(self.gh, self.config.runner_scaling)
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
        }

        # Add runners section if feature is enabled and observation succeeded
        if runners_data is not None:
            data["runners"] = runners_data

        return CommandResult(True, "status complete", data)

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
        }
        for label in self.config.labels.all:
            color = "0E8A16" if label == self.config.labels.ready else "5319E7"
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
                state = append_event(
                    state,
                    "intake_failed",
                    {"issue_number": failure["issue"], "error": failure["error"]},
                )
            if prose_only_deps_issues:
                state = append_event(
                    state,
                    "intake_prose_only_deps",
                    {"issue_numbers": sorted(prose_only_deps_issues)},
                )
            state = append_event(
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

    def dispatch(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
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
                    },
                )
        try:
            return self._dispatch_impl(limit, only_issues=only_issues)
        finally:
            if fleet_lock is not None:
                fleet_lock.release()

    def _dispatch_impl(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit

        # Gather sessions_dir for stall detection and live worker counting
        sessions_dir = self._resolve(self.config.devin.sessions_dir)

        # Detect and handle stalled sessions before applying concurrency governor
        # This must run once per dispatch() call, not twice (was duplicated in governor)
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
            merged_prs = self.gh.merged_pr_list()
            merged_pr_bound_issue_numbers, merged_pr_mention_only_issue_numbers = (
                self._merged_pr_referenced_issue_numbers(issues, merged_prs)
            )
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
            candidates, blocked_issues = self._filter_blocked_issues(candidates)

            # Sort candidates by dispatch order
            # Default (oldest) uses dependency-aware ordering; explicit newest uses creation date
            if self.config.dispatch.order == "newest":
                candidates = self._sort_by_dispatch_order(candidates)
            else:
                # Default: use dependency-aware ordering (out-degree) with oldest-first tiebreaker
                candidates = self._sort_by_dependency_depth(candidates)

            if only_issues:
                wanted = parse_issue_numbers(only_issues)
                by_number = {int(issue["number"]): issue for issue in candidates}
                selected = [by_number[number] for number in wanted if number in by_number]
                skipped_issue_numbers = sorted(set(wanted) - set(by_number))
                # Apply concurrency governor cap to explicit issue selection
                if len(selected) > dispatch_limit:
                    deferred_by_concurrency = [
                        int(issue["number"]) for issue in selected[dispatch_limit:]
                    ]
                    selected = selected[:dispatch_limit]
                else:
                    deferred_by_concurrency = []
            else:
                selected = candidates[:dispatch_limit]
                deferred_by_concurrency = []
            selected_issue_numbers = [int(issue["number"]) for issue in selected]

            # Compute would-be SessionRequests without state mutation
            session_requests: list[SessionRequest] = []
            full_issues: dict[int, dict[str, Any]] = {}
            for issue_number in selected_issue_numbers:
                full_issue = self.gh.issue_view(issue_number)
                full_issues[issue_number] = full_issue
                prompt_path = self._write_worker_prompt(full_issue)
                branch_name = self._branch_name(full_issue)

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
                "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                "merged_pr_mention_only_issue_numbers": sorted(
                    merged_pr_mention_only_issue_numbers
                ),
                "label_errors": [],
                "sessions": [asdict(request) for request in session_requests],
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
        merged_prs = self.gh.merged_pr_list()
        merged_pr_bound_issue_numbers, merged_pr_mention_only_issue_numbers = (
            self._merged_pr_referenced_issue_numbers(issues, merged_prs)
        )
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

        # Close ready issues whose merged PR safely binds to them (hijack-safe:
        # same-repo branch-prefix or closing-action verb — the same trust
        # level issue #220 uses to close at merge time). This is
        # belt-and-suspenders in case #220's merge-time close hasn't landed
        # yet. These are network calls, so they run outside the state lock;
        # the successful closures are persisted to state.json inside the lock
        # below, and the issue numbers are excluded from dispatch candidates
        # regardless of closure success.
        closed_merged_pr_issues: set[int] = set()
        for issue_number in merged_pr_bound_issue_numbers:
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
        # redispatch it.
        for issue_number in merged_pr_mention_only_issue_numbers:
            transition(self.gh, self.config.labels, issue_number, "merged_pr_mention_flagged")

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
                )
                save_state(self.paths.state_file, state)
            # Record a flag timestamp so operators/tooling (e.g. a doctor
            # check) can surface mention-only coverage without re-deriving
            # the mention scan. "status" is deliberately untouched — the
            # issue stays open and its normal state machine intact.
            for issue_number in merged_pr_mention_only_issue_numbers:
                _issue_key = str(issue_number)
                _issue_entry = state["issues"].get(_issue_key, {})
                state["issues"][_issue_key] = {
                    **_issue_entry,
                    "number": issue_number,
                    "merged_pr_mention_flagged_at": utc_now(),
                }
            if merged_pr_mention_only_issue_numbers:
                state = append_event(
                    state,
                    "dispatch_merged_pr_mention_flagged",
                    {"issue_numbers": sorted(merged_pr_mention_only_issue_numbers)},
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

        # Apply dependency gate: skip issues with open blockers
        # Done outside the lock to avoid holding it during GitHub API calls
        candidates, blocked_issues = self._filter_blocked_issues(candidates)

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

            # Log dispatch_skip_blocked events for blocked issues
            if blocked_issues:
                for issue_number, blockers in blocked_issues.items():
                    state = append_event(
                        state,
                        "dispatch_skip_blocked",
                        {"issue": issue_number, "blockers": blockers},
                    )
                save_state(self.paths.state_file, state)

            if only_issues:
                wanted = parse_issue_numbers(only_issues)
                by_number = {int(issue["number"]): issue for issue in candidates}
                selected = [by_number[number] for number in wanted if number in by_number]
                skipped_issue_numbers = sorted(set(wanted) - set(by_number))
                # Apply concurrency governor cap to explicit issue selection
                if len(selected) > dispatch_limit:
                    deferred_by_concurrency = [
                        int(issue["number"]) for issue in selected[dispatch_limit:]
                    ]
                    selected = selected[:dispatch_limit]
                else:
                    deferred_by_concurrency = []
            else:
                selected = candidates[:dispatch_limit]
                deferred_by_concurrency = []
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
                state["issues"][str(issue_number)] = entry
            save_state(self.paths.state_file, state)
        # Do all network calls, file writes, and worker launches outside the lock
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        for issue_number in selected_issue_numbers:
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            prompt_path = self._write_worker_prompt(full_issue)
            branch_name = self._branch_name(full_issue)

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
        manifest_path = self.repo_root / self.config.devin.session_manifest
        results_path = self.repo_root / self.config.devin.session_results
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
        live_worker_issue_numbers = {
            result.issue_number
            for result in dispatch_results
            if not result.ok and result.failure_kind == "live_worker_redispatch_averted"
        }
        failed_issue_numbers = {
            result.issue_number
            for result in dispatch_results
            if not result.ok and result.issue_number not in live_worker_issue_numbers
        }
        # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
        manual = self.config.devin.adapter == "manual"
        label_errors: list[int] = []
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            for request in session_requests:
                full_issue = full_issues[request.issue_number]
                ok = request.issue_number in successful_issue_numbers
                is_live_worker = request.issue_number in live_worker_issue_numbers
                if ok:
                    status = "manifest_written" if manual else "dispatched"
                    dispatched_at = utc_now()
                elif is_live_worker:
                    status = "dispatched"
                    prev_entry = previous_entries.get(request.issue_number, {})
                    dispatched_at = prev_entry.get("dispatched_at") or utc_now()
                else:
                    status = "dispatch_failed"
                    dispatched_at = None
                entry = {
                    **state["issues"].get(str(request.issue_number), {}),
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
                        entry["label_error"] = {
                            "edge": target,
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        label_errors.append(request.issue_number)
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
                        )
                        save_state(self.paths.state_file, state)
            state = append_event(
                state,
                "dispatch",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "live_worker_issue_numbers": sorted(live_worker_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "label_errors": sorted(label_errors),
                    "skipped_issue_numbers": skipped_issue_numbers,
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                    "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
                    "merged_pr_flagged_issue_numbers": sorted(
                        merged_pr_mention_only_issue_numbers
                    ),
                },
            )
            save_state(self.paths.state_file, state)
        result_dicts = [result.to_dict() for result in dispatch_results]
        message = "dispatch complete"
        if failed_issue_numbers:
            message = "dispatch completed with failures"
        elif live_worker_issue_numbers:
            message = "dispatch completed with live worker redispatch averted"
        if skipped_issue_numbers:
            message += f" (skipped non-dispatchable: {skipped_issue_numbers})"
        if label_errors:
            message += f" (launched but label write failed: {sorted(label_errors)})"
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "live_worker_count": len(live_worker_issue_numbers),
            "skipped_issue_numbers": skipped_issue_numbers,
            "deferred_by_concurrency": deferred_by_concurrency,
            "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
            "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
            "merged_pr_flagged_issue_numbers": sorted(merged_pr_mention_only_issue_numbers),
            "label_errors": sorted(label_errors),
            "session_manifest": str(manifest_path),
            "session_results": str(results_path),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": result_dicts,
            "stalled": stalled_entries,
            "blocked": [
                {"issue": issue_number, "blockers": blockers}
                for issue_number, blockers in sorted(blocked_issues.items())
            ],
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
                emit_digest(self.config.notify, digest)

        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

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
            was not found.
        """
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return CommandResult(False, f"PR #{pr_number} was not found", {})
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
        )
        issue = self.gh.issue_view(issue_number) if issue_number is not None else {}
        checks = self.gh.pr_checks(pr_number)

        # Load PR state for no-op rework detection (only if PR has verdict history)
        pr_state = None
        state = load_state_locked(self.paths.state_file)
        if str(pr_number) in state.get("prs", {}):
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})

        # Fetch diff for patch-id based no-op rework detection (issue #222)
        # This is needed before the janitor gate to detect actual content changes
        diff = self.gh.pr_diff(pr_number)

        # Deterministic janitor gate BEFORE any packet/cross-family spend: an
        # obviously-not-ready PR (draft, conflicting, red CI, no issue link)
        # must cost zero review tokens. Failures don't move labels — they are
        # the worker's/CI's to fix, not a review decision.
        verdict = run_janitor(
            pr, checks, self.config, pr_state=pr_state, repo_root=self.repo_root, pr_diff=diff
        )
        if not verdict.ok:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    "status": "janitor_blocked",
                    "janitor_ok": False,
                    "janitor_failures": list(verdict.failures),
                }
                state = append_event(
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
        self._write_json(pr_dir / "pr.json", pr)
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
                return self.record_review(pr_number, "request_changes", summary=summary)
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
                "checks_json_path": pr_dir / "checks.json",
                "diff_path": pr_dir / "diff.patch",
                "cross_family_section": cross_family_section,
                "janitor_section": _janitor_section(merged_warnings),
                "test_adequacy_section": test_adequacy_section,
                "decision_command": f"{CLI_NAME} verdict --pr {pr_number} --decision approved --summary-file <path>",
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
        decision_path = pr_dir / "review-decision.json"
        if not decision_path.exists():
            self._write_json(decision_path, decision_template)
        else:
            # An approval is pinned to a specific head. If the PR has moved on,
            # the old verdict is void and must not survive into the new packet.
            existing_decision = self._review_decision(pr_number)
            reviewed_head_sha = existing_decision.get("reviewed_head_sha")
            if existing_decision.get("decision") == "approved" and (
                reviewed_head_sha is None or reviewed_head_sha != pr.get("headRefOid")
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
                "janitor_warnings": list(merged_warnings),
                "cross_family_report": cf_result.report_path if cf_result else None,
                "cross_family_ok": cf_result.ok if cf_result else None,
                "consecutive_failed_merge_attempts": 0,
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

    def record_review(
        self,
        pr_number: int,
        decision: str,
        summary: str = "",
        summary_file: Path | None = None,
        comment: bool = False,
    ) -> CommandResult:
        if decision not in {"approved", "request_changes", "blocked"}:
            return CommandResult(
                False, "decision must be approved, request_changes, or blocked", {}
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
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
        # Issue #11: reject empty summary for request_changes/blocked decisions
        # before any state/label mutation
        if decision in {"request_changes", "blocked"} and not summary_text.strip():
            return CommandResult(
                False,
                f"--summary or --summary-file is required for decision '{decision}'",
                {},
            )
        reviewed_head_sha = pr.get("headRefOid") if pr else None
        # Calculate patch-id for the PR diff to detect actual content changes
        # (issue #222: base-update merges can advance head SHA without changing diff content)
        reviewed_patch_id = ""
        if pr and decision in {"request_changes", "approved"}:
            diff = self.gh.pr_diff(pr_number)
            reviewed_patch_id = _calculate_patch_id(diff)
        decision_payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "summary": summary_text,
            "required_changes": [],
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_at": utc_now(),
        }
        decision_path = pr_dir / "review-decision.json"
        self._write_json(decision_path, decision_payload)
        # Merge-update (never in-place assignment) and persist BEFORE any GitHub
        # label mutation: a label-write failure or crash must not desync the
        # durable decision/counter from what actually happened.
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            rework_path: str | None = None
            escalated = False
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
                # Only count a rework cycle when the PR head has actually advanced.
                # If the head is unchanged, the prior cycle's attempt was never
                # delivered (e.g., worker died orphaned), so re-issuing request_changes
                # should not consume the escalation budget. See issue #208.
                head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
                if not escalated and head_advanced:
                    request_changes_count += 1
                if not escalated:
                    rework_path = str(self._write_rework_prompt(pr, issue_number, summary_text))
            decision_payload["escalated"] = escalated
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "issue_number": issue_number,
                "decision": decision,
                "decision_path": str(decision_path),
                "reviewed_head_sha": reviewed_head_sha,
                "reviewed_patch_id": reviewed_patch_id,
                "request_changes_count": request_changes_count,
                "status": "escalated" if escalated else decision,
                "consecutive_failed_merge_attempts": 0,
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
            state = append_event(
                state,
                "record_review",
                {"pr_number": pr_number, "decision": decision, "escalated": escalated},
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
        message = (
            f"review recorded — rework cap ({self.config.review.max_rework_cycles}) reached, "
            "escalated to human"
            if escalated
            else "review recorded"
        )
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
                "rework_path": rework_path,
                "escalated": escalated,
                "request_changes_count": request_changes_count,
                "label_error": label_error,
            },
        )

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
        if approved:
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            if reviewed_head_sha is None or live_head_sha != reviewed_head_sha:
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
                    }
                    if issue_number is not None:
                        _issue_key = str(issue_number)
                        _issue_entry = state["issues"].get(_issue_key, {})
                        state["issues"][_issue_key] = {**_issue_entry, "merge_alert": "OK"}
                    state = append_event(
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
            # Head matches the approved SHA. In merge-train mode, only the head
            # of the approved queue is allowed to proceed, and it must be
            # up-to-date with main before checks are evaluated.
            update_open_prs = self.config.auto_merge.update_open_prs
            if update_open_prs == "next":
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
            if not sync_failed and update_open_prs in {"next", "all"}:
                base_current = self._is_base_current(pr)
                if self._should_update_pr_branch(pr, base_current):
                    if self.gh.pr_update_branch(pr_number):
                        new_head = self._verify_synced_head(pr_number, live_head_sha)
                        if new_head and new_head != live_head_sha:
                            self._update_approval_head(pr_number, decision, new_head)
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
                if base_current is None and update_open_prs not in {"next", "all"}:
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
                        existing_pr_state,
                        base_ref,
                        head_sha,
                        reason,
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
                state = append_event(
                    state,
                    "containment_check",
                    {
                        "pr_number": pr_number,
                        "warnings": list(containment_warnings),
                    },
                )
                save_state(self.paths.state_file, state)
        can_merge = (
            summary.ready
            and (approved or not self.config.auto_merge.require_approved_review)
            and not sync_failed
        )
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        branch_deleted: bool | None = None
        label_error: dict[str, Any] | None = None
        update_results: list[dict[str, Any]] | None = None
        cancel_results: dict[str, Any] | None = None
        if can_merge and should_merge:
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
            if self.config.auto_merge.update_open_prs in {"all", "next"}:
                update_results = self._update_open_agent_prs(pr_number)
            # Cancel superseded queued runs on default branch after successful merge (if configured)
            if self.config.runners.enabled and self.config.runners.cancel_superseded_main_runs:
                cancel_results = cancel_superseded_runs(
                    self.gh,
                    self.config.runners.default_branch,
                    self.config.runners.workflow_name,
                )
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing = state["prs"].get(str(pr_number), {})
            new_attempts = 0
            merge_attempt_alarm = False
            merge_attempt_warning: str | None = None
            if approved and not can_merge and not _is_pending_only(summary):
                new_attempts = int(existing.get("consecutive_failed_merge_attempts", 0)) + 1
                threshold = self.config.auto_merge.failed_attempt_alarm
                merge_attempt_alarm = threshold > 0 and new_attempts == threshold
                if merge_attempt_alarm:
                    merge_attempt_warning = _format_merge_attempt_alarm_message(
                        pr_number, new_attempts, summary
                    )
                    state = append_event(
                        state,
                        "merge_failed_attempt_alarm",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "attempts": new_attempts,
                            "threshold": threshold,
                            "checks_summary": asdict(summary),
                            "message": merge_attempt_warning,
                        },
                    )
            if approved and can_merge and merge_output is None:
                # merge=False / auto_merge.enabled=False: can_merge recovered but no
                # merge was attempted. Clear the merge alert so a subsequent
                # degradation can re-fire the digest (last_health == current_health
                # dedup would otherwise drop it).
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
            }
            if merge_output:
                prs_entry["status"] = "merged"
                prs_entry["merged"] = True
            state["prs"][str(pr_number)] = prs_entry
            state = append_event(
                state,
                "merge_ready",
                {
                    "pr_number": pr_number,
                    "can_merge": can_merge,
                    "merged": bool(merge_output),
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
            "merge_attempt_alarm": merge_attempt_alarm,
            "merge_attempt_warning": merge_attempt_warning,
        }
        message = "merge readiness evaluated"
        if checks_unavailable:
            message = "checks unavailable (gh failure)"
        elif label_error:
            message += f" (merged; post-merge label/branch cleanup failed: {label_error})"
        return CommandResult(not checks_unavailable, message, data)

    def spec_review(self, artifact_path: Path) -> CommandResult:
        """Run an explicit cross-family adversarial pass over a spec/plan file.

        Independent of ``cross_family.enabled`` (that flag governs the PR-auto path);
        this command is the pre-execution spec slot and always runs when invoked.
        """
        path = Path(artifact_path)
        if not path.exists():
            return CommandResult(False, f"spec artifact not found: {path}", {})
        cfg = self.config.cross_family
        reviews_dir = self.paths.root / "cross-family"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(path.stem)
        prompt_text = self._render(
            "cross_family_spec_review.md",
            {"artifact_label": f"`{path}`", "artifact_text": path.read_text(encoding="utf-8")},
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
            state = append_event(
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

    def reconcile(self, *, fix: bool = False) -> CommandResult:
        """Detect (and optionally repair) drift between GitHub reality and the
        orchestrator's labels/state — e.g. a PR merged by hand outside
        merge-ready leaving `agent:in-progress` stale forever. Read-only unless
        ``fix`` is passed."""
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            drift = detect_drift(self.gh, state, self.config, repo_root=self.repo_root)
            fixed = False
            post_fix_drift: list[DriftItem] = []
            if fix and drift:
                new_state = apply_drift_fixes(self.gh, state, drift, self.config)
                save_state(self.paths.state_file, new_state)
                # Post-#134: transition() returns TransitionResult with PARTIAL_FAILURE
                # for failed adds/removes, and apply_fixes records the outcome in the
                # reconcile event. Re-detect against the new state to verify the repairs
                # actually landed before reporting success.
                post_fix_drift = detect_drift(
                    self.gh, new_state, self.config, repo_root=self.repo_root
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

        Behavior is controlled by ``auto_merge.update_open_prs``:

        - "all" (legacy): update every open tracked PR that is not approved-pending-ship
          and has no required checks currently in-flight.
        - "next" (merge-train): update only the head of the approved queue, so a single
          merge only causes one CI reset (the next candidate) instead of N-1.
        - "off": do nothing.

        Per-PR failures (conflicts, network errors) are reported as values and
        never abort the batch operation. A GitHubError from pr_list is also
        reported as a value and never propagates.
        """
        results: list[dict[str, Any]] = []
        mode = self.config.auto_merge.update_open_prs
        if mode == "off":
            return results

        if mode == "next":
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

            self._update_approval_head(pr_number, decision, new_head)
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": True,
                    "new_head": new_head,
                }
            ]

        # mode == "all": legacy behavior — update every qualifying PR.
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

            # Skip approved-pending-ship PRs to avoid invalidating their approvals
            # These will get base-updated when they themselves are merged (GitHub merges
            # handle base freshness) or by a later pass after they merge.
            decision = self._review_decision(pr_number)
            if decision.get("decision") == "approved":
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

            # Use the same compare-derived base-current signal as the merge-train
            # path and merge_ready so "all" mode also skips up-to-date PRs and
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
            },
        )

    def _merge_deferred_stale_base_result(
        self,
        pr_number: int,
        issue_number: int | None,
        decision: dict[str, Any],
        existing_pr_state: dict[str, Any],
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
            )
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
                "consecutive_failed_merge_attempts": existing_pr_state.get(
                    "consecutive_failed_merge_attempts", 0
                ),
                "merge_attempt_alarm": False,
                "merge_attempt_warning": None,
            },
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

    def _update_approval_head(
        self,
        pr_number: int,
        decision: dict[str, Any],
        new_head: str,
    ) -> None:
        """Persist an updated review head for a PR whose branch was synced.

        Keeps the approval valid when the branch was base-updated without
        content changes. Updates both review-decision.json and state.json.
        """
        decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
        updated_decision = dict(decision)
        updated_decision["reviewed_head_sha"] = new_head
        self._write_json(decision_path, updated_decision)

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "decision": pr_state.get("decision") or decision.get("decision") or "approved",
                "status": pr_state.get("status") or decision.get("decision") or "approved",
                "reviewed_head_sha": new_head,
                "reviewed_patch_id": pr_state.get("reviewed_patch_id")
                or decision.get("reviewed_patch_id")
                or "",
            }
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
        """Append a merge_ready result to merges or errors if checks are unavailable."""
        if merge_result.data.get("checks_unavailable"):
            errors.append({"pr": merge_result.data.get("pr"), "error": merge_result.message})
        else:
            merges.append(merge_result.data)

    def loop(self, limit: int | None = None, *, merge: bool | None = None) -> CommandResult:
        # merge=False runs the full pass (intake, dispatch, reviews, readiness
        # evaluation + labels) but skips the actual `gh pr merge` — for
        # operators sequencing same-surface PR cascades by hand, where the
        # pr_list (newest-first) merge order would land PRs in the wrong order.
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
        # Unconditional sweep: reap stalled/orphaned sessions even when this pass
        # has zero ready/rework candidates and never reaches dispatch()'s reaper call.
        _detect_and_handle_stalled_sessions(sessions_dir, self.paths.state_file, self.config)
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
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
        reaped = _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, self.paths.state_file, self.gh, self.config
        )

        # Sweep for orphan processes in dead session worktrees (issue #139)
        # This catches detached/daemonized processes that survived session kills
        _sweep_orphan_processes_for_dead_sessions(sessions_dir, self.paths.state_file, self.config)

        # Detect and handle orphaned workers using state.json PID records (issue #207)
        # This fallback detects dead workers even when session sidecar files are orphaned
        _detect_and_handle_orphaned_workers(
            sessions_dir, self.paths.state_file, self.config, self.gh
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
                emit_digest(self.config.notify, digest)

        dispatch_rework = self.dispatch_rework(effective_limit)
        rework_count = dispatch_rework.data.get("selected_count", 0)
        fresh_limit = max(0, effective_limit - rework_count)
        dispatch = self.dispatch(fresh_limit)
        reviews: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        open_tracked_prs = 0
        skipped_reviews = 0
        prs = self.gh.pr_list()
        merge_train_head = (
            self._merge_train_head(prs)
            if self.config.auto_merge.update_open_prs == "next"
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
            # Count every PR with a resolvable linked issue (includes skipped ones)
            open_tracked_prs += 1
            pr_number = int(pr["number"])
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
            except GitHubError as exc:
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
                emit_digest(self.config.notify, digest)

        ok = intake.ok and dispatch.ok and dispatch_rework.ok and not errors
        message = "loop complete"
        if errors:
            message = f"loop completed with {len(errors)} PR error(s)"
        elif not intake.ok:
            message = "loop completed with intake failures"
        elif not dispatch.ok:
            message = "loop completed with dispatch failures"
        elif not dispatch_rework.ok:
            message = "loop completed with rework dispatch failures"
        data = {
            "intake": intake.data,
            "dispatch": dispatch.data,
            "dispatch_rework": dispatch_rework.data,
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
        return CommandResult(
            ok,
            message,
            data,
        )

    def dispatch_rework(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
        """Dispatch rework workers for issues in needs-rework state with open PRs."""
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
            return self._dispatch_rework_impl(limit, only_issues=only_issues)
        finally:
            if fleet_lock is not None:
                fleet_lock.release()

    def _dispatch_rework_impl(
        self, limit: int | None = None, *, only_issues: str | None = None
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

        sessions_dir = self._resolve(self.config.devin.sessions_dir)
        # Unconditional reaper call, matching dispatch()'s :773-775 — previously
        # this only ran when max_concurrent_sessions > 0 via the governor at :2189.
        _detect_and_handle_stalled_sessions(sessions_dir, self.paths.state_file, self.config)

        # Detect and handle orphaned workers using state.json PID records (issue #207)
        # This fallback detects dead workers even when session sidecar files are orphaned
        _detect_and_handle_orphaned_workers(
            sessions_dir, self.paths.state_file, self.config, self.gh
        )

        # Load state to find rework_requested issues (state-driven selection)
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)

        # Find issues with rework_requested status
        rework_issues = []
        for number, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "rework_requested":
                # Fetch the full issue from GitHub to get labels and other metadata
                try:
                    issue_number = int(number)
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

        # Filter to issues with open PRs
        prs = self.gh.pr_list()
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

        # pr_list() returns only open PRs by contract (--state open); its field
        # list does not include "state", so no per-PR state check here.
        candidates = [issue for issue in rework_issues if int(issue["number"]) in pr_by_issue]

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
                "deferred_by_concurrency": deferred_by_concurrency,
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
                state["issues"][str(issue_number)] = entry
            save_state(self.paths.state_file, state)

        if not selected_issue_numbers:
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "deferred_by_concurrency": deferred_by_concurrency,
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
                continue
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
                    },
                )
                save_state(self.paths.state_file, state)
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "deferred_by_concurrency": deferred_by_concurrency,
            }
            if gov.enabled or gov.fleet_enabled:
                data.update(gov.report_fields())
            return CommandResult(
                True,
                "no valid rework prompts found",
                data,
            )

        manifest_path = self.repo_root / self.config.devin.session_manifest
        results_path = self.repo_root / self.config.devin.session_results
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
        failed_issue_numbers = {
            result.issue_number for result in dispatch_results if not result.ok
        }

        # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
        label_errors: list[int] = []
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
                    window_start = now - timedelta(
                        minutes=self.config.watchdog.redispatch_window_minutes
                    )
                    prior = [
                        t
                        for t in entry.get("redispatch_at", [])
                        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
                    ]
                    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]
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
                            entry["label_error"] = {
                                "edge": "redispatch_escalated",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            label_errors.append(request.issue_number)
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
                            entry["label_error"] = {
                                "edge": "rework_dispatched",
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            label_errors.append(request.issue_number)
                            save_state(self.paths.state_file, state)
                else:
                    state["issues"][str(request.issue_number)] = entry
                    save_state(self.paths.state_file, state)
            state = append_event(
                state,
                "dispatch_rework",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "skipped_issue_numbers": sorted(skipped_issue_numbers),
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "label_errors": sorted(label_errors),
                },
            )
            save_state(self.paths.state_file, state)

        result_dicts = [result.to_dict() for result in dispatch_results]
        message = "rework dispatch complete"
        if failed_issue_numbers:
            message = "rework dispatch completed with failures"
        if label_errors:
            message += f" (launched but label write failed: {sorted(label_errors)})"
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "deferred_by_concurrency": deferred_by_concurrency,
            "label_errors": sorted(label_errors),
            "session_manifest": str(manifest_path),
            "session_results": str(results_path),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": result_dicts,
        }
        if gov.enabled or gov.fleet_enabled:
            data.update(gov.report_fields())

        # Emit notification digest if there are health transitions (stalled sessions)
        # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
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
                emit_digest(self.config.notify, digest)

        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

    def _merged_pr_referenced_issue_numbers(
        self,
        issues: list[dict[str, Any]],
        merged_prs: list[dict[str, Any]],
    ) -> tuple[set[int], set[int]]:
        """Return ready issues already covered by a merged PR, split by trust level.

        Returns a ``(bound, mention_only)`` pair:

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

        Both sets are intersected with the set of ready issues so a stray
        mention of an issue not in the dispatch queue does not get actioned.
        ``bound`` takes precedence: an issue bound by one merged PR but only
        mentioned by another is reported solely in ``bound``.
        """
        ready_issue_numbers = {int(issue["number"]) for issue in issues}
        bound: set[int] = set()
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
            # isCrossRepository describes the PR's own head-branch provenance
            # (fork vs. same-repo), not which repo a free-text "#N" refers to.
            # It cannot fully guard a cross-repo mention collision, but it does
            # guard the common case of a fork PR's text being trusted at all.
            if pr.get("isCrossRepository") is False:
                for mentioned in issue_numbers_mentioned_by_pr(pr):
                    if mentioned in ready_issue_numbers:
                        mention_only.add(mentioned)
        mention_only -= bound
        return bound, mention_only

    def _is_dispatchable(self, issue: dict[str, Any]) -> bool:
        names = label_names(issue)
        if self.config.labels.ready not in names:
            return False
        if names & self.config.labels.terminal:
            return False
        return not names & self.config.labels.active

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

    def _filter_blocked_issues(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
        """Filter out issues with open blockers from the candidate list.

        This is a shared helper used by both dry-run and real dispatch paths
        to ensure single-point-of-enforcement for the dependency gate logic.

        Args:
            candidates: List of candidate issue dicts from GitHub API

        Returns:
            Tuple of (filtered_candidates, blocked_issues). filtered_candidates is the
            input list with blocked issues removed. blocked_issues is a dict mapping
            blocked issue numbers to their declared blocker lists.
        """
        blocked_issues: dict[int, list[int]] = {}
        for issue in candidates:
            issue_number = int(issue["number"])
            declared_blockers, open_blockers = self._get_open_blockers(issue)
            if open_blockers:
                blocked_issues[issue_number] = declared_blockers

        filtered_candidates = [
            issue for issue in candidates if int(issue["number"]) not in blocked_issues
        ]
        return filtered_candidates, blocked_issues

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

    def _write_worker_prompt(self, issue: dict[str, Any]) -> Path:
        issue_number = int(issue["number"])
        issue_dir = self.paths.issues / f"issue-{issue_number}"
        issue_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = issue_dir / "worker-prompt.md"
        prompt = self._render(
            self.config.dispatch.worker_template,
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
        self, pr: dict[str, Any], issue_number: int | None, summary: str
    ) -> Path:
        pr_number = int(pr["number"])
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        prompt_path = pr_dir / "rework-prompt.md"
        prompt = self._render(
            "rework.md",
            {
                "pr_number": pr_number,
                "pr_title": pr.get("title", ""),
                "pr_url": pr.get("url", ""),
                "issue_number": issue_number or "UNKNOWN",
                "review_summary": summary,
                "branch_name": pr.get("headRefName", ""),
            },
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt_path

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
