from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import CLI_NAME
from .adapters import AdapterSettings, SessionRequest, dispatch_sessions
from .checks import summarize_checks
from .config import CrossFamilyConfig, OrchestratorConfig
from .cross_family import (
    CrossFamilyResult,
    extract_report_body,
    report_body_is_valid,
    run_cross_family_review,
)
from .github import (
    GitHub,
    GitHubError,
    get_github_issue_dependencies,
    label_names,
    linked_issue_number,
    parse_blockers,
)
from .janitor import check_operator_containment, run_janitor
from .labels import transition
from .paths import RuntimePaths
from .prompts import render_prompt
from .reconcile import DriftItem, apply_fixes as apply_drift_fixes, detect_drift
from .state import (
    append_event,
    is_claim_stale,
    is_throttled,
    load_state,
    save_state,
    state_lock,
    utc_now,
)
from .process_utils import is_session_stalled, kill_process_tree


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

    @property
    def enabled(self) -> bool:
        """Return True if the governor is enabled (max_concurrent > 0)."""
        return self.max_concurrent > 0

    def report_fields(self) -> dict[str, int]:
        """Return the fields to include in CommandResult.data when clamped."""
        return {
            "concurrency_limit": self.max_concurrent,
            "live_session_count": self.live_count,
            "available_slots": self.available_slots,
        }


def _janitor_section(warnings: tuple[str, ...]) -> str:
    if not warnings:
        return ""
    lines = "\n".join(f"- {warning}" for warning in warnings)
    return (
        "\n## Janitor warnings (non-blocking)\n\n"
        f"{lines}\n\n"
        "These deterministic pre-checks passed the gate but deserve reviewer attention.\n"
    )


def slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "work"


def parse_issue_numbers(only_issues: str) -> list[int]:
    return [int(part) for part in only_issues.replace(" ", "").split(",") if part]


def _count_live_sessions(sessions_dir: Path) -> int:
    """Count the number of currently alive worker sessions across both adapters.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the total count of sessions with alive PIDs.
    """
    from .devin_shell import is_session_alive, read_session_records
    from .claude_code import is_worker_alive, read_worker_records

    live_count = 0
    # Count devin-shell sessions
    for record in read_session_records(sessions_dir):
        if is_session_alive(record):
            live_count += 1
    # Count claude-code sessions
    for record in read_worker_records(sessions_dir):
        if is_worker_alive(record):
            live_count += 1
    return live_count


def _detect_stalled_sessions(sessions_dir: Path, config: OrchestratorConfig) -> set[int]:
    """Detect stalled sessions (live PID but dead agent) without handling them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. This is a read-only detection function for status/roll-call.

    Returns the set of issue numbers that are stalled.
    """
    from .devin_shell import is_session_alive, read_session_records
    from .claude_code import is_worker_alive, read_worker_records

    if not config.watchdog.enabled:
        return set()

    stalled_issues = set()
    stall_threshold = config.watchdog.stall_minutes

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue

        if not is_session_alive(record):
            continue

        log_path = Path(record.log_path)
        is_stalled, _ = is_session_stalled(log_path, stall_threshold)

        if is_stalled:
            stalled_issues.add(record.issue_number)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue

        if not is_worker_alive(record):
            continue

        log_path = Path(record.log_path)
        is_stalled, _ = is_session_stalled(log_path, stall_threshold)

        if is_stalled:
            stalled_issues.add(record.issue_number)

    return stalled_issues


def _detect_and_handle_stalled_sessions(
    sessions_dir: Path, state_file: Path, config: OrchestratorConfig
) -> set[int]:
    """Detect stalled sessions (live PID but dead agent) and handle them.

    A session is stalled when its PID is alive but its log file's mtime is
    older than the configured threshold, or the log contains a terminal error
    marker. On detection, the process tree is killed, the sidecar is marked
    with failure_kind: stalled, and a session_stalled event is logged.

    Returns the set of issue numbers that were stalled (for exclusion from
    dispatch in the same pass).
    """
    from .devin_shell import (
        is_session_alive,
        read_session_records,
        update_session_record_with_failure_classification,
    )
    from .claude_code import (
        is_worker_alive,
        read_worker_records,
        update_worker_record_with_failure_classification,
    )

    if not config.watchdog.enabled:
        return set()

    stalled_issues = set()
    stall_threshold = config.watchdog.stall_minutes

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue

        if not is_session_alive(record):
            continue

        log_path = Path(record.log_path)
        is_stalled, last_log_line = is_session_stalled(log_path, stall_threshold)

        if is_stalled:
            # Kill the process tree (with start-time verification to prevent PID recycling)
            killed_pids = kill_process_tree(record.pid, record.process_start_time)

            # Mark the sidecar with failure_kind: stalled
            update_session_record_with_failure_classification(
                sessions_dir, record.issue_number, failure_kind="stalled"
            )

            # Log the event
            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "session_stalled",
                    {
                        "issue_number": record.issue_number,
                        "pid": record.pid,
                        "log_mtime": str(datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)),
                        "last_log_line": last_log_line,
                        "killed_pids": killed_pids,
                    },
                )
                save_state(state_file, state)

            stalled_issues.add(record.issue_number)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.pid is None or record.error is not None:
            continue

        if not is_worker_alive(record):
            continue

        log_path = Path(record.log_path)
        is_stalled, last_log_line = is_session_stalled(log_path, stall_threshold)

        if is_stalled:
            # Kill the process tree (with start-time verification to prevent PID recycling)
            killed_pids = kill_process_tree(record.pid, record.process_start_time)

            # Mark the sidecar with failure_kind: stalled
            update_worker_record_with_failure_classification(
                sessions_dir, record.issue_number, failure_kind="stalled"
            )

            # Log the event
            with state_lock(state_file):
                state = load_state(state_file)
                state = append_event(
                    state,
                    "session_stalled",
                    {
                        "issue_number": record.issue_number,
                        "pid": record.pid,
                        "log_mtime": str(datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)),
                        "last_log_line": last_log_line,
                        "killed_pids": killed_pids,
                    },
                )
                save_state(state_file, state)

            stalled_issues.add(record.issue_number)

    return stalled_issues


def _classify_dead_sessions_and_update_throttle_state(
    sessions_dir: Path, state_file: Path, gh: GitHub, config: OrchestratorConfig
) -> None:
    """Check for dead sessions, classify their failures, and update throttle state.

    This is called from the production loop to detect provider throttling
    from worker deaths and set the cooldown window in state.json.

    Also reconciles labels for dead sessions with no open PR (issue #118):
    a dead worker with no open PR is recoverable and should be relabeled
    as dispatchable (remove active labels, ensure ready label present).
    """
    from .devin_shell import (
        is_session_alive,
        read_session_records,
        update_session_record_with_failure_classification,
    )
    from .claude_code import (
        is_worker_alive,
        read_worker_records,
        update_worker_record_with_failure_classification,
    )
    from .state import append_event, load_state, save_state, set_throttled_until, state_lock

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

    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if record.error is None and not is_session_alive(record):
            # Session exited without error - classify the failure
            failure_kind, throttled_until = update_session_record_with_failure_classification(
                sessions_dir, record.issue_number
            )
            if failure_kind and throttled_until:
                # Update state with throttle window
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(state, throttled_until)
                    save_state(state_file, state)

            # Issue #118: reconcile labels for dead sessions with no open PR
            if record.issue_number not in open_prs_by_issue:
                try:
                    issue = gh.issue_view(record.issue_number)
                except Exception:
                    # Issue may have been deleted or we lack access; skip relabel
                    continue
                issue_labels = label_names(issue)
                active_labels = issue_labels & config.labels.active
                if active_labels:
                    # Remove all active labels (error-as-value)
                    for label in sorted(active_labels):
                        gh.remove_issue_label(record.issue_number, label)
                    # Ensure ready label is present (error-as-value)
                    if config.labels.ready not in issue_labels:
                        gh.add_issue_label(record.issue_number, config.labels.ready)
                    # Record the relabel event
                    with state_lock(state_file):
                        state = load_state(state_file)
                        state = append_event(
                            state,
                            "session_failed_relabeled",
                            {
                                "issue_number": record.issue_number,
                                "failure_kind": failure_kind,
                                "removed_labels": sorted(active_labels),
                                "added_ready": config.labels.ready not in issue_labels,
                            },
                        )
                        save_state(state_file, state)

    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if record.error is None and not is_worker_alive(record):
            # Session exited without error - classify the failure
            failure_kind, throttled_until = update_worker_record_with_failure_classification(
                sessions_dir, record.issue_number
            )
            if failure_kind and throttled_until:
                # Update state with throttle window
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = set_throttled_until(state, throttled_until)
                    save_state(state_file, state)

            # Issue #118: reconcile labels for dead sessions with no open PR
            if record.issue_number not in open_prs_by_issue:
                try:
                    issue = gh.issue_view(record.issue_number)
                except Exception:
                    # Issue may have been deleted or we lack access; skip relabel
                    continue
                issue_labels = label_names(issue)
                active_labels = issue_labels & config.labels.active
                if active_labels:
                    # Remove all active labels (error-as-value)
                    for label in sorted(active_labels):
                        gh.remove_issue_label(record.issue_number, label)
                    # Ensure ready label is present (error-as-value)
                    if config.labels.ready not in issue_labels:
                        gh.add_issue_label(record.issue_number, config.labels.ready)
                    # Record the relabel event
                    with state_lock(state_file):
                        state = load_state(state_file)
                        state = append_event(
                            state,
                            "session_failed_relabeled",
                            {
                                "issue_number": record.issue_number,
                                "failure_kind": failure_kind,
                                "removed_labels": sorted(active_labels),
                                "added_ready": config.labels.ready not in issue_labels,
                            },
                        )
                        save_state(state_file, state)


def _issues_with_live_workers(sessions_dir: Path) -> set[int]:
    """Return the set of issue numbers that have currently alive worker sessions.

    Reads session sidecar files from both devin-shell and claude-code adapters,
    then checks each record's PID liveness using the adapter-specific liveness
    probe. Returns the set of issue numbers with alive PIDs.
    """
    from .devin_shell import is_session_alive, read_session_records
    from .claude_code import is_worker_alive, read_worker_records

    live_issues = set()
    # Check devin-shell sessions
    for record in read_session_records(sessions_dir):
        if is_session_alive(record):
            live_issues.add(record.issue_number)
    # Check claude-code sessions
    for record in read_worker_records(sessions_dir):
        if is_worker_alive(record):
            live_issues.add(record.issue_number)
    return live_issues


class OrchestratorApp:
    def __init__(
        self,
        repo_root: Path,
        paths: RuntimePaths,
        config: OrchestratorConfig,
        gh: GitHub,
        *,
        dry_run: bool = False,
    ):
        self.repo_root = repo_root
        self.paths = paths
        self.config = config
        self.gh = gh
        self.dry_run = dry_run
        prompts_dir = config.runtime.prompts_dir
        if prompts_dir:
            override = Path(prompts_dir)
            if not override.is_absolute():
                override = repo_root / override
            self.prompt_dirs: tuple[Path, ...] = (override,)
        else:
            self.prompt_dirs = ()
        self.paths.ensure()

    def _render(self, template_name: str, values: dict[str, Any]) -> str:
        return render_prompt(template_name, values, search_dirs=self.prompt_dirs)

    def _resolve(self, value: str) -> Path:
        # pathlib keeps an absolute right-hand side as-is, so this handles
        # both repo-relative and absolute config paths.
        return self.repo_root / value

    def _adapter_settings(self) -> AdapterSettings:
        claude = self.config.claude_code
        return AdapterSettings(
            adapter=self.config.devin.adapter,
            dispatch_command=self.config.devin.dispatch_command,
            command_timeout_seconds=self.config.devin.command_timeout_seconds,
            sessions_dir=self._resolve(self.config.devin.sessions_dir),
            shell_command=self.config.devin.shell_command,
            claude_command=claude.command,
            worktrees_dir=self._resolve(claude.worktrees_dir) if claude.worktrees_dir else None,
            venv_source=self._resolve(claude.venv_source) if claude.venv_source else None,
            worker_env=claude.worker_env,
            worker_model=self.config.devin.worker_model,
            dry_run=self.dry_run,
            base_ref=self.config.dispatch.base_ref,
        )

    def _apply_concurrency_governor(self, dispatch_limit: int) -> ConcurrencyGovernorResult:
        """Apply global concurrency governor cap to a dispatch limit.

        Returns a ConcurrencyGovernorResult with the potentially-clamped limit
        and all related fields. This eliminates Pyright's reportPossiblyUnbound
        warnings by ensuring live_count is always bound together with the
        clamped flag.
        """
        max_concurrent = self.config.dispatch.max_concurrent_sessions
        live_count = 0
        available_slots = dispatch_limit
        clamped = False

        if max_concurrent > 0:
            sessions_dir = self._resolve(self.config.devin.sessions_dir)
            # Detect and handle stalled sessions before counting live sessions
            _detect_and_handle_stalled_sessions(sessions_dir, self.paths.state_file, self.config)
            live_count = _count_live_sessions(sessions_dir)
            available_slots = max(0, max_concurrent - live_count)
            if available_slots < dispatch_limit:
                dispatch_limit = available_slots
                clamped = True

        return ConcurrencyGovernorResult(
            clamped=clamped,
            max_concurrent=max_concurrent,
            live_count=live_count,
            available_slots=available_slots,
            dispatch_limit=dispatch_limit,
        )

    def status(self) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        prs = self.gh.pr_list()
        state = load_state(self.paths.state_file)
        active_issues = [
            issue for issue in issues if label_names(issue) & self.config.labels.active
        ]
        available_issues = [issue for issue in issues if self._is_dispatchable(issue)]

        # Check for blocked issues (dependency gate)
        truly_available, blocked_issues = self._filter_blocked_issues(available_issues)

        # Check for stalled sessions (read-only for status/roll-call)
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
        stalled_issues = _detect_stalled_sessions(sessions_dir, self.config)

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
            "stalled": sorted(stalled_issues) if stalled_issues else [],
        }
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
            state = append_event(
                state, "intake", {"issue_count": len(issues), "failed_count": len(failed)}
            )
            save_state(self.paths.state_file, state)
        message = "intake complete"
        if failed:
            message = f"intake completed with {len(failed)} failure(s)"
        return CommandResult(
            not failed,
            message,
            {"issues": written, "failed": failed},
        )

    def dispatch(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit

        # Apply global concurrency governor cap
        gov = self._apply_concurrency_governor(dispatch_limit)
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
                if gov.clamped:
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
            # Gather network results outside the lock (matching intake pattern)
            sessions_dir = self._resolve(self.config.devin.sessions_dir)
            # Detect and handle stalled sessions before checking live workers
            stalled_issues = _detect_and_handle_stalled_sessions(
                sessions_dir, self.paths.state_file, self.config
            )
            live_worker_issues = _issues_with_live_workers(sessions_dir)
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
                        issue_number = int(number)
                        if issue_number in live_worker_issues or issue_number in pr_by_issue:
                            live_dispatched.add(issue_number)
                candidates = [
                    issue
                    for issue in issues
                    if self._is_dispatchable(issue)
                    and int(issue["number"]) not in live_dispatched
                    and int(issue["number"]) not in stalled_issues
                ]

            # Apply dependency gate: skip issues with open blockers (dry-run)
            # Done outside the lock to avoid holding it during GitHub API calls
            candidates, blocked_issues = self._filter_blocked_issues(candidates)

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
                "label_errors": [],
                "sessions": [asdict(request) for request in session_requests],
                "dispatch_results": [],
                "blocked": [
                    {"issue": issue_number, "blockers": blockers}
                    for issue_number, blockers in sorted(blocked_issues.items())
                ],
            }
            if gov.clamped:
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
        # Gather network results outside the lock (matching intake pattern)
        sessions_dir = self._resolve(self.config.devin.sessions_dir)
        # Detect and handle stalled sessions before checking live workers
        stalled_issues = _detect_and_handle_stalled_sessions(
            sessions_dir, self.paths.state_file, self.config
        )
        live_worker_issues = _issues_with_live_workers(sessions_dir)
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

        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
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
                    issue_number = int(number)
                    if issue_number in live_worker_issues or issue_number in pr_by_issue:
                        live_dispatched.add(issue_number)
            candidates = [
                issue
                for issue in issues
                if self._is_dispatchable(issue)
                and int(issue["number"]) not in live_dispatched
                and int(issue["number"]) not in stalled_issues
            ]

        # Apply dependency gate: skip issues with open blockers
        # Done outside the lock to avoid holding it during GitHub API calls
        candidates, blocked_issues = self._filter_blocked_issues(candidates)

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
                state["issues"][str(issue_number)] = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "dispatch_pending",
                    "dispatch_pending_at": utc_now(),
                }
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
        failed_issue_numbers = {
            result.issue_number for result in dispatch_results if not result.ok
        }
        # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
        manual = self.config.devin.adapter == "manual"
        label_errors: list[int] = []
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
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
                    "status": ("manifest_written" if manual else "dispatched")
                    if ok
                    else "dispatch_failed",
                    "dispatched_at": utc_now() if ok else None,
                }
                # Clear the claim timestamp on successful upgrade
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                state["issues"][str(request.issue_number)] = entry
                # Persist the launched worker BEFORE touching GitHub labels: a
                # transient label-write failure (or crash) must never leave a live
                # worker unrecorded and therefore re-dispatchable next wave. The
                # transition is isolated per-issue so one failure never aborts the
                # rest of the batch (orphaning already-launched workers).
                save_state(self.paths.state_file, state)
                if ok:
                    try:
                        transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            "queued" if manual else "dispatched",
                        )
                    except GitHubError as exc:
                        entry["label_error"] = str(exc)
                        label_errors.append(request.issue_number)
                        save_state(self.paths.state_file, state)
            state = append_event(
                state,
                "dispatch",
                {
                    "issue_numbers": sorted(successful_issue_numbers),
                    "failed_issue_numbers": sorted(failed_issue_numbers),
                    "label_errors": sorted(label_errors),
                    "skipped_issue_numbers": skipped_issue_numbers,
                    "deferred_by_concurrency": deferred_by_concurrency,
                },
            )
            save_state(self.paths.state_file, state)
        result_dicts = [result.to_dict() for result in dispatch_results]
        message = "dispatch complete"
        if failed_issue_numbers:
            message = "dispatch completed with failures"
        if skipped_issue_numbers:
            message += f" (skipped non-dispatchable: {skipped_issue_numbers})"
        if label_errors:
            message += f" (launched but label write failed: {sorted(label_errors)})"
        data = {
            "selected_count": len(successful_issue_numbers),
            "attempted_count": len(session_requests),
            "failed_count": len(failed_issue_numbers),
            "skipped_issue_numbers": skipped_issue_numbers,
            "deferred_by_concurrency": deferred_by_concurrency,
            "label_errors": sorted(label_errors),
            "session_manifest": str(manifest_path),
            "session_results": str(results_path),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": result_dicts,
        }
        if gov.clamped:
            data.update(gov.report_fields())
        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

    def review(self, pr_number: int, *, cross_family: bool | None = None) -> CommandResult:
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
        # Cheap existence check before acquiring lock
        state = load_state(self.paths.state_file)
        if str(pr_number) in state.get("prs", {}):
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                pr_state = state["prs"].get(str(pr_number), {})

        # Deterministic janitor gate BEFORE any packet/cross-family spend: an
        # obviously-not-ready PR (draft, conflicting, red CI, no issue link)
        # must cost zero review tokens. Failures don't move labels — they are
        # the worker's/CI's to fix, not a review decision.
        verdict = run_janitor(pr, checks, self.config, pr_state=pr_state, repo_root=self.repo_root)
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
                },
            )
        diff = self.gh.pr_diff(pr_number)
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(pr_dir / "pr.json", pr)
        self._write_json(pr_dir / "checks.json", checks)
        diff_path = pr_dir / "diff.patch"
        diff_path.write_text(diff, encoding="utf-8")
        # Run containment check for worker edits leaked into operator checkout
        containment_warnings = check_operator_containment(self.repo_root, diff, pr_number)
        # Merge containment warnings with janitor warnings
        merged_warnings = tuple(list(verdict.warnings) + list(containment_warnings))
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
            }
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
        label_error: str | None = None
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
                try:
                    transition(self.gh, self.config.labels, issue_number, "review_started")
                except GitHubError as exc:
                    label_error = str(exc)
                    with state_lock(self.paths.state_file):
                        state = load_state(self.paths.state_file)
                        state["prs"][str(pr_number)]["label_error"] = label_error
                        save_state(self.paths.state_file, state)
        message = "review packet generated"
        if label_error:
            message += f" (label update failed: {label_error})"
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
        decision_payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "summary": summary_text,
            "required_changes": [],
            "reviewed_head_sha": reviewed_head_sha,
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
                if not escalated:
                    request_changes_count += 1
                    rework_path = str(self._write_rework_prompt(pr, issue_number, summary_text))
            decision_payload["escalated"] = escalated
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "issue_number": issue_number,
                "decision": decision,
                "decision_path": str(decision_path),
                "reviewed_head_sha": reviewed_head_sha,
                "request_changes_count": request_changes_count,
                "status": "escalated" if escalated else decision,
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
                        }
                    else:
                        # Clear rework_requested status when escalated to prevent selection
                        state["issues"][str(issue_number)] = {
                            **state["issues"].get(str(issue_number), {}),
                            "number": issue_number,
                            "status": "escalated",
                        }
                elif decision == "approved":
                    state["issues"][str(issue_number)] = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "status": "approved",
                    }
                elif decision == "blocked":
                    state["issues"][str(issue_number)] = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "status": "blocked",
                    }
            state = append_event(
                state,
                "record_review",
                {"pr_number": pr_number, "decision": decision, "escalated": escalated},
            )
            save_state(self.paths.state_file, state)
        # GitHub label side effects are best-effort and isolated: the durable
        # decision above is the authority; a label failure is reported, not fatal.
        label_error: str | None = None
        try:
            if issue_number is not None:
                if decision == "request_changes":
                    transition(
                        self.gh,
                        self.config.labels,
                        issue_number,
                        "escalated" if escalated else "rework_requested",
                    )
                elif decision == "blocked":
                    transition(self.gh, self.config.labels, issue_number, "blocked")
                elif decision == "approved":
                    transition(self.gh, self.config.labels, issue_number, "review_approved")
            if decision == "request_changes" and comment and summary_text:
                self._comment_pr(pr_number, summary_text)
        except GitHubError as exc:
            label_error = str(exc)
        message = (
            f"review recorded — rework cap ({self.config.review.max_rework_cycles}) reached, "
            "escalated to human"
            if escalated
            else "review recorded"
        )
        if label_error:
            message += f" (label update failed: {label_error})"
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

    def merge_ready(self, pr_number: int, *, merge: bool | None = None) -> CommandResult:
        # Idempotence: if state already records this PR as merged, short-circuit
        # to a success no-op. Re-running `ship-it` on a completed PR must not
        # re-attempt `gh pr merge` (which fails on an already-merged PR and
        # propagates GitHubError → exit 2).
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing_pr_state = state["prs"].get(str(pr_number), {})
            if existing_pr_state.get("status") == "merged":
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
        if approved:
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            if reviewed_head_sha is None or live_head_sha != reviewed_head_sha:
                message = "PR head moved since approval — re-review required"
                label_error: str | None = None
                try:
                    if issue_number is not None:
                        transition(self.gh, self.config.labels, issue_number, "review_started")
                except GitHubError as exc:
                    label_error = str(exc)
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
                    }
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
        checks = self.gh.pr_checks(pr_number)
        summary = summarize_checks(checks, self.config.auto_merge.required_checks)
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
        can_merge = summary.ready and (
            approved or not self.config.auto_merge.require_approved_review
        )
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        branch_deleted: bool | None = None
        label_error: str | None = None
        update_results: list[dict[str, Any]] = []
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
                }
                save_state(self.paths.state_file, state)
            # Label + branch cleanup are best-effort; the merged fact is already
            # durable. A branch-deletion failure (head branch checked out in a
            # worktree) or label failure must never un-record the merge.
            try:
                if issue_number is not None:
                    transition(self.gh, self.config.labels, issue_number, "merged")
                if self.config.auto_merge.delete_branch:
                    head_ref = str(pr.get("headRefName") or "")
                    branch_deleted = self.gh.delete_branch(head_ref) if head_ref else False
            except GitHubError as exc:
                label_error = str(exc)
            # Update remaining open agent PRs after successful merge (if configured)
            if self.config.auto_merge.update_open_prs:
                update_results = self._update_open_agent_prs(pr_number)
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
            "label_error": label_error,
            "update_open_prs_results": update_results,
            "containment_warnings": list(containment_warnings),
        }
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            existing = state["prs"].get(str(pr_number), {})
            prs_entry: dict[str, Any] = {
                **existing,
                "number": pr_number,
                "issue_number": issue_number,
            }
            if merge_output:
                prs_entry["status"] = "merged"
                prs_entry["merged"] = True
            state["prs"][str(pr_number)] = prs_entry
            state = append_event(
                state,
                "merge_ready",
                {"pr_number": pr_number, "can_merge": can_merge, "merged": bool(merge_output)},
            )
            save_state(self.paths.state_file, state)
        message = "merge readiness evaluated"
        if label_error:
            message += f" (merged; post-merge label/branch cleanup failed: {label_error})"
        return CommandResult(True, message, data)

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
        if report_path.exists() and report_path.stat().st_size > 0:
            text = report_path.read_text(encoding="utf-8")
            first_line = text.splitlines()[0]
            # The file is a wrapped report (header + caveat + body).  Validate the
            # model body only, not the wrapper text that itself contains bold
            # markdown ("**leads, not verdicts**").
            body = extract_report_body(text)
            if "(UNAVAILABLE)" not in first_line and report_body_is_valid(body):
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
                # The label removals above use allow_failure=True, so a failed
                # removal is silently swallowed. Re-detect against the new state to
                # verify the repairs actually landed before reporting success.
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

        Calls `gh pr update-branch` on all open PRs that:
        - Are same-repo (not forks)
        - Have the configured branch prefix
        - Are not the just-merged PR
        - Are NOT approved-pending-ship (decision == "approved" with live head == reviewed_head_sha)

        Per-PR failures (conflicts, network errors) are reported as values and
        never abort the batch operation.
        """
        results: list[dict[str, Any]] = []
        prs = self.gh.pr_list()
        branch_prefix = self.config.dispatch.branch_prefix

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

    def loop(self, limit: int | None = None, *, merge: bool | None = None) -> CommandResult:
        # merge=False runs the full pass (intake, dispatch, reviews, readiness
        # evaluation + labels) but skips the actual `gh pr merge` — for
        # operators sequencing same-surface PR cascades by hand, where the
        # pr_list (newest-first) merge order would land PRs in the wrong order.
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
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, self.paths.state_file, self.gh, self.config
        )

        dispatch_rework = self.dispatch_rework(effective_limit)
        rework_count = dispatch_rework.data.get("selected_count", 0)
        fresh_limit = max(0, effective_limit - rework_count)
        dispatch = self.dispatch(fresh_limit)
        reviews: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for pr in self.gh.pr_list():
            issue_number = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            )
            if issue_number is None:
                continue
            pr_number = int(pr["number"])
            # Per-PR isolation: one PR's merge conflict or gh failure must not
            # abort review/merge of every remaining PR in the batch.
            try:
                # Idempotence: if the PR already has an approved decision in
                # state and isn't in a rework/blocked state, skip the expensive
                # review() pass (packet regeneration + label transitions) and
                # go straight to merge_ready. This prevents a second loop() pass
                # from rewriting the review packet or re-firing labels for a PR
                # that's simply waiting on pending checks.
                state = load_state(self.paths.state_file)
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
                    if head_matches:
                        merges.append(self.merge_ready(pr_number, merge=merge).data)
                    else:
                        review = self.review(pr_number)
                        reviews.append(review.data)
                        decision = self._review_decision(pr_number)
                        if decision.get("decision") == "approved":
                            merges.append(self.merge_ready(pr_number, merge=merge).data)
                else:
                    review = self.review(pr_number)
                    reviews.append(review.data)
                    decision = self._review_decision(pr_number)
                    if decision.get("decision") == "approved":
                        merges.append(self.merge_ready(pr_number, merge=merge).data)
            except GitHubError as exc:
                errors.append({"pr": pr_number, "error": str(exc)})
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
        }
        # Propagate concurrency info from dispatch results
        if gov.enabled:
            data.update(gov.report_fields())
        return CommandResult(
            ok,
            message,
            data,
        )

    def dispatch_rework(
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
                if gov.clamped:
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
            if gov.clamped:
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
                state["issues"][str(issue_number)] = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "dispatch_pending",
                    "dispatch_pending_at": utc_now(),
                }
            save_state(self.paths.state_file, state)

        if not selected_issue_numbers:
            data = {
                "adapter": self.config.devin.adapter,
                "selected_count": 0,
                "deferred_by_concurrency": deferred_by_concurrency,
            }
            if gov.clamped:
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
                # Skip if rework prompt doesn't exist — record as dispatch_failed
                # to release the claim and avoid blocking re-dispatch for 30 min
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
                        "status": "dispatch_failed",
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
            if gov.clamped:
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
            # Record skipped issues (missing rework prompt) as dispatch_failed
            # This handles the mixed case where some issues have prompts and some don't
            for issue_number in skipped_issue_numbers:
                full_issue = full_issues[issue_number]
                entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    "status": "dispatch_failed",
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
                    "status": "dispatched" if ok else "dispatch_failed",
                    "dispatched_at": utc_now() if ok else None,
                }
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                state["issues"][str(request.issue_number)] = entry
                save_state(self.paths.state_file, state)
                if ok:
                    try:
                        transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            "rework_dispatched",
                        )
                    except GitHubError as exc:
                        entry["label_error"] = str(exc)
                        label_errors.append(request.issue_number)
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
        if gov.clamped:
            data.update(gov.report_fields())
        return CommandResult(
            not failed_issue_numbers,
            message,
            data,
        )

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

    def _comment_pr(self, pr_number: int, summary: str) -> None:
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        body_path = pr_dir / "review-comment.md"
        body_path.write_text(summary, encoding="utf-8")
        self.gh.pr_comment(pr_number, body_path)

    def _summarize_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        return {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "url": issue.get("url"),
            "labels": sorted(label_names(issue)),
            "dispatchable": self._is_dispatchable(issue),
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

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
