from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
from .github import GitHub, GitHubError, label_names, linked_issue_number
from .janitor import run_janitor
from .labels import transition
from .paths import RuntimePaths
from .prompts import render_prompt
from .reconcile import DriftItem, apply_fixes as apply_drift_fixes, detect_drift
from .state import append_event, load_state, save_state, utc_now


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any]


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


class OrchestratorApp:
    def __init__(
        self, repo_root: Path, paths: RuntimePaths, config: OrchestratorConfig, gh: GitHub
    ):
        self.repo_root = repo_root
        self.paths = paths
        self.config = config
        self.gh = gh
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
        )

    def status(self) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        prs = self.gh.pr_list()
        state = load_state(self.paths.state_file)
        active_issues = [
            issue for issue in issues if label_names(issue) & self.config.labels.active
        ]
        available_issues = [issue for issue in issues if self._is_dispatchable(issue)]
        linked_prs = [self._summarize_pr(pr) for pr in prs if linked_issue_number(pr) is not None]
        data = {
            "ready_issue_count": len(issues),
            "available_issue_count": len(available_issues),
            "active_issue_count": len(active_issues),
            "open_linked_pr_count": len(linked_prs),
            "state_file": str(self.paths.state_file),
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "issues": [self._summarize_issue(issue) for issue in issues],
            "prs": linked_prs,
            "last_generated_at": state.get("generated_at"),
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
        state = load_state(self.paths.state_file)
        written: list[dict[str, Any]] = []
        for issue in issues:
            issue_number = int(issue["number"])
            full_issue = self.gh.issue_view(issue_number)
            issue_dir = self.paths.issues / f"issue-{issue_number}"
            issue_dir.mkdir(parents=True, exist_ok=True)
            issue_json = issue_dir / "issue.json"
            self._write_json(issue_json, full_issue)
            prompt_path = self._write_worker_prompt(full_issue)
            # Merge-update, never replace: intake used to clobber dispatch
            # status recorded by earlier passes (production-confirmed).
            state["issues"][str(issue_number)] = {
                **state["issues"].get(str(issue_number), {}),
                "number": issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "labels": sorted(label_names(full_issue)),
                "prompt_path": str(prompt_path),
                "updated_at": full_issue.get("updatedAt"),
            }
            written.append({"issue": issue_number, "prompt_path": str(prompt_path)})
        state = append_event(state, "intake", {"issue_count": len(issues)})
        save_state(self.paths.state_file, state)
        return CommandResult(True, "intake complete", {"issues": written})

    def dispatch(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit
        state = load_state(self.paths.state_file)
        # Defence-in-depth against double-dispatch: an issue whose state records
        # a live launched worker (status "dispatched") is not re-dispatchable
        # even if its GitHub label write failed after the worker launched.
        # _is_dispatchable is label-only; this closes the launched-but-unlabeled
        # window that would otherwise spawn a second worker on the same issue.
        live_dispatched = {
            int(number)
            for number, entry in state.get("issues", {}).items()
            if isinstance(entry, dict) and entry.get("status") == "dispatched"
        }
        candidates = [
            issue
            for issue in issues
            if self._is_dispatchable(issue) and int(issue["number"]) not in live_dispatched
        ]
        skipped_issue_numbers: list[int] = []
        if only_issues:
            wanted = parse_issue_numbers(only_issues)
            by_number = {int(issue["number"]): issue for issue in candidates}
            selected = [by_number[number] for number in wanted if number in by_number]
            skipped_issue_numbers = sorted(set(wanted) - set(by_number))
        else:
            selected = candidates[:dispatch_limit]
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        for issue in selected:
            issue_number = int(issue["number"])
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            prompt_path = self._write_worker_prompt(full_issue)
            branch_name = self._branch_name(full_issue)
            session_requests.append(
                SessionRequest(
                    issue_number=issue_number,
                    issue_title=str(full_issue.get("title") or ""),
                    prompt_path=prompt_path,
                    branch_name=branch_name,
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
        # The manual adapter's ok means "manifest written, awaiting the
        # operator" — no worker exists yet, so the issue is queued, not
        # in-progress. Only an adapter that actually launched something may
        # promote to in-progress.
        manual = self.config.devin.adapter == "manual"
        label_errors: list[int] = []
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
        return CommandResult(
            not failed_issue_numbers,
            message,
            {
                "selected_count": len(successful_issue_numbers),
                "attempted_count": len(session_requests),
                "failed_count": len(failed_issue_numbers),
                "skipped_issue_numbers": skipped_issue_numbers,
                "label_errors": sorted(label_errors),
                "session_manifest": str(manifest_path),
                "session_results": str(results_path),
                "sessions": [asdict(request) for request in session_requests],
                "dispatch_results": result_dicts,
            },
        )

    def review(self, pr_number: int, *, cross_family: bool | None = None) -> CommandResult:
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return CommandResult(False, f"PR #{pr_number} was not found", {})
        issue_number = linked_issue_number(pr)
        issue = self.gh.issue_view(issue_number) if issue_number is not None else {}
        checks = self.gh.pr_checks(pr_number)
        # Deterministic janitor gate BEFORE any packet/cross-family spend: an
        # obviously-not-ready PR (draft, conflicting, red CI, no issue link)
        # must cost zero review tokens. Failures don't move labels — they are
        # the worker's/CI's to fix, not a review decision.
        verdict = run_janitor(pr, checks, self.config)
        if not verdict.ok:
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
                "janitor_section": _janitor_section(verdict.warnings),
                "decision_command": f"{CLI_NAME} record-review --pr {pr_number} --decision approved --summary-file <path>",
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
        if issue_number is not None:
            transition(self.gh, self.config.labels, issue_number, "review_started")
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
            "janitor_warnings": list(verdict.warnings),
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
        return CommandResult(
            True,
            "review packet generated",
            {
                "pr": pr_number,
                "issue": issue_number,
                "prompt_path": str(prompt_path),
                "decision_path": str(decision_path),
                "cross_family_report": cf_result.report_path if cf_result else None,
                "cross_family_ok": cf_result.ok if cf_result else None,
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
        issue_number = linked_issue_number(pr) if pr else None
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
        reviewed_head_sha = pr.get("headRefOid") if pr else None
        decision_payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "summary": summary_text,
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_at": utc_now(),
        }
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
        decision_path = pr_dir / "review-decision.json"
        self._write_json(decision_path, decision_payload)
        # Merge-update (never in-place assignment) and persist BEFORE any GitHub
        # label mutation: a label-write failure or crash must not desync the
        # durable decision/counter from what actually happened.
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
        issue_number = linked_issue_number(pr)
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
        can_merge = summary.ready and (
            approved or not self.config.auto_merge.require_approved_review
        )
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        branch_deleted: bool | None = None
        label_error: str | None = None
        if can_merge and should_merge:
            # Merge, then labels, then best-effort branch deletion — in that
            # order. merge_pr is the irreversible step: persist status="merged"
            # to state IMMEDIATELY after it succeeds and BEFORE the label
            # transition, so a transition failure or Ctrl+C can't leave GitHub
            # merged while state.json still shows "reviewing" — which made
            # reconcile false-positive on every clean auto-merge and lost the
            # merged fact entirely on a crash between merge and save.
            merge_output = self.gh.merge_pr(pr_number, self.config.auto_merge.strategy)
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
        }
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
        )
        return self._cross_family_section(result.report_path), result

    def reconcile(self, *, fix: bool = False) -> CommandResult:
        """Detect (and optionally repair) drift between GitHub reality and the
        orchestrator's labels/state — e.g. a PR merged by hand outside
        merge-ready leaving `agent:in-progress` stale forever. Read-only unless
        ``fix`` is passed."""
        state = load_state(self.paths.state_file)
        drift = detect_drift(self.gh, state, self.config)
        fixed = False
        post_fix_drift: list[DriftItem] = []
        if fix and drift:
            new_state = apply_drift_fixes(self.gh, state, drift, self.config)
            save_state(self.paths.state_file, new_state)
            # The label removals above use allow_failure=True, so a failed
            # removal is silently swallowed. Re-detect against the new state to
            # verify the repairs actually landed before reporting success.
            post_fix_drift = detect_drift(self.gh, new_state, self.config)
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

    def loop(self, limit: int | None = None) -> CommandResult:
        intake = self.intake()
        dispatch = self.dispatch(limit)
        reviews: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for pr in self.gh.pr_list():
            issue_number = linked_issue_number(pr)
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
                        merges.append(self.merge_ready(pr_number).data)
                    else:
                        review = self.review(pr_number)
                        reviews.append(review.data)
                        decision = self._review_decision(pr_number)
                        if decision.get("decision") == "approved":
                            merges.append(self.merge_ready(pr_number).data)
                else:
                    review = self.review(pr_number)
                    reviews.append(review.data)
                    decision = self._review_decision(pr_number)
                    if decision.get("decision") == "approved":
                        merges.append(self.merge_ready(pr_number).data)
            except GitHubError as exc:
                errors.append({"pr": pr_number, "error": str(exc)})
        ok = dispatch.ok and not errors
        message = "loop complete"
        if not dispatch.ok:
            message = "loop completed with dispatch failures"
        if errors:
            message = f"loop completed with {len(errors)} PR error(s)"
        return CommandResult(
            ok,
            message,
            {
                "intake": intake.data,
                "dispatch": dispatch.data,
                "reviews": reviews,
                "merges": merges,
                "errors": errors,
            },
        )

    def _is_dispatchable(self, issue: dict[str, Any]) -> bool:
        names = label_names(issue)
        if self.config.labels.ready not in names:
            return False
        if names & self.config.labels.terminal:
            return False
        return not names & self.config.labels.active

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
            },
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt_path

    def _review_decision(self, pr_number: int) -> dict[str, Any]:
        decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
        if not decision_path.exists():
            return {"decision": "missing"}
        with decision_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
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
            "issue_number": linked_issue_number(pr),
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
