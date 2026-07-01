from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import CLI_NAME
from .adapters import SessionRequest, dispatch_sessions
from .checks import summarize_checks
from .config import CrossFamilyConfig, OrchestratorConfig
from .cross_family import CrossFamilyResult, run_cross_family_review
from .github import GitHub, label_names, linked_issue_number
from .paths import RuntimePaths
from .prompts import render_prompt
from .state import append_event, load_state, save_state, utc_now


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any]


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
        return CommandResult(True, "labels ensured", {"labels": self.config.labels.all})

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
            state["issues"][str(issue_number)] = {
                "number": issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "labels": sorted(label_names(full_issue)),
                "prompt_path": str(prompt_path),
                "updated_at": full_issue.get("updatedAt"),
            }
            written.append({"issue": issue_number, "prompt_path": str(prompt_path)})
        append_event(state, "intake", {"issue_count": len(issues)})
        save_state(self.paths.state_file, state)
        return CommandResult(True, "intake complete", {"issues": written})

    def dispatch(
        self, limit: int | None = None, *, only_issues: str | None = None
    ) -> CommandResult:
        issues = self.gh.issue_list(self.config.labels.ready)
        dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit
        candidates = [issue for issue in issues if self._is_dispatchable(issue)]
        if only_issues:
            wanted = parse_issue_numbers(only_issues)
            by_number = {int(issue["number"]): issue for issue in candidates}
            selected = [by_number[number] for number in wanted if number in by_number]
        else:
            selected = candidates[:dispatch_limit]
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        state = load_state(self.paths.state_file)
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
            self.config.devin.adapter,
            self.config.devin.dispatch_command,
            self.config.devin.command_timeout_seconds,
            session_requests,
        )
        successful_issue_numbers = {
            result.issue_number for result in dispatch_results if result.ok
        }
        failed_issue_numbers = {
            result.issue_number for result in dispatch_results if not result.ok
        }
        for request in session_requests:
            full_issue = full_issues[request.issue_number]
            if request.issue_number in successful_issue_numbers:
                self.gh.add_issue_label(request.issue_number, self.config.labels.in_progress)
                self.gh.remove_issue_label(request.issue_number, self.config.labels.queued)
            state["issues"][str(request.issue_number)] = {
                "number": request.issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "branch_name": request.branch_name,
                "prompt_path": str(request.prompt_path),
                "status": "dispatched"
                if request.issue_number in successful_issue_numbers
                else "dispatch_failed",
                "dispatched_at": utc_now()
                if request.issue_number in successful_issue_numbers
                else None,
            }
        append_event(
            state,
            "dispatch",
            {
                "issue_numbers": sorted(successful_issue_numbers),
                "failed_issue_numbers": sorted(failed_issue_numbers),
            },
        )
        save_state(self.paths.state_file, state)
        result_dicts = [result.to_dict() for result in dispatch_results]
        return CommandResult(
            not failed_issue_numbers,
            "dispatch complete"
            if not failed_issue_numbers
            else "dispatch completed with failures",
            {
                "selected_count": len(successful_issue_numbers),
                "attempted_count": len(session_requests),
                "failed_count": len(failed_issue_numbers),
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
        if issue_number is not None:
            self.gh.add_issue_label(issue_number, self.config.labels.pr_open)
            self.gh.add_issue_label(issue_number, self.config.labels.reviewing)
        state = load_state(self.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "url": pr.get("url"),
            "issue_number": issue_number,
            "prompt_path": str(prompt_path),
            "decision_path": str(decision_path),
            "status": "reviewing",
            "cross_family_report": cf_result.report_path if cf_result else None,
            "cross_family_ok": cf_result.ok if cf_result else None,
        }
        append_event(
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
        decision_payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "summary": summary_text,
            "reviewed_at": utc_now(),
        }
        decision_path = pr_dir / "review-decision.json"
        self._write_json(decision_path, decision_payload)
        rework_path: str | None = None
        if decision == "request_changes":
            rework_path = str(self._write_rework_prompt(pr, issue_number, summary_text))
            if issue_number is not None:
                self.gh.add_issue_label(issue_number, self.config.labels.needs_rework)
                self.gh.remove_issue_label(issue_number, self.config.labels.reviewing)
            if comment and summary_text:
                self._comment_pr(pr_number, summary_text)
        elif decision == "blocked" and issue_number is not None:
            self.gh.add_issue_label(issue_number, self.config.labels.human_needed)
        state = load_state(self.paths.state_file)
        state["prs"].setdefault(str(pr_number), {})["decision"] = decision
        state["prs"][str(pr_number)]["decision_path"] = str(decision_path)
        append_event(state, "record_review", {"pr_number": pr_number, "decision": decision})
        save_state(self.paths.state_file, state)
        return CommandResult(
            True,
            "review recorded",
            {
                "pr": pr_number,
                "decision": decision,
                "decision_path": str(decision_path),
                "rework_path": rework_path,
            },
        )

    def merge_ready(self, pr_number: int, *, merge: bool | None = None) -> CommandResult:
        pr = self.gh.pr_view(pr_number)
        if not pr:
            return CommandResult(False, f"PR #{pr_number} was not found", {})
        issue_number = linked_issue_number(pr)
        checks = self.gh.pr_checks(pr_number)
        summary = summarize_checks(checks, self.config.auto_merge.required_checks)
        decision = self._review_decision(pr_number)
        approved = decision.get("decision") == "approved"
        can_merge = summary.ready and (
            approved or not self.config.auto_merge.require_approved_review
        )
        should_merge = self.config.auto_merge.enabled if merge is None else merge
        merge_output: str | None = None
        branch_deleted: bool | None = None
        if can_merge and should_merge:
            # Merge, then labels, then best-effort branch deletion — in that
            # order. A branch-deletion failure (e.g. the head branch is checked
            # out in a local worktree) must never leave a merged PR's issue
            # labels un-updated, which is exactly what the old coupled
            # `gh pr merge --delete-branch` behavior did.
            merge_output = self.gh.merge_pr(pr_number, self.config.auto_merge.strategy)
            if issue_number is not None:
                self.gh.add_issue_label(issue_number, self.config.labels.done)
                for label in self.config.labels.active:
                    self.gh.remove_issue_label(issue_number, label)
            if self.config.auto_merge.delete_branch:
                head_ref = str(pr.get("headRefName") or "")
                branch_deleted = self.gh.delete_branch(head_ref) if head_ref else False
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
        }
        state = load_state(self.paths.state_file)
        state["prs"].setdefault(str(pr_number), {}).update(data)
        append_event(
            state,
            "merge_ready",
            {"pr_number": pr_number, "can_merge": can_merge, "merged": bool(merge_output)},
        )
        save_state(self.paths.state_file, state)
        return CommandResult(True, "merge readiness evaluated", data)

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
        append_event(
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
        # Idempotent: a non-empty report is reused, so repeated review()/loop() passes
        # don't re-burn the cross-family model on the same PR.
        if report_path.exists() and report_path.stat().st_size > 0:
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
        for pr in self.gh.pr_list():
            issue_number = linked_issue_number(pr)
            if issue_number is None:
                continue
            review = self.review(int(pr["number"]))
            reviews.append(review.data)
            decision = self._review_decision(int(pr["number"]))
            if decision.get("decision") == "approved":
                merges.append(self.merge_ready(int(pr["number"])).data)
        return CommandResult(
            dispatch.ok,
            "loop complete" if dispatch.ok else "loop completed with dispatch failures",
            {
                "intake": intake.data,
                "dispatch": dispatch.data,
                "reviews": reviews,
                "merges": merges,
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
