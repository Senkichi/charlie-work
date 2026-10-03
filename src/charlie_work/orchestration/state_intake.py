"""The ``intake`` delegate for ``OrchestratorApp``.

Extracted from ``OrchestratorApp`` in ``charlie_work.workflow`` during the
issue #2226 rework: this PR's ``ready_observed``/gate-routed additions pushed
``workflow.py`` past its file-size ratchet mark, so the intake lane lives
here as its own delegate module. The ``workflow_delegation`` installer
re-attaches the ``def`` onto the class; the body is verbatim apart from the
``_wf.``/``_labels.`` namespace rebinds (issue #1627) and the
``ready_observed_due`` helper now owning the new-episode computation in
``labels.py``.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work import labels as _labels


@_wf._guard_state_lock
def intake(self) -> _wf.CommandResult:
    issues = self.gh.issue_list(self.config.labels.ready)
    written: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    prose_only_deps_issues: list[int] = []
    # Gather all network results and write files outside the lock
    for issue in issues:
        issue_number = int(issue["number"])
        try:
            full_issue = self.gh.issue_view(issue_number)
        except _wf.GitHubError as exc:
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
        has_prose_deps = _wf.detect_prose_only_dependencies(body_text)
        has_structured_blockers = bool(_wf.parse_blockers(body_text))

        # If prose-only dependencies exist without structured blockers, label for human attention
        if has_prose_deps and not has_structured_blockers:
            prose_only_deps_issues.append(issue_number)
            try:
                # Issue #2226: gate-bound — state_path/repo auto-bound; dry-run no-ops.
                self.write_gate.apply_issue_labels(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    add=(self.config.labels.prose_only_deps,),
                    cause="intake_prose_only_deps",
                )
            except Exception:
                # Label add failure is non-blocking for intake
                pass

        written.append(
            {
                "issue": issue_number,
                "prompt_path": str(prompt_path),
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "labels": sorted(_wf.label_names(full_issue)),
                "updated_at": full_issue.get("updatedAt"),
            }
        )
    # Issue #1848: scan the open-issue blocker graph for cycles -- a loop
    # of open issues blocking each other (or an issue listing itself)
    # stalls every member forever while each still looks armed. Reporting
    # only: one warning per reported cycle is logged inside the scan, and
    # one blocker_cycle event per reported cycle is recorded below (the
    # report is bounded by MAX_REPORTED_CYCLES total plus per-component
    # cycle/DFS caps inside the scan; the intake event carries the
    # truncation marker). Runs with the rest of intake's reads, outside
    # the state lock; fail-open.
    blocker_cycle_scan = _wf.detect_open_blocker_cycles(self.gh)
    blocker_cycles = blocker_cycle_scan.cycles
    # Single lock for all state updates — skipped in dry-run (issue #618)
    if not self.dry_run:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for entry in written:
                issue_number = entry["issue"]
                # Merge-update, never replace: intake used to clobber dispatch
                # status recorded by earlier passes (production-confirmed).
                existing = state["issues"].get(str(issue_number), {})
                # Issue #2226: ``ready_observed`` fires once per issue per
                # Ready episode — the episode-boundary computation lives in
                # ``labels.ready_observed_due``.
                ready_is_new_episode = _labels.ready_observed_due(
                    self.paths.state_file, self.config.labels, issue_number, entry["labels"]
                )
                state["issues"][str(issue_number)] = {
                    **existing,
                    "number": issue_number,
                    "title": entry["title"],
                    "url": entry["url"],
                    "labels": entry["labels"],
                    "prompt_path": entry["prompt_path"],
                    "updated_at": entry["updated_at"],
                }
                if ready_is_new_episode:
                    state = self._record_event(
                        state,
                        "ready_observed",
                        {"issue_number": issue_number},
                    )
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
            for cycle in blocker_cycles:
                state = self._record_event(
                    state,
                    "blocker_cycle",
                    {"issue_numbers": cycle},
                )
            state = self._record_event(
                state,
                "intake",
                {
                    "issue_count": len(issues),
                    "failed_count": len(failed),
                    "blocker_cycles_reported": len(blocker_cycles),
                    "blocker_cycles_truncated": blocker_cycle_scan.truncated,
                },
            )
            # Issue #2226: gate-bound; a raw save_state here trips the exclusive-use check.
            self.write_gate.save_state(state)
    message = "intake complete"
    if self.dry_run:
        message = f"dry-run: would intake {len(written)} issue(s)"
    elif failed:
        message = f"intake completed with {len(failed)} failure(s)"
    if prose_only_deps_issues:
        message += f", {len(prose_only_deps_issues)} issue(s) labeled with prose-only dependencies"
    return _wf.CommandResult(
        not failed,
        message,
        {
            "issues": written,
            "failed": failed,
            "prose_only_deps_issues": prose_only_deps_issues,
            "blocker_cycles": blocker_cycles,
            "blocker_cycles_truncated": blocker_cycle_scan.truncated,
        },
    )
