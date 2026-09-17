"""Worker-prompt rendering moved out of ``OrchestratorApp`` (Track 2 Phase B, L07).

Bodies relocated verbatim from ``charlie_work.workflow`` (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). ``workflow_delegation._install_delegates`` re-attaches each
``def`` unwrapped onto ``OrchestratorApp``.

``_render`` reaches ``render_prompt`` through ``_wf.render_prompt`` rather than
importing it directly: ``tests/test_prompt_template_drift_check.py`` patches
``render_prompt`` on the ``charlie_work.workflow`` module object
(``monkeypatch.setattr(workflow_mod, "render_prompt", ...)``), so the call must
resolve through that seam (Tier D, #1627). ``render_prompt`` therefore stays a
deliberate re-export in ``charlie_work.workflow``. The remaining free functions
used here (``fenced_block`` and the four ``assert_*`` prompt contracts, plus
``prompt_template_digest``) are patched by no test, so they are imported
directly from their source modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.markdown_fence import fenced_block
from charlie_work.prompts import (
    assert_containment,
    assert_conventional_commit_title,
    assert_execution_contract,
    assert_no_merge_contract,
    prompt_template_digest,
)


def _render(self, template_name: str, values: dict[str, Any]) -> str:
    return _wf.render_prompt(template_name, values, search_dirs=self.prompt_dirs)


def _write_worker_prompt(self, issue: dict[str, Any], *, dry_run: bool = False) -> Path:
    issue_number = int(issue["number"])
    issue_dir = self.paths.issues / f"issue-{issue_number}"
    prompt_path = issue_dir / "worker-prompt.md"
    prompt = self._render(
        self.config.dispatch.worker_template,
        {
            "issue_number": issue_number,
            "issue_title": issue.get("title", ""),
            "issue_url": issue.get("url", ""),
            # The raw body stays available to templates; no shipped one
            # references it since #883, but rendering it bare is a
            # legitimate thing for a template to want.
            "issue_body": issue.get("body", ""),
            # Pre-fenced, not bare: the fence width depends on the body's
            # own backtick runs, so it cannot be written in the template
            # (#883). ``or ""`` guards a null body, which would otherwise
            # reach the regex as None.
            "issue_body_block": fenced_block(str(issue.get("body") or ""), "md"),
            "issue_comments": self._render_issue_comments(issue),
            "branch_name": self._branch_name(issue),
            # Issue #1444: the module-map section, derived from the live
            # tree at packet build time. Fail-soft: a parse failure yields
            # an empty string (omitted section) plus a
            # ``worker_module_map_failed`` warning event, never a dispatch
            # failure. ``_build_module_map_value`` is the single point of
            # enforcement for that fail-soft contract.
            "module_map": self._build_module_map_value(issue_number),
            # Issue #1460: the attachment-point placement clause, gated
            # solely on `.attachment-budgets.json`'s presence. Fail-soft
            # like module_map: a malformed baseline yields an empty
            # string (omitted clause) plus a
            # ``worker_attachment_budget_failed`` warning event.
            "attachment_budget": self._build_attachment_budget_value(issue_number),
        },
    )
    # Issue #714: enforce the no-merge contract on the *rendered output*
    # so a repo-local flat override that drops $section_no_merge_contract
    # is caught at the dispatch boundary rather than shipping a worker
    # with no instruction against merging/closing/relabeling its own PR.
    assert_no_merge_contract(prompt, context=f"worker prompt for issue #{issue_number}")
    # Issue #715: enforce the conventional-commit title instruction on the
    # *rendered output* so a repo-local flat override that mandates a stale
    # non-conventional-commit title format (e.g. 'Fix #N: ...') is caught
    # at the dispatch boundary rather than shipping a worker whose every PR
    # trips the janitor's _check_title_conventional warning.
    assert_conventional_commit_title(prompt, context=f"worker prompt for issue #{issue_number}")
    # Issue #717: enforce the execution-contract escalation trigger on the
    # *rendered output* so a repo-local flat override that drops
    # $section_execution_contract — leaving a blanket "never run the full
    # local suite" prohibition with no carve-out for contract-changing diffs
    # (public function signature/return shape, exception type/message
    # consumed elsewhere, DB schema, or module re-export) — is caught at the
    # dispatch boundary rather than shipping a worker who can change a
    # contract surface and push without ever exercising the wider suite.
    assert_execution_contract(prompt, context=f"worker prompt for issue #{issue_number}")
    # Issue #1010: enforce the widened containment clause on the *rendered
    # output* so a repo-local flat override that drops
    # $section_scope_contract or reverts to the old repo-scoped wording
    # (which does not cover a different repo) is caught at the dispatch
    # boundary rather than shipping a worker with no effective prohibition
    # against editing a sibling repo's checkout.
    assert_containment(prompt, context=f"worker prompt for issue #{issue_number}")
    # Issue #618: the dry-run dispatch branch promises "skip all state
    # writes, label transitions, and file mutations" — mkdir + write_text
    # here would violate that, and for a dead-worker recovery candidate
    # (previous status "dispatched", same branch) would silently overwrite
    # the prompt a crashed worker was launched with, destroying the
    # forensic record the preview was meant to inspect. The assertions
    # above are validation on the rendered text, not file mutations, so
    # they stay — a broken template should fail the preview too.
    if not dry_run:
        issue_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def _review_template_sha(self) -> str:
    """SHA-256 digest of the resolved review template + referenced section
    partials.

    Captures every input to ``review()``'s prompt render that lives in a
    template file: the ``review.md`` template (package default or repo-local
    override) and every ``worker_sections/*.md`` partial it references. A
    change in any of those files changes this digest, so ``loop()`` can
    treat a digest mismatch as packet staleness alongside a head-SHA
    mismatch (issue #592).
    """
    return prompt_template_digest("review.md", search_dirs=self.prompt_dirs)
