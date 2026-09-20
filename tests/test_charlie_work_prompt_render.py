"""Worker-prompt rendering and template-dir override/fallback.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    OrchestratorConfig,
    RuntimeConfig,
)
from charlie_work.markdown_fence import fenced_block
from charlie_work.paths import runtime_paths
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import render_prompt
from charlie_work.workflow import OrchestratorApp


def test_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "issue_body_block": fenced_block("Body text", "md"),
            "branch_name": "agent/issue-123-fix-search",
            "issue_comments": "",
            "module_map": "",
            "attachment_budget": "",
            **prompt_test_command_values("", None),
        },
    )

    assert "Issue #123" in prompt
    assert "agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_claude_code_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker_claude_code.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "issue_body_block": fenced_block("Body text", "md"),
            "branch_name": "agent/issue-123-fix-search",
            "issue_comments": "",
            "module_map": "",
            "attachment_budget": "",
            **prompt_test_command_values("", None),
        },
    )

    assert "Issue #123" in prompt
    assert "git switch -c agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_repo_local_prompt_dir_overrides_package_template(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "CUSTOM for #$issue_number on $branch_name", encoding="utf-8"
    )

    prompt = render_prompt(
        "worker.md",
        {"issue_number": 5, "branch_name": "agent/issue-5-x"},
        search_dirs=(override_dir,),
    )

    assert prompt == "CUSTOM for #5 on agent/issue-5-x"


def test_missing_repo_local_template_falls_back_to_package(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()

    prompt = render_prompt(
        "rework.md",
        {
            "pr_number": 9,
            "pr_title": "t",
            "pr_url": "u",
            "issue_number": 1,
            "dispatch_note": "s",
            "dispatch_note_block": fenced_block("s", "md"),
            "required_changes_section": "",
            "branch_name": "agent/issue-1-t",
            **prompt_test_command_values("", None),
        },
        search_dirs=(override_dir,),
    )

    assert "Rework Task: PR #9" in prompt


def test_app_prompts_dir_override_wins_for_worker_prompt(tmp_path: Path) -> None:
    override_dir = tmp_path / "orchestrator-prompts"
    override_dir.mkdir()
    # The override must carry the no-merge contract markers (issue #714),
    # the conventional-commit title instruction (issue #715), the
    # execution-contract escalation trigger (issue #717), and the widened
    # containment clause markers (issue #1010):
    # _write_worker_prompt's post-render guards reject a flat override that
    # drops any of these.
    (override_dir / "worker.md").write_text(
        "REPO-LOCAL #$issue_number\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n\n"
        "**Execution contract (self-detect from your diff):** the default is "
        "the targeted command. Only if the diff changes any public function "
        "signature/return shape, run the **FULL suite** locally at the final "
        "head before pushing.\n\n"
        "**Containment:** All file edits happen in the assigned worktree; "
        "never modify any path outside the assigned worktree root.\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir="orchestrator-prompts"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.gh.prs[0]["state"] = "CLOSED"
    app.dispatch(limit=1)

    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == (
        "REPO-LOCAL #123\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n\n"
        "**Execution contract (self-detect from your diff):** the default is "
        "the targeted command. Only if the diff changes any public function "
        "signature/return shape, run the **FULL suite** locally at the final "
        "head before pushing.\n\n"
        "**Containment:** All file edits happen in the assigned worktree; "
        "never modify any path outside the assigned worktree root.\n"
    )
