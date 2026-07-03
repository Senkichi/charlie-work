from __future__ import annotations

import re
from pathlib import Path

from charlie_work.prompt_sections import section_variables
from charlie_work.prompts import render_prompt

ISSUE_VALUES = {
    "issue_number": 123,
    "issue_title": "Fix search",
    "issue_url": "https://example.test/issues/123",
    "issue_body": "Body text",
    "branch_name": "agent/issue-123-fix-search",
    "worker_model_tier": "capable",
}


def _render_worker_with_sections(template_name: str) -> str:
    """Render a worker prompt with section variables merged in.

    `render_prompt` now handles section resolution internally, so this helper
    just passes the issue values directly.
    """
    return render_prompt(template_name, ISSUE_VALUES)


def test_section_variables_discovers_package_sections() -> None:
    sections = section_variables()

    assert "section_issue_metadata" in sections
    assert "section_scope_contract" in sections
    assert "Do not batch unrelated fixes." in sections["section_scope_contract"]


def test_section_variables_has_no_hardcoded_name_list(tmp_path: Path) -> None:
    worker_sections = tmp_path / "worker_sections"
    worker_sections.mkdir()
    (worker_sections / "totally_new_section.md").write_text(
        "Brand new shared policy text.", encoding="utf-8"
    )

    sections = section_variables(search_dirs=(tmp_path,))

    assert sections["section_totally_new_section"] == "Brand new shared policy text."


def test_repo_local_section_overrides_package_section_by_filename(tmp_path: Path) -> None:
    worker_sections = tmp_path / "worker_sections"
    worker_sections.mkdir()
    (worker_sections / "scope_contract.md").write_text(
        "## Scope contract\n\n- Custom repo-local override.", encoding="utf-8"
    )

    sections = section_variables(search_dirs=(tmp_path,))

    assert (
        sections["section_scope_contract"] == "## Scope contract\n\n- Custom repo-local override."
    )


def test_repo_local_dir_without_override_falls_back_to_package_sections(tmp_path: Path) -> None:
    # tmp_path has no worker_sections/ dir at all — every section must still
    # come from the package default.
    package_sections = section_variables()
    merged_sections = section_variables(search_dirs=(tmp_path,))

    assert merged_sections == package_sections


def test_worker_prompt_renders_issue_values_with_merged_sections() -> None:
    prompt = _render_worker_with_sections("worker.md")

    assert "Issue #123" in prompt
    assert "agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_claude_code_worker_prompt_renders_issue_values_with_merged_sections() -> None:
    prompt = _render_worker_with_sections("worker_claude_code.md")

    assert "Issue #123" in prompt
    assert "git switch -c agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_rendered_worker_prompts_contain_shared_section_text_and_no_placeholders() -> None:
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)

        assert "- Number: #123" in prompt
        assert "- Model tier target: capable" in prompt
        assert "## Scope contract" in prompt
        assert "- Solve only issue #123." in prompt
        assert "If the fix touches security-sensitive behavior" in prompt
        assert "**Containment:**" in prompt
        assert "never resolve, cd into, or modify any other checkout" in prompt
        assert not re.search(r"\$section_\w+", prompt), (
            f"leftover $section_ placeholder in rendered {template_name}"
        )


def test_worker_templates_reference_section_variables_in_source() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"

    for template_name in ("worker.md", "worker_claude_code.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")

        assert "$section_issue_metadata" in text
        assert "$section_scope_contract" in text
        # The extracted blocks must be gone from the template body itself —
        # otherwise this isn't deduplication, it's just an unused partial.
        assert "Do not perform opportunistic refactors." not in text


def test_attacker_controlled_placeholders_not_expanded() -> None:
    """Verify that attacker-controlled values containing $placeholders are not expanded.

    This is the security fix for issue #8: an issue body containing $section_* or
    $issue_number should render as literal text, not be expanded in a second pass.
    """
    malicious_values = {
        "issue_number": 8,
        "issue_title": "Test issue",
        "issue_url": "https://example.test/issues/8",
        "issue_body": "This contains $section_scope_contract and $issue_number",
        "branch_name": "agent/issue-8-test",
        "worker_model_tier": "capable",
    }

    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = render_prompt(template_name, malicious_values)

        # The literal placeholder text should appear in the prompt
        assert "$section_scope_contract" in prompt, "Expected $section_scope_contract in prompt"
        assert "$issue_number" in prompt, "Expected $issue_number in prompt"
        # The attacker's placeholders should NOT be expanded to their values
        # (i.e., we should NOT see "This contains [resolved section text] and 8")
        assert "This contains $section_scope_contract and $issue_number" in prompt


def test_legitimate_partial_placeholders_still_resolve() -> None:
    """Verify that $placeholders inside worker_sections/*.md partials still resolve.

    The fix must not break the legitimate use case: partials can contain $issue_number
    and other placeholders, which should be resolved when the partial is rendered.
    """
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = render_prompt(template_name, ISSUE_VALUES)

        # These should be resolved from the section partials
        assert "- Number: #123" in prompt
        assert "- Solve only issue #123." in prompt
        assert "- Model tier target: capable" in prompt
        # No leftover $section_ placeholders
        assert "$section_" not in prompt


def test_rework_prompt_includes_merge_main_instruction() -> None:
    """Verify that the rework prompt instructs workers to merge origin/main first."""
    from pathlib import Path

    rework_values = {
        "pr_number": 456,
        "pr_title": "fix: search is broken",
        "pr_url": "https://example.test/pull/456",
        "issue_number": 123,
        "review_summary": "Fix the typo in the search function.",
    }
    # The rework.md template is in the package prompts dir, not repo-local
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    prompt = render_prompt("rework.md", rework_values, search_dirs=(prompts_dir,))

    assert "merge the PR's base branch" in prompt
    assert "incorporate any base changes" in prompt
