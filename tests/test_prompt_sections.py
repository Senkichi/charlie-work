from __future__ import annotations

import re
from pathlib import Path
from string import Template

from devin_orchestrator.prompt_sections import section_variables
from devin_orchestrator.prompts import render_prompt

ISSUE_VALUES = {
    "issue_number": 123,
    "issue_title": "Fix search",
    "issue_url": "https://example.test/issues/123",
    "issue_body": "Body text",
    "branch_name": "agent/issue-123-fix-search",
    "worker_model_tier": "capable",
}


def _render_worker_with_sections(template_name: str) -> str:
    """Mirror the future `render_prompt` integration: merge section_variables()
    into the context, then substitute twice.

    `string.Template.safe_substitute` is not recursive — injected section text
    that itself contains `$issue_number`-style placeholders is only resolved
    on a second pass. `render_prompt` renders once, so until the orchestrator
    wires `section_variables()` + a second substitution pass into it, tests
    here perform that second pass explicitly to prove the templates are
    semantically correct once that wiring lands.
    """
    context = {**ISSUE_VALUES, **section_variables()}
    once = render_prompt(template_name, context)
    str_values = {key: str(value) for key, value in context.items()}
    return Template(once).safe_substitute(str_values)


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
        assert not re.search(r"\$section_\w+", prompt), (
            f"leftover $section_ placeholder in rendered {template_name}"
        )


def test_worker_templates_reference_section_variables_in_source() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "devin_orchestrator" / "prompts"

    for template_name in ("worker.md", "worker_claude_code.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")

        assert "$section_issue_metadata" in text
        assert "$section_scope_contract" in text
        # The extracted blocks must be gone from the template body itself —
        # otherwise this isn't deduplication, it's just an unused partial.
        assert "Do not perform opportunistic refactors." not in text
