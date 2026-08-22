from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from charlie_work.markdown_fence import fenced_block
from charlie_work.prompt_sections import section_variables
from charlie_work.prompts import render_prompt

ISSUE_VALUES = {
    "issue_number": 123,
    "issue_title": "Fix search",
    "issue_url": "https://example.test/issues/123",
    "issue_body": "Body text",
    "issue_body_block": fenced_block("Body text", "md"),
    "branch_name": "agent/issue-123-fix-search",
    "worker_model_tier": "capable",
    "issue_comments": "",
    "pr_number": 456,
    "pr_title": "fix: search is broken",
    "pr_url": "https://example.test/pull/456",
    "dispatch_note": "Fix the typo in the search function.",
    "dispatch_note_block": fenced_block("Fix the typo in the search function.", "md"),
    "required_changes_section": "",
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


def test_execution_contract_section_present_and_rendered() -> None:
    """Verify the conditional full-suite execution contract is a shared section and appears in all worker prompts."""
    sections = section_variables()

    assert "section_execution_contract" in sections
    contract = sections["section_execution_contract"]
    assert "self-detect from your diff" in contract
    assert "the default is the targeted command" in contract
    assert "public function signature" in contract
    assert "return shape" in contract
    assert "exception type" in contract
    assert "DB schema" in contract
    assert "module re-export" in contract
    assert "run the **FULL suite** locally at the final head before pushing" in contract
    assert "For all other diffs, do NOT run the full suite locally" in contract
    assert "CI runs it on every push and is the merge gate" in contract
    assert "Quote the exact command you ran" in contract

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert contract in prompt


def test_api_shape_validation_section_present_and_rendered() -> None:
    """Verify the live-payload API shape validation section is a shared section and appears in both worker prompts."""
    sections = section_variables()

    assert "section_api_shape_validation" in sections
    validation = sections["section_api_shape_validation"]
    assert "live call transcript" in validation
    assert "signature/docstring" in validation
    assert "tests/fixtures/" in validation

    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert validation in prompt


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
        assert (
            "never resolve, cd into, or modify any path outside the assigned worktree root"
            in prompt
        )
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
        "issue_body_block": fenced_block(
            "This contains $section_scope_contract and $issue_number", "md"
        ),
        "branch_name": "agent/issue-8-test",
        "worker_model_tier": "capable",
        "issue_comments": "",
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


def test_rework_prompt_includes_conditional_base_merge_instruction() -> None:
    """F4 (docs/plans/rework-findings-channel.md, #6): the rework prompt still
    instructs workers to merge the PR's base branch when it's actually needed,
    but only as a demoted, self-conditional trigger -- not as the mandated
    opening action of every rework."""
    from pathlib import Path

    rework_values = {
        "pr_number": 456,
        "pr_title": "fix: search is broken",
        "pr_url": "https://example.test/pull/456",
        "issue_number": 123,
        "dispatch_note": "Fix the typo in the search function.",
        "dispatch_note_block": fenced_block("Fix the typo in the search function.", "md"),
        "required_changes_section": "",
        "branch_name": "agent/issue-123-fix-search",
    }
    # The rework.md template is in the package prompts dir, not repo-local
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    prompt = render_prompt("rework.md", rework_values, search_dirs=(prompts_dir,))
    normalized = " ".join(prompt.split())

    assert "merge the PR's base branch" in normalized
    assert "If your branch is behind its base or the PR shows a merge conflict" in normalized


def test_rework_prompt_includes_push_then_verify_final_step() -> None:
    """Verify that the rework prompt includes the push-then-verify FINAL STEP with resolved placeholders.

    This test goes through the REAL call site (_write_rework_prompt) to ensure the branch_name
    is correctly extracted from PR headRefName and rendered without unresolved placeholders.
    """
    from pathlib import Path
    import tempfile

    # Use the real workflow._write_rework_prompt call site
    from charlie_work.workflow import OrchestratorApp
    from charlie_work.config import OrchestratorConfig, DevinConfig
    from charlie_work.paths import runtime_paths

    # Minimal mock GitHub client - only what _write_rework_prompt needs
    class MinimalFakeGitHub:
        def __init__(self):
            self.labels_added = []
            self.labels_removed = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = OrchestratorConfig(
            devin=DevinConfig(adapter="manual"),
        )
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        fake_gh = MinimalFakeGitHub()
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        # PR dict with headRefName (the real source of branch_name)
        pr = {
            "number": 456,
            "title": "fix: search is broken",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
        }

        # Call the real _write_rework_prompt method
        # The method expects the PR directory to exist
        pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
        pr_dir.mkdir(parents=True, exist_ok=True)
        rework_path = app._write_rework_prompt(pr, 123, "Fix the typo in the search function.")

        # Read the rendered prompt
        prompt = rework_path.read_text(encoding="utf-8")

        # Verify the FINAL STEP section exists
        assert "## FINAL STEP — push and verify" in prompt
        # Verify the key instruction about local commits not being done
        assert "Committing locally is NOT done" in prompt
        # Verify the canonical targeted test command (operator directive
        # 2026-07-11: local runs are targeted only, full suite stays on CI)
        assert "uv run --extra dev pytest tests/test_<touched_module>.py -q --tb=short" in prompt
        # Verify the push instruction with RESOLVED branch name (from headRefName)
        assert "git push origin agent/issue-123-fix-search" in prompt
        # Verify the PR head verification with RESOLVED PR number
        assert "gh pr view 456 --json headRefOid" in prompt
        # Verify the comparison instruction
        assert "headRefOid" in prompt
        assert "git rev-parse HEAD" in prompt
        # CRITICAL: assert NO unresolved $ placeholders remain anywhere. This is
        # narrower than a bare `"$" not in prompt` check: worker_sections
        # partials may legitimately contain a literal `$` that isn't a
        # `string.Template` placeholder at all -- e.g. `mutation_check.md`'s
        # shell snippet `git show $(git merge-base ...)` uses `$(` (command
        # substitution), which `get_identifiers()` correctly ignores because it
        # is not `$identifier`/`${identifier}` shaped.
        assert not string.Template(prompt).get_identifiers(), (
            f"Unresolved placeholders found in rendered prompt:\n{prompt}"
        )


def test_worker_prompt_includes_push_then_verify() -> None:
    """Verify that the worker.md prompt includes push-then-verify in the Done condition."""
    prompt = _render_worker_with_sections("worker.md")

    # Verify the key instruction about local commits not being done
    assert "Committing locally is NOT done" in prompt
    # Verify the push instruction with resolved branch name
    assert "git push origin agent/issue-123-fix-search" in prompt
    # Verify the PR head verification
    assert "gh pr view agent/issue-123-fix-search --json headRefOid" in prompt
    # Verify the comparison instruction
    assert "headRefOid" in prompt
    assert "git rev-parse HEAD" in prompt


def test_claude_code_worker_prompt_includes_push_then_verify() -> None:
    """Verify that the worker_claude_code.md prompt includes push-then-verify in the Done condition."""
    prompt = _render_worker_with_sections("worker_claude_code.md")

    # Verify the key instruction about local commits not being done
    assert "Committing locally is NOT done" in prompt
    # Verify the push instruction with resolved branch name
    assert "git push -u origin agent/issue-123-fix-search" in prompt
    # Verify the PR head verification
    assert "gh pr view agent/issue-123-fix-search --json headRefOid" in prompt
    # Verify the comparison instruction
    assert "headRefOid" in prompt
    assert "git rev-parse HEAD" in prompt


def test_worker_and_rework_templates_contain_identical_canonical_test_command() -> None:
    """Verify that worker.md and rework.md contain the exact same canonical test command string.

    This prevents silent drift where one template uses the full command and the other
    uses a partial variant (issue #91).
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    canonical_command = "uv run --extra dev pytest tests/test_<touched_module>.py -q --tb=short"

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")
        assert canonical_command in text, (
            f"Template {template_name} does not contain the canonical test command. "
            f"Expected to find: {canonical_command}"
        )


def test_rendered_worker_prompts_contain_canonical_test_command() -> None:
    """Verify that rendered worker prompts contain the canonical test command.

    This test goes through the real render_prompt call to ensure the command
    appears in the final rendered output that workers actually see.
    """
    canonical_command = "uv run --extra dev pytest tests/test_<touched_module>.py -q --tb=short"

    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert canonical_command in prompt, (
            f"Rendered {template_name} does not contain the canonical test command. "
            f"Expected to find: {canonical_command}"
        )


def test_rendered_worker_prompts_require_completion_report_with_command_and_count() -> None:
    """Verify that rendered worker prompts require the completion report to quote the command and count."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        # Verify the instruction to quote the exact command
        assert "Quote the exact command you ran" in prompt
        # Verify the instruction to quote the collected/passed count
        assert "collected/passed count" in prompt
        # Verify the example format
        assert "300 collected, 300 passed" in prompt


def test_rendered_worker_prompt_qualified_test_bullet() -> None:
    """Verify that the rendered worker.md prompt has a qualified /test bullet, not bare /test.

    This test goes through the real render path to ensure the rendered worker prompt
    no longer advertises unqualified /test (issue #95, AC2). The mutation gate:
    reverting the worker.md bullet edit must fail this test.
    """
    prompt = _render_worker_with_sections("worker.md")

    # Verify the qualified bullet text is present
    qualified_bullet = "`/test` - Run the test suite and verify all tests pass (only if it wraps the canonical command below)"
    assert qualified_bullet in prompt, (
        f"Qualified /test bullet not found in rendered worker.md. "
        f"Expected to find: {qualified_bullet}"
    )

    # Verify no bare /test bullet exists (without the qualification)
    # This regex matches a bullet line with /test that does NOT have the qualification
    bare_test_pattern = r"^- `/test`[^\(]*$"
    bare_test_matches = re.findall(bare_test_pattern, prompt, re.MULTILINE)
    assert not bare_test_matches, (
        f"Found bare /test bullet(s) without qualification in rendered worker.md: {bare_test_matches}"
    )


def test_worker_and_rework_templates_contain_body_reconciliation_requirement() -> None:
    """Verify that worker.md, worker_claude_code.md, and rework.md contain the body-reconciliation requirement.

    This prevents silent drift where workers don't reconcile PR body claims with the final pushed head
    (issue #99). The mutation gate: removing the clause from any one template must fail this test.
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    body_reconciliation_text = "After verifying the push, re-read your PR body and make every claim literally true at the pushed head"

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")
        assert body_reconciliation_text in text, (
            f"Template {template_name} does not contain the body-reconciliation requirement. "
            f"Expected to find: {body_reconciliation_text}"
        )


def test_worker_prompts_require_git_ls_remote_push_verification() -> None:
    """Verify that worker.md, worker_claude_code.md, and rework.md require a git ls-remote check after push (issue #256)."""
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "git ls-remote origin agent/issue-123-fix-search" in prompt
        assert "git rev-parse HEAD" in prompt
        assert "retry the push" in prompt


def test_worker_prompts_require_config_parity_check() -> None:
    """Verify that worker.md and worker_claude_code.md require config parity (issue #256)."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "config example file" in prompt
        assert "counterpart" in prompt
        assert "parity tests" in prompt


def test_worker_prompts_require_ruff_preflight_before_commit() -> None:
    """Verify that worker.md and worker_claude_code.md require ruff check + format before commit (issue #256)."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "uv run ruff check ." in prompt
        assert "uv run ruff format ." in prompt
        assert "Before committing" in prompt


def test_worker_prompts_require_parallel_investigation() -> None:
    """Verify that worker.md and worker_claude_code.md instruct parallel independent investigation (issue #256)."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "fan out independent investigation" in prompt
        assert "in parallel" in prompt


def test_worker_prompts_require_body_checklist_revalidation() -> None:
    """Verify that worker.md, worker_claude_code.md, and rework.md require checklist revalidation at the final head (issue #256)."""
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "including the checklist" in prompt


def test_worker_prompts_contain_no_merge_contract() -> None:
    """Verify that worker prompts carry the no-merge contract (issue #502)."""
    sections = section_variables()
    assert "section_no_merge_contract" in sections
    contract = sections["section_no_merge_contract"]
    assert "Your deliverable ENDS at pushing the branch and opening the PR" in contract
    assert "gh pr merge" in contract
    assert "never" in contract.lower()

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "## No-merge contract" in prompt
        assert "Your deliverable ENDS at pushing the branch and opening the PR" in prompt


def test_assert_no_merge_contract_passes_for_package_templates() -> None:
    """The guard accepts prompts rendered from the package templates (issue #714)."""
    from charlie_work.prompts import assert_no_merge_contract

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        # Must not raise — the package templates all reference
        # $section_no_merge_contract.
        assert_no_merge_contract(prompt)


def test_assert_no_merge_contract_rejects_flat_override_without_contract(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override that drops the no-merge contract (issue #714).

    This is the exact scenario the issue describes: a repo-local flat
    whole-file ``worker.md`` override that does not reference
    ``$section_no_merge_contract`` renders without the prohibition, and the
    guard must refuse to let it through.
    """
    from charlie_work.prompts import (
        MissingNoMergeContractError,
        assert_no_merge_contract,
    )

    # Simulate a flat override that carries no no-merge contract — the
    # kind job-cannon shipped before PR #1564 migrated it to package
    # templates.
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\nGo fix the issue. No safety sections here.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        {
            "issue_number": 42,
            "issue_title": "Test",
            "issue_url": "https://example.test/issues/42",
            "issue_body": "",
            "issue_body_block": "",
            "branch_name": "agent/issue-42-test",
            "worker_model_tier": "capable",
            "issue_comments": "",
        },
        search_dirs=(override_dir,),
    )
    assert "## No-merge contract" not in prompt

    with pytest.raises(MissingNoMergeContractError) as exc_info:
        assert_no_merge_contract(prompt, context="worker prompt for issue #42")
    assert "issue #714" in str(exc_info.value)
    assert "worker prompt for issue #42" in str(exc_info.value)


def test_assert_no_merge_contract_accepts_override_with_contract(
    tmp_path: Path,
) -> None:
    """The guard accepts a flat override that does carry the contract (issue #714).

    A repo-local override is not inherently wrong — it just must not drop
    the safety-critical no-merge prohibition.
    """
    from charlie_work.prompts import assert_no_merge_contract

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR. "
        "You must **never** merge or close PRs.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        {
            "issue_number": 42,
            "issue_title": "Test",
            "issue_url": "https://example.test/issues/42",
            "issue_body": "",
            "issue_body_block": "",
            "branch_name": "agent/issue-42-test",
            "worker_model_tier": "capable",
            "issue_comments": "",
        },
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the contract text.
    assert_no_merge_contract(prompt)


def test_worker_prompts_contain_conventional_commit_title_instruction() -> None:
    """Verify that worker prompts carry the conventional-commit title instruction (issue #715)."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "Conventional-Commits format" in prompt


def test_assert_conventional_commit_title_passes_for_package_templates() -> None:
    """The guard accepts prompts rendered from the package worker templates (issue #715)."""
    from charlie_work.prompts import assert_conventional_commit_title

    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        # Must not raise — the package worker templates all instruct
        # conventional-commit titles.
        assert_conventional_commit_title(prompt)


def test_assert_conventional_commit_title_rejects_stale_flat_override(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override with a stale title format (issue #715).

    This is the exact scenario the issue describes: a repo-local flat
    whole-file ``worker.md`` override that mandates ``Fix #$issue_number:
    <short title>`` instead of Conventional Commits, so every PR the worker
    opens trips the janitor's ``_check_title_conventional`` warning.
    """
    from charlie_work.prompts import (
        MissingConventionalTitleError,
        assert_conventional_commit_title,
    )

    # Simulate job-cannon's stale flat override — the kind the issue says
    # mandates ``Fix #$issue_number: <short title>``.
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task: Issue #$issue_number\n\n"
        "## PR requirements\n\n"
        "- Title format: `Fix #$issue_number: <short title>`.\n"
        "- Body must include `Closes #$issue_number`.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    assert "Conventional-Commits format" not in prompt
    assert "Fix #123" in prompt  # $issue_number substituted

    with pytest.raises(MissingConventionalTitleError) as exc_info:
        assert_conventional_commit_title(prompt, context="worker prompt for issue #123")
    assert "issue #715" in str(exc_info.value)
    assert "worker prompt for issue #123" in str(exc_info.value)


def test_assert_conventional_commit_title_accepts_override_with_correct_format(
    tmp_path: Path,
) -> None:
    """The guard accepts a flat override that does carry the correct format (issue #715).

    A repo-local override is not inherently wrong — it just must not drop
    the conventional-commit title instruction.
    """
    from charlie_work.prompts import assert_conventional_commit_title

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task: Issue #$issue_number\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n"
        "- Body must include `Closes #$issue_number`.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the conventional-commit title instruction.
    assert_conventional_commit_title(prompt)


def test_worker_prompts_contain_execution_contract() -> None:
    """Verify that worker/rework prompts carry the execution-contract carve-out (issue #717)."""
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "Execution contract (self-detect from your diff)" in prompt
        assert "run the **FULL suite** locally at the final head before pushing" in prompt


def test_assert_execution_contract_passes_for_package_templates() -> None:
    """The guard accepts prompts rendered from the package templates (issue #717)."""
    from charlie_work.prompts import assert_execution_contract

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        # Must not raise — the package templates all reference
        # $section_execution_contract.
        assert_execution_contract(prompt)


def test_assert_execution_contract_rejects_flat_override_without_carve_out(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override that drops the execution contract (issue #717).

    This is the exact scenario the issue describes: a repo-local flat
    whole-file ``worker.md`` override that states the blanket "never run the
    full local suite" prohibition with no carve-out for contract-changing diffs
    — the kind job-cannon shipped before PR #1564 migrated it to package
    templates.
    """
    from charlie_work.prompts import (
        MissingExecutionContractError,
        assert_execution_contract,
    )

    # Simulate a flat override that carries no execution-contract carve-out —
    # the kind job-cannon shipped before PR #1564 migrated it to package
    # templates.
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\n"
        "Never run the full local suite as your gate -- CI runs the full "
        "~90-file matrix on push and is the sole authority on wider "
        "regressions. No exception clause here.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    assert "Execution contract (self-detect from your diff)" not in prompt
    assert "run the **FULL suite** locally at the final head before pushing" not in prompt

    with pytest.raises(MissingExecutionContractError) as exc_info:
        assert_execution_contract(prompt, context="worker prompt for issue #123")
    assert "issue #717" in str(exc_info.value)
    assert "worker prompt for issue #123" in str(exc_info.value)


def test_assert_execution_contract_accepts_override_with_carve_out(
    tmp_path: Path,
) -> None:
    """The guard accepts a flat override that does carry the carve-out (issue #717).

    A repo-local override is not inherently wrong — it just must not drop the
    execution-contract escalation trigger for contract-changing diffs.
    """
    from charlie_work.prompts import assert_execution_contract

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task: Issue #$issue_number\n\n"
        "**Execution contract (self-detect from your diff):** the default is "
        "the targeted command. Only if the diff changes any public function "
        "signature/return shape, run the **FULL suite** locally at the final "
        "head before pushing.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the execution-contract carve-out.
    assert_execution_contract(prompt)


def test_worker_prompts_contain_widened_containment_clause() -> None:
    """Verify that worker/rework prompts carry the widened containment clause (issue #1010).

    The clause must forbid any path outside the assigned worktree root — not
    just other checkouts of the same repo.  A rework worker can wander to a
    sibling repo just as easily as a fresh-dispatch worker.
    """
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "**Containment:**" in prompt
        assert "any path outside the assigned worktree root" in prompt


def test_assert_containment_passes_for_package_templates() -> None:
    """The guard accepts prompts rendered from the package templates (issue #1010)."""
    from charlie_work.prompts import assert_containment

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        # Must not raise — the package templates all reference
        # $section_scope_contract with the widened wording.
        assert_containment(prompt)


def test_assert_containment_rejects_flat_override_without_clause(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override that drops the containment clause (issue #1010).

    A repo-local flat whole-file ``worker.md`` override that does not reference
    ``$section_scope_contract`` renders without the containment clause,
    dispatching workers with no prohibition against editing a sibling repo's
    checkout.  The guard must refuse to let it through.
    """
    from charlie_work.prompts import (
        MissingContainmentError,
        assert_containment,
    )

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\nGo fix the issue. No containment clause here.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    assert "**Containment:**" not in prompt

    with pytest.raises(MissingContainmentError) as exc_info:
        assert_containment(prompt, context="worker prompt for issue #123")
    assert "issue #1010" in str(exc_info.value)
    assert "worker prompt for issue #123" in str(exc_info.value)


def test_assert_containment_rejects_old_repo_scoped_wording(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override that reverts to the old repo-scoped wording (issue #1010).

    The old wording forbade only "any other checkout of the repo" — which does
    not cover a different repo at all.  A flat override that carries the old
    wording must be caught, since it leaves the exact gap issue #1010 describes.
    """
    from charlie_work.prompts import (
        MissingContainmentError,
        assert_containment,
    )

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task: Issue #$issue_number\n\n"
        "## Scope contract\n\n"
        "- **Containment:** All file edits happen in the worktree; never "
        "modify any other checkout of the repo.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    assert "**Containment:**" in prompt
    assert "any path outside the assigned worktree root" not in prompt

    with pytest.raises(MissingContainmentError):
        assert_containment(prompt, context="worker prompt for issue #123")


def test_assert_containment_accepts_override_with_widened_clause(
    tmp_path: Path,
) -> None:
    """The guard accepts a flat override that does carry the widened clause (issue #1010)."""
    from charlie_work.prompts import assert_containment

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task: Issue #$issue_number\n\n"
        "## Scope contract\n\n"
        "- **Containment:** All file edits happen in the worktree; never "
        "modify any path outside the assigned worktree root.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the widened containment clause.
    assert_containment(prompt)


def test_rework_template_references_scope_contract_section() -> None:
    """Verify that rework.md references $section_scope_contract (issue #1010).

    A rework worker can wander to a sibling repo just as easily as a
    fresh-dispatch worker, so the containment clause must appear in rework
    prompts too.
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "rework.md").read_text(encoding="utf-8")

    assert "$section_scope_contract" in text


def test_review_template_contains_test_adequacy_section_placeholder() -> None:
    """Verify that review.md contains the $test_adequacy_section placeholder (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "$test_adequacy_section" in text, (
        "review.md must contain $test_adequacy_section placeholder"
    )


def test_review_template_contains_hollow_test_heuristics() -> None:
    """Verify that review.md contains the four hollow-test heuristic phrases (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the four hollow-test heuristics are present
    assert "Asserts only that a mock/stub was called, without asserting on real behavior" in text
    assert "Re-asserts a constant the code already hardcodes" in text
    assert "Contains an assertion that cannot fail" in text
    assert "Never imports or exercises the changed symbol" in text


def test_review_template_contains_new_approval_criteria_bullet() -> None:
    """Verify that review.md contains the new test-adequacy approval-criteria bullet (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the new approval-criteria bullet is present
    assert "Every non-exempt changed behavior has a genuine regression test" in text


def test_review_template_retains_existing_test_bullet() -> None:
    """Verify that review.md retains the existing test-related approval-criteria bullet (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the existing bullet is still present (design §9/R4: flagged not removed)
    assert "Tests or a strong no-test rationale are present" in text


def test_review_template_contains_exemption_scrutiny_instruction() -> None:
    """Verify that review.md contains the exemption-scrutiny instruction (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the exemption-scrutiny instruction is present
    assert "If a `Test-exempt:` reason is present above, treat it as a claim to verify" in text
    assert "not a fact to accept" in text


def test_review_template_contains_self_report_distrust_rule() -> None:
    """Verify that review.md tells the reviewer not to trust the PR's self-report."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "## Do not trust the PR's self-report" in text
    assert "never by itself downgrades a finding's severity" in text


def test_review_template_contains_investigation_discipline() -> None:
    """Verify that review.md bounds out-of-diff investigation and forbids re-running the full suite."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "do not otherwise crawl the broader codebase" in text
    assert "Do not\nre-run the full test suite to confirm results already reported by CI" in text


def test_review_template_contains_calibration_section() -> None:
    """Verify that review.md has a severity-calibration section distinguishing Critical/Important/Minor."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "## Calibration" in text
    assert "Tag every finding Critical, Important, or Minor" in text
    assert "label it\nplan-mandated" in text
    assert "Acknowledge what was done well before listing issues" in text


def test_review_template_summary_requires_strengths_and_severity_tags() -> None:
    """Verify that review.md's required summary fields include Strengths and severity-tagged Findings."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "Strengths — what's done well, specifically" in text
    assert "Findings, each tagged Critical / Important / Minor" in text


def test_rework_template_severity_aware_required_behavior() -> None:
    """Verify that rework.md tells the worker Minor findings are optional to
    address, and that this guidance is self-conditional (F4) rather than an
    unconditional command to act on a possibly-empty findings set."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "rework.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Critical and Important items are mandatory" in normalized
    assert "a Minor item may be skipped only if you say so explicitly" in normalized
    # The old unconditional imperative must not reappear.
    assert "Address every Critical and Important finding directly" not in normalized


def test_cross_family_review_contains_self_report_distrust_rule() -> None:
    """Verify that cross_family_review.md tells the reviewer a stated rationale never lowers severity."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "cross_family_review.md").read_text(encoding="utf-8")

    assert "is the author grading their own work" in text
    assert "it never by itself lowers a finding's" in text
    assert "Verify the claim against the code; if it doesn't hold, the finding stands." in text


def test_review_rendered_with_populated_test_adequacy_section() -> None:
    """Verify that review.md renders with populated test-adequacy facts when enabled (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create a populated TestAdequacyFacts
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=("src/foo.py", "src/bar.py"),
        exempt=False,
        exempt_reason="",
    )
    warnings = ("Zero recognized assertions in added test lines",)

    # Render the section
    section = render_test_adequacy_section(facts, warnings)

    # Verify the facts appear in the rendered output
    assert "## Test-adequacy facts (Tier 1, deterministic)" in section
    assert "Added product LOC: 100" in section
    assert "Added test LOC: 50" in section
    assert "Assertion-bearing added test lines: 10" in section
    assert "Test files changed: 2" in section
    assert "Untested product files: src/foo.py, src/bar.py" in section
    assert "Zero recognized assertions in added test lines" in section


def test_review_rendered_with_disabled_test_adequacy_section() -> None:
    """Verify that review.md renders with empty test-adequacy section when disabled (issue #180)."""
    from charlie_work.workflow import render_test_adequacy_section

    # When facts is None (gate disabled), should return empty string
    section = render_test_adequacy_section(None, ())

    assert section == "", "Disabled gate should render empty string"


def test_review_rendered_with_exempt_claim() -> None:
    """Verify that review.md renders with exempt claim when present (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create facts with an exempt claim
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=0,
        assertion_count=0,
        test_files_changed=0,
        untested_product_files=(),
        exempt=True,
        exempt_reason="n/a - pure refactoring",
    )

    # Render the section
    section = render_test_adequacy_section(facts, ())

    # Verify the exempt claim appears
    assert 'Test-exempt claim: "n/a - pure refactoring" (verify against the diff)' in section


def test_review_rendered_with_no_untested_files() -> None:
    """Verify that review.md renders without untested files list when none exist (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create facts with no untested files
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=(),
        exempt=False,
        exempt_reason="",
    )

    # Render the section
    section = render_test_adequacy_section(facts, ())

    # Verify the untested files line does NOT appear
    assert "Untested product files:" not in section
