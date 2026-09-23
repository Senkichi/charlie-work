"""Rendered worker/rework prompt content requirements.

Split out of ``tests/test_prompt_sections.py`` (issue #1570, Track 1
shoulder) -- bodies are verbatim relocations; shared helpers live in
``tests/_prompt_sections_fixtures.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from charlie_work.markdown_fence import fenced_block
from charlie_work.prompt_sections import section_variables
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import render_prompt, unresolved_rendered_identifiers

from _prompt_sections_fixtures import (
    ISSUE_VALUES,
    TEST_COMMAND_VALUES,
    _render_worker_with_sections,
)


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

    # The partial names the full-suite command through a placeholder; the rendered
    # prompt carries it with the resolved command spliced in.
    assert "$full_suite_command" in contract
    rendered_contract = contract.replace(
        "$full_suite_command", TEST_COMMAND_VALUES["full_suite_command"]
    )
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert rendered_contract in prompt


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
        **TEST_COMMAND_VALUES,
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
        # A consumer whose ``dev`` extra lists pytest: the derived runner is what the
        # rework prompt must carry (``prompt_test_command``), not a hardcoded string.
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "0"\n'
            '[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
            encoding="utf-8",
        )
        config = OrchestratorConfig(
            devin=DevinConfig(),
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
        assert "uv run --extra dev pytest <impacted test files> -q --tb=short" in prompt
        assert "test_<touched_module>" not in prompt
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
        assert not unresolved_rendered_identifiers(prompt), (
            f"Unresolved placeholders found in rendered prompt:\n{prompt}"
        )


def test_worker_prompt_includes_push_then_verify() -> None:
    """Verify that the worker.md prompt includes push-then-verify in the Done condition.

    cw#1771: the worker's done condition is push + verify + write the outcome
    file -- it never runs ``gh pr view``/``gh pr create`` itself (issue #502:
    workers carry no ``gh`` credential, so both always fail on auth). The
    orchestrator opens the PR from ``.worker-outcome.json`` after the session
    ends.
    """
    prompt = _render_worker_with_sections("worker.md")

    # Verify the key instruction about local commits not being done
    assert "Committing locally is NOT done" in prompt
    # Verify the push instruction with resolved branch name
    assert "git push origin agent/issue-123-fix-search" in prompt
    # Verify the push-verification instruction (ls-remote, not a PR-existence probe)
    assert "git ls-remote origin agent/issue-123-fix-search" in prompt
    assert "git rev-parse HEAD" in prompt
    # The done condition routes to the outcome-file contract instead of
    # asking the worker to confirm a PR it never opens.
    assert ".worker-outcome.json" in prompt
    assert "gh pr view agent/issue-123-fix-search --json headRefOid" not in prompt


def test_claude_code_worker_prompt_includes_push_then_verify() -> None:
    """Verify that the worker_claude_code.md prompt includes push-then-verify in the Done condition.

    cw#1771: same contract as ``worker.md`` -- see that test's docstring.
    """
    prompt = _render_worker_with_sections("worker_claude_code.md")

    # Verify the key instruction about local commits not being done
    assert "Committing locally is NOT done" in prompt
    # Verify the push instruction with resolved branch name
    assert "git push -u origin agent/issue-123-fix-search" in prompt
    # Verify the push-verification instruction (ls-remote, not a PR-existence probe)
    assert "git ls-remote origin agent/issue-123-fix-search" in prompt
    assert "git rev-parse HEAD" in prompt
    assert ".worker-outcome.json" in prompt
    assert "gh pr view agent/issue-123-fix-search --json headRefOid" not in prompt


def test_worker_prompts_never_instruct_gh_pr_create() -> None:
    """cw#1771: the DEFAULT worker contract is commit/push/verify/write-outcome-file/stop.

    Workers carry no ``gh`` credential (issue #502), so ``gh pr create`` always
    fails on auth; the old instruction to run it burned 2-4 wasted tool calls
    on every single dispatch (measured 8/8 in the issue's session-log sample).
    This must fail against the pre-fix templates, which told the worker to
    "Open the pull request with `gh pr create`" as the very last loop step --
    the rendered prompt only ever mentions `gh pr create` now inside the
    explicit "do not run" instruction, never as something to do.
    """
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "Open the pull request with `gh pr create`" not in prompt
        assert "Open the PR (see requirements below)" not in prompt
        assert "do not run `gh pr create`" in prompt
        assert "pr_title" in prompt
        assert "pr_body" in prompt


def test_worker_and_rework_templates_contain_identical_canonical_test_command() -> None:
    """worker.md, worker_claude_code.md and rework.md take the test command from the
    same two placeholders, and none of them hardcodes a runner.

    The name is unchanged from when the command was a literal in each template: the
    invariant (one canonical command shared by every template) is the same, only its
    representation moved to a placeholder.

    This prevents silent drift where one template uses the full command and the other
    a partial variant (issue #91). The command used to be a literal in each template;
    it is now one derivation (``prompt_test_command``) spliced in through
    ``$targeted_test_command`` / ``$full_suite_command``, so drift means a template that
    stopped referencing them, or grew its own literal again.
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    sections_dir = prompts_dir / "worker_sections"

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")
        assert "$targeted_test_command" in text, (
            f"Template {template_name} does not reference $targeted_test_command"
        )
        assert "uv run --extra" not in text, f"Template {template_name} hardcodes a runner"
        assert "test_<touched_module>" not in text, (
            f"Template {template_name} still names the flat test_<touched_module>.py shape"
        )
    # The full-suite command is spliced in by the shared execution-contract partial,
    # which every one of those templates includes.
    contract = (sections_dir / "execution_contract.md").read_text(encoding="utf-8")
    assert "$full_suite_command" in contract
    assert "uv run --extra" not in contract
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        text = (prompts_dir / template_name).read_text(encoding="utf-8")
        assert "$section_execution_contract" in text


def test_rendered_worker_prompts_contain_canonical_test_command(tmp_path: Path) -> None:
    """Rendered worker and rework prompts all carry the same derived test command.

    A repo whose ``dev`` extra lists pytest yields
    ``uv run --extra dev pytest <impacted test files> -q --tb=short``; the full-suite
    parenthetical uses the same runner. This goes through the real ``render_prompt`` so
    the command is asserted in the output workers actually see (issue #91: the fresh and
    rework lanes must not drift on the command).
    """
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n'
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
        encoding="utf-8",
    )
    values = {**ISSUE_VALUES, **prompt_test_command_values("", tmp_path)}
    canonical_command = "uv run --extra dev pytest <impacted test files> -q --tb=short"
    full_suite = "(`uv run --extra dev pytest -q --tb=short`)"

    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = render_prompt(template_name, values)
        assert canonical_command in prompt, (
            f"Rendered {template_name} does not contain the canonical test command. "
            f"Expected to find: {canonical_command}"
        )
        assert full_suite in prompt, (
            f"Rendered {template_name} does not carry the derived full-suite command. "
            f"Expected to find: {full_suite}"
        )
        assert "test_<touched_module>" not in prompt


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
    # The skills list only renders under the ``skills`` variant: a consumer that ships
    # no skills gets the plain git/gh loop and no bullet at all (see prompt_skills).
    prompt = _render_worker_with_sections("worker.md", variants=("skills",))

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


def test_worker_prompts_contain_conventional_commit_title_instruction() -> None:
    """Verify that worker prompts carry the conventional-commit title instruction (issue #715)."""
    for template_name in ("worker.md", "worker_claude_code.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "Conventional-Commits format" in prompt


def test_worker_prompts_contain_execution_contract() -> None:
    """Verify that worker/rework prompts carry the execution-contract carve-out (issue #717)."""
    for template_name in ("worker.md", "worker_claude_code.md", "rework.md"):
        prompt = _render_worker_with_sections(template_name)
        assert "Execution contract (self-detect from your diff)" in prompt
        assert "run the **FULL suite** locally at the final head before pushing" in prompt


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


def test_rework_template_references_scope_contract_section() -> None:
    """Verify that rework.md references $section_scope_contract (issue #1010).

    A rework worker can wander to a sibling repo just as easily as a
    fresh-dispatch worker, so the containment clause must appear in rework
    prompts too.
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "rework.md").read_text(encoding="utf-8")

    assert "$section_scope_contract" in text


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
