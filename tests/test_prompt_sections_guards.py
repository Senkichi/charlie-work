"""``assert_*`` prompt-guard function tests (issues #714, #715, #717, #1010, #1780).

Split out of ``tests/test_prompt_sections.py`` (issue #1570, Track 1
shoulder) -- bodies are verbatim relocations; shared helpers live in
``tests/_prompt_sections_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.prompts import render_prompt

from _prompt_sections_fixtures import (
    ISSUE_VALUES,
    _render_worker_with_sections,
)


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
            "issue_comments": "",
        },
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the contract text.
    assert_no_merge_contract(prompt)


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


def test_assert_session_scratch_dir_passes_for_package_templates() -> None:
    """The guard accepts prompts rendered from the package templates (issue #1780)."""
    from charlie_work.prompts import assert_session_scratch_dir

    for template_name in (
        "worker.md",
        "worker_claude_code.md",
        "worker_local.md",
        "rework.md",
    ):
        prompt = _render_worker_with_sections(template_name)
        # Must not raise — every package worker/rework template references
        # $section_session_scratch_dir.
        assert_session_scratch_dir(prompt)


def test_assert_session_scratch_dir_rejects_flat_override_without_section(
    tmp_path: Path,
) -> None:
    """The guard catches a flat override that drops the scratch-dir rule (issue #1780).

    This is the exact scenario the issue describes: a repo-local flat
    whole-file ``worker.md`` override that does not reference
    ``$section_session_scratch_dir`` renders without the ``$TMPDIR``
    instruction, dispatching workers free to pick a literal ``/tmp/...``
    path that resolves through MSYS's install-wide mount (Git Bash) or the
    shared system temp — shared across every concurrent worker session,
    the residual class #1767's env fix could not close.
    """
    from charlie_work.prompts import (
        MissingSessionScratchDirError,
        assert_session_scratch_dir,
    )

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\nGo fix the issue. No scratch-file rule here.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    assert "$TMPDIR" not in prompt

    with pytest.raises(MissingSessionScratchDirError) as exc_info:
        assert_session_scratch_dir(prompt, context="worker prompt for issue #123")
    assert "issue #1780" in str(exc_info.value)
    assert "worker prompt for issue #123" in str(exc_info.value)


def test_assert_session_scratch_dir_accepts_override_with_rule(
    tmp_path: Path,
) -> None:
    """The guard accepts a flat override that does carry the scratch-dir rule (issue #1780).

    A repo-local override is not inherently wrong — it just must not drop
    the ``$TMPDIR``-only instruction and the literal-``/tmp`` prohibition.
    """
    from charlie_work.prompts import assert_session_scratch_dir

    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    # ``$$TMPDIR`` in template source renders as the literal ``$TMPDIR``
    # (string.Template escaping); a bare ``$TMPDIR`` is an unresolved
    # placeholder that strict rendering refuses before the guard ever runs.
    (override_dir / "worker.md").write_text(
        "# Worker Task\n\n"
        "## Scratch files\n\n"
        "Put every scratch file under `$$TMPDIR`; never a literal `/tmp/` "
        "path — it is shared across every concurrent worker session.\n",
        encoding="utf-8",
    )
    prompt = render_prompt(
        "worker.md",
        ISSUE_VALUES,
        search_dirs=(override_dir,),
    )
    # Must not raise — the override carries the scratch-dir rule.
    assert_session_scratch_dir(prompt)
