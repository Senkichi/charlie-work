"""Tests for issue #589: a prompt template referencing a placeholder the
orchestrator no longer supplies must fail loudly, not ship literal text.

Live incident (2026-07-22 -> 2026-07-25, Senkichi/job-cannon): the repo set
``prompts_dir: .devin/prompts``, and its ``review.md`` override -- an
unversioned local copy dated 2026-07-02, invisible to git because ``.devin/``
sat in ``.git/info/exclude`` -- still used the pre-#513 decision protocol:

    ## Decision output
    Write your review summary to a Markdown file, then record one decision:
    ```powershell
    $decision_command
    ```

#513 replaced that with a parsed fenced-JSON block on 2026-07-21 and stopped
supplying ``decision_command`` (and, earlier, ``checks_json_path``). Because
rendering used ``Template.safe_substitute``, both orphaned placeholders
rendered as literal text rather than failing.

Reviewers launch under ``--permission-mode plan`` and cannot execute commands
or write files, so the stale prompt instructed them to do the one thing they
are structurally forbidden from doing, while omitting the fenced-JSON block
that is their only recording channel. Twenty-one PRs ran full multi-turn
reviews, reached real verdicts, and had every one discarded -- three paid
sessions each before escalating the entire review queue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.prompts import PromptTemplateError, render_prompt


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_orphaned_placeholder_raises(tmp_path: Path) -> None:
    """The exact job-cannon shape: an override referencing a removed variable."""
    _write(
        tmp_path,
        "review.md",
        "# Review PR #$pr_number\n\nRecord one decision:\n\n"
        "```powershell\n$decision_command\n```\n",
    )

    with pytest.raises(PromptTemplateError) as excinfo:
        render_prompt("review.md", {"pr_number": 42}, search_dirs=(tmp_path,))

    assert excinfo.value.missing == ("decision_command",)
    # The message must name the offending file: the whole failure mode is that
    # a repo-local override silently shadows the packaged template.
    assert "review.md" in str(excinfo.value)


def test_every_orphaned_placeholder_is_reported(tmp_path: Path) -> None:
    """job-cannon's override had two orphans; reporting one at a time would
    turn a single fix into a guess-and-retry loop."""
    _write(tmp_path, "review.md", "$pr_number $decision_command $checks_json_path\n")

    with pytest.raises(PromptTemplateError) as excinfo:
        render_prompt("review.md", {"pr_number": 42}, search_dirs=(tmp_path,))

    assert excinfo.value.missing == ("checks_json_path", "decision_command")


def test_dollar_sign_in_a_value_never_trips_the_guard(tmp_path: Path) -> None:
    """A ``$word`` inside an attacker-controlled value is a leaf replacement,
    not an unresolved placeholder.

    Neither substitution pass re-scans values, so treating one as a missing
    identifier would let any issue deny service to its own dispatch just by
    putting a dollar sign in its title or body.
    """
    _write(tmp_path, "worker.md", "# Issue #$issue_number\n\n$issue_body\n")

    rendered = render_prompt(
        "worker.md",
        {
            "issue_number": 7,
            "issue_body": "Set $decision_command and $PATH, then run `echo $HOME`.",
        },
        search_dirs=(tmp_path,),
    )

    assert "$decision_command" in rendered
    assert "$PATH" in rendered
    assert "$HOME" in rendered


def test_strict_false_preserves_legacy_lenient_rendering(tmp_path: Path) -> None:
    """Callers that genuinely want partial rendering must opt out explicitly."""
    _write(tmp_path, "review.md", "PR #$pr_number\n$decision_command\n")

    rendered = render_prompt("review.md", {"pr_number": 42}, search_dirs=(tmp_path,), strict=False)

    assert "PR #42" in rendered
    assert "$decision_command" in rendered


def test_supplied_placeholders_render_normally(tmp_path: Path) -> None:
    _write(tmp_path, "review.md", "PR #$pr_number by $author\n")

    rendered = render_prompt(
        "review.md", {"pr_number": 42, "author": "octocat"}, search_dirs=(tmp_path,)
    )

    assert rendered.strip() == "PR #42 by octocat"


def test_stale_placeholder_inside_a_referenced_section_raises(tmp_path: Path) -> None:
    """Section partials ship into the prompt too, so drift there is just as fatal."""
    _write(tmp_path / "worker_sections", "scope_contract.md", "Scope for $branch_name.\n")
    _write(tmp_path, "worker.md", "# Work\n\n$section_scope_contract\n")

    with pytest.raises(PromptTemplateError) as excinfo:
        render_prompt("worker.md", {}, search_dirs=(tmp_path,))

    assert excinfo.value.missing == ("branch_name",)


def test_stale_placeholder_in_an_unreferenced_section_is_ignored(tmp_path: Path) -> None:
    """An unused partial never reaches the output, so it must not block an
    unrelated render -- otherwise one stale file breaks every template in the
    repo at once."""
    _write(tmp_path / "worker_sections", "unused.md", "Needs $long_gone_variable.\n")
    _write(tmp_path, "worker.md", "# Work on #$issue_number\n")

    rendered = render_prompt("worker.md", {"issue_number": 7}, search_dirs=(tmp_path,))

    assert "# Work on #7" in rendered


def test_packaged_review_template_is_satisfied_by_its_call_site() -> None:
    """The packaged templates must not themselves trip the guard.

    Values mirror ``OrchestratorApp.review``'s render call. If a future edit
    adds a ``$placeholder`` to review.md without supplying it, this fails here
    rather than in production against a live PR.
    """
    rendered = render_prompt(
        "review.md",
        {
            "pr_number": 1,
            "pr_title": "t",
            "pr_url": "u",
            "issue_number": 2,
            "issue_title": "it",
            "issue_url": "iu",
            "pr_json_path": "p.json",
            "diff_path": "d.patch",
            "cross_family_section": "",
            "janitor_section": "",
            "test_adequacy_section": "",
            "static_probe_section": "",
            "diff_size_section": "",
            "ci_status_section": "",
            "over_cap_section": "",
            "prior_review_section": "",
        },
    )

    assert "PR #1" in rendered
