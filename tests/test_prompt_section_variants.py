"""Tests for section variants: ``worker_sections/<variant>/<stem>.md`` overlays.

A variant is a subdirectory of a ``worker_sections/`` directory. While it is active,
``<variant>/<stem>.md`` replaces ``<stem>.md``; a stem that exists only inside a variant
directory renders as an empty string while the variant is inactive. Nothing declares
the set of variants -- it is whatever subdirectories exist -- so the strict-render
guard and the startup drift check must reach every variant on disk, not just the base
set: a stale placeholder inside an overlay would otherwise stay armed until the matching
consumer dispatches.

A search dir in these tests is a directory containing ``worker_sections/``, exactly as
``runtime.prompts_dir`` is in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.markdown_fence import fenced_block
from charlie_work.prompt_sections import section_variables, section_variant_names
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import PromptTemplateError, render_prompt, unsupplied_placeholders
from charlie_work.workflow import (
    REWORK_PROMPT_KEYS,
    WORKER_PROMPT_KEYS,
    check_prompt_template_drift,
)

VALUES = {
    "issue_number": 123,
    "issue_title": "Fix search",
    "issue_url": "https://example.test/issues/123",
    "issue_body": "Body text",
    "issue_body_block": fenced_block("Body text", "md"),
    "branch_name": "agent/issue-123-fix-search",
    "issue_comments": "",
    "module_map": "",
    "attachment_budget": "",
    "pr_number": 456,
    "pr_title": "fix: search is broken",
    "pr_url": "https://example.test/pull/456",
    "dispatch_note": "note",
    "dispatch_note_block": fenced_block("note", "md"),
    "required_changes_section": "",
    **prompt_test_command_values("", None),
}


def _prompt_dir(root: Path, files: dict[str, str]) -> Path:
    """A search dir whose ``worker_sections/`` holds ``files`` (keys are relative paths)."""
    for relative, text in files.items():
        path = root / "worker_sections" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# section_variables: overlay resolution
# ---------------------------------------------------------------------------


def test_default_call_equals_the_no_variant_call() -> None:
    assert section_variables() == section_variables(variants=())


def test_no_variants_ignores_overlays(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"foo.md": "BASE", "skills/foo.md": "OVERLAY"})

    assert section_variables((root,), variants=())["section_foo"] == "BASE"


def test_active_variant_selects_the_overlay_for_a_stem_in_both(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"foo.md": "BASE", "skills/foo.md": "OVERLAY"})

    assert section_variables((root,), variants=("skills",))["section_foo"] == "OVERLAY"


def test_overlay_only_stem_is_empty_when_inactive_and_text_when_active(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"skills/only_here.md": "OVERLAY ONLY"})

    inactive = section_variables((root,), variants=())
    active = section_variables((root,), variants=("skills",))

    assert inactive["section_only_here"] == ""
    assert active["section_only_here"] == "OVERLAY ONLY"


def test_repo_local_base_beats_the_package_overlay_when_the_variant_is_active(
    tmp_path: Path,
) -> None:
    """The documented rule: the more specific (repo-local) override wins, even against
    the package's own overlay for the active variant."""
    package_overlay = section_variables(variants=("skills",))["section_loop_finish_steps"]
    root = _prompt_dir(tmp_path, {"loop_finish_steps.md": "LOCAL BASE"})

    assert package_overlay != "LOCAL BASE"
    assert section_variables((root,), variants=("skills",))["section_loop_finish_steps"] == (
        "LOCAL BASE"
    )
    assert section_variables((root,), variants=())["section_loop_finish_steps"] == "LOCAL BASE"


def test_repo_local_overlay_beats_the_repo_local_base(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"x.md": "LOCAL BASE", "skills/x.md": "LOCAL OVERLAY"})

    assert section_variables((root,), variants=("skills",))["section_x"] == "LOCAL OVERLAY"
    assert section_variables((root,), variants=())["section_x"] == "LOCAL BASE"


def test_repo_local_overlay_beats_the_package_base(tmp_path: Path) -> None:
    package_base = section_variables()["section_loop_finish_steps"]
    root = _prompt_dir(tmp_path, {"skills/loop_finish_steps.md": "LOCAL OVERLAY"})

    active = section_variables((root,), variants=("skills",))["section_loop_finish_steps"]
    inactive = section_variables((root,), variants=())["section_loop_finish_steps"]

    assert active == "LOCAL OVERLAY"
    assert inactive == package_base


def test_earlier_variant_wins_over_a_later_one(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"a/pick.md": "FROM A", "b/pick.md": "FROM B"})

    assert section_variables((root,), variants=("a", "b"))["section_pick"] == "FROM A"
    assert section_variables((root,), variants=("b", "a"))["section_pick"] == "FROM B"


def test_naming_a_variant_that_exists_nowhere_is_harmless() -> None:
    assert section_variables(variants=("no-such-variant",)) == section_variables()


def test_non_markdown_files_in_a_variant_dir_are_ignored(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"skills/notes.txt": "x", "skills/real.md": "REAL"})

    active = section_variables((root,), variants=("skills",))

    assert active["section_real"] == "REAL"
    assert "section_notes" not in active


def test_package_skills_overlay_swaps_the_loop_text() -> None:
    plain = section_variables(variants=())
    skills = section_variables(variants=("skills",))

    assert "`/commit`" in skills["section_loop_finish_steps"]
    assert "`/commit`" not in plain["section_loop_finish_steps"]
    assert "git push -u origin" in plain["section_loop_finish_steps"]
    assert "git push -u origin" not in skills["section_loop_finish_steps"]
    assert "## Available skills" in skills["section_available_skills"]
    assert plain["section_available_skills"] == ""
    assert plain["section_loop_test_skill_note"] == ""
    assert "`/test` skill" in skills["section_loop_test_skill_note"]


# ---------------------------------------------------------------------------
# section_variant_names: discovery
# ---------------------------------------------------------------------------


def test_package_skills_variant_is_discovered() -> None:
    assert "skills" in section_variant_names()


def test_variant_names_are_sorted_and_unique() -> None:
    names = section_variant_names()

    assert names == tuple(sorted(set(names)))


def test_repo_local_variant_dir_is_discovered_alongside_the_package_one(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"extra/tail.md": "x", "skills/loop_finish_steps.md": "y"})

    names = section_variant_names((root,))

    assert "extra" in names and "skills" in names
    assert set(names) == set(section_variant_names()) | {"extra"}


def test_plain_files_beside_variant_dirs_are_not_variants(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"notes.md": "x", "README": "y", "real_variant/a.md": "z"})

    names = section_variant_names((root,))

    assert "real_variant" in names
    assert "notes.md" not in names and "notes" not in names and "README" not in names


def test_missing_search_dir_contributes_no_variants(tmp_path: Path) -> None:
    assert section_variant_names((tmp_path / "does-not-exist",)) == section_variant_names()


def test_repo_local_only_variant_overlay_stem_is_empty_until_activated(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"extra/tail.md": "EXTRA TAIL"})

    assert section_variables((root,), variants=())["section_tail"] == ""
    assert section_variables((root,), variants=("extra",))["section_tail"] == "EXTRA TAIL"


# ---------------------------------------------------------------------------
# render_prompt / unsupplied_placeholders with variants
# ---------------------------------------------------------------------------


def test_worker_render_differs_between_plain_and_skills_variant() -> None:
    plain = render_prompt("worker.md", VALUES, variants=())
    skills = render_prompt("worker.md", VALUES, variants=("skills",))

    assert plain != skills
    assert "## Available skills" in skills and "## Available skills" not in plain
    assert "Use `/commit` to commit your changes" in skills
    assert "Use `/commit`" not in plain
    # The loop's own push step (the end-of-prompt push/verify instructions are shared).
    assert "10. Push your branch: `git push -u origin agent/issue-123-fix-search`." in plain
    assert "10. Use `/push` to push your branch to GitHub." in skills
    assert "10. Use `/push`" not in plain
    assert "10. Push your branch:" not in skills
    assert render_prompt("worker.md", VALUES) == plain


def test_a_template_that_never_references_the_overlays_is_unchanged() -> None:
    for template in ("worker_claude_code.md", "worker_local.md"):
        assert render_prompt(template, VALUES, variants=("skills",)) == render_prompt(
            template, VALUES, variants=()
        )


@pytest.mark.parametrize(
    ("stem", "template", "keys"),
    [
        ("loop_finish_steps", "worker.md", WORKER_PROMPT_KEYS),
        ("loop_test_skill_note", "worker.md", WORKER_PROMPT_KEYS),
        ("available_skills", "worker.md", WORKER_PROMPT_KEYS),
        ("rework_preflight", "rework.md", REWORK_PROMPT_KEYS),
    ],
)
def test_unsupplied_placeholder_in_an_overlay_is_flagged_only_under_that_variant(
    tmp_path: Path, stem: str, template: str, keys: frozenset[str]
) -> None:
    root = _prompt_dir(tmp_path, {f"skills/{stem}.md": "step $bogus"})

    with_variant = unsupplied_placeholders(
        template, keys, search_dirs=(root,), variants=("skills",)
    )
    without = unsupplied_placeholders(template, keys, search_dirs=(root,))

    assert "bogus" in with_variant
    assert "bogus" not in without


def test_the_package_templates_are_clean_under_every_variant_on_disk() -> None:
    """Control for the flagging tests: nothing is unsupplied in the shipped surface."""
    for template, keys in (("worker.md", WORKER_PROMPT_KEYS), ("rework.md", REWORK_PROMPT_KEYS)):
        for variants in ((), *((name,) for name in section_variant_names())):
            assert unsupplied_placeholders(template, keys, variants=variants) == set()


def test_strict_render_rejects_a_bogus_placeholder_in_the_active_overlay(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"skills/loop_finish_steps.md": "8. Use $bogus."})

    with pytest.raises(PromptTemplateError, match="bogus"):
        render_prompt("worker.md", VALUES, search_dirs=(root,), variants=("skills",))

    assert "Issue #123" in render_prompt("worker.md", VALUES, search_dirs=(root,), variants=())


# ---------------------------------------------------------------------------
# workflow.check_prompt_template_drift covers variants, not just the base set
# ---------------------------------------------------------------------------


def _drift_errors(root: Path) -> list[PromptTemplateError]:
    return check_prompt_template_drift(OrchestratorConfig(), search_dirs=(root,))


def test_drift_check_is_clean_with_no_overlay_override(tmp_path: Path) -> None:
    assert _drift_errors(tmp_path) == []
    assert check_prompt_template_drift(OrchestratorConfig()) == []


def test_drift_check_flags_a_bogus_placeholder_in_a_skills_overlay(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"skills/loop_finish_steps.md": "8. Use `/commit` $bogus."})

    errors = _drift_errors(root)

    assert any("bogus" in error.missing for error in errors), errors
    # The base-only check misses exactly this: the overlay is unreachable without the
    # variant, so an implementation that skipped the variant loop would report clean.
    assert "bogus" not in unsupplied_placeholders(
        "worker.md", WORKER_PROMPT_KEYS, search_dirs=(root,)
    )


def test_drift_check_flags_a_bogus_placeholder_in_a_rework_overlay(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"skills/rework_preflight.md": "- run $bogus"})

    errors = _drift_errors(root)

    assert any("bogus" in error.missing and "rework.md" in str(error) for error in errors), errors


def test_drift_check_discovers_variants_from_disk_not_from_a_name(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"anything/loop_finish_steps.md": "8. Use $bogus."})

    errors = _drift_errors(root)

    assert any("bogus" in error.missing for error in errors), errors


def test_drift_check_still_flags_a_bogus_placeholder_in_the_base_set(tmp_path: Path) -> None:
    root = _prompt_dir(tmp_path, {"loop_finish_steps.md": "8. Use $bogus."})

    assert any("bogus" in error.missing for error in _drift_errors(root))
