"""Tests for ``prompt_skills``: when a worker prompt may name slash-command skills.

The skills are not a property of the harness. They exist only where the consumer ships
``<dir>/<name>/SKILL.md`` in a directory the harness loads skills from; a worker in a
repo that ships none gets ``Skill "create-branch" not found`` back after being told the
skill exists. ``active_prompt_variants`` therefore returns ``("skills",)`` only when the
harness loads skills AND the repo ships every skill the loop declares.

Which skills the loop needs is derived from the ``Available skills`` bullet list in the
overlay (``declared_skills``), so the declaration and the loop text cannot drift apart.
The guard tests at the bottom enforce exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from charlie_work.harnesses import HARNESS_REGISTRY, HarnessCapabilities
from charlie_work.markdown_fence import fenced_block
from charlie_work.prompt_skills import (
    DECLARATION_SECTION,
    SKILLS_VARIANT,
    active_prompt_variants,
    declared_skills,
    provisioned_skills,
)
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import TEMPLATE_DIR, render_prompt

SKILLS_OVERLAY_DIR = TEMPLATE_DIR / "worker_sections" / SKILLS_VARIANT

# A backticked slash-command token: `/commit`, `/create-branch`. Paths such as
# `/dev/null` do not match (the closing backtick must follow the first segment).
SLASH_SKILL = re.compile(r"`/([a-z0-9][a-z0-9-]*)`")

DEVIN = "devin-shell"
CLAUDE = "claude-code"

_RENDER_VALUES = {
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


def _ship(root: Path, skill_dir: str, names: tuple[str, ...]) -> None:
    """Make ``root`` ship ``<skill_dir>/<name>/SKILL.md`` for each name."""
    for name in names:
        skill = root / skill_dir / name
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")


def _override_declaration(root: Path, body: str) -> Path:
    """A repo-local prompt dir whose skills overlay declares ``body``."""
    overlay = root / "worker_sections" / SKILLS_VARIANT
    overlay.mkdir(parents=True)
    (overlay / "available_skills.md").write_text(body, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# declared_skills: the derivation and its positive control
# ---------------------------------------------------------------------------


def test_declared_skills_are_the_six_loop_skills_in_order() -> None:
    """Positive control: the bullet parser finds exactly what the overlay lists, so an
    empty tuple elsewhere is a verdict on the repo and not on a broken parser.

    cw#1771: ``create-pr`` was dropped from the declaration -- workers never open
    the PR themselves (see ``push_pr_outcome.md``), so a skill for it is never
    provisioned or needed.
    """
    assert declared_skills() == (
        "create-branch",
        "commit",
        "test",
        "preflight",
        "push",
        "complete",
    )


def test_declared_skills_come_from_the_declaration_partial() -> None:
    text = (SKILLS_OVERLAY_DIR / "available_skills.md").read_text(encoding="utf-8")

    assert declared_skills() == tuple(dict.fromkeys(re.findall(r"^- `/([a-z0-9-]+)`", text, re.M)))
    assert DECLARATION_SECTION == "section_available_skills"


def test_declared_skills_follow_a_repo_local_declaration(tmp_path: Path) -> None:
    override = _override_declaration(
        tmp_path, "## Available skills\n\n- `/alpha` - a\n- `/beta`\n"
    )

    assert declared_skills((override,)) == ("alpha", "beta")


def test_declared_skills_deduplicate_and_ignore_non_bullets(tmp_path: Path) -> None:
    override = _override_declaration(
        tmp_path,
        "Use `/prose-mention` inline.\n- `/one` - first\n  - `/nested-indented`\n- `/one` - again\n",
    )

    assert declared_skills((override,)) == ("one",)


# ---------------------------------------------------------------------------
# provisioned_skills
# ---------------------------------------------------------------------------


def test_provisioned_skills_unions_names_across_dirs(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", ("alpha", "beta"))
    _ship(tmp_path, ".claude/skills", ("beta", "gamma"))

    found = provisioned_skills(tmp_path, (".devin/skills", ".claude/skills"))

    assert found == frozenset({"alpha", "beta", "gamma"})
    assert isinstance(found, frozenset)


def test_provisioned_skills_only_reads_the_named_dirs(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", ("alpha",))
    _ship(tmp_path, ".claude/skills", ("beta",))

    assert provisioned_skills(tmp_path, (".claude/skills",)) == frozenset({"beta"})
    assert provisioned_skills(tmp_path, ()) == frozenset()
    assert provisioned_skills(tmp_path, (".no-such/skills",)) == frozenset()


def test_directory_without_skill_md_is_not_provisioned(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", ("real",))
    (tmp_path / ".devin" / "skills" / "hollow").mkdir()
    (tmp_path / ".devin" / "skills" / "hollow" / "README.md").write_text("x", encoding="utf-8")

    assert provisioned_skills(tmp_path, (".devin/skills",)) == frozenset({"real"})


def test_skill_md_that_is_a_directory_is_not_provisioned(tmp_path: Path) -> None:
    (tmp_path / ".devin" / "skills" / "odd" / "SKILL.md").mkdir(parents=True)

    assert provisioned_skills(tmp_path, (".devin/skills",)) == frozenset()


def test_plain_file_in_skills_dir_is_not_a_skill(tmp_path: Path) -> None:
    skills = tmp_path / ".devin" / "skills"
    skills.mkdir(parents=True)
    (skills / "commit").write_text("not a directory", encoding="utf-8")

    assert provisioned_skills(tmp_path, (".devin/skills",)) == frozenset()


def test_nested_skill_directories_are_not_provisioned(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills/group", ("deep",))

    assert provisioned_skills(tmp_path, (".devin/skills",)) == frozenset()


# ---------------------------------------------------------------------------
# active_prompt_variants: the decision matrix
# ---------------------------------------------------------------------------

_ALL = declared_skills()


def test_devin_with_all_skills_under_devin_dir(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", _ALL)

    assert active_prompt_variants(DEVIN, tmp_path) == (SKILLS_VARIANT,)


def test_devin_with_all_skills_under_claude_dir(tmp_path: Path) -> None:
    _ship(tmp_path, ".claude/skills", _ALL)

    assert active_prompt_variants(DEVIN, tmp_path) == (SKILLS_VARIANT,)


def test_devin_with_skills_split_across_both_dirs(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", _ALL[:3])
    _ship(tmp_path, ".claude/skills", _ALL[3:])

    assert active_prompt_variants(DEVIN, tmp_path) == (SKILLS_VARIANT,)


@pytest.mark.parametrize("missing", _ALL)
def test_devin_with_one_skill_missing_gets_the_plain_loop(tmp_path: Path, missing: str) -> None:
    _ship(tmp_path, ".devin/skills", tuple(name for name in _ALL if name != missing))

    assert active_prompt_variants(DEVIN, tmp_path) == ()


def test_devin_with_no_skills(tmp_path: Path) -> None:
    assert active_prompt_variants(DEVIN, tmp_path) == ()


def test_devin_with_extra_skills_beyond_the_declared_set(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", (*_ALL, "unrelated-extra"))

    assert active_prompt_variants(DEVIN, tmp_path) == (SKILLS_VARIANT,)


def test_claude_code_ignores_the_devin_skills_dir(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", _ALL)

    assert active_prompt_variants(CLAUDE, tmp_path) == ()


def test_claude_code_with_all_skills_under_claude_dir(tmp_path: Path) -> None:
    _ship(tmp_path, ".claude/skills", _ALL)

    assert active_prompt_variants(CLAUDE, tmp_path) == (SKILLS_VARIANT,)


@pytest.mark.parametrize("harness", ["api", "command", "manual", "no-such-harness", ""])
def test_harness_without_a_skill_loader_gets_the_plain_loop(tmp_path: Path, harness: str) -> None:
    _ship(tmp_path, ".devin/skills", _ALL)
    _ship(tmp_path, ".claude/skills", _ALL)

    assert active_prompt_variants(harness, tmp_path) == ()


def test_no_repo_root_gets_the_plain_loop() -> None:
    assert active_prompt_variants(DEVIN, None) == ()


def test_directory_without_skill_md_does_not_satisfy_the_declaration(tmp_path: Path) -> None:
    _ship(tmp_path, ".devin/skills", _ALL[:-1])
    (tmp_path / ".devin" / "skills" / _ALL[-1]).mkdir()

    assert active_prompt_variants(DEVIN, tmp_path) == ()


@pytest.mark.parametrize("stray_dir", [".agents/skills", ".github/skills", "skills"])
def test_skills_in_an_unrelated_dir_do_not_count(tmp_path: Path, stray_dir: str) -> None:
    _ship(tmp_path, stray_dir, _ALL)

    assert active_prompt_variants(DEVIN, tmp_path) == ()


def test_declaration_is_derived_so_a_repo_local_one_changes_the_requirement(
    tmp_path: Path,
) -> None:
    """The set the repo must ship is whatever the overlay declares, not a constant."""
    prompt_dir = _override_declaration(tmp_path / "prompts", "- `/only-one` - the sole skill\n")
    repo = tmp_path / "repo"
    _ship(repo, ".devin/skills", ("only-one",))

    assert active_prompt_variants(DEVIN, repo, (prompt_dir,)) == (SKILLS_VARIANT,)
    # The same repo does not satisfy the package's own (six-skill) declaration.
    assert active_prompt_variants(DEVIN, repo) == ()


def test_empty_declaration_never_activates_the_variant(tmp_path: Path) -> None:
    prompt_dir = _override_declaration(tmp_path / "prompts", "No bullets here.\n")
    repo = tmp_path / "repo"
    _ship(repo, ".devin/skills", _ALL)

    assert declared_skills((prompt_dir,)) == ()
    assert active_prompt_variants(DEVIN, repo, (prompt_dir,)) == ()


# ---------------------------------------------------------------------------
# Registry-derived matrix and invariant
# ---------------------------------------------------------------------------


def test_registry_pins_the_observed_skill_dirs() -> None:
    assert HARNESS_REGISTRY[DEVIN].skill_dirs == (".devin/skills", ".claude/skills")
    assert HARNESS_REGISTRY[CLAUDE].skill_dirs == (".claude/skills",)


@pytest.mark.parametrize("name", sorted(HARNESS_REGISTRY))
def test_skill_dirs_are_repo_relative_forward_slash_paths(name: str) -> None:
    skill_dirs = HARNESS_REGISTRY[name].skill_dirs

    assert isinstance(skill_dirs, tuple)
    assert len(set(skill_dirs)) == len(skill_dirs), f"{name}: duplicate skill dir"
    for skill_dir in skill_dirs:
        assert isinstance(skill_dir, str) and skill_dir
        assert "\\" not in skill_dir, f"{name}: {skill_dir!r} uses a backslash"
        assert not skill_dir.startswith("/"), f"{name}: {skill_dir!r} is rooted"
        assert not skill_dir.endswith("/"), f"{name}: {skill_dir!r} has a trailing slash"
        assert not PurePosixPath(skill_dir).is_absolute()
        assert not PureWindowsPath(skill_dir).is_absolute()
        assert not PureWindowsPath(skill_dir).drive, f"{name}: {skill_dir!r} has a drive"
        assert ".." not in PurePosixPath(skill_dir).parts, f"{name}: {skill_dir!r} escapes root"


def test_harness_capabilities_is_frozen_and_defaults_to_no_skill_dirs() -> None:
    caps = HarnessCapabilities(worker=True, review=False, adapter_kind="x")

    assert caps.skill_dirs == ()
    with pytest.raises(AttributeError):
        caps.skill_dirs = (".x",)  # type: ignore[misc]


def test_every_skill_loading_harness_activates_from_each_of_its_dirs(tmp_path: Path) -> None:
    """Derived from the registry, not from a list of harness names."""
    checked = 0
    for name, caps in sorted(HARNESS_REGISTRY.items()):
        for index, skill_dir in enumerate(caps.skill_dirs):
            repo = tmp_path / f"{name}-{index}"
            _ship(repo, skill_dir, _ALL)
            assert active_prompt_variants(name, repo) == (SKILLS_VARIANT,), (name, skill_dir)
            checked += 1
    assert checked >= 3


def test_every_harness_without_skill_dirs_gets_the_plain_loop(tmp_path: Path) -> None:
    every_dir = {skill_dir for caps in HARNESS_REGISTRY.values() for skill_dir in caps.skill_dirs}
    for skill_dir in every_dir:
        _ship(tmp_path, skill_dir, _ALL)

    blind = [name for name, caps in HARNESS_REGISTRY.items() if not caps.skill_dirs]
    assert blind, "expected at least one harness with no skill loader"
    for name in blind:
        assert active_prompt_variants(name, tmp_path) == (), name


# ---------------------------------------------------------------------------
# Guards: loop text cannot use an undeclared skill; every declared skill is used
# ---------------------------------------------------------------------------


def _overlay_files() -> list[Path]:
    return sorted(path for path in SKILLS_OVERLAY_DIR.rglob("*") if path.is_file())


def test_overlay_directory_is_found_and_not_empty() -> None:
    """Positive control for the guards below: the glob sees the overlay files."""
    names = {path.name for path in _overlay_files()}

    assert "available_skills.md" in names
    assert len(names) >= 4
    assert any(SLASH_SKILL.search(path.read_text(encoding="utf-8")) for path in _overlay_files())


def test_every_slash_token_in_the_skills_overlay_files_is_declared() -> None:
    declared = set(declared_skills())

    for path in _overlay_files():
        used = set(SLASH_SKILL.findall(path.read_text(encoding="utf-8")))
        assert used <= declared, f"{path.name} uses undeclared skill(s): {sorted(used - declared)}"


@pytest.mark.parametrize("template", ["worker.md", "rework.md"])
def test_every_slash_token_in_the_skills_render_is_declared(template: str) -> None:
    rendered = render_prompt(template, _RENDER_VALUES, variants=(SKILLS_VARIANT,))

    used = set(SLASH_SKILL.findall(rendered))

    assert used, f"{template} rendered under the skills variant names no skill at all"
    assert used <= set(declared_skills()), sorted(used - set(declared_skills()))


def test_every_declared_skill_is_referenced_by_the_skills_loop_text() -> None:
    """The converse: a bullet nobody uses would make the prompt demand a skill the loop
    never invokes."""
    loop_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _overlay_files()
        if path.name != "available_skills.md"
    )
    referenced = set(SLASH_SKILL.findall(loop_text))

    assert set(declared_skills()) <= referenced, sorted(set(declared_skills()) - referenced)


def test_the_skills_worker_render_lists_and_uses_every_declared_skill() -> None:
    rendered = render_prompt("worker.md", _RENDER_VALUES, variants=(SKILLS_VARIANT,))

    assert "## Available skills" in rendered
    for name in declared_skills():
        assert f"- `/{name}` - " in rendered, f"{name} missing from the bullet list"
        assert rendered.count(f"`/{name}`") >= 2, f"{name} is listed but never used in the loop"


@pytest.mark.parametrize(
    "template", ["worker.md", "worker_claude_code.md", "worker_local.md", "rework.md"]
)
def test_plain_render_names_no_declared_skill(template: str) -> None:
    rendered = render_prompt(template, _RENDER_VALUES)

    assert not (set(SLASH_SKILL.findall(rendered)) & set(declared_skills()))
    assert "## Available skills" not in rendered
