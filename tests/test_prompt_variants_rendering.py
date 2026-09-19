"""End-to-end: worker and rework prompts rendered through the REAL writers.

``OrchestratorApp._write_worker_prompt`` / ``_write_rework_prompt`` are driven against
consumer-shaped tmp repos, so what is asserted is the text a worker actually receives:

* *extra-style*: a ``dev`` extra lists pytest, and the repo ships every skill the loop
  declares -- the skills loop and ``--extra dev`` must survive unchanged.
* *group-style*: dev tools live in a PEP 735 ``[dependency-groups]`` group and the repo
  ships no skills -- the plain git/gh loop, ``uv run pytest``, and no ``--extra``.
* *bare*: no ``pyproject.toml`` at the root -- no command is invented; the prompt points
  at the repository's docs, and ``dispatch.test_command`` fills the gap when set.
"""

from __future__ import annotations

import dataclasses
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.prompt_skills import declared_skills
from charlie_work.prompt_test_command import UNRESOLVED_FULL_SUITE, UNRESOLVED_TARGETED
from charlie_work.workflow import OrchestratorApp

DEVIN = "devin-shell"
CLAUDE = "claude-code"

EXTRA_PYPROJECT = (
    '[project]\nname = "x"\nversion = "0"\n[project.optional-dependencies]\n'
    'dev = ["pytest>=8", "pytest-cov>=5", "pytest-xdist>=3"]\n'
)
GROUP_PYPROJECT = (
    '[project]\nname = "x"\nversion = "0"\n[dependency-groups]\ndev = ["pytest>=8"]\n'
)

ISSUE = {"number": 7, "title": "fix: x", "url": "https://example.test/7", "body": "b"}
PR = {"number": 8, "title": "t", "url": "https://example.test/8", "headRefName": "agent/issue-7-x"}

SLASH_SKILL = re.compile(r"`/([a-z0-9][a-z0-9-]*)`")
TARGETED_BLOCK = re.compile(
    r"not just the tests you wrote:\n\s*```bash\n\s*(?P<cmd>[^\n]+)\n\s*```"
)
FULL_SUITE = re.compile(r"before pushing \((?P<cmd>[^)]+)\)\.")

PRECEDENCE = "**Repository commands take precedence.**"
IMPACTED = "<impacted test files>"


@dataclass(frozen=True)
class Shape:
    """A consumer layout plus the config it runs under."""

    name: str
    pyproject: str | None
    skills: tuple[str, ...] = ()
    skill_dir: str = ".devin/skills"
    harness: str = DEVIN
    dispatch: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Rendered:
    worker: str
    rework: str


def _all_skills() -> tuple[str, ...]:
    return declared_skills()


SHAPES: tuple[Shape, ...] = (
    Shape("extra-style", EXTRA_PYPROJECT, _all_skills()),
    Shape("extra-style-claude-dir", EXTRA_PYPROJECT, _all_skills(), ".claude/skills"),
    Shape("group-style", GROUP_PYPROJECT),
    Shape("bare", None),
    Shape(
        "bare-with-override", None, dispatch={"test_command": "uv run --directory server pytest"}
    ),
    Shape("partial-skills", EXTRA_PYPROJECT, _all_skills()[:-1]),
    Shape("claude-code-devin-dir", EXTRA_PYPROJECT, _all_skills(), harness=CLAUDE),
)


def _render(
    root: Path, shape: Shape, worker_template: str | None = None, harness: str | None = None
) -> Rendered:
    """Build ``shape`` under ``root`` and render both prompts through the real writers."""
    root.mkdir(parents=True, exist_ok=True)
    if shape.pyproject is not None:
        (root / "pyproject.toml").write_text(shape.pyproject, encoding="utf-8")
    for name in shape.skills:
        skill = root / shape.skill_dir / name
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    base = OrchestratorConfig()
    dispatch = dict(shape.dispatch)
    if worker_template is not None:
        dispatch["worker_template"] = worker_template
    config = dataclasses.replace(
        base,
        worker=dataclasses.replace(base.worker, harness=harness or shape.harness),
        dispatch=dataclasses.replace(base.dispatch, **dispatch),
    )
    app = OrchestratorApp(root, runtime_paths(root, config.runtime.state_dir), config, gh=None)
    worker = app._write_worker_prompt(ISSUE).read_text(encoding="utf-8")
    (app.paths.prs / "pr-8").mkdir(parents=True, exist_ok=True)
    rework = app._write_rework_prompt(PR, 7, "note").read_text(encoding="utf-8")
    return Rendered(worker=worker, rework=rework)


def _shape(name: str) -> Shape:
    return next(shape for shape in SHAPES if shape.name == name)


def _targeted(prompt: str) -> str:
    match = TARGETED_BLOCK.search(prompt)
    assert match, "no fenced targeted-test command block found"
    return match.group("cmd").strip()


def _full_suite(prompt: str) -> str:
    match = FULL_SUITE.search(prompt)
    assert match, "no full-suite parenthetical found"
    return match.group("cmd")


def _skill_tokens(prompt: str) -> set[str]:
    return set(SLASH_SKILL.findall(prompt)) & set(declared_skills())


# ---------------------------------------------------------------------------
# extra-style consumer: skills loop and --extra dev are preserved
# ---------------------------------------------------------------------------


def test_extra_style_worker_gets_the_skills_loop(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("extra-style"))

    assert "## Available skills" in rendered.worker
    assert _skill_tokens(rendered.worker) == set(declared_skills())
    assert "1. Use `/create-branch` to ensure you're on the correct branch." in rendered.worker
    assert "The `/test` skill is a convenience shortcut" in rendered.worker
    assert "`/test` - Run the test suite and verify all tests pass (only if it wraps" in (
        rendered.worker
    )
    assert "git switch -c agent/issue-7-fix-x" not in rendered.worker


def test_extra_style_rework_names_preflight(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("extra-style"))

    assert "run `/preflight` (ruff + ruff-format + pre-commit) and COMMIT anything" in (
        rendered.rework
    )
    assert "pre-commit run --files" not in rendered.rework


def test_skills_loop_step_text_is_intact_for_a_skills_consumer(tmp_path: Path) -> None:
    """Steps 1 and 8-12 read exactly as they did before the loop became conditional."""
    worker = _render(tmp_path, _shape("extra-style")).worker

    assert "8. Use `/commit` to commit your changes with conventional format.\n" in worker
    assert (
        "9. Use `/preflight` to match CI (ruff, ruff-format, pre-commit). Commit anything it\n"
        "   fixes — an uncommitted reflow or an un-normalized fixture is the #1 cause of a\n"
        "   green-locally / red-on-CI PR, and the push/PR gate will block you on it.\n"
        "10. Use `/push` to push your branch to GitHub.\n"
        "11. Use `/create-pr` to create a pull request with proper formatting.\n"
        "12. Use `/complete` to finalize the session.\n"
    ) in worker


def test_extra_style_keeps_the_dev_extra_command(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("extra-style"))

    for prompt in (rendered.worker, rendered.rework):
        assert _targeted(prompt) == f"uv run --extra dev pytest {IMPACTED} -q --tb=short"
        assert _full_suite(prompt) == "`uv run --extra dev pytest -q --tb=short`"


def test_devin_shell_loads_skills_from_the_claude_dir_too(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("extra-style-claude-dir"))

    assert "## Available skills" in rendered.worker
    assert "`/preflight`" in rendered.rework


# ---------------------------------------------------------------------------
# group-style consumer: plain loop, no --extra
# ---------------------------------------------------------------------------


def test_group_style_prompts_name_no_slash_skill(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("group-style"))

    assert _skill_tokens(rendered.worker) == set()
    assert _skill_tokens(rendered.rework) == set()
    assert "## Available skills" not in rendered.worker
    assert "convenience shortcut" not in rendered.worker


def test_group_style_worker_carries_the_plain_git_gh_loop(tmp_path: Path) -> None:
    worker = _render(tmp_path, _shape("group-style")).worker

    assert "git switch -c agent/issue-7-fix-x origin/main" in worker
    assert "git push -u origin agent/issue-7-fix-x" in worker
    assert "gh pr create" in worker
    assert "git status --short" in worker
    assert "8. Commit your changes with a Conventional-Commits message" in worker


def test_group_style_rework_carries_the_plain_preflight(tmp_path: Path) -> None:
    rework = _render(tmp_path, _shape("group-style")).rework

    assert "`ruff check`, `ruff format --check`" in rework
    assert "pre-commit run --files" in rework
    assert "git push origin agent/issue-7-x" in rework


def test_group_style_commands_have_no_extra_flag_and_no_flat_test_path(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("group-style"))

    for prompt in (rendered.worker, rendered.rework):
        assert _targeted(prompt) == f"uv run pytest {IMPACTED} -q --tb=short"
        assert _full_suite(prompt) == "`uv run pytest -q --tb=short`"
        assert "--extra dev" not in prompt
        assert "test_<touched_module>" not in prompt


# ---------------------------------------------------------------------------
# bare consumer (no root pyproject): nothing invented; config fills the gap
# ---------------------------------------------------------------------------


def test_bare_repo_points_at_the_repositorys_own_docs(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("bare"))

    for prompt in (rendered.worker, rendered.rework):
        assert _targeted(prompt) == UNRESOLVED_TARGETED
        assert _full_suite(prompt) == UNRESOLVED_FULL_SUITE
        assert "CLAUDE.md / CONTRIBUTING.md" in _targeted(prompt)


def test_configured_test_command_reaches_both_lanes(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("bare-with-override"))
    runner = "uv run --directory server pytest"

    for prompt in (rendered.worker, rendered.rework):
        assert _targeted(prompt) == f"{runner} {IMPACTED} -q --tb=short"
        assert _full_suite(prompt) == f"`{runner} -q --tb=short`"


def test_configured_test_command_beats_derivation(tmp_path: Path) -> None:
    shape = dataclasses.replace(
        _shape("extra-style"), dispatch={"test_command": "uv run --directory server pytest"}
    )

    rendered = _render(tmp_path, shape)

    for prompt in (rendered.worker, rendered.rework):
        assert _targeted(prompt).startswith("uv run --directory server pytest ")
        assert "--extra dev" not in prompt


# ---------------------------------------------------------------------------
# The conservative edges of the skills decision
# ---------------------------------------------------------------------------


def test_partial_skills_get_the_plain_loop(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("partial-skills"))

    assert _skill_tokens(rendered.worker) == set()
    assert _skill_tokens(rendered.rework) == set()
    assert "git push -u origin agent/issue-7-fix-x" in rendered.worker


def test_claude_code_harness_ignores_the_devin_skills_dir(tmp_path: Path) -> None:
    rendered = _render(tmp_path, _shape("claude-code-devin-dir"))

    assert _skill_tokens(rendered.worker) == set()
    assert _skill_tokens(rendered.rework) == set()
    assert "## Available skills" not in rendered.worker


def test_claude_code_harness_loads_the_claude_skills_dir(tmp_path: Path) -> None:
    shape = dataclasses.replace(_shape("extra-style-claude-dir"), harness=CLAUDE)

    rendered = _render(tmp_path, shape)

    assert "## Available skills" in rendered.worker
    assert "`/preflight`" in rendered.rework


# ---------------------------------------------------------------------------
# Every shape, every template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.name)
def test_no_shape_leaves_a_placeholder_or_the_flat_test_path(tmp_path: Path, shape: Shape) -> None:
    rendered = _render(tmp_path, shape)

    for prompt in (rendered.worker, rendered.rework):
        assert not string.Template(prompt).get_identifiers()
        assert "test_<touched_module>" not in prompt
        assert PRECEDENCE in prompt
        assert "tests/<package>/test_<module>.py" in prompt


@pytest.mark.parametrize("template", ["worker.md", "worker_claude_code.md", "worker_local.md"])
@pytest.mark.parametrize("shape_name", ["extra-style", "group-style", "bare"])
def test_every_worker_template_carries_the_precedence_clause(
    tmp_path: Path, template: str, shape_name: str
) -> None:
    worker = _render(tmp_path, _shape(shape_name), worker_template=template).worker

    assert PRECEDENCE in worker
    assert "`CLAUDE.md` or `CONTRIBUTING.md` documents a different command" in worker
    assert not string.Template(worker).get_identifiers()
    assert (
        _full_suite(worker)
        == {
            "extra-style": "`uv run --extra dev pytest -q --tb=short`",
            "group-style": "`uv run pytest -q --tb=short`",
            "bare": UNRESOLVED_FULL_SUITE,
        }[shape_name]
    )


def test_claude_code_template_has_the_derived_command_and_no_skills(tmp_path: Path) -> None:
    shape = _shape("extra-style-claude-dir")

    worker = _render(
        tmp_path, shape, worker_template="worker_claude_code.md", harness=CLAUDE
    ).worker

    assert "/create-branch" not in worker
    assert _skill_tokens(worker) == set()
    assert _targeted(worker) == f"uv run --extra dev pytest {IMPACTED} -q --tb=short"
    assert "git switch -c agent/issue-7-fix-x origin/main" in worker


def test_claude_code_template_follows_a_dependency_group(tmp_path: Path) -> None:
    worker = _render(
        tmp_path, _shape("group-style"), worker_template="worker_claude_code.md", harness=CLAUDE
    ).worker

    assert _targeted(worker) == f"uv run pytest {IMPACTED} -q --tb=short"
    assert "--extra dev" not in worker
