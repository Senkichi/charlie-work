"""``resolve_test_commands`` precedence and the ``dispatch.test_command`` config key.

Precedence is fixed: the operator's ``dispatch.test_command`` first, then what the
consumer's own ``pyproject.toml`` declares, then neither -- in which case no command is
invented and the prompt points the worker at the repository's own docs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from charlie_work.config import ConfigError, DispatchConfig, build_config_from_data
from charlie_work.prompt_test_command import (
    IMPACTED_TESTS_PLACEHOLDER,
    PYTEST_FLAGS,
    UNRESOLVED_FULL_SUITE,
    UNRESOLVED_TARGETED,
    WorkerTestCommands,
    prompt_test_command_values,
    resolve_test_commands,
)
from charlie_work.workflow import REWORK_PROMPT_KEYS, WORKER_PROMPT_KEYS

_HEADER = '[project]\nname = "x"\nversion = "0"\n'


def _write_pyproject(root: Path, body: str, *, header: str = _HEADER) -> Path:
    (root / "pyproject.toml").write_text(header + body, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# resolve_test_commands: precedence, and the shape of both strings
# ---------------------------------------------------------------------------


def _dev_extra_repo(root: Path) -> Path:
    return _write_pyproject(root, '[project.optional-dependencies]\ndev = ["pytest"]\n')


def test_config_override_beats_derivation_and_is_stripped(tmp_path: Path) -> None:
    _dev_extra_repo(tmp_path)

    commands = resolve_test_commands("  uv run --directory server pytest \t", tmp_path)

    assert commands.targeted == (
        "uv run --directory server pytest <impacted test files> -q --tb=short"
    )
    assert commands.full_suite == "`uv run --directory server pytest -q --tb=short`"
    assert "--extra" not in commands.targeted + commands.full_suite


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \t\n "])
def test_blank_override_falls_through_to_derivation(tmp_path: Path, blank: str) -> None:
    _dev_extra_repo(tmp_path)

    assert resolve_test_commands(blank, tmp_path) == resolve_test_commands("", tmp_path)
    assert resolve_test_commands(blank, tmp_path).targeted.startswith("uv run --extra dev pytest ")


def test_override_works_without_a_repo_root() -> None:
    commands = resolve_test_commands("make test", None)

    assert commands.targeted == "make test <impacted test files> -q --tb=short"
    assert commands.full_suite == "`make test -q --tb=short`"


def test_no_override_and_no_repo_root_is_unresolved() -> None:
    commands = resolve_test_commands("", None)

    assert commands == WorkerTestCommands(
        targeted=UNRESOLVED_TARGETED, full_suite=UNRESOLVED_FULL_SUITE
    )


def test_no_override_and_nothing_derivable_is_unresolved(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, '[project.optional-dependencies]\ndev = ["ruff"]\n')

    assert resolve_test_commands("", tmp_path) == resolve_test_commands("", None)


def test_unresolved_forms_point_the_worker_at_the_repos_own_docs() -> None:
    assert "CLAUDE.md" in UNRESOLVED_TARGETED
    assert "CONTRIBUTING.md" in UNRESOLVED_TARGETED
    assert "CLAUDE.md" in UNRESOLVED_FULL_SUITE
    assert "CONTRIBUTING.md" in UNRESOLVED_FULL_SUITE
    # The targeted form sits inside a ```bash fence, so it must be a comment line and
    # must not read as an executable command.
    assert UNRESOLVED_TARGETED.startswith("# ")
    # No command is invented: neither form names a runner.
    for text in (UNRESOLVED_TARGETED, UNRESOLVED_FULL_SUITE):
        assert "uv run" not in text
        assert "pytest" not in text


def test_resolved_forms_have_the_documented_shape(tmp_path: Path) -> None:
    _dev_extra_repo(tmp_path)

    commands = resolve_test_commands("", tmp_path)

    assert (
        commands.targeted
        == f"uv run --extra dev pytest {IMPACTED_TESTS_PLACEHOLDER} {PYTEST_FLAGS}"
    )
    assert IMPACTED_TESTS_PLACEHOLDER == "<impacted test files>"
    assert PYTEST_FLAGS == "-q --tb=short"
    # The full-suite form sits inside a parenthetical, so it is one backtick-wrapped
    # span, and it must not name the impacted-files placeholder (it runs everything).
    assert commands.full_suite.startswith("`")
    assert commands.full_suite.endswith("`")
    assert commands.full_suite.count("`") == 2
    assert IMPACTED_TESTS_PLACEHOLDER not in commands.full_suite


def test_resolved_commands_never_use_the_flat_touched_module_shape(tmp_path: Path) -> None:
    _dev_extra_repo(tmp_path)

    commands = resolve_test_commands("", tmp_path)

    assert "test_<touched_module>" not in commands.targeted + commands.full_suite


def test_worker_test_commands_is_a_frozen_value_object() -> None:
    commands = WorkerTestCommands(targeted="a", full_suite="b")

    with pytest.raises(dataclasses.FrozenInstanceError):
        commands.targeted = "c"  # type: ignore[misc]


def test_prompt_values_keys_are_exactly_the_two_template_placeholders(tmp_path: Path) -> None:
    _dev_extra_repo(tmp_path)

    values = prompt_test_command_values("", tmp_path)

    assert set(values) == {"targeted_test_command", "full_suite_command"}
    commands = resolve_test_commands("", tmp_path)
    assert values["targeted_test_command"] == commands.targeted
    assert values["full_suite_command"] == commands.full_suite


def test_prompt_values_are_declared_in_both_writers_key_sets() -> None:
    """Wiring: the drift check subsets a template's placeholders against the writer's
    declared keys, so the two test-command keys must be declared for worker AND rework."""
    keys = set(prompt_test_command_values("", None))

    assert keys <= WORKER_PROMPT_KEYS
    assert keys <= REWORK_PROMPT_KEYS


# ---------------------------------------------------------------------------
# dispatch.test_command config key
# ---------------------------------------------------------------------------


def test_dispatch_config_test_command_defaults_to_empty() -> None:
    assert DispatchConfig().test_command == ""
    assert build_config_from_data({}).dispatch.test_command == ""


def test_dispatch_config_test_command_is_a_frozen_field() -> None:
    config = DispatchConfig(test_command="uv run --directory server pytest")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.test_command = "other"  # type: ignore[misc]


def test_config_loading_accepts_a_string_test_command() -> None:
    config = build_config_from_data(
        {"dispatch": {"test_command": "uv run --directory server pytest"}}
    )

    assert config.dispatch.test_command == "uv run --directory server pytest"


@pytest.mark.parametrize("bad", [5, True, ["uv", "run", "pytest"], {"cmd": "pytest"}, 1.5])
def test_config_loading_rejects_a_non_string_test_command(bad: object) -> None:
    with pytest.raises(ConfigError, match="dispatch.*test_command.*must be a string"):
        build_config_from_data({"dispatch": {"test_command": bad}})


def test_configured_test_command_reaches_the_prompt_values() -> None:
    """End to end from config data to the placeholder values the writers splice in."""
    config = build_config_from_data(
        {"dispatch": {"test_command": "uv run --directory server pytest"}}
    )

    values = prompt_test_command_values(config.dispatch.test_command, None)

    assert values["targeted_test_command"].startswith("uv run --directory server pytest ")
    assert values["full_suite_command"] == "`uv run --directory server pytest -q --tb=short`"


def test_blank_yaml_test_command_resolves_like_the_default() -> None:
    """A key left blank in YAML parses to None; config normalizes it to the empty default."""
    config = build_config_from_data({"dispatch": {"test_command": None}})

    assert config.dispatch.test_command == ""
    assert prompt_test_command_values(config.dispatch.test_command, None) == (
        prompt_test_command_values("", None)
    )
