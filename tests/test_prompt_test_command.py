"""Tests for ``prompt_test_command``: the test command a worker prompt carries.

The worker/rework templates used to hardcode ``uv run --extra dev pytest`` for every
consumer, which fails before pytest starts for a consumer whose dev tools live in a
PEP 735 ``[dependency-groups]`` group. The runner is now resolved once, with a fixed
precedence -- ``dispatch.test_command``, then what the consumer's own ``pyproject.toml``
declares, then nothing (no command is invented) -- and spliced into the templates
through ``$targeted_test_command`` / ``$full_suite_command``.

``derive_pytest_runner`` is table-driven over tmp ``pyproject.toml`` files. The
positive controls are shaped like the two real consumer layouts (a ``dev`` extra that
also lists ``pytest-cov`` / ``pytest-xdist``; a ``dev`` dependency group) so the
name matcher and the group logic are exercised against realistic input, and the
``pytest-cov``-alone case is the negative control proving the matcher is exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.prompt_test_command import derive_pytest_runner, resolve_test_commands

_HEADER = '[project]\nname = "x"\nversion = "0"\n'


def _write_pyproject(root: Path, body: str, *, header: str = _HEADER) -> Path:
    (root / "pyproject.toml").write_text(header + body, encoding="utf-8")
    return root


# (id, pyproject body after the [project] header, expected runner)
_DERIVATION_CASES: list[tuple[str, str, str | None]] = [
    (
        "dev-extra",
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "non-dev-extra-only",
        '[project.optional-dependencies]\ntest = ["pytest>=8"]\n',
        "uv run --extra test pytest",
    ),
    (
        "dev-and-test-extras-prefers-dev",
        '[project.optional-dependencies]\ntest = ["pytest>=8"]\ndev = ["pytest>=8"]\n',
        "uv run --extra dev pytest",
    ),
    (
        # ``ci`` sorts before ``dev``: the preference for dev is not an artefact of order.
        "dev-preferred-over-alphabetically-earlier-extra",
        '[project.optional-dependencies]\nci = ["pytest>=8"]\ndev = ["pytest>=8"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "first-sorted-extra-when-no-dev",
        '[project.optional-dependencies]\nzeta = ["pytest"]\nalpha = ["pytest"]\n',
        "uv run --extra alpha pytest",
    ),
    (
        "extra-not-listing-pytest-is-skipped",
        '[project.optional-dependencies]\nlint = ["ruff"]\ntest = ["pytest"]\n',
        "uv run --extra test pytest",
    ),
    (
        "dev-group-is-a-uv-default-group",
        '[dependency-groups]\ndev = ["pytest>=8"]\n',
        "uv run pytest",
    ),
    (
        "non-default-group-needs-explicit-flag",
        '[dependency-groups]\nqa = ["pytest"]\n',
        "uv run --group qa pytest",
    ),
    (
        "empty-default-groups-makes-dev-group-explicit",
        '[dependency-groups]\ndev = ["pytest"]\n[tool.uv]\ndefault-groups = []\n',
        "uv run --group dev pytest",
    ),
    (
        "non-default-group-with-default-groups-empty",
        '[dependency-groups]\nqa = ["pytest"]\n[tool.uv]\ndefault-groups = []\n',
        "uv run --group qa pytest",
    ),
    (
        "default-groups-all",
        '[dependency-groups]\nqa = ["pytest"]\n[tool.uv]\ndefault-groups = "all"\n',
        "uv run pytest",
    ),
    (
        "default-groups-names-the-group",
        '[dependency-groups]\nqa = ["pytest"]\n[tool.uv]\ndefault-groups = ["qa"]\n',
        "uv run pytest",
    ),
    (
        "default-groups-replaces-the-dev-default",
        '[dependency-groups]\ndev = ["pytest"]\n[tool.uv]\ndefault-groups = ["docs"]\n',
        "uv run --group dev pytest",
    ),
    (
        "include-group-chain-reaches-pytest",
        '[dependency-groups]\ndev = [{include-group = "test"}]\ntest = ["pytest>=8"]\n',
        "uv run pytest",
    ),
    (
        "two-hop-include-group-chain",
        "[dependency-groups]\n"
        'dev = [{include-group = "mid"}]\n'
        'mid = [{include-group = "leaf"}]\n'
        'leaf = ["pytest"]\n',
        "uv run pytest",
    ),
    (
        "include-group-of-a-group-without-pytest",
        '[dependency-groups]\ndev = [{include-group = "lint"}]\nlint = ["ruff"]\n',
        None,
    ),
    (
        "include-group-cycle-without-pytest-terminates",
        '[dependency-groups]\na = [{include-group = "b"}]\nb = [{include-group = "a"}]\n',
        None,
    ),
    (
        "include-group-cycle-with-pytest-terminates",
        '[dependency-groups]\na = [{include-group = "b"}]\nb = [{include-group = "a"}, "pytest"]\n',
        "uv run --group a pytest",
    ),
    (
        "include-group-self-reference-terminates",
        '[dependency-groups]\ndev = [{include-group = "dev"}]\n',
        None,
    ),
    (
        "include-group-to-missing-group",
        '[dependency-groups]\ndev = [{include-group = "nope"}]\n',
        None,
    ),
    (
        "legacy-tool-uv-dev-dependencies",
        '[tool.uv]\ndev-dependencies = ["pytest>=8"]\n',
        "uv run pytest",
    ),
    (
        "legacy-dev-dependencies-with-default-groups-empty",
        '[tool.uv]\ndev-dependencies = ["pytest"]\ndefault-groups = []\n',
        "uv run --group dev pytest",
    ),
    (
        "pytest-as-core-dependency",
        'dependencies = ["requests", "pytest>=8"]\n',
        "uv run pytest",
    ),
    (
        "core-dependency-beats-extra",
        'dependencies = ["pytest"]\n[project.optional-dependencies]\ndev = ["pytest"]\n',
        "uv run pytest",
    ),
    (
        "extra-beats-group",
        '[project.optional-dependencies]\ndev = ["pytest"]\n[dependency-groups]\ndev = ["pytest"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "version-specifier-range",
        '[project.optional-dependencies]\ndev = ["pytest>=8,<9"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "extras-on-the-requirement",
        '[project.optional-dependencies]\ndev = ["pytest[testing]>=8"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "environment-marker",
        "[project.optional-dependencies]\ndev = [\"pytest; python_version >= '3.10'\"]\n",
        "uv run --extra dev pytest",
    ),
    (
        "direct-url-reference",
        '[project.optional-dependencies]\ndev = ["pytest @ https://example.test/pytest.whl"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "case-and-separator-insensitive-name",
        '[project.optional-dependencies]\ndev = ["PyTest==8.0"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "leading-whitespace-in-requirement",
        '[project.optional-dependencies]\ndev = ["  pytest>=8"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "no-pytest-anywhere",
        'dependencies = ["requests"]\n[project.optional-dependencies]\ndev = ["ruff", "mypy"]\n'
        '[dependency-groups]\ndocs = ["sphinx"]\n',
        None,
    ),
    (
        "extra-value-not-a-list",
        '[project.optional-dependencies]\ndev = "pytest"\n',
        None,
    ),
    (
        "non-string-requirement-entries-ignored",
        '[project.optional-dependencies]\ndev = [1, {a = 2}, "pytest"]\n',
        "uv run --extra dev pytest",
    ),
]


@pytest.mark.parametrize(
    ("body", "expected"),
    [pytest.param(body, expected, id=case_id) for case_id, body, expected in _DERIVATION_CASES],
)
def test_derive_pytest_runner_table(tmp_path: Path, body: str, expected: str | None) -> None:
    _write_pyproject(tmp_path, body)

    assert derive_pytest_runner(tmp_path) == expected


def test_pyproject_without_a_project_table_still_reads_dependency_groups(tmp_path: Path) -> None:
    """PEP 735 does not require ``[project]``; the group must still be found."""
    _write_pyproject(tmp_path, '[dependency-groups]\ndev = ["pytest"]\n', header="")

    assert derive_pytest_runner(tmp_path) == "uv run pytest"


# ---------------------------------------------------------------------------
# The name matcher: negative control and its paired positive control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lookalike",
    ["pytest-cov", "pytest-xdist", "pytest_cov>=5", "pytest-asyncio", "pytester", "pytestx"],
)
def test_pytest_lookalike_does_not_count_as_pytest(tmp_path: Path, lookalike: str) -> None:
    """Negative control: a plugin alone must not conjure a ``pytest`` runner."""
    _write_pyproject(tmp_path, f'[project.optional-dependencies]\ndev = ["{lookalike}"]\n')

    assert derive_pytest_runner(tmp_path) is None


def test_pytest_plugins_alone_are_none_but_adding_pytest_flips_it(tmp_path: Path) -> None:
    """The paired control: the same table with ``pytest`` added *is* found, so the
    ``None`` above is the matcher's verdict and not a broken read of the file."""
    plugins = '"pytest-cov>=5", "pytest-xdist>=3"'
    _write_pyproject(tmp_path, f"[project.optional-dependencies]\ndev = [{plugins}]\n")
    assert derive_pytest_runner(tmp_path) is None

    _write_pyproject(tmp_path, f'[project.optional-dependencies]\ndev = ["pytest", {plugins}]\n')
    assert derive_pytest_runner(tmp_path) == "uv run --extra dev pytest"


def test_lookalike_group_entry_does_not_count(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, '[dependency-groups]\ndev = ["pytest-cov", "pytest-xdist"]\n')

    assert derive_pytest_runner(tmp_path) is None


# ---------------------------------------------------------------------------
# Positive controls shaped like the real consumer layouts
# ---------------------------------------------------------------------------


def test_extra_style_consumer_positive_control(tmp_path: Path) -> None:
    """A ``dev`` extra listing pytest next to its plugins: the layout charlie-work
    itself uses. Must resolve to ``--extra dev`` (what the old hardcoded string was)."""
    _write_pyproject(
        tmp_path,
        "[project.optional-dependencies]\n"
        "dev = [\n"
        '    "pytest>=8",\n'
        '    "pytest-cov>=5",\n'
        '    "pytest-xdist>=3",\n'
        '    "ruff>=0.5",\n'
        "]\n",
    )

    assert derive_pytest_runner(tmp_path) == "uv run --extra dev pytest"
    commands = resolve_test_commands("", tmp_path)
    assert commands.targeted == "uv run --extra dev pytest <impacted test files> -q --tb=short"
    assert commands.full_suite == "`uv run --extra dev pytest -q --tb=short`"


def test_group_style_consumer_positive_control(tmp_path: Path) -> None:
    """A PEP 735 ``dev`` group (no extra at all): ``--extra dev`` would fail before
    pytest starts here, so the runner must be a bare ``uv run pytest``."""
    _write_pyproject(tmp_path, '[dependency-groups]\ndev = ["pytest>=8"]\n')

    assert derive_pytest_runner(tmp_path) == "uv run pytest"
    commands = resolve_test_commands("", tmp_path)
    assert commands.targeted == "uv run pytest <impacted test files> -q --tb=short"
    assert commands.full_suite == "`uv run pytest -q --tb=short`"
    assert "--extra" not in commands.targeted + commands.full_suite
