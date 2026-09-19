"""``derive_pytest_runner`` on unreadable and structurally malformed ``pyproject.toml``.

The runner is derived at the dispatch boundary (both prompt writers call it), where an
exception would fail the dispatch, not just the test-command hint. So an absent,
undecodable, syntactically broken, or wrongly-*shaped* file (a scalar where a table
belongs) must read as "nothing derivable" -- ``None`` -- and never raise. The
well-formed controls prove the harness still finds pytest beside a broken sibling key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.prompt_test_command import derive_pytest_runner

_HEADER = '[project]\nname = "x"\nversion = "0"\n'


def _write_pyproject(root: Path, body: str, *, header: str = _HEADER) -> Path:
    (root / "pyproject.toml").write_text(header + body, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Unreadable input: never raises, never invents
# ---------------------------------------------------------------------------


def test_missing_pyproject_yields_none(tmp_path: Path) -> None:
    assert not (tmp_path / "pyproject.toml").exists()

    assert derive_pytest_runner(tmp_path) is None


def test_missing_repo_root_directory_yields_none(tmp_path: Path) -> None:
    assert derive_pytest_runner(tmp_path / "does-not-exist") is None


def test_malformed_toml_yields_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project\nname = "x"\ndev = ["pytest"\n', encoding="utf-8"
    )

    assert derive_pytest_runner(tmp_path) is None


def test_undecodable_pyproject_yields_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_bytes(b"\xff\xfe[project]\n\x80\x81")

    assert derive_pytest_runner(tmp_path) is None


def test_pyproject_that_is_a_directory_yields_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").mkdir()

    assert derive_pytest_runner(tmp_path) is None


def test_empty_pyproject_yields_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    assert derive_pytest_runner(tmp_path) is None


# Valid TOML whose *shape* is wrong: a scalar or array where a table belongs. This runs
# at the dispatch boundary, so it must read as "absent" rather than raise.
_MALFORMED_SHAPE_BODIES: list[tuple[str, str]] = [
    ("dependency-groups-int", "dependency-groups = 3\n"),
    ("dependency-groups-str", 'dependency-groups = "x"\n'),
    ("dependency-groups-array", "dependency-groups = [1]\n"),
    ("tool-int", "tool = 3\n"),
    ("tool-uv-int", "[tool]\nuv = 3\n"),
    ("project-int", "project = 3\n"),
    ("optional-dependencies-int", '[project]\nname="x"\noptional-dependencies = 3\n'),
    ("optional-dependencies-array", '[project]\nname="x"\noptional-dependencies = [1]\n'),
    ("core-dependencies-int", '[project]\nname="x"\ndependencies = 3\n'),
    ("legacy-dev-dependencies-int", "[tool.uv]\ndev-dependencies = 3\n"),
    ("default-groups-int", "[tool.uv]\ndefault-groups = 3\n"),
]


@pytest.mark.parametrize(
    "body",
    [pytest.param(body, id=case_id) for case_id, body in _MALFORMED_SHAPE_BODIES],
)
def test_malformed_table_shape_reads_as_absent_and_never_raises(tmp_path: Path, body: str) -> None:
    _write_pyproject(tmp_path, body, header="")

    assert derive_pytest_runner(tmp_path) is None


# Controls for the cases above: the same harness, with well-formed input, must resolve
# -- including when a *sibling* key is broken -- so the ``None`` results above are the
# malformed key being ignored, not the harness failing to find anything at all.
_WELL_FORMED_CONTROLS: list[tuple[str, str, str]] = [
    (
        "dev-group",
        '[dependency-groups]\ndev=["pytest>=8"]\n',
        "uv run pytest",
    ),
    (
        "dev-extra",
        '[project.optional-dependencies]\ndev=["pytest"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "legacy-path-survives-a-broken-dependency-groups-key",
        'dependency-groups = 3\n[tool.uv]\ndev-dependencies=["pytest"]\n',
        "uv run pytest",
    ),
    (
        "dependency-groups-survive-a-broken-project-key",
        'project = 3\n[dependency-groups]\ndev=["pytest"]\n',
        "uv run pytest",
    ),
    (
        "dependency-groups-survive-a-broken-tool-key",
        'tool = 3\n[dependency-groups]\ndev=["pytest"]\n',
        "uv run pytest",
    ),
    (
        "dev-extra-survives-a-broken-dependency-groups-key",
        'dependency-groups = "x"\n[project.optional-dependencies]\ndev=["pytest"]\n',
        "uv run --extra dev pytest",
    ),
    (
        "malformed-default-groups-falls-back-to-the-uv-default",
        '[dependency-groups]\ndev=["pytest"]\n[tool.uv]\ndefault-groups = 3\n',
        "uv run pytest",
    ),
]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(body, expected, id=case_id)
        for case_id, body, expected in _WELL_FORMED_CONTROLS
    ],
)
def test_well_formed_controls_resolve_beside_malformed_shapes(
    tmp_path: Path, body: str, expected: str
) -> None:
    _write_pyproject(tmp_path, body, header="")

    assert derive_pytest_runner(tmp_path) == expected
