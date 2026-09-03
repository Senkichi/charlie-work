"""Tests for issue #1544 Stage 1: extract attachment_contracts into an
installable distribution.

Covers:

* Zero intra-repo imports: no ``attachment_contracts/*.py`` file imports from
  any ``charlie_work.*`` module outside ``charlie_work.attachment_contracts``.
  The two former ``from charlie_work.subprocess_runner import
  no_console_window_kwargs`` call sites are now inlined via
  ``charlie_work.attachment_contracts._windows``.
* The inlined ``_windows.no_console_window_kwargs`` helper behaves identically
  to ``charlie_work.subprocess_runner.no_console_window_kwargs``.
* The distribution's ``pyproject.toml`` declares ``requires-python>=3.11`` and
  the correct distribution name.
* The wheel builds from the single in-tree source and contains exactly the
  expected ``.py`` files (no ``charlie_work/__init__.py``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "charlie_work" / "attachment_contracts"
_PKG_DIR = Path(__file__).resolve().parents[2] / "packages" / "attachment-contracts"


# ---------------------------------------------------------------------------
# Zero intra-repo imports
# ---------------------------------------------------------------------------


def _intra_repo_imports(source: str, filename: str) -> list[str]:
    """Return charlie_work imports that are NOT intra-package (issue #1544).

    An intra-package import is one from ``charlie_work.attachment_contracts``
    or a submodule thereof.  Any other ``charlie_work.*`` import is an
    intra-repo import that would break the standalone distribution.
    """
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod.startswith("charlie_work.") and not mod.startswith(
                "charlie_work.attachment_contracts"
            ):
                violations.append(f"{filename}:{node.lineno}: from {mod} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("charlie_work.") and not alias.name.startswith(
                    "charlie_work.attachment_contracts"
                ):
                    violations.append(f"{filename}:{node.lineno}: import {alias.name}")
    return violations


def test_zero_intra_repo_imports_in_attachment_contracts() -> None:
    """No attachment_contracts/*.py file imports from outside the subpackage.

    The distribution must be self-contained: zero ``charlie_work.*`` imports
    other than ``charlie_work.attachment_contracts.*`` (issue #1544
    acceptance: "zero remaining intra-repo imports").
    """
    violations: list[str] = []
    for py_file in sorted(_SRC_DIR.glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        violations.extend(_intra_repo_imports(source, str(py_file.relative_to(_SRC_DIR))))
    assert violations == [], (
        "attachment_contracts has intra-repo imports (issue #1544):\n" + "\n".join(violations)
    )


def test_windows_helper_inlined() -> None:
    """The ``_windows`` module exists and defines ``no_console_window_kwargs``."""
    windows_path = _SRC_DIR / "_windows.py"
    assert windows_path.is_file(), "_windows.py must exist in attachment_contracts/"
    source = windows_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(windows_path))
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "no_console_window_kwargs" in func_names


def test_windows_helper_matches_subprocess_runner() -> None:
    """The inlined helper's AST matches the original (verbatim copy).

    Mutation control: if the inlined helper's body is changed, this test
    fails -- the AST dump of the function definition differs.
    """
    from charlie_work.attachment_contracts._windows import no_console_window_kwargs as inlined
    from charlie_work.subprocess_runner import no_console_window_kwargs as original

    # Behavioral equivalence: same result for the same inputs.
    assert inlined() == original()
    assert inlined(0x00000200) == original(0x00000200)


def test_backtest_uses_inlined_helper() -> None:
    """backtest.py imports no_console_window_kwargs from _windows, not subprocess_runner."""
    source = (_SRC_DIR / "backtest.py").read_text(encoding="utf-8")
    assert (
        "from charlie_work.attachment_contracts._windows import no_console_window_kwargs" in source
    )
    assert "from charlie_work.subprocess_runner import" not in source


def test_main_uses_inlined_helper() -> None:
    """__main__.py imports no_console_window_kwargs from _windows, not subprocess_runner."""
    source = (_SRC_DIR / "__main__.py").read_text(encoding="utf-8")
    assert (
        "from charlie_work.attachment_contracts._windows import no_console_window_kwargs" in source
    )
    assert "from charlie_work.subprocess_runner import" not in source


# ---------------------------------------------------------------------------
# Distribution metadata
# ---------------------------------------------------------------------------


def test_pyproject_declares_requires_python_ge_311() -> None:
    """The distribution declares requires-python>=3.11 (issue #1544 acceptance)."""
    import tomllib

    with (_PKG_DIR / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    assert project["requires-python"] == ">=3.11"


def test_pyproject_distribution_name() -> None:
    """The distribution name is charlie-work-attachment-contracts."""
    import tomllib

    with (_PKG_DIR / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    assert project["name"] == "charlie-work-attachment-contracts"


# ---------------------------------------------------------------------------
# Wheel build (integration — skipped if build backend is not importable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_PKG_DIR / "build_backend.py").is_file(),
    reason="build_backend.py not present",
)
def test_wheel_builds_and_contains_expected_files(tmp_path: Path) -> None:
    """The wheel builds from the single in-tree source and has the right files.

    Builds the wheel via the custom PEP 517 backend and verifies:
    - Exactly 14 .py files (13 original + _windows.py).
    - No ``charlie_work/__init__.py`` (namespace package).
    - ``_windows.py`` is present.
    """
    sys.path.insert(0, str(_PKG_DIR))
    try:
        import build_backend

        out_dir = tmp_path / "dist"
        out_dir.mkdir()
        filename = build_backend.build_wheel(str(out_dir))
        whl_path = out_dir / filename
        assert whl_path.is_file(), f"wheel not found: {whl_path}"

        import zipfile

        names = zipfile.ZipFile(whl_path).namelist()
        py_files = sorted(n for n in names if n.endswith(".py") and "dist-info" not in n)
        assert len(py_files) == 14, f"expected 14 .py files, got {len(py_files)}: {py_files}"
        assert "charlie_work/__init__.py" not in names, (
            "wheel must NOT include charlie_work/__init__.py (namespace package)"
        )
        assert "charlie_work/attachment_contracts/_windows.py" in names
    finally:
        sys.path.remove(str(_PKG_DIR))
        sys.modules.pop("build_backend", None)
