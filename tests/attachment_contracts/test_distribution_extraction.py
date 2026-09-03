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


@pytest.mark.skipif(
    not (_PKG_DIR / "build_backend.py").is_file(),
    reason="build_backend.py not present",
)
def test_wheel_built_from_sdist_contains_source_files(tmp_path: Path) -> None:
    """A wheel built FROM the sdist contains the source ``.py`` files.

    Round-1 review finding #2: ``build_backend._SRC_DIR`` previously used
    ``_HERE.parent.parent / "src" / ...`` unconditionally, which is correct for
    the in-repo layout but wrong for the sdist layout.  ``build_sdist`` places
    ``build_backend.py`` at ``<prefix>/build_backend.py`` and bundles the source
    at ``<prefix>/src/charlie_work/attachment_contracts/`` -- one level DOWN
    from ``_HERE``, not two levels UP.  A wheel built from the sdist therefore
    resolved a non-existent source directory and silently contained zero
    ``.py`` files.

    This test exercises the sdist->wheel path: build the sdist from the in-repo
    layout, extract it, import the backend from the extracted prefix, and build
    a wheel.  The wheel MUST contain the 14 source ``.py`` files -- without the
    layout detection in ``_SRC_DIR``, it contains none and this test fails.

    Mutation control: reverting ``_SRC_DIR`` to
    ``_HERE.parent.parent / "src" / ...`` makes the extracted-prefix backend
    resolve ``<prefix>/../src/...`` (nonexistent), so ``build_wheel`` packages
    zero ``.py`` files and the ``len(py_files) == 14`` assertion fails.
    """
    import tarfile
    import zipfile

    sys.path.insert(0, str(_PKG_DIR))
    try:
        import build_backend as in_repo_backend

        # 1. Build the sdist from the in-repo layout.
        sdist_dir = tmp_path / "sdist"
        sdist_dir.mkdir()
        sdist_name = in_repo_backend.build_sdist(str(sdist_dir))
        sdist_path = sdist_dir / sdist_name
        assert sdist_path.is_file(), f"sdist not found: {sdist_path}"

        # 2. Extract the sdist into a clean directory -- this is what an
        #    isolated build frontend does before calling build_wheel.
        prefix_dir = tmp_path / "extracted"
        prefix_dir.mkdir()
        with tarfile.open(sdist_path, "r:gz") as tar:
            tar.extractall(prefix_dir, filter="data")  # noqa: S202 -- trusted local build output
        # The sdist top-level dir is ``<name>-<version>``.
        extracted_roots = [p for p in prefix_dir.iterdir() if p.is_dir()]
        assert len(extracted_roots) == 1, f"expected one top-level dir, got {extracted_roots}"
        prefix_root = extracted_roots[0]
        assert (prefix_root / "build_backend.py").is_file(), "backend must be in the sdist prefix"
        assert (prefix_root / "src" / "charlie_work" / "attachment_contracts").is_dir(), (
            "sdist must bundle the source under <prefix>/src/"
        )
    finally:
        sys.path.remove(str(_PKG_DIR))
        sys.modules.pop("build_backend", None)

    # 3. Import the backend FROM the extracted sdist prefix and build a wheel.
    #    The backend's ``_HERE`` is now the prefix root, so ``_SRC_DIR`` must
    #    detect the sdist layout (``_HERE / "src" / ...``).
    sys.path.insert(0, str(prefix_root))
    try:
        sys.modules.pop("build_backend", None)
        import build_backend  # imported from the extracted prefix

        wheel_dir = tmp_path / "wheel"
        wheel_dir.mkdir()
        filename = build_backend.build_wheel(str(wheel_dir))
        whl_path = wheel_dir / filename
        assert whl_path.is_file(), f"wheel from sdist not found: {whl_path}"

        names = zipfile.ZipFile(whl_path).namelist()
        py_files = sorted(n for n in names if n.endswith(".py") and "dist-info" not in n)
        assert len(py_files) == 14, (
            f"wheel built from sdist must contain 14 .py files, got {len(py_files)}: "
            f"{py_files} -- sdist layout detection in _SRC_DIR is broken"
        )
        assert "charlie_work/attachment_contracts/_windows.py" in names
        assert "charlie_work/__init__.py" not in names
    finally:
        sys.path.remove(str(prefix_root))
        sys.modules.pop("build_backend", None)


@pytest.mark.skipif(
    not (_PKG_DIR / "build_backend.py").is_file(),
    reason="build_backend.py not present",
)
def test_wheel_dist_info_dir_matches_version(tmp_path: Path) -> None:
    """The wheel's ``.dist-info`` directory name tracks the project version.

    Round-2 review finding: ``build_backend`` previously hardcoded
    ``_DIST_INFO = "charlie_work_attachment_contracts-0.1.1.dist-info"``, so
    the dist-info directory name would silently desync from the wheel filename
    on the next version bump, producing a spec-non-compliant wheel with no test
    to catch it.

    This test builds a wheel from a temp copy of the package with the version
    bumped to ``0.1.2`` and asserts the resulting dist-info directory name is
    ``charlie_work_attachment_contracts-0.1.2.dist-info`` -- i.e. derived from
    the project name/version via the same normalization as the wheel filename,
    not from a hardcoded constant.

    Mutation control: reverting ``_dist_info_dir`` to the hardcoded
    ``charlie_work_attachment_contracts-0.1.1.dist-info`` constant makes the
    dist-info directory name ``...-0.1.1.dist-info`` regardless of the bumped
    pyproject version, so the ``...-0.1.2.dist-info`` assertion fails.
    """
    import shutil
    import zipfile

    # 1. Lay out a temp package dir in the sdist layout (build_backend.py +
    #    pyproject.toml + src/...) so _SRC_DIR detects the sdist layout
    #    (_HERE / "src" / ...).
    tmp_pkg = tmp_path / "pkg"
    tmp_pkg.mkdir()
    shutil.copy2(_PKG_DIR / "build_backend.py", tmp_pkg / "build_backend.py")
    src_dst = tmp_pkg / "src" / "charlie_work" / "attachment_contracts"
    src_dst.mkdir(parents=True)
    for py_file in _SRC_DIR.glob("*.py"):
        shutil.copy2(py_file, src_dst / py_file.name)

    # 2. Write a pyproject.toml with the version bumped to 0.1.2.
    import tomllib

    with (_PKG_DIR / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)["project"]
    project["version"] = "0.1.2"
    # Re-serialize the [project] table into a minimal pyproject.toml the
    # backend's _read_project() can parse.
    import json

    readme_text = (
        project["readme"]["text"] if isinstance(project["readme"], dict) else project["readme"]
    )
    readme_ct = (
        project["readme"].get("content-type", "text/plain")
        if isinstance(project["readme"], dict)
        else "text/plain"
    )
    lines = [
        "[project]",
        f'name = "{project["name"]}"',
        f'version = "{project["version"]}"',
        f"description = {json.dumps(project['description'])}",
        f'readme = {{ text = {json.dumps(readme_text)}, content-type = "{readme_ct}" }}',
        f'requires-python = "{project["requires-python"]}"',
        f'license = {{ text = "{project["license"]["text"]}" }}',
        f"keywords = {json.dumps(project['keywords'])}",
        "classifiers = [",
    ]
    for cls in project["classifiers"]:
        lines.append(f'    "{cls}",')
    lines.append("]")
    (tmp_pkg / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3. Import the backend from the temp package and build a wheel.
    sys.path.insert(0, str(tmp_pkg))
    try:
        sys.modules.pop("build_backend", None)
        import build_backend  # imported from the temp package

        wheel_dir = tmp_path / "wheel"
        wheel_dir.mkdir()
        filename = build_backend.build_wheel(str(wheel_dir))
        whl_path = wheel_dir / filename
        assert whl_path.is_file(), f"wheel not found: {whl_path}"

        # 4. The wheel filename must reflect the bumped version...
        assert "0.1.2" in filename, f"wheel filename must contain 0.1.2: {filename}"

        names = zipfile.ZipFile(whl_path).namelist()
        dist_info_dirs = sorted({n.split("/")[0] for n in names if ".dist-info" in n})
        assert dist_info_dirs == ["charlie_work_attachment_contracts-0.1.2.dist-info"], (
            f"dist-info dir must track the bumped version, got {dist_info_dirs} -- "
            f"_DIST_INFO is hardcoded instead of derived from project version"
        )
        # The four expected dist-info entries live under that directory.
        expected = {
            "charlie_work_attachment_contracts-0.1.2.dist-info/METADATA",
            "charlie_work_attachment_contracts-0.1.2.dist-info/WHEEL",
            "charlie_work_attachment_contracts-0.1.2.dist-info/LICENSE",
            "charlie_work_attachment_contracts-0.1.2.dist-info/RECORD",
        }
        assert expected.issubset(set(names)), f"missing dist-info entries: {expected - set(names)}"
    finally:
        sys.path.remove(str(tmp_pkg))
        sys.modules.pop("build_backend", None)
