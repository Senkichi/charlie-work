"""Issue #1284: no test module may import from another test module.

PR 1 of N for #1284 hoisted every externally cross-imported fixture, helper,
and fake class out of the test_*.py file that used to define it and into a
dedicated ``tests/_*.py`` module (``_fakes_github.py``, ``_review_fixtures.py``,
``_reconcile_fixtures.py``, etc. -- following the bare-name-import convention
already established by ``_api_budget_fixtures.py``, ``_sessions_db_fixtures.py``,
and friends). The point of that hoist is to let each ``test_*.py`` file be
split independently in later PRs without dragging a hidden dependency on a
sibling test file along with it.

That invariant is only real if nothing can silently reintroduce a
``test_X -> test_Y`` import later. This test makes it mechanical: every
``tests/*.py`` file is parsed and its **full** AST is walked (``ast.walk``,
not just the module-level statement list) so that a function-local import --
the kind several call sites in this repo used before the hoist, e.g. the
``from test_reconcile import (...)`` blocks that used to live inside test
function bodies in ``test_closing_reference.py`` -- is caught exactly the same
as a module-level one.

Only ``test_*`` targets are flagged. Imports of the hoisted ``tests/_*.py``
fixture modules (bare name, no leading ``test_``) are the sanctioned pattern
and are not test modules themselves, so they never match.

Known limitation, stated rather than papered over: this only sees imports
that are literal ``ast.Import``/``ast.ImportFrom`` nodes with a statically
known module name. A dynamically constructed import (``importlib.import_module(
f"test_{x}")``) would not be caught. No cross-test-module dependency in this
repo is expressed that way today; the dynamic loader this suite does use
(``_script_loader.load_script_module``) takes a file path, not a module name,
so it is exempt from this concern entirely, not a gap in it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


@dataclass(frozen=True)
class CrossTestImportViolation:
    """One `from test_X import ...` / `import test_X` statement under tests/."""

    file: Path
    line: int
    statement: str

    def __str__(self) -> str:
        rel = self.file.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.line}: {self.statement}"


def _is_test_module_name(name: str | None) -> bool:
    """True for a dotted or bare module name that names a `test_*` module.

    Only the *first* dotted component matters: `import test_foo.bar` and
    `import test_foo` both name the test module `test_foo`. A name of
    `None` (a bare `from . import x`-style relative import with no module)
    never matches.
    """
    if not name:
        return False
    head = name.split(".", 1)[0]
    return head.startswith("test_")


def find_cross_test_import_violations(
    source: str, file_path: Path
) -> list[CrossTestImportViolation]:
    """Walk the full AST of `source` and report every cross-test-module import.

    Uses `ast.walk`, which descends into every nested scope (function bodies,
    class bodies, nested functions), so a `from test_X import y` written
    inside a test function -- not just at module top level -- is still
    caught.
    """
    tree = ast.parse(source, filename=str(file_path))
    violations: list[CrossTestImportViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _is_test_module_name(node.module):
                names = ", ".join(alias.name for alias in node.names)
                statement = f"from {node.module} import {names}"
                violations.append(CrossTestImportViolation(file_path, node.lineno, statement))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_test_module_name(alias.name):
                    statement = f"import {alias.name}"
                    violations.append(CrossTestImportViolation(file_path, node.lineno, statement))
    return violations


def _all_test_dir_files() -> list[Path]:
    files = sorted(TESTS_DIR.glob("*.py"))
    assert files, f"no .py files found under {TESTS_DIR} -- the guard protects nothing"
    return files


def test_files_scanned_non_empty_and_real() -> None:
    """The scan target list must be non-empty and every entry a real file.

    Without this, the scope could silently shrink to nothing (guard
    vacuously passes) while still reporting "PASSED".
    """
    files = _all_test_dir_files()
    missing = [f for f in files if not f.is_file()]
    assert not missing, f"scanned entries do not exist: {missing}"
    assert len(files) >= 100, (
        f"expected tests/ to contain well over 100 .py files, found {len(files)} -- "
        "the glob may be scoped wrong"
    )


def test_no_cross_test_module_imports_anywhere_under_tests() -> None:
    """Zero `from test_*` / `import test_*` statements anywhere under tests/.

    Covers module-level *and* function-local imports (`ast.walk`), across
    every file in the directory -- not just the files this PR's hoist
    touched, so a future test file that reintroduces a cross-import fails
    this test regardless of which file it is.
    """
    all_violations: list[CrossTestImportViolation] = []
    for file_path in _all_test_dir_files():
        source = file_path.read_text(encoding="utf-8")
        all_violations.extend(find_cross_test_import_violations(source, file_path))

    if all_violations:
        details = "\n".join(f"  - {v}" for v in all_violations)
        raise AssertionError(
            f"found {len(all_violations)} cross-test-module import(s) under tests/ "
            f"(every test_*.py file must be self-contained; shared fixtures belong "
            f"in a tests/_*.py module instead):\n{details}"
        )


def test_guard_catches_module_level_cross_import() -> None:
    """Self-test: a module-level `from test_X import y` is caught."""
    source = "from test_charlie_work import FakeGitHub\n"
    violations = find_cross_test_import_violations(source, Path("tests/probe.py"))
    assert len(violations) == 1
    assert violations[0].line == 1
    assert "test_charlie_work" in violations[0].statement


def test_guard_catches_function_local_cross_import() -> None:
    """Self-test: an import nested inside a test function body is caught.

    This is the exact shape the repo used before the #1284 hoist (e.g. the
    three `from test_reconcile import (...)` blocks that used to live
    inside test function bodies in test_closing_reference.py) -- a
    module-top-level-only scan would have missed it.
    """
    source = (
        "def test_something():\n"
        "    from test_reconcile import FakeGitHub\n"
        "    assert FakeGitHub is not None\n"
    )
    violations = find_cross_test_import_violations(source, Path("tests/probe.py"))
    assert len(violations) == 1
    assert violations[0].line == 2
    assert "test_reconcile" in violations[0].statement


def test_guard_catches_bare_import_statement() -> None:
    """Self-test: a bare `import test_X` (not `from ... import ...`) is caught."""
    source = "import test_cli\n"
    violations = find_cross_test_import_violations(source, Path("tests/probe.py"))
    assert len(violations) == 1
    assert violations[0].statement == "import test_cli"


def test_guard_allows_hoisted_fixture_module_imports() -> None:
    """Sanctioned pattern: importing a hoisted `tests/_*.py` module is fine.

    `_fakes_github`, `_review_fixtures`, `_sessions_db_fixtures`, etc. do not
    start with `test_`, so they must never be flagged -- that would make the
    guard fire on the very pattern #1284 introduced to fix the problem.
    """
    source = (
        "from _fakes_github import FakeGitHub\n"
        "from _sessions_db_fixtures import make_sessions_db\n"
        "import _script_loader\n"
    )
    violations = find_cross_test_import_violations(source, Path("tests/probe.py"))
    assert violations == []
