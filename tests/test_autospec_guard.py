"""Enforcement test: monkeypatch doubles must use the autospec helper.

Issue #988: ``autospec_patch`` (the ``autospec`` fixture in ``tests/conftest.py``)
makes a test double conform to the real method's signature, but it is opt-in.
A structural guard catches the drift-prone hand-rolled form

    monkeypatch.setattr(<target>, "<name>", <plain function / lambda>)

and fails the test that introduces it, rather than letting a narrow double
surface as a failure in some later unrelated test.

The guard is intentionally scoped, not a blanket ratchet over the ~434 existing
hand-rolled sites.  The scope is a growing allowlist of test files; within an
allowed file, only test functions that have already opted into the ``autospec``
fixture are enforced.  Both decisions are derived at runtime from the source, so
no line-number allowlist can rot.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Initial file allowlist.  A file is checked only once it is added here; within
# the file, only tests that request the ``autospec`` fixture are enforced, which
# identifies the converted region without hardcoding line numbers or test names.
#
# test_autospec_patch.py is the helper's own tests.  test_charlie_work.py's
# fleet-status tests were converted to ``autospec`` in #949 and are the first
# live region the guard watches.
ENFORCED_FILES: frozenset[str] = frozenset(
    {"tests/test_autospec_patch.py", "tests/test_charlie_work.py"}
)


@dataclass(frozen=True)
class AutospecViolation:
    """One hand-rolled ``monkeypatch.setattr`` that should use ``autospec``."""

    filename: str
    lineno: int
    funcname: str
    message: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}: in {self.funcname!r}: {self.message}"


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _nearest_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _collect_test_nodes(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> set[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every pytest test function or method in ``tree``."""
    test_nodes: set[ast.FunctionDef | ast.AsyncFunctionDef] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        parent = parents.get(node)
        if isinstance(parent, (ast.Module, ast.ClassDef)):
            test_nodes.add(node)
    return test_nodes


def _has_autospec_fixture(test_fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when the test requests the ``autospec`` fixture."""
    all_args = (
        *test_fn.args.posonlyargs,
        *test_fn.args.args,
        *test_fn.args.kwonlyargs,
    )
    return any(arg.arg == "autospec" for arg in all_args)


def _monkeypatch_fixture_names(test_fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names of the ``monkeypatch`` fixture in this test function."""
    names: set[str] = set()
    all_args = (
        *test_fn.args.posonlyargs,
        *test_fn.args.args,
        *test_fn.args.kwonlyargs,
    )
    for arg in all_args:
        if _is_monkeypatch_arg(arg):
            names.add(arg.arg)
    return names


def _is_monkeypatch_arg(arg: ast.arg) -> bool:
    if arg.arg in {"monkeypatch", "mp"}:
        return True
    ann = arg.annotation
    if ann is None:
        return False
    if isinstance(ann, ast.Name) and ann.id == "MonkeyPatch":
        return True
    if isinstance(ann, ast.Attribute) and ann.attr == "MonkeyPatch":
        return True
    return False


def _local_double_names(
    test_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    test_nodes: set[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    """Names bound to a plain function or lambda inside ``test_fn``.

    Module-level helpers and already-autospecced mocks are deliberately left
    alone; this only identifies doubles defined inline within the test.
    """
    names: set[str] = set()

    for node in ast.walk(test_fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not test_fn:
            if _enclosing_test(node, parents, test_nodes) is test_fn:
                names.add(node.name)

    for node in ast.walk(test_fn):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not isinstance(node.value, ast.Lambda):
            continue
        if _enclosing_test(node, parents, test_nodes) is not test_fn:
            continue
        if _nearest_function(node, parents) is not test_fn:
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)

    return names


def _enclosing_test(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    test_nodes: set[ast.FunctionDef | ast.AsyncFunctionDef],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if current in test_nodes:
            return current
        current = parents.get(current)
    return None


def _is_hand_rolled_double(value: ast.expr, local_double_names: set[str]) -> bool:
    if isinstance(value, ast.Lambda):
        return True
    if isinstance(value, ast.Name) and value.id in local_double_names:
        return True
    return False


def _is_monkeypatch_setattr_call(call: ast.Call, fixture_names: set[str]) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "setattr":
        return False
    if not isinstance(call.func.value, ast.Name):
        return False
    return call.func.value.id in fixture_names


def _value_node(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "value":
            return kw.value
    if len(call.args) >= 3:
        return call.args[2]
    if len(call.args) == 2:
        return call.args[1]
    return None


def find_autospec_violations(
    source: str,
    filename: str,
    *,
    enforced_files: frozenset[str] | None = ENFORCED_FILES,
) -> list[AutospecViolation]:
    """Scan ``source`` for hand-rolled ``monkeypatch.setattr`` doubles.

    ``enforced_files`` limits which files are checked.  ``None`` checks every
    test function (used for positive/negative controls).  Within an allowed
    file, only test functions that request the ``autospec`` fixture are
    enforced, which identifies the converted region without a line allowlist.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [AutospecViolation(filename, exc.lineno or 0, "<parse>", str(exc))]

    parents = _build_parent_map(tree)
    test_nodes = _collect_test_nodes(tree, parents)

    violations: list[AutospecViolation] = []

    for test_fn in test_nodes:
        if enforced_files is not None:
            if filename not in enforced_files:
                continue
            if not _has_autospec_fixture(test_fn):
                continue

        fixture_names = _monkeypatch_fixture_names(test_fn)
        if not fixture_names:
            continue

        local_double_names = _local_double_names(test_fn, parents, test_nodes)

        for node in ast.walk(test_fn):
            if not isinstance(node, ast.Call):
                continue
            if _nearest_function(node, parents) is not test_fn:
                continue
            if not _is_monkeypatch_setattr_call(node, fixture_names):
                continue
            value = _value_node(node)
            if value is None:
                continue
            if _is_hand_rolled_double(value, local_double_names):
                violations.append(
                    AutospecViolation(
                        filename,
                        node.lineno,
                        test_fn.name,
                        "hand-rolled monkeypatch double (use the ``autospec`` helper)",
                    )
                )

    return violations


def _format_violations(violations: list[AutospecViolation]) -> str:
    lines = "\n".join(f"  {violation}" for violation in violations)
    return f"{len(violations)} hand-rolled double(s):\n{lines}"


def _scan_enforced_test_files() -> list[AutospecViolation]:
    violations: list[AutospecViolation] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        violations.extend(find_autospec_violations(source, rel))
    return violations


# -----------------------------------------------------------------------------
# Real source-tree enforcement
# -----------------------------------------------------------------------------


def test_no_hand_rolled_doubles_in_enforced_regions() -> None:
    """No allowed-file ``autospec`` test may introduce a hand-rolled double."""
    violations = _scan_enforced_test_files()
    assert not violations, _format_violations(violations)


def test_converted_region_uses_autospec_fixture() -> None:
    """The fleet-status tests in test_charlie_work.py request ``autospec``.

    This is the anti-vacuity anchor for the file-level scope: the guard is not
    an empty allowlist, and the converted region is identified by the fixture
    request, not by a hand-maintained list of test names.
    """
    source = (TESTS_DIR / "test_charlie_work.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="tests/test_charlie_work.py")
    parents = _build_parent_map(tree)
    test_nodes = _collect_test_nodes(tree, parents)
    fleet_status_tests = [n for n in test_nodes if n.name.startswith("test_fleet_status_")]
    assert fleet_status_tests, "no fleet-status test functions found"
    missing = [n.name for n in fleet_status_tests if not _has_autospec_fixture(n)]
    assert not missing, f"fleet-status tests missing autospec fixture: {missing}"


def test_guard_only_enforces_autospec_opted_in_tests() -> None:
    """Within an allowed file, only tests that request ``autospec`` are checked."""
    source = (
        "def test_converted(monkeypatch, autospec):\n"
        '    monkeypatch.setattr("some.module", "method", lambda x: x)\n'
        "def test_unconverted(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", lambda x: x)\n'
    )
    violations = find_autospec_violations(
        source, "probe.py", enforced_files=frozenset({"probe.py"})
    )
    assert len(violations) == 1
    assert violations[0].funcname == "test_converted"


# -----------------------------------------------------------------------------
# Checker teeth: synthetic positive and negative controls
# -----------------------------------------------------------------------------
#
# A guard that asserts an absence is worthless if the detector has silently
# stopped matching.  These exercise ``find_autospec_violations`` against small
# in-memory samples, independent of the real source tree, so the enforcement
# above is proven to have teeth even if every current test happened to be clean.


MUST_FLAG: dict[str, str] = {
    "lambda_three_arg": (
        "def test_foo(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", lambda x: x)\n'
    ),
    "lambda_two_arg": (
        'def test_foo(monkeypatch):\n    monkeypatch.setattr("some.module.method", lambda x: x)\n'
    ),
    "local_def_double": (
        "def test_foo(monkeypatch):\n"
        "    def _double(self, x):\n"
        "        return x\n"
        '    monkeypatch.setattr("some.module", "method", _double)\n'
    ),
    "local_lambda_binding": (
        "def test_foo(monkeypatch):\n"
        "    _double = lambda self, x: x\n"
        '    monkeypatch.setattr("some.module", "method", _double)\n'
    ),
    "value_keyword": (
        "def test_foo(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", value=lambda x: x)\n'
    ),
    "lambda_four_arg": (
        "def test_foo(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", lambda x: x, True)\n'
    ),
}

MUST_ALLOW: dict[str, str] = {
    "autospec_returned_mock": (
        "def test_foo(monkeypatch, autospec):\n"
        "    from some.module import Class\n"
        '    mock = autospec(monkeypatch, Class, "method", return_value=[])\n'
        '    monkeypatch.setattr(Class, "method", mock)\n'
    ),
    "magic_mock": (
        "from unittest.mock import MagicMock\n"
        "def test_foo(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", MagicMock())\n'
    ),
    "constant_value": (
        'def test_foo(monkeypatch):\n    monkeypatch.setattr("some.module", "method", 42)\n'
    ),
    "module_level_helper": (
        "def _module_double(x):\n"
        "    return x\n"
        "def test_foo(monkeypatch):\n"
        '    monkeypatch.setattr("some.module", "method", _module_double)\n'
    ),
    "non_callable_name": (
        "def test_foo(monkeypatch):\n"
        "    fake_stdout = object()\n"
        '    monkeypatch.setattr("sys", "stdout", fake_stdout)\n'
    ),
    "setenv_not_setattr": ('def test_foo(monkeypatch):\n    monkeypatch.setenv("KEY", "value")\n'),
    "constant_four_arg": (
        'def test_foo(monkeypatch):\n    monkeypatch.setattr("some.module", "method", 42, True)\n'
    ),
}


@pytest.mark.parametrize("name", sorted(MUST_FLAG))
def test_guard_flags_hand_rolled_double(name: str) -> None:
    """Each known-bad shape must be reported."""
    violations = find_autospec_violations(MUST_FLAG[name], name, enforced_files=None)
    assert violations, f"guard did not flag the {name!r} shape"


@pytest.mark.parametrize("name", sorted(MUST_ALLOW))
def test_guard_allows_legitimate_shape(name: str) -> None:
    """Each legitimate shape must stay quiet."""
    violations = find_autospec_violations(MUST_ALLOW[name], name, enforced_files=None)
    assert not violations, _format_violations(violations)
