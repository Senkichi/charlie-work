"""Issue #1670: caplog must never be scoped to the ``charlie_work.workflow`` logger.

Phase B of the Track 2 delegation campaign moves ``OrchestratorApp`` method
bodies out of ``workflow.py`` into ``orchestration/*.py`` delegates. A moved
body that logs through ``logging.getLogger(__name__)`` changes the emitting
logger name from ``charlie_work.workflow`` to
``charlie_work.orchestration.<module>`` at the moment of the move -- while a
``caplog.at_level(..., logger="charlie_work.workflow")`` scope keeps passing
silently, because caplog's handler sits on the root logger and the moved
logger still propagates to it. The scope argument becomes inert rather than
wrong: the record is captured via propagation, not via the named logger, so
the test only fails when ``charlie_work`` itself is configured above the
asserted level -- and until then it misleads the reader about where the
record comes from.

Two sites went stale exactly this way (``citation drift comment post
failed`` moved to ``misc_citation`` in PR #1669; ``worker census failed``
moved to ``reap_dispatch`` in PR #1677), and remaining Phase B leaves move
more ``getLogger(__name__)`` bodies -- each delegate move can strand the
next ``logger="charlie_work.workflow"`` scope. Scoping to the package
logger -- ``logger="charlie_work"`` -- is stable across every such move and
still excludes third-party noise, so it is the only permitted scope.

This guard makes that mechanical: every ``*.py`` under ``tests/`` is parsed
and every ``logger="charlie_work.workflow"`` call argument is flagged --
the ``logger=`` keyword on any call, plus the positional ``logger`` slot of
``caplog.at_level``/``caplog.set_level``. AST rather than a substring scan,
so mentioning the literal in a docstring (as this file does) or passing it
to ``logging.getLogger`` -- the marker-logger pattern in
``test_logging_setup.py``, which is unaffected -- does not fire the guard;
only a real ``logger=`` call argument does.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

FORBIDDEN_LOGGER = "charlie_work.workflow"
# caplog methods whose second positional parameter is the logger name.
_LEVEL_METHODS = frozenset({"at_level", "set_level"})


@dataclass(frozen=True)
class WorkflowLoggerScopeViolation:
    """One ``logger="charlie_work.workflow"`` argument under ``tests/``."""

    file: Path
    line: int
    statement: str

    def __str__(self) -> str:
        rel = self.file.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.line}: {self.statement}"


def _is_forbidden(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == FORBIDDEN_LOGGER


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def find_workflow_logger_scope_violations(
    source: str, file_path: Path
) -> list[WorkflowLoggerScopeViolation]:
    """Walk the full AST of `source` and flag every forbidden logger scope.

    Two shapes are caught: a ``logger=`` keyword argument on ANY call
    (caplog's ``at_level``/``set_level`` are the known consumers, but the
    staleness hazard is the argument value itself, not the receiver), and
    the positional ``logger`` slot of an ``at_level``/``set_level`` call.
    ``ast.walk`` descends into every nested scope, so a scope written inside
    a helper or fixture is caught the same as one in a test body.
    """
    tree = ast.parse(source, filename=str(file_path))
    violations: list[WorkflowLoggerScopeViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "logger" and _is_forbidden(kw.value):
                violations.append(
                    WorkflowLoggerScopeViolation(
                        file_path, node.lineno, f'logger="{FORBIDDEN_LOGGER}"'
                    )
                )
        if (
            _call_name(node.func) in _LEVEL_METHODS
            and len(node.args) >= 2
            and _is_forbidden(node.args[1])
        ):
            violations.append(
                WorkflowLoggerScopeViolation(
                    file_path,
                    node.lineno,
                    f"{_call_name(node.func)}(_, {FORBIDDEN_LOGGER!r})",
                )
            )
    return violations


def _all_test_dir_files() -> list[Path]:
    files = sorted(TESTS_DIR.rglob("*.py"))
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


def test_no_caplog_scope_pins_workflow_logger_anywhere_under_tests() -> None:
    """Zero ``logger="charlie_work.workflow"`` arguments under tests/.

    Scope ``caplog`` to the package logger (``logger="charlie_work"``) or
    drop the ``logger=`` argument entirely; both survive delegate moves.
    """
    all_violations: list[WorkflowLoggerScopeViolation] = []
    for file_path in _all_test_dir_files():
        source = file_path.read_text(encoding="utf-8")
        all_violations.extend(find_workflow_logger_scope_violations(source, file_path))

    if all_violations:
        details = "\n".join(f"  - {v}" for v in all_violations)
        raise AssertionError(
            f"found {len(all_violations)} caplog scope(s) pinned to "
            f'"{FORBIDDEN_LOGGER}" under tests/ -- Phase B delegate moves '
            'change the emitting logger to "charlie_work.orchestration.<module>", '
            'leaving the scope inert; scope to logger="charlie_work" instead '
            f"(issue #1670):\n{details}"
        )


def test_guard_catches_keyword_scope() -> None:
    """Self-test: the canonical ``caplog.at_level`` keyword form is caught."""
    source = (
        'with caplog.at_level(logging.WARNING, logger="charlie_work.workflow"):\n'
        "    app.dispatch(limit=1)\n"
    )
    violations = find_workflow_logger_scope_violations(source, Path("tests/probe.py"))
    assert len(violations) == 1
    assert violations[0].line == 1


def test_guard_catches_scope_inside_helper() -> None:
    """Self-test: a scope nested inside a helper/fixture body is caught --
    a module-top-level-only scan would miss it."""
    source = (
        "def _dispatch(app, caplog):\n"
        "    with caplog.at_level(logging.WARNING, logger='charlie_work.workflow'):\n"
        "        return app.dispatch(limit=1)\n"
    )
    violations = find_workflow_logger_scope_violations(source, Path("tests/probe.py"))
    assert len(violations) == 1
    assert violations[0].line == 2


def test_guard_catches_positional_and_set_level() -> None:
    """Self-test: the positional ``logger`` slot and ``set_level`` are caught."""
    source = (
        "caplog.at_level(logging.WARNING, 'charlie_work.workflow')\n"
        "caplog.set_level(logging.WARNING, 'charlie_work.workflow')\n"
    )
    violations = find_workflow_logger_scope_violations(source, Path("tests/probe.py"))
    assert len(violations) == 2


def test_guard_allows_package_logger_and_marker_logger() -> None:
    """Sanctioned patterns must never be flagged.

    ``logger="charlie_work"`` is the fix this issue prescribes, and
    ``logging.getLogger("charlie_work.workflow")`` is the marker-logger
    pattern ``test_logging_setup.py`` uses -- a ``getLogger`` call takes the
    name positionally, not via ``logger=``, so it is not a scope.
    """
    source = (
        "with caplog.at_level(logging.WARNING, logger='charlie_work'):\n"
        "    app.dispatch(limit=1)\n"
        "with caplog.at_level(logging.WARNING):\n"
        "    app.dispatch(limit=1)\n"
        "logging.getLogger('charlie_work.workflow').info(marker)\n"
    )
    violations = find_workflow_logger_scope_violations(source, Path("tests/probe.py"))
    assert violations == []
