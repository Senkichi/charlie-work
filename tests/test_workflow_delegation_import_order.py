"""Regression tests for the partial-module import-order hazard (issue #1798).

``charlie_work.workflow``'s module-level ``discover_delegate_modules()`` call
walks every ``charlie_work.orchestration`` submodule and installs each
top-level ``def`` onto ``OrchestratorApp``. If an orchestration submodule is
imported *before* ``charlie_work.workflow`` anywhere in the process, the
submodule's own ``import charlie_work.workflow as _wf`` line suspends its
execution while workflow's module body -- including discovery -- runs. The
discover call then re-imports the still-partial submodule out of
``sys.modules`` and ``vars(module)`` only sees the names bound above that
import line; every ``def`` below it used to be silently dropped from
``OrchestratorApp``, surfacing far from the cause as a confusing
``AttributeError``.

The fix makes discovery fail loud instead: ``_assert_fully_initialized``
compares the submodule's own source-declared top-level ``def`` names against
its routable members and raises ``ImportError`` when the namespace is
incomplete. These tests pin that contract at three levels: the unit check on
synthetic modules, the real import-order reproduction in a subprocess, and a
repo-wide sweep covering every ``charlie_work.orchestration`` submodule in
isolation so a *future* submodule cannot reintroduce the silent drop.
"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import charlie_work.orchestration as _orchestration
import charlie_work.workflow_delegation as wd

_ORCHESTRATION_DIR = Path(_orchestration.__file__).resolve().parent

# The diagnostic phrase the ImportError must carry for a failure to count as
# the deliberate guard rather than an unrelated import breakage.
_PARTIAL_MARKER = "partially initialized"


def _seed_partial_module(name: str, path: Path, *defs: str) -> types.ModuleType:
    """Build a partially-populated module object as Python's import machinery
    leaves it mid-import: ``__file__`` points at the real source (which
    declares more defs) but ``vars(module)`` only holds the names bound so
    far."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    for def_name in defs:

        def fn(self=None) -> None:  # noqa: ARG001
            return None

        fn.__module__ = name
        setattr(module, def_name, fn)
    return module


def _defs_below_workflow_import(source_path: Path) -> frozenset[str]:
    """Top-level ``def`` names declared *below* the module's first top-level
    ``import charlie_work.workflow`` statement (empty set when there is none).

    This is the exact set the hazard drops: names bound only after the import
    line that suspends the submodule's own execution. Derived from the live
    source, never a hand-maintained module/name list (CLAUDE.md rule 9).
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    import_lines = [
        node.lineno
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "charlie_work.workflow" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "charlie_work.workflow")
        or (
            isinstance(node, ast.ImportFrom)
            and node.level == 2
            and node.module in (None, "workflow")
        )
    ]
    if not import_lines:
        return frozenset()
    first = min(import_lines)
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno > first
    )


# ---------------------------------------------------------------------------
# _declared_top_level_defs
# ---------------------------------------------------------------------------


def test_declared_top_level_defs_only_module_level_names(tmp_path: Path) -> None:
    """Only direct-child, non-dunder ``def``s count as declared: nested,
    conditional, class-body, and dunder defs are excluded (a conditional def
    still installs when present -- it just cannot be *required*)."""
    src = tmp_path / "defs_src.py"
    src.write_text(
        "def top_a(self):\n    pass\n"
        "def top_b(self):\n    pass\n"
        "def __dunder__(self):\n    pass\n"
        "if True:\n"
        "    def in_if(self):\n        pass\n"
        "class _C:\n"
        "    def method(self):\n        pass\n"
        "def outer(self):\n"
        "    def nested(self):\n        pass\n",
        encoding="utf-8",
    )
    module = types.ModuleType("defs_src")
    module.__file__ = str(src)
    assert wd._declared_top_level_defs(module) == frozenset({"top_a", "top_b", "outer"})


def test_declared_top_level_defs_no_source_is_empty() -> None:
    """A synthetic module with no ``__file__`` (or an unreadable one) yields an
    empty declared set -- nothing derivable, nothing checked."""
    module = types.ModuleType("no_file_synthetic")
    assert wd._declared_top_level_defs(module) == frozenset()
    module.__file__ = str(Path("nonexistent_dir") / "ghost.py")
    assert wd._declared_top_level_defs(module) == frozenset()


# ---------------------------------------------------------------------------
# _assert_fully_initialized / discover on synthetic partial modules
# ---------------------------------------------------------------------------


def test_assert_fully_initialized_raises_on_partial_module(tmp_path: Path) -> None:
    """A seeded sys.modules entry missing a source-declared def is rejected by
    discovery with an ImportError naming the dropped member -- the synthetic
    analog of the real mid-import partial."""
    pkg_dir = tmp_path / "partialpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    leaf = pkg_dir / "leaf.py"
    leaf.write_text(
        "def _alpha(self):\n    return 'a'\ndef _beta(self):\n    return 'b'\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        pkg = importlib.import_module("partialpkg")
        # Seed a partial module: only _alpha bound so far, _beta still below
        # the suspended line.
        partial = _seed_partial_module("partialpkg.leaf", leaf, "_alpha")
        sys.modules["partialpkg.leaf"] = partial

        with pytest.raises(ImportError) as exc_info:
            wd.discover_delegate_modules(pkg)
        message = str(exc_info.value)
        assert "partialpkg.leaf" in message
        assert _PARTIAL_MARKER in message
        assert "_beta" in message  # the dropped delegate is named
        assert "import charlie_work.workflow" in message  # the remedy is named
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("partialpkg", None)
        sys.modules.pop("partialpkg.leaf", None)
        importlib.invalidate_caches()


def test_assert_fully_initialized_accepts_complete_module(tmp_path: Path) -> None:
    """Positive control: a seeded module whose routable members cover every
    source-declared def passes the guard and is returned by discovery."""
    pkg_dir = tmp_path / "fullpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    leaf = pkg_dir / "leaf.py"
    leaf.write_text(
        "def _alpha(self):\n    return 'a'\ndef _beta(self):\n    return 'b'\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        pkg = importlib.import_module("fullpkg")
        complete = _seed_partial_module("fullpkg.leaf", leaf, "_alpha", "_beta")
        sys.modules["fullpkg.leaf"] = complete

        discovered = wd.discover_delegate_modules(pkg)
        assert discovered == (complete,)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("fullpkg", None)
        sys.modules.pop("fullpkg.leaf", None)
        importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Real reproduction: importing an orchestration submodule first must fail loud
# ---------------------------------------------------------------------------


def test_orchestration_submodule_first_import_fails_loud_subprocess() -> None:
    """The exact #1768 reproduction: ``state_maintenance`` imported before
    ``charlie_work.workflow`` used to silently drop
    ``_maybe_emit_operator_queue_impact`` from ``OrchestratorApp``; it now
    fails at import time naming the module, the dropped defs, and the fix."""
    result = subprocess.run(
        [sys.executable, "-c", "import charlie_work.orchestration.state_maintenance"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "importing an orchestration submodule before charlie_work.workflow "
        "must fail, not silently install an incomplete delegate set"
    )
    assert _PARTIAL_MARKER in result.stderr
    assert "charlie_work.orchestration.state_maintenance" in result.stderr
    assert "_maybe_emit_operator_queue_impact" in result.stderr


def test_workflow_first_import_order_still_clean_subprocess() -> None:
    """Positive control for the supported order: workflow first, then the
    submodule, imports cleanly and the delegate is installed."""
    code = (
        "import charlie_work.workflow, charlie_work.orchestration.state_maintenance\n"
        "from charlie_work.workflow import OrchestratorApp\n"
        "assert hasattr(OrchestratorApp, '_maybe_emit_operator_queue_impact')\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_every_orchestration_submodule_import_order_guarded_subprocess() -> None:
    """Repo-wide sweep: every ``charlie_work.orchestration`` submodule imported
    in isolation before ``charlie_work.workflow`` either fails with the
    partial-module ImportError or imports cleanly -- never silently succeeds
    with delegates dropped. Each iteration purges ``charlie_work.workflow``
    and the orchestration submodules from ``sys.modules`` so the submodule is
    genuinely first, matching the hazard's precondition.

    Expectations are derived, not hard-coded: a submodule whose source places
    top-level ``def``s below its first top-level ``import
    charlie_work.workflow`` *must* raise (those names provably cannot be in
    the partial namespace); any other submodule may import cleanly, and any
    ImportError it raises must still carry the partial-module marker rather
    than some unrelated failure.
    """
    submodule_paths = sorted(p for p in _ORCHESTRATION_DIR.glob("*.py") if p.stem != "__init__")
    must_raise = {p.stem for p in submodule_paths if _defs_below_workflow_import(p)}
    # The known victim from issue #1768 must be in the must-raise set --
    # positive control that the static derivation sees the real hazard shape.
    assert "state_maintenance" in must_raise

    names = sorted(
        info.name for info in pkgutil.iter_modules(_orchestration.__path__) if not info.ispkg
    )
    assert {p.stem for p in submodule_paths} == set(names)

    script = textwrap.dedent(
        """
        import importlib, json, sys

        names = json.loads(sys.argv[1])
        results = {}
        for name in names:
            full = "charlie_work.orchestration." + name
            for mod in [
                m
                for m in list(sys.modules)
                if m == "charlie_work.workflow"
                or m.startswith("charlie_work.orchestration.")
            ]:
                del sys.modules[mod]
            try:
                importlib.import_module(full)
            except ImportError as exc:
                results[name] = ["importerror", str(exc)]
            except BaseException as exc:  # noqa: BLE001
                results[name] = ["other", f"{type(exc).__name__}: {exc}"]
            else:
                results[name] = ["ok", ""]
        print("RESULTS=" + json.dumps(results))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(names)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    results_line = next(line for line in result.stdout.splitlines() if line.startswith("RESULTS="))
    outcomes = json.loads(results_line[len("RESULTS=") :])
    assert set(outcomes) == set(names)

    raised = set()
    for name, (status, detail) in outcomes.items():
        assert status != "other", f"{name} failed with a non-ImportError: {detail}"
        if status == "importerror":
            assert _PARTIAL_MARKER in detail, (
                f"{name} raised an ImportError that is not the partial-module guard: {detail}"
            )
            raised.add(name)
    # Every submodule whose declared defs sit below its workflow import must
    # have raised -- no silent pass-through anywhere in the package.
    assert must_raise <= raised, (
        f"submodules with defs below their workflow import that imported "
        f"without raising: {sorted(must_raise - raised)}"
    )
