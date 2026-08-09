"""Tests for the shared ``_script_loader`` helper and the structural guard that
prevents hand-rolled copies of the same recipe.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

from _script_loader import load_script_module


def test_load_script_module_loads_script(tmp_path: Path) -> None:
    """The helper executes a Python file and returns its module object."""
    script = tmp_path / "probe.py"
    script.write_text("value = 42\n", encoding="utf-8")

    module = load_script_module(script, "probe_load_script")

    assert isinstance(module, ModuleType)
    assert module.value == 42


def test_load_script_module_registers_module_before_exec_module(tmp_path: Path) -> None:
    """Regression for issue #1023.

    Scripts that mature into carrying a frozen dataclass with
    ``from __future__ import annotations`` need the module to be in
    ``sys.modules`` before ``exec_module`` runs. Without that, class creation
    dies with ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    script = tmp_path / "synthetic_dataclass_script.py"
    script.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RegressionProbe:\n"
        "    value: int\n",
        encoding="utf-8",
    )

    module = load_script_module(script, "probe_dataclass")

    probe = module.RegressionProbe(value=1)
    assert probe.value == 1


def test_load_script_module_restores_sys_modules_entry(tmp_path: Path) -> None:
    """A prior sys.modules entry is restored and a missing entry is removed."""
    script = tmp_path / "probe.py"
    script.write_text("x = 1\n", encoding="utf-8")

    prior = ModuleType("probe_reuse")
    sys.modules["probe_reuse"] = prior
    try:
        module = load_script_module(script, "probe_reuse")
        assert module.x == 1
        assert sys.modules["probe_reuse"] is prior
    finally:
        sys.modules.pop("probe_reuse", None)

    # With no prior entry, the helper should leave nothing behind.
    assert "probe_reuse2" not in sys.modules
    try:
        load_script_module(script, "probe_reuse2")
    finally:
        sys.modules.pop("probe_reuse2", None)
    assert "probe_reuse2" not in sys.modules


def test_load_script_module_sets_and_restores_argv(tmp_path: Path) -> None:
    """``argv`` is visible inside the loaded script and ``sys.argv`` is restored."""
    script = tmp_path / "probe_argv.py"
    script.write_text("import sys\ncaptured = sys.argv\n", encoding="utf-8")

    original = sys.argv
    try:
        module = load_script_module(script, "probe_argv", argv=["probe_argv", "first", "second"])
        assert module.captured == ["probe_argv", "first", "second"]
        assert sys.argv is original
    finally:
        sys.argv = original


def _collect_imported_names(source: str) -> dict[str, str]:
    """Map local names to their fully-qualified dotted form.

    Covers the import forms likely to be used for
    ``importlib.util.spec_from_file_location``:

    * ``import importlib`` -> ``{'importlib': 'importlib'}``
    * ``import importlib.util`` -> ``{'importlib.util': 'importlib.util'}``
    * ``import importlib.util as u`` -> ``{'u': 'importlib.util'}``
    * ``from importlib import util`` -> ``{'util': 'importlib.util'}``
    * ``from importlib.util import spec_from_file_location`` ->
      ``{'spec_from_file_location': 'importlib.util.spec_from_file_location'}``
    * ``from importlib.util import spec_from_file_location as sfl`` ->
      ``{'sfl': 'importlib.util.spec_from_file_location'}``
    """
    mapping: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    mapping[alias.asname] = alias.name
                else:
                    # `import a.b.c` binds the top-level name `a` in the
                    # module namespace, and attribute access follows from there.
                    mapping[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                mapping[local] = f"{node.module}.{alias.name}"
    return mapping


def _resolve_dotted(expr: ast.expr, mapping: dict[str, str]) -> str | None:
    """Return the fully-qualified name of a possibly-dotted expression."""
    if isinstance(expr, ast.Name):
        return mapping.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _resolve_dotted(expr.value, mapping)
        if base is not None:
            return f"{base}.{expr.attr}"
    return None


def test_no_hand_rolled_spec_from_file_location_in_tests() -> None:
    """Structural pin for issue #1028.

    The only place ``importlib.util.spec_from_file_location`` may be called in
    tests is ``_script_loader.py``. A text grep is not enough: docstrings in
    tests (e.g. ``tests/test_verify_events.py``) mention the function by name.
    This walks the AST and resolves the call target instead, matching the
    approach in ``tests/test_global_config.py`` after #1025.
    """
    tests_dir = Path(__file__).resolve().parent
    target = "importlib.util.spec_from_file_location"

    offenders: list[str] = []
    for source_file in sorted(tests_dir.rglob("*.py")):
        if source_file.name == "_script_loader.py":
            continue

        try:
            source_text = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            offenders.append(
                f"{source_file.relative_to(tests_dir.parent)}: could not decode as UTF-8 ({exc})"
            )
            continue

        tree = ast.parse(source_text, filename=str(source_file))
        mapping = _collect_imported_names(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                resolved = _resolve_dotted(node.func, mapping)
                if resolved == target:
                    rel_path = source_file.relative_to(tests_dir.parent)
                    offenders.append(f"{rel_path}:{node.lineno}: {resolved}")

    assert not offenders, (
        "Hand-rolled ``importlib.util.spec_from_file_location`` call sites "
        "found in tests. Use _script_loader.load_script_module instead:\n" + "\n".join(offenders)
    )
