"""Tests for the AST-equivalence gate (issue #1541).

Covers:

* :func:`extract_symbols` -- AST symbol extraction with ``ast.dump``.
* :func:`derive_moved_symbols` -- the diff-derived moved-symbol set (graft E,
  rule #9: no hardcoded symbol-name list).
* :func:`generate_pep562_shim_source` -- PEP 562 ``__getattr__`` facade
  generation, with a test confirming old and new import paths resolve to the
  same object post-move.
* :func:`find_stale_facade_shims` -- the vulture "forgotten facade" sweep,
  with mutation control (removing the sweep step lets a stale shim through).
* :func:`render_review_packet` -- review packet rendering (evidence, not
  enforcement).
* The CLI command ``charlie ast-equivalence-check``.

Mutation controls: each regression test names the exact edit it reverts and
verifies the test fails against the unfixed code.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap
from pathlib import Path

from charlie_work.ast_equivalence_gate import (
    GateResult,
    MovedSymbol,
    StaleShim,
    derive_moved_symbols,
    extract_symbols,
    find_stale_facade_shims,
    generate_pep562_shim_source,
    parse_shim_mapping,
    render_review_packet,
)
from charlie_work.ast_equivalence_gate_command import (
    run_ast_equivalence_check_command,
)
from charlie_work.subprocess_runner import RunResult


# ---------------------------------------------------------------------------
# extract_symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_top_level_function() -> None:
    source = "def foo():\n    return 42\n"
    syms = extract_symbols(source, filename="test.py")
    assert "foo" in syms
    # The dump should be the ast.dump with include_attributes=False
    tree = ast.parse(source)
    expected = ast.dump(tree.body[0], include_attributes=False)
    assert syms["foo"] == expected


def test_extract_symbols_class_and_methods() -> None:
    source = textwrap.dedent("""
        class MyClass:
            def method_a(self):
                return 1
            def method_b(self):
                return 2
    """)
    syms = extract_symbols(source, filename="test.py")
    assert "MyClass" in syms
    assert "MyClass.method_a" in syms
    assert "MyClass.method_b" in syms
    # The class dump should differ from the method dumps
    assert syms["MyClass"] != syms["MyClass.method_a"]


def test_extract_symbols_async_function() -> None:
    source = "async def aio():\n    return 1\n"
    syms = extract_symbols(source, filename="test.py")
    assert "aio" in syms


def test_extract_symbols_empty_file() -> None:
    assert extract_symbols("", filename="empty.py") == {}


# ---------------------------------------------------------------------------
# derive_moved_symbols -- the diff-derived moved-symbol set (graft E)
# ---------------------------------------------------------------------------


def test_verbatim_move_is_detected_as_equivalent() -> None:
    """A function moved verbatim from file_a to file_b is equivalent (graft G).

    This is the self-proving test case: the gate must pass a genuinely
    verbatim relocation (identical AST, different file path).
    """
    func_source = "def moved_func():\n    return 42\n"
    base = {"src/a.py": extract_symbols(func_source, "a.py")}
    head = {"src/b.py": extract_symbols(func_source, "b.py")}
    moved = derive_moved_symbols(base, head)
    assert len(moved) == 1
    assert moved[0].name == "moved_func"
    assert moved[0].source_file == "src/a.py"
    assert moved[0].dest_file == "src/b.py"
    assert moved[0].equivalent is True


def test_non_equivalent_move_is_flagged() -> None:
    """A function moved but modified is flagged as non-equivalent."""
    base_source = "def moved_func():\n    return 42\n"
    head_source = "def moved_func():\n    return 99\n"
    base = {"src/a.py": extract_symbols(base_source, "a.py")}
    head = {"src/b.py": extract_symbols(head_source, "b.py")}
    moved = derive_moved_symbols(base, head)
    assert len(moved) == 1
    assert moved[0].equivalent is False


def test_no_move_when_symbol_stays_in_place() -> None:
    """A symbol that stays in its original file is not a move."""
    source = "def stay():\n    return 1\n"
    base = {"src/a.py": extract_symbols(source, "a.py")}
    head = {"src/a.py": extract_symbols(source, "a.py")}
    moved = derive_moved_symbols(base, head)
    assert moved == []


def test_no_move_when_symbol_added_without_removal() -> None:
    """A symbol added to a file that didn't exist at base is not a move."""
    new_source = "def new_func():\n    return 1\n"
    base: dict[str, dict[str, str]] = {}
    head = {"src/new.py": extract_symbols(new_source, "new.py")}
    moved = derive_moved_symbols(base, head)
    assert moved == []


def test_no_move_when_symbol_deleted_without_addition() -> None:
    """A symbol deleted from a file without appearing elsewhere is not a move."""
    source = "def gone():\n    return 1\n"
    base = {"src/a.py": extract_symbols(source, "a.py")}
    head: dict[str, dict[str, str]] = {"src/a.py": {}}
    moved = derive_moved_symbols(base, head)
    assert moved == []


def test_class_method_move_between_classes() -> None:
    """A method moving from class A to class B is detected by bare name."""
    base_source = textwrap.dedent("""
        class OldClass:
            def helper(self):
                return 1
    """)
    head_source = textwrap.dedent("""
        class NewClass:
            def helper(self):
                return 1
    """)
    base = {"src/old.py": extract_symbols(base_source, "old.py")}
    head = {"src/new.py": extract_symbols(head_source, "new.py")}
    moved = derive_moved_symbols(base, head)
    # The method "helper" moved from OldClass in old.py to NewClass in new.py
    method_moves = [m for m in moved if m.name == "helper"]
    assert len(method_moves) == 1
    assert method_moves[0].source_class == "OldClass"
    assert method_moves[0].dest_class == "NewClass"
    assert method_moves[0].equivalent is True


def test_multiple_symbols_moved() -> None:
    """Multiple symbols moved from one file to another are all detected."""
    base_source = textwrap.dedent("""
        def func_a():
            return 1
        def func_b():
            return 2
    """)
    head_source = textwrap.dedent("""
        def func_a():
            return 1
        def func_b():
            return 2
    """)
    base = {"src/old.py": extract_symbols(base_source, "old.py")}
    head = {"src/new.py": extract_symbols(head_source, "new.py")}
    moved = derive_moved_symbols(base, head)
    names = {m.name for m in moved}
    assert names == {"func_a", "func_b"}
    assert all(m.equivalent for m in moved)


# ---------------------------------------------------------------------------
# Rule #9 compliance: no hardcoded symbol-name list in the gate source
# ---------------------------------------------------------------------------


def test_no_hardcoded_symbol_name_list_in_gate_source() -> None:
    """Rule #9: grep of the gate's own source contains no hardcoded symbol-name list.

    The moved-symbol set must be diff-derived, never hand-typed. This test
    verifies the gate module does not contain a literal list of specific
    symbol names (e.g. ``["foo", "bar", "baz"]``) that could be a hand-
    maintained moved-symbol set. It does this by checking that no string
    literal in the source looks like a list of function/class names -- the
    gate's code should only reference symbol names as variables (``node.name``,
    ``bare_a``, etc.), never as literals.
    """
    import charlie_work.ast_equivalence_gate as gate_mod

    source = Path(gate_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=gate_mod.__file__)

    # Collect all string-literal lists (list/tuple/set of string constants).
    suspicious: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elements = node.elts
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements):
                # A list of string literals -- could be a hardcoded name list.
                # Exclude known-safe patterns: empty lists, single-element doc
                # markers, or lists that are clearly not symbol names (contain
                # spaces, dots, or are too long to be identifiers).
                vals = [e.value for e in elements]
                if len(vals) >= 2 and all(v.isidentifier() for v in vals):
                    suspicious.append(f"line {node.lineno}: {vals}")

    assert suspicious == [], (
        f"Gate source contains potential hardcoded symbol-name lists "
        f"(rule #9 violation): {suspicious}"
    )


# ---------------------------------------------------------------------------
# generate_pep562_shim_source
# ---------------------------------------------------------------------------


def test_generate_module_level_shim() -> None:
    """Module-level __getattr__ is generated for top-level moved symbols."""
    moved = [
        MovedSymbol(
            name="relocated_func",
            source_file="src/charlie_work/old_module.py",
            dest_file="src/charlie_work/new_module.py",
            source_class=None,
            dest_class=None,
            equivalent=True,
        ),
    ]
    shim = generate_pep562_shim_source("src/charlie_work/old_module.py", moved)
    assert "def __getattr__(name):" in shim
    assert "relocated_func" in shim
    assert "charlie_work.new_module" in shim
    assert "importlib" in shim


def test_generate_class_level_shim() -> None:
    """Class-level __getattr__ is generated for moved class members."""
    moved = [
        MovedSymbol(
            name="moved_method",
            source_file="src/charlie_work/old_module.py",
            dest_file="src/charlie_work/new_module.py",
            source_class="OldClass",
            dest_class="NewClass",
            equivalent=True,
        ),
    ]
    shim = generate_pep562_shim_source("src/charlie_work/old_module.py", moved)
    assert "OldClass" in shim
    assert "moved_method" in shim
    assert "charlie_work.new_module" in shim


def test_generate_shim_no_moves() -> None:
    """No moves -> the shim source has a comment but no __getattr__."""
    shim = generate_pep562_shim_source("src/charlie_work/old_module.py", [])
    assert "No symbols moved" in shim
    assert "def __getattr__" not in shim


def test_pep562_shim_resolves_old_and_new_import_paths(tmp_path: Path) -> None:
    """Both old and new import paths resolve to the same object post-move.

    This is the acceptance test: "a test confirms both the old and new import
    paths resolve to the same object post-move."
    """
    # Create a "new module" with the relocated symbol.
    pkg = tmp_path / "shim_test_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    new_module_source = textwrap.dedent("""
        VALUE = 42

        def relocated_func():
            return VALUE
    """)
    (pkg / "new_module.py").write_text(new_module_source, encoding="utf-8")

    # Simulate a move: the symbol was in old_module, now it's in new_module.
    moved = [
        MovedSymbol(
            name="relocated_func",
            source_file="shim_test_pkg/old_module.py",
            dest_file="shim_test_pkg/new_module.py",
            source_class=None,
            dest_class=None,
            equivalent=True,
        ),
    ]
    shim_source = generate_pep562_shim_source("shim_test_pkg/old_module.py", moved, src_root="")
    (pkg / "old_module.py").write_text(shim_source, encoding="utf-8")

    # Add the temp package to sys.path and import both paths.
    sys.path.insert(0, str(tmp_path))
    try:
        old_mod = importlib.import_module("shim_test_pkg.old_module")
        new_mod = importlib.import_module("shim_test_pkg.new_module")

        old_func = getattr(old_mod, "relocated_func")
        new_func = getattr(new_mod, "relocated_func")

        # Both paths resolve to the same object.
        assert old_func is new_func
        # And the function works.
        assert old_func() == 42
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name.startswith("shim_test_pkg"):
                del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# find_stale_facade_shims -- the vulture "forgotten facade" sweep
# ---------------------------------------------------------------------------


def test_stale_shim_detected_when_nobody_imports_it(tmp_path: Path) -> None:
    """A facade re-export entry nobody imports is flagged as stale.

    Mutation control: if the sweep step is removed (i.e.
    find_stale_facade_shims returns []), this test fails -- the stale shim
    goes undetected.
    """
    # Create a facade shim module with a re-export entry.
    facade_source = textwrap.dedent("""
        import importlib

        _REEXPORTS = {
            'stale_name': 'some.other.module',
            'used_name': 'some.other.module',
        }

        def __getattr__(name):
            if name in _REEXPORTS:
                mod = importlib.import_module(_REEXPORTS[name])
                return getattr(mod, name)
            raise AttributeError(name)
    """)
    facade_path = tmp_path / "facade.py"
    facade_path.write_text(facade_source, encoding="utf-8")

    # Create a consumer that imports 'used_name' but NOT 'stale_name'.
    consumer_source = "from facade import used_name\n"
    (tmp_path / "consumer.py").write_text(consumer_source, encoding="utf-8")

    stale = find_stale_facade_shims(
        facade_file="facade.py",
        facade_source=facade_source,
        repo_root=tmp_path,
        exclude_paths=frozenset({"facade.py"}),
    )
    stale_names = {s.name for s in stale}
    assert "stale_name" in stale_names
    assert "used_name" not in stale_names


def test_no_stale_shims_when_all_imported(tmp_path: Path) -> None:
    """No stale shims when every re-export entry is imported somewhere."""
    facade_source = textwrap.dedent("""
        import importlib

        _REEXPORTS = {
            'name_a': 'some.module',
            'name_b': 'some.module',
        }

        def __getattr__(name):
            if name in _REEXPORTS:
                mod = importlib.import_module(_REEXPORTS[name])
                return getattr(mod, name)
            raise AttributeError(name)
    """)
    facade_path = tmp_path / "facade.py"
    facade_path.write_text(facade_source, encoding="utf-8")

    (tmp_path / "consumer.py").write_text("from facade import name_a, name_b\n", encoding="utf-8")

    stale = find_stale_facade_shims(
        facade_file="facade.py",
        facade_source=facade_source,
        repo_root=tmp_path,
        exclude_paths=frozenset({"facade.py"}),
    )
    assert stale == []


def test_parse_shim_mapping_extracts_names() -> None:
    source = textwrap.dedent("""
        _REEXPORTS = {
            'foo': 'mod.a',
            'bar': 'mod.b',
        }
        _CLS_REEXPORTS_MyClass = {
            'method1': 'mod.c',
        }
    """)
    names = parse_shim_mapping(source)
    assert set(names) == {"foo", "bar", "method1"}


def test_parse_shim_mapping_empty() -> None:
    assert parse_shim_mapping("# no shims here\n") == []


# ---------------------------------------------------------------------------
# render_review_packet
# ---------------------------------------------------------------------------


def test_render_review_packet_with_equivalent_move() -> None:
    result = GateResult(
        moved_symbols=(
            MovedSymbol(
                name="func",
                source_file="src/a.py",
                dest_file="src/b.py",
                source_class=None,
                dest_class=None,
                equivalent=True,
            ),
        ),
        stale_shims=(),
        base="abc123",
    )
    packet = render_review_packet(result)
    assert "AST-equivalence gate" in packet
    assert "func" in packet
    assert "verbatim" in packet
    assert "evidence" in packet.lower()


def test_render_review_packet_with_non_equivalent_move() -> None:
    result = GateResult(
        moved_symbols=(
            MovedSymbol(
                name="func",
                source_file="src/a.py",
                dest_file="src/b.py",
                source_class=None,
                dest_class=None,
                equivalent=False,
            ),
        ),
        stale_shims=(),
        base="abc123",
    )
    packet = render_review_packet(result)
    assert "NO" in packet or "review" in packet.lower()
    assert "non-equivalent" in packet.lower() or "0 equivalent" in packet


def test_render_review_packet_with_stale_shims() -> None:
    result = GateResult(
        moved_symbols=(),
        stale_shims=(StaleShim(name="stale", facade_module="old.mod", facade_file="old.py"),),
        base="abc123",
    )
    packet = render_review_packet(result)
    assert "stale" in packet.lower()
    assert "stale" in packet  # the name
    assert "vulture" in packet.lower()


def test_render_review_packet_no_moves() -> None:
    result = GateResult(
        moved_symbols=(),
        stale_shims=(),
        base="abc123",
    )
    packet = render_review_packet(result)
    assert "No symbols moved" in packet


# ---------------------------------------------------------------------------
# CLI command (charlie ast-equivalence-check)
# ---------------------------------------------------------------------------


def _make_run_result(stdout: str = "", ok: bool = True) -> RunResult:
    return RunResult(
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="",
        error=None if ok else "error",
    )


def test_cli_ast_equivalence_check_detects_verbatim_move(monkeypatch, tmp_path: Path) -> None:
    """The CLI command detects a verbatim move via git diff + AST comparison."""
    from charlie_work import cli as cli_module

    func_source = "def moved_func():\n    return 42\n"

    # Mock git diff --name-only: one file removed, one added.
    # Mock git show: base version of old.py has the function, new.py doesn't exist at base.
    # Mock working tree reads: old.py is empty at head, new.py has the function.

    def mock_run_captured(cmd, cwd, timeout_seconds=60, **kw):
        if "diff" in cmd and "--name-only" in cmd:
            return _make_run_result(stdout="src/old.py\nsrc/new.py\n")
        if "show" in cmd:
            ref = cmd[cmd.index("show") + 1]
            path = ref.split(":", 1)[1] if ":" in ref else ""
            if "old.py" in path and "base" in ref:
                return _make_run_result(stdout=func_source)
            # new.py doesn't exist at base
            return _make_run_result(stdout="", ok=False)
        return _make_run_result(stdout="", ok=False)

    def mock_bootstrap(args):
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub
        from charlie_work.paths import RuntimePaths

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=OrchestratorConfig(),
            paths=RuntimePaths.__new__(RuntimePaths),  # minimal
            gh=GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    # Create the working tree files.
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "old.py").write_text("# moved out\n", encoding="utf-8")
    (tmp_path / "src" / "new.py").write_text(func_source, encoding="utf-8")

    monkeypatch.setattr(cli_module, "run_captured", mock_run_captured)
    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)

    import argparse

    args = argparse.Namespace(
        command="ast-equivalence-check",
        base="base",
        shim_file=None,
        output=None,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )
    result = run_ast_equivalence_check_command(args)
    assert result.ok is True
    assert len(result.data["moved_symbols"]) == 1
    assert result.data["moved_symbols"][0]["name"] == "moved_func"
    assert result.data["moved_symbols"][0]["equivalent"] is True


def test_cli_ast_equivalence_check_always_ok(monkeypatch, tmp_path: Path) -> None:
    """The gate always returns ok=True (evidence, not enforcement -- graft C)."""
    from charlie_work import cli as cli_module

    def mock_run_captured(cmd, cwd, timeout_seconds=60, **kw):
        if "diff" in cmd and "--name-only" in cmd:
            return _make_run_result(stdout="")
        return _make_run_result(stdout="", ok=False)

    def mock_bootstrap(args):
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=OrchestratorConfig(),
            paths=None,
            gh=GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    monkeypatch.setattr(cli_module, "run_captured", mock_run_captured)
    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)

    import argparse

    args = argparse.Namespace(
        command="ast-equivalence-check",
        base="base",
        shim_file=None,
        output=None,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )
    result = run_ast_equivalence_check_command(args)
    # Even with no moves, the gate is ok=True (evidence, not enforcement).
    assert result.ok is True
    assert result.data["moved_symbols"] == []
