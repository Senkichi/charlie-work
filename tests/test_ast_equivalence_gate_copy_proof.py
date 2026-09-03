"""Copy-detection proof for the AST-equivalence gate (issue #1544 Stage 1).

Round-1 review finding #4: ``derive_moved_symbols`` only detects
removal+addition pairs, so a two-copy extraction (original retained, copy
added to a new file) produces ZERO moved symbols.  Issue #1544 Stage 1 is
exactly that pattern -- ``no_console_window_kwargs`` is inlined into
``charlie_work.attachment_contracts._windows`` while
``charlie_work.subprocess_runner`` keeps its copy unchanged -- so the gate
reported zero findings for the very diff it was built to self-prove against
(graft G: "the first thing this gate verifies is the verbatim move of
attachment_contracts into its own distribution").

``derive_copied_symbols`` fills that gap: it detects symbols ADDED to a file
that are verbatim copies of a SURVIVING original (present at both base and
head).  These tests prove:

* ``derive_copied_symbols`` detects a verbatim copy as equivalent and a
  modified copy as non-equivalent.
* It does NOT double-report a move (original removed) -- that stays
  ``derive_moved_symbols``'s job.
* The real #1544 extraction (``subprocess_runner`` -> ``_windows``) is a
  verbatim copy, proven by ``extract_symbols`` + ``ast.dump`` equality on the
  actual repo files.
* The CLI command emits the copy as evidence when run against a simulated
  #1544 diff (with the unchanged original providing the surviving source).

Mutation controls name the exact edit reverted and verify the test fails
against the unfixed code.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from charlie_work.ast_equivalence_gate import (
    CopiedSymbol,
    derive_copied_symbols,
    derive_moved_symbols,
    extract_symbols,
)
from charlie_work.ast_equivalence_gate_command import (
    run_ast_equivalence_check_command,
)
from charlie_work.subprocess_runner import RunResult

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# derive_copied_symbols -- unit tests
# ---------------------------------------------------------------------------


def test_verbatim_copy_is_detected_as_equivalent() -> None:
    """A function copied verbatim to a new file (original retained) is a copy.

    This is the self-proving shape: the original survives in file_a at both
    base and head, and an identical function is added to file_b at head.
    ``derive_moved_symbols`` reports zero (no removal); ``derive_copied_symbols``
    reports one equivalent copy.
    """
    func_source = "def helper():\n    return 42\n"
    base = {"src/a.py": extract_symbols(func_source, "a.py")}
    head = {
        "src/a.py": extract_symbols(func_source, "a.py"),  # original survives
        "src/b.py": extract_symbols(func_source, "b.py"),  # copy added
    }
    moved = derive_moved_symbols(base, head)
    assert moved == [], "a copy (original retained) is not a move"
    copied = derive_copied_symbols(base, head)
    assert len(copied) == 1
    assert copied[0].name == "helper"
    assert copied[0].source_file == "src/a.py"
    assert copied[0].dest_file == "src/b.py"
    assert copied[0].equivalent is True


def test_non_equivalent_copy_is_flagged() -> None:
    """A copy whose body differs from the surviving original is non-equivalent."""
    original = "def helper():\n    return 42\n"
    modified = "def helper():\n    return 99\n"
    base = {"src/a.py": extract_symbols(original, "a.py")}
    head = {
        "src/a.py": extract_symbols(original, "a.py"),
        "src/b.py": extract_symbols(modified, "b.py"),
    }
    copied = derive_copied_symbols(base, head)
    assert len(copied) == 1
    assert copied[0].equivalent is False


def test_no_copy_when_symbol_is_purely_new() -> None:
    """A symbol added with no surviving original anywhere is not a copy."""
    new_source = "def brand_new():\n    return 1\n"
    base: dict[str, dict[str, str]] = {}
    head = {"src/new.py": extract_symbols(new_source, "new.py")}
    copied = derive_copied_symbols(base, head)
    assert copied == []


def test_copy_not_double_reported_as_move() -> None:
    """A move (original removed) is reported by derive_moved, NOT derive_copied.

    Mutation control: if ``derive_copied_symbols`` matched added symbols
    against originals that were REMOVED (not surviving), it would double-report
    a move.  The surviving-original check (present in BOTH base and head of the
    same file) prevents this -- a removed original is absent at head, so it is
    not in the survivor index.
    """
    func_source = "def helper():\n    return 42\n"
    base = {"src/a.py": extract_symbols(func_source, "a.py")}
    head = {"src/b.py": extract_symbols(func_source, "b.py")}  # a.py gone at head
    moved = derive_moved_symbols(base, head)
    assert len(moved) == 1, "a removal+addition is a move"
    copied = derive_copied_symbols(base, head)
    assert copied == [], "a move must not be double-reported as a copy"


def test_class_method_copy_between_classes() -> None:
    """A method copied into a different class (original retained) is detected."""
    original = textwrap.dedent("""
        class Origin:
            def helper(self):
                return 1
    """)
    copy = textwrap.dedent("""
        class Replica:
            def helper(self):
                return 1
    """)
    base = {"src/origin.py": extract_symbols(original, "origin.py")}
    head = {
        "src/origin.py": extract_symbols(original, "origin.py"),
        "src/replica.py": extract_symbols(copy, "replica.py"),
    }
    copied = derive_copied_symbols(base, head)
    method_copies = [c for c in copied if c.name == "helper"]
    assert len(method_copies) == 1
    assert method_copies[0].source_class == "Origin"
    assert method_copies[0].dest_class == "Replica"
    assert method_copies[0].equivalent is True


# ---------------------------------------------------------------------------
# Real #1544 extraction proof (the self-proving case, graft G)
# ---------------------------------------------------------------------------


def test_real_1544_extraction_is_verbatim_copy() -> None:
    """The actual ``no_console_window_kwargs`` copy is verbatim (graft G).

    This is the gate's self-proving case: ``no_console_window_kwargs`` was
    copied from ``charlie_work.subprocess_runner`` into
    ``charlie_work.attachment_contracts._windows`` (issue #1544 Stage 1) while
    the original was retained.  ``extract_symbols`` + ``ast.dump`` equality on
    the REAL repo files proves the copy is verbatim -- the acceptance
    criterion "#1541's AST-equivalence gate run against the extraction and
    passes (verbatim move proven)" is evidenced by this AST equality.

    Mutation control: if ``_windows.no_console_window_kwargs``'s body is
    changed (e.g. a different creationflags expression), the dumps differ and
    this test fails -- proving the assertion is not vacuous.
    """
    runner_path = _REPO_ROOT / "src" / "charlie_work" / "subprocess_runner.py"
    windows_path = _REPO_ROOT / "src" / "charlie_work" / "attachment_contracts" / "_windows.py"
    assert runner_path.is_file(), "subprocess_runner.py must exist"
    assert windows_path.is_file(), "attachment_contracts/_windows.py must exist"

    runner_syms = extract_symbols(runner_path.read_text(encoding="utf-8"), str(runner_path))
    windows_syms = extract_symbols(windows_path.read_text(encoding="utf-8"), str(windows_path))

    assert "no_console_window_kwargs" in runner_syms, "original must define the helper"
    assert "no_console_window_kwargs" in windows_syms, "inlined copy must define the helper"
    # The AST dumps are identical -> verbatim copy (include_attributes=False
    # strips line/column metadata so only the structural AST is compared).
    assert runner_syms["no_console_window_kwargs"] == windows_syms["no_console_window_kwargs"], (
        "no_console_window_kwargs was NOT copied verbatim: the AST dumps of "
        "subprocess_runner's and _windows's definitions differ"
    )


def test_real_1544_extraction_detected_by_derive_copied_symbols() -> None:
    """``derive_copied_symbols`` detects the real #1544 copy as equivalent.

    Feeds the real repo files' symbol maps (subprocess_runner surviving at
    both base and head; _windows added at head) through
    ``derive_copied_symbols`` and asserts the copy is detected with
    ``equivalent=True``.  This is the function-level proof that the gate's
    data model represents the #1544 extraction.
    """
    runner_path = _REPO_ROOT / "src" / "charlie_work" / "subprocess_runner.py"
    windows_path = _REPO_ROOT / "src" / "charlie_work" / "attachment_contracts" / "_windows.py"
    runner_rel = "src/charlie_work/subprocess_runner.py"
    windows_rel = "src/charlie_work/attachment_contracts/_windows.py"
    runner_syms = extract_symbols(runner_path.read_text(encoding="utf-8"), runner_rel)
    windows_syms = extract_symbols(windows_path.read_text(encoding="utf-8"), windows_rel)

    # Base: subprocess_runner has the helper; _windows does not exist yet.
    # Head: subprocess_runner STILL has it (surviving); _windows has the copy.
    base = {runner_rel: runner_syms}
    head = {runner_rel: runner_syms, windows_rel: windows_syms}

    copied = derive_copied_symbols(base, head)
    ncwk = [c for c in copied if c.name == "no_console_window_kwargs"]
    assert len(ncwk) == 1
    assert ncwk[0].source_file == runner_rel
    assert ncwk[0].dest_file == windows_rel
    assert ncwk[0].equivalent is True


# ---------------------------------------------------------------------------
# CLI command emits the copy as evidence (simulated #1544 diff)
# ---------------------------------------------------------------------------


def _make_run_result(stdout: str = "", ok: bool = True) -> RunResult:
    return RunResult(
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="",
        error=None if ok else "error",
    )


def test_cli_emits_copied_symbol_for_1544_layout(monkeypatch, tmp_path: Path) -> None:
    """The CLI command emits a verbatim copy for the #1544 two-copy layout.

    Simulates the #1544 diff: ``subprocess_runner.py`` is UNCHANGED (it is not
    in ``git diff --name-only``) and keeps ``no_console_window_kwargs``; a new
    ``_windows.py`` is added (it IS in the diff) with an identical copy.  The
    command merges the unchanged ``subprocess_runner.py`` into the symbol maps
    via ``_merge_unchanged_src_symbols`` so ``derive_copied_symbols`` can match
    the added copy against the surviving original, and emits it in
    ``result.data["copied_symbols"]`` with ``equivalent=True``.

    Mutation control: reverting ``_merge_unchanged_src_symbols`` (so unchanged
    files are NOT merged) makes the survivor index empty, so
    ``derive_copied_symbols`` finds no surviving original and
    ``copied_symbols`` is empty -- the ``len == 1`` assertion fails.
    """
    from charlie_work import cli as cli_module

    func_source = "def no_console_window_kwargs():\n    return {}\n"

    # Working tree: subprocess_runner.py (unchanged, surviving original) +
    # attachment_contracts/_windows.py (new file with the verbatim copy).
    src = tmp_path / "src" / "charlie_work" / "attachment_contracts"
    src.mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "charlie_work" / "subprocess_runner.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "src" / "charlie_work" / "subprocess_runner.py").write_text(
        func_source, encoding="utf-8"
    )
    (src / "_windows.py").write_text(func_source, encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")

    # git diff --name-only reports ONLY the new file (subprocess_runner is
    # unchanged, so it is not in the diff -- this is the crux of finding #4).
    diff_stdout = "src/charlie_work/attachment_contracts/_windows.py\n"

    def mock_run_captured(cmd, cwd, timeout_seconds=60, **kw):
        if "diff" in cmd and "--name-only" in cmd:
            return _make_run_result(stdout=diff_stdout)
        # git show base:<new file> -> file did not exist at base -> empty/ok=False
        return _make_run_result(stdout="", ok=False)

    def mock_bootstrap(args):
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub
        from charlie_work.paths import RuntimePaths

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=OrchestratorConfig(),
            paths=RuntimePaths.__new__(RuntimePaths),
            gh=GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    monkeypatch.setattr(cli_module, "run_captured", mock_run_captured)
    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)

    args = argparse.Namespace(
        command="ast-equivalence-check",
        base="base",
        shim_file=None,
        output=None,
        generate_shims=None,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )
    result = run_ast_equivalence_check_command(args)
    assert result.ok is True
    copied = result.data["copied_symbols"]
    assert len(copied) == 1, (
        f"expected 1 copied symbol for the #1544 layout, got {len(copied)}: {copied}"
    )
    assert copied[0]["name"] == "no_console_window_kwargs"
    assert copied[0]["source_file"] == "src/charlie_work/subprocess_runner.py"
    assert copied[0]["dest_file"] == "src/charlie_work/attachment_contracts/_windows.py"
    assert copied[0]["equivalent"] is True
    # No moves -- the original was retained, so derive_moved reports nothing.
    assert result.data["moved_symbols"] == []


def test_copied_symbol_dataclass_is_frozen() -> None:
    """CopiedSymbol is a frozen dataclass (CLAUDE.md config/value-object invariant)."""
    sym = CopiedSymbol(
        name="x",
        source_file="a.py",
        dest_file="b.py",
        source_class=None,
        dest_class=None,
        equivalent=True,
    )
    try:
        sym.equivalent = False  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("CopiedSymbol must be frozen (mutation must raise)")
