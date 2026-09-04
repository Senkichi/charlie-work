"""CLI command layer for the AST-equivalence gate (issue #1541).

This module is the command wrapper for
:mod:`charlie_work.ast_equivalence_gate`, following the same split as
:mod:`charlie_work.private_slug_check_command` (subparser registration +
git I/O + exit-code decision here; pure scanning logic in the gate module).

``cli`` is imported lazily *inside* the functions that need it, for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .ast_equivalence_gate import (
    GateResult,
    StaleShim,
    derive_copied_symbols,
    derive_moved_symbols,
    extract_symbols,
    find_stale_facade_shims,
    generate_pep562_shim_source,
    render_review_packet,
)
from .workflow import CommandResult


def register_ast_equivalence_check_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``ast-equivalence-check`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "ast-equivalence-check",
        help=(
            "CI gate (issue #1541): detect symbols moved between files in the "
            "PR diff and verify each is a verbatim relocation by comparing "
            "ast.dump(node, include_attributes=False) at base vs head. "
            "Generates PEP 562 facade shims for moved symbols, runs a vulture "
            "sweep for stale shims, and renders a review packet. The packet "
            "is evidence for the human reviewer -- enforcement is the "
            "required_checks list (#1538), not this gate."
        ),
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Git ref to diff against (default: origin/main). Uses the "
        "two-dot diff (base..HEAD), same as mojibake-check, for the same "
        "shallow-clone reason.",
    )
    parser.add_argument(
        "--shim-file",
        default=None,
        action="append",
        help="Path to an existing PEP 562 facade shim file to run the "
        "vulture stale-shim sweep against. May be passed multiple times. "
        "When omitted, the sweep is skipped (no shims to check).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the review packet to this file (in addition to stdout). "
        "When set, also writes to $GITHUB_STEP_SUMMARY if that env var is "
        "set (CI step-summary rendering).",
    )
    parser.add_argument(
        "--generate-shims",
        default=None,
        metavar="DIR",
        help="Generate PEP 562 facade shim source for each module that had "
        "symbols moved OUT of it in this diff, writing one file per facade "
        "module under DIR (preserving the source-relative path). This is the "
        "production call site for generate_pep562_shim_source: a relocation PR "
        "author runs `charlie ast-equivalence-check --generate-shims shims/` to "
        "produce the compat shims that keep old import paths resolving, then "
        "reviews and places them. Not wired into CI -- shims are generated "
        "on-demand by the author of a move, not on every PR.",
    )


def _git_changed_py_files(
    run_captured: Any, repo_root: Path, base: str
) -> tuple[list[str], str | None]:
    """Return (changed .py files, error_message) via ``git diff --name-only``."""
    result = run_captured(
        ["git", "diff", "--name-only", f"{base}..HEAD"],
        cwd=repo_root,
        timeout_seconds=60,
    )
    if not result.ok:
        return [], result.error or result.stderr or "git diff --name-only failed"
    files = [f for f in result.stdout.splitlines() if f.endswith(".py")]
    return files, None


def _git_show_file(run_captured: Any, repo_root: Path, ref: str, path: str) -> str | None:
    """Return the content of *path* at *ref* via ``git show``, or None on error."""
    result = run_captured(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    if not result.ok:
        return None
    return result.stdout


def _read_working_tree_file(repo_root: Path, path: str) -> str | None:
    """Return the content of *path* from the working tree, or None on error."""
    abs_path = repo_root / path
    try:
        return abs_path.read_text(encoding="utf-8", errors="surrogateescape")
    except (OSError, UnicodeDecodeError):
        return None


def _build_symbol_maps(
    run_captured: Any, repo_root: Path, base: str, changed_files: list[str]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Build (base_symbols_by_file, head_symbols_by_file) from the diff.

    For each changed .py file, extract symbols from the base version (via
    ``git show base:path``) and the head version (working tree).  Files that
    don't exist at a given ref contribute an empty symbol map.
    """
    base_maps: dict[str, dict[str, str]] = {}
    head_maps: dict[str, dict[str, str]] = {}

    for path in changed_files:
        # Base version
        base_source = _git_show_file(run_captured, repo_root, base, path)
        if base_source is not None:
            try:
                base_maps[path] = extract_symbols(base_source, filename=path)
            except SyntaxError:
                base_maps[path] = {}
        else:
            base_maps[path] = {}

        # Head version (working tree)
        head_source = _read_working_tree_file(repo_root, path)
        if head_source is not None:
            try:
                head_maps[path] = extract_symbols(head_source, filename=path)
            except SyntaxError:
                head_maps[path] = {}
        else:
            head_maps[path] = {}

    return base_maps, head_maps


def _merge_unchanged_src_symbols(
    base_maps: dict[str, dict[str, str]],
    head_maps: dict[str, dict[str, str]],
    repo_root: Path,
    changed_files: list[str],
) -> None:
    """Merge unchanged ``src/**/*.py`` files into both symbol maps in place.

    ``derive_moved_symbols`` only needs changed files (a move is a
    removal+addition, both visible in the changed set).  But
    ``derive_copied_symbols`` matches an ADDED symbol against the SURVIVING
    original, which lives in an UNCHANGED file (the two-copy extraction
    pattern: the original is retained, so it never appears in the diff).  The
    #1544 Stage 1 extraction is exactly this -- ``no_console_window_kwargs`` is
    copied into ``attachment_contracts/_windows.py`` while
    ``charlie_work/subprocess_runner.py`` keeps its copy unchanged -- so
    without the unchanged files in the maps, the gate reports zero copied
    symbols for the very diff it was built to self-prove against (round-1
    review finding #4).

    For an unchanged file, base == head == the working-tree version, so a
    single read populates both maps identically (no ``git show`` needed).
    Unchanged files contribute no removals and no additions, so they do not
    perturb ``derive_moved_symbols`` (a symbol present in both base and head of
    the same file is neither removed nor added).
    """
    changed_set = set(changed_files)
    src_root = repo_root / "src"
    if not src_root.is_dir():
        return
    for py_file in src_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        if rel in changed_set:
            continue  # already in the maps via _build_symbol_maps
        if _is_under_excluded_dir(py_file.relative_to(repo_root).parts):
            continue
        source = _read_working_tree_file(repo_root, rel)
        if source is None:
            continue
        try:
            syms = extract_symbols(source, filename=rel)
        except SyntaxError:
            continue
        base_maps.setdefault(rel, syms)
        head_maps.setdefault(rel, syms)


def _is_under_excluded_dir(rel_parts: tuple[str, ...]) -> bool:
    """Skip non-source directory components (mirrors the gate's vulture sweep)."""
    _EXCLUDED = frozenset({".venv", ".git", "__pycache__", "build", "dist"})
    return any(part in _EXCLUDED for part in rel_parts[:-1])


def _generate_shim_files(moved: list, repo_root: Path, out_dir: str) -> list[str]:
    """Generate PEP 562 facade shims for each module symbols moved OUT of.

    Groups *moved* by ``source_file`` (the facade module the symbols left),
    calls :func:`generate_pep562_shim_source` for each group, and writes the
    generated source to ``out_dir/<source_file>`` (preserving the
    source-relative path so the file maps 1:1 to the facade module it
    replaces).  Returns the list of written paths (relative to *repo_root*).

    This is the production call site for ``generate_pep562_shim_source``: a
    relocation PR author invokes it via ``--generate-shims DIR`` to produce
    the compat shims that keep old import paths resolving post-move.
    """
    # Group moved symbols by the facade module they moved out of.
    by_facade: dict[str, list] = {}
    for sym in moved:
        by_facade.setdefault(sym.source_file, []).append(sym)

    written: list[str] = []
    dest_root = repo_root / out_dir
    for facade_path, syms in by_facade.items():
        shim_source = generate_pep562_shim_source(facade_path, syms)
        out_path = dest_root / facade_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(shim_source, encoding="utf-8")
        written.append(facade_path)
    return written


def run_ast_equivalence_check_command(
    args: argparse.Namespace,
) -> CommandResult:
    """CI gate (issue #1541): detect and verify symbol relocations in the PR diff.

    Runs ``git diff --name-only base..HEAD`` to find changed ``.py`` files,
    extracts symbols from each at base and head, derives the moved-symbol set
    (diff-derived, never hand-typed -- graft E), and renders a review packet.

    If ``--shim-file`` is passed, the vulture stale-shim sweep runs against
    each named facade file, flagging re-export entries nobody imports.

    The gate **always exits 0** (ok=True): the review packet is evidence for
    the human reviewer, not enforcement.  Enforcement is the
    ``required_checks`` list (#1538), not this gate (graft C).

    Errors as values (per CLAUDE.md): git failures come back as
    ``CommandResult(ok=False)`` -- never raised.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    # Issue #1600: this is a read-only diagnostic, not a state-mutating
    # command.  bootstrap_command defaults to redirecting a linked-worktree
    # cwd to the shared main worktree root (issue #648 state-safety), but
    # that redirect makes the gate silently inspect the main worktree's diff
    # instead of the worktree it was invoked from.  Pass
    # redirect_to_main_worktree=False so the gate honors the invoking worktree.
    ctx = cli.bootstrap_command(args, redirect_to_main_worktree=False)
    base = getattr(args, "base", "origin/main")

    changed_files, err = _git_changed_py_files(cli.run_captured, ctx.repo_root, base)
    if err is not None:
        return CommandResult(
            False,
            f"ast-equivalence-check: could not run git diff against {base}: {err}",
            {"base": base},
        )

    base_maps, head_maps = _build_symbol_maps(cli.run_captured, ctx.repo_root, base, changed_files)

    moved = derive_moved_symbols(base_maps, head_maps)

    # Copy detection (round-1 review finding #4): ``derive_moved_symbols``
    # only sees removal+addition pairs, so a two-copy extraction (original
    # retained, e.g. #1544 Stage 1) produces zero moved symbols.  Merge the
    # UNCHANGED src/ files into the maps so ``derive_copied_symbols`` can
    # match added symbols against surviving originals in unchanged files.
    # This is what evidences the gate's self-proving requirement (graft G) for
    # the #1544 extraction.  Unchanged files do not perturb ``moved`` (no
    # removals/additions), so this is safe to do unconditionally.
    _merge_unchanged_src_symbols(base_maps, head_maps, ctx.repo_root, changed_files)
    copied = derive_copied_symbols(base_maps, head_maps)

    # PEP 562 facade shim generation (production call site for
    # generate_pep562_shim_source). On-demand: the relocation PR author passes
    # --generate-shims DIR to emit compat shims for each module symbols moved
    # out of. Not wired into CI -- shims are generated by the move's author,
    # not on every PR.
    generated_shims: list[str] = []
    shim_out_dir = getattr(args, "generate_shims", None)
    if shim_out_dir:
        generated_shims = _generate_shim_files(moved, ctx.repo_root, shim_out_dir)

    # Vulture stale-shim sweep
    stale: list[StaleShim] = []
    shim_files = getattr(args, "shim_file", None) or []
    for shim_path in shim_files:
        shim_source = _read_working_tree_file(ctx.repo_root, shim_path)
        if shim_source is None:
            continue
        stale.extend(
            find_stale_facade_shims(
                facade_file=shim_path,
                facade_source=shim_source,
                repo_root=ctx.repo_root,
                exclude_paths=frozenset({shim_path}),
            )
        )

    result = GateResult(
        moved_symbols=tuple(moved),
        stale_shims=tuple(stale),
        base=base,
        copied_symbols=tuple(copied),
    )

    packet = render_review_packet(result)

    # Write to --output file and/or $GITHUB_STEP_SUMMARY (CI rendering).
    import os

    output_path = getattr(args, "output", None)
    if output_path:
        out = ctx.repo_root / output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(packet + "\n", encoding="utf-8")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write(packet + "\n")
        except OSError:
            pass  # non-fatal: packet is also on stdout

    data: dict[str, Any] = {
        "base": base,
        "changed_files": changed_files,
        "moved_symbols": [
            {
                "name": s.name,
                "source_file": s.source_file,
                "dest_file": s.dest_file,
                "source_class": s.source_class,
                "dest_class": s.dest_class,
                "equivalent": s.equivalent,
            }
            for s in result.moved_symbols
        ],
        "stale_shims": [
            {
                "name": s.name,
                "facade_module": s.facade_module,
                "facade_file": s.facade_file,
            }
            for s in result.stale_shims
        ],
        "copied_symbols": [
            {
                "name": s.name,
                "source_file": s.source_file,
                "dest_file": s.dest_file,
                "source_class": s.source_class,
                "dest_class": s.dest_class,
                "equivalent": s.equivalent,
            }
            for s in result.copied_symbols
        ],
        "generated_shims": generated_shims,
    }

    # Always ok=True: the packet is evidence, not enforcement (graft C).
    eq = len(result.equivalent_moves)
    neq = len(result.non_equivalent_moves)
    cpy_eq = len(result.equivalent_copies)
    cpy_neq = len(result.non_equivalent_copies)
    stale_count = len(result.stale_shims)
    shim_count = len(generated_shims)
    message = (
        f"ast-equivalence-check: {eq} equivalent move(s), "
        f"{neq} non-equivalent move(s), "
        f"{cpy_eq} verbatim copy(ies), {cpy_neq} non-equivalent copy(ies), "
        f"{stale_count} stale shim(s), "
        f"{shim_count} shim(s) generated (diff against {base})"
    )
    return CommandResult(True, message, data)
