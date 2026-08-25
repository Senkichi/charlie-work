"""Derive a module map from the live source tree at packet build time.

Issue #1444: worker agents land new code in ``workflow.py`` by default because
it is the minimum-risk completion -- no new imports to wire, no naming
decisions, no conventions to learn. The worker's dispatch prompt gave it no
picture of what modules exist or what belongs where, so the largest file won
by gravity. This module derives a "module map" section from the tree at packet
build time: for every module under ``src/charlie_work/``, it lists the module
name, the first line of its docstring, and its public-surface size.

The map is NEVER a hand-maintained list. A hardcoded map rots and is the very
antipattern this effort fights. ``build_module_map`` walks the package
directory with ``pathlib`` and parses each ``.py`` file with ``ast`` -- zero
hardcoded module names, zero imports of the modules themselves (which would
pull heavy deps and side effects into packet build). A newly added module
appears in the next built packet with no config change.

Fail-soft: a repo layout this module cannot parse yields an empty string (an
omitted section) plus a warning event recorded by the caller, never a dispatch
failure. ``build_module_map`` raises on parse errors so the caller can record
the event and degrade; it does not swallow exceptions itself, keeping the
single point of enforcement for "dispatch never fails on a map error" at the
call site (``_write_worker_prompt``).
"""

from __future__ import annotations

import ast
from pathlib import Path

# The package directory is walked non-recursively for ``.py`` files, then each
# subdirectory that is a Python package (contains ``__init__.py``) is walked
# the same way. ``__pycache__`` and other non-``.py`` entries are ignored by
# the glob. There is deliberately no hardcoded module list -- the set of
# modules is whatever ``*.py`` files exist on disk.


def _module_dotted_name(py_file: Path, src_root: Path) -> str:
    """Return the dotted module name for ``py_file`` relative to ``src_root``.

    ``src/charlie_work/foo.py`` -> ``charlie_work.foo``;
    ``src/charlie_work/__init__.py`` -> ``charlie_work``;
    ``src/charlie_work/prompts/__init__.py`` -> ``charlie_work.prompts``.
    """
    rel = py_file.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_docstring_first_line(tree: ast.Module) -> str:
    """Return the first non-empty line of the module docstring, or ``''``."""
    doc = ast.get_docstring(tree, clean=True)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _public_surface_size(tree: ast.Module) -> int:
    """Count the module's public top-level names.

    If ``__all__`` is defined as a list or tuple literal at module level, its
    length is the public surface. Otherwise, count top-level bindings
    (functions, async functions, classes, and assignment targets) whose name
    does not start with an underscore, deduplicated.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return len(node.value.elts)
                    # A dynamically-built __all__ (e.g. ``extend``) cannot be
                    # measured statically; fall through to the name-count path.
                    break
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.add(node.target.id)
    return len(names)


def _iter_package_py_files(package_dir: Path) -> list[Path]:
    """Yield every ``.py`` file under ``package_dir`` (recursive, sorted).

    Walks the package directory recursively, yielding ``.py`` files in
    deterministic (path-sorted) order. Subdirectories need not contain an
    ``__init__.py`` to be walked -- the map is a layout aid, not an import
    linter, and a namespace package or a stray ``.py`` under a data directory
    is still a file a worker might place code in.
    """
    return sorted(package_dir.rglob("*.py"))


def build_module_map(package_dir: Path, src_root: Path) -> str:
    """Build the module-map prompt section from the live tree.

    Walks ``package_dir`` (e.g. ``src/charlie_work``) for ``.py`` files, parses
    each with ``ast`` (no imports -- avoids side effects and heavy deps at
    packet build time), and returns a markdown table listing each module's
    dotted name, the first line of its docstring, and its public-surface size.

    Args:
        package_dir: The package directory to map (e.g. ``src/charlie_work``).
        src_root: The ``src`` directory ``package_dir`` lives under, used to
            derive dotted module names.

    Returns:
        The full module-map section text (header + table), or an empty string
        if ``package_dir`` does not exist or contains no ``.py`` files.

    Raises:
        OSError: if a file cannot be read.
        SyntaxError: if a file cannot be parsed.

        These are deliberately NOT caught here: the caller
        (``_write_worker_prompt``) wraps the call in a fail-soft try/except,
        records a ``worker_module_map_failed`` warning event, and ships the
        prompt with an omitted section. Keeping the raise here means the
        single point of enforcement for "dispatch never fails on a map error"
        is the caller, not scattered across this module.
    """
    if not package_dir.is_dir():
        return ""
    py_files = _iter_package_py_files(package_dir)
    if not py_files:
        return ""

    rows: list[tuple[str, str, int]] = []
    for py_file in py_files:
        dotted = _module_dotted_name(py_file, src_root)
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        doc_line = _module_docstring_first_line(tree)
        size = _public_surface_size(tree)
        rows.append((dotted, doc_line, size))

    # Sort by dotted module name for deterministic output across platforms
    # and processes (path iteration order is not guaranteed to be stable).
    rows.sort(key=lambda r: r[0])

    lines = [
        "## Module map",
        "",
        "Derived from the live tree at packet build time. Place new code in the "
        "module whose purpose matches it -- not in the largest file by default.",
        "",
        "| Module | Purpose (docstring first line) | Public surface |",
        "|---|---|---|",
    ]
    for dotted, doc_line, size in rows:
        # Escape pipe characters so they do not break the markdown table.
        purpose = doc_line.replace("|", "\\|") or "(no docstring)"
        lines.append(f"| `{dotted}` | {purpose} | {size} |")
    return "\n".join(lines) + "\n"
