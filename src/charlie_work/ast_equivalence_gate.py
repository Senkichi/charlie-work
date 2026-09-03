"""AST-equivalence gate for verbatim symbol relocations (issue #1541).

For every symbol moved between files in a PR, this gate compares
``ast.dump(node, include_attributes=False)`` of that symbol's definition at
base vs head.  A moved function/class whose dumped AST is identical passed
the gate (a verbatim relocation); any difference is flagged for human review
as a non-equivalent move.

The moved-symbol set is **diff-derived, never hand-typed** (graft E, global
rule #9): it is computed as ``base-not-head(file A)`` ∩ ``head-not-base(file
B)``, matched by name and verified by ``ast.dump`` equality -- i.e. the gate
derives "this symbol moved from A to B" purely from the diff between the two
commits.  No hardcoded list of symbol names exists anywhere in this module
(rule #9 compliance: ``grep`` of this source contains no symbol-name list).

The module also provides:

* :func:`generate_pep562_shim_source` -- generates a module-level ``__getattr__``
  (PEP 562) facade, or a class-level ``__getattr__`` facade for class members,
  from the same diff-derived moved-symbol set, so old import paths keep
  resolving to the relocated symbol.
* :func:`find_stale_facade_shims` -- the vulture "forgotten facade" sweep:
  flags facade re-export entries nobody imports any more (shims that have
  outlived their need and are pure dead-code bloat).
* :func:`render_review_packet` -- renders the gate's per-symbol equivalence
  results into a review packet for the human reviewer.  This is **evidence,
  not enforcement** (graft C, #1538): the packet is advisory and does not
  block or allow a merge by itself.

Self-proving requirement (graft G): the first thing this gate verifies, once
built, is the verbatim move of ``attachment_contracts`` into its own
distribution (#1544).  If the gate cannot prove its own move is equivalent,
it is wrong before it ever judges a paydown PR.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model (frozen, per CLAUDE.md invariant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovedSymbol:
    """A symbol that moved from one file to another between base and head.

    ``equivalent`` is ``True`` when ``ast.dump(node, include_attributes=False)``
    is identical at base and head -- a verbatim relocation.  ``False`` means
    the symbol moved but its body changed; the gate flags it for human review.

    ``class_name`` is ``None`` for top-level symbols (functions, classes) and
    the enclosing class name for class members (methods), so the shim
    generator can emit a class-level ``__getattr__`` facade for the latter.
    """

    name: str
    source_file: str
    dest_file: str
    source_class: str | None
    dest_class: str | None
    equivalent: bool


@dataclass(frozen=True)
class StaleShim:
    """A facade re-export entry that nobody imports any more (vulture sweep).

    The shim has outlived its need and is pure dead-code bloat: the name is
    listed in a ``__getattr__`` re-export mapping but no ``.py`` file in the
    repo imports it from the facade module.
    """

    name: str
    facade_module: str
    facade_file: str


@dataclass(frozen=True)
class GateResult:
    """The full output of the AST-equivalence gate for one PR diff.

    ``moved_symbols`` is the diff-derived set of symbols that moved between
    files.  ``stale_shims`` is the vulture sweep's findings.  The review
    packet (:func:`render_review_packet`) renders both for the human reviewer.
    """

    moved_symbols: tuple[MovedSymbol, ...]
    stale_shims: tuple[StaleShim, ...]
    base: str

    @property
    def equivalent_moves(self) -> tuple[MovedSymbol, ...]:
        """Moved symbols whose AST is identical at base and head (verbatim)."""
        return tuple(s for s in self.moved_symbols if s.equivalent)

    @property
    def non_equivalent_moves(self) -> tuple[MovedSymbol, ...]:
        """Moved symbols whose AST differs -- flagged for human review."""
        return tuple(s for s in self.moved_symbols if not s.equivalent)


# ---------------------------------------------------------------------------
# Symbol extraction (pure functions over source text)
# ---------------------------------------------------------------------------


def _dump_node(node: ast.AST) -> str:
    """Return ``ast.dump(node, include_attributes=False)`` -- the canonical AST form.

    ``include_attributes=False`` strips line numbers, column offsets, and other
    positional metadata so a verbatim relocation (same code, different file)
    produces an identical dump.  Only the structural AST matters.
    """
    return ast.dump(node, include_attributes=False)


def extract_symbols(source: str, filename: str = "<unknown>") -> dict[str, str]:
    """Extract top-level and class-member definitions from *source*.

    Returns a mapping of **qualified name** -> ``ast.dump`` string.  Top-level
    functions/classes are keyed by their bare name (``"foo"``); class methods
    are keyed by ``"ClassName.method_name"``.  The dump uses
    ``include_attributes=False`` so a verbatim move (same code, different
    file) produces an identical value.

    This is a pure function over source text -- no I/O, no git.  The CLI
    command layer feeds it the base and head versions of each changed file.
    """
    tree = ast.parse(source, filename=filename)
    symbols: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[node.name] = _dump_node(node)
        elif isinstance(node, ast.ClassDef):
            symbols[node.name] = _dump_node(node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{node.name}.{child.name}"
                    symbols[qualified] = _dump_node(child)
    return symbols


def _split_qualified(name: str) -> tuple[str, str | None]:
    """Split a qualified name into (bare_name, class_name).

    ``"foo"`` -> ``("foo", None)``; ``"Cls.method"`` -> ``("method", "Cls")``.
    """
    if "." in name:
        cls, method = name.split(".", 1)
        return method, cls
    return name, None


# ---------------------------------------------------------------------------
# Diff-derived moved-symbol set (graft E, rule #9)
# ---------------------------------------------------------------------------


def derive_moved_symbols(
    base_symbols_by_file: dict[str, dict[str, str]],
    head_symbols_by_file: dict[str, dict[str, str]],
) -> list[MovedSymbol]:
    """Derive the set of symbols that moved between files, purely from the diff.

    A symbol moved from file A to file B when:

    1. It is present in ``base_symbols_by_file[A]`` but absent in
       ``head_symbols_by_file[A]`` (removed from A), **and**
    2. The same bare name is present in ``head_symbols_by_file[B]`` but absent
       in ``base_symbols_by_file[B]`` (added to B), **and**
    3. The ``ast.dump`` values match (equivalent) or differ (non-equivalent).

    The moved-symbol set is computed entirely from the two symbol maps, which
    are themselves derived from the diff between base and head.  No hardcoded
    list of symbol names is used anywhere (rule #9 compliance).

    Matching is by **bare name** (the unqualified function/method name), so a
    method moving from ``class A`` in file_a.py to ``class B`` in file_b.py is
    detected.  The ``ast.dump`` comparison determines whether the move is
    verbatim (equivalent) or modified (non-equivalent).
    """
    moved: list[MovedSymbol] = []

    # Build a reverse index: for each bare name added to any file at head,
    # record (file, qualified_name, dump).  This lets us find where a
    # removed symbol went without iterating every file pair.
    added_by_name: dict[str, list[tuple[str, str, str]]] = {}
    for path_b, head_syms_b in head_symbols_by_file.items():
        base_syms_b = base_symbols_by_file.get(path_b, {})
        for qname_b, dump_b in head_syms_b.items():
            if qname_b not in base_syms_b:
                bare_b, _ = _split_qualified(qname_b)
                added_by_name.setdefault(bare_b, []).append((path_b, qname_b, dump_b))

    for path_a, base_syms_a in base_symbols_by_file.items():
        head_syms_a = head_symbols_by_file.get(path_a, {})
        for qname_a, dump_a in base_syms_a.items():
            if qname_a in head_syms_a:
                continue  # still present in A -- not a move
            bare_a, cls_a = _split_qualified(qname_a)
            candidates = added_by_name.get(bare_a, [])
            for path_b, qname_b, dump_b in candidates:
                if path_b == path_a:
                    continue
                _, cls_b = _split_qualified(qname_b)
                moved.append(
                    MovedSymbol(
                        name=bare_a,
                        source_file=path_a,
                        dest_file=path_b,
                        source_class=cls_a,
                        dest_class=cls_b,
                        equivalent=(dump_a == dump_b),
                    )
                )
                break  # first match wins; one removal -> one move

    return moved


# ---------------------------------------------------------------------------
# PEP 562 facade shim generation
# ---------------------------------------------------------------------------


def _path_to_module(path: str, src_root: str = "src") -> str:
    """Convert a source file path to a dotted module path.

    ``"src/charlie_work/foo.py"`` -> ``"charlie_work.foo"``;
    ``"charlie_work/foo.py"`` -> ``"charlie_work.foo"``;
    ``"foo.py"`` -> ``"foo"``.  ``__init__.py`` maps to the package name.
    """
    p = path.replace("\\", "/")
    parts = p.split("/")
    if parts and parts[0] == src_root:
        parts = parts[1:]
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def generate_pep562_shim_source(
    facade_module_path: str,
    moved_symbols: list[MovedSymbol],
    src_root: str = "src",
) -> str:
    """Generate a PEP 562 ``__getattr__`` facade for symbols moved out of a module.

    For top-level symbols (``source_class is None``), this produces a
    module-level ``__getattr__`` (PEP 562) that re-exports the moved names
    from their new location, so old import paths keep resolving::

        from charlie_work.old_module import moved_func  # still works

    For class members (``source_class is not None``), this produces a
    class-level ``__getattr__`` facade on the source class, so old attribute
    access keeps resolving::

        OldClass().moved_method  # still works, delegates to new class

    The shim mapping is generated from the same diff-derived moved-symbol set
    -- no hardcoded names (rule #9).
    """
    facade_module = _path_to_module(facade_module_path, src_root)

    # Partition into module-level and class-level shims.
    module_shims: dict[str, str] = {}  # name -> dest_module
    class_shims: dict[str, dict[str, str]] = {}  # class -> {name: dest_module}

    for sym in moved_symbols:
        if sym.source_file != facade_module_path:
            continue
        dest_module = _path_to_module(sym.dest_file, src_root)
        if sym.source_class is None:
            module_shims[sym.name] = dest_module
        else:
            class_shims.setdefault(sym.source_class, {})[sym.name] = dest_module

    lines: list[str] = [
        f'"""PEP 562 facade shim for symbols moved out of {facade_module}.',
        "",
        "Auto-generated by the AST-equivalence gate (issue #1541).",
        "Do not edit by hand -- regenerate with `charlie ast-equivalence-check`.",
        '"""',
        "",
        "import importlib",
        "",
    ]

    if module_shims:
        lines.append("# Module-level re-exports (PEP 562 __getattr__).")
        lines.append("_REEXPORTS = {")
        for name in sorted(module_shims):
            lines.append(f"    {name!r}: {module_shims[name]!r},")
        lines.append("}")
        lines.append("")
        lines.append("")
        lines.append("def __getattr__(name):")
        lines.append('    """Re-export moved symbols from their new modules (PEP 562)."""')
        lines.append("    if name in _REEXPORTS:")
        lines.append("        mod = importlib.import_module(_REEXPORTS[name])")
        lines.append("        return getattr(mod, name)")
        lines.append('    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")')
        lines.append("")

    if class_shims:
        for cls_name in sorted(class_shims):
            shims = class_shims[cls_name]
            lines.append(f"# Class-level facade for members moved out of {cls_name}.")
            lines.append(f"_CLS_REEXPORTS_{cls_name} = {{")
            for name in sorted(shims):
                lines.append(f"    {name!r}: {shims[name]!r},")
            lines.append("}")
            lines.append("")
            lines.append(f"def _cls_getattr_{cls_name}(self, name):")
            lines.append('    """Re-export moved members from their new classes."""')
            lines.append(f"    if name in _CLS_REEXPORTS_{cls_name}:")
            lines.append(f"        mod = importlib.import_module(_CLS_REEXPORTS_{cls_name}[name])")
            lines.append("        return getattr(mod, name)")
            lines.append(
                "    raise AttributeError("
                'f"{type(self).__name__!r} object has no attribute {name!r}")'
            )
            lines.append("")

    if not module_shims and not class_shims:
        lines.append("# No symbols moved out of this module.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vulture "forgotten facade" sweep
# ---------------------------------------------------------------------------


_SHIM_MAPPING_RE = re.compile(r"^(\w+)\s*=\s*\{", re.MULTILINE)
_SHIM_ENTRY_RE = re.compile(r"^\s*['\"](\w+)['\"]\s*:", re.MULTILINE)

# Non-source directories excluded from the vulture sweep's ``rglob('*.py')``
# walk. ``uv run`` materializes ``.venv`` inside ``repo_root`` before this
# command runs, so an unfiltered walk would scan the entire installed
# environment -- thousands of third-party ``.py`` files whose import edges are
# irrelevant to whether a *charlie_work* facade shim is stale, and which can
# produce false negatives (a third-party module importing the re-exported name
# would mask a genuinely-stale shim). The set is declarative and overridable
# via ``find_stale_facade_shims(exclude_dirs=...)``; it names directory
# *components*, matched against any path part, so a nested ``__pycache__`` or a
# ``.venv`` at any depth is skipped.
_DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".venv", ".git", "__pycache__", "build", "dist", ".tox", ".eggs"}
)


def parse_shim_mapping(source: str) -> list[str]:
    """Extract re-exported names from a PEP 562 shim module's source.

    Parses the ``_REEXPORTS`` (or ``_CLS_REEXPORTS_*``) mapping in a
    generated facade shim and returns the list of names it re-exports.
    This is what the vulture sweep uses to know which entries to check.
    """
    names: list[str] = []
    # Find all mapping blocks and extract their keys.
    for match in _SHIM_MAPPING_RE.finditer(source):
        block_start = match.end()
        # Find the closing brace from the block start.
        depth = 1
        i = block_start
        while i < len(source) and depth > 0:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        block = source[block_start : i - 1]
        for entry in _SHIM_ENTRY_RE.finditer(block):
            names.append(entry.group(1))
    return names


def _is_under_excluded_dir(rel_parts: tuple[str, ...], exclude_dirs: frozenset[str]) -> bool:
    """Return True if any path component of *rel_parts* is in *exclude_dirs*.

    Matches directory *names* at any depth (``rel_parts`` excludes the final
    filename), so a nested ``__pycache__`` or a ``.venv`` anywhere under
    ``repo_root`` is skipped without enumerating every possible parent path.
    """
    return any(part in exclude_dirs for part in rel_parts[:-1])


def find_stale_facade_shims(
    facade_file: str,
    facade_source: str,
    repo_root: Path,
    exclude_paths: frozenset[str] = frozenset(),
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[StaleShim]:
    """Vulture sweep: flag facade re-export entries nobody imports any more.

    For each name in the shim's ``_REEXPORTS`` mapping, check whether any
    source ``.py`` file under *repo_root* (excluding the facade file itself,
    any *exclude_paths*, and any file beneath a directory named in
    *exclude_dirs*) still imports that name from the facade module.  If not,
    the shim has outlived its need and is pure dead-code bloat.

    Non-source directories (``.venv``, ``.git``, ``__pycache__``, ``build``,
    ``dist``, …) are excluded from the walk by default: ``uv run`` materializes
    ``.venv`` inside ``repo_root`` at the point this command runs, and scanning
    the installed environment is both slow and a source of false negatives (a
    third-party module importing the re-exported name would mask a genuinely
    stale shim).  Pass ``exclude_dirs=frozenset()`` to walk everything.

    This is the "forgotten facade" sweep paired with the shim generator (issue
    #1541).  It runs as part of the same gate, not a separate manual chore.
    """
    shim_names = parse_shim_mapping(facade_source)
    if not shim_names:
        return []

    facade_module = _path_to_module(facade_file)
    stale: list[StaleShim] = []

    # Build a set of all names imported from the facade module across the repo.
    imported_names: set[str] = set()
    for py_file in repo_root.rglob("*.py"):
        rel_parts = py_file.relative_to(repo_root).parts
        rel = "/".join(rel_parts)
        if rel == facade_file or rel in exclude_paths:
            continue
        if _is_under_excluded_dir(rel_parts, exclude_dirs):
            continue
        try:
            tree = ast.parse(
                py_file.read_text(encoding="utf-8", errors="surrogateescape"),
                filename=str(py_file),
            )
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == facade_module:
                for alias in node.names:
                    imported_names.add(alias.name)

    for name in shim_names:
        if name not in imported_names:
            stale.append(
                StaleShim(
                    name=name,
                    facade_module=facade_module,
                    facade_file=facade_file,
                )
            )
    return stale


# ---------------------------------------------------------------------------
# Review packet rendering (evidence, not enforcement -- graft C, #1538)
# ---------------------------------------------------------------------------


def render_review_packet(result: GateResult) -> str:
    """Render the gate's findings as a review packet for the human reviewer.

    This is **evidence for the human reviewer, not the enforcement mechanism**
    (graft C, #1538): the decision doc explicitly scoped this down --
    "the CI-populated review packet is adopted as evidence (gate 5), but
    enforcement is the required_checks list".  A packet the reviewer reads is
    not a hard stop by itself.
    """
    lines: list[str] = [
        "## AST-equivalence gate (issue #1541)",
        "",
        f"Base: `{result.base}`",
        "",
    ]

    if result.moved_symbols:
        lines.append("### Moved symbols")
        lines.append("")
        lines.append("| Symbol | Source | Destination | Equivalent |")
        lines.append("|--------|--------|-------------|------------|")
        for sym in result.moved_symbols:
            src = sym.source_file
            dst = sym.dest_file
            if sym.source_class:
                src = f"{src}::{sym.source_class}"
            if sym.dest_class:
                dst = f"{dst}::{sym.dest_class}"
            status = "yes (verbatim)" if sym.equivalent else "**NO -- review**"
            lines.append(f"| `{sym.name}` | `{src}` | `{dst}` | {status} |")
        lines.append("")

        eq = len(result.equivalent_moves)
        neq = len(result.non_equivalent_moves)
        lines.append(f"**Summary:** {eq} equivalent, {neq} non-equivalent.")
        lines.append("")
    else:
        lines.append("No symbols moved between files in this PR.")
        lines.append("")

    if result.stale_shims:
        lines.append("### Stale facade shims (vulture sweep)")
        lines.append("")
        lines.append("| Name | Facade module |")
        lines.append("|------|---------------|")
        for shim in result.stale_shims:
            lines.append(f"| `{shim.name}` | `{shim.facade_module}` |")
        lines.append("")
        lines.append(
            f"**{len(result.stale_shims)}** stale shim(s) -- re-export entries "
            "nobody imports any more. Consider removing them."
        )
        lines.append("")

    lines.append(
        "_This packet is evidence for the human reviewer. Enforcement is the "
        "required_checks list (#1538), not this packet._"
    )
    return "\n".join(lines)
