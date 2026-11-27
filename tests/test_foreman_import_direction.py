"""Import-direction guard: the hub never imports ``charlie_work.foreman``.

ADR-0010: the foreman is a *client* of the hub — no hub-side module may
import the package. The sanctioned edges are the two CLI command modules —
``charlie_work.gang_command`` and ``charlie_work.scheme_command`` (#3122) —
whose handlers import the package *inside functions* so that ``import
charlie_work.cli`` never loads it.

The scan is AST-based (same approach as ``tests/test_no_path_literals.py``):
comments and docstrings are invisible, each import carries whether it sits
inside a function body, and the checker is exercised against synthetic
sources so it cannot silently lose teeth.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from _src_ast import parsed_source

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "charlie_work"
FOREMAN_PKG = SRC_ROOT / "foreman"

#: The modules allowed to import ``charlie_work.foreman`` — and only
#: inside a function body, so the package stays out of the CLI's import time.
#: ``gang_command`` owns the operator commands; ``scheme_command`` (#3122)
#: owns the foreman pass. The supervise loop is deliberately NOT here: it
#: spawns ``charlie scheme`` by argv and never imports the package.
_ALLOWED_IMPORTERS = frozenset({"gang_command.py", "scheme_command.py"})


@dataclass(frozen=True)
class Edge:
    filename: str
    lineno: int
    module: str
    in_function: bool

    def __str__(self) -> str:
        scope = "nested" if self.in_function else "top-level"
        return f"{self.filename}:{self.lineno}: {scope} import of {self.module}"


def _function_scopes(tree: ast.AST) -> set[int]:
    """``id()`` of every node inside a function body."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                ids.add(id(child))
    return ids


def _package_parts(filename: str) -> list[str]:
    """The containing package of ``filename`` under ``src/`` as dotted parts."""
    parts = list(Path(filename).parent.parts)
    return parts[1:] if parts and parts[0] == "src" else parts


def _resolve_from(filename: str, level: int, module: str | None) -> str:
    """Resolve a relative ``from ...`` to its absolute base module."""
    parts = _package_parts(filename)
    base = parts[: len(parts) - (level - 1)]
    return ".".join(base + (module.split(".") if module else []))


def _is_foreman_module(module: str) -> bool:
    return module == "charlie_work.foreman" or module.startswith("charlie_work.foreman.")


def find_foreman_edges(source: str, filename: str) -> list[Edge]:
    """Every import of ``charlie_work.foreman*`` in ``source``, with scope."""
    tree = parsed_source(source, filename)
    in_fn = _function_scopes(tree)
    edges: list[Edge] = []
    for node in ast.walk(tree):
        nested = id(node) in in_fn
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_foreman_module(alias.name):
                    edges.append(Edge(filename, node.lineno, alias.name, nested))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                base = _resolve_from(filename, node.level, node.module)
            # The statement may reach the package directly (``from .foreman
            # import store``) or via an alias (``from charlie_work import
            # foreman``). One edge per statement, naming what was reached.
            if _is_foreman_module(base):
                edges.append(Edge(filename, node.lineno, base, nested))
            else:
                for alias in node.names:
                    module = f"{base}.{alias.name}"
                    if _is_foreman_module(module):
                        edges.append(Edge(filename, node.lineno, module, nested))
                        break
    return edges


def find_violations(source: str, filename: str) -> list[Edge]:
    """Edges that break the rule: everything except the CLI seams' lazy imports."""
    basename = Path(filename).name
    return [
        edge
        for edge in find_foreman_edges(source, filename)
        if not (basename in _ALLOWED_IMPORTERS and edge.in_function)
    ]


def _scan_src() -> list[Edge]:
    violations: list[Edge] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if FOREMAN_PKG in path.parents or path.parent == FOREMAN_PKG:
            continue  # the package may import itself
        rel = path.relative_to(REPO_ROOT).as_posix()
        violations.extend(find_violations(path.read_text(encoding="utf-8"), rel))
    return violations


def test_no_module_outside_foreman_imports_the_package() -> None:
    violations = _scan_src()
    assert violations == [], (
        "ADR-0010: only the CLI command modules may import "
        "charlie_work.foreman, and only inside a function:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


@pytest.mark.parametrize("importer", sorted(_ALLOWED_IMPORTERS))
def test_command_module_edge_is_present_and_lazy(importer: str) -> None:
    """Each sanctioned seam must exist — and stay function-local so cli import stays cheap."""
    source = (SRC_ROOT / importer).read_text(encoding="utf-8")
    edges = find_foreman_edges(source, f"src/charlie_work/{importer}")
    assert edges, f"{importer} lost its foreman edge — the CLI cannot reach the package"
    assert all(e.in_function for e in edges), (
        f"{importer}'s foreman import must stay function-local: "
        + ", ".join(str(e) for e in edges if not e.in_function)
    )


def test_find_violations_flags_top_level_import() -> None:
    dirty = "import charlie_work.foreman.store\n"
    violations = find_violations(dirty, "src/charlie_work/anything.py")
    assert len(violations) == 1
    assert not violations[0].in_function


def test_find_violations_flags_relative_import_outside_gang() -> None:
    """A nested import is still a violation outside the one sanctioned seam."""
    dirty = "def f():\n    from .foreman import store\n"
    violations = find_violations(dirty, "src/charlie_work/state.py")
    assert len(violations) == 1
    assert violations[0].in_function


def test_find_violations_flags_relative_from_sibling_package() -> None:
    dirty = "from ..foreman import store\n"
    violations = find_violations(dirty, "src/charlie_work/dashboard/x.py")
    assert len(violations) == 1
    assert violations[0].module == "charlie_work.foreman"


def test_find_violations_flags_from_package_alias() -> None:
    dirty = "from charlie_work import foreman\n"
    violations = find_violations(dirty, "src/charlie_work/cli.py")
    assert len(violations) == 1


def test_find_violations_allows_gang_command_nested() -> None:
    clean = "def f():\n    from .foreman import store\n"
    assert find_violations(clean, "src/charlie_work/gang_command.py") == []


def test_find_violations_clean_source() -> None:
    clean = "from charlie_work import layout\nfrom . import state\n"
    assert find_violations(clean, "src/charlie_work/cli.py") == []
