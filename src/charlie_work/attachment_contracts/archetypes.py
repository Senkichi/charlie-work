"""AST archetype detection -> list[AttachmentPoint].

All detection is structural (AST shape), never a hand-maintained list of
module or class names. `scan_source` is a pure function over one file's text;
`scan_tree` walks `src/` and `tests/` under a repo root.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
from typing import Iterator

from charlie_work.attachment_contracts.excludes import Excludes
from charlie_work.attachment_contracts.ledger import classify_ledger
from charlie_work.attachment_contracts.model import AttachmentPoint, Kind, ScanResult

_ROUTE_VERBS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})


def _walk_dfs(node: ast.AST) -> Iterator[ast.AST]:
    """Deterministic pre-order walk (source order), unlike `ast.walk`'s BFS."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_dfs(child)


def _stem(path: str) -> str:
    return PurePosixPath(path).stem


def _decorator_base_attr(dec: ast.expr) -> tuple[str, str] | None:
    """For `@base.attr` or `@base.attr(...)`, return (base, attr); else None."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return None


def _call_func_attr(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_def(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _find_router_names(tree: ast.Module) -> dict[str, Kind]:
    """Variables assigned `typer.Typer()` / `click.Group()`, plus functions
    decorated `@x.group(...)` (click group defined via decorator)."""
    names: dict[str, Kind] = {}
    for node in _walk_dfs(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            attr = _call_func_attr(node.value)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if attr == "Typer":
                    names[target.id] = "typer_app"
                elif attr == "Group":
                    names[target.id] = "click_group"
        elif _is_def(node):
            for dec in node.decorator_list:
                ba = _decorator_base_attr(dec)
                if ba is not None and ba[1] == "group":
                    names.setdefault(node.name, "click_group")  # type: ignore[arg-type]
    return names


def _find_blueprint_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in _walk_dfs(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _call_func_attr(node.value) == "Blueprint":
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.append(target.id)
    return names


def _collect_decorated_members(
    tree: ast.Module, base_name: str, attrs: frozenset[str]
) -> list[str]:
    members: list[str] = []
    for node in _walk_dfs(tree):
        if not _is_def(node):
            continue
        for dec in node.decorator_list:
            ba = _decorator_base_attr(dec)
            if ba is not None and ba[0] == base_name and ba[1] in attrs:
                members.append(node.name)
                break
    return members


def _typer_click_points(tree: ast.Module, path: str) -> list[AttachmentPoint]:
    points: list[AttachmentPoint] = []
    for name, kind in _find_router_names(tree).items():
        members = _collect_decorated_members(tree, name, frozenset({"command"}))
        points.append(
            AttachmentPoint(
                kind=kind, identity=f"{_stem(path)}:{name}", file=path, members=tuple(members)
            )
        )
    return points


def _blueprint_points(tree: ast.Module, path: str) -> list[AttachmentPoint]:
    points: list[AttachmentPoint] = []
    verb_attrs = _ROUTE_VERBS | {"route"}
    for name in _find_blueprint_names(tree):
        members = _collect_decorated_members(tree, name, verb_attrs)
        points.append(
            AttachmentPoint(
                kind="blueprint",
                identity=f"{_stem(path)}:{name}",
                file=path,
                members=tuple(members),
            )
        )
    return points


def _class_points(tree: ast.Module, path: str) -> list[AttachmentPoint]:
    points: list[AttachmentPoint] = []
    for node in _walk_dfs(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        members = tuple(child.name for child in node.body if _is_def(child))
        if classify_ledger(members):
            points.append(
                AttachmentPoint(
                    kind="migration_runner",
                    identity=node.name,
                    file=path,
                    members=members,
                    is_linear_ledger=True,
                )
            )
        else:
            points.append(
                AttachmentPoint(kind="class", identity=node.name, file=path, members=members)
            )
    return points


def _module_ledger_points(tree: ast.Module, path: str) -> list[AttachmentPoint]:
    top_level = tuple(node.name for node in tree.body if _is_def(node))
    if classify_ledger(top_level):
        identity = f"module:{_stem(path)}"
        return [
            AttachmentPoint(
                kind="migration_runner",
                identity=identity,
                file=path,
                members=top_level,
                is_linear_ledger=True,
            )
        ]
    return []


def _looks_like_test_module(path: str) -> bool:
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return False
    stem = p.stem
    return stem.startswith("test_") or stem.endswith("_test")


def _test_module_point(tree: ast.Module, path: str) -> AttachmentPoint:
    members: list[str] = []
    for node in tree.body:
        if _is_def(node) and node.name.startswith("test_"):
            members.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            members.append(node.name)
    return AttachmentPoint(
        kind="test_module", identity=f"{path}::module", file=path, members=tuple(members)
    )


def scan_source(text: str, path: str) -> list[AttachmentPoint]:
    """Detect all attachment points in one file's source text.

    Raises `SyntaxError` on unparseable source; callers (`scan_tree`) route
    that into `ScanResult.parse_failures` (G6 — never silently dropped).
    """
    tree = ast.parse(text)
    points: list[AttachmentPoint] = []
    points.extend(_typer_click_points(tree, path))
    points.extend(_blueprint_points(tree, path))
    points.extend(_class_points(tree, path))
    points.extend(_module_ledger_points(tree, path))
    if _looks_like_test_module(path):
        points.append(_test_module_point(tree, path))
    return points


def scan_tree(root: Path, excludes: Excludes) -> ScanResult:
    """Walk `src/` and `tests/` under `root`, honoring `excludes`.

    Returns points sorted by (file, identity) for deterministic output.
    """
    points: list[AttachmentPoint] = []
    parse_failures: list[str] = []
    for base_name in ("src", "tests"):
        base_dir = root / base_name
        if not base_dir.is_dir():
            continue
        for file_path in sorted(base_dir.rglob("*.py")):
            rel = file_path.relative_to(root).as_posix()
            if excludes.is_excluded_path(rel):
                continue
            text = file_path.read_text(encoding="utf-8")
            try:
                points.extend(scan_source(text, rel))
            except SyntaxError:
                parse_failures.append(rel)
    points.sort(key=lambda p: (p.file, p.identity))
    return ScanResult(
        root=str(root), points=tuple(points), parse_failures=tuple(sorted(parse_failures))
    )
