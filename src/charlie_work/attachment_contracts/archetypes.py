"""AST archetype detection -> list[AttachmentPoint].

All detection is structural (AST shape), never a hand-maintained list of
module or class names. `scan_source` is a pure function over one file's text;
`scan_tree` walks `src/` and `tests/` under a repo root.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping

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


def _iter_classdefs(
    node: ast.AST, scope: tuple[str, ...] = ()
) -> Iterator[tuple[ast.ClassDef, str]]:
    """Yield (class_node, identity) for every `ClassDef` reachable from
    `node`, deterministic pre-order.

    A top-level class keeps a bare-name identity (`ClassName`) for baseline
    stability. A class nested inside a function or another class gets its
    enclosing scope path prefixed (`enclosing_fn.ClassName`,
    `Outer.Inner`, ...) -- structurally derived from the AST, not a
    hand-maintained list. Without this, two same-named local fixture classes
    defined in different test functions (extremely common in this repo's
    test suite, e.g. many `class FakeGitHub:` bodies scoped to different
    tests) collide on identity: baseline.py keys entries by (kind, file,
    identity), so a bare-name collision silently drops one on ratchet
    writeback.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            identity = ".".join((*scope, child.name)) if scope else child.name
            yield child, identity
            yield from _iter_classdefs(child, (*scope, child.name))
        elif _is_def(child):
            yield from _iter_classdefs(child, (*scope, child.name))
        else:
            yield from _iter_classdefs(child, scope)


def _class_points(tree: ast.Module, path: str) -> list[AttachmentPoint]:
    points: list[AttachmentPoint] = []
    for node, identity in _iter_classdefs(tree):
        members = tuple(child.name for child in node.body if _is_def(child))
        if classify_ledger(members):
            points.append(
                AttachmentPoint(
                    kind="migration_runner",
                    identity=identity,
                    file=path,
                    members=members,
                    is_linear_ledger=True,
                )
            )
        else:
            points.append(
                AttachmentPoint(kind="class", identity=identity, file=path, members=members)
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


def iter_source_files(root: Path, excludes: Excludes) -> tuple[str, ...]:
    """Repo-relative posix paths of every `.py` file under `src/` and
    `tests/` that `scan_tree` would visit, honoring `excludes`.

    Exposed separately from `scan_tree` so a caller that needs the full
    scanned-file universe -- not just the files that happened to produce an
    `AttachmentPoint` -- doesn't have to duplicate the walk (e.g. the G1
    backtest's Cluster-B probe needs to see files with NO archetype match at
    all, which never appear in `ScanResult.points`).
    """
    files: list[str] = []
    for base_name in ("src", "tests"):
        base_dir = root / base_name
        if not base_dir.is_dir():
            continue
        for file_path in sorted(base_dir.rglob("*.py")):
            rel = file_path.relative_to(root).as_posix()
            if excludes.is_excluded_path(rel):
                continue
            files.append(rel)
    return tuple(sorted(files))


def scan_tree(
    root: Path,
    excludes: Excludes,
    *,
    content_overrides: Mapping[str, str] | None = None,
) -> ScanResult:
    """Walk `src/` and `tests/` under `root`, honoring `excludes`.

    `content_overrides` maps a repo-relative posix path to source text that
    should be scanned INSTEAD of the file's current on-disk content -- used
    by the PreToolUse hook to evaluate a pending Write/Edit/MultiEdit's
    proposed content rather than the stale pre-edit file (see
    `hook_entry._compute_proposed_content`). A path not present in
    `content_overrides` is read from disk as usual.

    Returns points sorted by (file, identity) for deterministic output.
    """
    overrides = content_overrides or {}
    points: list[AttachmentPoint] = []
    parse_failures: list[str] = []
    for rel in iter_source_files(root, excludes):
        try:
            text = overrides[rel] if rel in overrides else (root / rel).read_text(
                encoding="utf-8"
            )
            points.extend(scan_source(text, rel))
        except (SyntaxError, UnicodeDecodeError, OSError):
            # Unparseable source, undecodable bytes, or an unreadable file
            # (permission / race with an editor / dangling path) all route
            # into parse_failures (G6 -- never silently dropped). The read is
            # inside this try, not just the parse: a decode or OS failure on
            # the read must not crash the scan any more than a SyntaxError
            # does.
            parse_failures.append(rel)
    points.sort(key=lambda p: (p.file, p.identity))
    return ScanResult(
        root=str(root), points=tuple(points), parse_failures=tuple(sorted(parse_failures))
    )
