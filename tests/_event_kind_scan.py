"""AST-extraction machinery for the event-kind consumer scanner.

The AST literal resolver, emit-site scanner, marker parser and consumer
detector that ``tests/test_event_kind_consumers.py``'s assertions run on live
here, under the ``tests/_*.py`` hoisted-helper convention (the sanctioned
way for test modules to share code; the zero-cross-test-import guard bans
one ``test_*.py`` module importing another). The machinery moved out of the
guard's monolith when it grew past its file-size ratchet high-water mark, so
the guarding assertions could keep pace without growing the module. No
pytest dependency, and no assertions: the assertions all live in the test
module.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from _dws_emit_scan import is_decide_module, sweep_emit_kind
from _src_ast import parsed, source_text
from _event_kind_resolve import (
    _collect_local_assignments,
    _collect_module_functions,
    _collect_param_names,
    _is_kind_field_access,
    _resolve_collection,
    _resolve_literal,
)

# The caller module this scan machinery serves: its own docstrings/assertions
# name the kinds it discusses, so `_analyze` excludes its literals from the
# test-consumed set it feeds report consumers.
_THIS_FILE_NAME = "test_event_kind_consumers.py"


# The god-object paydown (Track 2 Phase B) relocates OrchestratorApp method
# bodies -- event emitters among them -- into the ``orchestration/`` delegate
# subpackage, and nothing stops the next ``src/charlie_work/`` subdirectory
# from doing the same. The scanned set is therefore DERIVED from the package
# tree -- the top level plus every directory beneath it that is a package
# (contains ``__init__.py``), recursively -- never declared: a new subpackage's
# emit sites enter ``emitted_kinds`` automatically, so an unscanned file cannot
# sit silently outside every assertion below (the #1671 blind spot, generalized
# by #1687). Everything downstream keys files by path relative to ``src_root``
# (POSIX form), not by basename: ``checks.py``/``labels.py``/``__main__.py``
# collide across subpackages, and a basename key would misattribute a marker or
# an emit site to the wrong file with no error -- the reason the scan could not
# widen until the key changed.
def _src_py_files(src_root: Path) -> list[Path]:
    """Every ``*.py`` in the package tree under ``src_root``, sorted.

    ``src_root`` itself always contributes its top-level ``*.py`` (it IS the
    ``charlie_work`` package); a descendant directory contributes only when it
    is a package (``__init__.py`` present), which is also the only kind of
    directory worth descending into.
    """
    files = list(src_root.glob("*.py"))
    pending = [src_root]
    while pending:
        current = pending.pop()
        for child in current.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                files.extend(child.glob("*.py"))
                pending.append(child)
    return sorted(files)


# ---------------------------------------------------------------------------
# The three sanctioned emitters (per the issue) plus the shapes that share
# their exact forwarding signature in practice: ``WriteGate`` mirrors all
# three under the same method names (``append_event``, ``log_event``, and
# ``record_event`` for the ``_record_event`` shape) so a call site that has
# migrated onto the dry-run gate is still scanned; ``_route_to_rework`` is a
# named wrapper with its own (different) signature, forwarding through its
# 5th positional argument / ``event_kind`` keyword.
# ---------------------------------------------------------------------------
EMIT_FUNC_NAMES: frozenset[str] = frozenset(
    {"append_event", "log_event", "_record_event", "record_event"}
)
KIND_ARG_SPEC = (1, "kind")  # (positional index, keyword name) for the four names above
WRAPPER_FUNC_SPECS: dict[str, tuple[int, str]] = {
    "_route_to_rework": (4, "event_kind"),
}
ALL_EMIT_LIKE_NAMES: frozenset[str] = EMIT_FUNC_NAMES | frozenset(WRAPPER_FUNC_SPECS)

# The batching path in stalled_review_reap.py builds a list of
# ``(literal_kind, payload)`` tuples via ``sweep_events.append(...)`` and only
# later flushes them through ``write_gate.append_event`` inside
# ``_append_sweep_events`` -- by the time that call fires, ``kind`` is a loop
# variable, not a literal. The literal information lives at the *build* site,
# so it is harvested there directly rather than lost. This is scoped to the
# exact receiver name the codebase uses for this pattern (not a per-kind
# enumeration), so a new kind added the same way is picked up automatically.
SWEEP_APPEND_RECEIVER = "sweep_events"

MARKER_PREFIX = "# event-consumer:"
_MARKER_RE = re.compile(re.escape(MARKER_PREFIX) + r"\s*(?P<body>.+?)\s*$")
_PENDING_RE = re.compile(r"^pending(?:\s+#(?P<num>\d+))?\b")

# Issue #2886: the per-file emit walk is the scanner's dominant cost -- it visits
# every AST node of every package file and re-walks each function body it
# enters. A file whose text cannot contain a flagged site skips the walk:
# every site the walk can report names one of the tokens below in its own
# source (a flagged call's func name, the sweep receiver, or an
# ``# event-consumer:`` marker), so an absence yields zero sites.
# Over-matching is free -- the walk merely runs on the file. The name tokens
# are derived from the scanner's own flag tables so a future form added there
# enters this prefilter automatically; the marker token keeps every
# marker-bearing file in the walked set so the orphan check in ``_analyze``
# sees the same file set it always did; and decide modules are always walked
# because their keyword forms (``event_kind=``, ``event=(kind, payload)``)
# are not cheaply reducible to a token.
_PRESENCE_TOKENS: frozenset[str] = frozenset(
    [*ALL_EMIT_LIKE_NAMES, SWEEP_APPEND_RECEIVER, MARKER_PREFIX]
)


def needs_emit_walk(rel_path: str, text: str) -> bool:
    """Whether ``_scan_emit_sites_in_tree`` can find a site in a file this shape."""
    if is_decide_module(rel_path):
        return True
    return any(token in text for token in _PRESENCE_TOKENS)


def _locate_arg(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _call_func_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in ALL_EMIT_LIKE_NAMES:
        return call.func.id
    if isinstance(call.func, ast.Attribute) and call.func.attr in ALL_EMIT_LIKE_NAMES:
        return call.func.attr
    return None


# ---------------------------------------------------------------------------
# Emission-site scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmitSite:
    path: str  # POSIX-relative path within the scanned root
    scope: str  # enclosing function/method name, or "<module>"
    lineno: int
    end_lineno: int
    func_name: str
    kind_literal: frozenset[str] | None  # None => dynamic (unresolved)
    kind_source: str  # ast.unparse() of the kind expression, for messages


def _is_structural_forwarding(
    kind_node: ast.expr, local_params: frozenset[str], scope_name: str
) -> bool:
    """True for a call site that only forwards its OWN kind-shaped parameter.

    This is the exact shape ``state.append_event`` -> ``log_event``,
    ``OrchestratorApp._record_event`` -> ``append_event``, ``_route_to_rework``
    -> ``self._record_event``, and every ``WriteGate`` method -> its
    underlying primitive share: the enclosing function IS itself one of the
    sanctioned emitter/wrapper names, and the "dynamic" kind argument is
    nothing but that function's own parameter. The real literal is chosen at
    every *caller* of this function, which is separately scanned (this
    function's own name is itself in ``ALL_EMIT_LIKE_NAMES``/
    ``WRAPPER_FUNC_SPECS``) -- so this site needs no marker of its own.
    """
    return (
        isinstance(kind_node, ast.Name)
        and kind_node.id in local_params
        and scope_name in ALL_EMIT_LIKE_NAMES
    )


def _scan_emit_sites_in_tree(
    tree: ast.Module, rel_path: str
) -> tuple[
    list[EmitSite], dict[str, list[ast.expr]], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
]:
    module_local = _collect_local_assignments(tree)
    module_funcs = _collect_module_functions(tree)
    module_constants: dict[str, set[str]] = {}
    for name, value_nodes in module_local.items():
        parts: list[set[str]] = []
        ok = True
        for value_node in value_nodes:
            part = _resolve_literal(value_node, module_local, {}, frozenset(), module_funcs)
            if part is None:
                ok = False
                break
            parts.append(part)
        if ok and parts:
            module_constants[name] = set().union(*parts)

    sites: list[EmitSite] = []
    in_decide = is_decide_module(rel_path)

    def scan(
        node: ast.AST,
        local_assigns: dict[str, list[ast.expr]],
        local_params: frozenset[str],
        scope_name: str,
    ) -> None:
        sweep_site = sweep_emit_kind(node, rel_path.rpartition("/")[2]) if in_decide else None
        if sweep_site is not None:
            # The literal kind is chosen here; the apply shell only forwards it.
            _record_site(
                node,
                sweep_site[0],
                "emit" if isinstance(node, ast.Call) else "event_kind",
                local_assigns,
                local_params,
                module_constants,
                module_funcs,
                scope_name,
                rel_path,
                sites,
            )
        elif isinstance(node, ast.Call):
            fname = _call_func_name(node)
            if fname in EMIT_FUNC_NAMES:
                pos, kw = KIND_ARG_SPEC
                kind_node = _locate_arg(node, pos, kw)
                _record_site(
                    node,
                    kind_node,
                    fname,
                    local_assigns,
                    local_params,
                    module_constants,
                    module_funcs,
                    scope_name,
                    rel_path,
                    sites,
                )
            elif fname in WRAPPER_FUNC_SPECS:
                pos, kw = WRAPPER_FUNC_SPECS[fname]
                kind_node = _locate_arg(node, pos, kw)
                _record_site(
                    node,
                    kind_node,
                    fname,
                    local_assigns,
                    local_params,
                    module_constants,
                    module_funcs,
                    scope_name,
                    rel_path,
                    sites,
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == SWEEP_APPEND_RECEIVER
                and node.args
                and isinstance(node.args[0], ast.Tuple)
                and node.args[0].elts
            ):
                first = node.args[0].elts[0]
                literal = _resolve_literal(
                    first, local_assigns, module_constants, local_params, module_funcs
                )
                sites.append(
                    EmitSite(
                        path=rel_path,
                        scope=scope_name,
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        func_name="sweep_events.append",
                        kind_literal=frozenset(literal) if literal else None,
                        kind_source=ast.unparse(first),
                    )
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scoped_assigns = _collect_local_assignments(node)
            scoped_params = _collect_param_names(node)
            for child in ast.iter_child_nodes(node):
                scan(child, scoped_assigns, scoped_params, node.name)
            return
        for child in ast.iter_child_nodes(node):
            scan(child, local_assigns, local_params, scope_name)

    scan(tree, module_local, frozenset(), "<module>")
    return sites, module_local, module_funcs


def _record_site(
    call: ast.Call | ast.keyword,
    kind_node: ast.expr | None,
    func_name: str,
    local_assigns: dict[str, list[ast.expr]],
    local_params: frozenset[str],
    module_constants: dict[str, set[str]],
    module_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    scope_name: str,
    rel_path: str,
    sites: list[EmitSite],
) -> None:
    if kind_node is None:
        sites.append(
            EmitSite(
                path=rel_path,
                scope=scope_name,
                lineno=call.lineno,
                end_lineno=call.end_lineno or call.lineno,
                func_name=func_name,
                kind_literal=None,
                kind_source="<no kind argument located>",
            )
        )
        return
    if _is_structural_forwarding(kind_node, local_params, scope_name):
        # Not a real per-kind emission site -- see _is_structural_forwarding.
        # Deliberately NOT added to `sites` at all: it needs neither a
        # literal resolution nor a marker, since every real caller of this
        # function is itself scanned.
        return
    literal = _resolve_literal(
        kind_node, local_assigns, module_constants, local_params, module_funcs
    )
    sites.append(
        EmitSite(
            path=rel_path,
            scope=scope_name,
            lineno=call.lineno,
            end_lineno=call.end_lineno or call.lineno,
            func_name=func_name,
            kind_literal=frozenset(literal) if literal else None,
            kind_source=ast.unparse(kind_node),
        )
    )


def _scan_emit_sites(root: Path) -> list[EmitSite]:
    sites: list[EmitSite] = []
    for path in _src_py_files(root):
        rel_path = path.relative_to(root).as_posix()
        if not needs_emit_walk(rel_path, source_text(path)):
            continue
        try:
            tree = parsed(path)
        except SyntaxError:
            continue
        file_sites, _, _ = _scan_emit_sites_in_tree(tree, rel_path)
        sites.extend(file_sites)
    return sites


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Marker:
    kind: str  # "audit-only" | "pending" | "pointer"
    lineno: int
    raw: str
    pending_issue: int | None = None
    valid: bool = True
    error: str = ""


def _parse_marker(line: str, lineno: int) -> Marker | None:
    """Parse a single source line for an ``# event-consumer:`` marker.

    Returns ``None`` if the line carries no marker at all. An invalid marker
    (bare ``pending`` with no issue number, or a blank ``audit-only`` with no
    justification clause) is still returned, with ``valid=False`` -- callers
    surface ``error`` rather than silently ignoring it.
    """
    m = _MARKER_RE.search(line)
    if m is None:
        return None
    body = m.group("body")
    if body.startswith("audit-only"):
        justification = body[len("audit-only") :].strip(" -#\t")
        if not justification:
            return Marker(
                kind="audit-only",
                lineno=lineno,
                raw=body,
                valid=False,
                error="`audit-only` marker has no justification clause (blanket marking is not allowed)",
            )
        return Marker(kind="audit-only", lineno=lineno, raw=body)
    if body.startswith("pending"):
        pm = _PENDING_RE.match(body)
        if pm is None or pm.group("num") is None:
            return Marker(
                kind="pending",
                lineno=lineno,
                raw=body,
                valid=False,
                error="`pending` marker has no issue number (use `pending #NNNN`)",
            )
        return Marker(kind="pending", lineno=lineno, raw=body, pending_issue=int(pm.group("num")))
    # Anything else is a pointer to an external/unseen consumer.
    return Marker(kind="pointer", lineno=lineno, raw=body)


@dataclass(frozen=True)
class FileMarkers:
    by_line: dict[int, Marker] = field(default_factory=dict)


def _scan_markers(path: Path) -> FileMarkers:
    by_line: dict[int, Marker] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        marker = _parse_marker(line, i)
        if marker is not None:
            by_line[i] = marker
    return FileMarkers(by_line=by_line)


def _marker_for_site(markers: FileMarkers, site: EmitSite) -> Marker | None:
    for lineno in range(site.lineno, site.end_lineno + 1):
        if lineno in markers.by_line:
            return markers.by_line[lineno]
    return None


def _orphan_markers(markers: FileMarkers, sites: list[EmitSite]) -> list[int]:
    """Marker line numbers that decorate no real emission site in this file."""
    covered: set[int] = set()
    for site in sites:
        covered.update(range(site.lineno, site.end_lineno + 1))
    return [lineno for lineno in markers.by_line if lineno not in covered]


# ---------------------------------------------------------------------------
# Consumer detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerSite:
    kind: str
    path: str
    scope: str
    category: str  # "src" | "heartbeat" | "test"


def _collect_src_consumer_sites(tree: ast.Module, rel_path: str) -> list[ConsumerSite]:
    module_local = _collect_local_assignments(tree)
    module_funcs = _collect_module_functions(tree)
    found: list[ConsumerSite] = []

    def scan(node: ast.AST, local_assigns: dict[str, list[ast.expr]], scope_name: str) -> None:
        if isinstance(node, ast.Call):
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname == "query_events":
                for kw in node.keywords:
                    if kw.arg == "kind":
                        lit = _resolve_literal(
                            kw.value, local_assigns, {}, frozenset(), module_funcs
                        )
                        if lit:
                            found.extend(
                                ConsumerSite(
                                    kind=k, path=rel_path, scope=scope_name, category="src"
                                )
                                for k in lit
                            )
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            kindish = [o for o in operands if _is_kind_field_access(o)]
            others = [o for o in operands if not _is_kind_field_access(o)]
            if kindish:
                for other in others:
                    lit: set[str] | None
                    if isinstance(other, ast.Constant) and isinstance(other.value, str):
                        lit = {other.value}
                    else:
                        lit = _resolve_collection(
                            other, local_assigns, module_funcs
                        ) or _resolve_collection(other, module_local, module_funcs)
                    if lit:
                        found.extend(
                            ConsumerSite(kind=k, path=rel_path, scope=scope_name, category="src")
                            for k in lit
                        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scoped_assigns = _collect_local_assignments(node)
            for child in ast.iter_child_nodes(node):
                scan(child, scoped_assigns, node.name)
            return
        for child in ast.iter_child_nodes(node):
            scan(child, local_assigns, scope_name)

    scan(tree, module_local, "<module>")
    return found


def _collect_literal_strings(tree: ast.Module) -> set[str]:
    return {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


# ---------------------------------------------------------------------------
# The assembled analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnaccountedKind:
    kind: str
    sites: tuple[EmitSite, ...]


@dataclass(frozen=True)
class UnmarkedDynamicSite:
    site: EmitSite


@dataclass(frozen=True)
class PendingEntry:
    kind: str
    issue: int


@dataclass(frozen=True)
class Report:
    emitted_kinds: frozenset[str]
    emit_sites: tuple[EmitSite, ...]
    unaccounted: tuple[UnaccountedKind, ...]  # literal kind, no consumer, no marker anywhere
    unmarked_dynamic: tuple[UnmarkedDynamicSite, ...]  # dynamic kind, no marker on that site
    orphan_markers: tuple[tuple[str, int], ...]  # (path, lineno) marker decorating nothing
    invalid_markers: tuple[tuple[str, int, str], ...]  # (path, lineno, error)
    pending: tuple[PendingEntry, ...]
    test_only_consumed: frozenset[str]
    heartbeat_literals: frozenset[str]


def _analyze(
    src_root: Path,
    tests_root: Path | None,
    heartbeat_paths: Path | Sequence[Path] | None,
) -> Report:
    emit_sites: list[EmitSite] = []
    file_markers: dict[str, FileMarkers] = {}
    file_sites: dict[str, list[EmitSite]] = {}
    for path in _src_py_files(src_root):
        rel_path = path.relative_to(src_root).as_posix()
        try:
            tree = parsed(path)
        except SyntaxError:
            continue
        if needs_emit_walk(rel_path, source_text(path)):
            sites, _, _ = _scan_emit_sites_in_tree(tree, rel_path)
        else:
            # No flagged form in the text, so no site can hide here (see
            # ``needs_emit_walk``); the marker scan below still runs so the
            # orphan-marker accounting behaves exactly as before.
            sites = []
        emit_sites.extend(sites)
        file_sites[rel_path] = sites
        file_markers[rel_path] = _scan_markers(path)

    emitted_kinds: set[str] = set()
    emit_scope_by_kind: dict[str, set[tuple[str, str]]] = {}
    dynamic_sites: list[EmitSite] = []
    for site in emit_sites:
        if site.kind_literal is not None:
            emitted_kinds |= set(site.kind_literal)
            for k in site.kind_literal:
                emit_scope_by_kind.setdefault(k, set()).add((site.path, site.scope))
        else:
            dynamic_sites.append(site)

    # Consumer collection.
    consumer_sites: list[ConsumerSite] = []
    for path in _src_py_files(src_root):
        rel_path = path.relative_to(src_root).as_posix()
        try:
            tree = parsed(path)
        except SyntaxError:
            continue
        consumer_sites.extend(_collect_src_consumer_sites(tree, rel_path))

    if heartbeat_paths is None:
        hb_paths: tuple[Path, ...] = ()
    elif isinstance(heartbeat_paths, Path):
        hb_paths = (heartbeat_paths,)
    else:
        hb_paths = tuple(heartbeat_paths)

    heartbeat_literals: set[str] = set()
    for hb_path in hb_paths:
        if hb_path.is_file():
            hb_tree = parsed(hb_path)
            heartbeat_literals |= _collect_literal_strings(hb_tree)

    test_literals: set[str] = set()
    if tests_root is not None and tests_root.is_dir():
        for path in sorted(tests_root.glob("test_*.py")):
            if path.name == _THIS_FILE_NAME:
                # This module's own docstrings/assert-messages necessarily
                # name the very kinds it discusses (see the module docstring
                # and the assertions below) -- counting THAT as "a test
                # consumes this kind" is exactly the false-signal bug this
                # issue exists to prevent, just turned on itself. Every other
                # test file is fair game (a real assertion elsewhere).
                continue
            try:
                tree = parsed(path)
            except SyntaxError:
                continue
            test_literals |= _collect_literal_strings(tree)

    consumer_scope_by_kind: dict[str, set[tuple[str, str]]] = {}
    for cs in consumer_sites:
        consumer_scope_by_kind.setdefault(cs.kind, set()).add((cs.path, cs.scope))

    def is_consumed(kind: str) -> bool:
        if kind in heartbeat_literals or kind in test_literals:
            return True
        candidates = consumer_scope_by_kind.get(kind, set())
        self_sites = emit_scope_by_kind.get(kind, set())
        # Self-match guard: a "consumer" occurrence in the exact same
        # (file, function) as the kind's own emission proves nothing --
        # it's the emission's own bookkeeping, not a downstream reader.
        remaining = candidates - self_sites
        return bool(remaining)

    test_only_consumed = {
        k
        for k in emitted_kinds
        if k in test_literals
        and k not in heartbeat_literals
        and not (consumer_scope_by_kind.get(k, set()) - emit_scope_by_kind.get(k, set()))
    }

    unaccounted: list[UnaccountedKind] = []
    pending: list[PendingEntry] = []
    for kind in sorted(emitted_kinds):
        if is_consumed(kind):
            continue
        sites_for_kind = tuple(s for s in emit_sites if s.kind_literal and kind in s.kind_literal)
        marker = None
        for site in sites_for_kind:
            marker = _marker_for_site(file_markers[site.path], site)
            if marker is not None and marker.valid:
                break
        else:
            marker = None
        if marker is not None and marker.valid:
            if marker.kind == "pending" and marker.pending_issue is not None:
                pending.append(PendingEntry(kind=kind, issue=marker.pending_issue))
            continue
        unaccounted.append(UnaccountedKind(kind=kind, sites=sites_for_kind))

    unmarked_dynamic: list[UnmarkedDynamicSite] = []
    for site in dynamic_sites:
        marker = _marker_for_site(file_markers[site.path], site)
        if marker is None or not marker.valid:
            unmarked_dynamic.append(UnmarkedDynamicSite(site=site))

    orphan_markers: list[tuple[str, int]] = []
    invalid_markers: list[tuple[str, int, str]] = []
    for path_name, markers in file_markers.items():
        for lineno, marker in markers.by_line.items():
            if not marker.valid:
                invalid_markers.append((path_name, lineno, marker.error))
        for lineno in _orphan_markers(markers, file_sites[path_name]):
            orphan_markers.append((path_name, lineno))

    return Report(
        emitted_kinds=frozenset(emitted_kinds),
        emit_sites=tuple(emit_sites),
        unaccounted=tuple(unaccounted),
        unmarked_dynamic=tuple(unmarked_dynamic),
        orphan_markers=tuple(orphan_markers),
        invalid_markers=tuple(invalid_markers),
        pending=tuple(pending),
        test_only_consumed=frozenset(test_only_consumed),
        heartbeat_literals=frozenset(heartbeat_literals),
    )
