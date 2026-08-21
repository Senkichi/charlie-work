"""Issue #1364: every emitted event ``kind`` is either consumed or declared audit-only.

The single most recurrent defect class in this codebase's review history is
*signal without a consumer*: an event kind gets emitted (into ``state.json``'s
ring and ``events.db``) and nothing ever reads it. The signal exists, so the
author and reviewer both believe the condition is "handled"; in reality it is
write-only. Standing review guidance is "grep per EVENT KIND, not per
subsystem" -- this test applies that check structurally, once, the same way
``tests/test_dormant_fleet_marking.py`` derives the dormant-module set from
the import graph instead of hand-maintaining it.

Two sets are built by static analysis (AST, not regex -- kinds are string
literals but call *shapes* vary):

**Emitted kinds** -- every string-literal ``kind``/``event_kind`` argument
passed to the sanctioned emitters (``append_event``, ``log_event``,
``OrchestratorApp._record_event``, the ``WriteGate`` equivalents that share
their name, and the ``_route_to_rework`` wrapper), across every module under
``src/charlie_work``. A call site whose kind argument cannot be reduced to a
literal string is a **failure** unless the emitting line carries a
``# event-consumer:`` marker (see below) -- a variable or f-string kind is
exactly how a kind escapes the registry unnoticed.

**Consumed kinds** -- a kind counts as consumed if its literal string turns
up in a read position: a ``query_events(kind=...)`` call, an equality/
membership comparison against an event's ``kind`` field (including via a
dict/frozenset classification table), anywhere in ``scripts/heartbeat_check.py``,
or anywhere under ``tests/`` (a weak, test-only consumer, reported separately
so it can be upgraded deliberately).

**The escape hatch** is declared at the emission site, never in a
hand-maintained central list (which would itself rot): a trailing comment
``# event-consumer: audit-only <justification>``, ``# event-consumer:
<pointer>``, or ``# event-consumer: pending #NNNN``. The marker's own source
line is read directly (line-based, but only over a *comment*, never over code
structure) against the AST call site it decorates -- a marker that decorates
no real emission site is orphaned and fails the same as a missing one.

Baseline handling (issue's own explicit instruction): the backlog this test
surfaced was NOT mass-marked audit-only. Every one of the 13 ``audit-only``
markers below carries a one-clause justification tied to the surrounding
code (usually: the real state mutation already happened inline, or a sibling
kind is the actionable one). The two kinds that looked like they should have
a real consumer -- ``draft_pr_blocked`` (the code's own comment names the
consumer it doesn't have yet) and ``venv_editable_anchor_violation`` (a hard
supervisor-refusal safety gate with no confirmed alerting path) -- carry
``pending #1366`` markers, pointing at the grouped tracking issue filed
alongside this PR (see ``pending-kinds-inventory.md`` at the repo root for
the derived inventory and the two kinds' detail).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "charlie_work"
TESTS_DIR = REPO_ROOT / "tests"
HEARTBEAT = REPO_ROOT / "scripts" / "heartbeat_check.py"
_THIS_FILE_NAME = Path(__file__).name

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


# ---------------------------------------------------------------------------
# AST scope/assignment helpers -- mirrors the proven approach in
# tests/test_instrumentation.py's _resolve_literal (a different concern,
# registry completeness, built to solve the identical "reduce this
# expression to a finite literal string set, or admit failure" problem).
# Kept as an independent copy here rather than a cross-module import so this
# test's correctness never depends on that file's private internals.
# ---------------------------------------------------------------------------


def _is_scope_boundary(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))


def _collect_local_assignments(scope: ast.AST) -> dict[str, list[ast.expr]]:
    assigns: dict[str, list[ast.expr]] = {}

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_scope_boundary(child):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(child.value)
            walk(child)

    walk(scope)
    return assigns


def _collect_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return frozenset(names)


def _collect_module_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _resolve_literal(
    node: ast.expr,
    local_assigns: dict[str, list[ast.expr]],
    module_constants: dict[str, set[str]],
    local_params: frozenset[str],
    module_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    _depth: int = 0,
) -> set[str] | None:
    """Best-effort reduction of ``node`` to the finite set of strings it can be.

    ``None`` means "cannot prove this is a literal string set" -- callers must
    check ``is None``, never falsiness (an empty-but-resolved set is never
    produced by this function; every branch below is non-empty by
    construction or falls through to ``None``).
    """
    if _depth > 12:
        return None
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else None
    if isinstance(node, ast.IfExp):
        body = _resolve_literal(
            node.body, local_assigns, module_constants, local_params, module_funcs, _depth + 1
        )
        orelse = _resolve_literal(
            node.orelse, local_assigns, module_constants, local_params, module_funcs, _depth + 1
        )
        if body is None or orelse is None:
            return None
        return body | orelse
    if isinstance(node, ast.JoinedStr):
        parts: list[set[str]] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append({value.value})
                continue
            if (
                isinstance(value, ast.FormattedValue)
                and value.format_spec is None
                and value.conversion in (-1, ord("s"))
            ):
                resolved = _resolve_literal(
                    value.value,
                    local_assigns,
                    module_constants,
                    local_params,
                    module_funcs,
                    _depth + 1,
                )
                if resolved is not None:
                    parts.append(resolved)
                    continue
            return None
        combined = {""}
        for part in parts:
            combined = {p + s for p in combined for s in part}
        return combined or None
    if isinstance(node, ast.Name):
        if node.id in local_params:
            # A function parameter is caller-controlled; some path could
            # carry the un-reassigned value through, so treat conservatively
            # as unresolved rather than trusting a partial local shadow.
            return None
        if node.id in local_assigns:
            values: set[str] = set()
            for value_node in local_assigns[node.id]:
                resolved = _resolve_literal(
                    value_node,
                    local_assigns,
                    module_constants,
                    local_params,
                    module_funcs,
                    _depth + 1,
                )
                if resolved is None:
                    return None
                values |= resolved
            return values or None
        if node.id in module_constants:
            return module_constants[node.id] or None
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in module_funcs
    ):
        # A call to a locally-defined function: resolve every direct
        # ``return`` in its body (not descending into nested defs) the same
        # way -- this is what lets e.g. supervise.py's
        # ``_self_deploy_event_kind(result)`` resolve to
        # {self_deploy_failed, self_deploy_succeeded, self_deploy_skipped}.
        func_def = module_funcs[node.func.id]
        returns: list[ast.expr] = []

        def walk(n: ast.AST) -> None:
            for child in ast.iter_child_nodes(n):
                if _is_scope_boundary(child):
                    continue
                if isinstance(child, ast.Return) and child.value is not None:
                    returns.append(child.value)
                walk(child)

        walk(func_def)
        if not returns:
            return None
        func_local_assigns = _collect_local_assignments(func_def)
        func_local_params = _collect_param_names(func_def)
        results: set[str] = set()
        for ret in returns:
            resolved = _resolve_literal(
                ret,
                func_local_assigns,
                module_constants,
                func_local_params,
                module_funcs,
                _depth + 1,
            )
            if resolved is None:
                return None
            results |= resolved
        return results or None
    return None


def _resolve_collection(
    node: ast.expr,
    module_local: dict[str, list[ast.expr]],
    module_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    _depth: int = 0,
) -> set[str] | None:
    """Resolve a collection-valued expression to a finite literal string set.

    Handles the classification-table shape this codebase actually uses:
    ``frozenset({...})``, ``MappingProxyType({...})`` (keys), and unions of
    those (``A | B``) referenced by name -- e.g.
    ``ESCALATION_REASON_CLASS_BY_EVENT_KIND`` and
    ``DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS`` in ``state.py``.
    """
    if _depth > 8:
        return None
    if isinstance(node, ast.Dict):
        keys: list[str] = []
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                return None
        return set(keys)
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        elts: list[str] = []
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                elts.append(e.value)
            else:
                return None
        return set(elts)
    if isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else None
        if fname in ("frozenset", "set", "list", "tuple"):
            if not node.args:
                return set()
            return _resolve_collection(node.args[0], module_local, module_funcs, _depth + 1)
        if fname == "MappingProxyType" and node.args:
            return _resolve_collection(node.args[0], module_local, module_funcs, _depth + 1)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.Add)):
        left = _resolve_collection(node.left, module_local, module_funcs, _depth + 1)
        right = _resolve_collection(node.right, module_local, module_funcs, _depth + 1)
        if left is None or right is None:
            return None
        return left | right
    if isinstance(node, ast.Name):
        if node.id in module_local:
            parts = []
            for value_node in module_local[node.id]:
                part = _resolve_collection(value_node, module_local, module_funcs, _depth + 1)
                if part is None:
                    return None
                parts.append(part)
            return set().union(*parts) if parts else set()
        return None
    return None


def _is_kind_field_access(node: ast.expr) -> bool:
    """True for an expression that reads an *event's* ``kind`` field.

    Deliberately narrower than "any name containing 'kind'": a bare
    ``ast.Name`` (e.g. the ``kind`` parameter inside ``log_event`` itself, or
    a classification table's own lookup key at emission time) is emission-side
    plumbing, not a downstream reader, and must NOT count as a consumer --
    that would make every kind registered in ``instrumentation._LEVEL_BY_KIND``
    (i.e. nearly all of them) look "consumed" by the level-classification
    logic that runs at every single emission, defeating this test's purpose.
    A genuine read accesses a *field* of some event/record object:
    ``event["kind"]``, ``evt.kind``, or ``e.get("kind")``.
    """
    if isinstance(node, ast.Subscript):
        sl = node.slice
        return isinstance(sl, ast.Constant) and sl.value == "kind"
    if isinstance(node, ast.Attribute):
        return node.attr == "kind"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ):
        return bool(
            node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "kind"
        )
    return False


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

    def scan(
        node: ast.AST,
        local_assigns: dict[str, list[ast.expr]],
        local_params: frozenset[str],
        scope_name: str,
    ) -> None:
        if isinstance(node, ast.Call):
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
    call: ast.Call,
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
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        file_sites, _, _ = _scan_emit_sites_in_tree(tree, path.name)
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


def _analyze(src_root: Path, tests_root: Path | None, heartbeat_path: Path | None) -> Report:
    emit_sites: list[EmitSite] = []
    file_markers: dict[str, FileMarkers] = {}
    file_sites: dict[str, list[EmitSite]] = {}
    for path in sorted(src_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        sites, _, _ = _scan_emit_sites_in_tree(tree, path.name)
        emit_sites.extend(sites)
        file_sites[path.name] = sites
        file_markers[path.name] = _scan_markers(path)

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
    for path in sorted(src_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        consumer_sites.extend(_collect_src_consumer_sites(tree, path.name))

    heartbeat_literals: set[str] = set()
    if heartbeat_path is not None and heartbeat_path.is_file():
        hb_tree = ast.parse(heartbeat_path.read_text(encoding="utf-8"))
        heartbeat_literals = _collect_literal_strings(hb_tree)

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
                tree = ast.parse(path.read_text(encoding="utf-8"))
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
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_the_extractor_actually_reaches_things() -> None:
    """Positive control / canary (AC1).

    An empty or near-empty emitted set would mean the extractor is broken,
    and every assertion below would be measuring nothing -- the same "an
    absence is not evidence until you show it could have been non-empty"
    shape as ``test_dormant_fleet_marking.py``'s reachability control. Pin a
    floor, and pin the specific kinds this session's real incidents relied
    on, so a regression in the AST walk (or in the Call-to-local-function
    resolution ``self_deploy_{failed,succeeded}`` specifically exercises)
    fails loudly instead of silently under-collecting.
    """
    report = _analyze(SRC, TESTS_DIR, HEARTBEAT)
    assert len(report.emitted_kinds) > 100, (
        f"emitted-kind extraction found only {len(report.emitted_kinds)} kinds -- "
        "the AST walk is broken"
    )
    required = {
        "loop_started",
        "loop_completed",
        "fleet_pass_config_error",
        "review_verdict_missed",
        "self_deploy_failed",
        "self_deploy_succeeded",
    }
    missing = required - report.emitted_kinds
    assert not missing, f"canary kinds not found by the extractor: {sorted(missing)}"


def test_every_emitted_kind_is_consumed_or_declared(request: object) -> None:
    """The core assertion (AC5/AC6): the real repo's current inventory.

    Every literal emitted kind is either consumed (src/heartbeat/tests) or
    carries a valid ``# event-consumer:`` marker at one of its emission
    sites. Every dynamic (non-literal) call site carries a marker too (AC4).
    A failure here names the offending kind(s)/site(s) and the two remedies:
    add a consumer, or mark it (audit-only with a justification, or
    ``pending #NNNN`` pointing at a real tracking issue).
    """
    report = _analyze(SRC, TESTS_DIR, HEARTBEAT)

    if report.unaccounted:
        lines = []
        for entry in report.unaccounted:
            where = "; ".join(f"{s.path}:{s.lineno}" for s in entry.sites) or "<no emission site>"
            lines.append(f"  {entry.kind} (emitted at {where})")
        assert False, (
            "event kind(s) emitted with no consumer and no marker -- add a consumer "
            "(query_events/heartbeat_check.py/a test), or mark the emission site "
            "`# event-consumer: audit-only <reason>` or `# event-consumer: pending #NNNN`:\n"
            + "\n".join(lines)
        )

    if report.unmarked_dynamic:
        lines = [
            f"  {u.site.path}:{u.site.lineno} in {u.site.scope}() -- kind expression "
            f"`{u.site.kind_source}` is not a string literal (variable/f-string/call); "
            "dynamic kinds escape the registry and MUST carry a "
            "`# event-consumer:` marker"
            for u in report.unmarked_dynamic
        ]
        assert False, "non-literal kind argument(s) without a marker:\n" + "\n".join(lines)

    if report.orphan_markers:
        lines = [f"  {path}:{lineno}" for path, lineno in report.orphan_markers]
        assert False, (
            "`# event-consumer:` marker(s) decorate no real emission site (marker rot -- "
            "the emission was removed/moved, or the marker was misplaced):\n" + "\n".join(lines)
        )

    if report.invalid_markers:
        lines = [f"  {path}:{lineno}: {error}" for path, lineno, error in report.invalid_markers]
        assert False, "invalid `# event-consumer:` marker(s):\n" + "\n".join(lines)

    # Report the pending backlog every run (it must show up, not hide) --
    # issue #1364's own two entries are the expected floor right now.
    pending_kinds = {p.kind for p in report.pending}
    expected_pending = {"draft_pr_blocked", "venv_editable_anchor_violation"}
    missing_pending = expected_pending - pending_kinds
    assert not missing_pending, (
        f"expected pending-marker kinds no longer pending (resolve the tracking issue "
        f"instead of just deleting the marker, or update this test): {sorted(missing_pending)}"
    )


def test_new_unconsumed_kind_without_marker_fails(tmp_path: Path) -> None:
    """AC2: a fresh ``log_event(..., "totally_new_kind", ...)`` with no
    consumer and no marker must be caught, naming the kind and file:line --
    proven via a synthetic module rather than mutating the real source tree.
    """
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "totally_new_kind", payload)\n',
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    assert "totally_new_kind" in report.emitted_kinds
    offending = {u.kind: u for u in report.unaccounted}
    assert "totally_new_kind" in offending, (
        "a brand-new emitted kind with no consumer and no marker must be reported as unaccounted"
    )
    site = offending["totally_new_kind"].sites[0]
    assert site.path == "injected.py"
    assert site.lineno == 4


def test_new_unconsumed_kind_with_marker_is_accepted(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        "    log_event(\n"
        '        path, "totally_new_kind", payload  # event-consumer: audit-only test fixture\n'
        "    )\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    assert not report.unaccounted
    assert not report.unmarked_dynamic
    assert not report.orphan_markers


def test_non_literal_kind_argument_without_marker_fails(tmp_path: Path) -> None:
    """AC4: a variable/f-string kind with no marker fails, explaining why."""
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload, some_dynamic_kind):\n"
        "    log_event(path, some_dynamic_kind, payload)\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    assert len(report.unmarked_dynamic) == 1
    assert report.unmarked_dynamic[0].site.path == "injected.py"
    assert report.unmarked_dynamic[0].site.lineno == 4


def test_non_literal_kind_argument_with_marker_is_accepted(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload, some_dynamic_kind):\n"
        "    log_event(  # event-consumer: pending #9999\n"
        "        path, some_dynamic_kind, payload\n"
        "    )\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    assert not report.unmarked_dynamic


def test_marker_grammar_audit_only() -> None:
    marker = _parse_marker(
        '    log_event(path, "x", {})  # event-consumer: audit-only -- forensic record only',
        1,
    )
    assert marker is not None
    assert marker.kind == "audit-only"
    assert marker.valid


def test_marker_grammar_audit_only_without_justification_is_invalid() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: audit-only', 1)
    assert marker is not None
    assert not marker.valid


def test_marker_grammar_pending_with_issue() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: pending #123', 1)
    assert marker is not None
    assert marker.kind == "pending"
    assert marker.valid
    assert marker.pending_issue == 123


def test_marker_grammar_pending_without_issue_is_invalid() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: pending', 1)
    assert marker is not None
    assert not marker.valid


def test_marker_grammar_pointer_form() -> None:
    marker = _parse_marker(
        '    log_event(path, "x", {})  # event-consumer: dashboards.external_widget',
        1,
    )
    assert marker is not None
    assert marker.kind == "pointer"
    assert marker.valid


def test_orphan_marker_on_non_emitting_line_fails(tmp_path: Path) -> None:
    """AC3: a marker decorating a line with no real emission call site fails."""
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "x = 1  # event-consumer: audit-only nothing emits here\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    assert ("injected.py", 1) in report.orphan_markers


def test_self_match_is_not_a_consumer(tmp_path: Path) -> None:
    """The literal kind string appearing a second time in the SAME function
    that emits it (e.g. a comment, or an unrelated local comparison) must
    not be mistaken for a genuine downstream consumer.
    """
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "self_matching_kind", payload)\n'
        '    also_kind = {"self_matching_kind"}\n'
        '    if payload.get("kind") in also_kind:\n'
        "        pass\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    offending = {u.kind for u in report.unaccounted}
    assert "self_matching_kind" in offending, (
        "a same-function 'consumer' occurrence must not exempt a kind from the unaccounted list"
    )


def test_genuine_downstream_consumer_in_a_different_scope_counts(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "emitter.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "genuinely_consumed_kind", payload)\n',
        encoding="utf-8",
    )
    (module / "reader.py").write_text(
        "def check(events):\n"
        '    return [e for e in events if e["kind"] == "genuinely_consumed_kind"]\n',
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_path=None)
    offending = {u.kind for u in report.unaccounted}
    assert "genuinely_consumed_kind" not in offending
