"""AST literal-resolution helpers for the event-kind extraction scan."""

from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# AST scope/assignment helpers -- mirrors the proven approach in
# tests/_instrumentation_kind_scanner.py's _resolve_literal (a different concern,
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
