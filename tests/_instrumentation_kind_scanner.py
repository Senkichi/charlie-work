"""Shared AST emit-site kind scanner machinery for the instrumentation tests.

Hoisted verbatim out of ``tests/test_instrumentation.py`` (issue #1569,
Track-1 seam split) when that module was split into seam-named siblings
-- the ``tests/_*.py`` hoisted-helper convention is the sanctioned import
target for shared test helpers (see
``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from charlie_work.instrumentation import _LEVEL_BY_KIND


# ---------------------------------------------------------------------------
# Issue #910 / #995: event-level registry must cover all in-repo emit sites
#
# #910 added ``_LEVEL_BY_KIND`` and this guard as its enforcement point: an
# emit site's ``kind`` should either be a registry member, or be provably one
# of a small number of literal values that are.
#
# #995: the original scanner (now superseded) understood exactly two shapes
# -- a bare string constant and a ternary between two string constants -- and
# treated everything else as contributing *nothing*. That makes an emit site
# whose kind is a bare variable, an f-string, or a function call indistin-
# guishable from a site that legitimately had no kind to check: the guard
# matches the shapes it recognises and fails *open* on every other shape.
#
# The replacement below inverts that. ``_resolve_literal`` still reduces an
# expression to a finite set of literal strings where it can (constants,
# ternaries, f-strings built entirely from resolved parts, module-level
# constants, and local variable assignments -- including multi-branch
# if/elif chains, which generalises the old single-case ``event_kind``
# special-case) -- but anything it cannot reduce is recorded as *unresolved*
# instead of silently dropped. ``test_event_kind_registry_exhaustive`` fails
# the build on any unresolved site absent from ``_ALLOWED_UNRESOLVED_KIND_SITES``
# (a small, named, reasoned allow-list), and, symmetrically, on any allow-list
# entry that no longer matches a real unresolved site -- so a site that later
# becomes resolvable (or is rewritten) can't leave stale cover behind.
# ---------------------------------------------------------------------------

_EMIT_FUNCS = {"log_event", "append_event", "_record_event", "record_event"}

_WRAPPER_FUNCS = {"_route_to_rework", "_emit_circuit_event"}

_VALID_LEVELS = {"info", "warning", "error"}


def _is_scope_boundary(node: ast.AST) -> bool:
    """True for nodes that start a new variable-assignment scope."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))


def _collect_local_assignments(scope: ast.AST) -> dict[str, list[ast.expr]]:
    """Every ``name = <expr>`` assigned directly within ``scope``.

    Descends through control flow (if/for/while/try/with) but stops at any
    nested function, lambda, or class body -- those are separate scopes with
    their own assignment map, not this one.
    """
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
    """Every name bound as a parameter of ``node`` (positional, keyword-only, *args, **kwargs)."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return frozenset(names)


def _resolve_literal(
    node: ast.expr,
    local_assigns: dict[str, list[ast.expr]],
    module_constants: dict[str, set[str]],
    local_params: frozenset[str] = frozenset(),
) -> set[str] | None:
    """Best-effort resolution of ``node`` to the finite set of strings it can be.

    Returns ``None`` when ``node`` cannot be proven to reduce to a literal set
    of strings -- including when it provably reduces to something that is
    *not* a string (e.g. a bare ``None`` constant), or when any branch of a
    multi-branch expression (an ``IfExp``, or multiple assignments to the same
    local name) is itself unresolvable. Callers MUST check ``is None``, never
    falsiness: ``None`` and ``set()`` are different signals, and conflating
    "unresolvable" with "resolved to nothing" is the #995/#1029 bug shape --
    #995 for ``kind``, #1029 for ``level`` (a ``level="x" if cond else None``
    site unioned the ``None`` branch away and was wrongly treated as
    self-classifying). Every exit point is also guaranteed to never return a
    resolved-but-empty set: ``set().issubset(_VALID_LEVELS)`` is ``True``, so
    an empty set handed to an ``is not None`` consumer would silently
    re-open the same fail-open bug one level up. This is enforced here (see
    the ``or None`` coercions below) rather than left as an invariant callers
    must trust -- a future branch added to this function only needs to avoid
    returning ``set()`` on its own exit, not reason about every consumer.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return {node.value}
        # Any other constant (None, an int, a bool, ...) is provably not a
        # string literal -- not merely unrecognised syntax. This is what lets
        # `level="warning" if cond else None` fail closed: the `None` branch
        # resolves to `None` (unresolvable) instead of silently contributing
        # the empty set that a union would then discard.
        return None
    if isinstance(node, ast.IfExp):
        body = _resolve_literal(node.body, local_assigns, module_constants, local_params)
        orelse = _resolve_literal(node.orelse, local_assigns, module_constants, local_params)
        if body is None or orelse is None:
            return None
        return body | orelse
    if isinstance(node, ast.JoinedStr):
        # An f-string resolves only if every interpolated part resolves (no
        # format spec, no conversion beyond the str-identity ``!s``); the
        # result is the cross product of the static and resolved parts, e.g.
        # ``f"{kind}_sweep"`` with kind in {"a", "b"} resolves to
        # {"a_sweep", "b_sweep"}. This subsumes the old bespoke ``_sweep``
        # special-case in ``_known_level`` below with a general mechanism.
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
                    value.value, local_assigns, module_constants, local_params
                )
                if resolved is not None:
                    parts.append(resolved)
                    continue
            return None
        combined = {""}
        for part in parts:
            combined = {prefix + suffix for prefix in combined for suffix in part}
        # Every `parts` entry is proven non-empty above (Constant str is a
        # singleton; a FormattedValue only appends when `resolved is not
        # None`, and _resolve_literal never itself returns an empty set --
        # see the `or None` guards below), so `combined` cannot legitimately
        # come out empty. Coerce defensively anyway: `_resolve_literal` must
        # never hand a resolved-but-empty set to a caller that branches on
        # `is not None`, since `set().issubset(_VALID_LEVELS)` is True and an
        # empty set would silently re-open the #1029 fail-open one level up.
        return combined or None
    if isinstance(node, ast.Name):
        if node.id in local_params:
            # A function parameter is caller-controlled. A local reassignment
            # of the same name inside the body doesn't prove every path
            # reaches the emit call *after* that reassignment -- the
            # un-reassigned (or not-yet-reassigned) parameter value could
            # still be the one that flows through on some path. Treat
            # conservatively as unresolved rather than trusting a partial
            # local reassignment over the parameter's own (unknown) value.
            return None
        if node.id in local_assigns:
            values: set[str] = set()
            for value_node in local_assigns[node.id]:
                resolved = _resolve_literal(
                    value_node, local_assigns, module_constants, local_params
                )
                if resolved is None:
                    # One reassignment on this name is unresolvable -- some
                    # path through the function could carry that value to the
                    # emit call, so the name as a whole is unresolvable too,
                    # same as an IfExp branch that doesn't resolve.
                    return None
                values |= resolved
            # Same empty-set guard as the JoinedStr exit above: `values`
            # should be non-empty by construction (every resolved branch is
            # itself non-empty, and `local_assigns[node.id]` is never an
            # empty list -- `_collect_local_assignments` only creates the key
            # when appending a value), but the guarantee must live at this
            # producer boundary, not be assumed by every consumer.
            return values or None
        if node.id in module_constants:
            # Third empty-set guard, for the same reason as the two above.
            # `_scan_tree` only inserts a key when `parts` is non-empty and
            # every part came back non-empty, so this cannot be empty today --
            # but that reasoning lives in a *different* function, and the
            # docstring above promises the invariant is enforced at every exit
            # of *this* one. Enforce it here rather than leaving the promise
            # true only by inspection of a caller.
            return module_constants[node.id] or None
        return None
    return None


@dataclass(frozen=True)
class _UnresolvedKindSite:
    """An emit-site ``kind`` expression the scanner could not reduce to literals."""

    path: str  # POSIX-relative to the scanned root
    scope: str  # enclosing function/method name, or "<module>"
    source: str  # ast.unparse() of the expression -- stable across line drift
    lineno: int = 0  # human-readable only; deliberately excluded from ``key``
    reason: str = ""  # human-readable only; deliberately excluded from ``key``

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.scope, self.source)


def _emit_func_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in _EMIT_FUNCS | _WRAPPER_FUNCS:
        return node.func.id
    if isinstance(node.func, ast.Attribute) and node.func.attr in _EMIT_FUNCS | _WRAPPER_FUNCS:
        return node.func.attr
    return None


def _has_explicit_level(
    node: ast.Call,
    local_assigns: dict[str, list[ast.expr]],
    module_constants: dict[str, set[str]],
    local_params: frozenset[str],
) -> bool:
    """True if ``node`` passes a ``level=`` keyword that resolves to a valid level.

    ``log_event`` lets a call site declare its level explicitly instead of
    relying on the registry (``instrumentation.log_event``'s ``level``
    parameter). Such a site is legitimately self-classifying, so its
    (possibly unresolvable) ``kind`` need not be registered or allow-listed.
    """
    for kw in node.keywords:
        if kw.arg != "level":
            continue
        resolved = _resolve_literal(kw.value, local_assigns, module_constants, local_params)
        if resolved is not None and resolved.issubset(_VALID_LEVELS):
            return True
    return False


def _locate_arg(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """Find an argument by position or by keyword name, whichever the call used.

    A call that supplies the kind (or ``event_kind``) as a keyword argument
    -- ``log_event(state_path, kind=k, payload=p)`` -- must not silently fall
    through just because it doesn't match the common positional shape. That
    is the same fail-open pattern #995 was filed against, one layer down: a
    scanner that only recognizes the arity/shape it expects and drops
    everything else. No call site in this package currently uses ``*args``
    unpacking into these functions, so positional lookup by index is sound;
    if that ever changes, the returned ``None`` correctly routes the site to
    "unresolved" rather than silently skipping it.
    """
    if len(node.args) > position:
        return node.args[position]
    for kw in node.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _record_kind_site(
    kind_node: ast.expr | None,
    call_node: ast.Call,
    module_constants: dict[str, set[str]],
    local_assigns: dict[str, list[ast.expr]],
    local_params: frozenset[str],
    scope_name: str,
    rel_path: str,
    used: set[str],
    unresolved: list[_UnresolvedKindSite],
) -> None:
    if kind_node is None:
        # The call matched an emit function name but no kind/event_kind
        # argument could be located by position or keyword. Rather than
        # assume this isn't really one of our functions (the #995 failure
        # mode), record it as unresolved so it surfaces for review.
        unresolved.append(
            _UnresolvedKindSite(
                path=rel_path,
                scope=scope_name,
                source="<no kind argument located>",
                lineno=call_node.lineno,
            )
        )
        return
    resolved = _resolve_literal(kind_node, local_assigns, module_constants, local_params)
    if resolved is not None:
        used.update(resolved)
        return
    unresolved.append(
        _UnresolvedKindSite(
            path=rel_path,
            scope=scope_name,
            source=ast.unparse(kind_node),
            lineno=call_node.lineno,
        )
    )


def _scan_node(
    node: ast.AST,
    module_constants: dict[str, set[str]],
    local_assigns: dict[str, list[ast.expr]],
    local_params: frozenset[str],
    scope_name: str,
    rel_path: str,
    used: set[str],
    unresolved: list[_UnresolvedKindSite],
) -> None:
    if isinstance(node, ast.Call):
        func_name = _emit_func_name(node)
        if func_name in _EMIT_FUNCS:
            if not _has_explicit_level(node, local_assigns, module_constants, local_params):
                kind_node = _locate_arg(node, 1, "kind")
                _record_kind_site(
                    kind_node,
                    node,
                    module_constants,
                    local_assigns,
                    local_params,
                    scope_name,
                    rel_path,
                    used,
                    unresolved,
                )
        elif func_name == "_route_to_rework":
            kind_node = _locate_arg(node, 4, "event_kind")
            _record_kind_site(
                kind_node,
                node,
                module_constants,
                local_assigns,
                local_params,
                scope_name,
                rel_path,
                used,
                unresolved,
            )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Entering a new function scope: its local assignments and parameters
        # shadow the enclosing ones rather than extending them.
        scoped_assigns = _collect_local_assignments(node)
        scoped_params = _collect_param_names(node)
        for child in ast.iter_child_nodes(node):
            _scan_node(
                child,
                module_constants,
                scoped_assigns,
                scoped_params,
                node.name,
                rel_path,
                used,
                unresolved,
            )
        return

    for child in ast.iter_child_nodes(node):
        _scan_node(
            child,
            module_constants,
            local_assigns,
            local_params,
            scope_name,
            rel_path,
            used,
            unresolved,
        )


def _scan_tree(tree: ast.Module, rel_path: str) -> tuple[set[str], list[_UnresolvedKindSite]]:
    """Scan one already-parsed module for emit-site kinds.

    Exposed separately from ``_scan_event_kinds`` so tests can feed it a
    synthetic module without writing a file to disk.
    """
    used: set[str] = set()
    unresolved: list[_UnresolvedKindSite] = []
    module_local = _collect_local_assignments(tree)
    module_constants: dict[str, set[str]] = {}
    for name, value_nodes in module_local.items():
        # A module-level name assigned more than once (e.g. reassigned under
        # an `if`) is only a "constant" -- resolvable at every reference site
        # -- if *every* assignment resolves. One unresolvable assignment (a
        # function call, an unknown name, ...) means some path could carry an
        # unproven value, so `parts` is reset and abandoned rather than
        # unioning in only the assignments that happened to resolve: that
        # would silently drop the unresolvable branch the same way `None`
        # was dropped from a `level=` union (#1029).
        parts: list[set[str]] = []
        for value_node in value_nodes:
            part = _resolve_literal(value_node, module_local, {}, frozenset())
            if part is None:
                parts = []
                break
            parts.append(part)
        if parts:
            module_constants[name] = set().union(*parts)
    _scan_node(
        tree, module_constants, module_local, frozenset(), "<module>", rel_path, used, unresolved
    )
    return used, unresolved


def _scan_event_kinds(root: Path) -> tuple[set[str], list[_UnresolvedKindSite]]:
    """Walk every Python file under ``root``; return (used kinds, unresolved sites)."""
    used: set[str] = set()
    unresolved: list[_UnresolvedKindSite] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        file_used, file_unresolved = _scan_tree(tree, path.relative_to(root).as_posix())
        used |= file_used
        unresolved.extend(file_unresolved)
    return used, unresolved


def _known_level(kind: str) -> bool:
    """Return True if ``kind`` is in the registry or is a registered sweep."""
    if kind in _LEVEL_BY_KIND:
        return True
    if kind.endswith("_sweep") and kind[: -len("_sweep")] in _LEVEL_BY_KIND:
        return True
    return False


# Sites where the scanner cannot statically resolve the ``kind`` argument to a
# finite literal set, and why that's fine. Every entry must be independently
# justified: either the real literal is chosen at a call site the scanner
# already covers elsewhere (a forwarding wrapper -- the literal at the
# *caller* is what's checked), or it's covered by a dedicated test below.
# ``test_event_kind_registry_exhaustive`` enforces both directions: an
# unresolved site missing from this list fails the test, and so does an
# entry here that no longer matches a real unresolved site.
_ALLOWED_UNRESOLVED_KIND_SITES: tuple[_UnresolvedKindSite, ...] = (
    _UnresolvedKindSite(
        path="state.py",
        scope="append_event",
        source="kind",
        reason=(
            "append_event forwards its own `kind` parameter to log_event. "
            "The literal is chosen at each call site, and append_event is "
            "itself in _EMIT_FUNCS, so every call site is already scanned -- "
            "this is the same site observed from inside the callee."
        ),
    ),
    _UnresolvedKindSite(
        path="orchestration/helpers_merge_outcomes.py",
        scope="_record_event",
        source="kind",
        reason=(
            "OrchestratorApp._record_event forwards its own `kind` parameter "
            "to append_event. Same pass-through as append_event/log_event; "
            "every self._record_event(...) call site is scanned. "
            "Moved out of workflow.py to charlie_work.orchestration."
            "helpers_merge_outcomes by leaf L02 b3 (#1633); the allow-list "
            "path follows the member (design 4.1), scope/source/reason unchanged."
        ),
    ),
    _UnresolvedKindSite(
        path="orchestration/state_rework_routing.py",
        scope="_route_to_rework",
        source="event_kind",
        reason=(
            "_route_to_rework forwards its own `event_kind` parameter to "
            "self._record_event. Every self._route_to_rework(...) call site "
            "is scanned (it is in _WRAPPER_FUNCS, 5th positional argument). "
            "Moved out of workflow.py to charlie_work.orchestration."
            "state_rework_routing by leaf L01 b1 (#1632); the allow-list key "
            "follows the member to its new module -- scope/source/reason "
            "unchanged."
        ),
    ),
    _UnresolvedKindSite(
        path="github_capabilities/transport.py",
        scope="_emit_circuit_event",
        source="kind",
        reason=(
            "Transport._emit_circuit_event forwards its own `kind` parameter "
            "to log_event. Same pass-through as _route_to_rework/event_kind; "
            "it is in _WRAPPER_FUNCS, so every self._emit_circuit_event(...) "
            "call site is scanned instead (both current call sites in "
            "_circuit_breaker_note_result pass a literal: "
            '"github_circuit_opened" / "github_circuit_closed"). Added for '
            "issue #1833."
        ),
    ),
    _UnresolvedKindSite(
        path="stalled_review_reap.py",
        scope="_append_sweep_events",
        source="kind",
        reason=(
            "kind is the loop variable over sweep_events, a list built by "
            'many `sweep_events.append(("literal_kind", payload))` call '
            "sites elsewhere in this file. Those literals are independently "
            "verified by test_sweep_event_append_kinds_are_registered."
        ),
    ),
    _UnresolvedKindSite(
        path="stalled_review_reap.py",
        scope="_append_sweep_events",
        source="f'{kind}_sweep'",
        reason="Same `kind` loop variable as above, with the `_sweep` suffix appended.",
    ),
    _UnresolvedKindSite(
        path="supervise.py",
        scope="_log_self_deploy_outcome",
        source="_self_deploy_event_kind(result)",
        reason=(
            "_self_deploy_event_kind(result) returns one of "
            "self_deploy_{failed,succeeded,skipped}; every branch is "
            "verified by test_self_deploy_event_kind_only_returns_registered_kinds."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="append_event",
        source="kind",
        reason=(
            "WriteGate.append_event forwards its own `kind` parameter to "
            "state.append_event. Same pass-through as the existing "
            "state.py/append_event entry above; the literal is chosen at "
            "each call site, and every self.write_gate.append_event(...) "
            "call site is scanned there."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="record_event",
        source="kind",
        reason=(
            "WriteGate.record_event forwards its own `kind` parameter to "
            "state.append_event, mirroring OrchestratorApp._record_event's "
            "own forwarding shape (see the workflow.py/_record_event entry "
            "above). Every self.write_gate.record_event(...) call site is "
            "scanned there; `record_event` is itself in _EMIT_FUNCS so no "
            "coverage gap opens once a call site migrates onto it."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="log_event",
        source="kind",
        reason=(
            "WriteGate.log_event forwards its own `kind` parameter to "
            "instrumentation.log_event. The literal is chosen at each call "
            "site, and every self.write_gate.log_event(...) call site is "
            "scanned there."
        ),
    ),
    _UnresolvedKindSite(
        path="dead_worker_reap.py",
        scope="_attempt_salvage",
        source="salvage_skip_event_kind(skip_reason)",
        reason=(
            "Issue #1241: salvage_skip_event_kind maps skip_reason to one of "
            "two registered literals (salvage_skipped_already_landed for the "
            "#1221 reasons, salvage_skipped_superseded for the new "
            "commits_reachable reason), both in _LEVEL_BY_KIND. The mapping "
            "is verified by test_salvage_skip_event_kind_only_returns_registered_kinds "
            "in tests/test_salvage_superseded_1241.py."
        ),
    ),
    _UnresolvedKindSite(
        path="reconcile.py",
        scope="apply_fixes",
        source="salvage_skip_event_kind(skip_reason)",
        reason=(
            "Issue #1241: same salvage_skip_event_kind mapping as the "
            "dead_worker_reap.py/_attempt_salvage entry above -- the reconcile "
            "salvage lane and the workflow salvage lane share the single "
            "enforcement point in salvage_superseded.py. Both target literals "
            "are in _LEVEL_BY_KIND and verified by "
            "test_salvage_skip_event_kind_only_returns_registered_kinds."
        ),
    ),
)


def _scan_sweep_append_kinds(root: Path) -> tuple[set[str], list[str]]:
    """Scan every ``*.py`` under ``root`` for ``sweep_events.append((kind, payload))``
    call sites; return (resolved literal kinds, unresolvable-element descriptions).

    Factored out of ``test_sweep_event_append_kinds_are_registered`` so the
    fail-closed behavior on an unresolvable kind element (#1029) can be
    exercised directly against a synthetic tree, the same way
    ``_scan_event_kinds`` is factored out for the main scanner.
    """
    found: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sweep_events"
            ):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Tuple) and arg.elts:
                resolved = _resolve_literal(arg.elts[0], {}, {})
                if resolved is None:
                    # #1029: this scan's whole premise is that every literal
                    # at a `sweep_events.append((kind, payload))` site is
                    # registered -- an element the resolver can't prove a
                    # literal for undermines that premise and must surface,
                    # not silently contribute nothing to `found` (the same
                    # union-with-empty-set fail-open the `level=` bug had).
                    unresolved.append(
                        f"{path.relative_to(root).as_posix()}:{node.lineno}: "
                        f"{ast.unparse(arg.elts[0])}"
                    )
                    continue
                found |= resolved
    return found, unresolved


def _call_node_for(source: str) -> tuple[ast.Call, dict[str, list[ast.expr]], frozenset[str]]:
    """Parse ``source``, return the single ``log_event`` ``Call`` node plus its
    enclosing function's local assignments and parameter names.

    Test helper for exercising ``_has_explicit_level`` directly, without going
    through the full ``_scan_tree`` walk.
    """
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    call = next(
        n for n in ast.walk(func) if isinstance(n, ast.Call) and _emit_func_name(n) == "log_event"
    )
    return call, _collect_local_assignments(func), _collect_param_names(func)
