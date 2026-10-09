"""AST machinery behind the issue-#1683 structural guard in
``test_issue_1683_review_dispatch_deescalation.py``: derive every
``escalation_reason`` reachable with ``reason_class="mechanical"`` from
``_escalate_issue`` call sites, so the test can assert each one is mapped
or allowlisted.

The defect class the guard covers is "a mechanical escalation reason
whose lane gates on a per-mechanism counter had no map entry".  A
hand-maintained copy of the reason list in a test would silently drift
the moment a new lane is added, so the set is DERIVED from the source:
every ``_escalate_issue`` call in ``charlie_work`` whose ``reason_class``
argument can be ``"mechanical"``, with the ``reason`` argument resolved
through local assignments, if-expression branches, f-strings, and
function-parameter call-site propagation.

Reasons that cannot be reduced to a finite literal set are represented
by dynamic descriptors:

- ``failure_kind`` (bare name or ``*.failure_kind`` attribute): every
  mechanical-classed call site membership-gates it on
  ``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` (deterministic-judgment
  kinds select ``reason_class="judgment"`` instead), so the live
  frozenset is substituted -- a new kind added to that set is checked
  automatically.
- ``fstring:<skeleton>`` and ``expr:<source>``: genuinely unbounded
  domains, allowlisted per-shape in the test module with a justification.

``_SourceIndex`` answers "what literal strings can flow into the
``reason`` and ``reason_class`` kwargs of an ``_escalate_issue`` call?"
-- which requires resolving names through local assignments and (for
parameters like ``attempts_key``) through the enclosing function's own
call sites.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _src_ast import parsed, source_files, source_text

import charlie_work
from charlie_work.config import DETERMINISTIC_ESCALATION_FAILURE_KINDS


class _SourceIndex:
    """AST index over every ``charlie_work`` module: every call site in the
    package, plus scope, name-assignment, and parameter indexes that are
    derived lazily -- only for the scopes ``resolve`` actually visits.

    Nothing is walked eagerly.  ``call_sites`` answers "every call to
    ``name`` in the package" by first restricting candidate modules by
    source text -- a call to ``name`` requires the name to appear
    literally in the file -- and AST-walking only the modules that pass,
    bucketing every call they contain by callee name so one walk serves
    all later queries.  An earlier version instead recursed through every
    node of all ~470 modules up front to tag each call with its enclosing
    scope and eagerly indexed the body of all ~3800 functions; that
    per-node descent cost ~2 s per build while the actual resolution
    below takes ~0.1 s and touches a few dozen scopes out of thousands
    (issue #2722).  A call's innermost enclosing scope is likewise derived
    on demand through a per-module child->parent map, and each scope's
    name assignments and parameter list are indexed the first time a name
    lookup reaches that scope.
    """

    def __init__(self) -> None:
        pkg_dir = Path(charlie_work.__file__).resolve().parent
        self._paths = source_files(pkg_dir)
        # path -> callee-name-bucketed calls; a path enters only once a
        # queried name is found in its source text and it has been walked.
        self._module_calls: dict[Path, dict[str | None, list[ast.Call]]] = {}
        # path -> parsed module tree (from the shared _src_ast cache).
        self._trees: dict[Path, ast.Module] = {}
        # path -> source text; a local memo because every _src_ast read
        # pays a resolve() syscall per queried name otherwise.
        self._texts: dict[Path, str] = {}
        # Everything below is keyed by node identity and filled on demand.
        # scope node -> {name: [assigned exprs]}; module trees and
        # function defs are both scope nodes.
        self.assigns: dict[int, dict[str, list[ast.expr]]] = {}
        # func node -> (ordered parameter names, {param name: default expr}).
        self.params: dict[int, tuple[list[str], dict[str, ast.expr]]] = {}
        # func node -> enclosing module tree (for module-scope fallback).
        self.parent_scope: dict[int, ast.Module] = {}
        # module tree -> {id(child node): parent node}.
        self._parents: dict[int, dict[int, ast.AST]] = {}
        # call node -> innermost enclosing scope (FunctionDef or Module).
        self._call_scope: dict[int, ast.AST] = {}

    def call_sites(self, name: str) -> list[tuple[ast.Module, ast.Call]]:
        """Every ``(module tree, call)`` in the package whose callee is
        named ``name`` -- matching ``_callee_name`` exactly (bare ``name(...)``
        and ``anything.name(...)``).
        """
        out: list[tuple[ast.Module, ast.Call]] = []
        for path in self._paths:
            index = self._module_calls.get(path)
            if index is None:
                # A call to ``name`` cannot exist in a file whose source
                # lacks the name, and _callee_name renders callees as
                # literal name/attribute text, so the substring check
                # loses nothing; matches inside comments or longer
                # identifiers only cost a walk.
                text = self._texts.get(path)
                if text is None:
                    text = self._texts[path] = source_text(path)
                if name not in text:
                    continue
                tree = self._trees.setdefault(path, parsed(path))
                index = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        index.setdefault(_callee_name(node), []).append(node)
                self._module_calls[path] = index
            for call in index.get(name, ()):
                out.append((self._trees[path], call))
        return out

    # -- lazy scope/assignment/parameter indexes --

    def scope_of(self, call: ast.Call, tree: ast.Module) -> ast.AST:
        """The innermost ``FunctionDef`` enclosing ``call``, else ``tree``.

        ``ClassDef`` and ``Lambda`` do not open scopes here, matching the
        resolver's rule that a function's parent scope is always its
        module.
        """
        scope = self._call_scope.get(id(call))
        if scope is not None:
            return scope
        parents = self._parents.get(id(tree))
        if parents is None:
            parents = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[id(child)] = parent
            self._parents[id(tree)] = parents
        scope = tree
        node: ast.AST = call
        while True:
            parent = parents.get(id(node))
            if parent is None:
                break  # node is the module tree itself
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = parent
                break
            node = parent
        if not isinstance(scope, ast.Module):
            self.parent_scope[id(scope)] = tree
        self._call_scope[id(call)] = scope
        return scope

    def _scope_assigns(self, scope: ast.AST) -> dict[str, list[ast.expr]]:
        assigns = self.assigns.get(id(scope))
        if assigns is None:
            assigns = (
                self._module_assigns(scope)
                if isinstance(scope, ast.Module)
                else self._func_assigns(scope)
            )
            self.assigns[id(scope)] = assigns
        return assigns

    @staticmethod
    def _module_assigns(tree: ast.Module) -> dict[str, list[ast.expr]]:
        assigns: dict[str, list[ast.expr]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assigns.setdefault(node.target.id, []).append(node.value)
        return assigns

    @staticmethod
    def _func_assigns(func: ast.AST) -> dict[str, list[ast.expr]]:
        assigns: dict[str, list[ast.expr]] = {}

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                # Nested scopes bind their own names; do not descend.
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
                ):
                    continue
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigns.setdefault(target.id, []).append(child.value)
                elif (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.value is not None
                ):
                    assigns.setdefault(child.target.id, []).append(child.value)
                elif isinstance(child, (ast.For, ast.AsyncFor)) and isinstance(
                    child.target, ast.Name
                ):
                    # Loop targets are bound but not literal-resolvable.
                    assigns.setdefault(child.target.id, []).append(child.iter)
                elif isinstance(child, (ast.With, ast.AsyncWith)):
                    for item in child.items:
                        if isinstance(item.optional_vars, ast.Name):
                            assigns.setdefault(item.optional_vars.id, []).append(item.context_expr)
                visit(child)

        visit(func)
        return assigns

    def _params_of(self, func: ast.AST) -> tuple[list[str], dict[str, ast.expr]]:
        cached = self.params.get(id(func))
        if cached is not None:
            return cached
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = func.args
            names = [p.arg for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            defaults: dict[str, ast.expr] = {}
            positional = list(args.posonlyargs) + list(args.args)
            for param, default in zip(
                positional[len(positional) - len(args.defaults) :], args.defaults
            ):
                defaults[param.arg] = default
            for param, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None:
                    defaults[param.arg] = default
            cached = (names, defaults)
        else:
            cached = ([], {})
        self.params[id(func)] = cached
        return cached

    # -- expression resolution --

    def resolve(
        self, node: ast.expr, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        """Resolve ``node`` to (literal strings, dynamic descriptors).

        Over-approximates literals (union of every branch/assignment) so a
        missed string is never silently dropped; anything not reducible to
        a literal becomes a descriptor the caller must allowlist.
        """
        literals: set[str] = set()
        dynamics: set[str] = set()

        def merge(res: tuple[set[str], set[str]]) -> None:
            literals.update(res[0])
            dynamics.update(res[1])

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                literals.add(node.value)
        elif isinstance(node, ast.Name):
            merge(self._resolve_name(node.id, scope, in_progress))
        elif isinstance(node, ast.IfExp):
            merge(self.resolve(node.body, scope, in_progress))
            merge(self.resolve(node.orelse, scope, in_progress))
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                merge(self.resolve(elt, scope, in_progress))
        elif isinstance(node, ast.JoinedStr):
            merge(self._resolve_joined(node, scope, in_progress))
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict):
            # ``{...literal dict...}[key]``: the result is one of the dict's
            # values, so the union of every value over-approximates it (the
            # dead-worker sweep maps an escalation kind to its reason this
            # way).
            for value in node.value.values:
                merge(self.resolve(value, scope, in_progress))
        elif isinstance(node, ast.Attribute) and node.attr == "failure_kind":
            # ``foo.failure_kind`` reaching a mechanical escalation site is
            # membership-gated to DETERMINISTIC_ESCALATION_FAILURE_KINDS at
            # every call site (judgment kinds take reason_class="judgment").
            literals.update(DETERMINISTIC_ESCALATION_FAILURE_KINDS)
        else:
            dynamics.add(f"expr:{ast.unparse(node)}")
        return literals, dynamics

    def _resolve_joined(
        self, node: ast.JoinedStr, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        segments: list[set[str]] = []
        dynamic = False
        for value in node.values:
            if isinstance(value, ast.Constant):
                segments.append({str(value.value)})
                continue
            lits, dyn = self.resolve(value.value, scope, in_progress)
            if dyn:
                dynamic = True
            segments.append(lits)
        if dynamic:
            skeleton = "".join(
                value.value if isinstance(value, ast.Constant) else "{*}" for value in node.values
            )
            return set(), {f"fstring:{skeleton}"}
        combos = [""]
        for segment in segments:
            combos = [c + s for c in combos for s in segment]
            if len(combos) > 64:
                skeleton = "".join(
                    value.value if isinstance(value, ast.Constant) else "{*}"
                    for value in node.values
                )
                return set(), {f"fstring:{skeleton}"}
        return set(combos), set()

    def _resolve_name(
        self, name: str, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        # ``failure_kind`` is the one name that must never be chased
        # through the call graph: every mechanical-classed escalation site
        # membership-gates it on DETERMINISTIC_ESCALATION_FAILURE_KINDS
        # before it can reach ``_escalate_issue``, so the live frozenset
        # IS the mechanical domain (a new kind is picked up automatically).
        if name == "failure_kind":
            return set(DETERMINISTIC_ESCALATION_FAILURE_KINDS), set()
        seen_scope = scope
        while seen_scope is not None:
            scope_assigns = self._scope_assigns(seen_scope).get(name)
            if scope_assigns:
                literals: set[str] = set()
                dynamics: set[str] = set()
                for expr in scope_assigns:
                    lits, dyn = self.resolve(expr, seen_scope, in_progress)
                    literals |= lits
                    dynamics |= dyn
                # A name that is both assigned and a parameter can still
                # carry the caller's value into reads that precede the
                # assignment; union the parameter domain (over-approximate).
                if name in self._params_of(seen_scope)[0]:
                    lits, dyn = self._param_domain(seen_scope, name, in_progress)
                    literals |= lits
                    dynamics |= dyn
                return literals, dynamics
            if name in self._params_of(seen_scope)[0]:
                return self._param_domain(seen_scope, name, in_progress)
            seen_scope = self.parent_scope.get(id(seen_scope))
        return set(), {f"name:{name}"}

    def _param_domain(
        self, func: ast.AST, param: str, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        """Resolve a function parameter from every call site of ``func``."""
        key = (id(func), param)
        if key in in_progress:
            return set(), set()
        in_progress = in_progress | {key}
        func_name = getattr(func, "name", "")
        positional = [
            p.arg
            for p in (*func.args.posonlyargs, *func.args.args)  # type: ignore[attr-defined]
        ]
        literals: set[str] = set()
        dynamics: set[str] = set()
        bound_somewhere = False
        for tree, call in self.call_sites(func_name):
            arg = None
            for keyword in call.keywords:
                if keyword.arg == param:
                    arg = keyword.value
            if arg is None and param in positional:
                idx = positional.index(param)
                if idx < len(call.args):
                    arg = call.args[idx]
            if arg is None:
                # ``**kwargs`` could smuggle a binding we cannot see.
                if any(keyword.arg is None for keyword in call.keywords):
                    dynamics.add(f"param:{func_name}:{param}:**kwargs")
                continue
            bound_somewhere = True
            lits, dyn = self.resolve(arg, self.scope_of(call, tree), in_progress)
            literals |= lits
            dynamics |= dyn
        default = self._params_of(func)[1].get(param)
        if default is not None:
            lits, dyn = self.resolve(default, func, in_progress)
            literals |= lits
            dynamics |= dyn
        if not bound_somewhere and default is None:
            dynamics.add(f"param:{func_name}:{param}")
        return literals, dynamics


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ``_escalate_issue`` is the imperative escalation site; the dead-worker
# sweep's pure decide phase instead yields the ``Escalate`` commit value
# (same ``reason`` / ``reason_class`` kwargs), which the apply phase feeds
# to ``_escalate_issue`` through a port -- a call the scan cannot follow.
# Treat the commit constructor as an escalation site so its literals count.
_ESCALATION_SITE_CALLEES = frozenset({"_escalate_issue", "Escalate"})


def _mechanical_escalation_reasons() -> tuple[set[str], set[str]]:
    """Derive every ``escalation_reason`` reachable with
    ``reason_class="mechanical"`` from ``_escalate_issue`` call sites.

    Returns ``(literal_reasons, dynamic_descriptors)``.  A call site is
    mechanical-capable when its ``reason_class`` resolves to a set
    containing ``"mechanical"`` or to anything unresolvable (conservative:
    an unresolvable class could be mechanical at runtime).
    """
    index = _SourceIndex()
    literal_reasons: set[str] = set()
    dynamic_sites: set[str] = set()
    site_calls = [
        (tree, call)
        for callee in _ESCALATION_SITE_CALLEES
        for tree, call in index.call_sites(callee)
    ]
    for tree, call in site_calls:
        scope = index.scope_of(call, tree)
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        class_lits, class_dyn = (
            index.resolve(kwargs["reason_class"], scope, frozenset())
            if "reason_class" in kwargs
            else (set(), {"missing:reason_class"})
        )
        if "mechanical" not in class_lits and not class_dyn:
            continue
        if "reason" not in kwargs:
            dynamic_sites.add("missing:reason")
            continue
        reason_lits, reason_dyn = index.resolve(kwargs["reason"], scope, frozenset())
        literal_reasons |= reason_lits
        dynamic_sites |= reason_dyn
    return literal_reasons, dynamic_sites
