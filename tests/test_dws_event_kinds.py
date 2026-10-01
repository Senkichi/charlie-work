"""Every event kind the dead-worker sweep package decides to emit is a literal and registered.

The sweep's decide modules build ``Emit`` commits through ``emit("<kind>", ...)`` and
the stalled lane names its event in ``StateTxn(event_kind="<kind>")``; the apply shells
forward ``commit.kind`` / ``commit.event_kind`` to ``append_event``. The repo-wide kind
scanners cannot see through that forwarding (the shells are allow-listed pass-throughs),
so the literal's true origin is checked here instead: each kind must resolve to string
literals, and a kind emitted without an explicit ``level`` must be in the registry (the
registry is where ``append_event`` looks a level up).
(Named ``test_dws_*`` so the dormant module guard does not mistake this for a
one-module-one-test file.)
"""

from __future__ import annotations

import ast
from pathlib import Path

from _instrumentation_kind_scanner import _known_level

_PACKAGE = Path(__file__).parents[1] / "src" / "charlie_work" / "dead_worker_sweep"


def _literals(expr: ast.expr, tree: ast.Module) -> set[str] | None:
    """String literals ``expr`` can evaluate to, following one local assignment; else None."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        body, orelse = _literals(expr.body, tree), _literals(expr.orelse, tree)
        return None if body is None or orelse is None else body | orelse
    if isinstance(expr, ast.Name):
        values = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == expr.id for t in node.targets)
        ]
        resolved = [_literals(v, tree) for v in values]
        if values and all(r is not None for r in resolved):
            return set().union(*resolved)  # type: ignore[arg-type]
    return None


def _has_level(call: ast.Call) -> bool:
    return len(call.args) > 2 or any(kw.arg == "level" for kw in call.keywords)


def _sweep_kinds() -> tuple[set[str], set[str], list[str]]:
    """(kinds needing registration, kinds with an explicit level, unresolvable sites)."""
    needs_registry: set[str] = set()
    explicit_level: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(_PACKAGE.glob("decide*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "emit"
                and node.args
            ):
                kind_node, leveled = node.args[0], _has_level(node)
            elif isinstance(node, ast.keyword) and node.arg == "event_kind":
                kind_node, leveled = node.value, False
            elif (  # ``GuardedUpdate(event=(kind, payload))``: the kind rides the write
                isinstance(node, ast.keyword)
                and node.arg == "event"
                and isinstance(node.value, ast.Tuple)
            ):
                kind_node, leveled = node.value.elts[0], False
            else:
                continue
            if path.name == "decide_common.py" and isinstance(kind_node, ast.Name):
                continue  # ``events()`` forwards the kind its caller already chose
            kinds = _literals(kind_node, tree)
            if kinds is None:
                unresolved.append(f"{path.name}:{node.lineno}: {ast.unparse(kind_node)}")
            else:
                (explicit_level if leveled else needs_registry).update(kinds)
    return needs_registry, explicit_level, unresolved


def test_the_scan_reaches_both_emit_forms() -> None:
    # Positive control: the walk sees the ``emit(...)`` form, the ``event_kind=`` form
    # and the computed ``event_kind`` variable, so an empty result below is not a blind query.
    needs_registry, explicit_level, _ = _sweep_kinds()
    everything = needs_registry | explicit_level
    assert "session_budget_exceeded" in everything
    assert {"session_stalled", "session_exited"} <= everything
    assert len(everything) > 10


def test_every_emitted_kind_resolves_to_literals() -> None:
    _, _, unresolved = _sweep_kinds()
    assert unresolved == [], f"sweep event kinds that are not literals: {unresolved}"


def test_every_kind_emitted_without_a_level_is_registered() -> None:
    needs_registry, _, _ = _sweep_kinds()
    unregistered = sorted(k for k in needs_registry if not _known_level(k))
    assert unregistered == [], f"unregistered sweep event kinds: {unregistered}"
