"""Every event kind the dead-worker sweep package decides to emit is a literal and registered.

The sweep's decide modules build ``Emit`` commits through ``emit("<kind>", ...)`` and
the stalled lane names its event in ``StateTxn(event_kind="<kind>")``; the apply shells
forward ``commit.kind`` / ``commit.event_kind`` to ``append_event``. The repo-wide
consumer scanner (``test_event_kind_consumers.py``) treats those decide-module literals
as the emit sites (sharing ``_dws_emit_scan``) and the shells as forwarding only; this
file checks the literal-ness and level registration: each kind must resolve to string
literals, and a kind emitted without an explicit ``level`` must be in the registry (the
registry is where ``append_event`` looks a level up).
(Named ``test_dws_*`` so the dormant module guard does not mistake this for a
one-module-one-test file.)
"""

from __future__ import annotations

import ast
from pathlib import Path

from _dws_emit_scan import literals, sweep_emit_kind
from _instrumentation_kind_scanner import _known_level

_PACKAGE = Path(__file__).parents[1] / "src" / "charlie_work" / "dead_worker_sweep"


def _sweep_kinds() -> tuple[set[str], set[str], list[str]]:
    """(kinds needing registration, kinds with an explicit level, unresolvable sites)."""
    needs_registry: set[str] = set()
    explicit_level: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(_PACKAGE.glob("decide*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            site = sweep_emit_kind(node, path.name)
            if site is None:
                continue
            kind_node, leveled = site
            kinds = literals(kind_node, tree)
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
