"""AST guard: every ``transition``/``apply_issue_labels`` call in ``src/`` passes ``state_path=``.

Split out of ``tests/test_lifecycle_transition_events.py`` for #3214: charging
the scan to a module-scoped fixture pushed that file over the 800-line cap
with no ratchet baseline entry (the issue #1442 ratchet treats a file with no
entry as mark 0), and the ratchet's rule is that new code splits into
seam-named siblings rather than receiving a mark (the #2992 precedent for
over-cap test files; the same move #3157 made for the ``sanitize_env``
call-site guard). The leaf name ``test_production_label_seam_calls_pass_state_path``
is unchanged, so the collect-only gate sees a pure relocation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _src_ast import parsed, source_files, source_text


@pytest.fixture(scope="module")
def label_seam_call_sites() -> list[tuple[str, bool]]:
    """Non-exempt ``transition``/``apply_issue_labels`` calls under ``src/`` as
    ``(file:line, passes_state_path)``.

    A call spells its callee name verbatim in the file text (``ast.Name.id``
    /``ast.Attribute.attr``), so a file whose ``source_text`` lacks both
    markers cannot hold a call site and is skipped before ``parsed()``.
    Calls on a ``WriteGate`` receiver (``*.write_gate``, ``write_gate``,
    ``gate``, ``wg``) are excluded up front — the gate binds its own state
    path. Measured at this tree: 546 scanned files shrink to 112 candidates.

    Scanning the package is this guard's dominant cost, so it is charged to
    a module fixture — the test's ``setup`` — instead of the ``call`` phase
    the ledger times (the accepted #2662/#2867/#2875/#2882 remedy family,
    applied to this guard for #3214). The scan's output is shared, read-only
    data; if the fixture's setup ever flags, ``ci-fleet ledger report
    --section slow-fixtures`` is its owner.
    """
    found: list[tuple[str, bool]] = []
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    gate_receivers = {"write_gate", "gate", "wg"}
    for path in source_files(src_root):
        text = source_text(path)
        if "transition" not in text and "apply_issue_labels" not in text:
            continue
        tree = parsed(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_expr = node.func
            call_name = (
                func_expr.attr
                if isinstance(func_expr, ast.Attribute)
                else getattr(func_expr, "id", "")
            )
            if call_name not in ("transition", "apply_issue_labels"):
                continue
            if isinstance(func_expr, ast.Attribute):
                receiver = func_expr.value
                if isinstance(receiver, ast.Attribute) and receiver.attr == "write_gate":
                    continue
                if isinstance(receiver, ast.Name) and receiver.id in gate_receivers:
                    continue
            found.append(
                (f"{path.name}:{node.lineno}", "state_path" in {kw.arg for kw in node.keywords})
            )
    return found


def test_production_label_seam_calls_pass_state_path(
    label_seam_call_sites: list[tuple[str, bool]],
) -> None:
    """AST-derived: every ``transition``/``apply_issue_labels`` call in src/
    passes ``state_path=`` explicitly — the keyword is required (no default),
    so a missing ``state_path=`` means the call would fail open as a
    ``TypeError`` the moment it ran. Calls on a ``WriteGate`` receiver
    (``self.write_gate``, ``ctx.write_gate``, ``write_gate``, ``gate``) are
    exempt — the gate binds its own state path."""
    # Positive control: a scan that matched nothing reports "zero offenders"
    # the same as a clean tree, so the guard must prove it sees the known
    # call sites before its empty offender list means anything.
    assert label_seam_call_sites, (
        "the scan found no transition/apply_issue_labels call sites under src/ "
        "-- the guard is blind, not green"
    )
    assert [loc for loc, passes_state_path in label_seam_call_sites if not passes_state_path] == []
