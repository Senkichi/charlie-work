"""AST guard: every ``sanitize_env(`` call site under ``src/`` must pass ``config=``.

Split out of ``tests/test_env_sanitize.py`` in the #3086 rework: landing the
module-scoped call-site scan there pushed that file over the 800-line cap with
no ratchet baseline entry (the issue #1442 ratchet treats a file with no entry
as mark 0), and the ratchet's rule is that new code splits into seam-named
siblings rather than receiving a mark (the #2992 precedent for over-cap test
files). The fixture and the guard test move verbatim.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _src_ast import parsed, source_files, source_text

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"

# ---------------------------------------------------------------------------
# sanitize_env: call-site config= guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sanitize_env_call_sites() -> list[tuple[str, bool]]:
    """Every ``sanitize_env(`` call site under ``src/`` as ``(file:line, passes_config)``.

    A ``sanitize_env`` call spells its identifier verbatim in the file text
    -- the AST check below reads ``ast.Name.id`` and ``ast.Attribute.attr``,
    both of which hold the token -- so a file whose ``source_text`` lacks it
    cannot hold a call site and is skipped before ``parsed()`` (the
    sibling-filter shape of issue #2606's family; the residual -- an
    identifier assembled from adjacent string literals -- is out of scope:
    this guard does tripwire duty for accidental omissions, not adversarial
    ones). Measured at this tree: 519 scanned files shrink to 14 candidates.

    Scanning the package is this module's dominant cost, so it is charged to
    a module fixture -- the test's ``setup`` -- instead of the ``call`` phase
    the ledger times (the accepted #2662/#2867/#2875/#2882 remedy family,
    applied to this guard for #3086). The scan's output is shared, read-only
    data; if the fixture's setup ever flags, ``ci-fleet ledger report
    --section slow-fixtures`` is its owner.
    """
    found: list[tuple[str, bool]] = []
    for py_file in source_files(_SRC_ROOT):
        if "sanitize_env" not in source_text(py_file):
            continue
        tree = parsed(py_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (isinstance(func, ast.Name) and func.id == "sanitize_env") or (
                isinstance(func, ast.Attribute) and func.attr == "sanitize_env"
            )
            if called:
                location = f"{py_file.relative_to(_SRC_ROOT.parent)}:{node.lineno}"
                found.append((location, any(kw.arg == "config" for kw in node.keywords)))
    return found


def test_sanitize_env_call_sites_in_src_all_pass_config(
    sanitize_env_call_sites: list[tuple[str, bool]],
) -> None:
    """AST guard: every ``sanitize_env(`` call in src/ must pass ``config=``.

    ``sanitize_env``'s ``config`` default is None, which fails OPEN — a call
    site that forgets it hands the worker a base env that still carries the
    orchestrator's api-provider keys (issue #2795). Because every in-tree launch
    site plausibly has the orchestrator config in scope, forgetting ``config=``
    is always a bug, never a deliberate choice, so instead of relying on a
    reviewer noticing at diff time this test mechanically fails whenever a
    ``src/`` call site omits the keyword (or a new call site appears without
    it).
    """
    # Positive control: a scan that matched nothing reports "zero offenders"
    # the same as a clean tree, so the guard must prove it sees the known
    # call sites before its empty offender list means anything.
    assert sanitize_env_call_sites, (
        "the scan found no sanitize_env( call sites under src/ -- the guard is blind, not green"
    )
    bad = [location for location, passes_config in sanitize_env_call_sites if not passes_config]
    assert not bad, (
        "every sanitize_env( call in src/ must pass config= (the None default "
        f"fails open and would leave api-provider keys in the worker base env): "
        f"{bad}"
    )
