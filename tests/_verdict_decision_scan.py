"""Shared scanner helpers for the issue #1265 verdict-provenance AST scans.

Issue #3038: hoisted verbatim out of ``tests/test_verdict_provenance_enforcement.py``
(issue #1284's no-cross-test-import invariant, plus issue #1442's file-size
ratchet) so the write-site scan and its text gate can be guarded by a test
module other than the over-cap enforcement file. Names are underscore-private
by history; the bare-name import convention of ``_review_fixtures`` and
friends applies.
"""

from __future__ import annotations

import ast
import unicodedata

# Keep in sync with the AC5 scanner contract in
# tests/test_verdict_provenance_enforcement.py: the exemption matches by
# function *name*, so the exempt-site census (len == 1) is what stays
# load-bearing rather than this set's membership.
_PROVENANCE_WRITER_EXEMPT_FUNCTIONS = frozenset({"merge_authorize"})


# ``record_local_review`` (issue #1844's no-remote lane) is the same
# verdict-ingest contract under a different name -- required
# verdict_provenance kwarg, same provenance literals -- so its call
# sites are held to the identical mapping table. One tuple feeds both the
# AST matcher and ``_may_call_record_review``'s text gate, so a name added
# here cannot be matched by the one and dropped by the other.
_RECORD_REVIEW_ATTRS = ("record_review", "record_local_review")


def _nfkc_text(text: str) -> str:
    """NFKC-normalize ``text`` when it is not ASCII, else pass it through."""
    return unicodedata.normalize("NFKC", text)


def _may_call_record_review(text: str) -> bool:
    """Cheap, sound pre-filter: could ``text`` hold a ``record_review`` call site?

    Issue #2931: the call-site scan parsed and walked all ~510 files under
    ``src/`` to find the handful that call ``record_review`` /
    ``record_local_review``. An ``ast.Attribute`` matched on ``.attr`` must
    spell that name in its source, so a file whose text lacks every name
    cannot hold a site and need not be parsed -- the ``source_text`` needle
    gate the other AST-scanning tests use.

    Sound, not just fast: the parser NFKC-normalizes identifiers, so
    ``x.ｒecord_review(...)`` (fullwidth ``r``) is a ``record_review``
    attribute whose raw text lacks the needle. A guard that exists to catch
    bypasses must not fail open on that rephrase, so non-ASCII text is
    normalized before the check (``str.isascii`` keeps the common case free).
    """
    return any(attr in _nfkc_text(text) for attr in _RECORD_REVIEW_ATTRS)


def _may_write_review_decision(text: str) -> bool:
    """Cheap, sound pre-filter: could ``text`` hold a ``record_decision`` call?

    Same soundness argument as ``_may_call_record_review`` (issue #2931,
    applied to issue #3038's write-site half of the AC5 scan): ``ast.Call``
    sites for ``record_decision`` are matched by identifier spelling, either
    a bare ``Name`` (``record_decision(...)``) or an ``Attribute`` whose
    ``.attr`` spells the same name, and the parser NFKC-normalizes
    identifiers, so non-ASCII text is normalized before the needle check.
    A file whose (normalized) text lacks ``record_decision`` cannot hold a
    site this scanner matches, so skipping it drops nothing.
    """
    return "record_decision" in _nfkc_text(text)


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(id(current))
    return None


def _dict_var_has_key(func_node: ast.AST, var_name: str, key: str) -> bool:
    """True if ``var_name`` is ever assigned a Dict literal containing
    ``key``, OR ever subscript-assigned that key directly (the
    read-modify-write shape ``_update_approval_head`` uses: ``updated_decision
    = dict(current_decision)`` followed by
    ``updated_decision["verdict_provenance"] = "carried_forward"``).

    Covers ``ast.AnnAssign`` as well as ``ast.Assign`` -- the annotated
    binding ``decision_payload: dict[str, Any] = {...}`` is the same write
    with a type annotation (issue #1844's ``record_local_review`` binds it
    that way); an Assign-only walk would fail open on the rephrase."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == var_name
                and isinstance(node.value, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == key for k in node.value.keys)
            ):
                return True
            continue
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == var_name
                and isinstance(node.value, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == key for k in node.value.keys)
            ):
                return True
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == var_name
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key
            ):
                return True
    return False


def _write_site_supplies_provenance(call_node: ast.Call, func_node: ast.AST) -> bool:
    if len(call_node.args) < 2:
        return False
    value_arg = call_node.args[1]
    if isinstance(value_arg, ast.Dict):
        return any(
            isinstance(k, ast.Constant) and k.value == "verdict_provenance" for k in value_arg.keys
        )
    if isinstance(value_arg, ast.Name):
        return _dict_var_has_key(func_node, value_arg.id, "verdict_provenance")
    return False


def _review_decision_write_sites(
    tree: ast.AST,
) -> tuple[list[tuple[str, ast.Call, ast.AST]], list[tuple[str, ast.Call, ast.AST]]]:
    """Returns ``(checked_sites, exempted_sites)``.

    Issue #1362 Stage 2: matches calls to ``record_decision`` -- the single
    writer of review-decision.json -- rather than the old direct
    ``self._write_json(decision_path, ...)`` call sites (all four of which
    were converted to call ``record_decision`` instead). No separate
    "does this resolve to a review-decision.json path" check is needed the
    way the old scan required one: ``record_decision`` writes nothing else,
    so the function name alone scopes every match correctly.

    Issue #1650: receiver-agnostic, mirroring WriteGate R9's
    ``_primitive_name`` in ``tests/test_write_gate_enforcement.py``. A call
    is matched whether it is a bare ``Name`` (``record_decision(...)``) or
    an ``Attribute`` whose ``.attr == "record_decision"``
    (``_wf.record_decision(...)``). The #1627 ``_wf.`` namespace rebind that
    moved bodies apply must NOT hide a write site from this scan -- the
    prior receiver-sensitive (``ast.Name``-only) matcher silently dropped
    ``_wf.record_decision(...)`` calls from coverage, and the guard still
    passed (fewer sites to check), so the erosion was silent.

    Exempted sites are *collected*, not silently skipped, because the
    exemption in ``_PROVENANCE_WRITER_EXEMPT_FUNCTIONS`` matches by function
    *name*, not identity. Matching by name alone fails open: a future
    function anywhere under the scanned root that happens to be named
    ``merge_authorize`` would silently inherit the exemption and could write
    review-decision.json with no provenance while this scanner stays green.
    Returning the exempted sites lets the caller assert the exemption still
    resolves to exactly the one known, deliberate exclusion instead of
    trusting the name match blindly.
    """
    parents = _build_parent_map(tree)
    checked: list[tuple[str, ast.Call, ast.AST]] = []
    exempted: list[tuple[str, ast.Call, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        is_record_decision = (isinstance(func, ast.Name) and func.id == "record_decision") or (
            isinstance(func, ast.Attribute) and func.attr == "record_decision"
        )
        if not is_record_decision:
            continue
        func_node = _enclosing_function(node, parents)
        if func_node is None:
            continue
        if func_node.name in _PROVENANCE_WRITER_EXEMPT_FUNCTIONS:
            exempted.append((func_node.name, node, func_node))
        else:
            checked.append((func_node.name, node, func_node))
    return checked, exempted
