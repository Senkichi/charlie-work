"""Lint-style enforcement for issue #1362 Stage 3 (state as declared cache).

``state.json``'s ``prs[N].decision`` / ``reviewed_head_sha`` / ``decision_path``
fields are a *cache* of the file-first ``review_decision``/
``resolve_decision_payload`` resolution -- never a second source of truth.
Exactly six production sites are permitted to set new values for these keys
on a PR's ``state["prs"][pr_number]`` entry:

* ``review`` -- the new-dispatch placeholder write, immediately after (and
  sourced from) its own ``record_decision(..., archive_round=False)`` call.
* ``record_review`` -- the reviewer-verdict writer, immediately after its own
  ``record_decision(...)`` call.
* ``merge_ready`` -- the carry-forward branch, immediately after calling
  ``_update_approval_head`` (which itself calls ``record_decision``).
* ``_route_to_rework`` -- deliberately *preserves* prior values via
  ``pr_entry.get(...)``; the verdict file is untouched by design.
* ``_update_approval_head`` -- the carry-forward writer, immediately after
  its own ``record_decision(..., archive_round=False)`` call.
* ``_refresh_pr_decision_cache`` -- the Stage 3 loop-boundary refresh, the
  5th/6th sanctioned site: it mirrors ``review_decision()``'s file-first
  read into state for every PR that had no verdict activity this pass.

This mirrors ``test_no_unlocked_load_state_in_production_code``
(``tests/test_load_state_locked.py``): an AST scan over ``src/charlie_work``,
not a value-form grep -- rephrasing the write as ``.update()``,
``setdefault()``, a bare three-level subscript, or a conditional ``**``-spread
must still be caught, so a new writer growing back (the failure mode issue
#1362 exists to close off) fails the test until explicitly added, with
justification, to ``_SANCTIONED_FUNCTIONS`` above.

Deliberately scoped to targets shaped ``<expr>["prs"][<expr>]`` (the PR-state
dict itself), not any local dict that happens to reuse the key name
"decision" for an unrelated purpose (e.g. the unauthorized-merge audit
sidecar's ``record[...]  = {"decision": ...}``, or a tripwire-status report's
local ``entry.update(...)``) -- those are not part of the state.json cache
this issue governs.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

_DECISION_KEYS = {"decision", "reviewed_head_sha", "decision_path"}

# The exact, exhaustive set of functions permitted to introduce a NEW value
# for one of _DECISION_KEYS onto a state["prs"][pr_number] entry. Adding a
# name here must come with the same "sourced from a fresh record_decision
# call, never a stale re-read" justification the existing six sites document
# in their own docstrings/comments -- see this module's docstring.
_SANCTIONED_FUNCTIONS = frozenset(
    {
        "review",
        "record_review",
        "merge_ready",
        "_route_to_rework",
        "_update_approval_head",
        "_refresh_pr_decision_cache",
    }
)


def _is_prs_subscript(node: ast.AST) -> bool:
    """True for a target/receiver shaped ``<expr>["prs"][<expr>]``."""
    if not isinstance(node, ast.Subscript):
        return False
    outer = node.value
    if not isinstance(outer, ast.Subscript):
        return False
    key = outer.slice
    return isinstance(key, ast.Constant) and key.value == "prs"


def _dict_literal_decision_keys(node: ast.AST) -> list[str]:
    """Decision-key hits inside a ``Dict`` literal, including ``**`` spreads.

    Recurses into ``**{...}`` and ``**({...} if cond else {})`` spread
    entries (the shape ``review``'s new-dispatch site uses:
    ``**({"decision": "pending"} if voided_stale_verdict else {})``) so a
    key hidden behind a conditional unpack is not invisible to the scanner --
    a value-form grep would miss exactly this rephrasing.
    """
    hits: list[str] = []
    if not isinstance(node, ast.Dict):
        return hits
    for key, value in zip(node.keys, node.values):
        if key is not None:
            if isinstance(key, ast.Constant) and key.value in _DECISION_KEYS:
                hits.append(key.value)
            continue
        # A `**value` spread entry: recurse if the spread itself is (or
        # branches to) a dict literal.
        candidates: list[ast.AST] = []
        if isinstance(value, ast.Dict):
            candidates.append(value)
        elif isinstance(value, ast.IfExp):
            if isinstance(value.body, ast.Dict):
                candidates.append(value.body)
            if isinstance(value.orelse, ast.Dict):
                candidates.append(value.orelse)
        for candidate in candidates:
            hits.extend(_dict_literal_decision_keys(candidate))
    return hits


class _PrStateWriteVisitor(ast.NodeVisitor):
    """Find every new-value assignment to _DECISION_KEYS on a PR-state dict.

    Three shapes are covered, matching the constraint that a rephrased write
    (subscript vs ``.update()`` vs ``setdefault()`` vs a conditional
    ``**``-merge) must still be caught rather than passing by accident:

    1. ``x["prs"][y] = {...}`` -- a whole-dict-literal replace (every real
       site in this codebase today uses this shape) containing a decision
       key, direct or via a ``**`` spread (see :func:`_dict_literal_decision_keys`).
    2. ``x["prs"][y]["decision"] = ...`` -- a bare three-level subscript
       assignment straight onto one of the keys (the shape the ``record_decision``
       docstring says does not exist anywhere in ``src/`` today).
    3. ``x["prs"][y].update({...})`` / ``x["prs"][y].setdefault("decision", ...)``
       -- the two remaining standard-library ways to introduce a new key.
    """

    def __init__(self) -> None:
        self._func_stack: list[str] = []
        self.hits: list[tuple[int, str, str]] = []  # (lineno, func_name, key)

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def _current_function(self) -> str:
        return self._func_stack[-1] if self._func_stack else "<module>"

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if _is_prs_subscript(target):
                for key in _dict_literal_decision_keys(node.value):
                    self.hits.append((node.lineno, self._current_function(), key))
            elif (
                isinstance(target, ast.Subscript)
                and _is_prs_subscript(target.value)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value in _DECISION_KEYS
            ):
                self.hits.append((node.lineno, self._current_function(), target.slice.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and _is_prs_subscript(node.func.value):
            if node.func.attr == "update" and node.args and isinstance(node.args[0], ast.Dict):
                for key in _dict_literal_decision_keys(node.args[0]):
                    self.hits.append((node.lineno, self._current_function(), key))
            elif (
                node.func.attr == "setdefault"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in _DECISION_KEYS
            ):
                self.hits.append((node.lineno, self._current_function(), node.args[0].value))
        self.generic_visit(node)


def _scan(tree: ast.Module) -> list[tuple[int, str, str]]:
    visitor = _PrStateWriteVisitor()
    visitor.visit(tree)
    return visitor.hits


def test_pr_state_decision_writes_are_all_sanctioned() -> None:
    """Every real write site in src/charlie_work must be a sanctioned function."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "charlie_work"

    violations: list[str] = []
    functions_seen: set[str] = set()
    for source_file in sorted(src_root.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for lineno, func_name, key in _scan(tree):
            functions_seen.add(func_name)
            if func_name not in _SANCTIONED_FUNCTIONS:
                violations.append(
                    f"{source_file.relative_to(src_root)}:{lineno} in {func_name}() "
                    f"sets {key!r} on a PR-state dict but is not a sanctioned site"
                )

    assert not violations, (
        "New/unsanctioned production write(s) to a PR-state decision field:\n"
        + "\n".join(violations)
        + "\nIf this is a deliberate new writer, add it to _SANCTIONED_FUNCTIONS "
        "above with the same 'sourced from a fresh record_decision call, never "
        "a stale re-read' justification the existing six sites document."
    )

    # Every sanctioned name must still correspond to a real write in the
    # tree -- an entry that stops being used (the function was renamed,
    # inlined, or its write deleted) must not sit here silently allow-listing
    # nothing, mirroring the staleness check on
    # test_legacy_forwarding_wrapper_scopes_are_not_stale in
    # test_write_gate_enforcement.py.
    stale = _SANCTIONED_FUNCTIONS - functions_seen
    assert not stale, (
        f"_SANCTIONED_FUNCTIONS lists {sorted(stale)} but no production write to a "
        "PR-state decision field was found in that function anymore -- remove the "
        "stale entry (or the write it guarded was deleted/renamed and this list is "
        "now over-permissive)."
    )


def test_scanner_fails_closed_on_a_rephrased_write() -> None:
    """The scanner must catch the write regardless of its syntactic shape.

    An AST guard that only matches the CURRENT spelling of a forbidden write
    fails open the moment someone rephrases it (subscript -> .update() ->
    setdefault() -> a conditional **-merge). This pins all four shapes
    against a synthetic module so a future edit to the scanner itself that
    narrows it back to one shape is caught here, not in production.
    """
    samples = {
        "whole_dict_replace": """
            def _rogue_writer(state, pr_number):
                state["prs"][str(pr_number)] = {"decision": "approved"}
        """,
        "bare_subscript": """
            def _rogue_writer(state, pr_number):
                state["prs"][str(pr_number)]["reviewed_head_sha"] = "deadbeef"
        """,
        "dot_update": """
            def _rogue_writer(state, pr_number):
                state["prs"][str(pr_number)].update({"decision_path": "x"})
        """,
        "setdefault": """
            def _rogue_writer(state, pr_number):
                state["prs"][str(pr_number)].setdefault("decision", "pending")
        """,
        "conditional_spread": """
            def _rogue_writer(state, pr_number, cond):
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    **({"decision": "pending"} if cond else {}),
                }
        """,
    }
    for name, source in samples.items():
        tree = ast.parse(textwrap.dedent(source))
        hits = _scan(tree)
        assert hits, f"scanner failed to catch the {name!r} rephrasing: {ast.dump(tree)}"
        assert all(func_name == "_rogue_writer" for _, func_name, _ in hits), (
            f"{name!r}: hit attributed to the wrong enclosing function: {hits}"
        )


def test_scanner_ignores_unrelated_local_dicts() -> None:
    """A local dict that reuses the key name 'decision' for something else
    (not the state["prs"] PR-state cache) must not be flagged -- this is the
    real shape of ``_announce_unauthorized_merges``'s audit-sidecar record
    and ``tripwire_status``'s report entry, both of which use "decision" as a
    key on a dict that is NOT ``state["prs"][...]``.
    """
    source = """
        def _unrelated(state, key, candidate):
            record = state.get(key)
            record[str(candidate["pr"])] = {
                "decision": candidate.get("decision"),
                "reviewed_head_sha": candidate.get("reviewed_head_sha"),
            }
            entry = {"pr": candidate.get("pr")}
            entry.update({"decision": candidate.get("decision")})
    """
    tree = ast.parse(textwrap.dedent(source))
    assert _scan(tree) == []
