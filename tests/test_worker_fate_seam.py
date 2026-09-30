"""Guard test for ``worker_fate.PersistedFailure``'s single-point-of-enforcement claim.

wf-review-opus.md finding B7: the raw ``dead_worker_failure_kind`` state-entry
key must be read through ``worker_fate.persisted_failure()`` everywhere except
the two files that own it:

* ``state.py`` -- defines ``record_dead_worker_failure_kind`` /
  ``clear_dead_worker_failure_kind``, the only functions allowed to write it.
* ``worker_fate.py`` -- this module's own read accessor (``persisted_failure``).

Without this test the claim in ``PersistedFailure``'s docstring is just prose,
and a new direct ``entry.get("dead_worker_failure_kind")`` read lands
unguarded -- exactly the failure scenario B7 describes ("the 'single point of
enforcement' claim is false today").

The check is AST-based, not a text/regex grep over the quoted literal,
because the literal also appears legitimately in places that are not a
*read* of a persisted classification:

* prose in docstrings/comments illustrating the old call this guard replaced
  (``orphaned_worker_sweep.py``'s own migration comment quotes it);
* ``reconcile.py``, which *writes* the field through its drift-item apply
  path (a dict-literal key, not a ``.get()``/subscript read) and uses the
  same string as an unrelated ``DriftItem.kind`` discriminant tag;
* ``unescalate_reset_fields.py``, a tuple of field *names* to clear on
  re-arm, never read for its value.

An AST walk that only flags ``<expr>.get("dead_worker_failure_kind", ...)``
calls and ``<expr>["dead_worker_failure_kind"]`` reads (``Load`` context,
never ``Store``/``Del``) naturally excludes all of the above without a
hand-maintained exemption list beyond the two owner files -- matching the
comment already left in ``orphaned_worker_sweep.py``: "keeps the raw key
name out of every module but state.py/worker_fate.py".
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work

_ALLOWED_FILES = frozenset({"state.py", "worker_fate.py"})
_TARGET_KEY = "dead_worker_failure_kind"


def _src_root() -> Path:
    # __path__, not __file__: the latter can be None for a namespace package.
    return Path(next(iter(charlie_work.__path__)))


def _raw_read_lines(tree: ast.AST) -> list[int]:
    """Line numbers of any raw read of the ``dead_worker_failure_kind`` key.

    Flags exactly two shapes, both reads:
    ``<expr>.get("dead_worker_failure_kind"[, default])`` and
    ``<expr>["dead_worker_failure_kind"]`` in a ``Load`` context. A dict
    literal's key (construction/write), a keyword-argument value, and an
    equality comparison against an unrelated field are all different AST
    shapes and are not matched.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == _TARGET_KEY
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == _TARGET_KEY
        ):
            lines.append(node.lineno)
    return lines


def test_raw_dead_worker_failure_kind_key_is_confined_to_allowed_files() -> None:
    """Only the owner and this module's accessor may read the literal key;
    every other consumer must go through ``worker_fate.persisted_failure()``.
    """
    root = _src_root()
    offenders: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name in _ALLOWED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = _raw_read_lines(tree)
        if lines:
            offenders[str(path.relative_to(root))] = lines

    assert offenders == {}, (
        "raw 'dead_worker_failure_kind' reads outside the allow-list -- route "
        f"through worker_fate.persisted_failure() instead: {offenders}"
    )


def test_the_guard_catches_get_and_subscript_reads() -> None:
    """Positive control: both shapes the guard targets actually get flagged.

    Without this, a detector that stopped matching (e.g. after some future
    ast.Subscript/ast.Index version skew) would leave the guard above
    vacuously green forever -- an empty ``offenders`` dict is consistent with
    both "nothing violates the rule" and "the check stopped checking".
    """
    get_snippet = ast.parse('kind = entry.get("dead_worker_failure_kind")')
    subscript_snippet = ast.parse('kind = entry["dead_worker_failure_kind"]')

    assert _raw_read_lines(get_snippet) == [1]
    assert _raw_read_lines(subscript_snippet) == [1]


def test_the_guard_does_not_flag_writes_or_unrelated_comparisons() -> None:
    """Negative control: the exact shapes ``reconcile.py`` legitimately uses.

    Proves the detector is precise rather than merely narrow -- a dict-literal
    key (construction), a keyword-argument value, a subscript assignment, and
    an equality comparison against an unrelated field must all pass clean, or
    this guard would force ``reconcile.py`` onto a hand-maintained exemption
    list the way the retired text-regex version of this check needed.
    """
    dict_literal = ast.parse('new_issues[key] = {**entry, "dead_worker_failure_kind": kind}')
    kwarg_value = ast.parse('DriftItem(kind="dead_worker_failure_kind", issue_number=n)')
    subscript_write = ast.parse('entry["dead_worker_failure_kind"] = kind')
    subscript_delete = ast.parse('del entry["dead_worker_failure_kind"]')
    unrelated_compare = ast.parse('x = item.kind == "dead_worker_failure_kind"')

    for snippet in (dict_literal, kwarg_value, subscript_write, subscript_delete):
        assert _raw_read_lines(snippet) == []
    # The comparison's constant is still a Load-context Compare operand, not a
    # Subscript/`.get()` read -- it must not match either.
    assert _raw_read_lines(unrelated_compare) == []


def test_the_allow_listed_files_still_use_the_literal() -> None:
    """Sanity check the allow-list is not stale.

    If ``state.py`` or ``worker_fate.py`` stopped using the literal key (e.g.
    the logic was renamed or moved elsewhere), it would silently fall out of
    the set of usages this test has actually reviewed and excused, while
    still being exempted here on trust alone.
    """
    root = _src_root()
    for name in _ALLOWED_FILES:
        path = root / name
        assert path.exists(), f"allow-listed file no longer exists: {name}"
        assert f'"{_TARGET_KEY}"' in path.read_text(encoding="utf-8"), (
            f"allow-listed file no longer references the literal key: {name}"
        )


# --------------------------------------------------------------------------
# Write side (B7 / rule 6, wf-r2-s4): ``worker_fate.persist_failure`` is the
# single write primitive. The read guard above confines *reads* of the raw key;
# this confines *writes* the same way.
# --------------------------------------------------------------------------

_WRITE_PRIMITIVE = "record_dead_worker_failure_kind"


def _raw_write_lines(tree: ast.AST) -> list[int]:
    """Line numbers of any raw write of the ``dead_worker_failure_kind`` stamp.

    Flags three shapes: a call to ``record_dead_worker_failure_kind`` (by bare
    or attribute name), a subscript *assignment* to the key
    (``entry["dead_worker_failure_kind"] = ...``), and a dict-literal key
    (``{**entry, "dead_worker_failure_kind": kind}``). Clears
    (``entry.pop``/``del entry[...]``, ``clear_dead_worker_failure_kind``) and
    the ``UNESCALATE_ISSUE_RESET_FIELDS`` tuple of field *names* are different
    AST shapes and stay legal -- resetting is not writing a classification.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name == _WRITE_PRIMITIVE:
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == _TARGET_KEY
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == _TARGET_KEY:
                    lines.append(key.lineno)
    return lines


def test_dead_worker_failure_kind_writes_are_confined_to_persist_primitive() -> None:
    """Only ``state.py`` (the field's owner) and ``worker_fate.py`` (which owns
    ``persist_failure``) may write the stamp; every other writer -- the reap
    sites, ``dead_worker_classification``, ``reconcile`` -- calls
    ``worker_fate.persist_failure``.
    """
    root = _src_root()
    offenders: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name in _ALLOWED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = _raw_write_lines(tree)
        if lines:
            offenders[str(path.relative_to(root))] = lines

    assert offenders == {}, (
        "raw 'dead_worker_failure_kind' writes outside the allow-list -- route "
        f"through worker_fate.persist_failure() instead: {offenders}"
    )


def test_the_write_guard_catches_every_raw_write_shape() -> None:
    """Positive control, one snippet per shape -- including the exact shapes the
    pre-s4 ``dead_worker_reap.py`` / ``dead_worker_classification.py`` /
    ``reconcile.py`` used -- so an empty offenders dict above cannot mean the
    detector stopped detecting.
    """
    call = ast.parse("state = record_dead_worker_failure_kind(state, n, kind)")
    attr_call = ast.parse("state = st.record_dead_worker_failure_kind(state, n, kind)")
    subscript = ast.parse('entry["dead_worker_failure_kind"] = failure_kind')
    dict_literal = ast.parse('new[k] = {**existing, "dead_worker_failure_kind": kind}')

    for snippet in (call, attr_call, subscript, dict_literal):
        assert _raw_write_lines(snippet) == [1]


def test_the_write_guard_allows_clears_and_field_name_tuples() -> None:
    """Negative control: resets are not writes, and neither is naming the field."""
    for source in (
        'entry.pop("dead_worker_failure_kind", None)',
        'del entry["dead_worker_failure_kind"]',
        "clear_dead_worker_failure_kind(entry)",
        'FIELDS = ("dead_worker_failure_kind", "other")',
        'DriftItem(kind="dead_worker_failure_kind", failure_kind=kind)',
        'x = entry.get("dead_worker_failure_kind")',
    ):
        assert _raw_write_lines(ast.parse(source)) == [], source


# --------------------------------------------------------------------------
# Liveness (B6 / wf-r2-s6): a public ``worker_fate`` function nobody outside
# the module calls is a signal without a consumer -- ``stale_evidence_events``
# sat in exactly that state (tested, exported, never emitted from three of
# four ``resolve_fate`` consumers) until ``report_stale_evidence`` wired it.
# --------------------------------------------------------------------------


def _public_function_names(tree: ast.AST) -> set[str]:
    """Module-level, non-underscore function names defined in ``tree``."""
    return {
        node.name
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("_")
    }


def _referenced_names(tree: ast.AST) -> set[str]:
    """Every name ``tree`` references: bare names, attribute names, imports."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def _functions_without_production_caller(fate_source: str, other_sources: list[str]) -> set[str]:
    """Public functions unreachable from any production reference.

    A function is reachable when another ``src`` module references it, or when
    a reachable top-level def/class in ``worker_fate.py`` references it (so a
    helper such as ``stale_evidence_events`` counts once its caller
    ``report_stale_evidence`` is itself consumed).
    """
    fate_tree = ast.parse(fate_source)
    defs = {
        node.name: _referenced_names(node)
        for node in fate_tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    reachable: set[str] = set()
    for source in other_sources:
        reachable |= _referenced_names(ast.parse(source))
    frontier = [name for name in defs if name in reachable]
    reachable = set(frontier)
    while frontier:
        for ref in defs[frontier.pop()]:
            if ref in defs and ref not in reachable:
                reachable.add(ref)
                frontier.append(ref)
    return _public_function_names(fate_tree) - reachable


def _production_sources(root: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if path.name != "worker_fate.py"
    ]


def test_every_public_worker_fate_function_has_a_production_caller() -> None:
    """Every public ``worker_fate`` function is referenced by another ``src``
    module. The set is derived from the module's own AST, so a new function
    is covered the moment it is added -- there is no list to keep current.
    """
    root = _src_root()
    fate_source = (root / "worker_fate.py").read_text(encoding="utf-8")

    orphans = _functions_without_production_caller(fate_source, _production_sources(root))

    assert orphans == set(), (
        "public worker_fate functions with no production caller -- wire them into "
        f"a consumer or delete them: {sorted(orphans)}"
    )


def test_the_caller_guard_flags_a_function_nobody_calls() -> None:
    """Positive control: the pre-s6 shape (``stale_evidence_events`` defined and
    tested but not called by any production consumer) is flagged, while a
    function reached via attribute access or ``from`` import is not.
    """
    fate_source = (
        "def emit_thing(x): ...\n"
        "def used_attr(x): ...\n"
        "def used_import(x): ...\n"
        "def helper(x): ...\n"
        "def _private(): ...\n"
        "def entry(x):\n    return helper(x)\n"
    )
    consumers = [
        "worker_fate.used_attr(1)\nworker_fate.entry(0)",
        "from .worker_fate import used_import\nused_import(2)",
    ]

    # ``helper`` is reached transitively through the consumed ``entry``.
    assert _functions_without_production_caller(fate_source, consumers) == {"emit_thing"}
    assert _functions_without_production_caller(fate_source, []) == {
        "emit_thing",
        "used_attr",
        "used_import",
        "helper",
        "entry",
    }
