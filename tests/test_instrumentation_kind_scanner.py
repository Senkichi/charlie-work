"""Emit-site kind-scanner unit tests for ``charlie_work.instrumentation``.

Split out of ``tests/test_instrumentation.py`` (issue #1569, Track-1):
the fail-closed coverage of the AST scanner's unresolvable-form handling
(#995/#1029), exercised against synthetic sources through the machinery
in ``tests/_instrumentation_kind_scanner.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from _instrumentation_kind_scanner import (
    _call_node_for,
    _has_explicit_level,
    _scan_sweep_append_kinds,
    _scan_tree,
)


def test_sweep_append_kind_scan_flags_unresolvable_element_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """#1029 (sweep-kind direction): an unresolvable ``sweep_events.append((kind,
    payload))`` element must surface as unresolved, not vanish from ``found``
    via ``found |= _resolve_literal(...)`` unioning in an empty result.

    Asserts on the observable consequence -- the site is flagged -- rather
    than on ``found`` being empty, since an empty ``found`` is also what a
    correctly-behaving scan produces when nothing resolves; only the
    unresolved list distinguishes "silently dropped" from "surfaced".
    """
    (tmp_path / "fixture.py").write_text(
        "def _append(kind_choice):\n    sweep_events.append((kind_choice(), {'x': 1}))\n",
        encoding="utf-8",
    )
    found, unresolved = _scan_sweep_append_kinds(tmp_path)
    assert not found
    assert unresolved, "unresolvable sweep_events.append kind element was silently dropped"
    assert "kind_choice()" in unresolved[0]


def test_scanner_module_constant_with_unresolvable_value_not_silently_dropped() -> None:
    """#1029 (module-constant direction): a module-level name reassigned on
    two branches -- one a string literal, one unresolvable (a function call)
    -- must not be treated as resolved to just the literal branch.

    A single-assignment unresolvable constant doesn't discriminate this bug:
    an empty accumulator unioned with one unresolvable ``None``/empty result
    stays empty either way. The bug only shows up with a *mix*, exactly like
    ``level="warning" if cond else None`` -- the old code's
    ``resolved |= _resolve_literal(...) or set()`` would union in the
    resolvable branch (``{"kind_a"}``) and silently drop the unresolvable one,
    ending up "resolved" to ``{"kind_a"}`` when the true value could be
    anything ``_compute_kind()`` returns. Asserts the emit site is flagged
    *unresolved* -- not merely that ``"kind_a"`` is absent from ``used``,
    since that alone doesn't distinguish "silently dropped" (buggy: `used` is
    `{"kind_a"}`, non-empty) from "correctly surfaced" (fixed: `used` is
    empty because `unresolved` is used instead).
    """
    source = (
        "if _flag():\n"
        "    KIND = 'kind_a'\n"
        "else:\n"
        "    KIND = _compute_kind()\n\n\n"
        "def emit():\n    log_event(state_path, KIND, {})\n"
    )
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used, f"expected no resolvable kinds (one branch is unresolvable), got {used}"
    assert unresolved, (
        "module constant with a mixed resolvable/unresolvable value was dropped (#1029)"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def emit(kind):\n    log_event(state_path, kind, {})\n",
            id="bare-variable",
        ),
        pytest.param(
            "def emit(suffix):\n    log_event(state_path, f'job_{suffix}', {})\n",
            id="f-string-with-unresolvable-interpolation",
        ),
        pytest.param(
            "def emit(a, b):\n    log_event(state_path, a if _flag() else b, {})\n",
            id="conditional-expression-with-nonliteral-branches",
        ),
        pytest.param(
            "def emit(k):\n    log_event(state_path, kind=k, payload={})\n",
            id="keyword-passed-kind-argument",
        ),
        pytest.param(
            "def emit(kind, flag):\n    if flag:\n        kind = 'kind_a'\n    log_event(state_path, kind, {})\n",
            id="parameter-reassigned-on-one-branch-only",
        ),
    ],
)
def test_scanner_flags_unresolved_nonliteral_kind_forms(source: str) -> None:
    """#995 regression control: each non-literal form must be surfaced as
    *unresolved*, never silently dropped.

    This reproduces the exact #995 bug shape: the pre-fix scanner recognised
    only a bare string constant and a ternary between two literals, and
    treated every other shape -- a bare variable, an f-string, a ternary
    between two non-literal branches -- as contributing no kinds, which is
    indistinguishable from a legitimately kind-less call. A test that only
    exercises the literal case cannot detect a regression back to that
    behavior; this asserts the *unresolved* branch is reached instead.

    The keyword-passed-kind case guards the scanner's own `_locate_arg`
    fallback: a call site that passes ``kind=`` by keyword must be located
    the same as a positional one, not silently skipped because
    ``len(node.args) < 2``. The reassigned-on-one-branch case guards
    ``local_params``: a function parameter that gets a literal value on only
    one conditional path must still be treated as unresolved overall, since
    the parameter's own (unknown) value can reach the call site on the path
    that never executes the reassignment. Without the `local_params` check
    this would incorrectly resolve to ``{"kind_a"}``.
    """
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used, f"expected no resolvable kinds for a non-literal form, got {used}"
    assert unresolved, "non-literal kind form was silently dropped instead of flagged (#995)"


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            "def emit():\n    log_event(state_path, 'plain_kind', {})\n",
            {"plain_kind"},
            id="literal",
        ),
        pytest.param(
            "def emit(flag):\n    log_event(state_path, 'kind_a' if flag else 'kind_b', {})\n",
            {"kind_a", "kind_b"},
            id="ternary-of-literals",
        ),
        pytest.param(
            "def emit(flag):\n"
            "    if flag:\n"
            "        kind = 'kind_a'\n"
            "    else:\n"
            "        kind = 'kind_b'\n"
            "    log_event(state_path, kind, {})\n",
            {"kind_a", "kind_b"},
            id="multi-branch-local-assignment",
        ),
        pytest.param(
            "KIND = 'module_kind'\n\n\ndef emit():\n    log_event(state_path, KIND, {})\n",
            {"module_kind"},
            id="module-level-constant",
        ),
        pytest.param(
            "def emit(flag):\n"
            "    kind = 'kind_a' if flag else 'kind_b'\n"
            "    log_event(state_path, f'{kind}_sweep', {})\n",
            {"kind_a_sweep", "kind_b_sweep"},
            id="f-string-with-resolvable-interpolation",
        ),
    ],
)
def test_scanner_resolves_provably_finite_nonliteral_kind_forms(
    source: str, expected: set[str]
) -> None:
    """Contrast case for the regression control above: forms that ARE
    statically provable to a finite literal set must still resolve, not be
    over-eagerly flagged. Covers the generalised local-variable (including
    multi-branch if/elif), module-constant, and f-string tracing this fix
    adds -- each was a real emit site in this package before this fix.
    """
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert used == expected
    assert not unresolved


def test_scanner_accepts_explicit_level_without_requiring_kind_resolution() -> None:
    """A call site with a literal ``level=`` is self-classifying (mirrors
    ``log_event``'s runtime behavior) and does not need its kind registered
    or allow-listed, even if the kind itself is unresolvable."""
    source = (
        "def emit(dynamic_kind):\n    log_event(state_path, dynamic_kind, {}, level='warning')\n"
    )
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used
    assert not unresolved


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level='warning')\n",
            True,
            id="literal-str-only",
        ),
        pytest.param(
            "def emit(cond):\n"
            "    log_event(state_path, 'k', {}, level='warning' if cond else None)\n",
            False,
            id="str-or-none-conditional",
        ),
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level=None)\n",
            False,
            id="plain-none",
        ),
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level=_compute_level())\n",
            False,
            id="call-node",
        ),
        pytest.param(
            "def emit(lvl):\n    log_event(state_path, 'k', {}, level=lvl)\n",
            False,
            id="name-variable",
        ),
    ],
)
def test_has_explicit_level_fails_closed_on_none_admitting_expressions(
    source: str, expected: bool
) -> None:
    """#1029: a ``level=`` expression that can be ``None`` on some branch must
    not exempt the site from kind-registry verification.

    ``None`` is the documented "fall back to ``_LEVEL_BY_KIND``" signal (see
    ``instrumentation.log_event``), so a site whose level is conditionally
    ``None`` -- e.g. ``level="warning" if cond else None`` -- is still
    registry-dependent on the ``None`` path. Before this fix, ``_resolve_literal``
    unioned away the ``None`` branch (it only collects string constants) and
    ``_has_explicit_level`` saw only ``{"warning"}``, wrongly exempting the site.

    ``literal-str-only`` is the positive control: it must stay ``True`` so this
    test cannot pass merely because the helper started returning ``False``
    unconditionally. ``call-node`` and ``name-variable`` guard the same
    fail-closed behavior for expression shapes the resolver cannot statically
    reduce at all.
    """
    call, local_assigns, local_params = _call_node_for(source)
    assert _has_explicit_level(call, local_assigns, {}, local_params) is expected
