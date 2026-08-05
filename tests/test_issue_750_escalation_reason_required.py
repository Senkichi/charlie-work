"""Regression tests for issue #750.

Issue #750: status=escalated must always carry escalation_reason, and the
only place in workflow.py that constructs a ``status: "escalated"`` entry is
the shared ``_escalate_issue`` helper.

Issue #983: the structural guard below originally required the written value to
be an ``ast.Constant``, so any rephrasing of the same write -- a conditional
expression, a local binding, an f-string -- passed unseen. Matching the shape
of a forbidden value fails *open*; the scan now derives its exemption from the
call graph instead, which fails closed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp, _escalate_issue

from test_charlie_work import FakeGitHub


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_WORKFLOW = REPO_ROOT / "src" / "charlie_work" / "workflow.py"

FORBIDDEN_STATUS = "escalated"
HELPER_NAME = "_escalate_issue"


def _annotate_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


def _enclosing_function(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = getattr(current, "parent", None)
    return None


def _in_helper(node: ast.AST) -> bool:
    current: ast.AST | None = node
    while current is not None:
        if (
            isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef))
            and current.name == HELPER_NAME
        ):
            return True
        current = getattr(current, "parent", None)
    return False


def _function_calls_helper(fn: ast.AST | None) -> bool:
    if fn is None:
        return False
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name) and func.id == HELPER_NAME:
            return True
        if isinstance(func, ast.Attribute) and func.attr == HELPER_NAME:
            return True
    return False


def _is_direct_constant(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and value.value == FORBIDDEN_STATUS


def _mentions_forbidden(value: ast.AST) -> bool:
    """True when the forbidden constant appears anywhere in the value expression.

    Catches ``ast.IfExp`` (either branch), f-string literal parts, and any other
    expression that carries the literal syntactically. This is the property that
    makes the guard fail *closed*: a new way of phrasing the same write still
    contains the constant, so it is still seen.
    """
    return any(
        isinstance(sub, ast.Constant) and sub.value == FORBIDDEN_STATUS for sub in ast.walk(value)
    )


def _name_bound_to_forbidden(value: ast.AST, fn: ast.AST | None) -> bool:
    """True when ``value`` is a bare name the enclosing function binds to the
    forbidden constant -- the ``status = "escalated"`` local-binding form.

    The binding is tested with :func:`_mentions_forbidden`, not with a
    direct-constant check. Requiring the binding itself to be a plain constant
    would reintroduce the very fail-open bug this guard exists to close, one
    level down: ``status = "escalated" if flag else "escalated"`` binds the
    forbidden value without ever being an ``ast.Constant``.
    """
    if not isinstance(value, ast.Name) or fn is None:
        return False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and _mentions_forbidden(node.value):
            if any(isinstance(t, ast.Name) and t.id == value.id for t in node.targets):
                return True
    return False


def _status_writes(tree: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """Every syntactic write to a ``status`` key, as (node, value expression).

    Covers dict literals (which also catches ``**{...}`` unpacking and
    ``.update({...})``), subscript assignment, and keyword forms
    ``dict(status=...)`` / ``.update(status=...)``.
    """
    writes: list[tuple[ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "status":
                    writes.append((node, value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "status"
                ):
                    writes.append((node, node.value))
        elif isinstance(node, ast.Call):
            func = node.func
            is_dict_ctor = isinstance(func, ast.Name) and func.id == "dict"
            is_update = isinstance(func, ast.Attribute) and func.attr == "update"
            if is_dict_ctor or is_update:
                for kw in node.keywords:
                    if kw.arg == "status":
                        writes.append((node, kw.value))
    return writes


def find_escalation_violations(source: str, filename: str = "<src>") -> list[str]:
    """Two-tier scan for ``status="escalated"`` writes outside the helper.

    Tier A -- the value is the constant itself. Strictest rule: the write must
    live inside ``_escalate_issue``. This preserves the original guard's
    strength exactly.

    Tier B -- the value reaches the constant indirectly (conditional
    expression, local binding, f-string). Exempt when the enclosing function
    *is or calls* the helper, so a reason is provably written in the same
    frame. Deriving the exemption from the call graph -- rather than listing
    exempt file/line pairs -- is what keeps this from rotting as lines move.
    """
    tree = ast.parse(source, filename=filename)
    _annotate_parents(tree)
    violations: list[str] = []

    for node, value in _status_writes(tree):
        line = getattr(node, "lineno", "?")

        if _is_direct_constant(value):
            if not _in_helper(node):
                violations.append(f"tier-A direct constant at line {line}")
            continue

        fn = _enclosing_function(node)
        indirect = _mentions_forbidden(value) or _name_bound_to_forbidden(value, fn)
        if indirect and not (_in_helper(node) or _function_calls_helper(fn)):
            violations.append(f"tier-B indirect value at line {line}")

    return violations


def test_record_review_event_payload_includes_issue_number(tmp_path: Path) -> None:
    """The record_review event must include issue_number so events.db indexes it."""
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(456, "request_changes", summary="fix A")
    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(456, "request_changes", summary="fix B")
    fake_gh.pr_head_shas[456] = "sha-3"
    app.record_review(456, "request_changes", summary="fix C")

    state = load_state(paths.state_file)
    record_review_events = [e for e in state.get("events", []) if e.get("kind") == "record_review"]
    assert len(record_review_events) == 3
    for event in record_review_events:
        assert "issue_number" in event["payload"], event
        assert event["payload"]["issue_number"] == 123


def test_escalated_status_literal_is_only_in_helper() -> None:
    """Every hand-rolled ``status = "escalated"`` write must live inside the
    ``_escalate_issue`` helper.

    This is the structural backstop that prevents a future call site from
    hand-rolling a status="escalated" write without ``escalation_reason``.

    Two syntactic forms count as hand-rolling, because both are unambiguous
    and both occur in this file's own history:

    * a dict literal ``{"status": "escalated", ...}``
    * a subscript assignment ``record["status"] = "escalated"``

    Checking only the first form is not enough. After the #750 consolidation
    every escalating call site goes through the helper, which assigns from its
    ``status`` *parameter* rather than from a literal -- so zero dict literals
    remain and a literal-only check passes trivially, including if the helper
    were deleted and its call sites reverted to subscript assignment. The
    anti-vacuity assertion below anchors the guard against exactly that.

    A third class -- the value reaches the constant *indirectly* -- is handled
    by tier B rather than ignored. Issue #983: the original guard required the
    value node to be an ``ast.Constant``, so both

    * ``status = "escalated"`` bound to a local, then ``record["status"] = status``
      (workflow.py's dispatch-failure block), and
    * ``"status": "escalated" if escalated else decision`` (``record_review``)

    were invisible to it. Matching the *shape* of a bad value fails open the
    moment the value is rephrased; deriving the exemption from the call graph
    fails closed instead. Both sites above are legitimate -- each sits in a
    function that calls the helper, so a reason is written -- and tier B
    exempts them without an allowlist of line numbers to drift.
    """
    source = SRC_WORKFLOW.read_text(encoding="utf-8")
    violations = find_escalation_violations(source, str(SRC_WORKFLOW))

    assert not violations, (
        "Found status='escalated' write(s) outside _escalate_issue: " + ", ".join(violations)
    )

    # The scan above returning empty is only meaningful if it actually looked at
    # the indirect sites. Assert it saw them, so a future refactor that makes
    # them unparseable fails loudly instead of passing as "no violations".
    tree = ast.parse(source, filename=str(SRC_WORKFLOW))
    _annotate_parents(tree)
    indirect_seen = [
        node
        for node, value in _status_writes(tree)
        if not _is_direct_constant(value)
        and (
            _mentions_forbidden(value)
            or _name_bound_to_forbidden(value, _enclosing_function(node))
        )
    ]
    assert indirect_seen, (
        "No indirect status='escalated' writes found at all. Either workflow.py "
        "changed shape or the tier-B detector stopped matching; an empty result "
        "here is not evidence the invariant holds."
    )

    # Anti-vacuity anchor. The scan above legitimately finds nothing once every
    # call site routes through the helper, so an empty result is not by itself
    # evidence that the invariant holds -- it is equally consistent with the
    # helper having been removed. Assert the subject of the invariant still
    # exists and still owns the escalated status.
    helper = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_escalate_issue"
        ),
        None,
    )
    assert helper is not None, "_escalate_issue helper is gone; this guard would pass vacuously"

    kwonly = dict(zip(helper.args.kwonlyargs, helper.args.kw_defaults, strict=True))
    status_default = next(
        (default for arg, default in kwonly.items() if arg.arg == "status"),
        None,
    )
    assert isinstance(status_default, ast.Constant) and status_default.value == "escalated", (
        "_escalate_issue no longer defaults status to 'escalated'; the guard above "
        "would no longer be checking the escalation path"
    )
    assert any(arg.arg == "reason" for arg in helper.args.kwonlyargs), (
        "_escalate_issue no longer takes a keyword-only 'reason'; "
        "status='escalated' without a reason is representable again"
    )


GUARD_MUST_FLAG = {
    "ifexp": 'def leaky(entry, esc, decision):\n    return {"status": "escalated" if esc else decision}\n',
    "local_binding": (
        "def leaky(entry, flag):\n"
        '    status = "escalated" if False else "escalated"\n'
        '    entry["status"] = status\n'
    ),
    "name_bound_elsewhere": (
        "def leaky(entry, flag):\n"
        "    if flag:\n"
        '        status = "escalated"\n'
        "    else:\n"
        '        status = "dispatch_failed"\n'
        '    entry["status"] = status\n'
    ),
    "fstring": 'def leaky(entry):\n    entry["status"] = f"escalated"\n',
    "dict_ctor": 'def leaky():\n    return dict(status="escalated")\n',
    "update_kwarg": 'def leaky(entry):\n    entry.update(status="escalated")\n',
    "update_dict_literal": 'def leaky(entry):\n    entry.update({"status": "escalated"})\n',
    "plain_subscript": 'def leaky(entry):\n    entry["status"] = "escalated"\n',
}

GUARD_MUST_ALLOW = {
    "other_status": 'def fine(entry):\n    entry["status"] = "dispatch_failed"\n',
    "inside_helper": (
        "def _escalate_issue(state, n, *, reason):\n"
        '    return {"status": "escalated", "escalation_reason": reason}\n'
    ),
    "caller_exempt": (
        "def caller(state, n, esc, decision):\n"
        '    state = _escalate_issue(state, n, reason="x")\n'
        '    return {"status": "escalated" if esc else decision}\n'
    ),
}


@pytest.mark.parametrize("name", sorted(GUARD_MUST_FLAG))
def test_guard_flags_every_indirect_value_form(name: str) -> None:
    """Positive control for the guard itself.

    ``test_escalated_status_literal_is_only_in_helper`` asserts an *absence*,
    which is worthless if the detector cannot see anything. These synthetic
    sources are known-positive: each one is a way of writing
    ``status="escalated"`` outside the helper, and each must be reported. If a
    future edit narrows the detector back to constants-only, the rephrased
    forms here go silent and this test fails.
    """
    assert find_escalation_violations(GUARD_MUST_FLAG[name], name), (
        f"guard did not flag the {name!r} form; it has regressed to matching "
        "value shapes and will miss rephrased writes"
    )


@pytest.mark.parametrize("name", sorted(GUARD_MUST_ALLOW))
def test_guard_allows_legitimate_shapes(name: str) -> None:
    """Negative control: the tier-B widening must not manufacture false positives."""
    assert not find_escalation_violations(GUARD_MUST_ALLOW[name], name)


def test_escalate_issue_helper_requires_reason() -> None:
    """``reason`` is a required keyword-only argument."""
    state: dict[str, Any] = {"issues": {"1": {"number": 1}}, "prs": {}}
    with pytest.raises(TypeError):
        _escalate_issue(state, 1)  # type: ignore[call-arg]


def test_escalate_issue_helper_sets_issue_and_pr_reason() -> None:
    """The helper atomically writes status, escalation_reason, and reason_class."""
    state: dict[str, Any] = {
        "issues": {"1": {"number": 1, "status": "rework_requested"}},
        "prs": {"10": {"number": 10, "issue_number": 1, "status": "reviewing"}},
    }

    state = _escalate_issue(
        state,
        1,
        reason="test_reason",
        reason_class="mechanical",
        pr_number=10,
        pr_extra={"decision": "request_changes"},
    )

    issue = state["issues"]["1"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "test_reason"
    assert issue["reason_class"] == "mechanical"
    assert issue["merge_alert"] == "OK"

    pr = state["prs"]["10"]
    assert pr["status"] == "escalated"
    assert pr["escalation_reason"] == "test_reason"
    assert pr["number"] == 10
    assert pr["issue_number"] == 1
    assert pr["decision"] == "request_changes"
    # PRs do not carry reason_class; only the issue record does.
    assert "reason_class" not in pr
