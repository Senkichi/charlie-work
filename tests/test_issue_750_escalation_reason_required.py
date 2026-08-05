"""Regression tests for issue #750.

Issue #750: status=escalated must always carry escalation_reason, and the
only place in workflow.py that constructs a literal ``status: "escalated"``
entry is the shared ``_escalate_issue`` helper.
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

    A third form -- ``status = "escalated"`` bound to a local, then assigned as
    ``record["status"] = status`` -- is deliberately *not* flagged. It is
    control-flow dependent (the same local carries "dispatch_failed" on the
    other branch), so it cannot be judged syntactically without false
    positives.
    """
    source = SRC_WORKFLOW.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SRC_WORKFLOW))

    # Annotate each node with its parent so we can walk up to the enclosing
    # function/method scope.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)

    def _is_in_helper(node: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if current.name == "_escalate_issue":
                    return True
                # If the helper is ever made a method, also allow the same
                # method name inside a class.
                parent = getattr(current, "parent", None)
                if isinstance(parent, ast.ClassDef):
                    if current.name == "_escalate_issue":
                        return True
            current = getattr(current, "parent", None)
        return False

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "status"
                    and isinstance(value, ast.Constant)
                    and value.value == "escalated"
                    and not _is_in_helper(node)
                ):
                    violations.append(f"dict literal at line {getattr(node, 'lineno', '?')}")
        elif isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Constant) and node.value.value == "escalated"):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "status"
                    and not _is_in_helper(node)
                ):
                    violations.append(f"subscript assign at line {getattr(node, 'lineno', '?')}")

    assert not violations, (
        "Found status='escalated' write(s) outside _escalate_issue: " + ", ".join(violations)
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
