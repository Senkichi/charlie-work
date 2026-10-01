"""Side-effect order of the live Merge path, pinned to the pre-extraction order.

Two reads sit between effects that legacy ``merge_ready`` ordered deliberately:

* the cross-PR revert rework request is persisted BEFORE ``pr_checks`` /
  ``pr_diff`` (a refused read after the detection must not lose the route);
* the human-merge label read (``issue_view``) happens only AFTER the stall and
  infra exits (a stall pass costs no extra API call and a refused read cannot
  suppress the stall rework).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import _merge_path_facts as mf
import pytest
from _fakes_github import FakeGitHub, FakeGitHubWithMissingRequired
from _merge_path_characterization_harness import (
    _ISSUE,
    _REQUIRED,
    _cfg,
    _run,
)
from charlie_work import workflow as workflow_module
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.cross_pr_revert import CrossPrRevertResult, CrossPrRevertStatus
from charlie_work.merge_path import EffectResults, MergePlan, PlanKind, decide_accounting
from charlie_work.merge_path import apply_accounting
from charlie_work.merge_path.model import EventSpec
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.write_gate import WriteGate

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


class _ReadRefused(BaseException):
    """Stands in for ``PassDeadlineExceeded``: a BaseException out of a gh read."""


class _ChecksRefused(FakeGitHub):
    def pr_checks(self, number: int):  # type: ignore[override]
        raise _ReadRefused("pass deadline spent")


def test_revert_rework_is_persisted_before_pr_checks_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "detect_cross_pr_revert",
        lambda *_a, **_k: CrossPrRevertResult(
            CrossPrRevertStatus.REVERT_DETECTED, "reverts feature C"
        ),
    )
    config = _cfg()
    root = tmp_path / "live"

    with pytest.raises(_ReadRefused):
        _run(root, config, _ChecksRefused, dry_run=False)

    state = load_state(runtime_paths(root, config.runtime.state_dir).state_file)
    assert state["issues"][str(_ISSUE)]["status"] == "rework_requested"


class _StalledCounting(FakeGitHubWithMissingRequired):
    """No required check ever started, an ancient head, and a counted ``issue_view``."""

    def __init__(self) -> None:
        super().__init__()
        self.prs[0]["updatedAt"] = "2020-01-01T00:00:00Z"
        self.prs[0]["mergeStateStatus"] = "CLEAN"
        self.prs[0]["mergeable"] = "MERGEABLE"
        self.issues[0]["labels"] = [{"name": "automated-ready"}]
        self.issue_view_calls: list[int] = []

    def issue_view(self, number: int):
        self.issue_view_calls.append(number)
        return super().issue_view(number)


def _stall_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=_REQUIRED,
            require_approved_review=True,
            update_branch_strategy="off",
            require_current_base=False,
            failed_attempt_alarm=3,
            readiness_no_ci_minutes=15,
        ),
        dispatch=DispatchConfig(human_merge_labels=("needs-design",)),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )


def test_stall_exit_does_not_read_the_human_merge_labels(tmp_path: Path) -> None:
    def clear_setup_calls(_app: Any, _paths: Any, gh: _StalledCounting) -> None:
        gh.issue_view_calls.clear()

    out = _run(
        tmp_path / "live", _stall_config(), _StalledCounting, dry_run=False, pre=clear_setup_calls
    )

    assert out.data["readiness_no_ci_stall"] is True
    assert out.gh.issue_view_calls == []


def test_human_merge_labels_are_still_read_when_the_pass_proceeds(tmp_path: Path) -> None:
    """Positive control for the test above: the read exists and it is the one counted."""

    class _Counting(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issue_view_calls: list[int] = []

        def issue_view(self, number: int):
            self.issue_view_calls.append(number)
            return super().issue_view(number)

    def clear_setup_calls(_app: Any, _paths: Any, gh: _Counting) -> None:
        gh.issue_view_calls.clear()

    config = _cfg(dispatch=DispatchConfig(human_merge_labels=("needs-design",)))
    out = _run(tmp_path / "live", config, _Counting, dry_run=False, pre=clear_setup_calls)

    assert out.gh.issue_view_calls.count(_ISSUE) >= 1


# ---------------------------------------------------------------------------
# Accounting events keep literal kinds (event-kind guards can see them)
# ---------------------------------------------------------------------------


def _module_tree(name: str) -> ast.Module:
    import importlib

    module = importlib.import_module(name)
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _decided_event_kinds() -> set[str]:
    return {
        node.args[0].value
        for node in ast.walk(_module_tree("charlie_work.merge_path.decide"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "EventSpec"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def _recorded_event_kinds() -> set[str]:
    record = next(
        node
        for node in ast.walk(_module_tree("charlie_work.merge_path.apply_accounting"))
        if isinstance(node, ast.FunctionDef) and node.name == "_record"
    )
    return {
        call.args[1].value
        for call in ast.walk(record)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "record_event"
        and len(call.args) > 1
        and isinstance(call.args[1], ast.Constant)
    }


def test_every_decided_accounting_event_has_a_literal_record_call() -> None:
    decided = _decided_event_kinds()
    assert decided == {"merge_ready", "merge_succeeded", "merge_failed_attempt_alarm"}
    assert _recorded_event_kinds() == decided


def test_record_refuses_a_kind_it_has_no_literal_call_for(tmp_path: Path) -> None:
    gate = WriteGate(dry_run=False, state_path=tmp_path / "state.json", repo="r")
    with pytest.raises(ValueError, match="merge_handoff_reverted"):
        apply_accounting._record(gate, {"events": []}, EventSpec("merge_handoff_reverted", {}))


# ---------------------------------------------------------------------------
# Persisted counter coercion (legacy ``int(...)``)
# ---------------------------------------------------------------------------


def test_a_string_persisted_failed_attempt_counter_is_coerced_like_legacy() -> None:
    plan = MergePlan(kind=PlanKind.NONE, readiness=mf.readiness(gate=mf.gate(summary_ready=False)))
    facts = mf.accounting_facts(
        locked=mf.persisted(failed_attempts="2"), config=mf.cfg(failed_attempt_alarm=5)
    )

    acc = decide_accounting(plan, EffectResults(), facts)

    assert acc.failed_attempts == 3
