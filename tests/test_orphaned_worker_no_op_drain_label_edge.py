"""Issue #2783: the label edge of the no-op drain's provider-throttle re-arm.

``test_orphaned_worker_no_op_drain.py`` drives the whole sweep and pins the state,
refund and event outcomes of the re-arm. What it cannot reach cheaply is the
``rework_requested`` edge failing: the state write has already landed by then, so a
non-applied edge must be recorded as ``label_error`` instead of vanishing. These tests
call the drain's public entry point with a collected route.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg

from charlie_work import orphaned_worker_no_op_drain as drain_module
from charlie_work.config import OrchestratorConfig
from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.orphaned_worker_no_op_drain import NoOpReworkRoute, drain_no_op_rework_routes
from charlie_work.state import load_state, save_state
from charlie_work.write_gate import WriteGate


def _iso(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def _drain_throttle_death(tmp_path: Path) -> tuple[dict[str, Any], FakeGitHub, OrchestratorConfig]:
    """Drain one no-op route for a dispatched issue whose dead epoch was a quota throttle."""
    state_file = tmp_path / "state.json"
    state = load_state(state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "dispatched_at": _iso(-timedelta(hours=1)),
        "redispatch_at": [_iso(-timedelta(minutes=59))],
        "dead_worker_failure_kind": "quota_exhausted",
    }
    save_state(state_file, state)
    gh, config = FakeGitHub(), OrchestratorConfig()
    drain_no_op_rework_routes(
        [NoOpReworkRoute(207, 100, "def456", "dead_worker_with_head_change", "agent/issue-207")],
        gh=gh,
        config=config,
        state_file=state_file,
        write_gate=_wg(state_file),
        sessions_dir=tmp_path,
        repo_root=tmp_path,
        worktrees_dir=None,
        review_callback=None,
        record_review_callback=None,
        enrich_checks_callback=None,
    )
    return load_state(state_file)["issues"]["207"], gh, config


def test_applied_edge_leaves_no_label_error(tmp_path: Path) -> None:
    """Positive control: the identical drain with the real edge applies it cleanly."""
    entry, gh, config = _drain_throttle_death(tmp_path)

    assert entry["status"] == "rework_requested"
    assert entry["redispatch_at"] == []  # the dead session's own stamp was refunded
    assert (207, config.labels.needs_rework) in gh.labels_added
    assert "label_error" not in entry


@pytest.mark.parametrize(
    "outcome", [TransitionOutcome.PARTIAL_FAILURE, TransitionOutcome.NOTHING_CHANGED]
)
def test_non_applied_edge_is_recorded_as_label_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: TransitionOutcome
) -> None:
    """The edge returning a non-applied result must leave a ``label_error`` behind."""
    failed = [(207, "agent:needs-rework")]
    monkeypatch.setattr(
        WriteGate, "transition", lambda self, *a, **k: TransitionResult(outcome, failed, [])
    )
    entry, _gh, _config = _drain_throttle_death(tmp_path)

    assert entry["status"] == "rework_requested"  # the state write precedes the edge
    assert entry["label_error"]["edge"] == "rework_requested"
    assert entry["label_error"]["outcome"] == outcome.value
    assert entry["label_error"]["add_failures"] == [[207, "agent:needs-rework"]]


def test_non_applied_edge_goes_through_the_shared_label_error_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``label_error`` write is ``escalation._record_issue_label_error``, not a local copy.

    The helper is the single point of enforcement for persisting a failed edge. The spy
    replaces it, so ``label_error`` is absent from the entry unless some *other* code
    path (a re-introduced inline block) wrote it -- which is what this asserts against.
    """
    failed = TransitionResult(TransitionOutcome.PARTIAL_FAILURE, [(207, "agent:needs-rework")], [])
    monkeypatch.setattr(WriteGate, "transition", lambda self, *a, **k: failed)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(drain_module, "_record_issue_label_error", lambda *a: calls.append(a))

    entry, _gh, _config = _drain_throttle_death(tmp_path)

    assert len(calls) == 1
    state_file, write_gate, issue_number, edge, result = calls[0]
    assert Path(state_file) == tmp_path / "state.json"
    assert isinstance(write_gate, WriteGate)
    assert (issue_number, edge, result) == (207, "rework_requested", failed)
    assert entry["status"] == "rework_requested"
    assert "label_error" not in entry  # only the (stubbed) helper may write it
