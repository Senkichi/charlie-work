"""Tests for ``charlie_work.write_gate`` (issue #1264, W6 PR1).

Covers the design doc's §8 test plan items 1-5: frozen/immutability,
construction/equality, dry_run=True zero-call/nothing-happened behavior for
each of the 5 gated methods, dry_run=False exact passthrough (including
auto-bound ``state_path``/``repo``), ``require_write_gate``'s raise-on-
missing-gate contract, and that ``OrchestratorApp.__init__`` constructs
``self.write_gate`` with fields matching its own ``dry_run``/``paths``/
``repo_root``.

All external effects (state.py/instrumentation.py/labels.py primitives) are
monkeypatched at the module level ``write_gate.py`` looks them up at --
no real GitHub, no real subprocess, no real file writes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from charlie_work import write_gate
from charlie_work.config import OrchestratorConfig
from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate, require_write_gate

STATE_PATH = Path("/fake/state/state.json")
REPO = "charlie-work"


def _gate(*, dry_run: bool) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=STATE_PATH, repo=REPO)


# ---------------------------------------------------------------------------
# 1. Frozen/immutability -- repo idiom per tests/test_cli.py's CommandContext
#    test (pytest.raises(FrozenInstanceError)).
# ---------------------------------------------------------------------------


def test_write_gate_is_frozen() -> None:
    gate = _gate(dry_run=False)
    assert dataclasses.is_dataclass(gate)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.dry_run = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Construction/equality -- default frozen-dataclass behavior.
# ---------------------------------------------------------------------------


def test_write_gate_construction_and_equality() -> None:
    a = WriteGate(dry_run=True, state_path=STATE_PATH, repo=REPO)
    b = WriteGate(dry_run=True, state_path=STATE_PATH, repo=REPO)
    c = WriteGate(dry_run=False, state_path=STATE_PATH, repo=REPO)
    assert a == b
    assert a != c
    assert a.dry_run is True
    assert a.state_path == STATE_PATH
    assert a.repo == REPO


# ---------------------------------------------------------------------------
# 3. dry_run=True -- for each of the 5 methods, the underlying primitive is
#    NEVER called (genuinely zero calls, not called-then-discarded), and the
#    return value is the natural "nothing happened" value.
# ---------------------------------------------------------------------------


def test_save_state_dry_run_true_never_calls_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(write_gate, "save_state", lambda *a, **k: calls.append((a, k)))
    gate = _gate(dry_run=True)
    data = {"issues": {}}
    result = gate.save_state(data)
    assert calls == []
    assert result is data


def test_append_event_dry_run_true_never_calls_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(write_gate, "append_event", lambda *a, **k: calls.append((a, k)))
    gate = _gate(dry_run=True)
    data = {"events": []}
    result = gate.append_event(data, "spec_review", {"ok": True})
    assert calls == []
    assert result is data


def test_record_event_dry_run_true_never_calls_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(write_gate, "append_event", lambda *a, **k: calls.append((a, k)))
    gate = _gate(dry_run=True)
    state = {"events": []}
    result = gate.record_event(state, "spec_review", {"ok": True})
    assert calls == []
    assert result is state


def test_log_event_dry_run_true_never_calls_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(write_gate, "log_event", lambda *a, **k: calls.append((a, k)))
    gate = _gate(dry_run=True)
    result = gate.log_event("dispatch_backpressure", {"open_pr_count": 3})
    assert calls == []
    assert result is None


def test_transition_dry_run_true_never_calls_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(write_gate, "transition", lambda *a, **k: calls.append((a, k)))
    gate = _gate(dry_run=True)
    result = gate.transition(object(), object(), 123, "escalated")
    assert calls == []
    assert result == TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])


# ---------------------------------------------------------------------------
# 4. dry_run=False -- pure passthrough, exact args/kwargs, including the
#    auto-bound state_path/repo the real _record_event applies today.
# ---------------------------------------------------------------------------


def test_save_state_dry_run_false_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_save_state(path, data):
        calls.append((path, data))
        return {**data, "generated_at": "now"}

    monkeypatch.setattr(write_gate, "save_state", fake_save_state)
    gate = _gate(dry_run=False)
    data = {"issues": {}}
    result = gate.save_state(data)
    assert calls == [(STATE_PATH, data)]
    assert result == {"issues": {}, "generated_at": "now"}


def test_append_event_dry_run_false_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_append_event(
        data, kind, payload, max_size=None, *, state_path=None, repo=None, level=None
    ):
        calls.append(
            {
                "data": data,
                "kind": kind,
                "payload": payload,
                "max_size": max_size,
                "state_path": state_path,
                "repo": repo,
                "level": level,
            }
        )
        return {**data, "events": [*data.get("events", []), {"kind": kind}]}

    monkeypatch.setattr(write_gate, "append_event", fake_append_event)
    gate = _gate(dry_run=False)
    data = {"events": []}
    result = gate.append_event(data, "spec_review", {"ok": True}, level="warning")
    assert calls == [
        {
            "data": data,
            "kind": "spec_review",
            "payload": {"ok": True},
            "max_size": None,
            "state_path": STATE_PATH,
            "repo": REPO,
            "level": "warning",
        }
    ]
    assert result["events"] == [{"kind": "spec_review"}]


def test_record_event_dry_run_false_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_append_event(state, kind, payload, *, state_path=None, repo=None, level=None):
        calls.append(
            {
                "state": state,
                "kind": kind,
                "payload": payload,
                "state_path": state_path,
                "repo": repo,
                "level": level,
            }
        )
        return {**state, "events": [*state.get("events", []), {"kind": kind}]}

    monkeypatch.setattr(write_gate, "append_event", fake_append_event)
    gate = _gate(dry_run=False)
    state = {"events": []}
    result = gate.record_event(state, "spec_review", {"ok": True})
    assert calls == [
        {
            "state": state,
            "kind": "spec_review",
            "payload": {"ok": True},
            "state_path": STATE_PATH,
            "repo": REPO,
            "level": None,
        }
    ]
    assert result["events"] == [{"kind": "spec_review"}]


def test_log_event_dry_run_false_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_log_event(state_path, kind, payload, *, repo=None, correlation_id=None, level=None):
        calls.append(
            {
                "state_path": state_path,
                "kind": kind,
                "payload": payload,
                "repo": repo,
                "correlation_id": correlation_id,
                "level": level,
            }
        )
        return None

    monkeypatch.setattr(write_gate, "log_event", fake_log_event)
    gate = _gate(dry_run=False)
    result = gate.log_event("dispatch_backpressure", {"open_pr_count": 3}, correlation_id="abc")
    assert calls == [
        {
            "state_path": STATE_PATH,
            "kind": "dispatch_backpressure",
            "payload": {"open_pr_count": 3},
            "repo": REPO,
            "correlation_id": "abc",
            "level": None,
        }
    ]
    assert result is None


def test_transition_dry_run_false_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    expected = TransitionResult(TransitionOutcome.APPLIED, [], [])

    def fake_transition(gh, labels, issue_number, event):
        calls.append((gh, labels, issue_number, event))
        return expected

    monkeypatch.setattr(write_gate, "transition", fake_transition)
    gate = _gate(dry_run=False)
    gh_sentinel = object()
    labels_sentinel = object()
    result = gate.transition(gh_sentinel, labels_sentinel, 123, "escalated")
    assert calls == [(gh_sentinel, labels_sentinel, 123, "escalated")]
    assert result is expected


# ---------------------------------------------------------------------------
# 5. require_write_gate -- raises on None and on wrong type, returns a real
#    gate unchanged.
# ---------------------------------------------------------------------------


def test_require_write_gate_raises_on_none() -> None:
    with pytest.raises(TypeError, match="write_gate is required"):
        require_write_gate(None)


def test_require_write_gate_raises_on_wrong_type() -> None:
    with pytest.raises(TypeError, match="write_gate is required"):
        require_write_gate(False)  # e.g. a caller accidentally passes dry_run itself


def test_require_write_gate_returns_real_gate_unchanged() -> None:
    gate = _gate(dry_run=True)
    assert require_write_gate(gate) is gate


# ---------------------------------------------------------------------------
# OrchestratorApp.__init__ wiring -- reuses the lightweight construction
# pattern from tests/_helpers.py's _cross_family_app (a real git repo is
# not required: OrchestratorApp only touches git when self.gh is the real
# GitHub class, and FakeGitHub is not that).
# ---------------------------------------------------------------------------


def test_orchestrator_app_constructs_write_gate(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)
    assert isinstance(app.write_gate, WriteGate)
    assert app.write_gate.dry_run == app.dry_run
    assert app.write_gate.state_path == app.paths.state_file
    assert app.write_gate.repo == app.repo_root.name


def test_orchestrator_app_write_gate_dry_run_false(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=False)
    assert app.write_gate.dry_run is False
    assert app.write_gate.dry_run == app.dry_run
