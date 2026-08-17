"""Tests for OrchestratorApp._maybe_probe_quota_recovery, carved out of test_charlie_work.py (#1284) -- named for the quota_probe domain the lane wires around (charlie_work.claude_code's run_quota_probe is the stubbed dependency the lane exercises, supplying the filename), not a literal 1:1 extraction of claude_code.py itself."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def _quota_probe_app(
    tmp_path: Path, *, interval_minutes: int = 15, enabled: bool = True
) -> OrchestratorApp:
    from charlie_work.config import QuotaProbeConfig

    config = OrchestratorConfig(
        quota_probe=QuotaProbeConfig(enabled=enabled, interval_minutes=interval_minutes)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def test_maybe_probe_quota_recovery_noop_when_nothing_throttled(tmp_path: Path) -> None:
    from charlie_work import workflow as workflow_module

    app = _quota_probe_app(tmp_path)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("run_quota_probe must not be called when nothing is throttled")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("quota_probe", {}).get("next_probe_at") is None


def test_maybe_probe_quota_recovery_disarms_stale_schedule_when_throttle_cleared(
    tmp_path: Path,
) -> None:
    """A cooldown can expire naturally (root throttled_until passes) between
    passes, before the flat-interval probe schedule fires. The next pass
    must disarm the now-stale schedule rather than probing needlessly."""
    from charlie_work.state import arm_quota_probe, save_state

    app = _quota_probe_app(tmp_path)
    state = load_state(app.paths.state_file)
    state = arm_quota_probe(state, "2026-08-01T00:00:00Z")
    save_state(app.paths.state_file, state)

    app._maybe_probe_quota_recovery()

    state = load_state(app.paths.state_file)
    assert state.get("quota_probe", {}).get("next_probe_at") is None


def test_maybe_probe_quota_recovery_arms_on_first_pass_without_probing(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import save_state, set_throttled_until

    app = _quota_probe_app(tmp_path, interval_minutes=15)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    save_state(app.paths.state_file, state)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("first pass after onset must arm, not probe")

    # frozen_now (issue #828) is injected so the schedule assertion below is
    # exact instead of racing wall-clock time under CI runner contention --
    # no downstream real-clock-dependent step follows in this test, so no
    # offset is needed (contrast test_loop_classifies_dead_sessions_...,
    # which offsets +1h because a later dispatch() reads real wall clock).
    frozen_now = datetime.now(UTC)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery(now=frozen_now)
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    next_probe_at = state["quota_probe"]["next_probe_at"]
    expected = (
        (frozen_now + timedelta(minutes=15))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert next_probe_at == expected


def test_maybe_probe_quota_recovery_waits_until_due(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import arm_quota_probe, save_state, set_throttled_until

    app = _quota_probe_app(tmp_path)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    not_due = (
        (datetime.now(UTC) + timedelta(minutes=10))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, not_due)
    save_state(app.paths.state_file, state)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("must not probe before the scheduled time")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state["quota_probe"]["next_probe_at"] == not_due


def test_maybe_probe_quota_recovery_green_clears_all_throttles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import (
        arm_quota_probe,
        save_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    app = _quota_probe_app(tmp_path)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(
        load_state(app.paths.state_file), future, reason="rate_limited", adapter_kind="claude-code"
    )
    state = set_reviewer_quota_exhausted(state, throttled_until=future, probe_after=future)
    state = {
        **state,
        "reviewer_quota": {**state["reviewer_quota"], "consecutive_probe_failures": 3},
    }
    due = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, due)
    save_state(app.paths.state_file, state)

    calls: list[dict] = []

    def _fake_probe(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(workflow_module, "run_quota_probe", _fake_probe)

    app._maybe_probe_quota_recovery()

    assert len(calls) == 1
    assert calls[0]["repo_root"] == tmp_path
    state = load_state(app.paths.state_file)
    assert state["throttled_until"] is None
    assert "throttled_until" not in state.get("reviewer_quota", {})
    assert state.get("quota_probe", {}).get("next_probe_at") is None
    assert any(e["kind"] == "quota_probe_succeeded" for e in state["events"])
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 0


def test_maybe_probe_quota_recovery_red_reschedules_flat_interval_and_keeps_throttle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import arm_quota_probe, save_state, set_throttled_until

    app = _quota_probe_app(tmp_path, interval_minutes=15)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    due = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, due)
    save_state(app.paths.state_file, state)

    monkeypatch.setattr(workflow_module, "run_quota_probe", lambda **_kwargs: False)

    # frozen_now (issue #828) injected for an exact schedule assertion; no
    # downstream real-clock dependency follows, so no offset is needed.
    frozen_now = datetime.now(UTC)
    app._maybe_probe_quota_recovery(now=frozen_now)

    state = load_state(app.paths.state_file)
    # Flat interval: rescheduled ~15 minutes out again, not a growing backoff.
    next_probe_at = state["quota_probe"]["next_probe_at"]
    expected = (
        (frozen_now + timedelta(minutes=15))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert next_probe_at == expected
    assert state["throttled_until"] == future
    assert any(e["kind"] == "quota_probe_failed" for e in state["events"])


def test_maybe_probe_quota_recovery_red_also_defers_reviewer_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A red flat probe must also bump reviewer_quota.probe_after so
    dispatch_reviews's probe_mode gate defers instead of independently
    launching a real reviewer session into the same still-closed window
    (issue #663)."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import (
        arm_quota_probe,
        is_reviewer_probe_ready,
        save_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    app = _quota_probe_app(tmp_path, interval_minutes=15)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    # Reviewer quota exhausted with probe_after in the past (ready to probe).
    state = set_reviewer_quota_exhausted(
        state, throttled_until=future, probe_after="2020-01-01T00:00:00Z"
    )
    due = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, due)
    save_state(app.paths.state_file, state)

    monkeypatch.setattr(workflow_module, "run_quota_probe", lambda **_kwargs: False)

    app._maybe_probe_quota_recovery()

    state = load_state(app.paths.state_file)
    next_probe_at = state["quota_probe"]["next_probe_at"]
    # probe_after must now be bumped to the flat probe's next attempt, so
    # dispatch_reviews's is_reviewer_probe_ready returns False.
    assert state["reviewer_quota"]["probe_after"] == next_probe_at
    assert is_reviewer_probe_ready(state) is False
    # consecutive_probe_failures is not touched by the flat probe.
    assert state["reviewer_quota"].get("consecutive_probe_failures", 0) == 0


def test_maybe_probe_quota_recovery_red_does_not_shorten_existing_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the reviewer quota's own exponential backoff already pushed
    probe_after further out than the flat probe's interval, a red flat
    probe must not shorten it (issue #663)."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import (
        arm_quota_probe,
        save_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    app = _quota_probe_app(tmp_path, interval_minutes=15)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    far_future = (
        (datetime.now(UTC) + timedelta(hours=4))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    state = set_reviewer_quota_exhausted(state, throttled_until=future, probe_after=far_future)
    due = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, due)
    save_state(app.paths.state_file, state)

    monkeypatch.setattr(workflow_module, "run_quota_probe", lambda **_kwargs: False)

    app._maybe_probe_quota_recovery()

    state = load_state(app.paths.state_file)
    # probe_after stays at the existing (further-out) backoff target.
    assert state["reviewer_quota"]["probe_after"] == far_future


def test_maybe_probe_quota_recovery_red_skips_reviewer_probe_when_only_root_throttled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A red flat probe must not write reviewer_quota.probe_after when the
    reviewer quota is not exhausted -- only the root throttle is active
    (issue #663)."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import arm_quota_probe, save_state, set_throttled_until

    app = _quota_probe_app(tmp_path, interval_minutes=15)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(
        load_state(app.paths.state_file), future, reason="rate_limited", adapter_kind="claude-code"
    )
    due = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_quota_probe(state, due)
    save_state(app.paths.state_file, state)

    monkeypatch.setattr(workflow_module, "run_quota_probe", lambda **_kwargs: False)

    app._maybe_probe_quota_recovery()

    state = load_state(app.paths.state_file)
    # No reviewer_quota.probe_after written -- reviewer quota was not exhausted.
    assert "probe_after" not in state.get("reviewer_quota", {})


def test_maybe_probe_quota_recovery_disabled_config_never_probes(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import save_state, set_throttled_until

    app = _quota_probe_app(tmp_path, enabled=False)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(load_state(app.paths.state_file), future, reason="rate_limited")
    save_state(app.paths.state_file, state)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("disabled quota_probe must never call run_quota_probe")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("quota_probe", {}).get("next_probe_at") is None


def test_maybe_probe_quota_recovery_never_arms_for_provider_auth_throttle(
    tmp_path: Path,
) -> None:
    """A dead key does not self-heal within minutes, and ``clear_quota_throttles``
    deliberately leaves a provider_auth-reasoned root throttle untouched even
    on a green probe. The probe must therefore never arm for one in the first
    place -- arming/probing a throttle that can never be cleared would just
    burn a Haiku session every ``interval_minutes`` for the whole cooldown
    window with no possible benefit."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import save_state, set_throttled_until

    app = _quota_probe_app(tmp_path)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(
        load_state(app.paths.state_file),
        future,
        reason="provider_auth",
        adapter_kind="claude-code",
    )
    save_state(app.paths.state_file, state)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("provider_auth throttle must never arm the probe")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("quota_probe", {}).get("next_probe_at") is None
    assert state["throttled_until"] == future
    assert state["throttle_reason"] == "provider_auth"


def test_maybe_probe_quota_recovery_never_arms_for_non_claude_code_adapter_throttle(
    tmp_path: Path,
) -> None:
    """A devin/api-adapter throttle is not cleared by a claude-code ambient
    probe (different tool/credential entirely -- see
    ``clear_quota_throttles``), so the probe must never arm for one."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import save_state, set_throttled_until

    app = _quota_probe_app(tmp_path)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = set_throttled_until(
        load_state(app.paths.state_file),
        future,
        reason="rate_limited",
        adapter_kind="devin",
    )
    save_state(app.paths.state_file, state)

    def _fail_if_called(**_kwargs: object) -> bool:
        raise AssertionError("devin-adapter throttle must never arm the probe")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "run_quota_probe", _fail_if_called)
    try:
        app._maybe_probe_quota_recovery()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("quota_probe", {}).get("next_probe_at") is None
    assert state["throttled_until"] == future
    assert state["throttle_adapter_kind"] == "devin"
