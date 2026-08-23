"""Regression tests for issue #1314: operator-queue follow-ups deferred from #1266.

Covers all four items:

1. CLI subcommand (``charlie operator-queue``) — ``OrchestratorApp.operator_queue``.
2. Sweep cadence knob — ``DeescalationConfig.operator_queue_review_interval_minutes``
   + ``is_operator_queue_review_due`` / ``arm_operator_queue_review`` state helpers.
3. ``operator_queue_depth`` event + threshold — the gauge event emitted in
   ``_loop_impl`` when depth exceeds the configured threshold, registered in
   ``_LEVEL_BY_KIND`` and ``EXPECTED_OPERATIONAL_KINDS``.
4. ``escalation_parked_labels`` derived from ``ESCALATION_REASON_CLASSES``
   instead of a hand-picked ``"mechanical"`` literal in ``reconcile.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub
from _reconcile_fixtures import _issue
from charlie_work.config import DeescalationConfig, OrchestratorConfig, PostMortemConfig
from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS
from charlie_work.instrumentation import _LEVEL_BY_KIND, query_events
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import detect_drift
from charlie_work.state import (
    arm_operator_queue_review,
    empty_state,
    is_operator_queue_review_due,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp, operator_queue_depth


# ---------------------------------------------------------------------------
# Shared fixture: a minimal OrchestratorApp pointed at tmp_path.
# ---------------------------------------------------------------------------


def _app(
    tmp_path: Path,
    *,
    deescalation: DeescalationConfig | None = None,
) -> OrchestratorApp:
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        deescalation=deescalation or DeescalationConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _seed_operator_queue_issue(
    app: OrchestratorApp,
    issue_number: int,
    *,
    terminal_since: str | None = None,
    reason_class: str = "mechanical",
    status: str = "escalated",
) -> None:
    """Plant an issue in state.json that matches the operator-queue criteria."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        entry: dict[str, Any] = {
            "number": issue_number,
            "status": status,
            "reason_class": reason_class,
            "escalation_reason": "test escalation",
        }
        if terminal_since is not None:
            entry["terminal_since"] = terminal_since
        state.setdefault("issues", {})[str(issue_number)] = entry
        save_state(app.paths.state_file, state)


# ---------------------------------------------------------------------------
# Item 4: escalation_parked_labels derived from ESCALATION_REASON_CLASSES
# ---------------------------------------------------------------------------


def test_escalation_parked_labels_covers_all_reason_classes(tmp_path: Path) -> None:
    """Issue #1314 item 4: ``detect_drift``'s ``escalation_parked_labels`` set
    must be derived from ``ESCALATION_REASON_CLASSES``, not a hand-picked
    ``"mechanical"`` literal. The set must include the label for every
    reason_class in the enum (today: ``human_needed`` for ``judgment``,
    ``operator_queue`` for ``mechanical``), and must automatically include
    any future reason_class's label without a code change here.

    This test verifies the behavior end-to-end: an issue carrying
    ``operator_queue`` is detected as ``terminal_state_stale`` (the consumer
    of ``escalation_parked_labels``), proving the label is in the set.
    """
    config = OrchestratorConfig()
    gh = ReconcileFakeGitHub(
        prs=[],
        issues=[_issue(894, [config.labels.operator_queue])],
    )
    state = empty_state()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    state["issues"]["894"] = {
        "number": 894,
        "status": "escalated",
        "reason_class": "mechanical",
        "terminal_since": "2026-01-05T00:00:00Z",  # 5 days before `now`
    }

    drift = detect_drift(gh, state, config, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 894
    assert config.labels.operator_queue in matches[0].detail


def test_escalation_parked_labels_includes_human_needed(tmp_path: Path) -> None:
    """The judgment-escalation label (``human_needed``) must still be in the
    parked set — the enum-derived derivation must not accidentally drop it
    when replacing the hand-picked literal."""
    config = OrchestratorConfig()
    gh = ReconcileFakeGitHub(
        prs=[],
        issues=[_issue(895, [config.labels.human_needed])],
    )
    state = empty_state()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    state["issues"]["895"] = {
        "number": 895,
        "status": "escalated",
        "reason_class": "judgment",
        "terminal_since": "2026-01-05T00:00:00Z",
    }

    drift = detect_drift(gh, state, config, now=now)

    matches = [item for item in drift if item.kind == "terminal_state_stale"]
    assert len(matches) == 1
    assert matches[0].issue_number == 895
    assert config.labels.human_needed in matches[0].detail


# ---------------------------------------------------------------------------
# Item 3: operator_queue_depth event kind registration
# ---------------------------------------------------------------------------


def test_operator_queue_depth_registered_as_warning() -> None:
    """``operator_queue_depth`` must be registered in ``_LEVEL_BY_KIND`` at
    ``"warning"`` — it is a threshold alert, not a routine info gauge, and
    ``EXPECTED_OPERATIONAL_KINDS`` membership requires warning level."""
    assert "operator_queue_depth" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["operator_queue_depth"] == "warning"


def test_operator_queue_depth_in_expected_operational_kinds() -> None:
    """``operator_queue_depth`` must be in ``EXPECTED_OPERATIONAL_KINDS`` so
    ``heartbeat_check.py`` buckets it into a summarized count (the consumer
    surface the signal-without-a-consumer rule requires to land in the same
    PR as the signal)."""
    assert "operator_queue_depth" in EXPECTED_OPERATIONAL_KINDS


# ---------------------------------------------------------------------------
# Item 3: operator_queue_depth gauge function
# ---------------------------------------------------------------------------


def test_operator_queue_depth_counts_mechanical_escalated(tmp_path: Path) -> None:
    """``operator_queue_depth`` counts only issues with ``status ==
    "escalated"`` and ``reason_class == "mechanical"`` — the in-state mirror
    of the ``agent:operator-queue`` label."""
    app = _app(tmp_path)
    _seed_operator_queue_issue(app, 101, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 102, terminal_since="2026-01-02T00:00:00Z")
    # A judgment escalation must NOT be counted.
    _seed_operator_queue_issue(app, 103, reason_class="judgment")
    # A blocked issue must NOT be counted (blocked is always judgment).
    _seed_operator_queue_issue(app, 104, status="blocked")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)

    depth = operator_queue_depth(state)
    assert depth == {101, 102}


def test_operator_queue_depth_empty_state(tmp_path: Path) -> None:
    """An empty state must produce an empty depth set, not an error."""
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert operator_queue_depth(state) == set()


# ---------------------------------------------------------------------------
# Item 3: _maybe_emit_operator_queue_depth emission
# ---------------------------------------------------------------------------


def test_depth_gauge_emits_when_threshold_exceeded(tmp_path: Path) -> None:
    """When depth exceeds the configured threshold, the gauge event is
    emitted to events.db with the depth, threshold, and issue numbers."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=2,
        ),
    )
    _seed_operator_queue_issue(app, 201, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 202, terminal_since="2026-01-02T00:00:00Z")
    _seed_operator_queue_issue(app, 203, terminal_since="2026-01-03T00:00:00Z")

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["depth"] == 3
    assert payload["threshold"] == 2
    assert sorted(payload["issue_numbers"]) == [201, 202, 203]


def test_depth_gauge_no_emit_when_below_threshold(tmp_path: Path) -> None:
    """When depth is at or below the threshold, no event is emitted — the
    gauge is silent when the queue is manageable."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=5,
        ),
    )
    _seed_operator_queue_issue(app, 301, terminal_since="2026-01-01T00:00:00Z")

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert events == []


def test_depth_gauge_no_emit_when_depth_equals_threshold(tmp_path: Path) -> None:
    """When depth exactly equals the threshold, no event is emitted — the
    alert fires only when depth *exceeds* the threshold (depth > threshold),
    not when it merely meets it. This is the boundary that distinguishes
    ``>`` from ``>=`` and catches a ``depth < threshold`` mutation."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=2,
        ),
    )
    _seed_operator_queue_issue(app, 311, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 312, terminal_since="2026-01-02T00:00:00Z")

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert events == []


def test_depth_gauge_no_emit_when_threshold_disabled(tmp_path: Path) -> None:
    """Threshold 0 disables the alert entirely — no event regardless of depth."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=0,
        ),
    )
    _seed_operator_queue_issue(app, 401, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 402, terminal_since="2026-01-02T00:00:00Z")

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert events == []


def test_depth_gauge_respects_review_cadence(tmp_path: Path) -> None:
    """When ``operator_queue_review_interval_minutes > 0``, the gauge is
    gated by the ``next_operator_queue_review_at`` timestamp. A future
    timestamp means the gauge is not due and no event is emitted."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=1,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(app, 501, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 502, terminal_since="2026-01-02T00:00:00Z")

    # Arm the next review far in the future.
    future = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = arm_operator_queue_review(state, future)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert events == []


def test_depth_gauge_emits_when_review_cadence_due(tmp_path: Path) -> None:
    """When the review cadence is due (past timestamp), the gauge fires
    normally."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=1,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(app, 601, terminal_since="2026-01-01T00:00:00Z")
    _seed_operator_queue_issue(app, 602, terminal_since="2026-01-02T00:00:00Z")

    # Arm the next review in the past (due now).
    past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = arm_operator_queue_review(state, past)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_depth()

    events = query_events(app.paths.state_file, kind="operator_queue_depth")
    assert len(events) == 1
    assert events[0]["payload"]["depth"] == 2


# ---------------------------------------------------------------------------
# Item 2: state helpers for the review cadence
# ---------------------------------------------------------------------------


def test_is_operator_queue_review_due_no_timestamp_is_due() -> None:
    """A fresh state with no ``next_operator_queue_review_at`` is due
    immediately — same semantics as ``is_deescalation_due``."""
    state = empty_state()
    assert is_operator_queue_review_due(state) is True


def test_is_operator_queue_review_due_future_timestamp_not_due() -> None:
    """A future timestamp means the gauge is not due."""
    future = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    state = arm_operator_queue_review(empty_state(), future)
    assert is_operator_queue_review_due(state) is False


def test_is_operator_queue_review_due_past_timestamp_is_due() -> None:
    """A past timestamp means the gauge is due."""
    past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    state = arm_operator_queue_review(empty_state(), past)
    assert is_operator_queue_review_due(state) is True


def test_is_operator_queue_review_due_malformed_timestamp_is_due() -> None:
    """A malformed timestamp is treated as due, not as a wedge."""
    state = arm_operator_queue_review(empty_state(), "not-a-timestamp")
    assert is_operator_queue_review_due(state) is True


def test_arm_operator_queue_review_does_not_mutate() -> None:
    """``arm_operator_queue_review`` returns a new state dict; the original
    is not mutated."""
    original = empty_state()
    future = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    result = arm_operator_queue_review(original, future)
    assert "next_operator_queue_review_at" not in original.get("deescalation_pass", {})
    assert result["deescalation_pass"]["next_operator_queue_review_at"] == future


# ---------------------------------------------------------------------------
# Item 2: config parsing
# ---------------------------------------------------------------------------


def test_deescalation_config_defaults() -> None:
    """The new fields have the right defaults: review interval 0 (every pass),
    depth threshold 5."""
    cfg = DeescalationConfig()
    assert cfg.operator_queue_review_interval_minutes == 0
    assert cfg.operator_queue_depth_threshold == 5


def test_deescalation_config_parsed_from_yaml(tmp_path: Path) -> None:
    """The deescalation section is now parsed from YAML (previously always
    defaulted). The new fields are configurable."""
    from charlie_work.config import build_config_from_data

    data = {
        "deescalation": {
            "enabled": True,
            "interval_minutes": 15,
            "operator_queue_review_interval_minutes": 10,
            "operator_queue_depth_threshold": 3,
        }
    }
    config = build_config_from_data(data)
    assert config.deescalation.enabled is True
    assert config.deescalation.interval_minutes == 15
    assert config.deescalation.operator_queue_review_interval_minutes == 10
    assert config.deescalation.operator_queue_depth_threshold == 3


def test_deescalation_config_rejects_negative_review_interval(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_review_interval_minutes"):
        build_config_from_data({"deescalation": {"operator_queue_review_interval_minutes": -1}})


def test_deescalation_config_rejects_negative_depth_threshold(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_depth_threshold"):
        build_config_from_data({"deescalation": {"operator_queue_depth_threshold": -1}})


def test_deescalation_config_rejects_non_int_threshold(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_depth_threshold"):
        build_config_from_data({"deescalation": {"operator_queue_depth_threshold": "five"}})


# ---------------------------------------------------------------------------
# Item 1: CLI subcommand — OrchestratorApp.operator_queue
# ---------------------------------------------------------------------------


def _app_with_issues(
    tmp_path: Path,
    issues: list[dict[str, Any]],
) -> OrchestratorApp:
    """Build an app with a FakeGitHub whose issue_list returns ``issues``."""
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    gh.issues = issues
    return OrchestratorApp(tmp_path, paths, config, gh)


def test_operator_queue_command_lists_issues(tmp_path: Path) -> None:
    """``operator_queue()`` returns a sorted queue of issues parked on the
    operator queue, joining GitHub label data with state.json provenance."""
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    now = datetime.now(UTC)
    gh = FakeGitHub()
    gh.issues = [
        _issue(701, [config.labels.operator_queue]),
        _issue(702, [config.labels.operator_queue]),
    ]
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    _seed_operator_queue_issue(
        app, 701, terminal_since=(now - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    )
    _seed_operator_queue_issue(
        app, 702, terminal_since=(now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    )

    result = app.operator_queue()

    assert result.ok
    assert result.data["depth"] == 2
    queue = result.data["queue"]
    assert len(queue) == 2
    # Sorted by terminal_since ascending (oldest first).
    assert queue[0]["number"] == 701
    assert queue[1]["number"] == 702
    # Each entry carries the required provenance fields.
    for entry in queue:
        assert "reason_class" in entry
        assert "escalation_reason" in entry
        assert "terminal_since" in entry
        assert "age_days" in entry
        assert "last_escalation_event" in entry
        assert "labels" in entry
        assert "title" in entry
        assert "url" in entry


def test_operator_queue_command_includes_state_only_issues(tmp_path: Path) -> None:
    """Issues in state but missing from the GitHub label query (a label
    transition that has not yet propagated) are included from state alone."""
    app = _app_with_issues(tmp_path, issues=[])

    _seed_operator_queue_issue(app, 801, terminal_since="2026-01-01T00:00:00Z")

    result = app.operator_queue()

    assert result.ok
    assert result.data["depth"] == 1
    assert result.data["queue"][0]["number"] == 801
    assert result.data["queue"][0]["reason_class"] == "mechanical"


def test_operator_queue_command_includes_label_only_issues(tmp_path: Path) -> None:
    """Issues on the label but missing from state (a manual label add) are
    included with ``reason_class: None`` so they are visible."""
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    app = _app_with_issues(tmp_path, issues=[_issue(901, [config.labels.operator_queue])])

    result = app.operator_queue()

    assert result.ok
    assert result.data["depth"] == 1
    entry = result.data["queue"][0]
    assert entry["number"] == 901
    assert entry["reason_class"] is None
    assert entry["age_days"] is None


def test_operator_queue_command_empty(tmp_path: Path) -> None:
    """An empty queue returns depth 0 and an empty list."""
    app = _app_with_issues(tmp_path, issues=[])

    result = app.operator_queue()

    assert result.ok
    assert result.data["depth"] == 0
    assert result.data["queue"] == []


def test_operator_queue_command_excludes_judgment_escalations(tmp_path: Path) -> None:
    """A judgment escalation in state (``reason_class == "judgment"``) must
    NOT appear in the operator queue — it parks on ``human_needed``, not
    ``operator_queue``."""
    app = _app_with_issues(tmp_path, issues=[])

    _seed_operator_queue_issue(app, 1001, reason_class="judgment")

    result = app.operator_queue()

    assert result.ok
    assert result.data["depth"] == 0
