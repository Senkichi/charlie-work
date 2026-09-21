"""Regression tests for issue #1314's operator-queue follow-ups, and issue
#1768's edge-triggered rewrite of item 3.

Covers all four #1314 items plus #1768:

1. CLI subcommand (``charlie operator-queue``) — ``OrchestratorApp.operator_queue``.
2. Sweep cadence knob — ``DeescalationConfig.operator_queue_review_interval_minutes``
   + ``is_operator_queue_review_due`` / ``arm_operator_queue_review`` state helpers.
3. ``operator_queue_impact`` event + threshold (issue #1768 rewrite of the
   original #1314 item 3 ``operator_queue_depth`` gauge): edge-triggered on
   root-set change / impact-threshold crossing / age-bucket crossing,
   measuring the transitive count of automated-ready open issues blocked
   behind the sink roots rather than a raw root tally, and delivered
   through ``notify.AttentionDigest`` in addition to ``events.db``.
4. ``escalation_parked_labels`` derived from ``ESCALATION_REASON_CLASSES``
   instead of a hand-picked ``"mechanical"`` literal in ``reconcile.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub
from _reconcile_fixtures import _issue
from charlie_work.config import (
    DeescalationConfig,
    NotifyConfig,
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS
from charlie_work.github import GitHubError
from charlie_work.instrumentation import _LEVEL_BY_KIND, query_events
from charlie_work.notify import _DESKTOP_SEVERITIES
from charlie_work.operator_queue_impact import (
    OperatorQueueImpact,
    age_bucket_label,
    compute_operator_queue_impact,
    operator_queue_impact_signature,
    should_fire_operator_queue_impact,
)
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import detect_drift
from charlie_work.state import (
    arm_operator_queue_review,
    empty_state,
    is_operator_queue_review_due,
    load_state,
    operator_queue_impact_baseline,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp, is_operator_queue_issue, operator_queue_depth

# Imported after ``charlie_work.workflow`` deliberately: ``state_maintenance``
# does ``import charlie_work.workflow as _wf`` at module level, so importing
# it directly *before* ``charlie_work.workflow`` has been fully imported
# triggers Python's circular-import partial-module hazard --
# ``workflow``'s own module-level ``discover_delegate_modules`` call would
# then reimport this already-in-progress module and see only the portion
# defined above the ``import charlie_work.workflow as _wf`` line, silently
# dropping every delegate defined below it (including
# ``_maybe_emit_operator_queue_impact``) from ``OrchestratorApp``. Importing
# ``charlie_work.workflow`` first (immediately above) guarantees delegate
# installation has already completed by the time this line runs.
from charlie_work.orchestration.state_maintenance import _BLOCKED_READY_ISSUE_NUMBERS_LIMIT  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures: a minimal OrchestratorApp pointed at tmp_path, and a
# small blocker-graph builder for the impact-computation fixtures.
# ---------------------------------------------------------------------------


def _app(
    tmp_path: Path,
    *,
    deescalation: DeescalationConfig | None = None,
    notify: NotifyConfig | None = None,
    dry_run: bool = False,
    gh: Any = None,
) -> OrchestratorApp:
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        deescalation=deescalation or DeescalationConfig(),
        notify=notify or NotifyConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = gh if gh is not None else FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)


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


def _blocked_issue(
    number: int, labels: list[str], *, blocked_by: int | None = None, state: str = "OPEN"
) -> dict[str, Any]:
    """An ``_issue()`` fixture whose body declares a blocker, for the
    transitive-closure fixtures below."""
    issue = _issue(number, labels, state)
    if blocked_by is not None:
        issue["body"] = f"Blocked by #{blocked_by}"
    return issue


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
# Item 3 / #1768: event kind registry migration
# ---------------------------------------------------------------------------


def test_operator_queue_impact_registered_as_warning() -> None:
    """``operator_queue_impact`` (the #1768 replacement kind) must be
    registered in ``_LEVEL_BY_KIND`` at ``"warning"``."""
    assert "operator_queue_impact" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["operator_queue_impact"] == "warning"


def test_operator_queue_impact_not_in_expected_operational_kinds() -> None:
    """Unlike the old gauge, ``operator_queue_impact`` must NOT be bucketed
    into ``EXPECTED_OPERATIONAL_KINDS`` -- it is edge-triggered by design
    (issue #1768), so it should read like any other genuinely rare warning
    in ``heartbeat_check.py``'s flat listing, not be summarized away."""
    assert "operator_queue_impact" not in EXPECTED_OPERATIONAL_KINDS


def test_operator_queue_depth_kind_fully_retired() -> None:
    """The old level-triggered kind name must not survive the #1768
    migration anywhere a consumer would see it: not in the level registry,
    not in the operational-kinds bucket. (The distinct ``operator_queue_depth``
    *function* in ``workflow.py`` -- a census helper, not an event kind --
    is intentionally unaffected; see the tests below.)"""
    assert "operator_queue_depth" not in _LEVEL_BY_KIND
    assert "operator_queue_depth" not in EXPECTED_OPERATIONAL_KINDS


def test_operator_queue_impact_desktop_severity_registered() -> None:
    """Issue #1768 AC2: the impact signal must reach the desktop-toast
    pipeline, which severity-gates on ``_DESKTOP_SEVERITIES``."""
    assert "OPERATOR_QUEUE_IMPACT" in _DESKTOP_SEVERITIES


# ---------------------------------------------------------------------------
# Item 3: operator_queue_depth() census function (unchanged by #1768 --
# still a legitimate, narrower census; simply no longer the emitter's root
# source, which now uses the broader `sink_census`).
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


def test_is_operator_queue_issue_predicate() -> None:
    """Issue #1768 review finding 3: ``is_operator_queue_issue`` is the
    single shared predicate behind both ``operator_queue_depth`` (this
    module) and ``operator_queue`` command's state-side numbers
    (``github_ops_operator_queue.py``). Direct unit coverage on the
    predicate itself, independent of either caller."""
    assert is_operator_queue_issue({"status": "escalated", "reason_class": "mechanical"}) is True
    assert is_operator_queue_issue({"status": "escalated", "reason_class": "judgment"}) is False
    assert is_operator_queue_issue({"status": "blocked", "reason_class": "mechanical"}) is False
    assert is_operator_queue_issue({}) is False


def test_operator_queue_depth_empty_state(tmp_path: Path) -> None:
    """An empty state must produce an empty depth set, not an error."""
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert operator_queue_depth(state) == set()


# ---------------------------------------------------------------------------
# Issue #1768: compute_operator_queue_impact() transitive-closure unit tests
# ---------------------------------------------------------------------------


def test_compute_impact_diamond_dependency_transitive_closure() -> None:
    """Pin the transitive-closure semantics with a synthetic diamond:
    root -> A -> B, root -> C (per the issue's own "Test expectations").
    All of A/B/C are automated-ready; the root and an unrelated issue are
    not part of the closure."""
    config = OrchestratorConfig()
    ready = config.labels.ready
    gh = FakeGitHub()
    gh.issues = [
        _issue(1, []),  # root: not itself automated-ready, not counted
        _blocked_issue(2, [ready], blocked_by=1),  # A
        _blocked_issue(3, [ready], blocked_by=2),  # B (blocked by A, transitively by root)
        _blocked_issue(4, [ready], blocked_by=1),  # C
        _issue(5, [ready]),  # unrelated, no blockers
    ]

    impact = compute_operator_queue_impact(gh, config, {1})

    assert impact.roots == (1,)
    assert impact.blocked_ready_issue_numbers == (2, 3, 4)
    assert impact.blocked_ready_count == 3


def test_compute_impact_non_ready_intermediate_does_not_cut_the_chain() -> None:
    """A chain through a non-ready intermediate issue must still reach a
    ready issue further down -- traversal follows the full graph
    regardless of label; only the reported set is ready-filtered."""
    config = OrchestratorConfig()
    ready = config.labels.ready
    gh = FakeGitHub()
    gh.issues = [
        _issue(10, []),  # root
        _blocked_issue(11, [], blocked_by=10),  # intermediate, NOT ready
        _blocked_issue(12, [ready], blocked_by=11),  # ready, blocked via the chain
    ]

    impact = compute_operator_queue_impact(gh, config, {10})

    assert impact.blocked_ready_issue_numbers == (12,)
    assert impact.blocked_ready_count == 1


def test_compute_impact_fresh_eyes_shape_one_root_blocks_most_of_backlog() -> None:
    """The #1768 investigation's fresh-eyes shape: a single root
    transitively blocks the large majority of a small open backlog (16 of
    20 dependents here, plus the root itself = 21 open issues total,
    matching the live measurement of 17/21)."""
    config = OrchestratorConfig()
    ready = config.labels.ready
    gh = FakeGitHub()
    gh.issues = (
        [_issue(12, [])]
        + [
            _blocked_issue(n, [ready], blocked_by=12)
            for n in range(13, 29)  # 16 dependents
        ]
        + [_issue(n, [ready]) for n in range(29, 33)]
    )  # 4 unrelated ready issues

    impact = compute_operator_queue_impact(gh, config, {12})

    assert impact.blocked_ready_count == 16
    assert set(impact.blocked_ready_issue_numbers) == set(range(13, 29))


def test_compute_impact_empty_roots_is_zero_impact_no_gh_call() -> None:
    """An empty root set must short-circuit before any GitHub call."""
    config = OrchestratorConfig()
    gh = FakeGitHub()
    call_count = {"n": 0}
    original = gh.issue_list

    def _counting_issue_list(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return original(*args, **kwargs)

    gh.issue_list = _counting_issue_list  # type: ignore[method-assign]

    impact = compute_operator_queue_impact(gh, config, set())

    assert impact == OperatorQueueImpact((), (), 0, None)
    assert call_count["n"] == 0


def test_compute_impact_success_is_observed() -> None:
    """A successful fetch (even with dependents) must report ``observed=True``
    -- the normal, trustworthy-zero-or-nonzero case."""
    config = OrchestratorConfig()
    ready = config.labels.ready
    gh = FakeGitHub()
    gh.issues = [_issue(1, []), _blocked_issue(2, [ready], blocked_by=1)]

    impact = compute_operator_queue_impact(gh, config, {1})

    assert impact.observed is True


def test_compute_impact_empty_open_issue_list_reports_unobserved() -> None:
    """Issue #1768 review finding 2: an ambiguously-empty fetch (a ``gh``
    call that succeeds but returns no open issues at all, while a non-empty
    root set exists) must report ``observed=False``, not a trustworthy zero
    -- mirroring ``classify_backlog_reachability``'s identical reasoning for
    the identical fetch. A genuinely trustworthy zero can only come from an
    empty *root* set (see ``test_compute_impact_empty_roots_is_zero_impact_no_gh_call``),
    never from an empty issue list."""
    config = OrchestratorConfig()
    gh = FakeGitHub()
    gh.issues = []

    impact = compute_operator_queue_impact(gh, config, {99})

    assert impact.roots == (99,)
    assert impact.blocked_ready_count == 0
    assert impact.observed is False


def test_compute_impact_github_error_reports_unobserved() -> None:
    """Issue #1768 review finding 1: a ``GitHubError`` raised by
    ``gh.issue_list`` (timeout, missing binary, non-zero exit) must be
    contained and reported as ``observed=False``, never propagated -- this
    check runs every loop pass and must never be able to crash an otherwise
    fully-completed pass over an advisory signal."""
    config = OrchestratorConfig()
    gh = FakeGitHub()

    def _raising_issue_list(*args: Any, **kwargs: Any) -> Any:
        raise GitHubError("gh: command timed out")

    gh.issue_list = _raising_issue_list  # type: ignore[method-assign]

    impact = compute_operator_queue_impact(gh, config, {99})

    assert impact.roots == (99,)
    assert impact.blocked_ready_count == 0
    assert impact.observed is False


# ---------------------------------------------------------------------------
# Issue #1768: age_bucket_label() unit tests
# ---------------------------------------------------------------------------


def test_age_bucket_label_boundaries() -> None:
    assert age_bucket_label(None) == "unknown"
    assert age_bucket_label(0.5) == "<1d"
    assert age_bucket_label(0.999) == "<1d"
    assert age_bucket_label(1.0) == "<3d"
    assert age_bucket_label(2.9) == "<3d"
    assert age_bucket_label(3.0) == "<7d"
    assert age_bucket_label(29.9) == "<30d"
    assert age_bucket_label(30.0) == ">=30d"
    assert age_bucket_label(90.0) == ">=30d"


# ---------------------------------------------------------------------------
# Issue #1768: should_fire_operator_queue_impact() edge-detection unit tests
# ---------------------------------------------------------------------------


def test_should_fire_no_baseline_always_fires() -> None:
    """The first-ever observation of a non-empty root set fires
    unconditionally -- this is what makes a brand-new single-root queue
    (the fresh-eyes shape) fire even though it would never cross a
    raw-count threshold."""
    current = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
    )
    assert should_fire_operator_queue_impact(None, current, now=datetime.now(UTC)) is True


def test_should_fire_unchanged_below_threshold_does_not_fire() -> None:
    signature = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
    )
    baseline = {**signature, "alerted_at": "2026-01-01T00:00:00Z"}
    assert should_fire_operator_queue_impact(baseline, signature, now=datetime.now(UTC)) is False


def test_should_fire_root_set_change_fires() -> None:
    baseline = {
        **operator_queue_impact_signature(
            root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
        ),
        "alerted_at": "2026-01-01T00:00:00Z",
    }
    current = operator_queue_impact_signature(
        root_issue_numbers=[12, 13], over_threshold=False, age_bucket="<1d"
    )
    assert should_fire_operator_queue_impact(baseline, current, now=datetime.now(UTC)) is True


def test_should_fire_threshold_crossing_fires() -> None:
    baseline = {
        **operator_queue_impact_signature(
            root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
        ),
        "alerted_at": "2026-01-01T00:00:00Z",
    }
    current = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=True, age_bucket="<1d"
    )
    assert should_fire_operator_queue_impact(baseline, current, now=datetime.now(UTC)) is True


def test_should_fire_age_bucket_crossing_fires() -> None:
    baseline = {
        **operator_queue_impact_signature(
            root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
        ),
        "alerted_at": "2026-01-01T00:00:00Z",
    }
    current = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=False, age_bucket="<3d"
    )
    assert should_fire_operator_queue_impact(baseline, current, now=datetime.now(UTC)) is True


def test_should_fire_unchanged_over_threshold_reminder_not_elapsed() -> None:
    signature = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=True, age_bucket="<1d"
    )
    now = datetime.now(UTC)
    baseline = {
        **signature,
        "alerted_at": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    assert should_fire_operator_queue_impact(baseline, signature, now=now) is False


def test_should_fire_unchanged_over_threshold_reminder_elapsed() -> None:
    signature = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=True, age_bucket="<1d"
    )
    now = datetime.now(UTC)
    baseline = {
        **signature,
        "alerted_at": (now - timedelta(hours=25)).isoformat().replace("+00:00", "Z"),
    }
    assert should_fire_operator_queue_impact(baseline, signature, now=now) is True


def test_should_fire_no_baseline_not_qualifying_does_not_fire() -> None:
    """Issue #1768 review finding 7: a first-ever observation with zero
    impact (``qualifies=False``) must NOT fire -- a brand-new sink arrival
    that blocks nothing is not itself alert-worthy."""
    current = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
    )
    assert (
        should_fire_operator_queue_impact(None, current, now=datetime.now(UTC), qualifies=False)
        is False
    )


def test_should_fire_root_set_change_not_qualifying_does_not_fire() -> None:
    """Issue #1768 review finding 7: a root-set change that still carries
    zero impact must NOT fire."""
    baseline = {
        **operator_queue_impact_signature(
            root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
        ),
        "alerted_at": "2026-01-01T00:00:00Z",
    }
    current = operator_queue_impact_signature(
        root_issue_numbers=[12, 13], over_threshold=False, age_bucket="<1d"
    )
    assert (
        should_fire_operator_queue_impact(
            baseline, current, now=datetime.now(UTC), qualifies=False
        )
        is False
    )


def test_should_fire_threshold_crossing_fires_even_when_not_qualifying() -> None:
    """An ``over_threshold`` flip must fire regardless of ``qualifies`` --
    crossing the threshold in either direction is meaningful on its own
    (issue #1768 review finding 7's docstring)."""
    baseline = {
        **operator_queue_impact_signature(
            root_issue_numbers=[12], over_threshold=False, age_bucket="<1d"
        ),
        "alerted_at": "2026-01-01T00:00:00Z",
    }
    current = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=True, age_bucket="<1d"
    )
    assert (
        should_fire_operator_queue_impact(
            baseline, current, now=datetime.now(UTC), qualifies=False
        )
        is True
    )


def test_should_fire_malformed_alerted_at_fires() -> None:
    """A corrupt durable marker must fail toward firing, not toward
    silently wedging the reminder off forever."""
    signature = operator_queue_impact_signature(
        root_issue_numbers=[12], over_threshold=True, age_bucket="<1d"
    )
    baseline = {**signature, "alerted_at": "not-a-timestamp"}
    assert should_fire_operator_queue_impact(baseline, signature, now=datetime.now(UTC)) is True


# ---------------------------------------------------------------------------
# Issue #1768: _maybe_emit_operator_queue_impact() end-to-end behavior
# ---------------------------------------------------------------------------


def _fresh_eyes_gh(root: int, ready_label: str, dependents: range) -> FakeGitHub:
    gh = FakeGitHub()
    gh.issues = [_issue(root, [])] + [
        _blocked_issue(n, [ready_label], blocked_by=root) for n in dependents
    ]
    return gh


def test_impact_fires_on_fresh_eyes_shape_first_occurrence(tmp_path: Path) -> None:
    """AC-critical case: a single root transitively blocking 16 of 20
    dependent automated-ready issues MUST fire on its very first
    observation, even though "1 root" never crosses any raw-count
    threshold -- the failure mode the old gauge could never catch."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
        notify=NotifyConfig(enabled=True, sink="file", file_path=str(tmp_path / "digest.jsonl")),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    events = query_events(app.paths.state_file, kind="operator_queue_impact")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["root_issue_numbers"] == [12]
    assert payload["blocked_ready_count"] == 16
    assert sorted(payload["blocked_ready_issue_numbers"]) == list(range(13, 29))

    # The old kind name must never appear in a new emission.
    assert query_events(app.paths.state_file, kind="operator_queue_depth") == []


def test_impact_delivers_attention_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2: the computed payload must actually reach ``notify.emit_digest``,
    not just ``events.db``."""
    import charlie_work.workflow as _wf

    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
        notify=NotifyConfig(enabled=True, sink="file", file_path=str(tmp_path / "digest.jsonl")),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    captured: list[Any] = []
    original_emit = _wf.emit_digest

    def _capture(notify_config: Any, digest: Any) -> Any:
        captured.append(digest)
        return original_emit(notify_config, digest)

    monkeypatch.setattr(_wf, "emit_digest", _capture)

    app._maybe_emit_operator_queue_impact()

    assert len(captured) == 1
    digest = captured[0]
    assert len(digest.transitions) == 1
    entry = digest.transitions[0]
    assert entry.issue_number == 0
    assert entry.adapter_kind == "operator_queue"
    assert entry.health == "OPERATOR_QUEUE_IMPACT"
    assert "16" in (entry.last_log_line or "")


def test_impact_no_digest_when_notify_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``notify.enabled=False`` (the default) must not attempt delivery --
    ``events.db`` still gets the event either way."""
    import charlie_work.workflow as _wf

    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
        notify=NotifyConfig(enabled=False),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    calls: list[Any] = []
    monkeypatch.setattr(_wf, "emit_digest", lambda *a, **k: calls.append((a, k)))

    app._maybe_emit_operator_queue_impact()

    assert calls == []
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1


def test_impact_second_pass_unchanged_emits_nothing(tmp_path: Path) -> None:
    """The no-change case: two consecutive passes over an unchanged root
    set / impact / age bucket must fire exactly once, not twice."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1

    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1, (
        "an unchanged root set / impact / age bucket must not re-fire on the very next pass"
    )


def test_impact_fires_again_on_root_set_change(tmp_path: Path) -> None:
    """A genuine set change (a second root arrives) must fire again."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )
    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1

    _seed_operator_queue_issue(
        app, 13, terminal_since="2026-01-02T00:00:00Z", reason_class="mechanical"
    )
    # 13 is now both a root AND one of 12's dependents; that's fine -- the
    # sink census and the blocker graph are independent views.
    app._maybe_emit_operator_queue_impact()

    events = query_events(app.paths.state_file, kind="operator_queue_impact")
    assert len(events) == 2
    assert events[-1]["payload"]["root_issue_numbers"] == [12, 13]


def test_impact_no_emit_when_no_roots(tmp_path: Path) -> None:
    """An empty sink (nothing parked) must never emit -- and must not touch
    state.json at all (no baseline was ever recorded)."""
    app = _app(
        tmp_path,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    # Initialize state.json via the normal load/save path (a fresh app has
    # not written it yet), so "before" reflects a real baseline rather than
    # a nonexistent file.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        save_state(app.paths.state_file, state)
    before_bytes = app.paths.state_file.read_bytes()

    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []
    assert app.paths.state_file.read_bytes() == before_bytes


def test_impact_drain_then_refill_fires_fresh(tmp_path: Path) -> None:
    """A queue that empties out and later refills with the *same* root/impact
    shape must still fire on the refill -- it is a fresh event, not a
    continuation of the drained one (the baseline is cleared on drain)."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )
    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1

    # Drain: clear the issue's escalated status entirely.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        del state["issues"]["12"]
        save_state(app.paths.state_file, state)
    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1, (
        "draining to empty must not itself emit"
    )

    # Refill with the identical shape.
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-03T00:00:00Z", reason_class="judgment"
    )
    app._maybe_emit_operator_queue_impact()

    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 2, (
        "a drain-then-refill of the identical root/impact shape must fire again, "
        "not be silently suppressed by a stale baseline"
    )


def test_impact_disabled_when_threshold_zero(tmp_path: Path) -> None:
    """Threshold 0 disables the alert entirely -- no event regardless of
    impact, preserving the pre-feature silent-queue behavior."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=0),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []


def test_impact_respects_review_cadence(tmp_path: Path) -> None:
    """When ``operator_queue_review_interval_minutes > 0``, the check is
    gated by the ``next_operator_queue_review_at`` timestamp. A future
    timestamp means the check is not due and no event is emitted, even for
    a fresh-eyes-shaped root set."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=5,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    future = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = arm_operator_queue_review(state, future)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []


def test_impact_emits_when_review_cadence_due(tmp_path: Path) -> None:
    """When the review cadence is due (past timestamp), the check fires
    normally."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=5,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = arm_operator_queue_review(state, past)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_impact()

    events = query_events(app.paths.state_file, kind="operator_queue_impact")
    assert len(events) == 1
    assert events[0]["payload"]["blocked_ready_count"] == 16


def test_impact_silent_under_dry_run_with_high_impact_queue(tmp_path: Path) -> None:
    """``dry_run=True`` must short-circuit the check even for a
    fresh-eyes-shaped high-impact queue that would otherwise fire on its
    first observation -- the C1.2 "byte-identical to a pass that never
    ran" dry-run invariant."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        dry_run=True,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    before_bytes = app.paths.state_file.read_bytes()

    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == [], (
        "dry_run=True must not emit operator_queue_impact even for a "
        "first-occurrence fresh-eyes-shaped queue"
    )
    assert app.paths.state_file.read_bytes() == before_bytes, (
        "dry_run=True must not mutate state.json"
    )


def _raising_gh() -> FakeGitHub:
    gh = FakeGitHub()

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise GitHubError("gh: command timed out")

    gh.issue_list = _raise  # type: ignore[method-assign]
    return gh


def test_impact_check_failure_does_not_fire_or_touch_baseline(tmp_path: Path) -> None:
    """Issue #1768 review findings 1/2, end-to-end: a ``gh.issue_list``
    failure must not emit an event, must not record a baseline, and must
    not raise out of ``_maybe_emit_operator_queue_impact`` -- a fabricated
    zero here would flip ``over_threshold`` to False and cause a false
    "resolved" alert on the next successful pass."""
    app = _app(
        tmp_path,
        gh=_raising_gh(),
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert operator_queue_impact_baseline(state) is None


def test_impact_check_failure_still_rearms_review_cadence(tmp_path: Path) -> None:
    """A failing check must still re-arm ``operator_queue_review_interval_minutes``
    (issue #1768 review finding 4) so a persistently failing ``gh`` binary
    is retried on the configured cadence, not hammered every single pass."""
    app = _app(
        tmp_path,
        gh=_raising_gh(),
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=5,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert is_operator_queue_review_due(state) is False


def test_impact_rearms_cadence_on_non_firing_pass(tmp_path: Path) -> None:
    """Issue #1768 review finding 4: the cadence marker must re-arm on
    every *completed* check, not only on the fire path -- otherwise the
    steady state (checked, materially unchanged) is permanently "due" and
    the GitHub fetch + whole-backlog walk runs every single pass forever,
    exactly the per-pass cost this knob exists to bound."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(
            enabled=False,
            operator_queue_depth_threshold=5,
            operator_queue_review_interval_minutes=30,
        ),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    # First pass fires and arms the cadence marker.
    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1

    # Force the cadence due again, then run a second, materially-unchanged
    # (non-firing) pass.
    forced_past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = arm_operator_queue_review(state, forced_past)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_impact()

    # Still exactly one event (non-firing pass), but the cadence marker
    # must have moved forward past "now" again -- not stayed stuck at the
    # forced-past value, which is what the pre-fix "only re-arm on the fire
    # path" bug would leave behind on a non-firing pass.
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    second_next_review = state["deescalation_pass"]["next_operator_queue_review_at"]
    assert is_operator_queue_review_due(state) is False
    assert second_next_review != forced_past


def test_impact_zero_impact_root_never_fires(tmp_path: Path) -> None:
    """Issue #1768 review finding 7, end-to-end: a sink root with zero
    transitive automated-ready impact must never fire -- neither on its
    first observation nor across a root-set change -- and must never
    record a baseline."""
    config = OrchestratorConfig()
    gh = FakeGitHub()
    # A non-empty, but wholly unrelated, open-issue list: keeps the fetch
    # genuinely "observed" (an empty list is itself ambiguous -- see
    # test_compute_impact_empty_open_issue_list_reports_unobserved) while
    # having zero dependents on either sink root below.
    gh.issues = [_issue(999, [config.labels.ready])]
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 50, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()
    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []

    _seed_operator_queue_issue(
        app, 51, terminal_since="2026-01-02T00:00:00Z", reason_class="mechanical"
    )
    app._maybe_emit_operator_queue_impact()

    assert query_events(app.paths.state_file, kind="operator_queue_impact") == []
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert operator_queue_impact_baseline(state) is None


def test_impact_digest_includes_blocked_issue_numbers(tmp_path: Path) -> None:
    """Issue #1768 review finding 8: the digest's log line must actually
    carry the blocked-ready issue numbers the emitter computed, not just
    the count -- ``blocked_ready_issue_numbers`` had no consumer before
    this fix."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 16))  # 3 dependents
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=1),
        notify=NotifyConfig(enabled=True, sink="file", file_path=str(tmp_path / "digest.jsonl")),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    digest_lines = (tmp_path / "digest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(digest_lines) == 1
    last_log_line = json.loads(digest_lines[0])["transitions"][0]["last_log_line"]
    assert "blocked_issues=" in last_log_line
    for number in (13, 14, 15):
        assert str(number) in last_log_line


def test_impact_payload_bounds_blocked_ready_issue_numbers(tmp_path: Path) -> None:
    """Issue #1768 review finding 9: the ring-resident event payload's
    ``blocked_ready_issue_numbers`` must be bounded, with an explicit
    ``blocked_ready_truncated`` count -- mirroring
    ``summarize_loop_errors``'s bounded-list-plus-truncated-count
    precedent. The scalar ``blocked_ready_count`` stays the true,
    uncapped count."""
    total_dependents = _BLOCKED_READY_ISSUE_NUMBERS_LIMIT + 10
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 13 + total_dependents))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    events = query_events(app.paths.state_file, kind="operator_queue_impact")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["blocked_ready_count"] == total_dependents
    assert len(payload["blocked_ready_issue_numbers"]) == _BLOCKED_READY_ISSUE_NUMBERS_LIMIT
    assert payload["blocked_ready_truncated"] == 10


def test_impact_file_sink_end_to_end(tmp_path: Path) -> None:
    """Issue #1768 review finding 6: a real, unmocked ``sink: file`` delivery
    -- every fleet repo runs this sink today, not desktop toast, so this is
    the actual delivery path AC2 needs covered, not just the
    ``_DESKTOP_SEVERITIES`` frozenset-membership check."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    digest_path = tmp_path / "digest.jsonl"
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
        notify=NotifyConfig(enabled=True, sink="file", file_path=str(digest_path)),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()

    assert digest_path.exists()
    lines = digest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["repo"] == tmp_path.name
    assert len(record["transitions"]) == 1
    entry = record["transitions"][0]
    assert entry["health"] == "OPERATOR_QUEUE_IMPACT"
    assert entry["adapter_kind"] == "operator_queue"
    assert "16" in entry["last_log_line"]


def test_impact_emitter_low_rate_reminder(tmp_path: Path) -> None:
    """Issue #1768 review finding 10: ``LOW_RATE_REMINDER_HOURS`` must also
    be exercised at the ``_maybe_emit_operator_queue_impact`` emitter
    level, not only through ``should_fire_operator_queue_impact``'s direct
    unit tests -- a materially-unchanged, still-over-threshold queue must
    re-fire once the reminder window has elapsed."""
    config = OrchestratorConfig()
    gh = _fresh_eyes_gh(12, config.labels.ready, range(13, 29))
    app = _app(
        tmp_path,
        gh=gh,
        deescalation=DeescalationConfig(enabled=False, operator_queue_depth_threshold=5),
    )
    _seed_operator_queue_issue(
        app, 12, terminal_since="2026-01-01T00:00:00Z", reason_class="judgment"
    )

    app._maybe_emit_operator_queue_impact()
    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 1

    # Rewrite the durable marker to look like it fired 25h ago (past the 24h
    # reminder window), with the identical signature.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        stale_alerted_at = (
            (datetime.now(UTC) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
        )
        state = _wf_record_signature_at(state, stale_alerted_at)
        save_state(app.paths.state_file, state)

    app._maybe_emit_operator_queue_impact()

    assert len(query_events(app.paths.state_file, kind="operator_queue_impact")) == 2, (
        "an unchanged, still-over-threshold queue must re-fire once the "
        "low-rate reminder window has elapsed"
    )


def _wf_record_signature_at(state: dict[str, Any], alerted_at: str) -> dict[str, Any]:
    """Test helper: overwrite the recorded operator-queue-impact baseline's
    ``alerted_at`` while keeping the rest of the signature identical."""
    from charlie_work.state import record_operator_queue_impact_signature

    baseline = operator_queue_impact_baseline(state)
    assert baseline is not None
    return record_operator_queue_impact_signature(
        state,
        root_issue_numbers=baseline["root_issue_numbers"],
        over_threshold=baseline["over_threshold"],
        age_bucket=baseline["age_bucket"],
        alerted_at=alerted_at,
    )


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
    """Only the two new operator-queue follow-up knobs are parsed from the
    ``deescalation`` YAML section. The rest of the section was previously
    100% inert (always defaulted) and stays inert in this PR — full-section
    activation is a separate, explicitly-reviewed change. So ``enabled`` and
    ``interval_minutes`` overrides in the YAML are silently ignored (they
    keep their dataclass defaults), while the two new fields are honored."""
    from charlie_work.config import build_config_from_data

    data = {
        "deescalation": {
            "enabled": False,
            "interval_minutes": 15,
            "operator_queue_review_interval_minutes": 10,
            "operator_queue_depth_threshold": 3,
        }
    }
    config = build_config_from_data(data)
    # The two new fields are parsed and honored.
    assert config.deescalation.operator_queue_review_interval_minutes == 10
    assert config.deescalation.operator_queue_depth_threshold == 3
    # Pre-existing fields keep their defaults — overrides are ignored, same
    # as before this PR (the section was inert).
    assert config.deescalation.enabled is True
    assert config.deescalation.interval_minutes == 30


def test_deescalation_config_tolerates_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys in the ``deescalation`` section must NOT brick startup.
    Before this PR the section was never parsed (always defaulted), so a live
    config may already carry a ``deescalation:`` block with typo'd or
    extra keys. Full-section parsing via ``_build_section`` would
    hard-reject those; this PR scopes to the two new fields only, so unknown
    keys are silently ignored."""
    from charlie_work.config import build_config_from_data

    data = {
        "deescalation": {
            "typo_key": "whatever",
            "future_field": 99,
            "operator_queue_depth_threshold": 7,
        }
    }
    config = build_config_from_data(data)
    assert config.deescalation.operator_queue_depth_threshold == 7


def test_deescalation_config_ignores_enabled_override(tmp_path: Path) -> None:
    """A pre-existing ``enabled: false`` override must NOT flip the sweep's
    default-on behavior. Before this PR the section was inert, so operators
    who set ``enabled: false`` expecting it to be a no-op (it was) must not
    have their sweep silently disabled by this PR's scoped parsing."""
    from charlie_work.config import build_config_from_data

    config = build_config_from_data({"deescalation": {"enabled": False}})
    assert config.deescalation.enabled is True


def test_deescalation_config_ignores_interval_minutes_override(tmp_path: Path) -> None:
    """A pre-existing ``interval_minutes`` override must NOT change the
    sweep cadence. Before this PR the section was inert, so the override was
    silently ignored; this PR preserves that."""
    from charlie_work.config import build_config_from_data

    config = build_config_from_data({"deescalation": {"interval_minutes": 5}})
    assert config.deescalation.interval_minutes == 30


def test_deescalation_config_rejects_negative_review_interval(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_review_interval_minutes"):
        build_config_from_data({"deescalation": {"operator_queue_review_interval_minutes": -1}})


def test_deescalation_config_rejects_non_int_review_interval(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_review_interval_minutes"):
        build_config_from_data({"deescalation": {"operator_queue_review_interval_minutes": "ten"}})


def test_deescalation_config_rejects_negative_depth_threshold(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_depth_threshold"):
        build_config_from_data({"deescalation": {"operator_queue_depth_threshold": -1}})


def test_deescalation_config_rejects_non_int_threshold(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="operator_queue_depth_threshold"):
        build_config_from_data({"deescalation": {"operator_queue_depth_threshold": "five"}})


def test_deescalation_config_threshold_errors_name_the_unit(tmp_path: Path) -> None:
    """Issue #1768 review finding 5: ``operator_queue_depth_threshold``
    silently changed units (root-count -> blocked-ready-issue-count) with
    this PR. The validation error messages must say so, since an operator
    reading a rejection message has no other way to learn the unit
    changed."""
    from charlie_work.config import ConfigError, build_config_from_data

    with pytest.raises(ConfigError, match="blocked-ready-issue count"):
        build_config_from_data({"deescalation": {"operator_queue_depth_threshold": -1}})
    with pytest.raises(ConfigError, match="blocked-ready-issue count"):
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
