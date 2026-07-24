"""Tests for TASK W5: reviewer-quota probe exponential backoff.

Covers cost-spirals.md Finding 2: ``quota_probe_interval_minutes`` had "No
escalation backoff" (config.py comment, verbatim) -- a live provider outage
relaunched a real reviewer session into the wall every 15 minutes forever,
and provider-throttle stalls are explicitly exempt from the per-PR dispatch
attempt cap (workflow.py's stalled-review handling), so this was the one
failure mode that could not terminate on its own.

``_set_reviewer_quota_exhausted_with_backoff`` is a pure module-level
function (state, config, now) -> state, so the doubling/cap behavior is
tested directly without needing to drive the full dispatch_reviews()
subprocess machinery. The reset-on-success path lives inside
dispatch_reviews() itself (cleared only when a real verdict is reaped from a
dead reviewer -- the only proof the provider quota window is actually open),
so that one test reuses the existing ``_dispatch_reviews_app`` /
sidecar-log fixture pattern already established in test_charlie_work.py's
``test_dispatch_reviews_probe_success_clears_reviewer_quota``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import _set_reviewer_quota_exhausted_with_backoff

from test_charlie_work import _dispatch_reviews_app, _fake_claude_worker_record


def _probe_after(state: dict[str, Any]) -> datetime:
    raw = state["reviewer_quota"]["probe_after"]
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def test_probe_backoff_doubles_each_consecutive_failure() -> None:
    config = OrchestratorConfig()
    now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)
    state: dict[str, Any] = {}

    state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 1
    assert _probe_after(state) - now == timedelta(minutes=15)

    state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 2
    assert _probe_after(state) - now == timedelta(minutes=30)

    state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 3
    assert _probe_after(state) - now == timedelta(minutes=60)

    state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 4
    assert _probe_after(state) - now == timedelta(minutes=120)


def test_probe_backoff_caps_at_configured_max_interval() -> None:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            quota_probe_interval_minutes=15,
            quota_probe_max_interval_minutes=40,
        )
    )
    now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)
    state: dict[str, Any] = {}

    # 15 -> 30 -> 60 (capped to 40) -> 120 (capped to 40)
    for _ in range(4):
        state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    assert _probe_after(state) - now == timedelta(minutes=40)


def test_probe_backoff_uncapped_when_max_interval_is_zero() -> None:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            quota_probe_interval_minutes=15,
            quota_probe_max_interval_minutes=0,
        )
    )
    now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)
    state: dict[str, Any] = {}

    for _ in range(5):
        state = _set_reviewer_quota_exhausted_with_backoff(state, config, now)
    # 15 * 2^4 = 240, uncapped (0 disables the cap per the config docstring).
    assert _probe_after(state) - now == timedelta(minutes=240)


def test_probe_backoff_resets_on_successful_verdict_reap(monkeypatch, tmp_path) -> None:
    """A verdict recorded from a dead reviewer is the only proof the provider
    quota window is actually open (dispatch_reviews's own comment) -- it must
    reset ``consecutive_probe_failures`` back to 0 so the next outage starts
    from the configured base interval again instead of carrying forward an
    exponentially-grown one.
    """
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    future_throttle = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    past_probe = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["reviewer_quota"] = {
            "throttled_until": future_throttle,
            "probe_after": past_probe,
            "consecutive_probe_failures": 3,
        }
        state["prs"]["100"] = {
            **state["prs"].get("100", {}),
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
        }
        save_state(app.paths.state_file, state)

    reviews_dir = tmp_path / app.config.review_dispatch.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        'Review complete.\n```json\n{"decision": "approved", "summary": "LGTM"}\n```',
        encoding="utf-8",
    )
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt"),
                "command": ["claude"],
                "pid": 0,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "log_path": str(log_path),
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *args, **kwargs: _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        ),
    )

    app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    assert state.get("reviewer_quota", {}).get("throttled_until") is None
    assert state.get("reviewer_quota", {}).get("consecutive_probe_failures") == 0
