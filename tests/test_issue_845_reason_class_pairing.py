"""Tests for issue #845: escalation ``escalation_reason`` and ``reason_class`` must stay paired.

Issue #845: when a dispatch failure was non-terminal, the ``else`` branch in
``_update_dispatch_outcome`` popped ``escalation_reason`` but left the stale
``reason_class`` behind. That stale ``reason_class`` could mislead the
``_maybe_deescalate_mechanical`` sweep into treating a later escalation as
mechanical when it should be judgment.

The fix consolidates every escalation behind ``workflow._escalate_issue`` and
clearing behind ``state.clear_escalation``, so the two fields are written and
cleared as an atomic pair and a terminal status cannot be written without a
reason.

Issue #981: ``state.set_escalation`` was the pre-#750 half-write helper and is
now dead code; these tests exercise ``_escalate_issue`` instead.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import DevinConfig, OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    clear_escalation,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp, _escalate_issue

from test_charlie_work import FakeGitHub


def _closed_pr_app(tmp_path: Path) -> tuple[OrchestratorApp, FakeGitHub]:
    """A dispatchable ready issue #123, with the default fixture PR #456
    closed so it doesn't trip the open-PR exclusion."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _fake_dispatch_sessions_factory(failure_kind: str | None):
    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="launch failed",
                failure_kind=failure_kind,
            )
            for request in requests
        ]

    return fake_dispatch_sessions


def test_escalate_issue_and_clear_escalation_pair_fields() -> None:
    """``_escalate_issue`` writes status and the paired fields;
    ``clear_escalation`` removes the paired fields."""
    state: dict[str, Any] = {
        "issues": {"123": {"number": 123, "status": "rework_requested"}},
        "prs": {},
    }

    state = _escalate_issue(
        state,
        123,
        reason="some_reason",
        reason_class="mechanical",
    )
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "some_reason"
    assert issue["reason_class"] == "mechanical"

    clear_escalation(issue)
    assert "escalation_reason" not in issue
    assert "reason_class" not in issue

    clear_escalation({})  # safe on empty dicts

    with pytest.raises(ValueError):
        _escalate_issue(
            {"issues": {"1": {"number": 1}}, "prs": {}},
            1,
            reason="x",
            reason_class="invalid",
        )


def test_state_integrity_no_paired_field_without_the_other() -> None:
    """A state-integrity assertion: ``escalation_reason`` and ``reason_class``
    must appear and disappear as a pair on escalated/block issue entries, and
    ``escalation_reason`` must never appear on a non-escalated entry.
    """
    state = {
        "issues": {
            "123": {
                "number": 123,
                "status": "escalated",
                "escalation_reason": "review_request_changes_cap_exceeded",
                "reason_class": "mechanical",
            },
            "124": {
                "number": 124,
                "status": "blocked",
                "escalation_reason": "review_blocked",
                "reason_class": "judgment",
            },
            "125": {
                "number": 125,
                "status": "rework_requested",
                "escalation_reason": "stale_reason",
                "reason_class": "stale_class",
            },
        }
    }

    with pytest.raises(AssertionError):
        _assert_issue_escalation_pairing(state)

    # Once the non-terminal issue is cleared, the invariant holds.
    state["issues"]["125"] = clear_escalation(dict(state["issues"]["125"]))
    _assert_issue_escalation_pairing(state)


def _assert_issue_escalation_pairing(state: dict) -> None:
    """Check that issue escalation fields are kept paired.

    - ``escalation_reason`` must never appear without ``reason_class``.
    - ``escalation_reason`` must only appear on ``escalated`` or ``blocked`` issues.
    - ``escalated`` or ``blocked`` issues with ``reason_class`` must also have
      ``escalation_reason`` (legacy backfill may leave only ``reason_class``,
      but new code should never create that state).
    """
    for key, entry in state.get("issues", {}).items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        has_reason = "escalation_reason" in entry
        has_class = "reason_class" in entry

        if has_reason:
            assert has_class, f"issue {key}: escalation_reason without reason_class"
            assert status in ("escalated", "blocked"), (
                f"issue {key}: escalation_reason on status {status}"
            )

        if status in ("escalated", "blocked") and has_class:
            assert has_reason, (
                f"issue {key}: {status} issue with reason_class but no escalation_reason"
            )


def test_dispatch_non_terminal_failure_clears_stale_reason_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-terminal dispatch failure must clear both escalation fields, not
    leave a stale ``reason_class`` behind."""
    app, _ = _closed_pr_app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatch_failed",
            "dispatch_failed_at": [
                (datetime.now(UTC) - timedelta(hours=5)).isoformat().replace("+00:00", "Z")
            ],
            # Pre-existing stale pair: the bug used to pop escalation_reason
            # but leave reason_class.
            "escalation_reason": "redispatch_cap_exceeded",
            "reason_class": "mechanical",
        }
        save_state(app.paths.state_file, state)

    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_sessions_factory(None),
    )

    result = app.dispatch(limit=1)
    assert result.ok is False

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "dispatch_failed"
    assert "escalation_reason" not in issue
    assert "reason_class" not in issue


def test_record_review_request_changes_cap_writes_paired_fields(
    tmp_path: Path,
) -> None:
    """A request_changes verdict that hits the rework cap must write a
    non-null ``escalation_reason`` alongside ``reason_class == "mechanical"``."""
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
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "max_rework_cycles_exceeded"
    assert issue["reason_class"] == "mechanical"


def test_record_review_blocked_writes_paired_judgment_fields(
    tmp_path: Path,
) -> None:
    """A blocked verdict must write a non-null ``escalation_reason``
    alongside ``reason_class == "judgment"``."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "blocked", summary="security concern")
    assert result.ok is True

    state = load_state(paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "blocked"
    assert issue["escalation_reason"] == "review_blocked"
    assert issue["reason_class"] == "judgment"


def test_deescalation_sweep_skips_blocked_review_verdict(
    tmp_path: Path,
) -> None:
    """A blocked review verdict is ``judgment``; the mechanical de-escalation
    sweep must leave it untouched."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "blocked",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "blocked",
            "escalation_reason": "review_blocked",
            "reason_class": "judgment",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "blocked"
    assert issue["reason_class"] == "judgment"
    assert issue["escalation_reason"] == "review_blocked"
    assert "auto_deescalation_count" not in issue
