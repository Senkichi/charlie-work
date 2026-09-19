"""Throttle/failure-classification tests for the devin-shell adapter.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
``_classify_session_failure`` log-tail classification, rate-limit
``get_rate_limit_defer_until`` parsing, ``set_throttled_until``
accumulation, and ``update_session_record_with_failure_classification``
sidecar updates.
"""

from __future__ import annotations

import json
from pathlib import Path

from _devin_shell_fixtures import _make_session_sidecar

from charlie_work.config import OrchestratorConfig, RuntimeConfig
from charlie_work.devin_shell import (
    get_rate_limit_defer_until,
    update_session_record_with_failure_classification,
)
from charlie_work.state import set_throttled_until


def test_classify_session_failure_rate_limit_with_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors with 'resets in N minutes' are classified correctly."""
    from charlie_work.devin_shell import _classify_session_failure
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    # Verify it's a valid ISO timestamp
    assert "T" in throttled_until
    assert "Z" in throttled_until
    # Verify the cooldown reflects the parsed 10 minutes
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected_time = (now + timedelta(minutes=10)).replace(microsecond=0)
    assert throttle_time == expected_time


def test_classify_session_failure_rate_limit_without_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors without reset time use default cooldown."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\nError: Reached overall message rate limit. Please try again later.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    # Should use default 15 minute cooldown
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_quota_exhausted(tmp_path: Path) -> None:
    """Test that quota-exhaustion errors are classified correctly."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: daily usage quota has been exhausted. Please try again tomorrow.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "quota_exhausted"
    assert throttled_until is not None
    # Should use default 24 hour cooldown
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_no_throttle(tmp_path: Path) -> None:
    """Test that non-throttle errors return None."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\nError: something went wrong with the task\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_missing_log(tmp_path: Path) -> None:
    """Test that missing log files return None."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "nonexistent.log"

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_includes_resume_margin(tmp_path: Path) -> None:
    """Issue #499: killed-worker rate-limit classification must include the resume margin."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 3 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    failure_kind, throttled_until = _classify_session_failure(
        log_path, resume_margin_seconds=90, now=now
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=3, seconds=90)).replace(microsecond=0)
    assert parsed == expected


def test_get_rate_limit_defer_until_with_reset_time(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until returns a deadline offset by the parsed reset time plus slack."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    defer_until = get_rate_limit_defer_until(log_path, slack_minutes=2, now=now)

    assert defer_until is not None
    assert "T" in defer_until
    assert "Z" in defer_until
    expected = (now + timedelta(minutes=10 + 2)).replace(microsecond=0)
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    assert parsed == expected


def test_get_rate_limit_defer_until_without_reset_time(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until uses the default cooldown when no reset time is present."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    defer_until = get_rate_limit_defer_until(log_path, slack_minutes=2, now=now)

    assert defer_until is not None
    expected = (now + timedelta(minutes=15 + 2)).replace(microsecond=0)
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    assert parsed == expected


def test_get_rate_limit_defer_until_no_match(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until returns None for non-rate-limit logs."""
    log_path = tmp_path / "session.log"
    log_path.write_text("Working on task...\n", encoding="utf-8")

    assert get_rate_limit_defer_until(log_path, slack_minutes=2) is None


def test_get_rate_limit_defer_until_includes_resume_margin(tmp_path: Path) -> None:
    """Issue #499: provider reset estimates are floors; add a resume margin."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 3 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    defer_until = get_rate_limit_defer_until(
        log_path,
        slack_minutes=2,
        now=now,
        resume_margin_seconds=90,
    )

    assert defer_until is not None
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=3 + 2, seconds=90)).replace(microsecond=0)
    assert parsed == expected


def test_set_throttled_until_overwrites_no_accumulation() -> None:
    """set_throttled_until replaces the value; it does not accumulate margins."""
    from datetime import UTC, datetime, timedelta

    first = (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    second = (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")

    original = {}
    state = set_throttled_until(original, first)
    state = set_throttled_until(state, second)

    assert state["throttled_until"] == second
    assert original.get("throttled_until") is None


def test_update_session_record_with_failure_classification(tmp_path: Path) -> None:
    """Test that session records are updated with failure classification."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a session sidecar
    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with rate-limit error
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None

    # Verify the sidecar was updated
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_update_session_record_with_failure_classification_includes_resume_margin(
    tmp_path: Path,
) -> None:
    """Issue #499: update wrapper applies config.runtime.throttle_resume_margin_s."""
    from datetime import UTC, datetime, timedelta

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 5 minutes.\n",
        encoding="utf-8",
    )

    config = OrchestratorConfig(runtime=RuntimeConfig(throttle_resume_margin_s=90))
    now = datetime.now(UTC)
    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, config=config, now=now
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=5, seconds=90)).replace(microsecond=0)
    assert parsed == expected


def test_update_session_record_with_failure_classification_session_completed_skips_log_tail(
    tmp_path: Path,
) -> None:
    """Issue #656: a completed session's own prose must not be reclassified quota_exhausted.

    Sibling of the claude_code.py regression test -- same fix, devin adapter.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") that '
        "legitimately appear in this codebase's rate-limit/quota domain commentary.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="unpublished_work", session_completed=True
    )

    assert failure_kind == "unpublished_work"
    assert throttled_until is None

    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "unpublished_work"


def test_update_session_record_skips_already_classified(tmp_path: Path) -> None:
    """Test that already-classified records are not re-classified."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a session sidecar with existing classification
    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
                "failure_kind": "rate_limited",  # Already classified
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with a different error (should be ignored)
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: daily usage quota has been exhausted.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42
    )

    # Should return the existing classification, not re-classify
    assert failure_kind == "rate_limited"
    assert throttled_until is None  # No new throttled_until

    # Verify the sidecar was not changed
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_classify_session_failure_tool_rejected_is_not_throttle(tmp_path: Path) -> None:
    """Issue #260, corrected premise: 'A tool was rejected by the user' is the
    Devin CLI's own surfacing of a PreToolUse hook block, not a provider
    throttle condition — it must NOT classify as rate_limited (no retry
    semantics, no throttled_until). The original PR #263 premise treated
    this string as a throttle signature; a correction comment on issue #260
    established it is a hard failure that must instead route through
    post_mortem.classify_and_record's worker_blocked log-tail fallback (see
    test_post_mortem_log_tail_fallback.py), which composes with escalation, not cooldown."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_update_session_record_tool_rejected_is_not_rate_limited(tmp_path: Path) -> None:
    """Issue #260, corrected premise: a tool-rejected sidecar log must not be
    classified rate_limited by the adapter's own log-tail classifier — see
    test_classify_session_failure_tool_rejected_is_not_throttle."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_session_record_unknown_tail_falls_back_to_stalled(tmp_path: Path) -> None:
    """Unknown log tail should fall back to the provided fallback_kind."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: something completely unrelated went wrong\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_session_record_custom_throttle_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable without code changes."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: provider-specific frobnicate limit exceeded\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    config = OrchestratorConfig(
        runtime=RuntimeConfig(throttle_error_markers=("frobnicate limit exceeded",))
    )
    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, config=config
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"
