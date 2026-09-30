"""Throttle/failure-classification tests for the devin-shell adapter.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
``worker_fate.classify_for`` log-tail classification, rate-limit
``get_rate_limit_defer_until`` parsing, ``set_throttled_until``
accumulation, and ``update_session_record_with_failure_classification``
sidecar updates.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

from _devin_shell_fixtures import _make_session_sidecar

from charlie_work import worker_fate
from charlie_work.config import OrchestratorConfig, RuntimeConfig
from charlie_work.devin_shell import (
    get_rate_limit_defer_until,
    update_session_record_with_failure_classification,
)
from charlie_work.state import set_throttled_until

_classify_session_failure = partial(worker_fate.classify_for, "devin")


def test_classify_session_failure_rate_limit_with_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors with 'resets in N minutes' are classified correctly."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC).replace(microsecond=0)
    # Issue #1997: the window is anchored at the log's emission time (mtime),
    # so pin mtime to the frozen clock for an exact assertion.
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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

    log_path = tmp_path / "nonexistent.log"

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_includes_resume_margin(tmp_path: Path) -> None:
    """Issue #499: killed-worker rate-limit classification must include the resume margin."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 3 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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

    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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

    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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

    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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
    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))
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


def test_classify_session_failure_anchors_window_to_log_mtime(tmp_path: Path) -> None:
    """Issue #1997 criterion 1: the throttle window counts from when the
    provider emitted "reset in N minutes" (the log's mtime), not from when
    the orchestrator classifies the death.

    A log whose mtime is T and whose tail says "reset in 37 minutes",
    classified at T+45min, yields ``max(now, T+37min+margin) = T+45min``
    (already expired — the real reset has passed) — not ``T+45min+37min``
    (the old classification-time anchoring, which would idle the fleet ~45
    minutes past the real reset).
    """
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=45)
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached free model rate limit. Please try again later. "
        "Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (emitted_at.timestamp(), emitted_at.timestamp()))

    failure_kind, throttled_until = _classify_session_failure(
        log_path, resume_margin_seconds=90, now=now
    )

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    # T+37min+margin < T+45min: the anchored window ended before
    # classification, so the issue's max(now, anchored) formula clamps to
    # ``now`` — an already-expired deadline, never the old T+82min.
    assert parsed == now
    assert parsed != (now + timedelta(minutes=37, seconds=90)).replace(microsecond=0)
    # And provably not the raw anchored value either (it is ~8 min in the
    # past; the clamp floor is the later of the two).
    assert parsed > emitted_at + timedelta(minutes=37, seconds=90)


def test_classify_session_failure_anchored_window_still_future(tmp_path: Path) -> None:
    """Issue #1997: when the anchored window has not expired yet, the
    remaining time still gates dispatch — anchored at emission, the window
    end lands earlier than classification-time anchoring produced."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=45)
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached free model rate limit. Your limit will reset in 60 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (emitted_at.timestamp(), emitted_at.timestamp()))

    failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (emitted_at + timedelta(minutes=60)).replace(microsecond=0)
    assert parsed == expected
    assert parsed > now


def test_classify_session_failure_expired_window_clamps_to_now(tmp_path: Path) -> None:
    """Issue #1997 criterion 2: a window already in the past after anchoring
    clamps to ``now`` — an already-expired ``throttled_until`` (no deferral),
    per the issue's ``max(now, emitted_at + reset + margin)`` formula."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=45)
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached free model rate limit. Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (emitted_at.timestamp(), emitted_at.timestamp()))

    failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    # T+37min < T+45min: the reset has already happened by classification
    # time, so the stored window clamps to ``now`` — already expired
    # (state.is_throttled compares ``now < throttled_until``, which is never
    # true of the stored timestamp), and provably NOT the raw anchored value
    # (~8 minutes in the past).
    assert parsed == now
    assert parsed > emitted_at + timedelta(minutes=37)


def test_classify_session_failure_prefers_tail_line_timestamp(tmp_path: Path) -> None:
    """Issue #1997: a tz-aware timestamp on a tail line is a better emission
    anchor than the file mtime — it is the line's actual write time."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=10)
    ts_line = emitted_at.isoformat().replace("+00:00", "Z")
    log_path = tmp_path / "session.log"
    log_path.write_text(
        f"{ts_line} Error: Reached free model rate limit. Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    # mtime disagrees with the tail-line timestamp; the line wins. The
    # anchored window stays live (emitted_at+37 > now) so the three
    # candidates — tail timestamp, mtime, and classification time — each
    # produce a distinct deadline.
    later = now - timedelta(minutes=5)
    os.utime(log_path, (later.timestamp(), later.timestamp()))

    failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (emitted_at + timedelta(minutes=37)).replace(microsecond=0)
    assert parsed == expected
    assert parsed != (later + timedelta(minutes=37)).replace(microsecond=0)


def test_classify_session_failure_naive_tail_timestamp_uses_mtime(tmp_path: Path) -> None:
    """Issue #1997: a tail-line timestamp without a UTC offset is ambiguous —
    skipped in favor of the file mtime rather than a guessed zone."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=45)
    naive_ts = emitted_at.strftime("%Y-%m-%d %H:%M:%S")  # no offset
    log_path = tmp_path / "session.log"
    log_path.write_text(
        f"{naive_ts} Error: Reached free model rate limit. Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    mtime = now - timedelta(minutes=10)
    os.utime(log_path, (mtime.timestamp(), mtime.timestamp()))

    failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (mtime + timedelta(minutes=37)).replace(microsecond=0)
    assert parsed == expected


def test_classify_session_failure_falls_back_to_now_when_stat_fails(tmp_path: Path) -> None:
    """Issue #1997: with no usable mtime (stat fails) and no tail timestamp,
    the anchor falls back to ``now`` — the pre-#1997 behavior."""
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached free model rate limit. Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC).replace(microsecond=0)
    real_stat = Path.stat
    stat_calls = 0

    def flaky_stat(self: Path, *args: object, **kwargs: object) -> object:
        # exists() stats once; make every later stat (the anchor's) fail.
        nonlocal stat_calls
        stat_calls += 1
        if stat_calls > 1:
            raise OSError("simulated stat failure")
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", flaky_stat):
        failure_kind, throttled_until = _classify_session_failure(log_path, now=now)

    assert failure_kind == "rate_limited"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=10)).replace(microsecond=0)
    assert parsed == expected


def test_get_rate_limit_defer_until_anchors_to_log_mtime(tmp_path: Path) -> None:
    """Issue #1997 criterion 3: get_rate_limit_defer_until uses the same
    emission-time anchor — ``emitted_at + reset + slack + margin``, not
    ``now + reset + slack + margin``. The window is still live here, so the
    stored value IS the anchored deadline (no clamp)."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=10)
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached free model rate limit. Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (emitted_at.timestamp(), emitted_at.timestamp()))

    defer_until = get_rate_limit_defer_until(
        log_path, slack_minutes=2, now=now, resume_margin_seconds=30
    )

    assert defer_until is not None
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    expected = (emitted_at + timedelta(minutes=37 + 2, seconds=30)).replace(microsecond=0)
    assert parsed == expected
    # Provably emission-anchored, not classification-anchored (which would
    # store now + 39.5min — 10 minutes later).
    assert parsed != (now + timedelta(minutes=37 + 2, seconds=30)).replace(microsecond=0)


def test_get_rate_limit_defer_until_expired_window_clamps_to_now(tmp_path: Path) -> None:
    """Issue #1997 criterion 2 (defer variant): an anchored defer window
    already in the past clamps to ``now`` — the caller's ``now < defer_until``
    check then reports not-deferred (a zero-length window, already expired)."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).replace(microsecond=0)
    emitted_at = now - timedelta(minutes=45)
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached free model rate limit. Your limit will reset in 37 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (emitted_at.timestamp(), emitted_at.timestamp()))

    defer_until = get_rate_limit_defer_until(log_path, slack_minutes=2, now=now)

    assert defer_until is not None
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    assert parsed == now
    # Provably the clamp, not the raw anchored value (~6 minutes in the past).
    assert parsed > emitted_at + timedelta(minutes=37 + 2)
