"""N9: the throttle-window emission anchor, per log format.

The reader is chosen from the LOG'S SHAPE, not the harness: claude-code's
``.log`` is plain CLI prose unless ``tee_stream_json`` is on, so it gets the
tail-timestamp -> mtime chain (#1997); a stream-json log (api, reviewers) is read
event by event. In stream-json only the event carrying the throttle marker
counts: its own top-level ``"timestamp"``, else the log mtime when it is the
terminal ``result`` line (the CLI writes it last), else ``now`` -- never an
earlier assistant/user turn (retry backoff precedes the death) and never a
timestamp quoted inside a ``tool_result``.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work import worker_fate

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 29, 10, 20, 0, tzinfo=UTC)
EMITTED = datetime(2026, 9, 29, 10, 0, 0, tzinfo=UTC)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _copy_fixture(tmp_path: Path, name: str, *, mtime: datetime) -> Path:
    log_path = tmp_path / name
    shutil.copyfile(FIXTURES / name, log_path)
    os.utime(log_path, (mtime.timestamp(), mtime.timestamp()))
    return log_path


def test_stream_json_anchor_is_the_throttle_events_own_timestamp(tmp_path: Path) -> None:
    # mtime deliberately disagrees with the event's own timestamp.
    log_path = _copy_fixture(
        tmp_path,
        "claude_stream_json_throttle_assistant_ts.jsonl",
        mtime=NOW - timedelta(minutes=3),
    )

    for kind in ("claude-code", "api"):
        failure_kind, throttled_until = worker_fate.classify_for(kind, log_path, now=NOW)

        assert failure_kind == "rate_limited"
        assert throttled_until == _iso(EMITTED + timedelta(minutes=30))


def test_stream_json_ignores_a_timestamp_quoted_inside_a_tool_result(tmp_path: Path) -> None:
    # The throttle event has no top-level timestamp; the following
    # tool_result quotes an ancient ISO timestamp, and the mtime is old too.
    # The anchor must be classification time: now + cooldown, not the
    # quoted (or mtime) instant, which would collapse the window to `now`.
    log_path = _copy_fixture(
        tmp_path,
        "claude_stream_json_throttle_embedded_old_ts.jsonl",
        mtime=EMITTED,
    )

    failure_kind, throttled_until = worker_fate.classify_for("claude-code", log_path, now=NOW)

    assert failure_kind == "rate_limited"
    assert throttled_until == _iso(NOW + timedelta(minutes=30))


def test_devin_plain_text_anchor_is_the_last_timestamped_line(tmp_path: Path) -> None:
    log_path = _copy_fixture(
        tmp_path,
        "devin_plain_text_throttle.log",
        mtime=NOW - timedelta(minutes=3),
    )

    failure_kind, throttled_until = worker_fate.classify_for("devin", log_path, now=NOW)

    assert failure_kind == "rate_limited"
    assert throttled_until == _iso(EMITTED + timedelta(minutes=30))


def test_devin_plain_text_without_timestamp_still_anchors_at_log_mtime(tmp_path: Path) -> None:
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 30 minutes.\n",
        encoding="utf-8",
    )
    os.utime(log_path, (EMITTED.timestamp(), EMITTED.timestamp()))

    failure_kind, throttled_until = worker_fate.classify_for("devin", log_path, now=NOW)

    assert failure_kind == "rate_limited"
    assert throttled_until == _iso(EMITTED + timedelta(minutes=30))


def test_classify_for_unknown_adapter_kind_classifies_nothing(tmp_path: Path) -> None:
    log_path = _copy_fixture(
        tmp_path,
        "claude_stream_json_throttle_assistant_ts.jsonl",
        mtime=EMITTED,
    )

    assert worker_fate.classify_for("no-such-adapter", log_path, now=NOW) == (None, None)


def test_stream_json_result_without_timestamp_anchors_at_mtime_not_earlier_turn(
    tmp_path: Path,
) -> None:
    # Real CLI shape: the throttle `result` event has no timestamp; the only
    # stamped event is an assistant turn 20 minutes BEFORE the death. The
    # anchor must be the log mtime (the result line is written last), not
    # that turn's 09:40 -- which would undershoot the window by the backoff.
    log_path = _copy_fixture(
        tmp_path,
        "claude_stream_json_throttle_real_shape.jsonl",
        mtime=EMITTED,
    )

    for kind in ("claude-code", "api"):
        failure_kind, throttled_until = worker_fate.classify_for(kind, log_path, now=NOW)

        assert failure_kind == "rate_limited"
        assert throttled_until == _iso(EMITTED + timedelta(minutes=30))


def test_claude_code_plain_text_log_anchors_at_mtime(tmp_path: Path) -> None:
    # tee_stream_json defaults to False: a default claude-code log is prose.
    # It must keep the #1997 mtime anchor, not fall to classification time.
    log_path = _copy_fixture(
        tmp_path,
        "claude_plain_text_throttle.log",
        mtime=EMITTED,
    )

    failure_kind, throttled_until = worker_fate.classify_for("claude-code", log_path, now=NOW)

    assert failure_kind == "rate_limited"
    assert throttled_until == _iso(EMITTED + timedelta(minutes=30))
