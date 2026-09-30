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


# --- merged stderr (r3 B1) ------------------------------------------------
# ``claude_code`` launches with ``stderr=STDOUT``, so the ``.log`` interleaves
# the CLI's stream-json events with stderr prose; on this host most json-bearing
# logs END with a non-JSON ``SessionEnd hook ... failed`` line. The log's shape
# must be decided from the JSON lines anywhere in the tail, not the last line.

HOOK_STDERR = (
    "SessionEnd hook [pwsh session-end.ps1] failed: exit 1 at 2026-09-29T09:05:00+00:00\n"
)


def _with_trailing_stderr(tmp_path: Path, name: str, *, mtime: datetime) -> Path:
    log_path = _copy_fixture(tmp_path, name, mtime=mtime)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(HOOK_STDERR)
    os.utime(log_path, (mtime.timestamp(), mtime.timestamp()))
    return log_path


def test_trailing_stderr_line_does_not_change_the_stream_json_anchor(tmp_path: Path) -> None:
    # Same log as the real-shape test plus one trailing stderr line (which even
    # carries a timestamp): must equal the clean-log answer, never the 09:40
    # assistant turn nor the stderr line's 09:05.
    clean = _copy_fixture(tmp_path, "claude_stream_json_throttle_real_shape.jsonl", mtime=EMITTED)
    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    noisy = _with_trailing_stderr(
        noisy_dir, "claude_stream_json_throttle_real_shape.jsonl", mtime=EMITTED
    )

    for kind in ("claude-code", "api"):
        expected = worker_fate.classify_for(kind, clean, now=NOW)
        assert expected == ("rate_limited", _iso(EMITTED + timedelta(minutes=30)))
        assert worker_fate.classify_for(kind, noisy, now=NOW) == expected


def test_trailing_stderr_keeps_a_quoted_timestamp_from_becoming_the_anchor(
    tmp_path: Path,
) -> None:
    # The throttle event has no timestamp and a tool_result quotes 2026-01-01;
    # a stderr line follows. Anchor is classification time, exactly as without
    # the stderr line -- never the quoted instant (cooldown collapsed to 10:20).
    log_path = _with_trailing_stderr(
        tmp_path, "claude_stream_json_throttle_embedded_old_ts.jsonl", mtime=EMITTED
    )

    failure_kind, throttled_until = worker_fate.classify_for("claude-code", log_path, now=NOW)

    assert failure_kind == "rate_limited"
    assert throttled_until == _iso(NOW + timedelta(minutes=30))


def test_stream_json_event_timestamp_wins_over_trailing_stderr(tmp_path: Path) -> None:
    log_path = _with_trailing_stderr(
        tmp_path,
        "claude_stream_json_throttle_assistant_ts.jsonl",
        mtime=NOW - timedelta(minutes=3),
    )

    _, throttled_until = worker_fate.classify_for("claude-code", log_path, now=NOW)

    assert throttled_until == _iso(EMITTED + timedelta(minutes=30))


def test_plain_text_reader_ignores_json_lines_and_truncated_fragments() -> None:
    from charlie_work.failure_classifier import _is_stream_json, _tail_emission_timestamp

    tail = (
        '_cut","timestamp":"2026-01-01T00:00:00Z"}\n'  # first line cut by the byte slice
        "Error: rate limit hit at 2026-09-29T10:00:00+00:00\n"
        '{"type":"user","timestamp":"2026-09-29T09:59:00+00:00"}\n'
        '{"type":"user","message":{"content":"2026-09-29T10:19:00+00:00 quoted"}\n'  # fragment
    )

    assert _is_stream_json(tail)  # a whole event line anywhere decides the shape
    assert _tail_emission_timestamp(tail) == EMITTED


def test_prose_with_a_json_looking_word_is_not_stream_json() -> None:
    from charlie_work.failure_classifier import _is_stream_json

    assert not _is_stream_json("plain prose\n{not json}\nmore prose\n")
    assert not _is_stream_json('{"no_type": 1}\n')
