"""Issue #2813: the watchdog's loop/no-progress check (``worker`` Signal 5)
against the real Claude Code ``--output-format stream-json`` schema.

Claude Code's stream-json never emits a ``tool_call`` event type: a tool
call is an ``assistant`` event carrying a ``tool_use`` content block, timed
by that event's top-level ISO ``timestamp``. ``claude_code.
last_tool_call_timestamp`` is the single parser for that; ``worker.
classify_worker_health``'s Signal 5 reads it.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.claude_code import last_tool_call_timestamp
from charlie_work.config import OrchestratorConfig, WatchdogConfig
from charlie_work.post_mortem import _events_path_from_log
from charlie_work.worker import WorkerHealth, WorkerView, classify_worker_health


def _ts(dt: datetime) -> str:
    return dt.isoformat()


def _assistant_tool_use(timestamp: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]
        },
    }


def _assistant_text(timestamp: str, text: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _user_tool_result(timestamp: str) -> dict[str, object]:
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]},
    }


def _system_init() -> dict[str, object]:
    return {"type": "system", "subtype": "init", "cwd": ".", "session_id": "s"}


def _write_events(path: Path, lines: list[object]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def test_real_stream_json_tool_use_yields_assistant_timestamp(tmp_path: Path) -> None:
    """A tool_use block in an assistant event is the tool call; later text-only and tool_result events don't supersede it."""
    path = tmp_path / "issue-1.events.jsonl"
    t1 = datetime.now(UTC) - timedelta(minutes=5)
    t2 = t1 + timedelta(seconds=10)
    t3 = t2 + timedelta(seconds=10)
    _write_events(
        path,
        [
            _system_init(),
            _assistant_tool_use(_ts(t1)),
            _user_tool_result(_ts(t2)),
            _assistant_text(_ts(t3), "thinking plateau"),
        ],
    )
    assert last_tool_call_timestamp(path) == t1


def test_control_without_tool_use_block_yields_none(tmp_path: Path) -> None:
    """The same file with the tool_use block removed has no tool call at all."""
    path = tmp_path / "issue-1.events.jsonl"
    t2 = datetime.now(UTC) - timedelta(minutes=5)
    t3 = t2 + timedelta(seconds=10)
    _write_events(
        path,
        [
            _system_init(),
            _user_tool_result(_ts(t2)),
            _assistant_text(_ts(t3), "no tools here"),
        ],
    )
    assert last_tool_call_timestamp(path) is None


def test_legacy_tool_call_shape_still_yields_timestamp(tmp_path: Path) -> None:
    """The legacy ``{"type": "tool_call", ...}`` shape keeps working."""
    path = tmp_path / "issue-1.events.jsonl"
    moment = datetime.now(UTC) - timedelta(minutes=5)
    _write_events(path, [{"type": "tool_call", "timestamp": _ts(moment)}])
    assert last_tool_call_timestamp(path) == moment


def test_legacy_shape_after_real_shape_wins_in_file_order(tmp_path: Path) -> None:
    """Last valid tool call in file order wins, whichever schema it uses."""
    path = tmp_path / "issue-1.events.jsonl"
    t1 = datetime.now(UTC) - timedelta(minutes=10)
    t2 = datetime.now(UTC) - timedelta(minutes=3)
    _write_events(
        path,
        [
            _assistant_tool_use(_ts(t1)),
            {"type": "tool_call", "timestamp": _ts(t2)},
        ],
    )
    assert last_tool_call_timestamp(path) == t2


def test_truncated_final_line_yields_last_complete_tool_call(tmp_path: Path) -> None:
    """A session still writing: a partial trailing line is skipped, not raised."""
    path = tmp_path / "issue-1.events.jsonl"
    t1 = datetime.now(UTC) - timedelta(minutes=5)
    _write_events(path, [_assistant_tool_use(_ts(t1))])
    with path.open("a", encoding="utf-8") as handle:
        partial = {
            "type": "assistant",
            "timestamp": _ts(datetime.now(UTC)),
            "message": {"content": [{"type": "tool_use"}]},
        }
        handle.write(json.dumps(partial)[:40])
    assert last_tool_call_timestamp(path) == t1


def test_non_json_lines_are_skipped(tmp_path: Path) -> None:
    """Stderr noise merged into the same file never breaks the read."""
    path = tmp_path / "issue-1.events.jsonl"
    t1 = datetime.now(UTC) - timedelta(minutes=5)
    lines = [json.dumps(_assistant_tool_use(_ts(t1))), "not json at all", "{partial"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert last_tool_call_timestamp(path) == t1


def test_missing_file_yields_none(tmp_path: Path) -> None:
    assert last_tool_call_timestamp(tmp_path / "absent.events.jsonl") is None


def test_file_with_no_tool_call_yields_none(tmp_path: Path) -> None:
    path = tmp_path / "issue-1.events.jsonl"
    _write_events(path, [_system_init(), _assistant_text(_ts(datetime.now(UTC)), "hi")])
    assert last_tool_call_timestamp(path) is None


def _claude_view(tmp_path: Path, started_at: datetime) -> WorkerView:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_file = sessions_dir / "issue-1.claude.log"
    log_file.write_text('{"type":"system","subtype":"init"}\n', encoding="utf-8")
    fresh = datetime.now(UTC)
    os.utime(log_file, (time.time(), fresh.timestamp()))
    return WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=started_at.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path=str(tmp_path / "worktree"),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )


def _classify(
    tmp_path: Path,
    started_at: datetime,
    tool_call_at: datetime | None,
    *,
    loop_kill: bool,
) -> WorkerHealth:
    view = _claude_view(tmp_path, started_at)
    if tool_call_at is not None:
        events_file = _events_path_from_log(Path(view.log_path))
        _write_events(events_file, [_assistant_tool_use(_ts(tool_call_at))])
        os.utime(events_file, (time.time(), tool_call_at.timestamp()))
    config = OrchestratorConfig(
        watchdog=WatchdogConfig(loop_kill=loop_kill, worktree_mtime_enabled=False)
    )
    now = datetime.now(UTC)
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        return classify_worker_health(view, config, now)


def test_loop_check_reads_real_tool_use_recent_not_slow_with_loop_kill_false(
    tmp_path: Path,
) -> None:
    """60-minute-old worker, log advancing, tool call 5 minutes ago: not SLOW."""
    health = _classify(
        tmp_path,
        datetime.now(UTC) - timedelta(minutes=60),
        datetime.now(UTC) - timedelta(minutes=5),
        loop_kill=False,
    )
    assert health != WorkerHealth.SLOW
    assert health != WorkerHealth.RUNAWAY


def test_loop_check_reads_real_tool_use_recent_not_slow_with_loop_kill_true(
    tmp_path: Path,
) -> None:
    """Same healthy shape under ``loop_kill: true``: not RUNAWAY."""
    health = _classify(
        tmp_path,
        datetime.now(UTC) - timedelta(minutes=60),
        datetime.now(UTC) - timedelta(minutes=5),
        loop_kill=True,
    )
    assert health != WorkerHealth.RUNAWAY
    assert health != WorkerHealth.SLOW


def test_loop_check_stale_tool_use_is_slow_with_loop_kill_false(
    tmp_path: Path,
) -> None:
    """Control: tool call 45 minutes ago (past 2x stall_minutes) is SLOW."""
    health = _classify(
        tmp_path,
        datetime.now(UTC) - timedelta(minutes=60),
        datetime.now(UTC) - timedelta(minutes=45),
        loop_kill=False,
    )
    assert health == WorkerHealth.SLOW


def test_loop_check_stale_tool_use_is_runaway_with_loop_kill_true(
    tmp_path: Path,
) -> None:
    """Control: the same stalled shape under ``loop_kill: true`` is RUNAWAY."""
    health = _classify(
        tmp_path,
        datetime.now(UTC) - timedelta(minutes=60),
        datetime.now(UTC) - timedelta(minutes=45),
        loop_kill=True,
    )
    assert health == WorkerHealth.RUNAWAY
