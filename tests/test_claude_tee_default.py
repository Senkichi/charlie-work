"""Issue #2814: the stream-json tee is the default for claude-code sessions.

Shipped default ON with ``tee_stream_json: false`` in config as the kill
switch, so the watchdog has a live activity signal while a worker's first
minutes are spent reading and planning (a plain ``claude -p`` worker log
stays 0 bytes until the session ends).

The module covers:

* the defaults on the config section, the adapter settings, and the lambda
  launch command they build through the real dispatch path;
* ``classify_worker_health`` distinguishing a 30-minute-old default-tee
  worker (fresh ``tool_use`` event, untouched worktree) from the same
  worker with the tee off, which IS stalled;
* every log reader of a claude-code worker log being stream-json-aware on a
  tee'd stream-json log (issue #2814 rework): signal 2's terminal-marker
  scan, failure classification, the post-mortem log-tail fallback — a
  ``tool_result`` merely *quoting* a signature never matches, while a real
  CLI death (an error ``result`` event or a merged-stderr line) still
  classifies the same as the equivalent plain-text log — and completion
  detection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work import claude_code, tee_splitter
from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions
from charlie_work.config import (
    ClaudeCodeConfig,
    OrchestratorConfig,
    PostMortemConfig,
    SignatureRule,
    WatchdogConfig,
    load_config,
)
from charlie_work.claude_code import update_worker_record_with_failure_classification
from charlie_work.failure_classifier import (
    classify_failure,
)
from charlie_work.post_mortem import _classify_worker_blocked_from_log_tail
from charlie_work.process_utils import worker_terminal_status_path
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
    real_activity_probe_for,
)
from charlie_work.worktree import WorktreeInfo


def _dumps(value: object) -> str:
    return json.dumps(value)


def _tool_result_event(text: str) -> str:
    """A real-shape stream-json ``user`` event carrying a ``tool_result`` block
    (``claude --output-format stream-json`` tool output; see
    ``tests/test_claude_loop_check_schema.py`` for the same real schema)."""
    return _dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": text}],
            },
        }
    )


def _assistant_tool_use_event(minutes_ago: int) -> str:
    """A real-shape stream-json ``assistant`` event carrying a ``tool_use`` block."""
    timestamp = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    return _dumps(
        {
            "type": "assistant",
            "timestamp": timestamp,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "src/x.py"}}
                ],
            },
        }
    )


def _backdate(path: Path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


# --- defaults and the launch command ----------------------------------------


def test_claude_code_settings_default_tee_on() -> None:
    assert ClaudeCodeConfig().tee_stream_json is True
    assert AdapterSettings().tee_stream_json is True


def test_config_tee_stream_json_false_is_the_kill_switch(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
claude_code:
  tee_stream_json: false
""",
        encoding="utf-8",
    )
    assert load_config(config_file).claude_code.tee_stream_json is False


def test_default_settings_launch_command_carry_stream_json_flags(
    tmp_path: Path, monkeypatch
) -> None:
    """Dispatching through the real claude-code adapter lane with default
    ``AdapterSettings``: the launch argv carries --output-format stream-json
    and --verbose, and the events sidecar opens at launch time."""
    result, _, events_path = _dispatch_claude(tmp_path, monkeypatch, tee_stream_json=True)

    assert result.ok, result.error
    assert "--output-format" in result.command
    assert "stream-json" in result.command
    assert "--verbose" in result.command
    assert result.pid is not None
    assert _await_events_sidecar(events_path)


def test_tee_off_config_launches_without_stream_json_flags(tmp_path: Path, monkeypatch) -> None:
    """Control: settings built from a config with ``tee_stream_json: false``
    build a command with neither flag and write no events.jsonl — the exact
    pre-#2814 behaviour."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
claude_code:
  tee_stream_json: false
""",
        encoding="utf-8",
    )
    loaded_tee = load_config(config_file).claude_code.tee_stream_json
    assert loaded_tee is False

    result, sessions_dir, events_path = _dispatch_claude(
        tmp_path, monkeypatch, tee_stream_json=loaded_tee
    )

    assert result.ok, result.error
    assert "--output-format" not in result.command
    assert "stream-json" not in result.command
    assert "--verbose" not in result.command
    assert not events_path.exists()
    assert sessions_dir is not None


def _await_events_sidecar(events_path: Path, timeout_seconds: float = 10.0) -> bool:
    """Poll until the events sidecar exists. The launcher pre-creates both
    tee destinations synchronously (round 3), so the path appears at launch
    itself; the poll just absorbs any concurrency between the launch call
    returning and the file being observable on this host."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if events_path.exists():
            return True
        time.sleep(0.05)
    return False


def _dispatch_claude(
    tmp_path: Path, monkeypatch, *, tee_stream_json: bool
) -> tuple[object, Path | None, Path]:
    """One real ``_dispatch_claude_code`` dispatch with the default worker
    command template resolved to a test-double binary (a stdin-echoing python
    script), so ``result.command`` is the exact argv the default settings build."""
    script_path = tmp_path / "fake_claude.py"
    script_path.write_text("import sys\nsys.stdin.read()\nprint('ok')\n", encoding="utf-8")
    script_argv = (sys.executable, str(script_path))

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    def fake_create_worktree(repo_root, branch, **kwargs):
        worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
        worktree_path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
    # Default command template, resolved to the test double: launch never sees
    # a "is claude installed here" question — only what argv it builds.
    monkeypatch.setattr(claude_code, "_WORKER_COMMAND_TEMPLATE", script_argv)

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Do the thing.", encoding="utf-8")

    settings = AdapterSettings(
        adapter="claude-code", sessions_dir=sessions_dir, tee_stream_json=tee_stream_json
    )
    request = SessionRequest(
        issue_number=7,
        issue_title="t",
        prompt_path=prompt_path,
        branch_name="agent/issue-7-fix",
    )
    results = dispatch_sessions(
        tmp_path, tmp_path / "manifest.json", tmp_path / "results.json", settings, [request]
    )
    assert len(results) == 1
    events_path = sessions_dir / "issue-7.events.jsonl"
    return results[0], sessions_dir, events_path


# --- the watchdog: a default-tee worker is not reaped while it plans ---------


def _thirty_minute_worker(tmp_path: Path, *, tee_on: bool) -> WorkerView:
    """A 30-minute-old claude-code worker: no file in the worktree changed
    after its start (checked-out files backdated before start), and log set up
    per tee. With tee ON the log and events.jsonl carry a ``tool_use`` event 2
    minutes old while both files' mtimes are deliberately frozen at launch so
    signal 3 fires and has to be vetoed by the events source. With tee OFF the
    log stays 0 bytes since launch (``claude -p`` prints only its final
    result) and no events.jsonl exists — the pre-#2814 shape."""
    now = datetime.now(UTC)
    started_at = now - timedelta(minutes=30)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = sessions_dir / "issue-1.claude.log"
    log_path.write_text("", encoding="utf-8")
    _backdate(log_path, started_at)

    if tee_on:
        tool_use_event = _assistant_tool_use_event(minutes_ago=2) + "\n"
        log_path.write_text(tool_use_event, encoding="utf-8")
        events_path = sessions_dir / "issue-1.events.jsonl"
        events_path.write_text(tool_use_event, encoding="utf-8")
        _backdate(log_path, started_at)
        _backdate(events_path, started_at)

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    checked_out = worktree / "checked_out.py"
    checked_out.write_text("x", encoding="utf-8")
    _backdate(checked_out, started_at - timedelta(minutes=5))

    return WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=4242,
        started_at=started_at.isoformat(),
        process_start_time=started_at.timestamp(),
        log_path=str(log_path),
        worktree_path=str(worktree),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )


def _probe_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        watchdog=WatchdogConfig(stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )


def test_default_tee_worker_with_fresh_tool_use_not_stalled(tmp_path: Path) -> None:
    """Acceptance: 30 minutes after start, no post-start worktree writes, but
    the tee'd log and events.jsonl got a tool_use event 2 minutes ago — not
    STALLED (the events source vetoes signal 3)."""
    view, config, now = (
        _thirty_minute_worker(tmp_path, tee_on=True),
        _probe_config(tmp_path),
        datetime.now(UTC),
    )

    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
    assert health == WorkerHealth.HEALTHY


def test_tee_off_same_worker_is_stalled(tmp_path: Path) -> None:
    """Control: the same 30-minute worker with tee OFF (0-byte launch-time
    log, no events.jsonl, untouched worktree) IS STALLED."""
    view, config, now = (
        _thirty_minute_worker(tmp_path, tee_on=False),
        _probe_config(tmp_path),
        datetime.now(UTC),
    )

    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
    assert health == WorkerHealth.STALLED


# --- signal 2: stream-json event lines never quote terminal errors ------------


MARKER = "Error: A tool was rejected"


def _marker_view(tmp_path: Path, log_text: str) -> WorkerView:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    log_path = sessions_dir / "issue-2.claude.log"
    log_path.write_text(log_text, encoding="utf-8")
    return WorkerView(
        adapter_kind="claude-code",
        issue_number=2,
        repo_key="",
        pid=99,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=time.time(),
        log_path=str(log_path),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )


def test_signal2_streamjson_tool_result_quote_not_terminal(tmp_path: Path) -> None:
    """Acceptance: a stream-json log whose last line is a ``tool_result`` that
    quotes a terminal marker is not a terminal error — the worker observed an
    error, the CLI did not die mid-session."""
    log_text = (
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + _tool_result_event(f"hook output: {MARKER}")
        + "\n"
    )
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        health = classify_worker_health(
            _marker_view(tmp_path, log_text), OrchestratorConfig(), datetime.now(UTC)
        )
    assert health == WorkerHealth.HEALTHY


def test_signal2_plain_text_last_line_marker_is_dead(tmp_path: Path) -> None:
    """The equivalent plain-text log (the marker is the CLI's own last line)
    stays terminal: signal 2 keeps killing a wedged --print session."""
    log_text = f"passed shim\n{MARKER}\n"
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        health = classify_worker_health(
            _marker_view(tmp_path, log_text), OrchestratorConfig(), datetime.now(UTC)
        )
    assert health == WorkerHealth.DEAD


def test_signal2_streamjson_trailing_stderr_marker_is_dead(tmp_path: Path) -> None:
    """stderr=STDOUT: a stream-json log routinely ends on a plain stderr line;
    a real fatal marker there still kills exactly as in plain text."""
    log_text = _dumps({"type": "system", "subtype": "init"}) + f"\n{MARKER}\n"
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        health = classify_worker_health(
            _marker_view(tmp_path, log_text), OrchestratorConfig(), datetime.now(UTC)
        )
    assert health == WorkerHealth.DEAD


def test_signal2_plain_text_clean_log_stays_healthy(tmp_path: Path) -> None:
    """Parity guard: a clean plain log was healthy before and stays healthy."""
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        health = classify_worker_health(
            _marker_view(tmp_path, "Working...\nStill working\n"),
            OrchestratorConfig(),
            datetime.now(UTC),
        )
    assert health == WorkerHealth.HEALTHY


# --- failure classification: stream-json parity -------------------------------


def test_failure_classification_stream_json_matches_plain_text(tmp_path: Path) -> None:
    """``classify_failure`` is stream-json-aware (issue #2814 rework): a
    tee'd stream-json log where a ``tool_result`` merely *quotes* a throttle
    signature classifies as no failure — the worker observed an error, the
    CLI did not die (the issue's acceptance example). The equivalent plain
    text when the CLI itself dies is still classified, in either shape: the
    merged-stderr last line in the plain-text log and a real CLI error
    (``result`` event with ``is_error``) in the stream-json log answer with
    the same ``rate_limited`` kind."""
    throttle_markers = OrchestratorConfig().runtime.throttle_error_markers
    signature = "Rate limit exceeded. Resets in 15 minutes."

    quoted = tmp_path / "quoted-stream.log"
    quoted.write_text(
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + _tool_result_event(signature)
        + "\n",
        encoding="utf-8",
    )
    assert classify_failure(quoted, throttle_markers) == (None, None)

    plain = tmp_path / "plain.log"
    plain.write_text(f"shim\n{signature}\n", encoding="utf-8")
    stream_death = tmp_path / "stream-death.log"
    stream_death.write_text(
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + _dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": signature,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plain_kind, plain_throttled = classify_failure(plain, throttle_markers)
    stream_kind, stream_throttled = classify_failure(stream_death, throttle_markers)
    assert plain_kind == "rate_limited"
    assert stream_kind == "rate_limited"
    assert stream_kind == plain_kind
    assert plain_throttled is not None
    assert stream_throttled is not None


def test_failure_classification_clean_stream_json_log_not_failure(tmp_path: Path) -> None:
    """A tee'd worker log ending in a successful ``result`` event classifies
    as no failure — the same answer as the equivalent plain-text log (a
    completed session still counts as completed, nowhere a failure)."""
    markers = OrchestratorConfig().runtime.throttle_error_markers

    stream = tmp_path / "done.log"
    stream.write_text(
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + _dumps(
            {"type": "result", "subtype": "success", "is_error": False, "result": "All done."}
        )
        + "\n",
        encoding="utf-8",
    )
    plain = tmp_path / "done-plain.log"
    plain.write_text("All done.\n", encoding="utf-8")

    assert classify_failure(stream, markers) == (None, None)
    assert classify_failure(stream, markers) == classify_failure(plain, markers)


# --- post-mortem log-tail fallback: stream-json parity -------------------------


def test_post_mortem_log_tail_worker_blocked_stream_json_parity(tmp_path: Path) -> None:
    """``_classify_worker_blocked_from_log_tail`` is stream-json-aware (issue
    #2814 rework): a tee'd stream-json log where a ``tool_result`` merely
    *quotes* a block signature is NOT worker_blocked (the issue's acceptance
    example) — the worker observed a block, the CLI did not surface one. A
    real CLI surfacing of the block is still classified in either shape: a
    plain (non-event) line in the stream-json log (its own prose) and the
    equivalent plain-text log answer with the same True."""
    rules = (SignatureRule(pattern=r"Tool blocked:", kind="worker_blocked"),)

    plain_hit = tmp_path / "hit-plain.log"
    plain_hit.write_text("shim\nTool blocked: push not allowed\n", encoding="utf-8")
    stream_hit = tmp_path / "hit-stream.log"
    stream_hit.write_text(
        _dumps({"type": "system", "subtype": "init"}) + "\n" + "Tool blocked: push not allowed\n",
        encoding="utf-8",
    )
    stream_quote = tmp_path / "quote-stream.log"
    stream_quote.write_text(
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + _tool_result_event("Tool blocked: push not allowed")
        + "\n",
        encoding="utf-8",
    )
    plain_miss = tmp_path / "miss-plain.log"
    plain_miss.write_text("shim\nall good\n", encoding="utf-8")
    stream_miss = tmp_path / "miss-stream.log"
    stream_miss.write_text(_dumps({"type": "system", "subtype": "init"}) + "\n", encoding="utf-8")

    assert _classify_worker_blocked_from_log_tail(stream_quote, rules) is False
    assert _classify_worker_blocked_from_log_tail(stream_hit, rules) is True
    assert _classify_worker_blocked_from_log_tail(stream_hit, rules) == (
        _classify_worker_blocked_from_log_tail(plain_hit, rules)
    )
    assert _classify_worker_blocked_from_log_tail(stream_miss, rules) is False
    assert _classify_worker_blocked_from_log_tail(stream_miss, rules) == (
        _classify_worker_blocked_from_log_tail(plain_miss, rules)
    )


# --- completion detection -----------------------------------------------------


STREAM_COMPLETED_LOG = (
    _dumps({"type": "system", "subtype": "init"})
    + "\n"
    + _dumps({"type": "result", "subtype": "success", "is_error": False, "result": "All done."})
    + "\n"
)
PLAIN_COMPLETED_LOG = "All done.\n"


def _sidecar(tmp_path: Path, sessions_dir: Path, log_path: Path) -> Path:
    sidecar = sessions_dir / "issue-9.claude.json"
    sidecar.write_text(
        _dumps(
            {
                "issue_number": 9,
                "branch": "agent/issue-9",
                "worktree_path": "",
                "prompt_path": "",
                "command": [],
                "pid": 5,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_completed_session_stream_json_log_records_no_failure(tmp_path: Path) -> None:
    """Completion detection: a tee'd stream-json log whose last line is a
    successful ``result`` event — the terminal record proves this exact pid
    exited 0 with an outcome, so the log tail (which here carries a throttle
    phrase inside the result prose) is never failure-classified. The
    equivalent plain-text completed log answers identically."""
    config = OrchestratorConfig()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = sessions_dir / "completed.claude.log"
    record_path = worker_terminal_status_path(sessions_dir, 9, "claude")

    for log_text in (STREAM_COMPLETED_LOG, PLAIN_COMPLETED_LOG):
        # A throttle phrase even sits inside the result prose — completion
        # detection must skip the tail classification all the same.
        marker_log = log_text.replace("All done.", "Rate limit: all done.")
        log_path.write_text(marker_log, encoding="utf-8")
        sidecar = _sidecar(tmp_path, sessions_dir, log_path)
        record_path.write_text(
            _dumps(
                {
                    "pid": 5,
                    "exit_code": 0,
                    "started_at": datetime.now(UTC).isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": 1.0,
                    "worker_outcome": {"outcome": "approved_pr_created"},
                }
            ),
            encoding="utf-8",
        )

        kind, throttled = update_worker_record_with_failure_classification(
            sessions_dir, 9, config=config
        )

        assert (kind, throttled) == (None, None)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload.get("failure_kind") is None
        # Reset for the second log shape.
        record_path.unlink(missing_ok=True)


# --- the tee survives the launching supervisor exiting ------------------------
#
# Recorded experiment (issue #2814 rework): the old tee drained a
# ``subprocess.PIPE`` with a daemon thread INSIDE the launching supervisor.
# Killing the launcher after 0.4s while the worker printed every 50ms ended
# the worker's 7th line with ``OSError: [Errno 22] Invalid argument``
# (Windows broken pipe; POSIX would raise BrokenPipeError). Both the
# positive and the negative control below reproduce that experiment as
# tests: launcher-thread mode loses output and errors the worker;
# splitter-child mode (what ``launch_claude_worker`` now runs) keeps the
# worker alive and both files complete.


# --- the tee splitter child itself ---------------------------------------------


def test_tee_splitter_script_writes_all_lines_to_both_files(tmp_path: Path) -> None:
    """Unit contract for ``tee_splitter.py`` standalone: every stdin line is
    flushed to both destinations, byte-for-byte."""
    log = tmp_path / "log.txt"
    events = tmp_path / "events.txt"
    subprocess.run(
        [
            sys.executable,
            str(Path(tee_splitter.__file__)),
            str(log),
            str(events),
        ],
        input="line-a\nline-b\nline-c\n",
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    assert log.read_text(encoding="utf-8").splitlines() == ["line-a", "line-b", "line-c"]
    assert log.read_bytes() == events.read_bytes()


def _run_tee_survival_experiment(tmp_path: Path, *, mode: str) -> tuple[str, str]:
    """One launcher child process that spawns the worker (and, in splitter
    mode, the tee splitter), then exits 0.4s in. The worker prints a line
    every 50ms for 3s and records every stdout failure to a status file.
    Returns (worker status text, launcher log text)."""
    launcher = tmp_path / "exp_launcher.py"
    worker = tmp_path / "exp_worker.py"
    splitter_path = Path(tee_splitter.__file__)

    worker.write_text(
        textwrap.dedent(
            """
            import sys
            import time

            status = open(sys.argv[1], "w", encoding="utf-8")
            status.write("start\\n")
            status.flush()
            n = 0
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    sys.stdout.write(f"line {n} {time.time():.2f}\\n")
                    sys.stdout.flush()
                    status.write(f"wrote {n + 1}\\n")
                    status.flush()
                    n += 1
                except BaseException as exc:
                    status.write(f"ERROR {type(exc).__name__}: {exc}\\n")
                    status.flush()
                    break
                time.sleep(0.05)
            status.write(f"final lines {n}\\n")
            """
        ),
        encoding="utf-8",
    )
    launcher.write_text(
        textwrap.dedent(
            f"""
            import os
            import subprocess
            import sys
            import threading
            import time

            here = {str(tmp_path)!r}
            mode = {mode!r}
            worker_argv = [sys.executable, {str(worker)!r}, {str(tmp_path / "status.txt")!r}]

            splitter = None
            if mode == "splitter":
                splitter = subprocess.Popen(
                    [sys.executable, {str(splitter_path)!r},
                     {str(tmp_path / "log.txt")!r}, {str(tmp_path / "events.txt")!r}],
                    stdin=subprocess.PIPE,
                )
                proc_argv = [*worker_argv]
                stdout = splitter.stdin
            else:
                proc_argv = [*worker_argv]
                stdout = subprocess.PIPE

            proc = subprocess.Popen(proc_argv, stdout=stdout, stderr=subprocess.STDOUT, text=True)

            if mode == "thread":
                def drain():
                    try:
                        for _ in proc.stdout:
                            pass
                    except Exception:
                        pass

                threading.Thread(target=drain, daemon=True).start()
            if splitter is not None:
                splitter.stdin.close()

            time.sleep(0.4)
            os._exit(0)
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(launcher)],
        check=False,
        timeout=30,
        capture_output=True,
    )
    status = tmp_path / "status.txt"
    deadline = time.monotonic() + 10.0
    status_text = ""
    while time.monotonic() < deadline:
        if status.exists() and "final" in status.read_text(encoding="utf-8"):
            break
        time.sleep(0.1)
    status_text = status.read_text(encoding="utf-8") if status.exists() else ""
    log = (
        (tmp_path / "log.txt").read_text(encoding="utf-8")
        if (tmp_path / "log.txt").exists()
        else ""
    )
    return status_text, log


def test_tee_splitter_child_worker_survives_launcher_exit(tmp_path: Path) -> None:
    """Mitigation (issue #2814 rework): with the tee running as a splitter
    CHILD process, a tee'd worker survives the launching supervisor exiting —
    it keeps printing for its whole window with no stdout error, and both the
    log and the events sidecar carry every line."""
    status_text, log = _run_tee_survival_experiment(tmp_path, mode="splitter")

    assert "ERROR" not in status_text, status_text
    assert "final lines" in status_text, status_text
    # Contract-level threshold (round 3 review): the launcher exits at 0.4s
    # (≈8 lines printed by then) while the worker keeps running to 3s
    # (≈59 lines max) — asserting close to the max flakes under -n auto
    # scheduler jitter, so pin well above the pre-exit count but below max.
    assert log.count("line ") >= 20, log


def test_tee_launcher_thread_pipe_worker_dies_on_launcher_exit(tmp_path: Path) -> None:
    """Negative control / recorded experiment: with the tee drained by a
    daemon thread holding ``subprocess.PIPE`` in the launcher (the pre-rework
    shape), the launching supervisor's exit breaks the pipe and the tee'd
    worker's stdout errors out within the same window."""
    status_text, _ = _run_tee_survival_experiment(tmp_path, mode="thread")

    assert "ERROR" in status_text, status_text
