"""Issue #2814 rework round 3: byte-sliced-tail signature matching.

Split out of the growing ``tests/test_claude_tee_default.py`` (file-size
ratchet): the direct ``failure_classifier.signature_match_text`` units and
the >2KB ``tool_result`` straddling the 2KB tail window for every
``signature_match_text`` consumer — ``classify_failure``,
``post_mortem._classify_worker_blocked_from_log_tail`` and the
``dead_worker_classification`` clean-exit throttle check — plus the tee
splitter's drop-unopenable-destination contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work import tee_splitter
from charlie_work.adapter_fate_profile import profile_for
from charlie_work.config import OrchestratorConfig, SignatureRule
from charlie_work.dead_worker_classification import (
    _terminal_is_throttle_error,
    classify_clean_exit_throttle,
)
from charlie_work.failure_classifier import (
    classify_failure,
    log_is_stream_json,
    signature_match_text,
)
from charlie_work.post_mortem import _classify_worker_blocked_from_log_tail


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


# --- signature_match_text: direct units ----------------------------------------


def test_signature_match_text_keeps_cli_own_error_output() -> None:
    """Direct units (round 3 review): ``system`` and ``rate_limit_event`` are
    the CLI's own output — kept; a ``result`` with ``is_error`` true is kept,
    false dropped."""
    signature = "Rate limit exceeded. Resets in 15 minutes."
    system_line = _dumps({"type": "system", "subtype": "init", "message": signature})
    rate_line = _dumps({"type": "rate_limit_event", "message": signature})
    kept = signature_match_text(f"{system_line}\n{rate_line}\n")
    assert system_line in kept
    assert rate_line in kept

    error_result = _dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": signature,
        }
    )
    assert signature_match_text(f"{error_result}\n") == error_result

    success_result = _dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": signature}
    )
    assert signature_match_text(f"{success_result}\n") == ""


def test_signature_match_text_drops_assistant_and_user_bodies() -> None:
    """A stream-json log whose ONLY signature-bearing lines are ``assistant``/
    ``user`` event bodies (e.g. a ``tool_result`` quoting a throttle message)
    matches as nothing — the CLI did not produce the text."""
    text = (
        _assistant_tool_use_event(minutes_ago=1)
        + "\n"
        + _tool_result_event("Rate limit exceeded. Resets in 15 minutes.")
        + "\n"
    )
    assert signature_match_text(text) == ""


def test_signature_match_text_plain_tail_passthrough() -> None:
    """A plain (non-stream-json) tail is returned whole, byte-identical —
    the shape filter changes nothing for it."""
    plain = "shim started\nRate limit exceeded. Resets in 15 minutes.\n"
    assert signature_match_text(plain) == plain


def test_signature_match_text_cut_drops_partial_first_line() -> None:
    """``cut=True`` drops the byte-slice's partial first line of a STREAM-JSON
    tail: the fragment of a >2KB event line parses as no event (so it would
    be treated as merged stderr and matched raw), and its surviving half can
    carry a whole signature string. Control: the same tail without the cut
    flag still leaks ``signature`` (the pre-round-3 behaviour the flag
    closes). Round 4: the drop applies ONLY to a stream-json tail — a plain
    tail is returned whole even when cut (the tee-off kill-switch promise:
    plain-text matching is byte-identical to the pre-rework behaviour)."""
    signature = "Rate limit exceeded. Resets in 15 minutes."
    fragment = signature + " (the 2KB slice cut this event line)"
    tail = f"{fragment}\n{_dumps({'type': 'result', 'is_error': True, 'result': 'fatal line'})}\n"
    assert signature in signature_match_text(tail, stream_json=True)
    assert signature_match_text(tail, cut=True, stream_json=True) == _dumps(
        {"type": "result", "is_error": True, "result": "fatal line"}
    )

    plain = f"{fragment}\nWorker still healthy.\n"
    assert signature_match_text(plain, cut=True) == plain
    assert signature_match_text(plain) == plain


def test_classify_failure_plain_last_line_over_2kb_still_classifies(tmp_path: Path) -> None:
    """Round 4 regression (tee-off kill switch): a PLAIN-text log whose LAST
    line is longer than 2KB and carries the signature — the 2KB slice starts
    inside that line, and the un-cut behaviour must still match through the
    raw slice, byte-identical to the pre-rework classifier. Both a throttle
    and a permission signature variant."""
    throttle = ("x" * 2200) + "Rate limit exceeded. Resets in 15 minutes."
    throttle_log = tmp_path / "plain-throttle.log"
    throttle_log.write_text(f"shim started\n{throttle}", encoding="utf-8")
    kind, until = classify_failure(
        throttle_log, OrchestratorConfig().runtime.throttle_error_markers
    )
    assert kind == "rate_limited"
    assert until is not None

    denial = ("x" * 2200) + DENIAL_TEXT
    denial_log = tmp_path / "plain-denial.log"
    denial_log.write_text(f"shim started\n{denial}", encoding="utf-8")
    kind, until = classify_failure(denial_log, headless_permission_detection=True)
    assert kind == "permission_denied"
    assert until is None


DENIAL_TEXT = (
    "I could not run the tests. If you approve command execution, I can finish these steps."
)


# --- the >2KB tool_result straddling the 2KB window -----------------------------


def _straddle_log(tmp_path: Path, tail_line: str, signature: str) -> Path:
    """A stream-json log whose total length exceeds 2KB and whose ``tail_line``
    (the last line) makes the 2KB slice start INSIDE the previous line — a
    >2KB ``tool_result`` quoting ``signature`` straddles the window, so the
    slice's first line is a partial event fragment (the shape is stream-json
    by the log head, not the window)."""
    log_text = _test_straddle_text(tail_line, signature)
    path = tmp_path / "straddle.log"
    path.write_text(log_text, encoding="utf-8")
    text = log_text[-2048:]
    # Guard the preconditions: the log ends past the 2KB slice with the slice's
    # first line CUT (a partial fragment, not the whole tool_result line), and
    # the shape is stream-json.
    assert len(text) < len(log_text)
    assert text.startswith("x")
    assert signature in text
    assert log_is_stream_json(log_text)
    return path


def test_classify_failure_large_tool_result_straddling_window_not_failure(
    tmp_path: Path,
) -> None:
    """Round 3 regression (issue #2814 rework): a ~3.5KB ``tool_result``
    quoting 'Rate limit exceeded. Resets in 15 minutes' STRADDLES the 2KB tail
    slice — before the fix, the byte fragment's surviving half matched the
    signature (verified loss: the tail's cut first line was treated as merged
    stderr). With the cut flag the fragment is dropped and the trailing
    non-error ``result`` event means NO failure."""
    throttled = OrchestratorConfig().runtime.throttle_error_markers
    quoted = _straddle_log(
        tmp_path,
        _dumps({"type": "result", "subtype": "success", "is_error": False, "result": "All done."}),
        "Rate limit exceeded. Resets in 15 minutes.",
    )
    assert classify_failure(quoted, throttled) == (None, None)


def test_classify_failure_straddle_window_error_result_control(tmp_path: Path) -> None:
    """Positive control for the straddling window: a real CLI death — an error
    ``result`` event — still classifies as ``rate_limited`` with a 15-minute
    stated reset even when a >2KB tool_result straddles the 2KB slice."""
    throttled = OrchestratorConfig().runtime.throttle_error_markers
    death = _straddle_log(
        tmp_path,
        _dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "Rate limit exceeded. Resets in 15 minutes",
            }
        ),
        "",
    )
    kind, throttled_until = classify_failure(death, throttled)
    assert kind == "rate_limited"
    assert throttled_until is not None


def test_classify_failure_straddle_window_stderr_control(tmp_path: Path) -> None:
    """Positive control: a merged-stderr line carrying the throttle still
    classifies through a straddling window — non-event lines are kept by
    ``signature_match_text`` even when the slice cut the event before it."""
    throttled = OrchestratorConfig().runtime.throttle_error_markers
    death = _straddle_log(
        tmp_path,
        "Error: Rate limit exceeded. Resets in 15 minutes",
        "Rate limit exceeded. Resets in 15 minutes",
    )
    kind, throttled_until = classify_failure(death, throttled)
    assert kind == "rate_limited"
    assert throttled_until is not None


def test_post_mortem_log_tail_large_tool_result_straddling_not_blocked(
    tmp_path: Path,
) -> None:
    """Round 3 regression: the post-mortem log-tail fallback with a >2KB
    ``tool_result`` quoting a block signature straddling the 2KB slice —
    the cut fragment must not classify ``worker_blocked``. The trailing
    non-event (stderr) line carrying the signature is the positive control."""
    rules = (SignatureRule(pattern=r"Tool blocked:", kind="worker_blocked"),)
    signature = "Tool blocked: push not allowed"

    quoted = _straddle_log(
        tmp_path,
        _dumps({"type": "result", "subtype": "success", "is_error": False, "result": "All done."}),
        signature,
    )
    assert _classify_worker_blocked_from_log_tail(quoted, rules) is False

    hit = _straddle_log(tmp_path, f"{signature}\n", signature)
    assert _classify_worker_blocked_from_log_tail(hit, rules) is True

    plain = tmp_path / "hit-plain.log"
    plain.write_text(f"shim\n{signature}\n", encoding="utf-8")
    assert _classify_worker_blocked_from_log_tail(hit, rules) == (
        _classify_worker_blocked_from_log_tail(plain, rules)
    )


# --- clean-exit throttle check: cut window + stream-json units -------------------


def test_terminal_is_throttle_error_cut_and_stream_json_quote_not_throttle() -> None:
    """Direct unit (round 3 review; this is the test that fails if the
    ``dead_worker_classification`` ``signature_match_text`` routing is
    reverted): a terminal window whose first line is a cut fragment of a huge
    ``user`` event quoting a throttle signature, followed by a non-error
    ``result`` event, is NOT a throttle; the same window ending on a REAL
    stderr throttle line still is."""
    profile = profile_for("claude-code")
    signature = "Rate limit exceeded. Resets in 15 minutes."
    fragment = ("x" * 2048) + signature

    quoted_window = (
        f"{fragment}\n{_dumps({'type': 'result', 'is_error': False, 'result': 'all done'})}"
    )
    assert (
        _terminal_is_throttle_error(
            quoted_window, profile, OrchestratorConfig(), cut=True, stream_json=True
        )
        is False
    )

    stderr_window = f"{fragment}\nError: {signature}"
    assert (
        _terminal_is_throttle_error(
            stderr_window, profile, OrchestratorConfig(), cut=True, stream_json=True
        )
        is True
    )


def _clean_exit_sessions(tmp_path: Path, log_text: str) -> tuple[Path, Path]:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = sessions_dir / "issue-11.claude.log"
    log_path.write_text(log_text, encoding="utf-8")
    (sessions_dir / "issue-11.claude.json").write_text(
        _dumps(
            {
                "issue_number": 11,
                "branch": "agent/issue-11",
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
    return sessions_dir, log_path


def test_clean_exit_throttle_large_tool_result_straddling_window_none(
    tmp_path: Path,
) -> None:
    """``classify_clean_exit_throttle`` end-to-end round 3 regression: a log
    whose >2KB ``tool_result`` quote straddles the 2KB slice and ends on a
    NON-error ``result`` event is not a throttle death despite the quote."""
    sessions_dir, _ = _clean_exit_sessions(
        tmp_path,
        _test_straddle_text(
            _dumps(
                {"type": "result", "subtype": "success", "is_error": False, "result": "All done."}
            ),
            "Rate limit exceeded. Resets in 15 minutes",
        ),
    )
    assert classify_clean_exit_throttle({}, sessions_dir, 11, OrchestratorConfig()) is None


def test_clean_exit_throttle_stream_stderr_terminal_control(tmp_path: Path) -> None:
    """Positive control: the same straddling log cut by a REAL terminal
    stderr throttle line classifies a clean-exit ``rate_limited`` death."""
    sessions_dir, _ = _clean_exit_sessions(
        tmp_path,
        _test_straddle_text("Error: Rate limit exceeded. Resets in 15 minutes", ""),
    )
    resolved = classify_clean_exit_throttle({}, sessions_dir, 11, OrchestratorConfig())
    assert resolved is not None
    adapter_kind, evidence = resolved
    assert adapter_kind == "claude-code"
    assert evidence.kind == "rate_limited"


def _test_straddle_text(tail_line: str, signature: str) -> str:
    """The >-2KB text ``_straddle_log`` writes (shared by the clean-exit tests)."""
    big_result_line = _dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "x" * 3400 + signature,
                    }
                ],
            },
        }
    )
    return (
        _dumps({"type": "system", "subtype": "init"})
        + "\n"
        + big_result_line
        + "\n"
        + tail_line
        + "\n"
    )


# --- tee splitter hardening ------------------------------------------------------


def test_tee_splitter_drops_unopenable_destination_and_keeps_draining(
    tmp_path: Path,
) -> None:
    """Round 3: a destination that cannot be opened (events inside a missing
    directory) is DROPPED — the splitter still drains stdin to EOF, exits 0,
    and the openable destination receives the whole stream."""
    log = tmp_path / "log.txt"
    events = tmp_path / "missing-dir" / "events.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(tee_splitter.__file__)),
            str(log),
            str(events),
        ],
        input="line-a\nline-b\n",
        encoding="utf-8",
        check=False,
        timeout=30,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["line-a", "line-b"]
    assert not events.exists()
