"""Issue #2010: headless claude-code worker permissions."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from charlie_work import claude_code, worker_fate
from charlie_work.claude_code import (
    _WORKER_COMMAND_TEMPLATE,
    PROMPTING_PERMISSION_MODES,
    _sidecar_path,
    worker_permission_denied,
)
from charlie_work.config import ClaudeCodeConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.doctor_worker_permissions import (
    _check_claude_worker_permissions,
    effective_permission_mode,
)

DENIAL = "I could not run the tests. If you approve command execution, I can finish these steps."

_classify_session_failure = partial(worker_fate.classify_for, "claude-code")


def _run_check(repo: Path, config: OrchestratorConfig) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    _check_claude_worker_permissions(lambda n, ok, d, **_: out.append((n, ok, d)), repo, config)
    return out


def _cc_config(command: tuple[str, ...] = ()) -> OrchestratorConfig:
    return replace(
        OrchestratorConfig(),
        worker=replace(WorkerRoleConfig(), harness="claude-code"),
        claude_code=ClaudeCodeConfig(command=command),
    )


def test_default_worker_template_is_not_a_prompting_mode() -> None:
    mode = effective_permission_mode(_WORKER_COMMAND_TEMPLATE)
    assert mode == "bypassPermissions"
    assert mode not in PROMPTING_PERMISSION_MODES
    assert claude_code._REVIEW_COMMAND_TEMPLATE[-1] == "plan"


def test_doctor_flags_prompting_mode_without_bash_allow(tmp_path: Path) -> None:
    cfg = _cc_config(("claude", "-p", "--permission-mode", "acceptEdits"))
    ((name, ok, detail),) = _run_check(tmp_path, cfg)
    assert not ok and "acceptEdits" in detail


def test_doctor_flags_equals_form_and_missing_flag(tmp_path: Path) -> None:
    assert not _run_check(tmp_path, _cc_config(("claude", "-p", "--permission-mode=plan")))[0][1]
    assert not _run_check(tmp_path, _cc_config(("claude", "-p")))[0][1]


def test_doctor_passes_default_bypass_and_bash_allow(tmp_path: Path) -> None:
    assert _run_check(tmp_path, _cc_config())[0][1]
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(git status:*)"]}}), encoding="utf-8"
    )
    cfg = _cc_config(("claude", "-p", "--permission-mode", "acceptEdits"))
    assert _run_check(tmp_path, cfg)[0][1]


def test_doctor_skips_non_claude_code_harness(tmp_path: Path) -> None:
    assert _run_check(tmp_path, OrchestratorConfig()) == []


def test_denial_tail_classified_permission_denied_not_blocked(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    log.write_text("edits done\n" + DENIAL, encoding="utf-8")
    assert _classify_session_failure(log) == ("permission_denied", None)


def _seed_sidecar(sessions: Path, log: Path) -> None:
    path = _sidecar_path(sessions, 7, "claude-code")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"log_path": str(log)}), encoding="utf-8")


def test_worker_permission_denied_from_log_or_detail(tmp_path: Path) -> None:
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(DENIAL, encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)
    assert worker_permission_denied(tmp_path / "none", 8, DENIAL)
    assert not worker_permission_denied(tmp_path / "none", 8, "cross-repo scope")
    log.write_text("all good", encoding="utf-8")
    assert not worker_permission_denied(sessions, 7)


# Issue #2814 rework round 3: the stream-json tee is the default, so the
# log-tail check must not read a ``tool_result``'s QUOTE of the denial
# signature as a denial — event bodies are excluded via
# ``failure_classifier.signature_match_text``, exactly as in the other
# log-tail signature matchers.


def _stream_json_line(top_level: dict) -> str:
    return json.dumps(top_level)


def _quoted_denial_stream_log(quoting_text: str) -> str:
    """A stream-json log tail whose last line is a huge ``user`` event
    carrying a ``tool_result`` that merely quotes a denial signature inside a
    >2KB log (the quote survives into the 2KB byte slice, and — because the
    slice cuts the line — matches only through the cut fragment before the
    fix)."""
    tool_result = _stream_json_line(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "x" * 3400 + quoting_text,
                    }
                ],
            },
        }
    )
    return _stream_json_line({"type": "system", "subtype": "init"}) + "\n" + tool_result + "\n"


def test_worker_permission_denied_stream_json_quote_not_denied(tmp_path: Path) -> None:
    """A tee'd stream-json tail quoting 'requires your approval' inside a
    ``tool_result`` (>2KB log, the quote survives the slice) is NOT a denial —
    the worker observed a denial, the CLI did not surface one."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(_quoted_denial_stream_log(DENIAL), encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert not worker_permission_denied(sessions, 7)


def test_worker_permission_denied_stream_json_stderr_denial_is_denied(
    tmp_path: Path,
) -> None:
    """Real-death control: the CLI's own denial surfaced as a plain (merged-
    stderr) line in the same stream-json log tail still reads as a denial."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(_quoted_denial_stream_log("") + DENIAL + "\n", encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)


# Issue #2814 rework round 4: the REAL shape of a headless denial on a tee'd
# log is not an error — the CLI ends the session reporting it was denied, with
# the terminal ``result`` event carrying ``is_error: false`` (and
# ``permission_denials``), and the assistant's final turn carrying the ask.
# The strict is_error filter made those denials invisible; both entry points
# (``worker_permission_denied`` and ``worker_fate.classify_for`` ->
# ``failure_classifier.classify_failure``) must detect them in permission
# mode, while a ``tool_result`` that merely quotes the phrase stays excluded.


def _real_denial_stream_log(*, carrier: str, pad: int = 0) -> str:
    """The real headless-denial stream shape, with the denial signature placed
    by ``carrier``: ``"assistant"`` (the final assistant turn), ``"result"``
    (the terminal ``is_error: false`` ``result`` event, with
    ``permission_denials``), or ``"tool_result"`` (a ``user`` tool_result that
    merely QUOTES the phrase — the negative-control carrier). ``pad`` pads the
    carrying text so its line exceeds 2KB (when it is the log's last line, the
    2KB slice then starts INSIDE it — the straddling-window variant)."""
    init = _stream_json_line({"type": "system", "subtype": "init"})
    now_iso = datetime.now(UTC).isoformat()
    assistant_text = "I tried to run the test suite."
    if carrier == "assistant":
        assistant_text = "x" * pad + DENIAL
    assistant = _stream_json_line(
        {
            "type": "assistant",
            "timestamp": now_iso,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        }
    )
    result_text = "Denied. I need approval to run the test commands."
    if carrier == "result":
        result_text = "x" * pad + DENIAL
    result = _stream_json_line(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": False,
            "result": result_text,
            "permission_denials": (
                [{"type": "tool_use", "name": "Bash", "input": {"command": "pytest"}}]
                if carrier == "result"
                else []
            ),
        }
    )
    if carrier == "tool_result":
        tool_result = _stream_json_line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "x" * pad + DENIAL,
                        }
                    ],
                },
            }
        )
        return f"{init}\n{tool_result}\n{result}\n"
    return f"{init}\n{assistant}\n{result}\n"


def test_worker_permission_denied_stream_json_real_result_denial_is_denied(
    tmp_path: Path,
) -> None:
    """Real-shape control (round 4): the terminal ``result`` event carries the
    ask with ``is_error: false`` and ``permission_denials`` — the real headless
    denial shape — and must read as a denial on a tee'd log."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(_real_denial_stream_log(carrier="result"), encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)


def test_worker_permission_denied_stream_json_straddling_result_denial_is_denied(
    tmp_path: Path,
) -> None:
    """Real-shape control, straddling-window variant: a >2KB earlier event
    straddles the 2KB slice (its cut fragment — a benign tool_result — is
    DROPPED, so its text cannot leak either way), and the terminal result
    event carrying the ask still reads as a denial for both entry points."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    init = _stream_json_line({"type": "system", "subtype": "init"})
    huge_benign = _stream_json_line(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x" * 3400}],
            },
        }
    )
    terminal = _stream_json_line(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": False,
            "result": DENIAL,
            "permission_denials": [{"type": "tool_use", "name": "Bash", "input": {}}],
        }
    )
    log_text = f"{init}\n{huge_benign}\n{terminal}\n"
    assert len(log_text) > 2048
    assert log_text[-2048:].startswith("x")
    assert log_text[-2048:].endswith(terminal + "\n")
    log.write_text(log_text, encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)
    assert _classify_session_failure(log) == ("permission_denied", None)


def test_worker_permission_denied_stream_json_tool_result_quote_stays_excluded(
    tmp_path: Path,
) -> None:
    """Negative control in permission mode: a ``tool_result`` (a ``user``
    event body) that merely quotes the denial phrase — even with real
    assistant/result lines present around it — must NOT manufacture a
    denial when the CLI itself never carried one."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    line = _stream_json_line(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "x" * 3400 + DENIAL,
                    }
                ],
            },
        }
    )
    clean_result = _stream_json_line(
        {"type": "result", "subtype": "success", "is_error": False, "result": "All done."}
    )
    log.write_text(f"{line}\n{clean_result}\n", encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert not worker_permission_denied(sessions, 7)
    assert _classify_session_failure(log) == (None, None)


def test_worker_permission_denied_stream_json_assistant_ask_is_denied(tmp_path: Path) -> None:
    """The ask riding the assistant's final turn (the other half of the real
    shape) is admitted in permission mode — and the same turn carrying the
    phrase does not re-arm a THROTTLE check (strict filter unchanged)."""
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(_real_denial_stream_log(carrier="assistant"), encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)
    assert _classify_session_failure(log) == ("permission_denied", None)
