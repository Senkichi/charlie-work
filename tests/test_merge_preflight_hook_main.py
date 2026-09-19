"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``main()`` stdin protocol (fail-open rc=0, deny payload on
merge-shaped internals failure), ``_run_merge_check``'s
``charlie_work.cli.main`` delegation, and the
``.claude/settings.json`` wiring that activates the hook.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# main() -- stdin protocol
# ---------------------------------------------------------------------------


def _run_main(
    monkeypatch: pytest.MonkeyPatch, stdin_text: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(stdin_text))
    rc = hook.main()
    captured = capsys.readouterr()
    return rc, captured.out


def test_main_malformed_json_rc0_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = _run_main(monkeypatch, "{not json", capsys)
    assert rc == 0
    assert out == ""


def test_main_benign_bash_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out == ""


def test_main_merge_shaped_bash_decide_raises_denies_fail_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("registry blew up")

    monkeypatch.setattr(hook, "_decide", _boom)
    payload = json.dumps(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 1 --squash"}}
    )
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out.strip() != ""
    decision = json.loads(out)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "RuntimeError" in decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_non_merge_bash_internals_raise_rc0_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("should not matter, command is not merge-shaped")

    monkeypatch.setattr(hook, "_decide", _boom)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# _run_merge_check
# ---------------------------------------------------------------------------


class _FakeCliModule:
    """Stand-in for ``charlie_work.cli`` with a patchable ``main`` attribute."""


def test_run_merge_check_rc_zero_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import charlie_work.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv: 0)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is True
    assert isinstance(detail, str)


def test_run_merge_check_rc_nonzero_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv: 2)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False


def test_run_merge_check_system_exit_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    def _raise_system_exit(argv: list[str]) -> int:
        raise SystemExit(3)

    monkeypatch.setattr(cli_module, "main", _raise_system_exit)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False


def test_run_merge_check_runtime_error_denies_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    def _raise_runtime_error(argv: list[str]) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "main", _raise_runtime_error)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False
    assert "RuntimeError" in detail


# ---------------------------------------------------------------------------
# .claude/settings.json wiring (round-3 #1195): the hook activation itself
# ---------------------------------------------------------------------------


def test_claude_settings_wires_merge_preflight_hook() -> None:
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_tool_use = settings["hooks"]["PreToolUse"]
    matchers = {entry["matcher"]: entry for entry in pre_tool_use}
    assert set(matchers) == {"Bash", "mcp__github__merge_pull_request"}

    module_ref = f"-m {hook.__name__}"
    for matcher, entry in matchers.items():
        # The Bash matcher may carry additional PreToolUse hooks (e.g. the
        # git-push lint hook, #1309); find the merge_preflight_hook among
        # them rather than assuming it is the only one.
        command_hooks = entry["hooks"]
        merge_hooks = [h for h in command_hooks if module_ref in h.get("command", "")]
        assert len(merge_hooks) == 1, (
            f"{matcher} matcher must wire exactly one {module_ref} hook, got {len(merge_hooks)}"
        )
        command_hook = merge_hooks[0]
        assert command_hook["type"] == "command"
        command = command_hook["command"]
        assert command.strip().endswith("|| true"), f"{matcher} hook must fail open on crash"
