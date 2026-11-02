"""Review command-template helpers: ``_sanitize_review_command_template``
and the ``_apply_model_pin`` / ``_apply_effort_pin`` flag rewriters.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

from charlie_work.claude_code import (
    _apply_model_pin,
    _apply_effort_pin,
    _apply_tool_scope,
    _sanitize_review_command_template,
)


def test_sanitize_review_command_template_strips_duplicate_space_form_flags() -> None:
    """Round-3 review (PR #397): a template with duplicate space-form
    `--permission-mode` flags must not let the trailing occurrence survive —
    CLI parsers apply last-flag-wins semantics, so a naive first-match fix
    would still launch in acceptEdits mode."""
    template = (
        "claude",
        "-p",
        "--permission-mode",
        "plan",
        "--permission-mode",
        "acceptEdits",
    )

    result = _sanitize_review_command_template(template)

    assert result.count("--permission-mode") == 1
    idx = result.index("--permission-mode")
    assert result[idx + 1] == "plan"
    assert idx == len(result) - 2  # positioned last


def test_sanitize_review_command_template_strips_equals_joined_flag() -> None:
    """An equals-joined `--permission-mode=acceptEdits` token must be removed
    entirely, not merely left in place because the append-based happy path
    currently makes it look safe by accident."""
    template = ("claude", "-p", "--permission-mode=acceptEdits")

    result = _sanitize_review_command_template(template)

    assert not any(tok.startswith("--permission-mode=") for tok in result)
    assert result[-2:] == ("--permission-mode", "plan")


def test_sanitize_review_command_template_strips_mixed_forms() -> None:
    """Mixed equals-joined and space-form occurrences are all stripped,
    leaving a single trailing `--permission-mode plan`."""
    template = (
        "claude",
        "-p",
        "--permission-mode=acceptEdits",
        "--permission-mode",
        "acceptEdits",
    )

    result = _sanitize_review_command_template(template)

    assert result.count("--permission-mode") == 1
    assert not any(tok.startswith("--permission-mode=") for tok in result)
    assert result[-2:] == ("--permission-mode", "plan")


def test_sanitize_review_command_template_handles_bare_trailing_flag() -> None:
    """A malformed trailing `--permission-mode` with no value token must not
    raise (e.g. IndexError) — it is stripped like any other occurrence and
    the authoritative flag is appended."""
    template = ("claude", "-p", "--permission-mode")

    result = _sanitize_review_command_template(template)

    assert result == ("claude", "-p", "--permission-mode", "plan")


def test_sanitize_review_command_template_preserves_lookalike_token() -> None:
    """A token like `--permission-modex` must not be matched as the flag —
    only an exact `--permission-mode` token or exact `--permission-mode=`
    prefix count."""
    template = ("claude", "-p", "--permission-modex", "plan")

    result = _sanitize_review_command_template(template)

    assert "--permission-modex" in result
    assert result == ("claude", "-p", "--permission-modex", "plan", "--permission-mode", "plan")


def test_apply_model_pin_appends_to_template_without_model() -> None:
    """Issue #530: a bare template (no --model) must get the configured
    model pinned so the subprocess never falls back to ambient global CLI
    state (e.g. an interactive session's last `/model` choice)."""
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result == ("claude", "-p", "--permission-mode", "plan", "--model", "claude-sonnet-5")


def test_apply_model_pin_strips_existing_space_form_flag() -> None:
    template = ("claude", "-p", "--model", "claude-opus-4-8", "--permission-mode", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result.count("--model") == 1
    idx = result.index("--model")
    assert result[idx + 1] == "claude-sonnet-5"
    assert idx == len(result) - 2  # positioned last, last-flag-wins


def test_apply_model_pin_strips_equals_joined_flag() -> None:
    template = ("claude", "-p", "--model=claude-opus-4-8")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert not any(tok.startswith("--model=") for tok in result)
    assert result[-2:] == ("--model", "claude-sonnet-5")


def test_apply_model_pin_handles_bare_trailing_flag() -> None:
    template = ("claude", "-p", "--model")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result == ("claude", "-p", "--model", "claude-sonnet-5")


def test_apply_model_pin_preserves_lookalike_token() -> None:
    template = ("claude", "-p", "--modelx", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert "--modelx" in result
    assert result == ("claude", "-p", "--modelx", "plan", "--model", "claude-sonnet-5")


def test_apply_model_pin_handles_empty_template() -> None:
    assert _apply_model_pin((), "claude-sonnet-5") == ("--model", "claude-sonnet-5")


def test_apply_effort_pin_appends_to_template_without_effort() -> None:
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "medium")

    assert result == ("claude", "-p", "--permission-mode", "plan", "--effort", "medium")


def test_apply_effort_pin_strips_existing_space_form_flag() -> None:
    template = ("claude", "-p", "--effort", "high", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "medium")

    assert result.count("--effort") == 1
    idx = result.index("--effort")
    assert result[idx + 1] == "medium"
    assert idx == len(result) - 2


def test_apply_effort_pin_strips_equals_joined_flag() -> None:
    template = ("claude", "-p", "--effort=high")

    result = _apply_effort_pin(template, "medium")

    assert not any(tok.startswith("--effort=") for tok in result)
    assert result[-2:] == ("--effort", "medium")


def test_apply_effort_pin_empty_effort_is_noop() -> None:
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "")

    assert result == template


def test_apply_effort_pin_handles_bare_trailing_flag() -> None:
    template = ("claude", "-p", "--effort")

    result = _apply_effort_pin(template, "medium")

    assert result == ("claude", "-p", "--effort", "medium")


def test_apply_effort_pin_preserves_lookalike_token() -> None:
    template = ("claude", "-p", "--effortx", "plan")

    result = _apply_effort_pin(template, "medium")

    assert "--effortx" in result
    assert result == ("claude", "-p", "--effortx", "plan", "--effort", "medium")


def test_apply_effort_pin_handles_empty_template() -> None:
    assert _apply_effort_pin((), "medium") == ("--effort", "medium")


# Issue #2836: _apply_tool_scope -- the tool-allowlist + MCP-isolation pin.

_WORKER_TOOLS = "Bash,Read,Edit,Write,Grep,Glob,Monitor,TaskStop"
_REVIEWER_TOOLS = "Bash,Read,Grep,Glob,Write"
_ISOLATED_MCP_FLAGS = ("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}')


def test_apply_tool_scope_appends_tools_and_mcp_isolation() -> None:
    template = ("claude", "-p", "--permission-mode", "bypassPermissions")

    result = _apply_tool_scope(template, _WORKER_TOOLS, True)

    assert result == template + ("--tools", _WORKER_TOOLS) + _ISOLATED_MCP_FLAGS


def test_apply_tool_scope_reviewer_tools() -> None:
    result = _apply_tool_scope(("claude", "-p"), _REVIEWER_TOOLS, True)

    assert result.count("--tools") == 1
    assert result[-5:-3] == ("--tools", _REVIEWER_TOOLS)
    assert result[-3:] == _ISOLATED_MCP_FLAGS


def test_apply_tool_scope_empty_tools_is_kill_switch() -> None:
    """An empty *_tools string adds no --tools flag: CLI default applies."""
    template = ("claude", "-p")

    result = _apply_tool_scope(template, "", True)

    assert result == template + _ISOLATED_MCP_FLAGS
    assert not any(tok.startswith("--tools") for tok in result)


def test_apply_tool_scope_isolate_mcp_false_adds_no_mcp_flags() -> None:
    template = ("claude", "-p")

    result = _apply_tool_scope(template, _WORKER_TOOLS, False)

    assert result == template + ("--tools", _WORKER_TOOLS)
    assert "--strict-mcp-config" not in result
    assert "--mcp-config" not in result


def test_apply_tool_scope_strips_caller_space_form_tools() -> None:
    template = ("claude", "-p", "--tools", "Artifact,Workflow")

    result = _apply_tool_scope(template, _WORKER_TOOLS, True)

    assert "Artifact" not in result
    assert result.count("--tools") == 1
    assert result[result.index("--tools") + 1] == _WORKER_TOOLS


def test_apply_tool_scope_strips_equals_joined_tools() -> None:
    template = ("claude", "-p", "--tools=Artifact,Workflow")

    result = _apply_tool_scope(template, _WORKER_TOOLS, True)

    assert not any(tok.startswith("--tools=") for tok in result)
    assert "Artifact" not in result
    assert result.count("--tools") == 1


def test_apply_tool_scope_strips_caller_mcp_config() -> None:
    for template in (
        ("claude", "-p", "--mcp-config", "other.json"),
        ("claude", "-p", "--mcp-config=other.json"),
        ("claude", "-p", "--strict-mcp-config"),
    ):
        result = _apply_tool_scope(template, _REVIEWER_TOOLS, True)

        assert "other.json" not in result
        assert result.count("--strict-mcp-config") == 1
        assert result.count("--mcp-config") == 1
        assert result[result.index("--mcp-config") + 1] == '{"mcpServers":{}}'


def test_apply_tool_scope_strips_mcp_flag_from_isolate_off_template() -> None:
    """A caller-supplied --mcp-config is stripped even with isolation off —
    the config decision is authoritative, not a smuggled old value."""
    template = ("claude", "-p", "--mcp-config=other.json")

    result = _apply_tool_scope(template, _REVIEWER_TOOLS, False)

    assert "other.json" not in result
    assert "--mcp-config" not in result
