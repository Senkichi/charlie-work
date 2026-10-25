"""Doctor check: a claude-code worker that cannot run Bash headless (issue #2010).

A headless ``claude -p`` session cannot answer a permission prompt, so under a
prompting ``--permission-mode`` (``acceptEdits``/``default``/``plan``) every
Bash call not pre-allowed is denied: the worker edits but never tests,
commits or pushes. The repo's tracked ``.claude/settings.json`` is the only
other source of permissions.

Lives outside ``doctor`` because ``doctor.py`` is over the module cap and
pinned by the file-size ratchet (see ``doctor_local_backend``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .claude_code import _WORKER_COMMAND_TEMPLATE, PROMPTING_PERMISSION_MODES
from .config import OrchestratorConfig


def effective_permission_mode(command: tuple[str, ...]) -> str | None:
    """Last ``--permission-mode`` value in ``command`` (``None`` if absent)."""
    mode: str | None = None
    for index, token in enumerate(command):
        if token == "--permission-mode" and index + 1 < len(command):
            mode = command[index + 1]
        elif token.startswith("--permission-mode="):
            mode = token.split("=", 1)[1]
    return mode


def _has_bash_allow_rule(repo_root: Path) -> bool:
    try:
        data = json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return isinstance(allow, list) and any(
        isinstance(rule, str) and rule.startswith("Bash") for rule in allow
    )


def _check_claude_worker_permissions(
    add: Any, repo_root: Path, config: OrchestratorConfig
) -> None:
    if config.worker.harness != "claude-code":
        return
    command = config.claude_code.command or _WORKER_COMMAND_TEMPLATE
    _check_claude_worker_tool_scope(add, command, config)
    mode = effective_permission_mode(command)
    # No flag at all means the CLI's own default mode, which prompts.
    prompts = mode is None or mode in PROMPTING_PERMISSION_MODES
    if not prompts or _has_bash_allow_rule(repo_root):
        add(
            "claude-code worker permissions",
            True,
            f"permission mode `{mode}` (or a Bash allow rule) lets a headless worker run Bash",
        )
        return
    add(
        "claude-code worker permissions",
        False,
        f"permission mode `{mode or '(CLI default)'}` cannot answer prompts headless and "
        f"{repo_root / '.claude' / 'settings.json'} has no Bash allow rule: workers will "
        "edit but never test/commit/push. Use `--permission-mode bypassPermissions` in "
        "claude_code.command or add Bash allow rules.",
    )


def effective_tool_scope(command: tuple[str, ...], config: OrchestratorConfig) -> tuple[str, bool]:
    """Resolve the effective tool allowlist and MCP isolation for a
    claude-code worker launch (issue #2836).

    The argv-side pin (``claude_code._apply_tool_scope``) is appended by the
    adapter, not present in ``command`` — so a configured ``--tools`` in
    ``claude_code.command`` is the caller-tuned source, otherwise the config's
    ``worker_tools`` applies. Returns ``(tools, isolate_mcp)``: ``tools`` is
    the comma-separated allowlist, empty when the CLI default applies.
    """
    tools = ""
    for index, token in enumerate(command):
        if token == "--tools" and index + 1 < len(command):
            tools = command[index + 1]
        elif token.startswith("--tools="):
            tools = token.split("=", 1)[1]
    if not tools:
        tools = config.claude_code.worker_tools
    return tools, bool(config.claude_code.isolate_mcp)


def _check_claude_worker_tool_scope(
    add: Any, command: tuple[str, ...], config: OrchestratorConfig
) -> None:
    """Report the effective tool scope / MCP isolation and warn when an
    operator has turned either kill switch back on (issue #2836)."""
    tools, isolate_mcp = effective_tool_scope(command, config)
    findings: list[str] = []
    if not tools:
        findings.append(
            "claude_code.worker_tools is empty: no --tools pin is added and the CLI's "
            "full default tool set (~40k unused schema tokens per call) loads"
        )
    if not isolate_mcp:
        findings.append(
            "claude_code.isolate_mcp is false: user-scope MCP servers and claude.ai "
            "connectors load in every worker session that reads issue text under "
            "bypassPermissions"
        )
    detail = (
        f"--tools {tools}" if tools else "--tools (CLI default)"
    ) + f"; MCP isolation: {'on (--strict-mcp-config)' if isolate_mcp else 'off'}"
    if findings:
        add(
            "claude-code worker tool scope",
            False,
            f"{detail}; {'; '.join(findings)}",
            severity="warning",
        )
    else:
        add("claude-code worker tool scope", True, detail)
