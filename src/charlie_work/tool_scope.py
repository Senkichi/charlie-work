"""Tool-scope and MCP-isolation pin for Claude Code launch argv (issue #2836).

Extracted from ``claude_code.py`` (which re-exports the function): the
file-size ratchet forbids adding new code to that over-cap monolith, and a
single-purpose domain module is the right scope for a pure argv rewriter
consumed by both the adapter (launch) and the doctor (effective-scope
reporting).
"""

from __future__ import annotations


def _apply_tool_scope(
    command_template: tuple[str, ...],
    tools: str,
    isolate_mcp: bool,
) -> tuple[str, ...]:
    """Hard-pin the tool scope and MCP isolation onto ``command_template``.

    Issue #2836: fleet Claude sessions (workers and reviewers) used to launch
    with the operator's whole interactive tool surface — every user-scope MCP
    server, every claude.ai connector, and the full built-in tool schema block
    (Artifact/Skill/Workflow/Agent measured ~40k of ~57k first-call context
    tokens, then re-read from cache on every call for the session's lifetime).

    Coverage note: the worker role's usage over the trailing 48 h was fully
    covered by the worker allowlist. The reviewer role's usage was NOT:
    reviews also used PowerShell, WebFetch and Agent, which the reviewer
    allowlist drops. Reviews are expected to read GitHub through ``gh`` in
    Bash instead; a review that genuinely needs a dropped tool should argue
    that in code, not by widening this pin silently.

    Shaped like ``_apply_model_pin``: every occurrence of ``--tools`` (bare
    flag + value, or ``--tools=value``), ``--mcp-config`` (both forms), and
    the valueless ``--strict-mcp-config`` is stripped first — a
    caller-supplied ``command_template`` (``ClaudeCodeConfig.command``) must
    not be able to widen the scope, the same way
    ``_sanitize_review_command_template`` defeats ``--permission-mode``
    overrides. Then a single authoritative ``--tools <csv>`` is appended when
    ``tools`` is non-empty (an empty string is the operator kill switch: no
    flag, CLI default applies), and ``--strict-mcp-config --mcp-config
    {"mcpServers":{}}`` is appended when ``isolate_mcp`` is true (inline
    empty-server JSON; the CLI accepts a JSON string). Safe to apply uniformly
    to both roles: it only ever narrows.
    """
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token in ("--tools", "--mcp-config"):
            skip_next = True
            continue
        if token == "--strict-mcp-config":
            continue
        if token.startswith("--tools=") or token.startswith("--mcp-config="):
            continue
        filtered.append(token)
    if tools:
        filtered.extend(("--tools", tools))
    if isolate_mcp:
        filtered.extend(("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'))
    return tuple(filtered)
