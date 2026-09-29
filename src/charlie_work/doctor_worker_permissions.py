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
