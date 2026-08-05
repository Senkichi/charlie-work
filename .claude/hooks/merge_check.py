#!/usr/bin/env python3
"""Enforcing PreToolUse hook for raw ``gh pr merge``.

Issue #894. ``charlie merge-check <pr>`` exists and enforces the same
approved-at-head invariant as the ``ship-it`` and Aviator re-queue paths, but
nothing called it. This hook is the consumer: it runs the preflight for worker
PRs and denies the tool call when the PR is not authorized at its current head.

The hook only fires on ``gh pr merge`` Bash tool calls, and only consults
``merge-check`` when the PR's head branch starts with ``dispatch.branch_prefix``.
Orchestrator PRs on ``fix/*`` (or any other non-worker branch) bypass the check,
matching the tripwire's existing scoping.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Flags that take a value in ``gh pr merge``. We skip both the flag and its
# argument while looking for the positional PR number.
_GH_MERGE_FLAGS_WITH_VALUES = frozenset(
    {"--repo", "--body", "--body-file", "--subject", "--match-title", "--match-head-sha"}
)


def _repo_root() -> Path:
    """Resolve the repo root.

    The ``CLAUDE_PROJECT_DIR`` environment variable is provided by Claude Code
    to command hooks. When it is absent (e.g. manual testing or other CLIs), the
    script's own location inside ``.claude/hooks/`` gives the repo root.
    """
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if project_dir:
        return Path(project_dir).resolve()
    return Path(__file__).resolve().parents[2]


def _branch_prefix(repo_root: Path) -> str:
    """Load ``dispatch.branch_prefix`` from config, falling back to the default.

    The hook must not hardcode the worker prefix, because that is precisely the
    kind of second scoping notion the tripwire already encodes.
    """
    try:
        from charlie_work.config import find_config_path, load_config
    except ImportError:
        return "agent/issue"
    try:
        cfg = load_config(find_config_path(repo_root))
        return cfg.dispatch.branch_prefix
    except Exception:
        return "agent/issue"


def _is_gh_pr_merge(command: str) -> bool:
    """Return True if the Bash command is a ``gh pr merge`` invocation."""
    stripped = command.strip()
    return (
        stripped.startswith("gh pr merge")
        or re.search(r"\bgh\s+pr\s+merge\b", stripped) is not None
    )


def _parse_pr_number(command: str) -> int | None:
    """Extract a numeric PR number from a ``gh pr merge`` command string.

    Supports common invocations like ``gh pr merge 759 --squash`` and
    ``gh pr merge --repo owner/repo 759``. Returns None for branch- or URL-based
    merges; those are not checked by this hook.
    """
    tokens = command.strip().split()
    try:
        merge_idx = tokens.index("merge")
    except ValueError:
        return None

    idx = merge_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in ("&&", "||", ";", "|"):
            break
        if token.startswith("-"):
            if "=" in token:
                # ``--repo=owner/repo`` carries its own value.
                idx += 1
                continue
            if token in _GH_MERGE_FLAGS_WITH_VALUES:
                idx += 2
                continue
            idx += 1
            continue
        if "/pull/" in token:
            match = re.search(r"/pull/(\d+)", token)
            return int(match.group(1)) if match else None
        if token.isdigit():
            return int(token)
        # First non-flag positional was a branch name or URL without a number.
        return None
    return None


def _run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run an external command and return its result."""
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _gh_bin() -> list[str]:
    """Resolve the ``gh`` executable, honoring ``GH_BIN`` for tests."""
    override = os.environ.get("GH_BIN", "gh")
    if override.endswith(".py"):
        return [sys.executable, override]
    return [override]


def _charlie_bin(repo_root: Path) -> list[str]:
    """Resolve the ``charlie merge-check`` invocation, honoring ``CHARLIE_BIN``."""
    override = os.environ.get("CHARLIE_BIN")
    if override:
        if override.endswith(".py"):
            return [sys.executable, override]
        return [override]
    return [sys.executable, "-m", "charlie_work"]


def _gh_pr_view(pr_number: int, repo_root: Path) -> dict[str, object]:
    """Fetch PR metadata from ``gh`` and return a dict."""
    cmd = _gh_bin() + ["pr", "view", str(pr_number), "--json", "headRefName"]
    result = _run_command(cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh pr view failed")
    return json.loads(result.stdout or "{}")


def _run_merge_check(pr_number: int, repo_root: Path) -> dict[str, object]:
    """Call ``charlie merge-check --json <pr>`` and return the parsed result.

    ``charlie`` prints the result as a single pretty-printed JSON document
    (``print_result`` emits ``indent=2``). The hook must parse the whole stdout
    as one document, not scan lines, because every indented line is not itself
    valid JSON.

    If the subprocess produces no parseable JSON, this raises so the caller can
    fail open. A successful ``merge-check`` run always writes valid JSON even
    when it returns ``ok=False``.
    """
    base = _charlie_bin(repo_root)
    cmd = base + ["--json", "merge-check", str(pr_number)]
    result = _run_command(cmd, repo_root)
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("merge-check produced no stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"merge-check stdout is not valid JSON: {exc}") from exc


def _allow(message: str | None = None) -> None:
    """Emit an ``allow`` PreToolUse hook decision and exit.

    The hook allows the tool call. ``systemMessage`` carries the ``merge-check``
    verdict so the agent sees it in the transcript.
    """
    payload: dict[str, object] = {
        "hookSpecificOutput": {"permissionDecision": "allow"},
    }
    if message:
        payload["systemMessage"] = message
    print(json.dumps(payload))
    sys.exit(0)


def _deny(message: str) -> None:
    """Emit a ``deny`` PreToolUse hook decision and exit.

    The PR is a worker PR and the merge-check preflight refused authorization.
    The hook blocks the raw ``gh pr merge`` and surfaces the reason.
    """
    payload: dict[str, object] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        },
        "systemMessage": message,
    }
    print(json.dumps(payload))
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        _allow("merge-check hook: could not parse PreToolUse input")

    if payload.get("tool_name") != "Bash":
        _allow()

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        _allow()

    command = tool_input.get("command", "")
    if not isinstance(command, str) or not _is_gh_pr_merge(command):
        _allow()

    pr_number = _parse_pr_number(command)
    if pr_number is None:
        _allow("merge-check hook: could not parse PR number from `gh pr merge` command")

    repo_root = _repo_root()
    prefix = _branch_prefix(repo_root)

    try:
        pr_data = _gh_pr_view(pr_number, repo_root)
    except Exception as exc:
        _allow(f"merge-check hook: could not read PR #{pr_number} branch ({exc}); allowing")

    head_ref = str(pr_data.get("headRefName", ""))
    if not head_ref.startswith(prefix):
        _allow(
            f"merge-check hook: PR #{pr_number} branch {head_ref!r} does not match worker "
            f"prefix {prefix!r}; skipping authorization check"
        )

    try:
        verdict = _run_merge_check(pr_number, repo_root)
    except Exception as exc:
        _allow(
            f"merge-check hook: could not run merge-check for PR #{pr_number} ({exc}); allowing"
        )

    if not verdict.get("ok"):
        reason = verdict.get("data", {}).get("reason") or verdict.get("message", "unknown")
        message = (
            f"merge-check hook: PR #{pr_number} is NOT authorized at head (reason: {reason}). "
            f"Verdict: {verdict.get('message')}"
        )
        _deny(message)

    _allow(f"merge-check hook: PR #{pr_number} is authorized at current head.")


if __name__ == "__main__":
    main()
