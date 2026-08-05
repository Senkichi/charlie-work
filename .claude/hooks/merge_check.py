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
import shlex
import subprocess
import sys
from pathlib import Path


# Fallback for the set of ``gh pr merge`` flags that consume a value argument.
# The primary source is the installed ``gh`` help output; this fallback is only
# used when help cannot be read, so the parser degrades safely.
_FALLBACK_GH_PR_MERGE_VALUE_FLAGS = frozenset(
    {
        "-A",
        "--author-email",
        "-b",
        "--body",
        "-F",
        "--body-file",
        "-t",
        "--subject",
        "-R",
        "--repo",
        "--match-head-commit",
    }
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
    """Load ``dispatch.branch_prefix`` from the layered config.

    The hook must read the same config layer the orchestrator uses for the
    worker branch prefix. Reading a non-layered config would silently use the
    wrong prefix whenever the global fleet layer sets ``dispatch.branch_prefix``,
    causing the hook to skip the exact worker PRs it exists to guard.
    """
    from charlie_work.global_config import load_layered_config

    cfg = load_layered_config(repo_root)
    return cfg.dispatch.branch_prefix


def _is_gh_pr_merge(command: str) -> bool:
    """Return True if the Bash command is a ``gh pr merge`` invocation."""
    stripped = command.strip()
    return (
        stripped.startswith("gh pr merge")
        or re.search(r"\bgh\s+pr\s+merge\b", stripped) is not None
    )


def _parse_gh_pr_merge_help(help_text: str) -> frozenset[str]:
    """Parse ``gh pr merge --help`` and return the value-taking flags.

    A line in the FLAGS section looks like one of:

        --admin                   Use administrator privileges to merge ...
      -A, --author-email text     Email text for merge commit author
        --match-head-commit SHA   Commit SHA that the pull request head ...
      -R, --repo [HOST/]OWNER/REPO  Select another repository ...

    Boolean flags are followed directly by a capitalized description. Value
    flags are followed by a placeholder token: a lowercase word (``text``,
    ``file``), an all-caps abbreviation (``SHA``), or a bracketed/slashed
    template (``[HOST/]OWNER/REPO``).
    """
    value_flags: set[str] = set()
    in_flags = False
    for raw in help_text.splitlines():
        line = raw.rstrip()
        if line.strip() == "FLAGS":
            in_flags = True
            continue
        if not in_flags:
            continue
        if line.strip() == "":
            continue
        # The next section header ends the FLAGS list, but inherited flags
        # (e.g. ``-R, --repo``) are still value-taking and must be parsed.
        if line.strip() == "LEARN MORE":
            break
        if line.strip() == "INHERITED FLAGS":
            continue
        m = re.match(r"^\s*(?:-\w,\s*)?(--[\w-]+)(?:\s+(\S+))?", line)
        if not m:
            continue
        long_flag = m.group(1)
        short_flag = None
        short_m = re.match(r"^\s*(-\w),\s*", line)
        if short_m:
            short_flag = short_m.group(1)
        token = m.group(2)
        rest = line[m.end() :] if token else ""
        if token and _looks_like_value_placeholder(token, rest):
            value_flags.add(long_flag)
            if short_flag:
                value_flags.add(short_flag)
    return frozenset(value_flags)


def _looks_like_value_placeholder(token: str, rest: str) -> bool:
    """Return True if the token after a flag name is a value placeholder."""
    if not token:
        return False
    if token.startswith("[") or "/" in token:
        return True
    if token.isupper() and len(token) > 1:
        return True
    # Lowercase placeholder (e.g. ``text``, ``file``) followed by a description
    # that begins with an uppercase word.
    if token.islower() and rest.strip() and rest.strip()[0].isupper():
        return True
    return False


def _gh_pr_merge_value_flags(repo_root: Path) -> frozenset[str]:
    """Return the set of ``gh pr merge`` flags that consume a value argument.

    The authoritative source is the installed ``gh`` help output, parsed once
    per process. If help cannot be read, the function falls back to a built-in
    list so the parser still degrades safely.
    """
    cached = getattr(_gh_pr_merge_value_flags, "_cache", None)
    if cached is not None:
        return cached
    flags: frozenset[str] = _FALLBACK_GH_PR_MERGE_VALUE_FLAGS
    try:
        result = _run_command(_gh_bin() + ["pr", "merge", "--help"], repo_root)
        if result.returncode == 0:
            flags = _parse_gh_pr_merge_help(result.stdout)
    except Exception:
        pass
    _gh_pr_merge_value_flags._cache = flags
    return flags


def _extract_selector(command: str, repo_root: Path) -> str | None:
    """Return the first non-flag positional selector after ``gh pr merge``.

    The selector may be a PR number, a URL, or a branch name. ``None`` means
    the command had no positional argument (the form ``gh pr merge`` uses the
    current branch). Quoted flag values are preserved by ``shlex.split`` so a
    value like ``--body "merge this"`` is treated as one token and not mistaken
    for the PR number.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    try:
        merge_idx = tokens.index("merge")
    except ValueError:
        return None
    if merge_idx < 2 or tokens[merge_idx - 1] != "pr" or tokens[merge_idx - 2] != "gh":
        return None

    value_flags = _gh_pr_merge_value_flags(repo_root)
    stop_parsing_flags = False
    idx = merge_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in ("&&", "||", ";", "|"):
            break
        if token == "--":
            stop_parsing_flags = True
            idx += 1
            continue
        if not stop_parsing_flags and token.startswith("-"):
            if "=" in token:
                idx += 1
                continue
            if token in value_flags:
                idx += 2
                continue
            idx += 1
            continue
        return token
    return None


def _parse_pr_number(command: str, repo_root: Path) -> int | None:
    """Extract a numeric PR number from a ``gh pr merge`` command string.

    Supports common invocations like ``gh pr merge 759 --squash``,
    ``gh pr merge --repo owner/repo 759``, and ``gh pr merge --body "merge this" 759``.
    Returns None for branch- or URL-based merges; those are resolved via
    ``gh pr view`` instead.
    """
    selector = _extract_selector(command, repo_root)
    if not selector:
        return None
    if selector.isdigit():
        return int(selector)
    match = re.search(r"/pull/(\d+)", selector)
    return int(match.group(1)) if match else None


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


def _gh_pr_view(repo_root: Path, selector: str | None = None) -> dict[str, object]:
    """Fetch PR metadata from ``gh`` and return a dict."""
    cmd = _gh_bin() + ["pr", "view"]
    if selector is not None:
        cmd.append(selector)
    cmd.extend(["--json", "number,headRefName"])
    result = _run_command(cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh pr view failed")
    return json.loads(result.stdout or "{}")


def _current_branch(repo_root: Path) -> str:
    """Return the current git branch, or raise if it cannot be determined."""
    result = _run_command(["git", "branch", "--show-current"], repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git branch failed")
    return result.stdout.strip()


def _run_merge_check(pr_number: int, repo_root: Path) -> dict[str, object]:
    """Call ``charlie merge-check --json <pr>`` and return the parsed result.

    ``charlie`` prints the result as a single pretty-printed JSON document
    (``print_result`` emits ``indent=2``). The hook must parse the whole stdout
    as one document, not scan lines, because every indented line is not itself
    valid JSON.

    If the subprocess produces no parseable JSON, this raises so the caller can
    deny the merge. A successful ``merge-check`` run always writes valid JSON
    even when it returns ``ok=False``.
    """
    base = _charlie_bin(repo_root)
    cmd = base + ["--json", "merge-check", str(pr_number)]
    result = _run_command(cmd, repo_root)
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"merge-check produced no stdout (exit {result.returncode})")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"merge-check stdout is not valid JSON (exit {result.returncode}): {exc}"
        ) from exc


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

    repo_root = _repo_root()
    try:
        prefix = _branch_prefix(repo_root)
    except Exception as exc:
        _deny(
            f"merge-check hook: cannot load worker branch prefix from config ({exc}); "
            f"denying as a precaution"
        )

    try:
        selector = _extract_selector(command, repo_root)
    except Exception as exc:
        _allow(f"merge-check hook: could not parse `gh pr merge` command ({exc}); allowing")

    pr_data: dict[str, object] = {}
    if selector is None:
        # No positional PR number: ``gh pr merge`` infers from the current branch.
        try:
            pr_data = _gh_pr_view(repo_root)
        except Exception:
            try:
                branch = _current_branch(repo_root)
            except Exception:
                branch = ""
            if branch and branch.startswith(prefix):
                _deny(
                    f"merge-check hook: could not resolve a PR for current worker branch "
                    f"{branch!r}; denying"
                )
            _allow("merge-check hook: could not resolve a PR for the current branch; allowing")
    else:
        try:
            pr_data = _gh_pr_view(repo_root, selector)
        except Exception as exc:
            _allow(f"merge-check hook: could not resolve PR from command ({exc}); allowing")

    head_ref = str(pr_data.get("headRefName", ""))
    if not head_ref.startswith(prefix):
        _allow(
            f"merge-check hook: PR branch {head_ref!r} does not match worker "
            f"prefix {prefix!r}; skipping authorization check"
        )

    pr_number = int(pr_data.get("number") or 0)
    if not pr_number:
        _allow("merge-check hook: resolved PR has no number; allowing")

    try:
        verdict = _run_merge_check(pr_number, repo_root)
    except Exception as exc:
        _deny(f"merge-check hook: could not run merge-check for PR #{pr_number} ({exc}); denying")

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
