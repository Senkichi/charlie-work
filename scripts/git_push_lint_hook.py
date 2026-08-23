#!/usr/bin/env python3
"""PreToolUse hook: lint before ``git push`` leaves the machine (#1309).

The W2 design's second surface -- deferred from the shipped Stop-gate PRs
(cw #1305, jc Senkichi/job-cannon#1730) and confirmed unimplemented in both
repos by the jc port's adversarial review. The Stop gate lints the
committed-diff-vs-merge-base union at session end, so a push mid-session
gets caught at the next Stop; the remaining gap is a session that pushes
and then dies/aborts before any Stop fires (exactly the population the
salvage lane handles, whose pushes bypass the worker entirely). This hook
closes that gap by running the same scoped-ruff check at push time.

**Reuses** ``worker_stop_gate``'s changed-set derivation
(``_all_changed_files``) and scoped-ruff machinery (``_run_ruff``) rather
than a second implementation -- loaded via ``importlib`` from the sibling
script, never duplicated. Does NOT run targeted tests: the Stop gate
already does that at session end, and running tests before every push
would be too slow and too aggressive for a PreToolUse gate that fires on
every Bash command.

Registered as a PreToolUse hook on ``Bash`` in ``.claude/settings.json``.
Only fires on actual ``git push`` invocations (token-based detection --
see ``_is_git_push``), not on other git commands. A ``git push`` inside a
quoted string (a commit message, an echo) is a single shlex token and
therefore does not match -- the same trap the merge_preflight_hook's
first version fell into, avoided here by construction.

Honors the same fail-open fallbacks as ``worker_stop_gate``:

- **Branch-base derivation ambiguity** (detached HEAD, no ``origin/main``
  ref, ``HEAD`` == ``origin/main``, or any ``merge-base``/``diff``
  failure) -- handled inside ``_committed_diff_files``, which returns
  ``()`` and silently narrows the changed-set to working-tree-only scope.
  Inherited automatically by reusing ``_all_changed_files``.
- **Missing interpreter** -- the ``.claude/settings.json`` command line
  uses ``.venv/Scripts/python.exe``; if the venv is missing the hook
  cannot launch at all, which is the intentionally safe direction for an
  environment-setup failure that is not this gate's job to diagnose (same
  as the Stop gate).

Every other error path fails CLOSED (denies the push): a ``git status``
failure, a ``ruff`` crash, or any unexpected exception during a confirmed
``git push`` all emit a ``deny`` decision so the push does not leave the
machine with lint issues unchecked. This matches the Stop gate's own
fail-closed default (every error path in ``worker_stop_gate.py`` blocks,
with the branch-base exception called out above).

No bounded-retry / exhaustion counter (unlike the Stop gate): a
PreToolUse hook fires once per tool call, not in a retry loop. If the
push is denied, the agent sees the reason and can fix the issue before
trying again; there is no loop to cap.

Stdlib-only (same constraint as ``worker_stop_gate`` -- worker sessions
are not guaranteed to have the ``dev`` extra installed). It shells out
to ``ruff`` via ``uv run --no-sync`` through the reused ``_run_ruff``,
so the script itself never needs third-party packages to run.

Hook contract (Claude Code PreToolUse format, consistent with
``merge_preflight_hook``): blocking is done via stdout JSON
``{"hookSpecificOutput": {"hookEventName": "PreToolUse",
"permissionDecision": "deny", "permissionDecisionReason": "..."}}`` and
exit 0, never via exit code 2. The ``|| true`` suffix in
``settings.json`` is a belt-and-suspenders crash net -- a bare non-zero
exit is treated as a non-blocking hook error, so the push would proceed,
which is the safe direction for a crash that is not this gate's job to
diagnose (same reasoning as the Stop gate's missing-interpreter
fallback).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

#: Shell separators that delimit independent commands in a Bash invocation.
#: Used by the token-based ``git push`` detector to avoid matching across
#: command boundaries (e.g. ``git status ; echo git push`` must not fire).
_SHELL_SEPARATORS = frozenset({";", "|", "&", "&&", "||", "\n"})

#: Git global options that consume the next token as their value, so the
#: scanner can skip past them to reach the subcommand. Git's documented
#: global options (``git --help`` "Options") that take a separate value
#: argument: ``-C <path>``, ``-c <name>=<value>``, ``--git-dir <path>``,
#: ``--work-tree <path>``, ``--namespace <name>``, ``--exec-path <path>``.
#: The ``=`` forms (``--git-dir=<path>``) are self-contained and handled
#: separately. Unknown flags are treated as boolean (do not consume the
#: next token) -- the worst case is a false negative on a hypothetical
#: value-taking flag we do not list, which errs toward allow, never toward
#: a spurious deny.
_GIT_GLOBAL_VALUE_FLAGS = frozenset({"C", "c", "git-dir", "work-tree", "namespace", "exec-path"})

#: Regex fallback for commands that shlex cannot parse (unbalanced quotes).
#: Less precise than the token-based path (can match inside quoted strings)
#: but unparseable input is rare and the Stop gate still backstops whatever
#: slips through here.
_GIT_PUSH_RE = re.compile(r"\bgit(?:\s+-\S+)*\s+push\b")

#: Tokens that can precede ``git`` in command position without being a
#: shell separator: ``env`` (runs its args as a command) and ``NAME=VALUE``
#: env-var assignments (``VAR=val git push`` sets VAR for the command).
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _load_stop_gate() -> ModuleType:
    """Load ``worker_stop_gate.py`` from the sibling script path.

    Uses ``importlib.util.spec_from_file_location`` rather than polluting
    ``sys.path`` -- the script directory is not a package, and a bare
    ``sys.path.insert`` + ``import`` would shadow or collide with any
    same-named module in the worker's environment. The module name
    ``_worker_stop_gate_for_push_hook`` is deliberately unique to avoid
    collisions with the test loader's ``worker_stop_gate_under_test``.

    The module is registered in ``sys.modules`` *before* ``exec_module`` so
    that ``worker_stop_gate.py``'s ``@dataclass(frozen=True)`` classes can
    resolve their string annotations (``from __future__ import
    annotations``) during class creation -- the same fix
    ``_script_loader.py`` applies (issue #1023).
    """
    module_name = "_worker_stop_gate_for_push_hook"
    script_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(module_name, script_dir / "worker_stop_gate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load worker_stop_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _is_git_binary(tok: str) -> bool:
    """Does this token invoke ``git``? Handles bare ``git``, full paths
    (``/usr/bin/git``), and the Windows ``.exe`` suffix."""
    base = tok.replace("\\", "/").rsplit("/", 1)[-1]
    return base in ("git", "git.exe")


def _is_command_position(tokens: list[str], idx: int) -> bool:
    """Is ``tokens[idx]`` in command position (the first word of a command)?

    ``git`` must be the command itself, not an argument to another command
    (``echo git push`` must not fire). Command position means the token is
    either the first in the stream, or the preceding token is a shell
    separator (``;``, ``&&``, ``||``, ``|``, ``&``, newline), or the
    preceding token is an ``env`` prefix or a ``NAME=VALUE`` env-var
    assignment -- both of which transparently pass command position through
    to the next token.
    """
    if idx == 0:
        return True
    prev = tokens[idx - 1]
    if prev in _SHELL_SEPARATORS:
        return True
    if prev == "env":
        return True
    if _ENV_ASSIGNMENT_RE.match(prev):
        return True
    return False


def _is_git_push(command: str) -> bool:
    """Token-based detection of a ``git push`` invocation.

    Splits the command with ``shlex`` (``punctuation_chars=True`` so shell
    separators become their own tokens, preventing a first command's
    argument scan from swallowing a second command). A ``git push`` is:
    ``git`` (bare or as a path) in command position, then zero or more
    global flags (skipping value-taking flags and their values), then
    ``push`` as the first positional token -- i.e. the subcommand.

    A ``git push`` inside a quoted string (``echo "git push"``,
    ``git commit -m "git push"``) is a single shlex token and therefore
    does not match -- the same trap ``merge_preflight_hook``'s first
    version fell into, avoided here by construction. A bare ``git``
    appearing as an argument to a non-git command (``echo git push``)
    is not in command position and does not match either.

    Falls back to a regex for unparseable commands (unbalanced quotes).
    """
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return bool(_GIT_PUSH_RE.search(command))

    idx = 0
    while idx < len(tokens):
        if not _is_git_binary(tokens[idx]) or not _is_command_position(tokens, idx):
            idx += 1
            continue
        # Found `git` in command position -- scan forward for the subcommand.
        i = idx + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in _SHELL_SEPARATORS:
                break  # command boundary -- `git` here has no subcommand
            if tok.startswith("--") and "=" in tok:
                i += 1
                continue  # self-contained --flag=value
            if tok.startswith("-"):
                flag_name = tok.lstrip("-")
                if flag_name in _GIT_GLOBAL_VALUE_FLAGS and i + 1 < len(tokens):
                    i += 2  # consume flag + its value
                    continue
                i += 1
                continue
            # First positional token after global flags -- this is the
            # subcommand.
            if tok == "push":
                return True
            break  # subcommand is not `push` -- not our concern
        idx = i + 1 if i < len(tokens) else i
    return False


def _deny(reason: str) -> None:
    """Emit a Claude Code PreToolUse deny decision on stdout."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def _resolve_cwd(payload: dict[str, Any]) -> Path:
    """Resolve the working directory from the hook payload, falling back
    to ``Path.cwd()`` -- same logic as ``worker_stop_gate.main``."""
    cwd_value = payload.get("cwd")
    if isinstance(cwd_value, str) and cwd_value:
        return Path(cwd_value)
    return Path.cwd()


def _check_push_lint(command: str, cwd: Path) -> str | None:
    """Run the scoped-ruff lint check for a confirmed ``git push``.

    Returns a deny reason string, or ``None`` to allow the push. Reuses
    ``worker_stop_gate``'s ``_all_changed_files`` (changed-set derivation
    with the same fail-open branch-base fallback) and ``_run_ruff``
    (scoped ``ruff check`` + ``ruff format --check`` on exactly the
    changed ``.py`` files).
    """
    stop_gate = _load_stop_gate()
    repo_root = stop_gate._repo_root(cwd)
    changed = stop_gate._all_changed_files(repo_root)
    if not changed:
        return None  # nothing to lint -- allow
    py_files = tuple(
        sorted(cf.path for cf in changed if not cf.deleted and cf.path.endswith(".py"))
    )
    if not py_files:
        return None  # no .py files in the changed set -- allow
    result = stop_gate._run_ruff(repo_root, py_files)
    if result.block:
        return result.reason
    return None


def main() -> int:
    try:
        raw_stdin = sys.stdin.read()
    except OSError:
        raw_stdin = ""
    try:
        payload = json.loads(raw_stdin) if raw_stdin.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except json.JSONDecodeError:
        payload = {}

    tool_name = payload.get("tool_name") or ""
    if tool_name != "Bash":
        return 0  # not our tool -- leave undecided

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return 0
    command = tool_input.get("command") or ""
    if not isinstance(command, str) or not _is_git_push(command):
        return 0  # not a git push -- leave undecided

    # Confirmed git push -- run the scoped-ruff lint check.
    try:
        reason = _check_push_lint(command, _resolve_cwd(payload))
    except Exception as exc:  # noqa: BLE001 -- fail closed on any error during a push
        _deny(
            f"git push lint hook errored ({type(exc).__name__}: {exc}); "
            "failing closed -- fix the hook or push manually after verifying lint."
        )
        return 0
    if reason is not None:
        _deny(reason)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the hook subprocess
    sys.exit(main())
