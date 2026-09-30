"""Devin-shell review-mode (read-only reviewer) posture.

Extracted from ``devin_shell.py`` (PR #2069 rework, file-size ratchet #1442);
``devin_shell`` re-exports every public name so existing callers keep
resolving through ``charlie_work.devin_shell``.

This module owns everything that pins a devin *review* launch to read-only:
the sanitized command template (no ``--permission-mode dangerous``, ever),
the ``.devin/config.local.json`` exec allow-list written into the review
checkout, and the prompt section that states the allow-list to the reviewer.
The worker (non-review) posture — ``DEFAULT_COMMAND_TEMPLATE``'s
``--permission-mode dangerous`` — stays in ``devin_shell``; nothing here
applies to it.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Review-mode template: omits ``--permission-mode dangerous`` entirely. The
# Devin CLI's documented default when that flag is absent is ``auto``
# (read-only tools) -- exactly the posture a reviewer needs, since the review
# packet built by ``workflow.py`` pre-renders the diff, CI status, and
# test-adequacy sections directly into the prompt, so a reviewer never needs
# to shell out to git/uv/gh (the only calls ``auto`` mode stalls on). This is
# the devin-shell analogue of claude-code's hard-pinned ``--permission-mode
# plan`` for review launches (see ``claude_code._REVIEW_COMMAND_TEMPLATE`` /
# ``_sanitize_review_command_template``). Not used directly by
# ``launch_devin_session`` (which sanitizes whatever template it receives,
# including a caller-tuned one, via ``_sanitize_review_command_template``
# below) -- kept as a documented, test-comparable constant for what that
# sanitization produces from the worker default.
_REVIEW_COMMAND_TEMPLATE: tuple[str, ...] = (
    "devin",
    "{model_args}",
    "--prompt-file",
    "{prompt_path}",
    "--print",
    "--respect-workspace-trust",
    "false",
)


# Issue #2011: headless ``devin --print`` in ``auto`` mode auto-REJECTS any exec
# its classifier does not approve and ENDS the session with no verdict. Models
# shell out despite the pre-rendered packet, so review launches pre-approve a
# READ-ONLY exec allow-list via ``<cwd>/.devin/config.local.json`` (project
# local override; never clobbers a tracked ``.devin/config.json``). Rules are
# whole-word prefix matches; compound commands are checked per segment, so
# chaining (``git log && rm x``) cannot escape the list. Allow rules only: a
# deny rule also ends the session silently.
#
# Anything NOT listed still goes to Devin's own classifier, which judges the
# FULL command line: it already auto-approved ``git status``/``git diff``/
# ``git log``/``git rev-parse`` in the #2011 sessions, and it can refuse a
# flag-level escape (``git diff --output=<path>``) that a whole-word prefix
# rule cannot express. So the list holds only (a) what the classifier
# actually prompted on -- the read-only ``gh ... view`` family, which ended 4
# of the 5 missed #2011 sessions -- and (b) commands with no exec/write flag.
#
# Deliberately EXCLUDED (keep it that way; the reviewer reads attacker-
# influenced diffs, so every entry must be safe against prompt injection):
# - bare ``Exec(git)``/``Exec(gh)`` and any git/gh subcommand with a write or
#   exec flag: ``git diff|log|show`` (``--output=<path>`` writes anywhere),
#   ``git grep`` (``-O<cmd>`` runs a pager command), ``gh api`` (arbitrary
#   REST incl. writes).
# - ``rg`` (``--pre <cmd>`` runs a program), ``sort`` (``-o``,
#   ``--compress-program``), ``uniq`` (writes its 2nd arg), ``find``
#   (``-delete``/``-exec``), ``sed``/``awk`` (in-place writes, system()).
# - any interpreter or runner (python, uv, node, bash, sh, pwsh): arbitrary
#   code -- the #2011 mdls session was ended by ``uv run ... python -c``, and
#   that refusal is correct.
# ``--permission-mode dangerous`` stays impossible (see the sanitizer below).
_REVIEW_EXEC_ALLOWLIST: tuple[str, ...] = (
    "Exec(gh issue view)",
    "Exec(gh pr view)",
    "Exec(gh pr diff)",
    "Exec(gh pr checks)",
    "Exec(grep)",
    "Exec(cat)",
    "Exec(head)",
    "Exec(tail)",
    "Exec(wc)",
    "Exec(ls)",
    "Exec(pwd)",
)


def _write_review_permissions(checkout_path: Path) -> None:
    """Write the review exec allow-list into the review checkout (issue #2011).

    Never raises: a missing allow-list degrades to the pre-fix behavior, and
    adapters return errors as values. The file is untracked inside a review
    checkout that ``remove_review_checkout`` force-removes wholesale, and
    nothing computes dirtiness for review checkouts, so no separate cleanup
    is needed.
    """
    try:
        # Lazy: ``_write_json`` lives in ``devin_shell`` (the sidecar writer);
        # a top-level import would cycle (devin_shell imports this module),
        # and call-time resolution keeps ``devin_shell._write_json`` patches
        # reaching this path.
        from .devin_shell import _write_json

        _write_json(
            checkout_path / ".devin" / "config.local.json",
            {"permissions": {"allow": list(_REVIEW_EXEC_ALLOWLIST)}},
        )
    except OSError as exc:
        logger.warning(
            "could not write review exec allow-list in %s: %s (reviewer may hit "
            "an exec rejection)",
            checkout_path,
            exc,
        )


# Issue #2024: the allow-list alone does not stop a reviewer from *trying* a
# refused command -- Gemini re-ran ``uv run pytest`` on PR #2014 three reviews
# in a row, and each refusal ended the session with no verdict. So the review
# prompt states the same list, rendered from ``_REVIEW_EXEC_ALLOWLIST`` (never a
# second hand-kept list), plus what will be refused and why that is fine.
# Issue #2032: the section must present that list as the ONLY runnable set.
# Devin's own read-only auto-approval outside the list is unpredictable
# (`git log` passed nine times, and `git log -n 5 --stat` was refused), and
# advertising git reads as "normally allowed" cost 3 of the first 5 reviews.
REVIEW_EXEC_SECTION_HEADING = "## Shell commands in this review session"


def _review_exec_prompt_section() -> str:
    commands = ", ".join(f"`{entry[len('Exec(') : -1]}`" for entry in _REVIEW_EXEC_ALLOWLIST)
    return (
        f"{REVIEW_EXEC_SECTION_HEADING}\n\n"
        "This session is headless and read-only. Commands that need approval are "
        "REFUSED, and a refusal can end the session before you emit a verdict -- "
        "the review is then lost.\n\n"
        f"- The ONLY shell commands you may run: {commands}. Run nothing else -- not "
        "other `git` subcommands (not even read-only ones such as `git worktree` or "
        "`git log --stat`), not `sed`, `awk`, `jq`, and not a pipe into any "
        "unlisted program. Unlisted commands are refused unpredictably.\n"
        "- Read files (including this packet's `pr.json`, `diff.patch` and "
        "`interdiff.patch`) with your file-reading and search tools, not the shell. "
        "The packet already contains the PR's diff and metadata.\n"
        "- Do NOT run tests, linters, interpreters, or package managers (`pytest`, "
        "`uv`, `python`, `ruff`, `npm`, ...). CI has already run them: results are "
        "in the CI status section and `checks.json` of this packet. Judge test "
        "adequacy by reading the tests.\n"
    )


def _write_devin_review_prompt(prompt_path: Path) -> Path:
    """Return a sibling prompt with the exec section appended (issue #2024).

    The shared ``review-prompt.md`` packet file is left untouched -- other
    harnesses read it, and this section is devin-specific. Never raises: on
    any OSError the original prompt is returned, which is the pre-#2024
    behavior (the allow-list still applies).
    """
    derived = prompt_path.with_name(f"{prompt_path.stem}.devin{prompt_path.suffix}")
    try:
        text = prompt_path.read_text(encoding="utf-8")
        tmp = derived.with_suffix(derived.suffix + ".tmp")
        tmp.write_text(f"{text.rstrip()}\n\n{_review_exec_prompt_section()}", encoding="utf-8")
        tmp.replace(derived)
    except OSError as exc:
        logger.warning(
            "could not write devin review prompt %s: %s (using %s without the exec section)",
            derived,
            exc,
            prompt_path,
        )
        return prompt_path
    return derived


def _sanitize_review_command_template(command_template: tuple[str, ...]) -> tuple[str, ...]:
    """Hard-pin the read-only reviewer posture onto ``command_template``.

    A review launch must never carry ``--permission-mode dangerous`` -- this
    is an invariant, not a default a caller-supplied (or config-forwarded)
    ``command_template`` can defeat. ``DevinConfig.command`` is a single field
    shared with worker dispatch (workers need ``dangerous`` for write
    access); if an operator's worker-tuning override were honored verbatim
    for reviewers too, it would silently grant write access, defeating
    ``create_review_checkout``'s no-write guarantee. Mirrors
    ``claude_code._sanitize_review_command_template``'s pinning of
    ``--permission-mode plan`` for the identical reason.

    Every occurrence of ``--permission-mode`` (the flag plus its following
    value token, and any ``--permission-mode=<value>`` form) is stripped.
    Unlike claude-code's ``plan`` mode, nothing is appended in its place: the
    Devin CLI's own default when ``--permission-mode`` is entirely absent
    from argv is ``auto`` (read-only tools), which is the posture wanted here
    -- there is no "dangerous-but-read-only" flag value to pin to instead.
    """
    filtered: list[str] = []
    skip_next = False
    for token in command_template:
        if skip_next:
            skip_next = False
            continue
        if token == "--permission-mode":
            skip_next = True
            continue
        if token.startswith("--permission-mode="):
            continue
        filtered.append(token)
    return tuple(filtered)
