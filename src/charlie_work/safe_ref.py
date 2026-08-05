"""Single point of enforcement for git SHA / ref-name format validation.

Companion to ``safe_path.py`` (path-containment). Where ``safe_path`` guards
against directory traversal, this module guards against argument-injection via
git argv: a value read from persisted ``state.json`` or a GitHub API response
is format-checked *before* it reaches any ``subprocess`` argv list, so a
future refactor that drops an incidental protection (e.g. the ``^`` prefix on
``reviewed_head_sha`` in ``janitor._check_no_op_rework``) cannot silently turn
an attacker-influenced value into a parsed flag.

The SHA check is a pure format guard. The ref-name check mirrors the rules
enforced by ``git check-ref-format --allow-onelevel`` without shelling out,
so it can be used safely before any git argv is built (issue #659).

Git ref-name rules reference: ``git check-ref-format --help``.
"""

from __future__ import annotations

import re

# Git object SHA: abbreviated (min 4) to full (SHA-1 = 40, SHA-256 = 64).
# Hex-only means a flag prefix (``-``) is structurally impossible.
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{4,64}\Z")

# Forbidden characters in a git ref name: the set documented by
# ``git check-ref-format`` as rev-syntax / glob metacharacters.
_REF_FORBIDDEN_CHARS = frozenset("~^:?*[\\")

# ASCII whitespace that git rejects in ref names.
_REF_WHITESPACE = frozenset(" \t\n\r\f\v")


def _has_control_char(value: str) -> bool:
    return any(ord(c) < 32 or ord(c) == 127 for c in value)


def require_valid_sha(value: str, *, context: str) -> str:
    """Return ``value`` if it is a valid git object SHA, else raise ``ValueError``.

    Accepts 4-64 hex characters (covers abbreviated through full SHA-256).
    This is the defense-in-depth boundary that prevents a persisted or
    API-sourced value from being parsed as a git flag if a future refactor
    drops incidental prefix protection (see issue #659).
    """
    if not isinstance(value, str) or not _SHA_RE.match(value):
        raise ValueError(
            f"{context}: {value!r} is not a valid git object SHA (expected 4-64 hex characters)"
        )
    return value


def require_valid_ref_name(value: str, *, context: str) -> str:
    """Return ``value`` if it is a valid git ref name, else raise ``ValueError``.

    Enforces the same structural rules as ``git check-ref-format
    --allow-onelevel``: no leading ``.``/``-``/``/``, no ``..`` or ``@{``,
    no rev-syntax metacharacters, no control/whitespace, no empty path
    components, no ``.``-prefixed path components, no ``.lock``-suffixed path
    components, and no trailing ``.`` or ``/``. This tracks git's actual
    ref-name rules instead of a hand-maintained allowlist (issue #659).
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: ref name is empty or not a string")
    if value == "@":
        # "@" alone is the HEAD reflog shorthand, not a valid ref name.
        raise ValueError(f"{context}: {value!r} is not a valid git ref name")
    if value.startswith((".", "-", "/")):
        raise ValueError(f"{context}: {value!r} is not a valid git ref name")
    if value.endswith((".", "/")):
        raise ValueError(f"{context}: {value!r} ends with '/' or '.' (not a valid git ref name)")
    if ".." in value:
        raise ValueError(f"{context}: {value!r} contains '..' (not a valid git ref name)")
    if "@{" in value:
        raise ValueError(f"{context}: {value!r} contains '@{{' (not a valid git ref name)")
    if _has_control_char(value):
        raise ValueError(f"{context}: {value!r} is not a valid git ref name")
    if any(c in _REF_FORBIDDEN_CHARS or c in _REF_WHITESPACE for c in value):
        raise ValueError(f"{context}: {value!r} is not a valid git ref name")
    for part in value.split("/"):
        if not part:
            raise ValueError(f"{context}: {value!r} is not a valid git ref name")
        if part.startswith("."):
            raise ValueError(f"{context}: {value!r} is not a valid git ref name")
        if part.endswith(".lock"):
            raise ValueError(f"{context}: {value!r} is not a valid git ref name")
    return value


def require_valid_rev(value: str, *, context: str) -> str:
    """Return ``value`` if it is a valid git revision (SHA or ref name).

    For parameters like ``create_worktree``'s ``base_ref`` that legitimately
    accept either a commit SHA (e.g. ``"a1b2c3d4..."``) or a ref name
    (e.g. ``"HEAD"``, ``"origin/main"``). An empty string is rejected —
    callers that use ``""`` as a sentinel must check before calling.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: revision is empty or not a string")
    if _SHA_RE.match(value):
        return value
    return require_valid_ref_name(value, context=context)


__all__ = ["require_valid_ref_name", "require_valid_rev", "require_valid_sha"]
