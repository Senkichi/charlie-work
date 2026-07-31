"""Single point of enforcement for git SHA / ref-name format validation.

Companion to ``safe_path.py`` (path-containment). Where ``safe_path`` guards
against directory traversal, this module guards against argument-injection via
git argv: a value read from persisted ``state.json`` or a GitHub API response
is format-checked *before* it reaches any ``subprocess`` argv list, so a
future refactor that drops an incidental protection (e.g. the ``^`` prefix on
``reviewed_head_sha`` in ``janitor._check_no_op_rework``) cannot silently turn
an attacker-influenced value into a parsed flag.

The checks are format-only (defense-in-depth, see issue #659). Git's own
ref/SHA naming rules already reject flag-like strings today — a ref or SHA
cannot start with ``-`` — so none of the validated sites are exploitable now.
These validators exist so that stays true after future refactors.

Git ref-name rules reference: ``git check-ref-format --help``.
"""

from __future__ import annotations

import re

# Git object SHA: abbreviated (min 4) to full (SHA-1 = 40, SHA-256 = 64).
# Hex-only means a flag prefix (``-``) is structurally impossible.
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{4,64}\Z")

# Git ref-name: conservative allowlist covering all real branch names while
# rejecting every rev-syntax metacharacter. First char must be alphanumeric
# (rejects leading ``-`` for flag-injection prevention, and leading ``.``/``/``
# per git check-ref-format rules). Subsequent chars: alphanumerics plus
# ``. _ / - +``. Metacharacters ``~ ^ : ? * [ \`` and whitespace/control chars
# are excluded by the allowlist; ``..`` and ``@{`` are checked separately.
_REF_NAME_RE = re.compile(r"\A[0-9A-Za-z][0-9A-Za-z._+/-]*\Z")


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

    Rejects anything starting with ``-`` (flag injection) or containing
    rev-syntax metacharacters, ``..``, ``@{``, or trailing ``/``/``.``.
    Conservative allowlist: alphanumerics plus ``. _ / - +``. See issue #659.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: ref name is empty or not a string")
    if not _REF_NAME_RE.match(value):
        raise ValueError(
            f"{context}: {value!r} is not a valid git ref name "
            f"(must start alphanumeric; only alphanumerics, '.', '_', '/', '-', '+' allowed)"
        )
    if ".." in value:
        raise ValueError(f"{context}: {value!r} contains '..' (not a valid git ref name)")
    if value.endswith("/") or value.endswith("."):
        raise ValueError(f"{context}: {value!r} ends with '/' or '.' (not a valid git ref name)")
    if "@{" in value:
        raise ValueError(f"{context}: {value!r} contains '@{{' (not a valid git ref name)")
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
