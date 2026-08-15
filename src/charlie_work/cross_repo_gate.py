"""Pre-flight gate: refuse to dispatch when an issue's referenced code is not in the target repo.

Issue #1010: a dispatched worker edited a sibling repo's shared main checkout
because the issue's subject code (``suite_coverage.py``) does not exist in the
repo it was dispatched against.  The orchestrator created a proper isolated
worktree, but the file the worker was asked to change was not in it — so the
worker went to the sibling repo's shared checkout and worked there,
contaminating another agent's PR.

This module implements the "pre-flight gate" from the issue's proposed fix
(1): at dispatch time, extract file-path references from the issue body and
check whether any of them exist in the target repo.  If the issue references
file paths but *none* of them exist in the repo, the gate returns ``False``
— the caller should escalate to ``agent:human-needed`` with a
``cross_repo_target`` reason instead of burning a worker and a slot.

The gate is conservative by design:

- An issue that references **no** file paths passes (no evidence of a
  cross-repo target).
- An issue where **at least one** referenced path exists in the repo passes
  (the worker has something to work on here).
- Only an issue that references file paths where **none** exist in the repo
  is blocked.

Escalating to ``human-needed`` is a safe failure mode: a human can re-label
the issue after confirming the target repo, so a false positive costs one
manual triage action rather than a contaminated sibling checkout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .safe_path import contains

# A file extension: 1-10 word characters after a dot.  Bounds the length so
# the regex does not match version strings like ``1.2.3.4.5.6.7.8.9.0``.
_EXT = r"\.[a-zA-Z][a-zA-Z0-9]{0,9}"

# A relative path with at least one path separator and a file extension.
# Requires at least 2 segments (e.g. ``src/foo.py``, ``ci_fleet/suite_coverage.py``)
# to avoid matching bare filenames like ``main.py`` that may appear in prose
# without being file-path references.
_REL_PATH = rf"(?<![\w/.])((?:[\w.-]+/)+[\w.-]+{_EXT})(?![\w])"

# An absolute path: drive letter (Windows) or leading ``/`` (POSIX), followed
# by path segments and a file extension.
_ABS_WIN_PATH = rf"(?<![\w])([A-Za-z]:[\\/](?:[\w.-]+[\\/])+[\w.-]+{_EXT})(?![\w])"
_ABS_POSIX_PATH = rf"(?<![\w])(/(?:[\w.-]+/)+[\w.-]+{_EXT})(?![\w])"

# Backtick-quoted paths: `` `...ext` `` — catches paths quoted in markdown
# regardless of whether they are relative or absolute.
_TICK_PATH = rf"`([^`]*(?:/|\\)[^`]*{_EXT})`"

_PATH_RE = re.compile(
    "|".join((_TICK_PATH, _ABS_WIN_PATH, _ABS_POSIX_PATH, _REL_PATH)),
)


@dataclass(frozen=True)
class CrossRepoGateResult:
    """Outcome of the cross-repo pre-flight gate.

    Attributes:
        passed: ``True`` when the issue should be dispatched, ``False`` when
            it should be escalated to ``agent:human-needed``.
        referenced_paths: every candidate file path extracted from the issue
            body, as raw strings (may include paths that do not exist
            anywhere).
        missing_paths: the subset of ``referenced_paths`` that do not exist
            in the target repo.  When ``passed`` is ``False``, this equals
            ``referenced_paths`` (every referenced path was missing).
        reason: human-readable explanation for the gate's decision, suitable
            for an event payload.
    """

    passed: bool
    referenced_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    reason: str


def extract_referenced_paths(issue_body: str) -> list[str]:
    """Extract candidate file-path references from an issue body.

    Returns a de-duplicated list of raw path strings, preserving first-seen
    order.  Paths inside backtick quotes, absolute paths (Windows drive-letter
    or POSIX leading-slash), and relative paths with at least one ``/``
    separator and a file extension are all candidates.

    URLs (``http://``, ``https://``) are excluded — they are not file paths.
    """
    # Strip URLs before matching so the POSIX absolute-path alternation does
    # not capture the path portion of ``https://example.com/foo.py``.
    stripped = re.sub(r"https?://\S+", "", issue_body)
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _PATH_RE.finditer(stripped):
        # The regex has four alternation groups; pick the one that matched.
        raw = next((g for g in match.groups() if g is not None), "")
        if not raw:
            continue
        if raw not in seen:
            seen.add(raw)
            candidates.append(raw)
    return candidates


def _path_exists_in_repo(path_str: str, repo_root: Path) -> bool:
    """Return ``True`` when ``path_str`` resolves to an existing file inside ``repo_root``."""
    path = Path(path_str)
    if path.is_absolute():
        try:
            if not path.exists():
                return False
            return contains(repo_root, path)
        except (OSError, ValueError):
            return False
    # Relative path: resolve against the repo root.
    resolved = repo_root / path
    try:
        return resolved.exists()
    except OSError:
        return False


def cross_repo_gate(issue_body: str, repo_root: Path) -> CrossRepoGateResult:
    """Decide whether an issue should be dispatched or escalated as cross-repo.

    Returns a :class:`CrossRepoGateResult` with ``passed=True`` when the issue
    is safe to dispatch (it references no file paths, or at least one
    referenced path exists in ``repo_root``), and ``passed=False`` when every
    referenced path is missing from ``repo_root`` — the signal that the
    issue's subject code lives in a different repo.
    """
    referenced = extract_referenced_paths(issue_body)
    if not referenced:
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=(),
            missing_paths=(),
            reason="no file paths referenced in issue body",
        )
    missing = tuple(p for p in referenced if not _path_exists_in_repo(p, repo_root))
    # Block only when EVERY referenced path is absent — the issue's subject
    # code is not in this repo at all.  If even one referenced path exists,
    # the worker has something to work on here and the gate passes.
    if len(missing) < len(referenced):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(referenced),
            missing_paths=missing,
            reason="at least one referenced path exists in the target repo",
        )
    # Every referenced path is missing — the issue's subject code is not in
    # this repo.  Escalate instead of dispatching a worker that will wander.
    return CrossRepoGateResult(
        passed=False,
        referenced_paths=tuple(referenced),
        missing_paths=missing,
        reason=(
            f"cross_repo_target: all {len(missing)} referenced file path(s) "
            f"are absent from the target repo ({repo_root})"
        ),
    )


__all__ = [
    "CrossRepoGateResult",
    "cross_repo_gate",
    "extract_referenced_paths",
]
