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
- An issue with **exactly one** referenced path, that path missing, and the
  path not shaped like a reference to this repo (not a relative path whose
  first segment names a real top-level directory here) also passes — a
  single ambiguous fragment pulled out of prose is weak evidence on its own,
  and escalating on it wastes a human triage action for no reason.
- Otherwise — every referenced path is missing, and either there are
  multiple such paths or the sole path is repo-shaped-but-missing — the
  issue is blocked.

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

# A scheme-less domain-shaped token followed by a path, e.g.
# ``pultegroupinc.com/careers/default.aspx``.  This is not a file-path
# reference — it is a URL fragment with the ``https://`` scheme dropped (as
# happens routinely when a URL is pasted into prose or a markdown table
# cell). Mirrors the ``https?://`` strip below: the "host" portion is one or
# more dot-separated labels ending in an alpha-only 2-24 char label (TLD-
# shaped), immediately followed by ``/`` and the rest of the token. Stripped
# as a whole (host + path) before path extraction runs, exactly like the
# ``https?://\S+`` strip removes the scheme *and* its path together. The
# token stops at whitespace and markdown structure characters (``|``
# table-cell delimiters, backticks, closing brackets/parens) so a domain
# token packed tightly against a real path in a table cell —
# ``|domain.com/x.aspx|src/real.py |`` — does not swallow its neighbor.
_DOMAIN_PATH = r"\b(?:[\w-]+\.)+[a-zA-Z]{2,24}/[^\s|`)\]]*"


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
    Scheme-less domain-shaped tokens (e.g. a bare ``example.com/path`` pasted
    into prose, with no ``https://`` prefix) are excluded for the same
    reason.
    """
    # Strip URLs before matching so the POSIX absolute-path alternation does
    # not capture the path portion of ``https://example.com/foo.py``.
    stripped = re.sub(r"https?://\S+", "", issue_body)
    # Strip scheme-less domain-shaped tokens the same way — a "path" whose
    # leading segment is actually a hostname is not a file-path reference.
    stripped = re.sub(_DOMAIN_PATH, "", stripped)
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


def _top_level_dirs(repo_root: Path) -> set[str]:
    """Return the names of ``repo_root``'s immediate subdirectories.

    Derived dynamically from the live filesystem — never a hardcoded list —
    so the "is this a real repo-relative path" check tracks the repo's
    actual structure.
    """
    try:
        return {entry.name for entry in repo_root.iterdir() if entry.is_dir()}
    except OSError:
        return set()


def _is_absolute_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` is absolute, on any host platform.

    ``Path(candidate).is_absolute()`` alone is platform-dependent in a way
    that matters here: on Windows, ``PureWindowsPath`` only counts a path as
    absolute when it carries a drive letter, so a POSIX-style absolute path
    like ``/home/user/other-repo/foo.py`` reports ``is_absolute() is False``.
    Left unguarded, that misclassifies a genuinely absolute (and genuinely
    outside-the-repo) candidate as "relative", which would let it reach the
    repo-shape check, fail it (its "first segment" is the empty string
    before the leading slash), and incorrectly abstain instead of escalate.
    A leading path separator is unambiguously absolute regardless of host
    platform, so it is treated as absolute here even where ``Path.is_absolute``
    disagrees.
    """
    return Path(candidate).is_absolute() or bool(re.match(r"[\\/]", candidate))


def _is_repo_shaped_relative_candidate(candidate: str, repo_root: Path) -> bool:
    """Return ``True`` when ``candidate`` is a *relative* path whose first
    path segment names a directory that actually exists in ``repo_root``.

    Absolute candidates are never repo-shaped by this definition — callers
    that need to keep escalating on a missing absolute path must check
    ``_is_absolute_path(candidate)`` themselves before consulting this
    function (see the single-candidate exception in :func:`cross_repo_gate`).

    This is the shape test that distinguishes a genuine (but missing)
    repo-relative reference like ``src/charlie_work/nonexistent.py`` (first
    segment ``src`` is a real top-level directory) from an unrelated
    relative-looking token like ``Scripts/charlie.exe`` (first segment
    ``Scripts`` names no directory in this repo — it is a venv path, not a
    reference to this repo's code).
    """
    if _is_absolute_path(candidate):
        return False
    first_segment = re.split(r"[\\/]+", candidate, maxsplit=1)[0]
    return first_segment in _top_level_dirs(repo_root)


def cross_repo_gate(issue_body: str, repo_root: Path) -> CrossRepoGateResult:
    """Decide whether an issue should be dispatched or escalated as cross-repo.

    Returns a :class:`CrossRepoGateResult` with ``passed=True`` when the issue
    is safe to dispatch (it references no file paths, or at least one
    referenced path exists in ``repo_root``), and ``passed=False`` when every
    referenced path is missing from ``repo_root`` — the signal that the
    issue's subject code lives in a different repo.

    A single-candidate exception applies: when extraction found exactly one
    candidate and it is missing, the gate abstains (``passed=True``) unless
    that candidate is a relative path whose first segment names a directory
    that actually exists in ``repo_root`` (see
    :func:`_is_repo_shaped_relative_candidate`). One ambiguous fragment
    pulled out of prose (a venv-relative path, a config key, anything that
    merely *looks* path-shaped) is weak evidence of a cross-repo target on
    its own — escalating on it wastes a human triage action for no reason.
    An absolute path resolving outside the repo is not ambiguous in the same
    way and keeps escalating regardless of this exception.
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
    # NOTE: absoluteness is checked via ``_is_absolute_path``, not a bare
    # ``Path(...).is_absolute()`` — the latter is platform-dependent (a
    # POSIX-style absolute path is not "absolute" under Windows' drive-letter
    # rule) and would let a genuinely-absolute, outside-the-repo candidate
    # slip through as "relative" and incorrectly abstain. See
    # ``_is_absolute_path`` for the full rationale.
    if (
        len(referenced) == 1
        and not _is_absolute_path(referenced[0])
        and not _is_repo_shaped_relative_candidate(referenced[0], repo_root)
    ):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(referenced),
            missing_paths=missing,
            reason=(
                "single ambiguous candidate "
                f"({referenced[0]!r}) is not a repo-shaped relative path — "
                "abstaining rather than escalating on weak evidence"
            ),
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
