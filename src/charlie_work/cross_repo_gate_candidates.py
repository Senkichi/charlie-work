"""Candidate-shape machinery for the cross-repo gate.

Extracted from ``cross_repo_gate.py`` so the gate module stays under its
file-size high-water mark (issue #1442) — same split as
``cross_repo_gate_shorthand.py``: this module holds the "is this string a
path candidate at all" layer — the ``_PATH_RE`` extraction regex, the
placeholder/glob/launcher-owned drop filters, and the issue-#1790
whitespace-joined span splitter. The gate module owns the pipeline that
runs them (``_iter_candidate_matches``) and the pass/escalate verdicts.

Everything here is pure text/regex work — no I/O, no git, no imports back
into the gate module (a leaf module, so the dependency direction stays
one-way).
"""

from __future__ import annotations

import re

from .config import LAUNCHER_OWNED_DIRS

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

# A placeholder path segment: a template stand-in that can never name a real
# file. Issue #1343: a templated documentation path in an issue body (e.g. a
# runtime-state example ``<state-dir>/prs/pr-N/review-decision.json``) was
# extracted as a path candidate and, because the runtime state dir is a real
# top-level directory in the checkout, false-positived as a cross-repo
# target. A placeholder segment can never be a genuine cross-repo reference,
# so candidates containing one are dropped before existence checks.
#
# Matches either:
#   - an angle-bracket placeholder (``<state-dir>``, ``<pr-N>``, ``<...>``) —
#     any segment containing ``<`` or ``>``; or
#   - a placeholder-numbered segment (``pr-N``, ``issue-N``) — one or more
#     letters, a dash, then a literal capital ``N`` standing in for an
#     unknown number.
_PLACEHOLDER_SEGMENT = re.compile(r"^(?:[A-Za-z]+-N|.*[<>].*)$")

# Embedded whitespace of any width (issue #1756, and the single-space
# shape found live in cw #1518, ``_TICK_PATH`` capturing
# `` `tests/a.py tests/b.py` `` as one corrupted candidate): no real
# relative or absolute file path contains a space, tab, or line break. This
# is deliberately wider than #1756's own proposed "\r, \n, or 2+ whitespace"
# filter — a single embedded space is not "2+ whitespace" and would slip
# that narrower version, but is exactly the same class of corrupted,
# can-never-exist candidate.
#
# Issue #1790: a whitespace-containing candidate is no longer dropped
# unconditionally — it is first split on whitespace and each piece re-run
# through the normal pipeline (:func:`_split_whitespace_candidate`), so a
# genuine two-paths-in-one-span citation (the cw #1518 shape) is recovered
# as two candidates. The whole candidate is still dropped when the split
# yields fewer than 2 path-shaped pieces — the #1756 corrupted-single-path
# shape produces at most one.
_EMBEDDED_WHITESPACE_RE = re.compile(r"\s")

# Glob metacharacters: ``*``, ``?``, ``[``, ``]``. A candidate containing any
# of these is a glob pattern, not a literal file path — no file literally
# named ``*.py`` exists, so a glob candidate is always "missing" and would
# false-positive the gate. The backtick-quoted path regex (``_TICK_PATH``)
# captures globs because it matches anything inside backticks that contains a
# separator and ends with an extension; the non-tick regexes exclude ``*``
# from their character classes, so only backtick-quoted globs reach this
# filter (issue #1391).
_GLOB_METACHAR = re.compile(r"[*?\[\]]")


def _has_placeholder_segment(candidate: str) -> bool:
    """Return ``True`` when any segment of ``candidate`` is a template placeholder.

    A placeholder segment (``<state-dir>``, ``pr-N``, ``<...>``) can never
    name a real file — it is documentation template text, not a cross-repo
    reference.  Candidates containing one are dropped before existence
    checks so a templated example path in an issue body cannot fire the gate
    (issue #1343).
    """
    return any(_PLACEHOLDER_SEGMENT.match(seg) for seg in re.split(r"[\\/]+", candidate))


def _is_launcher_owned_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` is under a launcher-owned worktree dir.

    Paths under ``.devin/`` or ``.git_worktree_dir/`` live only inside agent
    worktrees — the shim materializes them on every dispatch — not in the
    repo tree. A candidate whose first path segment names one of these
    directories is not evidence of a cross-repo target: it will always be
    "missing" from the repo and would false-positive the gate (issue #1391).

    The launcher-owned directory set is sourced from
    :data:`charlie_work.config.LAUNCHER_OWNED_DIRS`, shared with
    :mod:`charlie_work.worktree`'s dirty check so the two modules share one
    definition of "launcher-owned, not evidence."
    """
    segments = re.split(r"[\\/]+", candidate, maxsplit=1)
    return bool(segments) and segments[0] in LAUNCHER_OWNED_DIRS


def _is_dropped_candidate(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` fails a drop filter.

    The single predicate behind the drop filters in
    ``cross_repo_gate._iter_candidate_matches``, shared with
    :func:`_split_whitespace_candidate` so a recovered piece faces exactly
    the same drops as a directly extracted candidate:

    - a placeholder segment (``pr-N``, ``<state-dir>``) is documentation
      template text, never a real file (issue #1343);
    - a glob metacharacter (``*``, ``?``, ``[``, ``]``) makes the candidate
      a pattern, not a literal path — always "missing" (issue #1391);
    - a launcher-owned first segment (``.devin/``, ``.git_worktree_dir/``)
      names a path that exists only inside agent worktrees, not in the
      repo tree (issue #1391).
    """
    return (
        _has_placeholder_segment(candidate)
        or _GLOB_METACHAR.search(candidate) is not None
        or _is_launcher_owned_path(candidate)
    )


def _split_whitespace_candidate(raw: str) -> list[str]:
    """Return the path-shaped pieces of a whitespace-joined candidate.

    Issue #1790: the embedded-whitespace filter (issue #1756, cw #1518)
    dropped an entire candidate the moment it contained any whitespace —
    including the deliberate cw #1518 shape where two real, distinct paths
    are cited together inside one backtick span separated by a single
    space (`` `tests/a.py tests/b.py` ``). Both paths were discarded rather
    than classified, silently losing the positive sibling-repo evidence
    either could have supplied.

    Each whitespace-separated piece is re-run through the same filters an
    un-split candidate faces in ``_iter_candidate_matches``: the
    ``_PATH_RE`` extraction shape check, then the shared drop predicate
    :func:`_is_dropped_candidate`. One asymmetry is structural, not a
    second filter chain: a whitespace-free piece contains no backticks, so
    only the non-tick ``_PATH_RE`` alternations can match it — and their
    character classes exclude glob metacharacters, so the predicate's glob
    arm can never fire on a piece (only ``_TICK_PATH`` captures globs,
    issue #1391).

    The call site reports every emitted piece at the whole span's
    ``(start, end)`` offsets rather than at per-piece offsets: the
    backtick span is a single citation unit, so a clause-local evidence
    marker, a suffix marker, or a citation-section heading that describes
    the span describes every piece of it. Per-piece offsets would let the
    first piece's own ``.ext`` period sever the clause for the second
    piece (``.`` is a clause boundary), inconsistently neutralizing only
    half of one citation.

    Falls back to the pre-#1790 "drop the whole thing" behavior — an empty
    list — unless the split yields 2+ path-shaped pieces: a single real
    path hard-wrapped mid-token (`` `dir/\\na.py` ``) or followed by prose
    inside the span yields at most one path-shaped piece, so the #1756
    corrupted-candidate protection is preserved.
    """
    path_shaped = 0
    pieces: list[str] = []
    for span in re.finditer(r"\S+", raw):
        for sub in _PATH_RE.finditer(span.group(0)):
            piece = next((g for g in sub.groups() if g is not None), "")
            if not piece:
                continue
            path_shaped += 1
            if _is_dropped_candidate(piece):
                continue
            pieces.append(piece)
    if path_shaped < 2:
        return []
    return pieces
