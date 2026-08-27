"""Pre-dispatch verification of ``path:line`` citations in issue bodies (issue #1000).

Issues in this repo cite defects as ``workflow.py:4746``. ``workflow.py`` is
~19,000 lines and changes on most merges, so those line numbers rot fast — fast
enough that an issue filed one day can point a worker at unrelated code the next.
The measured rate (issue #1000): 6 of 8 dispatch-ready issues needed correction,
and a single ordinary merge invalidated citations in 4 of 13 queued issues.

This module is the "cheap check" half of the fix. The other half is a filing
convention (see ``CONTRIBUTING.md``): **cite the symbol, not the line**, and
**stamp the commit** the citation was read against. A symbol citation
(``_collect_external_findings`` in ``workflow.py``) is stable across every edit
that does not rename it; a bare ``workflow.py:4746`` is stale the next time
anything above it grows. The convention prevents the drift; this check flags
what the convention did not prevent.

What the check can and cannot catch
-----------------------------------
It mechanically detects **coordinate drift**:

* ``FILE_MISSING`` — the cited file no longer exists at that path (renamed /
  deleted between filing and dispatch).
* ``OUT_OF_RANGE`` — the cited line number is beyond the file's current length
  (the file shrank, or the citation was always wrong).
* ``EMPTY_LINE`` — the cited line is blank; a citation meant to point at code
  now points at a gap between blocks, a strong drift signal.
* ``STALE_PREFIX`` — the citation carries a directory prefix whose asserted
  literal path does not exist, but the basename resolves uniquely via fallback
  (the file moved into a new subdirectory between filing and dispatch). The
  verdict's ``resolved_path`` surfaces where the file actually is. Bare
  (no-prefix) citations that resolve uniquely report ``RESOLVED_BY_BASENAME``
  instead -- only an asserted-and-now-false prefix flags as drift.
* ``RESOLVED_BY_BASENAME`` — info-level (not drift): the cited literal path
  does not exist, but exactly one tracked file shares the basename and the
  line range validates against it. The citation is usable, just imprecise
  (a bare basename rather than a full path). Excluded from
  ``drifted_verdicts`` so it surfaces without raising a false alarm.
* ``AMBIGUOUS_BASENAME`` — the cited literal path does not exist and more
  than one tracked file shares the basename. This is a real citation defect:
  the author must disambiguate with a directory prefix. The verdict's
  ``candidates`` tuple lists every match.

It **cannot** detect *in-range content drift* — the cited line is still a valid
line number but now contains unrelated, plausible code (the dangerous band in
issue #1000: a +148 shift lands on a valid line that reads as a plausible
"already fixed"). Distinguishing that from a correct citation requires comparing
the current line against the content at the stamped commit, which needs the
filing convention's commit stamp to be adopted first. ``verify_citations``
accepts an optional ``commit_sha`` + ``fetch_file_lines_at_commit`` so a caller
that has the stamp can do that comparison (``CONTENT_DRIFT``); the dispatch
integration does not wire this up yet and relies on the convention to prevent
content drift. The asymmetry is deliberate and documented: the convention is
the durable fix, the check is a backstop for issues that still cite bare line
numbers.

The check is pure: it reads the issue body and the working tree, makes no
network calls, writes nothing, and never raises on a missing file — that is a
verdict, not an error.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

__all__ = [
    "Citation",
    "CitationStatus",
    "CitationVerdict",
    "drift_fingerprint",
    "drifted_verdicts",
    "format_verdict_status_cell",
    "parse_citations",
    "verify_citations",
]


class CitationStatus(str, Enum):
    """Verdict on a single ``path:line`` citation.

    ``OK`` and ``RESOLVED_BY_BASENAME`` are the non-drift statuses. Every other
    value means the citation's coordinates no longer match the working tree and
    a worker should not trust the bare line number. ``RESOLVED_BY_BASENAME`` is
    info-level: the cited literal path did not exist, but a unique tracked file
    with that basename was found and the line range validates against it -- the
    citation is usable, just imprecise (a bare basename rather than a full
    path). It surfaces the citation-quality issue without a false missing-file
    claim, and is excluded from ``drifted_verdicts`` so it does not raise a
    drift alarm.
    """

    OK = "ok"
    FILE_MISSING = "file_missing"
    OUT_OF_RANGE = "out_of_range"
    EMPTY_LINE = "empty_line"
    STALE_PREFIX = "stale_prefix"
    CONTENT_DRIFT = "content_drift"
    RESOLVED_BY_BASENAME = "resolved_by_basename"
    AMBIGUOUS_BASENAME = "ambiguous_basename"


@dataclass(frozen=True)
class Citation:
    """A ``path:line`` (or ``path:start-end``) token parsed from an issue body."""

    raw: str
    path: str
    line: int
    end_line: int  # equal to ``line`` for a single-line citation


@dataclass(frozen=True)
class CitationVerdict:
    """The result of verifying one citation against the working tree."""

    citation: Citation
    status: CitationStatus
    current_line_text: str | None = None
    original_line_text: str | None = None
    resolved_path: str | None = None
    # Candidate paths for ``AMBIGUOUS_BASENAME`` -- the tracked files whose
    # basename matched the cited (literal-path-missing) citation. Populated
    # only for that status so the drift comment can surface the disambiguation
    # the author must perform. Stored as repo-root-relative POSIX strings.
    candidates: tuple[str, ...] | None = None


# A path-like token: an optional directory prefix, then a filename that carries
# an extension (the ``.\\w{1,6}`` requires a dot, which is what makes
# ``workflow.py:4746`` a citation and ``13:34`` a timestamp). The leading
# ``(?<![\\w/.:-])`` boundary stops the match inside a longer path or token
# (``src/charlie_work/workflow.py:4746`` matches as one citation, not two), and
# the ``:`` in the boundary class rejects URL scheme prefixes: in
# ``https://example.com:8080`` the ``//example.com`` token is preceded by ``:``
# (the ``://`` separator), so it cannot start a citation.
_CITATION_RE = re.compile(
    r"(?<![\w/.:-])"
    r"(?P<path>(?:[\w./-]+/)?[\w-]+\.\w{1,6})"
    r":(?P<start>\d{1,6})"
    r"(?:-(?P<end>\d{1,6}))?"
    r"(?![\d])"
)
# A commit stamp: a 7..40-char hex token. The dispatch integration does not
# require a stamp, but when one is present a caller can wire up content
# comparison. Kept permissive (7+) so short SHAs in prose still match.
_SHA_RE = re.compile(r"(?<![\w])(?P<sha>[0-9a-f]{7,40})(?![\w])", re.IGNORECASE)


def parse_citations(body: str) -> list[Citation]:
    """Return every ``path:line`` / ``path:start-end`` citation in ``body``.

    Deduplicated and order-preserving by ``(path, line, end_line)``. URL
    schemes (``https://example.com:8080``) are rejected by the regex's
    lookbehind boundary (``(?<![\\w/.:-])`` — the ``:`` in ``://`` precedes
    the path token, so the boundary fails). A citation is *parsed* on shape
    alone; whether it is *verified* is a separate step (``verify_citations``)
    that needs the working tree.
    """
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(body):
        path = m.group("path")
        start = int(m.group("start"))
        end_s = m.group("end")
        end = int(end_s) if end_s is not None else start
        # A range that runs backwards is not a real citation range.
        if end < start:
            continue
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), path=path, line=start, end_line=end))
    return out


def _git_ls_files(repo_root: Path) -> list[Path] | None:
    """Return every tracked file path via ``git ls-files``, or ``None`` on failure.

    Uses the git index rather than a filesystem walk so the basename index
    covers the *entire* tracked tree -- not a hardcoded set of source roots.
    A repo whose real source files live outside ``src/``/``scripts/``/``tests/``
    (e.g. job-cannon's ``job_finder/`` tree) is fully covered here, which is the
    root cause this function exists: the prior hardcoded-root walk missed those
    files and every bare-basename citation reported ``file_missing`` (issue
    #1452). Returns ``None`` when git is unavailable or ``repo_root`` is not a
    git worktree (e.g. a pytest ``tmp_path``); the caller falls back to a tree
    walk in that case. Never raises -- a subprocess failure is a value.
    """
    from .subprocess_runner import run_captured

    res = run_captured(["git", "ls-files"], cwd=repo_root, timeout_seconds=15)
    if not res.ok or not res.stdout.strip():
        return None
    out: list[Path] = []
    for line in res.stdout.splitlines():
        rel = line.strip()
        if rel:
            out.append(repo_root / rel)
    return out if out else None


def _walk_tree_files(repo_root: Path) -> list[Path]:
    """Fallback file list for non-git directories (e.g. pytest ``tmp_path``).

    Walks the whole tree but skips ``.git`` so the index still covers nested
    files when ``git ls-files`` is unavailable. This is the test path; in
    production the git index is the source. Build artifacts under a real repo
    are not a concern here because a real repo uses ``git ls-files``.
    """
    out: list[Path] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            out.append(p)
    return out


def _build_basename_index(repo_root: Path) -> dict[str, list[Path]]:
    """Map each bare filename to every tracked path with that basename.

    Derives the file set from ``git ls-files`` (the full tracked tree), never a
    hardcoded root list -- the prior ``src/``/``scripts/``/``tests/``-only walk
    missed files in other directories and produced false ``file_missing``
    verdicts for bare-basename citations (issue #1452). Returns
    ``dict[str, list[Path]]`` so a caller can distinguish a unique match (one
    entry) from an ambiguous one (multiple entries) -- the prior ``setdefault``
    first-match-wins behavior silently hid ambiguity. Built once per
    ``verify_citations`` call and reused across every citation, so the cost is
    one ``git ls-files`` per issue body, not per citation.
    """
    paths = _git_ls_files(repo_root)
    if paths is None:
        paths = _walk_tree_files(repo_root)
    index: dict[str, list[Path]] = {}
    for p in paths:
        index.setdefault(p.name, []).append(p)
    return index


def _literal_resolve(path: str, repo_root: Path) -> Path | None:
    """Resolve ``path`` literally against ``repo_root``, returning the file if it exists.

    Absolute paths are used as-is; relative paths are joined to ``repo_root``.
    Returns ``None`` when the literal path is not a file -- the caller then
    searches by basename. This is the literal-only half of the old ``_resolve``;
    the basename fallback moved into ``verify_citations`` so ambiguity can be
    detected and reported rather than silently resolved to the first match.
    """
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / path
    if p.is_file():
        return p
    return None


def _literal_path_exists(path: str, repo_root: Path) -> bool:
    """Return ``True`` if ``path`` exists as a file at its literal location.

    Mirrors the first check in ``_literal_resolve``: absolute paths are used as-is,
    relative paths are joined to ``repo_root``. Used to distinguish a
    citation whose asserted prefix is correct (literal path exists) from one
    whose prefix is stale (literal path missing, basename resolved via
    fallback) -- the latter is ``STALE_PREFIX``, not ``OK``.
    """
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / path
    return p.is_file()


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    except OSError:
        return []


def verify_citations(
    body: str,
    repo_root: Path,
    *,
    commit_sha: str | None = None,
    fetch_file_lines_at_commit: Callable[[str, str], list[str] | None] | None = None,
) -> list[CitationVerdict]:
    """Verify every citation in ``body`` against the working tree at ``repo_root``.

    Coordinate checks (``FILE_MISSING`` / ``OUT_OF_RANGE`` / ``EMPTY_LINE``) run
    unconditionally. Content comparison (``CONTENT_DRIFT``) runs only when both
    ``commit_sha`` and ``fetch_file_lines_at_commit`` are supplied: the fetcher
    returns the file's lines at the stamped commit (or ``None`` if the file or
    commit is unavailable), and a citation whose original line content no longer
    matches the current line is flagged. A citation that is out of range or in a
    missing file is never also checked for content drift -- the coordinate
    verdict already explains it.

    Resolution order for a citation whose literal path does not exist at
    ``repo_root`` (issue #1452):

    * **Zero** tracked files share the basename -> ``FILE_MISSING`` (genuinely
      absent -- the negative case stays detectable).
    * **Exactly one** tracked file shares the basename -> resolve against it
      and validate the line range there. A bare (no-prefix) citation that
      validates reports ``RESOLVED_BY_BASENAME`` (info-level -- the citation is
      usable, just imprecise); a prefixed citation reports ``STALE_PREFIX``
      (the asserted prefix is wrong). A range beyond EOF on the resolved file
      is still ``OUT_OF_RANGE`` -- line-range validation runs against the
      resolved path, not the missing literal one.
    * **Multiple** tracked files share the basename -> ``AMBIGUOUS_BASENAME``
      with the candidate list. This is a real citation defect the author must
      fix (disambiguate with a directory prefix); it is not a false alarm.

    Never raises on a missing/unreadable file; that is a verdict, not an error.
    """
    citations = parse_citations(body)
    verdicts: list[CitationVerdict] = []
    # Cache file reads (current and at-commit) by path so a file cited many
    # times is read once.
    current_lines_cache: dict[str, list[str]] = {}
    at_commit_cache: dict[str, list[str] | None] = {}
    # Build the basename index whenever there is at least one relative-path
    # citation. The index is derived from ``git ls-files`` (the full tracked
    # tree), so a bare basename resolves regardless of which directory the real
    # file lives in -- not just under a hardcoded set of source roots. A
    # citation with a stale/wrong directory prefix needs the index just as much
    # as a bare filename does, so the index is built for any relative-path
    # citation, not only bare ones. Built once per call, reused across every
    # citation: one ``git ls-files`` per issue body, not per citation.
    basename_index: dict[str, list[Path]] | None = None
    if any(not Path(c.path).is_absolute() for c in citations):
        basename_index = _build_basename_index(repo_root)

    for cite in citations:
        literal = _literal_resolve(cite.path, repo_root)
        if literal is not None:
            resolved = literal
            basename_resolved = False
        else:
            # Literal path missing -- search the tracked tree by basename.
            base = Path(cite.path).name
            matches = basename_index.get(base, []) if basename_index is not None else []
            if not matches:
                verdicts.append(CitationVerdict(cite, CitationStatus.FILE_MISSING))
                continue
            if len(matches) > 1:
                # Ambiguous: the author cited a bare basename (or a wrong
                # prefix) that matches more than one tracked file. This is a
                # real citation defect -- surface every candidate so the author
                # can disambiguate. Repo-root-relative POSIX strings keep the
                # comment readable on both Windows and POSIX.
                candidates = tuple(
                    str(p.relative_to(repo_root)).replace("\\", "/") for p in matches
                )
                verdicts.append(
                    CitationVerdict(
                        cite,
                        CitationStatus.AMBIGUOUS_BASENAME,
                        candidates=candidates,
                    )
                )
                continue
            resolved = matches[0]
            basename_resolved = True
            # Stale directory prefix: the citation asserted a path with a
            # directory prefix (``"/" in cite.path``), the literal path does not
            # exist, but the basename resolved uniquely via fallback. The prefix
            # was asserted and is now false -- flag it so a worker is not
            # silently pointed at the right file via a wrong path. Bare
            # (no-prefix) citations that resolve uniquely fall through to
            # line-range validation and report ``RESOLVED_BY_BASENAME``.
            # ``resolved_path`` surfaces where the file actually moved to.
            if "/" in cite.path:
                verdicts.append(
                    CitationVerdict(
                        cite,
                        CitationStatus.STALE_PREFIX,
                        resolved_path=str(resolved),
                    )
                )
                continue
        cache_key = str(resolved)
        if cache_key not in current_lines_cache:
            current_lines_cache[cache_key] = _read_lines(resolved)
        lines = current_lines_cache[cache_key]
        n = len(lines)
        # 1-based line numbers; end_line inclusive. Runs against the resolved
        # path, so a range beyond EOF on the resolved file is still flagged.
        if cite.line < 1 or cite.end_line > n:
            verdicts.append(CitationVerdict(cite, CitationStatus.OUT_OF_RANGE))
            continue
        current_slice = "\n".join(lines[cite.line - 1 : cite.end_line])
        if not current_slice.strip():
            verdicts.append(
                CitationVerdict(cite, CitationStatus.EMPTY_LINE, current_line_text=current_slice)
            )
            continue
        # Content drift needs the stamped commit's content. Skip when the caller
        # did not supply a stamp + fetcher -- coordinate checks are the backstop.
        original_slice: str | None = None
        if commit_sha and fetch_file_lines_at_commit is not None:
            if cite.path not in at_commit_cache:
                at_commit_cache[cite.path] = fetch_file_lines_at_commit(cite.path, commit_sha)
            at_commit = at_commit_cache[cite.path]
            if at_commit is not None and 1 <= cite.end_line <= len(at_commit):
                original_slice = "\n".join(at_commit[cite.line - 1 : cite.end_line])
        if original_slice is not None and original_slice != current_slice:
            verdicts.append(
                CitationVerdict(
                    cite,
                    CitationStatus.CONTENT_DRIFT,
                    current_line_text=current_slice,
                    original_line_text=original_slice,
                )
            )
            continue
        if basename_resolved:
            # Bare basename resolved uniquely and the line range validates.
            # Info-level: the citation is usable, just imprecise. Surface the
            # resolved path so a reader sees where the basename landed.
            verdicts.append(
                CitationVerdict(
                    cite,
                    CitationStatus.RESOLVED_BY_BASENAME,
                    current_line_text=current_slice,
                    resolved_path=str(resolved),
                )
            )
        else:
            verdicts.append(
                CitationVerdict(cite, CitationStatus.OK, current_line_text=current_slice)
            )
    return verdicts


# Statuses that are not drift: ``OK`` (a correct citation) and
# ``RESOLVED_BY_BASENAME`` (a bare basename that resolved uniquely and validates
# -- info-level, not an alarm). Everything else is a defect a worker or author
# must act on.
_NON_DRIFT = frozenset({CitationStatus.OK, CitationStatus.RESOLVED_BY_BASENAME})


def drifted_verdicts(verdicts: list[CitationVerdict]) -> list[CitationVerdict]:
    """Return only the verdicts that need attention -- the drift, not the info.

    ``OK`` and ``RESOLVED_BY_BASENAME`` are excluded: the former is a correct
    citation, the latter is an info-level bare-basename resolution that is
    usable as-is. Every other status (``FILE_MISSING``, ``OUT_OF_RANGE``,
    ``EMPTY_LINE``, ``STALE_PREFIX``, ``CONTENT_DRIFT``, ``AMBIGUOUS_BASENAME``)
    is a defect a worker or author must act on.
    """
    return [v for v in verdicts if v.status not in _NON_DRIFT]


def drift_fingerprint(verdicts: list[CitationVerdict]) -> str:
    """Stable hash of the drifted citations, for dedup across dispatch passes.

    Two passes that observe the same set of drifted citations produce the same
    fingerprint, so the flag-comment path fires once per drift state and re-alerts
    only when the drift actually changes (a new citation rots, or a stale one is
    corrected). ``OK`` verdicts are deliberately excluded: an issue whose
    citations are all valid has the empty fingerprint, distinct from one with
    drift, so resolving drift clears the flag.
    """
    parts = sorted(
        f"{v.citation.path}:{v.citation.line}-{v.citation.end_line}:{v.status.value}"
        for v in drifted_verdicts(verdicts)
    )
    if not parts:
        return ""
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def format_verdict_status_cell(v: CitationVerdict) -> str:
    """Render a verdict's status cell for the drift-comment table.

    Surfaces ``resolved_path`` (where the file actually moved to, for
    ``STALE_PREFIX`` and ``RESOLVED_BY_BASENAME``) and ``candidates`` (every
    tracked file sharing the basename, for ``AMBIGUOUS_BASENAME``) so the
    comment a worker reads via ``$issue_comments`` carries the disambiguation
    context. Kept here -- next to ``CitationVerdict`` -- rather than in the
    26k-line ``workflow.py`` monolith so the file-size ratchet is not tripped
    by presentation logic (issue #1452).
    """
    cell = v.status.value
    if v.resolved_path:
        resolved_display = v.resolved_path.replace("\\", "/")
        cell = f"{cell} (now at `{resolved_display}`)"
    if v.candidates:
        joined = ", ".join(f"`{c}`" for c in v.candidates)
        cell = (
            f"{cell} -- ambiguous basename, candidates: {joined}. "
            "Disambiguate the citation with a directory prefix."
        )
    return cell
