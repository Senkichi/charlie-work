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
    "parse_citations",
    "verify_citations",
]


class CitationStatus(str, Enum):
    """Verdict on a single ``path:line`` citation.

    ``OK`` is the only non-drift status. Every other value means the citation's
    coordinates no longer match the working tree and a worker should not trust
    the bare line number.
    """

    OK = "ok"
    FILE_MISSING = "file_missing"
    OUT_OF_RANGE = "out_of_range"
    EMPTY_LINE = "empty_line"
    CONTENT_DRIFT = "content_drift"


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


def _looks_like_url_context(path: str) -> bool:
    """Reject tokens that are really URL schemes (``https:``, ``http:``...).

    A citation path never contains ``://``; a URL always does. This is the
    single discriminator that keeps ``https://example.com:8080`` from matching
    as a citation of ``example.com:8080``.
    """
    return "://" in path


def parse_citations(body: str) -> list[Citation]:
    """Return every ``path:line`` / ``path:start-end`` citation in ``body``.

    Deduplicated and order-preserving by ``(path, line, end_line)``. Tokens whose
    path contains ``://`` (URLs) are dropped. A citation is *parsed* on shape
    alone; whether it is *verified* is a separate step (``verify_citations``)
    that needs the working tree.
    """
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(body):
        path = m.group("path")
        if _looks_like_url_context(path):
            continue
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


def _build_basename_index(repo_root: Path) -> dict[str, Path]:
    """Map each bare filename to its first path under the source roots.

    Walks ``src/``, ``scripts/``, and ``tests/`` recursively (bounded to those
    three roots) and records the first path found for each basename. Real
    source files in this repo live two levels deep -- ``src/charlie_work/<name>.py``
    -- so a one-level search misses every one of them. The walk is bounded to
    the three roots (not the whole tree) to stay cheap and to avoid matching
    files outside the package tree (build artifacts, vendored deps, etc.).

    Built once per ``verify_citations`` call and reused across every citation,
    so the cost is one walk per issue body, not one walk per citation. On a
    duplicate basename the first match (depth-first, deterministic ``rglob``
    order) wins; that is an acceptable ambiguity -- the prior behavior returned
    ``None`` (always wrong), and a unique basename is the common case.
    """
    index: dict[str, Path] = {}
    for root in ("src", "scripts", "tests"):
        base_dir = repo_root / root
        if not base_dir.is_dir():
            continue
        for p in base_dir.rglob("*"):
            if p.is_file():
                index.setdefault(p.name, p)
    return index


def _resolve(
    path: str,
    repo_root: Path,
    *,
    basename_index: dict[str, Path] | None = None,
) -> Path | None:
    """Resolve ``path`` against ``repo_root``, returning the file if it exists.

    Tries the literal path (absolute, or relative to ``repo_root``), then a
    bare-basename fallback (issues often cite ``workflow.py:4746`` without the
    ``src/charlie_work/`` prefix). The fallback first checks one level under the
    common source roots (cheap, preserves shallow-tree behavior), then falls back
    to ``basename_index`` -- a recursive index of the source roots -- so a bare
    citation resolves regardless of nesting depth (``src/charlie_work/<name>.py``
    as well as ``scripts/<name>.py``). Returns ``None`` when no file is found --
    the caller treats that as ``FILE_MISSING``.
    """
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / path
    if p.is_file():
        return p
    # Bare-filename fallback: try one level under the common source roots first
    # (cheap stat, no walk), then the recursive basename index for files nested
    # deeper (the real layout: src/charlie_work/<name>.py).
    base = Path(path).name
    for root in ("src", "scripts", "tests"):
        candidate = repo_root / root / base
        if candidate.is_file():
            return candidate
    if basename_index is not None:
        hit = basename_index.get(base)
        if hit is not None and hit.is_file():
            return hit
    return None


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

    Never raises on a missing/unreadable file; that is a verdict, not an error.
    """
    citations = parse_citations(body)
    verdicts: list[CitationVerdict] = []
    # Cache file reads (current and at-commit) by path so a file cited many
    # times is read once.
    current_lines_cache: dict[str, list[str]] = {}
    at_commit_cache: dict[str, list[str] | None] = {}
    # Build the recursive basename index whenever there is at least one
    # relative-path citation. The resolver's fallback strips the directory off
    # *any* unresolved citation (bare or prefixed) via ``Path(path).name``
    # before consulting this index, so a citation with a stale/wrong directory
    # prefix (e.g. a file that moved into a new subdirectory) needs the index
    # just as much as a bare filename does. Gating on "no directory prefix"
    # only would leave stale-prefixed citations reporting FILE_MISSING instead
    # of resolving -- the index would never be built for them. The walk is
    # bounded to the three source roots and runs once per issue body, so the
    # cost is one walk per ``verify_citations`` call, not per citation.
    basename_index: dict[str, Path] | None = None
    if any(not Path(c.path).is_absolute() for c in citations):
        basename_index = _build_basename_index(repo_root)

    for cite in citations:
        resolved = _resolve(cite.path, repo_root, basename_index=basename_index)
        if resolved is None:
            verdicts.append(CitationVerdict(cite, CitationStatus.FILE_MISSING))
            continue
        cache_key = str(resolved)
        if cache_key not in current_lines_cache:
            current_lines_cache[cache_key] = _read_lines(resolved)
        lines = current_lines_cache[cache_key]
        n = len(lines)
        # 1-based line numbers; end_line inclusive.
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
        verdicts.append(CitationVerdict(cite, CitationStatus.OK, current_line_text=current_slice))
    return verdicts


def drifted_verdicts(verdicts: list[CitationVerdict]) -> list[CitationVerdict]:
    """Return only the non-``OK`` verdicts -- the citations that need attention."""
    return [v for v in verdicts if v.status is not CitationStatus.OK]


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
