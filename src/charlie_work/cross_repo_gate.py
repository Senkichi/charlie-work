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

Issues #1452 and #1460: a referenced path is not automatically evidence of a
cross-repo target. Two shapes of "expected absent" candidate showed up in
production:

- **Evidence/authority citations** — a path cited as the source of the bug
  report or the design rationale (``Authority: <path> section 4 rows 5-6``),
  not as code the worker is meant to touch. #1452's original body cited
  ``job_finder/*`` paths this way to document where a false-alarm report
  came from; #1460 cited an ``llibrary`` decision doc as the design
  authority for the feature. The marker vocabulary is deliberately narrow
  (``authority``, ``evidence``, ``cited in``, ``rationale``, and the
  ``section N`` / ``rows N-M`` suffixes) — it excludes ``see``, ``per``, and
  ``line N``, which are also the ordinary way a bug report cites the file
  the worker must *edit* (``See `job_finder/matcher.py` line 42 — the loop
  never breaks``). Treating those as evidence markers would make a genuine
  wrong-repo issue cited that way go all-neutral and abstain into dispatch —
  the more expensive false-negative direction described above.
- **Runtime-artifact write destinations** — a path the issue's *own*
  deliverable will create at runtime (``advisories are logged to
  .var/attachment-contracts/advisories.jsonl``), which by definition cannot
  exist yet. A gitignored path (derived from the target repo's own
  ``.gitignore`` via ``git check-ignore``, never a hardcoded name list) is
  the same signal by construction — nothing under a gitignored path is a
  dispatch target in the first place.

Both shapes are classified **neutral**: excluded from the pass/escalate
decision and reported separately (``CrossRepoGateResult.neutral_paths``)
rather than folded into ``referenced_paths``/``missing_paths``. When every
extracted candidate is neutral, the gate abstains (``passed=True``) — the
same outcome as an issue that references no paths at all, since neutral
candidates carry no evidence either way about where the issue's subject code
lives. When at least one non-neutral ("surviving") candidate remains, the
existing pass/escalate rule above applies to the survivors unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import LAUNCHER_OWNED_DIRS
from .safe_path import contains
from .subprocess_runner import run_captured

#: Timeout for the ``git check-ignore`` invocation used to classify a
#: candidate as a gitignored runtime artifact. A single-path lookup is
#: sub-second in practice; this is a backstop against a wedged index lock.
_CHECK_IGNORE_TIMEOUT_SECONDS = 5

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

# Glob metacharacters: ``*``, ``?``, ``[``, ``]``. A candidate containing any
# of these is a glob pattern, not a literal file path — no file literally
# named ``*.py`` exists, so a glob candidate is always "missing" and would
# false-positive the gate. The backtick-quoted path regex (``_TICK_PATH``)
# captures globs because it matches anything inside backticks that contains a
# separator and ends with an extension; the non-tick regexes exclude ``*``
# from their character classes, so only backtick-quoted globs reach this
# filter (issue #1391).
_GLOB_METACHAR = re.compile(r"[*?\[\]]")

# Evidence/authority-citation markers (issues #1452, #1460): a candidate
# preceded by one of these words in its own clause is being cited as a
# reference that explains or backs the issue, not as a file the worker is
# expected to touch.
#
# Deliberately narrow. "see" and "per" were dropped: they are also the
# ordinary way a bug report cites the file the worker must EDIT ("See
# `job_finder/matcher.py` line 42 -- the loop never breaks"), so keeping
# them made a genuine wrong-repo issue whose only candidates are cited that
# way go all-neutral and abstain -- a false negative, which is the more
# expensive failure mode per the module docstring (it costs a contaminated
# sibling checkout, not just one manual triage action). Only markers that
# are unambiguously about *authority/rationale for the issue*, never about
# *where the bug lives*, stay in this list.
_EVIDENCE_MARKER_RE = re.compile(
    r"\b(?:authority|evidence|cited\s+in|rationale)\b",
    re.IGNORECASE,
)

# Evidence/authority citation SUFFIX markers: "section 4", "rows 5-6"
# following a candidate in the same paragraph is the same citation signal,
# just placed after the path instead of before it (issue #1460's
# "Authority: <path> section 4 rows 5-6").
#
# "line N" was dropped: it is the ordinary way a bug report pinpoints the
# file the worker must edit ("`job_finder/matcher.py` line 42 -- the loop
# never breaks"), not an evidence-citation signal -- keeping it risked the
# same false-negative direction as "see"/"per" above.
_EVIDENCE_SUFFIX_RE = re.compile(
    r"\b(?:section|rows?)\s+\d+",
    re.IGNORECASE,
)

# Runtime-artifact write-destination markers (issue #1460): a candidate
# preceded by one of these verbs in its own paragraph is something the
# issue's *own* future work will create or write, not evidence that already
# exists -- it cannot exist yet by definition, so its absence is not
# evidence of a cross-repo target.
_WRITE_DESTINATION_MARKER_RE = re.compile(
    r"\b(?:logged\s+to|writes?\s+to|written\s+to|will\s+create|"
    r"creates?|emits?\s+to|appends?\s+to)\b",
    re.IGNORECASE,
)

# A blank line (one or more) delimiting markdown paragraphs. Used to bound
# the context window for the evidence/write-destination checks to "this
# paragraph" -- wide enough that a marker word introducing a path on the
# very next line (a markdown soft-wrap, e.g. "Authority:\n<path>") is still
# caught, narrow enough that an unrelated marker word in a different
# paragraph of a long issue body cannot neutralize a candidate it has
# nothing to do with.
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")

# Clause-boundary punctuation *within* a paragraph. A preceding marker word
# must appear in the candidate's own clause, not merely somewhere earlier in
# the same paragraph -- "Authority: docs/decisions/x.md section 2; the
# actual bug is in `src/real.py`" must NOT neutralize ``src/real.py`` just
# because "Authority" opened an unrelated earlier clause in the same
# sentence.
#
# Deliberately excludes ``:`` and bare newlines: a label like "Authority:"
# or "Evidence:" is a single semantic unit with what follows it (including
# across a markdown soft-wrap onto the next line), not two separate
# clauses -- severing at the colon would make the label word unreachable
# from the clause search, defeating the exact pattern it exists to catch.
_CLAUSE_BOUNDARY_RE = re.compile(r"[.;,()]")


def _current_clause(text_before_in_paragraph: str) -> str:
    """Return the text since the last clause-boundary punctuation mark in
    *text_before_in_paragraph* (or the whole string, if none) -- the
    "current clause" a preceding marker word must appear in to describe the
    candidate that follows it, rather than an unrelated earlier clause in
    the same paragraph.
    """
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(text_before_in_paragraph))
    if not boundaries:
        return text_before_in_paragraph
    return text_before_in_paragraph[boundaries[-1].end() :]


@dataclass(frozen=True)
class CrossRepoGateResult:
    """Outcome of the cross-repo pre-flight gate.

    Attributes:
        passed: ``True`` when the issue should be dispatched, ``False`` when
            it should be escalated to ``agent:human-needed``.
        referenced_paths: candidate file paths extracted from the issue body
            that were *not* classified neutral (see ``neutral_paths``), as
            raw strings (may include paths that do not exist anywhere).
        missing_paths: the subset of ``referenced_paths`` that do not exist
            in the target repo.  When ``passed`` is ``False``, this equals
            ``referenced_paths`` (every referenced path was missing).
        reason: human-readable explanation for the gate's decision, suitable
            for an event payload.
        neutral_paths: candidate file paths extracted from the issue body
            that were classified as evidence/authority citations or
            gitignored runtime-artifact write destinations (issues #1452,
            #1460) -- excluded from ``referenced_paths``/``missing_paths``
            and from the pass/escalate decision entirely. Reported
            separately for observability, not because it drives any
            downstream behavior today.
    """

    passed: bool
    referenced_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    reason: str
    neutral_paths: tuple[str, ...] = ()


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

    Candidates containing a placeholder path segment (``<state-dir>``,
    ``pr-N``, ``issue-N``, ``<...>``) are dropped — a template stand-in can
    never name a real file, so a templated documentation path in an issue
    body cannot fire the gate (issue #1343).

    Candidates containing glob metacharacters (``*``, ``?``, ``[``, ``]``)
    are dropped — a glob pattern is not a literal file path, and no file
    literally named ``*.py`` exists, so a glob candidate is always "missing"
    and would false-positive the gate (issue #1391).

    Candidates whose first path segment is a launcher-owned worktree
    directory (``.devin``, ``.git_worktree_dir``) are dropped — these paths
    live only inside agent worktrees, not in the repo tree, so they are not
    evidence of a cross-repo target (issue #1391).
    """
    return [raw for raw, _start, _end in _iter_candidate_matches(issue_body)[0]]


def _strip_non_path_prose(issue_body: str) -> str:
    """Strip URLs and scheme-less domain-shaped tokens before path matching.

    Shared by :func:`_iter_candidate_matches` so the offsets it returns are
    relative to the same stripped text a caller uses for context lookups
    (see :func:`_paragraph_span`).
    """
    # Strip URLs before matching so the POSIX absolute-path alternation does
    # not capture the path portion of ``https://example.com/foo.py``.
    stripped = re.sub(r"https?://\S+", "", issue_body)
    # Strip scheme-less domain-shaped tokens the same way — a "path" whose
    # leading segment is actually a hostname is not a file-path reference.
    return re.sub(_DOMAIN_PATH, "", stripped)


def _iter_candidate_matches(issue_body: str) -> tuple[list[tuple[str, int, int]], str]:
    """Return ``(candidates, stripped_body)`` for every surviving candidate.

    ``candidates`` is a de-duplicated, first-seen-order list of
    ``(raw_path, start, end)`` where ``start``/``end`` are offsets into
    ``stripped_body`` — kept so callers can inspect the text surrounding a
    match (see :func:`_paragraph_span`) without re-deriving the stripped
    body themselves and risking an offset mismatch.
    """
    stripped = _strip_non_path_prose(issue_body)
    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for match in _PATH_RE.finditer(stripped):
        # The regex has four alternation groups; pick the one that matched.
        raw = next((g for g in match.groups() if g is not None), "")
        if not raw:
            continue
        # Drop templated/placeholder paths: a segment like ``pr-N`` or
        # ``<state-dir>`` is documentation template text, not a real file
        # reference, and can never be a genuine cross-repo target.
        if _has_placeholder_segment(raw):
            continue
        # Drop glob patterns: a candidate containing ``*``, ``?``, ``[``, or
        # ``]`` is a glob, not a literal path. No file named ``*.py`` exists,
        # so a glob is always "missing" and would false-positive the gate.
        if _GLOB_METACHAR.search(raw):
            continue
        # Drop launcher-owned worktree paths: paths under ``.devin/`` or
        # ``.git_worktree_dir/`` live only inside agent worktrees, not in the
        # repo tree, so they are not evidence of a cross-repo target.
        if _is_launcher_owned_path(raw):
            continue
        if raw not in seen:
            seen.add(raw)
            candidates.append((raw, match.start(), match.end()))
    return candidates, stripped


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


def _gitignored_top_level_names(repo_root: Path) -> set[str]:
    """Return top-level names ignored by the repo's ``.gitignore``.

    Derived from the repo's ``.gitignore`` — never a hardcoded name list —
    so the "repo-shaped" exclusion tracks the repo's actual ignore policy
    rather than a brittle hand-maintained set.  Issue #1343: the runtime
    state dir (``.var``) is a real top-level directory in the checkout but
    is gitignored; a templated documentation path keyed on its name must
    not count as "repo-shaped" evidence.

    Only simple literal patterns are collected — a bare ``name`` or
    ``name/`` line with no nested slash, no wildcard, and no negation.
    Complex patterns (globs, negations, nested paths) are left to git
    itself and do not participate in this exclusion; they cannot name a
    single top-level directory unambiguously, and over-collecting them
    would risk excluding a real tracked directory.
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return set()
    ignored: set[str] = set()
    try:
        text = gitignore.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    for line in text.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#") or pattern.startswith("!"):
            continue
        # Drop a trailing slash (directory marker); the bare name is what
        # would appear as the first segment of a relative path.
        name = pattern.rstrip("/")
        # Only collect simple literal top-level names: no slash (nested
        # patterns do not name a top-level dir), no wildcard/bracket sets
        # (they cannot name a single directory unambiguously).
        if "/" in name or "*" in name or "?" in name or "[" in name:
            continue
        ignored.add(name)
    return ignored


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
    path segment names a directory that actually exists in ``repo_root`` and
    is not gitignored.

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

    A real top-level directory that is *gitignored* (the runtime state dir,
    a venv, build artifacts) does not count as repo-shaped evidence: a path
    keyed on such a name is typically a templated documentation path (e.g. a
    runtime-state example), not a reference to this repo's tracked code
    (issue #1343).  The exclusion is derived from the repo's ``.gitignore``,
    not a hardcoded name list, so it tracks the repo's actual ignore policy.
    """
    if _is_absolute_path(candidate):
        return False
    first_segment = re.split(r"[\\/]+", candidate, maxsplit=1)[0]
    if first_segment not in _top_level_dirs(repo_root):
        return False
    if first_segment in _gitignored_top_level_names(repo_root):
        return False
    return True


def _paragraph_span(text: str, start: int, end: int) -> tuple[str, str]:
    """Return ``(text_before, text_after)`` within the paragraph containing ``[start, end)``.

    A paragraph is a blank-line-delimited block (see ``_PARAGRAPH_BREAK_RE``)
    rather than a single ``\\n``-delimited line — a markdown soft-wrap (one
    newline, no blank line) does not split a marker word from the path it
    introduces on the next line, which is exactly issue #1460's shape
    (``"Authority:\\nllibrary/docs/...\\n"``).
    """
    before_text = text[:start]
    after_text = text[end:]
    para_start = 0
    for m in _PARAGRAPH_BREAK_RE.finditer(before_text):
        para_start = m.end()
    after_match = _PARAGRAPH_BREAK_RE.search(after_text)
    para_end = after_match.start() if after_match else len(after_text)
    return before_text[para_start:], after_text[:para_end]


def _is_context_neutral(before: str, after: str) -> bool:
    """Return ``True`` when the candidate's own clause marks it as an
    evidence/authority citation or a runtime-artifact write destination
    rather than a dispatch target (issues #1452, #1460).

    The preceding-marker checks are scoped to :func:`_current_clause` of
    ``before`` (not the whole paragraph) so a marker word that opened an
    unrelated earlier clause in the same paragraph/sentence cannot
    neutralize a candidate it has nothing to do with. The suffix check
    (``section N`` / ``rows N-M``) stays paragraph-scoped: it only ever
    matches immediately after the candidate, so there is no equivalent
    "unrelated earlier clause" to guard against.
    """
    clause_before = _current_clause(before)
    if _EVIDENCE_MARKER_RE.search(clause_before) or _EVIDENCE_SUFFIX_RE.search(after):
        return True
    return bool(_WRITE_DESTINATION_MARKER_RE.search(clause_before))


def _is_gitignored(candidate: str, repo_root: Path) -> bool:
    """Return ``True`` when ``git check-ignore`` reports ``candidate`` as
    ignored by ``repo_root``'s gitignore rules.

    Works on non-existent paths — ``check-ignore`` matches by pathname
    pattern only, with no filesystem existence check. Any failure
    (``repo_root`` is not a git repository, the candidate resolves outside
    the repository, the git binary is missing, a timeout) falls back to
    ``False`` — "not ignored" — so a broken git invocation degrades to the
    *narrower* neutral set, not a wider one: a candidate that should
    escalate keeps escalating rather than being silently suppressed.
    """
    result = run_captured(
        ["git", "check-ignore", "-q", "--", candidate],
        cwd=repo_root,
        timeout_seconds=_CHECK_IGNORE_TIMEOUT_SECONDS,
    )
    return result.returncode == 0


def _split_survivors_and_neutral(issue_body: str, repo_root: Path) -> tuple[list[str], list[str]]:
    """Partition extracted candidates into survivors and neutral candidates.

    Survivors are the candidates that still count toward the pass/escalate
    decision. Neutral candidates (gitignored runtime artifacts or
    evidence/authority citations — see the module docstring) are excluded
    entirely; :func:`cross_repo_gate` reports them separately via
    ``CrossRepoGateResult.neutral_paths``.
    """
    candidates, stripped = _iter_candidate_matches(issue_body)
    survivors: list[str] = []
    neutral: list[str] = []
    for raw, start, end in candidates:
        before, after = _paragraph_span(stripped, start, end)
        if _is_context_neutral(before, after) or _is_gitignored(raw, repo_root):
            neutral.append(raw)
        else:
            survivors.append(raw)
    return survivors, neutral


def cross_repo_gate(issue_body: str, repo_root: Path) -> CrossRepoGateResult:
    """Decide whether an issue should be dispatched or escalated as cross-repo.

    Returns a :class:`CrossRepoGateResult` with ``passed=True`` when the issue
    is safe to dispatch (it references no file paths, or at least one
    referenced path exists in ``repo_root``), and ``passed=False`` when every
    referenced path is missing from ``repo_root`` — the signal that the
    issue's subject code lives in a different repo.

    Before the pass/escalate decision runs, extracted candidates are split
    into survivors and neutral candidates (see the module docstring and
    :func:`_split_survivors_and_neutral`) — a gitignored runtime artifact or
    an evidence/authority citation is excluded entirely rather than counted
    as a missing dispatch target. When every candidate is neutral, the gate
    abstains the same way it would for an issue with no candidates at all.
    The rules below then apply to the survivors.

    A single-candidate exception applies: when exactly one survivor remains
    and it is missing, the gate abstains (``passed=True``) unless that
    candidate is a relative path whose first segment names a directory that
    actually exists in ``repo_root`` (see
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
            neutral_paths=(),
        )
    survivors, neutral = _split_survivors_and_neutral(issue_body, repo_root)
    if not survivors:
        # Every candidate was neutral (a gitignored runtime artifact or an
        # evidence/authority citation, see the module docstring) — the same
        # outcome as no candidates at all, since none of them are evidence
        # of where the issue's subject code lives.
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=(),
            missing_paths=(),
            reason=(
                "abstaining: every referenced path was an evidence/authority "
                "citation or a gitignored runtime artifact, not a dispatch "
                "target"
            ),
            neutral_paths=tuple(neutral),
        )
    missing = tuple(p for p in survivors if not _path_exists_in_repo(p, repo_root))
    # Block only when EVERY surviving path is absent — the issue's subject
    # code is not in this repo at all.  If even one survivor exists, the
    # worker has something to work on here and the gate passes.
    if len(missing) < len(survivors):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(survivors),
            missing_paths=missing,
            reason="at least one referenced path exists in the target repo",
            neutral_paths=tuple(neutral),
        )
    # NOTE: absoluteness is checked via ``_is_absolute_path``, not a bare
    # ``Path(...).is_absolute()`` — the latter is platform-dependent (a
    # POSIX-style absolute path is not "absolute" under Windows' drive-letter
    # rule) and would let a genuinely-absolute, outside-the-repo candidate
    # slip through as "relative" and incorrectly abstain. See
    # ``_is_absolute_path`` for the full rationale.
    if (
        len(survivors) == 1
        and not _is_absolute_path(survivors[0])
        and not _is_repo_shaped_relative_candidate(survivors[0], repo_root)
    ):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(survivors),
            missing_paths=missing,
            reason=(
                "single ambiguous candidate "
                f"({survivors[0]!r}) is not a repo-shaped relative path — "
                "abstaining rather than escalating on weak evidence"
            ),
            neutral_paths=tuple(neutral),
        )
    # Every surviving path is missing — the issue's subject code is not in
    # this repo.  Escalate instead of dispatching a worker that will wander.
    return CrossRepoGateResult(
        passed=False,
        referenced_paths=tuple(survivors),
        missing_paths=missing,
        reason=(
            f"cross_repo_target: all {len(missing)} referenced file path(s) "
            f"are absent from the target repo ({repo_root})"
        ),
        neutral_paths=tuple(neutral),
    )


def cross_repo_scope_gate(
    issue_title: str,
    issue_body: str,
    dispatching_repo_name: str,
    managed_repo_names: frozenset[str],
) -> CrossRepoGateResult:
    """Decide whether an issue's *scope* targets another managed repo.

    This is the intake-time repo attribution from issue #1244 (Option 1):
    when an issue's deliverables live in a different managed repo, the lane
    that dispatches it can never finalize it — the worker either hops to
    the sibling repo's worktree (contaminating it) or exits with zero
    artifacts in the dispatching repo, and the single-remote orphan sweep
    declares it dead and redispatches, looping forever.

    The gate checks the issue **title** for a ``<repo-name>:`` prefix that
    names a managed repo other than the dispatching one.  This is the
    clearest signal that the issue's scope is that repo — the #709 case
    study had the title ``other-repo: docs/devin-orchestration/ ... stale``
    in the charlie-work tracker, but every deliverable was in that other
    repo.

    The managed-repo set **must** derive from the fleet registry (see
    :func:`charlie_work.fleet_registry.managed_repo_names`), never a
    hardcoded list — a literal list would require manual updates for every
    fleet change and break silently when a repo is added or removed.

    Conservative by design (mirrors the file-path gate's philosophy):

    - An empty managed-repo set (no fleet registry, single-repo) passes —
      no other repos to attribute the issue to.
    - An issue whose title does not start with ``<other-repo>:`` passes —
      the scope is not unambiguously another repo.
    - The dispatching repo's own name is excluded — an issue in
      charlie-work that says ``charlie-work: fix X`` is in-repo by
      definition.
    - Only a title that starts with ``<repo-name>:`` for a managed repo
      *other than* the dispatching one is blocked.  The body is not
      scanned for repo-name mentions because a passing reference
      (``coordinate with the other repo on this``) is not evidence of a
      cross-repo scope.

    Args:
        issue_title: The issue's title string.
        issue_body: The issue's body string (unused by the current
            title-prefix check, but accepted so callers can extend the
            detection without changing the call site).
        dispatching_repo_name: The repo-name segment of the dispatching
            repo (e.g. ``charlie-work``), extracted from the GitHub
            ``nameWithOwner`` or the repo root's directory name.
        managed_repo_names: The set of repo-name segments managed by the
            fleet, from :func:`fleet_registry.managed_repo_names`.

    Returns:
        A :class:`CrossRepoGateResult` with ``passed=False`` when the
        issue title names another managed repo, ``passed=True`` otherwise.
    """
    other_repos = managed_repo_names - {dispatching_repo_name}
    if not other_repos:
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=(),
            missing_paths=(),
            reason="no other managed repos in the fleet registry",
        )
    title_lower = issue_title.lower().lstrip()
    for repo_name in sorted(other_repos):
        # Match "repo-name:" at the start of the title (case-insensitive).
        # This is the "other-repo: docs/..." pattern from #709 — the
        # clearest signal that the issue's scope is that repo, not this
        # one.  A colon immediately after the repo name is the convention
        # for scope-prefixed issue titles in this fleet.
        prefix = f"{repo_name.lower()}:"
        if title_lower.startswith(prefix):
            return CrossRepoGateResult(
                passed=False,
                referenced_paths=(),
                missing_paths=(),
                reason=(
                    f"cross_repo_scope: issue title starts with "
                    f"{repo_name!r}: — the issue's deliverables target "
                    f"{repo_name}, not the dispatching repo "
                    f"({dispatching_repo_name})"
                ),
            )
    return CrossRepoGateResult(
        passed=True,
        referenced_paths=(),
        missing_paths=(),
        reason="issue title does not name another managed repo",
    )


__all__ = [
    "CrossRepoGateResult",
    "cross_repo_gate",
    "cross_repo_scope_gate",
    "extract_referenced_paths",
]
