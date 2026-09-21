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
- Otherwise — every referenced path is missing from this repo — the gate
  requires *positive* evidence of a sibling repo before escalating (see
  "Positive-evidence redesign" below) rather than treating bare absence as
  proof of a cross-repo target: found under exactly one *other* managed
  fleet repo's root -> escalate; found under zero or 2+ sibling repos ->
  abstain, the same "no evidence either way" outcome as an issue with no
  referenced paths at all.

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
- **Citation-section headings** (issue #1583) — a candidate whose nearest
  preceding markdown ``#``-heading *is* (not merely contains) ``provenance``,
  ``references``, ``sources``, or ``see also`` is being cited as evidence,
  not named as code to edit. A bullet list under a ``## Provenance`` heading
  is this fleet's house style for exactly this kind of citation
  (``- Numbers: raw/analyses/.../foo.json (live)``), and none of the
  clause-local marker words appear in a typical provenance bullet — so the
  most common citation shape in this repo's own issues was invisible to the
  neutral classifier and escalated at campaign scale. Section scope is a
  stronger signal than a clause-local word and is derived from the body's
  own structure rather than from prose. The heading match is anchored so a
  heading like ``## Code References That Must Change`` (which contains
  ``references`` as a substring) does NOT neutralize the genuine dispatch
  targets listed under it.
- **Shared-prefix shorthand citation items** (issue #1761) — a comma- or
  newline-separated run of backtick spans that spells a directory out once
  and then repeats only the remainder, prefixed with ``/`` or ``...``
  (`` `dir/sub/a.json`, `/b.json`, `.../c.json` ``). No file is literally
  named ``/b.json`` or ``.../c.json``, so the shorthand items extract as
  bogus missing candidates that survive every other neutralization arm. A
  candidate whose first non-separator segment is three or more dots
  (``...``/longer) is neutral by construction — it is never a real path
  segment. ``..`` is deliberately NOT shorthand: it is the real
  parent-directory segment, so ``../sibling-repo/x.py`` still escalates as
  a cross-repo target. A
  leading-separator candidate (``/x`` or ``/x/y``) that continues a
  backtick-span run is resolved against the nearest preceding
  non-shorthand span's directory prefix and the *resolved* path is
  classified through the normal pipeline — so `` `/a.json`, `/b.json` ``
  behaves exactly as if ``b.json`` had been spelled out in full, while a
  leading-``/`` candidate cited standalone (or one whose resolved form is
  genuinely missing) still escalates as before.

Positive-evidence redesign (issues #1756, #1757, #1758; design doc
"cross_repo_gate: root cause, precision audit, and redesign
recommendation", Option B): a 2026-09-21 precision audit found 0 confirmed
true positives across 29 firings (24 unique issues, all 5 managed repos,
all-time) — every measured escalation was a false positive, each one a
plain-absence misfire (a newline-corrupted candidate, a module-relative
citation missing only a path prefix, or a citation of a file the issue
itself is about to create). The root defect: "this repo's filesystem
doesn't recognize a candidate" was being read as "the candidate belongs to
a different repo," but absence here is not by itself evidence of presence
*anywhere in particular* — it is equally consistent with a corrupted
string, a not-yet-created file, or a citation this repo's tree simply
doesn't have.

Escalation now requires positive sibling-repo evidence instead of bare
absence: once every survivor is missing from this repo, each missing
survivor is checked against every *other* managed fleet repo's root
(:func:`charlie_work.fleet_registry.managed_repo_roots` — never a
hardcoded repo list) via :func:`_find_owning_repo`. Found under exactly
one sibling repo escalates, with that repo's name recorded on
``CrossRepoGateResult.found_in_repo``. Found under zero or 2+ sibling
repos abstains — which subsumes the corrupted-candidate (#1756) and
issue-authored-new-file (#1758) cases for free: both degrade to "not
found anywhere in the fleet" without any new prose-parsing heuristic,
exactly the way this module already collapses distinguishable-in-theory
cases into one bucket wherever the distinction carries no decision-layer
weight. (#1757's own module-relative *same-repo* fallback for
``_path_exists_in_repo`` — resolving ``schemas/coach.py`` against the real,
more deeply nested ``server/src/swole/schemas/coach.py`` in the *same*
repo — is a related but separate fix tracked on that issue and not
implemented here; this module's contribution toward it is the shared
segment-boundary suffix-match helper, :func:`_segment_boundary_suffix_match`,
used today by :func:`_find_owning_repo` and written so #1757's fix can
reuse it rather than duplicate the segment-boundary logic.)

Founding #1010/#953 protection, restored unconditionally (2026-09-21
review of the positive-evidence redesign, finding 5): narrowing escalation
to "found under one of the fleet's *registered* siblings" silently dropped
the original protection for a foreign checkout that is not itself a
managed fleet repo — exactly the founding incident's own shape
(``ci_runners`` was never registered). Independent of
``managed_repo_roots`` entirely, a missing survivor that is an absolute
path resolving outside ``repo_root`` and that exists on disk right now
(:func:`_is_confirmed_foreign_absolute_path`) escalates on its own terms —
a real, on-disk absolute path elsewhere is positive evidence of a foreign
checkout without needing a fleet registry lookup to confirm it.

The sibling-repo search itself is now also root-aware, not name-only
(review finding 3): :func:`_find_owning_repo` excludes a ``managed_roots``
entry both by ``dispatching_repo_name`` (the computed name, which can
mismatch the registry key on a transient lookup failure) and by whether
the entry's root resolves to (or contains) the dispatching repo's actual
``repo_root`` — so a name mismatch can never turn the dispatching repo
into its own reported "sibling". Each sibling's file listing
(:func:`_repo_tracked_files`, ``git ls-files`` — tracked files only, so a
nested ``.claude/worktrees/`` or ``.venv`` copy of the same leaf filename
never makes the suffix match ambiguous, review finding 2) is computed at
most once per root per :func:`cross_repo_gate` call, not once per missing
candidate (review finding 1: an uncached, unpruned full-tree walk repeated
per candidate measured 37 seconds against this fleet's largest managed
repo).

Also added as cheap defense-in-depth: a candidate containing *any* embedded
whitespace (not just runs of 2+, issue #1756's own narrower proposal) is
dropped at extraction, alongside the existing glob/placeholder/launcher-owned
filters. This closes #1756's newline-corrupted-candidate shape and a
related single-space multi-path-in-one-backtick-span shape found live in cw
issue #1518's body (`` `tests/a.py tests/b.py` `` extracting as one
corrupted candidate containing an embedded space, which a "2+ whitespace"
filter alone would not catch).

All of these shapes are classified **neutral**: excluded from the pass/escalate
decision and reported separately (``CrossRepoGateResult.neutral_paths``)
rather than folded into ``referenced_paths``/``missing_paths``. When every
extracted candidate is neutral, the gate abstains (``passed=True``) — the
same outcome as an issue that references no paths at all, since neutral
candidates carry no evidence either way about where the issue's subject code
lives. When at least one non-neutral ("surviving") candidate remains, the
existing pass/escalate rule above applies to the survivors unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import LAUNCHER_OWNED_DIRS
from .cross_repo_gate_shorthand import _is_dotdot_shorthand, _resolve_list_shorthand
from .safe_path import contains
from .subprocess_runner import run_captured

logger = logging.getLogger(__name__)

#: Timeout for the ``git check-ignore`` invocation used to classify a
#: candidate as a gitignored runtime artifact. A single-path lookup is
#: sub-second in practice; this is a backstop against a wedged index lock.
_CHECK_IGNORE_TIMEOUT_SECONDS = 5

#: Timeout for the ``git ls-files`` invocation used to list a sibling
#: repo's tracked files for the segment-boundary suffix match (review
#: findings 1/2). A single listing is sub-second in practice even against
#: a 750k-file tree (measured against this fleet's largest managed repo,
#: job-cannon) since git already indexes the tree; this is a backstop
#: against a wedged index lock, not the expected runtime.
_LS_FILES_TIMEOUT_SECONDS = 15

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

# Embedded whitespace of any width (issue #1756, and the single-space
# shape found live in cw #1518, ``_TICK_PATH`` capturing
# `` `tests/a.py tests/b.py` `` as one corrupted candidate): no real
# relative or absolute file path contains a space, tab, or line break. This
# is deliberately wider than #1756's own proposed "\r, \n, or 2+ whitespace"
# filter — a single embedded space is not "2+ whitespace" and would slip
# that narrower version, but is exactly the same class of corrupted,
# can-never-exist candidate.
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
#
# Issue #1583: ``provenance`` was added. A bullet under a ``## Provenance``
# heading is this fleet's house style for citing the evidence behind an issue
# (``- Evidence: raw/analyses/.../foo.json``), and ``provenance`` is the
# clause-local word that introduces such a citation when the heading itself
# is not present. It is not a word a bug report uses to pinpoint code the
# worker must edit, so it does not open the false-negative direction the way
# "see"/"per" do.
#
# ``origin`` was considered and rejected: it is an ordinary git-remote name
# (``origin/main``, ``git push origin``) that appears constantly in issue
# bodies near genuine dispatch targets. Adding it as a clause-local marker
# false-positives on that usage and neutralizes a real cross-repo target --
# the exact false-negative this module exists to prevent. The
# citation-section heading (``## Provenance`` / ``## References`` / ``## Sources``
# / ``## See Also``) and the ``provenance`` clause-local marker already cover
# the #1583 citation shape; ``origin`` is redundant with them and unsafe.
_EVIDENCE_MARKER_RE = re.compile(
    r"\b(?:authority|evidence|cited\s+in|rationale|provenance)\b",
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

# A markdown ATX heading line: 1-6 ``#`` followed by whitespace and heading
# text. Used to derive the enclosing section of a candidate from the body's
# own structure (issue #1583) -- a citation under a ``## Provenance``
# heading is neutral regardless of whether a clause-local marker word
# introduces it, because the heading itself is the citation signal.
#
# The leading-whitespace class is ``[ \t]*`` (any amount) only because the
# fence/indented-code filtering in :func:`_nearest_preceding_heading` strips
# code-block lines before this regex ever sees them; a real ATX heading
# cannot be indented 4+ spaces (that is an indented code block in CommonMark),
# and the function skips such lines before matching.
_HEADING_LINE_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$")

# A fenced-code-block delimiter (CommonMark): 0-3 leading spaces, then 3+
# backticks or 3+ tildes. An opener may carry an info string after the marker
# (e.g. `````python````); a closer is the marker alone (plus optional
# trailing whitespace). Used by :func:`_nearest_preceding_heading` to skip
# ``#``-prefixed lines inside a fenced block -- a code sample containing the
# literal line ``# References`` is not a structural heading and must not
# shadow a real preceding heading (review finding, PR #1584 round 3).
_FENCE_DELIM_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")

# Citation-section heading vocabulary (issue #1583): a candidate whose
# nearest preceding ``#``-heading is essentially one of these phrases is
# neutral. These are the fleet's house-style headings for evidence/reference
# sections (``## Provenance``, ``## References``, ``## Sources``, ``## See Also``)
# -- a path under such a heading is being cited, not named as code to edit.
# ``see also`` is two words and is matched as a phrase so a heading like
# ``## See Also`` (the fleet's cross-ref section) is caught but a heading
# like ``## See the bug`` is not.
#
# The regex is anchored with ``^...$`` so it matches only when the heading
# text *is* the citation phrase (optionally with a trailing ``:`` or ``.``),
# not when the heading merely *contains* one of the words as a substring.
# An unanchored ``\b...\b`` search would false-positive on headings like
# ``## Code References That Must Change`` -- neutralizing genuine dispatch
# targets listed under them -- which is the exact cross-repo-contamination
# risk this module exists to prevent. Review finding (PR #1584 round 1):
# the original unanchored regex substring-matched inside any heading text.
_CITATION_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:provenance|references|sources|see\s+also)\s*[:.]?\s*$",
    re.IGNORECASE,
)


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


def _nearest_preceding_heading(text_before_in_body: str) -> str | None:
    """Return the text of the nearest markdown ATX heading preceding the
    candidate's position, or ``None`` when no heading precedes it.

    *text_before_in_body* is the full stripped body up to (but not
    including) the candidate. The last heading match in that span is the
    section the candidate lives under -- markdown section scope runs from a
    heading until the next heading of the same or higher level, and a
    candidate's enclosing section is the most recent heading regardless of
    level (a ``### Sub`` under ``## Provenance`` is still in the Provenance
    section).

    The walk is fence/indented-code aware (review finding, PR #1584 round
    3): a ``#``-prefixed line inside a fenced code block (between
    ``\\`\\`\\``` / ``~~~`` delimiters) or an indented code block (4+ leading
    spaces or a tab) is code content, not a structural heading. Without
    this, a code sample containing the literal line ``# References`` would
    shadow a real preceding heading and wrongly neutralize a genuine
    dispatch target elsewhere in the body -- the same cross-repo-
    contamination risk class the gate exists to prevent. A real ATX
    heading is never indented 4+ spaces (that is an indented code block in
    CommonMark), so skipping such lines can only remove false positives,
    never a genuine heading.
    """
    last: str | None = None
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in text_before_in_body.splitlines():
        delim = _FENCE_DELIM_RE.match(line)
        if delim:
            marker = delim.group(1)
            char = marker[0]
            length = len(marker)
            if not in_fence:
                # Opener: enter the fence. An info string may follow the
                # marker (e.g. ```python); it is code, not prose.
                in_fence = True
                fence_char = char
                fence_len = length
            elif char == fence_char and length >= fence_len:
                # Closer: same fence character, at least as long as the
                # opener, and nothing but whitespace after the marker.
                if line[delim.end() :].strip() == "":
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
            continue
        if in_fence:
            # Inside a fenced block: every line is code, regardless of '#'.
            continue
        # Indented code block (CommonMark): 4+ leading spaces or a leading
        # tab is code, not prose -- a '# References' line so indented is a
        # code sample, not a heading.
        if line.startswith("    ") or line.startswith("\t"):
            continue
        match = _HEADING_LINE_RE.match(line)
        if match:
            last = match.group(1)
    return last


def _is_in_citation_section(text_before_in_body: str) -> bool:
    """Return ``True`` when the candidate's enclosing markdown section is a
    citation section (issue #1583).

    A citation section is one whose nearest preceding ``#``-heading *is*
    (not merely contains) one of the citation-heading phrases
    (``provenance``, ``references``, ``sources``, ``see also``), optionally
    with a trailing ``:`` or ``.``. The heading regex is anchored so a
    heading like ``## Code References That Must Change`` -- which contains
    ``references`` as a substring -- does NOT match: the paths under it are
    genuine dispatch targets, not citations. Section scope is a stronger
    signal than a clause-local marker word: a bullet under ``## Provenance``
    is being cited as evidence regardless of whether
    ``provenance``/``evidence`` appears in the bullet's own clause, because
    the heading itself declares the section's purpose. The signal is derived
    from the body's own structure rather than from prose, so it catches the
    fleet's house-style citation shape (``- Numbers:
    raw/analyses/.../foo.json (live)``) that no clause-local marker word
    introduces.
    """
    heading = _nearest_preceding_heading(text_before_in_body)
    if heading is None:
        return False
    return bool(_CITATION_SECTION_HEADING_RE.search(heading))


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
        found_in_repo: when ``passed`` is ``False``, the name of the single
            other managed fleet repo under whose root a missing survivor was
            positively found (issues #1756, #1757, #1758 -- the
            positive-evidence redesign; see the module docstring). ``None``
            when the gate passed, or when it was constructed by
            :func:`cross_repo_scope_gate` (which has no notion of a missing
            path's owning repo).
    """

    passed: bool
    referenced_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    reason: str
    neutral_paths: tuple[str, ...] = ()
    found_in_repo: str | None = None


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

    Candidates containing embedded whitespace of any width (a space, tab, or
    line break) are dropped — no real file path contains one, so a
    whitespace-corrupted candidate (a hard-wrapped backtick-quoted path, or
    two paths cited together in one backtick span separated by a single
    space) can never exist and would false-positive the gate (issue #1756).

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
        # Drop whitespace-corrupted candidates: a real path never contains a
        # space, tab, or line break. Catches both a hard-wrapped
        # backtick-quoted path (embedded \r\n, issue #1756) and a
        # single-space multi-path span in one backtick pair (cw #1518) —
        # deliberately wider than a "2+ whitespace" filter, which the
        # single-space shape would slip.
        if _EMBEDDED_WHITESPACE_RE.search(raw):
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


def _is_absolute_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` is absolute, on any host platform.

    ``Path(candidate).is_absolute()`` alone is platform-dependent in a way
    that matters here: on Windows, ``PureWindowsPath`` only counts a path as
    absolute when it carries a drive letter, so a POSIX-style absolute path
    like ``/home/user/other-repo/foo.py`` reports ``is_absolute() is False``.
    Left unguarded, that misclassifies a genuinely absolute (and genuinely
    outside-the-repo) candidate as "relative", which would let
    :func:`_resolve_within_root` join it onto a repo root instead of
    containment-checking it directly. A leading path separator is
    unambiguously absolute regardless of host platform, so it is treated as
    absolute here even where ``Path.is_absolute`` disagrees.
    """
    return Path(candidate).is_absolute() or bool(re.match(r"[\\/]", candidate))


def _resolve_within_root(root: Path, path_str: str) -> Path | None:
    """Return ``path_str`` resolved against ``root``, or ``None`` when it
    would not resolve to a path inside ``root``.

    Handles both relative candidates (joined onto ``root``, then
    containment-checked so a ``..`` segment walking out of ``root`` is
    caught) and absolute candidates (containment-checked directly, never
    joined — joining an absolute-looking string onto a base path is
    unreliable across platforms). Resolution and containment happen before
    any ``exists()`` check, so a traversal or absolute candidate can never
    be reported as "owned" by a root it does not actually resolve into,
    even if the resolved location happens to exist somewhere else on disk
    (:func:`charlie_work.safe_path.contains` resolves both sides, catching
    a lexically-contained-looking path that a symlink/junction or a ``..``
    collapse actually escapes).
    """
    path = Path(path_str)
    if _is_absolute_path(path_str):
        candidate = path
    else:
        candidate = root / path
    try:
        if contains(root, candidate):
            return candidate
    except (OSError, ValueError):
        return None
    return None


def _all_repo_files(repo_root: Path) -> list[str]:
    """Return every file under ``repo_root``, POSIX-separated and relative.

    Fallback listing used by :func:`_repo_tracked_files` when ``git
    ls-files`` fails (``repo_root`` is not a git repository, the git binary
    is missing, or the call times out) — see there for the primary listing
    every real call site uses today. Only ``.git`` is pruned during the
    walk (not filtered afterward), so unlike the tracked-files listing this
    can surface untracked/vendored duplicates (a nested ``.claude/worktrees/``
    or ``.venv`` copy of the same leaf filename); :func:`_segment_boundary_suffix_match`
    treats that ambiguity the same as "not found" rather than guessing, so
    this fallback degrades to strictly *less* precise, never wrong.

    Derived fresh from the live filesystem — never cached across calls —
    mirroring the rest of this module's "ask the filesystem, don't
    hand-maintain a list" convention. Best-effort: an unreadable
    subdirectory is skipped (logged, not silently dropped) rather than
    aborting the whole listing, and an ``OSError`` mid-walk (a repo root
    that vanishes concurrently) returns whatever was collected so far
    rather than raising, matching ``_top_level_dirs``'s historical
    "empty/partial on error" contract.
    """
    files: list[str] = []

    def _log_walk_error(exc: OSError) -> None:
        logger.debug(
            "cross_repo_gate: os.walk fallback skipped an unreadable path under %s: %s",
            repo_root,
            exc,
        )

    try:
        for dirpath, dirnames, filenames in os.walk(repo_root, onerror=_log_walk_error):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for filename in filenames:
                rel = Path(dirpath, filename).relative_to(repo_root).as_posix()
                files.append(rel)
    except OSError:
        return files
    return files


def _repo_tracked_files(repo_root: Path) -> list[str]:
    """Return every git-tracked file under ``repo_root``, POSIX-separated
    and relative to it.

    Uses ``git ls-files`` — one subprocess call, sub-second even against a
    750k-file repo (job-cannon, measured) — instead of walking the
    filesystem: tracked-only is worktree/``.venv``/``node_modules``/build-
    output-free *by construction*, with no prune list to hand-maintain and
    no chance of it drifting out of sync (review findings 1/2: an
    unpruned, uncached ``os.walk`` was both a 37-second-per-call liveness
    hazard against this fleet's largest repo and made every real candidate
    "ambiguous" for :func:`_segment_boundary_suffix_match` — every managed
    repo on this fleet has at least one nested ``.claude/worktrees/`` or
    ``.venv`` copy of *something*, so 2+ matches was the norm, not the
    exception).

    Falls back to :func:`_all_repo_files` (a pruned ``os.walk``) when ``git
    ls-files`` fails. The fallback is strictly noisier, never wrong: see
    its own docstring.
    """
    result = run_captured(
        ["git", "ls-files"],
        cwd=repo_root,
        timeout_seconds=_LS_FILES_TIMEOUT_SECONDS,
    )
    if result.ok:
        return [line.replace("\\", "/") for line in result.stdout.splitlines() if line]
    return _all_repo_files(repo_root)


def _segment_boundary_suffix_match(candidate: str, files: list[str]) -> str | None:
    """Return the single file in ``files`` whose path ends with ``candidate``
    on a path-segment boundary, or ``None`` when zero or 2+ files match.

    ``files`` are ``repo_root``-relative, POSIX-separated paths (see
    :func:`_all_repo_files`). A segment-boundary suffix match requires the
    match to start either at the beginning of the file's path or
    immediately after a ``/`` — a raw string suffix would wrongly match
    ``"ngs/coach.py"`` inside ``".../settings/coach.py"`` (issue #1757).
    Ambiguous suffixes (2+ files share one) are treated the same as no
    match at all: this module's existing conservative default for anything
    ambiguous (see the shorthand and evidence-marker classifiers) is to
    fall back to "not resolved" rather than silently pick one.

    Standalone — takes no ``repo_root`` and does no filesystem I/O itself —
    so it is shared as-is between :func:`_find_owning_repo`'s cross-repo
    check here and issue #1757's own planned same-repo nested-module
    fallback for :func:`_path_exists_in_repo`, rather than each
    reimplementing the segment-boundary comparison.
    """
    candidate_posix = candidate.replace("\\", "/").lstrip("/")
    if not candidate_posix:
        return None
    matches = {f for f in files if f == candidate_posix or f.endswith("/" + candidate_posix)}
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _excludes_dispatching_root(root: Path, dispatching_repo_root: Path) -> bool:
    """Return ``True`` when ``root`` is the dispatching repo's own root.

    Review finding 3: name-based exclusion alone (``repo_name == exclude``
    in :func:`_find_owning_repo`) compares ``_dispatching_repo_name``'s
    *computed* name against the registry key. That computation falls back
    to ``repo_root.name`` when the ``gh`` lookup fails (a transient API
    blip, offline), which need not match the repo's registered
    ``owner/repo`` segment — a mismatch there would otherwise let the
    dispatching repo's own registered entry be searched as if it were a
    foreign sibling, escalating the repo against itself. Root equality (or
    containment, via :func:`charlie_work.safe_path.contains`) is checked
    independently of whatever name was computed, so this exclusion holds
    even when the name-based one fails.

    Deliberately one-directional: only "the dispatching root is inside (or
    equal to) this candidate sibling root" excludes. The reverse — a
    genuine sibling repo that happens to live in a subdirectory of the
    dispatching repo's own tree — is not excluded, since that is real
    sibling-repo topology, not the dispatching repo appearing under its
    own name.
    """
    try:
        return contains(root, dispatching_repo_root)
    except (OSError, ValueError):
        return False


def _find_owning_repo(
    path_str: str,
    managed_roots: Mapping[str, Path],
    exclude: str,
    dispatching_repo_root: Path,
    file_listings: dict[str, list[str]],
) -> str | None:
    """Return the name of the single *other* managed repo that owns ``path_str``.

    Checks ``path_str`` against every repo in ``managed_roots`` except the
    dispatching repo itself — excluded both by name (``exclude``) and,
    independently, by resolved root (:func:`_excludes_dispatching_root`,
    review finding 3) — both as a literal repo-root-relative/absolute path
    (via :func:`_resolve_within_root`, which containment-checks before any
    ``exists()`` call — a ``..`` traversal or an absolute path can never be
    reported as resolving into a root it does not actually resolve into)
    and, when that misses, via :func:`_segment_boundary_suffix_match`
    against that repo's tracked-file listing — the module-relative-citation
    shape from issues #1757/#1758 (``shared/.../SettingsRepository.kt``
    citing the real, more deeply nested
    ``swole/app/mobile/.../SettingsRepository.kt``).

    ``file_listings`` memoizes each sibling root's tracked-file listing
    (:func:`_repo_tracked_files`) for the caller's lifetime — one listing
    per root per :func:`cross_repo_gate` call, not one per (missing
    candidate, root) pair (review finding 1). Callers share one dict across
    every missing candidate checked in a single :func:`cross_repo_gate`
    call.

    Returns the owning repo's name when exactly one *other* managed repo
    matches; ``None`` when zero or 2+ repos match. Ambiguous ownership is
    not escalation evidence — the same conservative "abstain rather than
    guess" default this module already uses for ambiguous suffix and
    shorthand resolutions.
    """
    owners: set[str] = set()
    for repo_name, root in managed_roots.items():
        if repo_name == exclude:
            continue
        if _excludes_dispatching_root(root, dispatching_repo_root):
            continue
        resolved = _resolve_within_root(root, path_str)
        if resolved is not None and resolved.exists():
            owners.add(repo_name)
            continue
        files = file_listings.get(repo_name)
        if files is None:
            files = _repo_tracked_files(root)
            file_listings[repo_name] = files
        if _segment_boundary_suffix_match(path_str, files) is not None:
            owners.add(repo_name)
    if len(owners) == 1:
        return next(iter(owners))
    return None


def _is_confirmed_foreign_absolute_path(path_str: str, repo_root: Path) -> bool:
    """Return ``True`` when ``path_str`` is an absolute candidate that
    resolves outside ``repo_root`` and exists on disk right now.

    Independent of the fleet registry entirely (review finding 5): a real,
    on-disk absolute path outside the target repo is positive evidence of a
    foreign checkout on its own terms — the exact founding #1010/#953
    shape, where the offending sibling (``ci_runners``) was never even a
    registered fleet member. The positive-evidence redesign's
    managed-repo-roots search only ever escalates for one of the fleet's
    *registered* siblings, which silently dropped this unconditional
    protection for every other checkout under the operator's ``repos``
    tree; this restores it as an independent, unconditional check that
    runs whether or not a fleet registry was supplied.

    Uses :func:`_is_absolute_path` (not ``Path.is_absolute()``) so a
    POSIX-style absolute candidate (no drive letter) is still recognized as
    absolute on Windows, matching :func:`_resolve_within_root`'s own
    absolute-path handling.
    """
    if not _is_absolute_path(path_str):
        return False
    path = Path(path_str)
    try:
        if contains(repo_root, path):
            # Resolves inside repo_root -- not foreign. Already excluded by
            # the missing-survivor filter in the caller's normal flow; a
            # redundant check here keeps this function correct standalone.
            return False
        return path.exists()
    except (OSError, ValueError):
        return False


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


def _split_survivors_and_neutral(
    issue_body: str, repo_root: Path
) -> tuple[list[tuple[str, str]], list[str]]:
    """Partition extracted candidates into survivors and neutral candidates.

    ``survivors`` is a list of ``(raw, effective)`` pairs: ``raw`` is the
    candidate as written (what ``referenced_paths``/``missing_paths``
    report) and ``effective`` is the path it denotes after shared-prefix
    shorthand resolution (issue #1761) — identical to ``raw`` for every
    non-shorthand candidate. Existence, gitignore, and repo-shape checks
    run against ``effective`` so a shorthand item behaves exactly as if the
    full path had been spelled out.

    Neutral candidates (gitignored runtime artifacts, evidence/authority
    citations, candidates under a citation-section heading, or shared-prefix
    shorthand items — see the module docstring) are excluded entirely;
    :func:`cross_repo_gate` reports them separately via
    ``CrossRepoGateResult.neutral_paths``.
    """
    candidates, stripped = _iter_candidate_matches(issue_body)
    survivors: list[tuple[str, str]] = []
    neutral: list[str] = []
    for raw, start, end in candidates:
        before, after = _paragraph_span(stripped, start, end)
        if _is_dotdot_shorthand(raw):
            neutral.append(raw)
            continue
        resolved = _resolve_list_shorthand(raw, stripped, start)
        effective = resolved if resolved is not None else raw
        if (
            _is_context_neutral(before, after)
            or _is_in_citation_section(stripped[:start])
            or _is_gitignored(effective, repo_root)
        ):
            neutral.append(raw)
        else:
            survivors.append((raw, effective))
    return survivors, neutral


def cross_repo_gate(
    issue_body: str,
    repo_root: Path,
    managed_repo_roots: Mapping[str, Path] | None = None,
    dispatching_repo_name: str = "",
) -> CrossRepoGateResult:
    """Decide whether an issue should be dispatched or escalated as cross-repo.

    Returns a :class:`CrossRepoGateResult` with ``passed=True`` when the issue
    is safe to dispatch (it references no file paths, at least one referenced
    path exists in ``repo_root``, or no missing path is positively found
    under exactly one other managed repo), and ``passed=False`` when every
    referenced path is missing from ``repo_root`` *and* at least one of them
    is found under exactly one other managed fleet repo — positive evidence
    that the issue's subject code lives there instead (see the module
    docstring's "Positive-evidence redesign" section, issues #1756, #1757,
    #1758).

    Before the pass/escalate decision runs, extracted candidates are split
    into survivors and neutral candidates (see the module docstring and
    :func:`_split_survivors_and_neutral`) — a gitignored runtime artifact or
    an evidence/authority citation is excluded entirely rather than counted
    as a missing dispatch target. When every candidate is neutral, the gate
    abstains the same way it would for an issue with no candidates at all.
    The rules below then apply to the survivors.

    ``managed_repo_roots`` (repo name -> repo root, from
    :func:`charlie_work.fleet_registry.managed_repo_roots`) and
    ``dispatching_repo_name`` (this repo's own name, excluded from the
    search) default to an empty mapping and the empty string — "no fleet
    information available" — which can only ever *abstain* on the
    every-survivor-missing branch's sibling-registry search, never
    escalate from it: that half of the decision requires positive
    sibling-repo evidence that only a real registry lookup can provide, so
    a caller that has not been wired to pass these two (tracked
    separately: this gate's own decision rule vs. threading the fleet
    registry through each call site) gets the strictly safer "abstain when
    we don't know" behavior rather than a crash or the old, imprecise
    "escalate on bare absence" rule. The founding #1010/#953 absolute-path
    protection (review finding 5, :func:`_is_confirmed_foreign_absolute_path`)
    is independent of both arguments and can still escalate with neither
    supplied.
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
                "citation, a citation-section heading, or a gitignored "
                "runtime artifact, not a dispatch target"
            ),
            neutral_paths=tuple(neutral),
        )
    missing_pairs = [
        (raw, effective)
        for raw, effective in survivors
        if not _path_exists_in_repo(effective, repo_root)
    ]
    missing = tuple(raw for raw, _ in missing_pairs)
    # Pass when at least one survivor exists here — the worker has something
    # to work on in this repo regardless of what else is missing.
    if len(missing) < len(survivors):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(raw for raw, _ in survivors),
            missing_paths=missing,
            reason="at least one referenced path exists in the target repo",
            neutral_paths=tuple(neutral),
        )
    # Every surviving path is missing from THIS repo. That alone is not
    # evidence of a cross-repo target (see "Positive-evidence redesign" in
    # the module docstring) — it is equally consistent with a corrupted
    # candidate, a citation of a file this issue is about to create, or a
    # citation this fleet has never seen anywhere. Escalate only when a
    # missing survivor is positively found under exactly one *other*
    # managed repo's root, OR (review finding 5, the founding #1010/#953
    # protection) is itself an absolute path that resolves outside
    # repo_root and exists on disk right now -- positive evidence of a
    # foreign checkout on its own terms, independent of whether that
    # foreign repo happens to be a registered fleet member.
    for raw, effective in missing_pairs:
        if _is_confirmed_foreign_absolute_path(effective, repo_root):
            return CrossRepoGateResult(
                passed=False,
                referenced_paths=tuple(r for r, _ in survivors),
                missing_paths=missing,
                reason=(
                    f"cross_repo_target: {effective!r} is an absolute path "
                    f"outside the target repo ({repo_root}) and exists on "
                    "disk -- positive evidence of a foreign checkout"
                ),
                neutral_paths=tuple(neutral),
            )
    found_in_repo: str | None = None
    if managed_repo_roots:
        # One tracked-file listing per sibling root for this whole call
        # (review finding 1) -- _find_owning_repo populates it lazily as
        # each root is actually checked.
        file_listings: dict[str, list[str]] = {}
        for raw, effective in missing_pairs:
            owner = _find_owning_repo(
                effective,
                managed_repo_roots,
                dispatching_repo_name,
                repo_root,
                file_listings,
            )
            if owner is not None:
                found_in_repo = owner
                break
    if found_in_repo is None:
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(raw for raw, _ in survivors),
            missing_paths=missing,
            reason=(
                f"abstaining: {len(missing)} referenced file path(s) are "
                "absent from this repo and were not found under exactly one "
                "other managed fleet repo — not positive evidence of a "
                "cross-repo target"
            ),
            neutral_paths=tuple(neutral),
        )
    return CrossRepoGateResult(
        passed=False,
        referenced_paths=tuple(raw for raw, _ in survivors),
        missing_paths=missing,
        reason=(
            f"cross_repo_target: a referenced file path is absent from the "
            f"target repo ({repo_root}) but found under exactly one other "
            f"managed fleet repo ({found_in_repo!r})"
        ),
        neutral_paths=tuple(neutral),
        found_in_repo=found_in_repo,
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
