"""Stale-open-issue-mention scanning primitives + exclusion machinery for ``heartbeat_check.py``.

The ``stale-open-issue-mentions`` check (issue #902, exclusions issue #2048)
surfaces open issues referenced by already-merged work with no closure path.
This module holds every piece of that check's scanning machinery: the
stdlib-only CommonMark block-structure scan (``_scan_markdown_structure``),
the ``#N`` mention extraction primitives (``_mentioned_issue_numbers``,
``_branch_issue_number``), the local ``git log`` reader
(``get_merged_commit_messages``), and -- since #2048 -- the three
false-positive exclusion classifiers (non-closing mention context, parked/
active issue labels, bot-authored PRs).

Loaded from ``heartbeat_check.py`` via ``importlib`` from the sibling script
path, never a bare ``import`` -- matching how #1895 loads
``heartbeat_event_alarms.py``, #1476 loads ``heartbeat_worktree.py``, and
#1861 loads ``heartbeat_local_repo.py``: ``scripts/`` is not a package and is
deliberately kept off ``sys.path`` by the test harness
(``tests/_script_loader.py``). ``heartbeat_check`` re-exports the names its
own code and tests reference, so ``hb.*`` attribute references keep
resolving unchanged. This module is never run standalone.

Extracted out of ``heartbeat_check.py`` for file-size ratchet headroom
(``file_size_ratchet_baseline/scripts/heartbeat_check.py.count`` was at
2987/3000 when issue #2048's exclusion machinery needed the room) -- the
moved blocks are verbatim relocations; the only new code is the #2048
exclusion machinery at the bottom.

Stdlib-only, same constraint as ``heartbeat_check.py`` itself
(``scripts/README.md``): no ``charlie_work`` or third-party imports, and
never an import back into ``heartbeat_check`` -- that would cycle through
its loader block, which is also why ``CREATE_NO_WINDOW``/
``GH_TIMEOUT_SECONDS`` are mirrored below rather than shared (same
treatment ``heartbeat_worktree.py`` gives them).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Mirrors heartbeat_check's CREATE_NO_WINDOW (``getattr``-guarded: the
# attribute only exists on Windows).
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Mirrors heartbeat_check's GH_TIMEOUT_SECONDS -- the same bounded-subprocess
# posture applies to the local `git log` scan below.
GH_TIMEOUT_SECONDS = 30


# --------------------------------------------------------------------------
# Markdown block-structure scan -- stdlib-only duplicate (architecture-
# deepening plan, candidate 3 "markdown structure";
# `docs/superpowers/plans/2026-09-29-architecture-deepening.md`)
#
# `charlie_work.markdown_fence.scan` is the shared CommonMark-correct scan
# side that the seven consumers in `src/charlie_work/` are wired onto (full
# inventory: `md-recon.md`, wave A scratchpad). This module cannot import
# it -- the heartbeat script family is deliberately stdlib-only (see the
# module docstring and `scripts/README.md:50-51`), with two narrow,
# documented, guarded exceptions (`charlie_work.event_kinds` via
# `heartbeat_event_alarms.py`, and the `notify_digest*` leaves, #1859) that
# this is not. So this is a genuine duplicate, not a reuse: the same
# algorithm, reimplemented from scratch against the same
# `tests/markdown_conformance_cases.py` ground-truth table
# (`tests/test_markdown_conformance.py` runs that table against both).
#
# `_mentioned_issue_numbers` below is wired onto this scan (via
# `_strip_fenced_code_blocks`), replacing the `_FENCED_CODE_BLOCK_RE` regex
# it used to strip fenced blocks with before matching bare `#N` references.
# `tests/test_markdown_structure_characterization.py`'s two
# `test_flip_heartbeat_check_*` tests now assert the CommonMark-correct
# (flipped) result instead of pinning the old regex's desync/tilde-blind
# behaviour.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _MarkdownStructure:
    """The three views every surveyed consumer needs, line-indexed against
    ``_md_split_lines(text)`` (0-indexed, no keepends) -- see `FenceSpan` in
    `charlie_work.markdown_fence` for the ``fenced_line_spans`` half-open
    convention this mirrors exactly.
    """

    fenced_line_spans: tuple[tuple[int, int], ...]
    quoted_lines: frozenset[int]
    heading_lines: tuple[int, ...]


# CommonMark line endings only: \n, \r\n, \r. See `_md_split_lines` --
# mirrors `charlie_work.markdown_fence._LINE_ENDING_RE` /
# ``split_lines`` exactly (adversarial review finding B2,
# architecture-deepening candidate 3).
_MD_LINE_ENDING_RE = re.compile(r"\r\n|\r|\n")

# The three markers below match against a line with its leading indent
# already stripped by `_md_strip_marker_indent` (0-3 CommonMark *columns*,
# tab stops of 4 -- NOT an `[ \t]{0,3}` *character* count, which would
# wrongly admit up to three tabs/12 columns; adversarial review finding B3).

# CommonMark fenced-code-block delimiter: 3+ of the same backtick or tilde
# character, then an optional info string.
_MD_FENCE_DELIM_RE = re.compile(r"^(`{3,}|~{3,})[ \t]*(.*)$")
# CommonMark blockquote marker: `>`.
_MD_BLOCKQUOTE_RE = re.compile(r"^>")
# CommonMark ATX heading: 1-6 `#`, then either end-of-line or at least one
# space/tab before whatever heading text follows.
_MD_ATX_HEADING_RE = re.compile(r"^#{1,6}([ \t].*)?$")


def _md_split_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split ``text`` on CommonMark line endings only (0-indexed). Mirrors
    `charlie_work.markdown_fence.split_lines` exactly (stdlib-only, so it
    cannot import that module -- see this section's header comment) --
    never ``str.splitlines()``, whose wider separator set (vertical tab,
    form feed, file/group/record separator, NEL, LINE/PARAGRAPH SEPARATOR)
    is not a CommonMark or GitHub line break and would desync these line
    indices from a real fence/heading position on such input (adversarial
    review finding B2).
    """
    if not text:
        return []
    lines: list[str] = []
    pos = 0
    for match in _MD_LINE_ENDING_RE.finditer(text):
        end = match.end() if keepends else match.start()
        lines.append(text[pos:end])
        pos = match.end()
    if pos < len(text):
        lines.append(text[pos:])
    return lines


def _md_strip_marker_indent(line: str, max_indent: int | None = 3) -> str | None:
    """Strip ``line``'s leading indent, or ``None`` when it is wider than
    ``max_indent`` CommonMark columns (a tab advances to the next multiple
    of 4). ``max_indent=None`` never rejects (container-tolerant mode; see
    `_scan_markdown_structure`). Mirrors
    `charlie_work.markdown_fence._strip_marker_indent` exactly.
    """
    columns = 0
    chars = 0
    for char in line:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4 - (columns % 4)
        else:
            break
        chars += 1
        if max_indent is not None and columns > max_indent:
            break
    if max_indent is not None and columns > max_indent:
        return None
    return line[chars:]


def _md_find_fence_close(
    lines: list[str], start: int, char: str, length: int, max_indent: int | None = 3
) -> int | None:
    """First line index >= start closing a ``char``-fence opened at ``length``.

    Only ever called for a *valid* opener -- see `_scan_markdown_structure`.
    See `charlie_work.markdown_fence._find_fence_close`, which this mirrors
    exactly, including the per-candidate-line indent check.
    """
    close_re = re.compile(rf"^{re.escape(char)}{{{length},}}[ \t]*$")
    for index in range(start, len(lines)):
        stripped = _md_strip_marker_indent(lines[index], max_indent)
        if stripped is not None and close_re.match(stripped):
            return index
    return None


def _scan_markdown_structure(text: str, *, max_indent: int | None = 3) -> _MarkdownStructure:
    """Scan ``text`` for CommonMark block-level structure.

    Stdlib-only reimplementation of `charlie_work.markdown_fence.scan`; see
    that function's docstring for the CommonMark rules implemented.
    ``max_indent`` bounds fence-opener, fence-closer and blockquote-marker
    indent (columns); ATX headings always keep the 0-3 bound.
    ``max_indent=None`` is the container-tolerant mode for *exclusion*
    consumers (this module's `_strip_fenced_code_blocks`): a fence nested
    under a list item is still a fence, and this scan models no list items.
    """
    lines = _md_split_lines(text)
    fenced_spans: list[tuple[int, int]] = []
    quoted: set[int] = set()
    heading_lines: list[int] = []

    line_count = len(lines)
    index = 0
    while index < line_count:
        stripped = _md_strip_marker_indent(lines[index])
        container_stripped = _md_strip_marker_indent(lines[index], max_indent)

        if container_stripped is not None:
            delimiter = _MD_FENCE_DELIM_RE.match(container_stripped)
            if delimiter is not None:
                run, info = delimiter.group(1), delimiter.group(2).strip()
                char, length = run[0], len(run)
                if char == "`" and "`" in info:
                    # Not a fence opener under CommonMark -- an ordinary
                    # line, not the start of a region to skip (adversarial
                    # review finding B1). Resume scanning on the next line.
                    index += 1
                    continue
                close_at = _md_find_fence_close(lines, index + 1, char, length, max_indent)
                end = close_at + 1 if close_at is not None else line_count
                fenced_spans.append((index, end))
                index = end
                continue

            if _MD_BLOCKQUOTE_RE.match(container_stripped):
                quoted.add(index)
                index += 1
                continue

        if stripped is not None:
            if _MD_ATX_HEADING_RE.match(stripped):
                heading_lines.append(index)
                index += 1
                continue

        index += 1

    return _MarkdownStructure(
        fenced_line_spans=tuple(fenced_spans),
        quoted_lines=frozenset(quoted),
        heading_lines=tuple(heading_lines),
    )


# Inline run of 3+ backticks paired with an equal-length closer, spanning lines.
_INLINE_TRIPLE_SPAN_RE = re.compile(r"(`{3,}).+?\1", re.DOTALL)


def _strip_fenced_code_blocks(text: str) -> str:
    """Remove every fenced code block from ``text`` (opener/closer lines
    included), via ``_scan_markdown_structure``.

    Used by ``_mentioned_issue_numbers`` so a `#N`-shaped literal inside a
    code sample is not read as a reference. Replaces the prior
    ``_FENCED_CODE_BLOCK_RE`` regex (```` ```.*?``` ```` DOTALL,
    non-line-anchored, nearest-pair) -- see
    `tests/test_markdown_structure_characterization.py`'s
    `test_flip_heartbeat_check_*` tests for the pinned-then-flipped
    behaviour this replaces.
    """
    # Container-tolerant (`max_indent=None`): this is an exclusion consumer,
    # and the old `_FENCED_CODE_BLOCK_RE` ignored indent, so a list-nested
    # fence must still be stripped (else a `#N` inside it is a false
    # stale-mention ANOMALY).
    structure = _scan_markdown_structure(text, max_indent=None)
    if structure.fenced_line_spans:
        lines = _md_split_lines(text, keepends=True)
        drop = [False] * len(lines)
        for start, end in structure.fenced_line_spans:
            for index in range(start, min(end, len(lines))):
                drop[index] = True
        text = "".join(line for line, is_dropped in zip(lines, drop) if not is_dropped)
    # A multi-line INLINE run of 3+ backticks (opener and closer of equal
    # length, mid-paragraph) is a code span, not a block: strip it too, as the
    # prior regex did (md-r3 N1).
    return _INLINE_TRIPLE_SPAN_RE.sub("", text)


# --------------------------------------------------------------------------
# Stale-open-issue-mention scanning primitives (issue #902)
#
# charlie_work.github already has `issue_numbers_mentioned_by_pr` (a same-repo
# PR title/body scanner) and `iter_unnegated_closing_keyword_matches` (a
# negation-aware `#N` scanner used by `closing_keyword_gate.py`). This module
# deliberately does NOT import charlie_work.github, or any other
# ci_fleet-reachable charlie_work module (see the module docstring and
# `scripts/README.md` for the two narrow, guarded, stdlib-only
# exceptions: `charlie_work.event_kinds` and the `notify_digest*` leaves), so
# the small negation/quote-stripping heuristics
# below are a minimal, self-contained reimplementation for this one check
# rather than a reuse of those functions. Two differences from `issue_numbers_mentioned_by_pr`
# are intentional, not drift:
#
# 1. Bare `#N` is matched, not just `issue #N` / closing-keyword `#N`. Issue
#    #866's only trace anywhere is its fix's commit message, "refs #866" --
#    neither "issue" nor a closing keyword precedes it, so the narrower
#    pattern used by dispatch's mention detector would miss the exact
#    reproduction this check exists to catch.
# 2. It also scans commit messages (via local `git log`), not just PR
#    title/body -- again, the #866 shape.
#
# Quote/negation suppression exists for the same reason #790 forced it onto
# `iter_unnegated_closing_keyword_matches`: a literal, quoted example like
# `"Fixes #649"` inside prose is not an intentional reference and must not
# be surfaced (issue #902 acceptance criterion 6).
# --------------------------------------------------------------------------

_ISSUE_REF_RE = re.compile(r"#(\d+)\b")
# The repo's own branch-naming convention for non-agent-dispatched work:
# `<type>/<issueNumber>-<slug>` (e.g. `fix/817-fleet-health-latch`). Matched
# separately from `_ISSUE_REF_RE` because there is no `#` in a branch name.
# Deliberately does NOT match `agent/issue-N-...` branches (digit is not
# immediately after the slash there) -- those are already covered by the
# normal branch-prefix binding path (`linked_issue_number`), so a miss here
# is not a gap, just redundant with machinery this check exists to backstop.
_BRANCH_ISSUE_NUMBER_RE = re.compile(r"^[A-Za-z][\w.]*/(\d+)(?=[-_/]|$)")
_NEGATION_WORDS = ("not", "never", "without", "cannot")
_NEGATION_CONTRACTION_SUFFIX = "n't"
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(_NEGATION_WORDS) + r")\b|" + re.escape(_NEGATION_CONTRACTION_SUFFIX),
    flags=re.IGNORECASE,
)
_NEGATION_LOOKBEHIND_CHARS = 32
_QUOTE_CHARS = "\"'`"
_QUOTE_LOOKAROUND_CHARS = 40


def _has_preceding_negation(text: str, match_start: int) -> bool:
    """True if a negation word/contraction appears shortly before match_start.

    Same 32-char lookback window as `charlie_work.github._has_preceding_negation`
    (kept in sync by convention, not import -- see the section docstring above).
    """
    window_start = max(0, match_start - _NEGATION_LOOKBEHIND_CHARS)
    return bool(_NEGATION_RE.search(text, window_start, match_start))


def _is_quoted(text: str, match_start: int, match_end: int) -> bool:
    """True if the match sits inside a quoted span on the same line.

    A bare `#N` match (unlike a `<keyword> #N` closing-keyword match) can sit
    arbitrarily far from the quote character that wraps the whole phrase --
    #790's incident was the literal text `"Fixes #649"`, where the opening
    quote is 7 characters before the `#`. So this looks for a quote character
    (`"`, `'`, or a backtick) within `_QUOTE_LOOKAROUND_CHARS` before the
    match AND a matching quote character within the same distance after it,
    both bounded to the current line so a quote on an unrelated line can
    never suppress a real reference.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_end)
    if line_end == -1:
        line_end = len(text)
    before = text[max(line_start, match_start - _QUOTE_LOOKAROUND_CHARS) : match_start]
    after = text[match_end : min(line_end, match_end + _QUOTE_LOOKAROUND_CHARS)]
    return any(q in before and q in after for q in _QUOTE_CHARS)


def _mentioned_issue_numbers(text: str) -> set[int]:
    """Return every bare `#N` reference in `text`, minus quoted/negated ones.

    Fenced code blocks are stripped first (a code sample containing the
    literal text `#123` is not a reference), via
    ``_strip_fenced_code_blocks`` -- the same CommonMark-correct fence model
    `charlie_work.github_body_scan`'s own mention scanner now shares too
    (both wired onto their respective `markdown_fence.scan`/
    `_scan_markdown_structure` implementations, architecture-deepening
    candidate 3).

    Deliberately context-agnostic (issue #2048): this is the raw-extraction
    surface the pre-#2048 tests pin -- a ``Refs #866``-style non-closing
    mention still counts as a *mention* here. Whether a mention counts as
    *closure evidence* is ``issue_mention_occurrences``' job, layered on
    top; do not fold the non-closing-context exclusion into this function.
    """
    return {number for number, _non_closing in issue_mention_occurrences(text)}


def _branch_issue_number(branch: str) -> int | None:
    match = _BRANCH_ISSUE_NUMBER_RE.match(branch)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# Issue #2048 exclusions: a mention stops counting as closure evidence when
# (1) it sits in a non-closing context, (2) the issue carries a parked or
# active label, or (3) the mentioning PR is bot-authored. The machinery for
# all three lives here; `heartbeat_check.check_stale_open_issue_mentions`
# applies them and reports the excluded counts.
#
# Exclusion 1: an explicit non-closing reference is not evidence
# that merged work was meant to close the issue. The keyword set is the one
# the issue enumerates (`Refs`/`Ref`/`Related`/`See`/`follow-up`/`deferred`/
# `part of`/`filed as`), loosened only where morphology demands it:
# `refs?` covers Ref/Refs, `defer(?:red|ral|s)?` covers defer/defers/
# deferred/deferral, `follow[-\s]?up` covers follow-up/followup/follow up,
# and `references?` is the unabbreviated Refs.
# --------------------------------------------------------------------------
_NON_CLOSING_KEYWORDS_RE = re.compile(
    r"\b(?:refs?|references?|related|see|follow[-\s]?up|defer(?:red|ral|s)?"
    r"|part\s+of|filed\s+as)\b",
    flags=re.IGNORECASE,
)

# A qualifier scopes to the clause that precedes the mention. Clause edges
# are sentence/structure boundaries (`.`, `;`, `!`, `?`, parens, dashes,
# newline) -- deliberately NOT `,` or `:`: `Refs #1, #2, #3` must keep the
# qualifier scoped over the whole enumeration, and `Refs: #N` /
# `For issue #817:` read the qualifier across the colon in both directions
# the rule needs (present for Refs, absent for the #817 regression shape).
_NON_CLOSING_CLAUSE_BOUNDARIES = frozenset(".;!?()—–")


def _is_non_closing_context(text: str, match_start: int) -> bool:
    """True if a non-closing keyword precedes the `#N` at ``match_start`` in
    the same clause.

    The keyword must come *before* the mention -- "For issue #817: the
    deferred refactor" keeps flagging #817 (the issue's own regression
    shape), while "Refs #817", "is a follow-up to #817", and "part of #817"
    do not count as closure evidence. Checking the preceding clause only
    (never the following text) mirrors the direction
    ``_has_preceding_negation`` already uses for the same reason: a
    qualifier after the reference does not mark it non-closing, and a
    keyword in a later clause of a long line ("Fixed #5; deferred work
    remains") must not eat a genuine fix mention.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    prefix = text[line_start:match_start]
    boundary = max(prefix.rfind(c) for c in _NON_CLOSING_CLAUSE_BOUNDARIES)
    clause = prefix[boundary + 1 :]
    return bool(_NON_CLOSING_KEYWORDS_RE.search(clause))


def issue_mention_occurrences(text: str) -> list[tuple[int, bool]]:
    """Every ``#N`` mention occurrence in ``text`` as ``(number, non_closing)``.

    Same suppression posture as ``_mentioned_issue_numbers`` (fenced code
    stripped, quoted/negated mentions dropped) -- but each surviving
    occurrence is classified by ``_is_non_closing_context`` (issue #2048):
    ``non_closing=True`` means the mention is an explicit non-closing
    reference (``Refs #N``, ``follow-up``, ``deferred``, ``part of``,
    ``filed as``, ...) and is not evidence that merged work intended to
    close the issue. Occurrence-level, deliberately: a PR body that says
    ``Refs #5`` in one clause and ``Fixed #6`` in another keeps #6.
    """
    stripped = _strip_fenced_code_blocks(text)
    occurrences: list[tuple[int, bool]] = []
    for match in _ISSUE_REF_RE.finditer(stripped):
        if _has_preceding_negation(stripped, match.start()):
            continue
        if _is_quoted(stripped, match.start(), match.end()):
            continue
        occurrences.append((int(match.group(1)), _is_non_closing_context(stripped, match.start())))
    return occurrences


# Issue #2048 exclusion 2: the *parked* label set is operator-configured via
# the ``heartbeat: stale_mention_parked_labels`` config knob
# (``charlie_work.config.HeartbeatConfig``, the source of truth). This module
# cannot import it (stdlib-only invariant, module docstring), so the default
# is mirrored here -- the same documented-mirror treatment
# ``heartbeat_worktree._slugify_branch`` gives ``charlie_work.worktree``.
# ``tests/test_heartbeat_check_stale_issue_mentions.py`` asserts the two
# stay equal, so a drift surfaces as a test failure, not a silent desync.
# The default carries the fleet's shared non-lifecycle triage taxonomy
# (``heartbeat_check``'s armable-pool gate declares the same set as
# ``ARMABLE_GATING_LABELS``) plus the conventional tracker/umbrella/epic
# names the issue enumerates -- none of the fleet's own repos defines a
# literal ``tracker`` label today (verified against the live label list).
STALE_MENTION_PARKED_LABELS_DEFAULT: frozenset[str] = frozenset(
    {
        "blocked",
        "needs-design",
        "human-action",
        "question",
        "wontfix",
        "duplicate",
        "invalid",
        "tracker",
        "umbrella",
        "epic",
    }
)


def parked_label_names(heartbeat_section: Any) -> frozenset[str]:
    """The configured parked-label set, or the mirrored default.

    ``heartbeat_section`` is the repo's serialized ``heartbeat:`` config
    mapping (``config.get("heartbeat")`` after ``load_orchestrator_config``);
    absent, non-mapping, or key-less input returns
    ``STALE_MENTION_PARKED_LABELS_DEFAULT`` so repos that predate the knob
    still get the issue's exclusion set. An explicitly configured list --
    including an empty one -- is authoritative, matching how
    ``HeartbeatConfig.__post_init__`` normalizes the same YAML value
    (a bare string is a one-element list).
    """
    if not isinstance(heartbeat_section, dict):
        return STALE_MENTION_PARKED_LABELS_DEFAULT
    configured = heartbeat_section.get("stale_mention_parked_labels")
    if configured is None:
        return STALE_MENTION_PARKED_LABELS_DEFAULT
    if isinstance(configured, str):
        configured = (configured,)
    try:
        return frozenset(str(name) for name in configured)
    except TypeError:  # non-iterable junk (int, dict): fall back to default
        return STALE_MENTION_PARKED_LABELS_DEFAULT


# Bot accounts whose merged PRs never count as closure evidence (issue #2048
# exclusion 3). GitHub Apps author as ``<name>[bot]`` -- the suffix match
# catches dependabot[bot]/renovate[bot] and every future app bot; the bare
# names cover the same tools running as ordinary accounts.
BOT_AUTHOR_LOGINS: frozenset[str] = frozenset({"dependabot", "renovate"})


def pr_author_is_bot(author: Any) -> bool:
    """True when a merged PR's author is a bot (dependabot, renovate, any
    ``[bot]`` account).

    ``author`` is the ``gh pr list --json author`` value: normally
    ``{"login": "..."}`` (newer gh builds also include ``is_bot``/``type``);
    ``None`` for a deleted account. A bare string is treated as the login
    for tolerance of older/alternate gh output shapes.
    """
    if isinstance(author, dict):
        if author.get("is_bot") is True or str(author.get("type") or "").lower() == "bot":
            return True
        login = str(author.get("login") or "")
    elif isinstance(author, str):
        login = author
    else:
        return False
    return login.endswith("[bot]") or login.lower() in BOT_AUTHOR_LOGINS


def lifecycle_label_names(labels_section: Any) -> frozenset[str]:
    """The repo's fleet-managed lifecycle label names, derived from its
    ``labels:`` config section -- the serialized form of
    ``charlie_work.config.LabelConfig``, which this stdlib-only module
    cannot import (scripts/README.md). Every field except ``ready`` is a
    lifecycle label the orchestrator manages; ``ready`` (``automated-ready``)
    is the *armed* marker -- an armed issue is neither active nor parked, so
    it stays flaggable.

    Fields left at their defaults all live under the ``agent:`` prefix, so
    the caller's prefix check covers them without this function mirroring
    LabelConfig's default values; what this set contributes on top of the
    prefix is configured *renames* (a repo that sets e.g.
    ``queued: fleet-queued`` still has its lifecycle labels excluded).
    """
    if not isinstance(labels_section, dict):
        return frozenset()
    return frozenset(
        value
        for key, value in labels_section.items()
        if key != "ready" and isinstance(value, str) and value
    )


def mention_exempt_by_label(names: set[str], managed: frozenset[str]) -> bool:
    """True when an open issue's label set marks it parked or active (issue
    #2048 exclusion 2): it intersects the caller's managed set (the config-
    derived parked set from ``parked_label_names`` plus the configured
    lifecycle renames from ``lifecycle_label_names``), or it carries any
    label under the ``agent:`` namespace -- the prefix is LabelConfig's
    lifecycle namespace, so every unconfigured default (``agent:queued``,
    ``agent:in-progress``, ``agent:operator-queue``, ...) is caught without
    a hard-coded list. Same prefix convention ``check_armable_backlog``
    already uses.
    """
    return bool(names & managed) or any(name.startswith("agent:") for name in names)


_GIT_LOG_RECORD_SEP = "\x1e"
_GIT_LOG_FIELD_SEP = "\x1f"


def get_merged_commit_messages(
    repo_root: Path, limit: int
) -> tuple[bool, list[tuple[str, str]], str]:
    """Return (ok, [(short_sha, full_message), ...], err) for the local checkout's history.

    Reads `git log` on the already-checked-out branch -- every commit on it is
    by definition already merged into that branch, so this needs no `--merged`
    flag and, crucially, no `gh` call at all (issue #902's "API economy"
    constraint: this is the free local source, not one of the two bulk `gh`
    calls). This is what catches issue #866's reproduction: its fix rode in
    as a commit inside PR #864, a PR *for a different issue*, so no scan of
    PR title/body/branch name (for #864 or any other PR) could ever find it --
    only a scan of #864's own commit messages can.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                f"-n{limit}",
                f"--pretty=format:%h{_GIT_LOG_FIELD_SEP}%B{_GIT_LOG_RECORD_SEP}",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GH_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, [], f"git log failed to run: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:200]
        return False, [], f"git log exited {proc.returncode}: {stderr}"

    commits: list[tuple[str, str]] = []
    for record in proc.stdout.split(_GIT_LOG_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, _, message = record.partition(_GIT_LOG_FIELD_SEP)
        commits.append((sha, message))
    return True, commits, ""
