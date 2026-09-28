"""Prose-dependency detection internals for ``github_body_scan``.

Extracted from ``github_body_scan.py`` during the issue #1949 rework: adding
the issue-ref ordering patterns pushed that module past the 800-line
file-size cap (``tests/test_file_size_ratchet.py``), so the prose-dependency
pattern tables and the shared quoted-prose judgement helpers live here.
``github_body_scan`` imports them back; nothing here imports
``github_body_scan``, so ``parse_blockers`` stays strictly downstream-free —
the detector needs the blocker set it produces, never the reverse.

The clause/quoting primitives (``_clause_bounds``, ``_inside_code_span``,
``_inside_quoted_span``, ``_is_blockquote_line``, ``_inside_quoted_prose``)
are shared ground: ``parse_blockers`` and ``issue_numbers_mentioned_by_pr``
apply them to blocker declarations and issue mentions exactly as
``detect_prose_only_dependencies`` applies them to dependency prose, so the
guard semantics cannot drift between the two scanners.
"""

from __future__ import annotations

import re

_CLAUSE_BOUNDARY_CHARS = ".!?\n"

# Markdown backtick code span: an opening run of backticks, content, and a
# closing run of the SAME length. Capturing group 2 is the span content.
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)(\1)", flags=re.DOTALL)
# Balanced straight-double-quote span. Group 1 is the quoted content.
_DOUBLE_QUOTE_SPAN_RE = re.compile(r'"([^"]*)"')

# A leading run of same-repo issue refs joined by ',' or 'and' separators
# ("#12, #13", "#12 and #13", "#12, #13, and #14"). A ref followed by
# non-separator prose ends the run ("#14 after #9 lands" -> "#14"), so
# trailing annotation text in an item is ignored rather than misparsed.
# The pattern text is shared with ``github_body_scan``'s blocker-section
# item scan so "wait for #12 and #13" contributes both refs there too.
_ISSUE_REF_RUN = r"#\d+(?:[ \t]*(?:,[ \t]*(?:and[ \t]+)?|and[ \t]+)#\d+)*"

# Issue #225 dependency-prose patterns (the detector's original surface).
# Each match is judged per-occurrence against the quoted-prose guards in
# ``detect_prose_only_dependencies`` — a phrase inside a code span, double
# quotes, or a blockquote line describes the detector rather than declaring
# a dependency (issue #1949).
_PROSE_DEPENDENCY_PATTERNS = [
    # "do not dispatch before" and variants
    re.compile(r"do\s+not\s+dispatch\s+before", flags=re.IGNORECASE),
    # Task references (P\d+-T\d+) only in dependency context — bare task
    # marker mentions like "implements P2-T4" are deliberately NOT matched.
    re.compile(r"depends\s+on\s+[^.\n]*P\d+-T\d+", flags=re.IGNORECASE),
    # "wait for ... P\d+-T\d+ ... <completion verb>" — "Wait for P1-T5 to
    # complete first."
    re.compile(
        r"wait\s+for\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        flags=re.IGNORECASE,
    ),
    # "before/until/after ... P\d+-T\d+ ... <completion verb>"
    re.compile(
        r"(?:before|until|after)\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        flags=re.IGNORECASE,
    ),
    # "wait for" before a PR or merge event (non-task dependency prose)
    re.compile(
        r"wait\s+for\s+(?:this|that|these|those)?\s*(?:PR|merge|land)",
        flags=re.IGNORECASE,
    ),
]

# Issue #1949: ordering prose next to a same-repo issue ref. ``parse_blockers``
# only reads "Blocked by"/"Depends on" declarations, so ordering written as
# "after #12 lands" or "wait for #12 to merge" used to slip through
# undetected. Capturing group 1 is the run of issue refs the phrase orders
# on. ``_ORDERING_REF_RUN`` anchors the run's first ref with a lookbehind so
# an ``owner/repo#N`` cross-repo reference is never captured (the spec scopes
# the detector to same-repo references; a cross-repo edge cannot be declared
# as a ``## Blocked by`` item anyway — it already reads as unreadable there).
# The gap between the phrase and the ref may contain neither another ref nor
# clause punctuation (`,`/`;`), and is length-bounded: "requires a race or
# permissions failure, ... issue #357" is incidental mention, not ordering.
_ORDERING_REF_RUN = r"(?<![\w/])" + _ISSUE_REF_RUN
# Same-repo ref extractor for the ordering patterns — a ``owner/repo#N``
# token inside a match (e.g. in the post-ref text before the verb) is not a
# same-repo reference and does not count toward the blocker-set check.
_ORDERING_ISSUE_REF_RE = re.compile(r"(?<![\w/])#\d+")
# A completion verb in a forward-looking form: bare present/noun
# ("lands", "the merge", "merge") or auxiliary + participle ("is merged",
# "are BOTH merged", "has landed", "to merge"). Bare past tense ("merged",
# "landed") is excluded — "after PR #920 merged" narrates history, it does
# not order this issue behind #920. The auxiliary alternative is anchored at
# a word boundary: without it "is"/"to" match inside words like "this" /
# "into" ("after PR #920 in this merged state" read as ordering).
_COMPLETION_VERB = (
    r"\b(?:lands?|merges?|completes?|done|ships?)\b"
    r"|\b(?:is|are|be|been|being|has|have|gets?|getting|to)\s+"
    r"(?:fully\s+|both\s+|all\s+)?"
    r"\b(?:lands?|landed|landing|merges?|merged|merging"
    r"|completes?|completed|completing|ships?|shipped|shipping|done)\b"
)
# A negation immediately before the ordering phrase inverts it: "this issue
# does not wait for #207" is an explicit non-dependency and must not park.
# The whole-word negations are anchored at a word boundary so a word that
# merely *ends* in one ("casino wait for #12") is not misread; ``cannot`` is
# listed explicitly because the anchor also stops ``not`` matching inside it
# ("cannot wait" was negated via that substring before the anchor). ``n't``
# is deliberately NOT anchored — it only ever occurs as a contraction whose
# ``n`` is itself mid-word ("doesn't" has no boundary before the "n").
_NEGATED_ORDERING_RE = re.compile(
    r"(?:\b(?:cannot|not|never|no)\s+|n't\s+)(?:longer\s+)?$", flags=re.IGNORECASE
)
# Each entry: (pattern, negation_sensitive). Only the modal-verb shapes
# ("wait for", "depends on", "requires") read as inverted under a leading
# negation — "do not merge before #12 lands" is still ordering prose.
_ORDERING_ISSUE_REF_PATTERNS = [
    # "wait for #12", "wait for the merge of #12" — the issue number is the
    # ordering target; no completion verb required.
    (
        re.compile(
            rf"\bwait\s+for\b[^.\n#,;]{{0,40}}?({_ORDERING_REF_RUN})",
            flags=re.IGNORECASE,
        ),
        True,
    ),
    # "depends on #12", "requires #12". "Depends on #N" is also a structured
    # blocker phrase; the blocker-set check in the caller keeps a properly
    # declared edge from being flagged.
    (
        re.compile(
            rf"\b(?:depends\s+on|requires)\b[^.\n#,;]{{0,40}}?({_ORDERING_REF_RUN})",
            flags=re.IGNORECASE,
        ),
        True,
    ),
    # "before/until/after ... #12 ... <completion verb>" — the issue-ref
    # counterpart of the P\d+-T\d+ ordering pattern above. The verb must sit
    # in the same sub-clause as the ref (no `,`/`;`/`:` between) and in a
    # forward-looking form, so retrospective narration does not park.
    (
        re.compile(
            rf"\b(?:before|until|after)\b[^.\n#,;]{{0,50}}?({_ORDERING_REF_RUN})"
            rf"[^.\n,;:]{{0,80}}?(?:{_COMPLETION_VERB})",
            flags=re.IGNORECASE,
        ),
        False,
    ),
]

# A bare "merge"/"merges" completion token can be the *event noun* rather
# than a verb: "after #990's merge" (possessive) or "after the #1500 merge"
# (determiner-preceded ref) both park on a noun, which contradicts the
# forward-looking-verb claim — the issue narrates a merge event, it does not
# order itself behind it. Verb uses ("after #336 and #329 merge", "after PR
# #1043 merges") carry neither marker and still flag. Determiners are a
# closed grammatical class, so a literal alternation is correct here.
_MERGE_NOUN_DETERMINER_RE = re.compile(
    r"\b(?:the|a|an|this|that|these|those|its|their|our)\s+$", flags=re.IGNORECASE
)
_POSSESSIVE_CLITIC_RE = re.compile(r"^[ \t]*['’]s[ \t]+merge\b", flags=re.IGNORECASE)
_BARE_MERGE_TOKEN_RE = re.compile(r"\bmerges?$", flags=re.IGNORECASE)
_COMPLETION_VERB_RE = re.compile(_COMPLETION_VERB, flags=re.IGNORECASE)


def _is_noun_merge_match(text: str, match: re.Match[str]) -> bool:
    """True when an ordering match ends in a bare ``merge``/``merges`` that
    reads as the event noun, not the completion verb (issue #1949 rollout
    step 3).

    Two noun markers are recognised: a possessive clitic that makes the ref
    own the merge (``after #990's merge`` — ``#12's dependents merge`` is a
    *verb* merge and still flags) and a determiner directly before the ref
    (``after the #1500 merge`` — ``match.start()`` is the ordering phrase, so
    the checked span is the gap between phrase and ref). Auxiliary forms
    ("is merged", "has merged") are never nouns and the match does not end
    in a bare token for them, so they return False before either marker is
    consulted. Only callable on ``_ORDERING_ISSUE_REF_PATTERNS`` matches —
    group 1 is the ref run in every entry.

    The noun token lazily claims the match's verb slot, so a genuine
    completion verb later in the same sub-clause is recovered before
    reporting noun: "until the #12 merge completes" still orders (the tail
    window mirrors the pattern's — bounded at 80 chars, ending at the first
    ``.``, ``,``, ``;``, ``:`` or line break).
    """
    if _BARE_MERGE_TOKEN_RE.search(match.group(0)) is None:
        return False
    if not (
        _POSSESSIVE_CLITIC_RE.match(text[match.end(1) :])
        or _MERGE_NOUN_DETERMINER_RE.search(text[match.start() : match.start(1)])
    ):
        return False
    tail = text[match.end() : match.end(1) + 80]
    stop = re.search(r"[.\n,;:]", tail)
    if stop:
        tail = tail[: stop.start()]
    for verb in _COMPLETION_VERB_RE.finditer(tail):
        # A bare merge again is another noun candidate — only a different
        # verb or an auxiliary form rescues the ordering reading.
        if _BARE_MERGE_TOKEN_RE.fullmatch(verb.group(0)) is None:
            return False
    return True


def _inside_code_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a Markdown backtick code span."""
    for m in _CODE_SPAN_RE.finditer(text):
        if m.start(2) <= start and end <= m.end(2):
            return True
    return False


def _inside_quoted_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a straight-double-quote span."""
    for m in _DOUBLE_QUOTE_SPAN_RE.finditer(text):
        if m.start(1) <= start and end <= m.end(1):
            return True
    return False


def _inside_quoted_prose(text: str, start: int, end: int) -> bool:
    """True if the ``[start, end)`` range reads as quoted/example prose.

    The same guards ``parse_blockers`` applies to a candidate blocker
    declaration (issues #1454/#1847), minus the fenced-block check — callers
    run on ``_strip_fenced_blocks`` output, which already removed fenced
    blocks:

    - blockquote line: the match's line starts with ``>`` after whitespace;
    - inline code span: the match falls inside a Markdown backtick code
      span, scoped to the containing clause so a stray backtick elsewhere in
      the body cannot pair across clauses and swallow a genuine match;
    - quoted span: same, for a balanced straight-double-quote span.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if _is_blockquote_line(text[line_start:start]):
        return True
    clause_start, clause_end = _clause_bounds(text, start, end)
    clause = text[clause_start:clause_end]
    return _inside_code_span(
        clause, start - clause_start, end - clause_start
    ) or _inside_quoted_span(clause, start - clause_start, end - clause_start)


def _clause_bounds(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Return the (start, end) offsets of the sentence/line containing a match.

    Bounded by the closest preceding AND following sentence terminator
    (".", "!", "?") or line break, so each bullet/sentence is judged
    independently. The boundary characters themselves are excluded from the
    returned range.
    """
    start_boundary = max(text.rfind(ch, 0, match_start) for ch in _CLAUSE_BOUNDARY_CHARS)
    start = start_boundary + 1 if start_boundary != -1 else 0
    end_candidates = [text.find(ch, match_end) for ch in _CLAUSE_BOUNDARY_CHARS]
    end_candidates = [p for p in end_candidates if p != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return start, end


def _is_blockquote_line(line: str) -> bool:
    """True if the line's first non-space character is ``>`` (a Markdown blockquote)."""
    return line.lstrip(" \t").startswith(">")
