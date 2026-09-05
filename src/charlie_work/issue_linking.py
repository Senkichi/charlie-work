"""Issue-linking and closing-keyword vocabulary (Track 2, issue #1613).

Relocated verbatim out of ``charlie_work.github`` (design doc
``docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md``,
Section 5, L06b). The L06 review (PR #1612, review 5110476115) adjudicated
that ``linked_issue_number`` and the closing-keyword chain it uses are a
text/policy primitive with zero ``gh`` coupling, so
``github_capabilities/_base.py`` -- the shared base for gh-transport
capability collaborators -- is the wrong home for it (unlike
``_LIST_LIMIT``/``_is_mutating``, which are genuine gh-transport concerns
sharing that base). ``closing_reference.py`` already duplicates this exact
keyword vocabulary on purpose "to stay free of a ``github.py`` import" -- the
codebase had already signalled that this cluster wants a neutral, non-``gh``
home.

This module has NO import of ``charlie_work.github`` or any ``gh``-coupled
module -- it is pure text/regex logic over a plain ``dict``. That neutrality
is what lets ``github_capabilities/pull_requests.py`` (``merged_prs_for_issue``,
moved there in this same leaf) import ``linked_issue_number`` from here
without a circular import through ``charlie_work.github`` (which imports
``github_capabilities`` at module load time).

``charlie_work.github`` re-exports ``linked_issue_number`` (27 external call
sites across ``workflow.py``, ``reconcile.py``, ``janitor.py``, ``cli.py``,
``dead_worker_reap.py``, ``backlog_reachability.py``, and ``worktree.py`` keep
resolving ``from charlie_work.github import linked_issue_number`` to this
module's object -- cold-import identity, not repointed to import from here
directly in this leaf) and ``iter_unnegated_closing_keyword_matches``
(``closing_keyword_gate.py``'s own import). Repointing those call sites to
import directly from ``charlie_work.issue_linking``, and letting
``closing_reference.py`` import ``_CLOSING_KEYWORDS_ALT`` from here instead
of duplicating it, are both deferred follow-ups -- issue #1613's own step 1
marks the repoint optional (same PR or a follow-up); this PR defers it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

# GitHub's own issue-closing keyword set, used here to decide whether a `#N`
# reference in freeform text actually links the PR to issue N. Shared between
# the matching regex below and `_CLOSING_KEYWORD_DEFANG_RE` so the two
# patterns cannot drift apart.
_CLOSING_KEYWORDS_ALT = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
_CLOSING_KEYWORD_REF = re.compile(_CLOSING_KEYWORDS_ALT + r"\s+#(\d+)", flags=re.IGNORECASE)

# The orchestrator's own branch convention (agent/issue-N-slug). A head ref is
# the trusted signal because the orchestrator created it at dispatch.
_BRANCH_ISSUE_REF = re.compile(r"issue[-_/](\d+)", flags=re.IGNORECASE)

# Negation words/contractions that, when found shortly before a closing
# keyword match, mean the keyword is being negated ("does not fix #649")
# rather than asserting a real closing action. Kept as a module-level
# constant — never inline literals at the match site — so the vocabulary is
# audited and extended in exactly one place.
_NEGATION_WHOLE_WORDS = ("not", "never", "without", "cannot")
# Matched as a bare substring (no leading \b): "doesn't"/"can't" have no word
# boundary immediately before the "n" in "n't" — the apostrophe is not a
# \w character, so "doesn" + "'t" is one continuous \w-run from \b's
# perspective and a \b-anchored "n't" would silently never match.
_NEGATION_CONTRACTION_SUFFIX = "n't"
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(_NEGATION_WHOLE_WORDS) + r")\b|" + re.escape(_NEGATION_CONTRACTION_SUFFIX),
    flags=re.IGNORECASE,
)
# How many characters back to look for a negation word before a closing
# keyword. 32 comfortably covers every negation phrase in the acceptance
# criteria ("does not " = 9 chars, "without " = 8) with headroom for a short
# intervening clause. The tradeoff is deliberate and biased toward the safe
# direction: at this width, "This is not a revert. Fixes #700" is treated as
# negated even though the negation and the keyword sit in different
# sentences. A missed binding leaves the issue in its current label state
# (safe); a false binding silently marks live work done (unsafe) — so
# over-triggering the guard is acceptable, under-triggering is not.
_NEGATION_LOOKBEHIND_CHARS = 32


def _has_preceding_negation(text: str, match_start: int) -> bool:
    """True if a negation word/contraction appears shortly before match_start."""
    window_start = max(0, match_start - _NEGATION_LOOKBEHIND_CHARS)
    # Pass pos/endpos (not a string slice) so \b at the window edge is still
    # resolved against the real surrounding text, not an artificial cut.
    return bool(_NEGATION_RE.search(text, window_start, match_start))


def iter_unnegated_closing_keyword_matches(text: str) -> Iterator[re.Match[str]]:
    """Yield every `_CLOSING_KEYWORD_REF` match in ``text`` not preceded by negation.

    This is the shared core scanning primitive (finditer over every
    keyword+``#N`` occurrence, filtered by the negation lookback) behind both
    consumers that need it:

    - `_first_unnegated_closing_keyword_match` (below) takes only the first —
      `linked_issue_number`'s label-transition binding only ever needs one
      match to bind an issue.
    - `closing_keyword_gate.find_unexpected_closing_references` (issue #790)
      needs *every* match across a whole PR body plus every commit message,
      because GitHub's native auto-close-on-merge scans all of those
      surfaces for every closing-keyword reference, not just the first.

    Both consume this one generator rather than each hand-rolling their own
    `finditer` + negation-lookback scan, so the two callers cannot drift
    apart on what counts as a live closing reference.
    """
    for match in _CLOSING_KEYWORD_REF.finditer(text):
        if not _has_preceding_negation(text, match.start()):
            yield match


def _first_unnegated_closing_keyword_match(text: str) -> re.Match[str] | None:
    """Return the first `_CLOSING_KEYWORD_REF` match not preceded by negation.

    A negated match (e.g. "does not fix #649") must not shadow a later,
    genuine match in the same field (e.g. "...but this PR also fixes #700") —
    scanning continues past it instead of giving up on the whole field.
    """
    return next(iter_unnegated_closing_keyword_matches(text), None)


def linked_issue_number(
    pr: dict[str, Any],
    *,
    is_cross_repository: bool | None,
    branch_prefix: str,
    branch_issue_validator: Callable[[int], bool] | None = None,
) -> int | None:
    """Resolve the issue a PR is bound to, safe against hijack.

    A bare ``#N`` substring in an attacker-controlled PR *title* must never
    bind the PR to issue N — that let any external PR author drive another
    issue's label/merge transitions. So: trust the head ref only when the PR
    is same-repo (isCrossRepository == false) AND the branch starts with the
    configured ``branch_prefix``. For fork PRs, never bind for lifecycle
    purposes — return None before any keyword scan. (GitHub's own auto-close
    on merge is GitHub's policy for issue state; the orchestrator's label
    lifecycle is ours.)

    When is_cross_repository is None (provenance unknown), treat as
    cross-repo for trust purposes — bind nothing via branch name or closing
    keyword (fail closed).

    A closing keyword preceded by a negation ("does not fix #649") also does
    not bind — see `_first_unnegated_closing_keyword_match`. This prevents a
    false LABEL TRANSITION in charlie-work's own state machine; it has no
    effect on GitHub's own issue auto-close, which is a separate mechanism
    charlie-work does not control.

    Issue #1229: ``branch_issue_validator``, when provided, is called with
    the candidate issue number parsed from the branch name. If it returns
    False (the number does not correspond to a real open issue), the
    branch-name binding is rejected and the function falls through to the
    closing-keyword path instead. This prevents a stale branch-name number
    (e.g. a branch ``agent/issue-709-…`` left over from a merged issue/PR
    #709, reused by an unrelated issue-less PR) from silently keying a
    rework episode under ``state["issues"]["709"]`` and colliding with the
    unrelated issue's lifecycle. When the validator is None (the default),
    the branch-name binding is trusted unconditionally — preserving the
    behavior of callers that do not need the validation.
    """
    # Cross-repo PRs or unknown provenance never bind for lifecycle purposes
    if is_cross_repository is True or is_cross_repository is None:
        return None

    head = str(pr.get("headRefName") or "")
    match = _BRANCH_ISSUE_REF.search(head)
    if match:
        # Only trust the branch ref when:
        # 1. PR is same-repo (is_cross_repository is not True)
        # 2. Branch starts with the configured prefix
        # 3. (Issue #1229) The parsed number is a real open issue, when a
        #    validator is supplied. Without a validator, trust unconditionally
        #    to preserve existing caller behavior.
        has_correct_prefix = head.startswith(branch_prefix)
        if has_correct_prefix:
            candidate = int(match.group(1))
            if branch_issue_validator is None or branch_issue_validator(candidate):
                return candidate
            # Branch-name number is stale/unmatched — fall through to the
            # closing-keyword path rather than binding to a non-existent or
            # closed issue.
    # For same-repo PRs, trust closing keywords in title/body — but a
    # negated keyword ("does not fix #649") must not bind; see
    # `_first_unnegated_closing_keyword_match`.
    for text in (str(pr.get("title") or ""), str(pr.get("body") or "")):
        match = _first_unnegated_closing_keyword_match(text)
        if match:
            return int(match.group(1))
    return None
