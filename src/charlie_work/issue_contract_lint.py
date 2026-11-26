"""Issue-body contract lint: the dispatchable-body rules, callable in-tree (issue #3115).

The rules for a dispatchable issue body were enforced only by an operator-local
script outside this repo. Anything in-tree that files issues (the Foreman's
``file_issue`` Apply step, docs/design/2026-11-25-foreman-layer-design.md
section 8.3) needs the same check as a library call, so this module moves it
in-tree. Every rule delegates to the code the dispatcher already uses —
``github_body_scan.parse_blockers``, ``github_body_scan.
detect_prose_only_dependencies``, ``citation_check.parse_citations`` — so the
lint cannot drift from the dispatcher; no rule is re-implemented here where a
dispatcher function exists.

The closing-keyword rule has no dispatcher function to delegate to: the
dispatcher needs only the first unnegated binding (``issue_linking.
iter_unnegated_closing_keyword_matches``), while the lint flags every
occurrence, including the colon form (``Closes: #N``), the qualified
``owner/repo#N`` form (``closing_reference._CLOSING_LINE_RE``'s qualifier), and
negated phrasing —
GitHub's auto-close-on-merge scans for the keyword syntax with no negation
awareness at all (issue #790), so an issue body must not carry the trigger
syntax in any form. The keyword *vocabulary* is still delegated: the pattern is
built from ``issue_linking._CLOSING_KEYWORDS_ALT``, the same GitHub keyword set
the dispatcher matches on, so the two cannot drift apart on which words count.
The whitespace between keyword and reference is ``\\s*`` — a superset of the
``\\s+`` in the dispatcher's ``issue_linking._CLOSING_KEYWORD_REF`` — so a
hard-wrapped ``fixes\\n#5`` the dispatcher binds is flagged here too: the lint
may forbid more than the dispatcher binds, never less.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .citation_check import parse_citations
from .github_body_scan import detect_prose_only_dependencies, parse_blockers
from .issue_linking import _CLOSING_KEYWORDS_ALT

# A closing keyword followed by an issue reference — the trigger syntax
# GitHub's auto-close-on-merge consumes regardless of authorial intent (issue
# #790). The keyword alternation is shared with the dispatcher via
# ``issue_linking._CLOSING_KEYWORDS_ALT``; the lint adds an optional colon
# (``Closes: #N``) and no negation filtering, because the contract forbids the
# syntax entirely rather than judging which occurrence would bind. The
# optional ``owner/repo`` qualifier mirrors ``closing_reference._CLOSING_LINE_RE``
# — ``Fixes owner/repo#N`` is the same trigger syntax aimed cross-repo.
# The whitespace is ``\s*``, a superset of the ``\s+`` in the dispatcher's
# ``issue_linking._CLOSING_KEYWORD_REF``: a hard-wrapped ``this fixes\n#5``
# binds for the dispatcher, so the lint must flag it too — a ``[ \t]`` class
# would leave the lint narrower than the code it claims to superset.
_CLOSING_KEYWORD_ISSUE_REF_RE = re.compile(
    r"\b" + _CLOSING_KEYWORDS_ALT + r"\b\s*:?\s*(?:[\w.-]+/[\w.-]+)?#\d+",
    flags=re.IGNORECASE,
)


def lint_issue_body(text: str, expected_blockers: frozenset[int]) -> tuple[str, ...]:
    """Check an issue body against the dispatchable-body contract.

    Returns one human-readable problem per violation; an empty tuple means the
    body is clean. Every check delegates to the parser the dispatcher already
    runs:

    - The blockers ``github_body_scan.parse_blockers`` reads must equal
      ``expected_blockers`` exactly — a body that silently declares a
      different edge set than the filer intended is reported with both sets.
    - ``github_body_scan.detect_prose_only_dependencies`` must be false —
      ordering written only in prose parks the issue for a human.
    - ``citation_check.parse_citations`` must find no ``path:line`` citation —
      line numbers rot; cite the symbol instead (docs/agents/issue-tracker.md
      convention, issue #1000).
    - No closing keyword followed by an issue reference (bare ``#N`` or
      qualified ``owner/repo#N``, including a hard-wrapped ``fixes\\n#5``) —
      the body must not carry syntax GitHub's auto-close would act on.
    """
    problems: list[str] = []

    parsed = frozenset(parse_blockers(text))
    expected = frozenset(expected_blockers)
    if parsed != expected:
        problems.append(
            f"blocker set mismatch: body parses to {sorted(parsed)}, expected {sorted(expected)}"
        )

    if detect_prose_only_dependencies(text):
        problems.append(
            "prose-only dependency declaration: the ordering is written in "
            "prose the dispatcher cannot parse; declare it with `Blocked by #N` "
            "lines under a `## Blocked by` heading instead"
        )

    for citation in parse_citations(text):
        problems.append(
            f"path-and-line citation {citation.raw!r}: line numbers rot; "
            "cite the symbol, not the line"
        )

    for match in _CLOSING_KEYWORD_ISSUE_REF_RE.finditer(text):
        problems.append(
            f"closing-keyword issue reference {match.group(0)!r}: an issue "
            "body must not carry auto-close syntax; drop the keyword or the #N"
        )

    return tuple(problems)


def render_blocker_section(blockers: Iterable[int]) -> str:
    """Render the ``## Blocked by`` section in the exact form the dispatcher reads.

    One blocker per line, ascending, each of the exact form ``Blocked by #N``
    that docs/agents/issue-tracker.md prescribes and the design's Apply step
    renders (``Blocked by #N`` lines, one per line — docs/design/
    2026-11-25-foreman-layer-design.md section 8.3); the single line ``None``
    when the set is empty. The returned text ends with a newline. Appended to
    a clean body, it lints clean against the same blocker set.
    """
    numbers = sorted({int(n) for n in blockers})
    lines = ["## Blocked by", ""]
    if numbers:
        lines.extend(f"Blocked by #{n}" for n in numbers)
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"
