"""Render issue comments into the worker prompt (issue #872).

A worker only ever saw ``issue.body``. Every correction, clarification, and
scope change written as a *comment* after filing was invisible to it, so a
worker could confidently implement a plan the humans had already talked it out
of. The motivating case is issue #868, where the operator corrected the
proposed approach in a comment and the dispatched worker never saw it.

Three design points that look arbitrary and are not:

**Why ``authorAssociation`` and not ``viewerDidAuthor``.** The obvious way to
drop bot chatter is to exclude comments the authenticated user wrote. That
inverts the filter here. The orchestrator authenticates as the *operator's own
account* (``gh api user`` -> ``Senkichi``), so ``viewerDidAuthor`` is ``true``
for exactly the human corrections this module exists to deliver, and ``false``
for real bots like ``aviator-app``. The GraphQL comment payload carries no
``is_bot`` field, so association is the discriminator that actually separates
the two: a human collaborator is OWNER/MEMBER/COLLABORATOR, an app is NONE.
The allow-list is config-driven rather than a hardcoded bot deny-list, so a new
bot needs no code change to be excluded.

**Why the fence width is computed, not fixed.** This is a correctness problem
first and a security problem second. A comment on a *developer* issue tracker
routinely contains a code block, and a body carrying its own ``` would close a
fixed three-backtick fence early -- dropping the "this is quoted material"
framing and letting the remainder render as prompt-level markdown, where a
heading in the comment becomes indistinguishable from a heading the
orchestrator wrote. Measured on this repo: 61 of 100 open issue *bodies*
contain a fence, so this is the common case, not an edge case. It is also the
injection that ``prompts.render_prompt``'s single-substitution design prevents
at the templating layer, reintroduced at the formatting layer.
``markdown_fence.fenced_block`` picks a delimiter longer than any backtick run
in the content, per CommonMark. (The identical defect in the fixed fence around
``$issue_body`` was tracked separately as #883 and fixed there, since it changed
the rendered prompt for those 61 issues -- a different blast radius than this
change, which is a no-op for an issue with no usable comments.)

**Why the block is empty rather than absent.** ``render_issue_comments``
returns ``""`` when nothing survives filtering, and the template attaches the
placeholder to the end of the preceding line, so an issue with no usable
comments renders a prompt that is byte-identical to the pre-#872 output. That
is a hard requirement, not a nicety: it keeps this change a no-op for the
overwhelmingly common case and makes any prompt diff attributable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .markdown_fence import fenced_block

__all__ = [
    "DEFAULT_INCLUDED_ASSOCIATIONS",
    "render_issue_comments",
    "select_comments",
]

# Author associations treated as "a human working on this repo". GitHub reports
# NONE for apps/bots (aviator-app) and for drive-by commenters; both are
# excluded by omission rather than by naming them.
DEFAULT_INCLUDED_ASSOCIATIONS: tuple[str, ...] = ("OWNER", "MEMBER", "COLLABORATOR")


def _normalized(values: Iterable[str]) -> frozenset[str]:
    return frozenset(value.strip().casefold() for value in values if value.strip())


def select_comments(
    comments: Sequence[Mapping[str, Any]] | None,
    *,
    included_associations: Iterable[str] = DEFAULT_INCLUDED_ASSOCIATIONS,
    excluded_authors: Iterable[str] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Filter ``comments`` down to human, on-topic, non-hidden entries.

    Order is preserved (the GitHub payload is chronological). Anything whose
    shape is unexpected is dropped rather than guessed at -- a malformed entry
    is not worth risking a garbled prompt over.
    """
    allowed = _normalized(included_associations)
    denied = _normalized(excluded_authors)
    selected: list[Mapping[str, Any]] = []
    for comment in comments or ():
        if not isinstance(comment, Mapping):
            continue
        # A minimized comment was explicitly hidden by a human (spam, outdated,
        # resolved). Honoring that is the whole point of the flag.
        if comment.get("isMinimized"):
            continue
        association = str(comment.get("authorAssociation") or "").casefold()
        if association not in allowed:
            continue
        author = comment.get("author") or {}
        login = str(author.get("login") or "").casefold() if isinstance(author, Mapping) else ""
        if login and login in denied:
            continue
        if not str(comment.get("body") or "").strip():
            continue
        selected.append(comment)
    return tuple(selected)


def _budgeted(
    comments: Sequence[Mapping[str, Any]],
    *,
    max_comments: int,
    max_chars: int,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Keep the most recent comments that fit, returning ``(kept, dropped)``.

    Newest-first accumulation, chronological output: when the budget binds, the
    *latest* correction is the one a worker most needs, and it is also the one a
    naive head-of-list truncation would discard.
    """
    if max_comments > 0:
        kept = list(comments[-max_comments:])
    else:
        kept = list(comments)
    dropped = len(comments) - len(kept)

    if max_chars > 0:
        budgeted: list[Mapping[str, Any]] = []
        used = 0
        for comment in reversed(kept):
            size = len(str(comment.get("body") or ""))
            # Always keep the newest comment even if it alone blows the budget;
            # an empty section would be a worse failure than an oversized one.
            if budgeted and used + size > max_chars:
                break
            budgeted.append(comment)
            used += size
        dropped += len(kept) - len(budgeted)
        kept = list(reversed(budgeted))

    return tuple(kept), dropped


def render_issue_comments(
    comments: Sequence[Mapping[str, Any]] | None,
    *,
    included_associations: Iterable[str] = DEFAULT_INCLUDED_ASSOCIATIONS,
    excluded_authors: Iterable[str] = (),
    max_comments: int = 20,
    max_chars: int = 12000,
    sanitize: Any = None,
) -> str:
    """Render a worker-prompt section for ``comments``, or ``""`` if none apply.

    ``sanitize`` is applied to each body before it is embedded (the caller
    passes ``github.defang_closing_keywords`` so a comment's "closes #123"
    cannot auto-close issues when a worker copies it into a PR body). It is a
    parameter rather than a direct import to keep this module free of the
    GitHub layer and trivially testable.

    The returned block begins with a blank line and does *not* end with one, so
    the template can attach it to the end of an existing line.
    """
    selected = select_comments(
        comments,
        included_associations=included_associations,
        excluded_authors=excluded_authors,
    )
    if not selected:
        return ""

    kept, dropped = _budgeted(selected, max_comments=max_comments, max_chars=max_chars)
    if not kept:
        return ""

    lines = [
        "",
        "",
        "## Issue comments",
        "",
        "These were posted after the issue body above. Treat them as amendments to"
        " it: where a comment conflicts with the body, **the comment wins**, and"
        " the latest comment wins over an earlier one.",
    ]
    if dropped:
        lines += [
            "",
            f"> _{dropped} earlier comment(s) omitted to fit the prompt budget."
            " Read the full thread on GitHub if the discussion below references"
            " something you cannot see._",
        ]

    for comment in kept:
        author = comment.get("author") or {}
        login = str(author.get("login") or "unknown") if isinstance(author, Mapping) else "unknown"
        association = str(comment.get("authorAssociation") or "").strip()
        created = str(comment.get("createdAt") or "").strip()
        heading = f"### @{login}"
        if association:
            heading += f" ({association})"
        if created:
            heading += f" - {created}"

        body = str(comment.get("body") or "").strip()
        if sanitize is not None:
            body = sanitize(body)
        lines += ["", heading, "", fenced_block(body, "md")]

    return "\n".join(lines)
