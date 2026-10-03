"""Exemption readers for the test-adequacy gate (issue #2220).

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark. ``janitor.check_test_adequacy`` imports
``exempt_reason_for_pr`` back (and re-exports
``exempt_reason_from_commit_trailers`` for callers that historically
imported it from ``janitor``), while ``workflow.review`` feeds the gate the
``commit.message`` strings from the PR's REST commits payload.
``janitor._check_no_op_rework`` imports ``newly_claimed_exempt_reason`` for
the issue #2281 exemption-claim escape. Nothing here imports back, so
there is no cycle.

Two channels claim the ``exempt_marker`` exemption: a ``<marker> <reason>``
line in the PR body, and the same line as a trailer in the last paragraph
of a commit message. The trailer channel exists because workers hold no
GitHub credential and cannot edit a PR body (an empty
``git commit --allow-empty`` commit carries it fine); the body marker is
checked first and wins on disagreement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def exempt_reason_for_pr(body: str, commit_messages: Sequence[str], marker: str) -> str:
    """Return the exemption reason claimed for a PR, or ``""``.

    The PR-body ``<marker> <reason>`` line is checked first; a commit trailer
    is the fallback channel — workers hold no GitHub credential and cannot
    edit the body (issue #2220). Only a non-empty reason exempts.
    """
    exempt_re = re.compile(rf"^{re.escape(marker)}\s*(?P<reason>.+)$", re.M)
    match = exempt_re.search(body)
    if match:
        reason = match.group("reason").strip()
        if reason:  # Non-empty reason required
            return reason
    return exempt_reason_from_commit_trailers(commit_messages, marker)


def exempt_reason_from_commit_trailers(messages: Sequence[str], marker: str) -> str:
    """Return the first non-empty ``<marker> <reason>`` trailer reason, or ``""``.

    A trailer is a line in the *last* paragraph of a commit message, and that
    paragraph must not be the subject (git's own trailer rule), so a message
    that merely quotes the marker in body prose does not count.
    """
    trailer_re = re.compile(rf"^{re.escape(marker)}\s*(?P<reason>.+)$")
    for message in messages:
        paragraphs = re.split(r"\n\s*\n", message.replace("\r\n", "\n").strip())
        if len(paragraphs) < 2:
            continue
        for line in paragraphs[-1].splitlines():
            match = trailer_re.match(line.strip())
            if match and match.group("reason").strip():
                return match.group("reason").strip()
    return ""


def newly_claimed_exempt_reason(
    body: str,
    commits: Sequence[Mapping[str, Any]],
    reviewed_head_sha: str,
    marker: str,
) -> str:
    """Return the exemption reason first claimed after ``reviewed_head_sha``, or ``""``.

    Companion to the #2220 trailer channel for the janitor's no-op rework
    gate (issue #2281): the gate compares patch-ids, but the remedy its own
    verdict text prescribes — an empty commit carrying a ``<marker>``
    trailer — produces no diff delta, so the patch-id alone flags the
    gate's own remedy as a no-op. This reader compares two claims: the
    PR's live claim (``exempt_reason_for_pr`` over ``body`` plus every
    commit message) and the claim as of ``reviewed_head_sha`` (the same
    function over the messages of commits at or before that head). A live
    claim the reviewed head did not already make was claimed by the rework
    itself and is returned; equal or absent claims return ``""`` — an
    exemption already present at the reviewed head must not satisfy the
    gate a second time.

    ``commits`` is the REST ``pulls/{n}/commits`` payload — oldest first,
    each entry carrying top-level ``sha`` and ``commit.message``, matching
    what ``GitHub.pr_commits`` returns. A ``reviewed_head_sha`` absent from
    the list (a force-push or rebase replaced the reviewed history), an
    empty list, an empty marker, or an empty ``reviewed_head_sha`` all
    yield ``""``: undeterminable newness fails closed, the same convention
    as every other baseline the no-op gate consults.

    ``body`` is the LIVE body on both sides deliberately: a body edit that
    added a marker already satisfied the #1939 body-change escape before
    this reader runs, and when that escape cannot decide (no recorded
    baseline) assuming the marker is old is the fail-closed direction.
    """
    if not marker or not reviewed_head_sha:
        return ""
    messages_now: list[str] = []
    messages_then: list[str] = []
    head_seen = False
    for entry in commits:
        if not isinstance(entry, Mapping):
            continue
        commit = entry.get("commit")
        message = commit.get("message") if isinstance(commit, Mapping) else None
        text = str(message or "")
        messages_now.append(text)
        if not head_seen:
            messages_then.append(text)
            if str(entry.get("sha") or "") == reviewed_head_sha:
                head_seen = True
    if not head_seen:
        return ""
    reason_now = exempt_reason_for_pr(body, messages_now, marker)
    if not reason_now:
        return ""
    if exempt_reason_for_pr(body, messages_then, marker):
        return ""
    return reason_now
