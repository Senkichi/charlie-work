"""Exemption readers for the test-adequacy gate (issue #2220).

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark. ``janitor.check_test_adequacy`` imports
``exempt_reason_for_pr`` back (and re-exports
``exempt_reason_from_commit_trailers`` for callers that historically
imported it from ``janitor``), while ``workflow.review`` feeds the gate the
``commit.message`` strings from the PR's REST commits payload. Nothing here
imports back, so there is no cycle.

Two channels claim the ``exempt_marker`` exemption: a ``<marker> <reason>``
line in the PR body, and the same line as a trailer in the last paragraph
of a commit message. The trailer channel exists because workers hold no
GitHub credential and cannot edit a PR body (an empty
``git commit --allow-empty`` commit carries it fine); the body marker is
checked first and wins on disagreement.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


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
