"""Commit-trailer exemption reader for the test-adequacy gate (issue #2220).

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark. ``janitor.check_test_adequacy`` imports
``exempt_reason_from_commit_trailers`` back (the janitor module also
re-exports it for callers that historically imported from there), and
``workflow.review`` feeds it the ``commit.message`` strings from the PR's
REST commits payload. Nothing here imports back, so there is no cycle.

The trailer channel exists because workers hold no GitHub credential and
cannot edit a PR body: the only exemption marker they can write is a
commit-message trailer (an empty ``git commit --allow-empty`` commit
carries it fine). The PR-body marker remains the other accepted channel
and is still read inline by ``check_test_adequacy``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


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
