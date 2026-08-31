"""CI gate: ratchet lint against new private-repo-slug mentions (issue #1502).

The public-flip decision (2026-08-28) accepted residual mentions of private
sibling repo slugs in tracked files (test fixtures, examples, a live
heartbeat-suppressions entry) but rejected *unbounded growth*: without a gate,
every future salvage PR can silently add new mentions.  This gate fails when
a PR adds a net-new mention of a configured private slug in a tracked file,
ratchet-style against a baseline file at the repo root.

The slug list (the "pattern source") is config, not hardcoded slugs in the
check body: it lives in ``.private-slug-baseline.json`` alongside a ``total``
count that forms the ratchet baseline.  A PR that adds a net-new mention MUST
also bump the baseline ``total`` by at least the net-new count -- that bump is
tamper-evident in diff review, the same way ``.attachment-budgets.json``
bumps are.  Without the bump, the gate fails; with it, a reviewer sees both
the new mention and the baseline increase in the same diff and can ask why.

Moves during a refactor (remove from one place, add to another) produce zero
net-new mentions and do not trigger the gate -- a pure added-lines scan
would false-positive on the moved line, which is why the gate counts
net-new (added minus removed) rather than added alone.  The baseline file
itself is excluded from the scan because it *lists* slugs as config -- those
are not "mentions" of the private repos.

The pure scanning function (:func:`find_slug_mentions_in_diff`) never raises
and never touches I/O; the CLI command
(:func:`run_private_slug_check_command` in ``cli.py``) owns the ``git diff``
subprocess, the baseline-file read, and the exit-code decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SlugMentionFinding:
    """A line in a diff that mentions a configured private slug (issue #1502).

    Frozen to match the project invariant that config/value objects are
    immutable.  ``line_number`` is the line's position in the new file (for
    added lines) or the old file (for removed lines), so the failure message
    can name the exact location.
    """

    path: str
    line_number: int
    slug: str
    content: str


@dataclass(frozen=True)
class SlugMentionDelta:
    """Net-new private-slug mentions in a diff (issue #1502).

    ``added`` and ``removed`` are the mentions found on added and removed
    lines respectively.  ``net_new`` is ``len(added) - len(removed)``: a
    move (remove + add) produces zero net-new, a pure addition produces
    positive net-new, and a pure removal produces negative net-new (the
    ratchet tightens).
    """

    added: list[SlugMentionFinding]
    removed: list[SlugMentionFinding]

    @property
    def net_new(self) -> int:
        return len(self.added) - len(self.removed)


def _slug_patterns(slugs: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile word-boundary regexes for each slug.

    Each slug is matched as a whole word (``\\bslug\\b``) so that a hyphenated
    slug matches inside ``Owner/slug-name`` (the ``/`` is a word boundary) but
    not inside a longer identifier, and an underscore slug matches in
    ``../slug_name`` but not in ``my_slug_name`` (the ``_`` is a word
    character, so no boundary inside the identifier).
    Hyphens and underscores in slugs are regex-escaped.
    """
    return [(slug, re.compile(r"\b" + re.escape(slug) + r"\b")) for slug in slugs]


def _line_mentions_slug(content: str, patterns: list[tuple[str, re.Pattern[str]]]) -> str | None:
    """Return the first slug in *content* that matches a pattern, or ``None``."""
    for slug, pattern in patterns:
        if pattern.search(content):
            return slug
    return None


def find_slug_mentions_in_diff(
    diff_text: str,
    slugs: list[str],
    *,
    exclude_paths: frozenset[str] | None = None,
) -> SlugMentionDelta:
    """Scan a unified diff for added/removed lines mentioning a private slug.

    Parses *diff_text* as a unified diff (the output of ``git diff``) and
    checks every added line (``+``) and removed line (``-``) for mentions of
    any slug in *slugs*.  Returns a :class:`SlugMentionDelta` whose
    ``net_new`` property is the count to check against the ratchet baseline.

    *exclude_paths* is a set of repo-relative paths to skip entirely (e.g.
    the baseline file itself, which lists slugs as config -- those are not
    "mentions" of the private repos).  Pass ``None`` for no exclusions.

    The diff is read as text (the caller decides encoding; ``git diff``
    output is UTF-8 in this repo).  Binary file diffs
    (``Binary files a/... and b/... differ``) carry no line-level content
    and are skipped naturally (no ``+++`` header means ``current_path``
    stays empty for that hunk's added lines).

    Never raises -- a malformed diff simply yields an empty delta.
    """
    patterns = _slug_patterns(slugs)
    added: list[SlugMentionFinding] = []
    removed: list[SlugMentionFinding] = []
    new_path = ""
    old_path = ""
    new_line_number = 0
    old_line_number = 0

    for line in diff_text.splitlines():
        # Track the old-file path from "--- a/path" (needed for removed-line
        # attribution, especially on file deletion where the new path is
        # /dev/null).
        if line.startswith("--- "):
            rest = line[4:]
            if rest == "/dev/null":
                old_path = ""
            elif rest.startswith("a/"):
                old_path = rest[2:]
            else:
                old_path = rest
            continue

        # Track the new-file path from "+++ b/path" (for added-line
        # attribution).  /dev/null means the file was deleted.
        if line.startswith("+++ "):
            rest = line[4:]
            if rest == "/dev/null":
                new_path = ""
            elif rest.startswith("b/"):
                new_path = rest[2:]
            else:
                new_path = rest
            continue

        # Hunk header: @@ -old,count +new,count @@ -- reset line counters.
        if line.startswith("@@"):
            new_line_number = _parse_new_start(line)
            old_line_number = _parse_old_start(line)
            continue

        # Added line ("+content").  The true "+++ b/path" header was already
        # consumed above (it requires a trailing space), so an added line
        # whose content starts with "++" is correctly treated as an added
        # line, not mistaken for a header.
        if line.startswith("+"):
            if new_path and not _excluded(new_path, exclude_paths):
                content = line[1:]
                line_no = new_line_number
                slug = _line_mentions_slug(content, patterns)
                if slug is not None:
                    added.append(
                        SlugMentionFinding(
                            path=new_path,
                            line_number=line_no,
                            slug=slug,
                            content=content,
                        )
                    )
            new_line_number += 1
            continue

        # Removed line ("-content").  The true "--- a/path" header was
        # already consumed above.
        if line.startswith("-"):
            if old_path and not _excluded(old_path, exclude_paths):
                content = line[1:]
                line_no = old_line_number
                slug = _line_mentions_slug(content, patterns)
                if slug is not None:
                    removed.append(
                        SlugMentionFinding(
                            path=old_path,
                            line_number=line_no,
                            slug=slug,
                            content=content,
                        )
                    )
            old_line_number += 1
            continue

        # Context line (" content") -- advance both counters.
        if line.startswith(" "):
            new_line_number += 1
            old_line_number += 1

    return SlugMentionDelta(added=added, removed=removed)


def _excluded(path: str, exclude_paths: frozenset[str] | None) -> bool:
    """Return ``True`` if *path* is in *exclude_paths*."""
    if exclude_paths is None:
        return False
    return path in exclude_paths


def count_slug_mentions_in_text(text: str, slugs: list[str]) -> int:
    """Count lines in *text* that mention any of *slugs*.

    Used by the baseline regeneration command to count existing mentions in
    a file's content.  A line mentioning the same slug twice counts once
    (consistent with :func:`find_slug_mentions_in_diff`, which produces one
    finding per line).  A line mentioning two different slugs also counts
    once (the first match wins, matching the diff scan's behavior).
    """
    patterns = _slug_patterns(slugs)
    count = 0
    for line in text.splitlines():
        if _line_mentions_slug(line, patterns) is not None:
            count += 1
    return count


def _parse_new_start(hunk_header: str) -> int:
    """Extract the starting line number of the new-file side from a ``@@`` header.

    A hunk header looks like ``@@ -10,5 +12,7 @@ optional context``.  The
    ``+12`` part is the starting line number in the new file.  Returns 0 if
    the header is malformed (the caller treats 0 as "unknown line number").
    """
    plus_idx = hunk_header.find("+", 3)  # skip "@@ "
    if plus_idx == -1:
        return 0
    rest = hunk_header[plus_idx + 1 :]
    num_str = ""
    for ch in rest:
        if ch.isdigit():
            num_str += ch
        else:
            break
    return int(num_str) if num_str else 0


def _parse_old_start(hunk_header: str) -> int:
    """Extract the starting line number of the old-file side from a ``@@`` header.

    A hunk header looks like ``@@ -10,5 +12,7 @@``.  The ``-10`` part is the
    starting line number in the old file.  Returns 0 if malformed.
    """
    minus_idx = hunk_header.find("-", 3)  # skip "@@ "
    if minus_idx == -1:
        return 0
    rest = hunk_header[minus_idx + 1 :]
    num_str = ""
    for ch in rest:
        if ch.isdigit():
            num_str += ch
        else:
            break
    return int(num_str) if num_str else 0
