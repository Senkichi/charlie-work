"""CI gate: detect mojibake introduced by worker edit tooling (issue #1057).

A worker agent editing a source file silently re-encoded every non-ASCII
character in it, converting all 27 em-dashes (U+2014) in
``src/charlie_work/fleet_dispatch.py`` into the three-byte UTF-8-read-as-cp1252
mojibake sequence (``\\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9d``).  Nothing in CI
caught it: ``ruff`` passes (mojibake inside a comment is valid UTF-8, just wrong
text), ``pytest`` passes (no test asserts on comment content), and human diff
review tends to skim comment-only hunks.

The corruption mechanism: a misconfigured editing path reads UTF-8 bytes as
cp1252 (the Windows code page), then re-encodes the result as UTF-8.  The
resulting text is valid UTF-8 — it just decodes to the wrong characters.  For
example, an em-dash ``—`` (U+2014, UTF-8 ``\\xe2\\x80\\x94``) becomes
``â€"`` (U+00E2, U+20AC, U+201D) when the three UTF-8 bytes are decoded as
cp1252, and that string re-encoded as UTF-8 is the
``\\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9d`` byte sequence the issue documents.

Detection is **derived from the encoding process itself**, not a hardcoded list
of bad byte sequences.  The reversal of the corruption is:

    text.encode("cp1252").decode("utf-8")

If that round-trip succeeds and produces *different* text, the original was
mojibake.  This catches any UTF-8/cp1252 round trip — not just the specific
em-dash sequence — so it does not fail open on a sequence nobody thought to
enumerate (the issue's stated preference).

The pure scanning function (:func:`is_mojibake` /
:func:`find_mojibake_in_diff`) never raises and never touches I/O; the CLI
command (:func:`run_mojibake_check_command` in ``cli.py``) owns the ``git diff``
subprocess and the exit-code decision.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class MojibakeFinding:
    """A line added in a diff that contains cp1252/UTF-8 mojibake (issue #1057).

    Frozen to match the project invariant that config/value objects are
    immutable.  ``recovered`` is the text the line *should* contain if the
    mojibake is reversed — shown to the operator so the fix is obvious
    (restore the original characters, never replace them with ASCII hyphens).
    """

    path: str
    line_number: int
    content: str
    recovered: str


def _try_recover(text: str) -> str | None:
    """Return the corrected text if *text* is cp1252/UTF-8 mojibake, else ``None``.

    Reverses the corruption by encoding *text* back as cp1252 (recovering the
    original UTF-8 bytes) and decoding as UTF-8 (recovering the original text).
    Returns ``None`` when the round-trip fails or produces identical text
    (i.e. the input was not mojibake).  Never raises.
    """
    try:
        recovered = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if recovered != text:
        return recovered
    return None


def _cp1252_chunks(text: str) -> Iterator[str]:
    """Yield maximal substrings of *text* whose characters are all in cp1252.

    Used as a fallback when the whole line cannot be encoded as cp1252 (e.g.
    it contains emoji alongside mojibake).  Splitting on non-cp1252 characters
    is safe because those characters cannot be part of a cp1252/UTF-8 round
    trip — they are outside the cp1252 code page entirely.
    """
    current: list[str] = []
    for ch in text:
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            if current:
                yield "".join(current)
                current = []
            continue
        current.append(ch)
    if current:
        yield "".join(current)


def is_mojibake(text: str) -> bool:
    """Return ``True`` if *text* contains cp1252/UTF-8 mojibake.

    Tries the whole-string round-trip first (the common case: a comment line
    with mojibake punctuation is entirely cp1252-encodable).  Falls back to
    scanning maximal cp1252-encodable substrings when the line also contains
    characters outside cp1252 (e.g. emoji), so mojibake mixed with legitimate
    non-cp1252 text is still caught.

    Pure ASCII text always round-trips identically and is never flagged.
    ``caf\\u00e9`` (a legitimate Latin-1 accented character) fails the
    round-trip because ``\\xe9`` is not a valid UTF-8 lead byte — it is not
    flagged.  The only theoretical false positive is a cp1252 string whose
    bytes happen to form valid UTF-8 that decodes to different text (e.g.
    ``\\u00c2\\u00a1`` → ``\\u00a1``); such sequences do not occur in real
    source code.
    """
    if _try_recover(text) is not None:
        return True
    return any(_try_recover(chunk) is not None for chunk in _cp1252_chunks(text))


def recover_mojibake(text: str) -> str | None:
    """Return the corrected text if *text* is mojibake, else ``None``.

    Like :func:`is_mojibake` but returns the recovered string rather than a
    boolean.  Scans cp1252-encodable chunks when the whole line cannot be
    encoded as cp1252; returns the first recovered chunk's correction.
    """
    recovered = _try_recover(text)
    if recovered is not None:
        return recovered
    for chunk in _cp1252_chunks(text):
        recovered = _try_recover(chunk)
        if recovered is not None:
            return recovered
    return None


def find_mojibake_in_diff(diff_text: str) -> list[MojibakeFinding]:
    """Scan a unified diff for added lines containing mojibake.

    Parses *diff_text* as a unified diff (the output of ``git diff``) and
    checks every added line (lines starting with ``+``) with
    :func:`is_mojibake`.  The ``+++ b/path`` file header is consumed separately
    (it requires a trailing space) before this check, so an added line whose
    content happens to start with ``++`` is correctly treated as an added line,
    not mistaken for a header.  Returns a list of findings, one per corrupted
    added line, in diff order.  An empty list means the diff is clean.

    The diff is read as UTF-8 text (``git diff`` output is UTF-8 when the
    repository's files are UTF-8, which this repo enforces structurally).
    Binary file diffs (``Binary files a/... and b/... differ``) are skipped
    because they carry no line-level content to scan.

    Never raises — a malformed diff simply yields no findings.
    """
    findings: list[MojibakeFinding] = []
    current_path = ""
    new_line_number = 0

    for line in diff_text.splitlines():
        # Track the file being diffed.  "+++ b/path" is the new-file header;
        # "/dev/null" means the file was deleted (no added lines to scan).
        if line.startswith("+++ "):
            rest = line[4:]
            if rest == "/dev/null":
                current_path = ""
            elif rest.startswith("b/"):
                current_path = rest[2:]
            else:
                current_path = rest
            continue

        # Hunk header: @@ -old,count +new,count @@ — reset the line counter.
        if line.startswith("@@"):
            new_line_number = _parse_new_start(line)
            continue

        # Only scan added lines; skip removed lines and context lines.
        # The true "+++ b/path" file header was already consumed above (it
        # requires a trailing space), so we must NOT re-check for "+++" here:
        # an added line whose content starts with "++" (e.g. "+++foo", which
        # is "+" + content "++foo") would collide with that check and be
        # silently skipped, desyncing the line counter for the rest of the
        # hunk.  Any line starting with "+" that reaches this point is an
        # added line.
        if not line.startswith("+"):
            # Advance the new-file line counter for context and removed lines
            # so that added-line numbers stay accurate.  Removed lines ("-")
            # do not advance the new-file counter.
            if line.startswith(" ") and not line.startswith("-"):
                new_line_number += 1
            elif line.startswith("-") and not line.startswith("---"):
                pass  # removed line: new-file counter unchanged
            continue

        if not current_path:
            continue

        content = line[1:]  # strip the leading "+"
        line_no = new_line_number  # this added line's position in the new file
        new_line_number += 1  # advance for the next line

        recovered = recover_mojibake(content)
        if recovered is not None:
            findings.append(
                MojibakeFinding(
                    path=current_path,
                    line_number=line_no,
                    content=content,
                    recovered=recovered,
                )
            )

    return findings


def _parse_new_start(hunk_header: str) -> int:
    """Extract the starting line number of the new-file side from a ``@@`` header.

    A hunk header looks like ``@@ -10,5 +12,7 @@ optional context``.  The
    ``+12`` part is the starting line number in the new file.  Returns 0 if
    the header is malformed (the caller treats 0 as "unknown line number").
    """
    # Find the "+" that starts the new-file section (after the old-file section).
    # The format is: @@ -old_start,old_count +new_start,new_count @@
    plus_idx = hunk_header.find("+", 3)  # skip "@@ "
    if plus_idx == -1:
        return 0
    rest = hunk_header[plus_idx + 1 :]
    # Read digits up to "," or " " or "@".
    num_str = ""
    for ch in rest:
        if ch.isdigit():
            num_str += ch
        else:
            break
    return int(num_str) if num_str else 0
