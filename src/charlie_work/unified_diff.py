"""Generic unified-diff section splitter.

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark (the #2220 trailer-reader extraction plus the
#2222 merge left it over its 2000-line mark). ``janitor.py`` re-imports
``iter_diff_files``, so every existing ``charlie_work.janitor.iter_diff_files``
consumer — ``checks``' deferred import, ``diff_coverage_probe``,
``orchestration/misc_attachment_budget.py``, the janitor's own gates, and
the test suite — keeps working unchanged. Nothing here imports back, so
there is no cycle.
"""

from __future__ import annotations

from collections.abc import Iterator


def iter_diff_files(diff: str) -> Iterator[tuple[str, bool, list[str]]]:
    """Split a unified diff into per-file hunk bodies.

    Yields ``(filename, is_new_file, hunk_lines)`` for each file section in
    ``diff``, where ``hunk_lines`` is every ``@@``-header and hunk-body line
    for that file (diff-metadata lines starting with ``\\`` are dropped).
    Sections with no discoverable ``+++ b/`` path are skipped. This performs
    structural splitting only — it does not tally added/removed lines or
    inspect hunk content beyond locating file/hunk boundaries; line counting
    is the caller's responsibility (see ``check_test_adequacy`` in a later
    module addition).
    """
    sections = diff.split("\ndiff --git")
    for section in sections:
        if not section.strip():
            continue
        if not section.startswith("diff --git"):
            section = "diff --git" + section

        current_file: str | None = None
        current_hunks: list[str] = []
        is_new_file = False

        for line in section.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("new file mode"):
                is_new_file = True
            elif line.startswith("@@"):
                current_hunks.append(line)
            elif current_hunks:
                if not line.startswith("\\"):
                    current_hunks.append(line)

        if current_file is not None:
            yield current_file, is_new_file, current_hunks
