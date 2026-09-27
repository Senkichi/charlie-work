"""Review-packet delta computation for `.attachment-budgets/` (issue #1460).

Pure text/data functions only -- no I/O, no subprocess, no AST, no line-count
arithmetic of any kind (this package's binding operator constraint: see
``model.py``). ``workflow.py``'s ``review()`` is the sole caller and owns all
I/O (reading the diff, reading the baseline files, reading advisories);
everything here is given already-fetched text and returns structured data for
``render_attachment_budget_section`` to format.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from charlie_work.attachment_contracts.baseline import (
    bump_ack_is_external,
    entries_of,
    new_bumps,
)
from charlie_work.attachment_contracts.model import (
    AdvisoryRecord,
    AttachmentPoint,
    BaselineEntry,
    Bump,
    Kind,
)

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def reconstruct_baseline_head_text(base_text: str | None, file_diff: str) -> str | None:
    """Apply ``file_diff``'s unified-diff hunks to ``base_text``, returning the
    resulting PR-head text of one baseline-store file (the legacy
    ``.attachment-budgets.json``, or one entry file / ``meta.json`` under
    ``.attachment-budgets/`` -- ``reconstruct_baseline_dir_head`` drives this
    per member).

    ``file_diff`` is the hunk-header-and-body text for a single file (an
    ``@@ ... @@`` header followed by ` `/`+`/`-`-prefixed body lines, as
    produced by ``janitor.iter_diff_files``'s ``hunk_lines`` -- may contain
    more than one hunk).

    ``base_text is None`` means the file is newly added in this PR: the
    reconstructed head text is simply the diff's added content (every ``+``
    line), since there is no base to apply hunks against.

    Returns ``None`` on ANY hunk/context mismatch (a context line in the diff
    that doesn't match ``base_text`` at the expected offset, an unparseable
    hunk header, or any other structural inconsistency) -- the caller renders
    a "could not evaluate" NOTE rather than trusting a corrupted
    reconstruction.
    """
    diff_lines = file_diff.splitlines()

    if base_text is None:
        # Newly-added file: no base to apply hunks to. Every `+` line in the
        # diff (skipping the `+++`/hunk-header lines) is the head content.
        added = [
            line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")
        ]
        return "\n".join(added) + ("\n" if added else "")

    base_lines = base_text.splitlines()
    result: list[str] = []
    base_index = 0  # 0-based cursor into base_lines
    hunk_index = 0
    n = len(diff_lines)

    while hunk_index < n:
        line = diff_lines[hunk_index]
        match = _HUNK_HEADER.match(line)
        if match is None:
            hunk_index += 1
            continue
        old_start = int(match.group(1))
        # Copy any untouched base lines before this hunk starts.
        target_index = old_start - 1
        if target_index < base_index or target_index > len(base_lines):
            return None
        result.extend(base_lines[base_index:target_index])
        base_index = target_index

        hunk_index += 1
        while hunk_index < n and not _HUNK_HEADER.match(diff_lines[hunk_index]):
            body_line = diff_lines[hunk_index]
            if body_line.startswith(" "):
                if base_index >= len(base_lines) or base_lines[base_index] != body_line[1:]:
                    return None
                result.append(base_lines[base_index])
                base_index += 1
            elif body_line.startswith("-"):
                if base_index >= len(base_lines) or base_lines[base_index] != body_line[1:]:
                    return None
                base_index += 1
            elif body_line.startswith("+"):
                result.append(body_line[1:])
            elif body_line == "" or body_line.startswith("\\"):
                pass  # blank separator / "no newline at end of file" marker
            else:
                return None
            hunk_index += 1

    result.extend(base_lines[base_index:])
    return "\n".join(result) + "\n"


def _split_diff_sections(diff_text: str) -> list[list[str]]:
    """Split a unified diff into per-file sections (each starting at its
    ``diff --git`` line). Unlike ``janitor.iter_diff_files`` this keeps the
    file-level headers (``new file mode`` / ``deleted file mode`` /
    ``rename`` lines) that a per-entry baseline directory needs to model
    adds, deletes, and renames -- not just content hunks."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git ") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return sections


def _section_path(line_value: str) -> str | None:
    """Decode a ``--- a/<path>`` / ``+++ b/<path>`` operand."""
    if line_value == "/dev/null":
        return None
    if line_value.startswith(("a/", "b/")):
        return line_value[2:]
    return line_value


def reconstruct_baseline_dir_head(
    base_files: Mapping[str, str] | None,
    diff_text: str,
    *,
    dir_prefix: str,
) -> dict[str, str] | None:
    """Apply a diff's hunks to the per-file map of a baseline directory.

    ``base_files`` maps paths relative to the baseline directory to base
    content (``baseline_dir.read_files`` output); ``dir_prefix`` is the
    directory's repo-relative name (``".attachment-budgets"``). Returns the
    head-side map in the same shape -- every file the directory holds after
    the diff -- or ``None`` on ANY structural inconsistency (a hunk whose
    context doesn't match base content, a section touching a file absent
    from the base snapshot, an unparseable header): the caller renders a
    "could not evaluate" NOTE rather than trusting a corrupted
    reconstruction, same contract as ``reconstruct_baseline_head_text``.

    Handles the file-level events a directory can produce that a single
    file never could: new entry files (``new file mode``), deleted ones
    (``deleted file mode``), and renames (``rename from``/``rename to`` --
    e.g. a host file move that re-keys its entry). Sections entirely
    outside ``dir_prefix`` are skipped.
    """
    if base_files is None:
        return None
    files = dict(base_files)
    prefix = dir_prefix.rstrip("/") + "/"

    for section in _split_diff_sections(diff_text):
        old_path: str | None = None
        new_path: str | None = None
        is_new = is_deleted = False
        saw_destination = False  # a "+++ b/..." or "rename to" line
        hunk_start: int | None = None
        for i, line in enumerate(section):
            if line.startswith("new file mode"):
                is_new = True
            elif line.startswith("deleted file mode"):
                is_deleted = True
            elif line.startswith("rename from "):
                old_path = line[len("rename from ") :]
            elif line.startswith("rename to "):
                saw_destination = True
                new_path = line[len("rename to ") :]
            elif line.startswith("--- "):
                old_path = _section_path(line[4:])
            elif line.startswith("+++ "):
                saw_destination = True
                new_path = _section_path(line[4:])
            elif line.startswith("@@"):
                hunk_start = i
                break
        hunks = "\n".join(section[hunk_start:]) if hunk_start is not None else ""

        old_rel = (
            old_path[len(prefix) :]
            if old_path is not None and old_path.startswith(prefix)
            else None
        )
        new_rel = (
            new_path[len(prefix) :]
            if new_path is not None and new_path.startswith(prefix)
            else None
        )
        if old_rel is None and new_rel is None:
            continue  # section touches no baseline-directory member
        if is_deleted:
            if old_rel is not None:
                files.pop(old_rel, None)
            continue
        if not saw_destination:
            # A mode-only section (no ---/+++ or rename lines): content is
            # unchanged, so the member carries over untouched.
            continue
        if new_rel is None:
            # Moved out of the baseline directory: the member leaves the
            # head map.
            if old_rel is not None:
                files.pop(old_rel, None)
            continue
        if is_new:
            base_text: str | None = None
        else:
            source_rel = old_rel if old_rel is not None else new_rel
            base_text = files.get(source_rel)
            if base_text is None:
                # A modified/renamed member absent from the base snapshot --
                # the map is stale or the diff mismatched; fail closed.
                return None
            if old_rel is not None and old_rel != new_rel:
                files.pop(old_rel)
        head_text = reconstruct_baseline_head_text(base_text, hunks)
        if head_text is None:
            return None
        files[new_rel] = head_text
    return files


@dataclass(frozen=True)
class RatchetablePoint:
    """A baselined attachment point whose live member count is strictly below
    its frozen baseline (issue #1539).

    The review packet renders a ratchet-remedy row for each such point,
    instructing the worker to run ``baseline --ratchet`` and commit the
    resulting ``.attachment-budgets/`` tightening in the same PR. A
    lowered count is a ratchet, not a bump -- G4 (workers may not self-ack
    bumps) governs raises only; CI re-verifies ``actual <= baseline``
    deterministically from the scan, so there is nothing for a worker to
    launder by self-committing a decrease.
    """

    kind: Kind
    identity: str
    file: str
    baseline_members: int
    live_count: int


@dataclass(frozen=True)
class BudgetSection:
    """Structured findings for the ``$attachment_budget_section`` packet block."""

    bumps: tuple[tuple[BaselineEntry, Bump], ...]
    blocking_bumps: tuple[tuple[BaselineEntry, Bump], ...]
    saturated_touched: tuple[BaselineEntry, ...]
    redirects_not_taken: tuple[AdvisoryRecord, ...]
    ratchetable: tuple[RatchetablePoint, ...] = ()
    head_unreadable: bool = False
    advisories_unavailable: bool = False


def build_budget_findings(
    *,
    base_document: dict[str, object] | None,
    head_document: dict[str, object] | None,
    changed_files: frozenset[str],
    baseline_touched: bool,
    advisories: tuple[AdvisoryRecord, ...] | None = None,
    ratchetable: tuple[RatchetablePoint, ...] = (),
) -> BudgetSection:
    """Build the structured findings for the review packet's budget section.

    ``base_document``/``head_document`` are the base-commit and PR-head
    baseline documents (already loaded -- ``baseline_dir.load``/``load_files``
    for the per-entry directory layout, ``baseline.loads`` for the legacy
    single file), or ``None`` when that side's baseline doesn't exist or
    failed to parse -- treated the same as an empty document. The caller is
    responsible for setting ``head_unreadable`` when the head baseline
    could not even be reconstructed (that's a NOTE, not silently
    swallowed).

    ``advisories is None`` means the advisories log was unavailable for this
    PR -- ``advisories_unavailable`` is set and redirects-not-taken is left
    empty (it cannot be computed without the log).

    ``ratchetable`` (issue #1539) is the pre-computed tuple of baselined
    points whose live member count is strictly below their frozen baseline.
    The caller (``_build_attachment_budget_section`` in ``workflow.py``)
    computes this from a PR-head scan -- this function stays pure (no AST,
    no scan) and just passes the tuple through to the ``BudgetSection``.
    """

    def _document_or_empty(document: dict[str, object] | None) -> dict[str, object]:
        return document if document is not None else {"version": 1, "entries": []}

    base_document = _document_or_empty(base_document)
    head_document = _document_or_empty(head_document)

    bumps = new_bumps(base_document, head_document)
    blocking_bumps = tuple(
        (entry, bump)
        for entry, bump in bumps
        if bump.actor == "worker" and not bump_ack_is_external(bump)
    )

    # G4-valid new bump identities suppress the corresponding
    # saturated-touched row for that identity: a legitimately-acked bump
    # already explains the growth, so flagging the same host file again as
    # "verify no new members were bound" would be redundant noise.
    g4_valid_bumped_identities = {
        entry.identity
        for entry, bump in bumps
        if not (bump.actor == "worker" and not bump_ack_is_external(bump))
    }

    saturated_touched = tuple(
        entry
        for entry in sorted(entries_of(head_document), key=lambda e: (e.kind, e.file, e.identity))
        if entry.file in changed_files and entry.identity not in g4_valid_bumped_identities
    )

    if advisories is None:
        redirects_not_taken: tuple[AdvisoryRecord, ...] = ()
        advisories_unavailable = True
    else:
        redirects_not_taken = tuple(
            record
            for record in advisories
            if record.redirect and record.redirect not in changed_files
        )
        advisories_unavailable = False

    return BudgetSection(
        bumps=bumps,
        blocking_bumps=blocking_bumps,
        saturated_touched=saturated_touched,
        redirects_not_taken=redirects_not_taken,
        ratchetable=ratchetable,
        head_unreadable=False,
        advisories_unavailable=advisories_unavailable,
    )


def compute_ratchetable(
    scan_points: tuple[AttachmentPoint, ...],
    head_document: dict[str, object],
    changed_files: frozenset[str],
) -> tuple[RatchetablePoint, ...]:
    """Compute the ratchetable-point tuple for the review packet (issue #1539).

    Pure: takes already-scanned ``AttachmentPoint`` objects (the caller in
    ``workflow.py`` runs the scan with PR-head ``content_overrides`` for
    touched baselined hosts) and the head baseline document, returns every
    baselined point whose live member count is strictly below its frozen
    baseline AND whose host file is in ``changed_files`` -- a shrink that
    happened in this PR, not a stale un-ratcheted point from a prior PR.

    Member counts only -- no line-count arithmetic of any kind (this
    package's binding operator constraint).
    """
    baseline_by_key: dict[tuple[str, str, str], BaselineEntry] = {
        (e.kind, e.file, e.identity): e for e in entries_of(head_document)
    }
    result: list[RatchetablePoint] = []
    for point in scan_points:
        entry = baseline_by_key.get((point.kind, point.file, point.identity))
        if entry is None:
            continue
        if point.file not in changed_files:
            continue
        if point.member_count < entry.member_count:
            result.append(
                RatchetablePoint(
                    kind=point.kind,
                    identity=point.identity,
                    file=point.file,
                    baseline_members=entry.member_count,
                    live_count=point.member_count,
                )
            )
    return tuple(sorted(result, key=lambda r: (r.kind, r.file, r.identity)))
