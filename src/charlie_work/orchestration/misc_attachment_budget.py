"""Attachment-budget review-section builder moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches it unwrapped onto
``OrchestratorApp`` (``self`` binds via the descriptor protocol).
"""

from __future__ import annotations

from charlie_work.attachment_budget_prompt import render_attachment_budget_section
from charlie_work.attachment_contracts import baseline as attachment_baseline
from charlie_work.attachment_contracts import baseline_dir as attachment_baseline_dir
from charlie_work.attachment_contracts import hook_entry as attachment_hook_entry
from charlie_work.attachment_contracts.model import AdvisoryRecord
from charlie_work.attachment_contracts.review_delta import (
    BudgetSection,
    build_budget_findings,
    reconstruct_baseline_dir_head,
    reconstruct_baseline_head_text,
)
from charlie_work.checks import compute_ratchetable_points
from charlie_work.janitor import _diff_content_signature, iter_diff_files


def _build_attachment_budget_section(self, diff: str, pr_number: int) -> str:
    """Build ``$attachment_budget_section`` for the review packet (#1460).

    Cheap gate first: if the checkout has no committed baseline, or this
    diff touches neither the baseline store itself nor any file that
    currently hosts a baselined attachment point, the section renders
    ``""`` -- the vast majority of PRs never approach this feature at
    all, so nothing past the gate (diff-hunk reconstruction, advisories
    read) runs for them.

    The baseline store is the per-entry ``.attachment-budgets/`` directory
    (issue #1839 -- one file per ``(kind, file, identity)`` entry, so PRs
    touching different attachment points merge cleanly), with the legacy
    ``.attachment-budgets.json`` file still honored on un-migrated
    checkouts.

    Once gated in: the base-commit baseline is read best-effort (a
    ``TamperError``/``OSError`` degrades to "no entries", same as a
    missing baseline -- this section is advisory-only and must never
    raise); the PR-head document is reconstructed from the diff (per-file
    for the directory layout, so entry adds/deletes/renames are modeled;
    or read straight off disk when the baseline itself isn't part of this
    diff); a reconstruction failure sets ``head_unreadable`` rather than
    guessing.

    Advisories are read best-effort, with "log file doesn't exist"
    (``advisory_log_exists`` False) distinguished from "log exists but
    has nothing relevant" -- only the former sets
    ``advisories_unavailable``.

    Issue #1466: the advisories source is now a two-tier fallback. The
    PR-comment channel is tried FIRST -- the worker publishes a single
    machine-readable PR comment (marker ``ADVISORY_COMMENT_MARKER`` +
    fenced JSON array of its ``AdvisoryRecord`` entries) at PR-open time
    and on subsequent pushes, because the worker's worktree-local
    ``.var/attachment-contracts/advisories.jsonl`` is generally invisible
    to the orchestrator's ``repo_root``. When a marker comment is present
    (``parse_advisories_comment`` returns a tuple, even ``()``), that
    channel wins and the local log is not consulted. Only when NO marker
    comment exists does the builder fall back to the local advisories
    file; and only when NEITHER channel is available does
    ``advisories_unavailable`` fire the "log not available" NOTE.
    """

    # Helpers are nested, not module-level: every top-level def in an
    # orchestration module is installed onto OrchestratorApp by
    # workflow_delegation._install_delegates, so a module-level helper
    # would silently grow the class's conserved member surface.
    def _head_document_for_dir_baseline(
        base_files: dict[str, str] | None,
    ) -> tuple[dict[str, object] | None, bool]:
        """Reconstruct the PR-head baseline document for a directory
        baseline: ``(document, head_unreadable)``. ``head_unreadable`` is
        True only when the diff can't be applied to the base snapshot at
        all (the caller renders a "could not evaluate" NOTE); a
        reconstructed map that merely fails schema validation degrades to
        ``None`` -- treated as an empty document downstream, same as an
        absent baseline.
        """
        if base_files is None:
            return None, True
        head_files = reconstruct_baseline_dir_head(
            base_files, diff, dir_prefix=attachment_baseline.BASELINE_DIRNAME
        )
        if head_files is None:
            return None, True
        try:
            return attachment_baseline_dir.load_files(head_files), False
        except attachment_baseline.TamperError:
            return None, False

    def _head_document_for_file_baseline(
        base_text: str | None,
    ) -> tuple[dict[str, object] | None, bool]:
        """Reconstruct the PR-head baseline document for the legacy single
        file -- same ``(document, head_unreadable)`` contract as above."""
        file_diff_lines: list[str] = []
        is_new_baseline_file = False
        for name, is_new, hunks in iter_diff_files(diff):
            if name == attachment_baseline.BASELINE_FILENAME:
                file_diff_lines = hunks
                is_new_baseline_file = is_new
                break
        head_text = reconstruct_baseline_head_text(
            None if is_new_baseline_file else base_text,
            "\n".join(file_diff_lines),
        )
        if head_text is None:
            return None, True
        try:
            return attachment_baseline.loads(head_text), False
        except attachment_baseline.TamperError:
            return None, False

    marker_path = attachment_baseline_dir.find_baseline(self.repo_root)
    if marker_path is None:
        return ""
    is_dir_baseline = marker_path.name == attachment_baseline.BASELINE_DIRNAME

    changed_files = _diff_content_signature(diff).changed_files
    if is_dir_baseline:
        dir_prefix = attachment_baseline.BASELINE_DIRNAME + "/"
        baseline_touched = any(path.startswith(dir_prefix) for path in changed_files)
    else:
        baseline_touched = attachment_baseline.BASELINE_FILENAME in changed_files

    try:
        base_document: dict[str, object] | None = attachment_baseline_dir.load(marker_path)
        base_entries = attachment_baseline.entries_of(base_document)
    except (attachment_baseline.TamperError, OSError):
        base_document = None
        base_entries = ()
    hosts_baselined = changed_files & {entry.file for entry in base_entries}

    if not (baseline_touched or hosts_baselined):
        return ""

    head_unreadable = False
    if not baseline_touched:
        # Baseline itself untouched by this diff: its head content is its
        # base content.
        head_document = base_document
    elif is_dir_baseline:
        try:
            base_files: dict[str, str] | None = attachment_baseline_dir.read_files(marker_path)
        except (attachment_baseline.TamperError, OSError):
            base_files = None
        head_document, head_unreadable = _head_document_for_dir_baseline(base_files)
    else:
        try:
            base_text: str | None = marker_path.read_text(encoding="utf-8")
        except OSError:
            base_text = None
        head_document, head_unreadable = _head_document_for_file_baseline(base_text)

    if head_unreadable:
        section = BudgetSection(
            bumps=(),
            blocking_bumps=(),
            saturated_touched=(),
            redirects_not_taken=(),
            head_unreadable=True,
            advisories_unavailable=False,
        )
    else:
        advisories: tuple[AdvisoryRecord, ...] | None
        # Issue #1466: prefer the worker-published PR-comment channel.
        # ``_read_advisories_from_pr_comment`` returns ``None`` when no
        # marker comment is present (fall back to the local log) and a
        # tuple (possibly ``()``) when one is (that channel wins, even
        # if empty -- a present marker with no records is a clean pass,
        # not an unavailable channel).
        pr_comment_advisories = self._read_advisories_from_pr_comment(pr_number)
        if pr_comment_advisories is not None:
            advisories = pr_comment_advisories
        elif attachment_hook_entry.advisory_log_exists(self.repo_root):
            advisories = attachment_hook_entry.read_advisories(self.repo_root)
        else:
            advisories = None
        # Issue #1539: compute ratchetable points (live member count
        # below baseline) for touched baselined hosts. The scan runs
        # with content_overrides for the touched host files so it
        # reflects PR-head member counts, not the base checkout's.
        # Advisory-only: any failure degrades to no ratchetable rows,
        # never raises -- mirroring the existing baseline-load
        # try/except above. Lives in ``checks.py`` (the attachment-
        # contracts tool's redirect destination) to keep
        # ``OrchestratorApp`` at its baselined member-count ceiling.
        ratchetable = compute_ratchetable_points(
            self.repo_root, diff, head_document, hosts_baselined, changed_files
        )
        section = build_budget_findings(
            base_document=base_document,
            head_document=head_document,
            changed_files=changed_files,
            baseline_touched=baseline_touched,
            advisories=advisories,
            ratchetable=ratchetable,
        )

    return render_attachment_budget_section(section)
