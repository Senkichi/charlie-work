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
from charlie_work.attachment_contracts import hook_entry as attachment_hook_entry
from charlie_work.attachment_contracts.model import AdvisoryRecord
from charlie_work.attachment_contracts.review_delta import (
    BudgetSection,
    build_budget_findings,
    reconstruct_baseline_head_text,
)
from charlie_work.checks import compute_ratchetable_points
from charlie_work.janitor import _diff_content_signature, iter_diff_files


def _build_attachment_budget_section(self, diff: str, pr_number: int) -> str:
    """Build ``$attachment_budget_section`` for the review packet (#1460).

    Cheap gate first: if `.attachment-budgets.json` is absent, or this
    diff touches neither the baseline file itself nor any file that
    currently hosts a baselined attachment point, the section renders
    ``""`` -- the vast majority of PRs never approach this feature at
    all, so nothing past the gate (diff-hunk reconstruction, advisories
    read) runs for them.

    Once gated in: the base-commit baseline text is read best-effort
    (a ``TamperError``/``OSError`` degrades to "no entries", same as a
    missing file -- this section is advisory-only and must never raise);
    the PR-head text is reconstructed from the diff (or read straight off
    disk when the baseline file itself isn't part of this diff); a
    reconstruction failure sets ``head_unreadable`` rather than guessing.

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
    marker_path = self.repo_root / attachment_baseline.BASELINE_FILENAME
    if not marker_path.is_file():
        return ""

    changed_files = _diff_content_signature(diff).changed_files
    baseline_touched = attachment_baseline.BASELINE_FILENAME in changed_files

    try:
        base_document = attachment_baseline.load(marker_path)
        base_entries = attachment_baseline.entries_of(base_document)
    except (attachment_baseline.TamperError, OSError):
        base_entries = ()
    hosts_baselined = changed_files & {entry.file for entry in base_entries}

    if not (baseline_touched or hosts_baselined):
        return ""

    try:
        base_baseline_text: str | None = marker_path.read_text(encoding="utf-8")
    except OSError:
        base_baseline_text = None

    if baseline_touched:
        file_diff_lines: list[str] = []
        is_new_baseline_file = False
        for name, is_new, hunks in iter_diff_files(diff):
            if name == attachment_baseline.BASELINE_FILENAME:
                file_diff_lines = hunks
                is_new_baseline_file = is_new
                break
        head_baseline_text = reconstruct_baseline_head_text(
            None if is_new_baseline_file else base_baseline_text,
            "\n".join(file_diff_lines),
        )
    else:
        # Baseline file itself untouched by this diff: its head content
        # is its base content.
        head_baseline_text = base_baseline_text

    if baseline_touched and head_baseline_text is None:
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
            self.repo_root, diff, head_baseline_text, hosts_baselined, changed_files
        )
        section = build_budget_findings(
            base_baseline_text=base_baseline_text,
            head_baseline_text=head_baseline_text,
            changed_files=changed_files,
            baseline_touched=baseline_touched,
            advisories=advisories,
            ratchetable=ratchetable,
        )

    return render_attachment_budget_section(section)
