"""Issue-citation drift-check delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each ``def`` unwrapped
onto ``OrchestratorApp``.

The workflow-defined ``ORCHESTRATOR_COMMENT_MARKER`` constant is reached through
``_wf.`` to preserve the ``charlie_work.workflow`` module namespace. The intra-module
call chain ``_check_issue_citations`` -> ``_post_citation_drift_comment`` ->
``_read_head_sha_best_effort`` stays ``self.`` calls (all three delegates live here).

Behavior delta: ``logging.getLogger(__name__)`` in ``_check_issue_citations`` and
``_post_citation_drift_comment`` now names the logger
``charlie_work.orchestration.misc_citation`` instead of ``charlie_work.workflow``
(``__name__`` is a module global that moves with the body). No test asserts on
either logger name.
"""

from __future__ import annotations

import logging
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.citation_check import (
    CitationVerdict,
    drift_fingerprint as citation_drift_fingerprint,
    drifted_verdicts as drifted_citation_verdicts,
    format_verdict_status_cell,
    verify_citations,
)
from charlie_work.subprocess_runner import run_captured


def _check_issue_citations(
    self,
    issue_number: int,
    full_issue: dict[str, Any],
    previous_entries: dict[int, dict[str, Any]],
) -> dict[int, tuple[str, list[CitationVerdict]]]:
    """Verify ``path:line`` citations in ``full_issue``'s body (issue #1000).

    Returns ``{issue_number: (fingerprint, drifted_verdicts)}`` when the
    drift fingerprint changed since the last pass (including a change to
    empty when drift resolved), and ``{}`` otherwise -- the caller only
    stamps/emit-vents on a change, so an unchanged-stale issue is not
    re-commented every pass. When drift is newly detected a flag comment is
    posted on the issue (best-effort; a failure is logged and swallowed so
    it can never abort dispatch). The comment is visible to the dispatched
    worker through ``$issue_comments`` (issue #872), which is the point: a
    worker that sees "these line numbers have drifted, grep for the symbol"
    is not silently misled onto unrelated code.

    This is the "cheap check" backstop. It catches coordinate drift
    (file missing / out of range / blank line); it does not catch in-range
    content drift, which the filing convention (``CONTRIBUTING.md``) is the
    durable fix for. See ``citation_check`` for the full rationale.
    """
    body = str(full_issue.get("body") or "")
    if not body.strip():
        return {}
    try:
        verdicts = verify_citations(body, self.repo_root)
    except Exception:
        logging.getLogger(__name__).warning(
            "citation_check failed issue=%d", issue_number, exc_info=True
        )
        return {}
    drifted = drifted_citation_verdicts(verdicts)
    new_fp = citation_drift_fingerprint(verdicts)
    prev_entry = previous_entries.get(issue_number, {})
    prev_fp = prev_entry.get("citation_drift_fingerprint")
    # ``None`` (never stamped) and "" (no drift) both mean "no current drift
    # marker" for the comparison; treat them as equal so the first pass on a
    # clean issue does not stamp an empty fingerprint as a "change".
    if (prev_fp or "") == new_fp:
        return {}
    if new_fp:
        self._post_citation_drift_comment(issue_number, drifted)
    return {issue_number: (new_fp, drifted)}


def _post_citation_drift_comment(self, issue_number: int, drifted: list[CitationVerdict]) -> None:
    """Post a best-effort flag comment listing drifted citations (issue #1000).

    Never raises: a comment-posting failure is logged and swallowed so it
    cannot abort the dispatch pass. The comment is stamped with the current
    ``HEAD`` sha (the commit the check ran against) per the issue's
    recommendation 2, so a reader can tell a drifted citation from one that
    was always wrong.
    """
    head_sha = self._read_head_sha_best_effort()
    lines = [
        f"{_wf.ORCHESTRATOR_COMMENT_MARKER}",
        "",
        "## Citation drift detected (issue #1000)",
        "",
        "One or more ``path:line`` citations in this issue no longer match the"
        " working tree. A worker reading the bare line number would land on"
        " unrelated code and may infer the defect was already fixed. **Grep for"
        " the cited symbol or surrounding context instead of trusting the line"
        " number**, and consider re-filing with a symbol citation (see"
        ' ``CONTRIBUTING.md`` -- "Citing code in issues").',
        "",
    ]
    if head_sha:
        lines.append(f"Verified against `HEAD` = `{head_sha}`.")
        lines.append("")
    lines.append("| citation | status |")
    lines.append("|---|---|")
    for v in drifted:
        lines.append(f"| `{v.citation.raw}` | {format_verdict_status_cell(v)} |")
    body = "\n".join(lines)
    issue_dir = self.paths.issues / f"issue-{issue_number}"
    try:
        issue_dir.mkdir(parents=True, exist_ok=True)
        body_path = issue_dir / "citation-drift-comment.md"
        body_path.write_text(body, encoding="utf-8")
        self.gh.issue_comment(issue_number, body_path)
    except Exception:
        logging.getLogger(__name__).warning(
            "citation drift comment post failed issue=%d", issue_number, exc_info=True
        )


def _read_head_sha_best_effort(self) -> str | None:
    """Return the current ``HEAD`` sha of ``repo_root``, or ``None``."""
    try:
        res = run_captured(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            timeout_seconds=15,
        )
        if not res.ok:
            return None
        return res.stdout.strip() or None
    except Exception:
        return None
