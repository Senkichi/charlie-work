"""Review-prompt builders moved out of ``OrchestratorApp`` (Track 2 Phase B, L03).

Bodies relocated verbatim from ``charlie_work.workflow`` (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). ``workflow_delegation._install_delegates`` re-attaches each
``def`` unwrapped onto ``OrchestratorApp``.

``_write_rework_prompt`` is a same-name thin wrapper around the free function of
the same name re-exported by ``charlie_work.workflow``. Reaching it through
``_wf._write_rework_prompt`` (rather than importing it here) is deliberate: a
direct import would bind the name to *this module's own def* and recurse. The
``_wf`` indirection also preserves every existing ``charlie_work.workflow``
monkeypatch seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.rework_prompts import _render_round_findings, _round_history_entries


def _build_prior_review_section(
    self,
    pr_dir: Path,
    prior_decision: dict[str, Any],
    new_head_sha: str | None,
) -> str:
    """Render ``$prior_review_section`` for a round-2+ review packet.

    Called when the prior decision is a terminal, non-pending verdict
    with a recorded ``reviewed_head_sha``. Issue #1270 (W13): surfaces
    EVERY archived round, not only the single most recent one -- a
    finding raised in round 1, dropped in round 2, and reintroduced in
    round 3 must be visible as such, not silently collapsed into "the
    latest decision". Round content is read exclusively from the
    ``rounds/round-K`` archive W11 built (``_round_history_entries`` in
    ``rework_prompts.py``) -- never events.db, never
    ``request_changes_count`` -- per the binding decision on #1270.
    ``prior_decision`` (the flat, latest-verdict mirror the caller
    already read) is consulted only as a fallback when the archive is
    empty -- see ``_round_history_entries``'s docstring.

    Two cases, the same distinction the pre-W13 single-round renderer
    drew, now keyed off the MOST RECENT archived round instead of the
    one ``prior_decision`` argument (which, once at least one round is
    archived, is always exactly that round's own payload -- writer and
    flat mirror are the same ``record_review`` call):

    - **Moved head** (latest round's head != live head): a genuine
      rework round. Every prior round's findings are listed, oldest
      first, followed by an interdiff (latest reviewed head -> new
      head) so the reviewer has somewhere to start, without losing
      sight of the full diff: the interdiff is "start here," never
      "only look here" -- the full diff stays attached and remains
      authoritative for findings outside it.
    - **Same head** (latest round's head == live head, issue #632
      defect 3): a PR parked on ``agent:human-needed`` whose head has
      not advanced, or an operator-corrected verdict. The diff is
      identical, so no interdiff is generated; every round's findings
      are still surfaced so a re-review can verify whether they still
      apply or whether the verdict was corrected. Without this branch
      the corrected verdict was invisible to the reviewer (the #510
      case).

    Fail-safe posture mirrors janitor.py's patch-id carry-forward
    (``_calculate_patch_id``/``_check_no_op_rework``): every I/O call
    here (``compare_diff``) already returns errors as values (``None``),
    never raises, so a failed/unavailable comparison (404, GC'd SHA,
    rebase/divergence, gh failure) just omits the interdiff and says so
    -- it never blocks packet generation.
    """
    rounds_dir = pr_dir / "rounds"
    entries = _round_history_entries(rounds_dir, fallback_decision=prior_decision)
    if not entries:
        return ""

    latest_decision = entries[-1][1]
    prior_head_sha = latest_decision.get("reviewed_head_sha")
    same_head = bool(prior_head_sha) and prior_head_sha == new_head_sha
    round_word = "round" if len(entries) == 1 else "rounds"

    lines = ["", f"## Prior review{' (same head)' if same_head else ''}", ""]
    if same_head:
        lines.append(
            f"A prior review of this head (`{prior_head_sha}`) recorded decision "
            f"**{latest_decision.get('decision') or 'unknown'}**. The diff has not "
            "changed since that review. Findings from every prior "
            f"{round_word} are listed below, oldest first -- verify whether "
            "each still applies or whether the verdict was corrected (e.g. "
            "by an operator hand-edit)."
        )
    else:
        lines.append(
            f"This PR has {len(entries)} prior review {round_word}. Findings "
            "from EVERY round are listed below, oldest first -- a finding "
            "raised in an earlier round is still in scope even if a later "
            "round's summary did not repeat it."
        )
    lines.append("")

    for round_number, round_decision in entries:
        decision_label = round_decision.get("decision") or "unknown"
        round_head = round_decision.get("reviewed_head_sha") or "unknown"
        lines.append(f"### Round {round_number} (decision: {decision_label}, head `{round_head}`)")
        lines.append("")
        findings = _render_round_findings(round_decision)
        lines.append(findings if findings else "_No findings recorded for this round._")
        lines.append("")

    # #1270 (W13): `_render_round_findings` strips
    # `_EXTERNAL_FINDINGS_POINTER` from each round's rendering -- its
    # wording is worker-brief-framed ("...none of which reach this
    # brief") and, unlike the brief (rendered once per PR), a
    # round-history section would otherwise repeat it once per round.
    # One reviewer-framed reminder for the whole section replaces it.
    lines.append(
        "The findings above come from the orchestrator's own reviewer, "
        "across every round. They are not necessarily the only ones -- "
        "a human or a peer agent may have posted verified findings as "
        "PR comments, review bodies, or inline review threads. Read the "
        "PR's review comments and review threads on GitHub before you "
        "start, and address what you find there too."
    )
    lines.append("")

    if same_head:
        lines.append(
            "No interdiff is needed -- the head is unchanged. Re-examine "
            "the full diff and confirm or overturn the prior verdict."
        )
        lines.append("")
        return "\n".join(lines)

    interdiff_text = None
    if prior_head_sha and new_head_sha:
        interdiff_text = self.gh.compare_diff(str(prior_head_sha), str(new_head_sha))
    if interdiff_text and interdiff_text.strip():
        interdiff_path = pr_dir / "interdiff.patch"
        interdiff_path.write_text(interdiff_text, encoding="utf-8")
        lines.append(
            f"Interdiff (most recent reviewed head to this head): `{interdiff_path}`. "
            "Verify the findings above are addressed there first -- but the "
            "full diff remains authoritative; findings outside the interdiff "
            "are still in scope."
        )
    else:
        lines.append(
            "Prior-head comparison was unavailable (rebase, divergence, or an "
            "API error) -- no interdiff could be generated. Review the full "
            "diff as usual, with the findings above in mind."
        )
    lines.append("")
    return "\n".join(lines)


def _write_rework_prompt(
    self, pr: dict[str, Any], issue_number: int | None, dispatch_note: str
) -> Path:
    return _wf._write_rework_prompt(
        self.paths.state_file,
        pr,
        issue_number,
        dispatch_note,
        self.config,
        repo_root=self.repo_root,
    )
