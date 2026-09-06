"""Check-evaluation and merge-readiness delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
here unwrapped onto ``OrchestratorApp`` (``self`` binds via the descriptor
protocol exactly as the lexical methods did).

Workflow-defined names are reached through ``_wf.<name>`` so every existing
``charlie_work.workflow`` monkeypatch seam keeps landing: the ``CarryForwardCheck``
and ``CommandResult`` classes, and ``_calculate_patch_id`` (a helper a test patches
on the ``charlie_work.workflow`` namespace). Every other free name is imported
directly from its defining module (none is patched on ``charlie_work.workflow``).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.checks import is_infra_blocked_check, summarize_checks
from charlie_work.janitor import (
    DiffContentSignature,
    _diff_content_signature,
    is_stale_ci_verdict,
    required_check_citation_names,
)


def _check_carry_forward(self, pr_number: int, decision: dict[str, Any]) -> _wf.CarryForwardCheck:
    """Determine whether ``decision``'s verdict can carry forward to the
    PR's live head, and via which tier (issues #411/#412, #414).

    Tier 1 (fast path): the live diff's stable patch-id equals the
    recorded ``reviewed_patch_id``. Issue #1187: ``git patch-id --stable``
    strips leading whitespace from ``+``/``-`` content lines, so two
    diffs that differ ONLY in indentation depth produce the identical
    patch-id — and in an indentation-sensitive language (Python) an
    indentation-only change can alter control flow (e.g. moving a
    ``return`` into or out of an ``if`` block). A tier-1 patch-id match
    alone is therefore NOT sufficient to carry forward a verdict: when
    a tier-2 line-content signature (which preserves whitespace verbatim)
    was recorded at review time, it is also validated and must match.
    If the tier-2 signatures differ — a whitespace-only change that
    patch-id collapsed — the check fails closed to stale rather than
    carrying forward an approved verdict across a semantically
    different, unreviewed head. Decisions that predate tier-2 (no
    signature stored) have no tier-2 baseline to consult and preserve
    #412's original patch-id-only carry-forward behavior. The tier-2
    binary eligibility gate does NOT apply here: patch-id already
    proved binary content identity, and tier-2 is consulted only for
    its text-line view, which is unaffected by binary payloads.

    Tier 2 (line-content, issue #414): patch-ids differ — which happens
    on every ordinary main advance, since ``git patch-id --stable``
    hashes hunk-boundary context and the merge-base moves whenever main
    does — but the ordered ``+``/``-`` line stream and changed-file set
    recorded at review time are identical to the live diff's. Reordered,
    added, removed, or altered lines, or a changed file set, are real
    content changes and do NOT carry forward. Tier 2 is INELIGIBLE
    whenever either side's diff touched a binary file: a binary payload
    emits no ``+``/``-`` lines, so the signature is blind to it — two
    diffs with genuinely different binary content at the same path
    would otherwise compare equal (review follow-up on issue #414).

    Fails closed (``tier=None``) on any missing data, diff-fetch
    failure, or binary content: a decision recorded before tier-2
    existed (no signature stored), a PR whose diff cannot be fetched,
    a binary file on either side, or a genuine content difference all
    report "cannot carry forward" — never carry forward on
    uncertainty. Tier 2 is pure string parsing of the diff text already
    fetched for tier 1 — it needs no additional git/gh calls and so has
    no failure mode of its own beyond that shared fetch.

    Eligibility for BOTH tiers is gated on ``reviewed_patch_id`` being
    recorded at all (matching #412's original behavior exactly): a
    "blocked" verdict, or any other decision that never computed one,
    has no baseline to compare against, full stop. A pure-rename or
    mode-only diff also has an empty ``reviewed_patch_id`` (no ``@@``
    hunk) despite having a valid tier-2 signature on file — that
    specific case is intentionally left conservative (stays stale)
    rather than gating on the signature fields' presence instead, which
    was tried and reverted: ``record_review`` unconditionally records a
    (possibly trivially-empty) signature for every approved/
    request_changes decision, so gating on "signature present" instead
    of "patch-id present" made an unrelated placeholder/no-op diff look
    like a valid tier-2 baseline and wrongly carried forward verdicts
    whose head had genuinely moved to unrelated content. Tracked as a
    narrow follow-up, not fixed here.
    """
    live_diff = self.gh.pr_diff(pr_number) or ""
    if not live_diff:
        return _wf.CarryForwardCheck(None, "", DiffContentSignature((), frozenset()))

    live_patch_id = _wf._calculate_patch_id(live_diff)
    live_signature = _diff_content_signature(live_diff)

    reviewed_patch_id = decision.get("reviewed_patch_id") or ""
    if not reviewed_patch_id:
        # No baseline recorded at all (e.g. a "blocked" verdict never
        # computes a patch-id) — nothing to compare against.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    if live_patch_id and live_patch_id == reviewed_patch_id:
        # Issue #1187: ``git patch-id --stable`` strips leading
        # whitespace from ``+``/``-`` content lines, so two diffs that
        # differ only in indentation depth produce the identical
        # patch-id. In an indentation-sensitive language (Python), an
        # indentation-only change can alter control flow. A patch-id
        # match alone must not carry forward a verdict: validate the
        # tier-2 line-content signature (which preserves whitespace
        # verbatim) when one was recorded. The tier-2 binary gate does
        # NOT apply here — patch-id already proved binary content
        # identity, and tier-2 is consulted only for its text-line view.
        reviewed_changed_lines = decision.get("reviewed_changed_lines")
        reviewed_changed_files = decision.get("reviewed_changed_files")
        if reviewed_changed_lines is None or reviewed_changed_files is None:
            # Decision predates tier-2 (no signature recorded) —
            # patch-id is the only available signal; preserve #412's
            # original carry-forward behavior for legacy decisions.
            return _wf.CarryForwardCheck("patch-id", live_patch_id, live_signature)
        lines_match = tuple(reviewed_changed_lines) == live_signature.changed_lines
        files_match = frozenset(reviewed_changed_files) == live_signature.changed_files
        if lines_match and files_match:
            return _wf.CarryForwardCheck("patch-id", live_patch_id, live_signature)
        # Patch-id matched but tier-2 signatures differ — a
        # whitespace-only change that patch-id collapsed (issue #1187).
        # Fail closed to stale rather than carrying forward an approved
        # verdict across a semantically different, unreviewed head.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    reviewed_changed_lines = decision.get("reviewed_changed_lines")
    reviewed_changed_files = decision.get("reviewed_changed_files")
    if reviewed_changed_lines is None or reviewed_changed_files is None:
        # Decision predates tier-2 (no signature recorded) — cannot
        # establish content identity; fail closed to stale.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    if decision.get("reviewed_has_binary") or live_signature.has_binary:
        # A binary payload emits no +/- content lines, so the signature
        # cannot see it — never rely on its silence for content it
        # never observed (issue #414 review follow-up).
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    lines_match = tuple(reviewed_changed_lines) == live_signature.changed_lines
    files_match = frozenset(reviewed_changed_files) == live_signature.changed_files
    if lines_match and files_match:
        return _wf.CarryForwardCheck("line-content", live_patch_id, live_signature)

    return _wf.CarryForwardCheck(None, live_patch_id, live_signature)


def _enrich_checks_infra_blocked(
    self, checks: list[dict[str, Any]] | None, required: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Reclassify FAILURE required checks as ``INFRA_BLOCKED`` at the
    check-ingestion data boundary (issue #1383).

    Single point of enforcement for the infra_blocked classification: a
    required check whose FAILED job shows structural evidence of a
    non-started job (zero non-setup steps / instant-fail) or a
    config-listed billing annotation is rewritten to the
    ``INFRA_BLOCKED`` marker state before ``summarize_checks`` (and
    therefore the janitor gate / rework-routing decision) ever sees it.
    ``summarize_checks`` then routes it to ``CheckSummary.infra_blocked``
    rather than ``failed``, so it never enters the "required checks
    failed -> dispatch rework" path and never burns a rework attempt.

    Kept as an OrchestratorApp method (not a pure function in
    ``checks.py``) because the structural + annotation signals require
    two ``gh`` API calls per FAILURE check (``actions_job`` /
    ``check_run_annotations``) -- I/O that the pure
    ``summarize_checks``/``is_infra_blocked_check`` layer must not
    perform. Both API methods return safe empty values (``None`` / ``[]``)
    on any GitHub failure and never raise, so an unenrichable check
    degrades to ordinary ``failed`` routing rather than crashing
    ingestion -- mirroring the existing ``merge_ready`` enrichment this
    replaces.

    Called before the janitor gate in ``review()`` (the rework-routing
    path) and in ``merge_ready()`` (the merge-execution path) so both
    paths classify budget-failed checks identically.
    """
    if not checks or not required:
        return list(checks or [])
    cfg = self.config.auto_merge.infra_blocked
    if not cfg.enabled:
        return list(checks)
    required_set = set(required)
    enriched: list[dict[str, Any]] = []
    for check in checks:
        name = str(check.get("name") or "")
        if name in required_set and str(check.get("state") or "").upper() == "FAILURE":
            check_run_id = check.get("databaseId")
            if isinstance(check_run_id, int):
                job = self.gh.actions_job(check_run_id)
                annotations = self.gh.check_run_annotations(check_run_id)
                if job is not None and is_infra_blocked_check(job, annotations, cfg):
                    check = {**check, "state": "INFRA_BLOCKED"}
        enriched.append(check)
    return enriched


def _is_stale_ci_request_changes(self, pr_number: int, decision: dict[str, Any]) -> bool:
    """True when ``decision`` is a request_changes verdict whose only
    findings cite required checks that are all green on the live head
    (issue #1111 staleness predicate, network half).

    The pure text-shape check (``required_check_citation_names``) runs
    first so ``gh pr checks`` is only fetched for the small set of
    verdicts that could possibly be stale. Checks-unavailable (``None``)
    fails closed to False — the verdict keeps its normal lifecycle.
    """
    required = self.config.auto_merge.required_checks
    if not required:
        return False
    if required_check_citation_names(decision, required) is None:
        return False
    checks = self.gh.pr_checks(pr_number)
    if checks is None:
        return False
    return is_stale_ci_verdict(decision, summarize_checks(checks, required))


def _merge_not_ready_result(
    self,
    pr_number: int,
    issue_number: int | None,
    decision: dict[str, Any],
    existing_pr_state: dict[str, Any],
) -> _wf.CommandResult:
    """Return a non-mergeable result for an approved PR that is not the train head."""
    return _wf.CommandResult(
        True,
        f"PR #{pr_number} is not the head of the merge-train queue",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": asdict(summarize_checks([], self.config.auto_merge.required_checks)),
            "checks_unavailable": False,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": [],
            "consecutive_failed_merge_attempts": existing_pr_state.get(
                "consecutive_failed_merge_attempts", 0
            ),
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": False,
        },
    )
