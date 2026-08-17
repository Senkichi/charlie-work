"""Rework-prompt-rendering free-function family (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: the free-function family that
renders and writes a PR's rework brief (``rework-prompt.md``) -- the
required-changes section's tiered fallback rendering, the review-decision
JSON reader and freshness comparison used to detect a stale brief, the
per-round retry/distinct-verdict archive numbering, and the pure-render /
atomic-write split that lets ``dispatch_rework`` detect renderer drift
without rewriting every brief it inspects. ``workflow.py`` re-exports every
symbol here via a facade import block (mirroring ``config.py``'s
``RunnerAllocationConfig`` re-export pattern and this repo's own
``dispatch_selection.py``/``escalation.py``/``verdict_parsing.py``
precedents), so existing import paths and monkeypatch targets keep working
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import OrchestratorConfig
from .cross_family import LEGACY_VACUOUS_SUMMARY
from .github import defang_closing_keywords
from .markdown_fence import fenced_block
from .prompts import (
    assert_containment,
    assert_execution_contract,
    assert_no_merge_contract,
    render_prompt,
)
from .verdict_parsing import body_has_crash_signature


def _rework_prompt_search_dirs(
    config: OrchestratorConfig, repo_root: Path | None = None
) -> tuple[Path, ...]:
    """Resolve the optional repo-local prompt override directory."""
    prompts_dir = config.runtime.prompts_dir
    if not prompts_dir:
        return ()
    path = Path(prompts_dir)
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    return (path,)


_EXTERNAL_FINDINGS_POINTER = (
    "## Also required: findings posted on the PR itself\n"
    "\n"
    "The findings above come from the orchestrator's own reviewer. They "
    "are not necessarily the only ones. A human or a peer agent may have "
    "posted verified findings as PR comments, review bodies, or inline "
    "review threads — none of which reach this brief. **Read the PR's "
    "review comments and review threads on GitHub before you start**, "
    "and address what you find there too.\n"
)


def _finish_required_changes_section(lines: list[str]) -> str:
    """Join a rendered findings section and append the external-findings pointer.

    Single point of enforcement for issue #950. Before this, the
    instruction to go read the PR's own review comments existed in exactly
    two places — the ``findings_channel == "vacuous"`` tier and the
    both-empty tier — and both fire *only when the internal channel came
    back empty*. The three tiers that render substantive content carried no
    pointer at all, so the sole reference to externally-posted findings was
    conditioned on the absence of internal ones: the better the
    orchestrator's own review, the more certainly a human's or a peer
    agent's comment was discarded. Observed on PR #948, where two verified
    findings posted as PR comments were invisible to the rework worker
    precisely because the internal verdict was substantive.

    The pointer is deliberately *not* a render-time fetch of the comments
    themselves. ``_render_rework_prompt`` is a pure function of the verdict
    file, the dispatch note, and the template set, and ``dispatch_rework``
    depends on that purity — it re-renders and diffs against the bytes on
    disk to detect renderer drift (#800). Live comment traffic as a render
    input would make every new comment read as drift, trading a silent drop
    for a permanently noisy detector. Telling the worker to fetch them
    itself costs no purity and no API call, and reads the freshest state.
    Folding comments into the verdict at ``record_review`` time is the
    separate half of #950.
    """
    return "\n".join([*lines, _EXTERNAL_FINDINGS_POINTER])


_EXTERNAL_FINDINGS_SECTION_INTRO = (
    "## Findings posted on the PR itself\n"
    "\n"
    "These are verified findings a human or peer agent posted on the PR as "
    "comments, review bodies, or inline review threads -- separate from the "
    "reviewer's findings above. Address each of them too.\n"
)


def _render_external_findings_section(
    reviewer_lines: list[str], external_findings: list[str]
) -> str:
    """Join the reviewer section and the external-findings section (issue #999).

    External findings render under their own heading, after the reviewer's
    section, as bullets -- each defanged so a closing keyword in a human
    comment cannot auto-close the linked issue from the worker's brief.

    The ``_EXTERNAL_FINDINGS_POINTER`` is deliberately *not* appended here.
    Its body states "none of which reach this brief", which this section
    makes false: the ingested findings are now rendered inline. Old-shape
    records (no ``external_findings`` field) never reach this function and
    keep the pointer exactly as before this fix.
    """
    lines = [*reviewer_lines, _EXTERNAL_FINDINGS_SECTION_INTRO]
    lines.extend(f"- {defang_closing_keywords(item)}" for item in external_findings)
    lines.append("")
    return "\n".join(lines)


def _render_required_changes_section(decision: dict[str, Any] | None) -> str:
    """Render the ``$required_changes_section`` for a rework brief.

    The findings are read from ``review-decision.json``. ``required_changes``
    -- the most actionable output the review pipeline produces -- is the
    primary source. Giving it its own rendered section (instead of
    multiplexing one prose slot) means an operational dispatch note can no
    longer displace it.

    Measured across the on-disk corpus, ``request_changes`` verdicts with a
    populated ``required_changes`` are the exception (0 of 20 observed):
    ``prompts/review.md`` historically documented the field as optional, so
    reviewers reliably fill in ``summary`` and skip ``required_changes``.
    That ``summary`` is real, substantive review content -- discarding it
    because the structured list is empty silently sends a worker a brief
    with nothing to act on. So for a ``request_changes`` verdict this
    function degrades through three tiers: (1) the enumerated
    ``required_changes`` list when non-empty, (2) the verdict's ``summary``
    rendered verbatim and clearly labelled as prose when the list is empty,
    (3) an explicit, loud "findings unavailable, check the PR on GitHub"
    instruction when BOTH are empty. Tier 3 is a hard invariant: this
    function must never render (or omit into silence) something a worker
    could read as "there is nothing to change" when a decision actively
    requires rework.

    Issue #792: ``record_review`` now derives ``required_changes`` from
    ``summary`` itself, at write time, and records which of two outcomes
    happened via ``decision["findings_channel"]``: ``"derived"`` (the
    summary was used as the findings, i.e. what used to be this function's
    own tier-2 fallback) or ``"vacuous"`` (nothing was derivable -- neither
    an itemized list nor a usable summary -- e.g. the historical
    cross-family placeholder, ``cross_family.LEGACY_VACUOUS_SUMMARY``, which
    is genuinely non-blank text but carries no reviewer content).

    Issue #950: when the PR itself carries verified human or peer-agent
    findings (issue comments, review bodies, inline review threads),
    ``record_review`` ingests them at write time.

    Issue #999: external findings no longer merge into ``required_changes``.
    They ride in their own ``external_findings`` field, and
    ``findings_channel`` continues to describe *only* the reviewer's list --
    so ``"derived"`` is never overwritten and keeps its tier-2 verbatim
    rendering even when external findings are present. This function renders
    them under their own heading (``_render_external_findings_section``)
    after the reviewer's section, instead of appending the
    ``_EXTERNAL_FINDINGS_POINTER`` (whose "none of which reach this brief"
    body the inline section makes false).

    Two shapes coexist for backward compatibility:

    * **New shape** (records written after #999, non-vacuous case): the
      ``external_findings`` field is present and ``findings_channel`` is
      ``None`` or ``"derived"`` (never ``"external"``). The reviewer's
      section renders per its marker, then the external section is
      appended.
    * **Old shape** (records written before #999, *and* the vacuous-replace
      case): external findings are already merged into
      ``required_changes`` with ``findings_channel == "external"`` and no
      ``external_findings`` field. These render exactly as before this fix
      -- the itemized tier with the external-aware intro and the pointer --
      so no content is lost from any verdict already on disk.

    The ``"vacuous"`` case is the one that must NOT become a separate
    section: a content-free reviewer summary has nothing worth rendering
    above the external items, so ``record_review`` still *replaces*
    ``required_changes`` with the external findings (flipping the channel
    to ``"external"``) and writes no ``external_findings`` field -- the
    old-shape ``"external"`` path handles it unchanged.

    Verdicts carrying the ``"vacuous"`` or ``"derived"`` markers are
    handled explicitly, before the shape-based tiers below:
    ``"vacuous"`` always renders tier 3 (a non-blank-but-content-free
    summary is strictly worse than an empty one -- rendering it as tier 2
    would silently present it as real findings), and ``"derived"`` always
    renders tier 2 verbatim rather than falling into tier 1's bullet list
    (a single derived item wrapped as a one-item bullet would otherwise dump
    an entire multi-paragraph summary onto one line). Verdicts with no
    marker at all -- every record written before #792 -- fall through
    unchanged to the original shape-based tiers.

    Rendered for ``request_changes`` and, defensively, ``blocked`` verdicts
    (routing to rework via the decision-agnostic janitor gates -- merge
    conflict / no-op-rework repair -- can carry forward whatever verdict was
    last on disk, including ``blocked``). ``approved`` never renders
    anything here; its findings (if any) reach the reviewer via
    ``$prior_review_section`` instead, not the worker's rework brief.

    For ``blocked`` specifically, the enumerated list and the summary
    fallback (tiers 1-2) are intentionally suppressed even when populated.
    ``prompts/review.md`` requires ``required_changes`` for ``blocked`` just
    as it does for ``request_changes``, but "what must change before this PR
    can be approved" language is the wrong framing for the routes that reach
    this decision-agnostic branch (merge-conflict / no-CI / cross-pr-revert
    routes, which explicitly tell the worker "do not re-litigate the
    review") -- an ``approved`` verdict can also legitimately carry a
    non-empty ``required_changes`` left over from an earlier round, which is
    the same contradiction. Tier 3 (the both-empty escape hatch) still
    applies to ``blocked`` -- suppressing content is fine, but suppressing
    it AND leaving the worker with no signal that something was withheld is
    not.

    Returns an empty string only when there is no decision, or the decision
    is not ``request_changes``/``blocked``, or it is ``blocked`` with some
    (but not zero) findings content -- every other case renders something.

    Reviewer-authored text (tiers 1-2) passes through
    ``defang_closing_keywords`` before rendering. A worker reads this brief
    and writes its own PR body/commit message from it -- text charlie-work
    does not control downstream and cannot re-check with
    ``linked_issue_number``'s hijack-safety guard. If reviewer prose contains
    a live closing keyword (``Fixes #649``), rewriting it to ``Fixes issue
    649`` keeps the issue number legible while removing the syntax that
    would trigger GitHub auto-close or a false label-transition binding.

    Issue #1269 (W12): both ``external_findings`` and (old-shape,
    ``findings_channel == "external"``) ``required_changes`` are filtered
    for ``verdict_parsing.body_has_crash_signature`` before rendering. This
    is the retroactive half of the crash-comment-noise fix -- the collector
    (``workflow._collect_external_findings``) only stops *new* ingestion, so
    this render path is what reaches records already persisted before that
    fix shipped (or before the provenance-marker stamp it also relies on,
    #1242, existed at all). This function is the single point every render
    path bottlenecks through (``_render_rework_prompt`` calls it once, and
    ``dispatch_rework``'s #800 drift reconciler re-renders and diffs against
    it every pass), so a guard here reaches every persisted record without
    needing the one-off backfill script to run against it.
    """
    if not isinstance(decision, dict):
        return ""
    verdict = decision.get("decision")
    if verdict not in ("request_changes", "blocked"):
        return ""

    findings_channel = decision.get("findings_channel")

    raw_required_changes = decision.get("required_changes")
    pre_filter_changes = (
        [str(item).strip() for item in raw_required_changes if str(item).strip()]
        if isinstance(raw_required_changes, list)
        else []
    )
    # Issue #1269 (W12): old-shape records merge external findings directly
    # into `required_changes` (see the "Old shape" section above), so a
    # reviewer-session-crash summary posted before the collector-side fix in
    # `workflow._collect_external_findings` -- or before the provenance-marker
    # stamping fix that predates it, #1242 -- can be sitting in this list on
    # an already-persisted record today. Filtered here too, defense-in-depth
    # for a reopened old-shape PR: the collector-side fix only prevents
    # *new* ingestion and cannot retroactively clean a record already
    # written before it shipped, and this render path is the single place
    # every one of them is read back.
    changes = (
        [item for item in pre_filter_changes if not body_has_crash_signature(item)]
        if findings_channel == "external"
        else pre_filter_changes
    )
    # LOAD-BEARING: body_has_crash_signature has no fallback for a
    # marker-STAMPED crash body (the ORCHESTRATOR_COMMENT_MARKER line
    # precedes the "## Reviewer session..." heading, and the predicate's
    # `lstrip()` does not remove a full preceding line, only leading
    # whitespace) -- that is safe here ONLY because a stamped body never
    # reaches this render path at all: the collector's
    # `_is_orchestrator_comment` marker filter drops every stamped body
    # before it is ever ingested into `required_changes`/`external_findings`,
    # so persisted records only ever contain pre-stamp body text, which is
    # exactly what this prefix match is written against. If that collector
    # filter is ever weakened or removed, this predicate would need a
    # marker-aware fallback too.
    #
    # Issue #999: new-shape records carry external findings in their own
    # field. Old-shape records (pre-#999, or the vacuous-replace case) have
    # no such field and render exactly as before this fix.
    #
    # Issue #1269 (W12): filtered for the same crash-signature content
    # unconditionally (not gated on `findings_channel`) -- a new-shape record
    # can carry a crash comment too, since the collector-side fix only stops
    # *future* ingestion; jc#1386 and jc#1394 both had crash comments already
    # persisted in this field from before it shipped. Computed here, ABOVE
    # the vacuous-old-shape guard below, so that guard can check it directly
    # rather than assuming (unasserted) that an "external"-channel record
    # never co-persists a populated `external_findings` alongside old-shape
    # `required_changes`.
    raw_external = decision.get("external_findings")
    external_findings = (
        [
            stripped
            for item in raw_external
            if (stripped := str(item).strip()) and not body_has_crash_signature(stripped)
        ]
        if isinstance(raw_external, list)
        else []
    )
    new_shape = bool(external_findings)

    raw_summary = decision.get("summary")
    summary_text = raw_summary.strip() if isinstance(raw_summary, str) else ""
    if (
        findings_channel == "external"
        and pre_filter_changes
        and not changes
        and not external_findings
        and (not summary_text or summary_text == LEGACY_VACUOUS_SUMMARY)
    ):
        # The filter above can legitimately empty `changes` entirely: the
        # vacuous-replace old-shape population (`record_review`'s
        # `findings_channel == "vacuous"` branch in workflow.py) unconditionally
        # replaces `required_changes` with `external_findings` whenever the
        # reviewer's own summary was vacuous, so an all-crash external-findings
        # set leaves nothing else behind. `summary_text` in that population is
        # exactly the placeholder `record_review` discarded to reach this path
        # (`cross_family.LEGACY_VACUOUS_SUMMARY`, or blank -- `record_review`
        # rejects a truly empty summary outright for `request_changes`/
        # `blocked`, so blank can only reach here on an older, hand-edited, or
        # differently-produced record) -- never real reviewer prose, since a
        # genuine non-vacuous summary is never routed through the
        # vacuous-replace branch (see `_summary_is_vacuous`). Treating it as
        # absent here routes to tier 3 below instead of rendering that
        # placeholder as if it were real findings -- exactly the failure the
        # `"vacuous"` marker branch below already guards against, just reached
        # through the old-shape `"external"` path instead of a fresh
        # `"vacuous"` marker. A pre-#999 record where the reviewer's OWN
        # summary was genuine but `required_changes` held only external items
        # is a different, legitimate population and is not swept in here: the
        # guard above only fires on the two known placeholder shapes, so that
        # case falls through unchanged to the summary_text fallback tier below.
        #
        # `and not external_findings` (added on review) makes this branch's
        # cross-field assumption explicit rather than implicit: every writer
        # today only ever populates `external_findings` on new-shape records,
        # never alongside an "external"-channel old-shape `required_changes`,
        # so this condition is currently always true when the others are --
        # but a hypothetical future writer that co-persisted both must not
        # have its genuine `external_findings` silently dropped by routing to
        # the "both empty" tier below when they are not, in fact, empty.
        summary_text = ""

    # issue #792: a verdict recorded by the current record_review carries an
    # explicit marker for exactly this distinction -- handle it before the
    # shape-based tiers below, which exist only to infer the same thing for
    # verdicts recorded before this marker existed.
    if findings_channel == "vacuous":
        lines = [
            "## Required changes",
            "",
            "**REVIEWER FINDINGS UNAVAILABLE.** No structured findings list "
            "and no summary were recorded for this verdict. This is NOT a "
            "signal that there is nothing to change — it means the findings "
            "did not make it into `review-decision.json`. Inspect the PR's "
            "review comments and review threads on GitHub directly before "
            "doing anything else.",
            "",
        ]
        return "\n".join(lines)

    if verdict == "request_changes" and findings_channel == "derived":
        lines = [
            "## Required changes",
            "",
            "The reviewer did not record a structured findings list for this "
            "verdict. This is their summary, rendered verbatim so it is not "
            "lost — treat it as the findings to address before this PR can "
            "be approved:",
            "",
            defang_closing_keywords(summary_text),
            "",
        ]
        if new_shape:
            return _render_external_findings_section(lines, external_findings)
        return _finish_required_changes_section(lines)

    if verdict == "request_changes" and changes:
        if findings_channel == "external":
            intro = (
                "Address every item below. These include the reviewer's "
                "own findings and verified findings posted on the PR itself as "
                "comments, review bodies, or inline review threads."
            )
        else:
            intro = (
                "Address every item below. These are the reviewer's structured "
                "findings — the authoritative list of what must change before this "
                "PR can be approved."
            )
        lines = [
            "## Required changes",
            "",
            intro,
            "",
        ]
        lines.extend(f"- {defang_closing_keywords(change)}" for change in changes)
        lines.append("")
        if new_shape:
            return _render_external_findings_section(lines, external_findings)
        return _finish_required_changes_section(lines)

    if verdict == "request_changes" and summary_text:
        lines = [
            "## Required changes",
            "",
            "The reviewer did not record a structured findings list for this "
            "verdict. This is their summary, rendered verbatim so it is not "
            "lost — treat it as the findings to address before this PR can "
            "be approved:",
            "",
            defang_closing_keywords(summary_text),
            "",
        ]
        if new_shape:
            return _render_external_findings_section(lines, external_findings)
        return _finish_required_changes_section(lines)

    if not changes and not summary_text:
        lines = [
            "## Required changes",
            "",
            "**REVIEWER FINDINGS UNAVAILABLE.** No structured findings list "
            "and no summary were recorded for this verdict. This is NOT a "
            "signal that there is nothing to change — it means the findings "
            "did not make it into `review-decision.json`. Inspect the PR's "
            "review comments and review threads on GitHub directly before "
            "doing anything else.",
            "",
        ]
        return "\n".join(lines)

    # blocked verdict carrying required_changes and/or a summary (but not
    # both empty): suppressed by design (see docstring) so nothing renders.
    return ""


def _read_review_decision(decision_path: Path) -> dict[str, Any] | None:
    """Read a PR's ``review-decision.json`` as a dict, or ``None`` if absent.

    Mirrors ``OrchestratorApp._review_decision``'s read shape but returns
    ``None`` (rather than a sentinel) for missing/invalid files so the caller
    can distinguish "no verdict on disk" from "a verdict with an empty
    findings list" — only the latter should render an (empty) section.
    """
    if not decision_path.exists():
        return None
    try:
        with decision_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_verdict_newer_than_brief(decision_path: Path, brief_path: Path) -> bool:
    """Return True when the verdict file is strictly newer than the brief.

    Used by ``dispatch_rework`` to detect a stale brief that has drifted from
    a corrected ``review-decision.json`` (issue #632: the brief on disk is
    authoritative and ``dispatch_rework`` reads it verbatim, so without this
    check a hand-corrected verdict — the #510 case — never reaches the
    worker). Comparison uses nanosecond mtimes; an equal timestamp (the
    normal verdict path writes the decision immediately before the brief) is
    treated as not-stale so the fresh brief is not pointlessly rewritten.
    """
    if not decision_path.exists() or not brief_path.exists():
        return False
    return decision_path.stat().st_mtime_ns > brief_path.stat().st_mtime_ns


def _write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + atomic rename.

    Module-level (not an ``OrchestratorApp`` method) because callers that
    need it -- the module-level ``_write_rework_prompt`` below, which has no
    ``self``, and ``record_review``'s per-round archive copies -- must not
    write a torn file to a path another process may poll mid-write (the
    same failure class as the exists-is-not-content-ready incident).
    Mirrors ``OrchestratorApp._write_json``'s tmp+replace shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    tmp_path.replace(path)


# Issue #1268 (W11): the field set that identifies a review round. Two
# ``record_review`` calls that agree on all four are the same round (a
# retry); any difference -- including on an unchanged head -- is a distinct
# verdict and must never overwrite a prior round's archived text.
# ``reviewed_at``/``reviewed_patch_id`` are deliberately excluded: both
# legitimately differ on a byte-identical retry (a fresh timestamp, a
# recomputed patch id for the same diff) and would otherwise defeat the
# retry check entirely.
_ROUND_COMPARE_KEYS = ("decision", "summary", "required_changes", "reviewed_head_sha")


def _existing_round_numbers(rounds_dir: Path) -> list[int]:
    """Return the round numbers already archived under ``rounds_dir``.

    Directory names are ``round-<K>``, K unpadded (no zero-padding). Any
    reader comparing round numbers -- including the W13/#1270 consumer that
    will build on this layout -- must parse the trailing digits as ``int``
    before ordering; a lexicographic string sort is wrong here (``"round-10"``
    sorts before ``"round-2"``).
    """
    if not rounds_dir.is_dir():
        return []
    numbers: list[int] = []
    for entry in rounds_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("round-"):
            suffix = entry.name[len("round-") :]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return numbers


def _next_round_number(rounds_dir: Path, decision_payload: Mapping[str, Any]) -> int:
    """Return the round-K under which ``decision_payload`` should be archived.

    Two independent requirements, not one:

    * RETRY -- the exact same write re-submitted (e.g. a crash between
      archiving round-K and the ``save_state`` call that would have recorded
      the head move, followed by a re-submission of the identical payload for
      the same head) must land back in the same round-K, never mint
      round-(K+1).
    * DISTINCT -- a different decision, summary, required_changes, or
      reviewed_head_sha must never overwrite a prior round's archived text,
      even when the head SHA has not advanced.

    Head advancement alone is deliberately NOT used as the sole
    discriminator: it satisfies the retry requirement (an advanced head is
    always a new round) but fails the distinct-verdict requirement -- a
    second, genuinely different verdict recorded on an *unchanged* head would
    be misread as "same round, overwrite in place", reproducing -- one layer
    down -- the exact data-loss bug this archive exists to fix. Comparing the
    archived content itself sidesteps that, and also reads only the archive
    already on disk (never ``pr_state``/``state.json``), so a crash between a
    prior archive write and the ``save_state`` call that would have persisted
    the head move cannot desync this decision from what is actually on disk.
    """
    highest = max(_existing_round_numbers(rounds_dir), default=0)
    if highest == 0:
        return 1
    prior_decision = _read_review_decision(
        rounds_dir / f"round-{highest}" / "review-decision.json"
    )
    is_retry = prior_decision is not None and all(
        prior_decision.get(key) == decision_payload.get(key) for key in _ROUND_COMPARE_KEYS
    )
    return highest if is_retry else highest + 1


def _render_rework_prompt(
    state_file: Path,
    pr: dict[str, Any],
    issue_number: int | None,
    dispatch_note: str,
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render the rework brief text for a PR, without writing anything.

    Split out of ``_write_rework_prompt`` for issue #800 so ``dispatch_rework``
    can compare the brief a *current* renderer would produce against the bytes
    already on disk. Rendering is a pure function of the verdict file, the
    dispatch note, and the template set, so re-rendering an unchanged brief
    reproduces it exactly — which is what lets the caller detect renderer drift
    without rewriting (and re-timestamping) every brief it inspects.
    """
    pr_number = int(pr["number"])
    pr_dir = state_file.parent / "prs" / f"pr-{pr_number}"
    decision = _read_review_decision(pr_dir / "review-decision.json")
    required_changes_section = _render_required_changes_section(decision)
    return render_prompt(
        config.dispatch.rework_template,
        {
            "pr_number": pr_number,
            "pr_title": pr.get("title", ""),
            "pr_url": pr.get("url", ""),
            "issue_number": issue_number or "UNKNOWN",
            # The raw note stays available to templates; no shipped one
            # references it since #883.
            "dispatch_note": defang_closing_keywords(dispatch_note),
            # Pre-fenced, not bare, for the same reason as ``issue_body_block``
            # (#883): reviewer prose quotes pytest output and shell commands,
            # so it carries its own fences -- measured at 16 of 289 summaries
            # on disk, with pr-182's brief a rendered example of the break.
            # The width depends on the note's own backtick runs, so it cannot
            # be written in the template.
            "dispatch_note_block": fenced_block(defang_closing_keywords(dispatch_note), "md"),
            "required_changes_section": required_changes_section,
            "branch_name": pr.get("headRefName", ""),
        },
        search_dirs=_rework_prompt_search_dirs(config, repo_root=repo_root),
    )


def _write_rework_prompt(
    state_file: Path,
    pr: dict[str, Any],
    issue_number: int | None,
    dispatch_note: str,
    config: OrchestratorConfig,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Write a rework brief for a PR under the shared ``prs/pr-{N}`` directory.

    This module-level helper lets both the OrchestratorApp review path and the
    dead-session recovery path produce the same ``rework-prompt.md`` artifact.

    Single point of enforcement for issue #632: the reviewer's structured
    ``required_changes`` are read from ``review-decision.json`` here — not
    threaded through by callers — so the three call sites cannot omit them.
    The ``dispatch_note`` (formerly the ``$review_summary`` slot) carries the
    operational/review-prose note and is kept separate from the findings, so
    a churn/rescue message accompanies the findings instead of replacing
    them. The raw note is also written to a sidecar
    (``rework-dispatch-note.txt``) so a stale brief can be regenerated at
    dispatch time without losing its note.

    Single point of enforcement for issue #781 (outbound defang): reviewer
    prose reaches this brief through two independent template slots —
    ``$required_changes_section`` (via ``_render_required_changes_section``,
    already defanged there) and ``$dispatch_note`` (this parameter, often the
    same reviewer summary text under a different name). Both slots land in a
    document a worker reads and copies from when authoring its own PR body,
    so ``dispatch_note`` is defanged here too, at render time only — the
    sidecar on disk stays raw so a future regeneration re-defangs from the
    same source instead of compounding a prior rewrite.
    """
    pr_number = int(pr["number"])
    pr_dir = state_file.parent / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "rework-prompt.md"
    prompt = _render_rework_prompt(
        state_file,
        pr,
        issue_number,
        dispatch_note,
        config,
        repo_root=repo_root,
    )
    # Issue #714: enforce the no-merge contract on the *rendered output* so a
    # repo-local flat rework override that drops $section_no_merge_contract is
    # caught at the dispatch boundary.
    assert_no_merge_contract(prompt, context=f"rework prompt for PR #{pr_number}")
    # Issue #717: enforce the execution-contract escalation trigger on the
    # *rendered output* so a repo-local flat rework override that drops
    # $section_execution_contract is caught at the dispatch boundary.
    assert_execution_contract(prompt, context=f"rework prompt for PR #{pr_number}")
    # Issue #1010: enforce the widened containment clause on the *rendered
    # output* so a repo-local flat rework override that drops
    # $section_scope_contract or reverts to the old repo-scoped wording is
    # caught at the dispatch boundary.
    assert_containment(prompt, context=f"rework prompt for PR #{pr_number}")
    # Issue #1268 (W11), item 1 binding-comment #4: both the live brief and
    # its sidecar are polled by dispatch-time readers (regeneration checks,
    # worker launch), so a plain write_text here is the same
    # exists-is-not-content-ready failure class as the atomic-write
    # invariant already covers for JSON state -- tmp+replace via
    # ``_write_text_atomic`` closes that gap for text artifacts too.
    _write_text_atomic(prompt_path, prompt)
    # Sidecar: the raw (non-defanged) dispatch note, so a dispatch-time
    # regeneration (when review-decision.json is newer than the brief) can
    # reproduce the note and re-defang it fresh rather than parsing already-
    # rewritten markdown back out.
    _write_text_atomic(pr_dir / "rework-dispatch-note.txt", dispatch_note)
    return prompt_path
