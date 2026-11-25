"""The mechanics that record a Verdict, shared by both entry points (issue #3102).

``record_review`` (``orchestration/state_record_review.py``, the PR path) and
``record_local_review`` (``orchestration/local_lanes.py``, the Local path) used to
carry the same handling twice. This module owns the part that is the same: decision
and provenance validation, the summary requirement, deriving ``required_changes``,
human-call reclassification, the terminal and escalated guards, head pinning and the
stale-review check, the patch-id and line-content signature, building and persisting
the decision payload, and the verdict comment. It is step 1 of a chain; the decision
rules (Rework cap, escalation, status and label-edge choice) and the Events are step 2,
in :mod:`charlie_work.verdict_rules` (issue #3103).

It lives at the package root, not under ``orchestration/``, because every top-level
function defined there is auto-installed on ``OrchestratorApp``.

Shape. The module imports nothing from :mod:`charlie_work.workflow` at import time (it
would be circular). Names defined there that tests replace by path are resolved at call
time through :func:`_workflow`; a name the caller must resolve itself (``reviewed_at``,
the round-number function) is handed in as a value.

Each path keeps its own sequencing and calls these steps in turn, then the decision rules
in :mod:`charlie_work.verdict_rules`. Each path describes the
thing under review once, as a :class:`VerdictSubject` (its live head, how to refuse a
terminal or escalated record, how to post the verdict comment), and states every
difference between the two paths in a :class:`VerdictDifferences`; nothing is inferred
from which path is calling. Nothing here writes ``state.json``: each path writes state
through its own route (``_wf.save_state`` on the PR path, ``write_gate.save_state`` on
the Local path).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from charlie_work import review_interdiff
from charlie_work.command_result import CommandResult
from charlie_work.review_decision import reclassify_human_call_verdict, record_decision


def _workflow() -> Any:
    # ``charlie_work.workflow`` imports the orchestration delegates, which import
    # this module -- resolve it at call time. Going through it keeps every existing
    # ``charlie_work.workflow.*`` monkeypatch site intercepting.
    import charlie_work.workflow as wf

    return wf


@dataclass(frozen=True)
class VerdictWording:
    """Refusal wording that differs between the two paths (declared, not fixed here)."""

    invalid_decision: str  # may use ``{decision}``
    invalid_provenance: str  # may use ``{provenance}`` and ``{allowed}``
    head_noun: str  # "PR" / "branch": "live <noun> head"
    reviewed_head_flag: str  # how the caller spells the reviewed-head argument
    choose_head_hint: str  # "use --reviewed-head to choose" / "pass reviewed_head to choose"


@dataclass(frozen=True)
class VerdictDifferences:
    """Every place the two paths record a Verdict differently, as explicit inputs."""

    wording: VerdictWording
    # The Local path coerces each ``required_changes`` item with ``str()``; the PR path
    # keeps the items as given.
    stringify_required_changes: bool
    # ``local: True`` and the ``human_call_*`` fields on the decision file (Local only).
    local_decision_fields: bool
    # The Local verdict comment lists the required changes even when they are the
    # summary itself (``findings_channel == "derived"``); the PR comment does not.
    comment_repeats_derived_summary: bool


@dataclass(frozen=True)
class VerdictSubject:
    """The thing under review, as one path sees it."""

    pr_number: int
    issue_number: int | None
    # Pre-built refusals: the wording and data keys differ per path. ``None`` when the
    # record is not terminal.
    terminal_refusal: CommandResult | None
    escalated_refusal: CommandResult
    # The state read the escalated guard checks (lazy: the PR path does not read it
    # when the PR is terminal).
    guard_state: Callable[[], dict[str, Any]]
    # The head the subject has right now (lazy: the Local path resolves it with git,
    # and only after its guards).
    live_head: Callable[[], str | None]
    # Posts a verdict comment; returns the error text on failure, ``None`` on success.
    post_comment: Callable[[str], str | None]


@dataclass(frozen=True)
class DerivedVerdict:
    decision: str  # after human-call reclassification
    summary_text: str
    required_changes: list[str]
    findings_channel: str | None
    reclassified_human_call: tuple[str, str] | None


@dataclass(frozen=True)
class PinnedHead:
    sha: str
    source: str  # "packet" | "live"
    packet_head_sha: str | None
    live_head_sha: str | None


@dataclass(frozen=True)
class ReviewedContent:
    patch_id: str
    signature: Any  # DiffContentSignature


@dataclass(frozen=True)
class DecisionRound:
    round_number: int
    round_dir: Path
    provenance_caveat: str | None


def derive_verdict(
    decision: str,
    *,
    verdict_provenance: str,
    summary: str,
    summary_file: Path | None,
    required_changes: Sequence[Any] | None,
    human_decision_markers: Any,
    differences: VerdictDifferences,
) -> DerivedVerdict | CommandResult:
    """Validate the call and derive what the decision file will carry."""
    wf = _workflow()
    wording = differences.wording
    if decision not in {"approved", "request_changes", "blocked"}:
        return CommandResult(
            False,
            wording.invalid_decision.format(decision=decision),
            {"reason": "invalid_decision"},
        )
    # Issue #1265: no default -- every caller must say where the verdict came from. This
    # membership check catches a garbage/typo'd literal (a missing argument is already
    # caught at the call boundary, by the keyword-only parameter having no default).
    if verdict_provenance not in wf.VERDICT_PROVENANCE_VALUES:
        return CommandResult(
            False,
            wording.invalid_provenance.format(
                provenance=verdict_provenance,
                allowed=", ".join(sorted(wf.VERDICT_PROVENANCE_VALUES)),
            ),
            {"reason": "invalid_verdict_provenance"},
        )
    summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
    # Issue #11: reject empty summary for request_changes/blocked decisions before any
    # state/label mutation.
    if decision in {"request_changes", "blocked"} and not summary_text.strip():
        return CommandResult(
            False,
            f"--summary or --summary-file is required for decision '{decision}'",
            {"reason": "missing_summary"},
        )

    # Issue #792: required_changes has a near-0% fill rate because prompts/review.md
    # historically documented it as optional, so reviewers reliably fill in `summary`
    # and skip the structured list. Derive here, at the point of record, so whatever
    # `required_changes` lands on disk is exactly what a worker's rework brief will show.
    #
    # This step never *rejects* a verdict for an empty required_changes -- see
    # test_record_review_never_rejects_for_empty_required_changes. A reject path here is
    # the tempting wrong fix: a False return writes no review-decision.json, the caller
    # logs review_verdict_missed and moves on, the next pass still sees the PR as
    # pending, and it gets re-dispatched to a fresh reviewer -- an unbounded re-review
    # loop with no cap and no escalation.
    if required_changes:
        effective = (
            [str(item) for item in required_changes]
            if (differences.stringify_required_changes)
            else list(required_changes)
        )
    else:
        effective = []
    findings_channel: str | None = None
    if decision in {"request_changes", "blocked"} and not effective:
        if wf._summary_is_vacuous(summary_text):
            # Nothing derivable. Persist the empty list anyway (never reject) with an
            # explicit marker so `_render_required_changes_section` can render its loud
            # tier-3 warning instead of treating this like a real derivation.
            findings_channel = "vacuous"
        else:
            # The F1 extractor (issue #792): "derived" tells the renderer this single
            # item is reviewer prose, not an itemized list.
            effective = [summary_text.strip()]
            findings_channel = "derived"

    # Issue #1642: a ``request_changes`` verdict whose own findings ask for a
    # human/operator decision is a misfiled ``blocked`` verdict. Reclassify here, at the
    # single choke point every verdict write passes through, so the verdict takes the
    # exact ``blocked`` path: operator queue, no rework brief, no rework budget. Runs on
    # the post-derivation list, so a human call written into ``summary`` is caught too.
    decision, reclassified = reclassify_human_call_verdict(
        decision,
        effective,
        markers=human_decision_markers,
        verdict_provenance=verdict_provenance,
    )
    return DerivedVerdict(decision, summary_text, effective, findings_channel, reclassified)


def check_recordable(subject: VerdictSubject) -> CommandResult | None:
    """The terminal and escalated guards: a refusal, or ``None`` when recording may go on."""
    # Issue #1131: a terminal-state record (merged or closed) must never have its decision
    # file overwritten or rework routed. Unconditional (not gated on ``allow_stale_head``):
    # even the operator CLI must not overwrite a merge-authorizing decision file.
    if subject.terminal_refusal is not None:
        return subject.terminal_refusal
    # Escalation is terminal for verdict recording too: without this, a late-arriving
    # verdict silently overwrites status="escalated" and re-enters the record into the
    # pipeline (observed re-escalating 2-3x). A human re-arms it with `charlie
    # unescalate`, after which verdicts record normally again.
    state = subject.guard_state()
    pr_state = (state.get("prs") or {}).get(str(subject.pr_number), {})
    issue_state = (
        (state.get("issues") or {}).get(str(subject.issue_number), {})
        if subject.issue_number is not None
        else {}
    )
    if pr_state.get("status") == "escalated" or issue_state.get("status") == "escalated":
        return subject.escalated_refusal
    return None


def pin_reviewed_head(
    subject: VerdictSubject,
    *,
    packet_head_sha: str | None,
    reviewed_head: str | None,
    allow_stale_head: bool,
    differences: VerdictDifferences,
) -> PinnedHead | CommandResult:
    """Pin the verdict to the head the reviewer read, refusing a stale review.

    ``reviewed_head_sha`` must reflect the packet the reviewer actually read, not a fresh
    fetch made at verdict time: a commit landing between packet generation and verdict
    recording would otherwise silently reattribute the decision to a head that was never
    reviewed. Fall back to the live head only when no packet exists.

    Issue #467/#1072: when the packet head and the live head disagree, the caller must
    choose explicitly; an automated caller that passes ``reviewed_head`` matching the
    packet is still refused when the live head has moved (``allow_stale_head`` is the
    operator CLI's explicit-choice design). The failure direction is toward redundant
    work (re-review next pass), never toward authorizing a merge on the wrong head.
    """
    wording = differences.wording
    live_head_sha = subject.live_head()
    if reviewed_head is not None:
        if packet_head_sha is not None and reviewed_head == packet_head_sha:
            if (
                not allow_stale_head
                and live_head_sha is not None
                and packet_head_sha != live_head_sha
            ):
                return CommandResult(
                    False,
                    f"reviewed head ({reviewed_head}) matches packet head "
                    f"({packet_head_sha}) but live {wording.head_noun} head has moved to "
                    f"({live_head_sha}); verdict not recorded — head moved "
                    "during build, will re-review next pass",
                    {
                        "pr": subject.pr_number,
                        "issue": subject.issue_number,
                        "reason": "head_moved_during_build",
                        "packet_head_sha": packet_head_sha,
                        "live_head_sha": live_head_sha,
                    },
                )
            return PinnedHead(reviewed_head, "packet", packet_head_sha, live_head_sha)
        if live_head_sha is not None and reviewed_head == live_head_sha:
            return PinnedHead(reviewed_head, "live", packet_head_sha, live_head_sha)
        options: list[str] = []
        if packet_head_sha is not None:
            options.append(f"packet head {packet_head_sha}")
        if live_head_sha is not None:
            options.append(f"live head {live_head_sha}")
        options_str = " or ".join(options) if options else "any available head"
        return CommandResult(
            False,
            f"{wording.reviewed_head_flag} {reviewed_head} does not match {options_str}",
            {"reason": "reviewed_head_mismatch"},
        )
    if (
        packet_head_sha is not None
        and live_head_sha is not None
        and (packet_head_sha != live_head_sha)
    ):
        return CommandResult(
            False,
            f"review packet head ({packet_head_sha}) differs from live "
            f"{wording.head_noun} head ({live_head_sha}); "
            f"{wording.choose_head_hint} the head the verdict applies to",
            {"reason": "head_moved_during_build"},
        )
    if packet_head_sha is not None:
        return PinnedHead(packet_head_sha, "packet", packet_head_sha, live_head_sha)
    if live_head_sha is not None:
        return PinnedHead(live_head_sha, "live", packet_head_sha, live_head_sha)
    return CommandResult(
        False,
        f"no packet or live {wording.head_noun} head available",
        {"reason": "no_head_available"},
    )


def reviewed_content(diff: str | None) -> ReviewedContent:
    """The patch-id and line-content signature of the reviewed diff.

    Patch-id detects actual content changes (issue #222: base-update merges can advance
    the head SHA without changing the diff). Every terminal decision persists it so the
    review-queue enumerator can carry it forward on content-identical heads. The tier-2
    content signature (issue #414) is persisted too: patch-id is unstable across every
    main advance, so the ordered +/- line stream and changed-file set are recorded here,
    while the diff is freshly known, and a later carry-forward never needs to
    reconstruct the historical diff. ``None`` or empty diff records an empty patch-id.
    """
    wf = _workflow()
    if not diff:
        return ReviewedContent("", wf.DiffContentSignature((), frozenset()))
    return ReviewedContent(wf._calculate_patch_id(diff), wf._diff_content_signature(diff))


def write_decision_round(
    pr_dir: Path,
    subject: VerdictSubject,
    *,
    decision: str,
    summary_text: str,
    required_changes: list[str],
    findings_channel: str | None,
    reclassified_human_call: tuple[str, str] | None,
    verdict_provenance: str,
    verdict_source: str | None,
    pinned: PinnedHead,
    content: ReviewedContent,
    body_sha256: str | None,
    external_findings: list[str] | None,
    ingestion_before: str | None,
    escalated: bool,
    reviewed_at: str,
    next_round_number: Callable[[Path, dict[str, Any]], int],
    differences: VerdictDifferences,
) -> DecisionRound:
    """Build the decision payload and persist it (decision file plus the round archive).

    Call under the calling function's state lock: the read of a prior decision file below is atomic
    against ``merge_authorize``, which holds the same lock. ``reviewed_at`` is the
    caller's own stamp, taken where it always was: it is the next round's ingestion lower
    bound when no ``before`` exists, so it must not be taken later.
    """
    wf = _workflow()
    decision_payload: dict[str, Any] = {
        "pr_number": subject.pr_number,
        "issue_number": subject.issue_number,
        "decision": decision,
        "summary": summary_text,
        "required_changes": required_changes,
        # Issue #1265: unconditional -- every caller must say where the verdict came from.
        "verdict_provenance": verdict_provenance,
        "reviewed_head_sha": pinned.sha,
        "reviewed_head_source": pinned.source,
        "reviewed_patch_id": content.patch_id,
        "reviewed_changed_lines": list(content.signature.changed_lines),
        "reviewed_changed_files": sorted(content.signature.changed_files),
        "reviewed_has_binary": content.signature.has_binary,
        "carried_forward_from": [],
        "reviewed_at": reviewed_at,
    }
    if differences.local_decision_fields:
        decision_payload["local"] = True
    # Issue #1939: a hash of the PR body the reviewer read (PR path only).
    if body_sha256 is not None:
        decision_payload["reviewed_body_sha256"] = body_sha256
    # Only present when the derivation ran (issue #792).
    if findings_channel is not None:
        decision_payload["findings_channel"] = findings_channel
    # Issue #999: external findings live in their own field, separate from the
    # reviewer's required_changes.
    if external_findings is not None:
        decision_payload["external_findings"] = external_findings
    # The ingestion upper bound (issue #998): the next round's ``since`` derives from it.
    if ingestion_before is not None:
        decision_payload["before"] = ingestion_before
    # Issue #1340: which extractor found the verdict block ("log", "events", "file:*").
    if verdict_source is not None:
        decision_payload["verdict_source"] = verdict_source
    # Issue #1485: a verdict extracted from a dead reviewer's session artifacts may be
    # incomplete or factually wrong; persist a caveat so every downstream surface can
    # render it. Only for blocked/request_changes.
    provenance_caveat: str | None = None
    if decision in {"request_changes", "blocked"}:
        provenance_caveat = wf.provenance_caveat_for(verdict_source)
        if provenance_caveat is not None:
            decision_payload["provenance_caveat"] = provenance_caveat
    if differences.local_decision_fields and reclassified_human_call is not None:
        matched_item, matched_marker = reclassified_human_call
        decision_payload["human_call_reclassified_from"] = "request_changes"
        decision_payload["human_call_marker"] = {"item": matched_item, "marker": matched_marker}
    decision_payload["escalated"] = escalated

    # Issue #934: ``decision_payload`` is a fresh dict that would silently discard an
    # ``authorized_override`` written by ``merge_authorize``; carry it forward so a later
    # verdict for the same record does not resurrect the false-positive tripwire finding.
    decision_path = pr_dir / "review-decision.json"
    if decision_path.exists():
        try:
            with decision_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict) and isinstance(
                existing.get("authorized_override"), dict
            ):
                decision_payload["authorized_override"] = existing["authorized_override"]
        except (OSError, json.JSONDecodeError):
            pass
    # Issue #1268 (W11): round_number comes from the archive already on disk, never from
    # state, so a crash between this write and ``save_state`` cannot desync it.
    rounds_dir = pr_dir / "rounds"
    round_number = next_round_number(rounds_dir, decision_payload)
    round_dir = rounds_dir / f"round-{round_number}"
    # Single writer (issue #1362 Stage 2): round-file-then-flat, so a crash between the
    # two writes still leaves a durable, readable verdict.
    record_decision(pr_dir, decision_payload, None)
    # Issue #2837: archive this round's diff.patch alongside the decision, so the next
    # packet can build an interdiff from it. The live pr_dir/diff.patch belongs to the
    # packet the reviewer just read and the next review() overwrites it in place.
    review_interdiff.archive_round_diff(pr_dir, round_dir)
    return DecisionRound(round_number, round_dir, provenance_caveat)


def post_verdict_comment(
    subject: VerdictSubject,
    *,
    round_number: int,
    decision: str,
    summary_text: str,
    provenance_caveat: str | None,
    required_changes: list[str],
    findings_channel: str | None,
    differences: VerdictDifferences,
) -> str | None:
    """Build the verdict comment and post it; returns the error text on failure."""
    body_parts = [f"## Fleet review - round {round_number} - {decision}"]
    if summary_text:
        body_parts.append(summary_text)
    # Issue #1485: the provenance caveat goes before the required-changes section so an
    # operator knows to re-verify factual claims before acting.
    if provenance_caveat is not None:
        body_parts.append(provenance_caveat)
    # A "derived" list is ``[summary_text.strip()]`` -- rendering it would post the
    # summary twice, so the PR comment skips it (the Local comment still lists it).
    if required_changes and (
        differences.comment_repeats_derived_summary or findings_channel != "derived"
    ):
        body_parts.append(
            "### Required changes\n" + "\n".join(f"- {rc}" for rc in required_changes)
        )
    return subject.post_comment("\n\n".join(body_parts))


def with_comment_error(label_error: dict[str, Any] | None, error: str) -> dict[str, Any]:
    """Fold a comment failure into the label error (separate from a transition failure)."""
    if label_error is None:
        return {"comment_error": error}
    label_error["comment_error"] = error
    return label_error
