"""Recorded review outcomes: required-changes derivation, external findings, dedup.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the required-changes / external-findings half of the ``test_record_review_*`` seam -- summary validation, derivation (issue #792), external-finding folding, crash-summary filtering (issue #1269), and dedup. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import _load_external_fixture
from charlie_work.config import OrchestratorConfig
from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.verdict_parsing import (
    REVIEW_SESSION_FAILED_HEADING,
    REVIEW_SESSION_SUMMARY_HEADING,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


# --- Issue #11: reject empty summary for request_changes/blocked decisions ----


def test_record_review_request_changes_rejects_empty_summary(tmp_path: Path) -> None:
    """Issue #11: request_changes with empty summary is rejected before state/label mutation."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "request_changes", summary="", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message
    # Verify no state/label mutation occurred
    assert load_state(paths.state_file).get("prs", {}).get("456") is None
    assert (123, "agent:needs-rework") not in fake_gh.labels_added
    # Verify no rework prompt was written
    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert not rework_prompt.exists()


def test_record_review_blocked_rejects_empty_summary(tmp_path: Path) -> None:
    """Issue #11: blocked with empty summary is rejected before state/label mutation."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "blocked", summary="", verdict_provenance="fresh_llm_review")

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message
    # Verify no state/label mutation occurred
    assert load_state(paths.state_file).get("prs", {}).get("456") is None
    assert (123, "agent:blocked") not in fake_gh.labels_added


def test_record_review_request_changes_rejects_whitespace_only_summary(tmp_path: Path) -> None:
    """Issue #11: request_changes with whitespace-only summary is rejected."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "request_changes", summary="   \n\t  ", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message


def test_record_review_approved_allows_empty_summary(tmp_path: Path) -> None:
    """Issue #11: approved with empty summary is allowed (no validation required)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "approved", summary="", verdict_provenance="fresh_llm_review")

    assert result.ok is True
    assert result.message == "review recorded (head from live)"
    # Verify state mutation occurred
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "approved"


def test_record_review_decision_payload_includes_required_changes(tmp_path: Path) -> None:
    """Issue #11: decision payload always includes required_changes field.

    Issue #792: a request_changes verdict with no required_changes no longer
    persists an empty list when `summary` has real content -- record_review
    now derives required_changes from summary at write time, so the
    persisted list here is `["fix A"]`, not `[]`. See
    test_record_review_derives_required_changes_from_summary and
    test_record_review_persists_vacuous_marker_when_nothing_derivable for the
    dedicated coverage of both derivation outcomes.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "required_changes" in decision
    assert decision["required_changes"] == []
    # approved is never subject to derivation: no marker at all (issue #792 AC-4).
    assert "findings_channel" not in decision

    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "required_changes" in decision
    assert decision["required_changes"] == ["fix A"]
    assert decision["findings_channel"] == "derived"


# --------------------------------------------------------------------------
# Issue #792: required_changes has a near-0% fill rate because reviewers
# reliably fill in `summary` and skip the structured list. record_review now
# derives required_changes from summary at write time instead of leaving it
# empty for a downstream renderer to paper over. These tests cover the 8
# acceptance criteria from the issue directly against the record_review
# entrypoint (not the decision-payload dict alone).
# --------------------------------------------------------------------------


def test_record_review_derives_required_changes_from_summary(tmp_path: Path) -> None:
    """AC-1: request_changes + empty required_changes + extractable prose ->
    persisted with required_changes populated from that prose, and
    findings_channel marks it as derived (not an itemized reviewer list)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    prose = "The null check in parse() is missing, causing a crash on empty input."
    result = app.record_review(
        456,
        "request_changes",
        summary=prose,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == [prose]
    assert decision["findings_channel"] == "derived"
    assert decision["summary"] == prose


def test_record_review_persists_vacuous_marker_when_nothing_derivable(
    tmp_path: Path,
) -> None:
    """AC-2: request_changes + empty required_changes + a summary with no
    extractable findings is still PERSISTED -- never rejected -- with
    required_changes: [], findings_channel: "vacuous", and a distinct
    required_changes_vacuous event alongside the general record_review
    event. A blank/whitespace-only summary cannot reach this derivation at
    all (issue #11's gate rejects it outright before any state mutation, so
    that shape can never produce a persisted vacuous marker); the only
    non-blank text this function is entitled to call vacuous is the one
    known historical placeholder, so that is what exercises this path here.
    See test_record_review_positive_control_legacy_vacuous_summary (AC-8)
    for the same literal used to prove the discriminator fires on real
    on-disk data, not just a fixture invented for this test.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == []
    assert decision["findings_channel"] == "vacuous"

    state = load_state(paths.state_file)
    kinds = [
        event["kind"] for event in state["events"] if event["payload"].get("pr_number") == 456
    ]
    assert "required_changes_vacuous" in kinds
    assert "record_review" in kinds


def test_record_review_positive_control_legacy_vacuous_summary(tmp_path: Path) -> None:
    """AC-8 positive control: the exact LEGACY_VACUOUS_SUMMARY literal --
    real text that shipped on six on-disk pre-#795 cross-family verdicts --
    fed through record_review must be classified vacuous, proving the
    discriminator actually fires on real historical data rather than only
    on a synthetic fixture invented for this test."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["findings_channel"] == "vacuous"
    assert decision["required_changes"] == []


def test_record_review_blocked_also_derives_required_changes(tmp_path: Path) -> None:
    """The derivation is not request_changes-specific: `blocked` verdicts go
    through the same rework-adjacent path (merge-conflict / janitor routes
    can carry a `blocked` decision forward) and must not silently drop the
    reviewer's stated reason either."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "blocked",
        summary="Security review flagged an unauthenticated endpoint.",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["findings_channel"] == "derived"
    assert decision["required_changes"] == ["Security review flagged an unauthenticated endpoint."]


def test_record_review_folds_external_findings_into_required_changes(
    tmp_path: Path,
) -> None:
    """Issue #950/#999: verified external PR comments, review bodies, and
    inline review threads are ingested into ``review-decision.json`` at
    record time. Since #999 they ride in their own ``external_findings``
    field (separate from the reviewer's ``required_changes``) so
    ``findings_channel`` keeps describing only the reviewer's list. Bot-
    authored content is filtered using the API ``user.type`` discriminator."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_external_issue_comments[456] = _load_external_fixture("issue_comments")
    fake_gh.pr_external_reviews[456] = _load_external_fixture("reviews")
    fake_gh.pr_external_review_comments[456] = _load_external_fixture("review_comments")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="The retry wrapper swallows the exception type (Fixes #649).",
        required_changes=["add a regression test"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    # The reviewer provided an itemized list, so findings_channel stays
    # unset (the derivation block only runs on an empty list) -- it is NOT
    # flipped to "external" anymore (issue #999).
    assert "findings_channel" not in decision
    assert decision["summary"] == "The retry wrapper swallows the exception type (Fixes #649)."
    # Internal finding is preserved untouched in required_changes.
    assert decision["required_changes"] == ["add a regression test"]
    # Human-authored external findings ride in their own field.
    external = decision["external_findings"]
    assert any("interactive PR list" in item for item in external)
    assert any("https://github.com/cli/cli/pull/14076" in item for item in external)
    assert any("whole test" in item for item in external)
    # Bot-authored bodies are skipped via user.type == "Bot".
    assert not any("v0.83.5" in item for item in external)
    assert not any("git.Client wrapper" in item for item in external)


def test_record_review_external_findings_override_vacuous_summary(
    tmp_path: Path,
) -> None:
    """Issue #950: an entirely external verdict must not be mis-binned as
    ``"vacuous"`` just because the internal summary is the legacy placeholder."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_external_issue_comments[456] = _load_external_fixture("issue_comments")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["findings_channel"] == "external"
    expected_body = _load_external_fixture("issue_comments")[0]["body"].strip()
    assert decision["required_changes"] == [expected_body]


def test_record_review_external_findings_scoped_to_previous_round(tmp_path: Path) -> None:
    """A comment already surfaced in a prior round's required_changes (or
    predating review entirely) must not be re-surfaced in a later round.

    Without a since-cutoff, `_collect_external_findings` re-fetches the PR's
    entire comment history on every `record_review` call, so a stale
    round-1 finding (and a worker's unmarked rework reply) pile up on top of
    the one new finding a human actually left for round 2 -- burying it in
    `_write_rework_prompt`. The fix scopes ingestion to comments posted after
    the previous round's `reviewed_at`, so a comment seen once can
    structurally never come back.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)

    # Round 1: an external comment predates the review packet and gets folded
    # in as usual, since no prior decision exists yet.
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": "The retry wrapper swallows the exception type.",
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result1 = app.record_review(
        456,
        "request_changes",
        summary="internal summary round 1",
        required_changes=["fix the internal thing"],
        verdict_provenance="fresh_llm_review",
    )
    assert result1.ok is True
    decision1 = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field, not required_changes.
    assert decision1["required_changes"] == ["fix the internal thing"]
    external1 = decision1["external_findings"]
    assert any("retry wrapper" in item for item in external1)
    reviewed_at_round1 = decision1["reviewed_at"]
    assert reviewed_at_round1

    # Round 2: the round-1 comment is still sitting on the PR (GitHub never
    # deletes it), plus a genuinely new comment posted after round 1 finished.
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": "The retry wrapper swallows the exception type.",
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "body": "Also missing: a regression test for the timeout path.",
            "user": {"login": "a-real-human", "type": "User"},
            # Far enough in the future to postdate round 1's reviewed_at
            # (real wall-clock time) regardless of when this test runs.
            "created_at": "2099-01-01T00:00:00Z",
        },
    ]
    result2 = app.record_review(
        456,
        "request_changes",
        summary="internal summary round 2",
        required_changes=["fix the internal thing, round 2"],
        verdict_provenance="fresh_llm_review",
    )
    assert result2.ok is True
    decision2 = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # The reviewer's required_changes stay separate from external findings.
    assert decision2["required_changes"] == ["fix the internal thing, round 2"]
    external2 = decision2["external_findings"]

    # The new, post-round-1 comment is ingested.
    assert any("regression test for the timeout path" in item for item in external2)
    # The stale round-1 comment (already surfaced once) is not re-ingested.
    assert not any("retry wrapper" in item for item in external2)


def test_record_review_derived_with_external_findings_preserves_both(
    tmp_path: Path,
) -> None:
    """Issue #999 core round-trip: a ``derived`` verdict (reviewer produced
    no itemized list, so the summary was back-derived) that ALSO carries
    external PR comments must persist the derived summary in
    ``required_changes`` with ``findings_channel == "derived"`` (NOT
    overwritten to ``"external"``) and the external findings in their own
    ``external_findings`` field. The rendered rework brief must show the
    derived summary verbatim (not as a single bullet) AND the external
    items as bullets under their own heading.

    Before #999, ``record_review`` merged the external items into
    ``required_changes`` and flipped the channel to ``"external"``, so the
    renderer took the itemized path and the multi-paragraph derived summary
    -- the only representation of what the reviewer wanted changed -- was
    emitted as one ``- {...}`` bullet.
    """
    from charlie_work.workflow import _render_required_changes_section

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    prose = (
        "The retry wrapper swallows the exception type. Callers cannot "
        "distinguish a transient failure from a permanent one, so every "
        "retry loop masks real bugs."
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": "The migration needs a rollback path before this can land.",
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2099-01-01T00:00:00Z",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # No itemized required_changes -> record_review derives from summary.
    result = app.record_review(
        456,
        "request_changes",
        summary=prose,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    # The channel stays "derived" -- NOT overwritten to "external".
    assert decision["findings_channel"] == "derived"
    # required_changes holds the derived summary, NOT the external item.
    assert decision["required_changes"] == [prose]
    # External findings ride in their own field.
    assert decision["external_findings"] == [
        "The migration needs a rollback path before this can land."
    ]

    # The rendered brief shows the derived summary verbatim (not a bullet)
    # and the external item under its own heading.
    section = _render_required_changes_section(decision)
    assert prose in section
    assert f"- {prose}" not in section
    assert "did not record a structured findings list" in section
    assert "## Findings posted on the PR itself" in section
    assert "- The migration needs a rollback path before this can land." in section


# --------------------------------------------------------------------------
# Issue #1269 (W12): _collect_external_findings filters out reviewer-session
# crash summaries -- posted by _extract_review_session_summary when a
# reviewer dies without a verdict -- so they are never ingested as if they
# were genuine human/peer-agent findings. This is the collector-side half of
# the fix; the render-side guard (tested above, near
# _render_required_changes_section) is what reaches records already
# persisted before this filter existed.
# --------------------------------------------------------------------------


def test_record_review_does_not_ingest_an_unstamped_crash_summary(tmp_path: Path) -> None:
    """A reviewer-session crash summary posted before the
    ORCHESTRATOR_COMMENT_MARKER provenance stamp existed (#1242/55cecd9) --
    or before this filter itself -- carries no marker for
    _is_orchestrator_comment to catch, so it used to be ingested as if it
    were a genuine external finding. It must not be, while a genuine human
    finding posted alongside it is still ingested."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    crash_body = (
        f"{REVIEW_SESSION_SUMMARY_HEADING}\n\n"
        "The automated reviewer ran for 4 turns (2 tool calls) but did not "
        "produce a structured verdict.\n"
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": crash_body,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        },
        {
            "body": "The migration script drops the index without a guard.",
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-08-09T12:05:00Z",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    external = decision.get("external_findings", [])
    assert not any(REVIEW_SESSION_SUMMARY_HEADING in item for item in external), (
        "the crash summary must not be ingested as an external finding"
    )
    assert any("migration script drops the index" in item for item in external), (
        "the genuine human finding must still be ingested"
    )


def test_record_review_does_not_ingest_a_synthetic_launch_failed_crash_summary(
    tmp_path: Path,
) -> None:
    """The other crash-summary heading (REVIEW_SESSION_FAILED_HEADING, for a
    reviewer that died before its first turn) is filtered too. No captured
    fixture carries this heading (the jc#1394 population happens to be
    entirely the summary variant), so this specimen is synthetic -- built
    from the shared constant, never a hardcoded copy."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    crash_body = (
        f"{REVIEW_SESSION_FAILED_HEADING}\n\n"
        "The automated reviewer exited before running a single turn, so no "
        "review was performed.\n"
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": crash_body,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision.get("external_findings", []) == []


def test_record_review_dedups_identical_bodies_within_a_round(tmp_path: Path) -> None:
    """Issue #1269 part (c): the same finding posted byte-for-byte on two
    different surfaces within one round (e.g. an issue comment and a review
    body -- the crash-recovery path reposts across more than one surface)
    collapses to a single entry, so a rework brief does not present N
    duplicate copies of the same finding as if they were N distinct ones."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    duplicated_body = "The migration script drops the index without a guard."
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": duplicated_body,
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        }
    ]
    fake_gh.pr_external_reviews[456] = [
        {
            "body": duplicated_body,
            "user": {"login": "a-real-human", "type": "User"},
            "submitted_at": "2026-08-09T12:01:00Z",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    external = decision.get("external_findings", [])
    assert external.count(duplicated_body) == 1, (
        f"expected the duplicated body collapsed to one entry, found: {external}"
    )


def test_record_review_dedup_does_not_over_merge_similar_but_distinct_findings(
    tmp_path: Path,
) -> None:
    """Over-aggression guard: two genuinely different findings that happen
    to share a substring must stay distinct -- the dedup is exact-string
    equality only (dict.fromkeys), never a whitespace/substring merge."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    finding_a = "The retry wrapper swallows the exception type in the fetch path."
    finding_b = "The retry wrapper swallows the exception type in the push path."
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": finding_a,
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        },
        {
            "body": finding_b,
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-08-09T12:01:00Z",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    external = decision.get("external_findings", [])
    assert finding_a in external
    assert finding_b in external
    assert len(external) == 2, f"two distinct findings must not be merged into one: {external}"


def test_record_review_never_rejects_for_empty_required_changes(tmp_path: Path) -> None:
    """AC-3 regression pin, referenced by name in record_review's derivation
    comment. A reject-on-empty-required_changes gate here would recreate the
    unbounded re-review loop this fix closes: a False CommandResult writes no
    review-decision.json, the caller logs review_verdict_missed, the PR still
    reads as pending next pass, and it gets re-dispatched to a fresh reviewer
    forever. Assert both the vacuous case and the derivable case return
    ok=True -- neither shape may ever produce ok=False on account of
    required_changes."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    vacuous_result = app.record_review(
        456,
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )
    assert vacuous_result.ok is True

    fake_gh.pr_head_shas[456] = "sha-2"
    derivable_result = app.record_review(
        456,
        "request_changes",
        summary="Real finding here.",
        required_changes=None,
        verdict_provenance="fresh_llm_review",
    )
    assert derivable_result.ok is True
