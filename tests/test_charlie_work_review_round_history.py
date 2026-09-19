"""Review prompt round history: prior-round sections, interdiffs, verdict block.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the prior-review / round-history half of the review-prompt seam -- prior-round sections, interdiffs, round-history summaries, and the fenced-JSON verdict block. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.verdict_parsing import REVIEW_SESSION_SUMMARY_HEADING
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_first_round_has_no_prior_review_section(tmp_path: Path) -> None:
    """No review-decision.json on disk yet: this is a first-round review, so
    $prior_review_section must render empty."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet
    assert not (paths.prs / "pr-456" / "interdiff.patch").exists()


def test_review_prior_verdict_pending_has_no_prior_review_section(tmp_path: Path) -> None:
    """A pending (never-recorded) prior decision must not be treated as round-2
    findings, even if it carries a reviewed_head_sha from a template write."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending", "reviewed_head_sha": "sha-old"}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet


def test_review_round2_successful_compare_includes_prior_findings_and_interdiff(
    tmp_path: Path,
) -> None:
    """Round-2 review (prior terminal verdict on an earlier head): the packet
    must surface round-1 decision/summary/required_changes and write/reference
    an interdiff between the prior reviewed head and the live head."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "fix the null check in validate()",
                "required_changes": ["add null check", "handle empty list"],
                "reviewed_head_sha": "sha-old",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "sha-old" in packet
    assert "fix the null check in validate()" in packet
    assert "add null check" in packet
    assert "handle empty list" in packet
    assert "still in scope" in packet  # anti-anchoring instruction

    interdiff_path = decision_dir / "interdiff.patch"
    assert interdiff_path.exists()
    interdiff_text = interdiff_path.read_text(encoding="utf-8")
    assert "sha-old" in interdiff_text
    assert "sha-abc123" in interdiff_text
    assert str(interdiff_path) in packet


def test_review_round2_failed_compare_omits_interdiff(tmp_path: Path) -> None:
    """When the prior-head comparison fails (404/GC'd SHA/API error), the
    packet must still carry the round-1 findings but state that no interdiff
    could be generated — it must never block packet generation."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    fake_gh.compare_diff_overrides[("sha-old", "sha-abc123")] = None
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "fix the null check",
                "required_changes": ["add null check"],
                "reviewed_head_sha": "sha-old",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "unavailable" in packet.lower()
    assert not (decision_dir / "interdiff.patch").exists()


def test_review_corrupted_decision_file_has_no_prior_review_section(tmp_path: Path) -> None:
    """A corrupted review-decision.json (_review_decision returns
    {"decision": "invalid"}) must not be mistaken for round-2 findings --
    pins the "invalid" member of the exclusion tuple in review()."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text("{not valid json", encoding="utf-8")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet


def test_review_round3_surfaces_findings_from_every_prior_round(tmp_path: Path) -> None:
    """Issue #1270 (W13): a round-3 review must surface findings from EVERY
    prior round, not only the most recent one -- a finding raised in round 1
    and not repeated in round 2 must still reach the round-3 reviewer.
    Reads exclusively from the rounds/round-K archive W11 built (#1268),
    mirroring what record_review actually writes to disk (the flat mirror
    review-decision.json always equals the latest archived round)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    round1_decision = {
        "decision": "request_changes",
        "summary": "round-1 summary text",
        "required_changes": ["round-1 finding: add null check"],
        "reviewed_head_sha": "sha-r1",
    }
    round2_decision = {
        "decision": "request_changes",
        "summary": "round-2 summary text",
        "required_changes": ["round-2 finding: handle empty list"],
        "reviewed_head_sha": "sha-r2",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round2_dir = decision_dir / "rounds" / "round-2"
    round1_dir.mkdir(parents=True)
    round2_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (round2_dir / "review-decision.json").write_text(json.dumps(round2_decision), encoding="utf-8")
    # The flat mirror _review_decision reads always equals the latest
    # archived round -- record_review writes both together, same call.
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round2_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "### Round 1" in packet
    assert "### Round 2" in packet
    assert packet.index("### Round 1") < packet.index("### Round 2")
    assert "round-1 finding: add null check" in packet
    assert "round-2 finding: handle empty list" in packet
    # #1270 follow-up: a round whose required_changes list is non-empty now
    # ALSO carries its own summary (appended after the itemized list) --
    # the pre-#792 tier-1/tier-2 exclusivity is a worker-brief-only
    # tradeoff; the reviewer's round-history entry shows both.
    assert "round-1 summary text" in packet
    assert "round-2 summary text" in packet

    # The interdiff compares the LATEST round's head to the live head, not
    # round 1's -- round 1's head must not leak into the interdiff.
    interdiff_path = decision_dir / "interdiff.patch"
    assert interdiff_path.exists()
    interdiff_text = interdiff_path.read_text(encoding="utf-8")
    assert "sha-r2" in interdiff_text
    assert "sha-abc123" in interdiff_text
    assert "sha-r1" not in interdiff_text


def test_review_round_history_strips_crash_signature_findings(tmp_path: Path) -> None:
    """Issue #1270 (W13) composes with the #1269 (W12) crash-signature
    guard: a prior round whose required_changes still carries a
    pre-collector-fix crash comment (old-shape findings_channel ==
    "external") must render with the crash text absent from the
    round-history section -- the same guarantee
    _render_required_changes_section already gives the worker's rework
    brief, now also true for the reviewer's aggregated prior-review
    section. Without this, aggregating every prior round (instead of only
    the latest decision) would widen a poisoned old round's exposure
    instead of narrowing it."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    round1_decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["fix the off-by-one", crash_body],
        "findings_channel": "external",
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round1_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "fix the off-by-one" in packet
    assert crash_body not in packet
    assert REVIEW_SESSION_SUMMARY_HEADING not in packet


def test_review_same_head_terminal_verdict_surfaces_findings(tmp_path: Path) -> None:
    """Issue #632 defect 3: a terminal verdict on disk for the SAME head (a
    PR parked on agent:human-needed whose head has not advanced, or an
    operator-corrected verdict) must still surface its findings to a
    re-review. The old is_round2_review gate required
    prior_reviewed_head_sha != headRefOid, so the corrected verdict was
    invisible and re-reviewing the unchanged diff started from scratch."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    # A corrected verdict pinned to the SAME head as the live PR.
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "the null check is still missing",
                "required_changes": ["add null check in validate()", "cover empty input"],
                "reviewed_head_sha": "sha-abc123",  # == live head
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    # The prior findings reach the reviewer even though the head hasn't moved.
    assert "## Prior review" in packet
    assert "same head" in packet.lower()
    assert "the null check is still missing" in packet
    assert "add null check in validate()" in packet
    assert "cover empty input" in packet
    # No interdiff is generated for a same-head review.
    assert not (decision_dir / "interdiff.patch").exists()
    assert "No interdiff is needed" in packet


def test_review_round_history_shows_approved_round_required_changes_with_summary(
    tmp_path: Path,
) -> None:
    """Issue #1270 review round 1 (blocker): an approved archived round can
    legitimately carry a non-empty required_changes left over from an
    earlier round (rework_prompts.py's own docstring names this population
    explicitly). _render_required_changes_section returns "" for `approved`
    by design -- that verdict is out of its scope, since its findings
    belong in $prior_review_section (i.e. exactly this round-history
    section) rather than the worker's brief. Before the fix,
    _render_round_findings inherited that "" unconditionally and the caller
    printed the affirmatively false "_No findings recorded for this
    round._" even though the round's archive held a real finding."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    round1_decision = {
        "decision": "approved",
        "summary": "looks good overall, one leftover item",
        "required_changes": ["leftover from earlier: tighten the input regex"],
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round1_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "leftover from earlier: tighten the input regex" in packet
    assert "looks good overall, one leftover item" in packet
    assert "_No findings recorded for this round._" not in packet


def test_review_round_history_shows_approved_round_required_changes_without_summary(
    tmp_path: Path,
) -> None:
    """Same population as the sibling test above (an approved round with
    leftover required_changes), but with an empty summary -- the bullets
    must render on their own, not only when a summary happens to also be
    present."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    round1_decision = {
        "decision": "approved",
        "summary": "",
        "required_changes": ["leftover finding: add a test for the empty-input case"],
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round1_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "leftover finding: add a test for the empty-input case" in packet
    assert "_No findings recorded for this round._" not in packet


def test_review_round_history_shows_blocked_round_required_changes_with_summary(
    tmp_path: Path,
) -> None:
    """Issue #1270 review round 1 (blocker): a blocked archived round's
    tiers 1-2 are intentionally suppressed by
    _render_required_changes_section for the WORKER's rework brief (the
    "what must change before this PR can be approved" framing is wrong for
    the decision-agnostic routes that produce a blocked verdict). That
    suppression is specific to the worker's brief -- it must not also
    delete the finding from the REVIEWER's round-history entry, which is a
    different audience with a different, legitimate use for the same
    content."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    round1_decision = {
        "decision": "blocked",
        "summary": "blocked pending an external dependency update",
        "required_changes": ["blocked-round finding: pin the vendored SDK version"],
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round1_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "blocked-round finding: pin the vendored SDK version" in packet
    assert "blocked pending an external dependency update" in packet
    assert "_No findings recorded for this round._" not in packet


def test_review_round_history_shows_blocked_round_required_changes_without_summary(
    tmp_path: Path,
) -> None:
    """Same population as the sibling test above (a blocked round with
    required_changes), but with an empty summary -- the bullets must render
    on their own."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    round1_decision = {
        "decision": "blocked",
        "summary": "",
        "required_changes": ["blocked-round finding: rotate the leaked credential"],
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)
    (round1_dir / "review-decision.json").write_text(json.dumps(round1_decision), encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(round1_decision), encoding="utf-8"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "blocked-round finding: rotate the leaked credential" in packet
    assert "_No findings recorded for this round._" not in packet


def test_review_round_history_falls_back_when_round_dir_unreadable(tmp_path: Path) -> None:
    """Issue #1270 review round 1 (fix 2): _round_history_entries used to
    gate its fallback-to-the-flat-mirror on `not numbers` (whether any
    round-K directory existed at all) rather than `not entries` (whether
    any round-K directory's decision file was actually readable).
    workflow.py's OrchestratorApp._write_json creates the round directory
    via mkdir strictly before its atomic tmp_path.replace(), so a crash in
    that window leaves an empty round-K/ with no review-decision.json
    inside -- exactly what this test simulates by mkdir-ing round-1/
    without writing anything into it. Before the fix, that left `numbers`
    non-empty (the directory exists) but `entries` empty (nothing readable
    inside it), so the fallback never fired and the whole prior-review
    section silently vanished even though the flat mirror
    (review-decision.json) held a perfectly valid verdict."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    decision_dir = paths.prs / "pr-456"
    flat_decision = {
        "decision": "request_changes",
        "summary": "fallback summary text",
        "required_changes": ["fallback finding: fix the retry loop"],
        "reviewed_head_sha": "sha-r1",
    }
    round1_dir = decision_dir / "rounds" / "round-1"
    round1_dir.mkdir(parents=True)  # directory exists, decision file does not
    (decision_dir / "review-decision.json").write_text(json.dumps(flat_decision), encoding="utf-8")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" in packet
    assert "fallback finding: fix the retry loop" in packet


def test_review_prompt_uses_fenced_json_verdict(tmp_path: Path) -> None:
    """Issue #507: the review packet must request a fenced JSON verdict block,
    not an unexecutable CLI command, because reviewers run in read-only plan mode."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Issue #507: reviewers cannot run CLI commands in read-only plan mode, so
    # the packet must ask for a fenced JSON verdict block instead.
    assert "```json" in packet_text, "fenced JSON verdict block not found in packet"
    assert '"decision"' in packet_text
    assert '"summary"' in packet_text
    assert '"required_changes"' in packet_text
    assert "$decision_command" not in packet_text
    assert "charlie verdict" not in packet_text
