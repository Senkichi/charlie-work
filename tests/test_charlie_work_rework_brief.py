"""Rework-brief rendering: verdict-driven required-changes sections.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_rework_brief_*`` seam -- required-changes extraction, dispatch-note
plus findings composition, approved-verdict omission, summary fallback, and
finding/bullet ordering. Brief *regeneration* lives in
``tests/test_charlie_work_dispatch_rework_briefs.py``; shared fakes and
helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _normalize_ws,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def test_rework_brief_contains_required_changes_from_verdict(tmp_path: Path) -> None:
    """Issue #632 defect 1: a request_changes verdict's required_changes must
    reach the rework brief. The brief reads review-decision.json itself (single
    point of enforcement), so asserting on the rendered content — not on the
    call having happened — is the right check."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    findings = ["add null check in parse()", "update tests for edge case"]
    app.record_review(
        456,
        "request_changes",
        summary="fix A",
        required_changes=findings,
        verdict_provenance="fresh_llm_review",
    )

    brief = (paths.prs / "pr-456" / "rework-prompt.md").read_text(encoding="utf-8")
    for finding in findings:
        assert finding in brief, f"finding missing from brief: {finding!r}"
    # The prose summary rides in the dispatch-note slot and must still appear.
    assert "fix A" in brief
    # The section header is present so the worker can locate the list.
    assert "## Required changes" in brief


def test_rework_brief_keeps_both_dispatch_note_and_findings(tmp_path: Path) -> None:
    """Issue #632 defect 2: the no-op-churn / merge-conflict routing path
    passes an operational note into the brief. That note must accompany the
    findings instead of displacing them — the #510 regression shipped a brief
    with the churn message and zero findings."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    # A pre-existing verdict with structured findings on disk.
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary",
                "required_changes": ["fix the off-by-one", "add a regression test"],
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
    }
    operational_note = (
        "The previous rework cycle produced no actual content change. Push the real fix."
    )
    app._write_rework_prompt(pr, 123, operational_note)

    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    # The operational note is preserved...
    assert "no actual content change" in brief
    # ...and the findings are still present alongside it.
    assert "fix the off-by-one" in brief
    assert "add a regression test" in brief
    # The sidecar note is written for dispatch-time regeneration.
    assert (pr_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8") == operational_note


def test_rework_brief_omits_required_changes_for_approved_verdict(tmp_path: Path) -> None:
    """Issue #632 edge case: an approved verdict may carry a non-empty
    required_changes list (the field is optional for every decision). The
    merge-conflict / no-CI / cross-pr-revert routes render a rework brief
    for an already-approved PR whose dispatch note explicitly says "do not
    re-litigate the review". Rendering a "Required changes ... before this
    PR can be approved" section into that brief would be contradictory.
    The section must only appear for a request_changes verdict."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    # An approved verdict that nonetheless carries required_changes (a
    # reviewer mistake, or stale carry-forward from a prior round).
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "summary": "lgtm",
                "required_changes": ["add a docstring", "rename foo to bar"],
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
    }
    # The merge-conflict operational note says "do not re-litigate".
    operational_note = (
        "The PR branch has a merge conflict. Merge the base branch and resolve. "
        "The code changes are already approved; do not re-litigate the review."
    )
    app._write_rework_prompt(pr, 123, operational_note)

    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    # The operational note is present.
    assert "do not re-litigate" in brief
    assert "merge conflict" in brief
    # The contradictory required-changes section is NOT rendered.
    assert "## Required changes" not in brief
    assert "before this PR can be approved" not in brief
    assert "add a docstring" not in brief
    assert "rename foo to bar" not in brief


def test_rework_brief_falls_back_to_summary_when_required_changes_empty(
    tmp_path: Path,
) -> None:
    """Bug reproducer (F1): measured across the on-disk corpus, request_changes
    verdicts with a populated required_changes are 0 of 20 -- prompts/review.md
    historically documented the field as optional, so reviewers reliably fill
    in `summary` and skip `required_changes`. Before F1,
    _render_required_changes_section returned "" whenever required_changes
    was empty, so the worker's rework brief contained none of that summary
    content even though it was real, substantive review findings sitting
    right there in review-decision.json. Must fail before F1, pass after.

    The reviewer_summary text here is the real PR #766 reproduction string
    from issue #781 -- it contains a live closing keyword ("does not fix
    #649") that must reach the brief DEFANGED (issue #781 outbound fix), so
    this test also doubles as an AC4-adjacent regression guard: the summary
    fallback tier must not leak `fix #649` verbatim into a worker's brief."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    reviewer_summary = (
        "BLOCKER - does not fix #649. The pin is a no-op over uv's resolver; "
        "the underlying version conflict is still unresolved."
    )
    app.record_review(
        456,
        "request_changes",
        summary=reviewer_summary,
        required_changes=[],
        verdict_provenance="fresh_llm_review",
    )

    brief = (paths.prs / "pr-456" / "rework-prompt.md").read_text(encoding="utf-8")
    assert "## Required changes" in brief
    # Content (the reviewer's findings) survives, defanged of the live
    # closing-keyword syntax that would otherwise let GitHub or a worker's
    # own PR body auto-close/mis-bind issue #649.
    assert "does not fix issue 649" in brief, "reviewer summary missing from brief"
    assert "does not fix #649" not in brief, "live closing keyword leaked into brief"


def test_rework_brief_orders_findings_before_demoted_base_merge_bullet(
    tmp_path: Path,
) -> None:
    """F4 (docs/plans/rework-findings-channel.md, #6): the base-merge bullet
    used to be the unconditional, first bullet of `## Required behavior`. When
    a verdict rendered zero findings, that made "merge the base branch" the
    only concrete instruction in the brief -- workers complied, the head
    advanced by a merge commit only, and the janitor correctly flagged the
    result a no-op. The findings-action bullet must now open the section, and
    the base-merge bullet must be demoted (present, but neither first nor
    unconditional)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary",
                "required_changes": ["fix the off-by-one", "add a regression test"],
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
    }
    app._write_rework_prompt(pr, 123, "")

    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    # The findings render well above `## Required behavior` at all.
    assert brief.index("fix the off-by-one") < brief.index("## Required behavior")

    behavior = brief.split("## Required behavior", 1)[1].split("\n## ", 1)[0]
    bullets = [_normalize_ws(b) for b in behavior.split("\n- ") if b.strip()]
    findings_bullet_idx = next(i for i, b in enumerate(bullets) if "the review above" in b)
    merge_bullet_idx = next(i for i, b in enumerate(bullets) if "merge the PR's base branch" in b)

    # The findings-action bullet opens the section; the base-merge bullet
    # comes strictly later -- reordered, not just co-present.
    assert findings_bullet_idx == 0
    assert findings_bullet_idx < merge_bullet_idx

    norm_behavior = _normalize_ws(behavior)
    # Demoted to a self-conditional trigger, not the mandated opening action.
    assert "If your branch is behind its base or the PR shows a merge conflict" in norm_behavior
    assert "First, merge the PR's base branch" not in norm_behavior


def test_rework_brief_omits_unconditional_address_findings_sentence_without_findings(
    tmp_path: Path,
) -> None:
    """F4: the brief used to assert 'Address every Critical and Important
    finding directly' unconditionally, instructing the worker to act on an
    empty set whenever the verdict rendered zero findings. That sentence must
    never render, in favor of self-conditional wording that references "the
    findings above" without asserting they exist. Uses an approved verdict
    (not request_changes with an empty required_changes list) so this keeps
    exercising the empty-findings path even after F1's summary fallback lands
    for request_changes verdicts."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "summary": "lgtm"}),
        encoding="utf-8",
    )
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
    }
    operational_note = (
        "The PR branch has a merge conflict. Merge the base branch and resolve. "
        "The code changes are already approved; do not re-litigate the review."
    )
    app._write_rework_prompt(pr, 123, operational_note)

    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    norm_brief = _normalize_ws(brief)
    assert "## Required changes" not in brief
    assert "Address every Critical and Important finding directly" not in norm_brief
    # The demoted base-merge bullet is still present for the route that
    # genuinely needs it (the merge-conflict rework path) -- demoted, not
    # deleted.
    assert "If your branch is behind its base or the PR shows a merge conflict" in norm_brief
