"""Tests for issue #1268 (W11), GitHub PR comment reformat + config toggle.

Third of three W11 sub-items -- see ``tests/test_review_round_archive.py``
(the round-numbered archive) and ``tests/test_review_event_payload.py``
(the events.db payload) for the overall issue context. This file covers:

  AC4 -- ``record_review``'s comment gate, previously
         ``decision == "request_changes" and comment and summary_text``
         (request_changes-only, silent unless a caller explicitly passed
         ``comment=True``), now fires for every terminal decision this call
         actually recorded (approved / request_changes / blocked), posting a
         "## Fleet review - round K - <decision>" header (K matching that
         round's own archive number -- see test_review_round_archive.py)
         followed by the summary and required-changes text. Still excludes
         an in-call ``escalated`` verdict: the rescue tier and the
         rework-cap escalation path already post their own comment for that
         case (see workflow.py's ``_process_rescue_review`` and the
         rework-cap block inside ``record_review`` itself), so the new gate
         must produce ZERO additional comment calls when this call's own
         verdict escalates. The test drives that via the real
         ``request_changes_count >= max_rework_cycles`` cap (rescue
         disabled, the default), never a literal ``escalated=True`` kwarg
         (``record_review`` has no such parameter) and never a
         ``decision == "escalated"`` branch (``escalated`` is a bool paired
         with a request_changes decision, not a decision value of its own).
         Also covers issue #792's ``findings_channel == "derived"`` shape
         (the dominant production case, per record_review's own #792
         comment): when no structured ``required_changes`` list is
         supplied, ``effective_required_changes`` becomes a verbatim copy
         of ``summary_text``, and the gate must render that text once, not
         twice.
  AC5 -- ``ReviewConfig.post_verdict_comment`` defaults to True; False
         suppresses the automatic comment while the existing ``comment=True``
         force-on override (the CLI's ``--comment`` flag, ``cli.py``) still
         posts regardless of the config value -- the OR gate is a superset,
         not a replacement.

Round archiving (first sub-item) and the events.db payload (second
sub-item) are already covered elsewhere and are not re-asserted here beyond
what's needed to pick the expected round number out of the comment header.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import ORCHESTRATOR_COMMENT_MARKER, OrchestratorApp

from _fakes_github import FakeGitHub
from _review_fixtures import _PR_NUMBER


def _body_after_marker(body: str) -> str:
    """`_comment_pr` (workflow.py) unconditionally prepends
    ``ORCHESTRATOR_COMMENT_MARKER`` + a newline ahead of whatever body this
    gate builds -- that stamping is a pre-existing, deliberately
    unconditional invariant (issue #950's external-findings ingestion keys
    off it) and out of scope for this item, so tests strip it before
    asserting on the "## Fleet review - round K - <decision>" header this
    item actually adds."""
    assert body.startswith(ORCHESTRATOR_COMMENT_MARKER + "\n")
    return body[len(ORCHESTRATOR_COMMENT_MARKER) + 1 :]


class _CapturingGitHub(FakeGitHub):
    """FakeGitHub subclass that records every ``pr_comment`` call's body,
    mirroring the existing capture pattern at test_charlie_work.py's own
    dispatch_reviews comment tests (and test_feat_rescue_tier.py's rescue
    escalation-comment tests) rather than inventing a new one."""

    def __init__(self) -> None:
        super().__init__()
        self.captured_comments: list[tuple[int, str]] = []

    def pr_comment(self, number: int, body_file: Path) -> None:
        self.captured_comments.append((number, body_file.read_text(encoding="utf-8")))


def _app_with_config(tmp_path: Path, config: OrchestratorConfig) -> tuple[OrchestratorApp, Any]:
    """Same minimal-fixture shape as test_review_round_archive._round_archive_app,
    parameterized on config (needed for AC5's post_verdict_comment toggle) and
    wired to a _CapturingGitHub instead of the base no-op FakeGitHub."""
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = _CapturingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _record_decision(
    app: OrchestratorApp,
    fake_gh: _CapturingGitHub,
    *,
    decision: str,
    head: str,
    summary: str,
    required_changes: list[str] | None = None,
    comment: bool = False,
):
    fake_gh.pr_head_shas[_PR_NUMBER] = head
    result = app.record_review(
        _PR_NUMBER,
        decision,
        summary=summary,
        required_changes=required_changes or [],
        comment=comment,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True, result.message
    return result


# ---------------------------------------------------------------------------
# AC4 -- comment gate fires per terminal decision, with the round-K header,
# and produces zero additional calls when this call's own verdict escalates.
# ---------------------------------------------------------------------------


def test_ac4_two_rounds_each_produce_one_comment_with_matching_round_header(
    tmp_path: Path,
) -> None:
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-r1",
        summary="round one summary",
        required_changes=["round one change"],
    )
    assert len(fake_gh.captured_comments) == 1
    number1, body1 = fake_gh.captured_comments[0]
    assert number1 == _PR_NUMBER
    assert _body_after_marker(body1).startswith("## Fleet review - round 1 - request_changes")
    assert "round one summary" in body1
    assert "round one change" in body1

    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-r2",
        summary="round two summary",
        required_changes=["round two change"],
    )
    assert len(fake_gh.captured_comments) == 2
    number2, body2 = fake_gh.captured_comments[1]
    assert number2 == _PR_NUMBER
    assert _body_after_marker(body2).startswith("## Fleet review - round 2 - request_changes")
    assert "round two summary" in body2
    assert "round two change" in body2
    # round-1's own comment is untouched by round-2's call.
    assert fake_gh.captured_comments[0] == (number1, body1)


def test_ac4_escalated_request_changes_produces_zero_new_comment_calls(
    tmp_path: Path,
) -> None:
    """max_rework_cycles=2 (default), rescue disabled (default): the third
    request_changes verdict on a third consecutive advanced head pushes
    request_changes_count to the cap and escalates -- driven through the
    real cap logic, never a literal escalated=True kwarg (record_review has
    none) and never branched on decision=="escalated" (escalated is a bool
    paired with request_changes, not a decision value)."""
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-1",
        summary="s1",
        required_changes=["c1"],
    )
    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-2",
        summary="s2",
        required_changes=["c2"],
    )
    assert len(fake_gh.captured_comments) == 2, "both below-cap rounds must comment"

    result3 = _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-3",
        summary="s3",
        required_changes=["c3"],
    )
    assert result3.data["escalated"] is True, "third advanced-head verdict must escalate"
    assert len(fake_gh.captured_comments) == 2, (
        "an escalated verdict must produce zero NEW comment calls from this "
        "gate -- the rework-cap/rescue paths already comment on escalation "
        "elsewhere, so this gate double-posting would duplicate that notice"
    )


def test_ac4_approved_decision_produces_exactly_one_comment_call(tmp_path: Path) -> None:
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    _record_decision(
        app,
        fake_gh,
        decision="approved",
        head="sha-approved",
        summary="looks good, shipping it",
    )

    assert len(fake_gh.captured_comments) == 1
    number, body = fake_gh.captured_comments[0]
    assert number == _PR_NUMBER
    assert _body_after_marker(body).startswith("## Fleet review - round 1 - approved")
    assert "looks good, shipping it" in body


def test_ac4_derived_required_changes_renders_summary_once_not_twice(
    tmp_path: Path,
) -> None:
    """Regression guard for issue #792's `findings_channel == "derived"`
    path -- the dominant production shape, since `required_changes` has a
    near-0% fill rate (record_review's own #792 comment). With no
    structured list supplied, record_review derives
    `effective_required_changes = [summary_text.strip()]`: the exact same
    text as the summary paragraph already posted above it. The gate must
    render that text exactly once (as prose), never also as a redundant
    one-item "### Required changes" bullet echoing the whole paragraph --
    mirroring `_render_required_changes_section`'s own tier-2 handling of
    the same marker."""
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    summary = "The retry loop never bounds its backoff; add a max-attempts cap."
    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-derived",
        summary=summary,
        required_changes=[],
    )

    assert len(fake_gh.captured_comments) == 1
    body = _body_after_marker(fake_gh.captured_comments[0][1])
    assert body.startswith("## Fleet review - round 1 - request_changes")
    assert body.count(summary) == 1, "derived summary text must appear exactly once"
    assert "### Required changes" not in body


def test_ac4_blocked_decision_produces_exactly_one_comment_call(tmp_path: Path) -> None:
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    _record_decision(
        app,
        fake_gh,
        decision="blocked",
        head="sha-blocked",
        summary="security concern, do not merge",
        required_changes=["explain the auth bypass"],
    )

    assert len(fake_gh.captured_comments) == 1
    number, body = fake_gh.captured_comments[0]
    assert number == _PR_NUMBER
    assert _body_after_marker(body).startswith("## Fleet review - round 1 - blocked")
    assert "security concern, do not merge" in body
    assert "explain the auth bypass" in body


# ---------------------------------------------------------------------------
# AC5 -- config toggle, and the CLI --comment force-on override layered on
# top of it (not replaced by it).
# ---------------------------------------------------------------------------


def test_review_config_post_verdict_comment_defaults_true() -> None:
    assert ReviewConfig().post_verdict_comment is True


def test_ac5_config_false_suppresses_automatic_comment(tmp_path: Path) -> None:
    config = OrchestratorConfig(review=ReviewConfig(post_verdict_comment=False))
    app, fake_gh = _app_with_config(tmp_path, config)

    _record_decision(
        app,
        fake_gh,
        decision="approved",
        head="sha-quiet",
        summary="quiet approval",
    )

    assert fake_gh.captured_comments == [], (
        "post_verdict_comment=False must suppress the automatic comment "
        "when the caller did not force it on"
    )


def test_ac5_comment_force_on_still_posts_when_config_disabled(tmp_path: Path) -> None:
    """The CLI's --comment flag (cli.py, comment=args.comment) must remain a
    strict superset of the config default, not be subsumed by it: even with
    post_verdict_comment=False, an explicit comment=True still posts."""
    config = OrchestratorConfig(review=ReviewConfig(post_verdict_comment=False))
    app, fake_gh = _app_with_config(tmp_path, config)

    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-forced",
        summary="forced comment",
        required_changes=["fix it"],
        comment=True,
    )

    assert len(fake_gh.captured_comments) == 1
    assert _body_after_marker(fake_gh.captured_comments[0][1]).startswith(
        "## Fleet review - round 1 - request_changes"
    )


def test_ac5_existing_comment_true_path_still_works_with_default_config(
    tmp_path: Path,
) -> None:
    """Regression guard for the pre-existing `charlie verdict --comment`
    request_changes path: with the new True default, comment=True and the
    config default together must still produce exactly one comment (an OR
    gate, not an additive one -- no double-post)."""
    app, fake_gh = _app_with_config(tmp_path, OrchestratorConfig())

    _record_decision(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-both",
        summary="both signals on",
        required_changes=["still one comment"],
        comment=True,
    )

    assert len(fake_gh.captured_comments) == 1
