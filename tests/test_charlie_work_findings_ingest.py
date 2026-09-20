"""External-finding ingestion from PR comments: orchestrator-own filtering, quote replies, timing gaps.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    OrchestratorConfig,
    ReviewConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.verdict_parsing import REVIEW_SESSION_SUMMARY_HEADING
from charlie_work.workflow import (
    ORCHESTRATOR_COMMENT_MARKER,
    OrchestratorApp,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_orchestrator_own_comment_is_not_reingested_as_external_finding(
    tmp_path: Path,
) -> None:
    """Issue #950 follow-up: the orchestrator's own PR comments must not come
    back as "external findings".

    The orchestrator authenticates with a *user* token -- ``gh api user``
    reports ``type=User`` -- so ``_is_bot_comment`` cannot see its output, and
    filtering by login would drop the genuine human findings this feature
    exists to ingest. Provenance therefore travels in the body via
    ``ORCHESTRATOR_COMMENT_MARKER``.

    This is deliberately a round trip rather than two assertions against a
    hardcoded marker string: the body is produced by the real ``_comment_pr``
    write path and then filtered by the real collection path, so the writer and
    the reader cannot drift apart without this failing.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Produce a comment body through the real posting path.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    app._comment_pr(456, "## Request changes\n\nThe retry wrapper swallows the type.")
    posted_body = (pr_dir / "review-comment.md").read_text(encoding="utf-8")
    assert "Request changes" in posted_body, "sanity: the summary survived stamping"

    # Feed it back as GitHub reports it: authored by a User, not a Bot.
    fake_gh.pr_external_issue_comments[456] = [
        {"body": posted_body, "user": {"login": "orchestrator-operator", "type": "User"}},
        {
            "body": "The migration needs a rollback path before this can land.",
            "user": {"login": "a-real-human", "type": "User"},
        },
    ]

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field.
    assert decision["required_changes"] == ["keep me"]
    external = decision["external_findings"]

    # The orchestrator's own echo is filtered out...
    assert not any("retry wrapper swallows the type" in item for item in external)
    # ...while a genuine human finding from an identical account type is kept.
    assert any("rollback path" in item for item in external)


def test_human_quote_reply_to_orchestrator_comment_is_still_ingested(
    tmp_path: Path,
) -> None:
    """The provenance marker must be matched as a *prefix*, not a substring.

    GitHub's "Quote reply" copies the raw markdown of the quoted comment --
    HTML comments included -- into a blockquote above the reply. Quote-replying
    to one of our ``request_changes`` comments is a natural way for a human to
    answer point by point, and it produces a body that *contains*
    ``ORCHESTRATOR_COMMENT_MARKER`` without starting with it.

    Under a substring test that comment is classified as ours and dropped, so
    the human's new finding below the quote never reaches ``required_changes``.
    Losing a genuine finding is the expensive direction of this filter, so the
    predicate must fail toward ingestion. This test fails if the check is
    weakened back to ``MARKER in body``.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    app._comment_pr(456, "## Request changes\n\nThe retry wrapper swallows the type.")
    posted_body = (pr_dir / "review-comment.md").read_text(encoding="utf-8")

    # Exactly what GitHub stores when a human uses "Quote reply": every line of
    # the quoted comment prefixed with "> ", then the human's own text.
    quoted = "\n".join(f"> {line}" for line in posted_body.splitlines())
    human_reply = f"{quoted}\n\nAgreed, and separately: the migration needs a rollback path."
    assert ORCHESTRATOR_COMMENT_MARKER in human_reply, (
        "sanity: the quote really does carry the marker -- otherwise this test "
        "would pass without exercising the prefix-vs-substring distinction"
    )
    assert not human_reply.startswith(ORCHESTRATOR_COMMENT_MARKER)

    fake_gh.pr_external_issue_comments[456] = [
        {"body": posted_body, "user": {"login": "orchestrator-operator", "type": "User"}},
        {"body": human_reply, "user": {"login": "a-real-human", "type": "User"}},
    ]

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field.
    assert decision["required_changes"] == ["keep me"]
    external = decision["external_findings"]

    # The human's finding survives even though their comment embeds our marker.
    assert any("rollback path" in item for item in external)
    # Our own unquoted comment is still filtered out.
    assert not any(item.lstrip().startswith(ORCHESTRATOR_COMMENT_MARKER) for item in external)


def test_worker_rework_reply_is_not_ingested_as_external_finding(
    tmp_path: Path,
) -> None:
    """Issue #998: a worker's own rework reply is machine-generated, posted
    through the worker's path (no ``ORCHESTRATOR_COMMENT_MARKER``), and posted
    *after* the rework commit it describes. It must not come back as a
    "required change" on the next ``request_changes`` verdict -- that would
    tell the worker to address its own completion report.

    The cutoff is temporal, not identity-based: the worker posts through a
    user token (same account / ``type=User`` as the human whose findings #950
    exists to capture), so the upper bound is the ``reviewed_head_sha``'s
    committer date. A genuine human comment from the *same* account and same
    ``type=User``, posted *before* the reviewed head, is still ingested --
    this is the regression any identity-based shortcut would cause, asserted
    positively rather than assumed.

    Mutation check: disabling the ``before`` upper bound (reverting
    ``_collect_external_findings`` to its merge-base form) makes this test
    fail, because the worker reply is then ingested alongside the human
    finding.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # The rework commit the reviewer is about to read. Its committer date is
    # the ingestion upper bound. Set it up as the live PR head with full
    # commit metadata so _commit_timestamp can resolve it.
    rework_sha = "rework-head-sha"
    fake_gh.pr_head_shas[456] = rework_sha
    fake_gh.commits[rework_sha] = {
        "parents": [{"sha": "base-sha"}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": "2026-08-10T10:00:00Z",
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": "2026-08-10T10:00:00Z",
            },
        },
    }

    # A genuine human finding from the SAME account and SAME type=User,
    # posted *before* the reviewed head commit -- must still be ingested.
    human_finding = "The migration script drops the index without a guard."
    # The worker pushed the rework at 10:00, then posted its completion reply
    # at 10:05 -- after the head commit, so outside the ingestion window. This
    # is exactly the real-world shape from PR #972's comment thread.
    worker_reply = (
        "Reworked in rework-head-sha. Summary of the changes addressing each "
        "point: added the missing rollback path and a regression test."
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": human_finding,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        },
        {
            "body": worker_reply,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-10T10:05:00Z",
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
    # Issue #999: external findings ride in their own field, not required_changes.
    changes = decision["required_changes"]
    external = decision.get("external_findings", [])

    # The worker's own rework reply is NOT fed back as an external finding.
    assert not any("Reworked in rework-head-sha" in item for item in external), (
        "worker rework reply must not be ingested as an external finding"
    )
    # The genuine human finding from the same account/type=User IS ingested
    # into the external_findings field.
    assert any("migration script drops the index" in item for item in external), (
        "genuine human comment before the reviewed head must still be ingested"
    )
    # The internal finding survives untouched in required_changes.
    assert "keep me" in changes


def test_human_comment_in_before_to_reviewed_at_gap_surfaces_next_round(
    tmp_path: Path,
) -> None:
    """Issue #998 rework: a genuine human comment posted in the gap
    ``(before, reviewed_at]`` -- after the reviewed head commit landed but
    before the verdict was written -- is excluded by ``before`` this round
    and MUST surface as a required_change in the following round.

    The per-round ingestion windows must be contiguous: the next round's
    ``since`` is this round's persisted ``before`` (not its ``reviewed_at``).
    Deriving ``since`` from ``reviewed_at`` instead would drop a gap comment
    forever -- it satisfies ``item_dt <= reviewed_at``, so the lower bound
    skips it in every subsequent round -- silently violating the
    fail-toward-ingestion invariant.

    Mutation check: reverting the ``since`` derivation in ``record_review``
    to ``previous_decision.get("reviewed_at")`` (the merge-base form, without
    the ``before`` fallback) makes this test fail, because the gap comment is
    then dropped by ``since`` in round 2 and never surfaces.
    """
    # max_rework_cycles bumped past 2 so the second request_changes round does
    # not escalate -- escalation does not short-circuit required_changes
    # persistence (the ingestion block runs before the escalation check), but
    # keeping the verdict non-escalated makes the assertion target unambiguous.
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    base = datetime.now(UTC)
    # Round-1 head commit landed 2 hours ago -- well before the verdict write.
    round1_commit_dt = base - timedelta(hours=2)
    # A genuine human finding posted 1 hour ago: strictly AFTER the round-1
    # head commit (so ``before`` excludes it in round 1) and strictly BEFORE
    # round-1's reviewed_at (utc_now() during round 1's record_review, i.e.
    # ~base). This is the (before, reviewed_at] gap that the discontinuity
    # silently dropped.
    gap_comment_dt = base - timedelta(hours=1)
    gap_finding = "Gap comment: the rollback path leaks a file handle on early return."

    round1_sha = "round1-head-sha"
    fake_gh.pr_head_shas[456] = round1_sha
    fake_gh.commits[round1_sha] = {
        "parents": [{"sha": "base-sha"}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": round1_commit_dt.isoformat(),
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": round1_commit_dt.isoformat(),
            },
        },
    }
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": gap_finding,
            "user": {"login": "operator", "type": "User"},
            "created_at": gap_comment_dt.isoformat(),
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Round 1: request_changes. The gap comment is after the round-1 head
    # commit, so ``before`` excludes it this round -- it must NOT appear yet.
    r1 = app.record_review(
        456,
        "request_changes",
        summary="round 1",
        required_changes=["internal-1"],
        verdict_provenance="fresh_llm_review",
    )
    assert r1.ok is True
    d1 = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8"))
    assert not any("Gap comment" in c for c in d1["required_changes"]), (
        "gap comment must be excluded by `before` in round 1 "
        "(it is strictly after the round-1 head commit)"
    )
    # The contiguity fix persists ``before`` so round 2 can derive ``since``
    # from it. This is the load-bearing persistence the next round reads back.
    assert d1.get("before") == round1_commit_dt.isoformat(), (
        "round-1 decision must persist the `before` upper bound for round-2 contiguity"
    )

    # Round 2: the worker pushed a new head. Its commit lands ~now (after the
    # gap comment), so ``before_2`` does not exclude the gap comment; and
    # ``since_2`` = round-1's persisted ``before`` = round1_commit_dt, which is
    # before the gap comment, so the lower bound does not exclude it either.
    round2_commit_dt = base
    round2_sha = "round2-head-sha"
    fake_gh.pr_head_shas[456] = round2_sha
    fake_gh.commits[round2_sha] = {
        "parents": [{"sha": round1_sha}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": round2_commit_dt.isoformat(),
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": round2_commit_dt.isoformat(),
            },
        },
    }

    r2 = app.record_review(
        456,
        "request_changes",
        summary="round 2",
        required_changes=["internal-2"],
        verdict_provenance="fresh_llm_review",
    )
    assert r2.ok is True
    d2 = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field, not required_changes.
    changes2 = d2["required_changes"]
    external2 = d2.get("external_findings", [])

    # THE regression assertion: the gap comment surfaces in round 2 rather
    # than being permanently dropped.
    assert any("Gap comment" in c for c in external2), (
        "human comment in the (before, reviewed_at] gap must surface in the next "
        "round, not be silently dropped forever"
    )
    # The round-2 internal finding survives alongside it.
    assert "internal-2" in changes2


def test_human_quote_reply_to_a_crash_summary_is_still_ingested(tmp_path: Path) -> None:
    """A genuine human reply that GitHub-quotes a crash summary (to discuss
    or dispute it) is preserved -- mirrors
    test_human_quote_reply_to_orchestrator_comment_is_still_ingested's
    rationale for ORCHESTRATOR_COMMENT_MARKER, applied to the crash-heading
    prefix check instead. A substring match would wrongly discard this
    reply along with the quoted heading; the prefix check must not."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    quoted_reply = (
        f"> {REVIEW_SESSION_SUMMARY_HEADING}\n"
        "> \n"
        "> The automated reviewer ran for 4 turns...\n"
        "\n"
        "This looks like a session crash, not a real review -- can we re-run it?"
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": quoted_reply,
            "user": {"login": "a-real-human", "type": "User"},
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
    external = decision.get("external_findings", [])
    assert any("can we re-run it" in item for item in external), (
        "a genuine human reply quoting a crash summary must still be ingested"
    )
