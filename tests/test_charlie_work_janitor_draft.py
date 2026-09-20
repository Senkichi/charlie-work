"""Janitor gate draft-PR handling: block without packet, #818 auto-ready actuation, merge-hold suppression, and packet warnings.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import _IssueViewCountingGitHub


def test_janitor_block_writes_no_review_packet(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue #818: draft is no longer alone here -- pair it with an empty body
    # (a second, unrelated janitor failure) so this stays a genuine
    # janitor_blocked park rather than the new pure-draft auto-ready path.
    # Empty body (not mergeable=CONFLICTING) deliberately avoids also
    # tripping the separate merge-conflict rework-routing special case.
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True, "body": ""}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # not "otherwise ready" -- never attempted
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()  # zero packet spend on a blocked PR
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"


def test_janitor_draft_only_block_auto_readies_pr(tmp_path: Path) -> None:
    """Issue #818: a draft PR that is otherwise mergeable is not a terminal
    park. When draft is the ONLY janitor failure, review() calls `gh pr
    ready` and defers the actual review to the next poll pass instead of
    writing status="janitor_blocked" -- pinning the actuation itself, not
    merely that the (pre-fix) failure string was appended.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False  # deferred to the next pass, not approved this pass
    assert result.data["draft_readied"] is True
    assert fake_gh.pr_ready_calls == [456]
    # GitHub's real effect: the PR is no longer a draft on the next fetch.
    assert fake_gh.pr_view(456)["isDraft"] is False
    # No packet spend -- the review itself is deferred, not performed now.
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()
    # Distinguishable event kind (issue #818 AC4): greppable via
    # query_events(kind=...) rather than a manual `gh pr list` sweep.
    recorded = query_events(paths.state_file, kind="draft_pr_ready_triggered")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["pr_number"] == 456
    # Issue #820 regression guard: no merge-hold anywhere means the new hold
    # check must not suppress the pre-existing auto-ready behavior, and must
    # not emit the hold-suppression event.
    assert query_events(paths.state_file, kind="draft_pr_ready_held") == []


def test_janitor_draft_only_block_gh_pr_ready_failure_stays_blocked(tmp_path: Path) -> None:
    """Issue #818 AC3: when `gh pr ready` itself fails, the PR must NOT be
    treated as ready -- the fleet must not proceed to merge on the
    assumption un-draft succeeded. Errors from external processes come back
    as values (GitHubRunResult.ok/.error), never exceptions.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    fake_gh.pr_ready_ok = False
    fake_gh.pr_ready_error = "gh: insufficient permissions to mark PR ready"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("draft_readied") is not True
    assert fake_gh.pr_ready_calls == [456]
    # GitHub-side state is unchanged: still a draft.
    assert fake_gh.pr_view(456)["isDraft"] is True
    # Parked exactly like the pre-fix behavior -- not silently advanced.
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["janitor_ok"] is False
    assert any("draft" in f.lower() for f in state["prs"]["456"]["janitor_failures"])
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()  # never routed toward review/merge
    # Distinguishable, deduped event (issue #818 AC4).
    recorded = query_events(paths.state_file, kind="draft_pr_ready_failed")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "warning"
    assert recorded[0]["payload"]["pr_number"] == 456
    assert "insufficient permissions" in recorded[0]["payload"]["error"]

    # A second pass with the same gh error does not re-fire the event
    # (cost-spirals.md dedup discipline: verdict.failures is byte-identical
    # every pass regardless of the actuator's outcome, so dedup is keyed on
    # the actuator's own error message, not on verdict.failures).
    app.review(456)
    # `gh pr ready` is attempted again -- it's the event that's deduped, not
    # the actuator call itself. Without this, a control-flow change that
    # skips the whole branch on pass 2 (e.g. an early return keyed off the
    # state just written) would still satisfy the event-count assertion
    # below, making it silently vacuous.
    assert fake_gh.pr_ready_calls == [456, 456]
    recorded_again = query_events(paths.state_file, kind="draft_pr_ready_failed")
    assert len(recorded_again) == 1


def test_janitor_draft_only_block_merge_hold_on_pr_suppresses_auto_ready(
    tmp_path: Path,
) -> None:
    """Issue #820: an operator parks a draft PR by applying the configured
    merge-hold label directly to the PR. The #818 auto-ready actuator must
    not un-draft it -- mirrors merge_ready's mergequeue-handoff hold check
    (workflow.py's merge_hold pattern) rather than inventing a variant.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _IssueViewCountingGitHub()
    fake_gh.prs[0] = {
        **fake_gh.prs[0],
        "isDraft": True,
        "labels": [{"name": config.labels.merge_hold}],
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # never attempted -- operator hold wins
    # The PR-label hold is resolved from `pr` alone, with no additional
    # issue_view call beyond review()'s own unconditional per-pass fetch
    # (used for packet building) -- the hold check must not double-fetch.
    assert fake_gh.issue_view_calls == [123]
    assert fake_gh.pr_view(456)["isDraft"] is True  # still parked as a draft
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "operator_merge_hold"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["pr_number"] == 456
    assert recorded[0]["payload"]["reason"] == "operator_merge_hold"
    assert recorded[0]["level"] == "warning"


def test_janitor_draft_only_block_merge_hold_on_issue_suppresses_auto_ready(
    tmp_path: Path,
) -> None:
    """Issue #820: the merge-hold label on the linked issue is an equally
    valid operator park signal (matching merge_ready's issue-side check)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.merge_hold},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []
    assert fake_gh.pr_view(456)["isDraft"] is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "operator_merge_hold"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["reason"] == "operator_merge_hold"


@pytest.mark.parametrize("degraded_payload", [{}, {"number": 123}])
def test_janitor_draft_only_block_merge_hold_check_unavailable_fails_safe(
    tmp_path: Path, degraded_payload: dict[str, Any]
) -> None:
    """Issue #820 fail-safe: when the linked issue's payload can't yield a
    usable "labels" set (empty dict, or a dict missing the "labels" key),
    auto-ready must be suppressed exactly like a confirmed hold -- never
    treated as "no hold" just because the check couldn't be evaluated.
    Mirrors merge_ready's merge_hold_check_unavailable degraded-payload arm
    (test_merge_ready_mergequeue_hold_issue_degraded_payload_fails_closed)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class DegradedIssueViewGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                return degraded_payload
            return super().issue_view(number)

    fake_gh = DegradedIssueViewGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # fail safe: do not act
    assert fake_gh.pr_view(456)["isDraft"] is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "merge_hold_check_unavailable"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["reason"] == "merge_hold_check_unavailable"


def test_janitor_draft_only_block_issue_view_raise_never_calls_pr_ready(
    tmp_path: Path,
) -> None:
    """Issue #820 fail-safe, raising arm: review() fetches the linked issue
    unconditionally near the top of the method (for packet building), before
    the janitor verdict is even computed. The merge-hold check added for
    #820 reuses that fetch rather than issuing its own redundant call (see
    the comment at the `is_draft_only_block` branch), so when the fetch
    itself raises, review() aborts before verdict computation and `gh pr
    ready` is never reached -- the pre-existing per-PR GitHubError handling
    in loop() (not a new in-branch code path) is what makes this fail safe.
    """
    from charlie_work.github import GitHubError as _GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class RaisingIssueViewGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                raise _GitHubError("transient gh issue view failure")
            return super().issue_view(number)

    fake_gh = RaisingIssueViewGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    with pytest.raises(_GitHubError):
        app.review(456)

    assert fake_gh.pr_ready_calls == []
    assert fake_gh.pr_view(456)["isDraft"] is True
    # No state was written -- the failure happened before any state_lock
    # section in review() runs for this pass.
    state = load_state(paths.state_file)
    assert "456" not in state.get("prs", {})


def test_janitor_draft_only_block_merge_hold_event_dedupes_and_refires_after_lift(
    tmp_path: Path,
) -> None:
    """Issue #820: the suppression event must dedupe across identical-hold
    passes (cost-spirals.md discipline) but must re-fire if the hold is
    lifted (the PR gets auto-readied) and the operator re-parks it later --
    a stale dedup marker must not silently swallow the second park."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    hold_label = config.labels.merge_hold
    fake_gh.prs[0] = {
        **fake_gh.prs[0],
        "isDraft": True,
        "labels": [{"name": hold_label}],
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    first = app.review(456)
    assert first.ok is False
    assert fake_gh.pr_ready_calls == []
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Same hold, second pass: dedup -- no new event.
    second = app.review(456)
    assert second.ok is False
    assert fake_gh.pr_ready_calls == []
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Operator lifts the hold: the PR is auto-readied as normal.
    fake_gh.prs[0]["labels"] = []
    lifted = app.review(456)
    assert lifted.data.get("draft_readied") is True
    assert fake_gh.pr_ready_calls == [456]
    assert fake_gh.pr_view(456)["isDraft"] is False
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Operator re-parks: re-drafts and re-applies the hold label. This must
    # emit a FRESH suppression event, not stay deduped against the first
    # park's stale reason.
    fake_gh.prs[0]["isDraft"] = True
    fake_gh.prs[0]["labels"] = [{"name": hold_label}]
    reparked = app.review(456)
    assert reparked.ok is False
    assert fake_gh.pr_ready_calls == [456]  # not called again
    held_events = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(held_events) == 2
    assert all(e["payload"]["pr_number"] == 456 for e in held_events)

    # A DIFFERENT route out of the held state: the operator bypasses our
    # actuator entirely -- manually un-drafts via the GitHub UI *and* lifts
    # the hold in the same step -- landing on the packet-success path
    # (verdict.ok True) rather than the `gh pr ready` branch exercised
    # above. This path never re-enters `is_draft_only_block`, so it only
    # proves the fix if the reason was reconciled to None unconditionally
    # rather than by a clear buried inside that branch.
    fake_gh.prs[0]["isDraft"] = False
    fake_gh.prs[0]["labels"] = []
    bypassed = app.review(456)
    assert bypassed.ok is True
    state_after_bypass = load_state(paths.state_file)
    assert state_after_bypass["prs"]["456"]["draft_ready_held_reason"] is None

    # Re-park a second time. If the packet-success pass above had left the
    # reason stale, this would wrongly dedupe against the pass-4 reason and
    # no third event would fire.
    fake_gh.prs[0]["isDraft"] = True
    fake_gh.prs[0]["labels"] = [{"name": hold_label}]
    reparked_again = app.review(456)
    assert reparked_again.ok is False
    assert fake_gh.pr_ready_calls == [456]  # still not called again
    held_events_final = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(held_events_final) == 3
    assert all(e["payload"]["pr_number"] == 456 for e in held_events_final)


def test_janitor_warnings_surface_in_review_packet(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "additions": 2000, "deletions": 10}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert "Janitor warnings" in packet.read_text(encoding="utf-8")
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["janitor_ok"] is True
    assert state["prs"]["456"]["janitor_warnings"]
