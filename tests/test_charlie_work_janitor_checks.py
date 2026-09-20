"""Janitor gate required-check verdicts: failure routing to rework, escalation on repeats, stale-packet handling, and co-occurring failure classification.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _fail_if_launched
from _fakes_github import FakeGitHubWithChecks
from _review_fixtures import _fake_claude_worker_record, _required_checks_config
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import ReviewDispatchConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHubWithRerunCapture


def test_janitor_required_check_failure_routes_to_rework(tmp_path: Path) -> None:
    """Issue #376: a definitive required-check failure on a linked issue routes to rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "CI failed on Tests passed; push a fix" in prompt_text

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["summary"] == "CI failed on Tests passed; push a fix"


def test_janitor_required_check_failure_after_stale_packet_routes_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #467: a stale review packet on disk must not block the automated
    check-failure rework path; review() must record the verdict against the
    live PR head, not the stale packet head."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    passing_checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    failing_checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    diff_a = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    diff_b = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix"
    fake_gh = FakeGitHubWithChecks(checks=passing_checks)
    fake_gh.diffs[456] = diff_a
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Round 1: clean review writes a packet at the original head.
    result1 = app.review(456)
    assert result1.ok is True
    packet = paths.prs / "pr-456" / "pr.json"
    assert packet.exists()
    assert json.loads(packet.read_text(encoding="utf-8"))["headRefOid"] == "sha-abc123"

    # Round 2: PR head advances and a required check fails.
    fake_gh.checks = failing_checks
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = diff_b

    result2 = app.review(456)

    assert result2.ok is True, result2.message
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-new-head"
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    assert "CI failed on Tests passed; push a fix" in rework_prompt.read_text(encoding="utf-8")

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-new-head"
    assert decision["reviewed_head_source"] == "live"


def test_janitor_required_check_failure_without_linked_issue_stays_blocked(
    tmp_path: Path,
) -> None:
    """Issue #376: a check-failure PR with no linked issue still dead-ends at the janitor gate."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    # Remove every issue reference so linked_issue_number returns None.
    fake_gh.prs[0]["headRefName"] = "misc/fix-search"
    fake_gh.prs[0]["title"] = "fix search"
    fake_gh.prs[0]["body"] = "No issue reference here."
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert "123" not in state.get("issues", {})
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_janitor_required_check_infra_failure_stays_blocked(tmp_path: Path) -> None:
    """Issue #376: an infrastructure check failure (CANCELLED) is never routed
    to code-fix rework -- that part of this test's original intent is
    unchanged. Issue #841: unlike issue #376's era, an infra failure no
    longer sits in janitor_blocked forever with zero remediation -- this
    fixture's check has no parseable Actions run id (no `link` field) at
    all, so it cannot be auto-retried and correctly escalates to a human on
    the very first pass rather than looping/blocking indefinitely (the bug
    issue #841 fixes). A check that DOES carry a run id instead gets one
    auto-rerun first -- see
    test_janitor_infra_failed_first_cancel_triggers_rerun_without_failed_flag.

    Issue #1266: this escalates through the same infra_rerun_cap_exceeded
    site as the cap-exhaustion case, which is a mechanical reason, so it
    lands agent:operator-queue, not agent:human-needed.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_escalated") is True
    state = load_state(paths.state_file)
    # Still never routed to the code-fix rework path -- issue #376's original
    # assertion, unchanged.
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    # Issue #841: escalated to a human instead of blocking silently forever.
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# Issue #1383: infra_blocked routing -- AC1 through AC4
# ---------------------------------------------------------------------------


def test_janitor_required_check_repeated_failure_escalates(tmp_path: Path) -> None:
    """Issue #376: repeated check-failure reworks escalate via the
    request_changes cap. Issue #1266: max_rework_cycles_exceeded is
    mechanical, so this lands agent:operator-queue, not agent:human-needed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithChecks(checks=checks)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["request_changes_count"] == 1

    fake_gh.pr_head_shas[456] = "sha-2"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix1"
    )
    result2 = app.review(456)
    assert result2.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["request_changes_count"] == 2

    fake_gh.pr_head_shas[456] = "sha-3"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix2"
    )
    result3 = app.review(456)
    assert result3.ok is True
    assert result3.data["escalated"] is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_janitor_required_check_failure_noop_does_not_reroute(tmp_path: Path) -> None:
    """Issue #376: a check-failure rework that produced no new content is not re-reviewed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithChecks(checks=checks)
    diff_text = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    fake_gh.diffs[456] = diff_text
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    request_count = state["prs"]["456"]["request_changes_count"]
    needs_rework_count = fake_gh.labels_added.count((123, config.labels.needs_rework))

    result2 = app.review(456)
    assert result2.ok is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["request_changes_count"] == request_count
    assert fake_gh.labels_added.count((123, config.labels.needs_rework)) == needs_rework_count
    assert any("unchanged" in f.lower() for f in result2.data["janitor_failures"])


def test_janitor_required_check_first_failure_triggers_rerun(tmp_path: Path) -> None:
    """Issue #391: first required-check failure triggers one auto-rerun and defers rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.rerun_calls[0][:3] == ["run", "rerun", "12345"]
    assert "--failed" in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["check_rerun_attempts"] == {"sha-abc123": {"Tests passed": [12345]}}
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert "123" not in state.get("issues", {})


def test_janitor_required_check_second_failure_routes_to_rework(tmp_path: Path) -> None:
    """Issue #391: the same check failing again on the same head is definitive and routes to rework.

    Issue #1258 extends this test (rather than adding a parallel one) to also
    cover: (a) AC2 -- the pre-existing sole-failure short-circuit and its
    one-time flake-debounce rerun are unchanged by the new co-occurring-
    failure branch (the rerun fires exactly once, on pass 1, never again on
    pass 2's definitive failure); (b) AC4 -- the new
    ``review_dispatch_skipped_ci_red`` provenance kind is emitted exactly
    once, alongside (additive to) this short-circuit's pre-existing
    ``record_review``-driven routing, tagged ``co_occurring: False`` because
    the required-check failure is this PR's sole janitor failure.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is False
    assert result1.data.get("rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    # Pass 1 is the one-time flake-debounce rerun itself, not the definitive
    # short-circuit -- no provenance event yet.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []

    result2 = app.review(456)
    assert result2.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    # No additional rerun was triggered on the second pass: AC2's "exactly
    # one rerun, not zero, not twice" across both passes.
    assert len(fake_gh.rerun_calls) == 1

    ci_red_events = query_events(paths.state_file, kind="review_dispatch_skipped_ci_red")
    assert len(ci_red_events) == 1
    payload = ci_red_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["failed_required_checks"] == ["Tests passed"]
    assert payload["co_occurring"] is False
    assert payload["co_occurring_failures"] == []


def test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC3): a required-check failure that is NOT the PR's sole
    janitor failure -- here, co-occurring with an empty PR body
    (``_check_body``) -- used to fall straight through
    ``is_check_failure_block`` (which requires the check failure be the SOLE
    blocker) into the passive ``janitor_blocked`` dead end: neither reviewed
    nor routed to rework, silently re-logging the same failure set forever.
    This must now route to rework via the same ``record_review(request_
    changes)`` machinery the sole-failure short-circuit uses, naming BOTH the
    failing check and the co-occurring janitor failure, and it must never
    launch a reviewer.

    The launch-avoidance assertion drives the real, end-to-end pipeline
    (``review()`` then ``dispatch_reviews()``), not just ``review()`` in
    isolation: launch happens in ``dispatch_reviews`` (workflow.py), which a
    ``review()``-only test never reaches, so a bare ``assert launched == []``
    against a monkeypatch that method can't trigger would pass for ANY
    mutation -- an inert control (see the AC1 sibling test immediately below,
    which drives the identical seam and gets ``launched_count == 1``; that is
    the positive-control proof this guard is live and this zero is real).
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    # Trip _check_body's "PR body is empty" failure alongside the red check.
    # This co-occurring failure is independent of issue_number binding
    # (unlike a missing-linked-issue fixture, which review() itself requires
    # non-None before reaching ANY routing branch -- see the janitor.py
    # cross-reference in _check_linked_issue) and is not a merge conflict or
    # a no-op-rework, so it exercises exactly the new branch, not the
    # existing sole-failure short-circuit or the merge-conflict/no-op-rework
    # routing block.
    fake_gh.prs[0]["body"] = ""
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    assert result.ok is True
    assert launched == []

    # Drive the real launch seam: no packet was written (the PR never left
    # the janitor-blocked/rework path), so dispatch_reviews must select and
    # launch nothing. This is what makes ``launched == []`` above meaningful
    # rather than vacuous.
    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "CI failed on Tests passed" in prompt_text
    assert "PR body is empty" in prompt_text

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "Tests passed" in decision["summary"]
    assert "PR body is empty" in decision["summary"]
    assert decision["required_changes"] == ["PR body is empty"]

    ci_red_events = query_events(paths.state_file, kind="review_dispatch_skipped_ci_red")
    assert len(ci_red_events) == 1
    payload = ci_red_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["failed_required_checks"] == ["Tests passed"]
    assert payload["co_occurring"] is True
    assert payload["co_occurring_failures"] == ["PR body is empty"]


def test_janitor_required_check_failure_with_co_occurring_infra_failure_stays_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC3 carve-out): unlike AC3's ``_check_body`` co-occurring
    failure immediately above, a genuine required-check FAILURE co-occurring
    with an INFRA-failed required check (CANCELLED/TIMED_OUT, issue #841/#847)
    must NOT route through the new ``is_co_occurring_check_failure_block``
    branch -- it has its own dedicated infra-rerun/escalation remediation
    that this fix must not shadow or double-dispatch against. This is the
    same combination the pre-existing
    ``test_janitor_mixed_genuine_failure_and_infra_failure_routes_to_rework_not_infra_rerun``
    (issue #847) pins from the infra side; this sibling pins it from #1258's
    side -- through the real ``review()`` + ``dispatch_reviews()`` pipeline,
    with the new provenance kind explicitly asserted ABSENT -- so the
    boundary is owned by this issue's own tests, not inferred from an
    unrelated suite.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "CANCELLED"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    # Falls through to the passive janitor_blocked path, unchanged by this
    # diff -- this combination is an accepted carve-out (infra remediation
    # owns it), not a new rework route.
    assert result.ok is False
    assert launched == []

    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"].get("decision") is None
    failures = state["prs"]["456"]["janitor_failures"]
    assert any("Tests passed" in f for f in failures)
    assert any("infrastructure" in f.lower() and "Lint & Format" in f for f in failures)

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert not rework_prompt.exists()

    # The new co-occurring-failure provenance kind must NOT fire for this
    # carve-out -- it is reserved for the code-fixable branch above.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []


def test_janitor_all_checks_green_dispatches_reviewer_ci_red_kind_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC1): the green-checks path is unaffected by the new
    co-occurring-failure branch and the new provenance kind -- both guards
    require ``verdict.failed_required_checks`` to be truthy, which an
    all-green PR never has, so neither new branch's body executes at all.
    Exercises the real, end-to-end pipeline (``review()`` packet-build then
    ``dispatch_reviews()`` launch), not just ``review()`` in isolation, so
    "the reviewer is launched with the same command as pre-diff" is an
    actual launch-call assertion, not merely an absence-of-packet inference.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = paths.prs / "pr-456" / "review-prompt.md"
    assert packet.exists()
    # The janitor gate never blocked this PR, so none of its routing kinds
    # (old or new) fired.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []
    assert not any(
        e["kind"] == "janitor_gate" for e in load_state(paths.state_file).get("events", [])
    )

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    dispatch_result = app.dispatch_reviews()

    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 1
    assert len(launched) == 1
    _launch_args, launch_kwargs = launched[0]
    assert launch_kwargs.get("review") is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["review_dispatch_status"] == "review_dispatch_dispatched"
    # Pre-existing dispatch-lane kinds still fire (additive, not replaced).
    assert query_events(paths.state_file, kind="review_dispatch_claim") != []
    # The new CI-red kind is scoped to the janitor gate and never fires on
    # the dispatch-claim/launch path itself (also AC6's redundant-gate claim,
    # from the launch side).
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []
