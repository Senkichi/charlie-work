"""Review-queue command: listing, read-only guarantees, stranded repairs.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the ``test_review_queue_*`` command seam -- queue listing and read-only guarantees, plus the stranded request_changes reroute repairs (issue #784 AC-8). Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from _review_fixtures import _dispatch_reviews_app, _write_review_packet, _review_queue_app
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_queue_includes_missing_pending_and_stale_decisions(
    tmp_path: Path,
) -> None:
    """Issue #369: review_queue enumerates current packets awaiting a verdict."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10: missing decision",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "Fix #20: pending decision",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "Fix #30: stale request_changes",
            "url": "https://example.test/pull/300",
            "headRefName": "agent/issue-30-fix",
            "baseRefName": "main",
            "headRefOid": "sha-300-new",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #30",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 400,
            "title": "Fix #40: approved on current head",
            "url": "https://example.test/pull/400",
            "headRefName": "agent/issue-40-fix",
            "baseRefName": "main",
            "headRefOid": "sha-400",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #40",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 500,
            "title": "Fix #50: stale packet",
            "url": "https://example.test/pull/500",
            "headRefName": "agent/issue-50-fix",
            "baseRefName": "main",
            "headRefOid": "sha-500-new",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #50",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 600,
            "title": "No linked issue",
            "url": "https://example.test/pull/600",
            "headRefName": "feature/unlinked",
            "baseRefName": "main",
            "headRefOid": "sha-600",
            "mergeStateStatus": "CLEAN",
            "body": "Some feature",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _review_queue_app(tmp_path, prs=prs)

    # PR 100: current packet, no decision -> decision missing
    _write_review_packet(tmp_path, 100, "sha-100")
    # PR 200: current packet, pending decision
    _write_review_packet(tmp_path, 200, "sha-200", {"decision": "pending"})
    # PR 300: current packet, request_changes from prior head -> stale.
    # Real (non-placeholder) summary so this exercises the stale/carry-
    # forward path under test, not issue #784's content-free "vacuous"
    # detection (a distinct concern covered by its own tests).
    _write_review_packet(
        tmp_path,
        300,
        "sha-300-new",
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-300-old",
            "summary": "some prior finding that needs a rework brief",
        },
    )
    # PR 400: current packet, approved on current head -> excluded
    _write_review_packet(
        tmp_path,
        400,
        "sha-400",
        {"decision": "approved", "reviewed_head_sha": "sha-400"},
    )
    # PR 500: stale packet (recorded head differs from live head) -> excluded
    _write_review_packet(tmp_path, 500, "sha-500-old")
    # PR 600: unlinked PR has no packet, so excluded

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": 100,
            "issue": 10,
            "packet_head_sha": "sha-100",
            "decision": "missing",
            "reviewed_head_sha": None,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        },
        {
            "pr": 200,
            "issue": 20,
            "packet_head_sha": "sha-200",
            "decision": "pending",
            "reviewed_head_sha": None,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        },
        {
            "pr": 300,
            "issue": 30,
            "packet_head_sha": "sha-300-new",
            "decision": "stale",
            "reviewed_head_sha": "sha-300-old",
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        },
    ]


def test_review_queue_is_read_only(tmp_path: Path) -> None:
    """Issue #369: review_queue must not mutate state.json or PR-directory files."""
    app = _review_queue_app(tmp_path)
    _write_review_packet(tmp_path, 456, "sha-abc123")
    before_state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    pr_json = (app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8")
    prompt = (app.paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")

    result = app.review_queue()

    assert result.ok is True
    after_state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    assert after_state == before_state
    assert (app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8") == pr_json
    assert (app.paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8") == prompt


def test_review_queue_lists_pending_pr_regardless_of_dispatch_enabled(tmp_path: Path) -> None:
    """Issue #868 AC4 (pin, not a new behavior): review_queue() must not
    filter on review_dispatch.enabled, so a PR awaiting review is queued the
    moment the flag flips back on -- nothing about candidate selection
    itself depends on the flag; only dispatch_reviews()'s own launch gate
    does.
    """
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)
    _write_review_packet(tmp_path, 100, "sha-100")

    result = app.review_queue()

    assert result.ok is True
    assert [c["pr"] for c in result.data["queue"]] == [100]


# --------------------------------------------------------------------------
# Issue #784 AC-8, Case 2: review_queue()'s head-unchanged shortcut must not
# treat "reviewed at live head" as "nothing to do" when a real, actionable,
# non-escalated request_changes verdict was never actually routed to
# rework_requested -- e.g. issue #789's reconcile one-way "closed" gate
# clobbering the issue status after record_review set it. This is a repair
# of a dropped transition (re-applying the same target record_review already
# decided via the same generic _route_to_rework entry point), never a new
# decision, and must be idempotent.
# --------------------------------------------------------------------------


def test_review_queue_reroutes_stranded_request_changes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """Motivating real-world case: PR #693 -> issue #649. A real, actionable
    request_changes verdict (non-empty prose summary, within the rework-cycle
    budget -- ``escalated: False``) is recorded at the live head, but the
    linked issue's status was left at (or clobbered to) something other than
    ``rework_requested`` -- so ``dispatch_rework``'s
    ``status == "rework_requested"`` selection never picks the issue up and
    the PR is stranded forever, even though nothing is actually wrong with
    the verdict itself.

    review_queue() must detect this at the head-unchanged shortcut and
    re-drive the SAME rework_requested transition via the SAME generic
    ``_route_to_rework`` entry point ``record_review`` itself uses -- never
    adding the PR to the returned ``queue`` (that would wrongly trigger a
    brand-new reviewer dispatch for a PR that already has a valid verdict).

    A second ``review_queue()`` call over the now-repaired state must not
    dispatch or mutate anything further (coordinator's explicit, non-
    negotiable requirement for this fix).
    """
    prs = [
        {
            "number": 693,
            "title": "Fix #649: some change",
            "url": "https://example.test/pull/693",
            "headRefName": "agent/issue-649-fix",
            "baseRefName": "main",
            "headRefOid": "sha-693",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #649",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _review_queue_app(tmp_path, prs=prs, dry_run=False)
    # Issue #1123: the restorer now consults the GitHub issue state before
    # re-activating. The motivating case (issue #789) is a GitHub-OPEN issue
    # whose state status was clobbered to "closed" by a reconcile bug -- so
    # the fake must report #649 as OPEN for the repair to fire.
    app.gh.issues = [
        {
            "number": 649,
            "title": "Fix search",
            "url": "https://example.test/issues/649",
            "body": "Search is broken",
            "labels": [],
            "state": "OPEN",
        }
    ]
    _write_review_packet(
        tmp_path,
        693,
        "sha-693",
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-693",
            "required_changes": [],
            "summary": (
                "The refactor drops the timeout guard added in #611; restore "
                "it before merging, otherwise a hung subprocess call blocks "
                "the orchestrator loop indefinitely."
            ),
            "escalated": False,
        },
    )
    # Simulate issue #789: the linked issue's status is something other than
    # rework_requested even though record_review's own contract says a
    # non-escalated request_changes verdict must have set it there.
    state = load_state(app.paths.state_file)
    state["issues"]["649"] = {"number": 649, "status": "closed"}
    save_state(app.paths.state_file, state)

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    state_after = load_state(app.paths.state_file)
    assert state_after["issues"]["649"]["status"] == "rework_requested"
    assert state_after["prs"]["693"]["status"] == "rework_requested"
    assert any(
        e["kind"] == "stranded_request_changes_rework_requested" for e in state_after["events"]
    )
    prompt_path = app.paths.prs / "pr-693" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (649, app.config.labels.needs_rework) in app.gh.labels_added

    # Idempotency: a second pass over the now-repaired state must dispatch
    # and mutate nothing further. "Without that test I will not merge this."
    events_before_second_pass = len(state_after["events"])
    labels_added_before_second_pass = list(app.gh.labels_added)

    result_2 = app.review_queue()

    assert result_2.ok is True
    assert result_2.data["queue"] == []
    state_final = load_state(app.paths.state_file)
    assert state_final["issues"]["649"]["status"] == "rework_requested"
    assert state_final["prs"]["693"]["status"] == "rework_requested"
    assert len(state_final["events"]) == events_before_second_pass
    assert app.gh.labels_added == labels_added_before_second_pass


def test_review_queue_does_not_reroute_stranded_request_changes_for_closed_issue(
    tmp_path: Path,
) -> None:
    """Issue #1123: the stranded request_changes restorer must never re-activate
    a CLOSED GitHub issue. The motivating real-world case is jc #1293 / PR #1394:
    the issue was closed as COMPLETED by the operator, but PR #1394 stayed open
    with a recorded ``request_changes`` verdict at an unchanged head -- the exact
    bait the stranded restorer keys on. Without consulting the GitHub issue
    state, the restorer flips the status back to ``rework_requested`` every pass,
    the no-op rework cap escalates it, and reconcile's
    ``state_active_status_issue_closed`` flips it back to "closed" -- a perpetual
    three-lane loop that emitted 56 bogus escalation events over 1.3 days.

    The fix: the restorer consults ``gh.issue_view`` and skips (emitting a
    ``stranded_request_changes_skipped_issue_closed`` event) when the linked
    issue is CLOSED on GitHub, converging the state status to "closed" and
    recording a ``stranded_skip_closed`` marker so subsequent passes
    short-circuit without a second ``gh.issue_view()`` call or a duplicate
    event. A second pass over the same stranded state must not re-emit the
    skip event or re-fetch the issue.
    """
    pr_number = 1394
    issue_number = 1293
    head = "2ee42f8d"
    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}: some change",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _review_queue_app(tmp_path, prs=prs, dry_run=False)
    # The linked issue is CLOSED on GitHub -- the operator closed it as
    # COMPLETED via a different PR. This is the ground truth the restorer
    # must consult.
    app.gh.issues = [
        {
            "number": issue_number,
            "title": f"Issue {issue_number}",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "Some issue",
            "labels": [],
            "state": "CLOSED",
        }
    ]
    _write_review_packet(
        tmp_path,
        pr_number,
        head,
        {
            "decision": "request_changes",
            "reviewed_head_sha": head,
            "required_changes": [],
            "summary": "Real reviewer prose describing a real problem to fix.",
            "escalated": False,
        },
    )
    # Plant the state status as "closed" -- the value reconcile's
    # state_active_status_issue_closed writes for a CLOSED GitHub issue.
    # "closed" is NOT in _REWORK_ALREADY_ROUTED_STATUSES, so the restorer's
    # idempotency guard does not short-circuit and the bug would fire
    # (flipping it back to "rework_requested") without the #1123 fix.
    state = load_state(app.paths.state_file)
    state["issues"][str(issue_number)] = {"number": issue_number, "status": "closed"}
    save_state(app.paths.state_file, state)

    # Track gh.issue_view calls to prove the second pass does not re-fetch.
    issue_view_calls: list[int] = []
    _original_issue_view = app.gh.issue_view

    def _counting_issue_view(number: int) -> Any:
        issue_view_calls.append(number)
        return _original_issue_view(number)

    app.gh.issue_view = _counting_issue_view  # type: ignore[method-assign]

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    # First pass fetched the issue once to confirm it is CLOSED.
    assert issue_view_calls == [issue_number]

    state_after = load_state(app.paths.state_file)
    # The restorer must NOT have re-activated the issue.
    assert state_after["issues"][str(issue_number)]["status"] == "closed"
    assert state_after["prs"].get(str(pr_number), {}).get("status") != "rework_requested"
    # The stranded_skip_closed marker must be set for idempotency.
    assert state_after["issues"][str(issue_number)].get("stranded_skip_closed") is True
    # No rework prompt written, no labels added.
    prompt_path = app.paths.prs / f"pr-{pr_number}" / "rework-prompt.md"
    assert not prompt_path.exists()
    assert app.gh.labels_added == []

    # A skip event must be recorded for observability.
    skip_events = query_events(
        app.paths.state_file, kind="stranded_request_changes_skipped_issue_closed"
    )
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["pr_number"] == pr_number
    assert skip_events[0]["payload"]["issue_number"] == issue_number
    assert skip_events[0]["payload"]["head_sha"] == head

    # No stranded_request_changes_rework_requested event must have fired.
    rework_events = query_events(
        app.paths.state_file, kind="stranded_request_changes_rework_requested"
    )
    assert rework_events == []

    # --- Second pass: idempotency ---
    # A subsequent review_queue() call over the same stranded state must not
    # re-fetch gh.issue_view or re-emit the skip event. The stranded_skip_closed
    # marker + status == "closed" short-circuits before the fetch.
    events_before_second_pass = len(
        query_events(app.paths.state_file, kind="stranded_request_changes_skipped_issue_closed")
    )

    result_2 = app.review_queue()

    assert result_2.ok is True
    assert result_2.data["queue"] == []
    # No additional gh.issue_view call on the second pass.
    assert issue_view_calls == [issue_number]
    # No duplicate skip event.
    skip_events_2 = query_events(
        app.paths.state_file, kind="stranded_request_changes_skipped_issue_closed"
    )
    assert len(skip_events_2) == events_before_second_pass
    # Status unchanged.
    state_final = load_state(app.paths.state_file)
    assert state_final["issues"][str(issue_number)]["status"] == "closed"
    assert state_final["prs"].get(str(pr_number), {}).get("status") != "rework_requested"


def test_review_queue_closed_issue_skip_converges_active_status_and_labels(
    tmp_path: Path,
) -> None:
    """Issue #1123 rework: when the restorer discovers a CLOSED issue whose
    state status is not yet ``"closed"`` (e.g. ``"reviewing"`` -- still in
    ACTIVE_STATE_STATUSES), it must converge the status to ``"closed"`` and
    strip active labels in the same pass, mirroring reconcile's
    ``state_active_status_issue_closed``. Without status convergence the
    ``stranded_skip_closed`` marker check (gated on
    ``status == "closed"``) would not short-circuit on the next pass and the
    restorer would re-fetch forever. Without label stripping the issue
    would carry active labels on a finalized state, since setting
    ``"closed"`` here prevents reconcile's
    ``state_active_status_issue_closed`` from firing (it requires
    ``status in ACTIVE_STATE_STATUSES``).
    """
    pr_number = 1395
    issue_number = 1294
    head = "3ff53f9e"
    active_label = "agent:in-progress"
    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}: another change",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.prs = prs
    # Issue is CLOSED on GitHub but still carries an active label.
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": f"Issue {issue_number}",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "Some issue",
            "labels": [{"name": active_label}],
            "state": "CLOSED",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)
    _write_review_packet(
        tmp_path,
        pr_number,
        head,
        {
            "decision": "request_changes",
            "reviewed_head_sha": head,
            "required_changes": [],
            "summary": "Real reviewer prose.",
            "escalated": False,
        },
    )
    # Plant status as "reviewing" -- an ACTIVE_STATE_STATUSES member that is
    # NOT in _REWORK_ALREADY_ROUTED_STATUSES, so the restorer proceeds past
    # the early-return and fetches the issue.
    state = load_state(app.paths.state_file)
    state["issues"][str(issue_number)] = {"number": issue_number, "status": "reviewing"}
    save_state(app.paths.state_file, state)

    issue_view_calls: list[int] = []
    _original_issue_view = app.gh.issue_view

    def _counting_issue_view(number: int) -> Any:
        issue_view_calls.append(number)
        return _original_issue_view(number)

    app.gh.issue_view = _counting_issue_view  # type: ignore[method-assign]

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    assert issue_view_calls == [issue_number]

    state_after = load_state(app.paths.state_file)
    # Status converged to "closed" and marker set.
    assert state_after["issues"][str(issue_number)]["status"] == "closed"
    assert state_after["issues"][str(issue_number)].get("stranded_skip_closed") is True
    # Active label stripped from the closed issue.
    assert (issue_number, active_label) in app.gh.labels_removed
    # No rework routed.
    assert state_after["prs"].get(str(pr_number), {}).get("status") != "rework_requested"
    prompt_path = app.paths.prs / f"pr-{pr_number}" / "rework-prompt.md"
    assert not prompt_path.exists()

    skip_events = query_events(
        app.paths.state_file, kind="stranded_request_changes_skipped_issue_closed"
    )
    assert len(skip_events) == 1

    # Second pass: idempotent -- no re-fetch, no duplicate event, no extra
    # label removal.
    labels_removed_before = list(app.gh.labels_removed)
    result_2 = app.review_queue()
    assert result_2.ok is True
    assert issue_view_calls == [issue_number]
    assert app.gh.labels_removed == labels_removed_before
    skip_events_2 = query_events(
        app.paths.state_file, kind="stranded_request_changes_skipped_issue_closed"
    )
    assert len(skip_events_2) == 1


def test_review_queue_does_not_reroute_escalated_request_changes(tmp_path: Path) -> None:
    """The Case 2 repair is scoped to the non-escalated lane only: when
    ``record_review`` already decided a verdict exceeded the rework-cycle
    budget (``escalated: True``), review_queue() must not independently
    re-decide that or invent a second rework lane bypassing
    ``max_rework_cycles`` -- it defers entirely to the existing
    escalation/rescue path.
    """
    prs = [
        {
            "number": 694,
            "title": "Fix #650: some other change",
            "url": "https://example.test/pull/694",
            "headRefName": "agent/issue-650-fix",
            "baseRefName": "main",
            "headRefOid": "sha-694",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #650",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _review_queue_app(tmp_path, prs=prs, dry_run=False)
    _write_review_packet(
        tmp_path,
        694,
        "sha-694",
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-694",
            "required_changes": [],
            "summary": "Real reviewer prose describing a real problem to fix.",
            "escalated": True,
        },
    )
    state = load_state(app.paths.state_file)
    state["issues"]["650"] = {"number": 650, "status": "closed"}
    save_state(app.paths.state_file, state)

    result = app.review_queue()

    assert result.data["queue"] == []
    state_after = load_state(app.paths.state_file)
    # Untouched -- Case 2 must not fire for an escalated verdict.
    assert state_after["issues"]["650"]["status"] == "closed"
    prompt_path = app.paths.prs / "pr-694" / "rework-prompt.md"
    assert not prompt_path.exists()
    assert app.gh.labels_added == []
