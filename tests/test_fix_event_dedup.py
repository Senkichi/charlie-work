"""Tests for TASK W5: event-log dedup (dispatch_skip_blocked, janitor_gate)
and blocked-chain-dead attention.

Covers cost-spirals.md Findings 2 and 3:

* ``dispatch_skip_blocked`` used to be an unconditional per-pass append --
  784 byte-identical events over 18h for 4 permanently-blocked issues in the
  investigated window. It must now emit only when the (issue, blockers)
  content actually changed since the last emission.
* ``janitor_gate`` similarly re-emitted an identical failure set every pass
  (699 events for 5 stuck PRs). It must emit only when the failure set
  changes from what is already on record in the PR's state.
* A blocked issue whose every currently-open blocker is itself dead
  (escalated, or its tracked PR is escalated/janitor_blocked) can never
  unblock through any automated path -- pr-lifecycle.md's "escalated is
  functionally invisible" finding, and cost-spirals.md Finding 3's 4+ day
  silent stuck chain. This must alert exactly once per transition into that
  state, not repeat every pass, and must re-alert if it recovers and then
  goes dead again.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub, FakeGitHubWithChecks, FakeGitHubWithMissingRequiredAndRuns
from _review_fixtures import _required_checks_config


def _events(state, kind: str) -> list[dict]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


class FakeGitHubWithBlockers(FakeGitHub):
    """A dependent issue (#752) blocked by an open foundation issue (#743).

    Mirrors test_charlie_work.py's existing
    ``test_dispatch_skips_when_any_blocker_open``-style fixtures exactly, so
    the blocker-parsing/``are_issues_open`` plumbing this dedup logic sits
    on top of is already proven correct elsewhere.
    """

    def __init__(
        self, blocker_body: str = "Blocked by #743", open_blockers: set[int] | None = None
    ):
        super().__init__()
        self._open_blockers = open_blockers if open_blockers is not None else {743}
        self.issues = [
            {
                "number": 752,
                "title": "Dependent issue",
                "url": "https://example.test/issues/752",
                "body": blocker_body,
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            },
            {
                "number": 743,
                "title": "Blocker issue",
                "url": "https://example.test/issues/743",
                "body": "Foundation work",
                "labels": [],
                "state": "OPEN",
            },
        ]

    def issue_list(self, labels=None, state=None):
        if isinstance(labels, str):
            ready_label = labels
            return [
                issue
                for issue in self.issues
                if ready_label in [label["name"] for label in issue.get("labels", [])]
            ]
        elif labels:
            return [
                issue
                for issue in self.issues
                if any(
                    label in [label_obj["name"] for label_obj in issue.get("labels", [])]
                    for label in labels
                )
            ]
        return self.issues

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        return self._open_blockers & set(issue_numbers)


def _blocked_app(tmp_path: Path, **kwargs) -> OrchestratorApp:
    config = OrchestratorConfig(devin=DevinConfig(adapter="manual"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithBlockers(**kwargs)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.gh.prs[0]["state"] = "CLOSED"  # keep the default PR/issue #123 out of the way
    return app


def test_dispatch_skip_blocked_emits_once_then_silent_until_content_changes(
    tmp_path: Path,
) -> None:
    app = _blocked_app(tmp_path)

    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_skip_blocked")) == 1

    # Second pass, identical blockers: must stay silent.
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_skip_blocked")) == 1

    # Third pass, a THIRD pass with truly identical blockers stays silent too
    # (not just a one-shot suppression of the second call).
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_skip_blocked")) == 1

    # Change the declared blocker set: content changed, must emit again.
    app.gh.issues[0]["body"] = "Blocked by #743, #744"
    app.gh.issues.append(
        {
            "number": 744,
            "title": "Second blocker",
            "url": "https://example.test/issues/744",
            "body": "More foundation work",
            "labels": [],
            "state": "OPEN",
        }
    )
    app.gh._open_blockers = {743, 744}

    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_skip_blocked")) == 2
    assert state["issues"]["752"]["last_skip_blocked_blockers"] == [743, 744]


def test_janitor_gate_emits_once_then_silent_until_failures_change(tmp_path: Path) -> None:
    """Uses a PR with no linked issue (issue #376's existing fixture shape)
    so the janitor gate falls straight to the still-unroutable
    ``janitor_blocked`` branch -- isolating the dedup behavior from the new
    conflict/no-op-rework routing this same fix adds.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks()
    fake_gh.prs[0]["headRefName"] = "misc/fix-search"
    fake_gh.prs[0]["title"] = "fix search"
    fake_gh.prs[0]["body"] = "No issue reference here."
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "janitor_gate")) == 1
    first_failures = state["prs"]["456"]["janitor_failures"]

    # Second pass, identical failure set: must stay silent.
    result2 = app.review(456)
    assert result2.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "janitor_gate")) == 1

    # Third identical pass: still silent.
    app.review(456)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "janitor_gate")) == 1

    # Change the failure set (also DIRTY, a genuinely new failure string):
    # must emit a fresh event.
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    result4 = app.review(456)
    assert result4.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "janitor_gate")) == 2
    assert state["prs"]["456"]["janitor_failures"] != first_failures


def test_ci_run_never_created_emitted_once_per_head_with_control(tmp_path: Path) -> None:
    """Job-cannon measured 11 PRs stuck behind "Required check(s) missing"
    for 4+ days with no run ever created for the head SHA -- indistinguishable
    from ordinary CI latency using check data alone. When the janitor gate
    reports missing required checks AND a direct Actions query finds zero
    workflow runs for the (grace-period-aged) head SHA, ``ci_run_never_created``
    must fire exactly once per (pr, head_sha); a head SHA with runs already
    present must never fire it (the control).
    """
    config = _required_checks_config(ci_run_never_created_grace_minutes=5)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    stale_updated_at = (
        (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )

    # Real GitHub head SHAs are always hex (safe_ref.require_valid_sha is a
    # defense-in-depth format guard on any value headed for a gh api argv,
    # per issue #659) -- use a conforming value rather than the suite-wide
    # "sha-abc123" placeholder, which is not valid hex.
    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[])
    fake_gh.prs[0]["headRefOid"] = "abc123abc123"
    fake_gh.prs[0]["updatedAt"] = stale_updated_at
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is False
    state = load_state(app.paths.state_file)
    events = _events(state, "ci_run_never_created")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["head_sha"] == "abc123abc123"
    assert payload["branch"] == "agent/issue-123-fix-search"
    assert set(payload["missing_checks"]) == {"Tests passed", "Lint & Format", "Pre-commit"}

    # Second pass, same head SHA: must stay silent (at most once per
    # (pr, head_sha)), even though the janitor gate re-evaluates every pass.
    result2 = app.review(456)
    assert result2.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "ci_run_never_created")) == 1

    # A subsequent push (new head SHA) is a fresh (pr, head_sha) pair and
    # must be able to alert again.
    fake_gh.prs[0]["headRefOid"] = "def456def456"
    fake_gh.prs[0]["updatedAt"] = stale_updated_at
    result3 = app.review(456)
    assert result3.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "ci_run_never_created")) == 2

    # Control: workflow runs DO exist for the head SHA -> never fires, even
    # though the janitor gate still reports the same "missing" required
    # checks and the head is equally stale. This isolates the query result
    # as the discriminator, not merely "checks are missing" or "PR is old".
    control_tmp = tmp_path / "control"
    control_paths = runtime_paths(control_tmp, config.runtime.state_dir)
    control_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[{"id": 1, "status": "queued"}])
    control_gh.prs[0]["headRefOid"] = "abc123abc123"
    control_gh.prs[0]["updatedAt"] = stale_updated_at
    control_app = OrchestratorApp(control_tmp, control_paths, config, control_gh)

    control_result = control_app.review(456)
    assert control_result.ok is False
    control_state = load_state(control_app.paths.state_file)
    assert len(_events(control_state, "ci_run_never_created")) == 0


def test_ci_run_never_created_stays_silent_when_the_runs_query_fails(tmp_path: Path) -> None:
    """``workflow_runs_for_head`` returning ``None`` means the query itself
    failed (rate limit, transient error) -- errors as values, per this
    repo's convention -- and must never be treated as "zero runs = never
    created". Only a successful, empty response is positive evidence of
    "never created"; a failed query is not evidence of anything. This guards
    the fail-closed branch a future ``if not head_runs:`` simplification
    (collapsing the `None` and `[]` cases) would silently flip to fail-open.
    """
    config = _required_checks_config(ci_run_never_created_grace_minutes=5)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    stale_updated_at = (
        (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )

    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=None)
    fake_gh.prs[0]["headRefOid"] = "abc123abc123"
    fake_gh.prs[0]["updatedAt"] = stale_updated_at
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)
    assert result.ok is False
    state = load_state(app.paths.state_file)
    assert len(_events(state, "ci_run_never_created")) == 0
    assert "ci_run_never_created_head" not in state["prs"]["456"]


def test_ci_run_never_created_fires_for_an_escalated_pr(tmp_path: Path) -> None:
    """A PR stuck 4+ days behind a missing-checks failure -- the exact
    condition ``ci_run_never_created`` detects -- is itself a strong
    escalation candidate. ``review()`` returns from an early branch for any
    escalated PR/issue (before the non-escalated janitor-gate block that the
    sibling test above exercises ever runs), so the detector must also fire
    from that branch or it is unreachable for the population it was built to
    diagnose. Seeds state the way the escalated-PR fixture above does (PR
    entry present, issue escalated) so ``review()`` takes the early-return
    path, then asserts the event still fires exactly once from there.
    """
    config = _required_checks_config(ci_run_never_created_grace_minutes=5)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    stale_updated_at = (
        (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )

    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[])
    fake_gh.prs[0]["headRefOid"] = "abc123abc123"
    fake_gh.prs[0]["updatedAt"] = stale_updated_at
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": [],
        }
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.review(456)
    assert result.ok is True
    assert result.data.get("pass_skipped") is True

    state = load_state(app.paths.state_file)
    events = _events(state, "ci_run_never_created")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["head_sha"] == "abc123abc123"
    assert payload["escalated"] is True
    assert state["prs"]["456"]["ci_run_never_created_head"] == "abc123abc123"

    # Second pass, still escalated, same head: must stay silent.
    result2 = app.review(456)
    assert result2.data.get("pass_skipped") is True
    state = load_state(app.paths.state_file)
    assert len(_events(state, "ci_run_never_created")) == 1


def test_blocked_chain_dead_alerts_once_per_transition(tmp_path: Path) -> None:
    app = _blocked_app(tmp_path)

    def _mark_blocker_escalated() -> None:
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["issues"]["743"] = {"number": 743, "status": "escalated"}
            save_state(app.paths.state_file, state)

    def _clear_blocker_escalation() -> None:
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["issues"]["743"] = {"number": 743, "status": "in_progress"}
            save_state(app.paths.state_file, state)

    # Blocker is alive: no chain-dead alert, ordinary skip-blocked only.
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 0

    # Blocker dies (escalated): transition into all-dead -> exactly one alert.
    _mark_blocker_escalated()
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    chain_events = _events(state, "dispatch_blocked_chain_dead")
    assert len(chain_events) == 1
    assert chain_events[0]["payload"] == {"issue": 752, "chain_root": [743]}

    # Still dead on the next pass: must NOT re-alert.
    app.dispatch(limit=10)
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 1

    # Blocker recovers: no new alert, but the transition marker clears
    # silently so a later re-death alerts again instead of staying silent
    # forever.
    _clear_blocker_escalation()
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 1
    assert state["issues"]["752"].get("chain_dead_alerted_blockers") is None

    # Blocker dies again: this is a FRESH transition into all-dead -> alerts
    # again (must not have been permanently suppressed by the first alert).
    _mark_blocker_escalated()
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 2


def _blocked_app_with_blocker_pr(tmp_path: Path) -> OrchestratorApp:
    """Like ``_blocked_app`` but the blocker issue #743 has a tracked open PR.

    The PR (#1627) links to #743 via its branch name (``agent/issue-743-...``)
    so ``linked_issue_number`` binds it, making ``pr_by_issue[743]`` resolve
    during dispatch. The PR's janitor-blocked status is seeded directly into
    state.json by each test (mirroring how ``review()`` persists it).
    """
    app = _blocked_app(tmp_path)
    # Add an open PR linked to the blocker issue #743.
    app.gh.prs.append(
        {
            "number": 1627,
            "title": "fix: foundation work for #743",
            "url": "https://example.test/pull/1627",
            "headRefName": "agent/issue-743-foundation",
            "baseRefName": "main",
            "headRefOid": "sha-743-head",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #743\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    )
    # Keep the default PR #456 (linked to #123) closed so it stays out of the
    # way -- ``_blocked_app`` already closes it, but ``pr_list`` filters on
    # OPEN so this is belt-and-suspenders.
    return app


def _seed_blocker_pr_state(
    app: OrchestratorApp,
    *,
    status: str,
    is_missing_checks_only_block: bool = False,
    ci_run_never_created_head: str | None = None,
    janitor_failures: list[str] | None = None,
) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        pr_entry: dict[str, Any] = {
            "number": 1627,
            "issue_number": 743,
            "status": status,
            "janitor_ok": False,
        }
        if janitor_failures is not None:
            pr_entry["janitor_failures"] = janitor_failures
        if is_missing_checks_only_block:
            pr_entry["is_missing_checks_only_block"] = True
        if ci_run_never_created_head is not None:
            pr_entry["ci_run_never_created_head"] = ci_run_never_created_head
        state["prs"]["1627"] = pr_entry
        save_state(app.paths.state_file, state)


def test_blocked_chain_dead_no_alert_for_transient_missing_checks_only_pr(
    tmp_path: Path,
) -> None:
    """Issue #1133: a blocker whose fresh PR is ``janitor_blocked`` ONLY
    because its required checks haven't reported yet (``is_missing_checks_only_block``
    with no ``ci_run_never_created_head``) is transient, not dead -- it
    self-heals within one CI cycle. ``dispatch_blocked_chain_dead`` must NOT
    fire for this population.
    """
    app = _blocked_app_with_blocker_pr(tmp_path)
    _seed_blocker_pr_state(
        app,
        status="janitor_blocked",
        is_missing_checks_only_block=True,
        janitor_failures=["Required check(s) missing: Tests passed"],
    )

    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 0
    # The chain-dead marker must not be set (no transition recorded).
    assert state["issues"]["752"].get("chain_dead_alerted_blockers") is None


def test_blocked_chain_dead_alerts_when_missing_checks_pr_confirmed_ci_never_created(
    tmp_path: Path,
) -> None:
    """Issue #1133 durable variant: the same missing-checks-only PR, but CI
    was confirmed to have never started for this head
    (``ci_run_never_created_head`` set by ``_detect_ci_run_never_created``).
    That is the durable population the alert exists for, so it MUST fire.
    """
    app = _blocked_app_with_blocker_pr(tmp_path)
    _seed_blocker_pr_state(
        app,
        status="janitor_blocked",
        is_missing_checks_only_block=True,
        ci_run_never_created_head="sha-743-head",
        janitor_failures=["Required check(s) missing: Tests passed"],
    )

    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    chain_events = _events(state, "dispatch_blocked_chain_dead")
    assert len(chain_events) == 1
    assert chain_events[0]["payload"] == {"issue": 752, "chain_root": [743]}


def test_blocked_chain_dead_alerts_for_durable_janitor_blocked_pr(
    tmp_path: Path,
) -> None:
    """Issue #1133: a ``janitor_blocked`` PR with a durable failure (merge
    conflict, not a missing-checks-only block) is dead and MUST alert.
    ``is_missing_checks_only_block`` is False here, so the transient carve-out
    does not apply regardless of ``ci_run_never_created_head``.
    """
    app = _blocked_app_with_blocker_pr(tmp_path)
    _seed_blocker_pr_state(
        app,
        status="janitor_blocked",
        is_missing_checks_only_block=False,
        janitor_failures=["PR has merge conflicts (mergeable=CONFLICTING)"],
    )

    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    chain_events = _events(state, "dispatch_blocked_chain_dead")
    assert len(chain_events) == 1
    assert chain_events[0]["payload"] == {"issue": 752, "chain_root": [743]}


def test_blocked_chain_dead_recovers_when_transient_pr_clears(
    tmp_path: Path,
) -> None:
    """Issue #1133 recovery: once the transient PR's checks report and the
    janitor clears it (status leaves ``janitor_blocked``), the blocker is
    alive again. A subsequent durable death (e.g. the issue escalates) must
    alert fresh, proving the transient carve-out did not permanently suppress
    the chain-dead machinery.
    """
    app = _blocked_app_with_blocker_pr(tmp_path)

    # Start transient: no alert.
    _seed_blocker_pr_state(
        app,
        status="janitor_blocked",
        is_missing_checks_only_block=True,
        janitor_failures=["Required check(s) missing: Tests passed"],
    )
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 0

    # CI reports, janitor passes -> PR is no longer blocked (reviewing).
    _seed_blocker_pr_state(
        app,
        status="reviewing",
        is_missing_checks_only_block=False,
        janitor_failures=[],
    )
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    assert len(_events(state, "dispatch_blocked_chain_dead")) == 0

    # Now the blocker issue escalates -> fresh death, must alert.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["743"] = {"number": 743, "status": "escalated"}
        save_state(app.paths.state_file, state)
    app.dispatch(limit=10)
    state = load_state(app.paths.state_file)
    chain_events = _events(state, "dispatch_blocked_chain_dead")
    assert len(chain_events) == 1
    assert chain_events[0]["payload"] == {"issue": 752, "chain_root": [743]}
