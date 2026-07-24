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

from pathlib import Path

from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub, FakeGitHubWithChecks


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
