"""Tests for issue #1132: transient GraphQL repo-resolution failure parks a PR
as ``foreign_issue_ref`` forever, invisibly, with no self-heal.

The root cause: ``GitHubNotFoundError`` conflates a permanent issue-level 404
("Could not resolve to a Issue with the number N") with a transient
repository-level resolution failure ("Could not resolve to a Repository with
the name 'owner/repo'"). Both match ``_is_not_found_gh_error``'s broad
"could not resolve to a" pattern, so both raised ``GitHubNotFoundError`` and
parked the PR durably — but only the issue-level 404 is permanent. A
repository-level failure is transient (the orchestrator just listed PRs from
this repo), and parking on it wedged the PR for ~32 hours with zero events.

Four fix directions, each covered here:
1. Classify before parking — transient repo-resolution failures route to the
   retry path, not the permanent park.
2. Require confirmation across passes — park only after N>=2 consecutive
   not-founds.
3. Bounded self-heal — re-probe parked markers on a slow cadence; clear if
   the issue now resolves.
4. Make the skip visible — parked PRs surface in ``loop_completed``'s payload.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub

from charlie_work.config import (
    NotifyConfig,
    OrchestratorConfig,
    ReviewConfig,
)
from charlie_work.github import (
    GitHubError,
    GitHubNotFoundError,
    is_transient_repo_resolution_failure,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp


def _foreign_pr_config(**review_kwargs: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        notify=NotifyConfig(enabled=True),
        review=ReviewConfig(**review_kwargs),
    )


def _make_app(
    tmp_path: Path,
    gh: FakeGitHub,
    config: OrchestratorConfig,
) -> OrchestratorApp:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh)


def _foreign_pr(number: int = 1586, issue: int = 1576) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{issue}: stuff",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue}-x",
        "baseRefName": "main",
        "headRefOid": f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


# ---------------------------------------------------------------------------
# Fix 1: classify before parking
# ---------------------------------------------------------------------------


class _RepoResolutionFailureGitHub(FakeGitHub):
    """Simulates a transient repository-level resolution failure.

    ``issue_view`` raises ``GitHubNotFoundError`` with a repository-level
    "Could not resolve to a Repository" message — the exact shape from the
    #1132 incident (a ~7-minute network/ISP dip).
    """

    def __init__(self) -> None:
        super().__init__()
        self.issues = []
        self.prs = [_foreign_pr()]
        self.issue_view_calls = 0

    def issue_view(self, number: int):
        if number == 1576:
            self.issue_view_calls += 1
            raise GitHubNotFoundError(
                "GraphQL: Could not resolve to a Repository with the name "
                "'Senkichi/job-cannon'. (repository)"
            )
        return super().issue_view(number)


def test_transient_repo_resolution_failure_not_parked(tmp_path: Path) -> None:
    """A ``GitHubNotFoundError`` whose message indicates a repository-level
    resolution failure (transient) must NOT park the PR as
    ``foreign_issue_ref``. It routes to the retry path (``errors`` list)
    instead, so the next pass re-attempts the lookup.

    This is the exact shape of the #1132 incident: the repo resolved moments
    ago (``pr_list`` succeeded), so a repository-level failure is a network
    dip, not a permanent foreign-issue condition.
    """
    config = _foreign_pr_config()
    fake_gh = _RepoResolutionFailureGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    result = app.loop(limit=0)

    # The error lands in result.data["errors"] (retry path), not a park.
    assert result.ok is False
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["pr"] == 1586
    assert "Could not resolve to a Repository" in result.data["errors"][0]["error"]

    # No foreign_issue_ref marker was written.
    state = load_state(app.paths.state_file)
    pr_entry = state["prs"].get("1586", {})
    assert "foreign_issue_ref" not in pr_entry

    # No parked PRs.
    assert result.data["parked_prs"] == []


def test_transient_repo_resolution_failure_retries_next_pass(tmp_path: Path) -> None:
    """On the next pass, if the repository resolves again, the PR is processed
    normally — the transient failure did not park it."""
    config = _foreign_pr_config()
    fake_gh = _RepoResolutionFailureGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    # Pass 1: transient failure → error, no park.
    app.loop(limit=0)
    assert fake_gh.issue_view_calls == 1

    # Simulate recovery: issue_view now succeeds.
    fake_gh.issues = [{"number": 1576, "state": "OPEN", "labels": [], "title": "test"}]
    fake_gh.issue_view_calls = 0

    # Pass 2: issue resolves → no error, no park.
    app.loop(limit=0)
    state = load_state(app.paths.state_file)
    assert "foreign_issue_ref" not in state["prs"].get("1586", {})


# ---------------------------------------------------------------------------
# Fix 2: require confirmation across passes
# ---------------------------------------------------------------------------


class _PermanentIssueNotFoundGitHub(FakeGitHub):
    """Simulates a permanent issue-level 404 (genuine foreign issue ref)."""

    def __init__(self) -> None:
        super().__init__()
        self.issues = []
        self.prs = [_foreign_pr()]
        self.issue_view_calls = 0

    def issue_view(self, number: int):
        if number == 1576:
            self.issue_view_calls += 1
            raise GitHubNotFoundError("could not resolve to a Issue with the number 1576.")
        return super().issue_view(number)


def test_permanent_issue_not_found_parks_after_confirmation(tmp_path: Path) -> None:
    """A permanent issue-level 404 parks the PR, but only after
    ``confirm_passes`` (default 2) consecutive not-found passes. The first
    pass writes an unconfirmed marker; the second confirms it and emits the
    one-shot digest.
    """
    config = _foreign_pr_config()
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    captured: list[Any] = []
    # Suppress the digest to avoid network/file I/O in the test.
    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest

    def _capture(notify_config, digest):
        captured.append(digest)

    wf_mod.emit_digest = _capture
    try:
        # Pass 1: unconfirmed marker.
        result1 = app.loop(limit=0)
        assert result1.ok is True
        assert fake_gh.issue_view_calls == 1
        assert len(captured) == 0
        state = load_state(app.paths.state_file)
        assert state["prs"]["1586"]["foreign_issue_ref"]["confirmations"] == 1

        # Pass 2: confirmed, digest emitted.
        result2 = app.loop(limit=0)
        assert result2.ok is True
        assert fake_gh.issue_view_calls == 2
        assert len(captured) == 1
        state = load_state(app.paths.state_file)
        assert state["prs"]["1586"]["foreign_issue_ref"]["confirmations"] == 2

        # Pass 3: skipped, zero GitHub calls.
        result3 = app.loop(limit=0)
        assert result3.ok is True
        assert fake_gh.issue_view_calls == 2
        assert result3.data["parked_prs"] == [1586]
    finally:
        wf_mod.emit_digest = original_emit


def test_confirm_passes_one_parks_immediately(tmp_path: Path) -> None:
    """With ``foreign_issue_ref_confirm_passes=1``, the original one-pass
    park behavior is preserved: the first not-found confirms the marker and
    emits the digest.
    """
    config = _foreign_pr_config(foreign_issue_ref_confirm_passes=1)
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    captured: list[Any] = []

    def _capture(notify_config, digest):
        captured.append(digest)

    wf_mod.emit_digest = _capture
    try:
        result = app.loop(limit=0)
        assert result.ok is True
        assert len(captured) == 1
        state = load_state(app.paths.state_file)
        assert state["prs"]["1586"]["foreign_issue_ref"]["confirmations"] == 1

        # Second pass: skipped.
        result2 = app.loop(limit=0)
        assert result2.data["parked_prs"] == [1586]
        assert fake_gh.issue_view_calls == 1
    finally:
        wf_mod.emit_digest = original_emit


# ---------------------------------------------------------------------------
# Fix 3: bounded self-heal
# ---------------------------------------------------------------------------


def test_self_heal_clears_marker_when_issue_resolves(tmp_path: Path) -> None:
    """A parked ``foreign_issue_ref`` marker is re-probed on a slow cadence.
    If the issue now resolves via ``issue_view``, the marker is cleared, an
    event is emitted, and per-PR processing resumes. A wrong park costs
    hours, not forever.
    """

    class ToggleableGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []
            self.prs = [_foreign_pr()]
            self.issue_view_calls = 0
            self._raise_not_found = True

        def issue_view(self, number: int):
            if number == 1576:
                self.issue_view_calls += 1
                if self._raise_not_found:
                    raise GitHubNotFoundError("could not resolve to a Issue with the number 1576.")
                return {"number": 1576, "state": "OPEN", "labels": [], "title": "test"}
            return super().issue_view(number)

    config = _foreign_pr_config(
        foreign_issue_ref_confirm_passes=1,
        foreign_issue_ref_reprobe_hours=24,
    )
    fake_gh = ToggleableGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Pass 1: park the PR (confirm_passes=1).
        app.loop(limit=0)
        state = load_state(app.paths.state_file)
        assert "foreign_issue_ref" in state["prs"]["1586"]

        # Plant an old detected_at so the reprobe cadence has elapsed.
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
        with wf_mod.state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] = old_ts
            save_state(app.paths.state_file, state)

        # Simulate recovery: issue_view now succeeds.
        fake_gh._raise_not_found = False

        # Pass 2: reprobe fires, issue resolves, marker cleared.
        result2 = app.loop(limit=0)
        state = load_state(app.paths.state_file)
        assert "foreign_issue_ref" not in state["prs"].get("1586", {})

        # A foreign_issue_ref_cleared event was emitted.
        cleared_events = query_events(app.paths.state_file, kind="foreign_issue_ref_cleared")
        assert len(cleared_events) == 1
        assert cleared_events[0]["payload"]["pr_number"] == 1586
        assert cleared_events[0]["payload"]["issue_number"] == 1576

        # The PR is no longer parked.
        assert 1586 not in result2.data.get("parked_prs", [])
    finally:
        wf_mod.emit_digest = original_emit


def test_self_heal_touches_clock_when_still_not_found(tmp_path: Path) -> None:
    """When a reprobe still raises ``GitHubNotFoundError`` (the issue is
    genuinely absent), the re-probe clock is reset so the next check is
    gated from now, not from the original park time. The marker stays.
    """
    from charlie_work.state import state_lock

    config = _foreign_pr_config(
        foreign_issue_ref_confirm_passes=1,
        foreign_issue_ref_reprobe_hours=24,
    )
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Park the PR.
        app.loop(limit=0)

        # Plant an old detected_at.
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] = old_ts
            save_state(app.paths.state_file, state)

        # Pass 2: reprobe fires, issue still not found, clock reset.
        result2 = app.loop(limit=0)
        state = load_state(app.paths.state_file)
        assert "foreign_issue_ref" in state["prs"]["1586"]
        assert "last_reprobe_at" in state["prs"]["1586"]["foreign_issue_ref"]
        # detected_at is preserved.
        assert state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] == old_ts

        # The PR is still parked.
        assert 1586 in result2.data.get("parked_prs", [])
    finally:
        wf_mod.emit_digest = original_emit


class _RepoResolutionFailureOnViewGitHub(FakeGitHub):
    """``issue_view`` raises a transient repository-level resolution failure.

    Used to exercise the reprobe path: a parked marker is re-probed, and the
    reprobe itself hits a transient ``Could not resolve to a Repository``
    failure (the same shape as the #1132 incident). This must NOT reset the
    re-probe clock — a transient failure is not evidence the issue is absent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.issues = []
        self.prs = [_foreign_pr()]
        self.issue_view_calls = 0

    def issue_view(self, number: int):
        if number == 1576:
            self.issue_view_calls += 1
            raise GitHubNotFoundError(
                "GraphQL: Could not resolve to a Repository with the name "
                "'Senkichi/job-cannon'. (repository)"
            )
        return super().issue_view(number)


def test_self_heal_transient_failure_during_reprobe_leaves_clock_untouched(
    tmp_path: Path,
) -> None:
    """A transient repository-level resolution failure occurring *during*
    reprobe must leave the marker and the re-probe clock untouched, matching
    the treatment the main per-PR park decision gives a transient failure.

    Before this fix, the reprobe path's ``except GitHubNotFoundError`` handler
    treated every not-found — including a transient repo-resolution failure —
    as "genuinely absent" and reset ``last_reprobe_at``. That reintroduced the
    #1132 root-cause conflation (bounded by the reprobe cadence, not infinite):
    a transient dip during reprobe would push the next reprobe out by a full
    cadence window even though the issue's absence was never confirmed.
    """
    from charlie_work.state import state_lock

    config = _foreign_pr_config(
        foreign_issue_ref_confirm_passes=1,
        foreign_issue_ref_reprobe_hours=24,
    )
    # Park with a permanent issue-level 404 first.
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Park the PR (confirm_passes=1, permanent issue-level 404).
        app.loop(limit=0)

        # Plant an old detected_at so the reprobe cadence has elapsed.
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] = old_ts
            save_state(app.paths.state_file, state)

        # Swap in a fake that raises a transient repo-resolution failure on
        # issue_view — the reprobe itself hits the #1132 incident shape.
        app.gh = _RepoResolutionFailureOnViewGitHub()

        # Pass 2: reprobe fires, hits a transient repo-resolution failure.
        result2 = app.loop(limit=0)
        state = load_state(app.paths.state_file)
        # Marker still present.
        assert "foreign_issue_ref" in state["prs"]["1586"]
        # The re-probe clock was NOT reset — a transient failure is not
        # evidence the issue is absent, so the next cadence window retries
        # from the same anchor (detected_at), not from now.
        assert "last_reprobe_at" not in state["prs"]["1586"]["foreign_issue_ref"]
        # detected_at is preserved untouched.
        assert state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] == old_ts

        # The PR is still parked.
        assert 1586 in result2.data.get("parked_prs", [])
    finally:
        wf_mod.emit_digest = original_emit


class _PlainGitHubErrorOnViewGitHub(FakeGitHub):
    """``issue_view`` raises a plain ``GitHubError`` (not
    ``GitHubNotFoundError``) — e.g. a network timeout, gh-not-installed, or
    any non-not-found gh failure.

    Used to exercise the reprobe path's ``except GitHubError:`` branch: a
    parked marker is re-probed, and the reprobe itself hits a transient
    non-not-found failure. This must NOT clear the marker and must NOT reset
    the re-probe clock — a transient failure is neither evidence the issue is
    absent nor present, so the next cadence window retries from the same
    anchor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.issues = []
        self.prs = [_foreign_pr()]
        self.issue_view_calls = 0

    def issue_view(self, number: int):
        if number == 1576:
            self.issue_view_calls += 1
            raise GitHubError("GraphQL: network timeout contacting api.github.com")
        return super().issue_view(number)


def test_self_heal_plain_github_error_during_reprobe_leaves_marker_and_clock_untouched(
    tmp_path: Path,
) -> None:
    """A plain ``GitHubError`` (not ``GitHubNotFoundError``) occurring *during*
    reprobe must leave the marker and the re-probe clock untouched, matching
    the ``GitHubNotFoundError`` transient-failure handler's discipline.

    The reprobe block's ``except GitHubError:`` branch is the catch-all for
    non-not-found gh failures (network timeouts, gh-not-installed, etc.). A
    regression there — e.g. clearing the marker or resetting the clock — would
    reintroduce the #1132 bug: a transient failure during reprobe would either
    resume per-PR processing against a still-foreign issue (clearing the
    marker) or push the next reprobe out by a full cadence window (resetting
    the clock) even though the issue's absence was never confirmed.
    """
    from charlie_work.state import state_lock

    config = _foreign_pr_config(
        foreign_issue_ref_confirm_passes=1,
        foreign_issue_ref_reprobe_hours=24,
    )
    # Park with a permanent issue-level 404 first.
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Park the PR (confirm_passes=1, permanent issue-level 404).
        app.loop(limit=0)

        # Plant an old detected_at so the reprobe cadence has elapsed.
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] = old_ts
            save_state(app.paths.state_file, state)

        # Swap in a fake that raises a plain GitHubError on issue_view — the
        # reprobe itself hits a transient non-not-found failure.
        app.gh = _PlainGitHubErrorOnViewGitHub()

        # Pass 2: reprobe fires, hits a plain GitHubError.
        result2 = app.loop(limit=0)
        state = load_state(app.paths.state_file)
        # Marker still present.
        assert "foreign_issue_ref" in state["prs"]["1586"]
        # The re-probe clock was NOT reset — a transient failure is not
        # evidence the issue is absent, so the next cadence window retries
        # from the same anchor (detected_at), not from now.
        assert "last_reprobe_at" not in state["prs"]["1586"]["foreign_issue_ref"]
        # detected_at is preserved untouched.
        assert state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] == old_ts

        # The PR is still parked.
        assert 1586 in result2.data.get("parked_prs", [])
    finally:
        wf_mod.emit_digest = original_emit


def test_self_heal_disabled_when_reprobe_hours_zero(tmp_path: Path) -> None:
    """When ``foreign_issue_ref_reprobe_hours=0``, self-heal is disabled —
    the marker is never re-probed and the PR stays parked."""
    config = _foreign_pr_config(
        foreign_issue_ref_confirm_passes=1,
        foreign_issue_ref_reprobe_hours=0,
    )
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Park the PR.
        app.loop(limit=0)

        # Plant an old detected_at (would trigger reprobe if enabled).
        old_ts = (datetime.now(UTC) - timedelta(hours=999)).isoformat().replace("+00:00", "Z")
        with wf_mod.state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["foreign_issue_ref"]["detected_at"] = old_ts
            save_state(app.paths.state_file, state)

        # issue_view now succeeds — but reprobe is disabled.
        fake_gh.issues = [{"number": 1576, "state": "OPEN", "labels": [], "title": "test"}]

        result2 = app.loop(limit=0)
        state = load_state(app.paths.state_file)
        # Marker still present.
        assert "foreign_issue_ref" in state["prs"]["1586"]
        assert 1586 in result2.data.get("parked_prs", [])
    finally:
        wf_mod.emit_digest = original_emit


# ---------------------------------------------------------------------------
# Fix 4: make the skip visible
# ---------------------------------------------------------------------------


def test_parked_prs_visible_in_loop_completed_event(tmp_path: Path) -> None:
    """Parked PR numbers surface in the ``loop_completed`` event payload so
    "PR untouched for days" is attributable from events.db.
    """
    config = _foreign_pr_config(foreign_issue_ref_confirm_passes=1)
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Pass 1: park.
        app.loop(limit=0)
        # Pass 2: skip (parked).
        app.loop(limit=0)

        # Query the loop_completed events.
        loop_events = query_events(app.paths.state_file, kind="loop_completed")
        # The last loop_completed should have parked_prs.
        last = loop_events[-1]
        assert last["payload"]["parked_prs_count"] == 1
        assert 1586 in last["payload"]["parked_prs"]
    finally:
        wf_mod.emit_digest = original_emit


# ---------------------------------------------------------------------------
# unescalate clears foreign_issue_ref
# ---------------------------------------------------------------------------


def test_unescalate_clears_foreign_issue_ref_marker(tmp_path: Path) -> None:
    """``charlie unescalate --pr`` must clear the ``foreign_issue_ref`` marker
    so the next pass re-probes the linked issue. Before #1132, unescalate
    flipped janitor_blocked -> open_passive but left the park in place, so
    the PR stayed invisible.
    """
    config = _foreign_pr_config(foreign_issue_ref_confirm_passes=1)
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    wf_mod.emit_digest = lambda notify_config, digest: None
    try:
        # Park the PR and mark it as janitor_blocked (the observed scenario).
        app.loop(limit=0)
        with wf_mod.state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"]["status"] = "janitor_blocked"
            save_state(app.paths.state_file, state)

        # Unescalate the PR.
        result = app.unescalate(pr_number=1586)
        assert result.ok is True

        state = load_state(app.paths.state_file)
        assert "foreign_issue_ref" not in state["prs"].get("1586", {})
    finally:
        wf_mod.emit_digest = original_emit


# ---------------------------------------------------------------------------
# Classifier unit test
# ---------------------------------------------------------------------------


def test_is_transient_repo_resolution_failure_classifier() -> None:
    """The classifier distinguishes repository-level (transient) from
    issue-level (permanent) resolution failures."""
    # Repository-level: transient.
    assert is_transient_repo_resolution_failure(
        "GraphQL: Could not resolve to a Repository with the name 'Senkichi/job-cannon'. (repository)"
    )
    assert is_transient_repo_resolution_failure(
        "could not resolve to a repository with the name 'foo/bar'"
    )
    # Issue-level: permanent.
    assert not is_transient_repo_resolution_failure(
        "could not resolve to a Issue with the number 4242."
    )
    # Unrelated: not transient repo failure.
    assert not is_transient_repo_resolution_failure("http 404")
    assert not is_transient_repo_resolution_failure("network timeout")


# ---------------------------------------------------------------------------
# Legacy marker backward-compat (pre-#1132 deploy-time state.json)
# ---------------------------------------------------------------------------


def test_legacy_marker_without_confirmations_skipped_silently(tmp_path: Path) -> None:
    """A pre-#1132 ``foreign_issue_ref`` marker planted directly into
    state.json (no ``confirmations`` key — the shape production state.json
    entries have on deploy) is treated as already-confirmed by the
    ``_loop_body`` early-skip: the PR is skipped with zero ``issue_view``
    calls and no digest re-emission, and the marker is left untouched (no
    ``confirmations`` key is written).

    This is the deploy-time backward-compat contract: existing parks are not
    re-processed, re-confirmed, or re-alerted. It also demonstrates that the
    ``same_issue and prev_conf is None`` branch in ``_mark_foreign_issue_ref``
    is unreachable via the loop — the early-skip (``confirmed =
    confirmations is None or ...``) handles legacy markers before that method
    is ever called, so the marker is never migrated.
    """
    config = _foreign_pr_config()
    fake_gh = _PermanentIssueNotFoundGitHub()
    app = _make_app(tmp_path, fake_gh, config)

    import charlie_work.workflow as wf_mod

    original_emit = wf_mod.emit_digest
    captured: list[Any] = []

    def _capture(notify_config, digest):
        captured.append(digest)

    wf_mod.emit_digest = _capture
    try:
        # Plant a pre-#1132 legacy marker directly into state.json: no
        # ``confirmations`` key. Use a recent detected_at so the reprobe
        # cadence has NOT elapsed (reprobe_hours default 24) and the
        # confirmed-marker skip path is taken without any issue_view call.
        now_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with wf_mod.state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["1586"] = {
                "number": 1586,
                "foreign_issue_ref": {
                    "issue": 1576,
                    "detected_at": now_ts,
                    "reason": "pre-#1132 legacy park",
                },
            }
            save_state(app.paths.state_file, state)

        # Run a loop pass.
        result = app.loop(limit=0)

        # The PR is skipped (parked) with zero issue_view calls — the
        # legacy marker is treated as confirmed and never re-probed (cadence
        # not elapsed) nor re-confirmed.
        assert fake_gh.issue_view_calls == 0
        assert result.data["parked_prs"] == [1586]

        # No digest re-emission — the legacy park was already alerted when
        # first written; it must not re-fire on every pass.
        assert captured == []

        # The marker is unchanged: still no ``confirmations`` key. The
        # early-skip path does not migrate the field, and
        # ``_mark_foreign_issue_ref`` was never called (proving its
        # ``same_issue and prev_conf is None`` branch is unreachable via
        # the loop).
        state = load_state(app.paths.state_file)
        marker = state["prs"]["1586"]["foreign_issue_ref"]
        assert "confirmations" not in marker
        assert marker["issue"] == 1576
        assert marker["detected_at"] == now_ts
    finally:
        wf_mod.emit_digest = original_emit
