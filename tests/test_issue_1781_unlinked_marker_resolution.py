"""Tests for issue #1781: the ``unlinked_pr_notice`` marker's missing falling edge.

``pr_unlinked_visibility.compute_unlinked_pr_transition`` is a rising-edge-only
detector invoked only for PRs that are STILL issue-less. Before this fix, a PR
that resolved a linked issue on a later pass (an operator edits the body to add
``Closes #N``, a closing-keyword comment is retrofitted, ...) left its persisted
``unlinked_pr_notice`` marker orphaned under ``state["prs"][<n>]`` forever:
``_record_unlinked_pr_skips`` only ever saw the still-unlinked set, so nothing
deleted the key and no event recorded the resolution.

Covered here:
1. The pure falling-edge helper (``compute_unlinked_pr_resolutions``): it diffs
   this pass's linked set against the standing markers, returns one
   event-payload dict per stale marker (``issue_number`` / ``first_seen_at`` /
   ``observed_days`` carried from the evicted marker), is read-only, and leaves
   markers on still-unlinked or absent PRs untouched.
2. The wired behavior through full ``OrchestratorApp.loop()`` passes: a PR that
   gains ``Closes #N`` gets its marker evicted and a ``pr_unlinked_resolved``
   (info) event emitted exactly once; a sibling still-unlinked PR is untouched;
   a re-unlink restarts the marker lifecycle cleanly; and a ``--dry-run`` pass
   persists no eviction and no event.
3. The pinned boundary of the fix (the issue's design question 3): a marker on
   a PR that leaves the open-PR list entirely (merged/closed while still
   issue-less) stays orphaned -- the same lifecycle ``foreign_issue_ref`` and
   the other per-PR marker keys already have.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub

from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND, query_events
from charlie_work.paths import runtime_paths
from charlie_work.pr_unlinked_visibility import (
    UNLINKED_PR_NOTICE_KEY,
    UNLINKED_PR_RESOLVED_EVENT_KIND,
    compute_unlinked_pr_resolutions,
)
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_app(
    tmp_path: Path,
    gh: FakeGitHub,
    config: OrchestratorConfig | None = None,
    *,
    dry_run: bool = False,
) -> OrchestratorApp:
    config = config or OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh, dry_run=dry_run)


def _dependabot_pr(
    number: int = 900,
    *,
    head_sha: str | None = None,
    body: str = "Bumps requests from 2.28.0 to 2.31.0.",
) -> dict[str, Any]:
    """An issue-less PR in the shape that rotted invisibly (#1766): same-repo,
    Dependabot's own branch convention, no closing keyword."""
    return {
        "number": number,
        "title": "Bump requests from 2.28.0 to 2.31.0",
        "url": f"https://example.test/pull/{number}",
        "headRefName": "dependabot/pip/requests-2.31.0",
        "baseRefName": "main",
        "headRefOid": head_sha or f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "body": body,
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
        "author": {"login": "dependabot", "type": "Bot"},
    }


def _issue(number: int = 123) -> dict[str, Any]:
    return {"number": number, "title": "Fix search", "state": "OPEN", "labels": []}


def _marker(pr_number: int, first_seen_at: str) -> dict[str, Any]:
    """A persisted ``unlinked_pr_notice`` marker in the shape
    ``compute_unlinked_pr_transition`` writes."""
    return {
        "first_seen_at": first_seen_at,
        "last_fingerprint": {
            "mergeable": "MERGEABLE",
            "merge_state_status": "CLEAN",
            "head_sha": f"sha-{pr_number}",
        },
        "last_emitted_at": first_seen_at,
    }


# ---------------------------------------------------------------------------
# Pure falling-edge helper
# ---------------------------------------------------------------------------


def test_resolutions_returns_only_linked_prs_with_markers() -> None:
    """The diff: markers on linked PRs are stale; markers elsewhere are not."""
    now = datetime(2026, 9, 22, tzinfo=UTC)
    state_prs: dict[str, Any] = {
        # Still unlinked (absent from `linked`): must NOT be reported.
        "900": {UNLINKED_PR_NOTICE_KEY: _marker(900, "2026-09-18T00:00:00+00:00")},
        # Linked this pass AND carrying a marker: the stale one.
        "901": {UNLINKED_PR_NOTICE_KEY: _marker(901, "2026-09-18T00:00:00+00:00")},
        # Linked this pass but never issue-less: no marker, nothing to evict.
        "902": {"decision": "approved"},
    }
    linked = {901: 55, 902: 56, 903: 57}
    original = copy.deepcopy(state_prs)

    resolutions = compute_unlinked_pr_resolutions(state_prs, linked, now=now)

    assert [r["pr_number"] for r in resolutions] == [901]
    assert resolutions[0]["issue_number"] == 55
    assert resolutions[0]["first_seen_at"] == "2026-09-18T00:00:00+00:00"
    assert resolutions[0]["observed_days"] == 4.0
    # Read-only contract: the helper diffs, the caller mutates.
    assert state_prs == original


def test_resolutions_empty_when_nothing_linked() -> None:
    """An all-unlinked pass (``linked_prs`` empty) must not evict anything --
    a standing issue-less backlog keeps its markers."""
    now = datetime(2026, 9, 22, tzinfo=UTC)
    state_prs = {"900": {UNLINKED_PR_NOTICE_KEY: _marker(900, "2026-09-18T00:00:00+00:00")}}

    assert compute_unlinked_pr_resolutions(state_prs, {}, now=now) == []


def test_resolutions_sorted_by_pr_number() -> None:
    """Deterministic emission order regardless of dict insertion order."""
    now = datetime(2026, 9, 22, tzinfo=UTC)
    ts = "2026-09-20T00:00:00+00:00"
    state_prs = {str(n): {UNLINKED_PR_NOTICE_KEY: _marker(n, ts)} for n in (910, 905, 907)}
    linked = {910: 1, 905: 2, 907: 3}

    resolutions = compute_unlinked_pr_resolutions(state_prs, linked, now=now)

    assert [r["pr_number"] for r in resolutions] == [905, 907, 910]


def test_resolutions_tolerates_non_dict_marker() -> None:
    """A malformed marker value still counts as stale (it is evicted), with a
    None ``first_seen_at`` rather than a crash."""
    now = datetime(2026, 9, 22, tzinfo=UTC)
    state_prs = {"900": {UNLINKED_PR_NOTICE_KEY: "not-a-dict"}}

    resolutions = compute_unlinked_pr_resolutions(state_prs, {900: 7}, now=now)

    assert [r["pr_number"] for r in resolutions] == [900]
    assert resolutions[0]["first_seen_at"] is None
    assert resolutions[0]["observed_days"] is None


def test_pr_unlinked_resolved_registered_as_info() -> None:
    """The falling edge is a self-heal completion, not a new fault -- info,
    sibling to ``foreign_issue_ref_cleared``."""
    assert _LEVEL_BY_KIND[UNLINKED_PR_RESOLVED_EVENT_KIND] == "info"


# ---------------------------------------------------------------------------
# Wired behavior: full OrchestratorApp.loop() passes
# ---------------------------------------------------------------------------


def test_resolution_evicts_marker_and_emits_resolved_event_once(tmp_path: Path) -> None:
    """The headline bug: pass 1 plants the marker on an issue-less PR; the
    operator then adds ``Closes #123`` to the body and pass 2 must evict the
    marker and emit ``pr_unlinked_resolved`` -- before the fix both were
    impossible (the PR never re-entered the detector)."""
    fake_gh = FakeGitHub()
    fake_gh.issues = [_issue()]
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    result1 = app.loop(limit=0)
    assert result1.ok is True
    state = load_state(app.paths.state_file)
    marker = state["prs"]["900"][UNLINKED_PR_NOTICE_KEY]
    first_seen_at = marker["first_seen_at"]

    # Operator edits the PR body: the closing keyword now binds issue #123.
    fake_gh.prs = [_dependabot_pr(body="Bumps requests.\n\nCloses #123")]
    result2 = app.loop(limit=0)
    assert result2.ok is True

    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY not in (state["prs"].get("900") or {})

    events = query_events(app.paths.state_file, kind="pr_unlinked_resolved")
    assert len(events) == 1
    assert events[0]["level"] == "info"
    payload = events[0]["payload"]
    assert payload["pr_number"] == 900
    assert payload["issue_number"] == 123
    # Carried from the evicted marker -- how long the PR stood issue-less.
    assert payload["first_seen_at"] == first_seen_at
    assert payload["observed_days"] == 0.0

    # The edge fired once: a third pass emits no further resolution and no
    # second skipped event.
    result3 = app.loop(limit=0)
    assert result3.ok is True
    assert len(query_events(app.paths.state_file, kind="pr_unlinked_resolved")) == 1
    assert len(query_events(app.paths.state_file, kind="pr_unlinked_skipped")) == 1


def test_resolution_leaves_other_unlinked_markers_untouched(tmp_path: Path) -> None:
    """Only the resolving PR's marker is evicted; a sibling still-issue-less
    PR keeps its marker and emits nothing."""
    fake_gh = FakeGitHub()
    fake_gh.issues = [_issue()]
    fake_gh.prs = [_dependabot_pr(900), _dependabot_pr(901)]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)
    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["900"]
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["901"]

    fake_gh.prs = [_dependabot_pr(900, body="Bumps.\n\nCloses #123"), _dependabot_pr(901)]
    result = app.loop(limit=0)
    assert result.ok is True

    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY not in (state["prs"].get("900") or {})
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["901"]

    resolved = query_events(app.paths.state_file, kind="pr_unlinked_resolved")
    assert [e["payload"]["pr_number"] for e in resolved] == [900]
    # The still-unlinked sibling emitted exactly its original first-sight
    # event and no transition since.
    skipped = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert len([e for e in skipped if e["payload"]["pr_number"] == 901]) == 1


def test_reunlinked_pr_restarts_marker_lifecycle(tmp_path: Path) -> None:
    """After eviction, a PR that loses its link again is a fresh first sight:
    new marker, new ``pr_unlinked_skipped`` with ``previous_state=None`` --
    the cleared marker must not leak forward as a bogus prior fingerprint."""
    fake_gh = FakeGitHub()
    fake_gh.issues = [_issue()]
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)
    fake_gh.prs = [_dependabot_pr(body="Bumps.\n\nCloses #123")]
    app.loop(limit=0)
    # Body reverted: issue-less again.
    fake_gh.prs = [_dependabot_pr()]
    result = app.loop(limit=0)
    assert result.ok is True

    skipped = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert len(skipped) == 2
    assert skipped[1]["payload"]["previous_state"] is None
    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["900"]


def test_marker_for_pr_leaving_open_list_remains_orphaned(tmp_path: Path) -> None:
    """Pinned boundary (issue #1781 design Q3): a PR that drops off the
    open-PR list entirely (merged/closed while still issue-less) keeps its
    marker -- the same permanent-but-bounded lifecycle ``foreign_issue_ref``
    and the other per-PR marker keys already have, and the same low-volume
    growth the issue sanctions leaving alone."""
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)
    # PR closed while still issue-less -> absent from pr_list() entirely.
    fake_gh.prs = []
    result = app.loop(limit=0)
    assert result.ok is True

    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["900"]
    assert query_events(app.paths.state_file, kind="pr_unlinked_resolved") == []


def test_dry_run_does_not_evict_marker_or_emit(tmp_path: Path) -> None:
    """The eviction shares ``write_gate.save_state``'s dry-run gate: a
    ``--dry-run`` pass where the PR resolved must leave the marker (and emit
    no event), so the next real pass still performs the resolution edge."""
    fake_gh = FakeGitHub()
    fake_gh.issues = [_issue()]
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)
    app.loop(limit=0)

    fake_gh.prs = [_dependabot_pr(body="Bumps.\n\nCloses #123")]

    dry_app = _make_app(tmp_path, fake_gh, dry_run=True)
    assert dry_app.loop(limit=0).ok is True
    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY in state["prs"]["900"]
    assert query_events(app.paths.state_file, kind="pr_unlinked_resolved") == []

    # The real pass then performs the edge exactly once.
    real_app = _make_app(tmp_path, fake_gh)
    assert real_app.loop(limit=0).ok is True
    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY not in (state["prs"].get("900") or {})
    assert len(query_events(app.paths.state_file, kind="pr_unlinked_resolved")) == 1
