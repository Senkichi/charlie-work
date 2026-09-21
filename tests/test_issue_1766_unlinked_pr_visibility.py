"""Tests for issue #1766: the merge lane's per-PR loop silently skips any
open PR with no resolvable linked issue.

Before this fix, ``reap_loop._loop_body`` fired a bare ``continue`` the
instant ``issue_linking.linked_issue_number`` returned ``None`` -- no event,
no operator-facing signal. A Dependabot PR (its own branch/body convention
never resolves an issue) or any other issue-less PR could rot unreviewed
indefinitely with nothing in ``events.db`` to show for it.

Covered here:
1. The pure edge-detector (``pr_unlinked_visibility.compute_unlinked_pr_transition``):
   first sight emits, an unchanged fingerprint does not, a changed
   fingerprint re-emits.
2. The wired behavior through a full ``OrchestratorApp.loop()`` pass: first
   sight emits ``pr_unlinked_skipped``, an unchanged second pass emits
   nothing new, a material state change (head SHA moves) re-emits, and a PR
   with a resolvable linked issue is completely unaffected (no marker, no
   event).
3. ``status()``'s ``unlinked_prs``/``unlinked_pr_count`` surfacing of the
   current standing set, independent of the last transition.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub

from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.pr_unlinked_visibility import (
    UNLINKED_PR_NOTICE_KEY,
    UnlinkedPrFingerprint,
    compute_unlinked_pr_transition,
    summarize_unlinked_prs,
)
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_app(
    tmp_path: Path, gh: FakeGitHub, config: OrchestratorConfig | None = None
) -> OrchestratorApp:
    config = config or OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh)


def _dependabot_pr(
    number: int = 900,
    *,
    head_sha: str = "sha-900",
    mergeable: str = "MERGEABLE",
    merge_state_status: str = "CLEAN",
) -> dict[str, Any]:
    """An issue-less PR in the exact shape that rotted invisibly (#1766):
    same-repo, Dependabot's own branch convention, no closing keyword.
    """
    return {
        "number": number,
        "title": "Bump requests from 2.28.0 to 2.31.0",
        "url": f"https://example.test/pull/{number}",
        "headRefName": "dependabot/pip/requests-2.31.0",
        "baseRefName": "main",
        "headRefOid": head_sha,
        "mergeStateStatus": merge_state_status,
        "mergeable": mergeable,
        "body": "Bumps requests from 2.28.0 to 2.31.0.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
        "author": {"login": "dependabot", "type": "Bot"},
    }


def _linked_pr(number: int = 456, issue: int = 123) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{issue}: search",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue}-fix-search",
        "baseRefName": "main",
        "headRefOid": f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "body": f"Closes #{issue}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
        "author": {"login": "some-worker", "type": "User"},
    }


# ---------------------------------------------------------------------------
# Pure edge-detector
# ---------------------------------------------------------------------------


def test_transition_first_sight_emits() -> None:
    now = datetime(2026, 9, 21, tzinfo=UTC)
    fingerprint = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1"
    )

    new_marker, event_extra = compute_unlinked_pr_transition(None, fingerprint, now=now)

    assert event_extra is not None
    assert event_extra["previous_state"] is None
    assert event_extra["age_days"] == 0.0
    assert new_marker["first_seen_at"] == now.isoformat()
    assert new_marker["last_fingerprint"] == fingerprint.as_dict()


def test_transition_unchanged_does_not_emit() -> None:
    now = datetime(2026, 9, 21, tzinfo=UTC)
    fingerprint = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1"
    )
    marker, _ = compute_unlinked_pr_transition(None, fingerprint, now=now)

    later = datetime(2026, 9, 22, tzinfo=UTC)
    unchanged_marker, event_extra = compute_unlinked_pr_transition(marker, fingerprint, now=later)

    assert event_extra is None
    # The caller must be handed back the marker verbatim (no rewrite) on a
    # no-op pass -- asserted by identity-of-content, not just "truthy".
    assert unchanged_marker == marker


def test_transition_state_change_reemits_and_preserves_first_seen() -> None:
    now = datetime(2026, 9, 21, tzinfo=UTC)
    fingerprint = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1"
    )
    marker, _ = compute_unlinked_pr_transition(None, fingerprint, now=now)

    later = datetime(2026, 9, 24, tzinfo=UTC)
    new_fingerprint = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-2"
    )
    new_marker, event_extra = compute_unlinked_pr_transition(marker, new_fingerprint, now=later)

    assert event_extra is not None
    assert event_extra["previous_state"] == fingerprint.as_dict()
    assert event_extra["age_days"] == 3.0
    # first_seen_at is the anchor from the ORIGINAL sighting, not reset by
    # the later transition.
    assert new_marker["first_seen_at"] == now.isoformat()
    assert new_marker["last_fingerprint"] == new_fingerprint.as_dict()


# ---------------------------------------------------------------------------
# Wired behavior: a full OrchestratorApp.loop() pass
# ---------------------------------------------------------------------------


def test_unlinked_pr_emits_event_on_first_pass(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    result = app.loop(limit=0)

    assert result.ok is True
    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 900
    assert payload["author"] == "dependabot"
    assert payload["is_bot"] is True
    assert payload["previous_state"] is None

    state = load_state(app.paths.state_file)
    marker = state["prs"]["900"][UNLINKED_PR_NOTICE_KEY]
    assert marker["last_fingerprint"] == {
        "mergeable": "MERGEABLE",
        "merge_state_status": "CLEAN",
        "head_sha": "sha-900",
    }


def test_unlinked_pr_unchanged_second_pass_does_not_reemit(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)
    result2 = app.loop(limit=0)

    assert result2.ok is True
    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    # Still exactly one -- the second, identical pass wrote nothing.
    assert len(events) == 1


def test_unlinked_pr_state_change_reemits(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_dependabot_pr(head_sha="sha-900")]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)

    # New commit pushed to the Dependabot branch -- head SHA moves.
    fake_gh.prs = [_dependabot_pr(head_sha="sha-901")]
    app.loop(limit=0)

    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert len(events) == 2
    assert events[0]["payload"]["head_sha"] == "sha-900"
    assert events[1]["payload"]["head_sha"] == "sha-901"
    assert events[1]["payload"]["previous_state"] == {
        "mergeable": "MERGEABLE",
        "merge_state_status": "CLEAN",
        "head_sha": "sha-900",
    }


def test_linked_pr_never_gets_unlinked_marker_or_event(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = [{"number": 123, "title": "Fix search", "state": "OPEN", "labels": []}]
    fake_gh.prs = [_linked_pr()]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)

    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert events == []
    state = load_state(app.paths.state_file)
    assert UNLINKED_PR_NOTICE_KEY not in (state["prs"].get("456") or {})


# ---------------------------------------------------------------------------
# status() surfacing
# ---------------------------------------------------------------------------


def test_status_surfaces_unlinked_prs(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = [{"number": 123, "title": "Fix search", "state": "OPEN", "labels": []}]
    fake_gh.prs = [_linked_pr(), _dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    # Run a pass first so the standing marker exists, matching production
    # sequencing (status() is read against a state a loop pass has touched).
    app.loop(limit=0)

    result = app.status(use_cache=False)

    assert result.data["unlinked_pr_count"] == 1
    assert [entry["pr_number"] for entry in result.data["unlinked_prs"]] == [900]
    assert result.data["unlinked_prs"][0]["author"] == "dependabot"
    assert result.data["unlinked_prs"][0]["is_bot"] is True
    # The linked PR must appear in the existing `prs` list, not here.
    # (`_summarize_pr`'s shape uses "number", not "pr_number".)
    assert [entry["number"] for entry in result.data["prs"]] == [456]


def test_summarize_unlinked_prs_pure_no_state_writes(tmp_path: Path) -> None:
    """`summarize_unlinked_prs` is a read-only view: calling it never touches
    `state["prs"]`, so `status()` reads (which may run on read replicas /
    concurrently with a loop pass) can never race a write."""
    now = datetime(2026, 9, 21, tzinfo=UTC)
    prs = [_dependabot_pr(), _linked_pr()]
    state_prs: dict[str, Any] = {}

    summary = summarize_unlinked_prs(prs, state_prs, branch_prefix="agent/issue", now=now)

    assert state_prs == {}
    assert [entry["pr_number"] for entry in summary] == [900]
