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
   fingerprint re-emits, and an ``UNKNOWN`` mergeability window (GitHub still
   computing the real verdict) neither emits nor overwrites the last settled
   value.
2. The wired behavior through a full ``OrchestratorApp.loop()`` pass: first
   sight emits ``pr_unlinked_skipped``, an unchanged second pass emits
   nothing new, a material state change (head SHA moves) re-emits, a
   mergeable/UNKNOWN flap with an unchanged head SHA emits exactly once, and
   a PR with a resolvable linked issue is completely unaffected (no marker,
   no event).
3. ``status()``'s ``unlinked_prs``/``unlinked_pr_count`` surfacing of the
   current standing set, independent of the last transition, and its
   agreement with ``prs``/linked-issue resolution on a stale branch-name PR.
4. Regression checks for the #1766 PR review's BLOCKER findings: a
   state-lock failure while recording the batch of skipped PRs must not
   abort the rest of the pass's review/merge work, and a ``--dry-run`` pass
   must never durably persist the edge-detector marker.
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
    UNLINKED_PR_SKIPPED_EVENT_KIND,
    UnlinkedPrFingerprint,
    compute_unlinked_pr_transition,
    summarize_unlinked_prs,
)
from charlie_work.state import StateLockBusy, load_state
from charlie_work.workflow import OrchestratorApp

# Deliberately imported AFTER `charlie_work.workflow` above (not
# alphabetized with the block above -- do not let a lint pass reorder this
# back next to the other `charlie_work.orchestration`-adjacent imports).
# Every `charlie_work.orchestration` submodule does `import charlie_work.workflow
# as _wf` at ITS OWN top level (the module-object seam every sibling uses).
# Importing this submodule directly BEFORE anything has imported
# `charlie_work.workflow` would recursively trigger `charlie_work.workflow`'s
# own import from inside this submodule's `_wf` line, and `charlie_work.workflow`'s
# own import runs `discover_delegate_modules()`, which would then re-enter and
# return THIS module while it is still mid-import (only its own import
# statements executed so far, none of its `def`s) -- permanently installing
# zero delegates from it onto `OrchestratorApp` for the rest of the process.
# The `OrchestratorApp` import above already fully imports
# `charlie_work.workflow` (which, in turn, fully and correctly imports this
# submodule via `discover_delegate_modules`), so by the time this line runs
# it is just a cheap `sys.modules` lookup.
import charlie_work.orchestration.github_ops_unlinked_prs as github_ops_unlinked_prs

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
    assert event_extra["observed_days"] == 0.0
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
    assert event_extra["observed_days"] == 3.0
    # first_seen_at is the anchor from the ORIGINAL sighting, not reset by
    # the later transition.
    assert new_marker["first_seen_at"] == now.isoformat()
    assert new_marker["last_fingerprint"] == new_fingerprint.as_dict()


def test_transition_unknown_mergeable_window_does_not_reemit() -> None:
    """MAJOR #3 (#1766 PR review): ``pr_list()`` reports ``UNKNOWN`` for both
    ``mergeable`` and ``merge_state_status`` for a window after ANY push
    while GitHub recomputes the real verdict (the same window
    ``state_stale_checks.py`` defers on, issue #1451) -- not just a push to
    this PR's own branches. Without carrying the last settled value forward
    through that window, a standing issue-less PR would "transition" (and
    re-emit) on every UNKNOWN sighting even though nothing an operator cares
    about changed.
    """
    now = datetime(2026, 9, 21, tzinfo=UTC)
    settled = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1"
    )
    marker, _ = compute_unlinked_pr_transition(None, settled, now=now)

    unknown = UnlinkedPrFingerprint(
        mergeable="UNKNOWN", merge_state_status="UNKNOWN", head_sha="sha-1"
    )
    later = datetime(2026, 9, 21, 0, 5, tzinfo=UTC)
    unchanged_marker, event_extra = compute_unlinked_pr_transition(marker, unknown, now=later)

    assert event_extra is None
    assert unchanged_marker == marker
    # The persisted marker keeps the last SETTLED fingerprint, not the
    # UNKNOWN placeholder -- so a later genuinely-settled sighting compares
    # against the real prior state instead of an unsettled one.
    assert unchanged_marker["last_fingerprint"] == settled.as_dict()


def test_transition_head_sha_change_is_always_real_even_when_unsettled() -> None:
    """A new commit is always a material change regardless of whether
    mergeability happens to read UNKNOWN at the same time -- unlike
    ``mergeable``/``merge_state_status``, ``head_sha`` is never carried
    forward by ``_settle_fingerprint``."""
    now = datetime(2026, 9, 21, tzinfo=UTC)
    settled = UnlinkedPrFingerprint(
        mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1"
    )
    marker, _ = compute_unlinked_pr_transition(None, settled, now=now)

    new_commit_unsettled = UnlinkedPrFingerprint(
        mergeable="UNKNOWN", merge_state_status="UNKNOWN", head_sha="sha-2"
    )
    later = datetime(2026, 9, 22, tzinfo=UTC)
    new_marker, event_extra = compute_unlinked_pr_transition(
        marker, new_commit_unsettled, now=later
    )

    assert event_extra is not None
    assert new_marker["last_fingerprint"]["head_sha"] == "sha-2"


def test_summarize_unlinked_prs_respects_branch_issue_validator() -> None:
    """MAJOR #4 (#1766 PR review): a PR whose branch name encodes a stale
    issue number (issue #1229 -- e.g. a merged/closed issue's branch reused
    by an unrelated PR) must resolve as UNLINKED here, exactly as
    ``linked_issue_number`` resolves it everywhere else once a validator
    says the candidate number is not a real open issue."""
    now = datetime(2026, 9, 21, tzinfo=UTC)
    stale_branch_pr: dict[str, Any] = {
        "number": 950,
        "title": "Unrelated change",
        "url": "https://example.test/pull/950",
        "headRefName": "agent/issue-709-old-fix",
        "baseRefName": "main",
        "headRefOid": "sha-950",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "body": "No closing keyword here.",
        "isCrossRepository": False,
        "author": {"login": "some-worker", "type": "User"},
    }

    # No validator supplied: the stale branch number is trusted
    # unconditionally, so this PR resolves as LINKED (to the wrong/closed
    # issue #709) and must NOT appear here.
    unvalidated = summarize_unlinked_prs(
        [stale_branch_pr], {}, branch_prefix="agent/issue", now=now
    )
    assert unvalidated == []

    # With a validator saying issue #709 is not open, the branch-name
    # binding is rejected and the PR falls through to unlinked.
    validated = summarize_unlinked_prs(
        [stale_branch_pr],
        {},
        branch_prefix="agent/issue",
        now=now,
        branch_issue_validator=lambda n: False,
    )
    assert [entry["pr_number"] for entry in validated] == [950]


def test_pr_unlinked_skipped_registered_as_warning() -> None:
    """Regression guard for MINOR #6 (#1766 PR review): ``_instrumentation_kind_scanner``
    only resolves module-level constant assignments WITHIN the file making the
    call, so ``UNLINKED_PR_SKIPPED_EVENT_KIND`` (imported from
    ``pr_unlinked_visibility`` into ``github_ops_unlinked_prs``) is invisible
    to it. The call site also passes an explicit ``level="warning"`` kwarg
    (satisfying the scanner directly), but that alone does not make
    ``charlie status``/heartbeat's warning-level surfaces see this kind --
    those read ``_LEVEL_BY_KIND`` itself, so the kind must be registered here
    too."""
    assert _LEVEL_BY_KIND[UNLINKED_PR_SKIPPED_EVENT_KIND] == "warning"


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
    # MINOR #6: must actually be classified as a warning at the events.db
    # row level, not just resolvable by the static kind-registry scanner.
    assert events[0]["level"] == "warning"

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


def test_unlinked_pr_flap_between_mergeable_and_unknown_emits_once(tmp_path: Path) -> None:
    """MAJOR #3, wired end to end: a PR that flaps
    MERGEABLE -> UNKNOWN -> MERGEABLE -> UNKNOWN -> MERGEABLE across 5 passes
    with an UNCHANGED head SHA must emit ``pr_unlinked_skipped`` exactly
    once (first sight), not once per flap."""
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    app = _make_app(tmp_path, fake_gh)

    sequence = ["MERGEABLE", "UNKNOWN", "MERGEABLE", "UNKNOWN", "MERGEABLE"]
    for mergeable in sequence:
        merge_state = "CLEAN" if mergeable == "MERGEABLE" else "UNKNOWN"
        fake_gh.prs = [_dependabot_pr(mergeable=mergeable, merge_state_status=merge_state)]
        result = app.loop(limit=0)
        assert result.ok is True

    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert len(events) == 1


def test_state_lock_busy_recording_unlinked_skip_does_not_abort_pass(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Regression check for BLOCKER #1 (#1766 PR review): the original
    per-PR ``_record_unlinked_pr_skip`` ran inline at the point of
    ``continue``, so a ``StateLockBusy`` there escaped ``_loop_body``
    uncaught and ``@_guard_state_lock`` converted the WHOLE pass into a
    success-shaped skip -- silently dropping review/merge work already
    completed for every OTHER PR in the same batch. ``_record_unlinked_pr_skips``
    now runs after that work, inside its own try/except, so a lock failure
    here must be swallowed and never surface as ``state_lock_busy`` on the
    pass's own result.
    """
    # Captured BEFORE patching: `github_ops_unlinked_prs._wf` is about to be
    # replaced with the proxy below, so this is the only remaining reference
    # to the real `charlie_work.workflow` module the proxy can delegate to.
    real_wf = github_ops_unlinked_prs._wf

    class _FaultyWf:
        """Proxies every attribute to the real ``charlie_work.workflow``
        module except ``state_lock``, which raises ``StateLockBusy`` --
        faults only THIS module's ``_wf.state_lock`` call.
        ``reap_loop.py``'s separate ``_wf`` name (the module-object seam:
        each importer resolves ``_wf.<name>`` from its own module globals at
        call time) is a different binding to the same module object and is
        untouched by this patch."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real_wf, name)

        @staticmethod
        def state_lock(_path: Path) -> Any:
            raise StateLockBusy("locked for test")

    monkeypatch.setattr(github_ops_unlinked_prs, "_wf", _FaultyWf())

    fake_gh = FakeGitHub()
    fake_gh.issues = [{"number": 123, "title": "Fix search", "state": "OPEN", "labels": []}]
    fake_gh.prs = [_linked_pr(), _dependabot_pr()]
    app = _make_app(tmp_path, fake_gh)

    result = app.loop(limit=0)

    assert result.ok is True
    assert result.data.get("state_lock_busy") is not True
    assert result.data.get("pass_skipped") is not True
    # The linked PR's per-PR work (counted here BEFORE the unlinked-skip
    # recording runs) still completed -- proving the fault in recording the
    # OTHER PR's skip did not abort this PR's own review/merge processing.
    assert result.data["open_tracked_prs"] == 1
    # The unlinked PR's skip was never durably recorded (the lock "failed"),
    # but that failure did not propagate out of the pass.
    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert events == []


def test_dry_run_pass_does_not_persist_marker_and_real_pass_still_emits(
    tmp_path: Path,
) -> None:
    """Regression check for BLOCKER #2 (#1766 PR review): the original
    ``_record_unlinked_pr_skip`` persisted the edge-detector marker via a
    raw ``_wf.save_state`` call that bypassed ``self.write_gate``'s dry-run
    gate, even though ``_record_event`` itself IS gated -- so a single
    ``--dry-run`` preview pass durably wrote ``last_fingerprint`` for an
    event that was never actually emitted, permanently suppressing the real
    first-sight event on every later non-dry-run pass.
    """
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_dependabot_pr()]

    dry_app = _make_app(tmp_path, fake_gh, dry_run=True)
    dry_result = dry_app.loop(limit=0)
    assert dry_result.ok is True

    # A fresh (non-dry-run) app instance against the SAME state file --
    # mirrors production, where `fleet_dispatch.fleet_loop` rebuilds the app
    # per repo per pass.
    real_app = _make_app(tmp_path, fake_gh, dry_run=False)
    real_result = real_app.loop(limit=0)
    assert real_result.ok is True

    events = query_events(real_app.paths.state_file, kind="pr_unlinked_skipped")
    # Exactly one real event -- the FIRST-sight emission on the real pass.
    # Before the fix, the dry-run pass durably wrote the marker via the raw
    # `_wf.save_state` call, so this real pass would see an (incorrectly)
    # unchanged fingerprint and emit NOTHING.
    assert len(events) == 1
    assert events[0]["payload"]["previous_state"] is None


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


def test_status_stale_branch_pr_appears_in_unlinked_not_linked(tmp_path: Path) -> None:
    """MAJOR #4, wired: ``status()`` must resolve ``linked_prs`` and
    ``unlinked_prs`` with the SAME branch-issue validator, or a stale-branch
    PR (#1229 -- a branch name encoding an issue that is not actually open)
    appears in both lists (double-counted) or neither (invisible) instead of
    exactly one."""
    fake_gh = FakeGitHub()
    # Issue #709 does not exist in this repo -- the branch name is stale.
    fake_gh.issues = []
    stale_branch_pr: dict[str, Any] = {
        "number": 950,
        "title": "Unrelated change",
        "url": "https://example.test/pull/950",
        "headRefName": "agent/issue-709-old-fix",
        "baseRefName": "main",
        "headRefOid": "sha-950",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "body": "No closing keyword here.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
        "author": {"login": "some-worker", "type": "User"},
    }
    fake_gh.prs = [stale_branch_pr]
    app = _make_app(tmp_path, fake_gh)

    app.loop(limit=0)
    result = app.status(use_cache=False)

    assert [entry["number"] for entry in result.data["prs"]] == []
    assert [entry["pr_number"] for entry in result.data["unlinked_prs"]] == [950]


def test_summarize_unlinked_prs_reads_state_without_mutating_it() -> None:
    """``summarize_unlinked_prs`` is a read-only view over ``state["prs"]``:
    it must surface an existing marker's ``first_seen_at`` in its output
    (proving it actually reads through the passed-in state, not merely
    ignoring the argument) and must never mutate ``state_prs`` -- ``status()``
    calls this against a state dict a loop pass may read concurrently, so a
    write here would be a correctness hazard, not just a style one."""
    now = datetime(2026, 9, 21, tzinfo=UTC)
    prs = [_dependabot_pr(), _linked_pr()]
    state_prs: dict[str, Any] = {
        "900": {
            UNLINKED_PR_NOTICE_KEY: {
                "first_seen_at": "2026-09-18T00:00:00+00:00",
                "last_fingerprint": {
                    "mergeable": "MERGEABLE",
                    "merge_state_status": "CLEAN",
                    "head_sha": "sha-900",
                },
                "last_emitted_at": "2026-09-18T00:00:00+00:00",
            }
        }
    }
    original = copy.deepcopy(state_prs)

    summary = summarize_unlinked_prs(prs, state_prs, branch_prefix="agent/issue", now=now)

    assert state_prs == original
    assert [entry["pr_number"] for entry in summary] == [900]
    # The pre-existing marker's `first_seen_at` -- not "now" -- proves this
    # reads the passed-in state rather than ignoring it.
    assert summary[0]["first_seen_at"] == "2026-09-18T00:00:00+00:00"
    assert summary[0]["observed_days"] == 3.0
