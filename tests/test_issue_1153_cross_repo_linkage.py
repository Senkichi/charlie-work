"""Tests for issue #1153: cross-repo issue/PR linkage is invisible to reconcile
and dispatch.

Two coupled blind spots, same root cause: nothing in the pipeline models an
issue and its fixing PR living in different repos.

Blind spot 1 (PR side, this repo): a same-repo PR whose closing-keyword
reference resolves to an issue number that does not exist in this repo (e.g.
``Closes #1497`` where 1497 is a sibling repo's issue) is invisible to every
issue-driven sweep. ``linked_issue_number`` returned a number, so the PR was
added to ``open_prs_by_issue[issue_number]``, but ``issues_by_number`` has no
entry for it -- no issue-side normalization, no review routing, no merge lane
ever sees it. The PR silently becomes ``open_passive`` (if tracked) or is
simply never tracked at all.

The fix (reconcile.py): when an OPEN PR has a ``linked_issue_number`` that
does not exist in the issue snapshot, emit a ``pr_linked_issue_not_in_repo``
drift item. ``apply_fixes`` tracks the PR in state with the cross-repo
``issue_number`` (for visibility and self-healing) but does NOT escalate --
the existing ``foreign_issue_ref`` parking mechanism in the per-PR review
loop (``_mark_foreign_issue_ref``) already handles parking and one-shot
digest alerting when ``issue_view`` raises ``GitHubNotFoundError``.
Escalating in reconcile would pre-empt that mechanism: ``review()`` treats
``status="escalated"`` as terminal and returns early, so ``issue_view`` is
never called and the foreign-PR digest is never emitted. The ``reconcile``
event (emitted for every drift item) is the visible signal that makes the
cross-repo linkage failure visible to operators and the janitor. A
self-healing condition in ``detect_drift`` skips re-emitting once the PR is
already tracked with the cross-repo ``issue_number``, so the drift item
does not re-fire on every pass.

Blind spot 2 (issue side, dispatch loop): an issue whose prior dispatch
attempts all show ``ahead_of_main: 0`` (the worker hopped to a sibling repo,
did the work there, and exited with zero commits in this repo's tree) is
swept as a dead worker with no open PR, relabeled to ``automated-ready``, and
redispatched indefinitely.

The fix (workflow.py): before the orphan sweep relabels to ``automated-ready``
for another redispatch, check the post-mortem sidecar. If ``>= 2`` attempts
all have ``ahead_of_main == 0``, escalate to ``agent:human-needed`` instead.

Evidence pair for the regression test: cw PR #1147 + jc #1497 (2026-08-12).
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import apply_fixes, detect_drift
from charlie_work.state import load_state, save_state

from test_reconcile import FakeGitHub, _issue, _pr


# ---------------------------------------------------------------------------
# Change 1: cross-repo linkage failure detection in reconcile
# ---------------------------------------------------------------------------


def test_open_pr_with_cross_repo_closing_ref_emits_linkage_drift() -> None:
    """A same-repo OPEN PR whose ``Closes #N`` reference resolves to an issue
    that does not exist in this repo must produce a
    ``pr_linked_issue_not_in_repo`` drift item, not be silently invisible.

    This is the exact shape of cw PR #1147: its body said ``Closes #1497``,
    but 1497 is a job-cannon issue number, not a charlie-work issue.
    ``linked_issue_number`` returned 1497, but ``issues_by_number`` had no
    entry for it -- the PR was invisible to every issue-driven sweep.
    """
    config = OrchestratorConfig()
    # PR #1147: same-repo, body says "Closes #1497" (a sibling repo's issue).
    # The issue snapshot contains only issue #123 (the default cw issue) --
    # issue #1497 is absent, simulating a cross-repo reference.
    gh = FakeGitHub(
        prs=[
            _pr(
                1147,
                "OPEN",
                head_ref="agent/issue-1497-fix-review-pipeline",
                body="Closes #1497\n\nFixes the review pipeline's blindness.",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {"issues": {}, "prs": {}, "events": []}

    drift = detect_drift(gh, state, config)

    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 1
    item = linkage_items[0]
    assert item.pr_number == 1147
    assert item.issue_number == 1497
    assert "1497" in item.detail
    assert "no such issue exists in this repo" in item.detail


def test_open_pr_with_resolvable_closing_ref_does_not_emit_linkage_drift() -> None:
    """A same-repo OPEN PR whose ``Closes #N`` reference resolves to an issue
    that DOES exist in this repo must NOT produce a
    ``pr_linked_issue_not_in_repo`` drift item -- the linkage is valid.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                456,
                "OPEN",
                head_ref="agent/issue-123-fix-search",
                body="Closes #123\n\nTests: regression coverage added.",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {"issues": {}, "prs": {}, "events": []}

    drift = detect_drift(gh, state, config)

    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 0


def test_linkage_drift_apply_fixes_tracks_pr_and_emits_event(tmp_path: Path) -> None:
    """``apply_fixes`` must track the PR in state (so it is visible to future
    sweeps) and emit a ``reconcile`` event that surfaces the cross-repo
    linkage failure. The PR is tracked with the cross-repo ``issue_number``
    but is NOT escalated -- the existing ``foreign_issue_ref`` parking
    mechanism in the per-PR review loop (``_mark_foreign_issue_ref``) handles
    parking and one-shot digest alerting when ``issue_view`` raises
    ``GitHubNotFoundError``. Escalating in reconcile would pre-empt that
    mechanism (``review()`` treats ``status="escalated"`` as terminal and
    returns early, so ``issue_view`` is never called and the digest is never
    emitted) and would break the ``dispatch_reviews`` lane (which skips
    escalated PRs).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(
        prs=[
            _pr(
                1147,
                "OPEN",
                head_ref="agent/issue-1497-fix-review-pipeline",
                body="Closes #1497\n\nFixes the review pipeline.",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {"issues": {}, "prs": {}, "events": []}
    save_state(paths.state_file, state)

    drift = detect_drift(gh, state, config)
    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 1

    new_state = apply_fixes(
        gh,
        state,
        linkage_items,
        config,
        state_path=paths.state_file,
    )

    # The PR must be tracked in state with the linked issue number.
    pr_entry = new_state["prs"].get("1147")
    assert pr_entry is not None
    assert pr_entry["number"] == 1147
    assert pr_entry["issue_number"] == 1497
    # The PR must NOT be escalated -- the existing ``foreign_issue_ref``
    # parking mechanism handles that in the per-PR review loop, and
    # escalating here would pre-empt it (see docstring above).
    assert pr_entry.get("status") != "escalated"
    assert "escalation_reason" not in pr_entry

    # A reconcile event must be emitted -- the visible signal.
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "pr_linked_issue_not_in_repo"
    assert reconcile_events[0]["payload"]["pr_number"] == 1147
    assert reconcile_events[0]["payload"]["issue_number"] == 1497


def test_linkage_drift_does_not_overwrite_existing_active_status() -> None:
    """If the PR is already tracked with a meaningful active status (e.g.
    ``reviewing``), ``apply_fixes`` must NOT overwrite it -- it only adds
    the cross-repo ``issue_number`` to the existing entry, preserving the
    active status for the per-PR review loop (which may park the PR via
    ``foreign_issue_ref`` when ``issue_view`` raises).
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1147,
                "OPEN",
                head_ref="agent/issue-1497-fix-review-pipeline",
                body="Closes #1497",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {
        "issues": {},
        "prs": {"1147": {"number": 1147, "status": "reviewing"}},
        "events": [],
    }

    drift = detect_drift(gh, state, config)
    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 1

    new_state = apply_fixes(gh, state, linkage_items, config)

    pr_entry = new_state["prs"]["1147"]
    assert pr_entry["status"] == "reviewing"
    assert pr_entry["issue_number"] == 1497


def test_linkage_drift_self_heals_after_apply_fixes(tmp_path: Path) -> None:
    """After ``apply_fixes`` has tracked the PR in state with the cross-repo
    ``issue_number``, a second ``detect_drift`` pass must NOT re-emit the
    ``pr_linked_issue_not_in_repo`` drift item.

    Without the self-healing condition in ``detect_drift``, this drift kind
    would re-fire on every reconcile pass for as long as the PR stays open
    (the issue never appears in this repo's snapshot), causing
    ``apply_fixes`` to emit a duplicate ``reconcile`` event each pass --
    unboundedly filling the capped events ring and ``events.db``. This test
    mirrors the self-healing pattern every sibling drift kind in the same
    function relies on (e.g. ``pr_status_normalized`` skips once status is no
    longer None).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(
        prs=[
            _pr(
                1147,
                "OPEN",
                head_ref="agent/issue-1497-fix-review-pipeline",
                body="Closes #1497\n\nFixes the review pipeline.",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {"issues": {}, "prs": {}, "events": []}
    save_state(paths.state_file, state)

    # First pass: detect + apply.
    drift = detect_drift(gh, state, config)
    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 1

    new_state = apply_fixes(
        gh,
        state,
        linkage_items,
        config,
        state_path=paths.state_file,
    )

    # Second pass: detect_drift on the post-apply state must not re-emit.
    second_drift = detect_drift(gh, new_state, config)
    second_linkage = [d for d in second_drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(second_linkage) == 0


def test_linkage_drift_self_heals_with_existing_active_status() -> None:
    """When the PR is already tracked with an active status (e.g.
    ``reviewing``) but no ``issue_number``, the first ``detect_drift`` pass
    emits the drift item (the issue_number is missing from state). After
    ``apply_fixes`` adds the ``issue_number`` (without overwriting the
    existing status), a second ``detect_drift`` pass must NOT re-emit --
    the PR is now tracked with the cross-repo ``issue_number``.
    """
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(
                1147,
                "OPEN",
                head_ref="agent/issue-1497-fix-review-pipeline",
                body="Closes #1497",
            )
        ],
        issues=[_issue(123, [config.labels.ready])],
    )
    state = {
        "issues": {},
        "prs": {"1147": {"number": 1147, "status": "reviewing"}},
        "events": [],
    }

    # First pass: emits because issue_number is not yet in the PR entry.
    drift = detect_drift(gh, state, config)
    linkage_items = [d for d in drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(linkage_items) == 1

    new_state = apply_fixes(gh, state, linkage_items, config)
    assert new_state["prs"]["1147"]["status"] == "reviewing"
    assert new_state["prs"]["1147"]["issue_number"] == 1497

    # Second pass: must not re-emit.
    second_drift = detect_drift(gh, new_state, config)
    second_linkage = [d for d in second_drift if d.kind == "pr_linked_issue_not_in_repo"]
    assert len(second_linkage) == 0


# ---------------------------------------------------------------------------
# Change 2: zero-artifact dispatch loop escalation
# ---------------------------------------------------------------------------


def _write_post_mortem_sidecar(
    sessions_dir: Path,
    issue_number: int,
    attempts: list[dict[str, Any]],
) -> None:
    """Write a post-mortem sidecar with the given attempt records."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"issue-{issue_number}.post-mortem.json"
    payload = {
        "issue_number": issue_number,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "db_path": "",
        "matched": False,
        "attempts": attempts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_is_zero_artifact_dispatch_loop_true_when_all_attempts_zero() -> None:
    """``_is_zero_artifact_dispatch_loop`` returns True when there are >= 2
    attempts and every attempt's ``ahead_of_main`` is 0.
    """
    from charlie_work.workflow import _is_zero_artifact_dispatch_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        _write_post_mortem_sidecar(
            sessions_dir,
            1497,
            [
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-1",
                    "ahead_of_main": 0,
                    "recorded_at": "2026-08-12T13:11:00Z",
                },
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-2",
                    "ahead_of_main": 0,
                    "recorded_at": "2026-08-12T14:00:00Z",
                },
            ],
        )
        assert _is_zero_artifact_dispatch_loop(sessions_dir, 1497) is True


def test_is_zero_artifact_dispatch_loop_false_when_one_attempt() -> None:
    """A single zero-artifact attempt is not yet a loop -- the threshold is 2."""
    from charlie_work.workflow import _is_zero_artifact_dispatch_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        _write_post_mortem_sidecar(
            sessions_dir,
            1497,
            [
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-1",
                    "ahead_of_main": 0,
                    "recorded_at": "2026-08-12T13:11:00Z",
                },
            ],
        )
        assert _is_zero_artifact_dispatch_loop(sessions_dir, 1497) is False


def test_is_zero_artifact_dispatch_loop_false_when_any_attempt_nonzero() -> None:
    """If any attempt has ``ahead_of_main != 0``, the loop is not all-zero --
    at least one attempt produced real artifacts."""
    from charlie_work.workflow import _is_zero_artifact_dispatch_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        _write_post_mortem_sidecar(
            sessions_dir,
            1497,
            [
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-1",
                    "ahead_of_main": 0,
                    "recorded_at": "2026-08-12T13:11:00Z",
                },
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-2",
                    "ahead_of_main": 3,
                    "recorded_at": "2026-08-12T14:00:00Z",
                },
            ],
        )
        assert _is_zero_artifact_dispatch_loop(sessions_dir, 1497) is False


def test_is_zero_artifact_dispatch_loop_false_when_no_sidecar() -> None:
    """No post-mortem sidecar means no evidence of a zero-artifact loop."""
    from charlie_work.workflow import _is_zero_artifact_dispatch_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        assert _is_zero_artifact_dispatch_loop(sessions_dir, 1497) is False


def test_is_zero_artifact_dispatch_loop_false_when_ahead_of_main_none() -> None:
    """An attempt with ``ahead_of_main: None`` is ambiguous (the count could
    not be computed) -- do not escalate on ambiguous evidence."""
    from charlie_work.workflow import _is_zero_artifact_dispatch_loop

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        _write_post_mortem_sidecar(
            sessions_dir,
            1497,
            [
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-1",
                    "ahead_of_main": None,
                    "recorded_at": "2026-08-12T13:11:00Z",
                },
                {
                    "ref": "refs/charlie/attempts/issue-1497/attempt-2",
                    "ahead_of_main": 0,
                    "recorded_at": "2026-08-12T14:00:00Z",
                },
            ],
        )
        assert _is_zero_artifact_dispatch_loop(sessions_dir, 1497) is False


def test_orphan_sweep_escalates_zero_artifact_loop(tmp_path: Path) -> None:
    """The orphan sweep must escalate to ``agent:human-needed`` instead of
    relabeling to ``automated-ready`` when the post-mortem sidecar shows
    a repeated zero-artifact dispatch loop (all attempts ``ahead_of_main: 0``).

    This is the exact shape of jc #1497: 10+ duplicate dispatch attempts, each
    producing ``ahead_of_main: 0`` in the jc tree, swept as
    ``dead_worker_no_open_pr_orphan_sweep``, relabeled, redispatched.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from test_charlie_work import FakeGitHub as CWFakeGitHub

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1497"] = {
        "status": "dispatched",
        "dispatched_at": "2026-08-12T13:11:00Z",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_post_mortem_sidecar(
        sessions_dir,
        1497,
        [
            {
                "ref": "refs/charlie/attempts/issue-1497/attempt-1",
                "ahead_of_main": 0,
                "recorded_at": "2026-08-12T13:11:00Z",
            },
            {
                "ref": "refs/charlie/attempts/issue-1497/attempt-2",
                "ahead_of_main": 0,
                "recorded_at": "2026-08-12T14:00:00Z",
            },
        ],
    )

    fake_gh = CWFakeGitHub()
    fake_gh.issues = [
        {
            "number": 1497,
            "title": "review pipeline blind to merge-conflicting PRs",
            "url": "https://example.test/issues/1497",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["1497"]

    # The issue must be escalated, not relabeled to ready.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "zero_artifact_dispatch_loop"

    # The ready label must NOT have been added.
    assert (1497, config.labels.ready) not in fake_gh.labels_added

    # The human-needed label must have been added.
    assert (1497, config.labels.human_needed) in fake_gh.labels_added

    # A session_failed_escalated event must be emitted with the reason.
    events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "zero_artifact_dispatch_loop"


def test_orphan_sweep_relabels_when_not_zero_artifact_loop(tmp_path: Path) -> None:
    """When the post-mortem sidecar does NOT show a zero-artifact loop (e.g.
    only one attempt, or an attempt with non-zero ``ahead_of_main``), the
    orphan sweep must still relabel to ``automated-ready`` as before.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from test_charlie_work import FakeGitHub as CWFakeGitHub

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "dispatched_at": "2026-08-12T13:11:00Z",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    # Only one attempt -- not yet a loop.
    _write_post_mortem_sidecar(
        sessions_dir,
        207,
        [
            {
                "ref": "refs/charlie/attempts/issue-207/attempt-1",
                "ahead_of_main": 0,
                "recorded_at": "2026-08-12T13:11:00Z",
            },
        ],
    )

    fake_gh = CWFakeGitHub()
    fake_gh.issues = [
        {
            "number": 207,
            "title": "some issue",
            "url": "https://example.test/issues/207",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # The issue must NOT be escalated -- it should be relabeled to ready.
    assert entry["status"] != "escalated"
    assert (207, config.labels.ready) in fake_gh.labels_added
    assert (207, config.labels.human_needed) not in fake_gh.labels_added
