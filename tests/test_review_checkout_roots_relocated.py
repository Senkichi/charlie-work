"""Orphan-review-checkout reap across the relocated checkout root (issue #3224).

``_reap_orphaned_review_checkouts`` scans both ``review_checkouts`` (the host
I/O volume root) and ``legacy_review_checkouts`` (``reviews_dir``, where
pre-switch checkouts remain); the liveness bookkeeping half stays in
``reviews_dir`` only. Split out of ``test_review_checkout_reap.py``, which is
over the file-size cap.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.state import empty_state, save_state
from charlie_work.workflow import _reap_orphaned_review_checkouts
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _merged_pr_gh() -> FakeGitHub:
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-1-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]
    return fake_gh


def test_reap_orphaned_review_checkouts_scans_the_legacy_root(monkeypatch, tmp_path: Path) -> None:
    """Issue #3224: a ``pr-<N>`` checkout left under ``reviews_dir`` from
    before the checkout root moved to the host I/O volume is still found and
    reaped -- the orphan scan covers both roots, and the teardown forwards
    them both."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkouts = tmp_path / "volume" / "repo" / "reviews"
    (reviews_dir / "pr-100").mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    save_state(state_file, empty_state())

    removed_calls: list[dict] = []

    def fake_remove_review_checkout(repo_root_arg: Path, pr_number: int, **kwargs) -> bool:
        removed_calls.append({"pr": pr_number, **kwargs})
        return True

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout",
        fake_remove_review_checkout,
    )

    reaped = _reap_orphaned_review_checkouts(
        _merged_pr_gh(),
        repo_root,
        reviews_dir,
        state_file,
        config,
        write_gate=_wg(state_file),
        review_checkouts=checkouts,
        legacy_review_checkouts=reviews_dir,
    )

    assert reaped == [100]
    assert removed_calls == [
        {
            "pr": 100,
            "reviews_dir": reviews_dir,
            "review_checkouts": checkouts,
            "legacy_review_checkouts": reviews_dir,
        }
    ]


def test_reap_orphaned_review_checkouts_liveness_reads_only_reviews_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """The relocated checkout root holds no bookkeeping: a sidecar under it
    is not a liveness signal (only ``reviews_dir`` is), so a ``pr-<N>``
    checkout on the volume is reaped for a merged PR despite the stray
    sidecar."""
    from charlie_work.claude_code import _get_process_start_time

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkouts = tmp_path / "volume" / "repo" / "reviews"
    (checkouts / "pr-100").mkdir(parents=True)
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    save_state(state_file, empty_state())

    # A live-pid sidecar under the checkouts root must NOT defer the reap:
    # liveness bookkeeping lives in reviews_dir only.
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-1-fix",
        "worktree_path": str(checkouts / "pr-100"),
        "prompt_path": "",
        "command": ["claude", "-p"],
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": "",
        "error": None,
        "process_start_time": _get_process_start_time(os.getpid()),
    }
    (checkouts / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    removed_calls: list[int] = []

    def fake_remove_review_checkout(repo_root_arg: Path, pr_number: int, **kwargs) -> bool:
        removed_calls.append(pr_number)
        return True

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout",
        fake_remove_review_checkout,
    )

    reaped = _reap_orphaned_review_checkouts(
        _merged_pr_gh(),
        repo_root,
        reviews_dir,
        state_file,
        config,
        write_gate=_wg(state_file),
        review_checkouts=checkouts,
        legacy_review_checkouts=reviews_dir,
    )

    assert reaped == [100]
    assert removed_calls == [100]


# -- kwargs forwarding and scan-root edge coverage (review rework findings) ----


def test_review_checkout_kwargs_filters_no_op_roots(tmp_path: Path) -> None:
    """``review_checkout_kwargs`` emits a key only when the value changes the
    outcome -- the historical ``reviews_dir=``-only call shape survives
    whenever the resolved roots coincide, which is what test doubles with a
    narrow signature on the checkout seams accept.
    """
    from charlie_work.worktree import review_checkout_kwargs

    reviews_dir = tmp_path / "reviews"
    checkouts = tmp_path / "volume" / "repo" / "reviews"

    # Nothing resolved differently: empty dict, the bare reviews_dir call.
    assert review_checkout_kwargs(reviews_dir) == {}
    # An active root equal to reviews_dir is the parameter's own fallback.
    assert review_checkout_kwargs(reviews_dir, reviews_dir) == {}
    # A relocated active root forwards; a legacy root equal to the active one
    # adds nothing (remove already covers it as candidate zero).
    assert review_checkout_kwargs(reviews_dir, checkouts) == {"review_checkouts": checkouts}
    assert review_checkout_kwargs(reviews_dir, checkouts, checkouts) == {
        "review_checkouts": checkouts
    }
    # A legacy root that IS the fallback adds nothing either.
    assert review_checkout_kwargs(reviews_dir, None, reviews_dir) == {}
    # Both genuinely different roots forward.
    assert review_checkout_kwargs(reviews_dir, checkouts, reviews_dir) == {
        "review_checkouts": checkouts,
        "legacy_review_checkouts": reviews_dir,
    }


def test_review_checkout_scan_roots_still_scan_both_when_nested(tmp_path: Path) -> None:
    """The ``pr-*`` match is a direct-children scan, so one root nested inside
    the other can never make its own ``pr-<N>`` children visible from the
    parent's listing: nested roots must scan both, not silently narrow to the
    active root (review rework finding)."""
    from charlie_work.paths import review_checkout_scan_roots

    active = tmp_path / "root" / "active"
    legacy_inside_active = tmp_path / "root" / "active" / "legacy"
    legacy_inside_active.mkdir(parents=True)
    (legacy_inside_active / "pr-1").mkdir()
    assert review_checkout_scan_roots(active, legacy_inside_active) == (
        active,
        legacy_inside_active,
    )

    active_inside_legacy = tmp_path / "outer" / "active"
    legacy_outer = tmp_path / "outer"
    active_inside_legacy.mkdir(parents=True)
    (legacy_outer / "pr-2").mkdir()
    assert review_checkout_scan_roots(active_inside_legacy, legacy_outer) == (
        active_inside_legacy,
        legacy_outer,
    )


def test_review_checkout_pr_numbers_dedupes_across_roots(tmp_path: Path) -> None:
    """The extracted dir scan: ``pr-<N>`` dirs from every scan root, deduped
    (a checkout can transiently sit under both roots mid-move), skipping
    non-``pr-`` entries, non-numeric suffixes and missing roots.
    """
    from charlie_work.paths import review_checkout_pr_numbers

    active = tmp_path / "volume" / "reviews"
    legacy = tmp_path / "reviews"
    (active / "pr-1").mkdir(parents=True)
    (active / "pr-2").mkdir()
    (active / "issue-1.claude.json").parent.mkdir(parents=True, exist_ok=True)
    (active / "issue-1.claude.json").write_text("{}", encoding="utf-8")
    (active / "pr-notanumber").mkdir()
    (legacy / "pr-2").mkdir(parents=True)
    (legacy / "pr-3").mkdir()

    assert review_checkout_pr_numbers((active, legacy)) == {1, 2, 3}
    assert review_checkout_pr_numbers((active, tmp_path / "missing")) == {1, 2}
    assert review_checkout_pr_numbers(()) == set()


def test_orphaned_reap_honours_an_explicit_scan_roots_tuple(monkeypatch, tmp_path: Path) -> None:
    """``checkout_scan_roots`` is the layout-derived scan set; when given it
    wins over the roots derivable from the bare path pair -- a ``pr-<N>``
    dir left under ``reviews_dir`` is invisible to a scan restricted to the
    volume root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkouts = tmp_path / "volume" / "repo" / "reviews"
    checkouts.mkdir(parents=True)
    (reviews_dir / "pr-100").mkdir()
    state_file = tmp_path / "state.json"
    save_state(state_file, empty_state())

    removed_calls: list[int] = []
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout",
        lambda *a, **k: removed_calls.append(1) or True,
    )

    reaped = _reap_orphaned_review_checkouts(
        _merged_pr_gh(),
        repo_root,
        reviews_dir,
        state_file,
        OrchestratorConfig(),
        write_gate=_wg(state_file),
        review_checkouts=checkouts,
        legacy_review_checkouts=reviews_dir,
        checkout_scan_roots=(checkouts,),
    )

    # The explicit tuple excludes the legacy root, so the pr-100 dir there is
    # never a candidate -- nothing is reaped despite the merged PR.
    assert reaped == []
    assert removed_calls == []


def test_completed_reap_forwards_the_resolved_roots(monkeypatch, tmp_path: Path) -> None:
    """The completed-verdict sweep is another ``remove_review_checkout`` call
    site: it must forward the resolved root pair, or a checkout on the volume
    survives a PR whose verdict already landed."""
    from charlie_work.workflow import _reap_completed_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkouts = tmp_path / "volume" / "repo" / "reviews"
    state_file = tmp_path / "state.json"
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_completed",
    }
    save_state(state_file, state)

    removed_calls: list[dict] = []
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout",
        lambda *a, **kw: removed_calls.append(kw) or True,
    )

    reaped = _reap_completed_review_checkouts(
        repo_root,
        reviews_dir,
        state_file,
        review_checkouts=checkouts,
        legacy_review_checkouts=reviews_dir,
    )

    assert reaped == [100]
    assert removed_calls == [
        {
            "reviews_dir": reviews_dir,
            "review_checkouts": checkouts,
            "legacy_review_checkouts": reviews_dir,
        }
    ]


def test_stalled_sweep_forwards_the_resolved_roots(monkeypatch, tmp_path: Path) -> None:
    """The stalled-claim sweep's dead-reviewer teardown (the dispatched-claim
    branch of ``_detect_and_handle_stalled_reviews``) forwards the resolved
    root pair through ``_remove_review_checkout_with_warning``."""
    from datetime import timedelta

    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkouts = tmp_path / "volume" / "repo" / "reviews"
    state_file = tmp_path / "state.json"
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": old_dispatched,
        "reviewer_pid": 999999999,  # not a real live pid
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    removed_calls: list[dict] = []
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout",
        lambda *a, **kw: removed_calls.append(kw) or True,
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir,
        state_file,
        OrchestratorConfig(),
        repo_root,
        write_gate=_wg(state_file),
        review_checkouts=checkouts,
        legacy_review_checkouts=reviews_dir,
    )

    assert any(entry.get("pr") == 100 for entry in stalled)
    assert removed_calls == [
        {
            "reviews_dir": reviews_dir,
            "review_checkouts": checkouts,
            "legacy_review_checkouts": reviews_dir,
        }
    ]
