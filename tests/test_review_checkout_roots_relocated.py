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
