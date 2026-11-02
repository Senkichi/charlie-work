"""Pass-end tracker-write flush wiring (issue #2434).

The orchestrator drains ``local_issue_commits``'s queue exactly once per
loop pass, at the end of ``_loop_body`` (``_flush_pass_tracker_writes`` --
the same batch-after-the-work shape the #1967 patch-equivalence drain uses).
These tests pin the wiring: the flush commits a local pass's writes and
stays a no-op on the remote ``gh`` backend and under ``dry_run``; the
deferred-flush event the doctor check reads lands when the flush cannot
commit and stays silent for a benign skip (the kill switch).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import charlie_work.orchestration.local_tracker_flush as flush_module
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.local_issue_commits import dirty_tracker_files
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _fakes_github import FakeGitHub
from _local_lane_fixtures import _app, _git, _init_repo, _local_config, _write_issue


def _seed_tracked_issues_dir(repo_root: Path, *paths: Path) -> None:
    _git(repo_root, "add", "--", *(str(p) for p in paths))
    _git(repo_root, "commit", "-m", "feat: seed issues dir")


def _tracker_commit_subjects(repo_root: Path) -> list[str]:
    return [
        line
        for line in _git(repo_root, "log", "--format=%s").stdout.splitlines()
        if line.startswith("chore(issues):")
    ]


def _event_kinds(state_file: Path) -> set[str]:
    return {e.get("kind") for e in load_state(state_file).get("events", [])}


def test_pass_flush_commits_tracker_writes(tmp_path: Path) -> None:
    """End to end through ``app.loop()``: a close issued to the client before
    the pass is committed once, at the pass end, and the tree is clean."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    issue_file = _write_issue(issues_dir, 3)
    _seed_tracked_issues_dir(repo_root, issue_file)
    app = _app(repo_root, issues_dir)

    assert app.gh.close_issue(3) is True
    assert "state: closed" in issue_file.read_text(encoding="utf-8")

    result = app.loop()

    assert result.ok is True
    assert dirty_tracker_files(repo_root, issues_dir) == ()
    assert _tracker_commit_subjects(repo_root) == ["chore(issues): close #3"]
    assert "local_tracker_writes_deferred" not in _event_kinds(app.paths.state_file)


def test_pass_flush_is_a_noop_on_a_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote ``gh`` backend has no ``issues_dir``: no flush call at all."""
    spy = Mock(wraps=flush_module.flush_tracker_writes)
    monkeypatch.setattr(flush_module, "flush_tracker_writes", spy)
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    gh = FakeGitHub()
    gh.issues = []
    gh.prs = []
    config = OrchestratorConfig()
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)

    app.loop()

    spy.assert_not_called()


def test_pass_flush_skips_under_dry_run_and_self_heals_on_a_real_pass(
    tmp_path: Path,
) -> None:
    """Dry-run: the zero-writes invariant holds -- no commit lands from a
    dry-run pass, and no deferred event fires (the delegate returns first).
    The dirt is still swept up by the first real pass, because the flush
    rescan reads the working tree."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    issue_file = _write_issue(issues_dir, 5, slug="hand")
    _seed_tracked_issues_dir(repo_root, issue_file)
    issue_file.write_text(
        '---\ntitle: "Test issue"\nstate: closed\nlabels: []\ncreated: "2026-09-17"\n'
        'author: "test"\n---\nBody text.\n',
        encoding="utf-8",
    )
    cfg = _local_config(repo_root, issues_dir)
    paths = runtime_paths(repo_root, cfg.runtime.state_dir)
    backend = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    dry_run_app = OrchestratorApp(repo_root, paths, cfg, backend, dry_run=True)

    dry_run_app.loop()

    assert dirty_tracker_files(repo_root, issues_dir) == (
        issue_file.relative_to(repo_root).as_posix(),
    )
    assert _tracker_commit_subjects(repo_root) == []
    assert "local_tracker_writes_deferred" not in _event_kinds(paths.state_file)

    real_app = OrchestratorApp(repo_root, paths, cfg, backend)
    real_app._flush_pass_tracker_writes()
    assert dirty_tracker_files(repo_root, issues_dir) == ()
    assert _tracker_commit_subjects(repo_root) == ["chore(issues): update #5"]


def test_pass_flush_records_deferred_event_on_detached_head(tmp_path: Path) -> None:
    """HEAD detached: no commit, and the pass records
    ``local_tracker_writes_deferred`` so the doctor's accumulation warning
    can see the deferral."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    issue_file = _write_issue(issues_dir, 7, slug="branch")
    _seed_tracked_issues_dir(repo_root, issue_file)
    app = _app(repo_root, issues_dir)
    body_file = repo_root / "comment.txt"
    body_file.write_text("hand-off note", encoding="utf-8")

    app.gh.issue_comment(7, body_file)
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    app._flush_pass_tracker_writes()

    assert dirty_tracker_files(repo_root, issues_dir) != ()
    deferred = query_events(app.paths.state_file, kind="local_tracker_writes_deferred", limit=5)
    assert len(deferred) == 1, deferred
    assert "HEAD is detached" in (deferred[0].get("payload") or {}).get("reason", "")


def test_kill_switch_leaves_writes_uncommitted_without_an_event(tmp_path: Path) -> None:
    """``local_issues.commit_writes: false``: the delegate's flush is a silent
    benign skip -- dirt stays, no deferred event fires (config is honored,
    not a failure), and doctor downgrades to a note on the same condition."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    issue_file = _write_issue(issues_dir, 9, slug="switch")
    _seed_tracked_issues_dir(repo_root, issue_file)
    cfg = _local_config(
        repo_root,
        issues_dir,
        local_issues={"enabled": True, "issues_dir": "docs/issues", "commit_writes": False},
    )
    app = _app(repo_root, issues_dir, cfg)

    assert app.gh.close_issue(9) is True

    app._flush_pass_tracker_writes()

    assert dirty_tracker_files(repo_root, issues_dir) == (
        issue_file.relative_to(repo_root).as_posix(),
    )
    assert _tracker_commit_subjects(repo_root) == []
    assert not [
        e
        for e in load_state(app.paths.state_file).get("events", [])
        if e.get("kind") == "local_tracker_writes_deferred"
    ]
