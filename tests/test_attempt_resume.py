"""Issue #2289: a throttle-killed attempt's preserved work seeds the redispatch.

Real temporary git repos, no mocks of git: a rescue ref is captured by the real
``_capture_worktree_work_to_rescue_ref`` and an attempt ref by the real
``snapshot_attempt_ref``; the dead worker's classification is a real
``session_exited`` row in ``events.db``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from _worktree_fixtures import _wt_scratch as _register_wt_scratch  # noqa: F401 -- registers the wt_scratch fixture (shallow tmp dir for real ``git worktree add``)
from charlie_work.attempt_refs import snapshot_attempt_ref
from charlie_work.attempt_resume import apply_resume_notice, render_resume_notice
from charlie_work.config import DispatchConfig, OrchestratorConfig
from charlie_work.instrumentation import log_event, query_events
from charlie_work.paths import runtime_paths
from charlie_work.worktree import (
    _capture_worktree_work_to_rescue_ref,
    create_worktree,
    remove_worktree,
)

ISSUE = 2289
BRANCH = "agent/issue-2289-resume"


def _state_file(repo_root: Path, config: OrchestratorConfig) -> Path:
    return runtime_paths(repo_root, config.runtime.state_dir).state_file


def _die(repo_root: Path, config: OrchestratorConfig, failure_kind: str | None) -> None:
    log_event(
        _state_file(repo_root, config),
        "session_exited",
        {"issue_number": ISSUE, "failure_kind": failure_kind},
    )
    time.sleep(1.1)  # ref clocks are whole seconds; the ref must post-date the death


def _worker_dirty_tree(repo_root: Path) -> Path:
    info = create_worktree(repo_root, BRANCH, base_ref="HEAD", issue_number=ISSUE)
    (info.path / "work.txt").write_text("partial work\nline two\n", encoding="utf-8")
    (info.path / "README.md").write_text("changed readme\n", encoding="utf-8")
    return info.path


def _rescue(repo_root: Path, wt_path: Path) -> str:
    capture = _capture_worktree_work_to_rescue_ref(repo_root, wt_path, ISSUE)
    assert capture.error is None and capture.ref_name is not None
    assert remove_worktree(repo_root, wt_path, force=True, branch=BRANCH)
    return capture.ref_name


def _events(repo_root: Path, config: OrchestratorConfig, kind: str) -> list[dict]:
    return query_events(_state_file(repo_root, config), kind=kind, issue_number=ISSUE)


@pytest.fixture
def repo(wt_scratch: Path) -> Path:
    root = wt_scratch / "repo"
    _init_repo(root)
    return root


def test_throttle_death_resumes_from_rescue_ref(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    ref = _rescue(repo, wt)

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is not None
    assert info.resumed_attempt.ref == ref
    assert info.resumed_attempt.ref_kind == "rescue"
    assert (info.path / "work.txt").read_text(encoding="utf-8") == "partial work\nline two\n"
    assert (info.path / "README.md").read_text(encoding="utf-8") == "changed readme\n"
    # On top of the base, as uncommitted work, exactly like the dead worker's tree.
    assert _git(info.path, "rev-parse", "HEAD").stdout == _git(repo, "rev-parse", "main").stdout
    assert _git(info.path, "diff", "--cached", "--name-only").stdout.strip() == ""
    assert info.resumed_attempt.files == 2
    [event] = _events(repo, config, "attempt_resumed")
    assert event["payload"]["ref"] == ref
    assert event["payload"]["files"] == 2
    assert event["payload"]["insertions"] == info.resumed_attempt.insertions
    assert _events(repo, config, "attempt_resume_failed") == []


def test_throttle_death_resumes_from_attempt_ref_commits(repo: Path) -> None:
    config = OrchestratorConfig()
    info1 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE)
    (info1.path / "feature.py").write_text("print('x')\n", encoding="utf-8")
    _git(info1.path, "add", "feature.py")
    _git(info1.path, "commit", "-m", "worker commit")
    _die(repo, config, "quota_exhausted")
    snap = snapshot_attempt_ref(repo, BRANCH, ISSUE, base_ref="main")
    assert snap.ref_name is not None
    assert remove_worktree(repo, info1.path, force=True, branch=BRANCH)

    info2 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info2.resumed_attempt is not None
    assert info2.resumed_attempt.ref == snap.ref_name
    assert info2.resumed_attempt.ref_kind == "attempt"
    # A branch tip is cherry-picked: the worker's commit survives.
    assert (info2.path / "feature.py").read_text(encoding="utf-8") == "print('x')\n"
    assert _git(info2.path, "log", "-1", "--format=%s").stdout.strip() == "worker commit"
    [event] = _events(repo, config, "attempt_resumed")
    assert event["payload"]["ref_kind"] == "attempt"


@pytest.mark.parametrize("failure_kind", ["stalled", "worker_crash", None])
def test_non_throttle_death_starts_clean(repo: Path, failure_kind: str | None) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, failure_kind)
    _rescue(repo, wt)

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None
    assert not (info.path / "work.txt").exists()
    assert _git(info.path, "status", "--porcelain").stdout.strip() == ""
    assert _events(repo, config, "attempt_resumed") == []
    assert _events(repo, config, "attempt_resume_failed") == []


def test_no_recorded_death_starts_clean(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _rescue(repo, wt)

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None
    assert not (info.path / "work.txt").exists()


def test_stale_rescue_ref_from_earlier_attempt_is_not_applied(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    ref = _rescue(repo, wt)
    # Re-date the ref to long before the death below: an unrelated earlier attempt.
    sha = _git(repo, "rev-parse", ref).stdout.strip()
    _git(repo, "update-ref", f"refs/charlie/rescue/issue-{ISSUE}-20200101T000000000000Z", sha)
    _git(repo, "update-ref", "-d", ref)
    _die(repo, config, "rate_limited")

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None
    assert not (info.path / "work.txt").exists()
    assert _events(repo, config, "attempt_resumed") == []


def test_other_issues_rescue_ref_is_not_applied(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    capture = _capture_worktree_work_to_rescue_ref(repo, wt, ISSUE + 1)
    assert capture.error is None
    assert remove_worktree(repo, wt, force=True, branch=BRANCH)

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None


def test_conflicting_resume_falls_back_to_clean_base_and_still_dispatches(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    ref = _rescue(repo, wt)
    # The base moves on and touches the same file the dead worker edited.
    (repo / "README.md").write_text("upstream readme change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "upstream change")

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.path.is_dir()
    assert info.resumed_attempt is None
    assert _git(info.path, "rev-parse", "HEAD").stdout == _git(repo, "rev-parse", "main").stdout
    assert _git(info.path, "status", "--porcelain").stdout.strip() == ""
    assert (info.path / "README.md").read_text(encoding="utf-8") == "upstream readme change\n"
    assert not (info.path / "work.txt").exists()
    [event] = _events(repo, config, "attempt_resume_failed")
    assert event["payload"]["ref"] == ref
    assert "conflict" in event["payload"]["reason"]
    assert _events(repo, config, "attempt_resumed") == []
    # The preserved work is untouched for an operator.
    assert _git(repo, "rev-parse", "--verify", ref).returncode == 0


def test_kill_switch_off_restores_clean_base(repo: Path) -> None:
    config = OrchestratorConfig(dispatch=DispatchConfig(resume_throttled_attempts=False))
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    _rescue(repo, wt)

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None
    assert not (info.path / "work.txt").exists()
    assert _events(repo, config, "attempt_resumed") == []


def test_resume_is_on_by_default() -> None:
    assert DispatchConfig().resume_throttled_attempts is True


def test_prompt_notice_says_continue_not_restart() -> None:
    from charlie_work.attempt_resume import ResumedAttempt

    resumed = ResumedAttempt("refs/charlie/rescue/issue-1-x", "rescue", 8, 603)
    notice = render_resume_notice(resumed)
    assert "rate limit" in notice
    assert "already in your working tree" in notice
    assert "Do not restart" in notice
    assert "603" in notice
    assert apply_resume_notice("ORIGINAL PROMPT\n", resumed).startswith("ORIGINAL PROMPT\n\n## ")
