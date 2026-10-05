"""Issue #2289: a throttle-killed attempt's preserved work seeds the redispatch.

Real temporary git repos, no mocks of git: a rescue ref is captured by the real
``_capture_worktree_work_to_rescue_ref`` and an attempt ref by the real
``snapshot_attempt_ref``; the dead worker's classification is a real
``session_exited`` row in ``events.db``.
"""

from __future__ import annotations

import sys
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
    # No sleep needed: selection compares at whole-second granularity, so a ref
    # made in the same second as the death event still belongs to it.


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
    # A branch tip is squash-merged like a rescue ref: the content is there,
    # as uncommitted work on top of the base.
    assert (info2.path / "feature.py").read_text(encoding="utf-8") == "print('x')\n"
    assert _git(info2.path, "rev-parse", "HEAD").stdout == _git(repo, "rev-parse", "main").stdout
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


def test_kindless_follow_up_events_cannot_mask_the_throttle_death(repo: Path) -> None:
    """Orphan-sweep batch events and relabel follow-ups carry no classification."""
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    state_file = _state_file(repo, config)
    _die(repo, config, "rate_limited")
    ref = _rescue(repo, wt)
    log_event(state_file, "session_failed_relabeled", {"issue_number": ISSUE, "reason": "x"})
    log_event(
        state_file,
        "session_failed_relabeled_sweep",
        {"issue_number": ISSUE, "failure_kind": None},
    )
    log_event(
        state_file,
        "session_failed_escalated",
        {"issue_number": ISSUE, "failure_kind": None},
    )

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is not None
    assert info.resumed_attempt.ref == ref


def test_later_classified_non_throttle_death_vetoes_the_resume(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    _rescue(repo, wt)
    log_event(
        _state_file(repo, config),
        "session_exited",
        {"issue_number": ISSUE, "failure_kind": "stalled"},
    )

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None


def test_kindless_session_exited_is_an_unclassified_death_and_vetoes(repo: Path) -> None:
    """A death with no classification is still the newest death (S1)."""
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    _rescue(repo, wt)
    log_event(
        _state_file(repo, config),
        "session_exited",
        {"issue_number": ISSUE, "failure_kind": None},
    )

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None
    assert not (info.path / "work.txt").exists()


def test_old_throttle_death_cannot_authorize_a_later_workers_work(repo: Path) -> None:
    """T0 throttle death -> T1 resumed redispatch -> T2 worker B dies unclassified
    -> T3 fresh dispatch must NOT apply B's work as a rate-limit resume."""
    config = OrchestratorConfig()
    state_file = _state_file(repo, config)
    wt_a = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")  # T0
    _rescue(repo, wt_a)
    resumed = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)
    assert resumed.resumed_attempt is not None  # T1
    log_event(state_file, "dispatch", {"issue_numbers": [ISSUE]})
    # Worker B edits further, then dies with NO death event at all (e.g. the
    # decide_stalled handoff skipped classification) and only a kind-less relabel.
    (resumed.path / "b_work.txt").write_text("worker B\n", encoding="utf-8")
    ref_b = _rescue(repo, resumed.path)  # T2
    log_event(state_file, "session_failed_relabeled", {"issue_number": ISSUE})

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)  # T3

    assert info.resumed_attempt is None
    assert not (info.path / "b_work.txt").exists()
    assert _git(repo, "rev-parse", "--verify", ref_b).returncode == 0


def test_launch_after_the_throttle_death_vetoes_even_without_a_newer_death(repo: Path) -> None:
    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    _rescue(repo, wt)
    log_event(_state_file(repo, config), "dispatch_rework", {"issue_numbers": [ISSUE]})

    info = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info.resumed_attempt is None


def test_empty_attempt_snapshot_cannot_beat_the_rescue_ref(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: ``create_worktree`` captures the rescue ref, removes the worktree, then
    snapshots the attempt ref. If removal crosses a second boundary the (whole
    second) attempt reflog reads newer, and the attempt ref can be empty or a
    subset. The real capture-then-snapshot sequence, with every git write after
    removal stamped 2 s ahead (``GIT_COMMITTER_DATE``) instead of a 1.1 s sleep."""
    from charlie_work import attempt_resume as attempt_resume_module
    from charlie_work import worktree as worktree_module

    config = OrchestratorConfig()
    info1 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE)
    (info1.path / "committed.txt").write_text("committed\n", encoding="utf-8")
    _git(info1.path, "add", "committed.txt")
    _git(info1.path, "commit", "-m", "worker commit")
    (info1.path / "work.txt").write_text("uncommitted work\n", encoding="utf-8")
    _die(repo, config, "rate_limited")

    real_remove = worktree_module.remove_worktree

    def late_remove(*args, **kwargs):
        monkeypatch.setenv("GIT_COMMITTER_DATE", f"@{int(time.time()) + 2} +0000")
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(worktree_module, "remove_worktree", late_remove)
    real_candidates = attempt_resume_module.preserved_ref_candidates
    offered: list[list] = []

    def recording_candidates(*args, **kwargs):
        found = real_candidates(*args, **kwargs)
        offered.append(found)
        return found

    monkeypatch.setattr(attempt_resume_module, "preserved_ref_candidates", recording_candidates)

    info2 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    # Positive control: the race B1 is about really happened -- the attempt ref
    # reads newer than the rescue ref -- and the rescue ref still wins.
    assert offered, "preserved_ref_candidates was never consulted"
    newest = {
        kind: max(ref.created_at for ref in offered[-1] if ref.kind == kind)
        for kind in ("rescue", "attempt")
    }
    assert newest["attempt"] > newest["rescue"]
    assert info2.rescue_capture is not None and info2.rescue_capture.ref_name is not None
    assert info2.attempt_snapshot is not None and info2.attempt_snapshot.ref_name is not None
    assert info2.resumed_attempt is not None
    assert info2.resumed_attempt.ref == info2.rescue_capture.ref_name
    assert info2.resumed_attempt.ref_kind == "rescue"
    assert (info2.path / "work.txt").read_text(encoding="utf-8") == "uncommitted work\n"
    assert (info2.path / "committed.txt").read_text(encoding="utf-8") == "committed\n"


def test_empty_preserved_ref_emits_resume_failed_instead_of_silence(repo: Path) -> None:
    config = OrchestratorConfig()
    info1 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE)
    _die(repo, config, "rate_limited")
    snap = snapshot_attempt_ref(repo, BRANCH, ISSUE, base_ref="main")  # 0 commits ahead
    assert snap.ref_name is not None
    assert remove_worktree(repo, info1.path, force=True, branch=BRANCH)

    info2 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info2.resumed_attempt is None
    [event] = _events(repo, config, "attempt_resume_failed")
    assert "empty_ref" in event["payload"]["reason"]


def test_attempt_ref_with_a_merge_commit_resumes(repo: Path) -> None:
    """S3: rework branches carry merge commits; ``base..ref`` cherry-pick fails on them."""
    config = OrchestratorConfig()
    info1 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE)
    (info1.path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(info1.path, "add", "a.txt")
    _git(info1.path, "commit", "-m", "A")
    _git(info1.path, "checkout", "-b", "side", "main")
    (info1.path / "b.txt").write_text("b\n", encoding="utf-8")
    _git(info1.path, "add", "b.txt")
    _git(info1.path, "commit", "-m", "B")
    _git(info1.path, "checkout", BRANCH)
    assert _git(info1.path, "merge", "--no-ff", "-m", "merge side", "side").returncode == 0
    _die(repo, config, "rate_limited")
    snap = snapshot_attempt_ref(repo, BRANCH, ISSUE, base_ref="main")
    assert snap.ref_name is not None
    assert remove_worktree(repo, info1.path, force=True, branch=BRANCH)

    info2 = create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert info2.resumed_attempt is not None
    assert info2.resumed_attempt.ref_kind == "attempt"
    assert (info2.path / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (info2.path / "b.txt").read_text(encoding="utf-8") == "b\n"


def test_unprovable_restore_raises_and_tears_the_worktree_down(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: never launch on a tree that may hold conflict markers or a wrong HEAD."""
    from charlie_work import attempt_resume

    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    _rescue(repo, wt)
    # Conflict at apply time, so a restore is attempted...
    (repo / "README.md").write_text("upstream readme change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "upstream change")
    # ...and the restore's hard reset silently does nothing.
    real_git = attempt_resume._git

    def broken_reset(repo_path, *args):
        if args[:2] == ("reset", "--hard"):
            return real_git(repo_path, "rev-parse", "--git-dir")
        return real_git(repo_path, *args)

    monkeypatch.setattr(attempt_resume, "_git", broken_reset)

    with pytest.raises(attempt_resume.ResumeRestoreError):
        create_worktree(repo, BRANCH, base_ref="HEAD", issue_number=ISSUE, config=config)

    assert _events(repo, config, "attempt_resume_failed")
    listing = _git(repo, "worktree", "list", "--porcelain").stdout
    assert "issue-2289" not in listing


def test_launch_claude_worker_puts_the_resume_notice_in_the_prompt_file(repo: Path) -> None:
    """Adapter level: real seeded worktree, real prompt file."""
    from charlie_work.claude_code import PROMPT_FILENAME, launch_claude_worker

    config = OrchestratorConfig()
    wt = _worker_dirty_tree(repo)
    _die(repo, config, "rate_limited")
    ref = _rescue(repo, wt)
    script = repo.parent / "fake_claude.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    record = launch_claude_worker(
        ISSUE,
        BRANCH,
        "Implement the issue.",
        repo_root=repo,
        sessions_dir=repo.parent / "sessions",
        command_template=(sys.executable, str(script)),
        config=config,
    )

    assert record.error is None
    prompt = (Path(record.worktree_path) / PROMPT_FILENAME).read_text(encoding="utf-8")
    assert prompt.startswith("Implement the issue.")
    assert "Continue that work" in prompt
    assert ref in prompt
    assert (Path(record.worktree_path) / "work.txt").exists()
