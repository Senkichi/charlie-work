"""Coverage for the rework pre-merge conflict fix.

Background: ``create_worktree``'s rework path used to raise
``ReworkBranchConflictError`` on an ordinary merge conflict against the base
ref, which meant the worker who was supposed to resolve the conflict never
launched (see the "rework-conflict" investigation). The fix makes
``_merge_update_rework_branch`` (worktree.py) abort the failed merge and
return a ``ReworkMergeConflict`` notice instead of raising, so the worktree —
and the worker — still launch. ``ReworkBranchConflictError`` is now reserved
for the genuinely unrecoverable case: ``git merge --abort`` itself fails.

Fixture style (``_init_repo``/``_git``/``_clone_repo``) is copied from
tests/test_worktree.py rather than imported, per instruction to leave that
file's own tests alone (its two pre-existing conflict tests were updated
in place there to match the new behavior, since they directly asserted the
old raise-on-ordinary-conflict contract -- this file adds the dedicated,
from-scratch coverage the fix task calls for).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from charlie_work import claude_code, devin_shell, worktree
from charlie_work.claude_code import PROMPT_FILENAME, launch_claude_worker
from charlie_work.devin_shell import launch_devin_session
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    ReworkBranchConflictError,
    ReworkMergeConflict,
    WorktreeInfo,
    create_worktree,
    remove_worktree,
    render_rework_conflict_notice,
)


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "file.txt").write_text("base line\n", encoding="utf-8")
    run(["git", "add", "file.txt"])
    run(["git", "commit", "-m", "initial commit"])


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _merge_head_present(worktree_path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# create_worktree: conflicting rework branch succeeds with a notice
# ---------------------------------------------------------------------------


def test_conflicting_rework_branch_succeeds_with_notice(tmp_path: Path) -> None:
    """A rework branch that conflicts with the base must not fail worktree
    creation: it returns successfully with WorktreeInfo.rework_conflict set,
    no MERGE_HEAD left behind, and the branch head unchanged."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-fix-conflict-1"
    info1 = create_worktree(repo_root, branch_name, base_ref="")
    (info1.path / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(info1.path, "add", "file.txt")
    _git(info1.path, "commit", "-m", "add feature")
    pre_merge_head = _git(info1.path, "rev-parse", "HEAD").stdout.strip()

    # Advance main with a conflicting edit to the same line.
    _git(repo_root, "checkout", "main")
    (repo_root / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "advance main")

    info2 = create_worktree(repo_root, branch_name, rework=True, base_ref="")

    assert isinstance(info2.rework_conflict, ReworkMergeConflict)
    assert info2.rework_conflict.conflicted_files == ("file.txt",)
    assert info2.rework_conflict.base_branch == "main"
    assert info2.rework_conflict.base_sha  # resolved to a real sha

    post_merge_head = _git(info2.path, "rev-parse", "HEAD").stdout.strip()
    assert post_merge_head == pre_merge_head, "branch head must be unchanged after the abort"
    assert not _merge_head_present(info2.path), "MERGE_HEAD must not remain after the abort"

    remove_worktree(repo_root, info1.path)


def test_clean_rework_branch_merges_as_before_notice_is_none(tmp_path: Path) -> None:
    """A rework branch with no conflict must still get the pre-merge (base
    changes land in the branch) and rework_conflict must be None."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-fix-conflict-clean"
    info1 = create_worktree(repo_root, branch_name, base_ref="")
    (info1.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info1.path, "add", "feature.txt")
    _git(info1.path, "commit", "-m", "add feature")

    # Advance main with a NON-conflicting change (different file).
    _git(repo_root, "checkout", "main")
    (repo_root / "other.txt").write_text("other\n", encoding="utf-8")
    _git(repo_root, "add", "other.txt")
    _git(repo_root, "commit", "-m", "advance main, non-conflicting")

    info2 = create_worktree(repo_root, branch_name, rework=True, base_ref="")

    assert info2.rework_conflict is None
    # The pre-merge must have actually happened: main's new file is present.
    assert (info2.path / "other.txt").exists()
    assert not _merge_head_present(info2.path)

    remove_worktree(repo_root, info1.path)


# ---------------------------------------------------------------------------
# render_rework_conflict_notice: wording/content
# ---------------------------------------------------------------------------


def test_render_rework_conflict_notice_content() -> None:
    conflict = ReworkMergeConflict(
        base_branch="main",
        base_sha="abc123def456abc123def456",
        conflicted_files=("file.txt", "other/path.py"),
    )
    notice = render_rework_conflict_notice(conflict)

    assert "ACTION REQUIRED" in notice
    assert "main" in notice
    assert "abc123def456" in notice  # short sha (first 12 chars)
    assert "- file.txt" in notice
    assert "- other/path.py" in notice
    assert "merge `origin/main`" in notice
    assert "THEN: address the review feedback" in notice


def test_render_rework_conflict_notice_handles_empty_files_and_sha() -> None:
    conflict = ReworkMergeConflict(base_branch="main", base_sha="", conflicted_files=())
    notice = render_rework_conflict_notice(conflict)

    assert "(conflicted paths unavailable)" in notice
    assert "unknown" in notice


# ---------------------------------------------------------------------------
# prompt injection: claude_code adapter (in-memory prompt_text, written fresh
# into the worktree)
# ---------------------------------------------------------------------------


def _fake_worktree(tmp_path: Path, branch: str, *, rework_conflict=None) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(
        path=worktree_path,
        branch=branch,
        venv_junction=None,
        rework_conflict=rework_conflict,
    )


def _fake_claude_script(tmp_path: Path) -> tuple[str, ...]:
    script = tmp_path / "fake_claude.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    return (sys.executable, str(script))


def test_launch_claude_worker_appends_notice_to_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    conflict = ReworkMergeConflict(
        base_branch="main", base_sha="deadbeef0000", conflicted_files=("file.txt",)
    )

    def fake_create_worktree(*args, **kwargs):
        return _fake_worktree(tmp_path, "agent/issue-1", rework_conflict=conflict)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    record = launch_claude_worker(
        1,
        "agent/issue-1",
        "Address the review feedback.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=True,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.error is None
    prompt_path = Path(record.worktree_path) / PROMPT_FILENAME
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "Address the review feedback." in prompt_text
    assert "ACTION REQUIRED" in prompt_text
    assert "file.txt" in prompt_text


def test_launch_claude_worker_prompt_file_unchanged_without_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    def fake_create_worktree(*args, **kwargs):
        return _fake_worktree(tmp_path, "agent/issue-2", rework_conflict=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    record = launch_claude_worker(
        2,
        "agent/issue-2",
        "Address the review feedback.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=True,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.error is None
    prompt_path = Path(record.worktree_path) / PROMPT_FILENAME
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert prompt_text == "Address the review feedback."
    assert "ACTION REQUIRED" not in prompt_text


# ---------------------------------------------------------------------------
# prompt injection: devin_shell adapter (mutates the caller-supplied
# prompt_path in place, since devin never copies it into the worktree)
# ---------------------------------------------------------------------------


def test_launch_devin_session_appends_notice_to_prompt_file_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Address the review feedback.", encoding="utf-8")
    conflict = ReworkMergeConflict(
        base_branch="main", base_sha="cafebabe0000", conflicted_files=("file.txt",)
    )

    def fake_create_worktree(*args, **kwargs):
        return _fake_worktree(tmp_path, "agent/issue-3", rework_conflict=conflict)

    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)

    script = tmp_path / "fake_devin.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    record = launch_devin_session(
        3,
        "agent/issue-3",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=True,
        command_template=(sys.executable, str(script)),
    )

    assert record.error is None
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "Address the review feedback." in prompt_text
    assert "ACTION REQUIRED" in prompt_text
    assert "file.txt" in prompt_text


def test_launch_devin_session_prompt_file_unchanged_without_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Address the review feedback.", encoding="utf-8")

    def fake_create_worktree(*args, **kwargs):
        return _fake_worktree(tmp_path, "agent/issue-4", rework_conflict=None)

    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)

    script = tmp_path / "fake_devin.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    record = launch_devin_session(
        4,
        "agent/issue-4",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=True,
        command_template=(sys.executable, str(script)),
    )

    assert record.error is None
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert prompt_text == "Address the review feedback."
    assert "ACTION REQUIRED" not in prompt_text


# ---------------------------------------------------------------------------
# Unrecoverable case: ReworkBranchConflictError still raised when
# `git merge --abort` itself fails.
# ---------------------------------------------------------------------------


def test_rework_conflict_raises_when_merge_abort_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the pre-merge conflicts AND `git merge --abort` cannot recover
    (simulated here since a real, deterministic abort failure is not cheaply
    reproducible with plain git), ReworkBranchConflictError must still raise
    -- the worktree is genuinely left mid-merge and unusable, so it must
    escalate rather than launch a worker into a broken workspace."""
    from charlie_work.subprocess_runner import run_captured as real_run_captured

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-fix-conflict-abort-fails"
    info1 = create_worktree(repo_root, branch_name, base_ref="")
    (info1.path / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(info1.path, "add", "file.txt")
    _git(info1.path, "commit", "-m", "add feature")

    _git(repo_root, "checkout", "main")
    (repo_root / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "advance main")

    def fake_run_captured(command, *, cwd, timeout_seconds, shell=False, stdin=None):
        if list(command) == ["git", "merge", "--abort"]:
            return RunResult(returncode=1, stdout="", stderr="simulated abort failure")
        return real_run_captured(
            command, cwd=cwd, timeout_seconds=timeout_seconds, shell=shell, stdin=stdin
        )

    monkeypatch.setattr(worktree, "run_captured", fake_run_captured)

    try:
        with pytest.raises(ReworkBranchConflictError) as exc_info:
            create_worktree(repo_root, branch_name, rework=True, base_ref="")
        assert "file.txt" in exc_info.value.conflicted_paths
        # The real abort never ran, so the worktree is genuinely mid-merge.
        assert _merge_head_present(info1.path)
    finally:
        # Leave the worktree clean for teardown regardless of the simulated
        # failure above (uses the real run_captured, not the monkeypatched one).
        subprocess.run(["git", "merge", "--abort"], cwd=info1.path, capture_output=True, text=True)
        monkeypatch.undo()
        remove_worktree(repo_root, info1.path)


def test_apply_rework_conflict_notice_is_idempotent_and_fresh() -> None:
    """devin_shell mutates the caller-supplied prompt file in place and the
    file is never regenerated per dispatch attempt, so a blind append
    stacked one notice per redispatch -- and the base branch can move
    between attempts, leaving contradictory base SHAs in the same prompt.
    apply_rework_conflict_notice must strip any previous notice block and
    leave exactly one, current notice.
    """
    from charlie_work.worktree import (
        REWORK_CONFLICT_NOTICE_BEGIN,
        ReworkMergeConflict,
        apply_rework_conflict_notice,
    )

    base_prompt = "# Rework PR #456\n\nAddress the review feedback."
    first = ReworkMergeConflict(
        base_branch="main", base_sha="a" * 40, conflicted_files=("src/foo.py",)
    )
    second = ReworkMergeConflict(
        base_branch="main", base_sha="b" * 40, conflicted_files=("src/foo.py", "src/bar.py")
    )

    once = apply_rework_conflict_notice(base_prompt, first)
    assert once.count(REWORK_CONFLICT_NOTICE_BEGIN) == 1
    assert ("a" * 12) in once

    twice = apply_rework_conflict_notice(once, second)
    assert twice.count(REWORK_CONFLICT_NOTICE_BEGIN) == 1
    assert ("b" * 12) in twice
    assert ("a" * 12) not in twice  # stale base SHA excised, not stacked
    assert "src/bar.py" in twice
    assert twice.startswith("# Rework PR #456")
    assert "Address the review feedback." in twice


def test_apply_rework_conflict_notice_recovers_from_torn_block() -> None:
    """A crashed attempt can leave a begin sentinel with no end; the helper
    must not let that garbage survive above the fresh notice.
    """
    from charlie_work.worktree import (
        REWORK_CONFLICT_NOTICE_BEGIN,
        REWORK_CONFLICT_NOTICE_END,
        ReworkMergeConflict,
        apply_rework_conflict_notice,
    )

    torn = (
        "# Rework PR #456\n\n"
        f"{REWORK_CONFLICT_NOTICE_BEGIN}\npartial stale garbage with no end sentinel"
    )
    conflict = ReworkMergeConflict(
        base_branch="main", base_sha="c" * 40, conflicted_files=("src/foo.py",)
    )

    repaired = apply_rework_conflict_notice(torn, conflict)
    assert repaired.count(REWORK_CONFLICT_NOTICE_BEGIN) == 1
    assert repaired.count(REWORK_CONFLICT_NOTICE_END) == 1
    assert "partial stale garbage" not in repaired
    assert repaired.startswith("# Rework PR #456")
