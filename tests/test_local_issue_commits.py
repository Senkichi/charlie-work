"""Tests for local_issue_commits: the per-pass tracker-write flush (issue #2434).

The scenario the issue reports: a consumer repo's tracked ``issues_dir``
files accumulate uncommitted backend writes (state flips, label edges,
appended comments) because nothing ever commits them. These tests run the
flush against real git repos (per-process templates per the #2425/#2387
spawn-cost convention), asserting the commit batch, the verb-bearing
messages, and every leave-it-to-the-next-pass skip the flush documents.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

import charlie_work.local_issue_commits as commits_module
from charlie_work.local_issue_commits import (
    dirty_tracker_files,
    flush_tracker_writes,
    queue_issue_write,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.subprocess_runner import RunResult
from charlie_work.local_issue_commits import _run_git
from _worktree_fixtures import _git, _init_repo


def _write_issue(
    issues_dir: Path, number: int, *, slug: str = "issue", state: str = "open", labels: str = "[]"
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_{slug}.md"
    path.write_text(
        f'---\ntitle: "issue {number}"\nstate: {state}\nlabels: {labels}\n---\nBody.\n',
        encoding="utf-8",
    )
    return path


def _seed_tracked(repo_root: Path, *paths: Path) -> None:
    """Commit baseline paths so the issues dir is tracked before the test dirties it."""
    _git(repo_root, "add", "--", *(str(p) for p in paths))
    _git(repo_root, "commit", "-m", "seed")


def _detach(repo_root: Path) -> None:
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


class _SeededTrackerRepo(NamedTuple):
    """What ``_seeded_tracker_repo`` arranges: a repo with one tracked issue file."""

    repo_root: Path
    issues_dir: Path
    issue_path: Path


@pytest.fixture
def _seeded_tracker_repo(tmp_path: Path) -> _SeededTrackerRepo:
    """One-commit repo whose ``docs/issues`` holds a single tracked issue file.

    Every git spawn of repo acquisition lives here so the measured ``call``
    phase holds only the backend mutation, ``flush_tracker_writes`` -- the
    property under test -- and the assertions (issue #2690). The repo
    materializes from the per-process ``plain`` git template rather than a
    fresh ``git init``+config+add+commit sequence.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 1)
    _seed_tracked(repo_root, path)
    return _SeededTrackerRepo(repo_root=repo_root, issues_dir=issues_dir, issue_path=path)


def test_mutation_writes_queue_into_the_pass_flush(
    _seeded_tracker_repo: _SeededTrackerRepo,
) -> None:
    """End to end through the backend write site: a close lands on disk and
    one flush pass later it is committed as ``chore(issues): close #1``."""
    repo_root = _seeded_tracker_repo.repo_root
    issues_dir = _seeded_tracker_repo.issues_dir
    path = _seeded_tracker_repo.issue_path
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.close_issue(1) is True
    assert "state: closed" in path.read_text(encoding="utf-8")
    assert dirty_tracker_files(repo_root, issues_dir) == ("docs/issues/001_issue.md",)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.committed == ("chore(issues): close #1",)
    assert not result.needs_attention
    assert dirty_tracker_files(repo_root, issues_dir) == ()
    log = _run_git(repo_root, ["git", "log", "-1", "--format=%s"]).stdout.strip()
    assert log == "chore(issues): close #1"


def test_label_edges_and_comments_batch_into_one_commit_per_issue(tmp_path: Path) -> None:
    """A label edge + an appended comment on the same issue during one pass
    batch into a single commit -- the anti-pattern the flush replaces was a
    commit per write, and a per-write design would land two."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 7, slug="queued")
    _seed_tracked(repo_root, path)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    body_file = repo_root / "comment.txt"
    body_file.write_text("verdict comment", encoding="utf-8")

    assert gh.add_issue_label(7, "agent:queued") is True
    gh.issue_comment(7, body_file)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.committed == ("chore(issues): label+comment #7",)
    assert dirty_tracker_files(repo_root, issues_dir) == ()
    log = _run_git(repo_root, ["git", "log", "-1", "--format=%s"]).stdout.strip()
    assert log == "chore(issues): label+comment #7"
    committed_text = _run_git(repo_root, ["git", "show", "HEAD:docs/issues/007_queued.md"]).stdout
    assert "agent:queued" in committed_text
    assert "verdict comment" in committed_text


def test_flush_sweeps_dirt_without_a_queue_entry(tmp_path: Path) -> None:
    """Self-healing: a hand-edited frontmatter (an operator flips a state by
    hand on this backend, per local_issues' own docstring) is committed at
    the next flush even though no write site queued it -- "update" verb."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 3, slug="hand")
    _seed_tracked(repo_root, path)

    path.write_text(
        '---\ntitle: "issue 3"\nstate: closed\nlabels: []\n---\nBody.\n',
        encoding="utf-8",
    )

    # No queue_issue_write call at all: the flush reads the working tree.
    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.committed == ("chore(issues): update #3",)
    assert dirty_tracker_files(repo_root, issues_dir) == ()


def test_flush_commits_a_wholly_untracked_issues_dir(tmp_path: Path) -> None:
    """A brand-new consumer repo whose ``issues_dir`` was never committed
    reaches porcelain as one collapsed ``?? dir/`` entry; the flush expands
    it and commits the files instead of silently reporting a clean tree."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1)
    _write_issue(issues_dir, 2, slug="second")
    (issues_dir / "README.md").write_text("template notes\n", encoding="utf-8")

    result = flush_tracker_writes(repo_root, issues_dir)

    assert "chore(issues): update #1" in result.committed
    assert "chore(issues): update #2" in result.committed
    # Only issue-shaped files were staged; the README stays for the operator.
    status = _run_git(repo_root, ["git", "status", "--porcelain", "--", "docs/issues"]).stdout
    assert status.strip() == "?? docs/issues/README.md"


def test_flush_skips_when_head_is_detached(tmp_path: Path) -> None:
    """Detached HEAD: defer, leave the dirt for the next pass, warn loudly."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 1)
    _seed_tracked(repo_root, path)
    _detach(repo_root)
    fresh = _write_issue(issues_dir, 2, slug="fresh")  # untracked on top
    assert fresh.name == "002_fresh.md"

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.deferred is True
    assert result.defer_reason == "HEAD is detached"
    assert result.left_dirty == ("docs/issues/002_fresh.md",)
    assert result.committed == ()
    # Nothing landed: HEAD's message is still the seed commit's.
    log = _run_git(repo_root, ["git", "log", "-1", "--format=%s"]).stdout.strip()
    assert log == "seed"


def test_flush_skips_when_a_merge_is_in_progress(tmp_path: Path) -> None:
    """Mid-merge (MERGE_HEAD present): defer even with clean issue files --
    committing inside an interrupted operation could entangle the tracker
    with the operator's merge."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 1)
    _seed_tracked(repo_root, path)
    readme = repo_root / "README.md"
    readme.write_text("main side\n", encoding="utf-8")
    _seed_tracked(repo_root, readme)
    _git(repo_root, "checkout", "-b", "other")
    readme.write_text("other side\n", encoding="utf-8")
    _seed_tracked(repo_root, readme)
    _git(repo_root, "checkout", "main")
    readme.write_text("conflicting main side\n", encoding="utf-8")
    _seed_tracked(repo_root, readme)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    assert gh.close_issue(1) is True
    merge = subprocess.run(
        ["git", "merge", "other"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge.returncode != 0  # conflict on README; merge state present
    assert dirty_tracker_files(repo_root, issues_dir) == ("docs/issues/001_issue.md",)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.deferred is True
    assert result.defer_reason == "MERGE_HEAD references an interrupted merge/rebase"
    assert result.committed == ()
    subprocess.run(
        ["git", "merge", "--abort"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("ref", ["MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD"])
def test_flush_defers_on_each_interrupted_operation_state_ref(
    ref: str,
    _seeded_tracker_repo: _SeededTrackerRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every slot of the batched ``rev-parse --git-path`` probe defers: a
    state file at any position must read as an interrupted operation, and
    all three resolve in one spawn (the #2690 batching, asserted on argv)."""
    repo_root = _seeded_tracker_repo.repo_root
    issues_dir = _seeded_tracker_repo.issues_dir
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    assert gh.close_issue(1) is True
    head = _run_git(repo_root, ["git", "rev-parse", "HEAD"]).stdout.strip()
    (repo_root / ".git" / ref).write_text(f"{head}\n", encoding="utf-8")
    real = commits_module.run_captured
    probes: list[list[str]] = []

    def recording_run_captured(command: list[str], **kwargs: object) -> RunResult:
        if "rev-parse" in command:
            probes.append(list(command))
        return real(command, **kwargs)

    monkeypatch.setattr(commits_module, "run_captured", recording_run_captured)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.deferred is True
    assert result.committed == ()
    assert result.defer_reason == f"{ref} references an interrupted merge/rebase"
    assert probes == [
        [
            "git",
            "rev-parse",
            "--git-path",
            "MERGE_HEAD",
            "--git-path",
            "REBASE_HEAD",
            "--git-path",
            "CHERRY_PICK_HEAD",
        ]
    ]


def test_flush_defers_when_the_state_ref_probe_fails(
    _seeded_tracker_repo: _SeededTrackerRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``--git-path`` probe defers rather than guessing: without a
    resolved path list the flush cannot prove the repo is safe to commit."""
    repo_root = _seeded_tracker_repo.repo_root
    issues_dir = _seeded_tracker_repo.issues_dir
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    assert gh.close_issue(1) is True
    real = commits_module.run_captured

    def refusing_probe(command: list[str], **kwargs: object) -> RunResult:
        if "--git-path" in command:
            return RunResult(
                returncode=128, stdout="", stderr="simulated probe refusal", error="boom"
            )
        return real(command, **kwargs)

    monkeypatch.setattr(commits_module, "run_captured", refusing_probe)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.deferred is True
    assert result.committed == ()
    assert "simulated probe refusal" in result.defer_reason


def test_flush_defers_on_conflicts_inside_issues_dir(tmp_path: Path) -> None:
    """A conflicted issue file is never half-committed: the whole flush defers."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 1)
    _seed_tracked(repo_root, path)
    _git(repo_root, "checkout", "-b", "other")
    path.write_text(
        '---\ntitle: "issue 1"\nstate: closed\nlabels: []\n---\nFrom other.\n',
        encoding="utf-8",
    )
    _seed_tracked(repo_root, path)
    _git(repo_root, "checkout", "main")
    path.write_text(
        '---\ntitle: "issue 1"\nstate: open\nlabels: [agent:queued]\n---\nFrom main.\n',
        encoding="utf-8",
    )
    _seed_tracked(repo_root, path)
    merge = subprocess.run(
        ["git", "merge", "other"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge.returncode != 0

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.deferred is True
    assert "unresolved merge conflicts" in result.defer_reason
    assert result.committed == ()
    subprocess.run(
        ["git", "merge", "--abort"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_flush_is_a_benign_skip_on_a_non_git_repo(tmp_path: Path) -> None:
    """Test doubles and non-git consumers: never raises, no commit attempted."""
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)

    result = flush_tracker_writes(tmp_path, issues_dir)

    assert result.skipped is True
    assert result.skip_reason == "not a git repository"
    assert result.committed == ()


def test_flush_honors_kill_switch(tmp_path: Path) -> None:
    """``commit_writes=False``: nothing commits, the dirt stays, and the
    result says so without an anomaly signal (config is not a failure)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    path = _write_issue(issues_dir, 1)
    _seed_tracked(repo_root, path)
    path.write_text(
        '---\ntitle: "issue 1"\nstate: closed\nlabels: []\n---\nBody.\n',
        encoding="utf-8",
    )
    queue_issue_write(repo_root, issues_dir, path, "close")

    result = flush_tracker_writes(repo_root, issues_dir, commit_writes=False)

    assert result.skipped is True
    assert result.skip_reason == "local_issues.commit_writes is false"
    assert result.committed == ()
    assert result.left_dirty == ("docs/issues/001_issue.md",)
    assert dirty_tracker_files(repo_root, issues_dir) == ("docs/issues/001_issue.md",)


def test_failing_group_does_not_block_other_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One issue's git add refusing must not stop the other issues committing."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    first = _write_issue(issues_dir, 1)
    second = _write_issue(issues_dir, 2, slug="stuck")
    _seed_tracked(repo_root, first, second)
    for candidate in (first, second):
        candidate.write_text(
            "---\ntitle: x\nstate: closed\nlabels: [agent:queued]\n---\nDone.\n",
            encoding="utf-8",
        )
    real = commits_module.run_captured

    def refusing_add_for_stuck(command: list[str], **kwargs: object) -> RunResult:
        if "add" in command and "docs/issues/002_stuck.md" in command:
            return RunResult(
                returncode=128, stdout="", stderr="simulated staging refusal", error="boom"
            )
        return real(command, **kwargs)

    monkeypatch.setattr(commits_module, "run_captured", refusing_add_for_stuck)

    result = flush_tracker_writes(repo_root, issues_dir)

    assert result.committed == ("chore(issues): update #1",)
    assert result.reason and "simulated staging refusal" in result.reason
    assert result.left_dirty == ("docs/issues/002_stuck.md",)
    assert dirty_tracker_files(repo_root, issues_dir) == ("docs/issues/002_stuck.md",)


def test_dirty_tracker_files_filters_non_issue_files(tmp_path: Path) -> None:
    """Only the NNN_*.md issue files are candidates; README/_template and
    stray files under issues_dir are left for the operator."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    issues_dir.mkdir(parents=True)
    _write_issue(issues_dir, 1)
    (issues_dir / "README.md").write_text("notes\n", encoding="utf-8")
    (issues_dir / "_template.md").write_text("---\nstate: open\n---\n", encoding="utf-8")

    assert dirty_tracker_files(repo_root, issues_dir) == ("docs/issues/001_issue.md",)


def test_queue_reuses_one_flush_per_repo_pair(tmp_path: Path) -> None:
    """The pending queue keys per (repo_root, issues_dir): a second consumer
    repo's queue is never drained by another repo's flush."""
    first_repo = tmp_path / "one"
    second_repo = tmp_path / "two"
    first_issues = first_repo / "docs" / "issues"
    second_issues = second_repo / "docs" / "issues"
    first_issue = _write_issue(first_issues, 1)
    second_issue = _write_issue(second_issues, 2)
    queue_issue_write(first_repo, first_issues, first_issue, "close")
    queue_issue_write(second_repo, second_issues, second_issue, "comment")

    from charlie_work.local_issue_commits import _pop_pending

    assert _pop_pending(first_repo, first_issues).keys() == {first_issue.as_posix()}
    # Popping the first repo's entry leaves the second repo's queue intact,
    # and the verbs recorded against each file survive the round trip.
    assert _pop_pending(second_repo, second_issues) == {second_issue.as_posix(): ["comment"]}
    assert _pop_pending(first_repo, first_issues) == {}
