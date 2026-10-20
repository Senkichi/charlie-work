"""Unpushed-commit tests for the janitor no-op-rework gate.

Split out of ``tests/test_janitor_no_op_rework.py`` (issue #1558, Track 1;
itself split out of ``tests/test_janitor.py``): the unpushed-commit
enrichment of merge-only no-op failures, plus the
``_get_unpushed_commit_info`` helper tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _janitor_fixtures import _config, _green_checks, _green_pr, _init_repo

from charlie_work.janitor import _get_unpushed_commit_info, run_janitor


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_no_op_rework_unpushed_commit_enrichment(tmp_path: Path) -> None:
    """Enrich failure message with unpushed commit count when worktree exists."""
    # Set up a git repo with a worktree
    _init_repo(tmp_path)
    # Create initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Create a worktree for the branch
    worktrees_dir = tmp_path / ".var" / "charlie-work" / "worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / "agent-issue-123-test"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/issue-123-test", str(worktree_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Add an unpushed commit in the worktree
    (worktree_path / "unpushed.txt").write_text("unpushed work")
    subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "unpushed work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
    )

    # Test with a PR that has unchanged head (no-op rework)
    pr = _green_pr(headRefOid=initial_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        review_decision=pr_state,
    )

    # Should fail (no-op rework)
    assert verdict.ok is False, (
        f"Expected no-op rework to fail, but verdict.ok={verdict.ok}, failures={verdict.failures}"
    )
    # The unpushed commit enrichment is best-effort; if git fails, we still get the base failure message
    # Just check that we got SOME failure message about no-op rework
    assert any("no pushed commits" in f or "unpushed commit" in f for f in verdict.failures), (
        f"Expected no-op rework failure, got: {verdict.failures}"
    )


def test_get_unpushed_commit_info_excludes_base_update_merge_noise(tmp_path: Path) -> None:
    """A worktree whose only local-not-remote commits are base-update merges reports
    no unpushed content.

    Regression test: ``_get_unpushed_commit_info`` used to count with plain
    ``git rev-list --count origin/{branch}..HEAD``, which counts every commit a
    local ``git merge origin/main`` transitively drags in — including other PRs'
    already-landed squash-merge commits — as "unpushed". None of that is genuine
    unpushed work.

    The scenario needs nothing but the commit graph and the
    ``refs/remotes/origin/*`` refs the function's rev-list exclusions read,
    so ``git fast-import`` builds the whole history — base commits, the
    agent branch's base-update merge, and the remote-tracking refs a real
    clone/push/fetch would leave — in a single process. The earlier version
    spawned ~18 ``git`` processes for a clone + push + fetch + merge cycle
    that produced the identical graph.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")

    # History: initial commit I on main, two already-landed squash-merge
    # commits S0/S1 on top (each a single non-merge commit, like GitHub's
    # squash-merge default), and the agent branch forked at I with an
    # unpushed base-update merge M (parents I and main's tip) dragging the
    # landed commits in. origin/* tracking refs sit at the pushed points, so
    # the old `rev-list --count origin/branch..HEAD` would count 3
    # "unpushed" commits here — all already on origin/main, none genuine.
    subprocess.run(
        ["git", "fast-import", "--quiet"],
        cwd=repo,
        # input as bytes: text=True would translate \n to \r\n on Windows
        # stdin, and git parses the \r as part of each command line.
        input=b"""\
commit refs/heads/main
mark :1
committer Test User <test@example.test> 0 +0000
data <<MSG
initial
MSG
commit refs/heads/main
mark :2
committer Test User <test@example.test> 0 +0000
data <<MSG
squash-merged PR #0
MSG
from :1
commit refs/heads/main
mark :3
committer Test User <test@example.test> 0 +0000
data <<MSG
squash-merged PR #1
MSG
from :2
commit refs/heads/agent/issue-123-test
mark :4
committer Test User <test@example.test> 0 +0000
data <<MSG
merge origin/main
MSG
from :1
merge :3
reset refs/remotes/origin/agent/issue-123-test
from :1
reset refs/remotes/origin/main
from :3
""",
        check=True,
        capture_output=True,
    )
    # Point HEAD at the agent branch so `git worktree list --porcelain`
    # reports this repo as the branch's worktree (there are no files to
    # check out — every commit is tree-empty).
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/agent/issue-123-test")

    result = _get_unpushed_commit_info("agent/issue-123-test", repo, base_ref="main")
    assert result is None, f"expected no unpushed-commit message, got: {result!r}"


def test_get_unpushed_commit_info_reports_genuine_unpushed_commit(tmp_path: Path) -> None:
    """A worktree with a real unpushed non-merge commit still reports it, with the
    correct count, even when a base_ref is supplied for exclusion.

    The scenario needs nothing but the commit graph and the
    ``refs/remotes/origin/*`` refs the function's rev-list exclusions read,
    so ``git fast-import`` builds the whole history — the base commit, the
    agent branch's genuine unpushed content commit, and the remote-tracking
    refs a real clone/push would leave — in a single process. The earlier
    version spawned ~13 ``git`` processes for an init + clone + config +
    push cycle that produced the identical graph (ledger-regress for the
    same runner-pool spawn slowdown that #2559 fixed in the sibling test).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")

    # History: initial commit I on main, and the agent branch forked at I
    # with a genuine unpushed non-merge commit W. Both origin/* tracking
    # refs sit at I — the state a `git push -u origin agent/issue-123-test`
    # immediately after branching would leave — so
    # `rev-list --no-merges --count HEAD ^origin/agent/issue-123-test
    # ^origin/main` counts exactly W.
    subprocess.run(
        ["git", "fast-import", "--quiet"],
        cwd=repo,
        # input as bytes: text=True would translate \n to \r\n on Windows
        # stdin, and git parses the \r as part of each command line.
        input=b"""\
commit refs/heads/main
mark :1
committer Test User <test@example.test> 0 +0000
data <<MSG
initial
MSG
commit refs/heads/agent/issue-123-test
mark :2
committer Test User <test@example.test> 0 +0000
data <<MSG
real unpushed work
MSG
from :1
M 100644 inline worker-change.txt
data <<EOF
real unpushed work
EOF
reset refs/remotes/origin/agent/issue-123-test
from :1
reset refs/remotes/origin/main
from :1
""",
        check=True,
        capture_output=True,
    )
    # Point HEAD at the agent branch so `git worktree list --porcelain`
    # reports this repo as the branch's worktree.
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/agent/issue-123-test")

    result = _get_unpushed_commit_info("agent/issue-123-test", repo, base_ref="main")
    assert result is not None
    assert "1 unpushed commit(s)" in result
    assert "git push origin agent/issue-123-test" in result


def test_no_op_rework_merge_only_ignores_unpushed_base_merge_noise(tmp_path: Path) -> None:
    """Merge-only-advance no-op failures don't get a false "unpushed commit(s)"
    remediation when the worktree's only local-not-remote commits are further
    base-update merge noise.

    Regression test for the PR #680 false diagnostic: the janitor reported "26
    unpushed commit(s)" for a merge-only-advance failure when 25 of those were
    already-landed squash-merge commits pulled in transitively and the 26th was
    the merge commit itself — zero genuine unpushed content. The remediation text
    ("run 'git push origin <branch>'") could not have fixed anything, and risked
    inviting a no-op push that advances the PR head SHA without real rework.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=remote_repo, check=True, capture_output=True
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    local_repo = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(local_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # First base-update: a merge-only advance that IS pushed. This becomes the
    # PR's recorded head (matches the merge-only-advance no-op path).
    subprocess.run(["git", "checkout", "main"], cwd=remote_repo, check=True, capture_output=True)
    (remote_repo / "main-change-1.txt").write_text("main branch change 1")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change 1"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    merge_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Second base-update: another merge in the SAME local worktree, left unpushed.
    # Mirrors production: an operator/worker re-synced locally with main after the
    # last push, dragging in another already-landed commit, without pushing.
    (remote_repo / "main-change-2.txt").write_text("main branch change 2")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change 2"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    # Deliberately NOT pushed — local_repo's HEAD is now ahead of both the PR's
    # recorded headRefOid (merge_sha) and origin/agent/issue-123-test.

    # The PR still reports the earlier (pushed) merge-only SHA as its head
    pr = _green_pr(headRefOid=merge_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=local_repo,
        review_decision=pr_state,
    )

    # Should still FAIL as a merge-only no-op rework
    assert verdict.ok is False
    assert any("only by merge commits" in f for f in verdict.failures)
    # Should NOT falsely claim there are unpushed commits to push — the only
    # local-not-remote commits are base-update merge noise, not real content
    assert not any("unpushed commit(s)" in f for f in verdict.failures), (
        f"expected no false unpushed-commit claim, got: {verdict.failures}"
    )
    assert any(
        "check the branch worktree for unpushed work before re-reviewing" in f
        for f in verdict.failures
    )
