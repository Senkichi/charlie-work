"""Shared git plumbing helpers for worktree-adjacent tests.

Hoisted out of ``test_worktree.py`` (issue #1284): a thin ``git`` subprocess
runner, a fresh-clone-with-identity builder, and a minimal repo initializer,
all imported by other test modules that need a real (non-bare) git repo to
exercise worktree behaviour against. ``test_worktree.py`` itself is one of
the three monoliths issue #1284 marks out of scope for a full split -- only
the exported symbols move; the rest of the file is untouched.

``_init_repo`` joined the hoist under the #1688 rework: the file-size
ratchet (issue #1442) required moving the new pre-merge-repair regression
tests out of the over-cap ``test_worktree.py`` into
``tests/test_worktree_launcher_owned_repair.py``, and the moved tests need
the same initializer, so it lives here rather than being duplicated.

Note: ``test_reconcile.py`` defines its own, byte-identical top-level
``_git`` helper. That copy has zero external importers (unlike this one)
so it stays where it is per the same out-of-scope-monolith carve-out;
``tests/_reconcile_fixtures.py`` imports this module's ``_git`` rather than
keeping a third copy.

``_init_bare_remote_and_clone`` and ``_setup_completed_worktree`` joined under
the #1548 Track-1 wave-2 split: a moved ``test_charlie_work.py`` test needs
them, and they are this module's domain -- a bare-remote-plus-clone builder
and a worktree-with-a-commit builder. ``test_charlie_work.py`` keeps its own
byte-identical ``_git`` for the same reason as ``test_reconcile.py``.
The template-unaware copy ``tests/_reconcile_fixtures.py`` used to carry
was removed under issue #2303; its former importers take this module's
builder directly so the reconcile suite shares the per-process git
template.

``_FakeGH`` joined under the #1713 rework: the new
``tests/test_worktree_clean_head_fallback.py`` needs the same
``clean_worktrees`` fake ``test_worktree.py`` defines, and the
cross-test-module import guard (``tests/test_zero_cross_test_import_guard.py``,
issue #1284) forbids a ``test_*.py -> test_*.py`` import, so the fake lives
here where both test modules can reach it.

``wt_scratch`` joined under #1944: worker sandboxes redirect
``TEMP``/``TMPDIR`` deep inside the checkout, where pytest's ``tmp_path``
overruns git's internal worktree-path buffer and ``git worktree add``
exits 128. Tests that create real linked worktrees use this fixture
instead of ``tmp_path`` for their repo roots.

``_clone_with_pushed_branch`` and ``_push_sibling_commit`` joined under the
#1476 rework: the file-size ratchet (issue #1442) required splitting
``tests/test_worktree_foreign_adoption.py`` into two thematic modules (gate /
rework adoption vs recovery / teardown / probe), and both sides need the same
bare-remote-plus-pushed-branch builders, so they live here rather than being
duplicated.

The repo builders (``_init_repo``, ``_init_repo_with_origin``,
``_clone_repo``, ``_init_bare_remote_and_clone``) copy a per-process template
when they can (HS-CW-4, ``tests/_git_templates.py``): same files, same
config, same refs as a fresh build, but one commit SHA shared by every copy
of a shape. Pass ``fresh=True`` where a test needs
independently-timestamped histories; the ``_fresh`` variants are the
original bodies, unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

import _git_templates
from charlie_work.github import GitHubRunResult
from charlie_work.worktree import WorktreeCleanGH, create_worktree


# Explicit fixture name: importing modules bring the (private) symbol into
# their namespace so pytest registers ``wt_scratch`` there — a public
# function name would collide with the fixture parameter in test
# signatures (ruff F811).
@pytest.fixture(name="wt_scratch")
def _wt_scratch() -> Iterator[Path]:
    """Per-test scratch dir shallow enough for real ``git worktree`` ops.

    Worker sandboxes redirect ``TEMP``/``TMPDIR`` deep inside the checkout
    (``.var/worker-tmp`` under the worktree), and pytest's ``tmp_path``
    inherits that depth — past ~225 chars ``git worktree add`` exits 128
    with a ``$GIT_DIR``-sized buffer failure even with ``core.longpaths``
    on. ``%LOCALAPPDATA%\\Temp`` is not redirected, so anchoring there
    stays shallow on Windows; on POSIX ``tempfile.gettempdir()`` is
    already short, and ``min()`` by path length picks the shallowest
    either way.

    The yielded path is canonicalized with ``Path.resolve()``: on
    GitHub-hosted Windows runners ``%TEMP%`` is an 8.3 short path
    (``C:\\Users\\RUNNER~1\\...``) while ``git worktree
    list --porcelain`` reports the canonical long spelling, so
    ``Path(wt["worktree"]) == worktree_path`` lookups inside
    ``create_worktree`` miss the registered worktree — it survives
    removal, and the fresh-dispatch ``git branch -D`` then fails on
    a branch still checked out in it. Resolving here keeps the
    spelling the tests pass in identical to the one git reports.
    """
    roots = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        roots.append(Path(local_appdata) / "Temp")
    roots.append(Path(tempfile.gettempdir()))
    root = min(roots, key=lambda p: len(str(p))) / "charlie-wt-scratch"
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="wt-", dir=root)).resolve()
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _seed_staged_addition_bubble(repo_root: Path) -> None:
    """Build the overwritten-paths commit bubble for the staged-addition
    scenario in one ``git fast-import`` process and check ``feature`` out.

    ``feature`` branches from the template's initial ``main`` commit; ``main``
    then gains a commit changing ``tracked.txt`` and one independently adding
    ``new_file.txt``. One spawn replaces the ~10 ``git`` children (add /
    commit / checkout per step) the inline version spawned — the per-spawn
    host cost was what the ledger's 2x call-phase regression on
    ``test_modified_paths_overwritten_by_ref_excludes_staged_addition``
    tracked (issue #2569).

    Caller leaves here exactly where it must before applying its local dirty
    state: on ``feature`` with the initial commit checked out.
    """
    subprocess.run(
        ["git", "fast-import", "--quiet"],
        cwd=repo_root,
        # input as bytes: text=True would translate \n to \r\n on Windows
        # stdin, and git parses the \r as part of each command line.
        input=b"""\
commit refs/heads/feature
mark :1
committer Test User <test@example.test> 0 +0000
data <<MSG
add ancestor file
MSG
from refs/heads/main
M 100644 inline tracked.txt
data <<EOT
ancestor v0
EOT
commit refs/heads/main
mark :2
committer Test User <test@example.test> 0 +0000
data <<MSG
base changes tracked file
MSG
from :1
M 100644 inline tracked.txt
data <<EOT
main committed v1
EOT
commit refs/heads/main
mark :3
committer Test User <test@example.test> 0 +0000
data <<MSG
base adds new file
MSG
from :2
M 100644 inline new_file.txt
data <<EOT
base version
EOT
""",
        check=True,
        capture_output=True,
    )
    # fast-import advances refs only — the index and worktree still reflect the
    # initial commit while main's tip moved, so a plain checkout would refuse
    # the stale state as local changes. ``checkout -f`` repopulates both from
    # the feature tree and moves HEAD to it.
    _git(repo_root, "checkout", "-f", "feature")


def _clone_repo(remote_repo: Path, repo_root: Path, *, fresh: bool = False) -> None:
    """``git clone`` plus the test identity; a template copy when ``remote_repo`` is an
    unchanged repo this process materialized (``_git_templates``)."""
    if not fresh and _git_templates.clone_repo(remote_repo, repo_root, build=_clone_repo_fresh):
        return
    _clone_repo_fresh(remote_repo, repo_root)


def _clone_repo_fresh(remote_repo: Path, repo_root: Path) -> None:
    subprocess.run(
        ["git", "clone", str(remote_repo), str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    # A fresh clone has no committer identity on CI runners.
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")


def _init_repo(repo_root: Path, bare: bool = False, *, fresh: bool = False) -> None:
    """One-commit ``main`` repo (``bare``: a bare clone of one); a template copy unless
    ``fresh`` or ``CI_FLEET_TEST_REUSE=off`` (``_git_templates``)."""
    if not fresh and _git_templates.init_repo(repo_root, bare=bare, build=_init_repo_fresh):
        return
    _init_repo_fresh(repo_root, bare)


def _init_repo_fresh(repo_root: Path, bare: bool = False) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    if bare:
        # Create a temporary non-bare repo, initialize it, then convert to bare
        temp_repo = repo_root.parent / f"{repo_root.name}-temp"
        temp_repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "--initial-branch=main"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.test"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        (temp_repo / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        # Convert to bare by cloning with --bare
        subprocess.run(
            ["git", "clone", "--bare", str(temp_repo), str(repo_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Clean up temp repo (ignore errors on Windows due to file locks)
        import shutil

        shutil.rmtree(temp_repo, ignore_errors=True)
    else:
        run(["git", "init", "--initial-branch=main"])
        run(["git", "config", "user.email", "test@example.test"])
        run(["git", "config", "user.name", "Test User"])
        (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
        run(["git", "add", "README.md"])
        run(["git", "commit", "-m", "initial commit"])


def _init_repo_with_origin(repo_root: Path, remote_url: str, *, fresh: bool = False) -> None:
    """``_init_repo`` plus ``git remote add origin <remote_url>``; the
    remote-add spawn is paid once per process inside the template build
    (``_git_templates.init_repo_with_origin``), so a materialized call spawns
    no process at all."""
    if not fresh and _git_templates.init_repo_with_origin(
        repo_root, remote_url, build=_init_repo_fresh
    ):
        return
    _init_repo_fresh(repo_root)
    _git(repo_root, "remote", "add", "origin", remote_url)


def _init_bare_remote_and_clone(tmp_path: Path, *, fresh: bool = False) -> tuple[Path, Path]:
    """Create a bare remote repo and a local clone, return (remote, clone).

    A template copy unless ``fresh`` or ``CI_FLEET_TEST_REUSE=off`` (``_git_templates``).
    """
    if not fresh:
        pair = _git_templates.bare_remote_and_clone(
            tmp_path, build=_init_bare_remote_and_clone_fresh
        )
        if pair is not None:
            return pair
    return _init_bare_remote_and_clone_fresh(tmp_path)


def _init_bare_remote_and_clone_fresh(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _clone_with_pushed_branch(tmp_path: Path, branch: str) -> tuple[Path, Path]:
    """Bare origin + clone + ``branch`` pushed to origin; return (remote, repo)."""
    remote = tmp_path / "remote.git"
    _init_repo(remote, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote, repo_root)
    _git(repo_root, "branch", branch)
    _git(repo_root, "push", "origin", branch)
    return remote, repo_root


def _push_sibling_commit(remote: Path, tmp_path: Path, branch: str, filename: str) -> str:
    """Advance ``origin/<branch>`` by one commit from a second clone; return the
    pushed SHA."""
    other = tmp_path / "other-clone"
    if not other.exists():
        _clone_repo(remote, other)
        _git(other, "checkout", "-b", branch, f"origin/{branch}")
    else:
        _git(other, "fetch", "origin", branch)
        _git(other, "reset", "--hard", f"origin/{branch}")
    (other / filename).write_text(f"{filename}\n", encoding="utf-8")
    _git(other, "add", filename)
    _git(other, "commit", "-m", f"remote advance {filename}")
    pushed_sha = _git(other, "rev-parse", "HEAD").stdout.strip()
    _git(other, "push", "origin", branch)
    return pushed_sha


def _setup_completed_worktree(
    repo_root: Path, issue_number: int, dirty: bool = False
) -> tuple[Path, str]:
    """Create a worktree with one commit beyond origin/main. Return (worktree_path, branch)."""
    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    if dirty:
        (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    return info.path, branch


class _FakeGH(WorktreeCleanGH):
    """Fake ``GitHub`` for ``clean_worktrees`` tests.

    Implements the ``WorktreeCleanGH`` protocol (the slice of ``GitHub`` the
    cleanup lane depends on) so it is statically assignable to
    ``clean_worktrees(..., gh=...)`` without ``cast`` (issue #641).

    ``available=False`` simulates ``gh`` itself failing/being unreachable
    (``GitHubRunResult(ok=False, ...)``), distinct from ``gh`` succeeding but
    reporting a PR state other than ``MERGED``.

    ``head_branch_prs`` maps a head-branch name to the PR numbers a
    ``gh pr list --head <branch> --state all`` call should report -- the
    issue-#1713 fallback lookup ``clean_worktrees`` makes when state.json
    has no linked PR. Every invocation is recorded in ``calls`` so tests
    can assert whether the fallback ran at all.

    ``issue_states`` maps an issue number to the ``state`` a
    ``gh issue view <n> --json state`` call should report -- the
    issue-#2487 closed-issue check ``clean_worktrees`` makes per
    candidate. Issues absent from the map report ``OPEN`` so the
    closed-issue lane stays inactive unless a test opts in.
    """

    def __init__(
        self,
        pr_state: str = "MERGED",
        merged_at: str | None = "2026-07-13T01:33:43Z",
        head_sha: str | None = None,
        *,
        available: bool = True,
        error: str = "gh: could not resolve to a PullRequest",
        head_branch_prs: dict[str, list[int]] | None = None,
        issue_states: dict[int, str] | None = None,
    ) -> None:
        self.pr_state = pr_state
        self.merged_at = merged_at
        self.head_sha = head_sha
        self.available = available
        self.error = error
        self.head_branch_prs = head_branch_prs or {}
        self.issue_states = issue_states or {}
        self.calls: list[list[str]] = []

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> GitHubRunResult:
        self.calls.append(list(args))
        if args[:2] == ["pr", "list"] and "--head" in args and json_output and allow_failure:
            if not self.available:
                return GitHubRunResult(
                    ok=False,
                    returncode=1,
                    stdout="",
                    stderr=self.error,
                    value=None,
                    error=self.error,
                )
            head = args[args.index("--head") + 1]
            return GitHubRunResult(
                ok=True,
                returncode=0,
                stdout="",
                stderr="",
                value=[{"number": number} for number in self.head_branch_prs.get(head, [])],
                error=None,
            )
        if args[:2] == ["pr", "view"] and json_output and allow_failure:
            if not self.available:
                return GitHubRunResult(
                    ok=False,
                    returncode=1,
                    stdout="",
                    stderr=self.error,
                    value=None,
                    error=self.error,
                )
            return GitHubRunResult(
                ok=True,
                returncode=0,
                stdout="",
                stderr="",
                value={
                    "state": self.pr_state,
                    "mergedAt": self.merged_at,
                    "headRefOid": self.head_sha,
                },
                error=None,
            )
        if args[:2] == ["issue", "view"] and json_output and allow_failure:
            if not self.available:
                return GitHubRunResult(
                    ok=False,
                    returncode=1,
                    stdout="",
                    stderr=self.error,
                    value=None,
                    error=self.error,
                )
            try:
                issue_number = int(args[2])
            except (IndexError, ValueError):
                issue_number = -1
            return GitHubRunResult(
                ok=True,
                returncode=0,
                stdout="",
                stderr="",
                value={"state": self.issue_states.get(issue_number, "OPEN")},
                error=None,
            )
        return GitHubRunResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            value=None,
            error="unexpected fake gh command",
        )
