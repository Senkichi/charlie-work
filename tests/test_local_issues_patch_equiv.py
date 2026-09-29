"""Issue #1967: a closed blocker whose work landed by rebase/cherry-pick stops blocking.

``_closed_issue_work_unlanded`` used ``git merge-base --is-ancestor`` as its
sole "did the work land?" test. Ancestry only recognises work that merged:
rebased or cherry-picked commits are new objects whose patches sit on HEAD
while the original tip is never an ancestor, so a correctly-landed branch
kept its closed issue blocking forever. The fallback
``local_lane.is_patch_equivalent`` (``git cherry <base> <tip>`` -- landed
iff no output line starts with ``+``) closes that gap; a ``git cherry``
failure returns False so the conservative unlanded verdict survives.

These tests run a real git repo (helpers mirror
``test_local_issues_merge_gate.py`` -- self-contained per the
zero-cross-test-import guard). The observability half is exercised through
the drain seam ``LocalFileGitHub`` exposes: ``are_issues_open`` collects
satisfied issues on the client, ``OrchestratorApp`` drains them into one
``local_blocker_satisfied_by_patch_equivalence`` event per issue.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import charlie_work.local_lane as local_lane
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub, drain_patch_equiv_satisfied
from charlie_work.paths import runtime_paths
from charlie_work.subprocess_runner import RunResult, run_captured
from charlie_work.workflow import OrchestratorApp


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (args, result.stderr)
    return result


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit on ``main`` -- the same shape
    ``test_local_issues_merge_gate._init_repo`` builds (``core.longpaths``
    for Windows basetemp paths, repo-local identity, ``GIT_*`` env scrub)."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "test")
    _git(repo_root, "commit", "--allow-empty", "-m", "chore: seed")


def _write_issue(issues_dir: Path, number: int, *, state: str, body: str = "Body.") -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_issue.md"
    path.write_text(f"---\nstate: {state}\nlabels: []\n---\n{body}\n", encoding="utf-8")
    return path


def _commit_on_branch(repo_root: Path, branch: str, filename: str) -> None:
    _git(repo_root, "switch", "-c", branch)
    (repo_root / filename).write_text("x = 1\n", encoding="utf-8")
    _git(repo_root, "add", filename)
    _git(repo_root, "commit", "-m", "work")
    _git(repo_root, "switch", "main")


def _advance_main(repo_root: Path) -> None:
    """One unrelated commit on main ahead of the branch's fork point.

    Required before any cherry-pick in these tests: ``git cherry-pick``
    fast-forwards when HEAD is the picked commit's parent, which would move
    main *to the branch tip object* and satisfy the ancestry check -- the
    exact shape this issue is NOT about. An intervening main commit forces
    every pick to become a new object with only a matching patch-id.
    """
    (repo_root / "main_only.py").write_text("m = 0\n", encoding="utf-8")
    _git(repo_root, "add", "main_only.py")
    _git(repo_root, "commit", "-m", "main advances")


@pytest.fixture
def local_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A git repo whose issue #1 is closed and issue #2 is ``Blocked by #1``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1, state="closed", body="Add helper.")
    _write_issue(issues_dir, 2, state="open", body="Blocked by #1\n\nUse the helper.")
    return repo_root, issues_dir


def test_closed_blocker_cherry_picked_work_counts_as_landed(
    local_repo: tuple[Path, Path],
) -> None:
    """The issue's shape: the branch's commit was cherry-picked onto main --
    new object, same patch-id. The tip is not an ancestor yet must not
    block."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    assert drain_patch_equiv_satisfied(gh) == {1: (("agent/issue-1-add-helper",), "HEAD")}


def test_closed_blocker_reapplied_diff_counts_as_landed(
    local_repo: tuple[Path, Path],
) -> None:
    """Rebase-shaped landing: a multi-commit branch whose commits were
    replayed onto HEAD as new objects (the ``rebase``/``cherry-pick``
    landing shape). Every original tip commit has a patch-id twin on HEAD,
    so the branch must not block even though its tip is not an ancestor."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "switch", "agent/issue-1-add-helper")
    (repo_root / "extra.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo_root, "add", "extra.py")
    _git(repo_root, "commit", "-m", "more work")
    _git(repo_root, "switch", "main")
    # Replay each branch commit onto main: new SHAs, same patch-ids.
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "main~1..agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    assert drain_patch_equiv_satisfied(gh) == {1: (("agent/issue-1-add-helper",), "HEAD")}


def test_closed_blocker_ancestry_merged_collects_nothing(
    local_repo: tuple[Path, Path],
) -> None:
    """An ancestry-merged branch satisfies the blocker on the first check
    alone -- the patch-equivalence fallback never runs for it, so nothing
    is collected for the drain (no event for ancestry-only landing)."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    assert drain_patch_equiv_satisfied(gh) == {}


def test_closed_blocker_partially_landed_still_blocks(
    local_repo: tuple[Path, Path],
) -> None:
    """A two-commit branch with only its first patch on HEAD still has a
    ``+`` line under ``git cherry`` -- patch equivalence is all-or-nothing,
    so the blocker stands."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "switch", "agent/issue-1-add-helper")
    (repo_root / "unlanded.py").write_text("z = 3\n", encoding="utf-8")
    _git(repo_root, "add", "unlanded.py")
    _git(repo_root, "commit", "-m", "unlanded work")
    _git(repo_root, "switch", "main")
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "agent/issue-1-add-helper~1")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == {1}
    assert drain_patch_equiv_satisfied(gh) == {}


def test_cherry_failure_keeps_blocker(
    local_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``git cherry`` returns False from ``is_patch_equivalent`` --
    the equivalence check can only ever *remove* a blocker on positive
    patch-id proof, never on an inconclusive subprocess."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "agent/issue-1-add-helper")
    real_run_captured = run_captured

    def _flaky_cherry(command: list[str] | str, **kwargs: object) -> RunResult:
        if isinstance(command, list) and command[:2] == ["git", "cherry"]:
            return RunResult(returncode=128, stdout="", stderr="simulated cherry failure")
        return real_run_captured(command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(local_lane, "run_captured", _flaky_cherry)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == {1}
    assert drain_patch_equiv_satisfied(gh) == {}


def test_collection_is_once_per_issue_per_instance(
    local_repo: tuple[Path, Path],
) -> None:
    """Repeated ``are_issues_open`` calls in one pass collect one record --
    ``_patch_equiv_noted`` dedupes like ``_unmerged_blocker_warned`` does
    for its warning."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    gh.are_issues_open([1])
    gh.are_issues_open([1])
    gh.are_issues_open([1])
    assert drain_patch_equiv_satisfied(gh) == {1: (("agent/issue-1-add-helper",), "HEAD")}
    assert drain_patch_equiv_satisfied(gh) == {}


def test_drain_is_a_no_op_on_a_backend_that_does_not_collect() -> None:
    """The remote ``GitHub`` client has no ``_patch_equiv_satisfied`` field;
    the drain must answer ``{}`` for any backend that lacks it."""

    class _RemoteLike:
        pass

    assert drain_patch_equiv_satisfied(_RemoteLike()) == {}  # type: ignore[arg-type]


def test_orchestrator_drain_emits_event_once_per_issue(
    local_repo: tuple[Path, Path],
) -> None:
    """The pass-level drain turns a collected satisfaction into exactly one
    ``local_blocker_satisfied_by_patch_equivalence`` event -- and the
    events.db dedupe keeps it one even when a *fresh* client re-collects it
    (the fleet loop rebuilds the client every pass)."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _advance_main(repo_root)
    _git(repo_root, "cherry-pick", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    config = OrchestratorConfig()
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)

    gh.are_issues_open([1])
    app._drain_local_blocker_patch_equiv()

    events = query_events(paths.state_file, kind="local_blocker_satisfied_by_patch_equivalence")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["issue_number"] == 1
    assert payload["branch"] == "agent/issue-1-add-helper"
    assert payload["base"] == "HEAD"

    # Second drain of the same client: nothing left to collect.
    app._drain_local_blocker_patch_equiv()
    # A fresh client on the same repo re-collects the satisfaction (its
    # `_patch_equiv_noted` is empty); the drain must still not re-emit --
    # the events.db record is the durable once-per-issue guard.
    gh2 = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    gh2.are_issues_open([1])
    assert drain_patch_equiv_satisfied(gh2) == {1: (("agent/issue-1-add-helper",), "HEAD")}
    app2 = OrchestratorApp(repo_root, paths, config, gh2)
    app2._drain_local_blocker_patch_equiv()
    assert (
        len(query_events(paths.state_file, kind="local_blocker_satisfied_by_patch_equivalence"))
        == 1
    )


def test_orchestrator_drain_emits_nothing_for_unmerged_blocker(
    local_repo: tuple[Path, Path],
) -> None:
    """A genuinely-unmerged branch collects nothing, so the drain emits
    nothing -- the signal must mean 'patch equivalence satisfied', not
    'are_issues_open ran'."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    config = OrchestratorConfig()
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)

    assert gh.are_issues_open([1]) == {1}
    app._drain_local_blocker_patch_equiv()

    assert (
        query_events(paths.state_file, kind="local_blocker_satisfied_by_patch_equivalence") == []
    )
