"""Tests for main_ci_reclaim.py (issues #863, #815).

Covers the safety invariant end to end: never cancel a started run, never
cancel main's current tip, only cancel a strict ancestor of tip -- re-checked
immediately before cancellation to close the list-then-cancel race window.
Never uses a real ``gh`` call; ``FakeGh`` below is a self-contained double
that never touches the network or a live GitHub Actions run. Git operations
run against real temporary repos (the ancestor logic shells out to real
``git``), never a live checkout.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from charlie_work.github import GitHubRunResult
from charlie_work.main_ci_reclaim import (
    _is_strict_ancestor,
    _object_exists,
    reclaim_superseded_main_ci_runs,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")


def _commit(repo_root: Path, message: str) -> str:
    _git(repo_root, "commit", "--allow-empty", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _clone(origin: Path, repo_root: Path) -> None:
    subprocess.run(
        ["git", "clone", str(origin), str(repo_root)], check=True, capture_output=True, text=True
    )
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")


@dataclass
class FakeGh:
    """Minimal GitHubLike double for main_ci_reclaim.py's three call shapes.

    Only implements ``run()`` and ``commit()`` -- the only two methods
    main_ci_reclaim.py calls. Always returns ``GitHubRunResult`` for
    ``allow_failure=True`` calls, matching the real ``GitHub.run()``'s
    contract exactly (the module under test always passes
    ``allow_failure=True``). ``commit()`` also returns ``GitHubRunResult``
    to match the real ``GitHub.commit()`` contract (issue #1140): callers
    read ``.value`` for the dict and ``.error`` for the failure reason.
    """

    dry_run: bool = False
    tip_commits: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    # branch -> error string to simulate a failed commit lookup. When set,
    # ``commit()`` returns a ``GitHubRunResult`` with ``ok=False`` and this
    # error, modeling a gh outage (TLS blip, rate limit, auth failure, etc.).
    tip_commit_errors: dict[str, str] = field(default_factory=dict)
    workflow_runs: list[dict[str, Any]] = field(default_factory=list)
    # run_id -> status to report on the pre-cancel re-fetch. Defaults to the
    # same status as in workflow_runs (no drift between list and re-fetch)
    # unless overridden here to simulate the started-in-between race.
    refetch_status_overrides: dict[int, str] = field(default_factory=dict)
    # run_id -> False to simulate a failed cancel call.
    cancel_ok_overrides: dict[int, bool] = field(default_factory=dict)
    cancel_calls: list[int] = field(default_factory=list)
    list_calls: int = 0
    refetch_calls: list[int] = field(default_factory=list)

    def commit(self, sha: str) -> GitHubRunResult:
        if sha in self.tip_commit_errors:
            return GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr=self.tip_commit_errors[sha],
                value=None,
                error=self.tip_commit_errors[sha],
            )
        commit = self.tip_commits.get(sha)
        if not isinstance(commit, dict):
            return GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="",
                value=None,
                error=f"commit {sha} not found",
            )
        return GitHubRunResult(ok=True, returncode=0, stdout="", stderr="", value=commit)

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        assert allow_failure is True, "main_ci_reclaim.py must always pass allow_failure=True"
        if args[0] == "api" and "/actions/workflows/" in args[1] and "/runs?" in args[1]:
            self.list_calls += 1
            return GitHubRunResult(
                ok=True,
                returncode=0,
                stdout="",
                stderr="",
                value={"workflow_runs": self.workflow_runs},
            )
        if args[0] == "api" and "/actions/runs/" in args[1]:
            run_id = int(args[1].rsplit("/", 1)[-1])
            self.refetch_calls.append(run_id)
            status = self.refetch_status_overrides.get(run_id)
            if status is None:
                status = next(
                    (r["status"] for r in self.workflow_runs if r["id"] == run_id), "queued"
                )
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value={"status": status}
            )
        if args[0] == "run" and args[1] == "cancel":
            run_id = int(args[2])
            self.cancel_calls.append(run_id)
            ok = self.cancel_ok_overrides.get(run_id, True)
            return GitHubRunResult(
                ok=ok,
                returncode=0 if ok else 1,
                stdout="",
                stderr="",
                error=None if ok else "cancel failed",
            )
        raise AssertionError(f"unexpected gh.run call: {args}")


@pytest.fixture
def repo_with_history(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    """A cloned repo with a linear main history c1 -> c2 -> c3(tip), plus a
    divergent branch commit ``d1`` whose object IS fetched (full clone, not
    ``--single-branch``) but is not an ancestor of tip. Returns
    (repo_root, c1, c2, c3, d1)."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    c1 = _commit(origin, "c1")
    c2 = _commit(origin, "c2")
    c3 = _commit(origin, "c3")
    _git(origin, "checkout", "-b", "feature")
    d1 = _commit(origin, "d1 (divergent)")
    _git(origin, "checkout", "main")

    repo_root = tmp_path / "repo"
    _clone(origin, repo_root)
    return repo_root, c1, c2, c3, d1


def test_object_exists_true_for_known_false_for_unknown(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, c1, _c2, _c3, _d1 = repo_with_history
    assert _object_exists(repo_root, c1) is True
    assert _object_exists(repo_root, "0" * 40) is False


def test_is_strict_ancestor_true_for_real_ancestor(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, c1, _c2, c3, _d1 = repo_with_history
    assert _is_strict_ancestor(repo_root, c1, c3) is True


def test_is_strict_ancestor_false_for_equal_shas(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """A commit is not its own STRICT ancestor -- guards independently of the
    caller's separate exact-tip-match check (see the function's docstring)."""
    repo_root, _c1, _c2, c3, _d1 = repo_with_history
    assert _is_strict_ancestor(repo_root, c3, c3) is False


def test_is_strict_ancestor_false_for_divergent_branch(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, _c1, _c2, c3, d1 = repo_with_history
    assert _is_strict_ancestor(repo_root, d1, c3) is False


def test_is_strict_ancestor_false_for_unfetched_object(tmp_path: Path) -> None:
    """Fail-safe direction: an object this checkout has never heard of must
    read as 'not an ancestor', never as a false 'yes, safe to cancel'."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    c1 = _commit(origin, "c1")
    c2 = _commit(origin, "c2")
    repo_root = tmp_path / "repo"
    _clone(origin, repo_root)

    unrelated = tmp_path / "unrelated"
    _init_repo(unrelated)
    stray = _commit(unrelated, "never pushed anywhere repo_root can see")

    assert _is_strict_ancestor(repo_root, stray, c2) is False
    assert _is_strict_ancestor(repo_root, c1, stray) is False


def test_reclaim_cancels_strict_ancestor_queued_run(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, c1, _c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 111, "head_sha": c1, "status": "queued", "created_at": "t1"},
        ],
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert result.tip_sha == c3
    assert [r.run_id for r in result.cancelled] == [111]
    assert gh.cancel_calls == [111]
    assert gh.refetch_calls == [111]


def test_reclaim_never_cancels_main_tip_run(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """The run for main's own current tip must never be touched, even if its
    status is technically in the cancelable set (e.g. re-queued)."""
    repo_root, _c1, _c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 222, "head_sha": c3, "status": "queued", "created_at": "t1"},
        ],
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert result.cancelled == ()
    assert gh.cancel_calls == []
    assert gh.refetch_calls == []  # never even re-checked; excluded before that


def test_reclaim_never_cancels_run_already_started(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """A run already in_progress at list time is filtered before the ancestor
    check even runs -- in_progress is not in _CANCELABLE_STATUSES."""
    repo_root, c1, _c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 333, "head_sha": c1, "status": "in_progress", "created_at": "t1"},
        ],
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert result.cancelled == ()
    assert gh.cancel_calls == []


def test_reclaim_never_cancels_run_that_started_between_list_and_cancel(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """The core race-safety property: queued at list time, in_progress by the
    time of the pre-cancel re-fetch -- must not be cancelled."""
    repo_root, c1, _c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 444, "head_sha": c1, "status": "queued", "created_at": "t1"},
        ],
        refetch_status_overrides={444: "in_progress"},
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert result.cancelled == ()
    assert result.skipped_started_before_cancel == 1
    assert gh.cancel_calls == []
    assert gh.refetch_calls == [444]


def test_reclaim_skips_non_ancestor_candidate(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, _c1, _c2, c3, d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 555, "head_sha": d1, "status": "queued", "created_at": "t1"},
        ],
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert result.cancelled == ()
    assert result.skipped_not_ancestor == 1
    assert gh.cancel_calls == []
    assert gh.refetch_calls == []  # ancestor check fails before the re-fetch


def test_reclaim_fails_safe_when_fetch_fails(tmp_path: Path) -> None:
    """No 'origin' remote configured at all -- git fetch fails, and the pass
    must report ok=False without calling gh at all."""
    repo_root = tmp_path / "lonely-repo"
    _init_repo(repo_root)
    _commit(repo_root, "only commit")
    gh = FakeGh()
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is False
    assert result.error is not None and "fetch" in result.error
    assert gh.list_calls == 0
    assert gh.cancel_calls == []


def test_reclaim_fails_safe_when_tip_resolution_fails(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, _c1, _c2, _c3, _d1 = repo_with_history
    gh = FakeGh(tip_commits={})  # gh.commit("main") -> None
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is False
    assert result.error is not None and "tip" in result.error
    assert gh.list_calls == 0
    assert gh.cancel_calls == []


def test_reclaim_tip_resolution_failure_carries_underlying_gh_error(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """Issue #1140: when ``gh.commit()`` fails, the pass-level error must
    include the underlying gh error string (TLS blip, rate limit, auth
    failure, ...) -- not just a fixed 'failed to resolve' message. Without
    this, a ``main_ci_reclaim_failed`` event during a GitHub-side outage
    cannot say *why* resolution failed, and attribution requires manual
    timestamp correlation against a separate probe."""
    repo_root, _c1, _c2, _c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commit_errors={
            "main": "TLS handshake timeout (x509: certificate signed by unknown authority)"
        }
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is False
    assert result.error is not None
    assert "tip" in result.error
    assert "TLS handshake timeout" in result.error
    assert "certificate signed by unknown authority" in result.error
    assert gh.list_calls == 0
    assert gh.cancel_calls == []


def test_reclaim_cancel_error_is_recorded_not_fatal(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    """One run's cancel call failing (e.g. it slipped to completed right as
    the API call landed) must not abort the rest of the pass -- mirrors the
    workflow's own try/catch + core.warning + continue."""
    repo_root, c1, c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 666, "head_sha": c1, "status": "queued", "created_at": "t1"},
            {"id": 777, "head_sha": c2, "status": "queued", "created_at": "t2"},
        ],
        cancel_ok_overrides={666: False},
    )
    result = reclaim_superseded_main_ci_runs(gh, repo_root)
    assert result.ok is True
    assert [r.run_id for r in result.cancelled] == [777]
    assert len(result.cancel_errors) == 1
    assert "666" in result.cancel_errors[0]
    assert sorted(gh.cancel_calls) == [666, 777]


def test_reclaim_uses_configured_workflow_filename_and_default_branch(
    repo_with_history: tuple[Path, str, str, str, str],
) -> None:
    repo_root, c1, _c2, c3, _d1 = repo_with_history
    gh = FakeGh(
        tip_commits={"main": {"sha": c3}},
        workflow_runs=[
            {"id": 888, "head_sha": c1, "status": "queued", "created_at": "t1"},
        ],
    )
    result = reclaim_superseded_main_ci_runs(
        gh, repo_root, default_branch="main", workflow_filename="tests.yml"
    )
    assert result.ok is True
    assert gh.list_calls == 1
    assert [r.run_id for r in result.cancelled] == [888]
