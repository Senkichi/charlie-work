"""PR-lifecycle mutation tests: ``pr_close``, ``pr_reopen``, and
``push_empty_commit`` (issue #1274, W17).

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module


# --- pr_close / pr_reopen / push_empty_commit (issue #1274, W17) -----------
#
# These three exercise only the GitHub-client-surface contract in isolation
# (dry-run synthetic-ok, allow_failure propagation, never-raises) -- nothing
# in review()'s janitor-gate path calls them yet. That wiring, and its own
# fixture-level tests (AC3-AC8/AC10/AC11), land in a later step of this
# item.


def test_pr_close_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    """Dry-run must short-circuit before any `gh` call, exactly like pr_ready."""

    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run pr_close must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.pr_close(42)

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_pr_close_success_returns_ok_result(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_close(42)

    assert calls == [["gh", "pr", "close", "42"]]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_close_failure_never_raises_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="gh: pull request #42 not found"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_close(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "not found" in (result.error or "")


def test_pr_reopen_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run pr_reopen must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.pr_reopen(42)

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_pr_reopen_success_returns_ok_result(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_reopen(42)

    assert calls == [["gh", "pr", "reopen", "42"]]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_reopen_failure_never_raises_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="gh: could not reopen"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_reopen(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "could not reopen" in (result.error or "")


def _make_fake_push_empty_commit_run(
    *,
    tip_sha: str = "tip-sha-abc",
    tree_sha: str = "tree-sha-def",
    new_sha: str = "new-sha-ghi",
    fail_at_step: int | None = None,
):
    """Build a fake ``subprocess.run`` covering ``push_empty_commit``'s four
    ordered `gh api` calls: GET ref -> GET commit -> POST commit -> PATCH
    ref. ``fail_at_step`` (1-indexed) makes that call return a nonzero exit
    so callers can assert the method stops there rather than proceeding
    against inconsistent state.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        step = len(calls)
        if fail_at_step is not None and step == fail_at_step:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=f"boom at step {step}"
            )
        joined = " ".join(cmd)
        if "-X" not in cmd and "refs/heads/" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"object": {"sha": tip_sha}}),
                stderr="",
            )
        if "-X" not in cmd and "git/commits/" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"tree": {"sha": tree_sha}}),
                stderr="",
            )
        if "-X" in cmd and "POST" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"sha": new_sha}), stderr=""
            )
        if "-X" in cmd and "PATCH" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"object": {"sha": new_sha}}),
                stderr="",
            )
        raise AssertionError(f"unexpected gh api call in push_empty_commit test: {cmd}")

    return fake_run, calls


def test_push_empty_commit_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    """Dry-run must short-circuit before ANY `gh` call -- including the
    read-only ref/commit lookups -- because the operation as a whole is
    unconditionally mutating.
    """

    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run push_empty_commit must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_push_empty_commit_success_walks_all_four_steps(monkeypatch, tmp_path: Path) -> None:
    fake_run, calls = _make_fake_push_empty_commit_run()
    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert len(calls) == 4
    # GET ref, GET commit -> read-only, no -X
    assert "-X" not in calls[0]
    assert "-X" not in calls[1]
    # POST new commit object, PATCH ref -> mutating
    assert "POST" in calls[2]
    assert "PATCH" in calls[3]


@pytest.mark.parametrize("fail_at_step", [1, 2, 3, 4])
def test_push_empty_commit_never_raises_stops_at_first_failure(
    fail_at_step: int, monkeypatch, tmp_path: Path
) -> None:
    """A failure at any of the four steps returns ok=False and does not
    attempt any later step against now-inconsistent state.
    """
    fake_run, calls = _make_fake_push_empty_commit_run(fail_at_step=fail_at_step)
    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.error
    assert len(calls) == fail_at_step
