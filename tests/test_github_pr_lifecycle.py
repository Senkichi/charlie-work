"""PR-lifecycle mutation tests: ``pr_close``, ``pr_reopen``, and
``push_empty_commit`` (issue #1274, W17).

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _fake_transport import FakeAdapter, make_github, ok, sent
from charlie_work import github as github_module


# --- pr_close / pr_reopen / push_empty_commit (issue #1274, W17) -----------
#
# These three exercise only the GitHub-client-surface contract in isolation
# (dry-run synthetic-ok, allow_failure propagation, never-raises) -- nothing
# in review()'s janitor-gate path calls them yet. That wiring, and its own
# fixture-level tests (AC3-AC8/AC10/AC11), land in a later step of this
# item.


def test_pr_close_dry_run_returns_synthetic_ok_without_subprocess_call(tmp_path: Path) -> None:
    """Dry-run must short-circuit before any request is sent (the transport owns the guard)."""
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.pr_close(42)

    assert result.ok is True
    assert result.error is None
    assert http.calls == []


def test_pr_close_success_returns_ok_result(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"state": "closed"})]))
    result = gh.pr_close(42)

    assert sent(http) == [("PATCH", "repos/{owner}/{repo}/pulls/42", {"state": "closed"})]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_close_failure_never_raises_returns_error_result(tmp_path: Path) -> None:
    reply = ok({"message": "Not Found"}, status=404)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    result = gh.pr_close(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "not found" in (result.error or "").lower()


def test_pr_reopen_dry_run_returns_synthetic_ok_without_subprocess_call(tmp_path: Path) -> None:
    """Dry-run must short-circuit before any request is sent (the transport owns the guard)."""
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.pr_reopen(42)

    assert result.ok is True
    assert result.error is None
    assert http.calls == []


def test_pr_reopen_success_returns_ok_result(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"state": "open"})]))
    result = gh.pr_reopen(42)

    assert sent(http) == [("PATCH", "repos/{owner}/{repo}/pulls/42", {"state": "open"})]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_reopen_failure_never_raises_returns_error_result(tmp_path: Path) -> None:
    reply = ok({"message": "could not reopen"}, status=422)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    result = gh.pr_reopen(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "could not reopen" in (result.error or "")


def _push_empty_commit_adapter(
    *,
    tip_sha: str = "tip-sha-abc",
    tree_sha: str = "tree-sha-def",
    new_sha: str = "new-sha-ghi",
    fail_at_step: int | None = None,
) -> FakeAdapter:
    """A scripted adapter covering ``push_empty_commit``'s four ordered
    requests: GET ref -> GET commit -> POST commit -> PATCH ref. ``fail_at_step``
    (1-indexed) makes that call answer 422 so callers can assert the method
    stops there rather than proceeding against inconsistent state.
    """
    script = [
        ok({"object": {"sha": tip_sha}}),
        ok({"tree": {"sha": tree_sha}}),
        ok({"sha": new_sha}, status=201),
        ok({"object": {"sha": new_sha}}),
    ]
    if fail_at_step is not None:
        script[fail_at_step - 1] = ok({"message": f"boom at step {fail_at_step}"}, status=422)
    return FakeAdapter("http", script)


def test_push_empty_commit_dry_run_returns_synthetic_ok_without_subprocess_call(
    tmp_path: Path,
) -> None:
    """Dry-run must short-circuit before ANY request -- including the
    read-only ref/commit lookups -- because the operation as a whole is
    unconditionally mutating.
    """
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )
    assert http.calls == []


def test_push_empty_commit_success_walks_all_four_steps(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=_push_empty_commit_adapter())
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    calls = sent(http)
    assert [method for method, _, _ in calls] == ["GET", "GET", "POST", "PATCH"]
    assert calls[2][2]["tree"] == "tree-sha-def"  # type: ignore[index]
    assert calls[3][2] == {"sha": "new-sha-ghi"}


@pytest.mark.parametrize("fail_at_step", [1, 2, 3, 4])
def test_push_empty_commit_never_raises_stops_at_first_failure(
    fail_at_step: int, tmp_path: Path
) -> None:
    """A failure at any of the four steps returns ok=False and does not
    attempt any later step against now-inconsistent state.
    """
    adapter = _push_empty_commit_adapter(fail_at_step=fail_at_step)
    gh, http, _ = make_github(tmp_path, http=adapter)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.error
    assert len(http.calls) == fail_at_step
