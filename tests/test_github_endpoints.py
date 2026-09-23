"""Single-endpoint wrapper tests: ``check_graphql_rate_limit``,
``compare_diff``, ``commit_check_runs``, and ``remove_pr_label`` -- thin
``gh``/``gh api`` calls and their return-shape contracts.

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig
from _github_fixtures import _read_fixture


def test_check_graphql_rate_limit_parses_live_payload(monkeypatch, tmp_path: Path) -> None:
    """Issue #398: the rate-limit guard must parse a live ``gh api rate_limit`` payload."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:3] == ["gh", "api", "rate_limit"]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=rate_limit_json, stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=1500)

    # Fixture has graphql.remaining == 4114, reset is a unix timestamp.
    assert sufficient is True
    assert remaining == 4114
    assert isinstance(reset_at, int)


def test_check_graphql_rate_limit_below_threshold_returns_insufficient(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #398: when remaining points are below the threshold the guard reports insufficient."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=rate_limit_json, stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=5000)

    assert sufficient is False
    assert remaining == 4114
    assert isinstance(reset_at, int)


def test_compare_diff_hits_three_dot_compare_with_diff_media_type(
    monkeypatch, tmp_path: Path
) -> None:
    """compare_diff must call the three-dot compare endpoint with the diff
    media type Accept header (not the default JSON compare metadata), and
    return the raw response body unwrapped from GitHubRunResult."""
    seen_cmd: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        seen_cmd.extend(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.compare_diff("sha-old", "sha-new")

    assert seen_cmd[:2] == ["gh", "api"]
    assert seen_cmd[2] == "repos/{owner}/{repo}/compare/sha-old...sha-new"
    assert "-H" in seen_cmd
    h_idx = seen_cmd.index("-H")
    assert seen_cmd[h_idx + 1] == "Accept: application/vnd.github.v3.diff"
    assert result == "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"


def test_compare_diff_returns_none_on_failure(monkeypatch, tmp_path: Path) -> None:
    """A failed compare (404, GC'd SHA, API error) must return None, never
    raise — errors are returned as values, per the GitHub wrapper's pattern."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="HTTP 404: Not Found",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    # gh_transport="gh": this test exercises the gh-subprocess failure path
    # directly via `fake_run`. `compare_diff`'s REST-GET-with-header shape is
    # an HTTP-transport candidate (issue #1834), and RuntimeConfig's
    # gh_transport field defaults to "http" -- pinning "gh" here keeps this
    # test exercising what it was written to test rather than the (already
    # separately covered, see tests/test_http_transport.py) HTTP path.
    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=0, gh_transport="gh"))
    result = gh.compare_diff("sha-old", "sha-new")

    assert result is None


def test_commit_check_runs_wraps_rest_endpoint(monkeypatch, tmp_path: Path) -> None:
    """reconcile.py's aviator_stale_blocked detection needs output.summary,
    which gh pr checks --json cannot surface (its description field is always
    empty for App-created Check Runs) -- this is the only path that can."""
    payload = {
        "check_runs": [
            {
                "id": 90085390042,
                "name": "aviator/checks",
                "status": "completed",
                "conclusion": "failure",
                "output": {
                    "title": "Aviator checks - blocked",
                    "summary": (
                        "This PR is not ready to merge (currently in state blocked): "
                        "PR has a blocked label, remove to re-queue."
                    ),
                },
            }
        ]
    }

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        assert cmd[2] == "repos/{owner}/{repo}/commits/abc123/check-runs"
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    check_runs = gh.commit_check_runs("abc123")

    assert check_runs is not None
    assert check_runs[0]["name"] == "aviator/checks"
    assert check_runs[0]["conclusion"] == "failure"
    assert "blocked label" in check_runs[0]["output"]["summary"]


def test_commit_check_runs_returns_none_on_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    assert gh.commit_check_runs("missing-sha") is None


def test_remove_pr_label_invokes_gh_pr_edit(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    ok = gh.remove_pr_label(1400, "blocked")

    assert ok is True
    assert calls[-1][:5] == ["gh", "pr", "edit", "1400", "--remove-label"]
    assert calls[-1][5] == "blocked"
