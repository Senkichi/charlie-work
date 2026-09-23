"""GitHub list/read endpoints: merged-PR pagination, issue/PR list caps, open-state cache, branch-protection cache.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
import pytest
from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig


def test_github_merged_pr_list_uses_rest_pagination(monkeypatch, tmp_path: Path) -> None:
    """merged_pr_list() now uses the REST pulls endpoint instead of the
    GraphQL-backed `gh pr list --state merged`, avoiding expensive field sets
    such as `statusCheckRollup` (issue #361).
    """
    captured_args: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merged_pr_list()

    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[:2] == ["gh", "api"]
    assert "pulls" in args[2]
    assert "state=closed" in args[2]
    assert not any(c[:2] == ["gh", "pr"] and "merged" in c for c in captured_args)


def test_github_merged_pr_list_retries_on_transient_gateway_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A transient 502/503/504 from the REST pulls endpoint retries
    in-pass (bounded) instead of immediately failing the whole fleet pass for
    that repo (issue #361). Succeeds on the 2nd attempt here.
    """
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == []
    assert call_count == 2
    assert len(sleeps) == 1


def test_github_merged_pr_list_gives_up_after_max_retries(monkeypatch, tmp_path: Path) -> None:
    """Persistent 502s must eventually raise GitHubError — never hang or retry
    forever — so the per-repo fleet-pass boundary can still catch it and move
    on to the next repo.
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=2))
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 3


def test_github_merged_pr_list_does_not_retry_non_transient_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A non-gateway error (e.g. bad credentials) must fail immediately rather
    than be swallowed into the transient-gateway retry loop.
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 1


def test_issue_list_raises_limit_to_500_and_warns_on_truncation(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT

    def fake_run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,
    ):
        assert json_output is True
        assert args[:2] == ["issue", "list"]
        assert str(limit) in args, f"expected --limit {limit} in {args}"
        return [{"number": i} for i in range(limit)]

    monkeypatch.setattr(github_module.GitHub, "run", fake_run)
    gh = github_module.GitHub(tmp_path)

    result = gh.issue_list("automated-ready")

    assert len(result) == limit
    assert any("truncated" in record.message for record in caplog.records)


def test_pr_list_raises_limit_to_500_and_warns_on_truncation(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT

    def fake_run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,
    ):
        assert json_output is True
        assert args[:2] == ["pr", "list"]
        assert str(limit) in args, f"expected --limit {limit} in {args}"
        return [{"number": i} for i in range(limit)]

    monkeypatch.setattr(github_module.GitHub, "run", fake_run)
    gh = github_module.GitHub(tmp_path)

    result = gh.pr_list()

    assert len(result) == limit
    assert any("truncated" in record.message for record in caplog.records)


def test_branch_protection_caches_per_pass(monkeypatch, tmp_path: Path) -> None:
    """Issue #812: branch_protection() must cost exactly one `gh api` call per
    base ref per orchestrator pass, not one per PR -- N callers sharing a base
    (e.g. N open PRs against main in one merge_ready/broadcast-sweep pass) must
    collapse to a single underlying read. The cache lives in GitHub._list_cache
    (the same dict pr_list/issue_list already use) and is cleared only by
    invalidate_list_cache(), which the orchestrator calls once per pass.
    """
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        payload = json.dumps({"required_status_checks": {"strict": True}})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    # Simulate N=5 PRs against the same base within one pass: 5 calls to the
    # method, but the underlying `gh api` subprocess must run exactly once.
    results = [gh.branch_protection("main") for _ in range(5)]
    assert all(r == {"required_status_checks": {"strict": True}} for r in results)
    assert len(calls) == 1
    # Pin the actual endpoint (and the {owner}/{repo} placeholder escaping),
    # not just "something got cached" -- a wrong URL would still pass a
    # call-count-only assertion.
    assert calls[0] == ["gh", "api", "repos/{owner}/{repo}/branches/main/protection"]

    # A different base ref is a distinct cache key, so it costs a fresh read.
    gh.branch_protection("develop")
    assert len(calls) == 2
    gh.branch_protection("develop")
    assert len(calls) == 2  # still cached

    # invalidate_list_cache() (called once at the top of every orchestrator
    # pass) must force a fresh read on the next call -- the cache is valid
    # only within a single pass, never leaking across passes.
    gh.invalidate_list_cache()
    gh.branch_protection("main")
    assert len(calls) == 3


def test_branch_protection_caches_failed_read_too(monkeypatch, tmp_path: Path) -> None:
    """A failed read (404/rate-limited) must also be cached as None for the
    rest of the pass -- otherwise every PR sharing a broken base ref retries
    the same doomed `gh api` call once each, turning one outage into N.
    """
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="HTTP 404: Not Found"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert len(calls) == 1


def test_github_are_issues_open_normalizes_uppercase_state(monkeypatch, tmp_path: Path) -> None:
    """Issue #173: Regression test for are_issues_open with realistic uppercase state.

    Exercises the production ``are_issues_open`` per-issue fallback path with
    realistic uppercase state field values (as returned by the real GitHub API),
    ensuring the ``.upper()`` normalization cannot silently regress.

    Post-L07 (issue #1591) ``are_issues_open`` lives on the ``Issues`` capability
    collaborator, and its thread-pool fallback closure resolves ``self.issue_view``
    on that collaborator -- so a GitHub *subclass* override of ``issue_view`` no
    longer intercepts that internal call (the disclosed interception-path
    relocation). The mock is therefore installed at the ``run`` layer, which
    ``issue_view`` forwards to through the collaborator->owner seam, and the
    batched GraphQL path is forced to fail so the per-issue fallback -- the code
    performing the ``.upper()`` normalization -- is the path under test.
    """
    from charlie_work.github import GitHub as RealGitHub
    from charlie_work.github import GitHubError

    # Realistic API states: uppercase from the real API, plus one lowercase (400)
    # that must still count as open via the ``.upper()`` normalization.
    states = {100: "OPEN", 200: "CLOSED", 300: "OPEN", 400: "open"}

    def _fail_graphql(self, numbers):
        # Force are_issues_open onto its per-issue ``issue_view`` fallback.
        raise GitHubError("forced batched-state failure -> per-issue fallback")

    def _fake_run(self, args, **kwargs):
        # issue_view builds ["issue", "view", str(number), "--json", ISSUE_VIEW_FIELDS];
        # args[2] is the issue number. Raise ValueError (not KeyError) for an
        # unexpected number so the failure mode matches the original mock: the
        # fallback's ``except (GitHubError, ValueError, TypeError)`` swallows it to
        # ``is_open=False`` rather than propagating out of ``pool.map``.
        number = int(args[2])
        if number not in states:
            raise ValueError(f"Unexpected issue number: {number}")
        return {"number": number, "state": states[number]}

    monkeypatch.setattr(RealGitHub, "_graphql_issue_states", _fail_graphql)
    monkeypatch.setattr(RealGitHub, "run", _fake_run)

    real_gh = RealGitHub(repo_root=tmp_path)

    # Call are_issues_open with a mix of open/closed issues
    result = real_gh.are_issues_open([100, 200, 300, 400])

    # Only the OPEN-state issues are returned: 100 and 300 (uppercase OPEN) and
    # 400 (lowercase "open", normalized via .upper()). 200 is CLOSED.
    assert result == {100, 300, 400}, f"Expected {{100, 300, 400}}, got {result}"


def test_are_issues_open_caches_per_pass_and_dedupes_shared_numbers(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #870: are_issues_open() was a fully serial, uncached, one
    `gh issue view` per number loop. Every distinct blocker issue number must
    now cost exactly one `gh issue view` call per status()/orchestrator pass,
    no matter how many separate callers ask about it or how much the
    requested number lists overlap -- mirroring the existing per-pass cache
    contract already proven for branch_protection()
    (test_branch_protection_caches_per_pass).
    """
    calls: list[int] = []

    def fake_run(command, **kwargs):
        # command: ["gh", "issue", "view", "<number>", "--json", ...]
        number = int(command[3])
        calls.append(number)
        payload = json.dumps({"number": number, "state": "OPEN" if number != 200 else "CLOSED"})
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    # Two overlapping requests, as _filter_blocked_issues and _summarize_issue
    # would each independently make for two issues sharing a blocker.
    first = gh.are_issues_open([100, 200])
    second = gh.are_issues_open([100, 200, 300])

    assert first == {100}
    assert second == {100, 300}
    # 100 and 200 must not be re-fetched by the second, overlapping call;
    # only the genuinely new number (300) costs a live call.
    assert sorted(calls) == [100, 200, 300]

    # invalidate_list_cache() (called once per orchestrator pass) must force
    # a fresh read on the next call -- never leaking across passes.
    gh.invalidate_list_cache()
    gh.are_issues_open([100])
    assert calls.count(100) == 2
