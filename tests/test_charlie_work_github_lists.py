"""GitHub list/read endpoints: merged-PR pagination, issue/PR list caps, open-state cache, branch-protection cache.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import logging
from pathlib import Path
import pytest
from _fake_transport import (
    FakeAdapter,
    graphql_failure,
    graphql_ok,
    graphql_variables,
    make_github,
    ok,
    paged_connection,
    sent,
)
from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig


def test_github_merged_pr_list_uses_rest_pagination(tmp_path: Path) -> None:
    """merged_pr_list() now uses the REST pulls endpoint instead of the
    GraphQL-backed `gh pr list --state merged`, avoiding expensive field sets
    such as `statusCheckRollup` (issue #361).
    """
    gh, http, gh_adapter = make_github(tmp_path, http=FakeAdapter("http", [ok([])]))
    gh.merged_pr_list()

    assert sent(http) == [("GET", "repos/{owner}/{repo}/pulls", None)]
    assert dict(http.api_requests[0].query)["state"] == "closed"  # type: ignore[union-attr]
    assert gh_adapter.api_requests == []


def test_github_merged_pr_list_retries_on_transient_gateway_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A transient 502/503/504 from the REST pulls endpoint retries
    in-pass (bounded) instead of immediately failing the whole fleet pass for
    that repo (issue #361). Succeeds on the 2nd attempt here.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    http = FakeAdapter("http", [ok("Bad Gateway", status=502), ok([])])
    gh, http, _ = make_github(tmp_path, http=http)

    result = gh.merged_pr_list()

    assert result == []
    assert len(http.api_requests) == 2
    assert len(sleeps) == 1


def test_github_merged_pr_list_gives_up_after_max_retries(monkeypatch, tmp_path: Path) -> None:
    """Persistent 502s must eventually raise GitHubError — never hang or retry
    forever — so the per-repo fleet-pass boundary can still catch it and move
    on to the next repo.
    """
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)
    http = FakeAdapter("http", [ok("Bad Gateway", status=502)])
    gh, http, _ = make_github(tmp_path, http=http, runtime=RuntimeConfig(gh_max_retries=2))

    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert len(http.api_requests) == 3


def test_github_merged_pr_list_does_not_retry_non_transient_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A non-gateway error (e.g. bad credentials) must fail immediately rather
    than be swallowed into the transient-gateway retry loop.
    """
    http = FakeAdapter("http", [ok({"message": "Bad credentials"}, status=401)])
    gh, http, _ = make_github(tmp_path, http=http)

    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    # 401 is terminal: the guard re-resolves the token once and resends once
    # (never the transient-gateway retry loop), so exactly two sends, then raise.
    assert [(r.method, r.route) for r in http.api_requests] == [  # type: ignore[union-attr]
        ("GET", "repos/octo/hello/pulls")
    ] * 2


def test_issue_list_raises_limit_to_500_and_warns_on_truncation(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    # The GraphQL path (REST open-issue list off, issue #2443) pages to the cap.
    monkeypatch.setenv("CHARLIE_WORK_ISSUE_LIST_REST", "off")
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT
    # More nodes than the cap exist: the read pages up to the cap, then warns.
    adapter = FakeAdapter("http", handler=paged_connection("issues", limit + 100))
    gh, http, _ = make_github(tmp_path, http=adapter)

    result = gh.issue_list("automated-ready")

    assert len(result) == limit
    assert len(http.api_requests) == limit // 100
    assert any("truncated" in record.message for record in caplog.records)


def test_pr_list_raises_limit_to_500_and_warns_on_truncation(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT
    # More nodes than the cap exist: the read pages up to the cap, then warns.
    adapter = FakeAdapter("http", handler=paged_connection("pullRequests", limit + 100))
    gh, http, _ = make_github(tmp_path, http=adapter)

    result = gh.pr_list()

    assert len(result) == limit
    assert len(http.api_requests) == limit // 100
    assert any("truncated" in record.message for record in caplog.records)


def test_branch_protection_caches_per_pass(tmp_path: Path) -> None:
    """Issue #812: branch_protection() must cost exactly one request per base
    ref per orchestrator pass, not one per PR -- N callers sharing a base
    (e.g. N open PRs against main in one merge_ready/broadcast-sweep pass) must
    collapse to a single underlying read. The cache lives in GitHub._list_cache
    (the same dict pr_list/issue_list already use) and is cleared only by
    invalidate_list_cache(), which the orchestrator calls once per pass.
    """
    http = FakeAdapter("http", [ok({"required_status_checks": {"strict": True}})])
    gh, http, _ = make_github(tmp_path, http=http)

    # Simulate N=5 PRs against the same base within one pass: 5 calls to the
    # method, but the underlying request must be sent exactly once.
    results = [gh.branch_protection("main") for _ in range(5)]
    assert all(r == {"required_status_checks": {"strict": True}} for r in results)
    assert len(http.calls) == 1
    # Pin the actual endpoint (and the {owner}/{repo} placeholder), not just
    # "something got cached" -- a wrong URL would still pass a count-only check.
    assert sent(http) == [("GET", "repos/{owner}/{repo}/branches/main/protection", None)]

    # A different base ref is a distinct cache key, so it costs a fresh read.
    gh.branch_protection("develop")
    assert len(http.calls) == 2
    gh.branch_protection("develop")
    assert len(http.calls) == 2  # still cached

    # invalidate_list_cache() (called once at the top of every orchestrator
    # pass) must force a fresh read on the next call -- the cache is valid
    # only within a single pass, never leaking across passes.
    gh.invalidate_list_cache()
    gh._list_cache[("_repo_owner_name",)] = ("octo", "hello")  # make_github seeds the slug
    gh.branch_protection("main")
    assert len(http.calls) == 3


def test_branch_protection_caches_failed_read_too(monkeypatch, tmp_path: Path) -> None:
    """A failed read (404/rate-limited) must also be cached as None for the
    rest of the pass -- otherwise every PR sharing a broken base ref retries
    the same doomed `gh api` call once each, turning one outage into N.
    """
    http = FakeAdapter("http", [ok({"message": "Not Found"}, status=404)])
    gh, http, _ = make_github(tmp_path, http=http)

    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert sent(http) == [("GET", "repos/{owner}/{repo}/branches/main/protection", None)]


def test_github_are_issues_open_normalizes_uppercase_state(tmp_path: Path) -> None:
    """Issue #173: Regression test for are_issues_open with realistic uppercase state.

    Exercises the production ``are_issues_open`` per-issue fallback path with
    realistic uppercase state field values (as returned by the real GitHub API),
    ensuring the ``.upper()`` normalization cannot silently regress. The batched
    GraphQL path is forced to fail so the per-issue fallback -- the code
    performing the ``.upper()`` normalization -- is the path under test; the
    per-issue ``issue_view`` reads go out as GraphQL requests.
    """
    from charlie_work.github import GitHub as RealGitHub
    from charlie_work.github import GitHubError

    # Realistic API states: uppercase from the real API, plus one lowercase (400)
    # that must still count as open via the ``.upper()`` normalization.
    states = {100: "OPEN", 200: "CLOSED", 300: "OPEN", 400: "open"}

    def _fail_graphql(self, numbers):
        # Force are_issues_open onto its per-issue ``issue_view`` fallback.
        raise GitHubError("forced batched-state failure -> per-issue fallback")

    def view(request):
        number = graphql_variables(request)["number"]
        return graphql_ok(
            {"repository": {"issueOrPullRequest": {"number": number, "state": states[number]}}}
        )

    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=view))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RealGitHub, "_graphql_issue_states", _fail_graphql)
        result = gh.are_issues_open([100, 200, 300, 400])

    # Only the OPEN-state issues are returned: 100 and 300 (uppercase OPEN) and
    # 400 (lowercase "open", normalized via .upper()). 200 is CLOSED.
    assert result == {100, 300, 400}, f"Expected {{100, 300, 400}}, got {result}"


def test_are_issues_open_caches_per_pass_and_dedupes_shared_numbers(tmp_path: Path) -> None:
    """Issue #870: are_issues_open() was a fully serial, uncached, one
    issue view per number loop. Every distinct blocker issue number must
    now cost exactly one per-issue read per status()/orchestrator pass,
    no matter how many separate callers ask about it or how much the
    requested number lists overlap -- mirroring the existing per-pass cache
    contract already proven for branch_protection()
    (test_branch_protection_caches_per_pass).
    """
    calls: list[int] = []

    def handler(request):
        if "$number" not in request.document:
            # The batched state query: fail it so each number is read per issue.
            return graphql_failure("batched state query unavailable", "INTERNAL")
        number = graphql_variables(request)["number"]
        calls.append(number)
        state = "OPEN" if number != 200 else "CLOSED"
        return graphql_ok(
            {"repository": {"issueOrPullRequest": {"number": number, "state": state}}}
        )

    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=handler))

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
    gh._list_cache[("_repo_owner_name",)] = ("octo", "hello")  # make_github seeds the slug
    gh.are_issues_open([100])
    assert calls.count(100) == 2
