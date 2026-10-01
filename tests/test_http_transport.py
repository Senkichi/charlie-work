"""The GitHub transport seen through ``GitHub.run`` and the argv table (issue #1834, ADR-0006).

No live network anywhere in this file. The pooled-HTTP code this file was
written against (``http_transport`` / ``http_translate``) is gone: a ``gh``
argv is now looked up in the closed ``legacy_argv`` table and sent as a typed
request through the ``GuardedTransport``. The adapters underneath are the
scripted ``FakeAdapter``s from ``tests/_fake_transport.py`` (one real
``HttpAdapter`` over a fake connection for the ETag case).

Covers, in order:
  * ``legacy_argv.request_for_argv`` -- the closed table: which shapes
    translate, which fail closed (no row).
  * ``http_cache`` -- round trip, corruption fail-closed, bounded eviction.
  * ``config.RuntimeConfig.gh_transport`` -- defaults to "http"; validation.
  * ``GitHub.run`` end to end against the fakes: success, REST/graphql error
    translation, 401 token re-resolution, pagination, ETag conditional GET,
    per-call fallback with the ``github_transport_fallback`` event,
    connection-level failures.
  * Parity: for each translated error shape, ``transient_errors.
    is_transient_network_error``, ``github._is_not_found_gh_error`` and
    ``circuit_breaker.classify_gh_failure`` agree with what the identical text
    coming from a real ``gh`` subprocess would already classify -- the "error
    translation at the seam" requirement.
  * The circuit breaker is one shared state across both adapters; the kill
    switch (``gh_transport: gh``) sends the same requests through gh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _fake_transport import (
    FakeAdapter,
    failure,
    gh_kill_switch_runtime,
    make_github,
    ok,
    token_ok,
)
from charlie_work import github as github_module
from charlie_work.config import ConfigError, RuntimeConfig, build_config_from_data
from charlie_work.github_capabilities import http_cache
from charlie_work.github_capabilities.circuit_breaker import (
    GhFailureClass,
    classify_gh_failure,
)
from charlie_work.github_transport import (
    FailureKind,
    GraphQLRequest,
    HttpAdapter,
    Response,
    RestRequest,
)
from charlie_work.github_transport.json_read import JsonRead
from charlie_work.github_transport.legacy_argv import request_for_argv
from charlie_work.instrumentation import query_events
from charlie_work.transient_errors import is_transient_network_error

# ---------------------------------------------------------------------------
# Helpers -- scripted adapters, no live network anywhere in this file.
# ---------------------------------------------------------------------------


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / ".var" / "charlie-work" / "state.json"


def _runtime(tmp_path: Path, **overrides: object) -> RuntimeConfig:
    """No retries (a scripted failure must not consume the next reply)."""
    return RuntimeConfig(
        state_dir=str(tmp_path / ".var" / "charlie-work"),
        gh_max_retries=0,
        **overrides,  # type: ignore[arg-type]
    )


def _run_http(
    tmp_path: Path,
    *,
    args: list[str],
    script: list,
    gh: FakeAdapter | None = None,
    runtime: RuntimeConfig | None = None,
):
    """``GitHub.run(args)`` over a scripted http adapter: ``(result, http, gh)``."""
    github, http, gh_adapter = make_github(
        tmp_path,
        http=FakeAdapter("http", script),
        gh=gh,
        runtime=runtime if runtime is not None else _runtime(tmp_path),
    )
    result = github.run(args, allow_failure=True)
    return result, http, gh_adapter


def _page(items: list, next_url: str | None = None):
    headers = {"Link": f'<{next_url}>; rel="next"'} if next_url else None
    return ok(items, headers=headers)


_NEXT = "https://api.github.com/repos/a/b/issues?page=2"


# ---------------------------------------------------------------------------
# Candidacy / translation (pure functions, no I/O)
# ---------------------------------------------------------------------------


def test_rest_get_is_candidate():
    translated = request_for_argv(["api", "rate_limit"])
    assert translated is not None
    assert isinstance(translated.request, RestRequest)


def test_graphql_query_is_candidate():
    translated = request_for_argv(["api", "graphql", "-f", "query=query { viewer { login } }"])
    assert translated is not None
    assert isinstance(translated.request, GraphQLRequest)


def test_graphql_mutation_is_not_candidate():
    assert request_for_argv(["api", "graphql", "-f", "query=mutation { addLabel }"]) is None


# Shapes that the table now models on purpose (a dialect read, a log
# download); every other shape below has no row at all.
_MODELLED_ELSEWHERE = {
    ("api", "repos/{owner}/{repo}/actions/jobs/1/logs"): RestRequest,
    ("issue", "list", "--json", "number"): JsonRead,
    ("pr", "view", "1", "--json", "state"): JsonRead,
}


@pytest.mark.parametrize(
    "args",
    [
        ["api", "repos/{owner}/{repo}/issues/1/labels", "-f", "labels[]=bug"],
        ["api", "repos/{owner}/{repo}/issues/1", "-X", "PATCH"],
        ["api", "repos/{owner}/{repo}/actions/jobs/1/logs"],
        ["issue", "list", "--json", "number"],
        ["pr", "view", "1", "--json", "state"],
        ["pr", "merge", "1", "--squash"],
        ["api", "repos/{owner}/{repo}/issues", "--jq", ".[].number"],
        # Issue #1834 review defect 1: unknown flags must fail CLOSED (not
        # be silently dropped and served over HTTP with different output
        # than gh's own).
        ["api", "repos/{owner}/{repo}/issues", "--slurp"],
        ["api", "repos/{owner}/{repo}/issues", "-i"],
        ["api", "repos/{owner}/{repo}/issues", "--include"],
        ["api", "repos/{owner}/{repo}/issues", "--template", "x"],
        ["api", "repos/{owner}/{repo}/issues", "-t", "x"],
        ["api", "repos/{owner}/{repo}/issues", "--method", "GET"],
        ["api", "repos/{owner}/{repo}/issues", "--hostname", "example.com"],
        ["api", "repos/{owner}/{repo}/issues", "--cache", "1h"],
        ["api", "repos/{owner}/{repo}/issues", "-p", "1"],
        ["api", "repos/{owner}/{repo}/issues", "--preview", "x"],
        ["api", "-q", ".foo", "repos/{owner}/{repo}/issues"],
        # A second positional (not a recognized flag's value) is also not a
        # single unambiguous endpoint.
        ["api", "repos/{owner}/{repo}/issues", "repos/{owner}/{repo}/pulls"],
        # graphql-query shape, issue #1834 review defect 1 (fail-closed
        # applies here too): only the query/owner/name -f fields the
        # graphql row reads are recognized.
        ["api", "graphql", "-f", "query=query { x }", "-H", "Accept: application/json"],
        ["api", "graphql", "-f", "query=query { x }", "-f", "unknownfield=bar"],
        ["api", "graphql", "-f", "query=query { x }", "--jq", ".data"],
    ],
)
def test_excluded_shapes_are_not_candidates(args):
    translated = request_for_argv(args)
    modelled = _MODELLED_ELSEWHERE.get(tuple(args))
    if modelled is None:
        assert translated is None
    else:
        assert isinstance(translated.request, modelled)


def test_paginate_before_path_is_candidate():
    """Issue #1834 review defect 2: `workflow._gh_api_list` -- the only
    `--paginate` call site in this codebase -- puts the flag before the
    path (`["api", "--paginate", path]`). The shape must translate, or
    pagination is dead code.
    """
    translated = request_for_argv(["api", "--paginate", "repos/{owner}/{repo}/pulls"])
    assert translated is not None
    assert translated.paginate is True


def test_build_request_plan_substitutes_owner_repo_and_headers(tmp_path: Path):
    github, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({})]))
    github.run(
        [
            "api",
            "repos/{owner}/{repo}/compare/a...b",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ],
        json_output=True,
    )
    (request,) = http.api_requests
    assert request.method == "GET"
    assert request.route == "repos/octo/hello/compare/a...b"
    assert request.accept == "application/vnd.github.v3.diff"


def test_build_request_plan_detects_paginate_flag():
    translated = request_for_argv(["api", "repos/{owner}/{repo}/issues", "--paginate"])
    assert translated.paginate is True


def test_build_request_plan_detects_paginate_flag_before_path():
    """Issue #1834 review defect 2: the real `--paginate` call shape
    (`workflow._gh_api_list`) puts the flag before the path."""
    translated = request_for_argv(["api", "--paginate", "repos/{owner}/{repo}/pulls"])
    assert translated.paginate is True
    assert translated.request.route == "repos/{owner}/{repo}/pulls"


def test_build_request_plan_graphql_body(tmp_path: Path):
    github, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"data": {}})]))
    github.run(
        ["api", "graphql", "-f", "query=query { x }", "-f", "owner=acme", "-f", "name=widgets"],
        json_output=True,
    )
    (request,) = http.api_requests
    assert isinstance(request, GraphQLRequest)
    assert request.document == "query { x }"
    assert json.loads(request.variables) == {"owner": "acme", "name": "widgets"}


# ---------------------------------------------------------------------------
# http_cache
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    assert http_cache.get_cached(path, "/repos/a/b") is None
    http_cache.record_response(path, "/repos/a/b", etag='"abc"', status=200, body='{"ok":true}')
    cached = http_cache.get_cached(path, "/repos/a/b")
    assert cached is not None
    assert cached.etag == '"abc"'
    assert cached.body == '{"ok":true}'


def test_cache_corrupt_file_fails_closed(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    path.write_text("not json{{{", encoding="utf-8")
    assert http_cache.load_cache(path) == {}
    assert http_cache.get_cached(path, "/x") is None


def test_cache_evicts_oldest_beyond_max_entries(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    monkeypatch.setattr(http_cache, "_MAX_ENTRIES", 2)
    http_cache.record_response(path, "/a", etag="1", status=200, body="a")
    http_cache.record_response(path, "/b", etag="2", status=200, body="b")
    http_cache.record_response(path, "/c", etag="3", status=200, body="c")
    entries = http_cache.load_cache(path)
    assert len(entries) == 2
    assert "/c" in entries  # most recent survives


# ---------------------------------------------------------------------------
# Config default
# ---------------------------------------------------------------------------


def test_gh_transport_defaults_to_http():
    assert RuntimeConfig().gh_transport == "http"


def test_gh_transport_rejects_unknown_value():
    with pytest.raises(ConfigError):
        build_config_from_data({"runtime": {"gh_transport": "carrier-pigeon"}})


def test_gh_transport_accepts_gh_kill_switch():
    config = build_config_from_data({"runtime": {"gh_transport": "gh"}})
    assert config.runtime.gh_transport == "gh"


def test_runtime_none_falls_back_to_gh_transport():
    """ADR-0006 inverted this: a GitHub built without a RuntimeConfig uses
    HTTP like every other instance (the name is kept for the collect-only
    gate; the kill switch is an explicit ``gh_transport: gh``).
    """
    assert github_module.GitHub(Path("."))._transport_v2.kill_switch is False
    assert (
        github_module.GitHub(Path("."), runtime=RuntimeConfig())._transport_v2.kill_switch is False
    )
    gh_runtime = RuntimeConfig(gh_transport="gh")
    assert github_module.GitHub(Path("."), runtime=gh_runtime)._transport_v2.kill_switch is True


# ---------------------------------------------------------------------------
# GitHub.run: success paths
# ---------------------------------------------------------------------------


def test_rest_get_success(tmp_path: Path):
    result, http, _ = _run_http(
        tmp_path, args=["api", "rate_limit"], script=[ok({"resources": {}})]
    )
    assert result.returncode == 0
    assert result.stdout == '{"resources": {}}'
    assert result.stderr == ""
    (request,) = http.api_requests
    assert request.method == "GET"
    assert request.route == "rate_limit"
    assert http.calls[0].token == "tok-1"  # the bearer token the adapter signs with


def test_graphql_success(tmp_path: Path):
    reply = ok({"data": {"viewer": {"login": "x"}}})
    result, http, _ = _run_http(
        tmp_path,
        args=["api", "graphql", "-f", "query=query { viewer { login } }"],
        script=[reply],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"data": {"viewer": {"login": "x"}}}
    (request,) = http.api_requests
    assert isinstance(request, GraphQLRequest)
    assert request.document == "query { viewer { login } }"


def test_non_candidate_never_touches_http(tmp_path: Path):
    """An argv with no row is refused outright: it reaches neither adapter,
    never resolving a token or opening a connection."""
    github, http, gh_adapter = make_github(tmp_path, runtime=_runtime(tmp_path))
    with pytest.raises(github_module.GitHubError, match="unsupported argv"):
        github.run(["pr", "merge", "1", "--squash"])
    assert http.calls == [] and gh_adapter.calls == []


def test_gh_kill_switch_never_touches_http(tmp_path: Path):
    gh_adapter = FakeAdapter("gh", [ok({})], token="tok-1")
    github, http, gh_adapter = make_github(
        tmp_path, gh=gh_adapter, runtime=gh_kill_switch_runtime()
    )
    result = github.run(["api", "rate_limit"])
    assert result == "{}"
    assert http.calls == []
    assert len(gh_adapter.api_requests) == 1


# ---------------------------------------------------------------------------
# Error translation parity -- the hard requirement.
# ---------------------------------------------------------------------------


def test_rest_404_translates_and_classifies_not_found(tmp_path: Path):
    result, _, _ = _run_http(
        tmp_path,
        args=["api", "repos/{owner}/{repo}/issues/999999"],
        script=[ok({"message": "Not Found"}, status=404)],
    )
    assert result.returncode == 1
    assert "HTTP 404" in result.stderr
    assert github_module._is_not_found_gh_error(result.stderr) is True
    assert is_transient_network_error(result.stderr) is False


def test_rest_401_translates_and_classifies_terminal(tmp_path: Path):
    bad = ok({"message": "Bad credentials"}, status=401)
    result, _, _ = _run_http(tmp_path, args=["api", "rate_limit"], script=[bad, bad])
    assert result.returncode == 1
    assert "bad credentials" in result.stderr.lower()
    assert "HTTP 401" in result.stderr
    assert is_transient_network_error(result.stderr) is False
    assert classify_gh_failure(result.stderr) is GhFailureClass.SEMANTIC


def test_rest_403_rate_limit_translates_and_classifies_transient(tmp_path: Path):
    result, _, _ = _run_http(
        tmp_path,
        args=["api", "rate_limit"],
        script=[ok({"message": "API rate limit exceeded for user"}, status=403)],
    )
    assert result.returncode == 1
    assert "rate limit" in result.stderr.lower()
    assert is_transient_network_error(result.stderr) is True


def test_rest_502_translates_and_classifies_transient(tmp_path: Path):
    result, _, _ = _run_http(
        tmp_path, args=["api", "rate_limit"], script=[ok("Bad Gateway", status=502)]
    )
    assert result.returncode == 1
    assert "HTTP 502" in result.stderr
    assert is_transient_network_error(result.stderr) is True


def test_graphql_not_found_error_matches_real_gh_format(tmp_path: Path):
    """Verified against this repo's own live `gh api graphql` output for a
    nonexistent object: `GraphQL: Could not resolve to a X with ... (path)`.
    """
    body = {
        "data": None,
        "errors": [
            {
                "message": "Could not resolve to a PullRequest with the number of 999999.",
                "path": ["repository", "pullRequest"],
            }
        ],
    }
    from charlie_work.github_transport import GraphQLError

    reply = Response(
        200,
        (),
        json.dumps(body),
        "http",
        graphql_errors=(
            GraphQLError(
                "Could not resolve to a PullRequest with the number of 999999.",
                "NOT_FOUND",
                ("repository", "pullRequest"),
            ),
        ),
    )
    result, _, _ = _run_http(
        tmp_path, args=["api", "graphql", "-f", "query=query { x }"], script=[reply]
    )
    assert result.returncode == 1
    assert result.stderr == (
        "GraphQL: Could not resolve to a PullRequest with the number of 999999. "
        "(repository.pullRequest)"
    )
    assert github_module._is_not_found_gh_error(result.stderr) is True


# ---------------------------------------------------------------------------
# 401 re-resolution, pagination, ETag conditional GET
# ---------------------------------------------------------------------------


def test_401_re_resolves_token_once_then_succeeds(tmp_path: Path):
    gh_adapter = FakeAdapter("gh", [token_ok("stale-token"), token_ok("fresh-token")])
    result, http, _ = _run_http(
        tmp_path,
        args=["api", "rate_limit"],
        script=[ok({"message": "Bad credentials"}, status=401), ok({"ok": True})],
        gh=gh_adapter,
    )
    assert result.returncode == 0
    assert [call.token for call in http.calls] == ["stale-token", "fresh-token"]


def test_paginate_follows_link_header_and_concatenates(tmp_path: Path):
    result, http, _ = _run_http(
        tmp_path,
        args=["api", "repos/{owner}/{repo}/issues", "--paginate"],
        script=[_page([{"n": 1}], _NEXT), _page([{"n": 2}])],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{"n": 1}, {"n": 2}]
    assert http.api_requests[1].route == "repos/a/b/issues"
    assert dict(http.api_requests[1].query) == {"page": "2"}


def test_paginate_flag_before_path_paginates_across_two_pages(tmp_path: Path):
    """Issue #1834 review defect 2, end to end: the real `--paginate` call
    shape (`workflow._gh_api_list`) is `["api", "--paginate", path]` -- flag
    before the path. It must actually paginate, not return page 1.
    """
    next_url = "https://api.github.com/repos/octo/hello/pulls?page=2"
    result, http, _ = _run_http(
        tmp_path,
        args=["api", "--paginate", "repos/{owner}/{repo}/pulls"],
        script=[_page([{"n": 1}], next_url), _page([{"n": 2}])],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{"n": 1}, {"n": 2}]
    assert http.api_requests[0].route == "repos/octo/hello/pulls"
    assert http.api_requests[1].route == "repos/octo/hello/pulls"
    assert dict(http.api_requests[1].query) == {"page": "2"}


# ---------------------------------------------------------------------------
# A partial pagination is never returned as a (truncated) success (issue
# #1834 review defect 3). B6: each page is its own guarded call, so a failing
# page IS the outcome -- there is no gh replay of the whole walk any more, and
# so no fallback event either.
# ---------------------------------------------------------------------------


def _run_paginate(tmp_path: Path, script: list):
    result, _, _ = _run_http(
        tmp_path, args=["api", "repos/{owner}/{repo}/issues", "--paginate"], script=script
    )
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    return result, events


def test_paginate_page_two_error_falls_back_to_gh_and_emits_event(tmp_path: Path):
    """A non-2xx later page must not be returned as a (silently partial)
    success -- before the fix, a break here returned page 1's items alone
    with returncode 0. Now the failing page is the outcome."""
    result, events = _run_paginate(
        tmp_path, [_page([{"n": 1}], _NEXT), ok("Internal Server Error", status=500)]
    )
    assert result.ok is False
    assert "HTTP 500" in result.stderr
    assert events == []  # an HTTP status is an answer, not a transport fallback


def test_paginate_page_two_unparseable_json_falls_back(tmp_path: Path):
    result, _ = _run_paginate(tmp_path, [_page([{"n": 1}], _NEXT), ok("not valid json")])
    assert result.ok is False
    assert result.stdout != json.dumps([{"n": 1}])


def test_paginate_page_two_non_list_falls_back(tmp_path: Path):
    result, _ = _run_paginate(tmp_path, [_page([{"n": 1}], _NEXT), ok({"n": 2})])
    assert result.ok is False
    assert result.stdout != json.dumps([{"n": 1}])


def test_paginate_cap_reached_with_next_link_falls_back(tmp_path: Path):
    """Hitting the page cap while a `rel="next"` link is still present means
    the result would have been truncated -- it is an error, never the pages
    collected so far."""
    from charlie_work.github_transport import pagination

    github, http, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [_page([{"n": 1}], _NEXT)]),
        runtime=_runtime(tmp_path),
    )
    outcome = pagination.paginate_rest(
        github._transport_v2,
        RestRequest.of("GET", "repos/{owner}/{repo}/issues"),
        max_pages=1,
    )
    assert not (isinstance(outcome, Response) and outcome.ok)


def test_paginate_object_first_page_falls_back(tmp_path: Path):
    """`gh --paginate` merges object-shaped pages by key; the transport does
    not replicate that, so an object first page is refused rather than being
    returned as page 1 unmodified (a different, and wrong, shape than gh's own
    merged-object output)."""
    result, _ = _run_paginate(tmp_path, [ok({"total_count": 1})])
    assert result.ok is False
    assert result.stdout != '{"total_count": 1}'


def test_etag_cache_serves_body_on_304(tmp_path: Path):
    """The real ``HttpAdapter`` + the on-disk ``FileEtagCache`` over a fake
    connection: the second GET carries the cached ETag and a 304 is answered
    from the cache as a 200."""
    from _fake_transport import FakeConn, FakeRaw

    conn = FakeConn([FakeRaw(200, {"ETag": '"v1"'}, b'{"n": 1}'), FakeRaw(304, {}, b"")])
    adapter = HttpAdapter(
        connection_factory=lambda host, timeout: conn,
        cache=http_cache.FileEtagCache(tmp_path / "http-etag-cache.json"),
    )
    request = RestRequest.of("GET", "rate_limit")
    first = adapter.send(request, token="tok", timeout=5.0)
    assert isinstance(first, Response) and first.body == '{"n": 1}'

    second = adapter.send(request, token="tok", timeout=5.0)
    assert isinstance(second, Response)
    assert second.status == 200
    assert second.body == '{"n": 1}'
    # The second request carried the cached ETag as If-None-Match.
    assert conn.requests[1][3]["If-None-Match"] == '"v1"'


# ---------------------------------------------------------------------------
# Per-call fallback: token failure, unexpected shape -- with event emission.
# ---------------------------------------------------------------------------


def test_token_resolution_failure_falls_back_to_gh_and_emits_event(tmp_path: Path):
    from charlie_work.github_transport import Response as _Response

    gh_adapter = FakeAdapter("gh", [_Response(1, (), "not logged in", "gh", returncode=1), ok({})])
    result, http, gh_adapter = _run_http(
        tmp_path, args=["api", "rate_limit"], script=[], gh=gh_adapter
    )
    assert result.stdout == "{}"
    assert http.calls == []  # no token: HTTP was never attempted
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    assert len(events) == 1
    assert "token" in events[0]["payload"]["reason"].lower()


def test_unexpected_graphql_shape_falls_back_and_emits_event(tmp_path: Path):
    """An unusable reply (adapter defect) is replayed through gh, once."""
    gh_adapter = FakeAdapter("gh", [ok({})], token="tok-1")
    result, _, _ = _run_http(
        tmp_path,
        args=["api", "graphql", "-f", "query=query { x }"],
        script=[failure(FailureKind.ADAPTER_DEFECT, "response was not JSON")],
        gh=gh_adapter,
    )
    assert result.stdout == "{}"
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Connection-level failures: translated for the retry path.
# ---------------------------------------------------------------------------


def test_connection_refused_is_translated_not_fallback(tmp_path: Path):
    """B4: a CONNECT failure is provably pre-send, so it replays through gh
    (the one failure kind that does), and a replay that also fails surfaces the
    translated "error connecting to" text the mutating-call retry path keys on.
    """
    gh_adapter = FakeAdapter(
        "gh",
        [failure(FailureKind.CONNECT, "error connecting to api.github.com", "gh")],
        token="tok-1",
    )
    result, _, _ = _run_http(
        tmp_path,
        args=["api", "rate_limit"],
        script=[failure(FailureKind.CONNECT, "error connecting to api.github.com")],
        gh=gh_adapter,
    )
    assert result.returncode == 1
    assert "error connecting to" in result.stderr.lower()
    assert is_transient_network_error(result.stderr) is True


def test_socket_timeout_raises_timeout_expired(tmp_path: Path):
    """A timeout maps to the 124 exit ``gh`` itself would report, and
    ``run()`` without ``allow_failure`` raises it as a ``GitHubError``."""
    github, _, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [failure(FailureKind.TIMEOUT, "timed out")]),
        runtime=_runtime(tmp_path),
    )
    result = github.run(["api", "rate_limit"], allow_failure=True)
    assert result.returncode == 124
    with pytest.raises(github_module.GitHubError):
        github.run(["api", "rate_limit"])


# ---------------------------------------------------------------------------
# GitHub.run() integration: shared circuit breaker, retry loop reuse.
# ---------------------------------------------------------------------------


def test_circuit_breaker_shared_across_http_and_gh_transports(tmp_path: Path):
    """One breaker state trips from HTTP-transport-class failures, and once
    open, a subsequent call that would have gone to the gh adapter also fails
    fast without reaching it -- proving the breaker is not per-adapter.
    """
    from charlie_work.config import GhCircuitBreakerConfig

    connect = failure(FailureKind.CONNECT, "error connecting to api.github.com")
    runtime = RuntimeConfig(
        gh_max_retries=0,
        gh_circuit_breaker=GhCircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0),
    )
    # The gh replay of each CONNECT failure fails the same way, so the breaker
    # counts the call as a transport-class failure.
    gh_adapter = FakeAdapter("gh", [connect], token="tok-1")
    github, http, gh_adapter = make_github(
        tmp_path, http=FakeAdapter("http", [connect]), gh=gh_adapter, runtime=runtime
    )
    for _ in range(3):
        result = github.run(["api", "rate_limit"], json_output=True, allow_failure=True)
        assert result.ok is False

    assert github._circuit_breaker_state.phase == "open"

    gh_calls_before = len(gh_adapter.calls)
    http_calls_before = len(http.calls)
    # A call routed to the *other* adapter (a gh-local command) must now also
    # fail fast, proving the shared breaker gates ``run()`` itself, upstream
    # of adapter selection.
    result = github.run(["auth", "status"], allow_failure=True)
    assert result.ok is False
    assert "circuit breaker open" in (result.error or "").lower()
    assert len(gh_adapter.calls) == gh_calls_before
    assert len(http.calls) == http_calls_before


def test_runtime_none_github_instance_uses_gh_subprocess_for_api_calls(tmp_path: Path):
    """Kill switch (issue #1834): `gh_transport: gh` sends `gh api`-shaped
    calls through the gh adapter. (Name kept for the collect-only gate; a
    runtime-less GitHub now uses HTTP, ADR-0006.)
    """
    gh_adapter = FakeAdapter("gh", [ok({})], token="tok-1")
    github, http, gh_adapter = make_github(
        tmp_path, gh=gh_adapter, runtime=gh_kill_switch_runtime()
    )
    result = github.run(["api", "rate_limit"])
    assert result == "{}"
    assert http.calls == []
