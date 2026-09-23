"""Pooled HTTP transport for `gh api` REST-GET/graphql calls (issue #1834).

No live network anywhere in this file: HTTP responses are supplied by
`_FakeConnection`, a small stand-in for `http.client.HTTPSConnection`
installed via monkeypatch in place of `http_transport.HTTPSConnection`.
`gh auth token` resolution goes through the real `subprocess` module object
(shared with `github.py`'s), so it is monkeypatched the same way this
repo's existing gh-subprocess tests patch `github_module.subprocess.run`.

Covers, in order:
  * `http_translate.is_http_candidate`/`build_request_plan` -- pure
    translation/candidacy logic, no I/O.
  * `http_cache` -- round trip, corruption fail-closed, bounded eviction.
  * `config.RuntimeConfig.gh_transport` -- defaults to "http"; validation.
  * `run_gh_command` end to end against the fake connection: success,
    REST/graphql error translation, 401 token re-resolution, pagination,
    ETag conditional GET, per-call fallback (token failure, unexpected
    shape) with `github_transport_fallback` event emission, connection-level
    failures translated (not fallback).
  * Parity: for each translated error shape, `transient_errors.
    is_transient_network_error`, `github._is_not_found_gh_error`,
    `github._should_retry`, and `circuit_breaker.classify_gh_failure` agree
    with what the identical text coming from a real `gh` subprocess would
    already classify -- the "error translation at the seam" requirement.
  * `GitHub.run()` integration: the circuit breaker is one shared state
    across both transports; `runtime=None` still uses the `gh` subprocess
    (existing test-suite compatibility).
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module
from charlie_work.config import ConfigError, RuntimeConfig, build_config_from_data
from charlie_work.github_capabilities import http_cache, http_transport
from charlie_work.github_capabilities.circuit_breaker import (
    GhFailureClass,
    classify_gh_failure,
)
from charlie_work.github_capabilities.http_translate import (
    build_request_plan,
    is_http_candidate,
)
from charlie_work.instrumentation import query_events
from charlie_work.transient_errors import is_transient_network_error


# ---------------------------------------------------------------------------
# Fakes -- no live network anywhere in this file.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None, body: bytes = b""):
        self.status = status
        self._headers = headers or {}
        self._body = body

    def getheaders(self):
        return list(self._headers.items())

    def read(self) -> bytes:
        return self._body


class _FakeConnection:
    """Stand-in for `http.client.HTTPSConnection`: a queue of canned
    responses (or exceptions to raise), with every sent request recorded.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.requests: list[tuple[str, str, bytes | None, dict]] = []
        self.closed = False

    def request(self, method, url, body=None, headers=None):
        self.requests.append((method, url, body, dict(headers or {})))

    def getresponse(self):
        if not self._responses:
            raise AssertionError("no more fake HTTP responses queued")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def _install_fake_connection(monkeypatch, responses: list) -> _FakeConnection:
    fake = _FakeConnection(responses)
    monkeypatch.setattr(http_transport, "HTTPSConnection", lambda host, timeout=None: fake)
    return fake


def _install_token(monkeypatch, token: str | None = "gho_faketoken") -> None:
    def fake_run(cmd, *args, **kwargs):
        assert cmd == ["gh", "auth", "token"]
        if token is None:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=token + "\n", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_run)


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / ".var" / "charlie-work" / "state.json"


def _owner_repo():
    return ("acme", "widgets")


def _run_http(
    monkeypatch,
    tmp_path: Path,
    *,
    args: list[str],
    responses: list,
    token: str | None = "gho_faketoken",
    runtime: RuntimeConfig | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[subprocess.CompletedProcess, _FakeConnection]:
    _install_token(monkeypatch, token)
    fake = _install_fake_connection(monkeypatch, responses)
    state = http_transport.build_http_transport_state()
    runtime = runtime if runtime is not None else RuntimeConfig()
    result = http_transport.run_gh_command(
        args=args,
        command=["gh", *args],
        cwd=tmp_path,
        timeout_seconds=timeout_seconds,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    return result, fake


# ---------------------------------------------------------------------------
# Candidacy / translation (pure functions, no I/O)
# ---------------------------------------------------------------------------


def test_rest_get_is_candidate():
    assert is_http_candidate(["api", "rate_limit"]) is True


def test_graphql_query_is_candidate():
    assert is_http_candidate(["api", "graphql", "-f", "query=query { viewer { login } }"]) is True


def test_graphql_mutation_is_not_candidate():
    assert is_http_candidate(["api", "graphql", "-f", "query=mutation { addLabel }"]) is False


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
        # applies here too): only the query/owner/name -f fields
        # `_build_graphql_plan` reads are recognized.
        ["api", "graphql", "-f", "query=query { x }", "-H", "Accept: application/json"],
        ["api", "graphql", "-f", "query=query { x }", "-f", "unknownfield=bar"],
        ["api", "graphql", "-f", "query=query { x }", "--jq", ".data"],
    ],
)
def test_excluded_shapes_are_not_candidates(args):
    assert is_http_candidate(args) is False


def test_paginate_before_path_is_candidate():
    """Issue #1834 review defect 2: `workflow._gh_api_list` -- the only
    `--paginate` call site in this codebase -- puts the flag before the
    path (`["api", "--paginate", path]`). Before the fix, `_rest_path`
    assumed `args[1]` was always the endpoint, so this exact production
    shape was never a candidate and pagination was dead code.
    """
    assert is_http_candidate(["api", "--paginate", "repos/{owner}/{repo}/pulls"]) is True


def test_build_request_plan_substitutes_owner_repo_and_headers():
    plan = build_request_plan(
        [
            "api",
            "repos/{owner}/{repo}/compare/a...b",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ],
        "acme",
        "widgets",
    )
    assert plan.method == "GET"
    assert plan.path == "/repos/acme/widgets/compare/a...b"
    assert plan.headers == (("Accept", "application/vnd.github.v3.diff"),)


def test_build_request_plan_detects_paginate_flag():
    plan = build_request_plan(
        ["api", "repos/{owner}/{repo}/issues", "--paginate"], "acme", "widgets"
    )
    assert plan.paginate is True


def test_build_request_plan_detects_paginate_flag_before_path():
    """Issue #1834 review defect 2: the real `--paginate` call shape
    (`workflow._gh_api_list`) puts the flag before the path."""
    plan = build_request_plan(
        ["api", "--paginate", "repos/{owner}/{repo}/pulls"], "acme", "widgets"
    )
    assert plan.paginate is True
    assert plan.path == "/repos/acme/widgets/pulls"


def test_build_request_plan_graphql_body():
    plan = build_request_plan(
        ["api", "graphql", "-f", "query=query { x }", "-f", "owner=acme", "-f", "name=widgets"],
        "acme",
        "widgets",
    )
    assert plan.method == "POST"
    assert plan.path == "/graphql"
    body = json.loads(plan.body)
    assert body["query"] == "query { x }"
    assert body["variables"] == {"owner": "acme", "name": "widgets"}


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
    """GitHub instances built without a RuntimeConfig (this repo's own
    dominant test-suite pattern) must keep using the gh subprocess -- see
    http_transport._DEFAULT_GH_TRANSPORT's docstring.
    """
    assert http_transport.gh_transport_mode(None) == "gh"
    assert http_transport.gh_transport_mode(RuntimeConfig()) == "http"


# ---------------------------------------------------------------------------
# run_gh_command: success paths
# ---------------------------------------------------------------------------


def test_rest_get_success(monkeypatch, tmp_path: Path):
    result, fake = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "rate_limit"],
        responses=[_FakeResponse(200, {}, b'{"resources": {}}')],
    )
    assert result.returncode == 0
    assert result.stdout == '{"resources": {}}'
    assert result.stderr == ""
    method, url, body, headers = fake.requests[0]
    assert method == "GET"
    assert url == "/rate_limit"
    assert headers["Authorization"] == "Bearer gho_faketoken"


def test_graphql_success(monkeypatch, tmp_path: Path):
    result, fake = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "graphql", "-f", "query=query { viewer { login } }"],
        responses=[_FakeResponse(200, {}, b'{"data": {"viewer": {"login": "x"}}}')],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"data": {"viewer": {"login": "x"}}}
    method, url, body, headers = fake.requests[0]
    assert method == "POST"
    assert url == "/graphql"
    assert json.loads(body)["query"] == "query { viewer { login } }"


def test_non_candidate_never_touches_http(monkeypatch, tmp_path: Path):
    """A mutating/--json call must go straight to the gh subprocess, never
    resolving a token or opening a connection."""

    def fail_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fail_run)

    def boom(*a, **k):
        raise AssertionError("HTTP path must not be attempted for a non-candidate call")

    monkeypatch.setattr(http_transport, "HTTPSConnection", boom)

    result = http_transport.run_gh_command(
        args=["issue", "list", "--json", "number"],
        command=["gh", "issue", "list", "--json", "number"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=RuntimeConfig(),
        transport_state=http_transport.build_http_transport_state(),
        resolve_owner_repo=_owner_repo,
    )
    assert result.stdout == "[]"


def test_gh_kill_switch_never_touches_http(monkeypatch, tmp_path: Path):
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_run)
    monkeypatch.setattr(
        http_transport, "HTTPSConnection", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    result = http_transport.run_gh_command(
        args=["api", "rate_limit"],
        command=["gh", "api", "rate_limit"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=RuntimeConfig(gh_transport="gh"),
        transport_state=http_transport.build_http_transport_state(),
        resolve_owner_repo=_owner_repo,
    )
    assert result.stdout == "{}"


# ---------------------------------------------------------------------------
# Error translation parity -- the hard requirement.
# ---------------------------------------------------------------------------


def test_rest_404_translates_and_classifies_not_found(monkeypatch, tmp_path: Path):
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "repos/{owner}/{repo}/issues/999999"],
        responses=[_FakeResponse(404, {}, b'{"message": "Not Found"}')],
    )
    assert result.returncode == 1
    assert "HTTP 404" in result.stderr
    assert github_module._is_not_found_gh_error(result.stderr) is True
    assert is_transient_network_error(result.stderr) is False


def test_rest_401_translates_and_classifies_terminal(monkeypatch, tmp_path: Path):
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "rate_limit"],
        responses=[
            _FakeResponse(401, {}, b'{"message": "Bad credentials"}'),
            _FakeResponse(401, {}, b'{"message": "Bad credentials"}'),
        ],
    )
    assert result.returncode == 1
    assert "bad credentials" in result.stderr.lower()
    assert "HTTP 401" in result.stderr
    assert is_transient_network_error(result.stderr) is False
    assert classify_gh_failure(result.stderr) is GhFailureClass.SEMANTIC


def test_rest_403_rate_limit_translates_and_classifies_transient(monkeypatch, tmp_path: Path):
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "rate_limit"],
        responses=[_FakeResponse(403, {}, b'{"message": "API rate limit exceeded for user"}')],
    )
    assert result.returncode == 1
    assert "rate limit" in result.stderr.lower()
    assert is_transient_network_error(result.stderr) is True


def test_rest_502_translates_and_classifies_transient(monkeypatch, tmp_path: Path):
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "rate_limit"],
        responses=[_FakeResponse(502, {}, b"Bad Gateway")],
    )
    assert result.returncode == 1
    assert "HTTP 502" in result.stderr
    assert is_transient_network_error(result.stderr) is True
    assert (
        github_module._should_retry(["api", "rate_limit"], result.stderr, is_mutating=False)
        is True
    )


def test_graphql_not_found_error_matches_real_gh_format(monkeypatch, tmp_path: Path):
    """Verified against this repo's own live `gh api graphql` output for a
    nonexistent object: `GraphQL: Could not resolve to a X with ... (path)`.
    """
    body = json.dumps(
        {
            "data": None,
            "errors": [
                {
                    "message": "Could not resolve to a PullRequest with the number of 999999.",
                    "path": ["repository", "pullRequest"],
                }
            ],
        }
    ).encode("utf-8")
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "graphql", "-f", "query=query { x }"],
        responses=[_FakeResponse(200, {}, body)],
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


def test_401_re_resolves_token_once_then_succeeds(monkeypatch, tmp_path: Path):
    tokens = iter(["stale-token", "fresh-token"])

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=next(tokens) + "\n", stderr=""
        )

    monkeypatch.setattr(http_transport.subprocess, "run", fake_run)
    fake = _install_fake_connection(
        monkeypatch,
        [
            _FakeResponse(401, {}, b'{"message": "Bad credentials"}'),
            _FakeResponse(200, {}, b'{"ok": true}'),
        ],
    )
    state = http_transport.build_http_transport_state()
    result = http_transport.run_gh_command(
        args=["api", "rate_limit"],
        command=["gh", "api", "rate_limit"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=RuntimeConfig(),
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    assert result.returncode == 0
    assert fake.requests[0][3]["Authorization"] == "Bearer stale-token"
    assert fake.requests[1][3]["Authorization"] == "Bearer fresh-token"


def test_paginate_follows_link_header_and_concatenates(monkeypatch, tmp_path: Path):
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    page2 = _FakeResponse(200, {}, b'[{"n": 2}]')
    result, fake = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "repos/{owner}/{repo}/issues", "--paginate"],
        responses=[page1, page2],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{"n": 1}, {"n": 2}]
    assert fake.requests[1][1] == "/repos/a/b/issues?page=2"


def test_paginate_flag_before_path_paginates_across_two_pages(monkeypatch, tmp_path: Path):
    """Issue #1834 review defect 2, end to end: the real `--paginate` call
    shape (`workflow._gh_api_list`) is `["api", "--paginate", path]` -- flag
    before the path. Before the fix this was never even a candidate (see
    test_paginate_before_path_is_candidate), so pagination silently never
    ran for it in production.
    """
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/acme/widgets/pulls?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    page2 = _FakeResponse(200, {}, b'[{"n": 2}]')
    result, fake = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "--paginate", "repos/{owner}/{repo}/pulls"],
        responses=[page1, page2],
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{"n": 1}, {"n": 2}]
    assert fake.requests[0][1] == "/repos/acme/widgets/pulls"
    assert fake.requests[1][1] == "/repos/acme/widgets/pulls?page=2"


# ---------------------------------------------------------------------------
# Partial pagination must fall back to gh, never return a truncated result
# (issue #1834 review defect 3).
# ---------------------------------------------------------------------------


def _run_paginate_fallback(
    monkeypatch, tmp_path: Path, responses: list, gh_stdout: str = "[]"
) -> tuple[subprocess.CompletedProcess, list]:
    _install_fake_connection(monkeypatch, responses)

    def fake_gh_run(cmd, *args, **kwargs):
        if cmd == ["gh", "auth", "token"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="gho_faketoken\n", stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=gh_stdout, stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_gh_run)

    state = http_transport.build_http_transport_state()
    runtime = RuntimeConfig(state_dir=str(tmp_path / ".var" / "charlie-work"))
    result = http_transport.run_gh_command(
        args=["api", "repos/{owner}/{repo}/issues", "--paginate"],
        command=["gh", "api", "repos/{owner}/{repo}/issues", "--paginate"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    return result, events


def test_paginate_page_two_error_falls_back_to_gh_and_emits_event(monkeypatch, tmp_path: Path):
    """A non-2xx later page must not be returned as a (silently partial)
    success -- before the fix, a break here returned page 1's items alone
    with returncode 0."""
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    page2 = _FakeResponse(500, {}, b"Internal Server Error")
    result, events = _run_paginate_fallback(
        monkeypatch, tmp_path, [page1, page2], gh_stdout='[{"n": 1}, {"n": 2}]'
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{"n": 1}, {"n": 2}]
    assert len(events) == 1
    assert "page" in events[0]["payload"]["reason"].lower()


def test_paginate_page_two_unparseable_json_falls_back(monkeypatch, tmp_path: Path):
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    page2 = _FakeResponse(200, {}, b"not valid json")
    result, events = _run_paginate_fallback(monkeypatch, tmp_path, [page1, page2])
    assert result.returncode == 0
    assert len(events) == 1
    assert "page" in events[0]["payload"]["reason"].lower()


def test_paginate_page_two_non_list_falls_back(monkeypatch, tmp_path: Path):
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    page2 = _FakeResponse(200, {}, b'{"n": 2}')
    result, events = _run_paginate_fallback(monkeypatch, tmp_path, [page1, page2])
    assert result.returncode == 0
    assert len(events) == 1
    assert "page" in events[0]["payload"]["reason"].lower()


def test_paginate_cap_reached_with_next_link_falls_back(monkeypatch, tmp_path: Path):
    """Hitting `_MAX_PAGINATE_PAGES` while a `rel="next"` link is still
    present means the result would have been truncated -- must fall back to
    gh rather than silently returning the pages collected so far."""
    monkeypatch.setattr(http_transport, "_MAX_PAGINATE_PAGES", 1)
    page1 = _FakeResponse(
        200,
        {"Link": '<https://api.github.com/repos/a/b/issues?page=2>; rel="next"'},
        b'[{"n": 1}]',
    )
    result, events = _run_paginate_fallback(monkeypatch, tmp_path, [page1])
    assert result.returncode == 0
    assert result.stdout == "[]"
    assert len(events) == 1
    assert "page" in events[0]["payload"]["reason"].lower()


def test_paginate_object_first_page_falls_back(monkeypatch, tmp_path: Path):
    """`gh --paginate` merges object-shaped pages by key; this module does
    not replicate that, so an object first page must fall back to gh rather
    than being returned as page 1 unmodified (which is a different, and
    wrong, shape than gh's own merged-object output)."""
    result, events = _run_paginate_fallback(
        monkeypatch,
        tmp_path,
        [_FakeResponse(200, {}, b'{"total_count": 1}')],
        gh_stdout='{"total_count": 1}',
    )
    assert result.returncode == 0
    assert result.stdout == '{"total_count": 1}'
    assert len(events) == 1
    assert "object" in events[0]["payload"]["reason"].lower()


def test_etag_cache_serves_body_on_304(monkeypatch, tmp_path: Path):
    _install_token(monkeypatch)
    fake = _install_fake_connection(
        monkeypatch, [_FakeResponse(200, {"ETag": '"v1"'}, b'{"n": 1}')]
    )
    state = http_transport.build_http_transport_state()
    runtime = RuntimeConfig(state_dir=str(tmp_path / ".var" / "charlie-work"))
    first = http_transport.run_gh_command(
        args=["api", "rate_limit"],
        command=["gh", "api", "rate_limit"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    assert first.stdout == '{"n": 1}'

    fake._responses.append(_FakeResponse(304, {}, b""))
    second = http_transport.run_gh_command(
        args=["api", "rate_limit"],
        command=["gh", "api", "rate_limit"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    assert second.returncode == 0
    assert second.stdout == '{"n": 1}'
    # The second request carried the cached ETag as If-None-Match.
    assert fake.requests[1][3]["If-None-Match"] == '"v1"'


# ---------------------------------------------------------------------------
# Per-call fallback: token failure, unexpected shape -- with event emission.
# ---------------------------------------------------------------------------


def test_token_resolution_failure_falls_back_to_gh_and_emits_event(monkeypatch, tmp_path: Path):
    _install_token(monkeypatch, token=None)
    monkeypatch.setattr(
        http_transport, "HTTPSConnection", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    def fake_gh_run(cmd, *args, **kwargs):
        if cmd == ["gh", "auth", "token"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="not logged in"
            )
        assert cmd == ["gh", "api", "rate_limit"]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_gh_run)

    state = http_transport.build_http_transport_state()
    runtime = RuntimeConfig(state_dir=str(tmp_path / ".var" / "charlie-work"))
    result = http_transport.run_gh_command(
        args=["api", "rate_limit"],
        command=["gh", "api", "rate_limit"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    assert result.stdout == "{}"
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    assert len(events) == 1
    assert "token" in events[0]["payload"]["reason"].lower()


def test_unexpected_graphql_shape_falls_back_and_emits_event(monkeypatch, tmp_path: Path):
    _install_token(monkeypatch)
    _install_fake_connection(monkeypatch, [_FakeResponse(200, {}, b"not valid json")])

    def fake_gh_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_gh_run)

    state = http_transport.build_http_transport_state()
    runtime = RuntimeConfig(state_dir=str(tmp_path / ".var" / "charlie-work"))
    result = http_transport.run_gh_command(
        args=["api", "graphql", "-f", "query=query { x }"],
        command=["gh", "api", "graphql", "-f", "query=query { x }"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=runtime,
        transport_state=state,
        resolve_owner_repo=_owner_repo,
    )
    assert result.stdout == "{}"
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Connection-level failures: translated, NOT a fallback trigger.
# ---------------------------------------------------------------------------


def test_connection_refused_is_translated_not_fallback(monkeypatch, tmp_path: Path):
    result, _ = _run_http(
        monkeypatch,
        tmp_path,
        args=["api", "rate_limit"],
        responses=[ConnectionRefusedError("refused")],
        runtime=RuntimeConfig(state_dir=str(tmp_path / ".var" / "charlie-work")),
    )
    assert result.returncode == 1
    # github._is_pre_connection_error's allowed phrases include "error
    # connecting to" (not the raw exception text) -- this is the exact
    # phrase the translation must produce for the mutating-call retry path
    # to recognize it.
    assert "error connecting to" in result.stderr.lower()
    assert is_transient_network_error(result.stderr) is True
    assert github_module._is_pre_connection_error(result.stderr) is True
    events = query_events(_state_path(tmp_path), kind="github_transport_fallback")
    assert events == []


def test_socket_timeout_raises_timeout_expired(monkeypatch, tmp_path: Path):
    _install_token(monkeypatch)
    _install_fake_connection(monkeypatch, [socket.timeout("timed out")])
    state = http_transport.build_http_transport_state()
    with pytest.raises(subprocess.TimeoutExpired):
        http_transport.run_gh_command(
            args=["api", "rate_limit"],
            command=["gh", "api", "rate_limit"],
            cwd=tmp_path,
            timeout_seconds=5.0,
            runtime=RuntimeConfig(),
            transport_state=state,
            resolve_owner_repo=_owner_repo,
        )


# ---------------------------------------------------------------------------
# GitHub.run() integration: shared circuit breaker, retry loop reuse.
# ---------------------------------------------------------------------------


def test_circuit_breaker_shared_across_http_and_gh_transports(monkeypatch, tmp_path: Path):
    """One breaker state trips from HTTP-transport-class failures, and once
    open, a subsequent call that would have used the `gh` subprocess path
    also fails fast without spawning gh -- proving the breaker is not
    per-transport.
    """
    from charlie_work.config import GhCircuitBreakerConfig

    _install_token(monkeypatch)
    _install_fake_connection(monkeypatch, [ConnectionRefusedError("refused")] * 3)

    gh_commands: list[list[str]] = []

    def fake_gh_run(cmd, *args, **kwargs):
        gh_commands.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    # `subprocess` is a shared module object: this patches both github.py's
    # and http_transport.py's `subprocess.run` (the latter used for
    # `gh auth token` resolution) in one call.
    monkeypatch.setattr(github_module.subprocess, "run", fake_gh_run)

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(
            gh_max_retries=0,
            gh_circuit_breaker=GhCircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0),
        ),
    )
    # Established pattern (tests/test_githublike_protocol_l09.py): seed the
    # shared _list_cache directly rather than touching a real git remote.
    gh._list_cache[("_repo_owner_name",)] = ("acme", "widgets")
    for _ in range(3):
        result = gh.run(["api", "rate_limit"], json_output=True, allow_failure=True)
        assert result.ok is False

    assert gh._circuit_breaker_state.phase == "open"

    # A non-HTTP-candidate call (would otherwise spawn `gh issue list`) must
    # now also fail fast without a subprocess call, proving the shared
    # breaker gates `run()` itself, upstream of transport selection. The one
    # recorded command is `gh auth token` from resolving the HTTP path's
    # token during the first of the three failing calls above -- not a
    # `gh issue list` invocation, which never happens.
    result = gh.run(["issue", "list"], json_output=True, allow_failure=True)
    assert result.ok is False
    assert "circuit breaker open" in (result.error or "").lower()
    assert gh_commands == [["gh", "auth", "token"]]


def test_runtime_none_github_instance_uses_gh_subprocess_for_api_calls(
    monkeypatch, tmp_path: Path
):
    """Existing-test-suite compatibility (issue #1834): `GitHub(tmp_path)`
    with no RuntimeConfig must keep reaching a monkeypatched
    `subprocess.run` for `gh api`-shaped calls, exactly as this repo's
    pre-#1834 test suite already relies on.
    """

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:3] == ["gh", "api", "rate_limit"]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        http_transport, "HTTPSConnection", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    gh = github_module.GitHub(tmp_path)
    result = gh.run(["api", "rate_limit"])
    assert result == "{}"
