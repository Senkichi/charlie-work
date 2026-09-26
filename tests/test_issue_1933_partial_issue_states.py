"""Issue #1933: one unresolvable issue number in the batched GraphQL
``are_issues_open`` query must not demote the whole batch to the per-issue
``issue_view`` fallback.

GitHub answers ``s_<n>: issue(number: <n>)`` aliases that cannot resolve with
a ``null`` data entry plus a per-node ``errors`` entry -- the HTTP status is
still 200 and every other alias resolves, but ``gh`` exits non-zero (and this
repo's pooled HTTP transport mirrors that by translating the error). Before
this fix the non-zero exit raised through ``_graphql_query`` and discarded
the partial ``data`` entirely, so ``are_issues_open`` re-fetched EVERY
uncached number through ``issue_view`` -- slow enough to blow the
``fleet status --json`` timeout whenever one stale issue number was in play.

The fix has three seams, each covered below:

* ``http_transport._execute_graphql`` preserves the response body on stdout
  for erroring GraphQL responses (real ``gh api`` does the same -- in
  cli/cli's ``pkg/cmd/api/api.go`` ``processResponse`` copies the body to the
  output writer before emitting the error), so the partial ``data`` reaches
  callers as ``GitHubRunResult.value``.
* ``github_capabilities.graphql_issue_states`` consumes that partial body:
  resolved aliases map normally; aliases with a null node are absent from
  the result.
* ``Issues.are_issues_open`` runs its per-issue ``issue_view`` fallback only
  over numbers absent from the batched result.

No live network anywhere: the HTTP transport's ``HTTPSConnection`` is faked
(the same stand-in shape as ``tests/test_http_transport.py``) and the ``gh``
subprocess path is monkeypatched.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work.config import RuntimeConfig
from charlie_work.github import GitHub, GitHubError, GitHubRunResult
from charlie_work.github_capabilities import http_transport


class _FakeResponse:
    def __init__(self, status: int, headers: dict | None = None, body: bytes = b""):
        self.status = status
        self._headers = headers or {}
        self._body = body

    def getheaders(self):
        return list(self._headers.items())

    def read(self) -> bytes:
        return self._body


class _FakeConnection:
    def __init__(self, responses: list):
        self._responses = list(responses)

    def request(self, method, url, body=None, headers=None):
        pass

    def getresponse(self):
        if not self._responses:
            raise AssertionError("no more fake HTTP responses queued")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        pass


def _partial_body(*, resolved: dict[str, dict], unresolved: list[int]) -> dict:
    """A real-shaped GraphQL response: every resolvable alias carries data,
    the failing alias is null, and the ``errors`` array names it by path."""
    repo = dict(resolved)
    errors = []
    for number in unresolved:
        repo[f"s_{number}"] = None
        errors.append(
            {
                "message": f"Could not resolve to an Issue with the number of {number}.",
                "path": ["repository", f"s_{number}"],
            }
        )
    return {"data": {"repository": repo}, "errors": errors}


def test_graphql_error_response_body_reaches_stdout(monkeypatch, tmp_path: Path) -> None:
    """``_execute_graphql`` must keep the response body on stdout for a 200
    response with an ``errors`` array -- that body is what carries the
    partial ``data`` ``GitHub.run`` parses into ``GitHubRunResult.value``.
    Real ``gh api graphql`` does the same (body copied to stdout before the
    error is emitted to stderr), so this is parity, not a new convention.
    """
    body = json.dumps(
        _partial_body(resolved={"s_1": {"number": 1, "state": "OPEN"}}, unresolved=[361])
    ).encode()

    def fake_run(cmd, *args, **kwargs):
        assert cmd == ["gh", "auth", "token"]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="t\n", stderr="")

    monkeypatch.setattr(http_transport.subprocess, "run", fake_run)
    fake = _FakeConnection([_FakeResponse(200, {}, body)])
    monkeypatch.setattr(http_transport, "HTTPSConnection", lambda host, timeout=None: fake)

    result = http_transport.run_gh_command(
        args=["api", "graphql", "-f", "query=query { x }"],
        command=["gh", "api", "graphql", "-f", "query=query { x }"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        runtime=RuntimeConfig(),
        transport_state=http_transport.build_http_transport_state(),
        resolve_owner_repo=lambda: ("acme", "widgets"),
    )

    assert result.returncode == 1
    assert result.stderr.startswith("GraphQL: Could not resolve to an Issue")
    parsed = json.loads(result.stdout)
    assert parsed["data"]["repository"]["s_1"] == {"number": 1, "state": "OPEN"}
    assert parsed["data"]["repository"]["s_361"] is None


def test_graphql_issue_states_omits_unresolved_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The batched function returns states for resolved aliases only; an
    unresolvable alias is absent from the mapping rather than forcing the
    whole result to fail."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    body = _partial_body(
        resolved={
            "s_1": {"number": 1, "state": "OPEN"},
            "s_2": {"number": 2, "state": "CLOSED"},
        },
        unresolved=[361],
    )

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        return GitHubRunResult(
            ok=False,
            returncode=1,
            stdout=json.dumps(body),
            stderr="GraphQL: Could not resolve to an Issue with the number of 361. "
            "(repository.s_361)",
            value=body,
            error="GraphQL: Could not resolve to an Issue with the number of 361. "
            "(repository.s_361)",
        )

    monkeypatch.setattr(GitHub, "run", fake_run)
    assert gh._graphql_issue_states([1, 2, 361]) == {1: True, 2: False}


def test_graphql_issue_states_still_raises_without_usable_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure with no partial ``data`` at all (transport error, or a
    body that never reached stdout) still raises -- the caller's whole-set
    fallback contract is unchanged for genuine batch failures."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        return GitHubRunResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
            value=None,
            error="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
        )

    monkeypatch.setattr(GitHub, "run", fake_run)
    with pytest.raises(GitHubError):
        gh._graphql_issue_states([1])


def test_are_issues_open_per_issue_fallback_covers_only_unresolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The incident shape: one unresolvable alias leaves the other numbers
    resolved by the single batched query, and ``issue_view`` runs only for
    the number the batch could not resolve."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    body = _partial_body(
        resolved={
            "s_1": {"number": 1, "state": "OPEN"},
            "s_2": {"number": 2, "state": "CLOSED"},
        },
        unresolved=[361],
    )
    issue_view_calls: list[int] = []

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        if args[:2] == ["api", "graphql"]:
            return GitHubRunResult(
                ok=False,
                returncode=1,
                stdout=json.dumps(body),
                stderr="GraphQL: Could not resolve to an Issue with the number of 361. "
                "(repository.s_361)",
                value=body,
                error="GraphQL: Could not resolve to an Issue with the number of 361. "
                "(repository.s_361)",
            )
        assert args[:2] == ["issue", "view"]
        number = int(args[2])
        issue_view_calls.append(number)
        # REST issue_view CAN resolve the number GraphQL choked on (issue
        # #1933's #361 evidence) -- and it turns out to be OPEN, so it must
        # land in the open set rather than being silently marked closed.
        return {"number": number, "state": "OPEN"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    assert gh.are_issues_open([1, 2, 361]) == {1, 361}
    assert issue_view_calls == [361]
    assert gh._list_cache[("issue_open", 361)] is True
    assert gh._list_cache[("issue_open", 2)] is False


def test_are_issues_open_full_fallback_when_batch_yields_no_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Contract preservation: a batch failure with no usable partial data
    still per-issue-fetches every uncached number (the pre-#1933 path)."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    issue_view_calls: list[int] = []

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        if args[:2] == ["api", "graphql"]:
            return GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="HTTP 502",
                value=None,
                error="HTTP 502",
            )
        assert args[:2] == ["issue", "view"]
        number = int(args[2])
        issue_view_calls.append(number)
        return {"number": number, "state": "OPEN" if number != 2 else "CLOSED"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    assert gh.are_issues_open([1, 2, 3]) == {1, 3}
    assert sorted(issue_view_calls) == [1, 2, 3]


def test_are_issues_open_emits_telemetry_for_partial_batch_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partial-batch fallback must be observable: the degraded condition is
    exactly what the incident needed telemetry for -- which numbers were bad
    and how large the requested set was. Whole-batch failures (the pre-#1933
    path) keep their existing log-line-only behavior."""
    from charlie_work.github_capabilities import issues as issues_module

    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    body = _partial_body(
        resolved={"s_1": {"number": 1, "state": "OPEN"}},
        unresolved=[361],
    )
    events: list[tuple[str, dict]] = []

    def fake_log_event(state_path, kind, payload, **kwargs):
        events.append((kind, payload))

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        if args[:2] == ["api", "graphql"]:
            return GitHubRunResult(
                ok=False,
                returncode=1,
                stdout=json.dumps(body),
                stderr="GraphQL: Could not resolve to an Issue with the number of 361. "
                "(repository.s_361)",
                value=body,
                error="GraphQL: Could not resolve to an Issue with the number of 361. "
                "(repository.s_361)",
            )
        return {"number": int(args[2]), "state": "CLOSED"}

    monkeypatch.setattr(GitHub, "run", fake_run)
    monkeypatch.setattr(issues_module, "log_event", fake_log_event)

    assert gh.are_issues_open([1, 361]) == {1}
    partial = [p for k, p in events if k == "github_issue_state_partial_fallback"]
    assert partial == [{"unresolved": [361], "requested": 2}]


def test_are_issues_open_end_to_end_over_http_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production stack, no seams mocked out: ``gh_transport="http"``
    (the ``RuntimeConfig`` default) -> ``_execute_graphql`` preserves the
    body -> ``run()`` parses it into ``GitHubRunResult.value`` -> partial
    states -> ``issue_view`` fallback for the unresolved number only."""
    gh = GitHub(repo_root=tmp_path, runtime=RuntimeConfig())
    gh._list_cache[("_repo_owner_name",)] = ("acme", "widgets")
    body = json.dumps(
        _partial_body(resolved={"s_1": {"number": 1, "state": "OPEN"}}, unresolved=[361])
    ).encode()
    issue_view_calls: list[int] = []

    def fake_run(cmd, *args, **kwargs):
        if cmd == ["gh", "auth", "token"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="t\n", stderr="")
        assert cmd[:3] == ["gh", "issue", "view"]
        number = int(cmd[3])
        issue_view_calls.append(number)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps({"number": number, "state": "OPEN"}),
            stderr="",
        )

    monkeypatch.setattr(http_transport.subprocess, "run", fake_run)
    fake = _FakeConnection([_FakeResponse(200, {}, body)])
    monkeypatch.setattr(http_transport, "HTTPSConnection", lambda host, timeout=None: fake)

    assert gh.are_issues_open([1, 361]) == {1, 361}
    assert issue_view_calls == [361]
