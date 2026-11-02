"""GhAdapter: request rendering and ``gh api --include`` parsing (ADR-0006).

``spawn`` is injected, so no subprocess runs and nothing depends on a
``gh`` binary being installed (host-independent).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work.github_transport import (
    CliCommand,
    CliRequest,
    FailureKind,
    GhAdapter,
    GraphQLRequest,
    Response,
    RestRequest,
    TransportFailure,
)
from charlie_work.github_transport.gh_adapter import parse_include_output, render_argv

GET = RestRequest.of("GET", "repos/o/r/pulls/1", query={"state": "all"})
POST = RestRequest.of("POST", "repos/o/r/issues", body={"title": "t"})
QUERY = GraphQLRequest.of("query Q($n: Int!) { a }", {"n": 1})


class Spawn:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[list[str], str | None, Path, float]] = []

    def __call__(self, argv, stdin, cwd, timeout):  # noqa: ANN001
        self.calls.append((argv, stdin, cwd, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def proc(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def send(request, result, tmp_path: Path):
    spawn = Spawn(result)
    out = GhAdapter(tmp_path, spawn=spawn).send(request, token="ignored", timeout=7.0)
    return out, spawn


def test_rest_get_renders_gh_api_include_with_the_query_in_the_target() -> None:
    argv, stdin = render_argv(GET)
    assert argv == [
        "gh", "api", "--include", "-X", "GET",
        "-H", "Accept: application/vnd.github+json",
        "repos/o/r/pulls/1?state=all",
    ]  # fmt: skip
    assert stdin is None


def test_rest_body_travels_on_stdin_never_as_fields() -> None:
    argv, stdin = render_argv(POST)
    assert "--input" in argv and argv[argv.index("--input") + 1] == "-"
    assert "-f" not in argv and "-F" not in argv
    assert json.loads(stdin) == {"title": "t"}


def test_graphql_renders_one_api_call_with_a_json_stdin() -> None:
    argv, stdin = render_argv(QUERY)
    assert argv[:4] == ["gh", "api", "--include", "graphql"]
    assert json.loads(stdin) == {"query": QUERY.document, "variables": {"n": 1}}


def test_cli_requests_render_their_command() -> None:
    assert render_argv(CliRequest(CliCommand.AUTH_TOKEN)) == (["gh", "auth", "token"], None)


def test_no_paginate_flag_is_ever_rendered() -> None:
    for request in (GET, POST, QUERY):
        assert "--paginate" not in render_argv(request)[0]


def test_include_output_is_split_into_status_headers_and_body() -> None:
    raw = (
        "HTTP/2.0 200 OK\r\nContent-Type: application/json\r\n"
        "X-RateLimit-Remaining: 42\r\n\r\n"
        '{"n": 1}'
    )
    status, headers, body = parse_include_output(raw)  # type: ignore[misc]
    assert status == 200 and body == '{"n": 1}'
    assert dict(headers)["x-ratelimit-remaining"] == "42"


def test_the_last_header_block_wins_after_a_redirect() -> None:
    raw = "HTTP/2.0 302 Found\nLocation: x\n\nHTTP/2.0 200 OK\nA: b\n\nbody"
    status, headers, body = parse_include_output(raw)  # type: ignore[misc]
    assert (status, body, dict(headers)) == (200, "body", {"a": "b"})


def test_output_without_a_status_line_is_none() -> None:
    assert parse_include_output("error connecting to api.github.com") is None


def test_a_successful_call_is_a_response_with_headers_for_the_budget(tmp_path: Path) -> None:
    raw = "HTTP/2.0 200 OK\nX-RateLimit-Limit: 5000\n\n{}"
    out, spawn = send(GET, proc(raw), tmp_path)
    assert isinstance(out, Response) and out.ok and out.adapter == "gh"
    assert out.header("x-ratelimit-limit") == "5000"
    argv, _, cwd, timeout = spawn.calls[0]
    assert argv[0] == "gh" and cwd == tmp_path and timeout == 7.0


def test_a_non_2xx_answer_is_a_response_even_though_gh_exits_nonzero(tmp_path: Path) -> None:
    raw = 'HTTP/2.0 404 Not Found\n\n{"message": "Not Found"}'
    out, _ = send(GET, proc(raw, "gh: Not Found (HTTP 404)", rc=1), tmp_path)
    assert isinstance(out, Response) and out.status == 404 and out.returncode == 1


def test_graphql_errors_in_a_200_are_parsed(tmp_path: Path) -> None:
    body = json.dumps({"data": None, "errors": [{"message": "x", "type": "NOT_FOUND"}]})
    out, _ = send(QUERY, proc(f"HTTP/2.0 200 OK\n\n{body}"), tmp_path)
    assert isinstance(out, Response) and not out.ok
    assert out.graphql_errors[0].type == "NOT_FOUND"


def test_a_graphql_200_with_a_non_object_body_is_an_adapter_defect(tmp_path: Path) -> None:
    out, _ = send(QUERY, proc("HTTP/2.0 200 OK\n\n[1]"), tmp_path)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


@pytest.mark.parametrize(
    ("stderr", "kind"),
    [
        ("error connecting to api.github.com", FailureKind.CONNECT),
        ("dial tcp: connection refused", FailureKind.CONNECT),
        ("read tcp: connection reset by peer", FailureKind.SENT_NO_RESPONSE),
        ("something unrecognised", FailureKind.ADAPTER_DEFECT),
    ],
)
def test_stderr_without_a_status_line_is_classified_with_the_shared_markers(
    stderr: str, kind: FailureKind, tmp_path: Path
) -> None:
    out, _ = send(GET, proc("", stderr, rc=1), tmp_path)
    assert isinstance(out, TransportFailure) and out.kind is kind and out.adapter == "gh"


def test_a_missing_binary_is_cli_missing(tmp_path: Path) -> None:
    out, _ = send(GET, FileNotFoundError("gh"), tmp_path)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.CLI_MISSING


def test_a_subprocess_timeout_is_timeout(tmp_path: Path) -> None:
    out, _ = send(GET, subprocess.TimeoutExpired(cmd="gh", timeout=7), tmp_path)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.TIMEOUT


def test_other_os_errors_are_an_adapter_defect(tmp_path: Path) -> None:
    out, _ = send(GET, PermissionError("denied"), tmp_path)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


def test_cli_results_carry_the_return_code_and_trimmed_output(tmp_path: Path) -> None:
    ok, _ = send(CliRequest(CliCommand.AUTH_TOKEN), proc("ghp_abc\n"), tmp_path)
    assert isinstance(ok, Response) and ok.ok and ok.body == "ghp_abc" and ok.returncode == 0
    bad, _ = send(CliRequest(CliCommand.AUTH_TOKEN), proc("", "not logged in", rc=1), tmp_path)
    assert isinstance(bad, Response) and not bad.ok and bad.returncode == 1
    assert bad.body == "not logged in"
