"""Regression tests for ``GitHub.pr_create``.

`pr_create` shipped passing ``--json number`` to ``gh pr create``, which has no
such flag. ``gh`` exited non-zero at argument parsing, before contacting the
API, so the method could never succeed and no PR was ever created -- the
orchestrator's whole "adopt a branch a worker pushed but could not open a PR
for" recovery lane (#935) was inert for every input.

Nothing caught it because every test in the suite substitutes a *fake* `gh`
object whose `pr_create` is a Python method returning a canned number. A fake
built alongside the caller is correct by construction and cannot disagree with
the real CLI about what flags exist -- no amount of coverage through that fake
would have found this. So the load-bearing test here is
``test_every_flag_pr_create_sends_is_accepted_by_the_installed_gh``, which
checks the arguments against ``gh`` itself rather than against our model of it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from _fake_transport import FakeAdapter, failure, make_github, ok
from charlie_work.github import GitHub, _pr_number_from_url
from charlie_work.github_transport import CliRequest, FailureKind, RestRequest

_URL = "https://github.com/Senkichi/charlie-work/pull/1234"


def _client(
    tmp_path: pathlib.Path, reply: object | None = None, *, dry_run: bool = False
) -> tuple[GitHub, FakeAdapter]:
    """A real ``GitHub`` over a scripted http adapter (no network, no gh)."""
    script = [reply if reply is not None else ok({"number": 1234, "html_url": _URL}, status=201)]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", script), dry_run=dry_run)
    return gh, http


# --------------------------------------------------------------------------
# The bug itself
# --------------------------------------------------------------------------


def test_pr_create_does_not_send_a_json_flag(tmp_path: pathlib.Path) -> None:
    """`pr_create` is a typed REST POST now: no gh argv exists to carry a bad
    flag, so the original `--json` bug class is unrepresentable (ADR-0006)."""
    gh, http = _client(tmp_path)
    gh.pr_create(head="agent/issue-1", base="main", title="t", body="b")

    assert len(http.calls) == 1
    request = http.api_requests[0]
    # Control: the request really is the one under test, so the absence of any
    # CLI argv is about pr_create's shape and not about an empty call list.
    assert isinstance(request, RestRequest)
    assert (request.method, request.route) == ("POST", "repos/octo/hello/pulls")
    assert not [r for r in http.requests if isinstance(r, CliRequest)]


def test_every_flag_pr_create_sends_is_accepted_by_the_installed_gh(
    tmp_path: pathlib.Path,
) -> None:
    """The REST successor of the installed-gh flag check: every field
    pr_create sends is one the documented create-pull-request endpoint accepts.
    (There is no CLI left to disagree with; the API's field list is the contract.)
    """
    gh, http = _client(tmp_path)
    gh.pr_create(head="agent/issue-1", base="main", title="t", body="b")

    sent_fields = set(json.loads(http.api_requests[0].body or "{}"))
    assert sent_fields == {"head", "base", "title", "body"}
    documented = {"title", "head", "base", "body", "draft", "maintainer_can_modify", "issue"}
    unsupported = sorted(sent_fields - documented)
    assert not unsupported, f"pr_create sends fields the pulls API does not accept: {unsupported}"


# --------------------------------------------------------------------------
# Parsing the number back out
# --------------------------------------------------------------------------


def test_pr_create_returns_the_number_from_the_url(tmp_path: pathlib.Path) -> None:
    """B9: the number comes from the JSON response, not from URL parsing."""
    gh, _ = _client(tmp_path, ok({"number": 1234, "html_url": _URL}, status=201))
    assert gh.pr_create(head="h", base="main", title="t", body="b") == 1234


def test_pr_create_prefers_the_last_url_in_the_output(tmp_path: pathlib.Path) -> None:
    """B9: a title or body echoed into the response can legitimately contain
    another PR link ("supersedes .../pull/900"). The structured `number` field
    is the created PR; no text in the body can displace it."""
    noisy = {
        "number": 1234,
        "html_url": _URL,
        "body": "note: supersedes https://github.com/Senkichi/charlie-work/pull/900",
    }
    gh, _ = _client(tmp_path, ok(noisy, status=201))
    assert gh.pr_create(head="h", base="main", title="t", body="b") == 1234


@pytest.mark.parametrize(
    "output, expected",
    [
        (_URL, 1234),
        (f"{_URL}\n", 1234),
        ("https://github.com/o/r/pull/7", 7),
        ("", None),
        ("no url here", None),
        ("https://github.com/o/r/issues/12", None),
    ],
)
def test_pr_number_from_url(output: str, expected: int | None) -> None:
    assert _pr_number_from_url(output) == expected


# --------------------------------------------------------------------------
# Failure paths return values, never raise
# --------------------------------------------------------------------------


def test_pr_create_returns_none_when_gh_fails(tmp_path: pathlib.Path) -> None:
    reply = ok({"message": "A pull request already exists"}, status=422)
    gh, http = _client(tmp_path, reply)
    assert gh.pr_create(head="h", base="main", title="t", body="b") is None
    assert len(http.calls) == 1  # control: the failure came from the answered request


def test_pr_create_logs_the_stderr_when_gh_fails(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The caller only sees None. Without the reason in the log, "not
    authenticated", "GitHub rejected it", and "we sent a bad request" are
    indistinguishable -- which is what made the original bug expensive. The
    reason is now the API's own message (B3)."""
    gh, _ = _client(tmp_path, ok({"message": "Validation Failed"}, status=422))
    with caplog.at_level("WARNING"):
        gh.pr_create(head="h", base="main", title="t", body="b")
    assert "Validation Failed" in caplog.text


def test_pr_create_returns_none_when_output_has_no_url(tmp_path: pathlib.Path) -> None:
    """A 2xx whose body carries no PR number must not be reported as a created
    PR. Returning a bogus number would put an unusable pr_number into state.json."""
    gh, _ = _client(tmp_path, ok({"unexpected": "shape"}, status=201))
    assert gh.pr_create(head="h", base="main", title="t", body="b") is None


def test_pr_create_is_a_no_op_under_dry_run(tmp_path: pathlib.Path) -> None:
    gh, http = _client(tmp_path, dry_run=True)
    assert gh.pr_create(head="h", base="main", title="t", body="b") == 0
    assert http.calls == []


def test_pr_create_returns_none_when_gh_is_missing(tmp_path: pathlib.Path) -> None:
    """A transport failure (no token, no connection, no gh) must be absorbed as
    a value, per the repo invariant that external-process errors come back as values."""
    gh, _ = _client(tmp_path, failure(FailureKind.CLI_MISSING, "gh not found"))
    assert gh.pr_create(head="h", base="main", title="t", body="b") is None
