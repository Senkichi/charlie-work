"""Tests for ``GitHub.create_issue`` and ``GitHub.edit_issue_body`` (issue #3116).

The two write requests the foreman's Apply step needs (ADR-0010). Each goes
through the guarded transport as a typed REST request, returns errors as
values, and passes title/body through ``outbound_body_guard`` before
anything leaves the process -- same enforcement shape ``pr_create`` has.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any

import pytest

from _fake_transport import FakeAdapter, failure, make_github, ok, sent
from charlie_work.github import GitHub
from charlie_work.github_transport import CliRequest, FailureKind, RestRequest
from charlie_work.instrumentation import query_events


def _client(
    tmp_path: pathlib.Path, reply: object | None = None, *, dry_run: bool = False
) -> tuple[GitHub, FakeAdapter]:
    """A real ``GitHub`` over a scripted http adapter (no network, no gh)."""
    script = [reply if reply is not None else ok({"number": 42}, status=201)]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", script), dry_run=dry_run)
    return gh, http


def _state_file(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "state.json"


# A 40-character high-entropy alphanumeric run -- distinct characters, so the
# vendored rules' Shannon-entropy gates pass (same construction as
# tests/test_outbound_body_guard.py's fixtures; synthetic, never a real token).
_MIX = "x7K9pQ2mN4vB8wL1cR3tY6uI0oP5aS2dF4gH9jZq"
_GHO = "gho_" + (_MIX * 2)[:36]


# --------------------------------------------------------------------------
# create_issue: request shape and number round-trip
# --------------------------------------------------------------------------


def test_create_issue_sends_rest_post_and_returns_number(tmp_path: Path) -> None:
    gh, http = _client(tmp_path)
    number = gh.create_issue(title="t", body="b", labels=["a", "b"])

    assert number == 42
    assert len(http.api_requests) == 1
    request = http.api_requests[0]
    assert isinstance(request, RestRequest)
    assert (request.method, request.route) == ("POST", "repos/octo/hello/issues")
    assert json.loads(request.body or "{}") == {"title": "t", "body": "b", "labels": ["a", "b"]}
    # No gh-CLI local commands went out for a pure REST write.
    assert not [r for r in http.requests if isinstance(r, CliRequest)]


def test_create_issue_default_labels_sends_empty_list(tmp_path: Path) -> None:
    gh, http = _client(tmp_path)
    gh.create_issue(title="t", body="b")

    assert sent(http) == [
        ("POST", "repos/{owner}/{repo}/issues", {"title": "t", "body": "b", "labels": []})
    ]


def test_create_issue_uses_label_strings_as_given(tmp_path: Path) -> None:
    """Labels pass through verbatim -- no label string is invented or rewritten."""
    gh, http = _client(tmp_path)
    gh.create_issue(title="t", body="b", labels=("foreman:alpha", "custom label"))

    assert sent(http)[0][2] == {
        "title": "t",
        "body": "b",
        "labels": ["foreman:alpha", "custom label"],
    }


def test_create_issue_success_without_number_returns_none(tmp_path: Path) -> None:
    gh, _ = _client(tmp_path, reply=ok({"id": 1}, status=201))
    assert gh.create_issue(title="t", body="b") is None


# --------------------------------------------------------------------------
# edit_issue_body: request shape
# --------------------------------------------------------------------------


def test_edit_issue_body_sends_rest_patch(tmp_path: Path) -> None:
    gh, http = _client(tmp_path, reply=ok({"number": 9}))
    assert gh.edit_issue_body(9, "new body") is True
    assert sent(http) == [("PATCH", "repos/{owner}/{repo}/issues/9", {"body": "new body"})]


# --------------------------------------------------------------------------
# Failure values, never raises
# --------------------------------------------------------------------------


def test_create_issue_transport_failure_returns_none(tmp_path: Path) -> None:
    # SENT_NO_RESPONSE is a non-fallback kind: the request may have landed, so
    # the transport reports failure rather than retrying through the gh adapter.
    gh, _ = _client(tmp_path, reply=failure(FailureKind.SENT_NO_RESPONSE))
    assert gh.create_issue(title="t", body="b") is None


def test_edit_issue_body_transport_failure_returns_false(tmp_path: Path) -> None:
    gh, _ = _client(tmp_path, reply=failure(FailureKind.SENT_NO_RESPONSE))
    assert gh.edit_issue_body(9, "new body") is False


def test_edit_issue_body_http_error_returns_false(tmp_path: Path) -> None:
    gh, _ = _client(tmp_path, reply=ok({"message": "Not Found"}, status=404))
    assert gh.edit_issue_body(9, "new body") is False


# --------------------------------------------------------------------------
# dry-run: nothing is sent, the failure value comes back
# --------------------------------------------------------------------------


def test_create_issue_dry_run_sends_nothing(tmp_path: Path) -> None:
    gh, http = _client(tmp_path, dry_run=True)
    assert gh.create_issue(title="t", body="b") is None
    assert http.calls == []


def test_edit_issue_body_dry_run_sends_nothing(tmp_path: Path) -> None:
    gh, http = _client(tmp_path, dry_run=True)
    assert gh.edit_issue_body(9, "new body") is False
    assert http.calls == []


# --------------------------------------------------------------------------
# Secret guard: refused before anything leaves the process
# --------------------------------------------------------------------------


def test_create_issue_refuses_secret_body_before_transport(tmp_path: Path) -> None:
    gh, http = _client(tmp_path)
    assert gh.create_issue(title="t", body=f"notes {_GHO}") is None
    assert http.calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "create_issue"


def test_create_issue_refuses_secret_title_before_transport(tmp_path: Path) -> None:
    """The title lands in the same visible edit history as the body."""
    gh, http = _client(tmp_path)
    assert gh.create_issue(title=f"wip {_GHO}", body="clean") is None
    assert http.calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "create_issue"


def test_edit_issue_body_refuses_secret_body_before_transport(tmp_path: Path) -> None:
    gh, http = _client(tmp_path)
    assert gh.edit_issue_body(9, f"notes {_GHO}") is False
    assert http.calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "edit_issue_body"
    assert events[0]["issue_number"] == 9


# --------------------------------------------------------------------------
# Secret guard: machinery failure fails closed (nothing sent)
# --------------------------------------------------------------------------


@pytest.fixture
def _broken_ruleset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the guard's rule loader at a malformed file so every check raises
    ``OutboundBodyGuardError`` (same construction as test_outbound_body_guard.py's
    fixture)."""
    import charlie_work.outbound_body_guard as obg

    bad = tmp_path / "broken-secrets.toml"
    bad.write_text("rules = [[[not toml", encoding="utf-8")
    monkeypatch.setattr(obg, "_RULES_PATH", bad)
    obg._rules.cache_clear()
    yield bad
    obg._rules.cache_clear()


def test_create_issue_guard_failure_returns_none_before_transport(
    tmp_path: Path, _broken_ruleset: Path
) -> None:
    """Fail-closed: a guard that cannot run cannot pass text through."""
    gh, http = _client(tmp_path)
    assert gh.create_issue(title="t", body="clean") is None
    assert http.calls == []


def test_edit_issue_body_guard_failure_returns_false_before_transport(
    tmp_path: Path, _broken_ruleset: Path
) -> None:
    gh, http = _client(tmp_path)
    assert gh.edit_issue_body(9, "clean") is False
    assert http.calls == []
