"""GitHub mutating calls: label add/remove, branch delete, dry-run mutating-command guard, token-mint retry.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from _fake_transport import FakeAdapter, failure, make_github, ok, sent
from charlie_work.github_transport import FailureKind, RestRequest
from charlie_work.github_transport.guarded import is_retryable
from charlie_work.github_transport.legacy_argv import request_for_argv


def _rejected():
    """A non-transient API rejection (never retried)."""
    return ok('{"message": "Validation Failed"}', status=422)


def test_github_delete_branch_failure_returns_false(tmp_path: Path) -> None:
    """A rejected delete (422) is a False result, not a raise."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [_rejected()]))

    assert gh.delete_branch("agent/issue-1-x") is False


def test_github_add_issue_label_failure_does_not_raise(tmp_path: Path) -> None:
    """C5 boundary test: a rejected add_issue_label returns a value, does not raise."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [_rejected()]))

    gh.add_issue_label(123, "agent:in-progress")


def test_github_remove_issue_label_failure_does_not_raise(tmp_path: Path) -> None:
    """C5 boundary test: a rejected remove_issue_label returns a value, does not raise."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [_rejected()]))

    gh.remove_issue_label(123, "agent:in-progress")


def test_github_add_issue_label_returns_false_on_failure(tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns False when the request is rejected."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [_rejected()]))

    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is False, "add_issue_label must return False on failure"


def test_github_add_issue_label_returns_true_on_success(tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns True on a successful request."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok([{"name": "x"}])]))
    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is True, "add_issue_label must return True on success"
    assert sent(http) == [
        ("POST", "repos/{owner}/{repo}/issues/123/labels", {"labels": ["agent:in-progress"]})
    ]


def test_github_remove_issue_label_returns_false_on_failure(tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns False when the request is rejected."""
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [_rejected()]))

    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is False, "remove_issue_label must return False on failure"


def test_github_remove_issue_label_returns_true_on_success(tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns True on a successful request."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok([])]))
    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is True, "remove_issue_label must return True on success"
    assert sent(http) == [
        ("DELETE", "repos/{owner}/{repo}/issues/123/labels/agent%3Ain-progress", None)
    ]


def test_github_dry_run_skips_mutating_command(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path, dry_run=True)

    out = gh.run(["run", "cancel", "7"])

    assert out.startswith("DRY-RUN:")
    assert http.calls == [] and gh_adapter.calls == []  # nothing sent for a mutation


def test_github_dry_run_allows_readonly_command(tmp_path: Path) -> None:
    reply = ok(
        {"data": {"repository": {"issues": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}
    )
    gh, http, _ = make_github(tmp_path, dry_run=True, http=FakeAdapter("http", [reply]))

    gh.run(["issue", "list", "--label", "x", "--json", "number"], json_output=True)

    assert len(http.calls) == 1  # read-only command still executes under dry-run


def test_is_mutating_classifies_readonly_and_mutating() -> None:
    """Mutation-ness is a property of the typed request (ADR-0006).

    A read-only gh shape translates to a request whose ``is_mutation`` is False;
    a mutating shape has no row at all, so ``GitHub.run`` refuses it rather than
    guessing at a classification.
    """
    for readonly in (
        ["issue", "list", "--json", "number"],
        ["pr", "view", "1", "--json", "state"],
        ["pr", "checks", "1", "--json", "name"],
        ["run", "list", "--json", "databaseId"],
    ):
        assert request_for_argv(readonly).request.is_mutation is False
    for mutating in (["pr", "merge", "1"], ["issue", "edit", "1"], ["label", "create", "x"]):
        assert request_for_argv(mutating) is None
    assert request_for_argv(["run", "cancel", "7"]).request.is_mutation is True


def test_is_mutating_blocks_the_argv_delete_branch_actually_builds(tmp_path: Path) -> None:
    """#914/#917: `-X DELETE` classified as read-only, so `--dry-run` really deleted
    PR head branches.

    The request is captured from `delete_branch` itself rather than written out by
    hand, so the gate cannot drift away from its most destructive caller. Mutation-ness
    is now derived from the request's method, and the transport -- not a per-caller
    guard -- suppresses the write under dry-run.
    """
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok("", status=204)]))
    assert gh.delete_branch("feature/x") is True
    assert len(http.api_requests) == 1
    assert http.api_requests[0].is_mutation is True

    dry, dry_http, _ = make_github(tmp_path, dry_run=True)
    assert dry.delete_branch("feature/x") is True
    assert dry_http.calls == []


def test_is_mutating_api_method_spellings_and_preserved_reads() -> None:
    """Every spelling `gh` accepts for a mutating method has no row, and the
    real read call sites keep theirs.

    The read-only half is the half that protects `--dry-run` from being tightened into
    uselessness: these are real live call sites, and without them a future "just deny
    all `gh api`" change passes every other test in the suite. The table fails
    CLOSED: an argv it cannot model is refused, never run as a guess.
    """
    for mutating in (
        ["api", "-X", "DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X=DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-XDELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X", "POST", "repos/o/r/actions/runners/remove-token"],
        ["api", "--method", "PATCH", "repos/o/r/issues/1"],
        ["api", "--method=PUT", "repos/o/r/branches/main/protection"],
        ["api", "-X"],  # named but valueless -> fail closed, not open
        ["api", "--method"],
        ["api", "repos/o/r/issues", "-f", "title=x"],  # params switch gh to POST
        ["api", "repos/o/r/issues", "--field=labels[]=bug"],
        # pflag takes an attached shorthand value here too, exactly as for -X (#919).
        ["api", "repos/o/r/issues", "-ftitle=x"],
        ["api", "repos/o/r/issues", "-Flabels[]=bug"],
        ["api", "repos/o/r/issues", "-F"],
        ["api", "repos/o/r/issues", "--raw-field", "title=x"],
        ["api", "repos/o/r/issues", "--input", "body.json"],
        ["api", "repos/o/r/issues", "--input=-"],
    ):
        assert request_for_argv(mutating) is None, mutating

    for readonly in (
        ["api", "rate_limit"],
        ["api", "repos/o/r/commits/abc/check-runs"],
        ["api", "repos/o/r/compare/main...topic"],
        ["api", "repos/o/r/branches/main/protection"],
        # A header is not a method -- github.py:1033 fetches a diff this way.
        ["api", "repos/o/r/pulls/1", "-H", "Accept: application/vnd.github.v3.diff"],
        ["api", "repos/o/r/issues", "--paginate"],
    ):
        translated = request_for_argv(readonly)
        assert translated is not None, readonly
        assert translated.request.is_mutation is False, readonly


def test_token_minting_posts_do_not_retry_post_send_failures() -> None:
    """Credential-minting POSTs must retry only on provable pre-send failures.

    The guard's retry policy (``is_retryable``) grants reads a retry on any
    transient failure and restricts mutations to failures that provably happened
    before the request was sent, so a request that may already have been applied
    is never re-sent.

    A post-send timeout on a request GitHub had actually served would mint a
    second token, so pin it here -- a future reclassification of a POST would
    otherwise reopen the loop with every other test still green (#919).
    """
    for route in (
        "repos/{owner}/{repo}/actions/runners/remove-token",
        "repos/{owner}/{repo}/actions/runners/registration-token",
    ):
        request = RestRequest("POST", route)
        assert request.is_mutation is True, route
        # Ambiguous: the request may have been served before the read timed out.
        assert is_retryable(failure(FailureKind.TIMEOUT, "i/o timeout"), is_mutation=True) is False
        # Provably pre-send -- no token can have been minted, so retrying is safe.
        assert (
            is_retryable(failure(FailureKind.CONNECT, "connection refused"), is_mutation=True)
            is True
        )

    # Control: a read is still granted the unconditional retry, so the assertions
    # above are about the mutating classification and not about the error kind.
    read = RestRequest("GET", "repos/{owner}/{repo}/actions/runners")
    assert read.is_mutation is False
    assert is_retryable(failure(FailureKind.TIMEOUT, "i/o timeout"), is_mutation=False) is True
