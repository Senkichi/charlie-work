"""REST mutations as typed requests (ADR-0006, flagged changes B2/B3/B8/B9/B11).

Every case drives a real ``GitHub`` over scripted adapters (``make_github``);
nothing touches the network, ``gh`` or the host clock.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _fake_transport import (
    FakeAdapter,
    graphql_ok,
    graphql_variables,
    make_github,
    ok,
    sent,
)
from charlie_work.config_validation import ConfigError


def test_merge_pr_returns_the_api_message(tmp_path: Path) -> None:
    reply = ok({"merged": True, "message": "Pull Request successfully merged"})
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.merge_pr(7, "squash") == "Pull Request successfully merged"
    assert sent(http) == [
        ("PUT", "repos/{owner}/{repo}/pulls/7/merge", {"merge_method": "squash"})
    ]


def test_merge_pr_without_an_api_message_falls_back_to_a_truthy_value(tmp_path: Path) -> None:
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"merged": True})]))

    assert gh.merge_pr(7, "merge") == "merged #7"


def test_merge_pr_match_head_commit_becomes_the_sha_guard(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"message": "done"})]))

    gh.merge_pr(7, "rebase", merge_flags=("--match-head-commit=abc123",))

    assert sent(http) == [
        (
            "PUT",
            "repos/{owner}/{repo}/pulls/7/merge",
            {"merge_method": "rebase", "sha": "abc123"},
        )
    ]


def test_merge_pr_admin_flag_is_the_plain_rest_merge(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"message": "done"})]))

    gh.merge_pr(7, "squash", admin=True)

    assert sent(http) == [
        ("PUT", "repos/{owner}/{repo}/pulls/7/merge", {"merge_method": "squash"})
    ]


def test_merge_pr_unknown_flag_raises_config_error_before_any_request(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path)

    with pytest.raises(ConfigError, match="--delete-branch"):
        gh.merge_pr(7, "squash", merge_flags=("--delete-branch",))

    assert http.calls == []
    assert gh_adapter.api_requests == []


def test_merge_pr_auto_still_goes_through_the_gh_cli(tmp_path: Path) -> None:
    """B10: ``--auto`` has no REST route, so it is a node-id read plus the
    ``enablePullRequestAutoMerge`` mutation, both over GraphQL (the leaf name
    predates the migration, when it was a ``gh pr merge --auto`` passthrough)."""
    replies = [
        graphql_ok({"repository": {"pullRequest": {"id": "PR_node7"}}}),
        graphql_ok({"enablePullRequestAutoMerge": {"pullRequest": {"number": 7}}}),
    ]
    gh, http, gh_adapter = make_github(tmp_path, http=FakeAdapter("http", replies))

    assert gh.merge_pr(7, "squash", merge_flags=("--auto",)) == "merged #7"

    read, mutation = http.api_requests
    assert graphql_variables(read) == {"owner": "octo", "name": "hello", "number": 7}
    assert "enablePullRequestAutoMerge" in mutation.document
    assert graphql_variables(mutation) == {"id": "PR_node7", "method": "SQUASH"}
    assert gh_adapter.api_requests == []


def test_merge_pr_dry_run_sends_nothing_and_still_reads_as_merged(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path, dry_run=True)

    assert gh.merge_pr(7, "squash") == "merged #7"

    assert http.calls == []
    assert gh_adapter.api_requests == []


def test_label_create_posts_once_when_the_label_is_new(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"name": "x"}, status=201)]))

    gh.label_create("agent:queued", "#0E8A16", "queued")

    assert sent(http) == [
        (
            "POST",
            "repos/{owner}/{repo}/labels",
            {"name": "agent:queued", "color": "0E8A16", "description": "queued"},
        )
    ]


def test_label_create_patches_the_existing_label_after_a_422(tmp_path: Path) -> None:
    replies = [ok({"message": "Validation Failed"}, status=422), ok({"name": "x"})]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", replies))

    gh.label_create("agent:queued", "#0E8A16", "queued")

    assert sent(http) == [
        (
            "POST",
            "repos/{owner}/{repo}/labels",
            {"name": "agent:queued", "color": "0E8A16", "description": "queued"},
        ),
        (
            "PATCH",
            "repos/{owner}/{repo}/labels/agent%3Aqueued",
            {"color": "0E8A16", "description": "queued"},
        ),
    ]


def test_label_create_does_not_patch_after_a_non_422_failure(tmp_path: Path) -> None:
    gh, http, _ = make_github(
        tmp_path, http=FakeAdapter("http", [ok({"message": "Forbidden"}, status=403)])
    )

    gh.label_create("agent:queued", "0E8A16", "queued")

    assert len(http.api_requests) == 1


def test_removing_an_absent_label_counts_as_removed(tmp_path: Path) -> None:
    gh, _, _ = make_github(
        tmp_path, http=FakeAdapter("http", [ok({"message": "Label does not exist"}, status=404)])
    )

    assert gh.remove_issue_label(5, "agent:queued") is True


def test_removing_a_label_reports_other_failures(tmp_path: Path) -> None:
    gh, _, _ = make_github(
        tmp_path, http=FakeAdapter("http", [ok({"message": "Forbidden"}, status=403)])
    )

    assert gh.remove_issue_label(5, "agent:queued") is False


def test_pr_create_takes_the_number_from_the_json_not_the_url(tmp_path: Path) -> None:
    reply = ok({"number": 42, "html_url": "https://github.com/octo/hello/pull/7"}, status=201)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.pr_create("agent/issue-1", "main", "title", "body") == 42


def test_pr_create_returns_none_on_a_failed_create(tmp_path: Path) -> None:
    reply = ok({"message": "Validation Failed"}, status=422)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.pr_create("agent/issue-1", "main", "title", "body") is None
